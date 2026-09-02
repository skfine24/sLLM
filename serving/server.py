"""OpenAI-compatible HTTP front-end (stdlib only).

Endpoints:
  GET  /health                    -> {"status": "ok", "model": ...}
  GET  /v1/models                 -> {"object": "list", "data": [{id, ...}]}
  POST /v1/completions            -> {prompt, max_tokens, temperature, ...}
  POST /v1/chat/completions       -> {messages, max_tokens, temperature, ...}

Responses follow the OpenAI schema (chat.completion / text_completion
objects, `usage` token counts, `finish_reason`). `stream: true` is served
as an SSE token stream (`data:` frames, `[DONE]` terminator) for both chat
and completions. A requested `model` different from the served one is
accepted permissively (single-model server). The model is decoupled via
`serving.executor.InferenceEngine`."""

from __future__ import annotations

import functools
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from env_config import get as _env
from env_config import get_int as _env_int
from .executor import InferenceEngine
from . import diag


def _error(status: int, message: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error",
                      "code": str(status)}}


# hard cap on request bodies: an unbounded Content-Length is a trivial DoS
# vector against a stdlib server (read() would allocate attacker-chimited bytes)
MAX_BODY_BYTES = 32 << 20  # 32 MiB


class InvalidRequestError(ValueError):
    """Client-side problem (bad params, prompt too long, cannot ever fit the
    KV budget) -> HTTP 400. Subclasses ValueError so the plain model paths
    (which raise ValueError) keep mapping to 400 unchanged."""


class SaturatedError(RuntimeError):
    """Server is at capacity; the request could run once others finish.
    -> HTTP 429 with a Retry-After hint. Raised by the batched serving
    facade's admission control (never by the single-shot engine)."""

    def __init__(self, message: str = "server at capacity", retry_after: int = 1):
        super().__init__(message)
        self.retry_after = int(retry_after)


def _http_logged(method):
    """Wrap a handler method: time it and emit one INFO http line (quiet
    servers stay silent; used by do_GET / do_POST)."""

    @functools.wraps(method)
    def wrapped(self, *a, **k):
        t0 = time.perf_counter()
        try:
            method(self, *a, **k)
        finally:
            status = getattr(self, "_status", 0)
            wall = time.perf_counter() - t0
            if not getattr(self.server, "quiet", False):
                diag.info("http", f"{self.command} {self.path} -> {status} "
                                  f"{wall * 1000:.1f}ms")
    return wrapped


def make_handler(engine: InferenceEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            # Slow-loris guard: a stalled client must not pin a handler thread
            # (and its connection slot) forever. The per-connection socket
            # timeout is server-configurable (SLLM_SOCKET_TIMEOUT, default 60s);
            # None disables it.
            t = getattr(self.server, "socket_timeout", None)
            if t is not None:
                self.connection.settimeout(t)

        def handle_one_request(self):
            # Bounded concurrency: a saturated server returns 503 immediately
            # instead of spawning/holding unbounded handler threads. The
            # semaphore is held for the whole request (including an SSE stream)
            # so concurrent in-flight generations stay capped.
            sem = getattr(self.server, "conn_sem", None)
            if sem is not None and not sem.acquire(blocking=False):
                # The request line has not been parsed yet; give the handler the
                # attributes send_response()/log_request() need, then reject.
                self.requestline = ""
                self.command = ""
                self.request_version = "HTTP/1.1"
                self._send_json(503, _error(503, "server at capacity"),
                                close=True)
                self.close_connection = True
                self.wfile.flush()
                return
            try:
                super().handle_one_request()
            finally:
                if sem is not None:
                    sem.release()

        @property
        def engine(self) -> InferenceEngine:
            return self.server.engine

        @property
        def model_name(self) -> str:
            name = getattr(self.server, "model_name", None)
            if name:
                return name
            rec = getattr(self.engine.model, "recipe", None)
            return getattr(rec, "model_id", None) or "sllm"

        def log_message(self, fmt, *args):  # keep stderr quiet in tests
            if getattr(self.server, "quiet", False):
                return
            super().log_message(fmt, *args)

        def _send_json(self, status: int, obj: dict, close: bool = False):
            self._status = status
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if close:
                # we did not consume the request body (413 / bad length):
                # keep-alive would desync the next request on this socket,
                # so announce the close and drop the connection after the resp
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _send_saturated(self, exc):
            """429 + Retry-After (admission control said 'not now')."""
            self._status = 429
            body = json.dumps(_error(429, str(exc)), ensure_ascii=False).encode("utf-8")
            self.send_response(429)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Retry-After", str(getattr(exc, "retry_after", 1)))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self, events):
            """Write OpenAI-style SSE frames for an iterable of event dicts
            (Connection: close, so no chunked framing is needed on 1.1). An
            engine failure AFTER the headers are committed cannot change the
            status line, so it is reported as an in-band error frame followed
            by [DONE] instead of propagating to do_POST (which would try to
            send a JSON status on an already-started 200 body)."""
            self._status = 200
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for obj in events:
                    payload = json.dumps(obj, ensure_ascii=False)
                    self.wfile.write(("data: %s\n\n" % payload).encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return  # client went away; nothing left to write
            except Exception as exc:  # noqa: BLE001 - headers already sent
                diag.error("http", f"stream aborted: {type(exc).__name__}: {exc}")
                try:
                    err = {"error": {"message": f"stream error: {exc}",
                                     "type": "server_error", "code": "500"}}
                    self.wfile.write(
                        ("data: %s\n\n" % json.dumps(err, ensure_ascii=False))
                        .encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _stream_common(self, events, chat: bool, body):
            """Yield OpenAI SSE event dicts from a (delta, finish_reason)
            token stream; _send_sse writes each frame eagerly so the client
            sees tokens as they are produced."""
            mid = f"chatcmpl-{uuid.uuid4().hex}" if chat else \
                f"cmpl-{uuid.uuid4().hex}"
            obj_kind = ("chat.completion.chunk" if chat else "text_completion")
            model = body.get("model") or self.model_name
            created = int(time.time())
            first = True
            for delta, reason in events:
                if first:
                    if chat:
                        yield {"id": mid, "object": obj_kind,
                               "created": created, "model": model,
                               "choices": [{"index": 0,
                                            "delta": {"role": "assistant",
                                                      "content": ""},
                                            "finish_reason": None}]}
                    else:
                        yield {"id": mid, "object": obj_kind,
                               "created": created, "model": model,
                               "choices": [{"index": 0, "text": "",
                                            "finish_reason": None}]}
                    first = False
                if delta:
                    if chat:
                        yield {"id": mid, "object": obj_kind,
                               "created": created, "model": model,
                               "choices": [{"index": 0,
                                            "delta": {"content": delta},
                                            "finish_reason": None}]}
                    else:
                        yield {"id": mid, "object": obj_kind,
                               "created": created, "model": model,
                               "choices": [{"index": 0, "text": delta,
                                            "finish_reason": None}]}
                if reason is not None:
                    empty = {} if chat else ""
                    key = "delta" if chat else "text"
                    yield {"id": mid, "object": obj_kind, "created": created,
                           "model": model,
                           "choices": [{"index": 0, key: empty,
                                        "finish_reason": reason}]}

        @_http_logged
        def do_GET(self):
            # Match the path component only: ignore a query string and a
            # trailing slash (/health/, /v1/models?x=1).
            path = urlsplit(self.path).path
            path = path.rstrip("/") if path != "/" else "/"
            if path == "/health":
                self._send_json(200, {"status": "ok", "model": self.model_name})
                return
            if path == "/v1/models":
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": self.model_name, "object": "model",
                              "created": int(self.server.started_at),
                              "owned_by": "sllm"}]})
                return
            self._send_json(404, _error(404, "not found"))

        @_http_logged
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                # body length unknown -> cannot keep-alive (leftover bytes
                # would be parsed as the next request line)
                self._send_json(400, _error(400, "invalid Content-Length"),
                                close=True)
                return
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                self._send_json(411, _error(411, "chunked Transfer-Encoding "
                                                 "is not supported"), close=True)
                return
            if length <= 0:
                self._send_json(400, _error(400, "empty body"))
                return
            if length > MAX_BODY_BYTES:
                # do not read a hostile/oversized body; close so the unread
                # bytes cannot poison a reused keep-alive connection
                self._send_json(413, _error(413, f"body exceeds {MAX_BODY_BYTES} bytes"),
                                close=True)
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise InvalidRequestError("request body must be a JSON object")
                if body.get("stream"):
                    if self.path == "/v1/chat/completions":
                        events = self._chat_stream(body)
                    elif self.path == "/v1/completions":
                        events = self._completion_stream(body)
                    else:
                        self._send_json(404, _error(404, "not found"))
                        return
                    self._send_sse(self._stream_common(events, chat=(
                        self.path == "/v1/chat/completions"), body=body))
                    return
                if self.path == "/v1/chat/completions":
                    obj = self._chat_completion(body)
                elif self.path == "/v1/completions":
                    obj = self._completion(body)
                else:
                    self._send_json(404, _error(404, "not found"))
                    return
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, _error(400, f"invalid JSON: {exc}"))
                return
            except InvalidRequestError as exc:
                self._send_json(400, _error(400, str(exc)))
                return
            except SaturatedError as exc:
                self._send_saturated(exc)
                return
            except (KeyError, ValueError, TypeError) as exc:
                # plain model-side validation (prompt exceeds max context,
                # bad message shape) is still the caller's fault -> 400
                self._send_json(400, _error(400, str(exc)))
                return
            except Exception as exc:  # noqa: BLE001 - keep the server alive
                # RuntimeError (GPU-only decode failure, a checkpoint without
                # a chat template) and any engine/handler bug are SERVER-side:
                # 500, never a silent 400 or a dropped keep-alive connection
                diag.error("http", f"internal error: {type(exc).__name__}: {exc}")
                self._send_json(500, {"error": {
                    "message": f"internal error: {exc}",
                    "type": "server_error", "code": "500"}})
                return
            self._send_json(200, obj)

        def _common(self, body) -> dict:
            default_max = getattr(self.server, "default_max_new", 16)
            try:
                max_tokens = int(body.get("max_tokens", body.get("max_new",
                                                                 default_max)))
            except (TypeError, ValueError):
                raise InvalidRequestError("max_tokens must be an integer") from None
            if max_tokens < 1:
                raise InvalidRequestError("max_tokens must be >= 1")
            try:
                temperature = float(body.get("temperature", 0.0))
            except (TypeError, ValueError):
                raise InvalidRequestError("temperature must be a number") from None
            if temperature < 0:
                raise InvalidRequestError("temperature must be >= 0")
            top_p = body.get("top_p")
            top_k = body.get("top_k")
            seed = body.get("seed")
            try:
                return {
                    "max_new": max_tokens, "temperature": temperature,
                    "top_p": float(top_p) if top_p is not None else None,
                    "top_k": int(top_k) if top_k is not None else None,
                    "seed": int(seed) if seed is not None else None,
                }
            except (TypeError, ValueError):
                raise InvalidRequestError(
                    "top_p/top_k/seed must be numeric") from None

        def _chat_completion(self, body) -> dict:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise InvalidRequestError("messages must be a non-empty list")
            params = self._common(body)
            # engine.chat_detail renders the template ONCE and reports the
            # real token ledger (no re-encode guessing, real finish_reason)
            d = self.engine.chat_detail(messages,
                                        add_generation_prompt=body.get("add_generation_prompt", True),
                                        **params)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or self.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": d["text"]},
                             "finish_reason": d["finish_reason"]}],
                "usage": {"prompt_tokens": d["prompt_len"],
                          "completion_tokens": d["completion_len"],
                          "total_tokens": d["prompt_len"] + d["completion_len"]},
            }

        def _chat_stream(self, body):
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise InvalidRequestError("messages must be a non-empty list")
            params = self._common(body)  # has max_new, temperature, ...
            return self.engine.stream_chat(
                messages,
                add_generation_prompt=body.get("add_generation_prompt", True),
                **params)

        def _completion_stream(self, body):
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                raise InvalidRequestError("prompt must be a string")
            return self.engine.stream_complete(prompt, **self._common(body))

        def _completion(self, body) -> dict:
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                raise InvalidRequestError("prompt must be a string")
            params = self._common(body)
            d = self.engine.complete_detail(prompt, **params)
            return {
                "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
                "created": int(time.time()),
                "model": body.get("model") or self.model_name,
                "choices": [{"index": 0, "text": d["text"],
                             "finish_reason": d["finish_reason"]}],
                "usage": {"prompt_tokens": d["prompt_len"],
                          "completion_tokens": d["completion_len"],
                          "total_tokens": d["prompt_len"] + d["completion_len"]},
            }

    return Handler


def create_server(engine: InferenceEngine, host: str = "127.0.0.1",
                  port: int = 0, quiet: bool = True,
                  model_name: str | None = None,
                  default_max_new: int | None = None):
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    server.engine = engine
    server.quiet = quiet
    server.model_name = model_name
    if default_max_new is not None:
        server.default_max_new = int(default_max_new)
    server.started_at = time.time()
    # Bounded concurrent handlers (SLLM_MAX_CONNECTIONS, default 64) and a
    # per-connection socket timeout (SLLM_SOCKET_TIMEOUT, default 60s).
    server.conn_sem = threading.BoundedSemaphore(
        max(1, _env_int("SLLM_MAX_CONNECTIONS", 64)))
    server.socket_timeout = _env_int("SLLM_SOCKET_TIMEOUT", 60)
    return server, server.server_address[1]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="dev serving stub")
    ap.add_argument("--host", default=None,
                    help="bind address (default $SLLM_HOST / 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None,
                    help="bind port (default $SLLM_PORT / 8000)")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args(argv)

    from .dev_model import build_dev_engine

    host = args.host or _env("SLLM_HOST") or "127.0.0.1"
    port = args.port or _env_int("SLLM_PORT", 8000)
    engine = build_dev_engine()
    server, port = create_server(engine, host, port, quiet=False)
    engine.show_banner(tag="dev-stub")
    diag.info("dev-stub", f"listening http://{host}:{port} (tiny model, "
                          f"max_new={args.max_new})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
