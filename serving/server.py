"""OpenAI-compatible HTTP front-end (stdlib only).

Endpoints:
  GET  /health                    -> {"status": "ok", "model": ...}
  GET  /v1/models                 -> {"object": "list", "data": [{id, ...}]}
  POST /v1/completions            -> {prompt, max_tokens, temperature, ...}
  POST /v1/chat/completions       -> {messages, max_tokens, temperature, ...}

Responses follow the OpenAI schema (chat.completion / text_completion
objects, `usage` token counts, `finish_reason`). `stream: true` is rejected
explicitly (400) until SSE lands; a requested `model` different from the
served one is accepted permissively (single-model server). The model is
decoupled via `serving.executor.InferenceEngine`.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .executor import InferenceEngine


def _error(status: int, message: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": status}}


def make_handler(engine: InferenceEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

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

        def _send_json(self, status: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _usage(self, prompt_text: str, out_text: str) -> dict:
            """Token counts via the engine tokenizer (best effort: v1 has no
            incremental token ledger)."""
            try:
                pt = len(self.engine.tokenizer.encode(prompt_text))
                ct = len(self.engine.tokenizer.encode(out_text))
            except Exception:
                pt = ct = 0
            return {"prompt_tokens": pt, "completion_tokens": ct,
                    "total_tokens": pt + ct}

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "model": self.model_name})
                return
            if self.path == "/v1/models":
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": self.model_name, "object": "model",
                              "created": int(self.server.started_at),
                              "owned_by": "sllm"}]})
                return
            self._send_json(404, _error(404, "not found"))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self._send_json(400, _error(400, "empty body"))
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, _error(400, f"invalid JSON: {exc}"))
                return
            try:
                if body.get("stream"):
                    self._send_json(400, _error(400, "streaming not supported"))
                    return
                if self.path == "/v1/chat/completions":
                    obj = self._chat_completion(body)
                elif self.path == "/v1/completions":
                    obj = self._completion(body)
                else:
                    self._send_json(404, _error(404, "not found"))
                    return
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                # RuntimeError: tokenizer.apply_chat_template on dirs without
                # a chat template (tokenizer.py).
                self._send_json(400, _error(400, str(exc)))
                return
            self._send_json(200, obj)

        def _common(self, body) -> dict:
            default_max = getattr(self.server, "default_max_new", 16)
            max_tokens = int(body.get("max_tokens", body.get("max_new",
                                                             default_max)))
            temperature = float(body.get("temperature", 0.0))
            top_p = body.get("top_p")
            top_k = body.get("top_k")
            seed = body.get("seed")
            return {
                "max_new": max_tokens, "temperature": temperature,
                "top_p": float(top_p) if top_p is not None else None,
                "top_k": int(top_k) if top_k is not None else None,
                "seed": seed,
            }

        def _chat_completion(self, body) -> dict:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            params = self._common(body)
            content = self.engine.chat(messages, add_generation_prompt=body.get("add_generation_prompt", True), **params)
            prompt_text = ""
            try:
                prompt_text = self.engine.tokenizer.apply_chat_template(
                    messages)
            except Exception:
                pass
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or self.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": self._usage(prompt_text, content),
            }

        def _completion(self, body) -> dict:
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            params = self._common(body)
            text = self.engine.complete(prompt, **params)
            return {
                "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
                "created": int(time.time()),
                "model": body.get("model") or self.model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": self._usage(prompt, text),
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
    return server, server.server_address[1]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="dev serving stub")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args(argv)

    from .dev_model import build_dev_engine

    engine = build_dev_engine()
    server, port = create_server(engine, args.host, args.port, quiet=False)
    print(f"[dev-stub] listening http://{args.host}:{port} (tiny model, max_new={args.max_new})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
