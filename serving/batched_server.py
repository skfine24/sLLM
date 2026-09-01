"""Continuous-batching HTTP serving facade.

Wraps a single-shot `InferenceEngine` behind the SAME method surface the
stdlib server (`serving.server`) already calls (`chat_detail` /
`complete_detail` / `stream_chat` / `stream_complete` / `describe` /
`show_banner` / `model` / `tokenizer`), but drives generation through the
`BatchedInferenceEngine` + `runtime.scheduler` continuous-batching loop.

Why this exists (design fix #5 + #6): the plain HTTP server spawns one thread
per connection and ran a full generation inline on the SHARED model with no
admission control -- unbounded concurrent generations (memory blow-up, GPU
device-resident double-upload/OOM). Here:

  * ONE daemon executor thread owns ALL model access (batched decode steps
    and exclusive multimodal/tool jobs), so the non-thread-safe oracle and its
    lazily built GPU weight table are never touched concurrently (fixes #6
    structurally; the model-side `_init_lock` is defense-in-depth).
  * Admission control is bounded by the recipe's `KVMemoryPlan`: a prompt that
    can never fit the KV budget is a 400; the server being at capacity is an
    immediate 429 (`SaturatedError` + Retry-After).
  * Per-token SSE streaming is preserved: text requests are batched and stream
    decoded deltas from the executor thread via a per-request queue.

Multimodal (vision) and tool-loop turns are NOT modeled by the block/state
scheduler, so they run as EXCLUSIVE jobs: the executor thread runs the inner
engine's single-shot path alone (between batched steps), still one thread on
the model.
"""

from __future__ import annotations

import math
import os
import queue
import threading

from env_config import get_int as _env_int

from .server import InvalidRequestError, SaturatedError

_HARD_MAX_BLOCKS = 200_000  # allocator list(range(cap)) memory guard


def _blocks_for(tokens: int, block_size: int) -> int:
    return (tokens + block_size - 1) // block_size if tokens > 0 else 0


class _Req:
    """One in-flight serving request: the scheduler sink + the handshake with
    the HTTP handler thread (queue for stream deltas, Event for completion)."""

    __slots__ = ("facade", "stream", "prompt_ids", "prompt_len", "q", "done",
                 "ids", "reason", "error", "result", "seq_id", "_released")

    def __init__(self, facade, stream: bool, prompt_ids: list[int] | None):
        self.facade = facade
        self.stream = stream
        self.prompt_ids = prompt_ids
        self.prompt_len = len(prompt_ids) if prompt_ids is not None else 0
        self.q: "queue.Queue" = queue.Queue()
        self.done = threading.Event()
        self.ids: list[int] = []
        self.reason: str | None = None
        self.error: BaseException | None = None
        self.result: dict | None = None
        self.seq_id: int | None = None
        self._released = False

    # -- scheduler sink API (called from the executor thread) --------------
    def on_token(self, tok: int) -> None:
        self.ids.append(tok)
        if self.stream:
            self.q.put(("tok_id", tok))

    def on_finish(self, reason: str) -> None:
        if self.done.is_set():
            return
        self.reason = reason
        if self.stream:
            self.q.put(("finish", reason))
        self.done.set()
        self.facade._release(self)

    def fail(self, exc: BaseException) -> None:
        if self.done.is_set():
            return
        self.error = exc
        if self.stream:
            self.q.put(("error", exc))
        self.done.set()
        self.facade._release(self)

    def _text_delta(self, gen_ids: list[int], last_full: list[str]) -> str:
        full = self.facade.tokenizer.decode(gen_ids)
        d = full[len(last_full[0]):]
        last_full[0] = full
        return d


class _Job:
    """An exclusive (multimodal / tool-loop) request run alone by the executor."""

    __slots__ = ("req", "func", "args", "kwargs")

    def __init__(self, req: _Req, func, args, kwargs):
        self.req = req
        self.func = func
        self.args = args
        self.kwargs = kwargs


class BatchedServingEngine:
    """Serving facade; construct AFTER the inner engine's banner (it warms the
    oracle/template lazily). Start/stop manage the executor thread; it is a
    daemon so a KeyboardInterrupt in `serve_forever` exits cleanly."""

    def __init__(self, engine, *, kv_capacity=None, state_capacity=None,
                 block_size=None, chunk_size=None, max_concurrency=None,
                 max_queued=None, req_timeout=None):
        self.engine = engine
        self.model = engine.model
        self.tokenizer = engine.tokenizer
        recipe = getattr(engine.model, "recipe", None)
        self.recipe = recipe

        # --- sizing: recipe KVMemoryPlan first, env override, hard caps ------
        block_size = int(block_size or _env_int("SLLM_BLOCK_SIZE", 16))
        chunk_size = int(chunk_size or _env_int("SLLM_CHUNK_SIZE", 32))
        max_concurrency = int(max_concurrency
                              or _env_int("SLLM_MAX_CONCURRENT", 8))
        if max_concurrency < 1:
            max_concurrency = 1
        if block_size < 1:
            block_size = 16
        if chunk_size < 1:
            chunk_size = 32
        state_capacity = int(state_capacity
                             or _env_int("SLLM_STATE_CAPACITY", max_concurrency))
        if state_capacity < 1:
            state_capacity = max_concurrency

        max_ctx = self._max_context()
        plan = self._kv_plan(recipe, block_size)
        # serving need: enough blocks for max_concurrency seqs of max_ctx, but
        # never more than the real budget the plan exposes; capped for the
        # allocator's list(range(cap)) footprint.
        need = _blocks_for(max_ctx, block_size) * max_concurrency
        if kv_capacity is None:
            if plan is not None and getattr(plan, "num_blocks", 0) > 0:
                kv_capacity = int(min(plan.num_blocks, need))
            else:
                kv_capacity = int(need)
        kv_capacity = max(1, min(int(kv_capacity), _HARD_MAX_BLOCKS))

        self.block_size = block_size
        self.kv_capacity = kv_capacity
        self.capacity_tokens = kv_capacity * block_size
        self.max_context = max_ctx
        self.max_concurrency = max_concurrency
        if max_queued is None:
            max_queued = _env_int("SLLM_MAX_QUEUED", max_concurrency * 4)
        self.max_queued = max(1, int(max_queued))
        self.req_timeout = float(req_timeout
                                 or _env_int("SLLM_SERVE_TIMEOUT", 600))

        from .executor import BatchedInferenceEngine
        self.batched = BatchedInferenceEngine(
            self.model, self.tokenizer, kv_capacity=kv_capacity,
            state_capacity=state_capacity, block_size=block_size,
            chunk_size=chunk_size, max_concurrency=max_concurrency)

        # --- executor-thread communication ---------------------------------
        self._submit_q: "queue.Queue" = queue.Queue()
        self._excl_q: "queue.Queue" = queue.Queue()
        self._abort_q: "queue.Queue" = queue.Queue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._inflight = 0
        self._served = 0
        self._warm()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="sllm-batched")
        self._thread.start()

    # -- engine interface the server needs ---------------------------------
    def describe(self):
        return self.engine.describe()

    def show_banner(self, tag: str = "sllm") -> None:
        self.engine.show_banner(tag=tag)
        from serving import diag
        diag.info(tag, f"serving mode: continuous batching "
                       f"(kv_capacity={self.kv_capacity} blocks x "
                       f"{self.block_size} tok, max_concurrency="
                       f"{self.max_concurrency}, max_queued={self.max_queued})")

    def summary(self) -> str:
        return (f"batched(max_concurrency={self.max_concurrency}, "
                f"max_queued={self.max_queued}, "
                f"kv_capacity={self.kv_capacity} blocks)")

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    # -- helpers -----------------------------------------------------------
    def _max_context(self) -> int:
        mc = getattr(self.model, "max_context", None)
        if callable(mc):
            try:
                return int(mc())
            except Exception:  # noqa: BLE001 - fall back to the recipe field
                pass
        return int(getattr(self.model, "recipe", None) and
                   getattr(self.recipe, "max_position_embeddings", 4096) or 4096)

    def _kv_plan(self, recipe, block_size):
        if recipe is None:
            return None
        try:
            from runtime.placement import KVMemoryPlan
            return KVMemoryPlan.from_recipe(recipe, block_size=block_size)
        except Exception:  # noqa: BLE001 - sizing heuristic only; use env caps
            return None

    def _warm(self) -> None:
        """Touch the lazily built shared objects on THIS (pre-thread) main
        thread so the executor never races a half-built oracle / jinja env."""
        # GPU availability + dsv4 oracle + q4 cfg may initialize here
        try:
            self.model._gpu_available()
        except Exception:  # noqa: BLE001
            pass
        try:  # build the chat-template jinja env once (concurrent renders
            # then only read the cache)
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}], add_generation_prompt=True)
        except Exception:  # noqa: BLE001 - a checkpoint without a template is
            pass           # a per-request error, not a startup failure

    # -- admission ---------------------------------------------------------
    def _admit(self) -> None:
        """Reserve one in-flight slot or raise 429 (immediate, no queueing)."""
        with self._lock:
            if self._inflight >= self.max_queued:
                raise SaturatedError(
                    f"server at capacity ({self._inflight} in flight, "
                    f"limit {self.max_queued})", retry_after=1)
            self._inflight += 1

    def _release(self, req: _Req) -> None:
        if req._released:
            return
        req._released = True
        with self._lock:
            self._inflight -= 1
            self._served += 1

    def _precheck(self, prompt_len: int, max_new: int) -> None:
        if max_new < 1:
            raise InvalidRequestError("max_tokens must be >= 1")
        if self.max_context and prompt_len + max_new > self.max_context:
            raise InvalidRequestError(
                f"prompt ({prompt_len}) + max_tokens ({max_new}) exceeds "
                f"max_position_embeddings ({self.max_context})")
        worst = _blocks_for(prompt_len + max_new, self.block_size)
        if worst > self.kv_capacity:
            raise InvalidRequestError(
                f"prompt ({prompt_len}) + max_tokens ({max_new}) needs "
                f"{worst} KV blocks > capacity {self.kv_capacity}; shorten the "
                f"prompt or lower max_tokens")

    # -- HTTP-facing API (same surface as InferenceEngine) -----------------
    def complete_detail(self, prompt, max_new=16, temperature=0.0, top_k=None,
                        top_p=None, seed=None, repetition_penalty=None) -> dict:
        ids = self.tokenizer.encode(prompt)
        self._precheck(len(ids), max_new)
        self._admit()
        try:
            req = _Req(self, stream=False, prompt_ids=ids)
            self._submit_q.put((req, dict(
                max_new=max_new, temperature=temperature, top_k=top_k,
                top_p=top_p, seed=seed, repetition_penalty=repetition_penalty)))
            self._wake.set()
            if not req.done.wait(self.req_timeout):
                self._request_abort(req)
                raise TimeoutError(f"generation timed out after {self.req_timeout}s")
            if req.error is not None:
                raise req.error
            return {"text": self.tokenizer.decode(req.ids),
                    "finish_reason": req.reason or "length",
                    "prompt_len": req.prompt_len,
                    "completion_len": len(req.ids)}
        except Exception:
            self._release(req)
            raise

    def stream_complete(self, prompt, max_new=16, temperature=0.0, top_k=None,
                        top_p=None, seed=None, repetition_penalty=None):
        ids = self.tokenizer.encode(prompt)
        self._precheck(len(ids), max_new)
        self._admit()
        try:
            req = _Req(self, stream=True, prompt_ids=ids)
            self._submit_q.put((req, dict(
                max_new=max_new, temperature=temperature, top_k=top_k,
                top_p=top_p, seed=seed, repetition_penalty=repetition_penalty)))
            self._wake.set()
        except Exception:
            self._release(req)
            raise
        return self._iter_batched_stream(req)

    def chat_detail(self, messages, add_generation_prompt=True, max_new=16,
                    temperature=0.0, top_k=None, top_p=None, seed=None,
                    repetition_penalty=None) -> dict:
        if self._is_vision(messages):
            return self._exclusive(
                self.engine.chat_detail, messages,
                add_generation_prompt=add_generation_prompt, max_new=max_new,
                temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
                repetition_penalty=repetition_penalty)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt)
        d = self.complete_detail(
            prompt_text, max_new=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, seed=seed,
            repetition_penalty=repetition_penalty)
        d["prompt_text"] = prompt_text
        return d

    def stream_chat(self, messages, add_generation_prompt=True, max_new=16,
                    temperature=0.0, top_k=None, top_p=None, seed=None,
                    repetition_penalty=None):
        if self._is_vision(messages):
            return self._exclusive_stream(
                self.engine.stream_chat, messages,
                add_generation_prompt=add_generation_prompt, max_new=max_new,
                temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
                repetition_penalty=repetition_penalty)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt)
        return self.stream_complete(
            prompt_text, max_new=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, seed=seed,
            repetition_penalty=repetition_penalty)

    # -- multimodal / exclusive path ---------------------------------------
    def _is_vision(self, messages) -> bool:
        try:
            return bool(self.engine._messages_have_images(messages)
                        and self.engine._model_has_vision())
        except Exception:  # noqa: BLE001 - never route a text turn to vision
            return False

    def _exclusive(self, func, *args, **kwargs) -> dict:
        self._admit()
        try:
            req = _Req(self, stream=False, prompt_ids=None)
            self._excl_q.put(_Job(req, func, args, kwargs))
            self._wake.set()
            if not req.done.wait(self.req_timeout):
                raise TimeoutError(f"generation timed out after {self.req_timeout}s")
            if req.error is not None:
                raise req.error
            return req.result
        except Exception:
            self._release(req)
            raise

    def _exclusive_stream(self, func, *args, **kwargs):
        self._admit()
        try:
            req = _Req(self, stream=True, prompt_ids=None)
            self._excl_q.put(_Job(req, func, args, kwargs))
            self._wake.set()
        except Exception:
            self._release(req)
            raise
        return self._iter_text_stream(req)

    # -- HTTP-side stream generators ---------------------------------------
    def _iter_batched_stream(self, req: _Req):
        """Drain raw token ids from the executor thread; decode deltas here so
        the executor stays lean and per-connection decoding is parallel."""
        gen_ids: list[int] = []
        last = [""]
        try:
            while True:
                try:
                    ev = req.q.get(timeout=self.req_timeout)
                except queue.Empty:
                    self._request_abort(req)
                    raise TimeoutError(
                        f"stream stalled after {self.req_timeout}s") from None
                kind = ev[0]
                if kind == "tok_id":
                    gen_ids.append(ev[1])
                    d = req._text_delta(gen_ids, last)
                    if d:
                        yield d, None
                elif kind == "finish":
                    yield "", ev[1]
                    return
                elif kind == "error":
                    raise ev[1]
        finally:
            # client disconnected (GeneratorExit) or error before finish: free
            # the scheduler blocks for a seq that is no longer being read
            if not req.done.is_set():
                self._request_abort(req)
                self._release(req)

    def _iter_text_stream(self, req: _Req):
        """Exclusive (vision) stream: the executor already decoded deltas."""
        try:
            while True:
                try:
                    ev = req.q.get(timeout=self.req_timeout)
                except queue.Empty:
                    raise TimeoutError(
                        f"stream stalled after {self.req_timeout}s") from None
                kind = ev[0]
                if kind == "tok_text":
                    if ev[1]:
                        yield ev[1], None
                elif kind == "finish":
                    yield "", ev[1]
                    return
                elif kind == "error":
                    raise ev[1]
        finally:
            if not req.done.is_set():
                self._release(req)

    def _request_abort(self, req: _Req) -> None:
        if req.seq_id is not None and not req.done.is_set():
            self._abort_q.put(req.seq_id)
            self._wake.set()

    # -- executor thread (single owner of the model) -----------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            got = False
            self._drain_aborts()
            got |= self._drain_submits()
            got |= self._drain_exclusive()
            if self.batched.queue_length > 0:
                try:
                    self.batched.step()
                except Exception as exc:  # noqa: BLE001 - fail the in-flight batch
                    self._fail_all_batched(exc)
                self.batched.reap()
                got = True
            if not got:
                self._wake.wait(0.01)
                self._wake.clear()

    def _drain_aborts(self) -> bool:
        n = 0
        while True:
            try:
                sid = self._abort_q.get_nowait()
            except queue.Empty:
                break
            self.batched.abort(sid)
            n += 1
        if n:
            self.batched.reap()
        return bool(n)

    def _drain_submits(self) -> bool:
        n = 0
        while True:
            try:
                req, params = self._submit_q.get_nowait()
            except queue.Empty:
                break
            try:
                req.seq_id = self.batched.submit(
                    "", max_new=params["max_new"],
                    temperature=params["temperature"], top_k=params["top_k"],
                    top_p=params["top_p"], seed=params["seed"],
                    repetition_penalty=params["repetition_penalty"],
                    sink=req, prompt_ids=req.prompt_ids)
            except Exception as exc:  # noqa: BLE001 - surface on this request
                req.fail(exc)
            n += 1
        return bool(n)

    def _drain_exclusive(self) -> bool:
        n = 0
        while True:
            try:
                job = self._excl_q.get_nowait()
            except queue.Empty:
                break
            self._run_job(job)
            n += 1
        return bool(n)

    def _run_job(self, job: _Job) -> None:
        req = job.req
        try:
            out = job.func(*job.args, **job.kwargs)
            if req.stream:
                last_reason = None
                for delta, reason in out:
                    if delta:
                        req.q.put(("tok_text", delta))
                    if reason is not None:
                        last_reason = reason
                req.on_finish(last_reason if last_reason is not None else "length")
            else:
                req.result = out
                req.reason = (out or {}).get("finish_reason", "stop") \
                    if isinstance(out, dict) else "stop"
                req.done.set()
                self._release(req)
        except Exception as exc:  # noqa: BLE001
            req.fail(exc)

    def _fail_all_batched(self, exc: BaseException) -> None:
        # A persistent step failure (e.g. a GPU decode fault) must not spin:
        # fail every in-flight request, then abort their scheduler slots so
        # queue_length drops to 0 and the executor stops stepping the batch.
        for sid in list(self.batched._seqs.keys()):
            info = self.batched._seqs.get(sid) or {}
            sink = info.get("sink")
            if isinstance(sink, _Req):
                sink.fail(exc)
        for sid in list(self.batched._seqs.keys()):
            self.batched.abort(sid)
        self.batched.reap()


def wrap_for_serving(engine, **kwargs):
    """Return a batched facade for real engines; leave dev/test single-shot
    stubs and already-wrapped facades untouched. Controlled by
    SLLM_SERVE_MODE (single|batch; default batch)."""
    mode = (os.environ.get("SLLM_SERVE_MODE") or "batch").strip().lower()
    if mode in ("single", "direct", "off", "0", "legacy") or isinstance(
            engine, BatchedServingEngine):
        return engine
    try:
        return BatchedServingEngine(engine, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a bad batch config must not
        from serving import diag    # prevent serving; fall back to direct
        diag.warn("sllm", f"continuous batching unavailable ({exc}); "
                          f"serving single-shot")
        return engine
