"""vLLM-style diagnostic / messaging layer for the sLLM engines.

A single module-level `diag` object writes tag-prefixed, level-gated messages
to STDERR, so stdout (CLI responses) and HTTP/SSE bodies stay clean. This is
the one channel every engine and the CLI/server wrappers share, which is what
keeps "verbose, uniform, multi-model debug output" consistent across future
arches.

Levels (SLLM_LOG_LEVEL env, `sllm ... --log-level`):
    TRACE  per-op/per-token detail
    DEBUG  per-token ids + arch card detail + fallback diagnostics
    INFO   engine startup banner + per-request stats (DEFAULT)
    WARNING  fallbacks, degraded paths, recoverable errors
    ERROR  request/runtime failures

Usage:
    from serving import diag
    diag.info("sllm", f"serving on {host}:{port}")
    with diag.stopwatch() as sw:
        out = engine.chat(...)
    diag.info("sllm", f"out={len(out)} tok in={sw.elapsed:.3f}s "
                      f"({diag.tps(len(out), sw.elapsed)})")
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time

_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_NAMES = {v: k for k, v in _LEVELS.items()}

_level = _LEVELS.get(os.environ.get("SLLM_LOG_LEVEL", "").upper(),
                     _LEVELS["INFO"])
_quiet = False
# None = write to sys.stderr AT EMIT TIME (so contextlib.redirect_stderr and
# a swapped sys.stderr are honored); `capture()` swaps this to a buffer while
# active (thread-safe: the target lives on this module).
_stream = None


def set_level(name: str | None) -> None:
    """Override the level (env takes precedence unless name is explicit)."""
    global _level
    if name is not None and str(name).upper() in _LEVELS:
        _level = _LEVELS[str(name).upper()]


def get_level() -> str:
    return _NAMES.get(_level, "INFO")


def set_quiet(quiet: bool) -> None:
    """Global mute (used by the HTTP server + tests)."""
    global _quiet
    _quiet = bool(quiet)


def is_enabled(level: str) -> bool:
    return not _quiet and _level <= _LEVELS.get(level.upper(), _level)


@contextlib.contextmanager
def capture(stream=None):
    """Route diagnostics into `stream` (default a fresh io.StringIO) for the
    duration; THREAD-SAFE (the target lives on this module, so handler
    threads keep writing to the same capture even after the capturing
    thread's sys.stderr redirect would have been torn down). Restores after.

        with diag.capture() as buf:
            ...
        assert "..." in buf.getvalue()
    """
    global _stream
    prev = _stream
    _stream = stream if stream is not None else io.StringIO()
    try:
        yield _stream
    finally:
        _stream = prev


def _emit(level: str, tag: str, msg: str) -> None:
    if not is_enabled(level):
        return
    ts = time.strftime("%H:%M:%S")
    head = f"{level:7s}[{tag}]"
    stream = _stream if _stream is not None else sys.stderr
    try:
        stream.write(f"{ts} {head} {msg}\n")
        stream.flush()
    except Exception:  # noqa: BLE001 - diagnostics must never crash the engine
        pass


def trace(tag: str, msg: str) -> None:
    _emit("TRACE", tag, msg)


def debug(tag: str, msg: str) -> None:
    _emit("DEBUG", tag, msg)


def info(tag: str, msg: str) -> None:
    _emit("INFO", tag, msg)


def warn(tag: str, msg: str) -> None:
    _emit("WARNING", tag, msg)


def error(tag: str, msg: str) -> None:
    _emit("ERROR", tag, msg)


def banner(tag: str, title: str, lines) -> None:
    """A vLLM-style startup block: title + indented, `key + pad + value`
    lines, each emitted on its own INFO line prefixed with `[tag]`."""
    if not is_enabled("INFO"):
        return
    info(tag, "=" * 64)
    info(tag, f"{title}")
    info(tag, "-" * 64)
    for ln in lines:
        info(tag, f"  {ln}")
    info(tag, "=" * 64)


class stopwatch:
    """Context manager: `with diag.stopwatch() as sw: ... ; sw.elapsed`."""

    def __init__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.0

    def __enter__(self):
        return self

    def reset(self) -> "stopwatch":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False


def tps(tokens: int, wall: float) -> str:
    if wall <= 0:
        return "0.0 tokens/s"
    return f"{tokens / wall:.1f} tokens/s"


def gib(nbytes) -> str:
    return f"{nbytes / (2 ** 30):.3g} GiB"
