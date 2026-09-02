"""ctypes bindings for the qwen4_exp kernel set (kernels/cuda/qwen4.cu).

Separate module from `_sllm_cuda` so a pre-A4 `sllm_gpu.so` (without the
`sllm_q4_*` symbols) keeps working unchanged: `available()` reports whether
the CURRENT build exports the q4 set, and every wrapper raises a clear
RuntimeError when it does not.

Host-pointer (transfer-era) API: parity surface for the numpy oracles
(tests/test_qwen4_kernels). Device-resident composition is C1.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char_p, c_float, c_int, c_long

import numpy as np

_SO = os.path.abspath(os.path.join(os.path.dirname(__file__), "cuda",
                                   "sllm_gpu.so"))

_Q4_ENTRIES = (
    "sllm_q4_grouped_gemma_rmsnorm", "sllm_q4_gemv_rows", "sllm_q4_hc_mix_apply",
    "sllm_q4_hc_combine", "sllm_q4_gemma_rmsnorm", "sllm_q4_rope_partial",
    "sllm_q4_qsa_pool_block", "sllm_q4_qsa_mqa_logits", "sllm_q4_qsa_topk",
    "sllm_q4_qsa_sparse_attn", "sllm_q4_moe_router", "sllm_q4_swiglu",
    "sllm_q4_axpy_rows", "sllm_q4_shared_gate_accum",
)

# device-pointer (C1) variants: the caller passes GPU pointers (from
# kernels._sllm_cuda.DeviceBuffer); no host malloc/copy. Kept as a separate
# symbol set so a pre-C1 .so still loads (available() reports the base set).
_Q4_DEV_ENTRIES = (
    "sllm_q4_grouped_gemma_rmsnorm_dev", "sllm_q4_gemv_rows_dev",
    "sllm_q4_hc_mix_apply_dev", "sllm_q4_hc_combine_dev",
    "sllm_q4_gemma_rmsnorm_dev", "sllm_q4_rope_partial_dev",
    "sllm_q4_qsa_pool_block_dev", "sllm_q4_qsa_mqa_logits_dev",
    "sllm_q4_qsa_topk_dev", "sllm_q4_qsa_sparse_attn_dev",
    "sllm_q4_moe_router_dev", "sllm_q4_swiglu_dev", "sllm_q4_axpy_rows_dev",
    "sllm_q4_shared_gate_accum_dev",
)

_g = None
_g_dev_ok = False
_lib_failed = False


def _setup():
    global _g, _lib_failed, _g_dev_ok
    if _g is not None or _lib_failed:
        return _g
    try:
        if not os.path.isfile(_SO):
            raise FileNotFoundError(_SO)
        lib = ctypes.CDLL(_SO)
        if not all(hasattr(lib, name) for name in _Q4_ENTRIES):
            raise AttributeError("sllm_q4_* symbols missing (rebuild "
                                 "kernels/cuda with qwen4.cu)")
        lib.sllm_q4_last_error.restype = c_char_p
        F, I, L, P = POINTER(c_float), c_int, c_long, POINTER(c_int)
        lib.sllm_q4_grouped_gemma_rmsnorm.argtypes = [F, F, F, I, I, I, c_float]
        lib.sllm_q4_gemv_rows.argtypes = [F, F, F, I, I, I, I, c_float]
        lib.sllm_q4_hc_mix_apply.argtypes = [F, F, F, I, I, I]
        lib.sllm_q4_hc_combine.argtypes = [F, F, F, F, I, I, I]
        lib.sllm_q4_gemma_rmsnorm.argtypes = [F, F, F, I, I, c_float]
        lib.sllm_q4_rope_partial.argtypes = [F, F, F, I, I, I]
        lib.sllm_q4_qsa_pool_block.argtypes = [F, F, I, I, I]
        lib.sllm_q4_qsa_mqa_logits.argtypes = [F, F, F, I, I, I, I, c_float]
        lib.sllm_q4_qsa_topk.argtypes = [F, P, I, I, I, I, I]
        lib.sllm_q4_qsa_sparse_attn.argtypes = [F, F, F, P, F, I, I, I, L, I,
                                                c_float]
        lib.sllm_q4_moe_router.argtypes = [F, F, P, I, I, I]
        lib.sllm_q4_swiglu.argtypes = [F, F, F, L]
        lib.sllm_q4_axpy_rows.argtypes = [F, F, F, I, I]
        lib.sllm_q4_shared_gate_accum.argtypes = [F, F, F, I, I]
        for n in _Q4_ENTRIES:
            getattr(lib, n).restype = c_int
        if all(hasattr(lib, name) for name in _Q4_DEV_ENTRIES):
            lib.sllm_q4_grouped_gemma_rmsnorm_dev.argtypes = [F, F, F, I, I, I,
                                                              c_float]
            lib.sllm_q4_gemv_rows_dev.argtypes = [F, F, F, I, I, I, I, c_float]
            lib.sllm_q4_hc_mix_apply_dev.argtypes = [F, F, F, I, I, I]
            lib.sllm_q4_hc_combine_dev.argtypes = [F, F, F, F, I, I, I]
            lib.sllm_q4_gemma_rmsnorm_dev.argtypes = [F, F, F, I, I, c_float]
            lib.sllm_q4_rope_partial_dev.argtypes = [F, F, F, I, I, I]
            lib.sllm_q4_qsa_pool_block_dev.argtypes = [F, F, I, I, I]
            lib.sllm_q4_qsa_mqa_logits_dev.argtypes = [F, F, F, I, I, I, I,
                                                       c_float]
            lib.sllm_q4_qsa_topk_dev.argtypes = [F, P, I, I, I, I, I]
            lib.sllm_q4_qsa_sparse_attn_dev.argtypes = [F, F, F, P, F, I, I, I,
                                                        L, I, c_float]
            lib.sllm_q4_moe_router_dev.argtypes = [F, F, P, I, I, I]
            lib.sllm_q4_swiglu_dev.argtypes = [F, F, F, L]
            lib.sllm_q4_axpy_rows_dev.argtypes = [F, F, F, I, I]
            lib.sllm_q4_shared_gate_accum_dev.argtypes = [F, F, F, I, I]
            for n in _Q4_DEV_ENTRIES:
                getattr(lib, n).restype = c_int
            _g_dev_ok = True
        else:
            _g_dev_ok = False
        _g = lib
    except Exception:
        _lib_failed = True
        _g = None
    return _g


def available() -> bool:
    return _setup() is not None


def dev_available() -> bool:
    """True when the loaded .so also exports the device-pointer (C1) variants."""
    return _setup() is not None and _g_dev_ok


def _lib():
    lib = _setup()
    if lib is None:
        raise RuntimeError(
            "qwen4 kernels unavailable: rebuild kernels/cuda (qwen4.cu) via "
            "kernels/cuda/build.sh and run on a CUDA box")
    return lib


def _err() -> str:
    try:
        return _lib().sllm_q4_last_error().decode()
    except Exception:
        return "q4 library unavailable"


def _check(rc: int) -> None:
    if rc != 0:
        raise RuntimeError(f"q4 kernel failed rc={rc}: {_err()}")


def _f(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _i(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def _fp(a):
    return _f(a).ctypes.data_as(POINTER(c_float))


def _ip(a):
    return _i(a).ctypes.data_as(POINTER(c_int))


# -- wrappers (host arrays in/out; keep the numpy-side temporaries alive) ----

def grouped_gemma_rmsnorm(x, w, hc, hs, eps=1e-6) -> np.ndarray:
    x, w = _f(x), _f(w)
    rows = int(np.prod(x.shape[:-1]))
    y = np.empty_like(x)
    _check(_lib().sllm_q4_grouped_gemma_rmsnorm(_fp(x), _fp(w),
                                                y.ctypes.data_as(POINTER(c_float)),
                                                rows, hc, hs, float(eps)))
    return y


def gemv_rows(A, W, post=0, scale=1.0) -> np.ndarray:
    A, W = _f(A), _f(W)
    rows, K = A.shape
    O = W.shape[0]
    out = np.empty((rows, O), np.float32)
    _check(_lib().sllm_q4_gemv_rows(_fp(A), _fp(W),
                                    out.ctypes.data_as(POINTER(c_float)),
                                    rows, K, O, int(post), float(scale)))
    return out


def hc_mix_apply(wgate, normed, hc, hs) -> np.ndarray:
    wgate, normed = _f(wgate), _f(normed)
    rows = wgate.shape[0]
    mixed = np.empty((rows, hs), np.float32)
    _check(_lib().sllm_q4_hc_mix_apply(_fp(wgate), _fp(normed),
                                       mixed.ctypes.data_as(POINTER(c_float)),
                                       rows, hc, hs))
    return mixed


def hc_combine(hyper, block_out, normed, inj_w, hc, hs) -> np.ndarray:
    hyper, block_out, normed, inj_w = map(_f, (hyper, block_out, normed, inj_w))
    rows = hyper.shape[0]
    out = hyper.copy()
    _check(_lib().sllm_q4_hc_combine(out.ctypes.data_as(POINTER(c_float)),
                                     _fp(block_out), _fp(normed), _fp(inj_w),
                                     rows, hc, hs))
    return out


def gemma_rmsnorm(x, w, eps=1e-6) -> np.ndarray:
    x, w = _f(x), _f(w)
    rows = int(np.prod(x.shape[:-1]))
    y = np.empty_like(x)
    _check(_lib().sllm_q4_gemma_rmsnorm(_fp(x), _fp(w),
                                        y.ctypes.data_as(POINTER(c_float)),
                                        rows, x.shape[-1], float(eps)))
    return y


def rope_partial(x, cos, sin, rot) -> np.ndarray:
    x, cos, sin = _f(x), _f(cos), _f(sin)
    rows = int(np.prod(x.shape[:-1]))
    out = x.copy()
    _check(_lib().sllm_q4_rope_partial(out.ctypes.data_as(POINTER(c_float)),
                                       _fp(cos), _fp(sin), rows, x.shape[-1],
                                       int(rot)))
    return out


def qsa_pool_block(tok_k, end, ratio) -> np.ndarray:
    tok_k = _f(tok_k).reshape(-1, tok_k.shape[-1])
    if not (0 < ratio <= end <= tok_k.shape[0]):
        raise ValueError(f"need 0 < ratio <= end <= {tok_k.shape[0]}")
    d = tok_k.shape[-1]
    out = np.empty(d, np.float32)
    _check(_lib().sllm_q4_qsa_pool_block(_fp(tok_k),
                                         out.ctypes.data_as(POINTER(c_float)),
                                         int(end), int(ratio), int(d)))
    return out


def qsa_mqa_logits(q, ck, start, end, scale) -> np.ndarray:
    q, ck = _f(q), _f(ck)
    nh, d = q.shape
    if not (0 <= start <= end <= ck.shape[0]):
        raise ValueError(f"bad range [{start},{end}) for {ck.shape[0]} rows")
    logits = np.full(ck.shape[0], -np.inf, np.float32)
    _check(_lib().sllm_q4_qsa_mqa_logits(_fp(q), _fp(ck),
                                         logits.ctypes.data_as(POINTER(c_float)),
                                         nh, d, int(start), int(end),
                                         float(scale)))
    return logits


def qsa_topk(logits, start, end, topk) -> np.ndarray:
    logits = _f(logits)
    m, stride = logits.shape
    if not (0 <= start < end <= stride):
        raise ValueError(f"bad range [{start},{end}) for stride {stride}")
    out = np.empty((m, topk), np.int32)
    rc = _lib().sllm_q4_qsa_topk(_fp(logits),
                                 out.ctypes.data_as(POINTER(c_int)),
                                 m, stride, int(start), int(end), int(topk))
    if rc == -2:
        raise RuntimeError("row range exceeds v1 smem topk (C1 radix kernel)")
    _check(rc)
    return out


def qsa_sparse_attn(q, k, v, slots, kvh, kcap, scale) -> np.ndarray:
    q, k, v, slots = _f(q), _f(k), _f(v), _i(slots)
    nh, hd = q.shape
    W = slots.shape[0]
    if k.shape != (kvh, kcap, hd) or v.shape != (kvh, kcap, hd):
        raise ValueError("k/v must be (kvh, kcap, hd) matching kcap")
    if W:
        smin, smax = int(slots.min()), int(slots.max())
        if smin < 0:
            raise ValueError(f"slot id negative ({smin}); uninitialised topk?")
        if smax >= kcap:
            raise ValueError("slot id exceeds kcap")
    out = np.empty((nh, hd), np.float32)
    rc = _lib().sllm_q4_qsa_sparse_attn(
        _fp(q), _fp(k), _fp(v), _ip(slots),
        out.ctypes.data_as(POINTER(c_float)),
        nh, kvh, hd, int(kcap), W, float(scale))
    if rc == -2:
        raise RuntimeError("slot width exceeds v1 smem bound")
    _check(rc)
    return out


def moe_router(logits, topk):
    logits = _f(logits)
    n, E = logits.shape
    w = np.empty((n, topk), np.float32)
    ids = np.empty((n, topk), np.int32)
    _check(_lib().sllm_q4_moe_router(_fp(logits),
                                     w.ctypes.data_as(POINTER(c_float)),
                                     ids.ctypes.data_as(POINTER(c_int)),
                                     n, E, int(topk)))
    return w, ids


def swiglu(g, u) -> np.ndarray:
    g, u = _f(g), _f(u)
    out = np.empty_like(g)
    _check(_lib().sllm_q4_swiglu(_fp(g), _fp(u),
                                 out.ctypes.data_as(POINTER(c_float)),
                                 int(g.size)))
    return out


def axpy_rows(out, y, w) -> np.ndarray:
    out, y, w = _f(out).copy(), _f(y), _f(w)
    _check(_lib().sllm_q4_axpy_rows(out.ctypes.data_as(POINTER(c_float)),
                                    _fp(y), _fp(w), out.shape[0],
                                    out.shape[1]))
    return out


def shared_gate_accum(out, shared, g) -> np.ndarray:
    out, shared, g = _f(out).copy(), _f(shared), _f(g)
    _check(_lib().sllm_q4_shared_gate_accum(
        out.ctypes.data_as(POINTER(c_float)), _fp(shared), _fp(g),
        out.shape[0], out.shape[1]))
    return out


# ---------------------------------------------------------------------------
# device-resident (C1) variants: take kernels._sllm_cuda.DeviceBuffer /
# DeviceView (use `.ptr`) or a raw ctypes pointer. No host copies -- the
# caller owns the buffers and the single ck.sync() per step.
# ---------------------------------------------------------------------------

def _dev(lib, name):
    fn = getattr(lib, name, None)
    if fn is None:
        raise RuntimeError(f"q4 device kernel {name} missing from sllm_gpu.so "
                           f"(rebuild kernels/cuda with the C1 qwen4.cu)")
    return fn


def _ptr(buf):
    """Resolve a DeviceBuffer/DeviceView (or raw pointer) to a ctypes pointer."""
    p = getattr(buf, "ptr", buf)
    return p


def grouped_gemma_rmsnorm_dev(x, w, y, hc, hs, eps=1e-6):
    rows = int(np.prod(x.shape[:-1])) if not isinstance(x, ctypes.c_void_p) \
        else 0
    _check(_dev(_lib(), "sllm_q4_grouped_gemma_rmsnorm_dev")(
        _ptr(x), _ptr(w), _ptr(y), rows, hc, hs, float(eps)))


def gemv_rows_dev(A, W, out, rows, K, O, post=0, scale=1.0):
    _check(_dev(_lib(), "sllm_q4_gemv_rows_dev")(_ptr(A), _ptr(W), _ptr(out),
                                                 rows, K, O, int(post),
                                                 float(scale)))


def hc_mix_apply_dev(wgate, normed, mixed, rows, hc, hs):
    _check(_dev(_lib(), "sllm_q4_hc_mix_apply_dev")(_ptr(wgate), _ptr(normed),
                                                    _ptr(mixed), rows, hc, hs))


def hc_combine_dev(hyper, block_out, normed, inj_w, rows, hc, hs):
    _check(_dev(_lib(), "sllm_q4_hc_combine_dev")(_ptr(hyper), _ptr(block_out),
                                                  _ptr(normed), _ptr(inj_w),
                                                  rows, hc, hs))


def gemma_rmsnorm_dev(x, w, y, rows, d, eps=1e-6):
    _check(_dev(_lib(), "sllm_q4_gemma_rmsnorm_dev")(_ptr(x), _ptr(w), _ptr(y),
                                                     rows, d, float(eps)))


def rope_partial_dev(x, cos, sin, rows, d, rot):
    _check(_dev(_lib(), "sllm_q4_rope_partial_dev")(_ptr(x), _ptr(cos),
                                                    _ptr(sin), rows, d,
                                                    int(rot)))


def qsa_pool_block_dev(tok_k, out, end, ratio, d):
    _check(_dev(_lib(), "sllm_q4_qsa_pool_block_dev")(_ptr(tok_k), _ptr(out),
                                                      int(end), int(ratio),
                                                      int(d)))


def qsa_mqa_logits_dev(q, ck, logits, nh, d, start, end, scale):
    _check(_dev(_lib(), "sllm_q4_qsa_mqa_logits_dev")(_ptr(q), _ptr(ck),
                                                      _ptr(logits), nh, d,
                                                      int(start), int(end),
                                                      float(scale)))


def qsa_topk_dev(logits, out, m, stride, start, end, topk):
    rc = _dev(_lib(), "sllm_q4_qsa_topk_dev")(_ptr(logits), _ptr(out), m,
                                              stride, int(start), int(end),
                                              int(topk))
    if rc == -2:
        raise RuntimeError("row range exceeds v1 smem topk (C1 radix kernel)")
    _check(rc)


def qsa_sparse_attn_dev(q, k, v, slots, out, nh, kvh, hd, kcap, W, scale):
    rc = _dev(_lib(), "sllm_q4_qsa_sparse_attn_dev")(_ptr(q), _ptr(k), _ptr(v),
                                                     _ptr(slots), _ptr(out),
                                                     nh, kvh, hd, int(kcap), W,
                                                     float(scale))
    if rc == -2:
        raise RuntimeError("slot width exceeds v1 smem bound")
    _check(rc)


def moe_router_dev(logits, w_out, id_out, n, E, topk):
    _check(_dev(_lib(), "sllm_q4_moe_router_dev")(_ptr(logits), _ptr(w_out),
                                                  _ptr(id_out), n, E,
                                                  int(topk)))


def swiglu_dev(g, u, out, n):
    _check(_dev(_lib(), "sllm_q4_swiglu_dev")(_ptr(g), _ptr(u), _ptr(out),
                                              int(n)))


def axpy_rows_dev(out, y, w, n, H):
    _check(_dev(_lib(), "sllm_q4_axpy_rows_dev")(_ptr(out), _ptr(y), _ptr(w),
                                                 n, H))


def shared_gate_accum_dev(out, shared, g, n, H):
    _check(_dev(_lib(), "sllm_q4_shared_gate_accum_dev")(_ptr(out),
                                                         _ptr(shared), _ptr(g),
                                                         n, H))
