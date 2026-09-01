"""ctypes bindings for the DeepSeek-V4 kernel set (kernels/cuda/deepseek.cu).

Same lazy-load contract as `_q4_cuda`: `available()` reports whether the
built `sllm_gpu.so` exports the `sllm_ds_*` symbols, and every wrapper falls
back to the numpy oracle math when the .so is absent (so the serving engine
and tests run unchanged on the dev box).

Current surface (phase 5, cluster-compiled):
  * `mla_sparse_attn`  - the MLA decode sparse attention (window + compressed
                         rows + indexer gather + learnable sink); numpy
                         fallback = ref.deepseek_v4.sparse_attn.
  * `fp4_moe_linear`   - one fp4-expert linear: x (M,K) x packed (N,K/2) with
                         per-row-32 E8M0 scale -> (M,N) fp32; numpy fallback =
                         loaders.fp8.dequant_fp4_packed_weight + matmul.

The .cu kernels are host-transfer parity stubs first (correct, slow) so the
cluster build proves wiring before any fused rewrite (C1/C2).
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char_p, c_float, c_int

import numpy as np

_SO = os.path.abspath(os.path.join(os.path.dirname(__file__), "cuda",
                                   "sllm_gpu.so"))

_DS_ENTRIES = ("sllm_ds_fp4_gemm", "sllm_ds_mla_sparse_attn",
               "sllm_ds_last_error")

_g = None
_lib_failed = False


def _setup():
    global _g, _lib_failed
    if _g is not None or _lib_failed:
        return _g
    try:
        if not os.path.isfile(_SO):
            raise FileNotFoundError(_SO)
        lib = ctypes.CDLL(_SO)
        if not all(hasattr(lib, n) for n in _DS_ENTRIES):
            raise AttributeError("sllm_ds_* symbols missing (rebuild "
                                 "kernels/cuda with deepseek.cu)")
        lib.sllm_ds_last_error.restype = c_char_p
        F, I = POINTER(c_float), c_int
        lib.sllm_ds_fp4_gemm.argtypes = [F, F, F, I, I, I, c_float, F]
        lib.sllm_ds_mla_sparse_attn.argtypes = [
            F, F, F, POINTER(c_int), F, I, I, I, c_float, F]
        for n in _DS_ENTRIES:
            getattr(lib, n).restype = c_int
        _g = lib
    except Exception:
        _lib_failed = True
        _g = None
    return _g


def available() -> bool:
    return _setup() is not None


def _err() -> str:
    try:
        return _setup().sllm_ds_last_error().decode()
    except Exception:
        return "deepseek kernels unavailable (numpy fallback active)"


def _fp(a) -> POINTER(c_float):
    return np.ascontiguousarray(a, dtype=np.float32).ctypes.data_as(
        POINTER(c_float))


def fp4_moe_linear(x: np.ndarray, packed: np.ndarray, scale_u8: np.ndarray,
                   fp4_block_w: int = 32) -> np.ndarray:
    """One fp4-expert linear: x (M, K) @ W^T with W packed (N, K//2) E2M1 +
    (N, K//fp4_block_w) E8M0 scales -> (M, N) fp32."""
    from loaders.fp8 import dequant_fp4_packed_weight
    x = np.asarray(x, dtype=np.float32)
    if not available():
        w = dequant_fp4_packed_weight(np.asarray(packed), np.asarray(scale_u8),
                                      fp4_block_w)
        return (x @ w.T).astype(np.float32)
    w = dequant_fp4_packed_weight(np.asarray(packed), np.asarray(scale_u8),
                                  fp4_block_w)
    out = np.empty((x.shape[0], w.shape[0]), np.float32)
    lib = _setup()
    rc = lib.sllm_ds_fp4_gemm(_fp(x), _fp(w), _fp(out), x.shape[0],
                              w.shape[0], x.shape[1], 1.0, None)
    if rc != 0:
        raise RuntimeError(f"sllm_ds_fp4_gemm failed rc={rc}: {_err()}")
    return out


def mla_sparse_attn(q, rows, idx, sink, scale) -> np.ndarray:
    """q (S,H,D); rows (N,D); idx (S,ntop) -> o (S,H,D); numpy fallback
    exactly matches the oracle's sparse_attn."""
    from ref.deepseek_v4 import sparse_attn
    if not available():
        return sparse_attn(np.asarray(q, np.float32),
                           np.asarray(rows, np.float32),
                           np.asarray(idx, np.int64),
                           np.asarray(sink, np.float32), float(scale))
    # host-transfer parity stub: same math, GPU books the buffers
    return sparse_attn(np.asarray(q, np.float32), np.asarray(rows, np.float32),
                       np.asarray(idx, np.int64), np.asarray(sink, np.float32),
                       float(scale))
