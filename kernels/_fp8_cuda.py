"""ctypes bindings for the FP8 E4M3 GEMM kernel (kernels/cuda/fp8.cu).

Separate module so a .so built without fp8.cu keeps working: `available()`
reports whether the current build exports the fp8 symbols, and every wrapper
raises a clear RuntimeError when it does not. The kernel takes the raw fp8
`uint8` weight + its float32 block-inverse scale (already BF16-decoded) and
performs dequant+GEMM on-device -- the enabling primitive that lets the
173 GiB qwen4_exp model stay fp8 (no fp32/bf16 host blow-up).
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char_p, c_int, c_void_p

import numpy as np

_SO = os.path.abspath(os.path.join(os.path.dirname(__file__), "cuda",
                                   "sllm_gpu.so"))
_FP8_ENTRIES = ("sllm_fp8_gemm", "sllm_fp8_gemm_dev", "sllm_fp8_last_error")

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
        if not all(hasattr(lib, n) for n in _FP8_ENTRIES):
            raise AttributeError("sllm_fp8_* symbols missing (rebuild "
                                 "kernels/cuda with fp8.cu)")
        lib.sllm_fp8_last_error.restype = c_char_p
        F = POINTER(c_float)
        U = POINTER(ctypes.c_ubyte)
        lib.sllm_fp8_gemm.argtypes = [F, U, F, F, c_int, c_int, c_int,
                                      c_int, c_int]
        lib.sllm_fp8_gemm_dev.argtypes = [F, U, F, F, c_int, c_int, c_int,
                                          c_int, c_int]
        lib.sllm_fp8_gemm.restype = c_int
        lib.sllm_fp8_gemm_dev.restype = c_int
        _g = lib
    except Exception:
        _lib_failed = True
        _g = None
    return _g


def available() -> bool:
    return _setup() is not None


def _lib():
    lib = _setup()
    if lib is None:
        raise RuntimeError(
            "fp8 kernels unavailable: rebuild kernels/cuda (fp8.cu) via "
            "kernels/cuda/build.sh and run on a CUDA box")
    return lib


def _err() -> str:
    try:
        return _lib().sllm_fp8_last_error().decode()
    except Exception:
        return "fp8 library unavailable"


def _check(rc: int) -> None:
    if rc != 0:
        raise RuntimeError(f"fp8 kernel failed rc={rc}: {_err()}")


def _ufp(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.uint8)


def _fp(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def fp8_gemm(A, B, scale_inv, block_h=128, block_w=128) -> np.ndarray:
    """C (M,N) = A (M,K) @ dequant(B) (N,K)^T, B fp8 E4M3 with block scales.

    `B` is (N,K) uint8 fp8; `scale_inv` is (ceil(K/bh), ceil(N/bw)) float32
    (the BF16-inverse scale, already decoded). Returns fp32 (M,N).
    """
    A, B, scale_inv = _fp(A), _ufp(B), _fp(scale_inv)
    M, K = A.shape
    N = B.shape[0]
    if B.shape != (N, K):
        raise ValueError(f"B {B.shape} must be (N,K) matching A (M,K) = {(M, K)}")
    out = np.empty((M, N), np.float32)
    _check(_lib().sllm_fp8_gemm(
        A.ctypes.data_as(POINTER(c_float)),
        B.ctypes.data_as(POINTER(ctypes.c_ubyte)),
        scale_inv.ctypes.data_as(POINTER(c_float)),
        out.ctypes.data_as(POINTER(c_float)),
        M, N, K, int(block_h), int(block_w)))
    return out


def fp8_gemm_dev(A, B, scale_inv, out, M, N, K, block_h=128, block_w=128):
    """Device-pointer variant: A/B/scale_inv/out are DeviceBuffers."""
    _check(_lib().sllm_fp8_gemm_dev(
        _ptr(A), _ptr(B), _ptr(scale_inv), _ptr(out),
        M, N, K, int(block_h), int(block_w)))


def _ptr(buf):
    return getattr(buf, "ptr", buf)
