"""ctypes binding to the sllm-node GPU kernels (kernels/cuda/sllm_gpu.so).

The .so is produced by `kernels/cuda/build.sh` (nvcc, CUDA 13.0 baseline) and
is built either on the host or inside the deploy Docker image. All kernels are
CUDA-ABI C functions; this module handles marshalling numpy arrays.

Functions with GPU access depend on a CUDA-capable runtime + driver; on a
non-GPU box `device_count()` returns -1 and callers should degrade to the
numpy reference.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char_p, c_float, c_int, c_long, c_void_p

import numpy as np

_SO = os.path.abspath(os.path.join(os.path.dirname(__file__), "cuda", "sllm_gpu.so"))


def _lib():
    if not os.path.isfile(_SO):
        raise FileNotFoundError(f"sllm_gpu.so not built: {_SO} (run kernels/cuda/build.sh)")
    lib = ctypes.CDLL(_SO)
    lib.sllm_last_error.restype = c_char_p
    lib.sllm_device_count.restype = c_int
    lib.sllm_rms_norm.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float),
                                  c_int, c_int, c_float]
    lib.sllm_rms_norm.restype = c_int
    lib.sllm_elwise_add.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_long]
    lib.sllm_elwise_add.restype = c_int
    lib.sllm_silu.argtypes = [POINTER(c_float), POINTER(c_float), c_long]
    lib.sllm_silu.restype = c_int
    lib.sllm_gemm.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float),
                              c_int, c_int, c_int]
    lib.sllm_gemm.restype = c_int
    lib.sllm_attention_decode.argtypes = [POINTER(c_float), POINTER(c_float),
                                          POINTER(c_float), POINTER(c_float),
                                          c_int, c_int, c_int, c_float]
    lib.sllm_attention_decode.restype = c_int
    lib.sllm_gated_delta_step.argtypes = [POINTER(c_float), POINTER(c_float),
                                          POINTER(c_float), POINTER(c_float),
                                          POINTER(c_float), POINTER(c_float),
                                          POINTER(c_float), c_int, c_int, c_int]
    lib.sllm_gated_delta_step.restype = c_int
    lib.sllm_buf_new.argtypes = [c_long]
    lib.sllm_buf_new.restype = c_void_p
    lib.sllm_mem_free_bytes.argtypes = []
    lib.sllm_mem_free_bytes.restype = c_long
    lib.sllm_buf_free.argtypes = [c_void_p]
    lib.sllm_buf_free.restype = c_int
    lib.sllm_buf_h2d.argtypes = [c_void_p, POINTER(c_float), c_long]
    lib.sllm_buf_h2d.restype = c_int
    lib.sllm_buf_d2h.argtypes = [POINTER(c_float), c_void_p, c_long]
    lib.sllm_buf_d2h.restype = c_int
    lib.sllm_buf_h2d_raw.argtypes = [c_void_p, c_void_p, c_long]
    lib.sllm_buf_h2d_raw.restype = c_int
    # -- device-resident API (all args are DEVICE pointers; see kernels.cu) --
    for name in ("sllm_sync",):
        getattr(lib, name).restype = c_int
    lib.sllm_gemm_dev.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
    lib.sllm_gemm_dev.restype = c_int
    lib.sllm_gemm_linear_dev.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
    lib.sllm_gemm_linear_dev.restype = c_int
    lib.sllm_bias_add_dev.argtypes = [c_void_p, c_void_p, c_int, c_int]
    lib.sllm_bias_add_dev.restype = c_int
    lib.sllm_elwise_mul_dev.argtypes = [c_void_p, c_void_p, c_void_p, c_long]
    lib.sllm_elwise_mul_dev.restype = c_int
    lib.sllm_elwise_add_dev.argtypes = [c_void_p, c_void_p, c_void_p, c_long]
    lib.sllm_elwise_add_dev.restype = c_int
    lib.sllm_silu_dev.argtypes = [c_void_p, c_void_p, c_long]
    lib.sllm_silu_dev.restype = c_int
    lib.sllm_rms_norm_dev.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_float]
    lib.sllm_rms_norm_dev.restype = c_int
    lib.sllm_gather_row.argtypes = [c_void_p, c_long, c_void_p, c_int]
    lib.sllm_gather_row.restype = c_int
    lib.sllm_rope_dev.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    lib.sllm_rope_dev.restype = c_int
    lib.sllm_kv_write.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int, c_int,
                                  c_void_p, c_int]
    lib.sllm_kv_write.restype = c_int
    lib.sllm_kv_relayout.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    lib.sllm_kv_relayout.restype = c_int
    lib.sllm_kv_relayout_w.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
    lib.sllm_kv_relayout_w.restype = c_int
    lib.sllm_attention_decode_dev.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p,
                                              c_void_p, c_int, c_int, c_int, c_int,
                                              c_int, c_float]
    lib.sllm_attention_decode_dev.restype = c_int
    # -- general-purpose fused + typed API (dtype-tagged device pointers) ----
    lib.sllm_gemm_ex.argtypes = [c_void_p, c_int, c_void_p, c_void_p, c_int, c_int, c_int]
    lib.sllm_gemm_ex.restype = c_int
    lib.sllm_gather_row_t.argtypes = [c_void_p, c_long, c_void_p, c_int, c_int]
    lib.sllm_gather_row_t.restype = c_int
    lib.sllm_add_rms.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
                                 c_int, c_int, c_float, c_int]
    lib.sllm_add_rms.restype = c_int
    lib.sllm_rope_bias.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
                                   c_int, c_int, c_int, c_int]
    lib.sllm_rope_bias.restype = c_int
    lib.sllm_silu_mul.argtypes = [c_void_p, c_void_p, c_void_p, c_long, c_int]
    lib.sllm_silu_mul.restype = c_int
    lib.sllm_kv_write_t.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p,
                                    c_int, c_int, c_void_p, c_int, c_int]
    lib.sllm_kv_write_t.restype = c_int
    lib.sllm_attention_decode_t.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p,
                                            c_void_p, c_int, c_int, c_int, c_int,
                                            c_int, c_float, c_int]
    lib.sllm_attention_decode_t.restype = c_int
    return lib


_g = None


def load():
    global _g
    if _g is None:
        _g = _lib()
    return _g


def device_count() -> int:
    """Number of CUDA devices visible (returns -1 when the driver/GPU is missing)."""
    try:
        return int(load().sllm_device_count())
    except Exception:
        return -1


def mem_free_bytes() -> int:
    """Free device memory in bytes; -1 when the info call fails."""
    try:
        return int(load().sllm_mem_free_bytes())
    except Exception:
        return -1


def _ptr(a: np.ndarray):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(POINTER(c_float))


def _check(rc: int) -> None:
    if rc != 0:
        try:
            err = load().sllm_last_error()
        except Exception:
            err = None
        raise RuntimeError(err.decode() if err else f"sllm call failed (rc={rc})")


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """GPU RMSNorm: out[row, :] = x * rsqrt(mean(x^2, -1) + eps) * weight."""
    xa = np.ascontiguousarray(x, dtype=np.float32)
    if xa.ndim != 2:
        raise ValueError("rms_norm expects a 2-D (rows, dim) array")
    rows, dim = xa.shape
    if weight.shape[0] != dim:
        raise ValueError("weight must be (dim,)")
    wa = np.ascontiguousarray(weight, dtype=np.float32)
    y = np.empty_like(xa)
    xa, xp = _ptr(xa)
    wa, wp = _ptr(wa)
    yp = y.ctypes.data_as(POINTER(c_float))
    _check(load().sllm_rms_norm(xp, wp, yp, rows, dim, float(eps)))
    return y


def elwise_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU y = a + b (same shape, float32)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("a and b must share shape")
    y = np.empty_like(a)
    _, ap = _ptr(a)
    _, bp = _ptr(b)
    yp = y.ctypes.data_as(POINTER(c_float))
    _check(load().sllm_elwise_add(ap, bp, yp, int(a.size)))
    return y


def silu(x: np.ndarray) -> np.ndarray:
    """GPU SiLU (Swish): y = x * sigmoid(x), float32, shape-preserving."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.empty_like(x)
    _, xp = _ptr(x)
    yp = y.ctypes.data_as(POINTER(c_float))
    _check(load().sllm_silu(xp, yp, int(x.size)))
    return y


# ---------------------------------------------------------------------------
# Device tensors + KV-placement building blocks (GPU-backed runs)
# ---------------------------------------------------------------------------

class DeviceBuffer:
    """Opaque device-side tensor (KV/recurrent state backing)."""

    def __init__(self, nbytes: int):
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError("nbytes must be a positive int")
        self.nbytes = nbytes
        self._ptr = load().sllm_buf_new(nbytes)
        if not self._ptr:
            raise RuntimeError("sllm_buf_new failed: "
                               + load().sllm_last_error().decode())
        self._closed = False

    def __del__(self):
        # GC safety net: exception paths that miss .free() must not strand
        # device memory for the process lifetime.
        try:
            self.free()
        except Exception:
            pass

    @property
    def ptr(self) -> int:
        if self._closed:
            raise RuntimeError("DeviceBuffer already freed")
        return self._ptr

    def free(self) -> None:
        if not self._closed:
            _check(load().sllm_buf_free(self._ptr))
            self._closed = True

    def copy_host(self) -> np.ndarray:
        """Return a float32 copy of the device bytes (row-major as stored)."""
        if self.nbytes % 4 != 0:
            raise ValueError("non-float32-sized buffer")
        n = self.nbytes // 4
        y = np.empty(n, dtype=np.float32)
        yp = y.ctypes.data_as(POINTER(c_float))
        _check(load().sllm_buf_d2h(yp, self._ptr, self.nbytes))
        return y

    def upload(self, arr: np.ndarray) -> None:
        """Overwrite the buffer from a float32 array of the same size."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.nbytes != self.nbytes:
            raise ValueError(f"size mismatch: {arr.nbytes} != {self.nbytes}")
        ap = arr.ctypes.data_as(POINTER(c_float))
        _check(load().sllm_buf_h2d(self._ptr, ap, arr.nbytes))

    def upload_raw(self, arr: np.ndarray) -> None:
        """Overwrite the buffer from a raw byte-image array (e.g. bf16 as
        uint16) of exactly this buffer's size."""
        arr = np.ascontiguousarray(arr)
        if arr.nbytes != self.nbytes:
            raise ValueError(f"size mismatch: {arr.nbytes} != {self.nbytes}")
        _check(load().sllm_buf_h2d_raw(self._ptr,
                                       arr.ctypes.data_as(c_void_p),
                                       arr.nbytes))


class DeviceView:
    """Non-owning byte window into a DeviceBuffer (fused-GEMM output slices).

    Keeps the base buffer alive; expose the same `.ptr`/`.nbytes` surface the
    device-op wrappers use, so callers can pass views anywhere a buffer fits.
    """

    def __init__(self, base: DeviceBuffer, off_bytes: int, nbytes: int):
        if off_bytes < 0 or nbytes < 0 or off_bytes + nbytes > base.nbytes:
            raise ValueError("view out of range")
        self._base = base
        self.nbytes = nbytes
        self._off = off_bytes
        base.ptr  # validates the base is live at construction time

    @property
    def ptr(self) -> int:
        # resolved through the base so a freed base raises instead of
        # handing out a stale (wild) device pointer
        return self._base.ptr + self._off


def to_bf16(arr: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 byte image (uint16, round-to-nearest-even)."""
    u = np.asarray(arr, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return np.ascontiguousarray(r)


def to_device(arr: np.ndarray) -> DeviceBuffer:
    """Upload a float32 array to the device and return a DeviceBuffer."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    buf = DeviceBuffer(arr.nbytes)
    ap = arr.ctypes.data_as(POINTER(c_float))
    _check(load().sllm_buf_h2d(buf.ptr, ap, arr.nbytes))
    return buf


def gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU c(m,n) = a(m,k) @ b(k,n), float32 (cuBLAS-backed).

    Feeds the kernel Fortran-order (column-major) buffers, which is cuBLAS's
    natural layout; returns a row-major numpy result.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"bad gemm shapes {a.shape} @ {b.shape}")
    m, n, k = a.shape[0], b.shape[1], a.shape[1]
    a_f = np.asfortranarray(a)
    b_f = np.asfortranarray(b)
    c_f = np.empty((m, n), dtype=np.float32, order="F")
    ap = a_f.ctypes.data_as(POINTER(c_float))
    bp = b_f.ctypes.data_as(POINTER(c_float))
    cp = c_f.ctypes.data_as(POINTER(c_float))
    _check(load().sllm_gemm(ap, bp, cp, m, n, k))
    return np.ascontiguousarray(c_f)


def attention_decode(q: np.ndarray, K: np.ndarray, V: np.ndarray,
                     scale: float = 1.0) -> np.ndarray:
    """GPU single-row decode attention: out[h] = softmax(q[h] . K[h] * scale) . V[h].

    q: (heads, D); K/V: (heads, S, D); out: (heads, D). Mirrors the numpy
    last-row attention used by incremental decode.
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    K = np.ascontiguousarray(K, dtype=np.float32)
    V = np.ascontiguousarray(V, dtype=np.float32)
    heads, S, D = K.shape
    if q.shape != (heads, D) or V.shape != (heads, S, D):
        raise ValueError("shape mismatch in attention_decode")
    out = np.empty((heads, D), dtype=np.float32)
    qp = q.ctypes.data_as(POINTER(c_float))
    kp = K.ctypes.data_as(POINTER(c_float))
    vp = V.ctypes.data_as(POINTER(c_float))
    op = out.ctypes.data_as(POINTER(c_float))
    _check(load().sllm_attention_decode(qp, kp, vp, op, heads, S, D, float(scale)))
    return out


def gated_delta_step(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                     g: np.ndarray, beta: np.ndarray, state: np.ndarray):
    """GPU GatedDeltaNet single-step decode (in-place state update).

    q/k: (Vh, Kd) [caller normalizes: L2 + kd^-0.5 and replicates to Vh];
    v: (Vh, Vd); g/beta: (Vh,); state: (Vh, Kd, Vd) -> updated IN PLACE;
    returns out (Vh, Vd).
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)
    # The kernel updates `state` IN PLACE. ascontiguousarray() would silently
    # copy a non-fp32/non-contiguous input and the recurrent state would then
    # FREEZE with rc==0 — refuse instead.
    if state.dtype != np.float32 or not state.flags["C_CONTIGUOUS"]:
        raise ValueError("gated_delta_step state must be fp32 C-contiguous "
                         "(it is updated in place; a copy would be discarded)")
    Vh, Kd, Vd = state.shape
    if q.shape != (Vh, Kd) or k.shape != (Vh, Kd) or v.shape != (Vh, Vd):
        raise ValueError("shape mismatch in gated_delta_step")
    out = np.empty((Vh, Vd), dtype=np.float32)
    _check(load().sllm_gated_delta_step(
        q.ctypes.data_as(POINTER(c_float)), k.ctypes.data_as(POINTER(c_float)),
        v.ctypes.data_as(POINTER(c_float)), g.ctypes.data_as(POINTER(c_float)),
        beta.ctypes.data_as(POINTER(c_float)), state.ctypes.data_as(POINTER(c_float)),
        out.ctypes.data_as(POINTER(c_float)), Vh, Kd, Vd))
    return out


# ---------------------------------------------------------------------------
# Device-resident ops (device-resident weights + persistent on-device KV).
# All buffers are DeviceBuffer objects; these calls enqueue on the default
# stream and do NOT synchronise — call sync() once at the end of a step.
# ---------------------------------------------------------------------------

def alloc(nfloats: int) -> DeviceBuffer:
    """Allocate an empty device buffer of nfloats float32 values."""
    return DeviceBuffer(int(nfloats) * 4)


def sync() -> None:
    """Blocking device synchronisation (one per decode step)."""
    _check(load().sllm_sync())


def gemm_dev(a: DeviceBuffer, b: DeviceBuffer, c: DeviceBuffer,
             m: int, n: int, k: int) -> None:
    """c(m,n) = a(m,k) @ b(k,n), row-major device buffers."""
    _check(load().sllm_gemm_dev(a.ptr, b.ptr, c.ptr, int(m), int(n), int(k)))


def gemm_linear_dev(hst: DeviceBuffer, w: DeviceBuffer, c: DeviceBuffer,
                    m: int, n: int, k: int) -> None:
    """c(m,n) = hst(m,k) @ w(n,k)^T; w device-resident in native (out,in) form."""
    _check(load().sllm_gemm_linear_dev(hst.ptr, w.ptr, c.ptr,
                                       int(m), int(n), int(k)))


def bias_add_dev(y: DeviceBuffer, bias: DeviceBuffer,
                 rows: int, cols: int) -> None:
    """In-place y(r,c) += bias(c)."""
    _check(load().sllm_bias_add_dev(y.ptr, bias.ptr, int(rows), int(cols)))


def elwise_mul_dev(a: DeviceBuffer, b: DeviceBuffer, y: DeviceBuffer) -> None:
    """y = a*b (y may alias a/b)."""
    if a.nbytes != b.nbytes or a.nbytes != y.nbytes:
        raise ValueError("a/b/y must share size")
    _check(load().sllm_elwise_mul_dev(a.ptr, b.ptr, y.ptr, a.nbytes // 4))


def elwise_add_dev(a: DeviceBuffer, b: DeviceBuffer, y: DeviceBuffer) -> None:
    """y = a+b (y may alias a/b)."""
    if a.nbytes != b.nbytes or a.nbytes != y.nbytes:
        raise ValueError("a/b/y must share size")
    _check(load().sllm_elwise_add_dev(a.ptr, b.ptr, y.ptr, a.nbytes // 4))


def silu_dev(x: DeviceBuffer, y: DeviceBuffer) -> None:
    """y = x*sigmoid(x) (y may alias x)."""
    _check(load().sllm_silu_dev(x.ptr, y.ptr, x.nbytes // 4))


def rms_norm_dev(x: DeviceBuffer, weight: DeviceBuffer, y: DeviceBuffer,
                 rows: int, dim: int, eps: float) -> None:
    """Device RMSNorm rows x dim -> y."""
    _check(load().sllm_rms_norm_dev(x.ptr, weight.ptr, y.ptr,
                                    int(rows), int(dim), float(eps)))


def gather_row(table: DeviceBuffer, row: int, out: DeviceBuffer,
               dim: int) -> None:
    """out(dim) = table(row, dim) — device embedding lookup."""
    _check(load().sllm_gather_row(table.ptr, int(row), out.ptr, int(dim)))


def rope_dev(q: DeviceBuffer, k: DeviceBuffer, cs: DeviceBuffer,
             qrows: int, krows: int, dim: int, rot: int) -> None:
    """In-place RoPE on q(qrows,dim) and k(krows,dim); cs = cos||sin (2*rot)."""
    _check(load().sllm_rope_dev(q.ptr, k.ptr, cs.ptr,
                                int(qrows), int(krows), int(dim), int(rot)))


def kv_write(kv: DeviceBuffer, cap: int, pos: int, row: DeviceBuffer,
             heads: int, dim: int, stage: DeviceBuffer | None = None,
             stage_off: int = 0) -> None:
    """Append row(heads,dim) into kv(heads,cap,dim) at pos; optional device
    staging mirror for the batched end-of-step host copy."""
    _check(load().sllm_kv_write(kv.ptr, int(cap), int(pos), row.ptr,
                                int(heads), int(dim),
                                stage.ptr if stage is not None else None,
                                int(stage_off)))


def kv_relayout(dst: DeviceBuffer, src: DeviceBuffer, heads: int,
                cap_old: int, cap_new: int, dim: int) -> None:
    """Copy kv (heads,cap_old,dim) into a larger (heads,cap_new,dim) buffer."""
    _check(load().sllm_kv_relayout(dst.ptr, src.ptr, int(heads),
                                   int(cap_old), int(cap_new), int(dim)))


def kv_relayout_w(dst, src, heads: int, cap_old: int, cap_new: int,
                  blk_words: int) -> None:
    """dtype-agnostic KV capacity growth: each (h,s) row is blk_words 32-bit
    words (dim*elem/4)."""
    _check(load().sllm_kv_relayout_w(_p(dst), _p(src), int(heads),
                                     int(cap_old), int(cap_new),
                                     int(blk_words)))


def attention_decode_dev(q: DeviceBuffer, K: DeviceBuffer, V: DeviceBuffer,
                         scores: DeviceBuffer, out: DeviceBuffer,
                         heads: int, kv_heads: int, S: int, stride: int,
                         D: int, scale: float) -> None:
    """Device GQA last-row decode attention (head h reads KV head h/(h//kvh))."""
    _check(load().sllm_attention_decode_dev(
        q.ptr, K.ptr, V.ptr, scores.ptr, out.ptr, int(heads), int(kv_heads),
        int(S), int(stride), int(D), float(scale)))


# ---------------------------------------------------------------------------
# General-purpose fused + typed ops (dtype-tagged; buffers may be
# DeviceBuffer or DeviceView; None-able args accept optional tensors).
# dtype tags: T_F32 / T_BF16.
# ---------------------------------------------------------------------------

T_F32 = 0
T_BF16 = 1


def _p(buf) -> int | None:
    return None if buf is None else buf.ptr


def alloc_n(nbytes: int) -> DeviceBuffer:
    """Allocate a raw device buffer of nbytes (typed tensors live in these)."""
    return DeviceBuffer(int(nbytes))


def gemm_ex(a, w, c, m: int, n: int, k: int, atype: int = T_F32) -> None:
    """c(m,n) fp32 = a(m,k) @ w(n,k)^T; a,w in dtype `atype`, fp32 accumulate."""
    _check(load().sllm_gemm_ex(_p(a), int(atype), _p(w), _p(c),
                               int(m), int(n), int(k)))


def gather_row_t(table, row: int, out, dim: int, t: int = T_F32) -> None:
    """out(dim) fp32 = table(row,dim) cast from dtype `t`."""
    _check(load().sllm_gather_row_t(_p(table), int(row), _p(out),
                                    int(dim), int(t)))


def add_rms(a, b, sum_out, norm_out, w, rows: int, dim: int, eps: float,
            norm_type: int = T_F32) -> None:
    """sum = a + (b or 0); norm_out = rmsnorm(sum) * w, stored dtype `norm_type`."""
    _check(load().sllm_add_rms(_p(a), _p(b), _p(sum_out), _p(norm_out), _p(w),
                               int(rows), int(dim), float(eps), int(norm_type)))


def rope_bias(q, k, qb, kb, cs, qrows: int, krows: int, dim: int,
              rot: int) -> None:
    """In-place q+=qb, k+=kb then partial RoPE on both (cs = cos||sin row)."""
    _check(load().sllm_rope_bias(_p(q), _p(k), _p(qb), _p(kb), _p(cs),
                                 int(qrows), int(krows), int(dim), int(rot)))


def silu_mul(g, u, out, out_type: int = T_F32) -> None:
    """out = silu(g)*u (fp32 math; g/u fp32; out stored dtype `out_type`)."""
    n = g.nbytes // 4
    _check(load().sllm_silu_mul(_p(g), _p(u), _p(out), int(n), int(out_type)))


def kv_write_t(kv, cap: int, pos: int, row, bias, heads: int, dim: int,
               stage=None, stage_off: int = 0,
               kv_type: int = T_F32) -> None:
    """Append fp32 row(+optional bias) into typed kv(heads,cap,dim) at pos
    with an fp32 staging mirror."""
    _check(load().sllm_kv_write_t(_p(kv), int(cap), int(pos), _p(row), _p(bias),
                                  int(heads), int(dim), _p(stage),
                                  int(stage_off), int(kv_type)))


def attention_decode_t(q, K, V, scores, out, heads: int, kv_heads: int,
                       S: int, stride: int, D: int, scale: float,
                       kv_type: int = T_F32) -> None:
    """GQA last-row decode attention with typed KV, fp32 math/out."""
    _check(load().sllm_attention_decode_t(
        _p(q), _p(K), _p(V), _p(scores), _p(out), int(heads), int(kv_heads),
        int(S), int(stride), int(D), float(scale), int(kv_type)))
