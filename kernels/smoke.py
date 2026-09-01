"""GPU kernel smoke: run the sllm-node kernels and validate against the numpy
reference. Requires kernels/cuda/sllm_gpu.so (built on the cluster) and a
visible CUDA device."""

from __future__ import annotations

import numpy as np

from kernels import _sllm_cuda as ck
from ref import standard as st


def main() -> int:
    print(f"CUDA devices visible: {ck.device_count()}")
    if ck.device_count() < 1:
        print("NO_GPU")
        return 1

    rng = np.random.default_rng(0)
    rows, dim = 4, 1024
    x = (rng.standard_normal((rows, dim)) * 2.0).astype(np.float32)
    w = np.abs(rng.standard_normal(dim)).astype(np.float32) + 0.5

    y_gpu = ck.rms_norm(x, w, eps=1e-6)
    y_ref = st.rms_norm_plain(x, w, eps=1e-6)
    maxdiff = float(np.abs(y_gpu - y_ref).max())
    ok = maxdiff < 1e-4

    a = (rng.standard_normal((rows, dim))).astype(np.float32)
    sum_gpu = ck.elwise_add(x, a)
    sum_ref = (x + a).astype(np.float32)
    maxdiff2 = float(np.abs(sum_gpu - sum_ref).max())
    ok2 = maxdiff2 < 1e-4

    # cuBLAS GEMM parity: the whole dense-forward claim rests on this op.
    m, kk, n = 4, 64, 32
    A = rng.standard_normal((m, kk)).astype(np.float32)
    B = rng.standard_normal((kk, n)).astype(np.float32)
    gemm_gpu = np.asarray(ck.gemm(A, B)).reshape(m, n)
    maxdiff3 = float(np.abs(gemm_gpu - A @ B).max())
    ok3 = maxdiff3 < 1e-3

    # last-row decode-attention parity (softmax(QK^T*scale)V, per head).
    heads, S, D = 3, 37, 32
    q = rng.standard_normal((heads, D)).astype(np.float32)
    K = rng.standard_normal((heads, S, D)).astype(np.float32)
    V = rng.standard_normal((heads, S, D)).astype(np.float32)
    scale = float(D) ** -0.5
    attn_gpu = ck.attention_decode(q, K, V, scale)
    sc = np.einsum('hd,hsd->hs', q, K) * scale
    sc = np.exp(sc - sc.max(axis=-1, keepdims=True))
    p = sc / sc.sum(axis=-1, keepdims=True)
    attn_ref = np.einsum('hs,hsd->hd', p, V)
    maxdiff4 = float(np.abs(attn_gpu - attn_ref).max())
    ok4 = maxdiff4 < 1e-4

    print(f"rms_norm maxdiff={maxdiff:.3e} ok={ok}")
    print(f"elwise_add maxdiff={maxdiff2:.3e} ok={ok2}")
    print(f"gemm maxdiff={maxdiff3:.3e} ok={ok3}")
    print(f"attention_decode maxdiff={maxdiff4:.3e} ok={ok4}")
    print("SMOKE_OK" if (ok and ok2 and ok3 and ok4) else "SMOKE_FAIL")
    return 0 if (ok and ok2 and ok3 and ok4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
