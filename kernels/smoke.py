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

    print(f"rms_norm maxdiff={maxdiff:.3e} ok={ok}")
    print(f"elwise_add maxdiff={maxdiff2:.3e} ok={ok2}")
    print("SMOKE_OK" if (ok and ok2) else "SMOKE_FAIL")
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
