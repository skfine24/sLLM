"""FP8 E4M3 block-dequant GEMM parity tests (SKIPPED unless the built
sllm_gpu.so exports the sllm_fp8_* set -- on a CUDA box after
`kernels/cuda/build.sh`).

The kernel keeps the fp8 uint8 weight + float32 block-inverse scale on device
and dequantizes inside the GEMM. Compared against the host reference
(loaders/fp8.dequant_weight_blocked + numpy matmul), the single source of
truth.
"""

from __future__ import annotations

import unittest

import numpy as np

from kernels import _fp8_cuda as fp8
from loaders.fp8 import decode_bf16_array, dequant_weight_blocked

_NEEDS_GPU = not fp8.available()


def _e4m3_rand(rng, n, k):
    """Random E4M3-representable float32 values -> raw uint8 bytes."""
    vals = rng.standard_normal((n, k)).astype(np.float32) * 0.05
    return np.clip(vals, -448.0, 448.0)


def _to_fp8_bytes(vals: np.ndarray) -> np.ndarray:
    """Encode float32 to E4M3FN uint8 (bias 7, 3 mantissa, no subnormals)."""
    v = np.asarray(vals, dtype=np.float32)
    s = (v < 0).astype(np.uint8) << 7
    a = np.abs(v)
    # find exponent e such that 2^(e-7) <= a < 2^(e-6); clamp to [1,14]
    e = np.floor(np.log2(np.maximum(a, 1e-12))).astype(np.int32) + 7
    e = np.clip(e, 1, 14).astype(np.uint8)
    m = np.round((a / np.exp2(e.astype(np.float32) - 7.0) - 1.0) * 8.0) \
        .astype(np.uint8) & 0x07
    # keep only the top 3 mantissa bits; ~round-to-nearest already applied
    return (s | (e << 3) | m).astype(np.uint8)


@unittest.skipIf(_NEEDS_GPU, "sllm_fp8_* kernels not built / no CUDA")
class TestFP8Gemm(unittest.TestCase):
    def test_fp8_gemm_matches_host_dequant(self):
        rng = np.random.default_rng(7)
        M, N, K, bh, bw = 4, 5, 64, 128, 128
        a = rng.standard_normal((M, K)).astype(np.float32) * 0.1
        bvals = _e4m3_rand(rng, N, K)
        b8 = _to_fp8_bytes(bvals)
        # block inverse scales (fp32): 1x1 since K,N < block size
        scale = (rng.standard_normal((1, 1)).astype(np.float32) * 0.5
                 + 1.0)
        want = dequant_weight_blocked(b8, scale, bh, bw)
        got = fp8.fp8_gemm(a, b8, scale, bh, bw)
        np.testing.assert_allclose(got, a @ want.T, rtol=2e-4, atol=2e-4)

    def test_fp8_gemm_bf16_scale_contract(self):
        # the real checkpoint stores BF16 inverse scales; decode on host first
        rng = np.random.default_rng(11)
        M, N, K = 2, 3, 32
        a = rng.standard_normal((M, K)).astype(np.float32) * 0.1
        b8 = _to_fp8_bytes(_e4m3_rand(rng, N, K))
        scale_bf16 = (rng.standard_normal((1, 1)).astype(np.float32) * 0.4
                      + 1.0).astype(np.float16)
        scale = decode_bf16_array(scale_bf16.view(np.uint16))
        want = dequant_weight_blocked(b8, scale, 128, 128)
        got = fp8.fp8_gemm(a, b8, scale, 128, 128)
        np.testing.assert_allclose(got, a @ want.T, rtol=3e-4, atol=3e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
