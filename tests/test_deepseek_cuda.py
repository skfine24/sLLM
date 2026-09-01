"""DeepSeek CUDA wrapper tests (phase 5, local == numpy fallback).

`kernels._deepseek_cuda` lazy-loads sllm_gpu.so; on the dev box the .so is
absent so these tests exercise the numpy FALLBACK path and assert the wrapper
contract (fp4 expert linear == dequant+matmul; MLA sparse attn == oracle).
When the cluster build exports `sllm_ds_*` the same tests run the device path
-- the parity surface.
"""

import unittest

import numpy as np

from kernels import _deepseek_cuda as dsc


class TestDeepseekCudaWrapper(unittest.TestCase):
    def test_fp4_moe_linear_fallback(self):
        from loaders.fp8 import dequant_fp4_packed_weight
        rng = np.random.default_rng(5)
        x = rng.standard_normal((4, 32), dtype=np.float32)
        # pack a small fp4 expert (N=6, K=32)
        w = rng.standard_normal((6, 32)).astype(np.float32)
        from tests.test_tp_deepseek import _pack_fp4
        packed, scale_u8 = _pack_fp4(w, rng)
        got = dsc.fp4_moe_linear(x, packed, scale_u8)
        wq = dequant_fp4_packed_weight(packed, scale_u8)
        want = (x @ wq.T).astype(np.float32)
        self.assertEqual(got.shape, (4, 6))
        np.testing.assert_allclose(got, want, rtol=0, atol=0)

    def test_mla_sparse_attn_matches_oracle(self):
        from ref.deepseek_v4 import sparse_attn
        rng = np.random.default_rng(6)
        S, H, D, N, ntop = 3, 2, 8, 20, 5
        q = rng.standard_normal((S, H, D)).astype(np.float32)
        rows = rng.standard_normal((N, D)).astype(np.float32)
        idx = rng.integers(-1, N, size=(S, ntop)).astype(np.int64)
        idx[idx == -1] = -1
        sink = rng.standard_normal(H).astype(np.float32)
        scale = D ** -0.5
        got = dsc.mla_sparse_attn(q, rows, idx, sink, scale)
        want = sparse_attn(q, rows, idx, sink, scale)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
