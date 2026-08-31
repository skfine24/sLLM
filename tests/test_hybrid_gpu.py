"""GPU hybrid (qwen3_5) decode parity: the GPU layer kernels must reproduce the
numpy incremental hybrid decode within fp tolerance. Runs only on a machine
with kernels/cuda/sllm_gpu.so built."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ref import incremental as inc  # noqa: E402
from serving.dev_model import tiny_recipe, tiny_weights  # noqa: E402

SO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "cuda", "sllm_gpu.so"))
need_gpu = os.path.isfile(SO)


@unittest.skipUnless(need_gpu, "sllm_gpu.so not built (no GPU toolchain)")
class TestGpuHybridDecode(unittest.TestCase):
    def setUp(self):
        self.recipe = tiny_recipe()
        self.weights = tiny_weights(np.random.default_rng(42))
        self.ids = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        from kernels import _sllm_cuda as ck
        self.assertTrue(ck.device_count() >= 1)

    def _gpu_decode_one(self, cache, last_id):
        from kernels.hybrid_decode import gpu_hybrid_decode_step
        return gpu_hybrid_decode_step(cache, self.weights, self.recipe, last_id)

    def test_gated_delta_step_matches_numpy(self):
        # direct kernel-vs-numpy check on the recurrent core
        from kernels import _sllm_cuda as ck
        from ref import qwen3_5 as qq
        Vh, Kd, Vd = 4, 8, 8
        rng = np.random.default_rng(0)
        q = qq.l2norm(rng.standard_normal((Vh, Kd)).astype(np.float32))
        k = qq.l2norm(rng.standard_normal((Vh, Kd)).astype(np.float32))
        v = rng.standard_normal((Vh, Vd)).astype(np.float32)
        g = rng.standard_normal(Vh).astype(np.float32)
        beta = np.abs(rng.standard_normal(Vh)).astype(np.float32) * 0.5
        state = rng.standard_normal((Vh, Kd, Vd)).astype(np.float32)

        state_np = state.copy()
        dec = np.exp(g)
        kv_mem = np.einsum("hde,hd->he", state_np * dec[:, None, None], k)
        delta = (v - kv_mem) * beta[:, None]
        state_np = state_np * dec[:, None, None] + k[..., None] * delta[:, None, :]
        out_np = np.einsum("hde,hd->he", state_np, q)

        state_gpu = state.copy()
        out_gpu = ck.gated_delta_step(q, k, v, g, beta, state_gpu)
        np.testing.assert_allclose(out_gpu, out_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(state_gpu, state_np, rtol=1e-3, atol=1e-3)

    def test_hybrid_gpu_logits_match_numpy(self):
        # Build a fresh numpy cache (numpy prefill) and run both decode paths on
        # separate caches so in-place state mutation never cross-contaminates.
        cache_np, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        cache_gpu, _ = inc.prefill(self.ids, self.weights, self.recipe)
        seq = list(self.ids[0])
        nxt = int(np.argmax(L0[0, -1]))
        for _ in range(4):
            seq.append(nxt)
            L_np = inc.decode_step(cache_np, self.weights, self.recipe, nxt)
            L_gpu = self._gpu_decode_one(cache_gpu, nxt)
            np.testing.assert_allclose(L_gpu, L_np, rtol=5e-3, atol=5e-3)
            self.assertEqual(int(np.argmax(L_gpu)), int(np.argmax(L_np)))
            nxt = int(np.argmax(L_np))

    def test_hybrid_gpu_deterministic(self):
        cache_gpu, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        nxt = int(np.argmax(L0[0, -1]))
        L1 = self._gpu_decode_one(cache_gpu, nxt)
        out1 = np.argmax(L1)
        cache_gpu2, L02 = inc.prefill(self.ids, self.weights, self.recipe)
        L2 = self._gpu_decode_one(cache_gpu2, nxt)
        np.testing.assert_array_equal(L1, L2)
        self.assertEqual(int(out1), int(np.argmax(L2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
