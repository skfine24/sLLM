"""GPU kernel tests: run only when kernels/cuda/sllm_gpu.so exists (built on a
CUDA 13.0-capable machine, e.g. the cluster). Skipped elsewhere."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernels import _sllm_cuda as ck  # noqa: E402
from ref import standard as st  # noqa: E402

SO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "cuda", "sllm_gpu.so"))
need_gpu = os.path.isfile(SO)


@unittest.skipUnless(need_gpu, "sllm_gpu.so not built (no GPU toolchain)")
class TestCudaKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(0)
        cls.rows, cls.dim = 8, 512

    def test_device_visible(self):
        self.assertGreaterEqual(ck.device_count(), 1)

    def test_rms_norm_matches_reference(self):
        x = (self.rng.standard_normal((self.rows, self.dim)) * 2.0).astype(np.float32)
        w = (np.abs(self.rng.standard_normal(self.dim)) + 0.5).astype(np.float32)
        y = ck.rms_norm(x, w, eps=1e-6)
        ref = st.rms_norm_plain(x, w, eps=1e-6)
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)

    def test_elwise_add(self):
        a = self.rng.standard_normal((self.rows, self.dim)).astype(np.float32)
        b = self.rng.standard_normal((self.rows, self.dim)).astype(np.float32)
        y = ck.elwise_add(a, b)
        np.testing.assert_allclose(y, (a + b), rtol=1e-5, atol=1e-6)

    def test_gemm_matches_numpy(self):
        m, n, k = 7, 9, 11
        a = self.rng.standard_normal((m, k)).astype(np.float32)
        b = self.rng.standard_normal((k, n)).astype(np.float32)
        c = ck.gemm(a, b)
        np.testing.assert_allclose(c, a @ b, rtol=1e-4, atol=1e-4)

    def test_attention_decode_matches_numpy_last_row(self):
        heads, S, D = 4, 13, 16
        q = self.rng.standard_normal((heads, D)).astype(np.float32)
        k = self.rng.standard_normal((heads, S, D)).astype(np.float32)
        v = self.rng.standard_normal((heads, S, D)).astype(np.float32)
        scale = D ** -0.5
        out = ck.attention_decode(q, k, v, scale)

        # numpy reference: softmax_s(q_h.k_s*scale) . v_s  (fp32)
        s = scale * np.einsum("hd,hsd->hs", q, k)
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(-1, keepdims=True)
        ref = np.einsum("hs,hsd->hd", p, v)
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_device_buffer_roundtrip(self):
        a = self.rng.standard_normal((3, 5)).astype(np.float32)
        buf = ck.to_device(a)
        got = buf.copy_host()
        np.testing.assert_array_equal(got.reshape(a.shape), a)
        buf.free()


if __name__ == "__main__":
    unittest.main(verbosity=2)
