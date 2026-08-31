"""A4 qwen4_exp CUDA kernel parity tests (SKIPPED unless the built
sllm_gpu.so exports the sllm_q4_* set — i.e. on a CUDA box after
`kernels/cuda/build.sh`; runs on the cluster in milestone B).

Every kernel is compared against the numpy oracle in ref/qwen4_exp.py
(which cites oracle/upstream/sglang/*), the single source of truth.
"""

from __future__ import annotations

import unittest

import numpy as np

from kernels import _q4_cuda as q4
from ref import qwen4_exp as qe

_NEEDS_GPU = not q4.available()


def _silu(x):
    return x / (1.0 + np.exp(-x))


@unittest.skipIf(_NEEDS_GPU, "sllm_q4_* kernels not built / no CUDA")
class TestHC(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.rng = rng
        self.rows, self.hc, self.hs, self.lr = 3, 2, 8, 4
        self.hyper = rng.standard_normal((self.rows, self.hc * self.hs)) \
            .astype(np.float32) * 0.3
        self.norm_w = rng.standard_normal(self.hc * self.hs).astype(np.float32) * 0.1
        self.down = rng.standard_normal((self.lr, self.hc * self.hs)) \
            .astype(np.float32) * 0.1
        self.up = rng.standard_normal((self.hc * self.hs, self.lr)) \
            .astype(np.float32) * 0.1
        self.inject = rng.standard_normal((self.hc, self.hc * self.hs)) \
            .astype(np.float32) * 0.1

    def test_grouped_gemma_rmsnorm(self):
        got = q4.grouped_gemma_rmsnorm(self.hyper, self.norm_w, self.hc,
                                       self.hs)
        want = qe.grouped_gemma_rmsnorm(self.hyper, self.norm_w, self.hs)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_hc_mix_full_chain(self):
        normed = q4.grouped_gemma_rmsnorm(self.hyper, self.norm_w, self.hc,
                                          self.hs)
        low = q4.gemv_rows(normed, self.down, post=1, scale=self.hc)
        wgate = q4.gemv_rows(low, self.up, post=2, scale=1.0)
        mixed = q4.hc_mix_apply(wgate, normed, self.hc, self.hs)
        w_mixed, w_normed = qe.hc_mix(self.hyper, self.norm_w, self.down,
                                      self.up, self.hc, self.hs)
        np.testing.assert_allclose(normed, w_normed, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(mixed, w_mixed, rtol=1e-4, atol=2e-6)

    def test_hc_combine(self):
        block = self.rng.standard_normal((self.rows, self.hs)).astype(
            np.float32) * 0.2
        _, normed = qe.hc_mix(self.hyper, self.norm_w, self.down, self.up,
                              self.hc, self.hs)
        normed = normed.astype(np.float32)
        got = q4.hc_combine(self.hyper, block, normed, self.inject, self.hc,
                            self.hs)
        want = qe.hc_combine(block, self.hyper, normed, self.inject, self.hc,
                             self.hs)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@unittest.skipIf(_NEEDS_GPU, "sllm_q4_* kernels not built / no CUDA")
class TestMoE(unittest.TestCase):
    def test_router_matches_oracle(self):
        rng = np.random.default_rng(33)
        logits = rng.standard_normal((5, 16)).astype(np.float32) * 2.0
        w, ids = q4.moe_router(logits, 4)
        w_ref, ids_ref = qe.moe_route(logits, 4)
        np.testing.assert_array_equal(ids, ids_ref.astype(np.int32))
        np.testing.assert_allclose(w, w_ref, rtol=1e-5, atol=1e-7)

    def test_swiglu_axpy_shared_gate(self):
        rng = np.random.default_rng(34)
        g = rng.standard_normal((3, 6)).astype(np.float32)
        u = rng.standard_normal((3, 6)).astype(np.float32)
        np.testing.assert_allclose(q4.swiglu(g, u), _silu(g) * u,
                                   rtol=1e-5, atol=1e-7)
        out = rng.standard_normal((3, 6)).astype(np.float32)
        y = rng.standard_normal((3, 6)).astype(np.float32)
        w = rng.random(3).astype(np.float32)
        np.testing.assert_allclose(q4.axpy_rows(out, y, w),
                                   out + w[:, None] * y, rtol=1e-6, atol=1e-7)
        gate = rng.standard_normal(3).astype(np.float32)
        shared = rng.standard_normal((3, 6)).astype(np.float32)
        np.testing.assert_allclose(
            q4.shared_gate_accum(out, shared, gate),
            out + (1.0 / (1.0 + np.exp(-gate)))[:, None] * shared,
            rtol=1e-5, atol=1e-7)


@unittest.skipIf(_NEEDS_GPU, "sllm_q4_* kernels not built / no CUDA")
class TestQSA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(35)
        self.rng = rng
        self.nh, self.d, self.rot = 4, 8, 4
        self.q = rng.standard_normal((self.nh, self.d)).astype(np.float32)
        self.nb = 7
        self.ck = rng.standard_normal((self.nb, self.d)).astype(np.float32)

    def test_gemma_and_rope(self):
        x = self.rng.standard_normal((5, self.d)).astype(np.float32)
        w = self.rng.standard_normal(self.d).astype(np.float32) * 0.2
        np.testing.assert_allclose(q4.gemma_rmsnorm(x, w),
                                   qe.gemma_rmsnorm(x, w), rtol=1e-5,
                                   atol=1e-6)
        pos = np.arange(5)
        half = self.rot // 2
        ang = pos[:, None] / (1e4 ** (np.arange(half)[None, :] / half))
        cos2 = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)
        sin2 = np.concatenate([np.sin(ang), np.sin(ang)], axis=1)
        got = q4.rope_partial(x, cos2, sin2, self.rot)
        want = qe.apply_rope_lastdim(
            x[:, None, :], cos2[:, None, :], sin2[:, None, :], self.rot)[:, 0]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_pool_block(self):
        tok = self.rng.standard_normal((8, self.d)).astype(np.float32)
        got = q4.qsa_pool_block(tok, 6, 4)
        np.testing.assert_allclose(got, qe.qsa_average_pool_keys(tok[2:6]),
                                   rtol=1e-6, atol=1e-7)

    def test_mqa_logits_and_topk(self):
        starts = np.zeros(1, np.int32)
        ends = np.array([self.nb], np.int32)
        scale = float(np.sqrt(self.d))
        got = q4.qsa_mqa_logits(self.q, self.ck, 0, self.nb, scale)
        want = qe.qsa_mqa_logits(self.q, self.ck, starts, ends)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        topk = q4.qsa_topk(got[None, :], 0, self.nb, 3)
        want_t = qe.qsa_fast_topk(want[None, :], starts, ends, 3)
        np.testing.assert_array_equal(topk, want_t)

    def test_sparse_attention_decode(self):
        nh, kvh, hd, S, W = 4, 2, 8, 6, 5
        q = self.rng.standard_normal((nh, hd)).astype(np.float32)
        k_tok = self.rng.standard_normal((S, kvh, hd)).astype(np.float32)
        v_tok = self.rng.standard_normal((S, kvh, hd)).astype(np.float32)
        slots = np.array([0, 2, 4, -1, 5], np.int32)
        k_head = np.ascontiguousarray(k_tok.transpose(1, 0, 2))
        v_head = np.ascontiguousarray(v_tok.transpose(1, 0, 2))
        scale = float(hd ** -0.5)
        got = q4.qsa_sparse_attn(q, k_head, v_head, slots, kvh, S, scale)
        want = qe.qsa_sparse_attention(
            q[None, :, :], k_tok, v_tok, slots[None, :], scale)[0]
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
