"""T0 parity tests for ref/qwen4_exp.py (oracle = upstream sglang formulas).

Synthetic tensors only; CPU-only. Each test recomputes the expected value
with an independent hand-rolled expression of the cited upstream math.
"""

from __future__ import annotations

import unittest

import numpy as np

from ref import qwen4_exp as qe


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class TestNorms(unittest.TestCase):
    def test_gemma_rmsnorm(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 8)).astype(np.float32)
        w = rng.normal(size=(8,)).astype(np.float32) * 0.1
        got = qe.gemma_rmsnorm(x, w, 1e-6)
        xf = x.astype(np.float64)
        want = xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + 1e-6)
        want = want * (1.0 + w.astype(np.float64))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_grouped_gemma_rmsnorm(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 12)).astype(np.float32)
        w = rng.normal(size=(12,)).astype(np.float32) * 0.1
        got = qe.grouped_gemma_rmsnorm(x, w, group_size=4, eps=1e-6)
        xf = x.astype(np.float64).reshape(2, 3, 4)
        vn = xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + 1e-6)
        want = vn.reshape(2, 12) * (1.0 + w.astype(np.float64))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


class TestHyperConnection(unittest.TestCase):
    def setUp(self):
        self.hc, self.hs, self.lr = 4, 16, 8
        rng = np.random.default_rng(2)
        self.hyper = rng.normal(size=(3, self.hc * self.hs)).astype(np.float32)
        self.norm_w = rng.normal(size=(self.hc * self.hs,)).astype(np.float32) * 0.2
        self.down = rng.normal(size=(self.lr, self.hc * self.hs)).astype(np.float32)
        self.up = rng.normal(size=(self.hc * self.hs, self.lr)).astype(np.float32)
        self.inject = rng.normal(size=(self.hc, self.hc * self.hs)).astype(np.float32)

    def test_mix_matches_hand_math(self):
        mixed, normed = qe.hc_mix(
            self.hyper, self.norm_w, self.down, self.up, self.hc, self.hs
        )
        nrm = qe.grouped_gemma_rmsnorm(self.hyper, self.norm_w, self.hs, 1e-6)
        np.testing.assert_allclose(normed, nrm, rtol=0, atol=0)
        low = nrm @ self.down.T / self.hc
        low = low / (1.0 + np.exp(-low))  # silu
        w = 1.0 / (1.0 + np.exp(-(low @ self.up.T)))
        want = (
            w.reshape(3, self.hc, self.hs) * nrm.reshape(3, self.hc, self.hs)
        ).mean(axis=1)
        np.testing.assert_allclose(mixed, want, rtol=1e-5, atol=1e-6)
        self.assertEqual(mixed.shape, (3, self.hs))

    def test_zero_weight_invariants(self):
        z = np.zeros
        mixed, normed = qe.hc_mix(
            self.hyper, z((self.hc * self.hs,)), z((self.lr, self.hc * self.hs)),
            z((self.hc * self.hs, self.lr)), self.hc, self.hs,
        )
        # silu(0)=0 -> up(0)=0 -> sigmoid(0)=0.5 -> mixed = mean(0.5*normed)
        np.testing.assert_allclose(
            mixed, 0.5 * normed.reshape(3, self.hc, self.hs).mean(axis=1),
            rtol=1e-6, atol=1e-7,
        )
        block = np.ones((3, self.hs), dtype=np.float32)
        out = qe.hc_combine(block, self.hyper, normed,
                            z((self.hc, self.hc * self.hs)), self.hc, self.hs)
        # inject=0 -> 2*sigmoid(0)=1 -> every branch receives the block once
        want = self.hyper.reshape(3, self.hc, self.hs) + block[:, None, :]
        np.testing.assert_allclose(out, want.reshape(3, self.hc * self.hs),
                                   rtol=0, atol=1e-6)


class TestMoE(unittest.TestCase):
    def test_route_renormalizes_and_orders(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(5, 8)).astype(np.float32)
        w, ids = qe.moe_route(logits, 3)
        p = _softmax(logits.astype(np.float64))
        for r in range(5):
            self.assertEqual(ids[r, 0], np.argmax(p[r]))
            order = np.argsort(-p[r], kind="stable")[:3]
            np.testing.assert_array_equal(ids[r], order)
            wr = p[r, order] / (p[r, order].sum() + 1e-20)
            np.testing.assert_allclose(w[r], wr, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(w[r].sum(), 1.0, rtol=1e-6)

    def test_route_ties_take_lowest_ids(self):
        logits = np.zeros((1, 6), dtype=np.float32)
        w, ids = qe.moe_route(logits, 3)
        np.testing.assert_array_equal(ids[0], [0, 1, 2])
        np.testing.assert_allclose(w[0], [1 / 3, 1 / 3, 1 / 3], rtol=1e-6)

    def test_block_matches_manual_loop(self):
        rng = np.random.default_rng(4)
        n, h, e, i, ts = 4, 6, 5, 3, 3
        x = rng.normal(size=(n, h)).astype(np.float32)
        router = rng.normal(size=(e, h)).astype(np.float32)
        eg = rng.normal(size=(e, i, h)).astype(np.float32) * 0.3
        eu = rng.normal(size=(e, i, h)).astype(np.float32) * 0.3
        ed = rng.normal(size=(e, h, i)).astype(np.float32) * 0.3
        sg = rng.normal(size=(ts, h)).astype(np.float32) * 0.3
        su = rng.normal(size=(ts, h)).astype(np.float32) * 0.3
        sd = rng.normal(size=(h, ts)).astype(np.float32) * 0.3
        sgl = rng.normal(size=(1, h)).astype(np.float32)
        got = qe.moe_block_forward(x, router, eg, eu, ed, sg, sgl, su, sd, top_k=2)

        p = _softmax(x @ router.T)
        want = np.zeros((n, h), dtype=np.float64)
        for r in range(n):
            ids = np.argsort(-p[r], kind="stable")[:2]
            wr = p[r, ids] / (p[r, ids].sum() + 1e-20)
            for eid, wgt in zip(ids, wr):
                g = (x[r] @ eg[eid].T) / (1.0 + np.exp(-(x[r] @ eg[eid].T)))
                u = x[r] @ eu[eid].T
                want[r] += wgt * ((g * u) @ ed[eid].T)
            gs = x[r] @ sg.T
            us = x[r] @ su.T
            sh = ((gs / (1.0 + np.exp(-gs))) * us) @ sd.T
            gate = 1.0 / (1.0 + np.exp(-(x[r] @ sgl.T)))
            want[r] += gate * sh
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


class TestQsa(unittest.TestCase):
    def test_mqa_logits_manual(self):
        rng = np.random.default_rng(5)
        q = rng.normal(size=(2, 4, 8)).astype(np.float32)
        k = rng.normal(size=(6, 8)).astype(np.float32)
        starts = np.array([0, 2], dtype=np.int32)
        ends = np.array([6, 6], dtype=np.int32)
        got = qe.qsa_mqa_logits(q, k, starts, ends)
        for r in range(2):
            for c in range(6):
                if starts[r] <= c < ends[r]:
                    s = np.maximum(q[r].astype(np.float64) @ k[c].astype(np.float64), 0)
                    np.testing.assert_allclose(
                        got[r, c], s.sum() / np.sqrt(8), rtol=1e-5, atol=1e-6)
                else:
                    self.assertEqual(got[r, c], -np.inf)

    def test_fast_topk_pads_and_relativizes(self):
        logits = np.full((1, 8), -np.inf, dtype=np.float32)
        logits[0, 2:5] = [0.5, 0.9, 0.7]
        starts = np.array([2], dtype=np.int32)
        ends = np.array([5], dtype=np.int32)
        out = qe.qsa_fast_topk(logits, starts, ends, 4)
        np.testing.assert_array_equal(out[0], [1, 2, 0, -1])  # relative order

    def test_expand_block_indices_hand_example(self):
        blocks = np.array([[1, 0]], dtype=np.int32)
        got = qe.qsa_expand_block_indices(blocks, np.array([9]), np.array([10]),
                                          compress_ratio=4, token_topk=8)
        self.assertEqual(got.shape, (1, 11))
        # visible=10 -> tail_start=8, tail_count=2 -> tail [8, 9]; width 11
        want = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1]
        np.testing.assert_array_equal(got[0], want)

    def test_expand_filters_invisible_and_compacts(self):
        # block_topk 2 but only blocks of a 6-token seq; block 0 full, and an
        # out-of-range block index (2 -> tokens 8..11) must be dropped.
        blocks = np.array([[2, 0]], dtype=np.int32)
        got = qe.qsa_expand_block_indices(blocks, np.array([5]), np.array([6]),
                                          compress_ratio=4, token_topk=8)
        toks = sorted(int(t) for t in got[0] if t >= 0)
        self.assertEqual(toks, [0, 1, 2, 3, 4, 5])
        # valid entries must be compacted in front
        first_pad = int(np.argmax(got[0] < 0))
        self.assertTrue(np.all(got[0, first_pad:] < 0))

    def test_select_tokens_covers_full_prefix(self):
        rng = np.random.default_rng(6)
        ratio, budget, heads, dim = 4, 16, 4, 8
        seqlen = 16  # 4 compressed blocks, budget >= coverage
        q = rng.normal(size=(4, heads, dim)).astype(np.float32)
        ck = rng.normal(size=(seqlen // ratio, dim)).astype(np.float32)
        qpos = np.array([3, 7, 11, 15], dtype=np.int32)
        ends = ((qpos + 1) // ratio).astype(np.int32)
        starts = np.zeros(4, dtype=np.int32)
        sel = qe.qsa_select_tokens(q, ck, starts, ends, qpos,
                                   np.full(4, seqlen, dtype=np.int32),
                                   ratio, budget)
        self.assertEqual(sel.shape, (4, budget + ratio - 1))
        for i, p in enumerate(qpos):
            toks = sorted(int(t) for t in sel[i] if t >= 0)
            self.assertEqual(toks, list(range(int(p) + 1)))

    def test_sparse_attention_manual(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(1, 4, 8)).astype(np.float32)
        kc = rng.normal(size=(6, 2, 8)).astype(np.float32)
        vc = rng.normal(size=(6, 2, 8)).astype(np.float32)
        slots = np.array([[0, 2, 5, -1]], dtype=np.int32)
        got = qe.qsa_sparse_attention(q, kc, vc, slots)
        ks = np.repeat(kc[[0, 2, 5]], 2, axis=1).astype(np.float64)
        vs = np.repeat(vc[[0, 2, 5]], 2, axis=1).astype(np.float64)
        s = np.einsum("hd,khd->hk", q[0].astype(np.float64), ks) / np.sqrt(8)
        e = np.exp(s - s.max(-1, keepdims=True))
        want = np.einsum("hk,khd->hd", e / e.sum(-1, keepdims=True), vs)
        np.testing.assert_allclose(got[0], want, rtol=1e-4, atol=1e-6)

    def test_sparse_attention_empty_row_is_zero(self):
        q = np.ones((1, 2, 4), dtype=np.float32)
        kc = np.ones((3, 1, 4), dtype=np.float32)
        got = qe.qsa_sparse_attention(q, kc, kc, np.array([[-1, -1]], dtype=np.int32))
        np.testing.assert_array_equal(got[0], 0.0)

    def test_index_project_qk_rope_tail_passthrough(self):
        rng = np.random.default_rng(8)
        tok, nh, hd, rot = 3, 4, 8, 4
        hidden = rng.normal(size=(tok, 32)).astype(np.float32)
        qk_w = rng.normal(size=((nh + 1) * hd, 32)).astype(np.float32) * 0.2
        qn = np.zeros(hd, dtype=np.float32)
        kn = np.zeros(hd, dtype=np.float32)
        pos = np.arange(tok)
        # zero-angle rotary: cos=1, sin=0 -> identity rotation, tail preserved
        cos, sin = np.ones((tok, rot)), np.zeros((tok, rot))
        q, tk = qe.qsa_index_project_qk(hidden, qk_w, qn, kn, cos, sin, rot, nh, hd)
        raw = (hidden @ qk_w.T)[:, : nh * hd].reshape(tok, nh, hd)
        nrm = raw / np.sqrt((raw.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(q[..., rot:], nrm[..., rot:], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(q[..., :rot], nrm[..., :rot], rtol=1e-5, atol=1e-6)
        self.assertEqual(tk.shape, (tok, 1, hd))

    def test_average_pool(self):
        g = np.stack([np.ones((2, 4)), np.full((2, 4), 3.0)], axis=1)
        np.testing.assert_allclose(qe.qsa_average_pool_keys(g),
                                   np.full((2, 4), 2.0), rtol=0, atol=0)


class TestApplyRope(unittest.TestCase):
    def test_matches_manual_neox_pair(self):
        rng = np.random.default_rng(9)
        tok, heads, dim, rot = 2, 3, 8, 6
        x = rng.normal(size=(tok, heads, dim)).astype(np.float32)
        theta = 10000.0
        inv = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
        pos = np.array([1.0, 5.0])
        ang = pos[:, None] * inv[None, :]
        cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
        sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
        got = qe.apply_rope_lastdim(x, cos, sin, rot)
        half = rot // 2
        x1 = x[..., :half].astype(np.float64)
        x2 = x[..., half:rot].astype(np.float64)
        c, s = np.cos(ang).astype(np.float64), np.sin(ang).astype(np.float64)
        r1 = x1 * c[:, None, :] - x2 * s[:, None, :]
        r2 = x2 * c[:, None, :] + x1 * s[:, None, :]
        want = np.concatenate([r1, r2, x[..., rot:].astype(np.float64)], axis=-1)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
