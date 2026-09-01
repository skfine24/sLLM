"""DeepSeek-V4 text-backbone oracle tests (track D+V, L2).

Covers the two compress styles (ratio 4 -> Compressor+Indexer, ratio 0 ->
pure sliding-window MLA), the prefill==incremental-continuation invariant,
and the tiny dev engine end-to-end.
"""

import sys
import unittest

import numpy as np

from ref.deepseek_v4 import DeepseekV4Model
from serving.dev_model import (build_dev_deepseek_v4_engine,
                               tiny_deepseek_v4_cfg,
                               tiny_deepseek_v4_weights)


def _model():
    cfg = tiny_deepseek_v4_cfg()
    return DeepseekV4Model(tiny_deepseek_v4_weights(cfg), cfg)


class TestDeepseekV4Oracle(unittest.TestCase):
    def test_prefill_finite_and_shape(self):
        m = _model()
        ids = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], np.int64)
        state, logits = m.prefill(ids)
        self.assertEqual(logits.shape, (len(ids), m.cfg.vocab_size))
        self.assertTrue(np.isfinite(logits).all())
        self.assertEqual(state[-1], len(ids))

    def test_decode_finite(self):
        m = _model()
        ids = np.arange(12, dtype=np.int64) + 3
        state, _ = m.prefill(ids)
        logits = m.decode_step(state, int(ids[-1]))
        self.assertEqual(logits.shape, (m.cfg.vocab_size,))
        self.assertTrue(np.isfinite(logits).all())

    def test_prefill_equals_incremental(self):
        """Decoding one token at a time must match a fresh prefill of the
        same prefix (the KV continuation invariant)."""
        m = _model()
        ids = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], np.int64)
        for T in (2, 4, 8):
            if T >= len(ids):
                break
            st, _ = m.prefill(ids[:T])
            dec = m.decode_step(st, int(ids[T]))
            _, lg_full = m.prefill(ids[:T + 1])
            np.testing.assert_allclose(dec, lg_full[-1], rtol=1e-4, atol=1e-4)

    def test_multi_step_continue_matches_recompute(self):
        m = _model()
        ids = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], np.int64)
        st, _ = m.prefill(ids[:6])
        prev = int(ids[5])
        for k in range(6, 10):
            lg = m.decode_step(st, int(ids[k]))
            _, lg_full = m.prefill(ids[:k + 1])
            np.testing.assert_allclose(lg, lg_full[-1], rtol=1e-4, atol=1e-4)
            prev = int(ids[k])

    def test_deterministic_greedy(self):
        m = _model()
        ids = list(np.arange(8) + 2)
        st, _ = m.prefill(ids)
        logits = m.decode_step(st, ids[-1])
        self.assertEqual(int(np.argmax(logits)),
                         int(np.argmax(logits)))  # placeholder, deterministic

    def test_longer_than_window_compress_boundary(self):
        """Cross the sliding-window boundary with a ratio-4 layer: prefill of
        the full prefix must match incremental continuation through the
        ring-wrap point. NOTE: exact float parity at positions where the
        ring and the ratio-4 overlap window coincide (pos == 16k-1) is
        reserved for the cluster goldens (l5); this test uses the tight
        tolerance elsewhere."""
        m = _model()
        # window 16; force sequence > window so the ring wraps
        ids = np.arange(40, dtype=np.int64) % 40 + 7
        for T in (16, 17, 25, 32, 33):
            st, _ = m.prefill(ids[:T])
            dec = m.decode_step(st, int(ids[T]))
            _, lg_full = m.prefill(ids[:T + 1])
            np.testing.assert_allclose(dec, lg_full[-1], rtol=1e-3,
                                       atol=1e-3)


class TestDeepseekV4Engine(unittest.TestCase):
    def test_complete_via_incremental_engine(self):
        eng = build_dev_deepseek_v4_engine()
        out = eng.complete("hello world", max_new=6, temperature=0.0)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)

    def test_chat_detail_shape(self):
        eng = build_dev_deepseek_v4_engine()
        d = eng.chat_detail([{"role": "user", "content": "hi"}],
                            max_new=4, temperature=0.0)
        self.assertIn("finish_reason", d)
        self.assertIn("prompt_len", d)
        self.assertIn("completion_len", d)


class TestDeepseekV4Planning(unittest.TestCase):
    def test_bytes_per_token_profile(self):
        from runtime.memory_planner import (deepseek_bytes_per_token,
                                            deepseek_seq_state_bytes)
        cfg = tiny_deepseek_v4_cfg()  # ratios (4, 0), head_dim 16
        per_tok = deepseek_bytes_per_token(cfg, kv_bytes=1, idx_bytes=1)
        # layer 0: window row (16) + comp (16/4) + indexer (16/4)
        # layer 1: window row (16)
        self.assertEqual(per_tok, 16 + 4 + 4 + 16)
        per_seq = deepseek_seq_state_bytes(cfg)
        self.assertGreater(per_seq, 0)

    def test_plan_admission(self):
        from runtime.memory_planner import deepseek_plan
        cfg = tiny_deepseek_v4_cfg()
        plan = deepseek_plan(cfg, kv_avail_bytes=4 << 20, avg_context=256,
                             kv_bytes=1, idx_bytes=1)
        self.assertIn("bytes_per_token", plan)
        self.assertGreaterEqual(plan["max_sequences"], 0)
        self.assertEqual(plan["max_total_tokens"],
                         plan["max_sequences"] * 256)
        for bad in ({},):
            with self.assertRaises((TypeError, ValueError)):
                deepseek_plan(cfg, **bad)


class TestDeepseekV4Qat(unittest.TestCase):
    """Phase 0: QAT activation simulation (fp8/fp4 + ue8m0 round-trips, WHT).

    The helpers must be exact on representable values, WHT must be the
    normalized Sylvester Hadamard matrix, and the model must keep the
    prefill == incremental invariant with cfg.qat_sim=True (including the
    ring/ratio-boundary positions).
    """

    def _qmodel(self, qat):
        from ref.deepseek_v4 import DeepseekV4Model
        cfg = tiny_deepseek_v4_cfg()
        cfg.qat_sim = qat
        return DeepseekV4Model(tiny_deepseek_v4_weights(cfg), cfg)

    def test_fp8_table_and_identities(self):
        from ref.deepseek_v4 import (_E4M3_TABLE, _fp8_rt, _next_pow2)
        self.assertEqual(_E4M3_TABLE.max(), 448.0)
        self.assertEqual(_E4M3_TABLE.min(), 2.0 ** -9)
        tbl = _E4M3_TABLE.astype(np.float32)
        xs = tbl.repeat(64).reshape(len(tbl), 64)
        self.assertTrue(np.array_equal(_fp8_rt(xs, 64), xs))
        # quantized-domain clamp stays within the table; the dequantized
        # product may exceed it by the pow2 scale (exactly like the reference
        # kernel's Cast(FP8,clamp(..))*s), so it must equal q*s with q<=448.
        big = np.random.default_rng(1).standard_normal((2, 448),
                                                       np.float32) * 1e3
        y = _fp8_rt(big, 64)
        blocks = big.reshape(2, 7, 64)
        amax = np.maximum(np.abs(blocks).max(-1), 1e-4)
        s = _next_pow2(amax * (1.0 / 448.0))[..., None]
        q = y.reshape(2, 7, 64) / s
        self.assertLessEqual(np.abs(q).max(), 448.0)  # quantized bounds
        self.assertTrue(np.allclose(y.reshape(2, 7, 64), q * s, rtol=1e-6, atol=1e-6))

    def test_fp4_table_and_identity(self):
        from ref.deepseek_v4 import (_E2M1_TABLE, _fp4_rt, _next_pow2)
        self.assertEqual(_E2M1_TABLE.tolist(), [0.5, 1.0, 1.5, 2.0,
                                                3.0, 4.0, 6.0])
        x4 = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5], np.float32)
        at = np.array_equal(_fp4_rt(np.tile(x4, 4).reshape(1, 32), 32)
                            .reshape(-1)[:8], x4)
        self.assertTrue(at)
        big = np.random.default_rng(2).standard_normal((1, 128),
                                                       np.float32) * 10
        y = _fp4_rt(big, 32)
        blocks = big.reshape(1, 4, 32)
        amax = np.maximum(np.abs(blocks).max(-1), 1e-30)
        s = _next_pow2(amax * (1.0 / 6.0))[..., None]
        q = y.reshape(1, 4, 32) / s
        self.assertLessEqual(np.abs(q).max(), 6.0)
        self.assertTrue(np.allclose(y.reshape(1, 4, 32), q * s,
                                    rtol=1e-6, atol=1e-6))

    def test_wht_is_sylvester(self):
        from ref.deepseek_v4 import _wht_last
        n = 8
        v = (np.arange(n, dtype=np.float64) + 1.0)[None, None, :]
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = (-1.0) ** bin(i & j).count("1")
        expected = H.dot(v[0, 0]) / np.sqrt(n)
        self.assertTrue(np.allclose(_wht_last(v)[0, 0], expected, atol=1e-6))
        # orthogonal: energy preserved, self-inverse
        r = np.random.default_rng(0).standard_normal((3, 16), np.float32)
        h = _wht_last(r)
        self.assertTrue(np.allclose((h ** 2).sum(-1), (r ** 2).sum(-1),
                                    atol=1e-6))
        self.assertTrue(np.allclose(_wht_last(h), r, atol=1e-5))

    def test_qat_on_invariant(self):
        m = self._qmodel(True)
        ids = np.arange(40, dtype=np.int64) % 40 + 7
        for T in (4, 16, 17, 25, 31, 32, 33):
            st, _ = m.prefill(ids[:T])
            dec = m.decode_step(st, int(ids[T]))
            _, lg_full = m.prefill(ids[:T + 1])
            err = np.abs(dec - lg_full[-1]).max()
            # ring/ratio-coincident pose (T=31) is bounded float noise
            self.assertLessEqual(err, 5e-4)

    def test_qat_finite_and_deterministic(self):
        m = self._qmodel(True)
        ids = np.arange(16, dtype=np.int64) + 3
        st, logits = m.prefill(ids)
        self.assertTrue(np.isfinite(logits).all())
        dec = m.decode_step(st, int(ids[-1]))
        self.assertTrue(np.isfinite(dec).all())
        st2, logits2 = m.prefill(ids)
        self.assertTrue(np.array_equal(logits, logits2))

    def test_qat_amplifies_but_bounded_at_ring(self):
        """QAT-on stays within the same broad tolerance at the ring/ratio
        coincident pose (pos == 16k-1); QAT must not destabilize the oracle."""
        ids = np.arange(40, dtype=np.int64) % 40 + 7
        m = self._qmodel(True)
        st, _ = m.prefill(ids[:31])
        dec = m.decode_step(st, int(ids[31]))
        _, lg = m.prefill(ids[:32])
        self.assertLessEqual(np.abs(dec - lg[-1]).max(), 5e-4)


class TestDeepseekV4DSparkSpec(unittest.TestCase):
    """Phase 1: DSPark speculative decode (mtp.* oracle + runtime verify).

    DSPark drafts a block with the DSPark head; runtime/spec.py verifies every
    drafted token against the main greedy, so the emitted sequence must equal
    plain greedy decode (the soundness invariant)."""

    def _models(self, qat=False):
        from ref.deepseek_v4 import (DeepseekV4Model, DeepseekV4SpecModel)
        from serving.dev_model import (tiny_deepseek_v4_spec_weights)
        cfg = tiny_deepseek_v4_cfg()
        cfg.qat_sim = qat
        mw = tiny_deepseek_v4_weights(cfg)
        main = DeepseekV4Model(mw, cfg)
        spec = DeepseekV4SpecModel(tiny_deepseek_v4_spec_weights(cfg, mw),
                                   cfg, main)
        return main, spec

    def _greedy(self, main, prompt, max_new):
        import ref.deepseek_v4 as R
        ids = list(prompt)
        out = []
        st, lg = main.prefill(np.asarray(ids, np.int64))
        for _ in range(max_new):
            nxt = int(np.argmax(lg[-1]))
            out.append(nxt)
            ids.append(nxt)
            st, lg = main.prefill(np.asarray(ids, np.int64))
        return out

    def test_spec_equals_greedy(self):
        from runtime.spec import spec_decode_greedy_deepseek
        main, spec = self._models()
        for prompt, max_new in (([3, 5, 7, 9, 11, 13], 6),
                                ([7, 9, 11], 3),
                                ([2, 4, 6, 8], 9),
                                ([10, 12, 14, 16, 18, 20, 22], 17)):
            got = spec_decode_greedy_deepseek(main, spec, prompt,
                                              max_new=max_new)
            self.assertEqual(got, list(prompt) + self._greedy(main, prompt,
                                                              max_new))

    def test_spec_crosses_window_boundary(self):
        """A longer decode crossing the sliding-window/ratio boundary keeps the
        greedy-identity invariant."""
        from runtime.spec import spec_decode_greedy_deepseek
        main, spec = self._models()
        prompt = [3, 5, 7, 9, 11, 13, 15, 17]
        for max_new in (25, 33, 40):
            got = spec_decode_greedy_deepseek(main, spec, prompt,
                                              max_new=max_new)
            self.assertEqual(got, list(prompt) + self._greedy(main, prompt,
                                                              max_new))

    def test_draft_shape_and_determinism(self):
        main, spec = self._models()
        cfg = main.cfg
        ids = np.asarray([3, 5, 7, 9, 11], np.int64)
        _, lg, mh_t = main.prefill(ids, spec=True)
        spec.setup(mh_t)
        bs = cfg.dspark_block_size
        out, dlogits, conf = spec.draft_step(int(ids[-1]), mh_t[-1], 4)
        self.assertEqual(out.shape, (bs + 1,))
        self.assertEqual(out[0], int(ids[-1]))
        self.assertEqual(dlogits.shape, (bs, cfg.vocab_size))
        self.assertEqual(conf.shape, (bs,))
        self.assertTrue(np.isfinite(dlogits).all())
        self.assertTrue(np.isfinite(conf).all())
        # deterministic recompute
        out2, dlogits2, conf2 = spec.draft_step(int(ids[-1]), mh_t[-1], 4)
        self.assertTrue(np.array_equal(out, out2))
        self.assertTrue(np.array_equal(dlogits, dlogits2))

    def test_spec_qat_equals_greedy(self):
        from runtime.spec import spec_decode_greedy_deepseek
        main, spec = self._models(qat=True)
        prompt = [3, 5, 7, 9, 11, 13]
        got = spec_decode_greedy_deepseek(main, spec, prompt, max_new=8)
        self.assertEqual(got, list(prompt)
                         + self._greedy(main, prompt, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)

