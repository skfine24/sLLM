"""Q2 pipeline parity tests: qwen4_exp prefill vs incremental decode.

Tiny fixture only (CPU, numpy). Gates (docs/design/04 tiers):
- T1-ish: prefill last-row logits == token-by-token decode logits
  (batched GDN-chunked + batched QSA vs recurrent/stepwise state machines)
- indexer state machine: prefix stability of compressed keys across runs
"""

from __future__ import annotations

import unittest

import numpy as np

from ref import qwen4_exp_pipeline as qp
from serving.dev_model import tiny_qwen4_exp_cfg, tiny_qwen4_exp_weights


class TestQwen4ExpPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_qwen4_exp_cfg()
        cls.w = tiny_qwen4_exp_weights(cls.cfg)
        cls.ids = [5, 9, 1, 7, 12, 3]

    def test_prefill_shape_finite_deterministic(self):
        _, lg1 = qp.prefill(self.ids, self.w, self.cfg)
        _, lg2 = qp.prefill(self.ids, self.w, self.cfg)
        self.assertEqual(lg1.shape, (len(self.ids), 32))
        self.assertTrue(np.isfinite(lg1).all())
        np.testing.assert_array_equal(lg1, lg2)

    def test_prefill_decode_parity(self):
        state, logits = qp.prefill(self.ids, self.w, self.cfg)
        want = logits[-1]
        st2 = qp.Qwen4ExpState(self.cfg)
        got = None
        for tid in self.ids:
            got = qp.decode_step(st2, self.w, self.cfg, tid)
        self.assertEqual(st2.n_ctx, state.n_ctx)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=2e-4)
        self.assertEqual(int(np.argmax(got)), int(np.argmax(want)))

    def test_indexer_prefix_stability(self):
        # layer outputs for tokens < 6 cannot depend on token 7 (causal), so
        # the compressed indexer cache of the 7-token run must match the
        # first 3 blocks of the 8-token run exactly (same fp32 ops).
        st7, _ = qp.prefill(list(range(7)), self.w, self.cfg)
        st8, _ = qp.prefill(list(range(8)), self.w, self.cfg)
        ck7 = st7.layers[2]["ck"]
        ck8 = st8.layers[2]["ck"]
        self.assertEqual(ck7.shape[0], 3)
        self.assertEqual(ck8.shape[0], 4)
        np.testing.assert_allclose(ck8[:3], ck7, rtol=0, atol=1e-6)
        self.assertEqual(st7.layers[2]["tok_k"].shape[0], 7)

    def test_kv_and_indexer_incremental_growth(self):
        st = qp.Qwen4ExpState(self.cfg)
        for t, tid in enumerate(self.ids):
            qp.decode_step(st, self.w, self.cfg, tid)
            l = st.layers[2]
            self.assertEqual(l["k"].shape[1], t + 1)
            self.assertEqual(l["tok_k"].shape[0], t + 1)
            self.assertEqual(l["ck"].shape[0], (t + 1) // 2)

    def test_generate_greedy_in_range(self):
        out = qp.generate_greedy(self.ids[:3], self.w, self.cfg, max_new=6)
        self.assertEqual(len(out), 6)
        self.assertTrue(all(0 <= t < 32 for t in out))

    def test_gdn_layers_carry_state(self):
        st, _ = qp.prefill(self.ids, self.w, self.cfg)
        for i in (0, 1):
            self.assertIsNotNone(st.layers[i]["gdn"])
            self.assertEqual(st.layers[i]["conv_win"].shape[2],
                             self.cfg.lin_conv - 1)

    def test_ple_guard(self):
        cfg = tiny_qwen4_exp_cfg()
        cfg.ple_layer_ids = (2,)
        with self.assertRaises(NotImplementedError):
            qp.prefill(self.ids, self.w, cfg)


if __name__ == "__main__":
    unittest.main()
