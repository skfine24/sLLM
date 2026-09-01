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

    def test_ple_prefill_decode_parity(self):
        from serving.dev_model import tiny_qwen4_exp_ple_cfg
        cfg = tiny_qwen4_exp_ple_cfg()
        w = tiny_qwen4_exp_weights(cfg)
        ids = [3, 11, 1, 7, 12, 3]
        state, logits = qp.prefill(ids, w, cfg)
        want = logits[-1]
        st2 = qp.Qwen4ExpState(cfg)
        got = None
        for tid in ids:
            got = qp.decode_step(st2, w, cfg, tid)
        self.assertEqual(st2.n_ctx, state.n_ctx)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=2e-4)
        self.assertEqual(int(np.argmax(got)), int(np.argmax(want)))

    def test_ple_changes_logits(self):
        from serving.dev_model import tiny_qwen4_exp_ple_cfg
        ple_cfg = tiny_qwen4_exp_ple_cfg()
        # same seed ⇒ w_ple main tensors == base weights + PLE extras
        w_ple = tiny_qwen4_exp_weights(ple_cfg)
        _, lg_ple = qp.prefill([3, 11, 1, 7], w_ple, ple_cfg)
        base_cfg = tiny_qwen4_exp_cfg()
        # main tensors only (no PLE tensors)
        main = {k: v for k, v in w_ple.items() if ".ple." not in k}
        _, lg_base = qp.prefill([3, 11, 1, 7], main, base_cfg)
        self.assertFalse(np.allclose(lg_ple, lg_base, atol=0.0),
                         "PLE must change the hyper stream / logits")

    def test_ple_incremental_batched_same_as_sequential(self):
        from serving.dev_model import tiny_qwen4_exp_ple_cfg
        cfg = tiny_qwen4_exp_ple_cfg()
        w = tiny_qwen4_exp_weights(cfg)
        ids = [3, 11, 8, 12]
        _, _, hyper_d = qp.prefill(ids, w, cfg, return_hyper=True)
        self.assertEqual(hyper_d.shape[1], len(ids))
        stb = qp.Qwen4ExpState(cfg)
        _, lg_batch = qp._forward(stb, w, cfg, hyper_in=hyper_d,
                                  ple_input_ids=np.array([ids]))
        sts = qp.Qwen4ExpState(cfg)
        lg_seq = []
        for p in range(len(ids)):
            _, lg_p = qp._forward(sts, w, cfg, hyper_in=hyper_d[:, p:p + 1],
                                  ple_input_ids=np.array([[ids[p]]]))
            lg_seq.append(lg_p[0])
        np.testing.assert_allclose(lg_batch, np.stack(lg_seq), rtol=0, atol=1e-4)
        self.assertEqual(stb.n_ctx, sts.n_ctx)

    def test_batched_hyper_injection_matches_sequential(self):
        import copy
        draft_ids = [3, 11, 2, 8]
        # the fused draft input: pre-final hyper tensor (1, S, hc*H) for the
        # draft positions, exactly what the MTP path feeds to `_forward`
        _, _, hyper_d = qp.prefill(draft_ids, self.w, self.cfg,
                                   return_hyper=True)
        self.assertEqual(hyper_d.shape[1], len(draft_ids))

        # batched draft-extend: one fused call with all S positions
        stb = qp.Qwen4ExpState(self.cfg)
        _, lg_batch, hyper_out = qp._forward(stb, self.w, self.cfg,
                                             hyper_in=hyper_d, return_hyper=True)

        # sequential reference: S single-token hyper-injections
        sts = qp.Qwen4ExpState(self.cfg)
        lg_seq = []
        for p in range(hyper_d.shape[1]):
            _, lg_p = qp._forward(sts, self.w, self.cfg,
                                  hyper_in=hyper_d[:, p:p + 1])
            lg_seq.append(lg_p[0])
        lg_seq = np.stack(lg_seq, axis=0)

        self.assertEqual(stb.n_ctx, sts.n_ctx)
        np.testing.assert_allclose(lg_batch, lg_seq, rtol=0, atol=1e-6)
        # identical cache growth per layer type
        for i, bt in enumerate(self.cfg.layer_types):
            if bt == "linear_attention":
                for key in ("gdn", "conv_win"):
                    self.assertEqual(stb.layers[i][key].shape,
                                     sts.layers[i][key].shape)
            else:
                for key in ("k", "v", "tok_k", "ck"):
                    self.assertEqual(stb.layers[i][key].shape,
                                     sts.layers[i][key].shape)
        # argmax agreement on the final draft position
        self.assertEqual(int(np.argmax(lg_batch[-1])),
                         int(np.argmax(lg_seq[-1])))


if __name__ == "__main__":
    unittest.main()
