"""A6 qwen4_exp MTP oracle: hybrid 2-fc fusion formula, draft step shapes,
and the spec-decode correctness gate (greedy identity, bit-for-bit)."""

from __future__ import annotations

import unittest

import numpy as np

from ref import qwen4_exp_mtp as qm
from ref import qwen4_exp_pipeline as qp
from runtime import spec as rspec
from serving.dev_model import (
    tiny_qwen4_exp_cfg,
    tiny_qwen4_exp_mtp_weights,
    tiny_qwen4_exp_weights,
)


def _setup(seed=5):
    rng = np.random.default_rng(seed)
    cfg = tiny_qwen4_exp_cfg()
    w = tiny_qwen4_exp_mtp_weights(cfg, tiny_qwen4_exp_weights(cfg, rng), rng)
    return cfg, w


class TestFusion(unittest.TestCase):
    def test_hybrid_two_fc_formula(self):
        cfg, w = _setup()
        mtp = qm.Qwen4ExpMTP(w, cfg)
        self.assertTrue(mtp.hybrid)
        s, h, hc = 4, cfg.hidden, cfg.hc_count
        emb = np.random.default_rng(1).standard_normal((s, h)).astype(np.float32)
        hyp = np.random.default_rng(2).standard_normal((s, hc * h)).astype(
            np.float32)
        eps = cfg.rms_norm_eps
        e = qm.qe.gemma_rmsnorm(emb, w["mtp.pre_fc_norm_embedding.weight"],
                                eps) @ w["mtp.fc_embedding.weight"].T
        hv = qm.qe.gemma_rmsnorm(hyp, w["mtp.pre_fc_norm_hidden.weight"], eps)
        enc = hv.reshape(s, hc, h) @ w["mtp.fc_hidden.weight"].T
        want = (e[:, None, :] + enc).reshape(s, hc * h)
        np.testing.assert_allclose(mtp.fuse(emb, hyp), want, rtol=1e-6,
                                   atol=1e-7)

    def test_standard_single_fc_fallback(self):
        cfg, w = _setup()
        cfg1 = qm.replace(cfg, hc_count=1)
        rng = np.random.default_rng(3)
        w2 = dict(w)  # hc_count == 1 -> hidden-norm is over H, not hc*H
        w2["mtp.pre_fc_norm_hidden.weight"] = \
            rng.standard_normal(cfg.hidden, dtype=np.float32) * 0.05
        w2["mtp.fc.weight"] = rng.standard_normal(
            (cfg.hidden, 2 * cfg.hidden), dtype=np.float32) * 0.1
        mtp = qm.Qwen4ExpMTP(w2, cfg1)
        self.assertFalse(mtp.hybrid)
        emb = rng.standard_normal((2, cfg.hidden), dtype=np.float32)
        hyp = emb.copy()  # hc_count == 1 -> hyper is (S, H)
        eps = cfg.rms_norm_eps
        cat = np.concatenate([
            qm.qe.gemma_rmsnorm(emb, w2["mtp.pre_fc_norm_embedding.weight"],
                                eps),
            qm.qe.gemma_rmsnorm(hyp, w2["mtp.pre_fc_norm_hidden.weight"], eps),
        ], axis=-1)
        np.testing.assert_allclose(mtp.fuse(emb, hyp),
                                   cat @ w2["mtp.fc.weight"].T,
                                   rtol=1e-6, atol=1e-7)


class TestDraft(unittest.TestCase):
    def test_step_shapes_positions_chain(self):
        cfg, w = _setup()
        mtp = qm.Qwen4ExpMTP(w, cfg)
        st = qp.Qwen4ExpState(mtp.mcfg)
        st.n_ctx = 7
        mtp.state = st
        emb_w = w["model.language_model.embed_tokens.weight"]
        hyper = np.random.default_rng(4).standard_normal(
            cfg.hc_count * cfg.hidden).astype(np.float32)
        logits, hyper_out = mtp.step(emb_w[3], hyper)
        self.assertEqual(logits.shape, (32,))
        self.assertEqual(hyper_out.shape, (cfg.hc_count * cfg.hidden,))
        self.assertEqual(mtp.state.n_ctx, 8)      # position continued
        logits2, _ = mtp.step(emb_w[5], hyper_out)
        self.assertEqual(mtp.state.n_ctx, 9)
        self.assertTrue(np.all(np.isfinite(logits)) and
                        np.all(np.isfinite(logits2)))

    def test_weights_map_overwrites_layer0_and_mixer(self):
        cfg, w = _setup()
        m = qm.mtp_weights_map(w)
        for key in ("model.language_model.layers.0.self_attn.q_proj.weight",
                    "model.language_model.hyper_connection_mixer.hc_norm.weight"):
            self.assertIn(key, m)
        self.assertIs(m["model.language_model.layers.0.self_attn.q_proj.weight"],
                      w["mtp.layers.0.self_attn.q_proj.weight"])


class TestGreedyIdentity(unittest.TestCase):
    """The spec-decode correctness gate: acceptance can only SKIP work,
    never change the emitted sequence."""

    def _run(self, seed):
        rng = np.random.default_rng(seed)
        cfg = tiny_qwen4_exp_cfg()
        w = tiny_qwen4_exp_mtp_weights(cfg, tiny_qwen4_exp_weights(cfg, rng),
                                       rng)
        ids = [1, 5, 9, 3, 27, 2, 8]
        want = qp.generate_greedy(ids, w, cfg, 10)
        got, drafted, accepted = qm.generate_greedy_mtp(ids, w, cfg, 10,
                                                        num_draft=2)
        self.assertEqual(got, want)
        self.assertGreater(drafted, 0)
        return accepted

    def test_greedy_identity_seeds(self):
        accepted_any = [self._run(seed) for seed in (5, 6, 7)]
        self.assertGreater(sum(accepted_any), 0)  # loop really exercises both
                                                  # accept and reject paths

    def test_max_new_boundary_never_overshoots(self):
        # a draft accepted exactly AT the limit must stop the draft loop
        # (regression: seeds 2/14 emitted max_new+1 tokens before the guard;
        # the stale `pending` re-emitted the just-accepted token).
        cfg = tiny_qwen4_exp_cfg()
        for seed in (2, 5, 14):
            rng = np.random.default_rng(seed)
            w = tiny_qwen4_exp_mtp_weights(cfg, tiny_qwen4_exp_weights(cfg,
                                                                       rng),
                                           rng)
            ids = [1, 5, 9, 3, 27, 2, 8]
            for max_new in range(1, 9):
                for nd in (1, 2, 3):
                    got, _, _ = qm.generate_greedy_mtp(ids, w, cfg, max_new,
                                                       num_draft=nd)
                    self.assertLessEqual(len(got), max_new,
                                         f"seed={seed} mn={max_new} nd={nd}")
                    self.assertEqual(got,
                                     qp.generate_greedy(ids, w, cfg,
                                                        max_new))


    def test_num_draft_one(self):
        rng = np.random.default_rng(8)
        cfg = tiny_qwen4_exp_cfg()
        w = tiny_qwen4_exp_mtp_weights(cfg, tiny_qwen4_exp_weights(cfg, rng),
                                       rng)
        ids = [4, 4, 12, 30]
        got, _, _ = qm.generate_greedy_mtp(ids, w, cfg, 6, num_draft=1)
        self.assertEqual(got, qp.generate_greedy(ids, w, cfg, 6))

    def test_mtp_greedy_identity_with_ple_main(self):
        # NEW (Phase 6): the MAIN model runs PLE (hyper += ngram feature on
        # layer 2) while the MTP draft driver is PLE-free (mcfg clears
        # ple_layer_ids) -- the two hyper streams must stay consistent across
        # the fused boundary (decode_step_full returns the PLE-augmented
        # post-step hyper row that the draft fuses).
        from serving.dev_model import tiny_qwen4_exp_ple_cfg
        rng = np.random.default_rng(9)
        cfg = tiny_qwen4_exp_ple_cfg()
        w = tiny_qwen4_exp_mtp_weights(cfg, tiny_qwen4_exp_weights(cfg, rng),
                                       rng)
        ids = [3, 11, 1, 7, 12, 3]
        want = qp.generate_greedy(ids, w, cfg, 10)
        got, drafted, accepted = qm.generate_greedy_mtp(ids, w, cfg, 10,
                                                        num_draft=2)
        self.assertEqual(got, want)
        self.assertGreater(drafted, 0)  # draft path actually exercised

    def test_runtime_spec_entry_matches_greedy(self):
        # runtime/spec.py entry wraps the incremental verify loop; the emitted
        # tail must equal plain greedy generation, with PLE in the main model
        from serving.dev_model import tiny_qwen4_exp_ple_cfg
        for make_cfg in (tiny_qwen4_exp_cfg, tiny_qwen4_exp_ple_cfg):
            rng = np.random.default_rng(4)
            cfg = make_cfg()
            w = tiny_qwen4_exp_mtp_weights(cfg, tiny_qwen4_exp_weights(cfg,
                                                                       rng),
                                           rng)
            ids = [3, 11, 1, 7, 12, 3]
            got = rspec.spec_decode_greedy_qwen4exp(cfg, w, ids, max_new=8,
                                                    num_draft=2)
            want = ids + qp.generate_greedy(ids, w, cfg, 8)
            self.assertEqual(got, want)
            self.assertLessEqual(len(got), len(ids) + 8)

    def test_runtime_spec_entry_never_emits_stop(self):
        rng = np.random.default_rng(12)
        cfg = tiny_qwen4_exp_cfg()
        w = tiny_qwen4_exp_mtp_weights(cfg, tiny_qwen4_exp_weights(cfg, rng),
                                       rng)
        ids = [1, 5, 9, 3, 27, 2, 8]
        stop = (27, 9)
        got = rspec.spec_decode_greedy_qwen4exp(cfg, w, ids, max_new=20,
                                                num_draft=2, stop_ids=stop)
        self.assertTrue(got[:len(ids)] == ids)
        self.assertFalse(any(t in stop for t in got[len(ids):]),
                         "a stop id was emitted")
        # prompt ids may contain stop ids; only the EMITTED tail is guarded


if __name__ == "__main__":
    unittest.main()
