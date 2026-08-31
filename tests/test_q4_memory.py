"""A5 qwen4_exp KV/state management: budget planner arithmetic + amortized
Qwen4ExpState growth (capacity-doubling buffers behind exact-length views).

Decode-output correctness under the growth path is pinned by
tests/test_qwen4_engine.py (engine greedy == pipeline) and
tests/test_qwen4_exp_pipeline.py (incremental == full prefill).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from ref import qwen4_exp_pipeline as qp
from recipes.schema import Recipe
from runtime.memory_planner import (
    qwen4_exp_bytes_per_token,
    qwen4_exp_plan,
    qwen4_exp_seq_state_bytes,
)
from serving.dev_model import tiny_qwen4_exp_cfg

RECIPE = os.path.join(os.path.dirname(__file__), "..", "recipes",
                      "Qwen3.8-Flash-Next-FP8.yaml")


class TestQ4Planner(unittest.TestCase):
    def test_per_token_formula(self):
        cfg = tiny_qwen4_exp_cfg()
        n_qsa = cfg.layer_types.count("full_attention")
        want = n_qsa * (2 * cfg.attn_kv_heads * cfg.attn_head_dim * 1
                        + cfg.idx_dim + cfg.idx_dim // cfg.idx_ratio)
        self.assertEqual(qwen4_exp_bytes_per_token(cfg), want)

    def test_per_sequence_state_formula(self):
        cfg = tiny_qwen4_exp_cfg()
        n_gdn = cfg.layer_types.count("linear_attention")
        c = 2 * cfg.lin_k_heads * cfg.lin_k_dim \
            + cfg.lin_v_heads * cfg.lin_v_dim
        want = n_gdn * (cfg.lin_v_heads * cfg.lin_v_dim * cfg.lin_k_dim * 4
                        + c * (cfg.lin_conv - 1) * 4)
        self.assertEqual(qwen4_exp_seq_state_bytes(cfg), want)

    def test_real_recipe_magnitudes(self):
        with open(RECIPE, encoding="utf-8") as f:
            cfg = qp.Qwen4ExpCfg.from_recipe(Recipe.from_yaml(f.read()))
        self.assertEqual(cfg.layer_types.count("full_attention"), 12)
        self.assertEqual(cfg.layer_types.count("linear_attention"), 36)
        # fp8 KV + fp8 indexer stream: 12 * (1024 + 128 + 32)
        self.assertEqual(qwen4_exp_bytes_per_token(cfg), 12 * 1184)
        # fp32 GDN state ~113 MB per sequence (admission must count it)
        per_seq = qwen4_exp_seq_state_bytes(cfg)
        self.assertGreater(per_seq, 100 * 1024 ** 2)
        self.assertLess(per_seq, 130 * 1024 ** 2)

    def test_plan_counts_state_not_just_tokens(self):
        cfg = tiny_qwen4_exp_cfg()
        per_tok = qwen4_exp_bytes_per_token(cfg)
        per_seq = qwen4_exp_seq_state_bytes(cfg)
        avg = 128
        budget = 7 * (per_seq + per_tok * avg)
        plan = qwen4_exp_plan(cfg, budget, avg, utilization=1.0)
        self.assertEqual(plan["max_sequences"], 7)
        self.assertEqual(plan["max_total_tokens"], 7 * avg)
        # token-only budgeting would admit far more: the state term must bite
        self.assertLess(plan["max_sequences"], budget // (per_tok * avg))
        with self.assertRaises(ValueError):
            qwen4_exp_plan(cfg, budget, 0)


class TestStateGrowth(unittest.TestCase):
    def test_append_axis_semantics_and_amortization(self):
        cfg = tiny_qwen4_exp_cfg()
        st = qp.Qwen4ExpState(cfg)
        lay = st.layers[st.cfg.layer_types.index("full_attention")]
        ref_rows = []
        caps = []
        for step in range(20):
            row = np.full((cfg.attn_kv_heads, 1, cfg.attn_head_dim),
                          step, np.float32)
            qp._append_axis(lay, "k", row, axis=1)
            ref_rows.append(row[0])
            lay["k_buf"]  # buffer-backed after the first append
            caps.append(lay["k_buf"].shape[1])
            # exact-length view with correct content
            self.assertEqual(lay["k"].shape,
                             (cfg.attn_kv_heads, step + 1, cfg.attn_head_dim))
            np.testing.assert_array_equal(
                lay["k"], np.stack(ref_rows, axis=1))
        # capacity doubling: no regrowth, and at most one doubling per append
        self.assertTrue(all(b >= a for a, b in zip(caps, caps[1:])))
        self.assertLessEqual(caps[-1], 2 * 20)

    def test_state_bytes_tracks_buffers(self):
        cfg = tiny_qwen4_exp_cfg()
        st = qp.Qwen4ExpState(cfg)
        base = st.state_bytes()
        lay = st.layers[st.cfg.layer_types.index("full_attention")]
        qp._append_axis(lay, "ck", np.ones((1, cfg.idx_dim), np.float32),
                        axis=0)
        self.assertGreater(st.state_bytes(), base)


if __name__ == "__main__":
    unittest.main()
