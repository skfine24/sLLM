"""A1 engine-integration tests: qwen4_exp through the serving executor.

Tiny fixture only (CPU, numpy). Gates (docs/design/09 Q4-CPU):
- Qwen4ExpCfg.from_recipe reproduces the tiny cfg and the real recipe knobs
- executor.generate() greedy == ref/qwen4_exp_pipeline.generate_greedy
  (bit-identical: the engine adds no extra math)
- BatchedInferenceEngine concurrent decode == sequential generation
- context clamp + PLE guard behaviour
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from ref import qwen4_exp_pipeline as qp
from recipes.schema import Recipe
from serving.dev_model import (
    TinyCharTokenizer,
    build_dev_qwen4_exp_engine,
    tiny_qwen4_exp_cfg,
    tiny_qwen4_exp_recipe,
    tiny_qwen4_exp_weights,
)
from serving.executor import BatchedInferenceEngine, ReferenceModel, generate

RECIPE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "recipes", "Qwen3.8-Flash-Next-FP8.yaml")


class TestCfgFromRecipe(unittest.TestCase):
    def test_tiny_recipe_reproduces_cfg(self):
        self.assertEqual(
            qp.Qwen4ExpCfg.from_recipe(tiny_qwen4_exp_recipe()),
            tiny_qwen4_exp_cfg())

    def test_real_recipe_knobs(self):
        with open(RECIPE_PATH, encoding="utf-8") as f:
            recipe = Recipe.from_yaml(f.read())
        cfg = qp.Qwen4ExpCfg.from_recipe(recipe)
        self.assertEqual(cfg.hidden, 2560)
        self.assertEqual(cfg.hc_count, 4)
        self.assertEqual(cfg.hc_lowrank, 320)
        self.assertEqual(len(cfg.layer_types), 48)
        self.assertEqual(sum(bt == "full_attention" for bt in cfg.layer_types), 12)
        self.assertEqual((cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim),
                         (24, 2, 256))
        self.assertEqual(cfg.rotary_dim, 64)
        self.assertEqual((cfg.idx_heads, cfg.idx_dim, cfg.idx_budget,
                          cfg.idx_ratio), (4, 128, 2048, 4))
        self.assertEqual((cfg.n_experts, cfg.top_k), (512, 10))
        # PLE is deferred to Q5: the real knobs must trip the guard, never
        # silently skip PLE layers.
        self.assertEqual(cfg.ple_layer_ids, (2,))
        with self.assertRaises(NotImplementedError):
            cfg.validate()


class TestEngineGreedyIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_qwen4_exp_cfg()
        cls.w = tiny_qwen4_exp_weights(cls.cfg)
        # cfg-less model: exercises Qwen4ExpCfg.from_recipe on the engine path
        cls.model = ReferenceModel(tiny_qwen4_exp_recipe(), cls.w)
        cls.prompt = [4, 8, 2, 13, 6]

    def test_supports_incremental(self):
        fresh = ReferenceModel(tiny_qwen4_exp_recipe(), self.w)
        self.assertTrue(fresh.supports_incremental)
        self.assertIsNone(fresh.q4_cfg)  # derived lazily, not required
        self.assertEqual(fresh._q4cfg(), self.cfg)

    def test_generate_matches_pipeline(self):
        out = generate(self.model, None, self.prompt, max_new=6,
                       temperature=0.0)
        want = qp.generate_greedy(self.prompt, self.w, self.cfg, 6)
        self.assertEqual(out, self.prompt + want)  # bit-identical token ids

    def test_logits_oracle_last_row(self):
        lg = self.model.logits(self.prompt)
        self.assertEqual(lg.shape, (1, len(self.prompt), 32))
        _, ref = qp.prefill(self.prompt, self.w, self.cfg)
        np.testing.assert_array_equal(lg[0], ref)

    def test_context_clamp(self):
        long_prompt = [3] * 256  # == max_position_embeddings (256)
        with self.assertRaises(ValueError):
            generate(self.model, None, long_prompt, max_new=4,
                     temperature=0.0)


class TestEngineBatch(unittest.TestCase):
    """Continuous batching must not change greedy results (no cross-talk)."""

    @classmethod
    def setUpClass(cls):
        cls.engine = build_dev_qwen4_exp_engine()

    def test_batch_matches_sequential(self):
        prompts = ["abcd", "aabbcad", "zz"]
        eng = BatchedInferenceEngine(self.engine.model, self.engine.tokenizer)
        for p in prompts:
            eng.submit(p, max_new=4, temperature=0.0)
        batched = eng.run_all()
        sequential = [
            self.engine.complete(p, max_new=4, temperature=0.0)
            for p in prompts
        ]
        self.assertEqual(batched, sequential)

    def test_stub_tokenizer_roundtrip(self):
        tok = TinyCharTokenizer()
        self.assertEqual(tok.decode(tok.encode("abcdef")), "abcdef")


if __name__ == "__main__":
    unittest.main()
