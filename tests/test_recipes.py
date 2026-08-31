"""Recipe schema tests: qwen3_5 example recipe loads and validates, and the
schema rejects malformed documents."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recipes.schema import Recipe, RecipeError, MemorySpec  # noqa: E402

RECIPE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recipes", "Qwen3.8-27B-FP8.yaml"))


class TestRecipeQwen3_5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(RECIPE, "r", encoding="utf-8") as f:
            cls.recipe = Recipe.from_yaml(f.read())

    def test_identity(self):
        self.assertEqual(self.recipe.model_id, "Qwen/Qwen3.8-27B-FP8")
        self.assertEqual(self.recipe.arch, "qwen3_5")

    def test_layer_schedule(self):
        self.assertEqual(self.recipe.num_layers, 64)
        self.assertEqual(len(self.recipe.layer_types), 64)
        linear_idx = [i for i in range(64) if i % 4 != 3]
        self.assertEqual(self.recipe.linear_attn_indices(), linear_idx)
        self.assertEqual(self.recipe.full_attn_indices(), [i for i in range(64) if i % 4 == 3])
        self.assertEqual(len(linear_idx), 48)
        self.assertEqual(len(self.recipe.full_attn_indices()), 16)

    def test_quant_spec(self):
        q = self.recipe.quant
        self.assertEqual(q.fmt, "e4m3")
        self.assertEqual(q.activation, "dynamic")
        self.assertEqual(q.weight_block_size, (128, 128))
        self.assertEqual(q.scale_tensor_suffix, "weight_scale_inv")

    def test_attention_knobs(self):
        la = self.recipe.linear_attention
        self.assertEqual((la.num_key_heads, la.key_head_dim), (16, 128))
        self.assertEqual((la.num_value_heads, la.value_head_dim), (48, 128))
        self.assertTrue(la.qk_l2norm)
        fa = self.recipe.full_attention
        self.assertEqual((fa.num_heads, fa.num_kv_heads, fa.head_dim), (24, 4, 256))
        self.assertEqual(self.recipe.rotary_dim(), 64)

    def test_mtp_and_vision(self):
        self.assertTrue(self.recipe.mtp.enabled)
        self.assertEqual(self.recipe.mtp.attention_type, "full")
        self.assertTrue(self.recipe.vision.enabled)
        self.assertEqual(self.recipe.vision.depth, 27)


class TestRecipeValidation(unittest.TestCase):
    @staticmethod
    def _base() -> dict:
        import yaml
        with open(RECIPE, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_bad_layer_count(self):
        doc = self._base()
        doc["text"]["layer_types"] = ["linear_attention"] * 10  # num_layers=64
        with self.assertRaises(RecipeError):
            Recipe.from_dict(doc)

    def test_unknown_layer_type(self):
        doc = self._base()
        doc["text"]["layer_types"] = ["mystery"] * 64
        with self.assertRaises(RecipeError):
            Recipe.from_dict(doc)

    def test_missing_model_id(self):
        doc = self._base()
        del doc["model_id"]
        with self.assertRaises(RecipeError):
            Recipe.from_dict(doc)


class TestSkeletonRecipes(unittest.TestCase):
    """The two non-28b recipes parse+validate as structure skeletons."""

    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recipes"))

    def _load(self, name):
        p = os.path.join(self.BASE, name)
        with open(p, "r", encoding="utf-8") as f:
            return Recipe.from_yaml(f.read())

    def test_qwen4_exp(self):
        r = self._load("Qwen3.8-Flash-Next-FP8.yaml")
        self.assertEqual(r.arch, "qwen4_exp")
        self.assertEqual(r.status, "skeleton")
        self.assertEqual(r.num_layers, 48)
        self.assertEqual(r.layer_types.count("linear_attention"), 36)
        self.assertEqual(r.layer_types.count("qsa_attention"), 12)
        self.assertEqual((r.mlp.type, r.mlp.num_experts, r.mlp.num_experts_per_tok), ("moe", 512, 10))
        self.assertIn("qsa", r.meta["spec"])
        self.assertIn("ngram", r.meta["spec"])

    def test_deepseek_v4(self):
        r = self._load("DeepSeek-V4-Flash-0731.yaml")
        self.assertEqual(r.arch, "deepseek_v4")
        self.assertEqual(r.status, "skeleton")
        self.assertEqual(r.num_layers, 43)
        self.assertTrue(all(t == "mla_attention" for t in r.layer_types))
        self.assertEqual((r.mlp.num_experts, r.mlp.num_experts_per_tok), (256, 6))
        self.assertEqual(r.meta["spec"]["expert_dtype"], "fp4")
        self.assertEqual(r.meta["spec"]["dspark"]["markov_rank"], 256)
        self.assertEqual(r.vocab_size, 129280)

    def test_status_vocabulary_per_capability(self):
        # ready = end-to-end runnable engine; partial = audited recipe,
        # gated real-checkpoint engine; skeleton = structure only.
        r = Recipe.from_yaml(open(RECIPE, encoding="utf-8").read())
        self.assertEqual(r.status, "partial")
        self.assertEqual(len(r.layer_types), 64)
        self.assertEqual(self._load("Qwen2.5-Coder-0.5B.yaml").status, "ready")
        self.assertEqual(self._load("Qwen3.8-Flash-Next-FP8.yaml").status,
                         "skeleton")
        self.assertEqual(self._load("DeepSeek-V4-Flash-0731.yaml").status,
                         "skeleton")

    def test_ready_recipes_carry_local_dir(self):
        for name, expected in {
            "Qwen3.8-27B-FP8.yaml": "Qwen3.8-27B-FP8",
            "Qwen2.5-Coder-0.5B.yaml": "Qwen2.5-Coder-0.5B",
            "Qwen3.8-Flash-Next-FP8.yaml": "Qwen3.8-Flash-Next-FP8",
            "DeepSeek-V4-Flash-0731.yaml": "DeepSeek-V4-Flash-0731",
        }.items():
            r = self._load(name)
            self.assertTrue(r.local_dir, f"{name} missing paths.local_dir")
            self.assertIn(expected, r.local_dir)


class TestMemorySpec(unittest.TestCase):
    def _doc(self, memory=None):
        import yaml
        with open(RECIPE, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if memory is not None:
            doc["memory"] = memory
        return doc

    def test_default_device_placement(self):
        r = Recipe.from_dict(self._doc())
        self.assertEqual(r.memory.kv_placement, "device")
        self.assertEqual(r.memory.kv_utilization, 0.9)
        self.assertNotIn("memory", r.meta)  # not swallowed as passthrough

    def test_host_placement_parsed(self):
        r = Recipe.from_dict(self._doc({"kv_placement": "host", "kv_host_bytes": 1 << 30}))
        self.assertEqual(r.memory.kv_placement, "host")
        self.assertEqual(r.memory.kv_host_bytes, 1 << 30)

    def test_invalid_placement_rejected(self):
        with self.assertRaises(RecipeError):
            Recipe.from_dict(self._doc({"kv_placement": "cpu"}))
        with self.assertRaises(RecipeError):
            Recipe.from_dict(self._doc({"kv_placement": "gpu"}))

    def test_invalid_utilization_rejected(self):
        with self.assertRaises(RecipeError):
            Recipe.from_dict(self._doc({"kv_placement": "host", "kv_utilization": 1.5}))

    def test_coder_recipe_memory_option(self):
        base = os.path.dirname(RECIPE)
        with open(os.path.join(base, "Qwen2.5-Coder-0.5B.yaml"), encoding="utf-8") as f:
            r = Recipe.from_yaml(f.read())
        self.assertEqual(r.memory.kv_placement, "device")
        self.assertIn(r.memory.kv_placement, MemorySpec.KV_PLACEMENTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
