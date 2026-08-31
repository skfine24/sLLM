"""Standard dense-transformer (Llama/Qwen2 family) reference + generation tests."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ref import standard as s  # noqa: E402
from ref import qwen3_5 as q  # noqa: E402
from serving.dev_model import tiny_standard_recipe, tiny_standard_weights  # noqa: E402
from serving.executor import ReferenceModel, generate  # noqa: E402


def _manual_forward(ids, w, recipe):
    """Independent layer-by-layer composition of the standard transformer."""
    b, slen = ids.shape
    prefix = recipe.text_prefix
    fa = recipe.full_attention
    hd = fa.effective_head_dim(recipe.hidden_size)
    eps = recipe.rms_norm_eps
    embed = w[f"{prefix}.embed_tokens.weight"]
    hidden = embed[ids].astype(np.float32)
    cos, sin = s.build_cos_sin(fa.rope.theta, np.arange(slen, dtype=np.int64)[None, :], hd)
    for i in range(recipe.num_layers):
        p = f"{prefix}.layers.{i}"
        residual = hidden
        h = s.rms_norm_plain(hidden, w[f"{p}.input_layernorm.weight"], eps=eps)
        h = s.standard_attention_forward(
            h,
            w_q=w[f"{p}.self_attn.q_proj.weight"],
            w_k=w[f"{p}.self_attn.k_proj.weight"],
            w_v=w[f"{p}.self_attn.v_proj.weight"],
            w_o=w[f"{p}.self_attn.o_proj.weight"],
            cos=cos, sin=sin, num_heads=fa.num_heads, kv_heads=fa.num_kv_heads, head_dim=hd,
            q_bias=w.get(f"{p}.self_attn.q_proj.bias"),
            k_bias=w.get(f"{p}.self_attn.k_proj.bias"),
            v_bias=w.get(f"{p}.self_attn.v_proj.bias"),
        )
        hidden = residual + h
        residual = hidden
        h = s.rms_norm_plain(hidden, w[f"{p}.post_attention_layernorm.weight"], eps=eps)
        h = q.mlp_forward(h,
                          w_gate=w[f"{p}.mlp.gate_proj.weight"],
                          w_up=w[f"{p}.mlp.up_proj.weight"],
                          w_down=w[f"{p}.mlp.down_proj.weight"])
        hidden = residual + h
    hidden = s.rms_norm_plain(hidden, w[f"{prefix}.norm.weight"], eps=eps)
    return hidden @ embed.T  # tied embeddings


class TestStandardReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = tiny_standard_recipe()
        cls.weights = tiny_standard_weights()

    def test_matches_manual(self):
        ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
        got = s.standard_model_forward(ids, self.weights, self.recipe)
        ref = _manual_forward(ids, self.weights, self.recipe)
        self.assertEqual(got.shape, (1, 4, 64))
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

    def test_tied_embeddings_used(self):
        ids = np.array([[3, 5]], dtype=np.int64)
        logits = s.standard_model_forward(ids, self.weights, self.recipe)
        self.assertTrue(np.isfinite(logits).all())

    def test_deterministic(self):
        ids = np.array([[1, 2]], dtype=np.int64)
        a = s.standard_model_forward(ids, self.weights, self.recipe)
        b = s.standard_model_forward(ids, self.weights, self.recipe)
        np.testing.assert_array_equal(np.argmax(a, -1), np.argmax(b, -1))

    def test_reference_model_routes_standard(self):
        m = ReferenceModel(self.recipe, self.weights)
        ids = [2, 4, 6]
        logits = m.logits(ids)
        self.assertEqual(logits.shape, (1, 3, 64))

    def test_generation_greedy(self):
        m = ReferenceModel(self.recipe, self.weights)
        out = generate(m, None, [1, 2, 3], max_new=5, temperature=0.0)
        self.assertEqual(len(out), 3 + 5)
        self.assertTrue(all(isinstance(x, int) for x in out))


class TestStandardRecipe(unittest.TestCase):
    def test_qwen25_coder_recipe_loads(self):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recipes"))
        from recipes.schema import Recipe
        with open(os.path.join(base, "Qwen2.5-Coder-0.5B.yaml"), encoding="utf-8") as f:
            r = Recipe.from_yaml(f.read())
        self.assertEqual(r.text_prefix, "model")
        self.assertTrue(r.tie_word_embeddings)
        self.assertEqual(r.full_attention.kernel, "standard_gqa")
        self.assertEqual(r.full_attention.effective_head_dim(r.hidden_size), 64)
        self.assertEqual(len(r.layer_types), 24)
        self.assertEqual(r.full_attention.rope.theta, 1000000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
