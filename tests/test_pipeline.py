"""Pipeline wiring tests: the checkpoint-named weight dictionary drives the
full text forward, and the pipeline agrees with an explicit layer-by-layer
composition using the reference modules directly."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from serving.dev_model import HIDDEN, VOCAB, tiny_recipe, tiny_weights  # noqa: E402
from ref import pipeline as pl  # noqa: E402
from ref import qwen3_5 as q  # noqa: E402


def _manual_forward(ids, w, recipe):
    """Explicit layer-by-layer composition with reference modules."""
    b, s = ids.shape
    hidden = w[f"model.language_model.embed_tokens.weight"][ids].astype(np.float32)
    cos, sin = pl.build_cos_sin_for_positions(recipe, s)
    for i, bt in enumerate(recipe.layer_types):
        kw = {"in_norm_w": w[f"model.language_model.layers.{i}.input_layernorm.weight"],
              "post_norm_w": w[f"model.language_model.layers.{i}.post_attention_layernorm.weight"],
              "block_type": bt, "rms_eps": recipe.rms_norm_eps}
        mlp = {"w_gate": w[f"model.language_model.layers.{i}.mlp.gate_proj.weight"],
               "w_up": w[f"model.language_model.layers.{i}.mlp.up_proj.weight"],
               "w_down": w[f"model.language_model.layers.{i}.mlp.down_proj.weight"]}
        if bt == "linear_attention":
            la = recipe.linear_attention
            p = f"model.language_model.layers.{i}.linear_attn"
            kw.update({
                "w_in_qkv": w[f"{p}.in_proj_qkv.weight"], "w_conv": w[f"{p}.conv1d.weight"],
                "w_z": w[f"{p}.in_proj_z.weight"], "w_b": w[f"{p}.in_proj_b.weight"],
                "w_a": w[f"{p}.in_proj_a.weight"], "a_log": w[f"{p}.A_log"],
                "dt_bias": w[f"{p}.dt_bias"], "norm_w": w[f"{p}.norm.weight"],
                "w_out": w[f"{p}.out_proj.weight"],
                "num_k_heads": la.num_key_heads, "head_k_dim": la.key_head_dim,
                "num_v_heads": la.num_value_heads, "head_v_dim": la.value_head_dim,
                "conv_kernel_size": la.conv_kernel_size,
                "use_qk_l2norm_in_kernel": la.qk_l2norm,
                "chunked": True, "chunk_size": 64, "rms_eps": recipe.rms_norm_eps,
                "mlp": mlp,
            })
            hidden = q.decoder_layer_forward(hidden, **kw)
        else:
            fa = recipe.full_attention
            p = f"model.language_model.layers.{i}.self_attn"
            kw.update({
                "w_q": w[f"{p}.q_proj.weight"], "w_k": w[f"{p}.k_proj.weight"],
                "w_v": w[f"{p}.v_proj.weight"], "w_o": w[f"{p}.o_proj.weight"],
                "q_norm_w": w[f"{p}.q_norm.weight"], "k_norm_w": w[f"{p}.k_norm.weight"],
                "cos": cos, "sin": sin,
                "num_heads": fa.num_heads, "kv_heads": fa.num_kv_heads,
                "head_dim": fa.head_dim, "rms_eps": recipe.rms_norm_eps,
                "mlp": mlp,
            })
            hidden = q.decoder_layer_forward(hidden, **kw)
    hidden = q.rms_norm(hidden, w["model.language_model.norm.weight"], eps=recipe.rms_norm_eps)
    return hidden @ w["lm_head.weight"].T


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.recipe = tiny_recipe()
        self.weights = tiny_weights(np.random.default_rng(42))
        self.ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

    def test_shape_and_finite(self):
        logits = pl.model_forward(self.ids, self.weights, self.recipe)
        self.assertEqual(logits.shape, (1, 5, VOCAB))
        self.assertTrue(np.isfinite(logits).all())

    def test_matches_manual_composition(self):
        got = pl.model_forward(self.ids, self.weights, self.recipe)
        ref = _manual_forward(self.ids, self.weights, self.recipe)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

    def test_deterministic_and_greedy_stable(self):
        logits1 = pl.model_forward(self.ids, self.weights, self.recipe)
        logits2 = pl.model_forward(self.ids, self.weights, self.recipe)
        np.testing.assert_array_equal(np.argmax(logits1, axis=-1), np.argmax(logits2, axis=-1))

    def test_state_collection_shapes(self):
        logits, states = pl.model_forward(self.ids, self.weights, self.recipe, return_states=True)
        self.assertEqual(logits.shape, (1, 5, VOCAB))
        self.assertIn(0, states)  # linear layer state
        self.assertEqual(states[0].shape, (1, 4, 8, 8))
        self.assertNotIn(1, states)  # full attention layer has no recurrent state


if __name__ == "__main__":
    unittest.main(verbosity=2)
