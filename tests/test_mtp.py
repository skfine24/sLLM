"""MTP reference + speculative-decoding tests.

Invariant: with full main-model greedy verification, MTP spec decode produces
EXACTLY the same tokens as plain greedy generation.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ref import mtp as m  # noqa: E402
from ref import pipeline as pl  # noqa: E402
from runtime import spec  # noqa: E402
from serving.dev_model import tiny_recipe, tiny_weights  # noqa: E402
from serving.executor import generate  # noqa: E402


class _TinyMTPModel:
    """Minimal model exposing logits + hidden for the MTP stack (tiny)."""

    def __init__(self, weights, recipe):
        self.weights = weights
        self.recipe = recipe

    def logits(self, ids):
        return pl.model_forward(np.asarray(ids, dtype=np.int64)[None, :], self.weights, self.recipe)


def _manual_mtp(ids, hidden, weights, recipe, cos, sin):
    """Independent re-implementation of the MTP forward for cross-checking."""
    from ref import qwen3_5 as qq

    eps = recipe.rms_norm_eps
    embed = weights["model.language_model.embed_tokens.weight"]
    e = qq.rms_norm(embed[ids], weights["mtp.pre_fc_norm_embedding.weight"], eps=eps)
    hn = qq.rms_norm(hidden, weights["mtp.pre_fc_norm_hidden.weight"], eps=eps)
    h = np.concatenate([e, hn], axis=-1) @ weights["mtp.fc.weight"].T
    p = "mtp.layers.0"
    from ref.qwen3_5 import decoder_layer_forward as dlf
    attn = {
        "w_q": weights[f"{p}.self_attn.q_proj.weight"],
        "w_k": weights[f"{p}.self_attn.k_proj.weight"],
        "w_v": weights[f"{p}.self_attn.v_proj.weight"],
        "w_o": weights[f"{p}.self_attn.o_proj.weight"],
        "q_norm_w": weights[f"{p}.self_attn.q_norm.weight"],
        "k_norm_w": weights[f"{p}.self_attn.k_norm.weight"],
        "cos": cos, "sin": sin, "num_heads": recipe.full_attention.num_heads,
        "kv_heads": recipe.full_attention.num_kv_heads,
        "head_dim": recipe.full_attention.head_dim,
        "mlp": {
            "w_gate": weights[f"{p}.mlp.gate_proj.weight"],
            "w_up": weights[f"{p}.mlp.up_proj.weight"],
            "w_down": weights[f"{p}.mlp.down_proj.weight"],
        },
    }
    h = dlf(h, in_norm_w=weights[f"{p}.input_layernorm.weight"],
            post_norm_w=weights[f"{p}.post_attention_layernorm.weight"],
            block_type="full_attention", rms_eps=eps, **attn)
    h = qq.rms_norm(h, weights["mtp.norm.weight"], eps=eps)
    return h @ weights["lm_head.weight"].T


class TestMTPReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = tiny_recipe()
        cls.weights = tiny_weights()
        cls.model = _TinyMTPModel(cls.weights, cls.recipe)

    def test_mtp_matches_manual(self):
        rng = np.random.default_rng(1)
        ids = np.array([[3, 5, 7]], dtype=np.int64)
        _, hidden = pl.model_forward(ids, self.weights, self.recipe, return_hidden_pre_norm=True)
        cos, sin = pl.build_cos_sin_for_positions(self.recipe, ids.shape[1])
        got = m.mtp_forward(ids, hidden, self.weights, self.recipe, cos, sin)
        ref = _manual_mtp(ids, hidden, self.weights, self.recipe, cos, sin)
        from serving.dev_model import VOCAB
        self.assertEqual(got.shape, (1, 3, VOCAB))
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

    def test_mtp_next_token_shapes_and_domain(self):
        tok = m.mtp_next_token(self.model, [3, 5], self.weights, self.recipe, temperature=0.0)
        self.assertIsInstance(tok, int)

    def test_hidden_pre_norm_return(self):
        ids = np.array([[2, 4, 6]], dtype=np.int64)
        logits, hidden = pl.model_forward(ids, self.weights, self.recipe, return_hidden_pre_norm=True)
        self.assertEqual(hidden.shape, (1, 3, hidden.shape[-1]))
        self.assertEqual(logits.shape, (1, 3, 248320))


class TestSpecDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = tiny_recipe()
        cls.weights = tiny_weights()
        cls.model = _TinyMTPModel(cls.weights, cls.recipe)

    def test_spec_equals_greedy(self):
        prompt = [4, 6, 8]
        for nd in (1, 2, 3):
            spec_ids = spec.spec_decode_greedy(self.model, self.weights, self.recipe,
                                               prompt, max_new=6, num_draft=nd)
            greedy_ids = generate(self.model, None, prompt, max_new=6, temperature=0.0)
            self.assertEqual(spec_ids, greedy_ids, f"num_draft={nd}")

    def test_result_shorter_or_equal(self):
        # corrective step guarantees progress: tokens >= prompt, <= prompt+max_new
        prompt = [1, 2, 3]
        out = spec.spec_decode_greedy(self.model, self.weights, self.recipe, prompt,
                                      max_new=5, num_draft=2)
        self.assertGreaterEqual(len(out), len(prompt))
        self.assertLessEqual(len(out), len(prompt) + 5)

    def test_max_new_one_never_overshoots(self):
        out = spec.spec_decode_greedy(self.model, self.weights, self.recipe,
                                      [1, 2, 3], max_new=1, num_draft=2)
        self.assertEqual(len(out), 4)  # exactly prompt + 1

    def test_verify_is_aligned_adversarial(self):
        # Main model: predicts 7 ONLY after a 7, else 9. Drafter: always 7.
        # A misaligned verifier (checking draft[k] with the logits AFTER it)
        # would accept the 7-chain; the aligned verifier must match greedy
        # (9, 9, 9, ...).
        import unittest.mock as mock

        def fake_forward(arr, weights, recipe):
            last = int(np.asarray(arr)[0, -1])
            pred = 7 if last == 7 else 9
            out = np.zeros((1, np.asarray(arr).shape[1], 10))
            out[0, -1, pred] = 1.0
            return out, None

        with mock.patch.object(spec._pipeline, "model_forward",
                               lambda *a, **k: fake_forward(*a, **k)[0]), \
             mock.patch.object(spec._mtp, "mtp_next_token",
                               lambda *a, **k: 7):
            out = spec.spec_decode_greedy(None, {}, None, [1], max_new=1,
                                          num_draft=2)
            self.assertEqual(out, [1, 9])
            out = spec.spec_decode_greedy(None, {}, None, [1], max_new=4,
                                          num_draft=3)
            self.assertEqual(out, [1, 9, 9, 9, 9])


if __name__ == "__main__":
    unittest.main(verbosity=2)
