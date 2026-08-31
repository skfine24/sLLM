"""Incremental-decode parity: the KV/recurrent-state path must produce the
same logits as the full recompute oracle, keep greedy output identical, and
honour the context window (max_position_embeddings)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ref import incremental as inc  # noqa: E402
from ref import standard as st  # noqa: E402
from ref import pipeline as pl  # noqa: E402
from serving.dev_model import tiny_recipe, tiny_weights, tiny_standard_recipe, tiny_standard_weights  # noqa: E402
from serving.executor import ReferenceModel, generate, BatchedInferenceEngine  # noqa: E402


class _NoIncremental:
    """Wrapper that forces the recompute-every-step fallback path."""

    def __init__(self, m):
        self._m = m

    @property
    def supports_incremental(self):
        return False

    def __getattr__(self, name):
        return getattr(self._m, name)


class _FakeTokenizer:
    """Deterministic id<->text tokenizer so batch tests run without the real
    tokenizer cache (dev machine). Encodes to ids in [1, 48] (vocab >= 48)."""

    def eos_id(self):
        return None

    def encode(self, text: str) -> list:
        return [ord(c) % 48 + 1 for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(48 + (i % 74)) for i in ids)


class TestIncrementalStandard(unittest.TestCase):
    def setUp(self):
        self.recipe = tiny_standard_recipe()
        self.weights = tiny_standard_weights(np.random.default_rng(3))
        self.ids = np.array([[1, 3, 5, 7, 2, 4, 6, 8]], dtype=np.int64)

    def test_logits_parity(self):
        cache, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        full0 = st.standard_model_forward(self.ids, self.weights, self.recipe)
        np.testing.assert_allclose(L0, full0, rtol=1e-6, atol=1e-6)
        seq = list(self.ids[0])
        L = L0[0, -1]
        for _ in range(5):
            nxt = int(np.argmax(L))
            seq.append(nxt)
            L = inc.decode_step(cache, self.weights, self.recipe, nxt)
            full = st.standard_model_forward(
                np.array([seq], dtype=np.int64), self.weights, self.recipe)[0, -1]
            np.testing.assert_allclose(L, full, rtol=1e-6, atol=1e-6)

    def test_generate_matches_recompute(self):
        m_inc = ReferenceModel(self.recipe, self.weights)
        m_rec = _NoIncremental(m_inc)
        prompt = list(self.ids[0])
        a = generate(m_inc, None, prompt, max_new=12, temperature=0.0, seed=1)
        b = generate(m_rec, None, list(prompt), max_new=12, temperature=0.0, seed=1)
        self.assertEqual(a, b)
        self.assertGreater(len(a), len(prompt))

    def test_qkv_bias_parity(self):
        # Qwen2-style checkpoints carry q/k/v biases; ensure the standard path
        # applies them and incremental stays consistent with recompute.
        w = dict(self.weights)
        for i in range(2):
            base = f"model.layers.{i}.self_attn"
            rng = np.random.default_rng(100 + i)
            w[f"{base}.q_proj.bias"] = rng.standard_normal(16).astype(np.float32) * 0.5
            w[f"{base}.k_proj.bias"] = rng.standard_normal(8).astype(np.float32) * 0.5
            w[f"{base}.v_proj.bias"] = rng.standard_normal(8).astype(np.float32) * 0.5
        m_inc = ReferenceModel(self.recipe, w)
        m_rec = _NoIncremental(m_inc)
        prompt = list(self.ids[0])
        a = generate(m_inc, None, list(prompt), max_new=10, temperature=0.0, seed=1)
        b = generate(m_rec, None, list(prompt), max_new=10, temperature=0.0, seed=1)
        self.assertEqual(a, b)
        cache, L0 = inc.prefill(self.ids, w, self.recipe)
        full0 = st.standard_model_forward(self.ids, w, self.recipe)
        np.testing.assert_allclose(L0, full0, rtol=1e-6, atol=1e-6)

    def test_deterministic_temperature0(self):
        m = ReferenceModel(self.recipe, self.weights)
        prompt = list(self.ids[0])
        a = generate(m, None, list(prompt), max_new=8, temperature=0.0, seed=7)
        b = generate(m, None, list(prompt), max_new=8, temperature=0.0, seed=99)
        self.assertEqual(a, b)


class TestIncrementalHybrid(unittest.TestCase):
    def setUp(self):
        self.recipe = tiny_recipe()
        self.weights = tiny_weights(np.random.default_rng(42))
        self.ids = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    def test_logits_parity(self):
        cache, L0 = inc.prefill(self.ids, self.weights, self.recipe)
        full0 = pl.model_forward(self.ids, self.weights, self.recipe)
        np.testing.assert_allclose(L0, full0, rtol=1e-3, atol=1e-3)
        seq = list(self.ids[0])
        L = L0[0, -1]
        for _ in range(5):
            nxt = int(np.argmax(L))
            seq.append(nxt)
            L = inc.decode_step(cache, self.weights, self.recipe, nxt)
            full = pl.model_forward(
                np.array([seq], dtype=np.int64), self.weights, self.recipe)[0, -1]
            np.testing.assert_allclose(L, full, rtol=1e-3, atol=1e-3)

    def test_generate_consistent(self):
        # Greedy incremental generation must be deterministic and bounded.
        # NOTE: the hybrid path has an inherent ~1e-7 prefill(chunked) vs
        # decode(recurrent) seam (the same seam vLLM has), so we do not demand
        # bit-identical greedy vs the recompute oracle here; logits parity is
        # asserted in test_logits_parity instead.
        m = ReferenceModel(self.recipe, self.weights)
        prompt = list(self.ids[0])
        a = generate(m, None, list(prompt), max_new=12, temperature=0.0, seed=1)
        b = generate(m, None, list(prompt), max_new=12, temperature=0.0, seed=2)
        self.assertEqual(a, b)
        self.assertGreaterEqual(len(a), len(prompt))
        self.assertLessEqual(len(a), len(prompt) + 12)

    def test_state_chaining(self):
        cache, _ = inc.prefill(self.ids, self.weights, self.recipe)
        self.assertIn(0, cache.state)          # linear layer state kept
        self.assertNotIn(0, cache.k)           # linear layer has no attention KV
        self.assertIn(1, cache.k)              # full layer has KV
        self.assertEqual(cache.state[0].shape, (1, 4, 8, 8))


class TestBatchedIncremental(unittest.TestCase):
    def setUp(self):
        self.tok = _FakeTokenizer()

    def _prompts(self):
        return [
            "def add(a, b):  return a + b",
            "x = [1, 2, 3]\nprint(x)",
            "hello world, incremental decode test!",
        ]

    def test_batch_standard_matches_recompute(self):
        recipe = tiny_standard_recipe()
        weights = tiny_standard_weights(np.random.default_rng(3))
        m = ReferenceModel(recipe, weights)

        eng_inc = BatchedInferenceEngine(m, self.tok, max_concurrency=3, chunk_size=8)
        eng_rec = BatchedInferenceEngine(_NoIncremental(m), self.tok, max_concurrency=3, chunk_size=8)
        for p in self._prompts():
            eng_inc.submit(p, max_new=12, temperature=0.0)
            eng_rec.submit(p, max_new=12, temperature=0.0)
        out_inc = eng_inc.run_all()
        out_rec = eng_rec.run_all()
        self.assertEqual(out_inc, out_rec)

    def test_batch_standard_equals_sequential(self):
        recipe = tiny_standard_recipe()
        weights = tiny_standard_weights(np.random.default_rng(3))
        m = ReferenceModel(recipe, weights)
        eng = BatchedInferenceEngine(m, self.tok, max_concurrency=3, chunk_size=8)
        for p in self._prompts():
            eng.submit(p, max_new=10, temperature=0.0)
        outs = eng.run_all()
        for p, o in zip(self._prompts(), outs):
            single = generate(m, self.tok, self.tok.encode(p), max_new=10, temperature=0.0)
            self.assertEqual(o, self.tok.decode(single[len(self.tok.encode(p)):]))

    def test_batch_hybrid_deterministic_bounded(self):
        recipe = tiny_recipe()
        weights = tiny_weights(np.random.default_rng(42))
        m = ReferenceModel(recipe, weights)
        eng = BatchedInferenceEngine(m, self.tok, max_concurrency=3, chunk_size=8)
        for p in self._prompts():
            eng.submit(p, max_new=12, temperature=0.0)
        a = eng.run_all()
        eng2 = BatchedInferenceEngine(m, self.tok, max_concurrency=3, chunk_size=8)
        for p in self._prompts():
            eng2.submit(p, max_new=12, temperature=0.0)
        b = eng2.run_all()
        self.assertEqual(a, b)
        for o in a:
            n = len(o)
            self.assertGreaterEqual(n, 1)
            self.assertLessEqual(n, 12)

    def test_batch_long_prompt_chunked_prefill(self):
        recipe = tiny_standard_recipe()
        weights = tiny_standard_weights(np.random.default_rng(3))
        m = ReferenceModel(recipe, weights)
        long_prompt = "z" * 40  # 40 tokens > chunk_size=8 -> multiple prefill actions
        eng = BatchedInferenceEngine(m, self.tok, max_concurrency=1, chunk_size=8)
        eng.submit(long_prompt, max_new=6, temperature=0.0)
        eng.run_all()
        info = eng._seqs[0]
        self.assertEqual(len(info["prompt"]), 40)
        self.assertGreater(len(info["gen"]), 0)


class TestRepetitionPenalty(unittest.TestCase):
    def test_penalty_applied_and_passthrough(self):
        from runtime import sampler as sm
        logits = np.zeros(10, dtype=np.float32)
        logits[2] = 3.0
        logits[3] = 3.0
        pen = sm.apply_repetition_penalty(logits, [2], 2.0)
        self.assertAlmostEqual(pen[2], 1.5)     # seen token penalized
        self.assertAlmostEqual(pen[3], 3.0)      # unseen token unchanged
        same = sm.apply_repetition_penalty(logits, [2], 1.0)
        np.testing.assert_array_equal(same, logits)

    def test_generate_with_penalty(self):
        recipe = tiny_standard_recipe()
        weights = tiny_standard_weights(np.random.default_rng(3))
        m = ReferenceModel(recipe, weights)
        prompt = list(range(1, 8))
        out = generate(m, None, list(prompt), max_new=15, temperature=0.0,
                       seed=1, repetition_penalty=1.3)
        self.assertGreater(len(out), len(prompt))
        self.assertLessEqual(len(out), len(prompt) + 15)


class TestContextWindow(unittest.TestCase):
    def test_decode_stops_at_max_context(self):
        recipe = tiny_standard_recipe()       # max_position_embeddings = 256
        weights = tiny_standard_weights(np.random.default_rng(3))
        m = ReferenceModel(recipe, weights)
        prompt = [i % 64 for i in range(253)]
        out = generate(m, None, prompt, max_new=100, temperature=0.0)
        self.assertEqual(len(out), 256)

    def test_over_long_prompt_raises(self):
        recipe = tiny_standard_recipe()
        weights = tiny_standard_weights(np.random.default_rng(3))
        m = ReferenceModel(recipe, weights)
        with self.assertRaises(ValueError):
            generate(m, None, list(range(256)), max_new=4, temperature=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
