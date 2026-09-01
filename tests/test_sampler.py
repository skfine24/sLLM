"""Sampler tests: greedy, temperature/top-k/top-p filtering, determinism."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime import sampler  # noqa: E402


class TestSampler(unittest.TestCase):
    def test_greedy_is_argmax(self):
        logits = np.array([0.1, 5.0, -1.0, 3.0])
        self.assertEqual(sampler.greedy(logits), 1)
        self.assertEqual(sampler.sample(logits, temperature=0.0), 1)

    def test_top_k_restricts_support(self):
        logits = np.array([10.0, 9.0, 1.0, 0.0])
        for _ in range(200):
            tok = sampler.sample(logits, temperature=1.0, top_k=2, rng=np.random.default_rng(0))
            self.assertIn(tok, (0, 1))

    def test_top_p_cumulative(self):
        # only the top token carries ~all probability
        logits = np.array([0.0] * 100)
        logits[0] = 40.0
        for _ in range(50):
            tok = sampler.sample(logits, temperature=1.0, top_p=0.9, rng=np.random.default_rng(1))
            self.assertEqual(tok, 0)

    def test_deterministic_with_seed_temperature(self):
        logits = np.arange(10, dtype=np.float64)
        rng = np.random.default_rng(7)
        a = sampler.sample(logits, temperature=0.8, top_p=1.0, rng=rng)
        b = sampler.sample(logits, temperature=0.8, top_p=1.0, rng=np.random.default_rng(7))
        self.assertEqual(a, b)

    def test_temperature_zero_returns_argmax_without_rng_error(self):
        logits = np.arange(5, dtype=np.float64)
        self.assertEqual(sampler.sample(logits, temperature=0.0), 4)

    def test_flat_returns_valid_id(self):
        logits = np.zeros(7)
        tok = sampler.sample(logits, temperature=1.0, rng=np.random.default_rng(3))
        self.assertIn(tok, range(7))

    def test_repetition_penalty_penalizes_negative_logits(self):
        # HF semantics: penalty > 1 must ALWAYS reduce probability. A
        # negative logit is multiplied (not divided), else it would grow.
        logits = np.array([-2.0, 2.0])
        out = sampler.apply_repetition_penalty(logits, [0], 2.0)
        self.assertEqual(out[0], -4.0)   # was -1.0 (rewarded) with the bug
        self.assertEqual(out[1], 2.0)
        p0_before = np.exp(logits - logits.max())[0]
        p0_after = np.exp(out - out.max())[0]
        self.assertLess(p0_after, p0_before)

    def test_repetition_penalty_always_returns_copy(self):
        logits = np.array([1.0, 2.0])
        for penalty in (1.0, 2.0):
            out = sampler.apply_repetition_penalty(logits, [0], penalty)
            self.assertIsNot(out, logits)
        float64_in = np.array([1.0, 2.0], dtype=np.float64)
        self.assertIsNot(sampler.apply_repetition_penalty(float64_in, None, 2.0),
                         float64_in)

    def test_repetition_penalty_ignores_out_of_range_ids(self):
        logits = np.array([1.0, 2.0])
        out = sampler.apply_repetition_penalty(logits, [0, 99, -3], 2.0)
        self.assertEqual(out[0], 0.5)
        self.assertEqual(out[1], 2.0)

    def test_top_p_renormalizes_after_top_k(self):
        # top-k=2 keeps probs ~ (0.73, 0.27); top_p=0.3 must then cut down to
        # the single top token (renormalized top prob 0.73 >= 0.3).
        logits = np.array([2.0, 1.0, -10.0, -10.0])
        for seed in range(20):
            tok = sampler.sample(logits, temperature=1.0, top_k=2, top_p=0.3,
                                 rng=np.random.default_rng(seed))
            self.assertEqual(tok, 0)

    def test_sample_never_returns_out_of_vocab(self):
        # extreme skewed probs: cumsum rounding must not yield index == V
        logits = np.full(4, -1000.0)
        logits[0] = 0.0
        for seed in range(50):
            tok = sampler.sample(logits, temperature=1.0,
                                 rng=np.random.default_rng(seed))
            self.assertIn(tok, range(4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
