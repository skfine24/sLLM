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


if __name__ == "__main__":
    unittest.main(verbosity=2)
