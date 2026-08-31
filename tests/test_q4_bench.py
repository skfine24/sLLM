"""Q3 bench plumbing test: the bench subset-parity pipeline run against the
F32 dev fixture must EXACTLY reproduce the in-memory pipeline (the loader
path adds no error), plus determinism/noise-floor reporting works."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench.q4_subset_parity as bp  # noqa: E402
from _synth import write_q4_dev_fixture  # noqa: E402
from ref import qwen4_exp_pipeline as qp  # noqa: E402


class TestQ4SubsetParityBench(unittest.TestCase):
    def test_fixture_run_matches_in_memory_oracle(self):
        cfg, w_mem = None, None
        with tempfile.TemporaryDirectory() as d:
            w_mem = write_q4_dev_fixture(d)
            cfg = qp.Qwen4ExpCfg.from_recipe(bp.load_recipe(tiny=True))
            r = bp.run_parity(d, tiny=True, layers=None, seq=8, seed=3,
                              out=os.path.join(d, "ref.npz"))
            self.assertTrue(r["determinism"])
            self.assertEqual(r["subset"], [0, 2])   # first linear + first QSA
            self.assertTrue(os.path.isfile(os.path.join(d, "ref.npz")))

        # in-memory twin: same weights, remapped to subset indices
        sub = r["subset"]
        _, sub_cfg = bp.pick_subset(cfg, sub)
        w = {}
        for j, i in enumerate(sub):
            for k, v in w_mem.items():
                if f".layers.{i}." in k:
                    w[k.replace(f".layers.{i}.", f".layers.{j}.")] = v
                elif f".layers." not in k:
                    w[k] = v
        ids = np.random.default_rng(3).integers(1, 32, size=8, dtype=np.int64)
        ids[0] = 1
        _, want = qp.prefill(ids, w, sub_cfg)
        np.testing.assert_array_equal(r["logits"], want)  # loader = identity

    def test_real_recipe_subset_selection_excludes_ple(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "recipes", "Qwen3.8-Flash-Next-FP8.yaml"),
                  encoding="utf-8") as f:
            recipe = bp.load_recipe(tiny=False)
        cfg = qp.Qwen4ExpCfg.from_recipe(recipe)
        subset, sub_cfg = bp.pick_subset(cfg, None)
        self.assertEqual(len(subset), 2)
        self.assertNotIn(subset[0], cfg.ple_layer_ids)
        self.assertEqual(sub_cfg.layer_types,
                         ("linear_attention", "full_attention"))


if __name__ == "__main__":
    unittest.main()
