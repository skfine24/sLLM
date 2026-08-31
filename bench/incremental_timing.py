"""Timing comparison: full-recompute vs incremental (KV-cached) decode.

Dev-machine indicative benchmark (plain numpy). Run:

    python bench/incremental_timing.py

Use the returned per-token times only as a relative measure on this machine;
the real speedup lands on the cluster with real checkpoints.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from serving.dev_model import tiny_standard_recipe, tiny_standard_weights  # noqa: E402
from serving.executor import ReferenceModel, generate  # noqa: E402


class _NoIncremental:
    def __init__(self, m):
        self._m = m

    @property
    def supports_incremental(self):
        return False

    def __getattr__(self, name):
        return getattr(self._m, name)


def main():
    recipe = tiny_standard_recipe()
    weights = tiny_standard_weights(np.random.default_rng(3))
    m = ReferenceModel(recipe, weights)
    m_rec = _NoIncremental(m)
    prompt = list(range(7))[::-1]

    print(f"model: {recipe.model_id} (hidden={recipe.hidden_size}, "
          f"layers={recipe.num_layers}, max_ctx={recipe.max_position_embeddings})")
    print(f"{'tokens':>8} {'recompute s':>12} {'incremental s':>14} {'speedup':>8}")
    for n in (50, 100, 200):
        t0 = time.perf_counter()
        generate(m_rec, None, list(prompt), max_new=n, temperature=0.0, seed=1)
        t_rec = time.perf_counter() - t0

        t0 = time.perf_counter()
        generate(m, None, list(prompt), max_new=n, temperature=0.0, seed=1)
        t_inc = time.perf_counter() - t0

        ratio = t_rec / t_inc if t_inc else float("inf")
        print(f"{n:>8} {t_rec:>12.3f} {t_inc:>14.3f} {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
