"""Deterministic and stochastic sampling over a final-logit vector."""

from __future__ import annotations

import numpy as np


def greedy(logits) -> int:
    return int(np.argmax(logits))


def apply_repetition_penalty(logits, penalized_ids, penalty: float) -> np.ndarray:
    """Penalize already-seen token ids: logit[id] /= penalty (penalty > 1).

    Applied to raw logits before temperature/softmax (HF-style). Returns a
    copy; `penalized_ids` is the token history (typically prompt + generated).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if penalty is None or penalty <= 0 or penalty == 1.0 or penalized_ids is None:
        return logits
    pids = np.asarray(penalized_ids, dtype=np.int64)
    if pids.size == 0:
        return logits
    out = logits.copy()
    out[pids] = out[pids] / penalty
    return out


def sample(
    logits,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token id (HF-style temperature + top-k + top-p filtering).

    - temperature <= 0 returns the argmax (deterministic greedy).
    - top_k keeps only the k largest logits (soft counts).
    - top_p keeps the smallest set whose cumulative prob >= top_p.
    """
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)
    if temperature is not None and temperature > 0:
        logits = logits / temperature
    else:
        return greedy(logits)

    scores = logits - logits.max()
    probs = np.exp(scores)
    probs = probs / probs.sum()

    if top_k is not None:
        k = min(int(top_k), int(probs.size))
        if k <= 0:
            raise ValueError("top_k must be >= 1")
        keep = np.argsort(probs)[-k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)

    if top_p is not None:
        order = np.argsort(probs)[::-1]
        sorted_p = probs[order]
        cumulative = np.cumsum(sorted_p)
        cutoff = int(np.searchsorted(cumulative, top_p) + 1)
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[:cutoff]] = True
        probs = np.where(mask, probs, 0.0)

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        return greedy(logits)
    probs = probs / total
    r = rng.random()
    return int(np.searchsorted(np.cumsum(probs), r))
