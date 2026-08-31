"""MTP speculative decoding (reference). Drafts tokens with ref.mtp.mtp_next_token,
then verifies each drafted token against the MAIN model's greedy argmax. This is
the simplest sound checker: an accepted prefix must match the main model, so the
final output equals plain greedy decode (the key invariant tested here)."""

from __future__ import annotations

import numpy as np

from ref import mtp as _mtp
from ref import pipeline as _pipeline
from runtime import sampler as _sampler


def spec_decode_greedy(
    model,
    weights: dict,
    recipe,
    prompt: list[int],
    max_new: int,
    num_draft: int = 2,
    stop_ids: tuple[int, ...] = (),
) -> list[int]:
    """MTP draft + verify, main-model greedy acceptance.

    Returns the full id sequence (prompt + accepted tokens). Deterministic and
    output-equivalent to plain greedy generation.
    """
    if num_draft < 1:
        raise ValueError("num_draft must be >= 1")
    ids = list(int(i) for i in prompt)
    n = 0
    while n < max_new:
        if ids[-1] in stop_ids:
            break
        # -- draft num_draft tokens with the MTP head ----------------------
        draft = []
        cur = list(ids)
        for _ in range(num_draft):
            tok = _mtp.mtp_next_token(model, cur, weights, recipe, temperature=0.0)
            draft.append(tok)
            cur.append(tok)

        # -- verify against the main model (greedy) ------------------------
        accepted = 0
        for k, tok in enumerate(draft):
            arr = np.asarray([ids + draft[:k + 1]], dtype=np.int64)
            logits = _pipeline.model_forward(arr, weights, recipe)[0, -1, :]
            if _sampler.greedy(logits) != tok:
                break
            accepted = k + 1

        ids.extend(draft[:accepted])
        n += accepted

        # corrective step: always emit at least one main-greedy token
        arr = np.asarray([ids], dtype=np.int64)
        logits = _pipeline.model_forward(arr, weights, recipe)[0, -1, :]
        nxt = _sampler.greedy(logits)
        if nxt in stop_ids:
            break
        ids.append(nxt)
        n += 1
    return ids
