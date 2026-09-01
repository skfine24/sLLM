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
        # -- draft with the MTP head (never more than we can still emit
        # while leaving room for the corrective greedy token) --------------
        nd = min(num_draft, max_new - n - 1)
        draft: list[int] = []
        cur = list(ids)
        for _ in range(nd):
            tok = _mtp.mtp_next_token(model, cur, weights, recipe, temperature=0.0)
            draft.append(tok)
            cur.append(tok)

        # -- verify against the main model (greedy) ------------------------
        # draft[k] is valid iff the MAIN model, run on (ids + draft[:k]),
        # greedily picks draft[k]. The logits of the first mismatching
        # prefix are exactly the corrective step's logits (reused below).
        accepted = 0
        mismatch: int | None = None
        for k, tok in enumerate(draft):
            arr = np.asarray([ids + draft[:k]], dtype=np.int64)
            logits = _pipeline.model_forward(arr, weights, recipe)[0, -1, :]
            if _sampler.greedy(logits) != tok:
                mismatch = _sampler.greedy(logits)
                break
            accepted = k + 1

        ids.extend(draft[:accepted])
        n += accepted
        if n >= max_new:
            break

        # corrective step: always emit at least one main-greedy token
        if mismatch is not None:
            nxt = mismatch
        else:
            arr = np.asarray([ids], dtype=np.int64)
            logits = _pipeline.model_forward(arr, weights, recipe)[0, -1, :]
            nxt = _sampler.greedy(logits)
        if nxt in stop_ids:
            break
        ids.append(nxt)
        n += 1
    return ids


def spec_decode_greedy_deepseek(
    model,
    spec_model,
    prompt: list[int],
    max_new: int,
    num_draft: int | None = None,
    stop_ids: tuple[int, ...] = (),
) -> list[int]:
    """DSPark draft + verify for DeepSeek-V4 (B=1, oracle).

    `model` is a `ref.deepseek_v4.DeepseekV4Model`; `spec_model` a
    `DeepseekV4SpecModel`. Drafts a block with the DSPark head, then verifies
    each drafted token against the MAIN model's greedy argmax (prefill
    recompute). An accepted prefix always matches plain greedy decode, so the
    returned sequence (prompt + emissions) is output-equivalent to greedy
    generation -- the key invariant asserted by the tests.

    Deterministic; `spec.setup` is re-seeded from the prefix each outer
    iteration, keeping the DSPark main-KV window consistent with `model`'s
    injected target-layer hidden means.
    """
    import numpy as np

    from ref.deepseek_v4 import DeepseekV4SpecModel  # noqa: F401  (type hint)
    from runtime import sampler as _sampler

    if num_draft is None:
        num_draft = int(spec_model.cfg.dspark_block_size)
    if num_draft < 1:
        raise ValueError("num_draft must be >= 1")
    ids = [int(i) for i in prompt]
    n = 0
    while n < max_new:
        if ids[-1] in stop_ids:
            break
        nd = min(num_draft, max_new - n)
        if nd < 1:
            break
        ids_arr = np.asarray(ids, dtype=np.int64)
        _, _, mh_t = model.prefill(ids_arr, spec=True)
        spec_model.setup(mh_t)
        prev_pos = len(ids) - 1
        out, _, _ = spec_model.draft_step(ids[-1], mh_t[-1], prev_pos)
        draft = [int(x) for x in out[1:1 + nd]]
        accepted = 0
        mismatch: int | None = None
        for k, tok in enumerate(draft):
            cand = np.asarray(ids + draft[:k], dtype=np.int64)
            _, lgk, _ = model.prefill(cand, spec=True)
            g = int(_sampler.greedy(lgk[-1]))
            if g != tok:
                mismatch = g
                break
            accepted = k + 1
        ids.extend(draft[:accepted])
        n += accepted
        if n >= max_new:
            break
        if mismatch is not None:
            nxt = mismatch
        else:
            arr = np.asarray(ids, dtype=np.int64)
            _, lgk, _ = model.prefill(arr, spec=True)
            nxt = int(_sampler.greedy(lgk[-1]))
        if nxt in stop_ids:
            break
        ids.append(nxt)
        n += 1
    return ids


def spec_decode_greedy_qwen4exp(
    cfg,
    weights: dict,
    prompt: list[int],
    max_new: int,
    num_draft: int = 2,
    stop_ids: tuple[int, ...] = (),
) -> list[int]:
    """MTP draft + verify for the qwen4_exp oracle (B=1).

    Delegates drafting/verification to the proven reference loop
    `ref.qwen4_exp_mtp.generate_greedy_mtp` (draft state rebuilt from the
    verified prefix each cycle, rope continuing the main context; every
    emitted token is the MAIN model's greedy argmax via incremental
    `decode_step_full`, so the PLE-augmented post-step hyper row feeds the
    next fusion). Stop ids are never emitted: the generated tail is trimmed at
    the first stop id. The returned sequence (prompt + emissions) equals plain
    greedy generation -- the key invariant asserted by the tests.
    """
    from ref import qwen4_exp_mtp as _qm

    if num_draft < 1:
        raise ValueError("num_draft must be >= 1")
    emitted = _qm.generate_greedy_mtp(list(int(i) for i in prompt), weights,
                                      cfg, max_new, num_draft=num_draft)[0]
    ids = list(int(i) for i in prompt) + [int(t) for t in emitted]
    if stop_ids:
        cut = len(ids)
        for i in range(len(prompt), len(ids)):
            if ids[i] in stop_ids:
                cut = i
                break
        ids = ids[:cut]
    return ids
