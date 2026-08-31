"""Qwen4_exp component reference math (numpy), the T0 oracle layer.

Ported from `oracle/upstream/sglang/` (sglang from the deployed
`lmsysorg/sglang:qwen38flashnext` image; see its README.md). Pure numpy,
torch-free. Each function cites its upstream source.

Scope (milestone Q1 of docs/design/09-roadmap-seq-qwen4exp-dsv4.md):
- hyper-connection mixer (`GatedResidual` with `hc_per_branch_norm=True`)
- MoE (router softmax -> top-k -> renormalize + swiglu experts + gated
  shared expert), dense emulation (no grouping/chunking)
- QSA indexer math (compressed-block sparse selection): MQA logits, average
  pooling, fast top-k, block->token expansion, sparse attention reference
- indexer q/k projection (GemmaRMSNorm + partial neox rotary, head_dim 128)

Out of scope here (later milestones): PLE/ngram, MTP-hybrid, vision, TP.
Tie-breaking note: `torch.topk` does not guarantee tie order; this oracle
uses stable sort (ties keep ascending index order). Measured tests use
random data without exact ties.
"""

from __future__ import annotations

import numpy as np

from . import qwen3_5 as qq


# --------------------------------------------------------------------------
# norms (Gemma style: out = rmsnorm(x) * (1 + weight), weight init zeros)
# --------------------------------------------------------------------------

def gemma_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """sglang `GemmaRMSNorm.forward_native` (layers/layernorm.py)."""
    xf = x.astype(np.float32)
    var = (xf * xf).mean(-1, keepdims=True)
    out = xf * (var + np.float32(eps)) ** -0.5
    out = out * (1.0 + weight.astype(np.float32))
    return out.astype(x.dtype)


def grouped_gemma_rmsnorm(
    x: np.ndarray, weight: np.ndarray, group_size: int, eps: float = 1e-6
) -> np.ndarray:
    """`hyperconnection.py: GroupedGemmaRMSNorm.forward` (non-JIT path).

    Variance is computed per `group_size` chunk of the last dim; the (1+w)
    scale is over the full last dim.
    """
    xf = x.astype(np.float32)
    n = xf.shape[-1]
    if n % group_size != 0:
        raise ValueError(f"last dim {n} not divisible by group_size {group_size}")
    xg = xf.reshape(*xf.shape[:-1], n // group_size, group_size)
    var = (xg * xg).mean(-1, keepdims=True)
    x_norm = (xg * (var + np.float32(eps)) ** -0.5).reshape(xf.shape)
    out = x_norm * (1.0 + weight.astype(np.float32))
    return out.astype(x.dtype)


# --------------------------------------------------------------------------
# hyper-connection (GatedResidual, hc_per_branch_norm=True)
# --------------------------------------------------------------------------
# upstream: oracle/upstream/sglang/hyperconnection.py (GatedResidual.mix /
# .combine and the _mix_compute / _combine_compute closures).
# Shapes here: hyper (..., hc*hs); weights: norm_w (hc*hs), down (lr, hc*hs),
# up (hc*hs, lr), inject (hc, hc*hs). `normed` is returned by hc_mix and fed
# back into hc_combine, matching upstream `residuals = (hyper_input,
# hyper_input_normed)`.

def hc_mix(
    hyper: np.ndarray,
    norm_w: np.ndarray,
    down_w: np.ndarray,
    up_w: np.ndarray,
    hc: int,
    hs: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Block input mixing: returns (mixed_input [.., hs], normed [.., hc*hs])."""
    if hyper.shape[-1] != hc * hs:
        raise ValueError(f"hyper last dim {hyper.shape[-1]} != hc*hs {hc * hs}")
    normed = grouped_gemma_rmsnorm(hyper, norm_w, hs, eps)  # per-branch groups
    low = qq.silu(normed @ down_w.T.astype(np.float32) / hc)  # (..., lr)
    w = qq.sigmoid(low @ up_w.T.astype(np.float32))           # (..., hc*hs)
    w = w.reshape(*w.shape[:-1], hc, hs)
    mixed = (w * normed.reshape(*normed.shape[:-1], hc, hs)).mean(axis=-2)
    return mixed.astype(hyper.dtype), normed


def hc_combine(
    block_out: np.ndarray,
    hyper: np.ndarray,
    normed: np.ndarray,
    inject_w: np.ndarray,
    hc: int,
    hs: int,
) -> np.ndarray:
    """Block output injection: returns updated hyper (..., hc*hs)."""
    inj = 2.0 * qq.sigmoid(normed @ inject_w.T.astype(np.float32) / hc)  # (..., hc)
    r = hyper.reshape(*hyper.shape[:-1], hc, hs)
    inj = inj.reshape(*inj.shape[:-1], hc, 1)
    out = (r + block_out[..., np.newaxis, :] * inj).reshape(*hyper.shape)
    return out.astype(hyper.dtype)


# --------------------------------------------------------------------------
# MoE (dense emulation)
# --------------------------------------------------------------------------
# upstream: models/qwen2_moe.py Qwen2MoeSparseMoeBlock (via sglang_qwen3_5
# .py line 781/1008) + layers/moe/topk.py (scoring_func="softmax" default,
# renormalize=norm_topk_prob default True, epsilon 1e-20 on the renorm sum).

def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def moe_route(
    router_logits: np.ndarray, top_k: int, eps: float = 1e-20
) -> tuple[np.ndarray, np.ndarray]:
    """softmax -> top-k (stable desc) -> renormalize. Returns (weights, ids).

    weights (n, top_k) float32; ids (n, top_k) int64 ascending on ties.
    """
    p = softmax_rows(router_logits)
    n, e = p.shape
    if top_k > e:
        raise ValueError(f"top_k {top_k} > num_experts {e}")
    order = np.argsort(-p, axis=-1, kind="stable")
    ids = order[:, :top_k].astype(np.int64)
    w = np.take_along_axis(p, ids, axis=-1)
    w = w / (w.sum(-1, keepdims=True) + np.float32(eps))
    return w.astype(np.float32), ids


def swiglu_mlp(
    x: np.ndarray, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
) -> np.ndarray:
    """Qwen2MoeMLP: down(silu(gate(x)) * up(x)); weights (out, in)."""
    g = x @ gate_w.T
    u = x @ up_w.T
    h = qq.silu(g) * u
    return h @ down_w.T


def moe_block_forward(
    x: np.ndarray,
    router_w: np.ndarray,
    expert_gate: np.ndarray,
    expert_up: np.ndarray,
    expert_down: np.ndarray,
    shared_gate_w: np.ndarray,
    shared_gate_linear_w: np.ndarray,
    shared_up_w: np.ndarray,
    shared_down_w: np.ndarray,
    top_k: int,
    eps: float = 1e-20,
) -> np.ndarray:
    """Dense emulation of Qwen2MoeSparseMoeBlock forward.

    x (n, H); router_w (E, H); expert_gate/up (E, I, H); expert_down (E, H, I);
    shared_gate_w (I_s, H); shared_gate_linear_w (1, H); shared_down_w (H, I_s).
    final = sum_k w_k * expert_k(x) + sigmoid(shared_gate_linear(x)) * shared(x)
    """
    n, h = x.shape
    weights, ids = moe_route(x @ router_w.T, top_k, eps)
    out = np.zeros((n, h), dtype=np.float32)
    for i in range(n):
        for j in range(top_k):
            e = ids[i, j]
            y = swiglu_mlp(x[i:i + 1], expert_gate[e], expert_up[e], expert_down[e])
            out[i] += weights[i, j] * y[0]
    if shared_up_w is not None:
        shared = swiglu_mlp(x, shared_gate_w, shared_up_w, shared_down_w)
        gate = qq.sigmoid(x @ shared_gate_linear_w.T)  # (n, 1)
        out += gate * shared.astype(np.float32)
    return out


# --------------------------------------------------------------------------
# QSA indexer (compressed sparse attention selection)
# --------------------------------------------------------------------------
# upstream: oracle/upstream/sglang/qsa/{mqa.py, kernel.py, qsa_indexer.py}.
# qwen4_exp knobs: n_heads=4, kv_heads=1, head_dim=128, budget=2048,
# compress_ratio=4 -> block_topk=512, final width = budget + ratio - 1 = 2051.

def apply_rope_lastdim(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray, rotary_dim: int
) -> np.ndarray:
    """NeoX-style partial rotary over the last dim.

    x (tokens, heads, dim); cos/sin (tokens, rotary_dim) (doubled-half form,
    same layout as `qwen3_5.compute_cos_sin`). Mirrors
    `QSAIndexer.apply_rope` (rotates [..., :rotary_dim], keeps the tail).
    """
    rot = x[..., :rotary_dim]
    tail = x[..., rotary_dim:]
    half = rotary_dim // 2
    c = cos[:, np.newaxis, :]
    s = sin[:, np.newaxis, :]
    rot1 = rot[..., :half]
    rot2 = rot[..., half:]
    rotated = np.concatenate(
        (rot1 * c[..., :half] - rot2 * s[..., :half],
         rot2 * c[..., half:] + rot1 * s[..., half:]),
        axis=-1,
    )
    return np.concatenate((rotated, tail), axis=-1)


def qsa_index_project_qk(
    hidden: np.ndarray,
    qk_w: np.ndarray,
    q_norm_w: np.ndarray,
    k_norm_w: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    rotary_dim: int,
    n_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """`QSAIndexer.project_qk` non-fused path + `normalize_compressed_keys`.

    Returns (q [tokens, n_heads, head_dim] rope'd, token_k
    [tokens, 1, head_dim] raw/unpooled). k is normalized+pooled+pooled-rope'd
    later by `qsa_normalize_compressed_keys` after average pooling.
    """
    qk = hidden @ qk_w.T  # (tokens, (n_heads + 1) * head_dim)
    q_raw = qk[:, : n_heads * head_dim].reshape(-1, n_heads, head_dim)
    q = gemma_rmsnorm(q_raw, q_norm_w, eps)
    q = apply_rope_lastdim(q, cos, sin, rotary_dim)
    token_k = qk[:, n_heads * head_dim:].reshape(-1, 1, head_dim)
    return q, token_k


def qsa_normalize_compressed_keys(
    pooled: np.ndarray,
    k_norm_w: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    rotary_dim: int,
    head_dim: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """`QSAIndexer.normalize_compressed_keys`: GemmaRMSNorm then rope.

    pooled (blocks, head_dim) -> (blocks, 1, head_dim).
    """
    k = gemma_rmsnorm(pooled[:, np.newaxis, :], k_norm_w, eps)
    return apply_rope_lastdim(k, cos, sin, rotary_dim)


def qsa_average_pool_keys(groups: np.ndarray) -> np.ndarray:
    """`kernel.py: average_pool_qsa_keys`: fp32 mean over the ratio axis."""
    return groups.astype(np.float32).mean(axis=-2)


def qsa_mqa_logits(
    q: np.ndarray,
    k: np.ndarray,
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    score_scale: float | None = None,
) -> np.ndarray:
    """`mqa.py: torch_qsa_mqa_prefill`.

    q (m, H, d), k (n, d) -> logits (m, n) fp32:
    sum_h relu(q_h . k) / scale, invalid columns -inf.
    """
    d = q.shape[-1]
    scale = float(score_scale) if score_scale else float(np.sqrt(d))
    scores = np.maximum(q.astype(np.float32) @ k.T.astype(np.float32), 0.0)
    logits = scores.sum(axis=1) / scale  # einsum mhd,nd->mn with relu+sum
    m, n = logits.shape
    cols = np.arange(n)[np.newaxis, :]
    valid = (cols >= row_starts.reshape(-1, 1)) & (cols < row_ends.reshape(-1, 1))
    return np.where(valid, logits, -np.inf)


def qsa_fast_topk(
    logits: np.ndarray, starts: np.ndarray, ends: np.ndarray, topk: int
) -> np.ndarray:
    """`kernel.py: qsa_fast_topk` CPU/reference path.

    Returns int32 (m, topk) indices RELATIVE to each row's start; -1 padded.
    """
    m = logits.shape[0]
    out = np.full((m, topk), -1, dtype=np.int32)
    for r in range(m):
        s, length = int(starts[r]), int(ends[r] - starts[r])
        width = min(length, topk)
        if width <= 0:
            continue
        row = logits[r, s:s + length]
        order = np.argsort(-row, kind="stable")
        out[r, :width] = order[:width].astype(np.int32)
    return out


def qsa_expand_block_indices(
    block_indices: np.ndarray,
    query_positions: np.ndarray,
    sequence_lengths: np.ndarray,
    compress_ratio: int,
    token_topk: int,
) -> np.ndarray:
    """`kernel.py: torch_expand_qsa_block_indices` (verbatim port).

    Relative compressed-block indices (+ optional absolute use; see note in
    QSAIndexer: rows index their own sequence so relative == in-sequence
    position) -> fixed-width token ids within the sequence, -1 padded and
    compacted; tail tokens of the current partial block are appended.
    """
    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got {block_indices.shape}"
        )
    rows = block_indices.shape[0]
    blocks = block_indices.astype(np.int64)
    offsets = np.arange(compress_ratio, dtype=np.int64)
    expanded = blocks[:, :, None] * compress_ratio + offsets
    expanded = np.where(blocks[:, :, None] >= 0, expanded, -1)
    expanded = expanded.reshape(rows, block_topk * compress_ratio)[:, :token_topk]

    qpos = query_positions.astype(np.int64)
    seqlen = sequence_lengths.astype(np.int64)
    expanded = np.where(
        (expanded >= 0) & (expanded < seqlen[:, None]), expanded, -1
    )

    tail_offsets = np.arange(compress_ratio - 1, dtype=np.int64)
    visible = qpos + 1
    tail_start = (visible // compress_ratio) * compress_ratio
    tail_count = visible - tail_start
    tail = tail_start[:, None] + tail_offsets[None, :]
    tail_valid = (
        (tail_offsets[None, :] < tail_count[:, None])
        & (tail < seqlen[:, None])
    )
    tail = np.where(tail_valid, tail, -1)

    result = np.concatenate([expanded, tail], axis=1)
    final_topk = token_topk + compress_ratio - 1
    order = np.tile(np.arange(final_topk, dtype=np.int64)[None, :], (rows, 1))
    sort_key = np.where(result >= 0, order, order + final_topk)
    perm = np.argsort(sort_key, axis=1, kind="stable")
    return np.take_along_axis(result, perm, axis=1).astype(np.int32)


def qsa_select_tokens(
    q: np.ndarray,
    compressed_keys: np.ndarray,
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    query_positions: np.ndarray,
    sequence_lengths: np.ndarray,
    compress_ratio: int,
    token_topk: int,
) -> np.ndarray:
    """`QSAIndexer.select_prefill_tokens`: logits -> top-k -> expand."""
    logits = qsa_mqa_logits(q, compressed_keys, row_starts, row_ends)
    block_topk = token_topk // compress_ratio
    blocks = qsa_fast_topk(logits, row_starts, row_ends, block_topk)
    return qsa_expand_block_indices(
        blocks, query_positions, sequence_lengths, compress_ratio, token_topk
    )


def qsa_sparse_attention(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    token_slots: np.ndarray,
    softmax_scale: float | None = None,
) -> np.ndarray:
    """`kernel.py: qsa_sparse_attention_reference` (GQA over selected slots)."""
    scale = float(softmax_scale) if softmax_scale else float(q.shape[-1] ** -0.5)
    repeats = q.shape[1] // k_cache.shape[1]
    out = np.zeros(q.shape, dtype=np.float32)
    for row in range(q.shape[0]):
        valid = token_slots[row] >= 0
        slots = token_slots[row, valid].astype(np.int64)
        if slots.size == 0:
            continue
        keys = np.repeat(k_cache[slots], repeats, axis=1)   # (n, Hq, d)
        values = np.repeat(v_cache[slots], repeats, axis=1)
        scores = np.einsum("hd,khd->hk", q[row].astype(np.float32),
                           keys.astype(np.float32)) * scale
        m = scores.max(-1, keepdims=True)
        e = np.exp(scores - m)
        probs = e / e.sum(-1, keepdims=True)
        out[row] = np.einsum("hk,khd->hd", probs, values.astype(np.float32))
    return out.astype(q.dtype)
