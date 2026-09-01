"""DeepSeek-V4 text backbone ??numpy oracle.

Faithful CPU port of `ref/hf_sources/dsv4/model.py` (+ tilelang `kernel.py`
math): MLA with sliding window + learned KV compression (Compressor/Indexer),
Hyper-Connections with Sinkhorn, 256-expert MoE, DSpark target blocks.

Assumptions / deltas vs. the reference:
- The serving engine runs B=1 prefill/decode per request; this module is B=1.
- Weights arrive dequantized to fp32 (done by the loader); `_w` reads them.
- QAT activation simulation is implemented (cfg.qat_sim=True): in-place
  FP8/E8M0 round-trips of non-rope KV dims (block 64) and of the compressed
  rows, plus WHT+FP4 round-trips in the indexer (matching kernel.py with
  scale_fmt="ue8m0"). Matmul activation-quant for the weight GEMMs (block 128)
  is deliberately NOT simulated here -- the loader dequantizes weights, so
  that path is a cluster/l5 golden. Bit-exact tie-breaking also stays l5.
- The `Compressor`/`Indexer` decode state lives in the model state, not in
  module buffers; `prefill()` builds the state that `decode_step()` consumes.

Known float characteristic:
- BLAS `x @ W` dispatches to different kernels for (S,K) batched vs (1,K)
  single-row leading dims, so `_linear` of the same row can differ by ~1 ulp
  between prefill and decode. For the learned indexer that ulp feeds the
  top-k comparator, so the narrow indexer feed paths (wq_b, weights_proj,
  indexer compressor wkv/wgate) use the deterministic `_mm_det` to keep
  prefill == incremental bitwise there. The big GEMMs (attention q/kv, MoE
  gates, HC) keep BLAS and can show ~1e-8 hidden-state noise; in a tiny
  random-seed model with near-degenerate MoE gates that noise can amplify to
  ~1e-4 logit drift at ring/ratio-boundary--coincident positions (pos == 16k-1).
  That is intrinsic execution-form float noise (the reference torch model has
  the same class of prefill-vs-incremental divergence), not an algorithmic bug;
  bit-level cluster goldens (l5) remain the source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _w(m: dict, key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=np.float32)


def rms_norm(x, weight, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    m = np.mean(np.square(x, dtype=np.float64), axis=-1, keepdims=True)
    return (x * 1.0 / np.sqrt(m.astype(np.float32) + float(eps)) * w).astype(np.float32)


def _linear(x, w) -> np.ndarray:
    """x (..., K) @ w^T (N, K) -> (..., N) fp32."""
    return (np.asarray(x, dtype=np.float32)
            @ np.asarray(w, dtype=np.float32).T).astype(np.float32)


def _mm_det(x, w) -> np.ndarray:
    """Deterministic matmul: x (..., K) against w (N, K) -> (..., N) fp32.

    Elementwise-products-then-fixed-axis-sum makes the reduction order depend
    only on K (not on the leading batch shape), so batched-prefill and
    single-token-decode produce BITWISE-identical rows. The BLAS `@` in
    `_linear` dispatches to different kernels for (M,K) vs (1,K) leading dims
    and yields ~1 ulp differences; for the learned indexer that ulp noise can
    flip a top-k rank and feed 1e-4-scale logit drift at ring boundaries. Only
    the narrow indexer feed paths use this (x stays sllm-shaped and identical
    between prefill and decode, verified)."""
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    return (x[..., None, :] * w).sum(-1).astype(np.float32)


def silu(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)


def softmax_rows(x) -> np.ndarray:
    """Row softmax over the last axis; fp64 internal, fp32 out."""
    return _softmax_axis(x, axis=-1)


def _softmax_axis(x, axis: int) -> np.ndarray:
    """Stable softmax over a chosen axis; -inf/-inf columns collapse to 0."""
    x = np.asarray(x, dtype=np.float64)
    safe = np.where(np.isfinite(x), x, -np.inf)
    m = np.max(safe, axis=axis, keepdims=True)
    e = np.exp(safe - m)
    e = np.where(np.isfinite(e), e, 0.0)  # all -inf column -> 0
    s = np.sum(e, axis=axis, keepdims=True)
    return (e / np.where(s == 0, 1.0, s)).astype(np.float32)


# ---------------------------------------------------------------------------
# QAT activation simulation (phase 0)
#
# Matches the reference (kernel.py act_quant / fp4_act_quant / rotate_activation
# with scale_fmt="ue8m0"): per-row block-wise round-to-nearest with power-of-2
# (E8M0) scales, applied as an in-place quant+dequant round-trip. `rotate`
# (indexer) additionally applies a (signed, normalized) Walsh-Hadamard
# transform of the full trailing dim before the FP4 round-trip. This is an
# exact CPU port of the cluster numerics except tilelang's tie-breaking, which
# is irrelevant for random activations; bit-level goldens stay a l5 task.
# ---------------------------------------------------------------------------


def _e4m3_magnitudes() -> np.ndarray:
    """Sorted positive magnitudes of float8 E4M3FN (bias 7; max 448).

    The FN variant has no inf/nan: stored exp 15 is finite, but its top
    mantissa pattern (m=7 -> 480) is excluded so the max is 448."""
    vals = []
    for e in range(0, 16):          # stored exponent 0..15
        if e == 0:                  # subnormals: (m/8) * 2^-6 -> m * 2^-9
            for m in range(1, 8):
                vals.append((m / 8.0) * 2.0 ** -6)
        elif e == 15:               # top exponent, mantissa capped at m=6
            for m in range(0, 7):
                vals.append((1.0 + m / 8.0) * 2.0 ** (e - 7))
        else:                       # normals: (1 + m/8) * 2^(e-7)
            for m in range(0, 8):
                vals.append((1.0 + m / 8.0) * 2.0 ** (e - 7))
    return np.array(sorted(vals), np.float64)


_E4M3_TABLE = _e4m3_magnitudes()
_E2M1_TABLE = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float64)


def _next_pow2(v) -> np.ndarray:
    """2^ceil(log2(v)) as float32 (E8M0 scale); exact powers stay fixed."""
    v = np.asarray(v, np.float64)
    m, e = np.frexp(np.maximum(v, 1e-300))
    e = np.where(m > 0.5, e, e - 1)
    return (2.0 ** e).astype(np.float32)


def _round_table(x, table, vmax: float) -> np.ndarray:
    """Round-to-nearest (ties -> smaller magnitude) against a sorted table,
    then clamp to [-vmax, vmax]. Returns the dequantized value."""
    ax = np.abs(np.asarray(x, np.float64))
    i = np.searchsorted(table, ax, side="left")
    lo = np.clip(i - 1, 0, table.size - 1)
    hi = np.clip(i, 0, table.size - 1)
    below = table[lo]
    above = table[hi]
    near = np.where((above - ax) < (ax - below), above, below)
    near = np.clip(near, 0.0, vmax)
    return (np.sign(np.asarray(x)) * near).astype(np.float32)


def _fp8_rt(x, block_size: int = 64) -> np.ndarray:
    """In-place-style FP8 (E4M3) round-trip with per-`block_size` power-of-2
    (E8M0) scales along the last dim (reference act_quant(..64.., inplace))."""
    x = np.asarray(x, dtype=np.float32)
    last = x.shape[-1]
    if last % block_size != 0:      # tiny dev configs: one whole-row group
        block_size = last
    y = x.reshape((-1, last)).copy()
    nblk = last // block_size
    blocks = y.reshape(y.shape[0], nblk, block_size)
    amax = np.maximum(np.abs(blocks).max(-1), 1e-4)   # (M, nblk) fp32
    s = _next_pow2(amax * (1.0 / 448.0))[..., None]
    q = _round_table(blocks / s, _E4M3_TABLE, 448.0)
    blocks[:] = (q * s).astype(np.float32)
    return y.reshape(x.shape)


def _fp4_rt(x, block_size: int = 32) -> np.ndarray:
    """In-place-style FP4 (E2M1) round-trip with per-`block_size` power-of-2
    (E8M0) scales along the last dim (reference fp4_act_quant(..32..))."""
    x = np.asarray(x, dtype=np.float32)
    last = x.shape[-1]
    if last % block_size != 0:      # tiny dev configs: one whole-row group
        block_size = last
    y = x.reshape((-1, last)).copy()
    nblk = last // block_size
    blocks = y.reshape(y.shape[0], nblk, block_size)
    amax = np.maximum(np.abs(blocks).max(-1), 1e-30)  # (M, nblk) fp32
    s = _next_pow2(amax * (1.0 / 6.0))[..., None]
    q = _round_table(blocks / s, _E2M1_TABLE, 6.0)
    blocks[:] = (q * s).astype(np.float32)
    return y.reshape(x.shape)


def _wht_last(x) -> np.ndarray:
    """Signed Walsh-Hadamard transform on the trailing (power-of-two) dim,
    normalized by 1/sqrt(n) -- the reference rotate_activation."""
    x = np.asarray(x, np.float64)
    n = x.shape[-1]
    y = x[..., :].copy()
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            lo = y[..., i:i + h]
            hi = y[..., i + h:i + 2 * h]
            t = lo.copy()
            hi_c = hi.copy()
            lo[...] = t + hi_c
            hi[...] = t - hi_c
        h *= 2
    return (y / np.sqrt(n)).astype(np.float32)


def rotate_rows(x) -> np.ndarray:
    """rotate_activation + FP4 round-trip (block 32) used by the indexer's
    rotated compressor and its query. No-op unless QAT is enabled (the caller
    gates on cfg.qat_sim)."""
    y = _wht_last(x)
    return _fp4_rt(y, 32)


# ---------------------------------------------------------------------------
# YaRN rotary (interleaved pairs, scope = trailing rope_head_dim)
# ---------------------------------------------------------------------------


def find_correction_dim(num_rotations: float, dim: int, base: float,
                        max_seq_len: int) -> int:
    return int(dim * math.log(max_seq_len / (num_rotations * 2 * math.pi))
               / (2 * math.log(base)))


def find_correction_range(low_rot: float, high_rot: float, dim: int,
                          base: float, max_seq_len: int) -> tuple[int, int]:
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def precompute_freqs(dim: int, seqlen: int, original_seq_len: int, base: float,
                     factor: float, beta_fast: int, beta_slow: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin (seqlen, dim/2) with YaRN linear ramp when original_seq_len."""
    half = dim // 2
    freqs = 1.0 / (base ** (np.arange(half, dtype=np.float32) / dim))
    if original_seq_len != 0:
        freqs = np.outer(np.arange(seqlen, dtype=np.float32), freqs)
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len)
        if low < high:
            ramp = np.minimum(1.0, np.maximum(0.0,
                (np.arange(half, dtype=np.float32) - low) / (high - low)))
        else:
            ramp = np.ones(half, np.float32)
        freqs *= ((1 - ramp) * (1.0 / factor) + ramp)[None]
    else:
        freqs = np.outer(np.arange(seqlen, dtype=np.float32), freqs)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def apply_rotary_last(x, cos, sin, inverse: bool = False) -> np.ndarray:
    """Rotate the trailing rd dims as interleaved pairs; cos/sin (.., rd/2)
    broadcast against x's leading dims."""
    c = np.asarray(cos, dtype=np.float32)
    s = np.asarray(sin, dtype=np.float32)
    if x.ndim == 1:  # single token: cos/sin (1, rd/2) -> (rd/2,)
        c = c.reshape(-1)[: x.shape[-1] // 2]
        s = s.reshape(-1)[: x.shape[-1] // 2]
    else:
        # align cos/sin after x's leading dims: (S, rd/2) -> (S, 1, ..., rd/2)
        missing = (x.ndim - 1) - (c.ndim - 1)
        for _ in range(missing):
            c = np.expand_dims(c, -2)
            s = np.expand_dims(s, -2)
        c = np.broadcast_to(c, x.shape[:-1] + (x.shape[-1] // 2,))
        s = np.broadcast_to(s, x.shape[:-1] + (x.shape[-1] // 2,))
    xr = x[..., 0::2].copy()
    xi = x[..., 1::2].copy()
    if inverse:
        nr = xr * c + xi * s
        ni = xi * c - xr * s
    else:
        nr = xr * c - xi * s
        ni = xi * c + xr * s
    out = np.empty_like(x)
    out[..., 0::2] = nr
    out[..., 1::2] = ni
    return out


# ---------------------------------------------------------------------------
# window / compression index helpers (B=1)
# ---------------------------------------------------------------------------


def window_topk_idxs(window: int, seqlen: int, start_pos: int) -> np.ndarray:
    """(seqlen, window) indices into the window ring / prefix; -1 pads."""
    if start_pos >= window - 1:
        start_pos %= window  # ring order starts at (start_pos+1) % window
        m = np.concatenate([np.arange(start_pos + 1, window),
                            np.arange(0, start_pos + 1)], 0).astype(np.int32)
        return np.broadcast_to(m[None, :], (seqlen, window)).copy()
    if start_pos > 0:
        m = np.full((seqlen, window), -1, np.int32)
        m[:, :start_pos + 1] = np.arange(start_pos + 1)[None]
        return m
    base = np.arange(seqlen).astype(np.int32)[:, None]
    m = (base - window + 1).clip(0) + np.arange(min(seqlen, window))[None]
    return np.where(m > base, -1, m).astype(np.int32)


def compress_topk_idxs(ratio: int, seqlen: int, start_pos: int,
                       offset: int) -> np.ndarray:
    """(seqlen, n_comp) indices of compressed rows; -1 pads early positions."""
    if start_pos > 0:
        n = max((start_pos + 1) // ratio, 1)
        m = (np.arange(n) + offset).astype(np.int32)
        return np.broadcast_to(m[None, :], (seqlen, n)).copy()
    n = max(seqlen // ratio, 1)
    m = np.broadcast_to(np.arange(n, dtype=np.int32)[None], (seqlen, n)).copy()
    mask = m >= (np.arange(1, seqlen + 1)[:, None] // ratio)
    m = np.where(mask, -1, m + offset).astype(np.int32)
    return m


def get_image_visible(input_ids, vocab_size: int, max_image_tokens: int):
    """Per-token visible counts left/right within each image span (B=1)."""
    seqlen = len(input_ids)
    idx = np.arange(seqlen, dtype=np.int64)
    is_start = np.asarray(input_ids) == vocab_size + IMAGE_START
    is_end = np.asarray(input_ids) == vocab_size + IMAGE_END
    valid = (is_start.cumsum() > is_end.cumsum()) | is_end
    starts = np.maximum.accumulate(np.where(is_start, idx, 0))
    left = (idx - starts) * valid
    ends = np.minimum.accumulate(np.where(is_end, idx, seqlen)[::-1])[::-1]
    right = (ends - idx) * valid
    return (left.clip(max=max_image_tokens - 1).astype(np.int64),
            right.clip(max=max_image_tokens).astype(np.int64))


def window_topk_visible(window: int, seqlen: int, left, right,
                        max_image_tokens: int) -> np.ndarray:
    """Window topk for a prompt with image spans (reference
    get_window_topk_idxs_visible, B=1)."""
    left = np.asarray(left, np.int64)
    right = np.asarray(right, np.int64)
    width = min(seqlen, window + max_image_tokens)
    idx = np.arange(seqlen, dtype=np.int64)
    left_add = (left - (window - 1)).clip(0)
    starts = (idx - (window - 1) - left_add).clip(0)
    m = starts[:, None] + np.arange(width)[None]
    m = np.where(m > (idx + right)[:, None], -1, m)
    return m[:, :window].astype(np.int64)


# the sentinel id bases for the merge (vocab_size + IMAGE_*)
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4Cfg:
    vocab_size: int = 129280
    dim: int = 4096
    n_layers: int = 43
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    window_size: int = 128
    compress_ratios: tuple = (0, 0, 4, 128)
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    n_routed_experts: int = 256
    n_activated_experts: int = 6
    n_shared_experts: int = 1
    moe_inter_dim: int = 2048
    swiglu_limit: float = 10.0
    route_scale: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    n_hash_layers: int = 3
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    original_seq_len: int = 65536
    max_seq_len: int = 1048576
    dspark_block_size: int = 5
    dspark_markov_rank: int = 256
    dspark_target_layer_ids: tuple = (40, 41, 42)
    dspark_noise_token_id: int = 0
    n_mtp_layers: int = 3
    qat_sim: bool = False
    vision: bool = False
    vision_max_n_token: int = 384

    @classmethod
    def from_recipe(cls, recipe) -> "DeepseekV4Cfg":
        t = getattr(recipe, "text", None) or {}
        if not t:
            t = {
                "vocab_size": recipe.vocab_size,
                "hidden_size": recipe.hidden_size,
                "num_layers": recipe.num_layers,
                "max_position_embeddings": recipe.max_position_embeddings,
                "rms_norm_eps": recipe.rms_norm_eps,
            }
        spec = recipe.meta.get("spec", {}) or {}
        rs = spec.get("rope_scaling") or {}
        c = cls()
        c.vocab_size = int(t.get("vocab_size", c.vocab_size))
        c.dim = int(t.get("hidden_size", c.dim))
        c.n_layers = int(t.get("num_layers", c.n_layers))
        c.n_heads = int(t.get("num_heads", c.n_heads))
        c.max_seq_len = int(t.get("max_position_embeddings", c.max_seq_len))
        c.norm_eps = float(t.get("rms_norm_eps", c.norm_eps))
        c.head_dim = int(spec.get("head_dim", c.head_dim))
        c.rope_head_dim = int(spec.get("qk_rope_head_dim", c.rope_head_dim))
        c.window_size = int(spec.get("sliding_window", c.window_size))
        if spec.get("compress_ratios"):
            c.compress_ratios = tuple(int(x) for x in spec["compress_ratios"])
        c.q_lora_rank = int(spec.get("q_lora_rank", c.q_lora_rank))
        c.o_lora_rank = int(spec.get("o_lora_rank", c.o_lora_rank))
        c.o_groups = int(spec.get("o_groups", c.o_groups))
        c.n_routed_experts = int(t.get("n_routed_experts", c.n_routed_experts))
        c.n_activated_experts = int(t.get("num_experts_per_tok",
                                          c.n_activated_experts))
        c.n_shared_experts = int(t.get("n_shared_experts", c.n_shared_experts))
        c.moe_inter_dim = int(t.get("moe_intermediate_size", c.moe_inter_dim))
        c.swiglu_limit = float(spec.get("swiglu_limit", c.swiglu_limit))
        c.route_scale = float(spec.get("routed_scaling_factor", c.route_scale))
        c.scoring_func = str(spec.get("scoring_func", c.scoring_func))
        c.n_hash_layers = int(spec.get("num_hash_layers", c.n_hash_layers))
        c.index_n_heads = int(spec.get("index_n_heads", c.index_n_heads))
        c.index_head_dim = int(spec.get("index_head_dim", c.index_head_dim))
        c.index_topk = int(spec.get("index_topk", c.index_topk))
        c.hc_mult = int(spec.get("hc_mult", c.hc_mult))
        c.hc_sinkhorn_iters = int(spec.get("hc_sinkhorn_iters",
                                           c.hc_sinkhorn_iters))
        c.hc_eps = float(spec.get("hc_eps", c.hc_eps))
        c.norm_eps = float(t.get("rms_norm_eps", c.norm_eps))
        c.rope_theta = float(spec.get("rope_theta", c.rope_theta))
        c.compress_rope_theta = float(spec.get("compress_rope_theta",
                                               c.compress_rope_theta))
        c.rope_factor = float(rs.get("factor", c.rope_factor))
        c.rope_beta_fast = int(rs.get("beta_fast", c.rope_beta_fast))
        c.rope_beta_slow = int(rs.get("beta_slow", c.rope_beta_slow))
        c.original_seq_len = int(rs.get("original_max_position_embeddings",
                                        c.original_seq_len))
        c.max_seq_len = int(t.get("max_position_embeddings", c.max_seq_len))
        c.dspark_block_size = int(spec.get("dspark_block_size",
                                           c.dspark_block_size))
        c.dspark_markov_rank = int(spec.get("dspark_markov_rank",
                                            c.dspark_markov_rank))
        c.dspark_noise_token_id = int(spec.get("dspark_noise_token_id",
                                               c.dspark_noise_token_id))
        if spec.get("dspark_target_layer_ids"):
            c.dspark_target_layer_ids = tuple(
                int(x) for x in spec["dspark_target_layer_ids"])
        c.n_mtp_layers = int(t.get("num_nextn_predict_layers", c.n_mtp_layers))
        c.qat_sim = bool(spec.get("qat_sim", c.qat_sim))
        return c


# ---------------------------------------------------------------------------
# Hyper-Connections
# ---------------------------------------------------------------------------


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int,
                      sinkhorn_iters: int, eps: float):
    mixes = np.asarray(mixes, dtype=np.float32)
    hc = hc_mult
    pre = np.asarray(1.0 / (1.0 + np.exp(
        -(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]))), np.float32) + eps
    post = 2.0 * np.asarray(1.0 / (1.0 + np.exp(
        -(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]))),
        np.float32)
    raw = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = softmax_rows(raw.reshape(raw.shape[:-1] + (hc, hc)))
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    return pre, post, comb


def hc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, eps, norm_eps):
    """x (b, s, hc, d) -> (y (b,s,d), post, comb)."""
    x = np.asarray(x, dtype=np.float32)
    b, s, hc, d = x.shape
    flat = x.reshape(b, s, hc * d).astype(np.float64)
    rsqrt = 1.0 / np.sqrt(np.mean(flat ** 2, -1, keepdims=True) + norm_eps)
    mixes = (flat @ fn.T * rsqrt).astype(np.float32)
    pre, post, comb = hc_split_sinkhorn(mixes, np.asarray(scale, np.float32),
                                        np.asarray(base, np.float32), hc_mult,
                                        sinkhorn_iters, eps)
    y = np.sum(pre[..., None] * x, axis=2)
    return y.astype(np.float32), post, comb


def hc_post(x, residual, post, comb):
    return (post[..., None] * x[..., None, :]
            + np.sum(comb[..., None] * residual[..., None, :], axis=2)
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def _gate_scores(x, w, bias, scoring: str) -> np.ndarray:
    scores = _linear(x, w)
    if scoring == "softmax":
        scores = softmax_rows(scores)
    elif scoring == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-scores.astype(np.float64))).astype(
            np.float32)
    else:  # sqrtsoftplus
        scores = np.sqrt(softplus(scores))
    return scores


def softplus(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    return y.astype(np.float32)


def moe_route(x, w, bias, bias_vl, ids, topk, hash_layer, tid2eid, scoring,
              route_scale):
    """Returns (weights (S,topk), indices (S,topk))."""
    scores = _gate_scores(x, w, bias, scoring)
    original = scores
    if hash_layer:
        if bias_vl is None:
            idx = np.asarray(tid2eid, np.int64)[np.clip(ids, 0,
                                                        tid2eid.shape[0] - 1)]
        else:
            # image branch exercised by the vision path; text uses hash
            idx = np.asarray(tid2eid, np.int64)[np.clip(ids, 0,
                                                        tid2eid.shape[0] - 1)]
    else:
        biased = scores + np.asarray(bias, np.float32) if bias is not None \
            else scores
        idx = np.argsort(-biased, axis=-1)[..., :topk]
    weights = np.take_along_axis(original, idx, axis=-1)
    if scoring != "softmax":
        weights = weights / weights.sum(-1, keepdims=True)
    weights = weights * route_scale
    return weights.astype(np.float32), idx.astype(np.int64)


def expert_ffn(x, ws, limit: float):
    gate = _linear(x, ws[0])
    up = _linear(x, ws[1])
    if limit > 0:
        up = np.clip(up, -limit, limit)
        gate = np.clip(gate, None, limit)
    return _linear(silu(gate) * up, ws[2])


# ---------------------------------------------------------------------------
# MLA attention (window + compress + indexer + sink + o-lora)
# ---------------------------------------------------------------------------


def sparse_attn(q, kv_rows, idx, sink, scale) -> np.ndarray:
    """q (S, H, D); kv_rows (N, D) the row pool; idx (S, n_top) -> gather.
    Returns o (S, H, D) with the learnable sink in the denominator only."""
    S, H, D = q.shape
    ntop = idx.shape[1]
    valid = idx != -1
    gathered = np.where(valid[:, :, None], kv_rows[np.clip(idx, 0, None)],
                        np.zeros((S, ntop, D)))
    # scores (S, H, ntop)
    scores = np.einsum("shd,snd->shn", q, gathered) * scale
    maxv = scores.max(-1, keepdims=True)
    e = np.exp(scores - maxv)
    e = np.where(valid[:, None, :], e, 0.0)
    denom = e.sum(-1, keepdims=True) + np.exp(sink[None, :, None] - maxv)
    o = (np.einsum("shn,snd->shd", e, gathered) / denom).astype(np.float32)
    return o


class Compressor:
    """Gated pooling of KV rows; ratio==4 uses the overlap (coff=2) layout."""

    def __init__(self, cfg, ratio: int, head_dim: int, rotate: bool = False):
        self.cfg = cfg
        self.ratio = ratio
        self.head_dim = head_dim
        self.overlap = ratio == 4
        self.rotate = rotate
        self.coff = 1 + self.overlap
        self.d = head_dim
        self.qat = bool(cfg.qat_sim)
        self._rd = cfg.rope_head_dim

    def build(self, w: dict):
        self.wkv = _w(w, "wkv.weight")
        self.wgate = _w(w, "wgate.weight")
        self.ape = _w(w, "ape")

    def _qat_rows(self, rows):
        """QAT round-trip of already norm+roped compressed rows (reference
        Compressor.forward lines 408-412): rotate=True -> WHT + FP4 block-32
        on the full row; else FP8 block-64 on the non-rope dims only."""
        if not getattr(self, "qat", False):
            return rows
        if self.rotate:
            return rotate_rows(rows)
        out = rows.copy()
        out[..., :-self._rd] = _fp8_rt(out[..., :-self._rd], 64)
        return out

    def _pool(self, kv, score):
        """kv/score (n, ratio, 2d) [overlap] or (n, ratio, d) -> (n, d).
        Gated pooling across the TOKEN axis (softmax over ratio, then sum)."""
        if self.overlap:
            # per reference overlap_transform: last `ratio` rows come from the
            # CURRENT group's second half (d:), rows 1..ratio-1 borrow the
            # PREVIOUS group's first half (:d)
            kv1 = kv[..., :self.d]  # first half (overlap half)
            kv2 = kv[..., self.d:]  # second half (normal half)
            kvw = np.zeros((kv.shape[0], 2 * self.ratio, self.d), np.float32)
            kvw[:, self.ratio:] = kv2
            kvw[1:, :self.ratio] = kv1[:-1]
            scw = np.full((score.shape[0], 2 * self.ratio, self.d), -np.inf,
                          np.float32)
            scw[:, self.ratio:] = score[..., self.d:]
            scw[1:, :self.ratio] = score[:-1, :, :self.d]
        else:
            kvw = kv
            scw = score
        p = _softmax_axis(scw, axis=-2)  # over the token axis
        return (kvw * p).sum(-2)

    def prefill(self, kv, score, start_pos: int, n_comp_target: int):
        """Compress a full prefill: kv/score (S, dd) -> (n, d) compressed rows
        and a decode-continuation (kv_state, score_state) pair."""
        ratio, dd = self.ratio, self.d * self.coff
        S = kv.shape[0]
        n = min(max(S // ratio, 0), n_comp_target)
        if n <= 0:
            return np.zeros((0, self.d), np.float32), self.state_init()
        kv_sel = kv[:n * ratio].reshape(n, ratio, dd)
        sc_pre = (score[:n * ratio].reshape(n, ratio, dd)
                  + self.ape[:ratio][None])
        comp = self._pool(kv_sel, sc_pre)  # (n, d)
        if start_pos != 0:
            raise NotImplementedError(
                "Compressor.prefill is only used at start_pos==0")
        state = self._state_from_tail(kv, score, n)
        return comp.astype(np.float32), state

    def state_init(self):
        return (np.zeros((self.coff * self.ratio, self.d * self.coff),
                         np.float32),
                np.full((self.coff * self.ratio, self.d * self.coff), -np.inf,
                        np.float32))

    def _state_from_tail(self, kv, score, n):
        """Decode continuation mirroring the reference kv_state/score_state
        after a prefill ending at group n, incl. the partial remainder."""
        ratio, dd = self.ratio, self.d * self.coff
        st = self.state_init()
        kv_state, sc_state = st
        used = n * ratio
        rem = kv.shape[0] - used
        if rem == 0 and used > 0:
            tail = kv[used - ratio: used]
            tail_sc = score[used - ratio: used]
            if self.overlap:
                # the state rows hold the FULL 2*d columns (both halves)
                kv_state[:ratio, :] = tail
                sc_state[:ratio, :] = tail_sc + self.ape[:ratio]
            else:
                kv_state[:ratio] = tail
                sc_state[:ratio] = tail_sc + self.ape[:ratio]
        elif rem > 0:
            offset = ratio if self.overlap else 0
            if self.overlap and used >= ratio:
                # seed the overlap half with the last FULL group (both halves)
                kv_state[:ratio, :] = kv[used - ratio: used]
                sc_state[:ratio, :] = (score[used - ratio: used]
                                       + self.ape[:ratio])
            kv_state[offset: offset + rem] = kv[used: used + rem]
            sc_state[offset: offset + rem] = (score[used: used + rem]
                                              + self.ape[:rem])
        return st

    def decode_step(self, kv_tok, score_tok, pos: int, state):
        """Incremental decode: returns (compressed row (1,d) or None, state)."""
        ratio, dd = self.ratio, self.d * self.coff
        kv_state, sc_state = state
        kv_state = kv_state.copy()
        sc_state = sc_state.copy()
        if self.overlap:
            kv_state[ratio + pos % ratio] = kv_tok
            sc_state[ratio + pos % ratio] = score_tok + self.ape[pos % ratio]
            if (pos + 1) % ratio != 0:
                return None, (kv_state, sc_state)
            a = np.concatenate([kv_state[:ratio, :self.d],
                                kv_state[ratio:, self.d:]], 0)  # (2r, d)
            b = np.concatenate([sc_state[:ratio, :self.d],
                                sc_state[ratio:, self.d:]], 0)
            p = _softmax_axis(b, axis=0)  # over the token axis
            comp = (a * p).sum(0, keepdims=True)
            kv_state[:ratio] = kv_state[ratio:]
            sc_state[:ratio] = sc_state[ratio:]
            kv_state[ratio:] = 0.0
            sc_state[ratio:] = -np.inf
            return comp.astype(np.float32), (kv_state, sc_state)
        kv_state[pos % ratio] = kv_tok
        sc_state[pos % ratio] = score_tok + self.ape[pos % ratio]
        if (pos + 1) % ratio != 0:
            return None, (kv_state, sc_state)
        p = _softmax_axis(sc_state, axis=0)
        comp = (kv_state * p).sum(0, keepdims=True)
        kv_state[:] = 0.0
        sc_state[:] = -np.inf
        return comp.astype(np.float32), (kv_state, sc_state)


class Indexer:
    """Learned top-k compressed-KV positions; only for ratio==4 layers.

    The indexer carries its OWN compressor (rotate=True) whose params live
    under `layers.N.attn.indexer.compressor.*`.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.rope_head_dim
        self.topk = cfg.index_topk
        self.ratio = 4
        self.softmax_scale = self.head_dim ** -0.5
        self.freqs_cos = None

    def build(self, w: dict):
        """w scoped to `layers.N.attn.indexer.`."""
        self.wq_b = _w(w, "wq_b.weight")
        self.weights_proj = _w(w, "weights_proj.weight")
        self.cmp = Compressor(self.cfg, self.ratio, self.head_dim, rotate=True)
        self.cmp.build(w["compressor"])
        # the indexer carries its OWN compressor norm (index_head_dim rows)
        self.cmp.norm_w = _w(w["compressor"], "norm.weight")

    def comp_rotate(self, rows, freqs_cos, freqs_sin, pos):
        """norm (with the INDEXER's own compressor norm) + rope + QAT rotate:
        WHT + FP4 block-32 round-trip, matching the reference indexer path
        (the main MLP's comp_rotate must NOT be reused here: index_head_dim
        differs from head_dim for the real checkpoint)."""
        rows = rows.copy()
        rows = rms_norm(rows, self.cmp.norm_w, self.cfg.norm_eps)
        rows[..., -self.rope_dim:] = apply_rotary_last(
            rows[..., -self.rope_dim:],
            np.asarray(freqs_cos, np.float32)[pos],
            np.asarray(freqs_sin, np.float32)[pos])
        if self.cfg.qat_sim:
            rows = rotate_rows(rows)
        return rows

    def run(self, x, qr, start_pos: int, offset: int, comp_rows, freqs_cos,
            freqs_sin):
        """x (S, D), qr (S, lora); comp_rows (N, index_head_dim) are the
        compressed rows so far (maintained by the caller). -> (S, topk) idx."""
        S = x.shape[0]
        ratio = self.ratio
        q = _mm_det(qr, self.wq_b).reshape(S, self.n_heads, self.head_dim)
        q[..., -self.rope_dim:] = apply_rotary_last(
            q[..., -self.rope_dim:], freqs_cos[start_pos:start_pos + S],
            freqs_sin[start_pos:start_pos + S])
        if self.cfg.qat_sim:
            # reference: q = rotate_activation(q); fp4_act_quant(q, 32, True)
            q = rotate_rows(q)
        weights = _mm_det(x, self.weights_proj) * (self.softmax_scale
                                                   * self.n_heads ** -0.5)
        n = comp_rows.shape[0]
        topk = min(self.topk, max(n, 1))
        # index_score (S, H, N) = relu(q_h . comp_t) weighted by token head w
        sc = np.maximum(np.einsum("shd,td->sht", q, comp_rows[:n]), 0.0)
        sc = (sc * weights[:, :, None]).sum(1)  # (S, N)
        idx = np.argsort(-sc, axis=-1)[:, :topk].astype(np.int64)
        if start_pos == 0:
            # cannot attend to compressed tokens derived from future tokens
            future = idx >= (np.arange(1, S + 1)[:, None] // ratio)
            idx = np.where(future, -1, idx + offset)
        else:
            idx = idx + offset
        return idx


# ---------------------------------------------------------------------------
# per-layer MLA + MoE + HC block, and the model assembler (B=1)
# ---------------------------------------------------------------------------


@dataclass
class _LayerKV:
    """Decode-time per-layer KV state (ring + compressed + indexer)."""
    kv: np.ndarray                 # (win + comp_cap, head_dim)
    n_comp: int = 0                # compressed rows present
    cmp_kv: np.ndarray = None      # main compressor incremental state
    cmp_score: np.ndarray = None
    idx_comp: np.ndarray = None    # (comp_cap, index_head_dim) or None
    n_idx: int = 0
    idx_kv: np.ndarray = None      # indexer compressor incremental state
    idx_score: np.ndarray = None

    @property
    def win(self) -> int:
        return self.kv.shape[0] - (self.kv.shape[0] - self.win_size) if False else 0


class MLA:
    """One attention block. Pure math; decode KV state is carried by the
    caller and threaded through `decode()`."""

    def __init__(self, cfg: DeepseekV4Cfg, layer_id: int, w: dict):
        self.cfg = cfg
        self.layer_id = layer_id
        self.ratio = cfg.compress_ratios[layer_id] if layer_id < len(
            cfg.compress_ratios) else 0
        self.win = cfg.window_size
        self.rd = cfg.rope_head_dim
        self.H = cfg.n_heads
        self.D = cfg.head_dim
        self.scale = self.D ** -0.5
        self.n_groups = cfg.o_groups
        self.dg = (self.H * self.D) // self.n_groups
        self.o_lora = cfg.o_lora_rank
        self.dim = cfg.dim
        self.wq_a = _w(w, "wq_a.weight")
        self.q_norm = _w(w, "q_norm.weight")
        self.wq_b = _w(w, "wq_b.weight")
        self.wkv_w = _w(w, "wkv.weight")
        self.kv_norm = _w(w, "kv_norm.weight")
        self.wo_a = _w(w, "wo_a.weight")
        self.wo_b = _w(w, "wo_b.weight")
        self.sink = _w(w, "attn_sink")
        self.cmp = None
        self.idx = None
        if self.ratio:
            cw = w["compressor"]
            c = Compressor(cfg, self.ratio, self.D)
            c.build(cw)
            c.norm_w = _w(cw, "norm.weight")
            self.cmp = c
            if self.ratio == 4:
                self.idx = Indexer(cfg)
                self.idx.build(w["indexer"])

    def comp_rotate(self, rows, freqs_cos, freqs_sin, pos):
        rows = rows.copy()
        rows = rms_norm(rows, self.cmp.norm_w, self.cfg.norm_eps)
        rows[..., -self.rd:] = apply_rotary_last(rows[..., -self.rd:],
                                                 freqs_cos[pos],
                                                 freqs_sin[pos])
        # QAT: the main (non-rotate) compressor act-quants the non-rope dims
        return self.cmp._qat_rows(rows)

    def o_proj(self, o):
        S = o.shape[0]
        o = o.reshape(S, self.n_groups, self.dg)
        o = np.einsum("sgd,grd->sgr", o, self.wo_a.reshape(self.n_groups,
                                                           self.o_lora,
                                                           self.dg))
        return _linear(o.reshape(S, self.n_groups * self.o_lora), self.wo_b)

    # -- prefill (full forward for one sequence) -----------------------------

    def prefill(self, x, start_pos: int, cos, sin, visible=None) -> tuple[np.ndarray, _LayerKV]:
        """x (S, D); returns (o (S, D), layer decode-state)."""
        cfg = self.cfg
        S = x.shape[0]
        ratio = self.ratio
        win = self.win
        q = rms_norm(_linear(x, self.wq_a), self.q_norm, cfg.norm_eps)  # (S,lora)
        qr = q
        q = _linear(q, self.wq_b).reshape(S, self.H, self.D)
        q = q / np.sqrt(np.mean(q.astype(np.float64) ** 2, -1, keepdims=True)
                        .astype(np.float32) + cfg.norm_eps)
        q[..., -self.rd:] = apply_rotary_last(q[..., -self.rd:],
                                              cos[start_pos:start_pos + S],
                                              sin[start_pos:start_pos + S])
        kv = rms_norm(_linear(x, self.wkv_w), self.kv_norm, cfg.norm_eps)
        kv[..., -self.rd:] = apply_rotary_last(kv[..., -self.rd:],
                                               cos[start_pos:start_pos + S],
                                               sin[start_pos:start_pos + S])
        if cfg.qat_sim:
            # reference Attention.forward: act_quant(kv[..., :-rd], 64, .., True)
            kv = kv.copy()
            kv[..., :-self.rd] = _fp8_rt(kv[..., :-self.rd], 64)
        if visible is None:
            topk = window_topk_idxs(win, S, 0)
        else:
            topk = window_topk_visible(win, S, *visible,
                                       cfg.vision_max_n_token)
        n_comp = S // ratio if ratio else 0
        comp = None
        if ratio:
            kvs = _linear(x, self.cmp.wkv)
            sc = _linear(x, self.cmp.wgate)
            pooled, cstate = self.cmp.prefill(kvs, sc, 0, n_comp)
            if pooled.shape[0]:
                pos = np.arange(pooled.shape[0], dtype=np.int64) * ratio
                pooled = self.comp_rotate(pooled, cos, sin, pos)
                comp = pooled
            if self.idx is not None:
                ikvs = _mm_det(x, self.idx.cmp.wkv)
                isc = _mm_det(x, self.idx.cmp.wgate)
                ipool, istate = self.idx.cmp.prefill(ikvs, isc, 0, n_comp)
                ipool = self.idx.comp_rotate(ipool, cos, sin,
                                             np.arange(ipool.shape[0]) * ratio)
                cidx = self.idx.run(x, qr, 0, S, ipool, cos, sin)
            else:
                cidx = compress_topk_idxs(ratio, S, 0, S)
            topk = np.concatenate([topk, cidx], -1)
        rows = np.concatenate([kv, comp], 0) if comp is not None else kv
        o = sparse_attn(q, rows, topk, self.sink, self.scale)
        o[..., -self.rd:] = apply_rotary_last(o[..., -self.rd:],
                                              cos[start_pos:start_pos + S],
                                              sin[start_pos:start_pos + S],
                                              inverse=True)
        out = self.o_proj(o)
        # build decode state mirroring the reference prefill writes
        cap = cfg.max_seq_len // ratio + 1 if ratio else 0
        st = _LayerKV(kv=np.zeros((win + cap, self.D), np.float32))
        if S <= win:
            st.kv[:S] = kv
        else:
            cutoff = S % win
            st.kv[cutoff:win] = kv[-win:][:win - cutoff]
            st.kv[:cutoff] = kv[-win:][win - cutoff:]
        if ratio:
            st.n_comp = n_comp
            if comp is not None:
                st.kv[win: win + n_comp] = comp
            st.cmp_kv, st.cmp_score = cstate
            if self.idx is not None:
                capi = cfg.max_seq_len // 4 + 1
                st.idx_comp = np.zeros((capi, self.idx.head_dim), np.float32)
                st.n_idx = n_comp
                st.idx_comp[:n_comp] = ipool
                # indexer owns its compressor state (separate wkv/wgate/ape)
                st.idx_kv, st.idx_score = istate
        return out, st

    # -- decode (single token) -----------------------------------------------

    def decode(self, x_tok, pos: int, st: _LayerKV, cos, sin) -> np.ndarray:
        cfg = self.cfg
        ratio = self.ratio
        win = self.win
        q = rms_norm(_linear(x_tok[None], self.wq_a), self.q_norm,
                     cfg.norm_eps)[0]
        qr = q
        q = _linear(q[None], self.wq_b).reshape(self.H, self.D)
        q = q / np.sqrt(np.mean(q.astype(np.float64) ** 2, -1, keepdims=True)
                        .astype(np.float32) + cfg.norm_eps)
        q[..., -self.rd:] = apply_rotary_last(q[..., -self.rd:],
                                              cos[pos:pos + 1],
                                              sin[pos:pos + 1])
        kv = rms_norm(_linear(x_tok[None], self.wkv_w), self.kv_norm,
                      cfg.norm_eps)[0]
        kv[..., -self.rd:] = apply_rotary_last(kv[..., -self.rd:],
                                               cos[pos:pos + 1],
                                               sin[pos:pos + 1])
        if cfg.qat_sim:
            kv = kv.copy()
            kv[..., :-self.rd] = _fp8_rt(kv[..., :-self.rd], 64)
        st.kv[pos % win] = kv
        topk = window_topk_idxs(win, 1, pos)
        comp = None
        if ratio:
            kvs = _linear(x_tok[None], self.cmp.wkv)[0]
            sc = _linear(x_tok[None], self.cmp.wgate)[0]
            crow, (st.cmp_kv, st.cmp_score) = self.cmp.decode_step(
                kvs, sc, pos, (st.cmp_kv, st.cmp_score))
            if crow is not None:
                crow = self.comp_rotate(crow, cos, sin,
                                        np.array([pos + 1 - ratio]))
                st.kv[win + st.n_comp] = crow[0]
                st.n_comp += 1
            if self.idx is not None:
                ikvs = _mm_det(x_tok[None], self.idx.cmp.wkv)[0]
                isc = _mm_det(x_tok[None], self.idx.cmp.wgate)[0]
                irow, (st.idx_kv, st.idx_score) = self.idx.cmp.decode_step(
                    ikvs, isc, pos, (st.idx_kv, st.idx_score))
                if irow is not None:
                    irow = self.idx.comp_rotate(irow, cos, sin,
                                                np.array([pos + 1 - 4]))
                    st.idx_comp[st.n_idx] = irow[0]
                    st.n_idx += 1
                cidx = self.idx.run(x_tok[None], qr[None], pos, win,
                                    st.idx_comp[:st.n_idx], cos, sin)
            else:
                cidx = compress_topk_idxs(ratio, 1, pos, win)
            topk = np.concatenate([topk, cidx.reshape(1, -1)], -1)
        rows = st.kv[: win + st.n_comp]
        o = sparse_attn(q[None], rows, topk, self.sink, self.scale)[0]  # (H,D)
        o[..., -self.rd:] = apply_rotary_last(o[..., -self.rd:],
                                              cos[pos:pos + 1],
                                              sin[pos:pos + 1], inverse=True)
        return self.o_proj(o[None])[0]



def _collect_expert_weights(w: dict, p: str, cfg) -> list:
    experts = []
    for e in range(cfg.n_routed_experts):
        ep = f"{p}ffn.experts.{e}."
        experts.append((_w(w, ep + "w1.weight"), _w(w, ep + "w3.weight"),
                        _w(w, ep + "w2.weight")))
    return experts


class Block:
    """One layer: HC-pre -> attn_norm -> MLA -> HC-post -> HC-pre -> ffn_norm
    -> MoE -> HC-post. Stateless besides the MLA KV state passed per call."""

    def __init__(self, model, i: int):
        self.m = model
        cfg = model.cfg
        self.layer_id = i
        self.p = f"layers.{i}."
        self.w_attn = model.w_attn(self.p)
        self.attn = MLA(cfg, i, self.w_attn)
        self.attn_norm = _w(model.w, self.p + "attn_norm.weight")
        self.ffn_norm = _w(model.w, self.p + "ffn_norm.weight")
        self.ha_fn = _w(model.w, self.p + "hc_attn_fn")
        self.ha_sc = _w(model.w, self.p + "hc_attn_scale")
        self.ha_bs = _w(model.w, self.p + "hc_attn_base")
        self.hf_fn = _w(model.w, self.p + "hc_ffn_fn")
        self.hf_sc = _w(model.w, self.p + "hc_ffn_scale")
        self.hf_bs = _w(model.w, self.p + "hc_ffn_base")
        self.gate_w = _w(model.w, self.p + "ffn.gate.weight")
        gb = model.w.get(self.p + "ffn.gate.bias")
        self.gate_bias = np.asarray(gb, np.float32) if gb is not None else None
        self.tid2eid = model.w.get("tid2eid")
        self.experts = _collect_expert_weights(model.w, self.p, cfg)
        self.shared = (_w(model.w, self.p + "ffn.shared_experts.w1.weight"),
                       _w(model.w, self.p + "ffn.shared_experts.w3.weight"),
                       _w(model.w, self.p + "ffn.shared_experts.w2.weight"))

    def hc_attn(self, hidden):
        return hc_pre(hidden, self.ha_fn, self.ha_sc, self.ha_bs,
                      self.m.cfg.hc_mult, self.m.cfg.hc_sinkhorn_iters,
                      self.m.cfg.hc_eps, self.m.cfg.norm_eps)

    def hc_ffn(self, hidden):
        return hc_pre(hidden, self.hf_fn, self.hf_sc, self.hf_bs,
                      self.m.cfg.hc_mult, self.m.cfg.hc_sinkhorn_iters,
                      self.m.cfg.hc_eps, self.m.cfg.norm_eps)

    def moe(self, x, ids):
        cfg = self.m.cfg
        xf = np.asarray(x, np.float32).reshape(-1, cfg.dim)
        weights, idx = moe_route(
            xf, self.gate_w, self.gate_bias, None, ids,
            cfg.n_activated_experts, self.layer_id < cfg.n_hash_layers,
            self.tid2eid, cfg.scoring_func, cfg.route_scale)
        S = weights.shape[0]
        y = np.zeros((S, cfg.dim), np.float32)
        y += expert_ffn(xf, self.shared, cfg.swiglu_limit)
        for e in range(cfg.n_routed_experts):
            rows = np.where(idx == e)[0]
            if rows.size == 0:
                continue
            sel_w = weights[rows, [int((idx[rows] == e).argmax())]][:, None]
            y[rows] += expert_ffn(xf[rows], self.experts[e],
                                  cfg.swiglu_limit) * sel_w
        return y.astype(np.float32).reshape(x.shape[:-1] + (cfg.dim,))

    def prefill(self, hidden, ids, cos, sin, state, collect, visible=None):
        # hidden (1,S,hc,d); returns (1,S,d)
        vv, pp, cc = self.hc_attn(hidden)
        vv = rms_norm(vv, self.attn_norm, self.m.cfg.norm_eps)
        o, st = self.attn.prefill(np.asarray(vv[0], np.float32), 0, cos, sin,
                                  visible)
        if collect:
            state[self.layer_id] = st
        vv = hc_post(o, hidden[0], pp[0], cc[0])
        r0 = np.asarray(vv, np.float32)
        vv, pp, cc = self.hc_ffn(r0[None])
        vv = rms_norm(vv, self.ffn_norm, self.m.cfg.norm_eps)
        o = self.moe(vv, ids)
        return hc_post(o[0], r0, pp[0], cc[0])[None]

    def decode(self, hidden, ids, pos, cos, sin, st):
        # hidden (1,S,hc,d) with S=1
        vv, pp, cc = self.hc_attn(hidden)
        vv = rms_norm(vv, self.attn_norm, self.m.cfg.norm_eps)
        o = self.attn.decode(np.asarray(vv[0, 0], np.float32), pos, st, cos,
                             sin)
        vv = hc_post(o[None], hidden[0], pp[0], cc[0])
        r0 = np.asarray(vv, np.float32)
        vv, pp, cc = self.hc_ffn(r0[None])
        vv = rms_norm(vv, self.ffn_norm, self.m.cfg.norm_eps)
        o = self.moe(vv, ids)
        return hc_post(o[0], r0, pp[0], cc[0])[None]


class DeepseekV4Model:
    """embed -> HC-expand -> N Blocks -> HC-head -> norm -> head.

    `prefill(ids)` returns (state, logits (1,S,V)); `decode_step(state,
    last_id)` returns logits (V,). `supports_incremental=True`.
    """

    def __init__(self, weights: dict, cfg: DeepseekV4Cfg, recipe=None,
                 **kwargs):
        self.w = weights
        self.cfg = cfg
        self.embed_w = _w(weights, "embed.weight")
        self.head_w = _w(weights, "head.weight")
        self.final_norm = _w(weights, "norm.weight")
        self.hh_fn = _w(weights, "hc_head_fn")
        self.hh_sc = _w(weights, "hc_head_scale")
        self.hh_bs = _w(weights, "hc_head_base")
        self.hc_mult = cfg.hc_mult
        n = int(min(cfg.max_seq_len, 4096))
        self.cos, self.sin = precompute_freqs(
            cfg.rope_head_dim, n, cfg.original_seq_len, cfg.rope_theta,
            cfg.rope_factor, cfg.rope_beta_fast, cfg.rope_beta_slow)
        self.layers = [Block(self, i) for i in range(cfg.n_layers)]
        self.vision = None
        if "vision.patch_embed.proj.weight" in weights:
            from .vision_deepseek import DeepseekVisionEncoder, VisionCfg
            vdim = np.asarray(weights["vision.patch_embed.proj.weight"],
                              np.float32).shape[1]
            vision_cfg = kwargs.get("vision_cfg")
            self.vision = DeepseekVisionEncoder(
                weights, vision_cfg or VisionCfg(dim=vdim),
                bf16=bool(kwargs.get("vision_bf16", False)))
            self.image_params = {
                "image_start": np.asarray(weights["image_start"], np.float32),
                "image_pad": np.asarray(weights["image_pad"], np.float32),
                "image_end": np.asarray(weights["image_end"], np.float32),
                "image_newline": np.asarray(weights["image_newline"],
                                            np.float32),
            }
        self.supports_incremental = True
        self.recipe = recipe

    def max_context(self) -> int:
        return int(self.cfg.max_seq_len)

    def encode_image(self, patches, n_vit_h, n_vit_w) -> np.ndarray:
        return self.vision.align(self.vision.vit(patches, n_vit_h, n_vit_w),
                                 n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h) -> np.ndarray:
        """Write each image's token block into h in place (B=1)."""
        from serving.image_processor import (IMAGE, IMAGE_END, IMAGE_NEW_LINE,
                                             IMAGE_PAD, IMAGE_START)
        h = np.asarray(h, np.float32)
        params = np.stack([self.image_params["image_start"],
                           self.image_params["image_pad"],
                           self.image_params["image_pad"],
                           self.image_params["image_newline"],
                           self.image_params["image_end"]])
        for img in images:
            embeds = self.encode_image(img.patches, img.n_vit_h,
                                       img.n_vit_w)[img.perm]
            types = np.asarray(img.types, np.int64)
            block = params[types].copy()
            block[types == IMAGE] = embeds
            h[img.start: img.start + block.shape[0]] = block
        return h

    def w_attn(self, layer_p: str) -> dict:
        a = layer_p + "attn."
        out = {}
        for k, v in self.w.items():
            if not k.startswith(a):
                continue
            rel = k[len(a):]
            if rel.startswith("compressor."):
                out.setdefault("compressor", {})[rel[len("compressor."):]] = \
                    np.asarray(v, np.float32)
            elif rel.startswith("indexer."):
                inner = rel[len("indexer."):]
                if inner.startswith("compressor."):
                    out.setdefault("indexer", {}).setdefault(
                        "compressor", {})[inner[len("compressor."):]] = \
                        np.asarray(v, np.float32)
                else:
                    out.setdefault("indexer", {})[inner] = np.asarray(v,
                                                                      np.float32)
            else:
                out[rel] = np.asarray(v, np.float32)
        return out

    def _expand(self, h):
        # h (1, S, d) -> (1, S, hc, d)
        return np.broadcast_to(h[:, :, None, :],
                               (h.shape[0], h.shape[1], self.hc_mult,
                                h.shape[2]))

    def _hc_head(self, hidden):
        b, s, hc, d = hidden.shape
        flat = hidden.reshape(b, s, hc * d).astype(np.float64)
        rsqrt = 1.0 / np.sqrt(np.mean(flat ** 2, -1, keepdims=True)
                         + self.cfg.norm_eps)
        mixes = flat @ self.hh_fn.T * rsqrt
        pre = 1.0 / (1.0 + np.exp(-(mixes * self.hh_sc[0] + self.hh_bs)))
        pre = pre.astype(np.float32) + self.cfg.hc_eps
        return np.sum(pre[..., None] * hidden, axis=2).astype(np.float32)

    def prefill(self, ids, images=None, spec=False):
        cfg = self.cfg
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if (ids >= cfg.vocab_size).any() and not images:
            raise ValueError(
                "image sentinel tokens need their ImageInput list; pass "
                "images= to prefill()")
        S = ids.size
        h = self.embed_w[np.minimum(ids, self.embed_w.shape[0] - 1)].astype(
            np.float32)
        visible = None
        if images:
            h = self.merge_image_embeddings(images, h)
            visible = get_image_visible(ids, cfg.vocab_size,
                                        cfg.vision_max_n_token)
        h = h[None]
        hidden = self._expand(h)
        state = [None] * cfg.n_layers
        targets = cfg.dspark_target_layer_ids
        main_hiddens = [] if spec and targets else None
        for i, layer in enumerate(self.layers):
            hidden = layer.prefill(hidden, ids, self.cos, self.sin, state,
                                   True, visible)
            if main_hiddens is not None and i in targets:
                main_hiddens.append(hidden[0].mean(1))  # (S, d)
        hidden = self._hc_head(hidden)
        hidden = rms_norm(hidden, self.final_norm, cfg.norm_eps)
        logits = _linear(hidden[0], self.head_w)  # (S, V)
        state.append(S)
        if spec:
            mh = (np.concatenate(main_hiddens, -1) if main_hiddens
                  else np.zeros((S, 0), np.float32))
            return state, logits, mh
        return state, logits

    def decode_step(self, state, last_id: int, spec=False):
        cfg = self.cfg
        pos = state[-1]
        if int(last_id) >= cfg.vocab_size:
            raise ValueError(
                "image spans must be prefilled in a single chunk; cannot "
                "decode an image sentinel token")
        ids = np.asarray([last_id], dtype=np.int64)
        h = self.embed_w[ids].astype(np.float32)[None]
        hidden = self._expand(h)
        targets = cfg.dspark_target_layer_ids
        main_hiddens = [] if spec and targets else None
        for i, layer in enumerate(self.layers):
            hidden = layer.decode(hidden, ids, pos, self.cos, self.sin,
                                  state[i])
            if main_hiddens is not None and i in targets:
                main_hiddens.append(hidden[0, 0].mean(0))  # (d,)
        hidden = self._hc_head(hidden)
        hidden = rms_norm(hidden, self.final_norm, cfg.norm_eps)
        logits = _linear(hidden[0, 0], self.head_w)
        state[-1] = pos + 1
        if spec:
            mh = (np.concatenate(main_hiddens) if main_hiddens
                  else np.zeros((0,), np.float32))
            return logits, mh
        return logits

# ---------------------------------------------------------------------------
# DSPark speculative-decode stages (phase 1)
#
# Ports the reference model.py DSparkAttention/DSparkMarkovHead/
# DSparkConfidenceHead/DSparkBlock/forward_embed/forward_head math under the
# `mtp.*` checkpoint namespace. Two deliberate, documented deltas:
#   - The reference window topk for the draft block uses raw KV slots with no
#     ring rotation (get_dspark_topk_idxs), which is only valid when the ring
#     has not wrapped; we reuse the window_topk ring subscripts from the main
#     MLA so the draft attends the actual last `win` real positions. Draft
#     exactness does not affect correctness -- runtime/spec.py verifies every
#     drafted token against the main greedy (accepted prefixes are equivalent
#     to plain greedy by construction).
#   - The reference forward_embed seeds the draft block with a noise token and
#     MOE-routes draft tokens with the single input id (a length mismatch for
#     block_size>1); we route the draft block with its own block ids.
#
# `DeepseekV4SpecModel` is stateful (own per-stage main-KV window); the main
# decoder must be run with spec=True to expose the target-layer hidden means
# that forward_embed consumes.
# ---------------------------------------------------------------------------


class DSparkMLA:
    """Sliding-window draft attention: attends the stored main-KV window plus
    the draft block's own KV (no compression)."""

    def __init__(self, cfg, w: dict):
        self.cfg = cfg
        self.win = cfg.window_size
        self.rd = cfg.rope_head_dim
        self.H = cfg.n_heads
        self.D = cfg.head_dim
        self.scale = self.D ** -0.5
        self.n_groups = cfg.o_groups
        self.dg = (self.H * self.D) // self.n_groups
        self.o_lora = cfg.o_lora_rank
        self.wq_a = _w(w, "wq_a.weight")
        self.q_norm = _w(w, "q_norm.weight")
        self.wq_b = _w(w, "wq_b.weight")
        self.wkv_w = _w(w, "wkv.weight")
        self.kv_norm = _w(w, "kv_norm.weight")
        self.wo_a = _w(w, "wo_a.weight")
        self.wo_b = _w(w, "wo_b.weight")
        self.sink = _w(w, "attn_sink")

    def main_kv_row(self, main_x, cos, sin, pos):
        kv = rms_norm(_linear(np.asarray(main_x, np.float32)[None],
                              self.wkv_w), self.kv_norm, self.cfg.norm_eps)[0]
        kv[..., -self.rd:] = apply_rotary_last(kv[..., -self.rd:],
                                               cos[pos:pos + 1],
                                               sin[pos:pos + 1])
        if self.cfg.qat_sim:
            kv = kv.copy()
            kv[..., :-self.rd] = _fp8_rt(kv[..., :-self.rd], 64)
        return kv

    def forward(self, x_blk, kv_cache, start_pos, cos, sin):
        """x_blk (bs, D); kv_cache (win, D) of main rows (mutated in place ->
        returns a copy of the concat rows for attention). Returns o (bs, D)."""
        bs = x_blk.shape[0]
        cfg = self.cfg
        q = rms_norm(_linear(x_blk, self.wq_a), self.q_norm, cfg.norm_eps)
        q = _linear(q, self.wq_b).reshape(bs, self.H, self.D)
        q = q / np.sqrt(np.mean(q.astype(np.float64) ** 2, -1, keepdims=True)
                        .astype(np.float32) + cfg.norm_eps)
        q[..., -self.rd:] = apply_rotary_last(
            q[..., -self.rd:], cos[start_pos + 1:start_pos + 1 + bs],
            sin[start_pos + 1:start_pos + 1 + bs])
        kv = rms_norm(_linear(x_blk, self.wkv_w), self.kv_norm, cfg.norm_eps)
        kv[..., -self.rd:] = apply_rotary_last(
            kv[..., -self.rd:], cos[start_pos + 1:start_pos + 1 + bs],
            sin[start_pos + 1:start_pos + 1 + bs])
        if cfg.qat_sim:
            kv = kv.copy()
            kv[..., :-self.rd] = _fp8_rt(kv[..., :-self.rd], 64)
        topk = window_topk_idxs(self.win, 1, start_pos)  # (1, win) ring slots
        draft_slots = self.win + np.arange(bs, dtype=np.int32)[None, :]
        topk = np.concatenate([topk, draft_slots], -1)
        rows = np.concatenate([kv_cache, kv], 0)          # (win+bs, D)
        o = sparse_attn(q, rows, topk, self.sink, self.scale)
        o[..., -self.rd:] = apply_rotary_last(
            o[..., -self.rd:], cos[start_pos + 1:start_pos + 1 + bs],
            sin[start_pos + 1:start_pos + 1 + bs], inverse=True)
        o = o.reshape(bs, self.n_groups, self.dg)
        o = np.einsum("sgd,grd->sgr", o,
                      self.wo_a.reshape(self.n_groups, self.o_lora, self.dg))
        return _linear(o.reshape(bs, self.n_groups * self.o_lora), self.wo_b)


class DSparkStage:
    """One mtp.{i} stage: optional embed head (stage 0) / head (last), an
    HC-attn (DSparkMLA) + HC-ffn block with a MoE. Holds its own main-KV
    window."""

    def __init__(self, model, stage_id: int):
        self.m = model
        self.cfg = model.cfg
        self.stage_id = stage_id
        self.p = f"mtp.{stage_id}."
        w = model.w
        self.attn = DSparkMLA(self.cfg, model.scope(self.p + "attn."))
        self.attn_norm = _w(w, self.p + "attn_norm.weight")
        self.ffn_norm = _w(w, self.p + "ffn_norm.weight")
        self.ha_fn = _w(w, self.p + "hc_attn_fn")
        self.ha_sc = _w(w, self.p + "hc_attn_scale")
        self.ha_bs = _w(w, self.p + "hc_attn_base")
        self.hf_fn = _w(w, self.p + "hc_ffn_fn")
        self.hf_sc = _w(w, self.p + "hc_ffn_scale")
        self.hf_bs = _w(w, self.p + "hc_ffn_base")
        self.gate_w = _w(w, self.p + "ffn.gate.weight")
        gb = w.get(self.p + "ffn.gate.bias")
        self.gate_bias = np.asarray(gb, np.float32) if gb is not None else None
        self.tid2eid = w.get("tid2eid")
        self.experts = _collect_expert_weights(w, self.p, self.cfg)
        self.shared = (_w(w, self.p + "ffn.shared_experts.w1.weight"),
                       _w(w, self.p + "ffn.shared_experts.w3.weight"),
                       _w(w, self.p + "ffn.shared_experts.w2.weight"))
        self.kv_cache = np.zeros((self.cfg.window_size, self.cfg.head_dim),
                                 np.float32)
        self.n_main = 0
        # stage heads
        self.main_proj = None
        self.main_norm_w = None
        self.head_norm_w = None
        self.markov_w1 = None
        self.markov_w2 = None
        self.conf_proj = None
        self.hc_head_fn = None
        self.hc_head_base = None
        self.hc_head_scale = None
        if stage_id == 0:
            self.main_proj = _w(w, self.p + "main_proj.weight")
            self.main_norm_w = _w(w, self.p + "main_norm.weight")
        if stage_id == self.cfg.n_mtp_layers - 1:
            self.head_norm_w = _w(w, self.p + "norm.weight")
            self.markov_w1 = _w(w, self.p + "markov_head.markov_w1.weight")
            self.markov_w2 = _w(w, self.p + "markov_head.markov_w2.weight")
            self.conf_proj = _w(w, self.p + "confidence_head.proj.weight")
            self.hc_head_fn = _w(w, self.p + "hc_head_fn")
            self.hc_head_base = _w(w, self.p + "hc_head_base")
            self.hc_head_scale = _w(w, self.p + "hc_head_scale")

    def hc_attn(self, hidden):
        return hc_pre(hidden, self.ha_fn, self.ha_sc, self.ha_bs,
                      self.cfg.hc_mult, self.cfg.hc_sinkhorn_iters,
                      self.cfg.hc_eps, self.cfg.norm_eps)

    def hc_ffn(self, hidden):
        return hc_pre(hidden, self.hf_fn, self.hf_sc, self.hf_bs,
                      self.cfg.hc_mult, self.cfg.hc_sinkhorn_iters,
                      self.cfg.hc_eps, self.cfg.norm_eps)

    def moe(self, x, ids):
        cfg = self.cfg
        xf = np.asarray(x, np.float32).reshape(-1, cfg.dim)
        weights, idx = moe_route(
            xf, self.gate_w, self.gate_bias, None, ids,
            cfg.n_activated_experts, self.stage_id < cfg.n_hash_layers,
            self.tid2eid, cfg.scoring_func, cfg.route_scale)
        S = weights.shape[0]
        y = np.zeros((S, cfg.dim), np.float32)
        y += expert_ffn(xf, self.shared, cfg.swiglu_limit)
        for e in range(cfg.n_routed_experts):
            rows = np.where(idx == e)[0]
            if rows.size == 0:
                continue
            sel_w = weights[rows, [int((idx[rows] == e).argmax())]][:, None]
            y[rows] += expert_ffn(xf[rows], self.experts[e],
                                  cfg.swiglu_limit) * sel_w
        return y.astype(np.float32).reshape(x.shape[:-1] + (cfg.dim,))

    def forward(self, h, start_pos, ids, main_x, cos, sin):
        """h (1, bs, hc, d); main_x (1, d) for the pos being extended."""
        cfg = self.cfg
        bs = h.shape[1]
        # update this stage's main-KV window with the current position row
        row = self.attn.main_kv_row(main_x[0], cos, sin, start_pos)
        self.kv_cache[start_pos % self.win] = row
        self.n_main += 1
        vv, pp, cc = self.hc_attn(h)
        vv = rms_norm(vv, self.attn_norm, cfg.norm_eps)
        right = self.win + bs if (self.win + bs) <= self.kv_cache.shape[0] \
            else self.kv_cache.shape[0]
        o = self.attn.forward(np.asarray(vv[0], np.float32), self.kv_cache,
                              start_pos, cos, sin)
        vv = hc_post(o[None], h, pp, cc)
        r0 = np.asarray(vv, np.float32)
        vv, pp, cc = self.hc_ffn(r0)
        vv = rms_norm(vv, self.ffn_norm, cfg.norm_eps)
        o = self.moe(vv, ids)
        return hc_post(o[0], r0[0], pp[0], cc[0])[None]

    @property
    def win(self):
        return self.cfg.window_size

    def forward_embed(self, main_hidden_p, prev_id):
        """main_hidden_p (Td,); prev_id scalar. Returns (main_x (1, d),
        draft h (1, bs, hc, d))."""
        mx = rms_norm(_linear(main_hidden_p[None], self.main_proj),
                      self.main_norm_w, self.cfg.norm_eps)[0]
        bs = self.cfg.dspark_block_size
        draft = np.full((bs,), self.cfg.dspark_noise_token_id, np.int64)
        draft[0] = prev_id
        x = self.m.embed_w[np.minimum(draft, self.m.embed_w.shape[0] - 1)]
        x = np.broadcast_to(x[:, None, :], (bs, self.cfg.hc_mult,
                                            self.cfg.dim)).copy()
        return mx[None], x[None]

    def forward_head(self, h, prev_id, sample="greedy"):
        """h (1, bs, hc, d). Returns (draft_ids (bs+1,), logits (bs, V),
        confidence (bs,))."""
        cfg = self.cfg
        bs = h.shape[1]
        x = self.hc_head(h)
        logits = _linear(rms_norm(x[0], self.head_norm_w, cfg.norm_eps),
                         self.m.head_w)  # (bs, V)
        out = np.empty(bs + 1, np.int64)
        out[0] = prev_id
        embeds = np.zeros((bs, cfg.dspark_markov_rank), np.float32)
        for i in range(bs):
            e = np.asarray(self.markov_w1[np.minimum(out[i],
                                                     self.markov_w1.shape[0] -
                                                     1)], np.float32)
            bias = _linear(e[None], self.markov_w2)[0]
            logits[i] = logits[i] + bias
            embeds[i] = e
            out[i + 1] = (int(np.argmax(logits[i]))
                          if sample == "greedy" else int(np.argmax(logits[i])))
        conf = _linear(np.concatenate([x[0], embeds], -1), self.conf_proj)
        return out, logits, np.asarray(conf, np.float32).reshape(bs)

    def hc_head(self, h):
        b, s, hc, d = h.shape
        flat = h.reshape(b, s, hc * d).astype(np.float64)
        rsqrt = 1.0 / np.sqrt(np.mean(flat ** 2, -1, keepdims=True)
                              + self.cfg.norm_eps)
        mixes = flat @ self.hc_head_fn.T * rsqrt
        pre = (1.0 / (1.0 + np.exp(-(mixes * self.hc_head_scale[0]
                                     + self.hc_head_base)))
               .astype(np.float32) + self.cfg.hc_eps)
        return np.sum(pre[..., None] * h, axis=2).astype(np.float32)


class DeepseekV4SpecModel:
    """DSPark speculative head over a `DeepseekV4Model` (B=1).

    Usage (matching runtime/spec.py's verify loop):
        st, _, mh_t = main.prefill(prefix, spec=True)
        spec = DeepseekV4SpecModel(weights, cfg, main)
        spec.setup(mh_t)
        # per step: main decode_step(state, tok, spec=True) -> (logits, mh)
        #           spec.draft_step(prev_id, mh, pos) -> (draft_ids, logits, conf)
    """

    def __init__(self, weights: dict, cfg, main):
        self.w = weights
        self.cfg = cfg
        self.main = main
        self.stages = [DSparkStage(self, i)
                       for i in range(cfg.n_mtp_layers)]

    def scope(self, prefix):
        out = {}
        for k, v in self.w.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    @property
    def embed_w(self):
        return self.main.embed_w

    @property
    def head_w(self):
        return self.main.head_w

    def setup(self, main_hiddens):
        """Seed each stage's main-KV window from the prefix's target-hidden
        means (S, Td), window ring layout identical to the main MLA."""
        cfg = self.cfg
        win = cfg.window_size
        S = main_hiddens.shape[0]
        main_x = rms_norm(_linear(main_hiddens, self.stages[0].main_proj),
                          self.stages[0].main_norm_w, cfg.norm_eps)
        for st in self.stages:
            rows = self._stage_main_rows(st, main_x, S)
            if S <= win:
                st.kv_cache[:S] = rows
            else:
                cutoff = S % win
                st.kv_cache[cutoff:win] = rows[-win:][:win - cutoff]
                st.kv_cache[:cutoff] = rows[-win:][win - cutoff:]
            st.n_main = S

    def _stage_main_rows(self, st, main_x, S):
        kvs = rms_norm(_linear(main_x, st.attn.wkv_w), st.attn.kv_norm,
                       self.cfg.norm_eps)
        kvs = kvs.copy()
        kvs[..., -self.cfg.rope_head_dim:] = apply_rotary_last(
            kvs[..., -self.cfg.rope_head_dim:],
            self.main.cos[0:S], self.main.sin[0:S])
        if self.cfg.qat_sim:
            kvs[..., :-self.cfg.rope_head_dim] = _fp8_rt(
                kvs[..., :-self.cfg.rope_head_dim], 64)
        return kvs

    def draft_step(self, prev_id, main_hidden, pos, sample="greedy"):
        """Returns (draft_ids (bs+1,), logits (bs, V), confidence (bs,))."""
        cfg = self.cfg
        bs = cfg.dspark_block_size
        main_x, h = self.stages[0].forward_embed(main_hidden, prev_id)
        ids = np.full((bs,), cfg.dspark_noise_token_id, np.int64)
        ids[0] = prev_id
        for st in self.stages:
            h = st.forward(h, pos, ids, main_x, self.main.cos, self.main.sin)
        return self.stages[-1].forward_head(h, prev_id, sample)
