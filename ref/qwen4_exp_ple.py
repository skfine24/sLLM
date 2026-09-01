"""P++LE (Pronomial Lexical Embedding) numpy oracle for qwen4_exp -- Phase 6.

Extracts the deterministic n-gram machinery + the PLE feature cell from the
vendored upstream `ref/hf_sources/modeling_qwen4_exp.py` (line refs below)
into pure numpy, so the pipeline can be integrated and parity-tested on CPU
without torch.

Everything here is EXACT integer/float replication of the upstream code:

  * prime/head layout            -- modeling_qwen4_exp.py:998-1051
  * splitmix64 / multipliers     -- :972-995
  * shift-right-ignore-eos       -- :1053-1067
  * n-gram hashing + lookup      -- :1069-1114
  * PLE feature cell             -- :1117-1189 (norm_key/value/norm_query/
                                      gate/conv + short conv state)

NOT included (documented successor work -- needs token-id threading through
the hyper-injection path + synthetic PLE weights in the tiny fixture):
wiring into `ref/qwen4_exp_pipeline._forward` (`hidden_states += PLE` at the
top of each PLE layer, DecoderLayer.forward:1217-1220) and the ngram-protected
TopKRouter scoring (SparseMoeBlock router sees the ngram embedding).

Shapes (following upstream, hidden_size==hs, hc==hc_count):
  ngram embedding table        (padded_vocab, head_dim_per_ngram)
  PLE output                   (B, S, hc*hs) -- sits on the hyper stream
"""

from __future__ import annotations

import math

import numpy as np

# upstream constants (modeling_qwen4_exp.py:972-976)
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def splitmix64(value: int) -> int:
    """stateless splitmix64 mix, upstream :979-983."""
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    """upstream :998-1006."""
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for d in range(3, math.isqrt(value) + 1, 2):
        if value % d == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    """upstream :1009-1015."""
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int,
                            ple_layer_index: int, seed: int) -> np.ndarray:
    """Int64 multipliers, upstream :986-995."""
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = np.empty(ngram_size, dtype=np.int64)
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out[index] = 2 * (splitmix64(value) % half_bound) + 1
    return out


class NGramLayout:
    """Head vocab sizes/offsets + padded total (Qwen4ExpTextNGramEmbedding
    __init__, upstream :1018-1051)."""

    def __init__(self, vocab_size: int, ngram_size: int, heads_per_ngram: int,
                 ngram_vocab_size_base: int, ple_layer_index: int,
                 make_ngram_vocab_size_divisible_by: int = 64, seed: int = 0):
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_heads = (ngram_size - 1) * heads_per_ngram
        self.context_len = ngram_size - 1
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.ple_layer_index = ple_layer_index
        sizes, offsets = [], []
        total = 0
        for head_idx in range(self.ngram_heads):
            g = self.ple_layer_index
            size = find_nth_prime_after(ngram_vocab_size_base - 1,
                                        g * self.ngram_heads + head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.head_vocab_sizes = np.asarray(sizes, dtype=np.int64)
        self.head_offsets = np.asarray(offsets, dtype=np.int64)
        self.total_vocab_size = total
        self.padded_vocab_size = (int(math.ceil(total / make_ngram_vocab_size_divisible_by))
                                  * make_ngram_vocab_size_divisible_by)
        self.layer_multipliers_int = build_layer_multipliers(
            vocab_size, ngram_size, ple_layer_index, seed)


def shift_right_ignore_eos(token_ids: np.ndarray, shift: int,
                           eos_token_id: int) -> np.ndarray:
    """upstream :1053-1067. token_ids (B, S) int64 -> shifted (B, S),
    positions crossing an EOS boundary (or shifting past position 0) are
    EOS-filled."""
    if shift == 0:
        return token_ids
    b, seq = token_ids.shape
    positions = np.arange(seq, dtype=np.int64)[None, :]
    eos_positions = np.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = np.maximum.accumulate(eos_positions, axis=1)
    previous_eos = np.concatenate(
        [np.full((b, 1), -1, np.int64), previous_eos_inclusive[:, :-1]], axis=1)
    segment_start = previous_eos + 1
    position_in_segment = positions - segment_start
    source_positions = positions - shift
    gather_positions = np.clip(source_positions, 0, None)
    shifted = token_ids[np.arange(b)[:, None], gather_positions]
    valid = (position_in_segment >= shift) & (source_positions >= 0)
    return np.where(valid, shifted, eos_token_id)


def ngram_ids(input_ids: np.ndarray, layout: NGramLayout,
              previous_context: np.ndarray | None = None,
              eos_token_id: int = 151643) -> np.ndarray:
    """N-gram head indices for each token. upstream :1093-1112.

    input_ids (B, S); returns (B, S, total_heads) int64 in [0, padded_vocab).
    `previous_context` is the right-aligned trailing tokens from before the
    window (B, context_len) -- pass the LAST `context_len` tokens of the prior
    chunk (or None at stream start -> all-EOS padding), later slices drop it.
    """
    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    b, s = ids.shape
    ctx = layout.context_len
    if previous_context is None:
        previous_context = np.full((b, ctx), eos_token_id, np.int64)
    else:
        previous_context = np.asarray(previous_context, dtype=np.int64)
    token_history = np.concatenate([previous_context, ids], axis=-1)
    shifted = [shift_right_ignore_eos(token_history, sh, eos_token_id)
               for sh in range(layout.ngram_size)]
    blocks = []
    for ngram in range(2, layout.ngram_size + 1):
        lo = (ngram - 2) * layout.heads_per_ngram
        hi = lo + layout.heads_per_ngram
        mixed = shifted[0] * layout.layer_multipliers_int[0]
        for position in range(1, ngram):
            mixed = np.bitwise_xor(
                mixed, shifted[position] * layout.layer_multipliers_int[position])
        head_sizes = layout.head_vocab_sizes[lo:hi]      # (heads,)
        head_offsets = layout.head_offsets[lo:hi]        # (heads,)
        h = np.remainder(mixed[..., None], head_sizes[None, None, :])
        blocks.append(h + head_offsets[None, None, :])
    cat = np.concatenate(blocks, axis=-1)                # (B, ctx+S, heads)
    return cat[:, -s:].astype(np.int64)


def ngram_embeddings(input_ids: np.ndarray, layout: NGramLayout,
                     table: np.ndarray, previous_context: np.ndarray | None = None,
                     eos_token_id: int = 151643) -> np.ndarray:
    """Per-token concatenated n-gram embeddings (B, S, ngram_heads*d).
    `table` (padded_vocab, d); upstream returns the flattened(-2) embedding
    lookup of ngram_ids (:1112-1114)."""
    idx = ngram_ids(input_ids, layout, previous_context, eos_token_id)
    flat = table[idx]                                   # (B, S, heads, d)
    return flat.reshape(flat.shape[0], flat.shape[1], -1)


def ple_feature(hidden_states: np.ndarray, embeddings: np.ndarray,
                w_key: np.ndarray, w_value: np.ndarray,
                gamma_conv: np.ndarray, norm_key_w: np.ndarray,
                norm_query_w: np.ndarray, norm_conv_w: np.ndarray,
                hc: int, hs: int, eps: float = 1e-6, dilation: int = 3,
                conv_state: np.ndarray | None = None,
                ) -> tuple[np.ndarray, np.ndarray]:
    """The PLE feature cell, upstream :1176-1189.

    hidden_states (B, S, hc*hs) -- the hyper stream; embeddings (B, S, pd) --
    the ngram embedding concat. Returns (output (B, S, hc*hs), new conv state
    (channels, state_len)) -- exactly the `gated_value + short_conv(gated)`
    additive feature added to the hyper stream by DecoderLayer.forward:1218.

    w_key (hc*hs, pd), w_value (hs, pd), conv kernel gamma (C, K) [or
    (C,1,K)] with groupwise RMSNorm from `norm_*_w` (group_size=hs).
    `dilation` = ngram_size (upstream :1134), kernel K = gamma columns.
    """
    B, S, _ = hidden_states.shape
    key_normed = _group_rmsnorm(np.asarray(embeddings) @ w_key.T,
                                norm_key_w, hs, eps)
    key_normed = key_normed.reshape(B, S, hc, hs)
    value = np.asarray(embeddings) @ w_value.T                  # (B,S,hs)
    query_normed = _group_rmsnorm(np.asarray(hidden_states), norm_query_w,
                                  hs, eps)
    query_normed = query_normed.reshape(B, S, hc, hs)
    gate = (key_normed * query_normed).sum(axis=-1, keepdims=True) / math.sqrt(hs)
    gate = np.abs(gate).clip(min=1e-6)**0.5 * np.sign(gate)
    gated_value = scipy_sigmoid(gate) * value[..., None, :]      # (B,S,hc,hs)
    flat = gated_value.reshape(B, S, hc * hs)
    flat_normed = _group_rmsnorm(flat, norm_conv_w, hs, eps)
    out_conv, state = _short_conv(flat_normed, gamma_conv, dilation,
                                  conv_state=conv_state)
    return flat + out_conv, state


def _group_rmsnorm(x: np.ndarray, weight: np.ndarray, group: int,
                   eps: float) -> np.ndarray:
    """Qwen4ExpTextRMSNorm(dim, group_size=group), forward = norm * (1+w).

    `x` (..., dim) with dim % group == 0; RMS is computed per group of
    `group` features (hc groups), the (dim,) weight is applied ELEMENTWISE
    via `(1 + weight)` -- matches modeling_qwen4_exp.py RMSNorm.forward.
    """
    x = np.asarray(x)
    xr = x.reshape(*x.shape[:-1], -1, group)
    rms = np.sqrt((xr.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)
    inv = xr * np.float32(1.0 / rms)
    inv = inv.reshape(*x.shape[:-1], -1)
    return inv * (1.0 + np.asarray(weight, dtype=inv.dtype))


def _short_conv(x: np.ndarray, gamma: np.ndarray, dilation: int,
                conv_state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dilated depthwise short conv, upstream :1150-1167. x (B, S, C).
    `gamma` (C, K) per-channel taps at offsets {0, d, .., (K-1)d}.
    Returns ((B,S,C) silu-conv output, updated conv state (C, state_len))
    where state_len = (K-1)*dilation.
    """
    gamma = np.asarray(gamma)
    if gamma.ndim == 3:
        gamma = gamma[:, 0, :]                       # (C, 1, K) -> (C, K)
    K = gamma.shape[1]
    C, S = x.shape[2], x.shape[1]
    state_len = (K - 1) * dilation
    if conv_state is None:
        history = np.zeros((C, state_len), x.dtype)
    else:
        history = np.asarray(conv_state, dtype=x.dtype)
        assert history.shape == (C, state_len), history.shape
    xt = x.transpose(0, 2, 1)                                  # (B,C,S)
    # causal dilated conv over [prior state + current chunk]; the prior
    # state (carried history) is the PREFIX, not zero padding
    xp = np.concatenate(
        [np.broadcast_to(history[None], (x.shape[0], C, state_len)), xt],
        axis=-1)
    xp = xp[..., -(state_len + S):]
    out = np.zeros((x.shape[0], C, S), x.dtype)
    for k in range(K):
        out += gamma[:, k][None, :, None] * xp[..., k * dilation: k * dilation + S]
    out = out * scipy_sigmoid(out)                            # silu
    new_state = _sliding_state(history, xt[0], state_len)
    return out.transpose(0, 2, 1), new_state


def _sliding_state(old: np.ndarray, new: np.ndarray, state_len: int
                   ) -> np.ndarray:
    """Right-aligned trailing window: concat(old, new), keep last state_len
    columns (C, state_len)."""
    joined = np.concatenate([old, new], axis=-1)
    return joined[:, -state_len:]


def scipy_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
