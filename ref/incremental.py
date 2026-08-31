"""Incremental (KV-cached) decode for the reference engine.

Replaces the recompute-every-step loop inside `serving.executor` with two
phases that keep the loop-carried "memory" (attention KV cache + recurrent
state + causal-conv window) continuous across decode steps:

  * `prefill(ids, weights, recipe)`: run the EXACT existing full forward once
    (same functions, same op order -> same output) while capturing per-layer
    attention K/V, linear recurrent state, and the causal-conv1d window.
    Returns `(cache, logits)` where `logits` is the full (1, S, V) tensor.
  * `decode_step(cache, weights, recipe, last_id)`: embed one new token and
    produce its logits using only the cached state. Full-attention layers do a
    last-row attention over cached K/V (O(context) instead of O(context^2));
    linear (GatedDeltaNet) layers run the existing single-step recurrent path
    (O(1)). Returns (V,).

Parity guarantees
-----------------
* "standard_gqa" (Llama/Qwen2 family): the decode path is bit-identical to the
  last token of the full forward (same fp32 operations; the causal mask never
  touches the last row, and cached K/V are exactly the post-norm/post-rotary
  tensors the eager oracle used).
* "paged_flash" (qwen3_5 hybrid): full-attention layers are bit-identical;
  GatedDeltaNet layers use the recurrent single-step path, which matches the
  chunked prefill path to ~1e-8 (the same parity already validated in
  `test_ref_qwen3_5`).

The full-recompute functions are untouched and remain the oracle (used by the
fallback path in `serving.executor` and by parity tests).
"""

from __future__ import annotations

import numpy as np

from . import qwen3_5 as qq
from . import standard as st
from .pipeline import (
    _TEXT_PREFIX,
    _full_attn_weights,
    _linear_attn_weights,
    _mlp_weights,
    build_cos_sin_for_positions,
)

SUPPORTED_KERNELS = ("standard_gqa", "paged_flash")


class IncrementalCache:
    """Opaque per-sequence runtime memory captured by `prefill` / updated by
    `decode_step`. Not thread-safe; one cache per sequence."""

    def __init__(self, kernel: str, n_ctx: int, eps: float, num_heads: int,
                 kv_heads: int, head_dim: int, theta: float,
                 rotary_dim: int | None = None):
        self.kernel = kernel
        self.n_ctx = n_ctx
        self.eps = eps
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.theta = theta
        self.rot = int(rotary_dim) if rotary_dim else head_dim  # RoPE width
        self.k: dict[int, np.ndarray] = {}      # full-attn layer -> (kv_heads, S, head_dim)
        self.v: dict[int, np.ndarray] = {}      # full-attn layer -> (kv_heads, S, head_dim)
        self.state: dict[int, np.ndarray] = {}  # linear layer -> (1, v_heads, k_dim, v_dim)
        self.conv_win: dict[int, np.ndarray] = {}  # linear layer -> (1, C, K-1) raw mixed


def _pos_cos_sin(theta: float, rotary_dim: int, pos: int) -> tuple[np.ndarray, np.ndarray]:
    cos, sin = qq.compute_cos_sin(theta, np.array([[pos]], dtype=np.int64), rotary_dim)
    return cos[0, 0], sin[0, 0]


def _last_row_attention(q3: np.ndarray, kt: np.ndarray, vt: np.ndarray,
                        scale: float) -> np.ndarray:
    """Exact last row of `qwen3_5.eager_attention`.

    q3: (num_heads, 1, head_dim) rotated query; kt/vt: (num_heads, S, head_dim)
    already replicated. Returns (num_heads, 1, head_dim) output, matching the
    last slice of eager_attention element-for-element.
    """
    scores = np.matmul(q3, np.swapaxes(kt, -1, -2)).astype(np.float32) * np.float32(scale)
    scores = scores - scores.max(-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(-1, keepdims=True)
    return np.matmul(probs, vt)


def _replicate(kt: np.ndarray, vt: np.ndarray, n_rep: int):
    if n_rep > 1:
        kt = np.repeat(kt, n_rep, axis=0)
        vt = np.repeat(vt, n_rep, axis=0)
    return kt, vt


# ---------------------------------------------------------------------------
# standard (Llama/Qwen2-family) path
# ---------------------------------------------------------------------------

def _prefill_standard(ids, weights, recipe):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    b, s = ids.shape
    prefix = recipe.text_prefix
    fa = recipe.full_attention
    nh, kvh = fa.num_heads, fa.num_kv_heads
    hd = fa.effective_head_dim(recipe.hidden_size)
    eps = recipe.rms_norm_eps
    theta = fa.rope.theta
    rot = recipe.rotary_dim()  # RoPE width (== head_dim unless partial rotary)

    embed_w = weights[f"{prefix}.embed_tokens.weight"]
    hidden = embed_w[ids].astype(np.float32)
    cos, sin = st.build_cos_sin(theta, np.arange(s, dtype=np.int64)[None, :], rot)

    cache = IncrementalCache("standard_gqa", n_ctx=s, eps=eps, num_heads=nh,
                             kv_heads=kvh, head_dim=hd, theta=theta,
                             rotary_dim=rot)
    for i in range(recipe.num_layers):
        p = f"{prefix}.layers.{i}"
        residual = hidden
        h_in = st.rms_norm_plain(hidden, weights[f"{p}.input_layernorm.weight"], eps=eps)
        wq = weights[f"{p}.self_attn.q_proj.weight"]
        wk = weights[f"{p}.self_attn.k_proj.weight"]
        wv = weights[f"{p}.self_attn.v_proj.weight"]
        wo = weights[f"{p}.self_attn.o_proj.weight"]
        qb = weights.get(f"{p}.self_attn.q_proj.bias")
        kb = weights.get(f"{p}.self_attn.k_proj.bias")
        vb = weights.get(f"{p}.self_attn.v_proj.bias")
        h = st.standard_attention_forward(
            h_in, wq, wk, wv, wo, cos, sin, nh, kvh, hd,
            q_bias=qb, k_bias=kb, v_bias=vb)
        kt = np.transpose(qq.linear(h_in, wk, kb).reshape(b, s, kvh, hd), (0, 2, 1, 3))
        vt = np.transpose(qq.linear(h_in, wv, vb).reshape(b, s, kvh, hd), (0, 2, 1, 3))
        _, kt = qq.apply_rotary_pos_emb(kt, kt, cos, sin)
        cache.k[i] = kt[0]
        cache.v[i] = vt[0]
        hidden = residual + h

        residual = hidden
        h = st.rms_norm_plain(hidden, weights[f"{p}.post_attention_layernorm.weight"], eps=eps)
        h = qq.mlp_forward(
            h,
            w_gate=weights[f"{p}.mlp.gate_proj.weight"],
            w_up=weights[f"{p}.mlp.up_proj.weight"],
            w_down=weights[f"{p}.mlp.down_proj.weight"],
        )
        hidden = residual + h

    hidden = st.rms_norm_plain(hidden, weights[f"{prefix}.norm.weight"], eps=eps)
    if recipe.tie_word_embeddings:
        logits = hidden @ embed_w.T
    else:
        logits = hidden @ weights["lm_head.weight"].T
    return cache, logits


def _decode_standard(cache: IncrementalCache, weights, recipe, last_id: int) -> np.ndarray:
    prefix = recipe.text_prefix
    nh, kvh, hd = cache.num_heads, cache.kv_heads, cache.head_dim
    eps, theta = cache.eps, cache.theta
    n_rep = nh // kvh

    embed_w = weights[f"{prefix}.embed_tokens.weight"]
    hidden = embed_w[[last_id]].astype(np.float32)
    pos = cache.n_ctx
    cache.n_ctx += 1
    cos, sin = _pos_cos_sin(theta, cache.rot, pos)

    for i in range(recipe.num_layers):
        p = f"{prefix}.layers.{i}"
        residual = hidden
        h_in = st.rms_norm_plain(hidden, weights[f"{p}.input_layernorm.weight"], eps=eps)
        wq = weights[f"{p}.self_attn.q_proj.weight"]
        wk = weights[f"{p}.self_attn.k_proj.weight"]
        wv = weights[f"{p}.self_attn.v_proj.weight"]
        wo = weights[f"{p}.self_attn.o_proj.weight"]
        qb = weights.get(f"{p}.self_attn.q_proj.bias")
        kb = weights.get(f"{p}.self_attn.k_proj.bias")
        vb = weights.get(f"{p}.self_attn.v_proj.bias")

        q = qq.linear(h_in, wq, qb).reshape(1, nh, 1, hd)
        k = qq.linear(h_in, wk, kb).reshape(1, kvh, 1, hd)
        v = qq.linear(h_in, wv, vb).reshape(1, kvh, 1, hd)
        q, k = qq.apply_rotary_pos_emb(q, k, cos[None, None], sin[None, None])
        cache.k[i] = np.concatenate([cache.k[i], k[0]], axis=1)
        cache.v[i] = np.concatenate([cache.v[i], v[0]], axis=1)
        kt, vt = _replicate(cache.k[i], cache.v[i], n_rep)
        attn = _last_row_attention(q[0], kt, vt, hd ** -0.5)
        attn = np.transpose(attn, (1, 0, 2)).reshape(1, nh * hd)
        hidden = residual + qq.linear(attn, wo).astype(hidden.dtype)

        residual = hidden
        h = st.rms_norm_plain(hidden, weights[f"{p}.post_attention_layernorm.weight"], eps=eps)
        h = qq.mlp_forward(
            h,
            w_gate=weights[f"{p}.mlp.gate_proj.weight"],
            w_up=weights[f"{p}.mlp.up_proj.weight"],
            w_down=weights[f"{p}.mlp.down_proj.weight"],
        )
        hidden = residual + h

    hidden = st.rms_norm_plain(hidden, weights[f"{prefix}.norm.weight"], eps=eps)
    if recipe.tie_word_embeddings:
        return (hidden @ embed_w.T)[0]
    return (hidden @ weights["lm_head.weight"].T)[0]


# ---------------------------------------------------------------------------
# qwen3_5 hybrid path (GatedDeltaNet linear + gated full attention)
# ---------------------------------------------------------------------------

def _prefill_hybrid(ids, weights, recipe):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    b, s = ids.shape
    fa = recipe.full_attention
    eps = recipe.rms_norm_eps

    embed_w = weights[f"{_TEXT_PREFIX}.embed_tokens.weight"]
    hidden = embed_w[ids].astype(np.float32)
    cos, sin = build_cos_sin_for_positions(recipe, s)

    cache = IncrementalCache(
        "paged_flash", n_ctx=s, eps=eps,
        num_heads=fa.num_heads, kv_heads=fa.num_kv_heads,
        head_dim=fa.head_dim, theta=fa.rope.theta,
    )
    for i, bt in enumerate(recipe.layer_types):
        in_norm_w = weights[f"{_TEXT_PREFIX}.layers.{i}.input_layernorm.weight"]
        post_norm_w = weights[f"{_TEXT_PREFIX}.layers.{i}.post_attention_layernorm.weight"]
        residual = hidden
        h_in = qq.rms_norm(hidden, in_norm_w, eps=eps)
        if bt == "linear_attention":
            la = recipe.linear_attention
            fw = _linear_attn_weights(weights, i)
            out, state = qq.gated_delta_net_forward(
                h_in, **fw,
                num_k_heads=la.num_key_heads, head_k_dim=la.key_head_dim,
                num_v_heads=la.num_value_heads, head_v_dim=la.value_head_dim,
                conv_kernel_size=la.conv_kernel_size,
                use_qk_l2norm_in_kernel=la.qk_l2norm,
                chunked=True, chunk_size=64, rms_eps=eps,
            )
            cache.state[i] = state
            mixed = np.transpose(qq.linear(h_in, fw["w_in_qkv"]), (0, 2, 1))
            kc = la.conv_kernel_size
            cache.conv_win[i] = mixed[:, :, -(kc - 1):] if kc > 1 else np.zeros((1, 0, 1))
            h = out
        else:
            fw = _full_attn_weights(weights, i)
            h = qq.full_attention_forward(
                h_in, **fw, cos=cos, sin=sin,
                num_heads=fa.num_heads, kv_heads=fa.num_kv_heads,
                head_dim=fa.head_dim, rms_eps=eps,
            )
            kt = np.transpose(qq.rms_norm(
                qq.linear(h_in, fw["w_k"]).reshape(b, s, fa.num_kv_heads, fa.head_dim),
                fw["k_norm_w"], eps=eps), (0, 2, 1, 3))
            vt = np.transpose(qq.linear(h_in, fw["w_v"]).reshape(b, s, fa.num_kv_heads, fa.head_dim),
                              (0, 2, 1, 3))
            _, kt = qq.apply_rotary_pos_emb(kt, kt, cos, sin)
            cache.k[i] = kt[0]
            cache.v[i] = vt[0]
        hidden = residual + h

        residual = hidden
        h = qq.rms_norm(hidden, post_norm_w, eps=eps)
        h = qq.mlp_forward(h, **_mlp_weights(weights, i))
        hidden = residual + h

    hidden = qq.rms_norm(hidden, weights[f"{_TEXT_PREFIX}.norm.weight"], eps=eps)
    logits = hidden @ weights["lm_head.weight"].T
    return cache, logits


def _decode_hybrid(cache: IncrementalCache, weights, recipe, last_id: int) -> np.ndarray:
    eps, theta = cache.eps, cache.theta
    fa = recipe.full_attention
    nh = fa.num_heads
    hd = fa.head_dim

    embed_w = weights[f"{_TEXT_PREFIX}.embed_tokens.weight"]
    hidden = embed_w[[last_id]].astype(np.float32)
    pos = cache.n_ctx
    cache.n_ctx += 1
    rot = recipe.rotary_dim()
    cos, sin = _pos_cos_sin(theta, rot, pos)

    for i, bt in enumerate(recipe.layer_types):
        in_norm_w = weights[f"{_TEXT_PREFIX}.layers.{i}.input_layernorm.weight"]
        post_norm_w = weights[f"{_TEXT_PREFIX}.layers.{i}.post_attention_layernorm.weight"]
        residual = hidden
        h_in = qq.rms_norm(hidden, in_norm_w, eps=eps)
        if bt == "linear_attention":
            la = recipe.linear_attention
            fw = _linear_attn_weights(weights, i)
            key_dim = la.num_key_heads * la.key_head_dim
            value_dim = la.num_value_heads * la.value_head_dim
            kc = la.conv_kernel_size

            mixed = np.transpose(qq.linear(h_in, fw["w_in_qkv"]), (1, 0))
            win = np.concatenate([cache.conv_win[i], mixed[None]], axis=-1)
            conv = qq.causal_conv1d_depthwise(win[:, :, -kc:], fw["w_conv"], activation="silu")
            mixed_conv = np.transpose(conv[:, :, -1:], (0, 2, 1))
            cache.conv_win[i] = win[:, :, -(kc - 1):]

            query, key, value = qq._split(mixed_conv, [key_dim, key_dim, value_dim])
            query = query.reshape(1, 1, -1, la.key_head_dim)
            key = key.reshape(1, 1, -1, la.key_head_dim)
            value = value.reshape(1, 1, la.num_value_heads, la.value_head_dim)
            ratio = la.num_value_heads // la.num_key_heads
            if ratio > 1:
                query = np.repeat(query, ratio, axis=2)
                key = np.repeat(key, ratio, axis=2)

            z = qq.linear(h_in, fw["w_z"]).reshape(1, 1, la.num_value_heads, la.value_head_dim)
            bvec = qq.linear(h_in, fw["w_b"])
            avec = qq.linear(h_in, fw["w_a"])
            beta = qq.sigmoid(bvec)[None]  # (1, 1, Vh)
            g = (-np.exp(fw["a_log"]).astype(np.float32)
                 * qq.softplus(avec.astype(np.float32) + fw["dt_bias"]))[None]

            core, state = qq.gated_delta_rule_recurrent(
                query, key, value, g=g, beta=beta,
                initial_state=cache.state[i], output_final_state=True,
                use_qk_l2norm_in_kernel=la.qk_l2norm,
            )
            cache.state[i] = state
            core = qq.rms_norm_gated(core, fw["norm_w"], z, eps=eps)
            core = core.reshape(1, 1, -1)
            h = qq.linear(core, fw["w_out"]).astype(hidden.dtype)[0]
        else:
            fw = _full_attn_weights(weights, i)
            kvh = fa.num_kv_heads
            n_rep = nh // kvh
            qg = qq.linear(h_in, fw["w_q"]).reshape(1, nh, hd * 2)
            q2, gate = qq._split(qg, [hd, hd])
            q2 = qq.rms_norm(q2, fw["q_norm_w"], eps=eps).reshape(1, nh, 1, hd)
            k = qq.rms_norm(qq.linear(h_in, fw["w_k"]).reshape(1, kvh, 1, hd), fw["k_norm_w"], eps=eps)
            v = qq.linear(h_in, fw["w_v"]).reshape(1, kvh, 1, hd)
            q2, k = qq.apply_rotary_pos_emb(q2, k, cos[None, None], sin[None, None])
            cache.k[i] = np.concatenate([cache.k[i], k[0]], axis=1)
            cache.v[i] = np.concatenate([cache.v[i], v[0]], axis=1)
            kt, vt = _replicate(cache.k[i], cache.v[i], n_rep)
            attn = _last_row_attention(q2[0], kt, vt, hd ** -0.5)
            attn = np.transpose(attn, (1, 0, 2)).reshape(1, nh * hd)
            attn = attn * qq.sigmoid(gate.reshape(1, -1))
            h = qq.linear(attn, fw["w_o"]).astype(hidden.dtype)
        hidden = residual + h

        residual = hidden
        h = qq.rms_norm(hidden, post_norm_w, eps=eps)
        h = qq.mlp_forward(h, **_mlp_weights(weights, i))
        hidden = residual + h

    hidden = qq.rms_norm(hidden, weights[f"{_TEXT_PREFIX}.norm.weight"], eps=eps)
    return (hidden @ weights["lm_head.weight"].T)[0]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def prefill(ids, weights, recipe):
    """Run a full forward once and return (cache, logits)."""
    if recipe.full_attention.kernel == "standard_gqa":
        return _prefill_standard(ids, weights, recipe)
    return _prefill_hybrid(ids, weights, recipe)


def decode_step(cache: IncrementalCache, weights, recipe, last_id: int) -> np.ndarray:
    """Logits (V,) for the token following `last_id` using only cached state."""
    if cache.kernel == "standard_gqa":
        return _decode_standard(cache, weights, recipe, last_id)
    return _decode_hybrid(cache, weights, recipe, last_id)

