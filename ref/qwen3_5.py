"""Qwen3_5 reference math (numpy), ported from transformers
`modeling_qwen3_5.py` (5.8.0.dev0).

Purpose: numeric-parity source of truth for the engine kernels. Pure numpy,
torch-free, so it runs on the dev machine and inside the cluster container.

Porting notes (keep in sync with upstream semantics):
- Decoder norms and self-attn q/k norms use `out = norm(x) * (1.0 + weight)`
  (weight initialized to zeros). The GatedDeltaNet internal norm is a *gated*
  RMSNorm: `out = weight * norm(x) * silu(gate)` (weight initialized to ones).
- Linear attention is a GatedDeltaNet = gated delta rule (FLA-style), NOT a
  Mamba selective scan. `beta = sigmoid(in_proj_b(x))` and
  `g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)` (decay in log space).
- Full attention projects Q plus a gate from one projection
  (`q_proj` out = heads*head_dim*2), normalizes q/k per head, applies partial
  rotary (mrope), and multiplies the attention output by `sigmoid(gate)`.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Qwen3_5RMSNorm: norm in fp32, out = norm(x) * (1 + weight)."""
    xf = x.astype(np.float32)
    var = (xf * xf).mean(-1, keepdims=True)
    out = xf * (var + np.float32(eps)) ** -0.5
    out = out * (1.0 + weight.astype(np.float32))
    return out.astype(x.dtype)


def rms_norm_gated(x: np.ndarray, weight: np.ndarray, gate: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Qwen3_5RMSNormGated: `weight * rmsnorm(x) * silu(gate)`."""
    xf = x.astype(np.float32)
    var = (xf * xf).mean(-1, keepdims=True)
    out = xf * (var + np.float32(eps)) ** -0.5
    out = out * weight.astype(np.float32)
    out = out * silu(gate.astype(np.float32))
    return out.astype(x.dtype)


def linear(x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """x: (..., in), w: (out, in) -> (..., out)."""
    out = x @ w.T
    if bias is not None:
        out = out + bias
    return out


def causal_conv1d_depthwise(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    activation: str | None = None,
) -> np.ndarray:
    """Depthwise causal conv1d on (B, C, S), weight (C, 1, K), padding K-1.

    Mirrors `causal_conv1d_fn` (F.conv1d padding=K-1, then [: , :, :S]).
    Assumes the weight has already been expanded for the group layout.
    """
    b, c, s = x.shape
    k = weight.shape[-1]
    out = np.zeros((b, c, s), dtype=np.float64)
    for t in range(s):
        acc = np.zeros((b, c), dtype=np.float64)
        for j in range(k):
            src = t - (k - 1) + j
            if src >= 0:
                acc += weight[:, 0, j] * x[:, :, src]
        out[:, :, t] = acc
    if bias is not None:
        out = out + bias[None, :, None]
    if activation == "silu":
        out = silu(out)
    return out.astype(x.dtype)


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    inv = ((x * x).sum(axis=axis, keepdims=True) + np.float32(eps)) ** -0.5
    return x * inv


def _split(x: np.ndarray, sizes) -> list[np.ndarray]:
    parts = []
    start = 0
    for sz in sizes:
        parts.append(x[..., start:start + sz])
        start += sz
    return parts


# --------------------------------------------------------------------------
# gated delta rule (linear attention core)
# --------------------------------------------------------------------------

def _solve_unit_lower(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch solve L x = b for unit-lower-triangular L (..., n, n), b (...,n,m).

    Forward substitution over the last row index (unit diagonal).
    """
    x = b.astype(np.float64).copy()
    for i in range(1, L.shape[-2]):
        sub = L[..., i, :i]  # (..., i)
        x[..., i, :] = x[..., i, :] - np.einsum("...i,...im->...m", sub, x[..., :i, :])
    return x.astype(b.dtype)


def gated_delta_rule_recurrent(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Port of `torch_recurrent_gated_delta_rule` (single-step decode path).

    Shapes in: query/key (B,S,K_heads,K_dim), value (B,S,V_heads,V_dim),
    g/beta (B,S,V_heads). Out: (B,S,V_heads,V_dim); final state
    (B, V_heads, K_dim, V_dim) or None.
    """
    b, s, _, kd = key.shape
    vh, vd = value.shape[-2], value.shape[-1]

    q = np.transpose(query, (0, 2, 1, 3)).astype(np.float32)
    k = np.transpose(key, (0, 2, 1, 3)).astype(np.float32)
    v = np.transpose(value, (0, 2, 1, 3)).astype(np.float32)
    bt = np.transpose(beta, (0, 2, 1)).astype(np.float32)
    dec = np.transpose(g, (0, 2, 1)).astype(np.float32)

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)
    q = q / np.float32(np.sqrt(kd))

    if initial_state is None:
        state = np.zeros((b, vh, kd, vd), dtype=np.float32)
    else:
        state = initial_state.astype(np.float32)

    out = np.zeros((b, vh, s, vd), dtype=np.float32)
    for i in range(s):
        decay_t = np.exp(dec[:, :, i])[..., None, None]
        state = state * decay_t
        beta_t = bt[:, :, i][..., None]  # (B,Vh,1)
        kv_mem = (state * k[:, :, i, :, None]).sum(-2)
        delta = (v[:, :, i] - kv_mem) * beta_t
        state = state + k[:, :, i, :, None] * delta[:, :, None, :]
        out[:, :, i] = (state * q[:, :, i, :, None]).sum(-2)

    final = state if output_final_state else None
    out = np.transpose(out, (0, 2, 1, 3))
    return out.astype(query.dtype), (None if final is None else final.astype(np.float32))


def gated_delta_rule_chunked(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    chunk_size: int = 64,
    initial_state: np.ndarray | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Port of `torch_chunk_gated_delta_rule` (prefill path)."""
    b, s, _, kd = key.shape
    vh, vd = value.shape[-2], value.shape[-1]

    q = np.transpose(query, (0, 2, 1, 3)).astype(np.float32)
    k = np.transpose(key, (0, 2, 1, 3)).astype(np.float32)
    v = np.transpose(value, (0, 2, 1, 3)).astype(np.float32)
    bt = np.transpose(beta, (0, 2, 1)).astype(np.float32)
    dec = np.transpose(g, (0, 2, 1)).astype(np.float32)

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)
    q = q * np.float32(kd ** -0.5)

    pad = (chunk_size - s % chunk_size) % chunk_size
    q = np.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
    k = np.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
    v = np.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
    bt = np.pad(bt, ((0, 0), (0, 0), (0, pad)))
    dec = np.pad(dec, ((0, 0), (0, 0), (0, pad)))

    total = s + pad
    nc = total // chunk_size

    v_beta = v * bt[..., None]
    k_beta = k * bt[..., None]

    q = q.reshape(b, vh, -1, chunk_size, kd)
    k = k.reshape(b, vh, -1, chunk_size, kd)
    k_beta = k_beta.reshape(b, vh, -1, chunk_size, kd)
    v_beta = v_beta.reshape(b, vh, -1, chunk_size, vd)
    dec = dec.reshape(b, vh, -1, chunk_size)

    upper = np.triu(np.ones((chunk_size, chunk_size), dtype=np.float32), 1).astype(bool)

    cum_decay = dec.cumsum(axis=-1)

    pairwise = cum_decay[..., None] - cum_decay[..., None, :]
    pairwise = np.where(upper, np.float32("-inf"), pairwise)
    pairwise = np.exp(pairwise)

    ut_system = np.matmul(k_beta, np.swapaxes(k, -1, -2)) * pairwise
    intra = np.matmul(q, np.swapaxes(k, -1, -2)) * pairwise
    decayed_k_beta = k_beta * np.exp(cum_decay)[..., None]

    new_values = _solve_unit_lower(ut_system, v_beta)
    k_cumdecay = _solve_unit_lower(ut_system, decayed_k_beta)

    if initial_state is None:
        state = np.zeros((b, vh, kd, vd), dtype=np.float32)
    else:
        state = initial_state.astype(np.float32)

    q = q * np.exp(cum_decay)[..., None]
    k = k * np.exp(cum_decay[..., -1:] - cum_decay)[..., None]
    chunk_decay = np.exp(cum_decay[..., -1])[..., None, None]

    core = np.zeros_like(new_values)
    for c in range(nc):
        v_new = new_values[:, :, c] - np.matmul(k_cumdecay[:, :, c], state)
        inter = np.matmul(q[:, :, c], state)
        core[:, :, c] = inter + np.matmul(intra[:, :, c], v_new)
        state = state * chunk_decay[:, :, c] + np.matmul(
            np.swapaxes(k[:, :, c], -1, -2), v_new
        )

    out = core.reshape(b, vh, total, vd)[:, :, :s, :]
    final = state if output_final_state else None
    out = np.transpose(out, (0, 2, 1, 3))
    return out.astype(query.dtype), (None if final is None else final.astype(np.float32))


def gated_delta_net_forward(
    hidden: np.ndarray,
    w_in_qkv: np.ndarray,
    w_conv: np.ndarray,
    w_z: np.ndarray,
    w_b: np.ndarray,
    w_a: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    norm_w: np.ndarray,
    w_out: np.ndarray,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    conv_kernel_size: int,
    use_qk_l2norm_in_kernel: bool = True,
    chunked: bool = True,
    chunk_size: int = 64,
    initial_state: np.ndarray | None = None,
    rms_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Port of `Qwen3_5GatedDeltaNet.forward` (prefill/chunked path).

    hidden: (B, S, hidden). Returns (B, S, hidden) and final recurrent state
    (B, num_v_heads, head_k_dim, head_v_dim) or None.
    """
    b, s, _ = hidden.shape
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    mixed = linear(hidden, w_in_qkv)  # (B,S, key*2+value)
    mixed = np.transpose(mixed, (0, 2, 1))
    mixed = causal_conv1d_depthwise(mixed, w_conv, activation="silu")
    mixed = np.transpose(mixed, (0, 2, 1))

    query, key, value = _split(mixed, [key_dim, key_dim, value_dim])
    query = query.reshape(b, s, -1, head_k_dim)
    key = key.reshape(b, s, -1, head_k_dim)
    value = value.reshape(b, s, num_v_heads, head_v_dim)

    z = linear(hidden, w_z).reshape(b, s, -1, head_v_dim)
    bvec = linear(hidden, w_b)
    avec = linear(hidden, w_a)

    beta = sigmoid(bvec)
    g = -np.exp(a_log).astype(np.float32) * softplus(avec.astype(np.float32) + dt_bias)

    ratio = num_v_heads // num_k_heads
    if ratio > 1:
        query = np.repeat(query, ratio, axis=2)
        key = np.repeat(key, ratio, axis=2)

    if chunked:
        core, state = gated_delta_rule_chunked(
            query, key, value, g=g, beta=beta,
            chunk_size=chunk_size, initial_state=initial_state,
            output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    else:
        core, state = gated_delta_rule_recurrent(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    core = core.reshape(-1, head_v_dim)
    z = z.reshape(-1, head_v_dim)
    core = rms_norm_gated(core, norm_w, z, eps=rms_eps)
    core = core.reshape(b, s, -1)

    return linear(core, w_out).astype(hidden.dtype), state


# --------------------------------------------------------------------------
# rotary (partial / mrope) + full attention
# --------------------------------------------------------------------------

def rotate_half(x: np.ndarray) -> np.ndarray:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate((-x2, x1), axis=-1)


def compute_cos_sin(theta: float, positions: np.ndarray, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Simplified (B,S) partial rotary: inv_freq over rotary_dim/2, emb doubled.

    positions: (B, S) int64. Returns cos/sin (B, S, rotary_dim).
    """
    dim = rotary_dim
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    freqs = positions.astype(np.float64)[..., :, None] * inv_freq[None, None, :]  # (B,S, dim/2)
    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.cos(emb), np.sin(emb)


def apply_rotary_pos_emb(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    unsqueeze_dim: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply partial rotary to q/k. q/k: (B, H, S, head_dim); cos/sin (B,S,D_rot)."""
    cos = np.expand_dims(cos, unsqueeze_dim)
    sin = np.expand_dims(sin, unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return np.concatenate((q_embed, q_pass), axis=-1), np.concatenate((k_embed, k_pass), axis=-1)


def eager_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
    causal: bool = True,
) -> np.ndarray:
    """q/k/v: (B,H,S,D). Returns (B,H,S,D) with fp32 softmax."""
    b, h, s, d = q.shape
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(scale)
    if causal:
        mask = np.triu(np.ones((s, s), dtype=bool), 1)
        scores = np.where(mask, np.float32("-inf"), scores)
    scores = scores.astype(np.float32)
    scores = scores - scores.max(-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(-1, keepdims=True)
    return np.matmul(probs, v)


def full_attention_forward(
    hidden: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    q_norm_w: np.ndarray,
    k_norm_w: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    rms_eps: float = 1e-6,
) -> np.ndarray:
    """Port of `Qwen3_5Attention.forward` (Q+gate projection, q/k norm, partial
    rotary, gated output). hidden: (B,S,H). cos/sin: (B,S,rotary_dim)."""
    b, s, _ = hidden.shape

    qg = linear(hidden, w_q)  # (B,S, H*2Hd)
    qg = qg.reshape(b, s, num_heads, head_dim * 2)
    q, gate = _split(qg, [head_dim, head_dim])
    q = q.reshape(b, s, num_heads, head_dim)
    gate = gate.reshape(b, s, -1)

    q = rms_norm(q, q_norm_w, eps=rms_eps)  # per-head (weight broadcast)
    k = rms_norm(linear(hidden, w_k).reshape(b, s, kv_heads, head_dim), k_norm_w, eps=rms_eps)
    v = linear(hidden, w_v).reshape(b, s, kv_heads, head_dim)

    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    n_rep = num_heads // kv_heads
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=1)
        v = np.repeat(v, n_rep, axis=1)

    attn = eager_attention(q, k, v, scale=head_dim ** -0.5, causal=True)
    attn = np.transpose(attn, (0, 2, 1, 3)).reshape(b, s, num_heads * head_dim)
    attn = attn * sigmoid(gate)

    return linear(attn, w_o).astype(hidden.dtype)


# --------------------------------------------------------------------------
# dense MLP + decoder layer
# --------------------------------------------------------------------------

def mlp_forward(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    """Qwen3_5MLP: down(silu(gate(x)) * up(x))."""
    return linear(silu(linear(x, w_gate)) * linear(x, w_up), w_down).astype(x.dtype)


def decoder_layer_forward(
    hidden: np.ndarray,
    in_norm_w: np.ndarray,
    post_norm_w: np.ndarray,
    block_type: str,
    rms_eps: float = 1e-6,
    **block_kwargs,
) -> np.ndarray:
    """One decoder layer; dispatches linear/full attention + dense MLP + residuals.

    block_kwargs: for
      linear_attention -> gated_delta_net_forward kwargs (minus hidden)
      full_attention  -> full_attention_forward kwargs (minus hidden)
    """
    residual = hidden
    h = rms_norm(hidden, in_norm_w, eps=rms_eps)
    mlp_kw = block_kwargs.pop("mlp", {})
    if block_type == "linear_attention":
        h, _ = gated_delta_net_forward(h, **block_kwargs)
    elif block_type == "full_attention":
        h = full_attention_forward(h, **block_kwargs)
    else:  # pragma: no cover
        raise ValueError(f"unknown block_type {block_type!r}")
    h = residual + h

    residual = h
    h = rms_norm(h, post_norm_w, eps=rms_eps)
    h = mlp_forward(h, **mlp_kw)
    return residual + h
