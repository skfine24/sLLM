"""Standard dense-transformer reference (Llama/Qwen2 family: GQA attention,
full-dim RoPE, plain RMSNorm, SwiGLU MLP, optional tied embeddings).

Used to serve generic dense checkpoints (e.g. Qwen2.5-Coder-0.5B) through the
same engine. Weight naming follows the `Qwen2ForCausalLM` layout with a
configurable prefix:
    {prefix}.embed_tokens.weight
    {prefix}.layers.{i}.input_layernorm.weight
    {prefix}.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    {prefix}.layers.{i}.mlp.{gate,up,down}_proj.weight
    {prefix}.layers.{i}.post_attention_layernorm.weight
    {prefix}.norm.weight
    lm_head.weight            (or tied to embed when tie_word_embeddings)

The attention is GQA without the qwen3_5 output gate / per-head q/k norms.
"""

from __future__ import annotations

import numpy as np

from . import qwen3_5 as qq


def rms_norm_plain(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Llama/Qwen2 RMSNorm: out = x * rsqrt(var+eps) * weight (no (1+w))."""
    xf = x.astype(np.float32)
    var = (xf * xf).mean(-1, keepdims=True)
    out = xf * (var + np.float32(eps)) ** -0.5
    out = out * weight.astype(np.float32)
    return out.astype(x.dtype)


def standard_attention_forward(
    hidden: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Llama/Qwen2-style GQA attention (no output gate, full-dim RoPE).

    Optional q/k/v biases (Qwen2 checkpoints ship them for the QKV
    projections); all-default-None keeps the Llama (bias-free) behaviour.
    """
    b, s, _ = hidden.shape
    q = qq.linear(hidden, w_q, q_bias).reshape(b, s, num_heads, head_dim)
    k = qq.linear(hidden, w_k, k_bias).reshape(b, s, kv_heads, head_dim)
    v = qq.linear(hidden, w_v, v_bias).reshape(b, s, kv_heads, head_dim)

    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    q, k = qq.apply_rotary_pos_emb(q, k, cos, sin)

    n_rep = num_heads // kv_heads
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=1)
        v = np.repeat(v, n_rep, axis=1)

    attn = qq.eager_attention(q, k, v, scale=head_dim ** -0.5, causal=True)
    attn = np.transpose(attn, (0, 2, 1, 3)).reshape(b, s, num_heads * head_dim)
    return qq.linear(attn, w_o).astype(hidden.dtype)


def build_cos_sin(theta: float, positions: np.ndarray, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return qq.compute_cos_sin(theta, positions, head_dim)


def standard_model_forward(
    ids: np.ndarray,
    weights: dict,
    recipe,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Full dense-transformer forward. ids: (B,S). Returns logits (B,S,V)."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    b, s = ids.shape
    prefix = recipe.text_prefix
    fa = recipe.full_attention
    head_dim = fa.effective_head_dim(recipe.hidden_size)
    eps = recipe.rms_norm_eps

    embed_w = weights[f"{prefix}.embed_tokens.weight"]
    hidden = embed_w[ids].astype(np.float32)

    if positions is None:
        positions = np.arange(s, dtype=np.int64)[None, :]
    cos, sin = build_cos_sin(fa.rope.theta, positions, recipe.rotary_dim())

    for i in range(recipe.num_layers):
        p = f"{prefix}.layers.{i}"
        residual = hidden
        h = rms_norm_plain(hidden, weights[f"{p}.input_layernorm.weight"], eps=eps)
        h = standard_attention_forward(
            h, cos=cos, sin=sin, **{
                "w_q": weights[f"{p}.self_attn.q_proj.weight"],
                "w_k": weights[f"{p}.self_attn.k_proj.weight"],
                "w_v": weights[f"{p}.self_attn.v_proj.weight"],
                "w_o": weights[f"{p}.self_attn.o_proj.weight"],
                "num_heads": fa.num_heads, "kv_heads": fa.num_kv_heads,
                "head_dim": head_dim,
                "q_bias": weights.get(f"{p}.self_attn.q_proj.bias"),
                "k_bias": weights.get(f"{p}.self_attn.k_proj.bias"),
                "v_bias": weights.get(f"{p}.self_attn.v_proj.bias"),
            },
        )
        hidden = residual + h

        residual = hidden
        h = rms_norm_plain(hidden, weights[f"{p}.post_attention_layernorm.weight"], eps=eps)
        h = qq.mlp_forward(
            h,
            w_gate=weights[f"{p}.mlp.gate_proj.weight"],
            w_up=weights[f"{p}.mlp.up_proj.weight"],
            w_down=weights[f"{p}.mlp.down_proj.weight"],
        )
        hidden = residual + h

    hidden = rms_norm_plain(hidden, weights[f"{prefix}.norm.weight"], eps=eps)
    if recipe.tie_word_embeddings:
        return hidden @ embed_w.T
    return hidden @ weights["lm_head.weight"].T
