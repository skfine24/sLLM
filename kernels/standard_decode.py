"""Standard (Llama/Qwen2-family) GPU decode driver.

Mirrors `ref/incremental._decode_standard` with the heavy ops moved to GPU:
dense GEMMs (cuBLAS `sllm_gemm`), last-row attention (`sllm_attention_decode`),
residual adds (`sllm_elwise_add`) and MLP SiLU (`sllm_silu`); norms, RoPE and
bias adds stay on the host (cheap, elementwise). Kernel absent -> callers
degrade to the numpy path.
"""

from __future__ import annotations

import numpy as np

from kernels import _sllm_cuda as ck
from ref import qwen3_5 as qq
from ref import standard as st


def _proj(h: np.ndarray, w: np.ndarray, bias=None) -> np.ndarray:
    """c = h (1,in) @ w.T (in,out) via cuBLAS -> (1, out), plus optional bias."""
    h = np.ascontiguousarray(h, dtype=np.float32)
    wt = np.ascontiguousarray(np.asarray(w, dtype=np.float32).T, dtype=np.float32)
    c = ck.gemm(h, wt)
    if bias is not None:
        c = c + np.ascontiguousarray(bias, dtype=np.float32).reshape(1, -1)
    return c


def gpu_standard_decode_step(cache, weights, recipe, last_id: int) -> np.ndarray:
    """Logits (V,) for the token after `last_id` via the GPU standard path."""
    from ref import incremental as inc  # local import to avoid a cycle

    prefix = recipe.text_prefix
    nh, kvh, hd = cache.num_heads, cache.kv_heads, cache.head_dim
    eps, theta = cache.eps, cache.theta
    n_rep = nh // kvh

    embed_w = weights[f"{prefix}.embed_tokens.weight"]
    hidden = embed_w[[last_id]].astype(np.float32)

    pos = cache.n_ctx
    cache.n_ctx += 1
    cos, sin = inc._pos_cos_sin(theta, cache.rot, pos)

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

        q = _proj(h_in, wq, qb).reshape(1, nh, 1, hd)
        k = _proj(h_in, wk, kb).reshape(1, kvh, 1, hd)
        v = _proj(h_in, wv, vb).reshape(1, kvh, 1, hd)
        q, k = qq.apply_rotary_pos_emb(q, k, cos[None, None], sin[None, None])
        cache.k[i] = np.concatenate([cache.k[i], k[0]], axis=1)
        cache.v[i] = np.concatenate([cache.v[i], v[0]], axis=1)
        kt, vt = inc._replicate(cache.k[i], cache.v[i], n_rep)
        attn = ck.attention_decode(q[0, :, 0, :], kt, vt, hd ** -0.5)
        attn = attn.reshape(1, nh * hd)
        hidden = ck.elwise_add(residual, _proj(attn, wo))

        residual = hidden
        h = st.rms_norm_plain(hidden, weights[f"{p}.post_attention_layernorm.weight"], eps=eps)
        gs = _proj(h, weights[f"{p}.mlp.gate_proj.weight"])
        us = _proj(h, weights[f"{p}.mlp.up_proj.weight"])
        hl = _proj(ck.silu(gs) * us, weights[f"{p}.mlp.down_proj.weight"])
        hidden = ck.elwise_add(residual, hl)

    hidden = st.rms_norm_plain(hidden, weights[f"{prefix}.norm.weight"], eps=eps)
    if recipe.tie_word_embeddings:
        return _proj(hidden, embed_w)[0]
    return _proj(hidden, weights["lm_head.weight"])[0]
