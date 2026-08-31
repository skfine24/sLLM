"""qwen3_5 hybrid GPU decode driver (GatedDeltaNet + gated full attention).

Replaces the heavy per-token ops inside the hybrid incremental decode with GPU
kernels while keeping the exact numpy scaffold around them for parity:

  * dense projections/MLP/lm_head  -> cuBLAS `sllm_gemm`
  * GatedDeltaNet single-step      -> `sllm_gated_delta_step` (in-place state)
  * full-attention last-row        -> `sllm_attention_decode`
  * residual adds                  -> `sllm_elwise_add`
  * norms / rotary / silu / gating / conv-window stay on the host (cheap,
    elementwise; fused kernels are a later increment).

Numerics mirror `ref/incremental._decode_hybrid` so GPU logits match the numpy
oracle within fp tolerance. Requires a built `sllm_gpu.so`; callers should
degrade to the numpy path when the kernels/device are absent.
"""

from __future__ import annotations

import numpy as np

from kernels import _sllm_cuda as ck
from ref import qwen3_5 as qq

_TEXTP = "model.language_model"


def _proj(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    """c = h (1,in) @ w.T (in,out) via cuBLAS -> (1, out)."""
    h = np.ascontiguousarray(h, dtype=np.float32)
    wt = np.ascontiguousarray(np.asarray(w, dtype=np.float32).T, dtype=np.float32)
    return ck.gemm(h, wt)


def gpu_hybrid_decode_step(cache, weights, recipe, last_id: int) -> np.ndarray:
    """Logits (V,) for the token after `last_id`, matching the GPU hybrid path.

    `cache` is the numpy `IncrementalCache` from a numpy `prefill`; its K/V,
    recurrent state and conv window are updated in place (same contract as the
    numpy decode path).
    """
    from ref import incremental as inc  # local import to avoid a cycle

    eps = cache.eps
    fa = recipe.full_attention
    nh, hd = fa.num_heads, fa.head_dim

    embed_w = weights[f"{_TEXTP}.embed_tokens.weight"]
    hidden = embed_w[[last_id]].astype(np.float32)  # (1, H)

    pos = cache.n_ctx
    cache.n_ctx += 1
    cos, sin = inc._pos_cos_sin(cache.theta, recipe.rotary_dim(), pos)

    for i, bt in enumerate(recipe.layer_types):
        in_norm = weights[f"{_TEXTP}.layers.{i}.input_layernorm.weight"]
        post_norm = weights[f"{_TEXTP}.layers.{i}.post_attention_layernorm.weight"]
        residual = hidden
        h_in = qq.rms_norm(hidden, in_norm, eps=eps)

        if bt == "linear_attention":
            la = recipe.linear_attention
            fw = inc._linear_attn_weights(weights, i)
            key_dim = la.num_key_heads * la.key_head_dim
            value_dim = la.num_value_heads * la.value_head_dim
            kc = la.conv_kernel_size

            mixed = np.transpose(_proj(h_in, fw["w_in_qkv"]), (1, 0))  # (C, 1)
            win = np.concatenate([cache.conv_win[i], mixed[None]], axis=-1)
            conv = qq.causal_conv1d_depthwise(win[:, :, -kc:], fw["w_conv"], activation="silu")
            mixed_conv = np.transpose(conv[:, :, -1:], (0, 2, 1))  # (1,1,C)
            cache.conv_win[i] = win[:, :, -(kc - 1):]

            query, key, value = qq._split(mixed_conv, [key_dim, key_dim, value_dim])
            query = query.reshape(1, 1, -1, la.key_head_dim)
            key = key.reshape(1, 1, -1, la.key_head_dim)
            value = value.reshape(1, 1, la.num_value_heads, la.value_head_dim)
            ratio = la.num_value_heads // la.num_key_heads
            if ratio > 1:
                query = np.repeat(query, ratio, axis=2)
                key = np.repeat(key, ratio, axis=2)

            # L2-norm + kd^-0.5 query scaling, exactly like the numpy recurrent
            # path, then feed the GPU single-step kernel (state updated in place).
            q = qq.l2norm(query[0, 0]).astype(np.float32)
            k = qq.l2norm(key[0, 0]).astype(np.float32)
            q = q / np.float32(np.sqrt(la.key_head_dim))
            v = np.ascontiguousarray(value[0, 0], dtype=np.float32)

            z = _proj(h_in, fw["w_z"]).reshape(1, 1, la.num_value_heads, la.value_head_dim)
            bvec = _proj(h_in, fw["w_b"])
            avec = _proj(h_in, fw["w_a"])
            beta = qq.sigmoid(bvec)[0].astype(np.float32)
            g = (-np.exp(fw["a_log"]).astype(np.float32)
                 * qq.softplus(avec.astype(np.float32) + fw["dt_bias"]))[0].astype(np.float32)

            out = ck.gated_delta_step(q, k, v, g, beta, cache.state[i][0])
            core = qq.rms_norm_gated(out[None, None], fw["norm_w"], z, eps=eps)
            core = core.reshape(1, 1, -1)
            h = _proj(core.reshape(1, -1), fw["w_out"])[0][None]  # -> (1, H)

        else:
            fw = inc._full_attn_weights(weights, i)
            kvh = fa.num_kv_heads
            n_rep = nh // kvh
            qg = _proj(h_in, fw["w_q"]).reshape(1, nh, hd * 2)
            q2, gate = qq._split(qg, [hd, hd])
            q2 = qq.rms_norm(q2, fw["q_norm_w"], eps=eps).reshape(1, nh, 1, hd)
            k = qq.rms_norm(
                _proj(h_in, fw["w_k"]).reshape(1, kvh, 1, hd), fw["k_norm_w"], eps=eps)
            v = _proj(h_in, fw["w_v"]).reshape(1, kvh, 1, hd)
            q2, k = qq.apply_rotary_pos_emb(q2, k, cos[None, None], sin[None, None])
            cache.k[i] = np.concatenate([cache.k[i], k[0]], axis=1)
            cache.v[i] = np.concatenate([cache.v[i], v[0]], axis=1)
            kt, vt = inc._replicate(cache.k[i], cache.v[i], n_rep)
            attn = ck.attention_decode(q2[0, :, 0, :], kt, vt, hd ** -0.5)
            attn = attn.reshape(1, nh * hd) * qq.sigmoid(gate.reshape(1, -1))
            h = _proj(attn, fw["w_o"])

        hidden = ck.elwise_add(residual, h)

        residual = hidden
        hh = qq.rms_norm(hidden, post_norm, eps=eps)
        gs = _proj(hh, weights[f"{_TEXTP}.layers.{i}.mlp.gate_proj.weight"])
        us = _proj(hh, weights[f"{_TEXTP}.layers.{i}.mlp.up_proj.weight"])
        hl = _proj(qq.silu(gs) * us, weights[f"{_TEXTP}.layers.{i}.mlp.down_proj.weight"])
        hidden = ck.elwise_add(residual, hl)

    hidden = qq.rms_norm(hidden, weights[f"{_TEXTP}.norm.weight"], eps=eps)
    return _proj(hidden, weights["lm_head.weight"])[0]
