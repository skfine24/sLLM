"""Qwen3_5 full-model reference forward (numpy), T2-ready.

Wires the reference modules in `ref/qwen3_5.py` into a checkpoint-named weight
dictionary (the same names the loader produces) and runs the text transformer:
    embed -> 64 decoder layers (linear/full per recipe) -> final RMSNorm -> lm_head

Weight keys follow the checkpoint layout:
    model.language_model.embed_tokens.weight
    model.language_model.layers.{i}.input_layernorm.weight
    model.language_model.layers.{i}.linear_attn.*        (linear layers)
    model.language_model.layers.{i}.self_attn.*          (full layers)
    model.language_model.layers.{i}.mlp.*
    model.language_model.norm.weight
    lm_head.weight

All weights are expected ALREADY dequantized to float32 (see
`loaders.weights.dequant_tensors`). Recurrent state is per-layer and discarded
unless requested (prefill path).

Context: text-only for now. MTP spec-decode and vision are later phases.
"""

from __future__ import annotations

import numpy as np

from . import qwen3_5 as q

_TEXT_PREFIX = "model.language_model"


def build_cos_sin_for_positions(recipe, num_tokens: int, positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Partial-rotary cos/sin for a (1, num_tokens) text sequence. Real M-RoPE
    position accounting is a vision/serving concern; text uses arange."""
    if positions is None:
        positions = np.arange(num_tokens, dtype=np.int64)[None, :]
    theta = recipe.full_attention.rope.theta
    rotary_dim = recipe.rotary_dim()
    return q.compute_cos_sin(theta, positions, rotary_dim)


def _linear_attn_weights(weights: dict, layer: int) -> dict:
    p = f"{_TEXT_PREFIX}.layers.{layer}.linear_attn"
    return {
        "w_in_qkv": weights[f"{p}.in_proj_qkv.weight"],
        "w_conv": weights[f"{p}.conv1d.weight"],
        "w_z": weights[f"{p}.in_proj_z.weight"],
        "w_b": weights[f"{p}.in_proj_b.weight"],
        "w_a": weights[f"{p}.in_proj_a.weight"],
        "a_log": weights[f"{p}.A_log"],
        "dt_bias": weights[f"{p}.dt_bias"],
        "norm_w": weights[f"{p}.norm.weight"],
        "w_out": weights[f"{p}.out_proj.weight"],
    }


def _full_attn_weights(weights: dict, layer: int) -> dict:
    p = f"{_TEXT_PREFIX}.layers.{layer}.self_attn"
    return {
        "w_q": weights[f"{p}.q_proj.weight"],
        "w_k": weights[f"{p}.k_proj.weight"],
        "w_v": weights[f"{p}.v_proj.weight"],
        "w_o": weights[f"{p}.o_proj.weight"],
        "q_norm_w": weights[f"{p}.q_norm.weight"],
        "k_norm_w": weights[f"{p}.k_norm.weight"],
    }


def _mlp_weights(weights: dict, layer: int) -> dict:
    p = f"{_TEXT_PREFIX}.layers.{layer}.mlp"
    return {
        "w_gate": weights[f"{p}.gate_proj.weight"],
        "w_up": weights[f"{p}.up_proj.weight"],
        "w_down": weights[f"{p}.down_proj.weight"],
    }


def model_forward(
    ids: np.ndarray,
    weights: dict,
    recipe,
    positions: np.ndarray | None = None,
    return_states: bool = False,
    return_hidden_pre_norm: bool = False,
):
    """Text forward. ids: (B, S) int tokens.

    Returns logits (B, S, vocab); if return_hidden_pre_norm, also returns the
    hidden BEFORE the final RMSNorm (needed by the MTP predictor).
    """
    b, s = ids.shape
    embed_w = weights[f"{_TEXT_PREFIX}.embed_tokens.weight"]
    hidden = embed_w[ids].astype(np.float32)

    cos, sin = build_cos_sin_for_positions(recipe, s, positions)

    states = {}
    for i, block_type in enumerate(recipe.layer_types):
        in_norm_w = weights[f"{_TEXT_PREFIX}.layers.{i}.input_layernorm.weight"]
        post_norm_w = weights[f"{_TEXT_PREFIX}.layers.{i}.post_attention_layernorm.weight"]
        eps = recipe.rms_norm_eps

        residual = hidden
        h = q.rms_norm(hidden, in_norm_w, eps=eps)
        if block_type == "linear_attention":
            la = recipe.linear_attention
            out, state = q.gated_delta_net_forward(
                h, **_linear_attn_weights(weights, i), **{
                    "num_k_heads": la.num_key_heads,
                    "head_k_dim": la.key_head_dim,
                    "num_v_heads": la.num_value_heads,
                    "head_v_dim": la.value_head_dim,
                    "conv_kernel_size": la.conv_kernel_size,
                    "use_qk_l2norm_in_kernel": la.qk_l2norm,
                    "chunked": True,
                    "chunk_size": 64,
                    "rms_eps": eps,
                },
            )
            if return_states:
                states[i] = state
            h = out
        elif block_type == "full_attention":
            fa = recipe.full_attention
            h = q.full_attention_forward(
                h, **_full_attn_weights(weights, i), cos=cos, sin=sin,
                num_heads=fa.num_heads,
                kv_heads=fa.num_kv_heads,
                head_dim=fa.head_dim,
                rms_eps=eps,
            )
        else:  # pragma: no cover
            raise ValueError(block_type)
        hidden = residual + h

        residual = hidden
        h = q.rms_norm(hidden, post_norm_w, eps=eps)
        h = q.mlp_forward(h, **_mlp_weights(weights, i))
        hidden = residual + h

    hidden_pre_norm = hidden
    hidden = q.rms_norm(hidden, weights[f"{_TEXT_PREFIX}.norm.weight"], eps=recipe.rms_norm_eps)
    logits = hidden @ weights["lm_head.weight"].T

    if return_states and return_hidden_pre_norm:
        return logits, states, hidden_pre_norm
    if return_states:
        return logits, states
    if return_hidden_pre_norm:
        return logits, hidden_pre_norm
    return logits
