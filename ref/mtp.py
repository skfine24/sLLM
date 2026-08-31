"""Qwen3_5 MTP (Multi-Token Prediction) reference math, ported from vLLM's
`vllm/model_executor/models/qwen3_5_mtp.py` (Qwen3_5MultiTokenPredictor).

Single MTP layer math (num_mtp_layers = 1, mtp layer type = full_attention):

    e  = embed(tokens)
    h' = concat([ pre_fc_norm_embedding(e), pre_fc_norm_hidden(hidden) ], last dim)
    h' = fc(h')                      # 2H -> H
    y  = full_attention_decoder_layer(h')
    z  = norm(y)
    logits = lm_head(z)

`hidden` is the main model's hidden BEFORE its final RMSNorm, for the same
positions. `ids[i]` is the token at position i, so `logits[i]` predicts
token i+1 (a next-token head). Weights reuse the main embedding + LM head.
"""

from __future__ import annotations

import numpy as np

from . import pipeline as _pipeline
from . import qwen3_5 as q

_MTP_PREFIX = "mtp"
_LAYER = "mtp.layers.0"


def mtp_layer_weights(weights: dict) -> tuple[dict, dict]:
    """Full-attention decoder-layer weights + the mlp subset for the MTP layer."""
    p = _LAYER
    attn = {
        "w_q": weights[f"{p}.self_attn.q_proj.weight"],
        "w_k": weights[f"{p}.self_attn.k_proj.weight"],
        "w_v": weights[f"{p}.self_attn.v_proj.weight"],
        "w_o": weights[f"{p}.self_attn.o_proj.weight"],
        "q_norm_w": weights[f"{p}.self_attn.q_norm.weight"],
        "k_norm_w": weights[f"{p}.self_attn.k_norm.weight"],
    }
    mlp = {
        "w_gate": weights[f"{p}.mlp.gate_proj.weight"],
        "w_up": weights[f"{p}.mlp.up_proj.weight"],
        "w_down": weights[f"{p}.mlp.down_proj.weight"],
    }
    return attn, mlp


def mtp_forward(
    ids: np.ndarray,
    hidden: np.ndarray,
    weights: dict,
    recipe,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Run the MTP predictor. ids: (B, S); hidden: (B, S, H) pre-final-norm.
    Returns logits (B, S, vocab)."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    hidden = np.asarray(hidden, dtype=np.float32)
    b, s, _ = hidden.shape
    eps = recipe.rms_norm_eps

    embed = weights["model.language_model.embed_tokens.weight"]
    e = q.rms_norm(embed[ids].astype(np.float32), weights["mtp.pre_fc_norm_embedding.weight"], eps=eps)
    hn = q.rms_norm(hidden, weights["mtp.pre_fc_norm_hidden.weight"], eps=eps)
    cat = np.concatenate([e, hn], axis=-1).astype(np.float32)
    h = q.linear(cat, weights["mtp.fc.weight"]).astype(np.float32)

    attn, mlp = mtp_layer_weights(weights)
    fa = recipe.full_attention
    h = q.decoder_layer_forward(
        h,
        in_norm_w=weights[f"{_LAYER}.input_layernorm.weight"],
        post_norm_w=weights[f"{_LAYER}.post_attention_layernorm.weight"],
        block_type="full_attention",
        rms_eps=eps,
        **{**attn, "cos": cos, "sin": sin,
           "num_heads": fa.num_heads, "kv_heads": fa.num_kv_heads,
           "head_dim": fa.head_dim, "mlp": mlp},
    )
    h = q.rms_norm(h, weights["mtp.norm.weight"], eps=eps)
    return h @ weights["lm_head.weight"].T


def mtp_next_token(model, ids: list[int], weights: dict, recipe,
                   temperature: float | None = None) -> int:
    """Draft the next token with the MTP head (greedy by default)."""
    ids_arr = np.asarray([list(ids)], dtype=np.int64)
    _, hidden = _pipeline.model_forward(ids_arr, weights, recipe, return_hidden_pre_norm=True)
    cos, sin = _pipeline.build_cos_sin_for_positions(recipe, ids_arr.shape[1])
    logits = mtp_forward(ids_arr, hidden, weights, recipe, cos, sin)[0, -1, :]
    if temperature is not None and temperature > 0:
        from runtime.sampler import sample
        return sample(logits, temperature=temperature)
    return int(np.argmax(logits))
