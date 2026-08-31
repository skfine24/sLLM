"""Shared synthetic-checkpoint helpers for tests (not a test module)."""

from __future__ import annotations

import json
import os
import struct

import numpy as np


def write_safetensors(path: str, tensors: dict) -> None:
    """Minimal safetensors writer (tests only).

    tensors: name -> (dtype_str, np.ndarray of the STORAGE dtype)
    F8_E4M3 -> uint8 array, BF16 -> uint16 array, natives -> native arrays.
    """
    header, blobs, off = {}, [], 0
    for n, (dt, arr) in tensors.items():
        raw = arr.tobytes()
        header[n] = {"dtype": dt, "shape": list(arr.shape),
                     "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_tiny_qwen4_checkpoint(d: str, cfg, tp_hint: int = 2) -> dict:
    """A 2-shard, checkpoint-named qwen4_exp fixture with REAL fp8 block
    layout (F8_E4M3 + F32 weight_scale_inv on quantizable 2-D weights, BF16
    embed/norms, F32 experts) sized from cfg. Returns the source arrays dict
    (raw storage arrays, same dtype as on "disk") for assertions.

    Shard split mimics the real index: layer tensors spread over both shards,
    embed on shard 1, lm_head on shard 2.
    """
    rng = np.random.default_rng(123)
    H = cfg.hidden
    d1, d2 = {}, {}

    def rand_fp8(shape):
        a = rng.integers(0, 255, size=shape, dtype=np.uint8)
        a[a == 127] = 126  # avoid the NaN byte pattern
        return a

    def quantized(store, name, shape):
        store[name] = ("F8_E4M3", rand_fp8(shape))
        store[name + "_scale_inv"] = ("F32", (rng.random(
            ((shape[0] + 127) // 128, (shape[1] + 127) // 128),
            dtype=np.float32) + 0.1))

    d1["model.language_model.embed_tokens.weight"] = (
        "BF16", fp32_to_bf16_bits(rng.standard_normal((32, H))))
    d2["lm_head.weight"] = (
        "BF16", fp32_to_bf16_bits(rng.standard_normal((32, H))))

    kd = cfg.lin_k_heads * cfg.lin_k_dim
    vd = cfg.lin_v_heads * cfg.lin_v_dim
    nh, kvh, hd = cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
    L = "model.language_model.layers.{}."
    for i, bt in enumerate(cfg.layer_types):
        store = d1 if i % 2 == 0 else d2
        if bt == "linear_attention":
            quantized(store, L.format(i) + "linear_attn.in_proj_qkv.weight",
                      (2 * kd + vd, H))
            quantized(store, L.format(i) + "linear_attn.out_proj.weight",
                      (H, vd))
            store[L.format(i) + "linear_attn.A_log"] = (
                "F32", rng.standard_normal(cfg.lin_v_heads).astype(np.float32))
        else:
            quantized(store, L.format(i) + "self_attn.q_proj.weight",
                      (nh * 2 * hd, H))
            quantized(store, L.format(i) + "self_attn.k_proj.weight",
                      (kvh * hd, H))
            quantized(store, L.format(i) + "self_attn.o_proj.weight",
                      (H, nh * hd))
        for hcname in ("attn_hyper_connection", "mlp_hyper_connection"):
            store[L.format(i) + hcname + ".hc_norm.weight"] = (
                "BF16", fp32_to_bf16_bits(rng.standard_normal(2 * H)))
        quantized(store, L.format(i) + "mlp.experts.0.gate_proj.weight",
                  (cfg.moe_inter, H))
        quantized(store, L.format(i) + "mlp.experts.1.down_proj.weight",
                  (H, cfg.moe_inter))
        quantized(store, L.format(i) + "mlp.shared_expert.gate_proj.weight",
                  (cfg.shared_inter, H))
        store[L.format(i) + "mlp.gate.weight"] = (
            "F32", rng.standard_normal((cfg.n_experts, H)).astype(np.float32))

    write_safetensors(os.path.join(d, "shard1.safetensors"), d1)
    write_safetensors(os.path.join(d, "shard2.safetensors"), d2)
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 1},
                   "weight_map": {**{n: "shard1.safetensors" for n in d1},
                                  **{n: "shard2.safetensors" for n in d2}}}, f)
    src = {}
    for store in (d1, d2):
        for n, (_, arr) in store.items():
            src[n] = arr
    return src


def write_q4_dev_fixture(d: str, cfg=None) -> dict:
    """FULL tiny qwen4_exp dev weights as a single-shard safetensors
    checkpoint (F32 storage, no quant companions) + index.json, so loader
    paths and bench scripts can run against a real on-disk layout locally.
    Returns the source weight dict (default rng) for output comparisons."""
    from serving.dev_model import tiny_qwen4_exp_cfg, tiny_qwen4_exp_weights

    cfg = cfg or tiny_qwen4_exp_cfg()
    w = tiny_qwen4_exp_weights(cfg)
    os.makedirs(d, exist_ok=True)
    write_safetensors(os.path.join(d, "q4fix.safetensors"),
                      {k: ("F32", v) for k, v in w.items()})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 1},
                   "weight_map": {k: "q4fix.safetensors" for k in w}}, f)
    return w
