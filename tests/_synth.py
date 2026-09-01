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


def write_tiny_deepseek_checkpoint(d: str) -> dict:
    """Single-shard DeepSeek-V4-style fixture: E4M3 + E8M0(ue8m0) scales for
    attn/gate/shared-expert matrices, FP4-packed (E2M1) routed experts with
    ue8m0 scales, BF16 embed/norm/head/vision/aligner. Returns the raw
    storage arrays (same dtype as on "disk") for assertions."""
    rng = np.random.default_rng(202)
    H, NH, HD, KEXP = 256, 4, 512, 48
    nexp = 2
    store: dict[str, tuple[str, np.ndarray]] = {}

    def bf16(name, shape):
        if shape == ():
            arr = rng.standard_normal(size=1)[0].astype(np.float32)
        else:
            arr = rng.standard_normal(size=shape).astype(np.float32)
        store[name] = ("BF16", fp32_to_bf16_bits(arr))

    def f32(name, shape):
        store[name] = ("F32", rng.standard_normal(size=shape).astype(np.float32))

    def fp8(name, shape):
        a = rng.integers(0, 255, size=shape, dtype=np.uint8)
        a[a == 127] = 126
        bh, bw = (shape[0] + 127) // 128, (shape[1] + 127) // 128
        s = rng.integers(1, 255, size=(bh, bw), dtype=np.uint8)
        s[s == 255] = 254
        store[name] = ("F8_E4M3", a)
        store[name[: -len(".weight")] + ".scale"] = ("F8_E8M0", s)

    def fp4exp(name, shape):
        packed = rng.integers(0, 256, size=(shape[0], (shape[1] + 1) // 2),
                              dtype=np.uint8)
        s = rng.integers(1, 255, size=(shape[0], (shape[1] + 31) // 32),
                         dtype=np.uint8)
        s[s == 255] = 254
        store[name] = ("U8", packed)
        store[name[: -len(".weight")] + ".scale"] = ("F8_E8M0", s)

    def fp4_group(name, shape):
        packed = rng.integers(0, 256, size=(shape[0], (shape[1] + 1) // 2),
                              dtype=np.uint8)
        s = rng.integers(1, 255, size=(shape[0], (shape[1] + 31) // 32),
                         dtype=np.uint8)
        s[s == 255] = 254
        store[name] = ("U8", packed)
        store[name[: -len(".weight")] + ".scale"] = ("F8_E8M0", s)

    bf16("embed.weight", (32, H))
    bf16("head.weight", (32, H))
    bf16("norm.weight", (H,))
    f32("hc_head_scale", (1,))
    f32("hc_head_base", (4,))
    f32("hc_head_fn", (4, H * 4))
    f32("attn_sink", (NH,))
    f32("tid2eid", (32, 4))

    # one MLA-ish block + one DSpark-ish block share the same tensor names
    for lidx in (0, 1):
        L = f"layers.{lidx}."
        bf16(L + "attn_norm.weight", (H,))
        fp8(L + "attn.wq_a.weight", (128, H))
        bf16(L + "attn.q_norm.weight", (128,))
        fp8(L + "attn.wq_b.weight", (NH * HD // 2, 128))
        fp8(L + "attn.wkv.weight", (HD, H))
        bf16(L + "attn.kv_norm.weight", (HD,))
        bf16(L + "attn.wo_a.weight", (64, NH * HD // 8))
        fp8(L + "attn.wo_b.weight", (H, 64))
        fp8(L + "ffn.gate.weight", (16, H))
        fp4_group(L + "ffn.shared_experts.w1.weight", (KEXP, H))
        fp4_group(L + "ffn.shared_experts.w3.weight", (KEXP, H))
        fp4_group(L + "ffn.shared_experts.w2.weight", (H, KEXP))
        for e in range(nexp):
            fp4exp(L + f"ffn.experts.{e}.w1.weight", (KEXP, H))
            fp4exp(L + f"ffn.experts.{e}.w3.weight", (KEXP, H))
            fp4exp(L + f"ffn.experts.{e}.w2.weight", (H, KEXP))
        bf16(L + "ffn_norm.weight", (H,))
        f32(L + "hc_attn_scale", (3,))
        f32(L + "hc_ffn_scale", (3,))
        f32(L + "hc_attn_base", (24,))
        f32(L + "hc_ffn_base", (24,))
        f32(L + "hc_attn_fn", (24, H * 4))
        f32(L + "hc_ffn_fn", (24, H * 4))

    # minimal vision tower (1 block) + aligner, all BF16
    bf16("vision.patch_embed.proj.weight", (16, 588))
    bf16("vision.patch_embed.proj.bias", (16,))
    bf16("vision.blocks.0.norm1.weight", (16,))
    bf16("vision.blocks.0.attn.wqkv.weight", (48, 16))
    bf16("vision.blocks.0.attn.wqkv.bias", (48,))
    bf16("vision.blocks.0.attn.wo.weight", (16, 16))
    bf16("vision.blocks.0.attn.wo.bias", (16,))
    bf16("vision.blocks.0.norm2.weight", (16,))
    bf16("vision.blocks.0.mlp.w1.weight", (32, 16))
    bf16("vision.blocks.0.mlp.w2.weight", (16, 32))
    bf16("vision.norm.weight", (16,))
    bf16("aligner.w1.weight", (16, 16 * 9))
    bf16("aligner.w1.bias", (16,))
    bf16("aligner.w2.weight", (16, 16))
    bf16("aligner.w2.bias", (16,))
    bf16("image_start", (H,))
    bf16("image_pad", (H,))
    bf16("image_end", (H,))
    bf16("image_newline", (H,))

    os.makedirs(d, exist_ok=True)
    write_safetensors(os.path.join(d, "dsv4.safetensors"), store)
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 1},
                   "weight_map": {k: "dsv4.safetensors" for k in store}}, f)
    return {k: v for k, (_, v) in store.items()}
