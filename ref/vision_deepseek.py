"""DeepSeek-V4 vision encoder: numpy oracle (ViT + Aligner).

Faithful port of `ref/hf_sources/dsv4/vision.py`: SigLIP-style tower with
2D RoPE (per-pixel), full bidirectional attention, then a 3x downsample
aligner into LLM hidden dim. Weights arrive dequantized to fp32 (loader).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l(x, w) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32)
            @ np.asarray(w, dtype=np.float32).T).astype(np.float32)


def silu(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)


def gelu(x) -> np.ndarray:
    return (0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)


def rms_norm(x, weight, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.mean(np.square(x, dtype=np.float64), -1, keepdims=True)
    return (x * (1.0 / np.sqrt(m.astype(np.float32) + eps))
            * np.asarray(weight, np.float32)).astype(np.float32)


def _bf16_rt(v: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 (truncate) -> fp32: numpy has no bfloat16 storage."""
    u = np.asarray(v, np.float32).view(np.uint32)
    return ((u >> 16) << 16).view(np.float32).reshape(v.shape)


def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float = 10000.0):
    # reference get_vision_cos_sin: per-pixel 2D rope, h/w interleaved per pair
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    hp = np.broadcast_to(np.arange(n_h, dtype=np.float32)[:, None, None],
                         (n_h, n_w, dim // 2))
    wp = np.broadcast_to(np.arange(n_w, dtype=np.float32)[None, :, None],
                         (n_h, n_w, dim // 2))
    freqs = np.stack([hp * inv[None, None, :], wp * inv[None, None, :]],
                     axis=-1).reshape(n_h * n_w, dim)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def apply_rotary(x, cos, sin) -> np.ndarray:
    """x (n, H?, D) -> rotate via chunked (first-half/last-half) 2D scheme."""
    c = np.asarray(cos, dtype=np.float32)
    s = np.asarray(sin, dtype=np.float32)
    missing = (x.ndim - 1) - (c.ndim - 1)
    for _ in range(missing):
        c = np.expand_dims(c, -2)
        s = np.expand_dims(s, -2)
    h = x.shape[-1] // 2
    c = np.broadcast_to(c, x.shape[:-1] + (h,))
    s = np.broadcast_to(s, x.shape[:-1] + (h,))
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s],
                          -1).astype(np.float32)


@dataclass
class VisionCfg:
    dim: int = 1024
    n_layers: int = 32
    n_heads: int = 16
    inter_dim: int = 2816
    patch_size: int = 14
    downsample_ratio: int = 3
    rope_theta: float = 10000.0

    @property
    def rope_dim(self) -> int:
        return self.dim // self.n_heads // 2


class DeepseekVisionEncoder:
    """ViT tower + aligner; all weights read from the flat checkpoint dict.

    `bf16=True` simulates the checkpoint's BF16 storage + the reference's
    per-op cast-to-bf16: weights are round-tripped at load and activation
    boundaries (embed/attn/mlp/aligner outputs) are bf16-cast between ops,
    giving the noise floor a BF16 GEMM engine will see on the cluster.
    """

    def __init__(self, w: dict, cfg: VisionCfg, bf16: bool = False):
        self.cfg = cfg
        self.bf16 = bf16
        rt = _bf16_rt if bf16 else (lambda v: v)
        self.embed = lambda x: _l(x, rt(w["vision.patch_embed.proj.weight"])) + \
            rt(np.asarray(w["vision.patch_embed.proj.bias"], np.float32))
        self.block_weights = []
        for i in range(cfg.n_layers):
            b = f"vision.blocks.{i}."
            self.block_weights.append({
                "norm1": rt(np.asarray(w[b + "norm1.weight"], np.float32)),
                "wqkv": rt(np.asarray(w[b + "attn.wqkv.weight"], np.float32)),
                "qkv_b": rt(np.asarray(w[b + "attn.wqkv.bias"], np.float32)),
                "wo": rt(np.asarray(w[b + "attn.wo.weight"], np.float32)),
                "wo_b": rt(np.asarray(w[b + "attn.wo.bias"], np.float32)),
                "norm2": rt(np.asarray(w[b + "norm2.weight"], np.float32)),
                "w1": rt(np.asarray(w[b + "mlp.w1.weight"], np.float32)),
                "w2": rt(np.asarray(w[b + "mlp.w2.weight"], np.float32)),
            })
        self.norm = rt(np.asarray(w["vision.norm.weight"], np.float32))
        self.aw1 = rt(np.asarray(w["aligner.w1.weight"], np.float32))
        self.aw1b = rt(np.asarray(w["aligner.w1.bias"], np.float32))
        self.aw2 = rt(np.asarray(w["aligner.w2.weight"], np.float32))
        self.aw2b = rt(np.asarray(w["aligner.w2.bias"], np.float32))

    def vit(self, patches, n_h: int, n_w: int) -> np.ndarray:
        """patches (n_patch, 3, p, p) -> (n_patch, vision_dim)."""
        n = patches.shape[0]
        x = self.embed(patches.reshape(n, -1))
        rt = _bf16_rt if self.bf16 else (lambda v: v)
        x = rt(x)
        cos, sin = vision_cos_sin(n_h, n_w, self.cfg.rope_dim,
                                  self.cfg.rope_theta)
        # 2D rope is only applied to a slice (rope_dim) of each patch token
        rd = self.cfg.rope_dim
        for b in self.block_weights:
            hn = self.cfg.dim // self.cfg.n_heads
            x = x + self._attn(rms_norm(x, b["norm1"]), b, cos, sin, rd, hn)
            x = rt(x)
            x = x + self._mlp(rms_norm(x, b["norm2"]), b)
            x = rt(x)
        return rms_norm(x, self.norm)

    def _attn(self, x, b, cos, sin, rd, hn):
        n = x.shape[0]
        qkv = _l(x, b["wqkv"]) + b["qkv_b"]
        q, k, v = [t.reshape(n, self.cfg.n_heads, hn) for t in
                   np.split(qkv, 3, axis=-1)]
        # reference rotates the WHOLE head dims (second half pairs) with the
        # 2D rope cos/sin (length == head_dim // 2 == rope_dim)
        q = apply_rotary(q, cos[:n], sin[:n])
        k = apply_rotary(k, cos[:n], sin[:n])
        q = q.transpose(1, 0, 2)  # (H, n, hn)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        sc = np.einsum("hnd,hmd->hnm", q, k) / np.sqrt(hn)
        p = sc - sc.max(-1, keepdims=True)
        e = np.exp(p.astype(np.float64))
        e = (e / e.sum(-1, keepdims=True)).astype(np.float32)
        o = np.einsum("hnm,hmd->hnd", e, v)
        o = o.transpose(1, 0, 2).reshape(n, -1)
        return _l(o, b["wo"]) + b["wo_b"]

    def _mlp(self, x, b):
        gate, up = np.split(_l(x, b["w1"]), 2, axis=-1)
        return _l(silu(gate) * up, b["w2"])

    def align(self, x, n_h: int, n_w: int) -> np.ndarray:
        """(n_patch, vdim) -> aligned (n_h/r * n_w/r, llm_dim) via 3x pooling."""
        r = self.cfg.downsample_ratio
        dim = self.cfg.dim
        g = x.reshape(n_h, n_w, dim).transpose(2, 0, 1)  # (d, nh, nw)
        pad_h = (-n_h) % r
        pad_w = (-n_w) % r
        if pad_h or pad_w:
            g = np.pad(g, ((0, 0), (0, pad_h), (0, pad_w)))
        # unfold: r*r neighboring patches -> (n_h/r * n_w/r, r*r*d)
        gh, gw = g.shape[1] // r, g.shape[2] // r
        blocks = g.reshape(dim, gh, r, gw, r).transpose(1, 3, 2, 4, 0)
        blocks = blocks.reshape(gh * gw, r * r * dim)
        out = _l(gelu(_l(blocks, self.aw1) + self.aw1b), self.aw2) + self.aw2b
        return _bf16_rt(out) if self.bf16 else out.astype(np.float32)
