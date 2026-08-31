"""Checkpoint weight loading for the reference pipeline.

Combines the minimal safetensors reader with the FP8 blocked dequant to turn a
set of shard paths into a flat fp32 weight dictionary shaped exactly like the
checkpoint tensor names (T2 / parity-ready).
"""

from __future__ import annotations

import numpy as np

from . import safetensors_reader as sr
from .fp8 import dequant_weight_blocked


def dequant_tensors(tensors: dict[str, object], block_h: int = 128, block_w: int = 128) -> dict[str, object]:
    """Dequantize a flat tensor dict in place (returning a new dict).

    For every pair `X.weight` (F8_E4M3, uint8) + `X.weight_scale_inv`
    (float32), replace `X.weight` with the block-dequantized float32 tensor.
    Every other tensor (BF16 already decoded to fp32, norms, embeddings,
    lm_head, mtp.fc) is passed through unchanged.
    """
    out = dict(tensors)
    names = [n for n in tensors if n.endswith(".weight")]
    for n in names:
        scale_name = n + "_scale_inv"
        if scale_name not in tensors:
            continue  # non-quantized weight
        fp8 = tensors[n]
        scale = tensors[scale_name]
        if fp8.dtype != np.uint8 or scale.dtype != np.float32:
            raise TypeError(f"unexpected dtypes for {n}: {fp8.dtype}, {scale.dtype}")
        out[n] = dequant_weight_blocked(fp8, scale, block_h, block_w)
    return out


def load_recipe_weights(
    shard_paths: list[str],
    block_h: int = 128,
    block_w: int = 128,
) -> dict[str, object]:
    """Load and dequantize all tensors from a list of shard paths."""
    all_t = {}
    for path in shard_paths:
        all_t.update(sr.load_tensors(path))
    return dequant_tensors(all_t, block_h, block_w)
