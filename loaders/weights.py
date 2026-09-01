"""Checkpoint weight loading for the reference pipeline.

Combines the minimal safetensors reader with the FP8 blocked dequant to turn a
set of shard paths into a flat fp32 weight dictionary shaped exactly like the
checkpoint tensor names (T2 / parity-ready).
"""

from __future__ import annotations

import numpy as np

from . import safetensors_reader as sr
from .fp8 import dequant_weight_auto


def dequant_tensors(tensors: dict[str, object], block_h: int = 128,
                    block_w: int = 128,
                    scale_suffix: str = "_scale_inv") -> dict[str, object]:
    """Dequantize a flat tensor dict IN PLACE (and return it).

    For every pair `X.weight` + scale companion (`scale_suffix`:
    `_scale_inv` APPENDS -> `X.weight_scale_inv`; a leading-dot suffix like
    `.scale` REPLACES `.weight` -> `X.scale`), replace `X.weight` with the
    block-dequantized float32 tensor and DROP the scale entry. The block
    format is inferred from the scale (F32/BF16 inverse scales -> legacy
    128x128; E8M0 u8 -> FP8-E4M3 or packed FP4-E2M1).
    """
    names = [n for n in tensors if n.endswith(".weight")]
    for n in names:
        scale_name = (n[: -len(".weight")] + scale_suffix
                      if scale_suffix.startswith(".") else n + scale_suffix)
        if scale_name not in tensors:
            continue  # non-quantized weight
        w = tensors[n]
        scale = tensors.pop(scale_name)
        try:
            tensors[n] = dequant_weight_auto(w, scale, (block_h, block_w))
        except BaseException:
            tensors[scale_name] = scale
            raise
    return tensors


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
