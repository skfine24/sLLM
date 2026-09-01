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
    scale_suffix: str = "_scale_inv",
    progress=None,
) -> dict[str, object]:
    """Load + DEQUANT-INLINE a set of shards into a flat fp32 dict.

    Each tensor is dequantized as soon as it is read, so the peak host RAM is
    ~the fp32 result alone (no full fp8 copy is held alongside it -- the
    previous read-all-then-dequant path spiked ~2x and gave no output while a
    multi-GiB 27B model loaded). Results are identical to
    `load_tensors` + `dequant_tensors` (same flat names, scales dropped).

    `progress(path, done, total)` is called periodically so a long build shows
    where it is (the serving/main.py qwen3_5 path throttles it into INFO).
    """
    out: dict[str, object] = {}
    for path in shard_paths:
        with open(path, "rb") as f:
            header = sr.read_header(f)
            names = list(header.tensors)
            total = len(names)

            # scale companion names (both suffix forms: appending _scale_inv
            # or a leading-dot suffix that REPLACES .weight); these are never
            # stored in the result.
            scale_names = set()
            for w in names:
                if not w.endswith(".weight"):
                    continue
                sn = (w[: -len(".weight")] + scale_suffix
                      if scale_suffix.startswith(".") else w + scale_suffix)
                if sn in names:
                    scale_names.add(sn)

            def _read_raw(spec):
                f.seek(header.data_offset + spec.begin)
                span = spec.end - spec.begin
                data = f.read(span)
                if len(data) != span:
                    raise sr.SafetensorsError(
                        f"{spec.name!r}: truncated tensor data")
                return data

            store_names = [n for n in names if n not in scale_names]
            total = len(store_names)
            for i, name in enumerate(store_names):
                spec = header.spec(name)
                tensor = sr.decode_tensor_bytes(_read_raw(spec), spec)
                if name.endswith(".weight"):
                    scale_name = (name[: -len(".weight")] + scale_suffix
                                  if scale_suffix.startswith(".")
                                  else name + scale_suffix)
                    if scale_name in header.tensors:
                        scale = sr.decode_tensor_bytes(
                            _read_raw(header.spec(scale_name)),
                            header.spec(scale_name))
                        tensor = dequant_weight_auto(tensor, scale,
                                                     (block_h, block_w))
                out[name] = tensor
                if progress is not None:
                    progress(path, i + 1, total)
    return out
