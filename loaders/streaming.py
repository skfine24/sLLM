"""Memory-safe streaming access to big checkpoints (operational track A2).

The 173 GiB qwen4_exp / 156 GiB deepseek_v4 checkpoints cannot be held as
dequantized fp32/bf16 dictionaries (full bf16 dequant is ~2x the fp8 size and
exceeds the dual-node memory), so this module never materializes the whole
model:

- `ShardFile`  mmaps one safetensors shard; tensors are read lazily. F8_E4M3
  payloads are returned as RAW uint8 views (no decode), BF16/natives as
  decoded arrays per tensor.
- `CheckpointIndex` resolves tensor name -> shard via the HF
  `*.safetensors.index.json` weight map (or a single-file checkpoint).
- `LazyWeightTable` is the loader front end for the runtime: per-tensor
  `get()` (fp8 stays uint8), `dequant()` on demand with the checkpoint's
  per-[128,128] `weight_scale_inv`, and per-layer fetches.

The fp8 bytes are the persistent on-host/pinned representation the TP2 loader
(C2) shards and uploads; dequantization happens per tensor/block, never as a
whole-model pass.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .fp8 import dequant_weight_blocked
from .safetensors_reader import (
    TensorSpec,
    decode_bf16_array,
    read_header,
)


class CheckpointError(ValueError):
    pass


class ShardFile:
    """Lazily readable safetensors shard backed by a file mmap.

    `raw(name)` returns an np.memmap view (no copy) of the tensor in its
    storage dtype: F8_E4M3/F8_E5M2/BF16 -> uint8/uint16, natives -> native.
    `get(name)` decodes BF16 -> float32 and passes everything else through
    (F8_E4M3 stays RAW uint8: decode is the caller's explicit step).
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.header = read_header(f)
        self._mm: dict[str, np.ndarray] = {}
        self._closed = False

    def spec(self, name: str) -> TensorSpec:
        return self.header.spec(name)

    def raw(self, name: str) -> np.ndarray:
        if self._closed:
            raise CheckpointError(f"shard closed: {self.path}")
        spec = self.spec(name)
        hit = self._mm.get(name)
        if hit is not None:
            return hit
        dtype = {
            "F8_E4M3": np.uint8, "F8_E5M2": np.uint8, "BF16": np.uint16,
            "F64": np.float64, "F32": np.float32, "F16": np.float16,
            "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
            "U64": np.uint64, "U32": np.uint32, "U16": np.uint16,
            "U8": np.uint8, "BOOL": np.bool_,
        }[spec.dtype]
        n_items = int(np.prod(spec.shape)) if spec.shape else 1
        arr = np.memmap(self.path, dtype=dtype, mode="r",
                        offset=self.header.data_offset + spec.begin,
                        shape=(n_items,)).reshape(spec.shape)
        self._mm[name] = arr
        return arr

    def get(self, name: str) -> np.ndarray:
        spec = self.spec(name)
        arr = self.raw(name)
        if spec.dtype == "BF16":
            return decode_bf16_array(arr)
        if spec.dtype == "F8_E4M3":
            return np.asarray(arr)  # raw uint8, dequant is explicit
        return np.asarray(arr)

    def names(self) -> list[str]:
        return list(self.header.tensors.keys())

    def close(self) -> None:
        self._mm.clear()
        self._closed = True


class CheckpointIndex:
    """Tensor-name -> shard resolution for a checkpoint directory."""

    def __init__(self, model_dir: str):
        self.model_dir = os.path.expanduser(model_dir)
        if not os.path.isdir(self.model_dir):
            raise CheckpointError(f"not a directory: {self.model_dir}")
        self.weight_map: dict[str, str] = {}
        for fn in sorted(os.listdir(self.model_dir)):
            if fn.endswith(".safetensors.index.json"):
                with open(os.path.join(self.model_dir, fn), encoding="utf-8") as f:
                    doc = json.load(f)
                wm = doc.get("metadata", {}).get("total_size")
                self.total_size = wm if isinstance(wm, int) else None
                self.weight_map = dict(doc["weight_map"])
                break
        if not self.weight_map:
            singles = [fn for fn in sorted(os.listdir(self.model_dir))
                       if fn.endswith(".safetensors")]
            if len(singles) != 1:
                raise CheckpointError(
                    f"{self.model_dir}: no index json and {len(singles)} "
                    "loose safetensors files (need exactly 1 or an index)")
            self.weight_map = {n: singles[0] for n in
                               self._shard_names(os.path.join(self.model_dir, singles[0]))}
            self.total_size = None
        self._shards: dict[str, ShardFile] = {}

    def _shard_names(self, path: str) -> list[str]:
        with open(path, "rb") as f:
            return list(read_header(f).tensors.keys())

    def shard_file(self, shard: str) -> ShardFile:
        hit = self._shards.get(shard)
        if hit is None:
            hit = ShardFile(os.path.join(self.model_dir, shard))
            self._shards[shard] = hit
        return hit

    def names(self) -> list[str]:
        return list(self.weight_map.keys())

    def filter(self, prefix: str) -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]

    def shard_of(self, name: str) -> str:
        try:
            return self.weight_map[name]
        except KeyError as exc:
            raise CheckpointError(f"unknown tensor: {name!r}") from exc

    def spec(self, name: str) -> TensorSpec:
        return self.shard_file(self.shard_of(name)).spec(name)

    def get(self, name: str) -> np.ndarray:
        """Decoded tensor (F8_E4M3 -> raw uint8, BF16 -> float32)."""
        return self.shard_file(self.shard_of(name)).get(name)

    def close(self) -> None:
        for s in self._shards.values():
            s.close()
        self._shards.clear()


class LazyWeightTable:
    """Per-tensor on-demand weight access for the runtime.

    Quantized tensors (`X.weight` with an `X.weight_scale_inv` companion) are
    stored on disk as F8_E4M3 + per-[block] inverse scales; `get()` keeps them
    raw and `dequant()` materializes float32 for exactly one tensor. Scales
    may be stored as F32 or BF16; both surface as float32.
    """

    def __init__(self, index: CheckpointIndex,
                 block: tuple[int, int] = (128, 128)):
        self.index = index
        self.block = block

    def names(self) -> list[str]:
        return self.index.names()

    def is_quantized(self, name: str) -> bool:
        return (name.endswith(".weight")
                and name + "_scale_inv" in self.index.weight_map)

    def get(self, name: str) -> np.ndarray:
        return self.index.get(name)

    def scale(self, name: str) -> np.ndarray:
        return self.index.get(name + "_scale_inv")

    def dequant(self, name: str) -> np.ndarray:
        if not self.is_quantized(name):
            return self.index.get(name)
        return dequant_weight_blocked(self.index.get(name), self.scale(name),
                                      self.block[0], self.block[1])

    def layer(self, layer_idx: int, prefix: str = "model.language_model") -> dict:
        """All tensors of decoder layer `layer_idx` (fp8 tensors dequantized,
        scale companions folded away), named by full checkpoint name."""
        p = f"{prefix}.layers.{layer_idx}."
        out = {}
        for n in self.index.filter(p):
            if n.endswith("_scale_inv"):
                continue
            out[n] = self.dequant(n)
        return out

    def embeddings_head(self, prefix: str = "model.language_model") -> dict:
        emb = f"{prefix}.embed_tokens.weight"
        out = {emb: self.dequant(emb)}
        if "lm_head.weight" in self.index.weight_map:
            out["lm_head.weight"] = self.dequant("lm_head.weight")
        return out
