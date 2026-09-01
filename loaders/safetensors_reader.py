"""Minimal safetensors reader (dependency-free).

Supports reading individual tensors from a shard, with explicit handling for
the dtypes used by Qwen checkpoints:
- F8_E4M3 -> raw uint8 (call `loaders.fp8.dequant_weight_blocked` to dequantize
  with its `weight_scale_inv` companion)
- BF16   -> decoded to float32 via `loaders.fp8.decode_bf16_array`
- standard numpy dtypes mapped directly

Also usable in "header-only" mode for remote/Range reads: parse the header
from a short byte prefix, then fetch the tensor bytes at their offsets.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from .fp8 import decode_bf16_array, decode_e5m2_array

_DTYPES = {
    "F64": (np.float64, 8),
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": ("bf16", 2),
    "F8_E5M2": ("e5m2", 1),
    "F8_E4M3": ("e4m3", 1),
    "F8_E8M0": ("e8m0", 1),
    "I64": (np.int64, 8),
    "I32": (np.int32, 4),
    "I16": (np.int16, 2),
    "I8": (np.int8, 1),
    "U64": (np.uint64, 8),
    "U32": (np.uint32, 4),
    "U16": (np.uint16, 2),
    "U8": (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


class SafetensorsError(ValueError):
    pass


# safetensors headers are tiny JSON; cap them so a corrupt/hostile u64 length
# cannot drive an unbounded read()/allocation (convention: < a few hundred MB).
MAX_HEADER_BYTES = 1 << 28  # 256 MiB


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    def nbytes(self) -> int:
        itemsize = _DTYPES[self.dtype][1]
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * itemsize


@dataclass
class Header:
    tensors: dict[str, TensorSpec]
    data_offset: int

    def spec(self, name: str) -> TensorSpec:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise SafetensorsError(f"no tensor named {name!r}") from exc


def parse_header_bytes(prefix: bytes) -> tuple[Header, int]:
    """Parse the header from the first 8+ bytes of a safetensors file.

    Returns (Header, header_length_consumed). `data_offset` in the Header is
    relative to the file start (8 + header length).
    """
    if len(prefix) < 8:
        raise SafetensorsError("prefix too short to contain header length")
    (hlen,) = struct.unpack("<Q", prefix[:8])
    if hlen > MAX_HEADER_BYTES:
        raise SafetensorsError(f"header length {hlen} exceeds cap {MAX_HEADER_BYTES}")
    header_bytes = prefix[8 : 8 + hlen]
    if len(header_bytes) < hlen:
        raise SafetensorsError("header longer than the provided prefix")
    try:
        doc = json.loads(header_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetensorsError(f"header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SafetensorsError("safetensors header root must be a JSON object")
    tensors = {}
    for name, info in doc.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise SafetensorsError(f"bad tensor entry for {name!r}")
        try:
            d, s, o = info["dtype"], info["shape"], info["data_offsets"]
        except KeyError as exc:
            raise SafetensorsError(f"incomplete tensor entry for {name!r}") from exc
        if d not in _DTYPES:
            raise SafetensorsError(f"unsupported dtype {d!r} in {name!r}")
        if not isinstance(o, (list, tuple)) or len(o) != 2:
            raise SafetensorsError(f"data_offsets must be a 2-list for {name!r}")
        begin, end = int(o[0]), int(o[1])
        if begin < 0 or end < begin:
            raise SafetensorsError(
                f"bad data_offsets [{begin}, {end}) for {name!r}")
        try:
            shape = tuple(int(x) for x in s)
        except (TypeError, ValueError) as exc:
            raise SafetensorsError(f"bad shape for {name!r}: {s!r}") from exc
        if any(d < 0 for d in shape):
            raise SafetensorsError(f"negative dimension in shape for {name!r}")
        itemsize = _DTYPES[d][1]
        if (end - begin) % itemsize:
            raise SafetensorsError(
                f"byte span {end - begin} not a multiple of itemsize "
                f"{itemsize} for {name!r}")
        tensors[name] = TensorSpec(name=name, dtype=d, shape=shape,
                                   begin=begin, end=end)
        # cross-check declared shape vs byte span; a mismatch is corruption.
        expected = tensors[name].nbytes()
        if expected != end - begin:
            raise SafetensorsError(
                f"{name!r}: shape {shape} x itemsize {itemsize} = {expected} "
                f"but data_offsets span {end - begin}")
    return Header(tensors=tensors, data_offset=8 + hlen)


def read_header(stream: BinaryIO) -> Header:
    head = stream.read(8)
    if len(head) < 8:
        raise SafetensorsError("file too short to contain a safetensors header length")
    (hlen,) = struct.unpack("<Q", head)
    if hlen > MAX_HEADER_BYTES:
        raise SafetensorsError(f"header length {hlen} exceeds cap {MAX_HEADER_BYTES}")
    body = stream.read(hlen)
    if len(body) < hlen:
        raise SafetensorsError("truncated safetensors header")
    header = parse_header_bytes(head + body)
    return header


def decode_tensor_bytes(data: bytes, spec: TensorSpec) -> np.ndarray:
    """Decode a tensor's raw bytes into a numpy array.

    F8_E4M3 is returned as raw uint8; BF16/E5M2 as float32; standard dtypes
    are mapped natively. E5M2 decodes to float32 (not raw uint8) so it can
    never be silently treated as the E4M3 decoder's input.
    """
    mt, itemsize = _DTYPES[spec.dtype]
    arr = np.frombuffer(data, dtype=np.uint8)
    count = len(arr) // itemsize
    if mt == "e4m3":
        return arr[:count * itemsize].reshape(spec.shape).astype(np.uint8)
    if mt == "e8m0":
        # E8M0 is a block-scale tensor; keep it raw uint8 (decode_ue8m0 is
        # the caller's explicit step).
        return arr[:count * itemsize].reshape(spec.shape).astype(np.uint8)
    if mt == "e5m2":
        u8 = arr[:count * itemsize].reshape(spec.shape)
        return decode_e5m2_array(u8)
    if mt == "bf16":
        u16 = np.frombuffer(data, dtype=np.uint16)
        return decode_bf16_array(u16.reshape(spec.shape))
    return np.frombuffer(data, dtype=mt).reshape(spec.shape)


def load_tensors(path: str, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load tensors from a safetensors file. `names=None` loads all."""
    with open(path, "rb") as f:
        header = read_header(f)
        file_size = f.seek(0, 2)
        want = list(names) if names is not None else list(header.tensors.keys())
        out = {}
        for name in want:
            spec = header.spec(name)
            if header.data_offset + spec.end > file_size:
                raise SafetensorsError(
                    f"{name!r}: offsets [{spec.begin}, {spec.end}) exceed the "
                    f"file ({file_size} B)")
            f.seek(header.data_offset + spec.begin)
            span = spec.end - spec.begin
            data = f.read(span)
            if len(data) != span:
                raise SafetensorsError(f"{name!r}: truncated tensor data")
            out[name] = decode_tensor_bytes(data, spec)
        return out


def load_tensors_from_url(
    url: str,
    names: list[str],
    prefix_len: int = 1 << 20,
) -> dict[str, np.ndarray]:
    """Load specific tensors from a remote safetensors shard via HTTP Range.

    Reads only the header prefix plus the requested tensor byte ranges, so a
    dev machine can validate real checkpoint bytes without a full download.
    """
    import urllib.request

    def _get(begin: int, length: int) -> bytes:
        if begin < 0 or length < 0 or length > MAX_HEADER_BYTES:
            raise SafetensorsError(f"bad range begin={begin} length={length}")
        req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{begin + length - 1}"})
        with urllib.request.urlopen(req) as r:
            # cap the read at `length`: a server IGNORING the Range header
            # answers 200 with the whole shard, and an uncapped read() would
            # slurp it all into RAM.
            return r.read(length)

    prefix = _get(0, prefix_len)
    header = parse_header_bytes(prefix)
    out = {}
    for name in names:
        spec = header.spec(name)
        span = spec.end - spec.begin
        raw = _get(header.data_offset + spec.begin, span)
        if len(raw) != span:
            raise SafetensorsError(
                f"{name!r}: server returned {len(raw)} B, expected {span} B "
                "(Range not honoured?)")
        out[name] = decode_tensor_bytes(raw, spec)
    return out
