"""load_recipe_weights: INLINE dequant must equal the old
load_tensors + dequant_tensors path (scales dropped, same flat fp32 dict),
and must call the progress hook for every tensor (so a multi-GiB 27B load
shows where it is instead of sitting silent)."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loaders import safetensors_reader as sr  # noqa: E402
from loaders.weights import dequant_tensors, load_recipe_weights  # noqa: E402


def _write_safetensors(path: str, tensors: dict[str, tuple[str, tuple, bytes]]):
    header, offset = {}, 0
    for name, (dt, shape, raw) in tensors.items():
        nbytes = len(raw)
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    enc = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(enc)))
        f.write(enc)
        for _, _, raw in tensors.values():
            f.write(raw)


def _bf16_bytes(x: float) -> bytes:
    # bf16 = TOP 16 bits of the fp32 bit pattern, little-endian on disk
    u32 = struct.unpack(">I", struct.pack(">f", x))[0]
    top16 = (u32 >> 16) & 0xFFFF
    return struct.pack("<H", top16)


class TestLoadRecipeWeightsInline(unittest.TestCase):
    def _fixture(self):
        rng = np.random.default_rng(0)
        # E4M3 without the special 0x7F/0xFF NaN payloads (finite only)
        self._w128 = rng.integers(0, 127, size=(128 * 128),
                                  dtype=np.uint8).tobytes()
        w128 = self._w128
        scale = _bf16_bytes(0.5)                       # (1,1) inverse scale
        plain = np.arange(8, dtype=np.uint16).astype('<u2').tobytes()  # BF16
        common = np.arange(4, dtype=np.int32).tobytes()
        path = tempfile.mktemp(suffix=".safetensors")
        _write_safetensors(path, {
            "blk.weight": ("F8_E4M3", [128, 128], w128),
            "blk.weight_scale_inv": ("BF16", [1, 1], scale),
            "plain.weight": ("BF16", [8], plain),
            "common.i": ("I32", [4], common),
        })
        return path

    def test_inline_equals_old_path(self):
        path = self._fixture()
        old = dequant_tensors(sr.load_tensors(path), 128, 128)
        new = load_recipe_weights([path], 128, 128)
        self.assertEqual(set(old), set(new))
        for name in old:
            np.testing.assert_array_equal(np.asarray(new[name]),
                                          np.asarray(old[name]))
        # scales are dropped, quantized weight is dequantized == f8(*0.5)
        self.assertNotIn("blk.weight_scale_inv", new)
        from loaders import fp8
        raw = np.frombuffer(self._w128, np.uint8).reshape(128, 128)
        np.testing.assert_array_equal(new["blk.weight"],
                                      0.5 * fp8.decode_e4m3fn_array(raw))
        self.assertNotIn("plain.weight_scale_inv", new)
        os.remove(path)

    def test_progress_hook_fires(self):
        path = self._fixture()
        seen = []
        load_recipe_weights([path], 128, 128,
                            progress=lambda p, d, t: seen.append((p, d, t)))
        self.assertEqual(len(seen), 3)                 # stored tensors only
        self.assertEqual(seen[-1][1:], (3, 3))
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
