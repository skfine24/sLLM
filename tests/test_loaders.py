"""Loader tests: FP8/BF16 decode (vs ml_dtypes oracle) and the minimal
safetensors reader (roundtrip against a self-written file)."""

import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import ml_dtypes  # noqa: E402
except ImportError:  # keep suite discovery green on nodes without it
    raise unittest.SkipTest("ml_dtypes not installed")

from loaders import fp8  # noqa: E402
from loaders import safetensors_reader as sr  # noqa: E402

E4M3 = ml_dtypes.float8_e4m3fn
BF16 = ml_dtypes.bfloat16


def _write_safetensors(path: str, tensors: dict[str, tuple[str, tuple, bytes]]):
    """Minimal safetensors writer for tests.

    tensors: name -> (dtype_str, shape, raw_bytes)
    """
    header = {"__metadata__": {"format": "test"}}
    offset = 0
    data = b""
    for name, (dt, shape, raw) in tensors.items():
        nbytes = len(raw)
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        data += raw
    enc = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(enc)))
        f.write(enc)
        f.write(data)


class TestFp8Decode(unittest.TestCase):
    def test_all_bytes_match_ml_dtypes(self):
        u8 = np.arange(256, dtype=np.uint8)
        got = fp8.decode_e4m3fn_array(u8)
        ref = u8.view(E4M3).astype(np.float32)
        nan_ref = np.isnan(ref)
        nan_got = np.isnan(got)
        self.assertTrue(np.array_equal(nan_ref, nan_got), "NaN pattern mismatch")
        np.testing.assert_allclose(got[~nan_got], ref[~nan_got], rtol=0, atol=0)

    def test_known_values(self):
        cases = {
            0x00: 0.0, 0x01: 2.0 ** -9, 0x07: 7.0 * 2.0 ** -9,
            0x08: 2.0 ** -6, 0x38: 1.0, 0x3A: 1.25, 0x3C: 1.5,
            0x68: 64.0, 0x78: 256.0, 0x7E: 448.0,
        }
        for byte, expected in cases.items():
            self.assertEqual(fp8.decode_e4m3fn(byte), expected, f"byte 0x{byte:02x}")

    def test_nan_and_signs(self):
        self.assertTrue(np.isnan(fp8.decode_e4m3fn(0x7F)))
        self.assertTrue(np.isnan(fp8.decode_e4m3fn(0xFF)))
        self.assertEqual(fp8.decode_e4m3fn(0x80), -0.0)
        self.assertEqual(fp8.decode_e4m3fn(0xB8), -1.0)

    def test_bf16_decode_matches_ml_dtypes(self):
        rng = np.random.default_rng(0)
        u16 = rng.integers(0, 65536, size=(4, 8)).astype(np.uint16)
        got = fp8.decode_bf16_array(u16)
        ref = u16.view(BF16).astype(np.float32)
        np.testing.assert_array_equal(got, ref)

    def test_bf16_decode_known(self):
        # 1.5 in bf16: fp32 0x3FC00000 -> upper 16 bits 0x3FC0
        arr = fp8.decode_bf16_array(np.array([0x3FC0], dtype=np.uint16))
        self.assertEqual(arr[0], 1.5)


class TestDequantBlocked(unittest.TestCase):
    def test_matches_loop(self):
        rng = np.random.default_rng(1)
        h, w = 300, 250
        fp8w = rng.integers(0, 256, size=(h, w)).astype(np.uint8)
        bh, bw = 128, 128
        scale = (rng.standard_normal((np.ceil(h / bh).astype(int), np.ceil(w / bw).astype(int))) * 0.1
                 + 0.5).astype(np.float32)
        got = fp8.dequant_weight_blocked(fp8w, scale, bh, bw)
        ref = fp8.dequant_weight_blocked_loop(fp8w, scale, bh, bw)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            fp8.dequant_weight_blocked(
                np.zeros((128, 128), dtype=np.uint8), np.zeros((2, 2), dtype=np.float32), 128, 128
            )

    def test_block_dimensions_match_checkpoint(self):
        # Audited shapes: [10240, 5120] weight, [80, 40] scale -> 128x128 blocks
        w = np.zeros((10240, 5120), dtype=np.uint8)
        s = np.zeros((80, 40), dtype=np.float32)
        out = fp8.dequant_weight_blocked(w, s, 128, 128)
        self.assertEqual(out.shape, (10240, 5120))


class TestSafetensorsReader(unittest.TestCase):
    def _make_file(self):
        rng = np.random.default_rng(2)
        f32 = rng.standard_normal((3, 5)).astype(np.float32)
        # generate bf16-exact values so the byte roundtrip is lossless
        bf16 = rng.integers(0, 65536, size=(2, 7)).astype(np.uint16).view(BF16).astype(np.float32)
        f8 = rng.integers(0, 256, size=(4, 4)).astype(np.uint8)
        u32 = bf16.view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        bf16_bytes = u16.astype("<u2").tobytes()
        f32_bytes = f32.tobytes()
        f8_bytes = f8.tobytes()
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "t.safetensors")
        _write_safetensors(
            p,
            {
                "a.f32": ("F32", (3, 5), f32_bytes),
                "b.bf16": ("BF16", (2, 7), bf16_bytes),
                "c.f8": ("F8_E4M3", (4, 4), f8_bytes),
            },
        )
        return p, f32, bf16, f8

    def test_roundtrip(self):
        p, f32, bf16, f8 = self._make_file()
        out = sr.load_tensors(p)
        self.assertEqual(set(out.keys()), {"a.f32", "b.bf16", "c.f8"})
        np.testing.assert_array_equal(out["a.f32"], f32)
        np.testing.assert_array_equal(out["c.f8"], f8)
        # bf16 decode: 16-bit precision -> tolerance at 16-bit relative level
        np.testing.assert_allclose(out["b.bf16"], bf16, rtol=2e-3, atol=1e-3)

    def test_selective_load(self):
        p, f32, bf16, f8 = self._make_file()
        out = sr.load_tensors(p, names=["c.f8"])
        self.assertEqual(list(out.keys()), ["c.f8"])
        np.testing.assert_array_equal(out["c.f8"], f8)

    def test_missing_tensor(self):
        p, *_ = self._make_file()
        with self.assertRaises(sr.SafetensorsError):
            sr.load_tensors(p, names=["nope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
