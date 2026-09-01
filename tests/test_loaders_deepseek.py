"""DeepSeek-V4 block-format loader tests: ue8m0 (E8M0) decode, FP4-E2M1 packed
unpack, FP8-E4M3 block dequant, and automatic format dispatch (L1)."""

import os
import tempfile
import unittest

import numpy as np

from loaders import fp8
from loaders.safetensors_reader import SafetensorsError, parse_header_bytes
from loaders.streaming import CheckpointIndex, LazyWeightTable
from loaders.weights import dequant_tensors

from tests._synth import write_tiny_deepseek_checkpoint


class TestUE8M0(unittest.TestCase):
    def test_value_table(self):
        v = fp8.decode_ue8m0(np.array([127, 128, 126, 129, 0, 255], np.uint8))
        np.testing.assert_allclose(v[:4], [1.0, 2.0, 0.5, 4.0], rtol=1e-6)
        self.assertEqual(v[4], 0.0)
        self.assertTrue(np.isnan(v[5]))

    def test_vectorized(self):
        x = np.arange(1, 255, dtype=np.uint8)  # 0x00 and 0xFF handled separately
        got = fp8.decode_ue8m0(x)
        np.testing.assert_allclose(got, 2.0 ** (x.astype(np.float64) - 127.0),
                                   rtol=1e-6)


class TestFP8MXFPDequant(unittest.TestCase):
    def test_matches_manual_loop(self):
        rng = np.random.default_rng(1)
        w8 = rng.integers(0, 255, size=(260, 400), dtype=np.uint8)  # non-divisible
        w8[w8 == 127] = 126
        bh, bw = (260 + 127) // 128, (400 + 127) // 128
        s = rng.integers(1, 255, size=(bh, bw), dtype=np.uint8)
        got = fp8.dequant_fp8_mxfp_weight(w8, s)
        rows = np.repeat(np.arange(bh), 128)[:260]
        cols = np.repeat(np.arange(bw), 128)[:400]
        ref = fp8.decode_e4m3fn_array(w8) * fp8.decode_ue8m0(s)[rows][:, cols]
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_shape_mismatch_rejected(self):
        w8 = np.zeros((128, 128), np.uint8)
        s = np.zeros((1, 2), np.uint8)
        with self.assertRaises(ValueError):
            fp8.dequant_fp8_mxfp_weight(w8, s)


class TestFP4PackedDequant(unittest.TestCase):
    def test_unpack_matches_manual(self):
        rng = np.random.default_rng(2)
        N, K = 5, 200
        packed = rng.integers(0, 256, size=(N, K // 2), dtype=np.uint8)
        s = rng.integers(1, 255, size=(N, (K + 31) // 32), dtype=np.uint8)
        got = fp8.dequant_fp4_packed_weight(packed, s)
        scale_rows = fp8.decode_ue8m0(s)  # (N, cols)
        # manual reference (byte-split nibbles + per-32-col scale)
        exp = np.empty((N, K), np.float32)
        for i in range(N):
            k = 0
            for j in range(K // 2):
                b = int(packed[i, j])
                for nib in (b & 0x0F, (b >> 4) & 0x0F):
                    exp[i, k] = fp8.FP4_TABLE[nib] * scale_rows[i, k // 32]
                    k += 1
        np.testing.assert_allclose(got, exp, rtol=1e-6)

    def test_scale_rows_mismatch_rejected(self):
        # fp4 path: scale rows must equal the weight rows
        with self.assertRaises(ValueError):
            fp8.dequant_fp4_packed_weight(np.zeros((2, 4), np.uint8),
                                          np.zeros((3, 1), np.uint8))


class TestAutoDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="dsv4_synth")
        cls.src = write_tiny_deepseek_checkpoint(cls.dir)
        cls.index = CheckpointIndex(cls.dir)

    def test_synth_layout_sanity(self):
        self.assertIn("layers.0.attn.wkv.scale", self.src)
        self.assertEqual(self.src["layers.0.attn.wkv.weight"].dtype, np.uint8)
        self.assertEqual(self.src["layers.0.ffn.experts.0.w1.weight"].dtype,
                         np.uint8)

    def test_fp4_expert_auto(self):
        table = LazyWeightTable(self.index, scale_suffix=".scale")
        name = "layers.0.ffn.experts.0.w1.weight"
        got = table.dequant(name)
        w = self.src[name]
        s = self.src[name[: -len(".weight")] + ".scale"]
        ref = fp8.dequant_fp4_packed_weight(w, s)
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_fp8_auto(self):
        table = LazyWeightTable(self.index, scale_suffix=".scale")
        for name in ("layers.0.attn.wkv.weight", "layers.0.ffn.gate.weight",
                     "layers.1.attn.wo_b.weight"):
            got = table.dequant(name)
            sc = name[: -len(".weight")] + ".scale"
            ref = fp8.dequant_fp8_mxfp_weight(self.src[name], self.src[sc])
            np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_shared_experts_fp4_auto(self):
        table = LazyWeightTable(self.index, scale_suffix="scale")
        name = "layers.0.ffn.shared_experts.w1.weight"
        self.assertEqual(table.dequant(name).ndim, 2)
        self.assertEqual(table.dequant(name).shape, self.src[name].shape)

    def test_unquantized_passthrough(self):
        table = LazyWeightTable(self.index, scale_suffix="scale")
        self.assertFalse(table.is_quantized("embed.weight"))
        self.assertEqual(table.dequant("embed.weight").ndim, 2)

    def test_dequant_tensors_uses_suffix(self):
        all_t = {k: self.src[k].copy() for k in self.src}
        n = "layers.0.ffn.experts.1.w2.weight"
        sc = n[: -len(".weight")] + ".scale"
        self.assertIn(sc, all_t)
        dequant_tensors(all_t, scale_suffix=".scale")
        self.assertNotIn(sc, all_t)
        self.assertEqual(all_t[n].dtype, np.float32)


class TestE8M0ReaderDtype(unittest.TestCase):
    def test_reader_registers_e8m0(self):
        import struct
        js = b'{"x":{"dtype":"F8_E8M0","shape":[2],"data_offsets":[0,2]}}'
        h = parse_header_bytes(struct.pack("<Q", len(js)) + js)
        self.assertEqual(h.tensors["x"].dtype, "F8_E8M0")

    def test_load_raw_through_shardfile(self):
        import numpy as np
        from tests._synth import write_tiny_deepseek_checkpoint
        import tempfile
        d = tempfile.mkdtemp(prefix="dsv4_raw")
        write_tiny_deepseek_checkpoint(d)
        index = CheckpointIndex(d)
        raw = index.get("layers.0.ffn.gate.scale")
        np.testing.assert_array_equal(
            raw, np.asarray(  # stored uint8 in the synth file
                index.shard_file("dsv4.safetensors").raw(
                    "layers.0.ffn.gate.scale")))
        self.assertEqual(raw.dtype, np.uint8)


class TestE5M2Decode(unittest.TestCase):
    """Phase 2: F8_E5M2 decoding (sign 5-exp bias15 2-mant, IEEE inf/NaN)."""

    def test_value_patterns(self):
        from loaders.fp8 import decode_e5m2_array
        u8 = np.array([0x00, 0x38, 0x39, 0x01, 0x82, 0x3C, 0x7C, 0x7F,
                       0x3F, 0x7B, 0xFC], np.uint8)
        y = decode_e5m2_array(u8)
        np.testing.assert_allclose(y[:5], [0.0, 0.5, 0.625, 2.0 ** -16,
                                           -2.0 * 2.0 ** -16], rtol=1e-6)
        self.assertEqual(y[5], 1.0)         # exp14 mant0 -> 2^-1
        self.assertTrue(np.isposinf(y[6]))  # +inf
        self.assertTrue(np.isnan(y[7]))     # nan
        self.assertEqual(y[8], 1.75)        # exp15 mant3
        self.assertEqual(y[9], 57344.0)     # max finite
        self.assertTrue(np.isneginf(y[10]))  # -inf

    def test_vectorized_symmetry(self):
        from loaders.fp8 import decode_e5m2_array
        x = np.arange(256, dtype=np.uint8)
        y = decode_e5m2_array(x)
        # sign flip symmetry for non-special values
        flip = decode_e5m2_array((x ^ 0x80).astype(np.uint8))
        norm = np.isfinite(y) & (x.astype(np.uint8) != 0)
        np.testing.assert_allclose(flip[norm], -y[norm], rtol=1e-6)

    def test_reader_and_streaming_decode_e5m2(self):
        import tempfile
        from tests._synth import write_safetensors
        from loaders.safetensors_reader import read_header, decode_tensor_bytes
        from loaders.streaming import ShardFile
        d = tempfile.mkdtemp(prefix="e5m2")
        path = os.path.join(d, "w.safetensors")
        arr_u8 = np.array([0x7B, 0x7C, 0x38, 0x01, 0x80],
                          dtype=np.uint8).reshape(5, 1)
        write_safetensors(path, {"e5": ("F8_E5M2", arr_u8)})
        with open(path, "rb") as f:
            header = read_header(f)
        spec = header.spec("e5")
        dec = decode_tensor_bytes(b"\x7b\x7c\x38\x01\x80", spec)
        self.assertEqual(dec.dtype, np.float32)
        self.assertTrue(np.isposinf(dec[1, 0]))
        # streaming path decodes E5M2 too (never handed out as raw E4M3)
        sf = ShardFile(path)
        got = sf.get("e5")
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(int(got[0, 0]), 57344)
        self.assertTrue(np.isposinf(got[1, 0]))
        sf.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
