"""A2 loader/TP-shard tests: streaming fp8-persistent loader + scale-aware
tensor-parallel slicing (dev machine, numpy only).

Gates (operational track phase A, docs/design/09 sec.4 methodology):
- ShardFile/LazyWeightTable: F8_E4M3 payloads stay RAW uint8, dequant matches
  the reference blocked loop, BF16 surfaces as float32, index resolution works
  for multi-shard (index.json) and single-file checkpoints.
- Qwen4ExpSharding: plans for real qwen4_exp geometry validate block-safe for
  every split tensor; per-rank dequant(weight, scale) equals the matching
  slice(s) of the full dequant (the contract the C2 loader relies on) for
  both real-size aligned tensors and sub-block tiny tensors; column/row GEMM
  partition identities and the MoE expert partition hold numerically.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest

import numpy as np

from loaders.fp8 import decode_bf16_array, dequant_weight_blocked, dequant_weight_blocked_loop
from loaders.streaming import CheckpointError, CheckpointIndex, LazyWeightTable
from loaders.tp_shard import Qwen4ExpSharding, ShardError
from ref.qwen4_exp_pipeline import Qwen4ExpCfg


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


class TestStreamingLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        rng = np.random.default_rng(7)
        cls.fp8 = rng.integers(0, 255, size=(256, 128), dtype=np.uint8)
        cls.fp8[cls.fp8 == 127] = 126  # avoid the single NaN byte pattern
        scale = rng.random((2, 1), dtype=np.float32) + 0.01
        cls.scale_bits = fp32_to_bf16_bits(scale)
        cls.scale_f32 = decode_bf16_array(cls.scale_bits)
        t1 = {
            "model.language_model.layers.0.self_attn.q_proj.weight":
                ("F8_E4M3", cls.fp8),
            "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv":
                ("BF16", cls.scale_bits),
            "model.language_model.layers.0.input_layernorm.weight":
                ("BF16", fp32_to_bf16_bits(np.ones(128))),
        }
        t2 = {
            "model.language_model.embed_tokens.weight":
                ("F32", rng.standard_normal((64, 128), dtype=np.float32)),
            "model.language_model.layers.1.mlp.gate.weight":
                ("F32", rng.standard_normal((8, 128), dtype=np.float32)),
        }
        write_safetensors(os.path.join(d, "shard1.safetensors"), t1)
        write_safetensors(os.path.join(d, "shard2.safetensors"), t2)
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": 1},
                       "weight_map": {**{n: "shard1.safetensors" for n in t1},
                                      **{n: "shard2.safetensors" for n in t2}}}, f)
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_index_resolution_and_fp8_raw(self):
        idx = CheckpointIndex(self.dir)
        self.assertEqual(len(idx.names()), 5)
        self.assertEqual(idx.shard_of(
            "model.language_model.layers.1.mlp.gate.weight"),
            "shard2.safetensors")
        got = idx.get("model.language_model.layers.0.self_attn.q_proj.weight")
        np.testing.assert_array_equal(got, self.fp8)  # RAW bytes, no decode
        self.assertEqual(got.dtype, np.uint8)
        idx.close()

    def test_dequant_matches_reference_loop(self):
        idx = CheckpointIndex(self.dir)
        tab = LazyWeightTable(idx, block=(128, 128))
        n = "model.language_model.layers.0.self_attn.q_proj.weight"
        self.assertTrue(tab.is_quantized(n))
        dq = tab.dequant(n)
        ref = dequant_weight_blocked_loop(self.fp8, self.scale_f32, 128, 128)
        np.testing.assert_allclose(dq, ref, rtol=0, atol=1e-6)
        ln = tab.get("model.language_model.layers.0.input_layernorm.weight")
        self.assertEqual(ln.dtype, np.float32)
        np.testing.assert_allclose(ln, np.ones(128), atol=1e-2)
        idx.close()

    def test_layer_fetch_folds_scales(self):
        idx = CheckpointIndex(self.dir)
        tab = LazyWeightTable(idx, block=(128, 128))
        layer = tab.layer(0)
        self.assertNotIn(
            "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv",
            layer)
        w = layer["model.language_model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(w.dtype, np.float32)
        self.assertEqual(w.shape, (256, 128))
        idx.close()

    def test_single_file_checkpoint_and_errors(self):
        tmp = tempfile.TemporaryDirectory()
        write_safetensors(os.path.join(tmp.name, "only.safetensors"), {
            "lm_head.weight": ("F32", np.ones((4, 4), np.float32))})
        idx = CheckpointIndex(tmp.name)
        self.assertEqual(idx.names(), ["lm_head.weight"])
        with self.assertRaises(CheckpointError):
            idx.get("nope")
        tmp.cleanup()


# ---------------------------------------------------------------------------
# TP sharding
# ---------------------------------------------------------------------------

L = "model.language_model.layers.0."


def _real_cfg() -> Qwen4ExpCfg:
    """Real qwen4_exp knobs (recipes/Qwen3.8-Flash-Next-FP8.yaml), one linear + one QSA."""
    return Qwen4ExpCfg(
        hidden=2560, hc_count=4, hc_lowrank=320,
        layer_types=("linear_attention", "full_attention"),
        attn_heads=24, attn_kv_heads=2, attn_head_dim=256,
        idx_budget=2048, idx_ratio=4,
        n_experts=512, top_k=10, moe_inter=640, shared_inter=640)


def _tp_cfg() -> Qwen4ExpCfg:
    """Small TP-feasible geometry (all head counts divisible by tp=2)."""
    return Qwen4ExpCfg(
        hidden=8, hc_count=2, hc_lowrank=2,
        layer_types=("linear_attention", "full_attention"),
        lin_k_heads=2, lin_k_dim=2, lin_v_heads=4, lin_v_dim=2, lin_conv=3,
        attn_heads=2, attn_kv_heads=2, attn_head_dim=4, rotary_factor=0.5,
        idx_heads=2, idx_dim=4, idx_budget=4, idx_ratio=2,
        n_experts=4, top_k=2, moe_inter=8, shared_inter=8,
    )


class TestTPPlansRealGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _real_cfg()
        cls.sh = Qwen4ExpSharding(cls.cfg, tp=2)

    def test_all_split_tensors_block_safe(self):
        for name, shape in {
                L + "self_attn.q_proj.weight": (12288, 2560),
                L + "self_attn.k_proj.weight": (512, 2560),
                L + "self_attn.v_proj.weight": (512, 2560),
                L + "self_attn.o_proj.weight": (2560, 6144),
                L + "linear_attn.in_proj_qkv.weight": (10240, 2560),
                L + "linear_attn.in_proj_z.weight": (6144, 2560),
                L + "linear_attn.in_proj_a.weight": (48, 2560),
                L + "linear_attn.out_proj.weight": (2560, 6144),
                "model.language_model.embed_tokens.weight": (248320, 2560),
                "lm_head.weight": (248320, 2560)}.items():
            if self.sh.plan_for(name).kind in ("split_out", "split_in"):
                self.sh.validate_tensor(name, shape, quantized=True)

    def test_plan_kinds(self):
        p = self.sh.plan_for
        self.assertEqual(p(L + "self_attn.q_proj.weight").kind, "split_out")
        self.assertEqual(p(L + "self_attn.o_proj.weight").kind, "split_in")
        self.assertEqual(p(L + "mlp.experts.7.gate_proj.weight").kind,
                         "experts")
        self.assertEqual(p(L + "mlp.experts.7.gate_proj.weight").expert, 7)
        # shared expert replicates: 640/2=320 is NOT 128-block aligned
        self.assertEqual(p(L + "mlp.shared_expert.gate_proj.weight").kind,
                         "replicated")
        self.assertEqual(p(L + "mlp.gate.weight").kind, "replicated")
        self.assertEqual(p(L + "self_attn.indexer.index_qk_proj.weight").kind,
                         "replicated")
        self.assertEqual(p(L + "attn_hyper_connection.hc_norm.weight").kind,
                         "replicated")

    def test_rank_ranges_head_granular(self):
        rr = self.sh.rank_ranges(L + "self_attn.q_proj.weight", (12288, 2560))
        self.assertEqual(rr[0], [(0, 6144)])      # 12 heads x 512
        self.assertEqual(rr[1], [(6144, 12288)])
        rr = self.sh.rank_ranges(L + "linear_attn.in_proj_qkv.weight",
                                 (10240, 2560))
        # rank 1 rows: q half [1024,2048), k half [3072,4096), v [7168,10240)
        self.assertEqual(rr[1], [(1024, 2048), (3072, 4096), (7168, 10240)])

    def test_real_size_qkv_shard_dequant_contract(self):
        """Real-geometry multi-segment fp8 slicing: rank dequant == full
        dequant sliced the same way; scales reassemble bit-exactly (all
        boundaries block-aligned at real sizes)."""
        rng = np.random.default_rng(21)
        name = L + "linear_attn.in_proj_qkv.weight"
        w = rng.integers(0, 255, size=(10240, 2560), dtype=np.uint8)
        scale = rng.random((80, 20), dtype=np.float32) + 0.1
        pairs = self.sh.shard_pair(name, w, scale)
        np.testing.assert_array_equal(
            self.sh.full_scale_from_shards(name, [p[1] for p in pairs]), scale)
        dq = dequant_weight_blocked(w, scale)
        for r, (wr, sr) in enumerate(pairs):
            dqr = dequant_weight_blocked(wr, sr)
            rr = self.sh.rank_ranges(name, w.shape)[r]
            want = np.concatenate([dq[b:e] for b, e in rr], axis=0)
            np.testing.assert_allclose(dqr, want, rtol=0, atol=0)
        np.testing.assert_array_equal(
            self.sh.full_from_shards(name, [p[0] for p in pairs]), w)

    def test_expert_partition_exact_cover(self):
        owners = [self.sh.owner(e) for e in range(512)]
        self.assertEqual(owners.count(0), 256)
        self.assertEqual(owners.count(1), 256)

    def test_misaligned_split_is_rejected(self):
        # the shared-expert intermediate (640) must NOT be column-split: the
        # halves (320) would cut fp8 block 2 and be scale-ambiguous.
        with self.assertRaises(ShardError):
            self.sh.segment_ranges(640, 128)


class TestTPSlicingTiny(unittest.TestCase):
    """Sub-block tiny tensors: the rank slice stays inside one fp8 block, so
    both ranks legitimately share a scale row (no bit-exact scale
    reassembly here; the contract is dequant-per-rank == sliced full dequant).
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = _tp_cfg()
        cls.sh = Qwen4ExpSharding(cls.cfg, tp=2)

    def test_rank_dequant_matches_sliced_full_dequant(self):
        rng = np.random.default_rng(5)
        cases = [
            (L + "self_attn.q_proj.weight", (16, 8)),   # 2 heads x (2*4)
            (L + "self_attn.o_proj.weight", (8, 8)),    # split_in (nh*hd=8)
            (L + "linear_attn.in_proj_a.weight", (4, 8)),  # 4 head-rows
            (L + "linear_attn.out_proj.weight", (8, 8)),   # split_in (vd=8)
        ]
        for name, shape in cases:
            w = rng.integers(0, 255, size=shape, dtype=np.uint8)
            scale = rng.random(
                ((shape[0] + 127) // 128, (shape[1] + 127) // 128),
                dtype=np.float32) + 0.01
            self.sh.validate_tensor(name, shape, quantized=True)
            pairs = self.sh.shard_pair(name, w, scale)
            np.testing.assert_array_equal(
                self.sh.full_from_shards(name, [p[0] for p in pairs]), w)
            dq = dequant_weight_blocked(w, scale)
            kind = self.sh.plan_for(name).kind
            for r, (wr, sr) in enumerate(pairs):
                dqr = dequant_weight_blocked(wr, sr)
                rr = self.sh.rank_ranges(name, shape)[r]
                if kind == "split_out":
                    want = np.concatenate([dq[b:e] for b, e in rr], axis=0)
                else:
                    want = np.concatenate([dq[:, b:e] for b, e in rr], axis=1)
                np.testing.assert_allclose(dqr, want, rtol=0, atol=1e-9)

    def test_column_row_gemm_partition_identities(self):
        rng = np.random.default_rng(6)
        # column parallel (q_proj plan: rows 16 = 2 heads x 8)
        x = rng.standard_normal((3, 16)).astype(np.float32)
        W = rng.standard_normal((16, 16), dtype=np.float32)
        parts = [x @ ws.T for ws in self.sh.shard(
            L + "self_attn.q_proj.weight", W)]
        np.testing.assert_allclose(np.concatenate(parts, axis=1), x @ W.T,
                                   rtol=1e-5, atol=1e-5)
        # row parallel (o_proj plan: in-dim nh*hd = 8)
        x2 = rng.standard_normal((3, 8)).astype(np.float32)
        Wo = rng.standard_normal((8, 8), dtype=np.float32)
        ws = self.sh.shard(L + "self_attn.o_proj.weight", Wo)
        rr = self.sh.rank_ranges(L + "self_attn.o_proj.weight", (8, 8))
        part2 = [x2[:, b:e] @ wsr.T
                 for r, wsr in enumerate(ws) for (b, e) in rr[r]]
        np.testing.assert_allclose(part2[0] + part2[1], x2 @ Wo.T,
                                   rtol=1e-5, atol=1e-5)

    def test_moe_expert_partition_matches_full(self):
        """Sharded MoE (experts by owner + replicated router weights) ==
        full MoE: rank-partial outputs sum (all-reduce) to the full result."""
        rng = np.random.default_rng(8)
        H, E, I, K = 8, self.cfg.n_experts, self.cfg.moe_inter, self.cfg.top_k
        x = rng.standard_normal((3, H)).astype(np.float32)
        gate = rng.standard_normal((E, H)).astype(np.float32) * 0.1
        eg = rng.standard_normal((E, I, H)).astype(np.float32) * 0.2
        eu = rng.standard_normal((E, I, H)).astype(np.float32) * 0.2
        ed = rng.standard_normal((E, H, I)).astype(np.float32) * 0.2

        logits = gate @ x.T
        probs = np.exp(logits - logits.max(0))
        probs /= probs.sum(0)
        idx = np.argsort(-probs, axis=0)[:K]
        wsel = np.take_along_axis(probs, idx, axis=0)
        wsel /= wsel.sum(0)

        def act(a):  # silu
            return a / (1.0 + np.exp(-a))

        def contrib(e, t, w):
            h = act(eg[e] @ x[t]) * (eu[e] @ x[t])
            return w * (ed[e] @ h)

        full = np.zeros_like(x)
        partials = [np.zeros_like(x) for _ in range(self.sh.tp)]
        for col in range(x.shape[0]):
            for j in range(K):
                e = int(idx[j, col])
                c = contrib(e, col, float(wsel[j, col]))
                full[col] += c
                partials[self.sh.owner(e)][col] += c
        np.testing.assert_allclose(partials[0] + partials[1], full,
                                   rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
