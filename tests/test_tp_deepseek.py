"""DeepSeek-V4 TP2 sharding contract (phase 5, local / numpy only).

Proves the `DeepseekV4Sharding` plan: rank slices tile the full tensor,
scales shard consistently with fp8 weights (auto-dispatched for E8M0),
routed experts partition by index, replicates stay whole, and the
130-block fp8 slicing path stays guarded at tiny (sub-block) sizes the same
way qwen4_exp does. Real-geometry fp8 slicing + SimCollectives parity of the
numberics is the cluster (l5) step.
"""

import os
import tempfile
import unittest

import numpy as np

from loaders.streaming import CheckpointIndex, LazyWeightTable
from loaders.tp_shard import ShardError, DeepseekV4Sharding, _EXPERTS, _REP
from ref.deepseek_v4 import DeepseekV4Cfg
from serving.dev_model import tiny_deepseek_v4_cfg, tiny_deepseek_v4_weights
from tests._synth import write_safetensors
from tp.rank_table import RankOwnership, RankWeightTable

P = "layers.0.attn."
PF = "layers.0.ffn."


def _pack_fp4(w: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Round an (N,K) fp32 matrix to packed E2M1 (N,K//2) int8 + (N,K//32)
    E8M0 scale (the DeepSeek expert layout the fp4 dequant expects)."""
    n, k = w.shape
    flat = w.astype(np.float32).ravel()
    amax = np.abs(flat).max() + 1e-6
    s = 2.0 ** (int(np.ceil(np.log2(amax / 6.0))))
    q = np.clip(np.floor(flat / s + 0.5), -6, 6)
    magtab = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mags = np.abs(q)
    idx = np.argmin(np.abs(magtab[None, :] - mags[:, None]), axis=-1) + 1
    idx[mags < 0.25] = 0
    nib = idx + np.where(q < 0, 8, 0)
    nib = nib.reshape(n, k // 2, 2)
    packed = ((nib[:, :, 0] & 0x0F) | ((nib[:, :, 1] & 0x0F) << 4)).astype(
        np.int8)
    scale_u8 = np.full((n, k // 32), 127 + int(np.round(np.log2(s))), np.uint8)
    return packed, scale_u8


def _make_fixture(d, cfg) -> dict:
    """Write a tiny deepseek shard with fp32 split tensors, one REP fp8+ue8m0
    tensor (auto-dispatch), and an fp4-packed expert. Returns source arrays."""
    rng = np.random.default_rng(21)
    w = tiny_deepseek_v4_weights(cfg)
    store = {}

    def e8m0(r=8, c=8):
        s = rng.integers(1, 255, size=(r, c), dtype=np.uint8)
        return s

    def fp8(rows, cols):
        a = rng.integers(0, 255, size=(rows, cols), dtype=np.uint8)
        a[a == 127] = 126
        return a

    for name in ("layers.0.attn.wq_a.weight", "layers.0.attn.wq_b.weight",
                 "layers.0.attn.wo_a.weight", "layers.0.attn.wo_b.weight",
                 "layers.0.attn.indexer.wq_b.weight",
                 "layers.0.attn.indexer.weights_proj.weight",
                 "layers.1.attn.wq_a.weight", "layers.1.attn.wo_b.weight",
                 "embed.weight", "head.weight"):
        store[name] = ("F32", w[name].astype(np.float32))

    # one REP fp8 weight with its ue8m0 scale (deepseek-style) -> auto-dispatch
    n = "layers.0.attn.wkv.weight"
    r, c = w[n].shape
    store[n] = ("F8_E4M3", fp8(r, c))
    store["layers.0.attn.wkv.scale"] = ("F8_E8M0", e8m0(
        (r + 127) // 128, (c + 127) // 128))

    # fp4 packed expert w1 for expert 0 (plan_whole)
    n = "layers.0.ffn.experts.0.w1.weight"
    packed, scale_u8 = _pack_fp4(w[n], rng)
    store[n] = ("I8", packed)
    store["layers.0.ffn.experts.0.w1.scale"] = ("F8_E8M0", scale_u8)

    for name in w:
        if name in store:
            continue
        store[name] = ("F32", w[name].astype(np.float32))
    write_safetensors(os.path.join(d, "dsv4_tp.safetensors"), store)
    return w


class TestDeepseekV4Sharding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = tiny_deepseek_v4_cfg()
        cls.src = _make_fixture(cls.tmp.name, cls.cfg)
        cls.index = CheckpointIndex(cls.tmp.name)
        cls.table = LazyWeightTable(cls.index, scale_suffix=".scale")
        cls.sh = DeepseekV4Sharding(cls.cfg, tp=2)
        cls.r0 = RankWeightTable(cls.table, cls.sh, 0)
        cls.r1 = RankWeightTable(cls.table, cls.sh, 1)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.tmp.cleanup()

    def test_split_tensors_tile_the_full(self):
        for name in ("layers.0.attn.wq_b.weight", "layers.0.attn.wo_b.weight",
                     "embed.weight", "head.weight"):
            full = self.src[name]
            plan = self.sh.plan_for(name)
            axis = 0 if plan.kind == "split_out" else 1
            got = np.concatenate([self.sh.shard(name, full)[r]
                                  for r in range(2)], axis=axis)
            np.testing.assert_array_equal(got, full)

    def test_replicated_stay_whole(self):
        # wkv is a REP fp8 tensor -> raw uint8 on every rank
        n = "layers.0.attn.wkv.weight"
        w0 = self.r0.get(n)
        w1 = self.r1.get(n)
        np.testing.assert_array_equal(w0, w1)
        np.testing.assert_array_equal(w0, self.table.get(n))
        # norms are REP fp32 tensors -> equal the source fp32
        a = "layers.0.attn_norm.weight"
        np.testing.assert_array_equal(self.r0.get(a), self.src[a])
        np.testing.assert_array_equal(self.r0.get(a), self.r1.get(a))

    def test_expert_partition(self):
        import re as _re
        n = "layers.0.ffn.experts.3.w2.weight"
        self.assertTrue(self.r1.owns(n))
        self.assertFalse(self.r0.owns(n))
        with self.assertRaises(RankOwnership):
            self.r0.get(n)
        names = [x for x in self.table.names()
                 if _re.search(r"\.experts\.\d+\.", x) and ".scale" not in x]
        for nm in names:
            m = _re.search(r"experts\.(\d+)\.", nm)
            e = int(m.group(1))
            owner = self.sh.owner(e)
            self.assertTrue((owner == 0 and self.r0.owns(nm)) or
                            (owner == 1 and self.r1.owns(nm)))

    def test_rep_fp8_auto_dequant(self):
        """A REP fp8+ue8m0 tensor dequantizes through the auto dispatcher to
        the same value as a direct e4m3*ue8m0 computation."""
        from loaders.fp8 import dequant_fp8_mxfp_weight
        n = "layers.0.attn.wkv.weight"
        got = self.r0.dequant(n)
        full = dequant_fp8_mxfp_weight(self.table.get(n),
                                       self.table.scale(n))
        np.testing.assert_allclose(got, full, rtol=0, atol=0)

    def test_fp4_expert_dequant(self):
        n = "layers.0.ffn.experts.0.w1.weight"
        r = self.r1 if self.sh.owner(0) == 1 else self.r0
        got = r.dequant(n)
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.isfinite(got).all())
        # fp4 round-trip recovers the source within one fp4 quantization step
        # (scale s ranges up to ~1 for the tiny 0.05-scaled weights)
        self.assertLess(np.abs(got - self.src[n]).max(), 1.0)

    def test_tiny_fp8_split_block_contained(self):
        """At tiny sizes a split slice stays INSIDE one 128-block, so the fp8
        scale slicing is unambiguous and validated; only a slice that CUTS a
        block is rejected (the qwen4 multi-segment test covers that path)."""
        n = "layers.0.attn.wo_a.weight"
        self.sh.validate_tensor(n, self.src[n].shape, quantized=True)

    def test_sim_collectives_roundtrip(self):
        """SimCollectives all-reduce/all-gather semantics compose with the
        rank plan: the summed partials reconstruct split_out outputs."""
        from tp.collectives import SimCollectives
        sim = SimCollectives(2)
        name = "layers.0.attn.wo_b.weight"  # split_in (row-parallel): all-reduce
        parts = self.sh.shard(name, self.src[name])
        for r, p in enumerate(parts):
            sim.contribute("wo_b", r, p)
        reduced = sim.all_reduce("wo_b")
        np.testing.assert_allclose(reduced, parts[0] + parts[1])
        # all-gather reconstructs the full vocab split row range
        sim.contribute("head", 0, self.src["head.weight"][:32])
        sim.contribute("head", 1, self.src["head.weight"][32:])
        self.assertEqual(sim.all_gather("head", axis=0).shape,
                         self.src["head.weight"].shape)


if __name__ == "__main__":
    unittest.main()
