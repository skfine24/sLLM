"""A3 tp-module tests: cluster topology, simulated collectives, and the
per-rank weight view over the streaming loader (dev machine, numpy only).

Gates:
- topology derives the verified DGX Spark pair from the recipe and rejects a
  tp.size/world mismatch;
- SimCollectives performs the exact NCCL SUM/CONCAT semantics and catches
  missing/duplicate/mis-shaped rank contributions;
- RankWeightTable: rank sees only its slices; fp8 dequant happens on the rank
  slice with the co-sharded scales and equals the correspondingly sliced
  full-tensor dequant; non-owned experts raise.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from loaders.fp8 import decode_bf16_array, dequant_weight_blocked
from loaders.streaming import CheckpointIndex, LazyWeightTable
from loaders.tp_shard import Qwen4ExpSharding
from ref.qwen4_exp_pipeline import Qwen4ExpCfg
from recipes.schema import Recipe
from tests._synth import write_tiny_qwen4_checkpoint
from tp.collectives import (
    CollectivesError, SimCollectives, all_gather_parts, all_reduce_partials,
)
from tp.rank_table import RankOwnership, RankWeightTable
from tp.topology import ClusterTopology, TopologyError

L = "model.language_model.layers.0."


def _cfg() -> Qwen4ExpCfg:
    return Qwen4ExpCfg(
        hidden=8, hc_count=2, hc_lowrank=2,
        layer_types=("linear_attention", "full_attention"),
        lin_k_heads=2, lin_k_dim=2, lin_v_heads=4, lin_v_dim=2, lin_conv=3,
        attn_heads=2, attn_kv_heads=2, attn_head_dim=4, rotary_factor=0.5,
        idx_heads=2, idx_dim=4, idx_budget=4, idx_ratio=2,
        n_experts=4, top_k=2, moe_inter=8, shared_inter=8,
    )


class TestTopology(unittest.TestCase):
    def test_from_qwen4_recipe(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "recipes", "Qwen3.8-Flash-Next-FP8.yaml"),
                encoding="utf-8") as f:
            recipe = Recipe.from_yaml(f.read())
        topo = ClusterTopology.from_recipe(recipe)
        self.assertEqual(topo.world_size, 2)
        self.assertTrue(topo.ranks[0].is_head)
        self.assertEqual(topo.ranks[0].host, "192.168.0.250")
        self.assertEqual(topo.ranks[1].pair_ip, "10.100.25.2")

    def test_tp_size_mismatch_rejected(self):
        recipe = Recipe.from_dict({"model_id": "x", "arch": "qwen4_exp",
                                   "tp": {"size": 1}})
        with self.assertRaises(TopologyError):
            ClusterTopology.from_recipe(recipe)


class TestSimCollectives(unittest.TestCase):
    def test_allreduce_allgather_semantics(self):
        sim = SimCollectives(2)
        p0 = np.ones((2, 3), np.float32)
        p1 = np.full((2, 3), 2.0, np.float32)
        sim.contribute("o_proj", 0, p0)
        sim.contribute("o_proj", 1, p1)
        np.testing.assert_allclose(sim.all_reduce("o_proj"), 3.0)
        sim.contribute("logits", 0, np.zeros((1, 4)))
        sim.contribute("logits", 1, np.ones((1, 4)))
        self.assertEqual(sim.all_gather("logits", axis=1).shape, (1, 8))
        sim.reset("o_proj")
        with self.assertRaises(CollectivesError):
            sim.all_reduce("o_proj")

    def test_missing_duplicate_and_bad_rank(self):
        sim = SimCollectives(2)
        sim.contribute("t", 0, np.ones(3))
        with self.assertRaises(CollectivesError):
            sim.all_reduce("t")                      # rank 1 missing
        sim.contribute("t", 1, np.ones(3))
        with self.assertRaises(CollectivesError):
            sim.contribute("t", 1, np.ones(3))       # double contribute
        with self.assertRaises(CollectivesError):
            sim.contribute("t", 5, np.ones(3))       # outside world
        with self.assertRaises(CollectivesError):
            all_reduce_partials([np.ones(2), np.ones(3)])  # shape mismatch

    def test_reference_helpers_match_sim(self):
        sim = SimCollectives(3)
        parts = [np.full((2,), float(r)) for r in range(3)]
        for r, p in enumerate(parts):
            sim.contribute("s", r, p)
        np.testing.assert_allclose(sim.all_reduce("s"),
                                   all_reduce_partials(parts))
        np.testing.assert_allclose(sim.all_gather("s"),
                                   all_gather_parts(parts))


class TestRankWeightTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _cfg()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = write_tiny_qwen4_checkpoint(cls.tmp.name, cls.cfg)
        cls.index = CheckpointIndex(cls.tmp.name)
        cls.table = LazyWeightTable(cls.index)
        cls.sh = Qwen4ExpSharding(cls.cfg, tp=2)
        cls.r0 = RankWeightTable(cls.table, cls.sh, 0)
        cls.r1 = RankWeightTable(cls.table, cls.sh, 1)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.tmp.cleanup()

    def test_fp8_stays_raw_uint8_per_rank(self):
        n = "model.language_model.layers.1.self_attn.q_proj.weight"
        w = self.r0.get(n)
        self.assertEqual(w.dtype, np.uint8)
        self.assertEqual(w.shape, (8, 8))  # 1 of 2 heads x (2*4) rows
        np.testing.assert_array_equal(w, self.table.get(n)[:8])

    def test_rank_dequant_equals_sliced_full_dequant(self):
        L1 = "model.language_model.layers.1."  # the QSA layer in the fixture
        # (multi-segment in_proj_qkv dequant is covered at REAL block-aligned
        # geometry in test_streaming_tp; tiny sub-block multi-segment scale
        # slicing is contractually rejected by shard_scale)
        for name, shape in [
                (L1 + "self_attn.q_proj.weight", (16, 8)),
                (L1 + "self_attn.o_proj.weight", (8, 8)),
                (L + "mlp.experts.0.gate_proj.weight", (8, 8)),
                (L + "mlp.shared_expert.gate_proj.weight", (8, 8))]:
            full_dq = dequant_weight_blocked(
                self.src[name], self.src[name + "_scale_inv"])
            kind = self.sh.plan_for(name).kind
            for r, rt in ((0, self.r0), (1, self.r1)):
                if not rt.owns(name):
                    continue
                got = rt.dequant(name)
                if kind == "replicated" or kind == "experts":
                    want = full_dq
                else:
                    rr = self.sh.rank_ranges(name, shape)[r]
                    axis = 0 if kind == "split_out" else 1
                    want = np.concatenate(
                        [(full_dq[b:e] if axis == 0 else full_dq[:, b:e])
                         for b, e in rr], axis=axis)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-9,
                                           err_msg=name)

    def test_multisegment_subblock_scale_rejected(self):
        from loaders.tp_shard import ShardError
        with self.assertRaises(ShardError):
            self.sh.shard_scale(L + "linear_attn.in_proj_qkv.weight",
                                np.ones((1, 1), np.float32))

    def test_expert_ownership(self):
        self.assertTrue(self.r0.owns(L + "mlp.experts.1.down_proj.weight"))
        self.assertFalse(self.r1.owns(L + "mlp.experts.1.down_proj.weight"))
        self.assertEqual(self.sh.owner(2), 1)  # plan-level: expert 2 -> rank 1
        with self.assertRaises(RankOwnership):
            self.r1.get(L + "mlp.experts.1.down_proj.weight")
        owned = self.r1.owned_names()
        self.assertNotIn(L + "mlp.experts.0.gate_proj.weight", owned)

    def test_replicated_and_vocab_split(self):
        n = L + "mlp.gate.weight"
        np.testing.assert_array_equal(self.r0.get(n), self.r1.get(n))
        e = "model.language_model.embed_tokens.weight"
        w0, w1 = self.r0.get(e), self.r1.get(e)
        full = decode_bf16_array(self.src[e])
        self.assertEqual(w0.shape, (16, 8))
        np.testing.assert_array_equal(np.concatenate([w0, w1], axis=0), full)


if __name__ == "__main__":
    unittest.main()
