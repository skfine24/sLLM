"""Tensor-parallel sharding plans + slicing for qwen4_exp (operational A2).

Semantics follow docs/design/03 sec.4: column-parallel splits the output dim,
row-parallel splits the input dim (all-reduce later), FP8 weights shard
TOGETHER WITH their per-block inverse scales (scales are part of the shard
layout, never recomputed), and MoE experts partition by expert index.

Split units come from the model geometry (Qwen4ExpCfg), never from guessing:
attention splits at head granularity, GDN at key/value-head granularity,
experts at whole-expert granularity. Small control tensors replicate
(routers, hyper-connection mixers, norms, indexer projections, dt/A_log) —
and the shared expert replicates as a deliberate exception: its intermediate
(640) does not split into 128-block-aligned halves for tp=2 (320 % 128 != 0),
and 3 x 640 x 2560 fp8 x 48 layers ~ 236 MB of replication is the cheap,
block-safe answer.

FP8 block-scale alignment rule (loaders/fp8, block (128,128)): a rank's
row/col range may be cut mid-block ONLY when the whole range stays inside one
block; `validate_tensor` proves this before any slicing. Scales then slice by
scale_row = weight_row // 128 (floor at the begin, ceil at the end).

Pure numpy + names: this defines the contract the C2 loader and grouped-GEMM
kernels implement, unit-tested on synthetic tensors on the dev machine.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_REP = "replicated"
_OUT = "split_out"
_IN = "split_in"
_EXPERTS = "experts"


class ShardError(ValueError):
    pass


@dataclass(frozen=True)
class TensorPlan:
    """How one checkpoint tensor shards over TP ranks.

    kind:
      "replicated" - full copy on every rank
      "split_out"  - axis 0 (output features); a rank's rows are the
                     concatenation of its per-segment slices
      "split_in"   - axis 1 (input features), single segment
      "experts"    - owned by exactly one rank (expert partition)
    segments: ((size, unit), ...) along the split axis; unit = alignment
    granularity (head dim / block size); size may be None for dynamic sizes
    (vocab), which always take a single even segment.
    """

    kind: str
    segments: tuple = ()
    expert: int | None = None

    @property
    def unit(self) -> int:
        return self.segments[0][1] if self.segments else 1


class Qwen4ExpSharding:
    """Name-driven TP plan for the qwen4_exp checkpoint tensor namespace."""

    def __init__(self, cfg, tp: int = 2, block: int = 128):
        if tp <= 1:
            raise ShardError("tp must be > 1")
        self.cfg, self.tp, self.block = cfg, tp, block
        for what, n in (("attn_heads", cfg.attn_heads),
                        ("attn_kv_heads", cfg.attn_kv_heads),
                        ("lin_k_heads", cfg.lin_k_heads),
                        ("lin_v_heads", cfg.lin_v_heads),
                        ("n_experts", cfg.n_experts)):
            if n % tp:
                raise ShardError(f"{what}={n} not divisible by tp={tp}")

    # -- naming patterns (docs/design/09 sec.1, names verified on head) -------

    _RE = {
        "embed": re.compile(r"^model\.language_model\.embed_tokens\.weight$"),
        "lm_head": re.compile(r"^lm_head\.weight$"),
        "q": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"),
        "kv": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.self_attn\.[kv]_proj\.weight$"),
        "o": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),
        "gdn_qkv": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$"),
        "gdn_z": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$"),
        "gdn_ba": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_[ab]\.weight$"),
        "gdn_conv": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.linear_attn\.conv1d\.weight$"),
        "gdn_dt": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.linear_attn\.(A_log|dt_bias)$"),
        "gdn_out": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$"),
        "expert": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"),
        "shared": re.compile(
            r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_expert\..*\.weight$"),
    }

    def plan_for(self, name: str) -> TensorPlan:
        c = self.cfg
        R = self._RE
        if R["embed"].match(name) or R["lm_head"].match(name):
            return TensorPlan(_OUT, ((None, 1),))       # vocab-row split
        if R["q"].match(name):
            return TensorPlan(_OUT, ((c.attn_heads * 2 * c.attn_head_dim,
                                      2 * c.attn_head_dim),))
        if R["kv"].match(name):
            return TensorPlan(_OUT, ((c.attn_kv_heads * c.attn_head_dim,
                                      c.attn_head_dim),))
        if R["o"].match(name):
            return TensorPlan(_IN, ((c.attn_heads * c.attn_head_dim,
                                     c.attn_head_dim),))
        if R["gdn_qkv"].match(name) or R["gdn_conv"].match(name):
            kd = c.lin_k_heads * c.lin_k_dim
            vd = c.lin_v_heads * c.lin_v_dim
            return TensorPlan(_OUT, ((kd, c.lin_k_dim), (kd, c.lin_k_dim),
                                     (vd, c.lin_v_dim)))
        if R["gdn_z"].match(name):
            return TensorPlan(_OUT, ((c.lin_v_heads * c.lin_v_dim,
                                      c.lin_v_dim),))
        if R["gdn_ba"].match(name) or R["gdn_dt"].match(name):
            return TensorPlan(_OUT, ((c.lin_v_heads, 1),))
        if R["gdn_out"].match(name):
            return TensorPlan(_IN, ((c.lin_v_heads * c.lin_v_dim,
                                     c.lin_v_dim),))
        m = R["expert"].match(name)
        if m:
            return TensorPlan(_EXPERTS, expert=int(m.group(2)))
        # shared expert + everything else (routers, HC, norms, indexer, ...):
        # replicate (see module docstring for the fp8-block reason on shared).
        return TensorPlan(_REP)

    # -- ranges ------------------------------------------------------------------

    def segment_ranges(self, size: int, unit: int) -> list[tuple[int, int]]:
        per = size // self.tp
        if size % self.tp:
            raise ShardError(f"segment size {size} not divisible by tp={self.tp}")
        if per % unit:
            raise ShardError(
                f"segment {size} (unit {unit}) splits into misaligned "
                f"{per}-slices for tp={self.tp}")
        return [(r * per, (r + 1) * per) for r in range(self.tp)]

    def rank_row_ranges(self, plan: TensorPlan) -> list[list[tuple[int, int]]]:
        """Per-rank list of (begin,end) row ranges, one per segment."""
        out = []
        for r in range(self.tp):
            rr, base = [], 0
            for size, unit in plan.segments:
                if size is None:
                    raise ShardError("dynamic-size plan needs tensor shape")
                b, e = self.segment_ranges(size, unit)[r]
                rr.append((base + b, base + e))
                base += size
            out.append(rr)
        return out

    def _col_ranges(self, plan: TensorPlan) -> list[list[tuple[int, int]]]:
        size, unit = plan.segments[0]
        rng = self.segment_ranges(size, unit)
        return [[rng[r]] for r in range(self.tp)]

    def rank_ranges(self, name: str, shape: tuple) -> list[list[tuple[int, int]]]:
        """Per-rank (begin,end) ranges on the split axis; validates sizes."""
        plan = self.plan_for(name)
        if plan.kind == _OUT:
            if plan.segments[0][0] is None:
                if shape[0] % self.tp:
                    raise ShardError(f"vocab {shape[0]} not divisible by tp")
                per = shape[0] // self.tp
                return [[(r * per, (r + 1) * per)] for r in range(self.tp)]
            total = sum(s for s, _ in plan.segments)
            if total != shape[0]:
                raise ShardError(f"plan expects out-dim {total}, got {shape[0]}")
            return self.rank_row_ranges(plan)
        if plan.kind == _IN:
            if plan.segments[0][0] != shape[1]:
                raise ShardError(f"plan expects in-dim {plan.segments[0][0]}, "
                                 f"got {shape[1]}")
            return self._col_ranges(plan)
        raise ShardError(f"no ranges for kind {plan.kind}")

    def validate_tensor(self, name: str, shape: tuple, quantized: bool = True) -> None:
        """Prove the plan is usable for this tensor; for fp8 tensors also
        prove every rank slice is block-aligned or block-contained."""
        plan = self.plan_for(name)
        if plan.kind in (_REP, _EXPERTS):
            return
        axis = 0 if plan.kind == _OUT else 1
        for rr in self.rank_ranges(name, shape):
            for b, e in rr:
                if e <= b:
                    raise ShardError(f"empty rank slice for {name}")
                if quantized:
                    bl0, bl1 = b // self.block, (e - 1) // self.block
                    if b % self.block and bl1 != bl0:
                        raise ShardError(
                            f"{name}: rank slice [{b},{e}) on axis {axis} cuts "
                            f"block {bl0} without staying inside it "
                            "(fp8 scale would be ambiguous)")

    # -- slicing -------------------------------------------------------------------

    def shard(self, name: str, arr: np.ndarray) -> list:
        """Rank slices of a weight tensor (None for non-owned experts)."""
        plan = self.plan_for(name)
        arr = np.asarray(arr)
        if plan.kind == _REP:
            return [arr] * self.tp
        if plan.kind == _EXPERTS:
            return [arr if self.owner(plan.expert) == r else None
                    for r in range(self.tp)]
        axis = 0 if plan.kind == _OUT else 1
        return [np.concatenate(
            [(sl[b:e] if axis == 0 else sl[:, b:e]) for b, e in rr], axis=axis)
            for rr in self.rank_ranges(name, arr.shape)
            for sl in [arr]]

    def shard_scale(self, name: str, scale: np.ndarray) -> list:
        """Rank slices of the per-block inverse scale, matching `shard()`
        exactly (floor at the begin, ceil at the end of every range)."""
        plan = self.plan_for(name)
        scale = np.asarray(scale)
        if plan.kind == _REP:
            return [scale] * self.tp
        if plan.kind == _EXPERTS:
            raise ShardError("expert tensors must be fp8-block quantized; "
                             "scale slicing is not plan-driven there")
        blk = self.block
        if plan.kind == _OUT:
            if len(plan.segments) > 1:
                # A concatenated multi-segment slice can only carry a
                # block-scale grid of its own when every segment boundary is
                # block-aligned (the real qwen4_exp geometry guarantees this:
                # all GDN segment sizes are multiples of 128).
                for size, _u in plan.segments:
                    if size % blk or (size // self.tp) % blk:
                        raise ShardError(
                            "multi-segment scale slicing needs block-aligned "
                            f"segments (got size {size}, block {blk})")
            return [np.concatenate([scale[b // blk: math.ceil(e / blk)]
                                    for b, e in rr], axis=0)
                    for rr in self.rank_row_ranges(plan)]
        return [np.concatenate([scale[:, b // blk: math.ceil(e / blk)]
                                for b, e in rr], axis=1)
                for rr in self._col_ranges(plan)]

    def shard_pair(self, name: str, fp8: np.ndarray,
                   scale: np.ndarray) -> list:
        """Consistent (weight, scale) shard pairs for one fp8 tensor."""
        w = self.shard(name, fp8)
        s = self.shard_scale(name, scale)
        return [(w[r], None if w[r] is None else s[r]) for r in range(self.tp)]

    # -- MoE expert partition ----------------------------------------------------------

    def owner(self, expert: int) -> int:
        per = self.cfg.n_experts // self.tp
        return min(expert // per, self.tp - 1)

    # -- reassembly (validation harness / tests) -----------------------------------------

    def full_from_shards(self, name: str, shards: list) -> np.ndarray:
        """Inverse of `shard()`. Multi-segment split_out plans scatter each
        rank's concatenated segment rows back to their global segment
        positions (simple concat would interleave segments wrongly)."""
        plan = self.plan_for(name)
        if plan.kind == _REP:
            return shards[0]
        if plan.kind == _EXPERTS:
            raise ShardError("expert tensors reassemble by expert index")
        if plan.kind == _IN:
            return np.concatenate(shards, axis=1)
        if plan.segments[0][0] is None or len(plan.segments) == 1:
            return np.concatenate(shards, axis=0)
        total = sum(s for s, _ in plan.segments)
        out = np.empty((total,) + tuple(shards[0].shape[1:]),
                       dtype=shards[0].dtype)
        pos, base = [0] * self.tp, 0
        for size, _unit in plan.segments:
            per = size // self.tp
            for r in range(self.tp):
                b, e = base + r * per, base + (r + 1) * per
                out[b:e] = shards[r][pos[r]:pos[r] + (e - b)]
                pos[r] += e - b
            base += size
        return out

    def full_scale_from_shards(self, name: str, shards: list) -> np.ndarray:
        """Inverse of `shard_scale()` for block-aligned plans (all rank scale
        rows map to exact global block ranges)."""
        plan = self.plan_for(name)
        if plan.kind == _IN:
            return np.concatenate(shards, axis=1)
        if plan.segments[0][0] is None or len(plan.segments) == 1:
            return np.concatenate(shards, axis=0)
        blk = self.block
        total = sum(s for s, _ in plan.segments)
        if total % blk:
            raise ShardError("scale reassembly needs block-aligned segments")
        out = np.empty((total // blk, shards[0].shape[1]), dtype=shards[0].dtype)
        pos, base = [0] * self.tp, 0
        for size, _unit in plan.segments:
            per = size // self.tp
            for r in range(self.tp):
                b, e = base + r * per, base + (r + 1) * per
                sb, se = b // blk, e // blk
                out[sb:se] = shards[r][pos[r]:pos[r] + (se - sb)]
                pos[r] += se - sb
            base += size
        return out


class DeepseekV4Sharding:
    """Name-driven TP2 plan for the DeepSeek-V4 checkpoint namespace (flat
    names: `embed.weight`, `layers.N.attn.*`, `mtp.*`, ...), mirroring the
    reference model.py TP semantics:

    - q/wq_b/wo_a + embed/head + indexer.wq_b/weights_proj: COLUMN parallel
      (split output rows) at head/group/vocab granularity;
    - wo_b: ROW parallel (split input cols) at group-lora granularity;
    - wkv, q_norm/kv_norm, compressor.*, hc_*, ffn.gate, shared experts,
      norms: REPLICATED (plain Linear/RMSNorm in the reference);
    - routed experts (w1/w3/w2 + their E8M0 scale): partition by expert;
    - fmt: fp8 (E4M3+E8M0, 128x128 blocks) or fp4 (packed E2M1 + per-row 32-col
      E8M0). Scales shard with weights; the 128-unit block rule of
      `validate_tensor` applies to fp8 splits (all DeepSeek split sizes are
      ert 128-aligned: 512/1024/4096/64640...).
    """

    def __init__(self, cfg, tp: int = 2, block: int = 128,
                 fp4_block_w: int = 32):
        if tp <= 1:
            raise ShardError("tp must be > 1")
        self.cfg, self.tp, self.block = cfg, tp, block
        self.fp4_block_w = fp4_block_w
        for what, n in (("n_heads", cfg.n_heads), ("n_groups", cfg.o_groups),
                        ("index_n_heads", cfg.index_n_heads),
                        ("n_experts", cfg.n_routed_experts)):
            if n % tp:
                raise ShardError(f"{what}={n} not divisible by tp={tp}")

    _RE = {
        "embed": re.compile(r"^embed\.weight$"),
        "head": re.compile(r"^head\.weight$"),
        "wq_a": re.compile(r"^layers\.\d+\.attn\.wq_a\.weight$"),
        "wq_b": re.compile(r"^layers\.\d+\.attn\.wq_b\.weight$"),
        "wo_a": re.compile(r"^layers\.\d+\.attn\.wo_a\.weight$"),
        "wo_b": re.compile(r"^layers\.\d+\.attn\.wo_b\.weight$"),
        "idx_wq_b": re.compile(
            r"^layers\.\d+\.attn\.indexer\.wq_b\.weight$"),
        "idx_wproj": re.compile(
            r"^layers\.\d+\.attn\.indexer\.weights_proj\.weight$"),
        "expert": re.compile(r"^layers\.\d+\.ffn\.experts\.(\d+)\.w[123]\.weight$"),
        "mtp_wq_b": re.compile(r"^mtp\.\d+\.attn\.wq_b\.weight$"),
        "mtp_wo_a": re.compile(r"^mtp\.\d+\.attn\.wo_a\.weight$"),
        "mtp_wo_b": re.compile(r"^mtp\.\d+\.attn\.wo_b\.weight$"),
        "markov_w1": re.compile(r"^mtp\.\d+\.markov_head\.markov_w1\.weight$"),
        "markov_w2": re.compile(r"^mtp\.\d+\.markov_head\.markov_w2\.weight$"),
    }

    def plan_for(self, name: str) -> TensorPlan:
        c = self.cfg
        R = self._RE
        if R["embed"].match(name) or R["head"].match(name):
            return TensorPlan(_OUT, ((None, 1),))       # vocab-row split
        if R["wq_a"].match(name):
            return TensorPlan(_OUT, ((c.q_lora_rank, 1),))
        if R["wq_b"].match(name):
            return TensorPlan(_OUT, ((c.n_heads * c.head_dim, c.head_dim),))
        if R["wo_a"].match(name):
            return TensorPlan(_OUT, ((c.o_groups * c.o_lora_rank,
                                      c.o_lora_rank),))
        if R["wo_b"].match(name):
            return TensorPlan(_IN, ((c.o_groups * c.o_lora_rank,
                                     c.o_lora_rank),))
        if R["idx_wq_b"].match(name):
            return TensorPlan(_OUT, ((c.index_n_heads * c.index_head_dim,
                                      c.index_head_dim),))
        if R["idx_wproj"].match(name):
            return TensorPlan(_OUT, ((c.index_n_heads, 1),))
        if R["mtp_wq_b"].match(name):
            return TensorPlan(_OUT, ((c.n_heads * c.head_dim, c.head_dim),))
        if R["mtp_wo_a"].match(name):
            return TensorPlan(_OUT, ((c.o_groups * c.o_lora_rank,
                                      c.o_lora_rank),))
        if R["mtp_wo_b"].match(name):
            return TensorPlan(_IN, ((c.o_groups * c.o_lora_rank,
                                     c.o_lora_rank),))
        if R["markov_w1"].match(name) or R["markov_w2"].match(name):
            return TensorPlan(_OUT, ((None, 1),))       # vocab-row split
        m = R["expert"].match(name)
        if m:
            return TensorPlan(_EXPERTS, expert=int(m.group(1)))
        # everything else (wkv, norms, compressor, indexer.compressor, hc_*,
        # ffn.gate, shared experts, attn_sink, ape, hc_head/markov-head proj
        # internals, main_proj/main_norm): replicate (reference plain Linear).
        return TensorPlan(_REP)

    # -- ranges / slicing / reassembly reuse the qwen4 implementation ---------

    def segment_ranges(self, size, unit):
        per = size // self.tp
        if size % self.tp:
            raise ShardError(f"segment size {size} not divisible by tp={self.tp}")
        if per % unit:
            raise ShardError(
                f"segment {size} (unit {unit}) splits into misaligned "
                f"{per}-slices for tp={self.tp}")
        return [(r * per, (r + 1) * per) for r in range(self.tp)]

    def rank_row_ranges(self, plan):
        out = []
        for r in range(self.tp):
            rr, base = [], 0
            for size, unit in plan.segments:
                if size is None:
                    raise ShardError("dynamic-size plan needs tensor shape")
                b, e = self.segment_ranges(size, unit)[r]
                rr.append((base + b, base + e))
                base += size
            out.append(rr)
        return out

    def _col_ranges(self, plan):
        size, unit = plan.segments[0]
        rng = self.segment_ranges(size, unit)
        return [[rng[r]] for r in range(self.tp)]

    def rank_ranges(self, name: str, shape: tuple):
        plan = self.plan_for(name)
        if plan.kind == _OUT:
            if plan.segments[0][0] is None:
                if shape[0] % self.tp:
                    raise ShardError(f"vocab {shape[0]} not divisible by tp")
                per = shape[0] // self.tp
                return [[(r * per, (r + 1) * per)] for r in range(self.tp)]
            total = sum(s for s, _ in plan.segments)
            if total != shape[0]:
                raise ShardError(f"plan expects out-dim {total}, got {shape[0]}")
            return self.rank_row_ranges(plan)
        if plan.kind == _IN:
            if plan.segments[0][0] != shape[1]:
                raise ShardError(f"plan expects in-dim {plan.segments[0][0]}, "
                                 f"got {shape[1]}")
            return self._col_ranges(plan)
        raise ShardError(f"no ranges for kind {plan.kind}")

    def validate_tensor(self, name: str, shape: tuple,
                        quantized: bool = True) -> None:
        plan = self.plan_for(name)
        if plan.kind in (_REP, _EXPERTS):
            return
        axis = 0 if plan.kind == _OUT else 1
        for rr in self.rank_ranges(name, shape):
            for b, e in rr:
                if e <= b:
                    raise ShardError(f"empty rank slice for {name}")
                if quantized:
                    bl0, bl1 = b // self.block, (e - 1) // self.block
                    if b % self.block and bl1 != bl0:
                        raise ShardError(
                            f"{name}: rank slice [{b},{e}) on axis {axis} cuts "
                            f"block {bl0} without staying inside it "
                            "(fp8 scale would be ambiguous)")

    def owner(self, expert: int) -> int:
        per = self.cfg.n_routed_experts // self.tp
        return min(expert // per, self.tp - 1)

    def shard(self, name: str, arr):
        plan = self.plan_for(name)
        arr = np.asarray(arr)
        if plan.kind == _REP:
            return [arr] * self.tp
        if plan.kind == _EXPERTS:
            return [arr if self.owner(plan.expert) == r else None
                    for r in range(self.tp)]
        axis = 0 if plan.kind == _OUT else 1
        return [np.concatenate(
            [(sl[b:e] if axis == 0 else sl[:, b:e]) for b, e in rr], axis=axis)
            for rr in self.rank_ranges(name, arr.shape)
            for sl in [arr]]

    def shard_scale(self, name: str, scale):
        """Rank slices of the E8M0 (128x128) scale grid, matching `shard()`.
        fp4 experts are EXPERTS-planned (no scale slicing here); their
        per-row-32-col scales stay whole with the owned expert tensors."""
        plan = self.plan_for(name)
        scale = np.asarray(scale)
        if plan.kind == _REP:
            return [scale] * self.tp
        if plan.kind == _EXPERTS:
            raise ShardError("expert tensors are plan-whole; scale slicing is "
                             "not needed (per-rank dequant uses the full pair)")
        blk = self.block
        if plan.kind == _OUT:
            return [np.concatenate([scale[b // blk: int(np.ceil(e / blk))]
                                    for b, e in rr], axis=0)
                    for rr in self.rank_row_ranges(plan)]
        return [np.concatenate([scale[:, b // blk: int(np.ceil(e / blk))]
                                for b, e in rr], axis=1)
                for rr in self._col_ranges(plan)]

    def shard_pair(self, name, fp8, scale):
        w = self.shard(name, fp8)
        s = self.shard_scale(name, scale)
        return [(w[r], None if w[r] is None else s[r]) for r in range(self.tp)]

    def full_from_shards(self, name, shards):
        plan = self.plan_for(name)
        if plan.kind == _REP:
            return shards[0]
        if plan.kind == _EXPERTS:
            raise ShardError("expert tensors reassemble by expert index")
        axis = 0 if plan.kind == _OUT else 1
        if plan.kind == _OUT and plan.segments[0][0] is None:
            axis = 0
        return np.concatenate(shards, axis=axis)

    def full_scale_from_shards(self, name, shards):
        return np.concatenate(shards, axis=0 if self.plan_for(name).kind == _OUT
                              else 1)
