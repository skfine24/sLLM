"""Per-rank weight view (operational track A3).

`RankWeightTable` binds the streaming loader (A2) to the TP sharding plan
(A2): rank r sees exactly the tensor slices it owns, and — critically for
the 173 GiB model — `dequant()` materializes ONLY the rank's own slice via
(scale-consistent) shard-then-dequant, never the full tensor. This is the
memory contract C2's loader implements on top of.
"""

from __future__ import annotations

import numpy as np

from loaders.fp8 import dequant_weight_blocked
from loaders.tp_shard import _EXPERTS, _REP


class RankOwnership(KeyError):
    pass


class RankWeightTable:
    def __init__(self, table, sharding, rank: int):
        if not (0 <= rank < sharding.tp):
            raise ValueError(f"rank {rank} outside tp={sharding.tp}")
        self.table = table
        self.sharding = sharding
        self.rank = rank

    # -- ownership -----------------------------------------------------------

    def owns(self, name: str) -> bool:
        plan = self.sharding.plan_for(name)
        if plan.kind == _EXPERTS:
            return self.sharding.owner(plan.expert) == self.rank
        return True

    def owned_names(self) -> list[str]:
        return [n for n in self.table.names() if self.owns(n)]

    def plan_for(self, name: str):
        return self.sharding.plan_for(name)

    # -- raw access (fp8 stays uint8) ------------------------------------------

    def _plan_slice(self, name: str, arr):
        plan = self.sharding.plan_for(name)
        if plan.kind == _REP:
            return arr
        if plan.kind == _EXPERTS:
            if not self.owns(name):
                raise RankOwnership(name)
            return arr
        return self.sharding.shard(name, arr)[self.rank]

    def get(self, name: str) -> np.ndarray:
        """Rank-owned raw slice (F8_E4M3 as uint8, BF16/natives decoded)."""
        if not self.owns(name):
            raise RankOwnership(name)
        return self._plan_slice(name, self.table.get(name))

    def dequant(self, name: str) -> np.ndarray:
        """float32 slice of the rank's shard. For SPLIT fp8 tensors the block
        scales are sharded WITH the weight and dequant runs on the rank slice
        only (the full tensor is never materialized); replicated/owned-expert
        tensors are single tensors by definition, so the plain per-tensor
        dequant already is the rank view. Scale format is auto-dispatched
        (fp32 inverse for Qwen-style, E8M0/FP4 for DeepSeek-style)."""
        if not self.owns(name):
            raise RankOwnership(name)
        plan = self.sharding.plan_for(name)
        if not self.table.is_quantized(name):
            return self._plan_slice(name, self.table.get(name))
        if plan.kind in (_REP, _EXPERTS):
            return self.table.dequant(name)
        fp8 = self.table.get(name)
        scale = self.table.scale(name)
        self.sharding.validate_tensor(name, fp8.shape, quantized=True)
        w_r, s_r = self.sharding.shard_pair(name, fp8, scale)[self.rank]
        # deepseek stores E8M0 uint8 scales (or FP4 packed weights whose expert
        # pairs are plan-whole); auto-dispatch avoids a silent fp32 misread.
        if np.asarray(scale).dtype == np.uint8:
            from loaders.fp8 import dequant_weight_auto
            blk = getattr(self.sharding, "block", 128)
            return dequant_weight_auto(w_r, s_r, block=(blk, blk))
        return dequant_weight_blocked(w_r, s_r)

    def close(self) -> None:
        self.table.index.close()
