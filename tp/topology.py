"""Cluster topology for tensor parallel runs (operational track A3).

Facts from docs/design/08-cluster-layout.md: two GB10 nodes (head n1 /
worker n2) reachable from the dev machine over 192.168.0.x, talking to each
other over the internal 10.100.25.x pair link. TP2 world = head rank0 +
worker rank1; the HTTP server binds on the head, the worker is compute-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from env_config import get as _env


class TopologyError(ValueError):
    pass


# Defaults from docs/design/08 sec.1 (verified 2026-08-29); overridable per
# deployment via config.env / environment variables (env_config precedence).
HEAD_EXTERNAL = _env("SLLM_HEAD_IP", "192.168.0.250")
WORKER_EXTERNAL = _env("SLLM_WORKER_IP", "192.168.0.231")
HEAD_PAIR_IP = _env("SLLM_HEAD_PAIR_IP", "10.100.25.1")
WORKER_PAIR_IP = _env("SLLM_WORKER_PAIR_IP", "10.100.25.2")


@dataclass(frozen=True)
class RankSpec:
    rank: int
    host: str          # ssh/coordination address
    pair_ip: str       # NCCL address on the internal pair link
    device: int = 0    # GPU index on the node (GB10: single GPU)

    @property
    def is_head(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class ClusterTopology:
    ranks: tuple

    @classmethod
    def dgx_spark_pair(cls) -> "ClusterTopology":
        return cls(ranks=(
            RankSpec(0, HEAD_EXTERNAL, HEAD_PAIR_IP),
            RankSpec(1, WORKER_EXTERNAL, WORKER_PAIR_IP),
        ))

    @classmethod
    def from_recipe(cls, recipe) -> "ClusterTopology":
        topo = cls.dgx_spark_pair()
        topo.validate(recipe)
        return topo

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    def validate(self, recipe) -> None:
        tp = recipe.tp.size
        if tp != self.world_size:
            raise TopologyError(
                f"recipe tp.size={tp} != topology world_size={self.world_size} "
                "(full-model qwen4_exp/deepseek_v4 need the 2-node pair)")
        ranks = sorted(r.rank for r in self.ranks)
        if ranks != list(range(self.world_size)):
            raise TopologyError(f"ranks must be 0..world_size-1, got {ranks}")
