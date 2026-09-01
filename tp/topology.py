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


def _names() -> dict:
    """Resolve cluster names lazily (env/config.env can change after import;
    an import-time read would bake stale values and mask configuration
    problems). Defaults from docs/design/08 sec.1 (verified 2026-08-29)."""
    return {
        "head": _env("SLLM_HEAD_IP", "192.168.0.250"),
        "worker": _env("SLLM_WORKER_IP", "192.168.0.231"),
        "head_pair": _env("SLLM_HEAD_PAIR_IP", "10.100.25.1"),
        "worker_pair": _env("SLLM_WORKER_PAIR_IP", "10.100.25.2"),
    }


# Back-compat module constants (resolved once; dgx_spark_pair() re-resolves
# lazily so a config.env change after import is honoured).
HEAD_EXTERNAL = _names()["head"]
WORKER_EXTERNAL = _names()["worker"]
HEAD_PAIR_IP = _names()["head_pair"]
WORKER_PAIR_IP = _names()["worker_pair"]


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
        n = _names()
        return cls(ranks=(
            RankSpec(0, n["head"], n["head_pair"]),
            RankSpec(1, n["worker"], n["worker_pair"]),
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
