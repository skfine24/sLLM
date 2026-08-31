"""Collectives interface + single-process simulation backend (A3).

C2 runs one process per rank and implements `Collectives` over NCCL (the
CUDA extension can dlopen libnccl directly, keeping the engine torch-free;
the pair-link bandwidth is measured in B2 before committing). Locally,
`SimCollectives` executes the SAME reductions a real collective performs —
over explicitly registered per-rank partials — so rank-logic bugs (missing
partials, wrong rank count, wrong gather order) are caught on the dev
machine before any cluster time is spent.
"""

from __future__ import annotations

import numpy as np


class CollectivesError(RuntimeError):
    pass


class Collectives:
    """Real cross-rank transport interface implemented by NCCL in C2."""

    rank: int = 0
    world_size: int = 1

    def all_reduce(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def all_gather(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError


class NcclCollectives(Collectives):
    """Placeholder bound at C2 (cluster, exclusive window)."""

    def __init__(self, rank: int, world_size: int, pair_ips: tuple):
        self.rank, self.world_size = rank, world_size
        self.pair_ips = pair_ips

    def _no(self):
        raise CollectivesError(
            "NCCL backend lands in milestone C2 on the cluster (needs the "
            "libnccl transport + the measured pair link); use SimCollectives "
            "for local development")

    def all_reduce(self, x):
        self._no()

    def all_gather(self, x, axis=0):
        self._no()

    def barrier(self):
        self._no()


def all_reduce_partials(partials) -> np.ndarray:
    """Reference semantic of all_reduce(SUM) for tests/simulation."""
    if not partials:
        raise CollectivesError("no partials to reduce")
    out = np.array(partials[0], dtype=np.float64)
    for p in partials[1:]:
        p = np.asarray(p)
        if p.shape != out.shape:
            raise CollectivesError(
                f"partial shape {p.shape} != {out.shape} in all-reduce")
        out = out + p
    return out.astype(np.asarray(partials[0]).dtype, copy=False)


def all_gather_parts(parts, axis: int = 0) -> np.ndarray:
    """Reference semantic of all_gather(CONCAT in rank order)."""
    if not parts:
        raise CollectivesError("no parts to gather")
    return np.concatenate([np.asarray(p) for p in parts], axis=axis)


class SimCollectives:
    """Single-process simulated world keyed by a tag.

    Each rank calls `contribute(tag, rank, partial)`; `all_reduce(tag)` sums
    exactly like an NCCL SUM all-reduce and errors when any rank's partial is
    missing or shapes disagree; `all_gather(tag)` concatenates in rank order.
    """

    def __init__(self, world_size: int):
        if world_size < 1:
            raise CollectivesError("world_size must be >= 1")
        self.world_size = world_size
        self._tags: dict[str, dict[int, np.ndarray]] = {}

    def contribute(self, tag: str, rank: int, arr: np.ndarray) -> None:
        if not (0 <= rank < self.world_size):
            raise CollectivesError(f"rank {rank} outside world "
                                   f"{self.world_size}")
        slot = self._tags.setdefault(tag, {})
        if rank in slot:
            raise CollectivesError(f"tag {tag!r}: rank {rank} contributed twice")
        slot[rank] = np.asarray(arr)

    def _parts(self, tag: str) -> list:
        slot = self._tags.get(tag)
        if slot is None:
            raise CollectivesError(f"unknown tag {tag!r}")
        missing = [r for r in range(self.world_size) if r not in slot]
        if missing:
            raise CollectivesError(
                f"tag {tag!r}: missing contributions from ranks {missing}")
        return [slot[r] for r in range(self.world_size)]

    def all_reduce(self, tag: str) -> np.ndarray:
        return all_reduce_partials(self._parts(tag))

    def all_gather(self, tag: str, axis: int = 0) -> np.ndarray:
        return all_gather_parts(self._parts(tag), axis=axis)

    def reset(self, tag: str | None = None) -> None:
        if tag is None:
            self._tags.clear()
        else:
            self._tags.pop(tag, None)
