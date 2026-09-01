"""Paged KV block + recurrent-state allocation (hybrid bookkeeping).

Pure-python, GPU-free: these allocators own the bookkeeping that the design
doc carries under "block allocator / hybrid state coordinator". They track
*who* owns *which* slots; the device kernels later read/write the backing
tensors addressed by these ids.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


class OutOfCapacity(Exception):
    pass


@dataclass
class BlockTable:
    """A sequence's list of KV block ids + optional recurrent-state slot."""
    seq_id: int
    blocks: list[int] = field(default_factory=list)
    state_slot: int | None = None

    def length_tokens(self, block_size: int) -> int:
        """Allocated CAPACITY in tokens (not the sequence's live length)."""
        return len(self.blocks) * block_size


class KVBlockAllocator:
    """Paged KV block allocator with per-sequence ownership tracking.

    Free ids live in a min-heap so allocation is O(count log F) and always
    reuses the lowest free ids (no O(F log F) sort per call)."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError("capacity must be >= 0")
        self._free: list[int] = list(range(self.capacity))  # valid min-heap
        self._owner: dict[int, int] = {}  # block_id -> seq_id

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def used_count(self) -> int:
        return self.capacity - len(self._free)

    def allocate(self, seq_id: int, count: int) -> list[int]:
        if count > len(self._free):
            raise OutOfCapacity(
                f"need {count} blocks, {len(self._free)} free (cap {self.capacity})"
            )
        ids = [heapq.heappop(self._free) for _ in range(count)]
        for b in ids:
            self._owner[b] = seq_id
        return ids

    def free(self, seq_id: int, ids: list[int]) -> None:
        # validate EVERYTHING first, then mutate: a bad id must never leave
        # the earlier ids of the batch already moved back to the free set.
        seen: set[int] = set()
        for b in ids:
            if b in seen:
                raise ValueError(f"block {b} listed twice")
            seen.add(b)
            if self._owner.get(b) != seq_id:
                raise ValueError(f"block {b} not owned by {seq_id}")
        for b in ids:
            del self._owner[b]
            heapq.heappush(self._free, b)

    def owned(self, seq_id: int) -> list[int]:
        return [b for b, s in self._owner.items() if s == seq_id]


class StateAllocator:
    """Per-sequence recurrent-state slot allocator (one slot per sequence;
    the per-layer state layout inside a slot is a later kernel concern)."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError("capacity must be >= 0")
        self._free: list[int] = list(range(self.capacity))  # valid min-heap
        self._owner: dict[int, int] = {}

    @property
    def free_count(self) -> int:
        return len(self._free)

    def allocate(self, seq_id: int) -> int:
        if not self._free:
            raise OutOfCapacity("no recurrent-state slots left")
        slot = heapq.heappop(self._free)
        self._owner[slot] = seq_id
        return slot

    def free(self, seq_id: int, slot: int | None) -> None:
        if slot is None:
            return
        if self._owner.get(slot) != seq_id:
            raise ValueError(f"state slot {slot} not owned by {seq_id}")
        del self._owner[slot]
        heapq.heappush(self._free, slot)

    def owned(self, seq_id: int) -> list[int]:
        return [s for s, o in self._owner.items() if o == seq_id]


class HybridKVCoordinator:
    """Owns both allocators and pairs them per sequence (hybrid attention +
    recurrent-state memory, mirroring vLLM's HybridKVCacheCoordinator role).

    The block IDs / state slot produced here feed the kernel layer's layout
    function; this object has no tensor storage.
    """

    def __init__(self, kv_capacity: int, state_capacity: int):
        self.kv = KVBlockAllocator(kv_capacity)
        self.state = StateAllocator(state_capacity)
        self.tables: dict[int, BlockTable] = {}

    @staticmethod
    def blocks_for_tokens(tokens: int, block_size: int) -> int:
        return (tokens + block_size - 1) // block_size if tokens > 0 else 0

    def new_sequence(self, seq_id: int, tokens: int, block_size: int,
                     use_state: bool = True) -> BlockTable:
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already registered")
        kv_blocks = self.blocks_for_tokens(tokens, block_size)
        table = BlockTable(seq_id=seq_id)
        try:
            if kv_blocks:
                table.blocks = self.kv.allocate(seq_id, kv_blocks)
            if use_state:
                table.state_slot = self.state.allocate(seq_id)
        except BaseException:
            # a failed state-slot allocation must not strand the KV blocks
            if table.blocks:
                self.kv.free(seq_id, table.blocks)
            raise
        self.tables[seq_id] = table
        return table

    def grow(self, seq_id: int, target_tokens: int, block_size: int) -> None:
        """Grow the sequence's KV table to cover `target_tokens` tokens."""
        if seq_id not in self.tables:
            raise KeyError(f"seq {seq_id} not registered")
        table = self.tables[seq_id]
        want = self.blocks_for_tokens(target_tokens, block_size)
        have = len(table.blocks)
        if want > have:
            table.blocks.extend(self.kv.allocate(seq_id, want - have))

    def free_sequence(self, seq_id: int) -> None:
        table = self.tables.get(seq_id)
        if table is None:
            return
        # free the resources BEFORE dropping the metadata: an ownership
        # error must not destroy the bookkeeping needed to recover.
        self.kv.free(seq_id, table.blocks)
        self.state.free(seq_id, table.state_slot)
        self.tables.pop(seq_id, None)

    def kv_used(self, seq_id: int) -> int:
        return len(self.tables[seq_id].blocks)

    @property
    def kv_used_total(self) -> int:
        return self.kv.used_count

    @property
    def state_used_total(self) -> int:
        return sum(1 for s in self.tables.values() if s.state_slot is not None)
