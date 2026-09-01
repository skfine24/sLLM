"""KV / recurrent-state memory placement: device (all-on-GPU) vs host (KV in RAM).

The recipe exposes `memory.kv_placement` — this module turns it into a sized
plan (`KVMemoryPlan`) and a concrete backing store (`KVBackend`) with two
implementations:

* `HostKVBackend`   — numpy arrays in host RAM. Used by the CPU/numpy path
  always ("CPU mode uses only RAM"); also the safe fallback when a GPU
  placement is requested but no device is available. Capacity over-run raises
  `OutOfCapacity` (recoverable, process-level, optionally swap-backed).
* `DeviceKVBackend` — KV stored on the GPU (device buffers via
  `kernels._sllm_cuda`). Mirrors the conventional all-on-GPU layout; the
  planned budget must never be exceeded (admission rejects overflow) because
  on GB10 over-subscribing device memory can hang the node until power off
  (docs/design/03).

The GPU decode loop that consumes `gather()` slices is the GPU-kernel phase;
this module provides the sizing, selection, and (recoverable) allocation
semantics.
"""

from __future__ import annotations

import warnings

import numpy as np

from .blocks import OutOfCapacity
from .memory_planner import kv_bytes_per_token

# Default KB budgets when a recipe does not pin them.
DEVICE_DEFAULT_BYTES = int(4 * 1024 ** 3)   # 4 GiB (override via memory.kv_device_bytes)
HOST_DEFAULT_BYTES = int(8 * 1024 ** 3)     # 8 GiB (override via memory.kv_host_bytes)


class KVMemoryPlan:
    """Sized KV budget for a placement mode."""

    def __init__(self, placement: str, utilization: float, bytes_per_token: int,
                 budget_bytes: int, block_size: int = 16):
        if placement not in ("device", "host"):
            raise ValueError(f"unknown kv_placement {placement!r}")
        self.placement = placement
        self.utilization = float(utilization)
        self.bytes_per_token = int(bytes_per_token)
        self.budget_bytes = int(budget_bytes)
        self.block_size = int(block_size)
        per_block = self.bytes_per_token * self.block_size
        budget = int(self.budget_bytes * self.utilization)
        self.num_blocks = budget // per_block if per_block else 0
        self.max_tokens = self.num_blocks * self.block_size

    def describe(self) -> dict:
        return {
            "placement": self.placement,
            "utilization": self.utilization,
            "bytes_per_token": self.bytes_per_token,
            "budget_bytes": self.budget_bytes,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_recipe(cls, recipe, block_size: int = 16,
                    kv_avail_bytes: int | None = None) -> "KVMemoryPlan":
        mem = recipe.memory
        # The `device` backend stores fp32 buffers (kernels._sllm_cuda.to_device
        # casts to fp32), so its plan must charge 4 B/element or the GB10 KV
        # budget is silently 2x over-subscribed (the exact failure mode this
        # module is meant to prevent). Host RAM is abundant: keep the nominal
        # 2 B/element (BF16) figure there.
        kv_bytes = 4 if mem.kv_placement == "device" else 2
        bpt = kv_bytes_per_token(recipe, kv_bytes=kv_bytes)
        if kv_avail_bytes is not None:
            budget = int(kv_avail_bytes)
        elif mem.kv_placement == "host":
            budget = mem.kv_host_bytes or HOST_DEFAULT_BYTES
        else:
            budget = mem.kv_device_bytes or DEVICE_DEFAULT_BYTES
        return cls(mem.kv_placement, mem.kv_utilization, bpt, budget, block_size)


class KVBackend:
    """Backing store for per-sequence attention K/V (and recurrent state)."""

    supports_gpu: bool = False
    name: str = "kv"

    def __init__(self, plan: KVMemoryPlan):
        self.plan = plan
        self._blocks: dict[int, int] = {}   # seq_id -> reserved blocks
        self._free_blocks = plan.num_blocks

    # -- capacity -----------------------------------------------------------

    def reserve(self, seq_id: int, num_blocks: int) -> None:
        if seq_id in self._blocks:
            raise ValueError(f"seq {seq_id} already reserved")
        if num_blocks > self._free_blocks:
            raise OutOfCapacity(
                f"need {num_blocks} KV blocks, {self._free_blocks} free "
                f"(placement={self.plan.placement}, cap {self.plan.num_blocks})")
        self._blocks[seq_id] = num_blocks
        self._free_blocks -= num_blocks

    def release(self, seq_id: int) -> None:
        n = self._blocks.pop(seq_id, 0)
        self._free_blocks += n

    @property
    def free_blocks(self) -> int:
        return self._free_blocks

    # -- storage (subclass required) ----------------------------------------

    def store(self, seq_id: int, layer: int, k, v) -> None:
        raise NotImplementedError

    def gather(self, seq_id: int, layer: int):
        raise NotImplementedError

    def free(self, seq_id: int) -> None:
        raise NotImplementedError


class HostKVBackend(KVBackend):
    """KV stored as numpy arrays in host RAM (used by CPU mode and as the
    safe fallback). Over-capacity raises OutOfCapacity (recoverable)."""

    supports_gpu: bool = False
    name: str = "host"

    def __init__(self, plan: KVMemoryPlan):
        super().__init__(plan)
        self._store: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    def store(self, seq_id: int, layer: int, k, v) -> None:
        self._store.setdefault(seq_id, {})[layer] = (
            np.asarray(k, dtype=np.float32), np.asarray(v, dtype=np.float32))

    def gather(self, seq_id: int, layer: int):
        try:
            return self._store[seq_id][layer]
        except KeyError as exc:
            raise KeyError(f"no KV for seq {seq_id} layer {layer}") from exc

    def free(self, seq_id: int) -> None:
        self._store.pop(seq_id, None)
        self.release(seq_id)


class DeviceKVBackend(KVBackend):
    """KV stored in GPU device buffers (conventional all-on-GPU layout).

    Construction requires a CUDA device and a built `sllm_gpu.so`; otherwise
    `build_kv_backend` falls back to the host backend with a warning.
    """

    supports_gpu: bool = True
    name: str = "device"

    def __init__(self, plan: KVMemoryPlan):
        from kernels import _sllm_cuda as ck  # lazy; needs built .so
        if ck.device_count() < 1:
            raise RuntimeError("no CUDA device for device KV placement")
        free = ck.mem_free_bytes()
        # Never pick "device" KV unless the device can actually hold the
        # planned budget: on a busy GPU (e.g. shared with vLLM) this is the
        # over-subscription case that can hang the GB10 node. Fall back to
        # host RAM instead (recoverable), per the placement option's contract.
        if free >= 0 and free < plan.budget_bytes:
            raise RuntimeError(
                f"insufficient free device memory for device KV placement: "
                f"{free} B free < {plan.budget_bytes} B planned budget")
        self._ck = ck
        self._bufs: dict[int, dict[int, tuple]] = {}  # seq_id -> layer -> (k_buf, v_buf)
        super().__init__(plan)

    def store(self, seq_id: int, layer: int, k, v) -> None:
        ck = self._ck
        kb = ck.to_device(np.ascontiguousarray(k, dtype=np.float32))
        vb = ck.to_device(np.ascontiguousarray(v, dtype=np.float32))
        layers = self._bufs.setdefault(seq_id, {})
        stale = layers.get(layer)
        if stale is not None:
            # a re-prefill overwrites: free the old device buffers first,
            # otherwise every retry leaks 2 device buffers (OOM on GB10).
            stale[0].free()
            stale[1].free()
        layers[layer] = (kb, vb)

    def gather(self, seq_id: int, layer: int):
        try:
            kb, vb = self._bufs[seq_id][layer]
        except KeyError as exc:
            raise KeyError(f"no device KV for seq {seq_id} layer {layer}") from exc
        return kb.copy_host(), vb.copy_host()

    def free(self, seq_id: int) -> None:
        for kb, vb in self._bufs.pop(seq_id, {}).values():
            kb.free()
            vb.free()
        self.release(seq_id)


def build_kv_backend(plan: KVMemoryPlan) -> KVBackend:
    """Instantiate the backend for a plan; device placement degrades to the
    host backend when the GPU/.so is unavailable (graceful, CPU-mode OK)."""
    if plan.placement == "host":
        return HostKVBackend(plan)
    try:
        return DeviceKVBackend(plan)
    except Exception as exc:  # noqa: BLE001 - degrade, never hard-fail serving
        warnings.warn(
            f"device KV placement requested but unavailable ({exc}); "
            f"falling back to host RAM KV", RuntimeWarning)
        return HostKVBackend(plan)
