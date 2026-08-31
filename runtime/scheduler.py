"""Continuous-batching scheduler (bookkeeping only, GPU-free).

Implements the design-doc runtime rules on top of `runtime.blocks`:

- admission control (max concurrency, KV blocks, recurrent-state slots)
- chunked prefill (a long prompt is split into chunk_size pieces)
- prefill -> decode phase transition per sequence
- finish -> resource release -> new admission (continuous batching)

The scheduler does NOT run any math; it returns a `Schedule` of actions that a
real executor would carry out (forward etc.). Tests emulate execution by
calling `advance` with the consumed token counts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .blocks import BlockTable, HybridKVCoordinator, OutOfCapacity


@dataclass
class Request:
    seq_id: int
    prompt_len: int
    max_new: int
    position: int = 0            # prompt tokens already prefilled
    tokens_generated: int = 0
    finished: bool = False


@dataclass
class Action:
    seq_id: int
    phase: str                   # "prefill" | "decode"
    from_tok: int = 0            # prefill: prompt start index
    to_tok: int = 0              # prefill: prompt end index (exclusive)
    length: int = 0


@dataclass
class Schedule:
    actions: list[Action] = field(default_factory=list)

    def prefills(self) -> list[Action]:
        return [a for a in self.actions if a.phase == "prefill"]

    def decodes(self) -> list[Action]:
        return [a for a in self.actions if a.phase == "decode"]

    @property
    def num_sequences(self) -> int:
        return len(self.actions)

    def tokens_this_step(self) -> int:
        return sum(a.length for a in self.actions)


class Scheduler:
    def __init__(
        self,
        kv_capacity: int,
        state_capacity: int,
        block_size: int = 16,
        chunk_size: int = 32,
        max_concurrency: int = 4,
    ):
        if chunk_size < 1 or block_size < 1 or max_concurrency < 1:
            raise ValueError("sizes must be >= 1")
        self.coord = HybridKVCoordinator(kv_capacity, state_capacity)
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.max_concurrency = max_concurrency
        self._waiting: deque[Request] = deque()
        self._running: dict[int, Request] = {}
        self.done: list[Request] = []

    # -- public api ---------------------------------------------------------

    def add(self, seq_id: int, prompt_len: int, max_new: int) -> None:
        self._waiting.append(Request(seq_id=seq_id, prompt_len=prompt_len, max_new=max_new))

    def table(self, seq_id: int) -> BlockTable:
        return self.coord.tables[seq_id]

    @property
    def running(self) -> list[Request]:
        return list(self._running.values())

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    def pump(self) -> list[int]:
        """Admit waiting requests while resources allow. Returns admitted ids."""
        admitted = []
        while self._waiting and self._accepts(self._waiting[0]):
            req = self._waiting.popleft()
            self._admit(req)
            admitted.append(req.seq_id)
        return admitted

    def schedule(self) -> Schedule:
        sched = Schedule()
        for req in self._running.values():
            if req.finished:
                continue
            if req.position < req.prompt_len:
                n = min(self.chunk_size, req.prompt_len - req.position)
                sched.actions.append(Action(
                    req.seq_id, "prefill", from_tok=req.position, to_tok=req.position + n, length=n
                ))
            else:
                sched.actions.append(Action(req.seq_id, "decode", length=1))
        return sched

    def step(self) -> Schedule:
        """pump() then schedule(): the standard per-iteration entry point."""
        self.pump()
        return self.schedule()

    def advance(self, action: Action, eos: bool = False) -> None:
        """Move bookkeeping forward after the executor ran `action`."""
        req = self._running[action.seq_id]
        if action.phase == "prefill":
            req.position = min(req.prompt_len, req.position + action.length)
        else:
            req.tokens_generated += 1
            req.position = req.prompt_len
        if eos or (req.position >= req.prompt_len and req.tokens_generated >= req.max_new):
            self._finish(req)

    def _admit(self, req: Request) -> None:
        worst = self.coord.blocks_for_tokens(req.prompt_len + req.max_new, self.block_size)
        self.coord.new_sequence(req.seq_id, worst * self.block_size, self.block_size)
        self._running[req.seq_id] = req

    def _accepts(self, req: Request) -> bool:
        if len(self._running) >= self.max_concurrency:
            return False
        if not self.coord.state.free_count:
            return False
        worst = self.coord.blocks_for_tokens(req.prompt_len + req.max_new, self.block_size)
        return self.coord.kv.free_count >= worst

    def _finish(self, req: Request) -> None:
        if req.finished:
            return
        req.finished = True
        self.coord.free_sequence(req.seq_id)
        self._running.pop(req.seq_id, None)
        self.done.append(req)
