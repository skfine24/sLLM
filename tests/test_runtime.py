"""Runtime bookkeeping tests: block/state allocators, hybrid coordinator,
memory planner, and the continuous-batching scheduler (CPU-only)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.blocks import BlockTable, HybridKVCoordinator, KVBlockAllocator, OutOfCapacity, StateAllocator  # noqa: E402
from runtime.memory_planner import fp8_kv_bytes_per_token, plan_block_count, qwen3_5_kv_profile  # noqa: E402
from runtime.scheduler import Action, Scheduler  # noqa: E402


def run_schedule(sched: Scheduler, max_steps: int = 1000) -> list[Action]:
    """Emulate a real executor: loop step/advance until only finished remain."""
    seen = []
    for _ in range(max_steps):
        actions = sched.step().actions
        if not actions:
            break
        for a in actions:
            sched.advance(a)
            seen.append(Action(a.seq_id, a.phase, a.from_tok, a.to_tok, a.length))
    return seen


class TestAllocators(unittest.TestCase):
    def test_kv_alloc_free_reuse(self):
        a = KVBlockAllocator(8)
        ids = a.allocate(0, 5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(a.free_count, 3)
        a.free(0, ids)
        self.assertEqual(a.free_count, 8)
        ids2 = a.allocate(1, 5)
        self.assertEqual(ids2, ids)  # reuse freed ids

    def test_kv_ownership_violation(self):
        a = KVBlockAllocator(4)
        ids = a.allocate(0, 2)
        with self.assertRaises(ValueError):
            a.free(1, ids)

    def test_kv_oom(self):
        a = KVBlockAllocator(2)
        a.allocate(0, 2)
        with self.assertRaises(OutOfCapacity):
            a.allocate(1, 1)

    def test_state_alloc_free(self):
        a = StateAllocator(2)
        s0 = a.allocate(0)
        s1 = a.allocate(1)
        self.assertEqual((s0, s1), (0, 1))
        a.free(0, s0)
        self.assertEqual(a.allocate(2), s0)
        with self.assertRaises(OutOfCapacity):
            a.allocate(3)

    def test_block_table_tokens(self):
        t = BlockTable(seq_id=0, blocks=[1, 2, 3])
        self.assertEqual(t.length_tokens(16), 48)


class TestHybridCoordinator(unittest.TestCase):
    def test_new_grow_free(self):
        c = HybridKVCoordinator(kv_capacity=100, state_capacity=4)
        t = c.new_sequence(0, tokens=30, block_size=16, use_state=True)
        self.assertEqual(len(t.blocks), 2)  # ceil(30/16)
        self.assertIsNotNone(t.state_slot)
        c.grow(0, target_tokens=70, block_size=16)
        self.assertEqual(c.kv_used(0), 5)
        c.free_sequence(0)
        self.assertNotIn(0, c.tables)
        self.assertEqual(c.kv_used_total, 0)
        self.assertEqual(c.state_used_total, 0)

    def test_blocks_for_tokens(self):
        f = HybridKVCoordinator.blocks_for_tokens
        self.assertEqual(f(0, 16), 0)
        self.assertEqual(f(16, 16), 1)
        self.assertEqual(f(17, 16), 2)


class TestMemoryPlanner(unittest.TestCase):
    def test_qwen3_5_bytes_per_token(self):
        self.assertEqual(fp8_kv_bytes_per_token(16, 4, 256, kv_bytes=1), 16 * 2 * 4 * 256)

    def test_plan_block_count(self):
        # 4 GiB, 32768 B/token, block 16 -> 0.9 budget
        from math import floor
        per = 16 * 2 * 4 * 256
        blocks = plan_block_count(4 * (1024 ** 3), per, 16, utilization=0.9)
        self.assertEqual(blocks, floor(int(4 * (1024 ** 3) * 0.9) // (per * 16)))

    def test_kv_profile_shape(self):
        p = qwen3_5_kv_profile(block_size=16, kv_avail_gib=41)
        self.assertEqual(p["bytes_per_token"], 32768)
        self.assertEqual(p["max_total_tokens"], p["num_blocks"] * 16)


class TestScheduler(unittest.TestCase):
    def make(self, kv=1000, state=4, block=16, chunk=8, conc=4):
        return Scheduler(kv_capacity=kv, state_capacity=state,
                         block_size=block, chunk_size=chunk, max_concurrency=conc)

    def test_admission_and_prefill(self):
        s = self.make()
        s.add(0, prompt_len=10, max_new=5)
        acts = run_schedule(s)
        self.assertEqual({a.seq_id for a in acts}, {0})
        # 10 prompt -> ceil(10/8)=2 prefill chunks + 5 decodes
        prefills = [a for a in acts if a.phase == "prefill"]
        decodes = [a for a in acts if a.phase == "decode"]
        self.assertEqual(sum(a.length for a in prefills), 10)
        self.assertEqual(len(decodes), 5)
        self.assertTrue(all(r.finished for r in s.done))

    def test_max_concurrency(self):
        s = self.make(state=2, conc=2)
        for i in range(3):
            s.add(i, 1, 1)
        self.assertEqual(s.pump(), [0, 1])
        self.assertEqual(s.waiting_count, 1)
        self.assertEqual(len(s.running), 2)

    def test_fifo_admission_order(self):
        s = self.make(state=2, conc=2)
        for i in range(2):
            s.add(i, 1, 1)
        self.assertEqual(s.pump(), [0, 1])

    def test_chunked_prefill_bounds(self):
        s = self.make(chunk=8)
        s.add(0, 20, 0)
        first = s.step().prefills()[0]
        self.assertEqual((first.from_tok, first.to_tok, first.length), (0, 8, 8))
        s.advance(first)
        second = s.step().prefills()[0]
        self.assertEqual((second.from_tok, second.to_tok), (8, 16))
        s.advance(second)
        third = s.step().prefills()[0]
        self.assertEqual((third.from_tok, third.to_tok), (16, 20))
        s.advance(third)
        self.assertEqual(s.step().decodes(), [])  # max_new=0 -> finished

    def test_state_capacity_blocks_admission(self):
        s = self.make(state=1, conc=4)
        s.add(0, 1, 1)
        s.add(1, 1, 1)
        self.assertEqual(s.pump(), [0])
        self.assertEqual(s.waiting_count, 1)

    def test_kv_capacity_blocks_admission(self):
        # block 16; worst = ceil((prompt+max)/16). cap enough for one req's worst.
        s = Scheduler(kv_capacity=2, state_capacity=4, block_size=16, chunk_size=8, max_concurrency=4)
        s.add(0, 20, 10)   # worst ceil(30/16)=2 -> fits exactly
        s.add(1, 20, 10)   # would need 2 more -> must stay waiting
        self.assertEqual(s.pump(), [0])
        self.assertEqual(s.waiting_count, 1)

    def test_finish_frees_for_next(self):
        s = self.make(state=1, conc=1)
        s.add(0, 1, 1)
        s.add(1, 1, 1)
        # only one fits at a time; the second must wait and run afterwards
        run_schedule(s)
        self.assertEqual(s.waiting_count, 0)
        self.assertEqual(len(s.running), 0)
        self.assertEqual([r.seq_id for r in s.done], [0, 1])
        self.assertEqual(s.coord.kv_used_total, 0)
        self.assertEqual(s.coord.state_used_total, 0)

    def test_eos_finishes_early(self):
        s = self.make()
        s.add(0, 5, 10)
        done = False
        for _ in range(100):
            acts = s.step().actions
            if not acts:
                break
            # finish at the first decode with eos=True
            for a in acts:
                s.advance(a, eos=(a.phase == "decode"))
            if any(r.finished for r in s.done):
                done = True
                break
        self.assertTrue(done)
        self.assertTrue(all(r.tokens_generated <= 1 for r in s.done))


if __name__ == "__main__":
    unittest.main(verbosity=2)
