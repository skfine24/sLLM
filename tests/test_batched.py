"""Continuous-batching integration tests: the runtime scheduler drives real
generation with several concurrent requests on the dev machine."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.dev_model import build_dev_engine  # noqa: E402
from serving.executor import BatchedInferenceEngine, generate_batch  # noqa: E402

TOK_DIR = r"C:\Users\skfin\AppData\Local\Temp\opencode\qwen27b_tok"


@unittest.skipUnless(os.path.isdir(TOK_DIR), "tokenizer files not cached")
class TestBatchedEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = build_dev_engine()
        cls.model, cls.tok = cls.eng.model, cls.eng.tokenizer

    def test_two_requests_complete_in_order(self):
        out = generate_batch(self.model, self.tok, ["hello", "world"], max_new=5, temperature=0.0)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(t, str) for t in out))

    def test_result_is_generated_tokens(self):
        batch = BatchedInferenceEngine(self.model, self.tok, max_concurrency=2)
        sid = batch.submit("hi", max_new=4, temperature=0.0)
        batch.run_all()
        info = batch._seqs[sid]
        self.assertEqual(len(info["gen"]), 4)  # no eos hit for tiny random model
        self.assertEqual(batch.result_text(sid), self.tok.decode(info["gen"]))

    def test_continuous_interleaving(self):
        # max_concurrency=2 admits both; decode actions for both sequences must
        # appear within the SAME scheduler steps (true continuous batching).
        batch = BatchedInferenceEngine(self.model, self.tok, max_concurrency=2, chunk_size=8)
        shortest_first = None
        for p, n in (("one", 3), ("two", 3)):
            sid = batch.submit(p, max_new=n, temperature=0.0)
        interleaved = False
        steps = 0
        while batch.queue_length > 0 and steps < 1000:
            res = batch.step()
            steps += 1
            decodes = [a.seq_id for a in res["actions"] if a.phase == "decode"]
            if len(set(decodes)) == 2:
                interleaved = True
                break
        self.assertTrue(interleaved, "two sequences never shared a decode step")

    def test_queueing_when_capacity_limited(self):
        # state cap 1 -> requests must serialize; both still complete in order.
        eng = BatchedInferenceEngine(self.model, self.tok, state_capacity=1, max_concurrency=1)
        for i in range(3):
            eng.submit(f"p{i}", max_new=2, temperature=0.0)
        eng.run_all()
        self.assertEqual(len(eng.results_in_order()), 3)
        # resources fully recycled
        self.assertEqual(eng.sched.coord.kv_used_total, 0)
        self.assertEqual(eng.sched.coord.state_used_total, 0)

    def test_chunked_prefill_splits_across_steps(self):
        chunk = 8
        eng = BatchedInferenceEngine(self.model, self.tok, chunk_size=chunk, max_concurrency=1)
        long_prompt = "the quick brown fox " * 30          # some hundreds of chars
        sid = eng.submit(long_prompt, max_new=0, temperature=0.0)
        from math import ceil
        exp_steps = ceil(len(self.tok.encode(long_prompt)) / chunk)
        steps = 0
        while eng.queue_length > 0 and steps < 500:
            res = eng.step()
            steps += 1
        self.assertEqual(steps, exp_steps,
                         "prefill must be split into exactly ceil(len/chunk) steps")

    def test_seeded_temperature_matches_sequential(self):
        seq = generate_batch(self.model, self.tok, ["same"], max_new=6,
                             temperature=0.9, seed=123)
        bat = generate_batch(self.model, self.tok, ["same"], max_new=6,
                             temperature=0.9, seed=123)
        self.assertEqual(seq, bat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
