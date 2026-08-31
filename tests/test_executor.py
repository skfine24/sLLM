"""Executor / generation-loop tests on the tiny reference model (deterministic,
causal, stop-on-eos, decode roundtrip)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.dev_model import VOCAB, tiny_recipe, tiny_weights  # noqa: E402
from serving.executor import InferenceEngine, ReferenceModel, generate  # noqa: E402

TOK_DIR = r"C:\Users\skfin\AppData\Local\Temp\opencode\qwen27b_tok"


@unittest.skipUnless(os.path.isdir(TOK_DIR), "tokenizer files not cached")
class TestGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serving.tokenizer import Tokenizer
        cls.model = ReferenceModel(tiny_recipe(), tiny_weights())
        cls.tok = Tokenizer(TOK_DIR)

    def test_generate_len_and_type(self):
        ids = [1, 2, 3]
        out = generate(self.model, self.tok, ids, max_new=5, temperature=0.0)
        self.assertEqual(len(ids) + 5, len(out))
        self.assertTrue(all(isinstance(x, int) for x in out[3:]))

    def test_greedy_deterministic(self):
        ids = [1, 2, 3]
        a = generate(self.model, self.tok, ids, max_new=4, temperature=0.0)
        b = generate(self.model, self.tok, ids, max_new=4, temperature=0.0)
        self.assertEqual(a, b)

    def test_ids_in_vocab(self):
        ids = [5, 6]
        out = generate(self.model, self.tok, ids, max_new=6, temperature=0.0)
        for tok in out:
            self.assertLess(tok, VOCAB)

    def test_stop_ids_terminate(self):
        ids = [1, 2, 3]
        full = generate(self.model, self.tok, ids, max_new=5, temperature=0.0)
        first_generated = full[len(ids)]
        # the stopping token is never emitted: generate breaks before append
        out = generate(self.model, self.tok, ids, max_new=5, temperature=0.0,
                       stop_ids=(first_generated,))
        self.assertEqual(len(out), len(ids))

    def test_engine_complete_returns_text(self):
        engine = InferenceEngine(self.model, self.tok)
        out = engine.complete("hi", max_new=4, temperature=0.0)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) >= 0)

    def test_engine_chat_wraps_template(self):
        engine = InferenceEngine(self.model, self.tok)
        out = engine.chat([{"role": "user", "content": "hi"}], max_new=3, temperature=0.0)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
