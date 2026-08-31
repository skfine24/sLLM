"""Tokenizer tests.

Tier A (offline, synthetic): byte-level BPE algorithm, bytes_to_unicode, merge
ranking, encode/decode roundtrip.
Tier B (real files + oracle): encode/decode (and special-token behavior) of the
real Qwen3.8-27B-FP8 tokenizer, asserted against the official `tokenizers`
library. These are skipped when the cached tokenizer files are not present so
the suite stays network-free.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.bpe import BPETokenizer, bytes_to_unicode  # noqa: E402

TOK_DIR = r"C:\Users\skfin\AppData\Local\Temp\opencode\qwen27b_tok"
Q25_TOK_DIR = r"C:\Users\skfin\AppData\Local\Temp\opencode\q25c"


# ---------------------------------------------------------------------------
# Tier A: offline synthetic
# ---------------------------------------------------------------------------

class TestBytesToUnicode(unittest.TestCase):
    def test_all_bytes_unique(self):
        m = bytes_to_unicode()
        self.assertEqual(len(m), 256)
        self.assertEqual(len(set(m.values())), 256)

    def test_ascii_printable_identity(self):
        m = bytes_to_unicode()
        for code in range(ord("!"), ord("~") + 1):
            self.assertEqual(m[code], chr(code))
        # newline is remapped out of the printable range
        self.assertNotEqual(m[10], chr(10))


class TestBPESynthetic(unittest.TestCase):
    def _tokenizer(self):
        # byte-level with two merges: ("ab" -> "ab") with "a b" at rank 0 and
        # "ab c" at rank 1 so "abc" merges to one symbol.
        vocab = {
            "a": 1, "b": 2, "c": 3, "d": 4, "e": 9, "f": 10,
            "ab": 5, "abc": 6, "Ġ": 7, "Ġhi": 8,
        }
        vocab.update({f"<0x{b:02X}>": 1000 + b for b in range(256)})
        merges = [("a", "b"), ("ab", "c"), ("Ġ", "h"), ("Ġh", "i")]
        pattern = r"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"
        return BPETokenizer(vocab, merges, pattern)

    def test_bpe_merge_ranking(self):
        tok = self._tokenizer()
        # "abc" -> parts ["a","b","c"]; "a b" rank 0 first => "ab","c"; "ab c" rank1 => "abc"
        ids = tok.encode_piece("abc")
        self.assertEqual(ids, [6])

    def test_no_merge_buffer(self):
        tok = self._tokenizer()
        ids = tok.encode_piece("ab")
        self.assertEqual(ids, [5])

    def test_unknown_falls_back_to_bytes(self):
        vocab = {f"<0x{b:02X}>": 100 + b for b in range(256)}
        vocab.update({"a": 1})
        tok = BPETokenizer(vocab, [], r"[\p{L}\p{M}]+")
        ids = tok.encode_piece("z")  # 'z' not in vocab -> byte fallback
        self.assertEqual(ids, [100 + ord("z")])

    def test_pretokenize_split(self):
        tok = self._tokenizer()
        parts = tok.pretokenize("abc def")
        self.assertEqual(parts, ["abc", " def"])

    def test_roundtrip(self):
        tok = self._tokenizer()
        text = "abc abc def"
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)


# ---------------------------------------------------------------------------
# Tier B: real files, oracle-checked
# ---------------------------------------------------------------------------

def _real_files_present() -> bool:
    return all(os.path.isfile(os.path.join(TOK_DIR, f)) for f in
               ("vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json"))


@unittest.skipUnless(_real_files_present(), "tokenizer files not cached")
class TestRealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serving.tokenizer import Tokenizer
        cls.tok = Tokenizer(TOK_DIR)
        from tokenizers import Tokenizer as HFTok
        cls.oracle = HFTok.from_file(os.path.join(TOK_DIR, "tokenizer.json"))

    def test_encode_matches_oracle(self):
        cases = [
            "Hello, world!",
            "안녕하세요, DGX Spark!",
            "Qwen 3.8-27B (FP8) test 123",
            "  leading and trailing  spaces  ",
            "line1\nline2\n\nline3",
            "mixed\t tabs and   multiple   spaces",
            "naïve déjà vu 日本語 🚀",
            "a" * 50,
            "some punctuation: ;:,./?!-+=_@#$%^&*()",
        ]
        for text in cases:
            got = self.tok.encode(text)
            exp = self.oracle.encode(text).ids
            self.assertEqual(got, exp, f"mismatch for {text!r}")

    def test_special_tokens_single_ids(self):
        # split_special_tokens=False: role markers stay single ids
        text = "<|im_start|>user\nhi<|im_end|>"
        got = self.tok.encode(text)
        exp = self.oracle.encode(text).ids
        self.assertEqual(got, exp, "special-token splitting mismatch")

    def test_roundtrip_real(self):
        texts = ["Hello, world!", "한국어 테스트 123", "\n\n  spaced  out\n"]
        for t in texts:
            self.assertEqual(self.tok.decode(self.tok.encode(t)), t)

    def test_eos_id_resolves(self):
        self.assertIsNotNone(self.tok.eos_id())
        self.assertEqual(self.tok.eos_token, "<|im_end|>")


@unittest.skipUnless(_real_files_present(), "tokenizer files not cached")
class TestChatTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serving.tokenizer import Tokenizer
        cls.tok = Tokenizer(TOK_DIR)

    def test_render_simple_conversation(self):
        messages = [{"role": "user", "content": "hi"}]
        rendered = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        self.assertIn("<|im_start|>user", rendered)
        self.assertIn("<|im_start|>assistant", rendered)  # generation prompt
        self.assertIn("hi", rendered)

    def test_render_and_encode_consistency(self):
        text = self.tok.apply_chat_template([{"role": "user", "content": "안녕"}],
                                            add_generation_prompt=False)
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)


@unittest.skipUnless(_real_files_present(), "tokenizer files not cached")
class TestQwen25TokenizerOracle(unittest.TestCase):
    """Qwen2.5-Coder-0.5B tokenizer: exercises the default pretokenize_regex
    fallback (its tokenizer_config has no `pretokenize_regex`)."""

    @classmethod
    def setUpClass(cls):
        from serving.tokenizer import Tokenizer
        cls.tok = Tokenizer(Q25_TOK_DIR)
        from tokenizers import Tokenizer as HFTok
        cls.oracle = HFTok.from_file(os.path.join(Q25_TOK_DIR, "tokenizer.json"))

    def test_encode_matches_oracle(self):
        cases = [
            "Hello, world!",
            "def add(a, b):\n    return a + b",
            "python code 123 # comment",
            "x = [i**2 for i in range(10)]",
            "국제화 테스트",
            "  indented   code\n\n",
        ]
        for text in cases:
            self.assertEqual(self.tok.encode(text), self.oracle.encode(text).ids, f"mismatch {text!r}")

    def test_roundtrip_and_eos(self):
        t = "print('hi')  # test"
        self.assertEqual(self.tok.decode(self.tok.encode(t)), t)
        self.assertEqual(self.tok.eos_token, "<|endoftext|>")


if __name__ == "__main__":
    unittest.main(verbosity=2)
