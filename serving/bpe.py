"""Self-made byte-level BPE (Qwen2Tokenizer-style), numpy/stdlib-free beyond
the `regex` module (used only as the Unicode-aware regex engine for the stored
pre-tokenization pattern; the pattern itself comes from the checkpoint's
tokenizer_config.json).

The merge algorithm is implemented here (no HF library involved): byte-to-
unicode mapping, pre-tokenization, rank-based greedy merging, encode/decode
with byte-level losslessness. Numeric parity against the official `tokenizers`
library is asserted in the test suite.
"""

from __future__ import annotations

import regex as _regex

# Official "bytes_to_unicode" mapping used by GPT-2 / Qwen byte-level BPEs.
def bytes_to_unicode() -> dict:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00A1"), ord("\u00AC") + 1))
        + list(range(ord("\u00AE"), ord("\u00FF") + 1))
    )
    cs = [chr(n) for n in bs]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(chr(2 ** 8 + n))
            n += 1
    return dict(zip(bs, cs))


class BPETokenizer:
    """Byte-level BPE over vocab.json + merges.txt + a pre-tokenize pattern."""

    def __init__(self, vocab: dict, merges: list, pattern: str):
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError("vocab must be a non-empty dict[int-able -> id]")
        self.vocab = {str(k): int(v) for k, v in vocab.items()}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.rank = {}
        for i, pair in enumerate(merges):
            a, b = pair
            self.rank[(str(a), str(b))] = i
        self.b2u = bytes_to_unicode()
        self.u2b = {c: b for b, c in self.b2u.items()}
        try:
            self.pattern = _regex.compile(pattern)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid pre-tokenize pattern: {exc}") from exc

    # -- pre-tokenization ---------------------------------------------------

    def pretokenize(self, text: str) -> list[str]:
        return [m.group(0) for m in self.pattern.finditer(text)]

    # -- bpe on one pre-token ----------------------------------------------

    def _bpe_symbols(self, tok: str) -> list[str]:
        data = tok.encode("utf-8")
        sym = "".join(self.b2u[b] for b in data)
        words = list(sym)
        while len(words) > 1:
            best_rank = None
            best_i = None
            for i in range(len(words) - 1):
                r = self.rank.get((words[i], words[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            words[best_i] += words[best_i + 1]
            del words[best_i + 1]
        return words

    def _bpe_to_ids(self, tok: str) -> list[int]:
        ids = []
        for sym in self._bpe_symbols(tok):
            if sym in self.vocab:
                ids.append(self.vocab[sym])
                continue
            # fallback to per-byte tokens (rare; Qwen vocab covers the JPEG-few
            # symbols after byte mapping, but keep the GPT-2 style fallback)
            raw = bytes(self.u2b[c] for c in sym)
            for byte in raw:
                ids.append(self.vocab[f"<0x{byte:02X}>"])
        return ids

    # -- public api ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        out = []
        for piece in self.pretokenize(text):
            out.extend(self._bpe_to_ids(piece))
        return out

    def decode(self, ids) -> str:
        buf = bytearray()
        for i in ids:
            token = self.id_to_token.get(int(i))
            if token is None:
                continue
            for ch in token:
                b = self.u2b.get(ch)
                if b is not None:
                    buf.append(b)
        return bytes(buf).decode("utf-8", errors="replace")

    def encode_piece(self, text: str) -> list[int]:
        return self._bpe_to_ids(text)
