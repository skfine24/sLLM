"""Tokenizer wrapper for the engine.

Loads a Qwen2Tokenizer-style checkpoint tokenizer (vocab.json, merges.txt,
tokenizer_config.json) into the self-made `serving.bpe.BPETokenizer`, and adds
the serving-layer concerns over the raw BPE:
- special-token splitting (split_special_tokens=False semantics)
- chat-template rendering (jinja2) for generate requests

Optional: tokenizer.json may be used by tests as the parity oracle.
"""

from __future__ import annotations

import json
import os

from .bpe import BPETokenizer

# Qwen2/2.5/3 byte-level GPT-2 style pre-tokenize pattern (used as a fallback
# when tokenizer_config.json does not carry `pretokenize_regex`).
DEFAULT_PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def load_merges(path: str) -> list:
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#version"):
                continue
            if not line.strip():
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                raise ValueError(f"bad merges.txt line in {path}: {line!r}")
            a, b = parts
            merges.append((a, b))
    return merges


class Tokenizer:
    def __init__(self, model_dir: str):
        vocab_path = os.path.join(model_dir, "vocab.json")
        merges_path = os.path.join(model_dir, "merges.txt")
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        for p in (vocab_path, merges_path, config_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"missing tokenizer file: {p}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        pattern = None
        cfg = None
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        pattern = cfg.get("pretokenize_regex") or DEFAULT_PRETOKENIZE_PATTERN

        self.bpe = BPETokenizer(vocab, load_merges(merges_path), pattern)
        self.config = cfg
        self._build_special()

    # -- special tokens ------------------------------------------------------

    def _build_special(self):
        # Qwen2-style: <|...|> tokens are ADDED tokens (ids >= vocab.json size),
        # carried only in added_tokens_decoder, not in vocab.json.
        decoder = self.config.get("added_tokens_decoder") or {}
        special_ids = {}
        strings = set()
        for id_str, info in decoder.items():
            if isinstance(info, dict) and info.get("content"):
                content = info["content"]
                strings.add(content)
                special_ids[content] = int(id_str)
        strings.update(self.config.get("additional_special_tokens") or [])
        for key in ("eos_token", "bos_token", "pad_token", "unk_token"):
            val = self.config.get(key)
            if val:
                strings.add(val)
        strings.discard(None)

        # Inject added tokens into the BPE vocab/id map so encode+decode treat
        # them as ordinary single ids (special-token splitting is done first);
        # only then select the special strings that actually resolve to ids.
        for content, _id in special_ids.items():
            if content in self.bpe.vocab:
                continue  # a vocab token already covers it (normal token path)
            self.bpe.vocab[content] = _id
            self.bpe.id_to_token[_id] = content
        self.special_strings = sorted(
            (s for s in strings if s in self.bpe.vocab), key=len, reverse=True
        )
        self.special_ids = {
            s: self.bpe.vocab[s] for s in self.special_strings if s in self.bpe.vocab
        }
        self._id_to_special = {v: k for k, v in self.special_ids.items()}

    @property
    def eos_token(self):
        return self.config.get("eos_token")

    def eos_id(self):
        tok = self.eos_token
        return self.special_ids.get(tok) if tok else None

    # -- encode / decode -----------------------------------------------------

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Encode text to ids with split_special_tokens=False semantics:
        special token strings are emitted as single ids and never BPE-split."""
        ids = []
        cur = []  # current non-special char buffer
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for s in self.special_strings:
                if text.startswith(s, i):
                    matched = s
                    break
            if matched is not None:
                if cur:
                    ids.extend(self.bpe.encode("".join(cur)))
                    cur = []
                ids.append(self.special_ids[matched])
                i += len(matched)
            else:
                cur.append(text[i])
                i += 1
        if cur:
            ids.extend(self.bpe.encode("".join(cur)))
        return ids

    def decode(self, ids) -> str:
        """Decode ids to text, emitting special-token id sequences as their
        literal strings (byte-level BPE would otherwise byte-decode the
        added-token chars into garbage)."""
        out = []
        cur = []
        for i in ids:
            special = self._id_to_special.get(int(i))
            if special is not None:
                if cur:
                    out.append(self.bpe.decode(cur))
                    cur = []
                out.append(special)
            else:
                cur.append(int(i))
        if cur:
            out.append(self.bpe.decode(cur))
        return "".join(out)

    def token_to_id(self, token: str) -> int | None:
        return self.bpe.vocab.get(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.bpe.id_to_token.get(int(token_id))

    # -- chat template -------------------------------------------------------

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        template = self.config.get("chat_template")
        if not template:
            raise RuntimeError("this tokenizer has no chat_template in tokenizer_config.json")
        import jinja2.sandbox

        # sandboxed: the template comes from a (semi-trusted) checkpoint dir;
        # compiled once per template string (per-request recompilation is waste)
        env = getattr(self, "_jenv", None)
        if env is None:
            env = self._jenv = jinja2.sandbox.ImmutableSandboxedEnvironment()
        cache = getattr(self, "_jtmpl", None)
        if cache is None:
            cache = self._jtmpl = {}
        render = cache.get(template)
        if render is None:
            render = cache[template] = env.from_string(template)
        return render.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.config.get("bos_token"),
            eos_token=self.config.get("eos_token"),
            add_special_tokens=True,
            echo=False,
        )
