"""Dev CLI: end-to-end prompt->completion on the tiny reference model.

Usage:
    python -m serving.cli --prompt "Hello" --max-new 12
    python -m serving.cli --chat "Hi there" --max-new 12 --temperature 0.8
    python -m serving.cli --batch "one two" "three four" --max-new 8
"""

from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(description="dev serving CLI (tiny model)")
    ap.add_argument("--prompt", type=str, default=None, help="raw completion prompt")
    ap.add_argument("--chat", type=str, default=None, help="user message for chat template")
    ap.add_argument("--batch", nargs="+", default=None, help="prompts served with continuous batching")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--qwen4", action="store_true",
                    help="serve the tiny qwen4_exp fixture (HC+GDN+QSA+MoE)")
    args = ap.parse_args(argv)

    if args.qwen4:
        from .dev_model import build_dev_qwen4_exp_engine as _build
    else:
        from .dev_model import build_dev_engine as _build

    engine = _build()
    if args.batch:
        from .executor import generate_batch
        outs = generate_batch(
            engine.model, engine.tokenizer, args.batch, max_new=args.max_new,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
        )
        for prompt, out in zip(args.batch, outs):
            print(f"[{prompt!r}] -> {out!r}")
    elif args.chat is not None:
        out = engine.chat(
            [{"role": "user", "content": args.chat}],
            max_new=args.max_new, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, seed=args.seed,
        )
        print(f"assistant: {out}")
    elif args.prompt is not None:
        out = engine.complete(
            args.prompt, max_new=args.max_new, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, seed=args.seed,
        )
        print(out)
    else:
        ap.error("provide --prompt, --chat, or --batch")


if __name__ == "__main__":
    main()
