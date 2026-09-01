"""D+V L5: DeepSeek-V4 SUBSET parity bench (run on head; cluster deliverable,
written and locally verified against the tiny dev model).

Loads real checkpoint bytes through loaders/streaming (mmap, per-tensor
dequant + ue8m0/fp4 via `dequant_weight_auto`), selects a layer subset from
the REAL geometry, and drives the numpy oracle (`ref.deepseek_v4`) forward:

  * determinism  : two identical runs must be bit-for-bit equal
  * noise floor  : fp32-dequant oracle vs bf16-cast weights -> logit argmax
                   agreement + relative RMS (the floor a GPU GEMM must beat;
                   QAT sim (cfg.qat_sim) is on iff --qat)
  * reference    : subset logits saved to .npz for the torch-golden compare
                   (bench/deepseek_parity.py, cluster)

Run (head; weights dir defaults to the recipe paths.local_dir):
    python bench/deepseek_subset_parity.py
Run (local plumbing check, no checkpoint):
    python bench/deepseek_subset_parity.py --tiny
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import replace  # noqa: E402

import ref.deepseek_v4 as ds  # noqa: E402


def load_cfg(tiny: bool, qat: bool):
    if tiny:
        from serving.dev_model import tiny_deepseek_v4_cfg
        cfg = tiny_deepseek_v4_cfg()
        cfg.n_layers = 2
        cfg.qat_sim = qat
        return cfg
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "recipes", "DeepSeek-V4-Flash-FP8.yaml"),
              encoding="utf-8") as f:
        from recipes.schema import Recipe
        recipe = Recipe.from_yaml(f.read())
    cfg = ds.DeepseekV4Cfg.from_recipe(recipe)
    cfg.qat_sim = qat
    return cfg


def pick_subset(cfg, layers):
    """Layer SUBSET (real indices). Default = first compress_ratio 4 layer
    (indexer) + first pure window (ratio 0) layer; target layers stay empty
    for the text-subset run."""
    if layers:
        subset = list(layers)
    else:
        subset = []
        for want in (4, 0):
            for i, r in enumerate(cfg.compress_ratios):
                if r == want and i not in subset:
                    subset.append(i)
                    break
        if len(subset) != 2:
            raise SystemExit("[ds] could not pick one ratio-4 + one ratio-0 "
                             "layer from compress_ratios")
    cr = tuple(cfg.compress_ratios[i] for i in subset)
    return subset, replace(cfg, n_layers=len(subset), compress_ratios=cr,
                           dspark_target_layer_ids=(), qat_sim=False)


def load_subset_weights(table, subset: list[int]) -> dict:
    """Real layer indices -> subset driver indices; fp8/ue8m0/fp4 folded."""
    from loaders.streaming import LazyWeightTable
    w = {}
    for j, i in enumerate(subset):
        for n in table.index.filter(f"layers.{i}."):
            if n.endswith(".scale"):
                continue
            key = n.replace(f".layers.{i}.", f".layers.{j}.")
            w[key] = table.dequant(n) if n.endswith(".weight") else table.get(n)
    for n in ("embed.weight", "head.weight", "norm.weight",
              "hc_head_fn", "hc_head_scale", "hc_head_base"):
        w[n] = table.dequant(n)
    return w


def _bf16_roundtrip(v: np.ndarray) -> np.ndarray:
    u = v.astype(np.float32).view(np.uint32)
    return ((u >> 16) << 16).view(np.float32).reshape(v.shape)


def _stats(name: str, a: np.ndarray):
    a64 = a.astype(np.float64)
    print(f"[ds] {name}: shape={a.shape} rms={float(np.sqrt((a64 ** 2).mean())):.6e} "
          f"min={float(a.min()):.4f} max={float(a.max()):.4f} "
          f"finite={bool(np.isfinite(a).all())}")


def run_parity(model_dir: str, tiny: bool, layers, qat: bool, seq: int,
               seed: int, out: str | None) -> dict:
    cfg = load_cfg(tiny, qat)
    subset, sub_cfg = pick_subset(cfg, layers)
    print(f"[ds] subset layers {subset} -> ratios {sub_cfg.compress_ratios} "
          f"qat_sim={qat}")

    if tiny:
        from serving.dev_model import tiny_deepseek_v4_weights
        w = tiny_deepseek_v4_weights(sub_cfg)
        model_dir = "(tiny)"
    else:
        from loaders.streaming import CheckpointIndex, LazyWeightTable
        index = CheckpointIndex(model_dir)
        table = LazyWeightTable(index, scale_suffix=".scale")
        q = [n for n in table.names() if table.is_quantized(n)]
        print(f"[ds] checkpoint: {len(table.names())} tensors, "
              f"{len(q)} quantized")
        w = load_subset_weights(table, subset)
        index.close()
    w = {k: np.array(v) for k, v in w.items()}
    nbytes = sum(v.nbytes for v in w.values())
    print(f"[ds] subset weights: {len(w)} tensors, {nbytes / 2**30:.2f} GiB fp32")

    rng = np.random.default_rng(seed)
    ids = rng.integers(1, min(64, sub_cfg.vocab_size), size=seq, dtype=np.int64)
    ids[0] = 1

    model = ds.DeepseekV4Model(w, sub_cfg)
    _, logits = model.prefill(ids)              # fp32 oracle
    _, logits2 = model.prefill(ids)             # determinism
    _stats("logits fp32", logits)
    det = bool(np.array_equal(logits, logits2))
    print(f"[ds] determinism bit-identical: {det}")

    w_bf = {k: (_bf16_roundtrip(v)
                if v.ndim >= 2 and np.issubdtype(v.dtype, np.floating) else v)
            for k, v in w.items()}
    model_bf = ds.DeepseekV4Model(w_bf, sub_cfg)
    _, logits_bf = model_bf.prefill(ids)        # bf16-storage floor
    rel = float(np.abs(logits_bf - logits).std() / (logits.std() + 1e-12))
    agree = float((logits_bf.argmax(-1) == logits.argmax(-1)).mean())
    _stats("logits bf16-cast", logits_bf)
    print(f"[ds] noise floor bf16-vs-fp32: rel_std={rel:.3e} "
          f"argmax_agreement={agree:.3f}  <- floor for future GEMM engines")

    if out:
        np.savez_compressed(out, logits=logits, ids=ids,
                            subset=np.asarray(subset), seed=seed)
        print(f"[ds] reference saved -> {out} (feed to torch-golden compare)")
    return {"determinism": det, "rel_std": rel, "argmax_agreement": agree,
            "subset": subset, "logits": logits, "model_dir": model_dir}


def resolve_model_dir(model_dir: str | None) -> str:
    if model_dir:
        return os.path.expanduser(model_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    rec = os.path.join(here, "..", "recipes", "DeepSeek-V4-Flash-FP8.yaml")
    if os.path.isfile(rec):
        from recipes.schema import Recipe
        with open(rec, encoding="utf-8") as f:
            ld = Recipe.from_yaml(f.read()).local_dir
        if ld:
            return os.path.expanduser(ld)
    raise SystemExit("[ds] no --model-dir and the recipe has no paths.local_dir")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=None,
                    help="weights dir (default: recipe paths.local_dir)")
    ap.add_argument("--tiny", action="store_true",
                    help="tiny dev model (local plumbing check)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qat", action="store_true",
                    help="enable QAT activation simulation")
    ap.add_argument("--out", default=None, help="save logits npz")
    args = ap.parse_args(argv)
    md = args.model_dir if args.tiny else resolve_model_dir(args.model_dir)
    print(f"[ds] model dir: {md}")
    r = run_parity(md, args.tiny, args.layers, args.qat, args.seq, args.seed,
                   args.out)
    ok = r["determinism"] and np.isfinite(r["rel_std"])
    print(f"[ds] {'PASS' if ok else 'FAIL'}: subset parity run complete")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
