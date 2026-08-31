"""Q3: real-checkpoint SUBSET parity for qwen4_exp (run on head; the
cluster-window milestone-B deliverable, written and locally verified here
against a synthetic checkpoint).

Loads actual fp8 checkpoint bytes through loaders/streaming (mmap, per-
tensor dequant), selects a layer subset from the REAL geometry (default:
first linear_attention + first full_attention, PLE layers excluded), and
drives the numpy pipeline forward on the subset model:

  * determinism  : two identical runs must be bit-for-bit equal
  * noise floor  : fp32-dequant oracle vs bf16-cast weights (what a future
                   bf16 GEMM engine would see) -> logit argmax agreement +
                   relative RMS; this is the floor any kernel must beat
  * reference    : subset logits saved to .npz for T1/T2 comparison against
                   sglang on the same ids/seed

Memory (real geometry, vLLM stopped): embed+lm_head ~5 GB fp32 + one MoE
layer ~10 GB fp32 -> printed up front.

Run (head; weights dir defaults to the recipe's paths.local_dir):
    python bench/q4_subset_parity.py
Run (local plumbing check):
    python tests/make_q4_fixture.py /tmp/q4fix && \
    python bench/q4_subset_parity.py --model-dir /tmp/q4fix --tiny
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import replace  # noqa: E402

from loaders.streaming import CheckpointIndex, LazyWeightTable  # noqa: E402
from ref import qwen4_exp_pipeline as qp  # noqa: E402

_PREFIX = "model.language_model"


def load_recipe(tiny: bool):
    if tiny:
        from serving.dev_model import tiny_qwen4_exp_recipe
        return tiny_qwen4_exp_recipe()
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "recipes", "Qwen3.8-Flash-Next-FP8.yaml"),
              encoding="utf-8") as f:
        from recipes.schema import Recipe
        return Recipe.from_yaml(f.read())


def pick_subset(cfg, layers):
    """Layer SUBSET (real indices). Default = first linear_attention + first
    full_attention; PLE layers are excluded (not implemented; Q5)."""
    if layers:
        subset = list(layers)
    else:
        subset = []
        for want in ("linear_attention", "full_attention"):
            for i, bt in enumerate(cfg.layer_types):
                if bt == want and i not in subset and i not in cfg.ple_layer_ids:
                    subset.append(i)
                    break
        if len(subset) != 2:
            raise SystemExit("[q3] could not pick one linear + one full layer "
                             "outside PLE layer ids")
    types = tuple(cfg.layer_types[i] for i in subset)
    return subset, replace(cfg, layer_types=types, ple_layer_ids=())


def load_subset_weights(table: LazyWeightTable, subset: list[int]) -> dict:
    """Real layer indices -> subset driver indices; fp8 folded, per tensor."""
    w = {}
    for j, i in enumerate(subset):
        for k, v in table.layer(i, _PREFIX).items():
            w[k.replace(f".layers.{i}.", f".layers.{j}.")] = v
    for n in table.index.filter(f"{_PREFIX}.hyper_connection_mixer"):
        if n.endswith("_scale_inv"):
            continue
        w[n] = table.dequant(n)
    w.update(table.embeddings_head(_PREFIX))
    return w


def _bf16_roundtrip(v: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 (truncate, numpy has no bfloat16) -> fp32."""
    u = v.astype(np.float32).view(np.uint32)
    return ((u >> 16) << 16).view(np.float32).reshape(v.shape)


def _stats(name: str, a: np.ndarray):
    a64 = a.astype(np.float64)
    print(f"[q3] {name}: shape={a.shape} rms={float(np.sqrt((a64 ** 2).mean())):.6e} "
          f"min={float(a.min()):.4f} max={float(a.max()):.4f} "
          f"finite={bool(np.isfinite(a).all())}")


def run_parity(model_dir: str, tiny: bool, layers, seq: int, seed: int,
               out: str | None) -> dict:
    cfg = qp.Qwen4ExpCfg.from_recipe(load_recipe(tiny))
    subset, sub_cfg = pick_subset(cfg, layers)
    print(f"[q3] subset layers {subset} -> types {sub_cfg.layer_types} "
          f"(PLE guard cleared for subset; PLE itself is unimplemented)")

    index = CheckpointIndex(model_dir)
    table = LazyWeightTable(index)
    q = [n for n in table.names() if table.is_quantized(n)]
    print(f"[q3] checkpoint: {len(table.names())} tensors, "
          f"{len(q)} fp8-quantized")
    w = load_subset_weights(table, subset)
    nbytes = sum(v.nbytes for v in w.values())
    print(f"[q3] subset weights: {len(w)} tensors, {nbytes / 2**30:.2f} GiB fp32")
    # materialize: native-dtype loads are memmap views; copying releases the
    # shard handles so the mmap lifetime never spans the forward.
    w = {k: np.array(v) for k, v in w.items()}
    index.close()

    rng = np.random.default_rng(seed)
    ids = rng.integers(1, 32 if tiny else 248320, size=seq, dtype=np.int64)
    ids[0] = 1

    _, logits = qp.prefill(ids, w, sub_cfg)          # fp32 oracle
    _, logits2 = qp.prefill(ids, w, sub_cfg)         # determinism
    _stats("logits fp32", logits)
    det = bool(np.array_equal(logits, logits2))
    print(f"[q3] determinism bit-identical: {det}")

    w_bf = {k: (_bf16_roundtrip(v)
                if v.ndim >= 2 and np.issubdtype(v.dtype, np.floating) else v)
            for k, v in w.items()}
    _, logits_bf = qp.prefill(ids, w_bf, sub_cfg)    # bf16-storage floor
    rel = float(np.abs(logits_bf - logits).std() / (logits.std() + 1e-12))
    agree = float((logits_bf.argmax(-1) == logits.argmax(-1)).mean())
    _stats("logits bf16-cast", logits_bf)
    print(f"[q3] noise floor bf16-vs-fp32: rel_std={rel:.3e} "
          f"argmax_agreement={agree:.3f}  <- floor for future GEMM engines")

    if out:
        np.savez_compressed(out, logits=logits, ids=ids,
                            subset=np.asarray(subset), seed=seed)
        print(f"[q3] reference saved -> {out} (feed to T1/T2 sglang compare)")
    return {"determinism": det, "rel_std": rel, "argmax_agreement": agree,
            "subset": subset, "logits": logits}


def resolve_model_dir(model_dir: str | None, tiny: bool) -> str:
    """Weights location comes from the recipe (paths.local_dir); an explicit
    --model-dir still wins. Model info lives in recipes, cluster info in
    config.env — this script needs no node knowledge."""
    if model_dir:
        return os.path.expanduser(model_dir)
    ld = load_recipe(tiny).local_dir
    if not ld:
        raise SystemExit("[q3] no --model-dir and the recipe has no "
                         "paths.local_dir")
    return os.path.expanduser(ld)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=None,
                    help="weights dir (default: recipe paths.local_dir)")
    ap.add_argument("--tiny", action="store_true",
                    help="tiny dev recipe (fixture plumbing check)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save logits npz")
    args = ap.parse_args(argv)
    md = resolve_model_dir(args.model_dir, args.tiny)
    print(f"[q3] model dir: {md}")
    r = run_parity(md, args.tiny, args.layers, args.seq, args.seed, args.out)
    ok = r["determinism"] and np.isfinite(r["rel_std"])
    print(f"[q3] {'PASS' if ok else 'FAIL'}: subset parity run complete")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
