"""D+V L5: DeepSeek-V4 torch-golden parity harness (RUN ON CLUSTER, torch box).

Loads the real checkpoint + the vendored reference (`ref/hf_sources/dsv4/
model.py`) in torch, runs the SAME ids that bench/deepseek_subset_parity.py
saved to an .npz, and reports logit delta vs the numpy oracle:

    # 1. on head (numpy box): produce the reference
    python bench/deepseek_subset_parity.py --seq 128 --out /tmp/dsv4_ref.npz
    # 2. on the torch box (weights dir): golden vs that reference
    python bench/deepseek_parity.py --npz /tmp/dsv4_ref.npz --model-dir <dir>

The subset default matches `pick_subset` in the numpy bench: first ratio-4
layer + first ratio-0 layer. QAT sim must be OFF on the numpy side for a
meaningful comparison (both sides fp32-dequant math; torch runs the vendored
bf16/native module numerics, so expect the documented float floor, not
bit-exactness -- bit-golden status is the l5 deliverable).

Only imports torch when the npz is present and the target layers actually
exist in the checkpoint; safe to commit alongside CI that never runs it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True, help="reference npz from the "
                                                 "numpy subset bench")
    ap.add_argument("--model-dir", required=True, help="real checkpoint dir")
    ap.add_argument("--ckpt-file", default=None,
                    help="safetensors file name (default: model0-mp1.safetensors)")
    ap.add_argument("--config", default="config.json",
                    help="model config json in the model dir")
    args = ap.parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"[ds-golden] torch unavailable ({exc}) -- cluster only")
        return 2

    ref = np.load(args.npz, allow_pickle=True)
    ids = ref["ids"].astype(np.int64)
    subset = [int(x) for x in ref["subset"]]
    torch_logits = _torch_forward(args, ids, subset)
    if torch_logits is None:
        return 2
    oracle = ref["logits"]
    if torch_logits.shape != oracle.shape:
        print(f"[ds-golden] shape mismatch torch {torch_logits.shape} vs "
              f"oracle {oracle.shape}")
        return 1
    diff = np.abs(torch_logits - oracle)
    rel = float(np.std(torch_logits - oracle) / (float(np.std(oracle)) + 1e-12))
    agree = float((torch_logits.argmax(-1) == oracle.argmax(-1)).mean())
    print(f"[ds-golden] max_abs={float(diff.max()):.3e} rel_std={rel:.3e} "
          f"argmax_agreement={agree:.3f}")
    ok = float(diff.max()) < 1e-2 and agree == 1.0
    print(f"[ds-golden] {'PASS' if ok else 'FAIL'}: torch vs numpy oracle")
    return 0 if ok else 1


def _torch_forward(args, ids, subset) -> np.ndarray | None:
    import json
    import torch
    from safetensors.torch import load_file

    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "ref", "hf_sources", "dsv4")))
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "ref", "hf_sources", "dsv4",
        "encoding")))  # not used; harmless if missing

    with open(os.path.join(args.model_dir, args.config), encoding="utf-8") as f:
        model_kw = json.load(f)
    model_kw["max_batch_size"] = 1
    from model import ModelArgs, Transformer  # noqa: E402

    args_obj = ModelArgs(**model_kw)
    if not set(subset).issubset(range(args_obj.n_layers)):
        print("[ds-golden] subset layers not present in the config layers "
              f"{args_obj.n_layers}")
        return None
    torch.set_default_dtype(torch.bfloat16)
    ckpt = args.ckpt_file or "model0-mp1.safetensors"
    state = load_file(os.path.join(args.model_dir, ckpt))
    # subset the checkpoint: keep whole layers remapped to 0..len-1 plus the
    # top-level tensors the oracle weights use.
    keep = {"embed.weight", "head.weight", "norm.weight",
            "hc_head_fn", "hc_head_scale", "hc_head_base"}
    for j, i in enumerate(subset):
        for k in state:
            if k.startswith(f"layers.{i}."):
                keep.add(k.replace(f"layers.{i}.", f"layers.{j}."))
    # mtp/main-projection placeholders: the subset has no target layers, so
    # drop every mtp.* and top-level extra except the kept set.
    pruned = {k: v for k, v in state.items() if k in keep}
    args_obj.n_layers = len(subset)
    args_obj.dspark_block_size = 0
    args_obj.dspark_target_layer_ids = ()
    args_obj.n_mtp_layers = 0
    with torch.device("cpu"):
        model = Transformer(args_obj)
    missing, unexpected = model.load_state_dict(pruned, strict=False)
    if missing:
        print(f"[ds-golden] missing tensors: {sorted(missing)[:5]} ...")
    with torch.inference_mode():
        x = torch.from_numpy(ids).unsqueeze(0)
        _, logits, _ = model(x)
    return logits[0].float().numpy()


if __name__ == "__main__":
    raise SystemExit(main())
