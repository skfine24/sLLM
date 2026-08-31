"""Decode-path timing: numpy incremental vs GPU transfer vs device-resident.

Run on a cluster node with a built sllm_gpu.so and (for the real model) the
checkpoint available. Small synthetic model (default):

    python bench/gpu_decode_timing.py --steps 32

Real Qwen2.5-Coder-0.5B (preferred, on the head node):

    python bench/gpu_decode_timing.py --model-dir ~/models/Qwen2.5-Coder-0.5B --steps 32

All three paths decode the SAME prompt and are checked to agree greedy; the
point is the per-token time (device-resident should win once per-op transfers +
syncs are removed). A busy GPU may make the resident build fall back — the
script prints which paths actually ran on the GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from ref import incremental as inc  # noqa: E402
from serving.dev_model import tiny_standard_recipe, tiny_standard_weights  # noqa: E402


def _load_real(model_dir):
    from loaders.weights import load_recipe_weights
    from recipes.schema import Recipe

    recipe_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "recipes", "Qwen2.5-Coder-0.5B.yaml"))
    with open(recipe_path, encoding="utf-8") as f:
        recipe = Recipe.from_yaml(f.read())
    weights = load_recipe_weights([os.path.join(model_dir, "model.safetensors")])
    return recipe, weights, list(range(10, 20))


def _time(fn, steps, cache, weights, recipe, start_id):
    t0 = time.perf_counter()
    ids = [start_id]
    for _ in range(steps):
        L = fn(cache, weights, recipe, ids[-1])
        ids.append(int(np.argmax(L)))
    return time.perf_counter() - t0, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="device-resident weights/KV dtype")
    ap.add_argument("--no-bf16", action="store_true",
                    help="skip the extra bf16 resident run")
    ap.add_argument("--no-transfer", action="store_true",
                    help="skip the slow GPU transfer baseline")
    args = ap.parse_args()

    if args.model_dir:
        recipe, weights, prompt = _load_real(os.path.expanduser(args.model_dir))
    else:
        recipe, weights, prompt = (tiny_standard_recipe(),
                                   tiny_standard_weights(np.random.default_rng(3)),
                                   list(range(1, 9)))

    from kernels import _sllm_cuda as ck
    gpu = ck.device_count() >= 1
    print(f"model={recipe.model_id} steps={args.steps} "
          f"gpu={gpu} free_bytes={ck.mem_free_bytes() if gpu else 'n/a'}")

    # numpy incremental
    cache, L0 = inc.prefill(prompt, weights, recipe)
    t_np, ids_np = _time(inc.decode_step, args.steps, cache, weights, recipe,
                         int(np.argmax(L0[0, -1])))

    results = [("numpy incremental", t_np, ids_np)]

    if gpu:
        # GPU transfer path (weights H2D each call)
        if not args.no_transfer:
            from kernels.standard_decode import gpu_standard_decode_step
            cache, L0 = inc.prefill(prompt, weights, recipe)
            t_tf, ids_tf = _time(gpu_standard_decode_step, args.steps, cache,
                                 weights, recipe, int(np.argmax(L0[0, -1])))
            results.append(("GPU transfer", t_tf, ids_tf))

        # device-resident path (weights + KV on device, one sync/step)
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable
        dtypes = [args.dtype] + ([] if args.no_bf16 or args.dtype == "bf16"
                                 else ["bf16"])
        for dtype in dtypes:
            cache, L0 = inc.prefill(prompt, weights, recipe)
            try:
                table = DeviceWeightTable(weights, recipe, dtype=dtype)
            except Exception as exc:  # busy GPU: guard rejected
                print(f"GPU resident/{dtype}: skipped ({exc})")
                continue
            state = DeviceDecodeState(table, cache, recipe)
            try:
                t0 = time.perf_counter()
                ids_rs = [int(np.argmax(L0[0, -1]))]
                for _ in range(args.steps):
                    ids_rs.append(int(np.argmax(state.step(ids_rs[-1]))))
                t_rs = time.perf_counter() - t0
                results.append((f"GPU resident/{dtype}", t_rs, ids_rs))
            finally:
                state.free()
                table.free()

    print(f"{'path':>22} {'s':>9} {'tok/s':>8}  vs numpy(fp32) greedy")
    base = results[0][2]
    for name, t, ids in results:
        diff = [i for i, (a, b) in enumerate(zip(ids, base)) if a != b]
        if not diff:
            tag = "identical"
        else:
            tag = f"diverges at step {diff[0]} of {len(ids)} (precision trade-off)"
        print(f"{name:>22} {t:>9.3f} {args.steps / t if t else float('inf'):>8.2f}  {tag}")
    print("note: bf16 rounds weights/KV to 8-bit mantissa; greedy can branch "
          "from the fp32 oracle once a near-tie flips. fp32 stays the default.")


if __name__ == "__main__":
    main()
