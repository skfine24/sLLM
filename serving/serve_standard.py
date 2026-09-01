"""Serve a real standard (Llama/Qwen2-family) checkpoint with the reference
engine: load BF16 weights, build the engine, then either one-shot generate or
start the stdlib HTTP server.

Usage:
    python -m serving.serve_standard --model-dir <dir> --prompt "hello" --max-new 8
    python -m serving.serve_standard --model-dir <dir> --port 8002 --serve
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loaders.weights import load_recipe_weights  # noqa: E402
from recipes.schema import Recipe  # noqa: E402
from env_config import get as _env  # noqa: E402
from env_config import get_int as _env_int  # noqa: E402
from serving import diag  # noqa: E402
from serving.executor import InferenceEngine, ReferenceModel  # noqa: E402
from serving.tokenizer import Tokenizer  # noqa: E402

RECIPE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recipes", "Qwen2.5-Coder-0.5B.yaml"))


def load_standard_engine(model_dir: str, gpu_dtype: str | None = None
                         ) -> tuple[InferenceEngine, "ReferenceModel", Recipe]:
    with open(RECIPE, encoding="utf-8") as f:
        recipe = Recipe.from_yaml(f.read())
    shard = os.path.join(model_dir, "model.safetensors")
    weights = load_recipe_weights([shard])
    tokenizer = Tokenizer(model_dir)
    model = ReferenceModel(recipe, weights, gpu_dtype=gpu_dtype)
    return InferenceEngine(model, tokenizer), model, recipe


def main(argv=None):
    ap = argparse.ArgumentParser(description="serve a standard dense checkpoint")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--batch", nargs="+", default=None,
                    help="several prompts served with continuous batching (BatchedInferenceEngine)")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--host", default=None,
                    help="bind address (default $SLLM_HOST / 0.0.0.0)")
    ap.add_argument("--port", type=int, default=None,
                    help="bind port (default $SLLM_PORT / 8002)")
    ap.add_argument("--kv-placement", choices=["device", "host"], default=None,
                    help="override the recipe memory.kv_placement (device|host)")
    ap.add_argument("--use-gpu", action="store_true",
                    help="force the GPU decode kernels when available "
                         "(default is AUTO: GPU if CUDA+.so, else CPU)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (numpy) decode even when CUDA is present "
                         "(same as SLLM_USE_GPU=0 / --placement um)")
    ap.add_argument("--placement", choices=("device", "um"), default=None,
                    help="device = GPU-RESIDENT weights+KV (DEFAULT when "
                         "CUDA+.so present); um = host RAM / unified-memory "
                         "mode (model+KV in RAM, CPU compute)")
    ap.add_argument("--gpu-dtype", choices=["fp32", "bf16"], default=None,
                    help="device-resident decode dtype for weights/KV "
                         "(env SLLM_GPU_DTYPE; default fp32)")
    ap.add_argument("--serve", action="store_true", help="start HTTP server instead of one-shot")
    args = ap.parse_args(argv)

    if args.placement:
        os.environ["SLLM_PLACEMENT"] = args.placement
    if args.placement == "um" or args.cpu:
        os.environ["SLLM_USE_GPU"] = "0"
    elif args.use_gpu:
        os.environ["SLLM_USE_GPU"] = "1"
    if args.gpu_dtype:
        os.environ["SLLM_GPU_DTYPE"] = args.gpu_dtype
    engine, model, recipe = load_standard_engine(args.model_dir, args.gpu_dtype)

    if args.kv_placement:
        recipe.memory.kv_placement = args.kv_placement
    from runtime.placement import KVMemoryPlan
    plan = KVMemoryPlan.from_recipe(recipe)
    engine.show_banner(tag="serve-standard")
    diag.info("serve-standard", f"{recipe.model_id} kv_placement="
                               f"{plan.placement} plan={plan.describe()} "
                               f"(CPU mode uses host RAM only)")

    if args.serve:
        from serving.server import create_server
        host = args.host or _env("SLLM_HOST") or "0.0.0.0"
        port = args.port or _env_int("SLLM_PORT", 8002)
        server, port = create_server(engine, host=host, port=port,
                                     quiet=False)
        diag.info("serve-standard", f"{recipe.model_id} on "
                                    f"http://{host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return

    if args.batch:
        from serving.executor import generate_batch
        outs = generate_batch(model, engine.tokenizer, args.batch, max_new=args.max_new,
                              temperature=0.0)
        for p, o in zip(args.batch, outs):
            print(f"[{p!r}] -> {o!r}")
        return

    if args.prompt is None:
        ap.error("provide --prompt (or --batch / --serve)")
    out = engine.complete(args.prompt, max_new=args.max_new, temperature=0.0)
    print(f"prompt   : {args.prompt}")
    print(f"assistant: {out}")


if __name__ == "__main__":
    main()
