"""`sllm <recipe>` — recipe-driven startup (the Docker entrypoint target).

Usage (inside the sllm-node container or natively from the repo root):
    python -m serving.main recipes/Qwen3.8-Flash-Next-FP8.yaml [--tp 1|2] [--mode plan|run|serve]
        [--model-dir D] [--chat TEXT] [--max-new N]

Division of inputs: the RECIPE is the model (identity, geometry, tp.size,
weights location paths.local_dir); config.env/env vars are the common SLLM
settings (nodes, pair IPs, port). `--nodes` selects the drive shape and is
validated against the recipe: a tp.size=2 model cannot run on one node.

Modes:
  plan   (default)  resolve everything and print the run plan (rank table,
                    weights, per-token/per-sequence cache bytes). No GPU,
                    safe anywhere.
  run               execute what the current engine supports:
                    qwen4_exp dev weights (CPU oracle) end-to-end, or a
                    standard-arch checkpoint via serve_standard. Real
                    qwen4_exp weights and TP2 execution are gated to their
                    milestones (C phase) with an explicit error, never a
                    silent fallback.
  serve             OpenAI-compatible HTTP server (GET /v1/models,
                    POST /v1/chat/completions, POST /v1/completions) on
                    the resolved host:port, engine as in `run`.

TP mode: `--tp|--nodes N` selects TP1/TP2. Default = recipe
`defaults.nodes` (fallback tp.size); a lower value is accepted only when
defaults.weights_gib fits one node (SLLM_NODE_WEIGHT_BUDGET_GIB, default
110 GiB of the 119 GiB (128 GB) coherent pool); otherwise the arithmetic
rejection names the numbers.
"""

from __future__ import annotations

import argparse
import os

from env_config import get as _env
from env_config import get_int as _env_int
from recipes.schema import Recipe
from runtime.memory_planner import (kv_bytes_per_token, qwen4_exp_bytes_per_token,
                                    qwen4_exp_seq_state_bytes)
from tp.topology import ClusterTopology

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(recipe_arg: str) -> tuple[Recipe, str]:
    path = recipe_arg if os.path.isfile(recipe_arg) else \
        os.path.join(_ROOT, "recipes", recipe_arg)
    if not os.path.isfile(path):
        raise SystemExit(f"sllm: recipe not found: {recipe_arg}")
    with open(path, encoding="utf-8") as f:
        recipe = Recipe.from_yaml(f.read())
    return recipe, path


def _check_nodes(recipe: Recipe, nodes: int) -> int:
    """TP1/TP2 drive validation. tp.size is the recipe's designed shape;
    driving with fewer nodes is allowed only when the weights provably fit
    (defaults.weights_gib arithmetic), never by silent fallback."""
    tp = recipe.tp.size if recipe.tp else 1
    if nodes < 1:
        raise SystemExit(f"sllm: --tp must be >= 1 (got {nodes})")
    if nodes > tp:
        raise SystemExit(
            f"sllm: --tp {nodes} but recipe tp.size={tp}; multi-node is "
            f"only for tp{tp} recipes")
    if nodes < tp:
        budget = _env_int("SLLM_NODE_WEIGHT_BUDGET_GIB", 110)
        w = recipe.defaults.get("weights_gib")
        if w is None:
            raise SystemExit(
                f"sllm: {recipe.model_id} is tp.size={tp}; driving it with "
                f"{nodes} node(s) needs a defaults.weights_gib fact in the "
                f"recipe to prove it fits")
        try:
            per_node = float(w) / nodes
        except (TypeError, ValueError):
            raise SystemExit(
                f"sllm: defaults.weights_gib must be numeric "
                f"(got {w!r} in {recipe.model_id})") from None
        if per_node > budget:
            raise SystemExit(
                f"sllm: {recipe.model_id} cannot run on {nodes} node(s): "
                f"{w} GiB weights -> {per_node:.0f} GiB/node exceeds the "
                f"{budget} GiB budget (use --tp {tp})")
        print(f"sllm: TP{nodes} override: {w} GiB weights fit one node "
              f"({per_node:.0f} GiB <= {budget} GiB budget)")
    return nodes


def build_plan(recipe: Recipe, path: str, model_dir: str | None,
               host: str, port: str, nodes: int):
    md = os.path.expanduser(model_dir or recipe.local_dir or "")
    lines = [f"sllm plan: {recipe.model_id} (arch={recipe.arch}, status="
             f"{recipe.status})",
              f"  recipe   : {path}",
              f"  weights  : {md or '(none)'}"
              + ("" if md and os.path.isdir(md) else "  [MISSING on this node]")]
    if recipe.arch == "qwen4_exp":
        from ref.qwen4_exp_pipeline import Qwen4ExpCfg
        cfg = Qwen4ExpCfg.from_recipe(recipe)
        w = recipe.defaults.get("weights_gib")
        lines += [
            f"  cache    : {qwen4_exp_bytes_per_token(cfg)} B/token (fp8 KV+"
            f"indexer), {qwen4_exp_seq_state_bytes(cfg) / 2**20:.0f} MiB/"
            f"sequence fixed GDN state (fp32)",
        ]
        if w is not None:
            lines.append(f"  weights  : {w} GiB fp8 -> ~{float(w) / nodes:.0f} "
                         f"GiB/rank at tp{nodes}")
        if nodes > 1:
            topo = ClusterTopology.dgx_spark_pair()
            for r in topo.ranks:
                lines.append(f"  rank {r.rank}: host={r.host} pair={r.pair_ip} "
                             f"dev={r.device}")
            from tp.topology import HEAD_PAIR_IP, WORKER_PAIR_IP
            lines.append(f"  pair link: {HEAD_PAIR_IP} <-> {WORKER_PAIR_IP} "
                         f"(bandwidth measured in milestone B via "
                         f"bench/probe_pair_link.sh)")
        else:
            lines.append("  rank 0   : single node (TP1 override validated "
                         "against defaults.weights_gib)")
    elif recipe.arch in ("qwen3_5", "qwen3_5_moe"):
        per_tok = kv_bytes_per_token(recipe, kv_bytes=1)
        lines.append(f"  cache    : {per_tok} B/token (fp8 KV over "
                     f"{len(recipe.full_attn_indices())} full-attn layers)")
    lines.append(f"  serving  : {host}:{port}")
    if recipe.launch_env:
        lines.append("  env      : " + ", ".join(f"{k}={v}" for k, v
                                                 in recipe.launch_env.items()
                                                 if v))
    return "\n".join(lines)


def _build_engine(recipe: Recipe, model_dir: str | None):
    """Engine for run/serve, or an honest SystemExit (never a silent
    fallback to a smaller model)."""
    if recipe.model_id.startswith("tiny/"):
        from serving.dev_model import (build_dev_engine,
                                       build_dev_qwen4_exp_engine,
                                       build_dev_standard_engine)
        if recipe.arch == "qwen4_exp":
            return build_dev_qwen4_exp_engine()
        if recipe.arch in ("qwen2", "llama"):
            from serving.dev_model import TinyCharTokenizer
            from serving.executor import InferenceEngine
            return InferenceEngine(build_dev_standard_engine(),
                                   TinyCharTokenizer())
        return build_dev_engine()
    if recipe.arch == "qwen4_exp":
        raise SystemExit(
            "sllm: real qwen4_exp weights need the exclusive-GPU "
            "milestones (C phase: fp8 GEMM kernels + device-resident state). "
            "Use --mode plan, or a tiny/ recipe for the dev engine.")
    if recipe.arch not in ("qwen2", "llama"):
        raise SystemExit(
            f"sllm: arch {recipe.arch!r} has no executable engine yet "
            f"(served now: qwen2/llama standard, tiny/* dev engines)")
    md = os.path.expanduser(model_dir or recipe.local_dir or "")
    if not md or not os.path.isdir(md):
        raise SystemExit(f"sllm: weights not found: {md or '(no path)'}")
    from serving.serve_standard import load_standard_engine  # existing engine
    return load_standard_engine(md)[0]


def run_model(recipe: Recipe, model_dir: str | None, chat: str | None,
              max_new: int) -> int:
    eng = _build_engine(recipe, model_dir)
    out = eng.chat([{"role": "user", "content": chat or "hello"}],
                   max_new=max_new)
    print(out if isinstance(out, str) else str(out))
    return 0


def serve_model(recipe: Recipe, model_dir: str | None, host: str,
                port: str) -> int:
    engine = _build_engine(recipe, model_dir)
    from serving.server import create_server
    server, bound = create_server(engine, host=host, port=int(port),
                                  quiet=False,
                                  model_name=recipe.name or recipe.model_id)
    print(f"[sllm] {recipe.name or recipe.model_id}: OpenAI-compatible API "
          f"on http://{host}:{bound} "
          f"(/v1/models, /v1/chat/completions, /v1/completions)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def resolve(recipe: Recipe, nodes: int | None, host: str | None,
            port: str | None):
    """Precedence: CLI > recipe `defaults:` > config.env (SLLM_PORT) >
    built-in. host has no config.env layer (bind address is a CLI/recipe
    concern)."""
    de = recipe.defaults
    try:
        n = nodes if nodes is not None else int(
            de.get("nodes", recipe.tp.size if recipe.tp else 1))
    except (TypeError, ValueError):
        raise SystemExit(f"sllm: defaults.nodes must be an integer "
                         f"(got {de.get('nodes')!r})") from None
    h = host or de.get("host") or "0.0.0.0"
    p = port or str(de.get("port") or _env("SLLM_PORT", "8002"))
    try:
        if not 0 < int(p) < 65536:
            raise ValueError
    except (TypeError, ValueError):
        raise SystemExit(f"sllm: port must be a valid TCP port "
                         f"(got {p!r})") from None
    return n, h, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sllm", description=__doc__.splitlines()[0])
    ap.add_argument("recipe", help="recipe yaml (path or name under recipes/)")
    ap.add_argument("--tp", "--nodes", dest="nodes", type=int, default=None,
                    metavar="N",
                    help="TP mode (1 or 2); default = recipe "
                         "defaults.nodes / tp.size")
    ap.add_argument("--mode", choices=("plan", "run", "serve"), default="plan")
    ap.add_argument("--model-dir", default=None,
                    help="override recipe paths.local_dir")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--chat", default=None)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args(argv)

    recipe, path = _resolve(args.recipe)
    nodes, host, port = resolve(recipe, args.nodes, args.host, args.port)
    _check_nodes(recipe, nodes)
    if args.mode == "plan":
        print(build_plan(recipe, path, args.model_dir, host, port, nodes))
        tp = recipe.tp.size if recipe.tp else 1
        print(f"  nodes    : {nodes} (recipe tp={tp}, mode=plan: nothing "
              f"executed)")
        return 0
    if args.mode == "serve":
        return serve_model(recipe, args.model_dir, host, port)
    return run_model(recipe, args.model_dir, args.chat, args.max_new)


if __name__ == "__main__":
    raise SystemExit(main())
