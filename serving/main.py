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
from serving import diag
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
    else:
        diag.info("sllm", f"TP{nodes} (recipe tp.size={tp})")
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
            lines.append(f"  pair link: {topo.ranks[0].pair_ip} <- -> "
                         f"{topo.ranks[1].pair_ip} (bandwidth measured in "
                         f"milestone B via bench/probe_pair_link.sh)")
        else:
            lines.append("  rank 0   : single node (TP1 override validated "
                         "against defaults.weights_gib)")
    elif recipe.arch == "deepseek_v4":
        from ref.deepseek_v4 import DeepseekV4Cfg
        from runtime.memory_planner import (deepseek_bytes_per_token,
                                            deepseek_seq_state_bytes)
        cfg = DeepseekV4Cfg.from_recipe(recipe)
        per_tok = deepseek_bytes_per_token(cfg, kv_bytes=1, idx_bytes=1)
        per_seq = deepseek_seq_state_bytes(cfg)
        lines.append(
            f"  cache    : {per_tok} B/token (MLA window + compressed KV + "
            f"indexer), {per_seq / 2**20:.0f} MiB/sequence compressor state "
            f"(fp32)")
        w = recipe.defaults.get("weights_gib")
        if w is not None:
            lines.append(f"  weights  : {w} GiB fp8/fp4 -> ~{float(w) / nodes:.0f} "
                         f"GiB/rank at tp{nodes}")
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


def _ram_gib_available() -> float | None:
    """Best-effort free host RAM in GiB (/proc/meminfo; None on non-Linux)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 ** 2)  # kB -> GiB
    except Exception:  # noqa: BLE001 - guard is best-effort
        return None
    return None


def _check_weight_mem(recipe: Recipe) -> None:
    """Fast-fail BEFORE the fp8->fp32 dequant OOM-thrashes the node (the
    loader now streams tensors one at a time, but a 27B still needs ~2x its
    fp8 store in host RAM)."""
    wg = float((recipe.defaults or {}).get("weights_gib") or 0)
    if wg <= 0:
        return
    need = wg * 2.05 + 2.0                       # fp8->fp32 + loader/slack
    avail = _ram_gib_available()
    if avail is not None and need > avail:
        raise SystemExit(
            f"sllm: {recipe.model_id} needs ~{need:.0f} GiB host RAM for the "
            f"fp32 reference load; ~{avail:.0f} GiB available. Stop the "
            f"shared vLLM on ALL TP2 nodes for a quiet window, then retry "
            f"(--mode plan never loads weights).")


def _load_progress(path, done: int, total: int) -> None:
    """Throttled loader progress (start/end + every 200th tensor) so a
    multi-GiB build shows where it is instead of sitting silent."""
    name = os.path.basename(path)
    if done in (1, total) or done % 200 == 0:
        diag.info("sllm", f"load {name}: {done}/{total} tensors")


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
    if recipe.arch == "qwen3_5":
        # Qwen3.8-27B-FP8 real engine: GDN linear-attn + paged full-attn via
        # the numpy incremental path (paged_flash + gated_delta_rule are in
        # ref/incremental.SUPPORTED_KERNELS). Weights are fp8 e4m3;
        # load_recipe_weights dequants to fp32 (~2x the 29 GiB store), so the
        # node needs the quiet window (stop the shared vLLM container).
        md = os.path.expanduser(model_dir or recipe.local_dir or "")
        if not md or not os.path.isdir(md):
            raise SystemExit(f"sllm: weights not found: {md or '(no path)'}")
        shards = [os.path.join(md, f) for f in sorted(os.listdir(md))
                  if f.endswith(".safetensors")]
        if not shards:
            raise SystemExit(f"sllm: no *.safetensors in {md}")
        _check_weight_mem(recipe)
        from loaders.weights import load_recipe_weights
        from serving import diag
        from serving.executor import InferenceEngine, ReferenceModel
        from serving.tokenizer import Tokenizer
        weights = load_recipe_weights(shards, progress=_load_progress)
        total_bytes = sum(int(np.asarray(v).nbytes) for v in weights.values())
        diag.info("sllm", f"loaded {len(weights)} tensors "
                          f"({diag.gib(total_bytes)} fp32)")
        return InferenceEngine(ReferenceModel(recipe, weights), Tokenizer(md))
    if recipe.arch not in ("qwen2", "llama"):
        raise SystemExit(
            f"sllm: arch {recipe.arch!r} has no executable engine yet "
            f"(served now: qwen2/llama standard, tiny/* dev engines)")
    md = os.path.expanduser(model_dir or recipe.local_dir or "")
    if not md or not os.path.isdir(md):
        raise SystemExit(f"sllm: weights not found: {md or '(no path)'}")
    _check_weight_mem(recipe)
    from serving.serve_standard import load_standard_engine  # existing engine
    return load_standard_engine(md)[0]


def run_model(recipe: Recipe, model_dir: str | None, chat: str | None,
              max_new: int) -> int:
    eng = _build_engine(recipe, model_dir)
    show = getattr(eng, "show_banner", None)
    if callable(show):
        show()
    out = eng.chat([{"role": "user", "content": chat or "hello"}],
                   max_new=max_new)
    print(out if isinstance(out, str) else str(out))
    return 0


def serve_model(recipe: Recipe, model_dir: str | None, host: str,
                port: str) -> int:
    engine = _build_engine(recipe, model_dir)
    show = getattr(engine, "show_banner", None)
    if callable(show):
        show()
    from serving.server import create_server
    server, bound = create_server(engine, host=host, port=int(port),
                                  quiet=False,
                                  model_name=recipe.name or recipe.model_id)
    diag.info("sllm", f"{recipe.name or recipe.model_id}: OpenAI-compatible "
                      f"API on http://{host}:{bound} (/v1/models, "
                      f"/v1/chat/completions, /v1/completions)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def resolve(recipe: Recipe, nodes: int | None, host: str | None,
            port: str | None):
    """Precedence: CLI > recipe `defaults:` > config.env (SLLM_HOST /
    SLLM_PORT) > built-in. The serve bind address is a recipe concern
    (`defaults.host`/`defaults.port`); node-pair IPs live in config.env
    (SLLM_{HEAD,WORKER}_{,PAIR_}IP, tp/topology.py)."""
    de = recipe.defaults
    try:
        n = nodes if nodes is not None else int(
            de.get("nodes", recipe.tp.size if recipe.tp else 1))
    except (TypeError, ValueError):
        raise SystemExit(f"sllm: defaults.nodes must be an integer "
                         f"(got {de.get('nodes')!r})") from None
    h = host or de.get("host") or _env("SLLM_HOST") or "0.0.0.0"
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
    ap.add_argument("recipe", nargs="?",
                    help="recipe yaml (path or name under recipes/)")
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
    ap.add_argument("--log-level", default=None,
                    choices=("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"),
                    help="diagnostic verbosity (default $SLLM_LOG_LEVEL/INFO)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (numpy) decode even when CUDA is present "
                         "(same as SLLM_USE_GPU=0; default is AUTO: GPU if "
                         "CUDA+.so, else CPU)")
    ap.add_argument("--version", "-V", action="store_true",
                    help="print the sLLM version and exit")
    args = ap.parse_args(argv)
    diag.set_level(args.log_level)
    if args.cpu:
        os.environ["SLLM_USE_GPU"] = "0"
    if args.version:
        from serving.version import version_string
        print(f"sllm {version_string()}")
        return 0
    if args.recipe is None:
        ap.error("the following arguments are required: recipe")

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
