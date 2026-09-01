"""Per-architecture engine "cards" for the startup diagnostic banner.

`describe(recipe, weights, model)` renders a uniform, multi-line engine
summary that every engine prints at INFO in vLLM fashion:

    [sllm] ===== <model_id> (arch=<arch>) =====
    [sllm]   architecture  : qwen4_exp
    [sllm]   embed         : 2560 x 248320
    ...

The ARCH REGISTRY (`_DETAIL`) is the single extension point for new models:
to add a model family, implement `def card_<arch>(recipe, model, weights) ->
list[str]` here (or provide a recipe.meta card) and register it. Common
lines (identity, weights footprint, backend, cache bytes) are computed once
in `common()` and shared by every arch so output stays uniform.

All heavy imports are lazy so loading a card never pulls in an arch's
oracle/kernel stack on import.
"""

from __future__ import annotations

import numpy as np


def _weights_footprint(weights) -> tuple[int, int]:
    """(n_tensors, bytes). memmap-backed arrays report nbytes without loading."""
    n = len(weights)
    b = 0
    for v in weights.values():
        try:
            b += int(np.asarray(v).nbytes)
        except Exception:  # noqa: BLE001 - a broken weight must not kill the banner
            pass
    return n, b


def _backend(model) -> str:
    """Best-effort runtime backend label (auto-detected; numpy default)."""
    arch = getattr(model, "recipe", None)
    arch = getattr(arch, "arch", None)
    isq = getattr(model, "_is_q4", False)
    isd = getattr(model, "_is_dsv4", False)
    if isd:
        return "numpy (oracle + device-resident state)"
    if isq:
        return "numpy (CPU pipeline; GPU kernels are milestone Q4-GPU)"
    mode = getattr(model, "gpu_mode", "auto")
    if getattr(model, "use_gpu", False):
        if getattr(model, "_gpu_available", lambda: False)():
            # resident table is built lazily on the first device-resident step
            if (getattr(model, "_dev_table", None) is not None
                    and getattr(model, "_resident_off", True) is False):
                return f"gpu (device-resident {getattr(model, 'gpu_dtype', 'fp32')})"
            return "gpu (per-step kernel transfer)"
        if mode == "off":
            return "numpy (CPU forced)"
        return "numpy (CPU; GPU unavailable, auto fallback)"
    if mode == "off":
        return "numpy (CPU forced)"
    return "numpy (recompute/incremental)"


def _vision(recipe, model) -> str:
    try:
        if getattr(model, "_is_dsv4", False):
            m = model._dsv4model()
            return "yes (SigLIP tower, bf16)" if getattr(m, "vision", None) \
                else "no"
    except Exception:  # noqa: BLE001
        pass
    return "no"


def common(recipe, weights, model) -> list[str]:
    n, b = _weights_footprint(weights)
    ctx = getattr(recipe, "max_position_embeddings", "?")
    vocab = getattr(recipe, "vocab_size", "?")
    from serving.version import version_string
    out = [
        f"version       : {version_string()}",
        f"architecture  : {recipe.arch}",
        f"model         : {recipe.model_id}",
        f"vocab / ctx   : {vocab} / {ctx}",
        f"backend       : {_backend(model)}",
        f"vision        : {_vision(recipe, model)}",
        f"weights       : {n} tensors, {_gib(b)}",
    ]
    return out


def _gib(b: int) -> str:
    from serving import diag
    return diag.gib(b)


def _cache_line(arch: str, recipe) -> str | None:
    try:
        from runtime import memory_planner as mp
        if arch == "qwen4_exp":
            from ref.qwen4_exp_pipeline import Qwen4ExpCfg
            cfg = Qwen4ExpCfg.from_recipe(recipe)
            return (f"cache         : {mp.qwen4_exp_bytes_per_token(cfg)} "
                    f"B/token KV+indexer, "
                    f"{mp.qwen4_exp_seq_state_bytes(cfg) / 2 ** 20:.0f} "
                    f"MiB/seq GDN state")
        if arch == "deepseek_v4":
            from ref.deepseek_v4 import DeepseekV4Cfg
            cfg = DeepseekV4Cfg.from_recipe(recipe)
            per_tok = mp.deepseek_bytes_per_token(cfg, kv_bytes=1, idx_bytes=1)
            per_seq = mp.deepseek_seq_state_bytes(cfg)
            return (f"cache         : {per_tok} B/token MLA window+compressed "
                    f"KV+indexer, {per_seq / 2 ** 20:.0f} MiB/seq compressor "
                    f"state")
        if arch in ("qwen3_5", "qwen3_5_moe", "qwen2", "llama"):
            per_tok = mp.kv_bytes_per_token(recipe, kv_bytes=1)
            return (f"cache         : {per_tok} B/token fp8 KV over "
                    f"{len(recipe.full_attn_indices())} full-attn layers")
    except Exception:  # noqa: BLE001 - cache sizing must never kill the banner
        return None
    return None


def _qwen4_exp_detail(recipe, model, weights) -> list[str]:
    from ref.qwen4_exp_pipeline import Qwen4ExpCfg
    cfg = Qwen4ExpCfg.from_recipe(recipe)
    layers = f"{len(cfg.layer_types)} ({', '.join(cfg.layer_types)})"
    lines = [
        f"layers        : {layers}",
        f"hidden / hc   : {cfg.hidden} / {cfg.hc_count}",
        f"moe           : {cfg.n_experts} experts, top-{cfg.top_k}",
    ]
    if cfg.ple_layer_ids:
        lines.append(
            f"ple           : layers {list(cfg.ple_layer_ids)} "
            f"(ngram={cfg.ngram_size}, heads/ngram={cfg.heads_per_ngram})")
    else:
        lines.append("ple           : off")
    return lines


def _deepseek_v4_detail(recipe, model, weights) -> list[str]:
    from ref.deepseek_v4 import DeepseekV4Cfg
    cfg = DeepseekV4Cfg.from_recipe(recipe)
    min_ratio = min((r for r in cfg.compress_ratios if r), default=0)
    lines = [
        f"layers        : {cfg.n_layers}",
        f"hidden        : {cfg.dim}, heads {cfg.n_heads} (head_dim "
        f"{cfg.head_dim}, rope {cfg.rope_head_dim})",
        f"window        : {cfg.window_size}",
        f"indexer       : top-{cfg.index_topk} over ratio-{min_ratio} "
        f"blocks ({cfg.index_n_heads} idx heads)",
        f"compress      : ratios {list(cfg.compress_ratios[:6])}...",
        f"moe           : {cfg.n_routed_experts + cfg.n_shared_experts} "
        f"experts, top-{cfg.n_activated_experts} (fp4 packed + fp8)",
        f"hc_mult       : {cfg.hc_mult}",
    ]
    return lines


def _standard_detail(recipe, model, weights) -> list[str]:
    la = recipe.linear_attention
    fa = recipe.full_attention
    lines = [
        f"layers        : {recipe.num_layers} "
        f"(full_attn {len(recipe.full_attn_indices())})",
        f"hidden/heads  : {recipe.hidden_size} / {fa.num_heads} "
        f"(kv {getattr(fa, 'num_kv_heads', fa.num_heads)})",
        f"rope          : theta={fa.rope.theta}, "
        f"partial={fa.rope.partial_rotary_factor}",
        f"attn kernel   : {fa.kernel}",
    ]
    if la is not None:
        lines.append(f"linear_attn   : {la.num_key_heads}x{la.key_head_dim} "
                     f"heads, conv {la.conv_kernel_size}")
    return lines


_DETAIL = {
    "qwen4_exp": _qwen4_exp_detail,
    "deepseek_v4": _deepseek_v4_detail,
    "qwen3_5": _standard_detail,
    "qwen3_5_moe": _standard_detail,
    "standard": _standard_detail,
    # qwen2/llama map onto the standard card via the generic fallback below
}


def describe(recipe, weights, model) -> list[str]:
    """Full banner lines for the running engine."""
    out = common(recipe, weights, model)
    cache = _cache_line(recipe.arch, recipe)
    if cache:
        out.append(cache)
    fn = _DETAIL.get(recipe.arch)
    if fn is not None:
        try:
            out.extend(fn(recipe, model, weights))
        except Exception as e:  # noqa: BLE001
            out.append(f"arch detail   : (error {e})")
    return out
