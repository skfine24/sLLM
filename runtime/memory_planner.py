"""Memory planner: derive the KV block budget (and recurrent-state slot count)
from a memory budget and the model's full-attention geometry.

The planner only does arithmetic; the scheduler consumes its result as
capacity. The formula follows the design doc: KV bytes per token =
num_full_attention_layers * 2 (k+v) * num_kv_heads * head_dim * kv_bytes.
"""

from __future__ import annotations


def fp8_kv_bytes_per_token(num_full_attn_layers: int, num_kv_heads: int,
                           head_dim: int, kv_bytes: int = 1) -> int:
    """Storage bytes for one token's KV across all full-attention layers."""
    return num_full_attn_layers * 2 * num_kv_heads * head_dim * kv_bytes


def plan_block_count(kv_avail_bytes: int, bytes_per_token: int,
                     block_size: int, utilization: float = 0.9) -> int:
    """Number of KV blocks that fit in the available bytes."""
    if bytes_per_token <= 0 or block_size <= 0:
        raise ValueError("bytes_per_token and block_size must be > 0")
    if kv_avail_bytes < 0:
        raise ValueError("kv_avail_bytes must be >= 0")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    per_block = bytes_per_token * block_size
    budget = int(kv_avail_bytes * utilization)
    return budget // per_block


def qwen3_5_kv_profile(block_size: int = 16, kv_bytes: int = 1,
                       kv_avail_gib: float | None = None, utilization: float = 0.9):
    """Convenience for the audited Qwen3.8-27B-FP8 numbers.

    Full attention: 16 layers, 4 KV heads, head_dim 256. Returns a dict with
    per-token bytes and (if kv_avail_gib given) the resulting block budget.
    """
    per_token = fp8_kv_bytes_per_token(
        num_full_attn_layers=16, num_kv_heads=4, head_dim=256, kv_bytes=kv_bytes
    )
    result = {"bytes_per_token": per_token, "block_size": block_size}
    if kv_avail_gib is not None:
        budget = plan_block_count(
            int(kv_avail_gib * (1024 ** 3)), per_token, block_size, utilization
        )
        result["num_blocks"] = budget
        result["max_total_tokens"] = budget * block_size
    return result


def qwen4_exp_bytes_per_token(cfg, kv_bytes: int = 1, idx_bytes: int = 1) -> int:
    """Per-token cache bytes for a qwen4_exp (QSA) config, summed over all
    full_attention layers: dense KV (k+v) + indexer token stream (tok_k) +
    compressed keys (one idx_dim row per idx_ratio tokens).

    GDN layers contribute ZERO per-token bytes (their state is per-sequence;
    see qwen4_exp_seq_state_bytes)."""
    n_qsa = sum(1 for t in cfg.layer_types if t == "full_attention")
    return n_qsa * (2 * cfg.attn_kv_heads * cfg.attn_head_dim * kv_bytes
                    + cfg.idx_dim * idx_bytes
                    + (-(-cfg.idx_dim // cfg.idx_ratio)) * idx_bytes)


def qwen4_exp_seq_state_bytes(cfg, state_bytes: int = 4) -> int:
    """Per-SEQUENCE fixed state bytes for a qwen4_exp config (GDN layers):
    recurrent state (lin_v_heads x lin_v_dim x lin_k_dim) + conv window,
    summed over linear_attention layers. With the real geometry this is
    ~113 MB/sequence in fp32 — admission must account for it, not just for
    token-based KV."""
    n_gdn = sum(1 for t in cfg.layer_types if t == "linear_attention")
    c = 2 * cfg.lin_k_heads * cfg.lin_k_dim + cfg.lin_v_heads * cfg.lin_v_dim
    return n_gdn * (cfg.lin_v_heads * cfg.lin_v_dim * cfg.lin_k_dim * state_bytes
                    + c * (cfg.lin_conv - 1) * state_bytes)


def qwen4_exp_plan(cfg, kv_avail_bytes: int, avg_context: int,
                   kv_bytes: int = 1, idx_bytes: int = 1,
                   state_bytes: int = 4, utilization: float = 0.9) -> dict:
    """Admission budget for qwen4_exp: how many concurrent sequences of
    ~avg_context tokens fit into kv_avail_bytes, counting BOTH per-token
    cache and the (large) per-sequence GDN state."""
    if avg_context <= 0:
        raise ValueError("avg_context must be > 0")
    if kv_avail_bytes < 0:
        raise ValueError("kv_avail_bytes must be >= 0")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    per_tok = qwen4_exp_bytes_per_token(cfg, kv_bytes, idx_bytes)
    per_seq = qwen4_exp_seq_state_bytes(cfg, state_bytes)
    budget = int(kv_avail_bytes * utilization)
    per_seq_full = per_seq + per_tok * avg_context
    n_seqs = budget // per_seq_full if per_seq_full else 0
    return {
        "bytes_per_token": per_tok,
        "bytes_per_sequence": per_seq,
        "bytes_per_sequence_at_context": per_seq_full,
        "max_sequences": int(n_seqs),
        "max_total_tokens": int(n_seqs * avg_context),
    }


def kv_bytes_per_token(recipe, kv_bytes: int = 2) -> int:
    """KV bytes per token for a recipe's full-attention layers (placement
    planning). kv_bytes: bytes per scalar KV element (2 = BF16, 1 = FP8)."""
    return fp8_kv_bytes_per_token(
        num_full_attn_layers=len(recipe.full_attn_indices()),
        num_kv_heads=recipe.full_attention.num_kv_heads,
        head_dim=recipe.full_attention.effective_head_dim(recipe.hidden_size),
        kv_bytes=kv_bytes,
    )


def deepseek_bytes_per_token(cfg, kv_bytes: int = 2, idx_bytes: int = 2,
                             bool_bytes: int = 1) -> int:
    """Per-token cache bytes for a DeepSeek-V4 config over ALL layers.

    Each MLA layer writes ONE latent KV row (head_dim) into a shared window
    ring per new token (the ring is reused, so the incremental cost is one
    row, not `window` rows) plus ~1/ratio compressed rows, plus the
    learned-indexer compressed token stream for ratio-4 layers. Sink/ape and
    the per-sequence compressor buffers are counted via seq-state.
    """
    total = 0
    for ratio in cfg.compress_ratios:
        total += kv_bytes * cfg.head_dim  # window row
        if ratio:
            total += int(kv_bytes * (cfg.head_dim / ratio))
            if ratio == 4:
                total += int(idx_bytes * (cfg.index_head_dim / ratio))
    return int(total)


def deepseek_seq_state_bytes(cfg, state_bytes: int = 4, kv_bytes: int = 2,
                             idx_bytes: int = 2) -> int:
    """Per-SEQUENCE fixed state bytes: compressor+indexer incremental buffers
    (overlap keeps 2x rows on ratio-4 layers) over the compressed layers."""
    total = 0
    for ratio in cfg.compress_ratios:
        if ratio == 4:
            total += 2 * (2 * ratio) * (2 * cfg.head_dim) * state_bytes
            total += 2 * (2 * ratio) * (2 * cfg.index_head_dim) * state_bytes
        elif ratio:
            total += 2 * ratio * cfg.head_dim * state_bytes
    return int(total)


def deepseek_plan(cfg, kv_avail_bytes: int, avg_context: int,
                  kv_bytes: int = 2, idx_bytes: int = 2,
                  state_bytes: int = 4, utilization: float = 0.9) -> dict:
    """Admission budget for deepseek_v4: concurrent sequences of ~avg_context
    tokens into kv_avail_bytes (per-token cache + per-sequence compressor
    state)."""
    if avg_context <= 0 or kv_avail_bytes < 0 or not 0 < utilization <= 1:
        raise ValueError("invalid deepseek_plan args")
    per_tok = deepseek_bytes_per_token(cfg, kv_bytes, idx_bytes)
    per_seq = deepseek_seq_state_bytes(cfg, state_bytes, kv_bytes, idx_bytes)
    budget = int(kv_avail_bytes * utilization)
    per_seq_full = per_seq + per_tok * avg_context
    n_seqs = budget // per_seq_full if per_seq_full else 0
    return {
        "bytes_per_token": per_tok,
        "bytes_per_sequence": per_seq,
        "bytes_per_sequence_at_context": per_seq_full,
        "max_sequences": int(n_seqs),
        "max_total_tokens": int(n_seqs * avg_context),
    }


