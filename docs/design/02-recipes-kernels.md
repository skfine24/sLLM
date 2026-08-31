# 02 — Recipe Schema and Per-Model Kernel Inventory

## 1. What a recipe is

A recipe is a declarative document (JSON/YAML) that maps a model checkpoint to
the engine: architecture knobs, layer-type schedule, kernel choices, FP8
layout, loader plan, TP sharding plan, tokenizer/chat-template, and sampling
defaults. The engine contains no model-specific hard-coding; it executes the
recipe.

Suggested schema (draft):

```yaml
model_id: Qwen/Qwen3.8-27B-FP8
arch: qwen3_5
dtype: bfloat16
quant:
  method: fp8            # fmt e4m3
  activation: dynamic
  scale: TBD             # weight_block_size / per-tensor — verify from config
  exclude: [<modules_to_not_convert list>]
layers:
  count: 64
  schedule: {pattern: "L" x3 + "F", repeat: 16}   # linear/full attention
  linear_attention:
    kernel: linear_attn_ssm      # conv(4) + SSM state, float32 state
    state_dtype: float32
    key_heads: 16, key_head_dim: 128
    value_heads: 48, value_head_dim: 128
  full_attention:
    kernel: flashattn_paged
    heads: 24, kv_heads: 4, head_dim: 256
    rope: {type: mrope, section: [11,11,10], interleaved: true, partial: 0.25}
    output_gate: true
mlp: {type: moe_shared_gate, ...}
mtp: {enabled: true, layers: 1}
vision: {enabled: true, type: qwen3_5_vision, ...}
tp: {size: 2, map: <layer-split>, shard_axes: {...}}
tokenizer: {path: ..., chat_template: ...}
defaults: {max_seq_len, block_size, kv_dtype, sampling}
```

## 2. Kernel inventory — shared (built once)

| Kernel / component | Notes |
|---|---|
| FP8 dequant + dynamic quant (e4m3, per-block/per-tensor scales) | consumed by GEMM path |
| RMSNorm / activation / SiLU/Swish fused helpers | |
| Paged KV block allocator + gather/scatter | model-agnostic layout |
| Sampling (top-k/top-p/temperature, speculative accept) | |
| Grouped-GEMM glue for MoE (routes per token) | GEMM math via cuBLAS |
| Tokenizer / chat-template runner | per-recipe tokenizer data |

## 3. Kernel inventory — `qwen3_5` (Qwen3.8-27B-FP8) — FIRST TARGET

From `config.json` (`model_type=qwen3_5`):

- **Linear attention (48 of 64 layers)**: **GatedDeltaNet (gated delta rule,
  FLA-style)** — NOT a Mamba selective scan (audited from upstream
  `modeling_qwen3_5.py`; see `05-audit-qwen3-5.md`). Depthwise causal conv1d
  (kernel 4) + SiLU on the QKV projection, `A_log`/`dt_bias`/`in_proj_a`/
  `in_proj_b`/`in_proj_z` gating, L2-normed q/k, `head_k_dim**-0.5` query
  scaling, float32 recurrent state `(B, 48, 128, 128)`.
  - Kernel: **stateful gated-delta-rule forward** — chunked parallel variant
    (prefill) and single-step recurrence (decode); float32 state.
  - Reference math ported to `ref/qwen3_5.py` and parity tests in
    `tests/test_ref_qwen3_5.py`.
- **Full attention (16 layers, every 4th)**: GQA `24 / 4` heads, head_dim 256,
  M-RoPE (`mrope_section [11,11,10]`, interleaved), `partial_rotary_factor 0.25`,
  `attn_output_gate: true` (Q+gate from one projection; per-head q/k RMSNorm).
  - Kernel: paged FlashAttention (FP8 KV cache), RoPE fused.
- **MLP**: **dense** gate/up/down (`intermediate_size 17408`) — audited from
  the checkpoint index: no expert tensors exist, despite template names in
  `modules_to_not_convert`.
- **MTP**: 1 MTP layer of **full attention** + dense MLP + `mtp.fc`
  (`mtp_num_hidden_layers: 1`; audited).
- **Vision**: Qwen3.5 vision (`vision_config`: 27 layers, hidden 1152, 16
  heads, patch 16, deepstack-templated merger) + M-RoPE in text. First four
  ViT blocks and patch_embed stay non-quantized (BF16).

## 4. Kernel inventory — `qwen4_exp` (Qwen3.8-Flash-Next-FP8) — SECOND TARGET

From local docs (`kv-cache-architecture.md`) and config:

- 48 layers in a repeating unit: `3 x (GatedDeltaNet -> MoE) -> 1 x Qwen Sparse
  Attention -> MoE` (12 QSA + 36 GDN).
- **GDN / Gated DeltaNet (36 layers)**: delta-rule recurrent state updates;
  requires an **associative-scan / delta-rule kernel** (not a plain Mamba scan).
  Recurrent state + conv state managed by a hybrid state manager (vLLM calls it
  the Mamba-style state manager).
- **QSA sparse attention (12 layers)**: token-block sparse attention with an
  indexer that selects blocks; FP8 KV; needs block-sparse paged attention
  kernel + indexer kernel.
- **MoE** inside the repeating unit; **MTP3**; **vision** (own encoder);
  multimodal M-RoPE.

## 5. Kernel inventory — `deepseek_v4` (DeepSeek-V4-Flash-0731) — THIRD TARGET

From `config.json` (`model_type=deepseek_v4`, 304B):

- **MLA (Multi-head Latent Attention)**: `q_lora_rank/o_lora_rank 1024`,
  `head_dim 512`, `num_attention_heads 64`, `num_key_value_heads 1`,
  `qk_rope_head_dim 64`, low-rank KV compression with per-head RoPE.
- **HDC / head-level chunking**: `hc_mult 4`, `hc_sinkhorn_iters 20` — chunked
  attention with sinkhorn-balancing; custom chunk attention + balancing kernel.
- **Sparse / hashed attention**: `num_hash_layers 3`, `sliding_window 128`,
  `compress_ratios` per layer, `index_topk 512`, `index_n_heads 64`,
  `index_head_dim 128` — hash-guided block selection kernels.
- **MoE**: `n_routed_experts 256`, `num_experts_per_tok 6` (top5+shared),
  `n_shared_experts 1`, `moe_intermediate_size 2048`, `expert_dtype fp4`,
  `topk_method noaux_tc`, `scoring_func sqrtsoftplus` — needs **FP4 expert
  GEMM** path + routing kernels.
- **DSPark speculative decoding**: `dspark_markov_rank 256`, `dspark_block_size 5`,
  `dspark_target_layer_ids [40,41,42]`, `dspark_noise_token_id` — a Markov-chain
  draft module with the main checkpoint; needs draft+verify pipeline.
- **nextn (MTP)**: `num_nextn_predict_layers 1`.
- FP8 e4m3 + ue8m0 scales, `weight_block_size [128,128]`.
- Chat encoding is custom (`encoding_dsv4.py`), not a Jinja template — the
  serving layer must load it as a recipe component.

## 6. Kernel reuse cross-model (reality check)

| Kernel family | qwen3_5 | qwen4_exp | deepseek_v4 |
|---|---|---|---|
| Recurrent state | **gated delta rule** | **gated delta rule (GDN)** | — (chunked) |
| Dense attention | paged FA | — | — |
| Sparse attention | — | QSA block | hash-guided windowed |
| Latent attention | — | — | MLA |
| Grouped MoE GEMM | — (dense MLP) | yes | yes (fp4 experts) |
| Spec decode | MTP | MTP3 | DSPark Markov |

The two Qwen models share the same gated-delta-rule kernel family (qwen3_5's
GatedDeltaNet and qwen4_exp's GDN), which lowers the `qwen4_exp` phase cost;
DeepSeek remains largely disjoint.

## 7. Open items

Resolved in P0 (see `05-audit-qwen3-5.md`):
- `qwen3_5` FP8 weight scale format -> **per-128x128-block inverse scales**
  (`weight_scale_inv`, BF16), dynamic activation.
- Exact linear-attention math -> **GatedDeltaNet** (gated delta rule), ported
  to `ref/qwen3_5.py`.
- `qwen3_5` MLP -> **dense** (no expert tensors in the checkpoint).
- MTP layer type -> **full attention** + dense MLP + fc.

Deferred (cluster-only, later phases):
- DeepSeek expert dtype fp4 layout in safetensors.
- Memory: peak activation/mem for `qwen4_exp` at long context under TP2
  (existing NV_ERR_NO_MEMORY history in `qwen38/docs`).
