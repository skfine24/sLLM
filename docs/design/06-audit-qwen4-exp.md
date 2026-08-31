# 06 — Checkpoint Audit: Qwen/Qwen3.8-Flash-Next-FP8 (arch qwen4_exp)

Evidence record derived from `config.json`, `model.safetensors.index.json`, and
safetensors headers of `Qwen/Qwen3.8-Flash-Next-FP8` (not assumptions).

## 1. Top level

- `model_type=qwen4_exp`, multimodal (vision). Text config nested
  (`qwen4_exp_text`); `transformers_version 5.8.0.dev0`.
- Weights ≈ 174 GB (dual-node TP2 fits; matches the deployed
  vLLM `qwen38/doc` numbers).

## 2. Layer schedule (48 layers)

`12 x ( 3 x linear_attention + full_attention )` — i.e. 36 GatedDeltaNet
linear-attention layers and 12 Qwen Sparse Attention (QSA) full-attention
layers. Verified: explicit `layer_types` list; `full_attention_interval 4`.

## 3. Linear attention = GatedDeltaNet (same family as qwen3_5)

| knob | value |
|---|---|
| num_key_heads / head_k_dim | 16 / 128 (key_dim 2048) |
| num_value_heads / head_v_dim | 48 / 128 (value_dim 6144) |
| conv_kernel_size | 4 |
| state dtype | float32 |
| output_gate | sigmoid |

Weight inventory per linear layer (index): `linear_attn.{A_log, conv1d,
dt_bias, in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, norm, out_proj}` —
identical structure to `qwen3_5`. Kernel reuse with qwen3_5 is therefore
direct (delta-rule family).

## 4. QSA full attention (12 layers)

- 24 heads, 2 KV heads, head_dim 256, partial M-RoPE (0.25).
- **indexer** (per-layer, 3 tensors): `indexer.*` with knobs `indexer_n_heads 4,
  indexer_kv_heads 1, indexer_head_dim 128, indexer_budget 2048,
  indexer_compress_ratio 4` — token-block sparse attention selector.
- **hc / hyperchunking**: `hc_count 4`, `hc_lowrank 320` (head-level chunking
  low-rank), `heads_per_ngram 8`.
- Per-layer `self_attn.{q,k,v,o_proj, q_norm, k_norm, indexer.*}` audited.

## 5. Hyper-connection (new vs qwen3_5)

Both a model-level `hyper_connection_mixer` (hc_norm + input_mix_weight_up/
down) and per-layer `attn_hyper_connection` / `mlp_hyper_connection`
(`block_inject_weight`, `hc_norm`, `input_mix_weight_up/down`). These are
exempt from FP8 (`modules_to_not_convert`). This is a residual-stream mixer
(streaming-style, cf. DeepSeek hyperconnection concept); the mixer is a
required engine component for qwen4_exp.

## 6. MoE MLP (all 48 layers)

- 512 routed experts, 10 per token, `moe_intermediate_size 640`,
  `shared_expert_intermediate_size 640`, 1 shared expert,
  `output_router_logits` (router `mlp.gate`).
- Per layer: `mlp.experts` (3072 = 512x6 tensors incl. FP8 scales),
  `mlp.gate`, `mlp.shared_expert{,_gate}`.

## 7. Speculative: PLE + ngram + MTP

- **PLE** (`ple_layer_ids [2]`): `layers.1.ple.*` (conv1d, key_proj,
  norm_conv/norm_key/norm_query) + `ple_embedding.ngram_embedding` sharded
  (ngram_size 3, ngram_vocab_size_base 20M split into 128 shards,
  heads_per_ngram 8).
- **MTP**: 1 full-attention layer, `hybrid: true`, but with an MTP-level
  `hyper_connection_mixer` and its own MoE (`mtp.layers.0.*`, 3094 tensors);
  `mtp.fc_embedding.weight` + `mtp.fc_hidden.weight` (two 2H->H fcs, instead of
  qwen3_5's single `mtp.fc`).

## 8. FP8 layout

`quantization_config`: `fp8 / e4m3 / dynamic`, `weight_block_size [128,128]`,
`weight_per_tensor false` (per-block, like qwen3_5's `weight_scale_inv`).
Exempt (`modules_to_not_convert`): lm_head, embed, hyper-connection mixers,
per-layer attn/mlp hyper-connection, PLE, and more.

## 9. Vision

Same shape as qwen3_5's VP170 (27 blocks, hidden 1152, 16 heads, patch 16,
temporal 2, position 2304) but `out_hidden_size 2560`.

## 10. Consequences

- qwen3_5 delta-rule kernel + loader/FP8 infraststruction reuse is near-total
  for the linear layers.
- New engine work: QSA indexer + sparse paged attention, hyper-connection
  mixer, MoE (512 experts) routing/grouped GEMM, PLE/ngram + MTP
  (hybrid, MoE) speculative path.
