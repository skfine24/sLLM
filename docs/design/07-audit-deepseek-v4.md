# 07 — Checkpoint Audit: deepseek-ai/DeepSeek-V4-Flash-0731 (arch deepseek_v4)

Evidence record derived from `config.json` + `model.safetensors.index.json`
of `deepseek-ai/DeepSeek-V4-Flash-0731` (304 B params, ~166 GB).

## 1. Top level

- `model_type=deepseek_v4`, 43 layers, hidden 4096, vocab 129280, max_seq
  1,048,576, yarn RoPE (factor 16, theta 1e4).
- Weights ≈ 166 GB (fits dual-node TP2 with ~45 GB headroom/node).

## 2. Attention = MLA + sparse (all 43 layers, dense layers)

- **MLA**: 64 heads / 1 KV head, head_dim 512,
  `q_lora_rank / o_lora_rank 1024`, `qk_rope_head_dim 64`.
- Weight inventory per layer (`attn.*`): `wq_a/wq_b` (q lora up/down),
  `wkv` (compressed KV), `wo_a/wo_b` (o lora up/down), `q_norm`,
  `kv_norm`, `attn_sink` (a learned sink/decay state), plus `_scale_inv` for
  FP8. (2 tensors each = weight + scale.)
- **Sparse compression**: `num_hash_layers 3`, `sliding_window 128`,
  `compress_ratios` per layer (values 4/128 alternating; 0 at the first two
  and tail layers), `compress_rope_theta 160000`, `index_n_heads 64`,
  `index_head_dim 128`, `index_topk 512`.
- **HDC (head-level chunking)**: per-layer `hc_attn_*` / `hc_ffn_*`
  (base/fn/scale) plus model-level `hc_head_base/fn/scale`;
  `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps 1e-6`, `o_groups 8`.

## 3. MoE (all 43 layers)

- `n_routed_experts 256`, `num_experts_per_tok 6`, `n_shared_experts 1`,
  `moe_intermediate_size 2048`, `expert_dtype fp4`,
  `topk_method noaux_tc`, `scoring_func sqrtsoftplus`,
  `routed_scaling_factor 1.5`, `norm_topk_prob true`,
  `swiglu_limit 10.0`.
- Per layer (`ffn.*`): `experts` (1536 = 256x6 tensors incl. fp4/fp8
  scales), `gate`, `shared_experts` (6).

## 4. nextn (MTP)

- `num_nextn_predict_layers 1` but the checkpoint carries **three** MTP
  blocks `mtp.0 / mtp.1 / mtp.2`, each a full layer (attn + ffn + hc) with
  `main_norm` + `main_proj` to the main hidden; `mtp.2` contains the
  **DSPark markov_head** (`markov_w1`, `markov_w2`).

## 5. DSPark speculative decoding

- `dspark_markov_rank 256`, `dspark_block_size 5`,
  `dspark_target_layer_ids [40, 41, 42]`, `dspark_noise_token_id 128799`.
- The Markov draft head lives in the MTP stack (`mtp.2.markov_head.*`),
  matching the vLLM changelog ("DSpark speculative decoding for
  DeepSeek-V4").

## 6. FP8 layout

- e4m3 dynamic, `scale_fmt ue8m0`, `weight_block_size [128,128]`;
  expert weights are **fp4** (`expert_dtype fp4`). Distinct from Qwen's
  `weight_scale_inv` BF16 convention, so the loader needs an FP4 + ue8m0
  path.

## 7. Consequences

- No recurrent/linear layers: the whole model is dense-attention-family
  (MLA). The hybrid KV/recurrent bookkeeping is NOT needed for deepseek_v4;
  instead sparse-MLA page tables + HDC chunking are required.
- New engine work: MLA (lora compression + per-head RoPE), sparse/hash-guided
  windowed attention, HDC sinkhorn chunking, FP4 MoE, DSPark Markov draft +
  verify (block-wise), MTP 3-layer.
- Loader: contrast with Qwen (BF16 `weight_scale_inv`) — needs fp4 decode and
  ue8m0 scales.
- Tokenizer/encoding: custom `encoding_dsv4.py` (not a Jinja chat_template).
