# 05 — Checkpoint Audit: Qwen/Qwen3.8-27B-FP8 (arch qwen3_5)

Evidence record for P0. Everything below is derived from the live artifacts
(config.json, model.safetensors.index.json, and safetensors shard headers) of
`Qwen/Qwen3.8-27B-FP8`, not from assumptions.

Sources:
- `config.json` (51 KB), `model.safetensors.index.json` (137 KB)
- safetensors headers of `layers-0.safetensors`, `layers-3.safetensors`
- upstream `transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
  (5.8.0.dev0 era)

## 1. Architecture (config)

- `model_type=qwen3_5`, `Qwen3_5ForConditionalGeneration` (multimodal).
- Text: hidden 5120, 64 layers, 24 attn heads, 4 KV heads, head_dim 256,
  vocab 248320, max_seq 262144, M-RoPE (`section [11,11,10]`, interleaved,
  `partial_rotary_factor 0.25`, `theta 1e7`).
- Layer schedule: `3001 pattern repeated 16x` => 48 linear_attention +
  16 full_attention (every 4th layer).
- `attn_output_gate: true`, `rms_norm_eps 1e-6`, `mamba_ssm_dtype float32`.
- MTP: `mtp_num_hidden_layers 1`.
- Vision: `depth 27`, hidden 1152, 16 heads, patch 16, temporal_patch 2,
  spatial_merge 2, intermediate 4304, pos_emb 2304, out_hidden 5120.

## 2. Linear attention is a GatedDeltaNet (NOT Mamba)

FACT (upstream `modeling_qwen3_5.py`, class `Qwen3_5GatedDeltaNet`): the
"linear_attention" block is a gated delta-rule (FLA-style Gate-dDeltaNet), not
a Mamba selective scan.

Math (upstream):
- `in_proj_qkv` hidden -> (key*2+value); depthwise causal `conv1d` (groups=C,
  kernel 4, bias=False) + SiLU on the QKV channels; then split into
  query/key (num_k_heads, head_k_dim) / value (num_v_heads, head_v_dim).
- `beta = sigmoid(in_proj_b(x))`; `g = -exp(A_log) * softplus(in_proj_a(x) +
  dt_bias)` (decay in log space, negative).
- query/key are L2-normalized (fp32, eps 1e-6); query pre-scaled by
  `head_k_dim**-0.5`. When `num_v_heads > num_k_heads`, q/k are
  repeat_interleaved to `num_v_heads`.
- Recurrent state `(B, num_v_heads, head_k_dim, head_v_dim)`, float32.
- Output gate: `rms_norm_gated(core, weight, z)` where
  `z = silu(in_proj_z(x))` (swish output gate), then `out_proj`.

Config numbers: num_k_heads 16 x head_k_dim 128 (key_dim 2048); num_v_heads 48
x head_v_dim 128 (value_dim 6144).

## 3. MLP is DENSE (not MoE)

FACT (index): every layer has exactly `mlp.gate_proj`, `mlp.up_proj`,
`mlp.down_proj` (+ `weight_scale_inv`). There are **no** `mlp.gate` /
`shared_expert_gate` / `experts.*` tensors, despite those names appearing in
`quantization_config.modules_to_not_convert` (that list is a blanket template
and includes dead names such as `linear_attn.in_proj_ba` which do not exist in
the checkpoint). Decision: the recipe records `mlp.type: dense`.

## 4. FP8 layout (RESOLVED open item)

FACT (safetensors headers):

| tensor | dtype | shape |
|---|---|---|
| `layers.N.linear_attn.in_proj_qkv.weight` | F8_E4M3 | [10240, 5120] |
| `...weight_scale_inv` | BF16 | [80, 40] |
| `layers.N.mlp.gate_proj.weight` | F8_E4M3 | [17408, 5120] |
| `...weight_scale_inv` | BF16 | [136, 40] |
| `layers.N.self_attn.q_proj.weight` | F8_E4M3 | [12288, 5120] |
| `...weight_scale_inv` | BF16 | [96, 40] |
| `layers.N.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] |
| `layers.N.self_attn.q_norm.weight` | BF16 | [256] |

- Weights are stored as e4m3 with **per-128x128-block inverse scales**
  (`weight_scale_inv`, BF16 values): 10240/80 = 128, 5120/40 = 128,
  17408/136 = 128, 12288/96 = 128.
- `activation_scheme: dynamic` => activations are quantized at runtime
  (not stored).
- Non-quantized (str) names: conv1d, A_log, dt_bias, norms, embeddings,
  lm_head, vision patch embed / first ViT blocks per the exemption list.
- lm_head, mtp.fc carry no `weight_scale_inv` => BF16.

## 5. Full attention structure

FACT (headers + code): `q_proj` out = heads*head_dim*2 = 12288
(Q + output gate from one projection); `k_proj`/`v_proj` out = kv_heads*head_dim
= 1024; per-head `q_norm`/`k_norm` (RMSNorm, `(1+weight)` form); partial rotary
on the first `head_dim * partial_rotary_factor = 64` dims; attention output
multiplied by `sigmoid(gate)` before `o_proj`.

## 6. MTP structure

FACT (index): MTP = 1 layer of **full attention** (self_attn) + dense MLP,
plus `mtp.fc`, `mtp.norm`, `mtp.pre_fc_norm_embedding`,
`mtp.pre_fc_norm_hidden`. MTP GEMMs are FP8 (have `weight_scale_inv`);
`mtp.fc` is BF16.

## 7. Vision

FACT (index/config): patch_embed (Conv3d) + pos_embed + 27 blocks + merger;
per-block tensors `norm1/norm2/attn.qkv/attn.proj/mlp.linear_fc1/linear_fc2` ->
12 tensors x 27 = 324. First four ViT blocks and patch_embed are exempt from
FP8 (BF16) per `modules_to_not_convert`.

## 8. Status

- Resolved by this audit: FP8 scale format (per-128x128 block, dynamic
  activation), MLP dense-vs-MoE, MTP attention type, GatedDeltaNet semantics.
- Still open (cluster-only): real weight loading, tokenizer, vision preprocessing,
  and numeric parity against the live vLLM baseline.
