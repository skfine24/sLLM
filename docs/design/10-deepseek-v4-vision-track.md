# 10. DeepSeek-V4-Flash-Vision-Exp Track (D+V)

- Date: 2026-08-31
- Status: approved / in progress
- Scope: full multimodal support (text backbone + vision), numpy-CPU oracle first, CUDA later
- Reference checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (MIT, 168 GB, 48 shards)
- Vendored reference: `ref/hf_sources/dsv4/` (README, config.json, encode, image_processor,
  model.py, vision.py, convert.py, kernel.py, generate.py, tests, examples)

## 1. Checkpoint facts (audited from config.json + model.safetensors.index.json + shard headers)

- Total 72,633 tensors; **the checkpoint already uses inference-ready naming** (no `model.`
  prefix; `layers.N.attn.*`, `layers.N.ffn.*`, `vision.*`, `aligner.*`, `embed.weight`,
  `head.weight`, `mtp.*`). `weight_scale_inv` is already `scale`.
- Dtype split: text weights FP8-E4M3 (scales **F8_E8M0**), experts **FP4-E2M1 packed as
  int8**, vision tower + aligner + embed + lm_head **BF16**.
- Shard layout: shard 1 (~2 GB) holds the ENTIRE vision tower (259 tensors, all BF16) +
  aligner + `embed.weight`. Shard 45 holds `hc_head_*`, `head.weight`, `norm.weight`
  (tail). shard 46+ has `F8_E8M0` scales (`mtp.0.attn.wkv.scale`).
- Size: `total_size 167811372792` -> ~156.3 GiB; TP2 => ~78.2 GiB/rank (within the
  110 GiB/node SLLM_NODE_WEIGHT_BUDGET_GIB); KV/cache headroom is tight.

## 2. Architecture (pinned from ref/hf_sources/dsv4/model.py)

Text backbone (`deepseek_v4`):
- hyper-dim 4096, 43 layers, vocab 129280, head_dim 512, rope_head_dim 64, q_lora_rank
  1024, o_lora_rank 1024, o_groups 8, window_size 128, compress_ratios per layer
  (0,0,4,128,...), max_seq 1,048,576 (YaRN factor 16, original 65536, compress_rope_theta
  160000), rms_norm_eps 1e-20.
- MLA attention: `wq_a -> q_norm -> wq_b`, q rms-normed + rotated (last 64 dims); `wkv ->
  kv_norm -> rotate`; non-rope KV dims FP8-simulated (64-block, ue8m0); sliding-window
  128 + compressed KV (ratio 4 -> learned Indexer; ratio 128 -> positional) + sink bias;
  `sparse_attn` (gather top-k KV rows, online softmax); grouped `wo_a` (8 groups, LoRA
  rank 1024) + `wo_b`.
- Compressor: wkv/wgate(dim -> 2*head_dim), ape, gated pooling, overlap when ratio==4,
  incremental decode state (kv_state/score_state), rotate (Hadamard) for indexer only.
- Indexer: index_n_heads 64 x index_head_dim 128, index_topk 512, wq_b + weights_proj.
- MoE: gate (score_func sqrtsoftplus; first 3 layers hash-routed via tid2eid; vision
  bias_vl path), 256 routed + 1 shared, 6 active, moe_inter 2048, swiglu_limit 10,
  routed_scaling_factor 1.5, norm_topk_prob; **expert weights FP4 (E2M1, block-32)**.
- Hyper-Connections: hc_mult 4, sinkhorn 20 iters, hc_eps 1e-6; gated hc-pre/post;
  hc_head (sigmoid pre) -> final norm -> ParallelHead.
- DSPark: 3 target layers (40-42), block_size 5, markov_rank 256, main_proj, markov
  w1/w2 heads, confidence head, forward_spec next-token prediction.

Vision:
- ViT (SigLIP-2 style, no pos-embed): PatchEmbed = Linear(3*14^2=588 -> 1024) over
  14x14 patches; 32 blocks; Attention wqkv(1024->3*1024)/wo, 2D RoPE (rope_dim = 32,
  theta 10000) on q,k; MLP w1(1024->2*2816) silu(up? no: silu(gate)*up)/w2; RMSNorm.
- Aligner: w1(1024*9 -> 4096) + GELU + w2(4096->4096); downsample_ratio 3 unfold.
- Image processor: patch 14, down 3, max_n_token 384, min_pixels 147456, max_wh_ratio 8;
  5 sentinel token ids (image_start/pad/newline/end at vocab 129280+) + per-heart
  learnable `image_*` embeddings; N-layout grid; interleaved multi-image.
- The 4 text/image `visible` window math (`get_image_visible` / `get_window_topk_idxs_visible`)
  makes image spans visible to narrow windows; image blocks MUST be prefilled in one chunk.

## 3. Track plan (L0..L5) and current progress

- **L0 — Ingest/audit (DONE):** vendored sources; pinned arch + loader facts above.
  - Gap: F8_E8M0 dtype missing from `loaders/safetensors_reader.py`; no fp4/e8m0 math;
    `weight_scale_inv` hardcoded in `loaders/weights.py`/`streaming.py`.
- **L1 — Loader groundwork (DONE):**
  - `loaders/safetensors_reader.py`: added `F8_E8M0` (ue8m0) dtype (1 byte) + raw map
    in `streaming.py`.
  - `loaders/fp8.py`: `decode_ue8m0` (2^(x-127), 0x00->0, 0xFF->nan), `dequant_fp8_mxfp_weight`
    (E4M3+E8M0, 128-blk), `dequant_fp4_packed_weight` (packed E2M1 nibbles + E8M0 32-blk on
    K, `FP4_TABLE`), `dequant_weight_auto` (fp8-vs-fp4 inferred from scale geometry).
  - `loaders/weights.py`/`streaming.py`: quant companion is parameterized by
    `scale_suffix` (Qwen `_scale_inv` APPENDS; DeepSeek `.scale` REPLACES `.weight`),
    format auto-detected. Qwen path green.
  - `tests/_synth.write_tiny_deepseek_checkpoint` (fp8+ue8m0 / fp4-packed layout),
    `tests/test_loaders_deepseek.py` (14 cases: ue8m0 table, fp8/fp4 dequant vs manual
    loops, auto dispatch, reader dtype).
- **L2 — Text backbone oracle (DONE):** `ref/deepseek_v4.py` numpy port (MLA window +
  Compressor (overlap ratio-4) + Indexer + MoE + Hyper-Connections sinkhorn + DSPark
  target-hidden collection); prefill==incremental-continuation invariant verified
  (incl. ring-wrap positions); `serving/dev_model.py` tiny_deepseek_v4 dev model/recipe/
  engine; `serving/executor.py` `deepseek_v4` arch branches. Tests: `tests/test_deepseek_v4.py`.
  - NOTE (parity): exact float equality at ring positions where the 4-ratio overlap
    window coincides (pos == 16k-1) still shows ~2e-4 drift; reserved for l5 goldens.
  - DSPark spec `mtp.*` (markov/confidence) is NOT in the serving path (l5 speculative
    loop); main_hiddens are collected at the target layers.
- **L3 — Vision pipeline (DONE, oracle + serving wire-up):** `serving/image_processor.py`
  numpy port (Pillow decode-only; OpenAI blob/url/path; min-pixels/max-wh-ratio;
  N-layout grid); `ref/vision_deepseek.py` ViT + Aligner numpy (2D interleaved rope,
  full-head rotation, 3x downsample); `DeepseekV4Model` optional `images=` prefill
  (merge_image_embeddings + `get_image_visible` + `window_topk_visible`); decode rejects
  image sentinels (one-chunk invariant). Serving wire-up: `serving/encoding_dsv4.py`
  (adopted reference encoder: OpenAI content blocks -> prompt text + image records),
  `VisionSpec` (schema.py) DeepSeek SigLIP fields, `InferenceEngine.vl_chat_detail`
  routing (content-block messages -> expand placeholders -> `generate(images=...)`),
  `dev_model.py` TinyVLTokenizer + `build_dev_deepseek_v4_vl_engine`. Tests:
  `tests/test_vision_deepseek.py` (processor, encoder adoption, ViT/aligner, multimodal
  prefill, VL engine E2E via image_url content blocks).
  - Remaining (l5/cluster): exact bf16 image pipeline, live OpenAI server image tests
    against the real checkpoint, `IMAGE_TAG_PATTERN` path<->data-uri uniformity.
- **L4 — Memory & placement (DONE):** `runtime/memory_planner.py` `deepseek_bytes_per_token`
  / `deepseek_seq_state_bytes` / `deepseek_plan` (MLA window row + compressed + indexer
  stream per token; compressor buffers per seq); `serving/main.py` plan rendering for
  `deepseek_v4`.
- **L5 — GPU/cluster (deferred):** `kernels/cuda/deepseek.cu`, `_deepseek_cuda.py`
  (MLA device decode, fp4-MoE GEMM, ViT/apply-position kernels, DSPark spec), TP2
  RankWeightTable DeepSeek sharding, streaming Range convert, run.sh/entrypoint, goldens
  via torch on cluster; exact QAT (+Hadamard `rotate_activation`) parity;
  DSPark spec-decode acceleration; ground the loc-31 float residual.

## 4. Key decisions (approved)

- numpy-CPU oracle first for all of L1-L4; CUDA only in L5.
- Todo track runs in parallel with the existing qwen4_exp (Track Q); DeepSeek-V4-Flash
  Vision is the priority when the two conflict.
- Image decoding uses Pillow (dev + cluster image must add Pillow); all subsequent
  preprocessing is numpy.
- DSPark is implemented as full forward (target layers + main_hidden collection); the
  speculative acceleration loop is deferred to L5.
- Tokenizer compat reads HF `tokenizer.json` (BPE merged model) by reusing
  `BPETokenizer`.
- Large audit artifacts (model.safetensors.index.json etc.) live under
  `$TEMP/opencode/dsv4`; only distilled facts are committed.

## 5. Risks / notes

- FP8/FP4 numerics are QAT-matched: unquantized paths must reproduce
  `act_quant(..., round_scale=True, inplace=True)` (fp8) and fp4-act-quant in the oracle
  for parity; the fp8 "simulation" of non-rope KV dims is required for exact logits.
- Hadamard rotation (`rotate_activation`) for the indexer/compressor has no numpy oracle
  yet; implement an exact Walsh-Hadamard transform in ref (scale = dim^-0.5).
- Hyper-Connections sinkhorn (20 iters on hc_mult=4) plus 256-expert fp4 MoE make the
  numpy oracle slow; parity uses tiny-synth geometries; per-layer goldens generated on
  the cluster in L5.
- `image_visible` window math and the one-chunk prefill invariant constrain the batched
  scheduler (prefill chunks for an image turn must not split image blocks).
- 156 GiB at TP2 => ~78 GiB/rank: GPU-resident weights + KV budget must be re-planned in
  L4 (LF immediately; NVLink pair headroom low).
- legacy `mitigation`: model.safetensors.index.json asserts complete shard sets; our
  loader reuse `CheckpointIndex` for full-file validation.

## 6. Out of scope (this track)

- Training/RL, DSpark inference-time speculative acceleration, tools/agents, TP3+,
  tensor sharding of the vision tower (grid stays; only aligner/wo etc. shard), and any
  other V4 modal (audio/video).
