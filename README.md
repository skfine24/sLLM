# sLLM

A from-scratch model serving engine for dual-node NVIDIA DGX Spark (GB10),
designed to serve three models, one at a time, under a single engine.

> **Naming**: the serving program is temporarily called **sLLM**; the
> project directory is `sLLM` (renamed from `dgxspark-serve`). Cluster
> deploy paths that still use `dgxspark-serve` refer to the prior checkout
> location.

Target models (recipes):

> Configuration split: `recipes/<ModelName>.yaml` = MODEL + launch info
> (`recipe_version/name/description`, execution options, `defaults:`,
> `env:`, `command:` template, weights location `paths.local_dir`);
> repo-root `config.env` = SLLM COMMON info (head/worker IPs, pair link,
> fallback port, `NVCC_ARCH`, tokenizer dir). Environment variables override
> `config.env` (see `env_config.py`).
>
> Build & run (Docker-first):
>
> ```bash
> deploy/run.sh build                                   # build the image
> sllm recipes/Qwen3.8-Flash-Next-FP8.yaml              # plan (tp from recipe)
> sllm recipes/Qwen2.5-Coder-0.5B.yaml --mode serve     # OpenAI API on :8002
> sllm recipes/Qwen3.8-27B-FP8.yaml --tp 1 --mode plan  # TP mode option
> ```
>
> `sllm <recipe> [--tp|--nodes 1|2] [--mode plan|run|serve] ...` —
> precedence CLI > recipe defaults > config.env. TP1/TP2 are first-class:
> tp.mode defaults from the recipe, `--tp 1` is accepted only when
> `defaults.weights_gib` provably fits one node (arithmetic, never silent).
> `serve` exposes the OpenAI-compatible API (`GET /v1/models`,
> `POST /v1/chat/completions`, `POST /v1/completions`, usage counts;
> `stream:true` explicitly rejected until SSE). Native equivalent:
> `python -m serving.main <recipe>` (same entry the container runs).

| Recipe | Model | Architecture | Weights |
|---|---|---|---|
| `Qwen3.8-27B-FP8.yaml` | Qwen/Qwen3.8-27B-FP8 | hybrid linear-attention (48/64 GDN + 16 full) + dense MLP + MTP + vision | ~29 GiB |
| `Qwen3.8-Flash-Next-FP8.yaml` | Qwen/Qwen3.8-Flash-Next-FP8 | GDN/DeltaNet (36/48) + QSA sparse (12/48) + MoE + MTP(1) + vision | ~173 GiB |
| `DeepSeek-V4-Flash-0731.yaml` | deepseek-ai/DeepSeek-V4-Flash-0731 | MLA + HDC/sparse + MoE(256, fp4) + DSPark spec decode | ~156 GiB |

Weights are `du -h` sizes on head (GiB; docs/design/08). Status: P0 done
(deploy path); qwen4_exp CPU oracle (Q1-Q3) + tooling done; GPU kernel and
TP2 execution milestones (B/C) await cluster access.

## P0 progress

- [x] Design documents (01–04)
- [x] Checkpoint audit — `docs/design/05-audit-qwen3-5.md`
      (linear attention = **GatedDeltaNet**, dense MLP, FP8 per-128x128 block
      inverse scales, MTP full-attention)
- [x] Recipe schema v1 + `recipes/Qwen3.8-27B-FP8.yaml`
- [x] Reference math `ref/qwen3_5.py` (numpy, torch-free) + T0 parity tests
      (`tests/test_ref_qwen3_5.py`, `tests/test_recipes.py`)
- [x] Loaders: FP8/BF16 decode + blocked dequant (`loaders/fp8.py`), minimal
      safetensors reader (`loaders/safetensors_reader.py`), checkpoint weight
      loader (`loaders/weights.py`)
- [x] Full-model reference forward (`ref/pipeline.py`), ready for T2
- [x] Real-shard validation: `bench/audit_shard_fp8.py` fetched the actual
      `layers-0.safetensors` over HTTP Range and proved header parse + F8_E4M3
      decode + 128x128-block dequant on real bytes (all blocks saturate near
      the fp8 ceiling, dequant magnitudes sane)
- [x] Real-weight layer T1: `bench/audit_real_forward.py` ran layer-0
      GatedDeltaNet and layer-3 full attention forwards on the actual
      checkpoint weights (chunked-vs-recurrent diff 4.5e-8, state shape
      (1,48,128,128), finite outputs)
- [x] Tokenizer: self-made byte-level BPE (`serving/bpe.py`) + wrapper with
      special-token splitting and chat-template rendering
      (`serving/tokenizer.py`) — encode matches the official `tokenizers`
      library on the real Qwen3.8-27B-FP8 tokenizer (unicode/Korean/emoji/
      specials/whitespace)
- [x] Dev serving stub (E2E on the dev machine):
      `runtime/sampler.py` (greedy/top-k/top-p), `serving/executor.py`
      (generation loop + InferenceEngine), `serving/server.py` (stdlib HTTP,
      /v1/chat/completions + /v1/completions + /health), `serving/cli.py`,
      `serving/dev_model.py` (real-vocab-size tiny model so the real tokenizer
      ids fit) — `python -m serving.cli --chat "hi"` runs chat-template ->
      tokenize -> forward -> sample -> detokenize end to end
- [x] Runtime bookkeeping (CPU-testable): `runtime/blocks.py` (KV block +
      recurrent-state allocators + HybridKVCoordinator), `runtime/memory_planner.py`
      (FP8 KV budget for qwen3_5 geometry), `runtime/scheduler.py`
      (continuous batching: admission control, chunked prefill, prefill->decode,
      finish->free->re-admit)
- [x] Continuous-batching E2E: `serving/executor.py::BatchedInferenceEngine`
      drives real generation from the scheduler (admission, chunked prefill,
      interleaved decode of concurrent requests); `python -m serving.cli
      --batch "a" "b"` serves several prompts with continuous batching
- [x] MTP + speculative decoding: `ref/mtp.py` (vLLM-derived MTP math:
      concat(embed,hidden)-fc + full-attn layer + norm + lm_head),
      `runtime/spec.py` (MTP draft + main-model greedy verify) — invariant
      test proves spec-decode output == plain greedy for num_draft 1..3
- [x] Other-model audits + skeleton recipes: `docs/design/06-audit-qwen4-exp.md`
      (qwen4_exp: 36 GDN + 12 QSA, hyper-connection, 512-expert MoE, PLE/ngram,
      hybrid MTP), `docs/design/07-audit-deepseek-v4.md` (deepseek_v4: MLA +
      sparse/hash, HDC, FP4 MoE, DSPark + 3-layer nextn); recipe schema
      extended (MoE MLP, qsa/mla block types, status/meta) and
      `recipes/Qwen3.8-Flash-Next-FP8.yaml` + `recipes/DeepSeek-V4-Flash-0731.yaml` skeletons parse
- [x] Standard dense-transformer support + REAL-MODEL operating test:
      `ref/standard.py` (Llama/Qwen2 family GQA + RoPE + plain RMSNorm + tied
      embeddings), `recipes/Qwen2.5-Coder-0.5B.yaml`,
      `serving/serve_standard.py`. Deployed dgxspark-serve to the DGX Spark
      head (ARM64), downloaded Qwen2.5-Coder-0.5B, ran the full test suite
      (105 OK on the cluster), one-shot real-weight generation, and a live
      HTTP serve (health + /v1/chat/completions + /v1/completions).
- [x] Cross-validation vs transformers (WSL): our engine's logits and greedy
      output match transformers exactly (logit diff <= 2e-4; token-identical
      generation across many prompts). Root cause of an apparent mismatch:
      transformers 5.16.1 `Qwen2Attention` hardcodes q/k/v `bias=True` while
      the checkpoint has `attention_bias=False`, leaving random-init biases;
      zeroing them makes the baselines agree -> our engine is checkpoint-correct.
- [x] Cluster layout recorded: `docs/design/08-cluster-layout.md` (head
      192.168.0.250/worker 192.168.0.231, shared `~/models/<model>`,
      vLLM coexistence rules). Recipes carry `paths.local_dir` per model.
- [x] Cluster continuous-batching operating test on the real 0.5B: 3 prompts
      served concurrently via `BatchedInferenceEngine` through
      `python -m serving.serve_standard --batch ...` (6.7s, outputs identical
      to transformers), test suite 106 OK on ARM64.
- [x] GPU kernels + containerised operation (`sllm-node`): `kernels/cuda/`
      self-contained CUDA kernels (RMSNorm, elwise add) built with nvcc
      (CUDA 13.0 baseline) into `sllm_gpu.so`; `deploy/Dockerfile`
      (multi-stage: nvcc build -> slim python runtime) builds the
      `sllm-node:latest` image (230MB); `deploy/run.sh` launches it with
      `--gpus all`. Verified on the cluster: kernels run on the GB10 GPU
      (rms_norm vs numpy ref 4.8e-7, device_count=1) and `nvidia-smi` shows
      the sllm-node `python` process using the GPU (11 MiB) alongside vLLM.
- [x] 109 tests pass (`tests/`)
- [x] Incremental decode (loop-carried runtime memory): `ref/incremental.py`
      keeps full-attn KV cache + GatedDeltaNet recurrent state + conv window
      across steps (`prefill` once, `decode_step` O(context) last-row attention).
      Standard (Llama/Qwen2) matches full recompute to ~1e-6 (tiny) / ~2.5e-4
      absolute logits (real 0.5B) — the known fp32 GEMM prefill-vs-decode seam
      (M=1 vs M=S contraction), same class as vLLM's; greedy is identical on the
      tiny model and on the real 0.5B for 12+ steps until a near-tie argument
      flips. Real-model operating test (head): incremental 200 tokens in 7.5 s
      (26.7 tok/s) vs recompute 4.0 tok/s at only 20 tokens. Generation stops
      at `max_position_embeddings` (context clamp). 117 tests OK;
      `bench/incremental_timing.py` shows the speedup growing with context
      (6.6x at 200 tokens on the tiny dev model).
- [x] Batch incremental decode: `BatchedInferenceEngine` keeps a per-sequence
      `IncrementalCache` (built on the first prefill action), so interleaved
      decode of concurrent requests is O(context) per step with no cross-talk.
      Batch output is identical to single-sequence generation (real 0.5B:
      3 concurrent prompts, 120 tokens total, 4.69 s; batch == sequential).
      Fixes an off-by-one in the first-decode step (prefill logits now select
      the first token; `decode_step` only for subsequent tokens) — the earlier
      real-model greedy mismatch was this bug, not the fp32 seam (now 0/38
      greedy mismatches on the real 0.5B). 121 tests OK.
- [x] QKV-bias correctness fix + repetition penalty + 0.5B game demo:
      Qwen2.5-Coder-0.5B ships q/k/v biases (72 tensors) that the standard
      path previously ignored → engine logits now match transformers exactly
      (cosine 1.0, top-5 identical). Sampler gained HF-style
      `repetition_penalty` (breaks greedy loops). First real-model artifact:
      `demo/snake_0.5b.html` (complete snake game generated by the 0.5B via
      the engine in ~42 s). **124 tests OK.**
- [ ] Cluster-only: full 64-layer qwen3_5 kernels, TP2, live parity vs vLLM
- [x] Operational track (user decision 2026-08-31): bring `qwen4_exp`
       (Qwen3.8-Flash-Next-FP8, FP8 173 GiB) to real serving on the TP2 pair
       (vLLM/sglang containers stopped for the live test window; parity
       target). Phase plan in `docs/design/09` as resequenced: A = local CPU,
       B = head-CPU corrections + risk probes, C = exclusive-GPU TP2.
       Phase A delivered:
       - A1 `qwen4_exp` runs through the executor/serving stack (CPU
         incremental): `Qwen4ExpCfg.from_recipe`, engine greedy == pipeline
         (`python -m serving.cli --qwen4 ...`)
       - A2 fp8-persistent streaming loader (`loaders/streaming.py`: mmap,
         F8_E4M3 stays raw, per-tensor dequant) + scale-aware TP2 sharding
         (`loaders/tp_shard.py`: head-granular splits, fp8 block-scale
         alignment PROVEN, MoE expert partition; shared expert replicates —
         640/2 is not 128-block-aligned)
       - A3 `tp/` module: cluster topology from the recipe, `Collectives`
         iface + `SimCollectives` (exact NCCL semantics locally),
         `RankWeightTable` (per-rank slice, shard-then-dequant memory contract)
       - A4 qwen4_exp CUDA kernel set (`kernels/cuda/qwen4.cu` +
         `kernels/_q4_cuda.py`): HC mix/combine, MoE router/swiglu/accum,
         QSA pool/MQA-logits/topk/sparse decode attention — oracle-parity
         tests (skip without .so; compile+run in milestone B)
       - A5 KV/state management: `qwen4_exp` budget planner (per-token QSA
         cache + ~113 MB/seq fp32 GDN state — admission must count both),
         amortized `Qwen4ExpState` growth (capacity doubling, exact views)
       - A6 MTP: `ref/qwen4_exp_mtp.py` oracle (hybrid 2-fc fusion, one-layer
         QSA+HC+MoE draft, hyper handoff) + greedy-identity spec loop
         (output == plain greedy bit-for-bit; batched verify = C-phase)
       - A7 milestone-B tooling, locally verified, execute-only in the
         window: `bench/q4_subset_parity.py` (Q3 real-checkpoint subset
         parity + bf16 noise floor + T1/T2 reference npz),
         `bench/probe_cublaslt_fp8.cu` (GB10 fp8 block-scale support
         matrix), `bench/probe_pair_link.sh` (pair bandwidth measurement)
       - **239 tests OK (33 skipped: GPU/cluster-only)**
- [x] qwen3_5 hybrid GPU kernels: `sllm_gated_delta_step` (GatedDeltaNet
      single-step decode recurrence) + `kernels/hybrid_decode.py` GPU driver
      (cuBLAS GEMMs + last-row attention + delta step + elwise). GPU decode
      logits match the numpy incremental decoder (argmax-equal) on the tiny
      hybrid model; 143 tests OK. Real 27B run needs a quiet GPU window.
- [x] GPU decode loop wired into serving: `ReferenceModel(use_gpu=...)` / env
      `SLLM_USE_GPU=1` / `--use-gpu` route every `decode_step` through the GPU
      kernels (standard + hybrid; fused `sllm_silu` added), with transparent
      numpy fallback. Real 0.5B GPU decode output == numpy output (0/24
      mismatches). Correctness milestone; the transfer design (per-op H2D/D2H +
      per-kernel sync, host-resident weights) is slower than CPU, so it is the
      fallback only — see device-resident decode below.
- [x] Device-resident GPU decode (standard path): `kernels/device_decode.py`
      uploads all weights once (`DeviceWeightTable`, free-memory-guarded) and
      keeps the attention KV on the GPU with capacity-doubling
      (`DeviceDecodeState`); every activation stays in reused on-device scratch
      and the loop synchronises **once per step** (logits + the step's appended
      KV rows copied back, so the host `IncrementalCache` stays authoritative
      and a runtime failure degrades to the transfer/numpy path). New kernels:
      GQA device attention (`sllm_attention_decode_dev`, capacity-stride),
      device RoPE, KV append/relayout, row-major cuBLAS GEMM. Real 0.5B
      `bench/gpu_decode_timing.py`: **89 tok/s vs 29 numpy / 0.4 transfer**,
       greedy-identical single + 3-prompt continuous batch. Env
       `SLLM_GPU_RESIDENT=0` forces the transfer path. **152 tests OK.**
- [x] General-purpose fused + dtype decode engine (for broad deployment): the
       device-resident path is no longer fp32 / full-rotary / bias-free only.
       Each decoder layer fuses to **11 device ops** (q|k|v and gate|up each one
       cuBLAS `GemmEx`, bias+RoPE, residual+RMSNorm and silu*mul fused); model
       dtype is selectable **fp32 | bf16** for weights/embedding/KV/GEMM
       operands (fp32 residual + logits, RNE-rounded, bit-exact for BF16
       checkpoints) via `--gpu-dtype` / `SLLM_GPU_DTYPE`; partial RoPE honoured
       via `recipe.rotary_dim()` across oracle + both GPU paths. Fixed three
       latent correctness bugs surfaced only by the widened matrix (q/k bias
       per-head offset, K-cache double-bias, partial-rotary bias tail). Real
       0.5B: fp32 teacher-forced max|Δlogit| **2.5e-5**, greedy `generate()`
       **identical to numpy** for fp32 and bf16; `bench/gpu_decode_timing.py`
       **fp32 58 / bf16 90 tok/s vs 29 numpy**, greedy-identical. **161 tests
       OK (head + local).**

## KV memory placement (recipe option)

| `memory.kv_placement` | Behaviour (GPU-backed runs) | OOM safety |
|---|---|---|
| `device` (default) | conventional all-on-GPU: weights + compute + KV | hard upper bound from the planned budget; admission rejects overflow so the node is never over-subscribed (GB10 can hang until power off otherwise) |
| `host` | KV / recurrent state in host RAM, gathered per decode step | recoverable process-level failure / swap |

- CPU/numpy mode always uses host RAM only, so the option is a no-op there.
- Backed by `runtime/placement.py`: `KVMemoryPlan` (sizing) + `KVBackend`
  (`HostKVBackend`/`DeviceKVBackend`); device placement degrades to the host
  backend gracefully when no GPU is present. CLI override:
  `serve_standard --kv-placement device|host`.
- GPU tensors: `kernels/cuda` adds a cuBLAS `sllm_gemm`, an O(S) last-row
  decode-attention kernel, and opaque device buffers
  (`kernels/_sllm_cuda.py` → `to_device`/`gemm`/`attention_decode`).
  Freely verified on the cluster: gemm and attention_decode match numpy
  (attention maxdiff 3.3e-7); on this busy GPU (vLLM holds it) `device` KV
  placement auto-degrades to host RAM via a `cudaMemGetInfo` free-memory guard —
  the over-subscription case that can hang GB10 is avoided by design.

## Container operation (sllm-node)

```bash
# build once (on the cluster)
deploy/run.sh build                      # -> sllm-node:latest (CUDA 13.0 base)
# run modes
deploy/run.sh kernel                     # GPU kernel smoke (image already built sllm_gpu.so)
SLLM_BATCH="p1;p2" deploy/run.sh batch   # continuous-batch one-shot on a standard model
deploy/run.sh serve                      # HTTP serving (mounts $HOME/models ro; MODEL_DIR=... override)
```
- Image/container name: `sllm-node`. Base: `nvidia/cuda:13.0.0-devel-ubuntu24.04`
  (build stage) + `python:3.12-slim` (runtime stage); kernels self-contained,
  cudart shipped into the runtime image.
- The kernel set currently covers norm/elwise; attention/GEMM kernels (cuBLAS
  reuse for dense GEMM per the design) are the next increment toward a full
  GPU forward of the 0.5B model.

Dev dependencies (tests/dev only): `ml_dtypes` (FP8 oracle), `tokenizers`/
`regex`/`jinja2` (tokenizer oracle/rendering).

## Design documents

- [`docs/design/01-architecture.md`](docs/design/01-architecture.md) — system architecture, data flow, directory layout
- [`docs/design/02-recipes-kernels.md`](docs/design/02-recipes-kernels.md) — recipe schema and per-model kernel inventory
- [`docs/design/03-runtime-tp.md`](docs/design/03-runtime-tp.md) — batching/scheduling, KV/memory, tensor parallelism
- [`docs/design/04-validation-phases.md`](docs/design/04-validation-phases.md) — parity methodology, perf methodology, phase plan P0-P5, risks

## Scope decisions (confirmed)

- Engine supports 3 model "recipes"; exactly one model is resident at a time.
- Development order: `qwen3_5` -> `qwen4_exp` -> `deepseek_v4`.
- Performance target: parity with the existing vLLM + FlashInfer stack on the
  same cluster.
- "Self-made kernels" scope: attention / recurrent-state / sparse / FP8 /
  fused kernels. Dense GEMM (cuBLAS) and communication (NCCL) are reused from
  the CUDA toolchain, matching common industry practice.
