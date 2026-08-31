# 01 — System Architecture (sLLM)

The serving program is temporarily named **sLLM** (project directory
`sLLM`, renamed from `dgxspark-serve`).

## 1. Goals

- Serve Qwen3.8-27B, Qwen3.8-Flash-Next, and DeepSeek-V4-Flash on a dedicated
  dual-node DGX Spark (GB10, ARM64, CUDA 13.x) pair.
- One engine, three model "recipes"; exactly one model loaded at a time
  (resident model pool with reload).
- Tensor-parallel 2 (TP2) across the two nodes via NCCL.
- Performance parity (throughput / TTFT) with the existing vLLM + FlashInfer
  0.6.17 stack on the same hardware.
- Self-made kernels for every part that is not a dense GEMM: recurrent-state,
  linear attention, sparse attention, FP8 paths, fused ops, sampling.

## 2. Hardware resource model

- Per node: 128 GB unified memory, no swap, single ARM64 SoC (GB10).
- Two nodes connected as a peer pair (existing cluster: head `10.100.25.1`,
  worker `10.100.25.2`).

Memory budget (TP2, weights split 50/50 across nodes):

| Recipe | Weights total | Weights/node | Headroom/node (128 GB) |
|---|---|---|---|
| `qwen3_5` | ~31 GB | ~16 GB | ~112 GB |
| `qwen4_exp` | ~174 GB | ~87 GB | ~41 GB |
| `deepseek_v4` | ~166 GB | ~83 GB | ~45 GB |

Consequence: the three models cannot be resident simultaneously (sum > 256 GB).
The engine therefore owns exactly one model at a time and exposes a model
switch (reload) rather than concurrent multi-model residency.

## 3. Data flow (per request)

```
client
  -> OpenAI-compatible API (HTTP/SSE)
  -> tokenizer + chat template  (serving/)
  -> continuous-batch scheduler (runtime/)   : admission, prefill/decode split,
                                                chunked prefill, preemption policy
  -> executor (model kernel)   (kernels/ + ref/)
     model forward -> attention/recurrent-state -> MoE/MLP -> logits
     stats: block allocator, sampling, spec-decode (MTP / DSPark)
  -> output tokens -> de-tokenize -> stream back to client
```

Data owned by the engine:

| Data | Owner | Notes |
|---|---|---|
| Paged KV / recurrent state | runtime/ block allocator | per-sequence blocks, hybrid state coordinator |
| Weights (FP8) | loaders/ | loaded per recipe, sharded across TP2 |
| Active sequence set | runtime/ scheduler | states: running / preempted / finished |
| Model config + kernel map | recipes/ | declarative recipe -> kernel dispatch |

## 4. Directory layout

```
recipes/         # 3 recipe files: structure, kernel map, loader, TP plan
runtime/         # batching scheduler, prefill/decode split, KV page allocator,
                 #   memory planner, preemption
tp/              # NCCL world init, weight sharding, fused all-reduce points,
                 #   dual-node launch
kernels/         # CUDA kernels: linear/recurrent scan, full attention,
                 #   sparse attention, FP8 quant/dequant, MoE grouped GEMM,
                 #   fused norms/gates, sampling
ref/             # PyTorch reference math (numeric-parity harness source of truth)
serving/         # HTTP API, SSE streaming, tokenizer, chat templates
loaders/         # safetensors + FP8 weight loader + weight sharding
bench/           # parity harness + perf bench (reuses cluster bench harness)
tests/           # unit + parity tests (CPU-runnable where possible)
```

## 5. Key design principles

- **Reference-first**: `ref/` is the golden numeric implementation, ported
  verbatim from the upstream modeling code, and every kernel has a parity test
  against it before it is trusted.
- **Recipe as kernel map**: no hard-coded model in the engine. The recipe
  declares which kernels run per layer type, so adding a model is adding a
  recipe + the missing kernels.
- **One-time shared infra**: tokenizer/API/scheduler/KV-planner/loader plumbing
  is model-agnostic and built once.
- **cuBLAS + NCCL reuse**: dense GEMM and cross-node comms are the only
  non-self-made pieces (industry norm; even NVIDIA does not rebuild cuBLAS).
- **No silent behavior change**: FP8 dynamic quantization, weight scale
  format, and non-quantized module lists are read from each model's
  `config.json` / index, never assumed.

## 6. Launch model (target)

Dual-node launch follows the proven cluster pattern (head + worker via a
wrapper script, rank 0 = head, rank 1 = worker), but the engine owns its own
NCCL init (no vLLM). Deployment artifacts (container, SSH, env) are modeled on
`vllm_qwen38_* .sh` / `vllm-docker-um` conventions but are out of scope until
Phase P3.
