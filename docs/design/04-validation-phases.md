# 04 — Validation Strategy and Phase Plan

## 1. Numerical parity methodology

Source of truth: a PyTorch reference in `ref/`, ported to match the upstream
modeling code (transformers `modeling_qwen3_5.py`, `qwen4_exp`, DeepSeek
`deepseek_v4`) exactly for this checkpoint. Kernels are trusted only after
parity checks against this reference.

Tiers:

| Tier | Scope | Tolerance / Gate |
|---|---|---|
| T0 kernel | single kernel vs its reference slice | structured error, per-kernel (e.g. rel error < 1e-3 for FP8 paths) |
| T1 layer | one layer end-to-end vs reference layer | logit diff below FP8 noise floor |
| T2 model | greedy decode, short context vs reference | token-identical (reference in BF16/F32) |
| T3 live | full prompt samples vs vLLM on the same cluster | token-identical or documented diff policy |

Rules:

- Reference math for `qwen3_5` linear attention is extracted in P0 from upstream
  code (roles of `A_log`, `dt_bias`, `in_proj_a/b/ba`, conv, norm, gate
  ordering are taken from the code, not guessed).
- All FP8 paths compare against a BF16 reference with a documented tolerance
  budget; the tolerance is measured, not assumed.
- Every parity test is CPU-runnable where hardware-independent (reference vs a
  mocked kernel), so the suite runs on the dev machine and in CI without a GPU
  window.

## 2. Performance methodology (parity target)

Baseline: the existing vLLM + FlashInfer 0.6.17 stack serving the same recipe
on the same cluster pair (the `qwen38` / `vllm-docker-um` artifacts preserve
the exact launch and image). Comparison metrics:

- TTFT (time to first token) across context lengths (short .. long).
- Throughput (tokens/s) under a concurrent workload (existing `bench/`
  harnesses are reused/extended which already cover concurrency and mem-limit
  A/B).
- Memory headroom at the deployed `gpu_memory_utilization` / cgroup settings.
- Decode latency percentiles (P50/P95) — page-fault / state-launch stalls are
  an explicit risk area for recurrent/state kernels.

Gate: recipe X is "parity" when throughput within +-10% and TTFT within +-20%
of the vLLM baseline at the same max-concurrency and context mix.

## 3. Phase plan

| Phase | Scope | Exit criteria |
|---|---|---|
| **P0** | Reference extraction (`modeling_qwen3_5.py` math to `ref/`), parity harness, recipe schema v1, project skeleton, memory/FP8-scale audit of the checkpoint | T0/T1 pass for `qwen3_5` core kernels; recipe documents load; open items in `02-recipes-kernels.md` resolved or explicitly deferred |
| **P1** | Single-node eager model: `qwen3_5` FP8, greedy, short context, no batching | T2 pass (token-identical vs reference) |
| **P2** | Single-node serving: scheduler, paged KV, prefill/decode split, OpenAI API, streaming, MTP | T3 pass; concurrent multi-request correctness; allocator invariants green |
| **P3** | Dual-node TP2 + performance tuning | Perf parity gates (section 2) on `qwen3_5`; `qwen4_exp`/`deepseek_v4` recipes validated for load + memory planning |
| **P4** | `qwen4_exp` kernels: GDN delta-rule scan + QSA sparse paged attention + hybrid coordinator + MTP3 | T2/T3 parity for Flash-Next; perf parity |
| **P5** | `deepseek_v4` kernels: MLA, HDC chunk attention, hash/windowed sparse, fp4 MoE, DSPark spec decode, custom message encoder | T2/T3 parity for DeepSeek; perf parity |

Each phase ends with (a) parity evidence and (b) regression evidence that prior
phases still pass. No phase implies a cluster GPU window unless it says so:
P0-P1 are designed to be mostly verifiable on the dev machine (CPU parity) and
only the list checkpoints consume approved GPU-free windows.

## 4. Regression rules

- The full CPU parity suite runs before and after every phase; no kernel change
  may silently alter earlier-tier behavior.
- Reload/pool behavior is tested on CPU-only mode (drain/free/re-plan) before
  it is ever run with real weights.
- Failure paths are exercised: OOM planning must fail clearly (no
  over-subscription), mid-stream reload must either finish or return a clean
  error, and end-of-generation must return state to the allocator.

## 5. Risks

| Risk | Mitigation |
|---|---|
| `qwen3_5` linear-attention math ambiguity | P0 reference extraction from upstream code; defer to code-not-guess |
| FP8 scale format unknown (`weight_block_size`) | P0 weight-header audit |
| Recurrent kernel perf (scan) below FlashInfer/vLLM | chunked parallel scan kernel; fused recurrence; benchmark early (T0) rather than at the end |
| GB10 unified-memory ceiling / cgroup cap (NV_ERR_NO_MEMORY history) | explicit planner headroom checks; respect existing mem-limit A/B discipline |
| `qwen4_exp` / `deepseek_v4` kernel development cost | schedule acknowledges near-zero cross-model kernel reuse; phases are sequential by design |
| Reload downtime for large recipes | cold-reload path documented; not real-time by design |
| Spec-decode (MTP3 / DSPark) parity | implement after base model parity; draft acceptance logic only after throughput gate |

## 6. Repository conventions

- Documentation in English; user-facing summaries in Korean.
- Code comments in English; identifiers/APIs in original form (no translation).
- No code exists yet — this design is the approval artifact for P0.
