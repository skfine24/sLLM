# 03 — Runtime and Tensor Parallelism

## 1. Scheduler (continuous batching)

- Admission control: accept a new sequence only when blocks and compute budget
  allow; otherwise queue.
- Two-mode execution loop:
  - **Prefill** (chunked): process a sequence in chunks up to a max prefill
    chunk (`chunked_prefill_size`), so a long prompt does not monopolize the
    GPU (matches the existing stack's behavior).
  - **Decode**: one forward step per sequence per scheduler tick (decode slab).
- Scheduling policy (initial): FCFS with chunked prefill; later switch to a
  length-aware / priority heuristic only after the baseline is validated in P3.
- Preemption: swap-out supported at block granularity; state save/restore for
  recurrent modules must be included (recurrent state is per-sequence and is
  not paged like attention KV). First version may forbid preemption and rely on
  admission control; preemption is a stretch item.

## 2. KV / recurrent-state memory

Two distinct state kinds, mirroring the hybrid nature of the target models:

- **Attention KV (paged)**: dense or sparse attention KV in pinned blocks,
  `kv_dtype` FP8, block-level page table per sequence. FP8 conversion happens
  at write time (dynamic quant), dequant at read time — matching FlashInfer's
  FP8 KV conventions where the recipe opts in.
- **Recurrent state (hybrid manager)**: for `qwen3_5` linear attention and
  `qwen4_exp` GDN layers. Recurrent state is kept per sequence and lives in a
  state pool, not in the attention page table. The block manager exposes both
  allocators behind one API (modeled on the Map/Mamba-style split proven in
  the existing vLLM stack).

Lifecycle invariants:

- Allocate happens only after memory planning (below); never oversized.
- A block/state slot is returned before a finishing sequence is deleted.
- Preemption and reload must free all per-sequence state.

## 3. Memory planner

- Profile run at engine start (like vLLM `memory_profiling`): measure
  weights + activation peak, then derive available KV bytes, applying a
  configurable utilization factor.
- Manual override supported (`kv_cache_memory_bytes`) to reproduce cluster
  scenarios, and needed because the GB10 unified-memory pool + cgroup cap
  (110 GB in the existing environment) can force driver-level
  `NV_ERR_NO_MEMORY` when over-subscribed (see `qwen38/docs/gb10-um-analysis.md`).
- The planner must report expected headroom per recipe before launch and fail
  clearly if the fleet leaves no safe margin (no silent over-subscription).

## 4. Tensor parallelism (TP2)

- World = 2 ranks (head = rank 0, worker = rank 1). NCCL init over the two
  nodes; a run guard refuses to launch if a conflicting engine occupies the
  pair (same spirit as the existing `vllm_qwen38_*` guard).
- Sharding (per recipe):
  - Column-parallel linear: split out-dim across ranks (row halves for
    attention QKV and MLP in-proj).
  - Row-parallel linear: split in-dim, all-reduce the output.
  - FP8 weights are sharded together with their scale tensors; scales are
    part of the shard layout, not recomputed after splitting.
  - Embedding / LM head: row-split + all-reduce logits (or reduce-scatter
    partial logits when fused sampling is enabled later).
- Recurrent-state layers shard cleanly across ranks because the state
  dimension is the feature dimension (state is not sequence-global), which is
  one reason the hybrid layout is feasible on TP2.
- All-reduce points are fused with the following GEMM where possible
  (two all-reduces per transformer layer in the naive case: one after QKV/MLP
  column-parallel, one after the row-parallel output).

## 5. Model reload / pool

- Exactly one recipe resident. A reload request (`model=A` -> `model=B`):
  1. Drain in-flight sequences (graceful, configurable timeout).
  2. Free weights + state + KV.
  3. Run memory planner for B, load + shard B, re-init NCCL buffers if layout
     uses different graph.
- Reload is a cold path: target seconds-to-minutes for the 31 GB model, and
  explicitly not real-time for the big two. The API exposes the resident model
  and reload progress.

## 6. Serving layer

- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, streaming SSE,
  `model` field selects the recipe.
- Chat templating per recipe. `deepseek_v4` uses a custom encoder
  (`encoding_dsv4.py`) instead of a Jinja template — the serving layer keeps a
  pluggable "message encoding" interface, with Jinja and script-mode
  implementations.
- Regional concern (existing cluster uses a head/worker pair with a stable
  node topology): the HTTP server binds on the head node; the worker is a
  compute-only rank reached over NCCL.

## 7. Concurrency model

- Single GPU worker thread owns the CUDA stream and the scheduler loop; API
  handlers enqueue requests (no shared mutable state across the GPU loop).
- Async token streaming from a dedicated deque consumed by the SSE writers.
- Enforced by tests: two streams of concurrent requests must never touch the
  same block/state slot (allocator invariants unit-tested on CPU).

## 8. Hybrid state coordinator

For `qwen4_exp` (and partially `qwen3_5`) the engine needs the dual bookkeeping
proven by the existing vLLM `HybridKVCacheCoordinator`:

- Separate allocators for attention blocks and recurrent-state slots.
- A layout function that maps (layer, token) -> contiguous slot, so kernels
  read state with a fixed stride instead of per-token indirection.
- Unit-test the coordinator on CPU (pure bookkeeping) before any kernel work.
