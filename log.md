# log.md

## 2026-08-31 - Project directory renamed: dgxspark-serve -> sLLM

### Summary
Renamed `pc_app/dgx_spark/dgxspark-serve` to `pc_app/dgx_spark/sLLM`
(program name = directory name). No code change needed (everything is
HERE/`__file__`-relative); updated the operative naming notes in README
and docs/design/01, and flagged the head-side deploy path (docs/design/08:
re-sync `/home/sklee/dgxspark-serve` -> `/home/sklee/sLLM` at next deploy).
Historical mentions in log.md/README checklist left intact.
Validation: full suite 262 OK (skipped 33) + `sllm --dry` + native plan all
run identically from the new path.

## 2026-08-31 - Full-code inspection: P1/P2/P3 fix round (launcher wiring, deploy path, gates)

### Summary
Two-axis read-only audit (Python core + deploy/launcher/recipes/tests) found
and fixed, in priority order:
P1 (broken paths):
- `serving/main.py` run mode passed a raw string to `chat()` (list-of-dicts
  contract; crashed every engine) -> wrap `[{"role":"user","content":...}]`;
  regression test with engine stub.
- `sllm`: `--chat` silently dropped in the `command:`-template branch and
  `--model-dir` never reached the container (recipe `~` expanded against
  the container HOME -> README's own serve example failed). Now both are
  ALWAYS appended (template-rendered command + CLI-suffix merge, no flag
  duplication), weights dir mounted at the same path, recipes outside the
  repo are mounted in. Port precedence fixed to CLI > recipe > config.env
  (was inverted; config.env used to beat recipe `defaults.port`). TTY-aware
  `-i -t`, PyYAML/port validation, recipe path realpath.
- `_check_nodes`: nodes<1 and non-numeric `weights_gib` now exit cleanly
  (was ZeroDivisionError/ValueError tracebacks). `resolve()` coerces
  nodes/port with named exits. `build_plan` consumes `defaults.weights_gib`,
  skips the 2-rank table on TP1 override, prints topology fallback IPs.
- `serve_standard.load_standard_engine` annotation fixed (3-tuple).
- `ref/qwen4_exp_mtp.py`: draft loop could emit `max_new+1` tokens when a
  draft was accepted exactly at the limit (stale `pending` re-emitted the
  just-accepted token). Guard added; boundary test pins the violation seeds
  (2, 14 - verified: old code overshoots, new code is greedy-identical).
P2 (deploy path):
- `env_source.sh` (new): shell-side loader that does NOT clobber real env
  vars (plain `. config.env` overwrote them, violating the documented
  precedence in sllm/entrypoint/run.sh/build.sh/probe_pair_link.sh - all
  migrated), tolerates CRLF. Verified: env 9999 wins over file 8002.
- `deploy/run.sh`: stale `/home/sklee/models` mount + port 8003 removed;
  now sources config.env, `$HOME/models` mount, per-model MODEL_DIR, SLLM_PORT.
- `deploy/Dockerfile`: runtime image now ships libcublas/libcublasLt (the
  kernels link them; dlopen would fail without); NVCC_ARCH as build ARG
  (docker build has no GPU -> default sm_121 instead of native); stage 1
  sees config.env/env_source.sh. NOTE: sm_121 validity + ldd cleanliness
  ASSUMPTION - confirm at B gate `docker build` + `ldd sllm_gpu.so`.
- `deploy/entrypoint.sh`: NCCL_SOCKET_IFNAME defaults from SLLM_PAIR_IFACE
  (the documented-but-unimplemented mapping). `config.env`: added SLLM_IMAGE,
  SLLM_NODE_WEIGHT_BUDGET_GIB, removed fictitious SLLM_ROLE note.
  Root `.dockerignore` added (build context is the repo root; the old
  deploy/.dockerignore was inert - removed).
P3 (truth/tests/docs):
- Size truth unified on `du -h` GiB facts from docs/design/08: field renamed
  `weights_gib` = 173 (q4) / 29 (27B) / 156 (DSv4; docs/07 166 GB decimal) /
  0.95 (0.5B); 173-vs-174 and 156-vs-166 splits were decimal-GB vs GiB unit
  mixing.
- 27B recipe `status: partial` (ready = runnable engine; skeleton = structure
  only) + status-vocabulary test; `TPSpec.size` default 2->1 (launcher and
  schema now agree); recipes note that defaults.max_*/kv_utilization are
  planning values until C-phase admission wiring.
- server: RuntimeError (missing chat template) now returns JSON 400;
  `--max-new` CLI flag actually works (server.default_max_new).
- tests: test_server TOK_DIR via Q27B_TOKENIZER_DIR chain (was a hardcoded
  dev-machine path); test_envconfig isolated from operator-exported SLLM_*;
  unused imports (main.sys, dev_model.os, bench ReferenceModel), truncated
  debug print, tiny-standard dispatch wired to `build_dev_standard_engine`.

### Files Changed
- `serving/main.py`, `serving/server.py`, `serving/serve_standard.py`,
  `serving/dev_model.py`, `serving/executor.py`, `sllm`, `env_source.sh`
  (new), `.dockerignore` (new), `deploy/run.sh`, `deploy/Dockerfile`,
  `deploy/entrypoint.sh`, `deploy/.dockerignore` (removed - inert),
  `config.env`, `recipes/*.yaml` (weights_gib/status/placeholder notes),
  `recipes/schema.py` (TP default), `ref/qwen4_exp_mtp.py`,
  `loaders/streaming.py`, `tp/rank_table.py`, `kernels/cuda/build.sh`,
  `bench/probe_pair_link.sh`, `bench/gpu_decode_timing.py`,
  tests: `test_main_cli.py`, `test_q4_mtp.py`, `test_recipes.py`,
  `test_server.py`, `test_envconfig.py`, README

### Validation
- Full suite 262 OK (skipped 33). MTP boundary test proven non-vacuous
  (old code overshoots on pinned seeds; new code passes 72 combos).
- `bash -n` all shell files; env_source precedence live-tested;
  `sllm --dry` matrix: plan/serve/run + tp1 override + chat quoting.
- Cluster-side to confirm at B gate (ASSUMPTIONs): sm_121 flag valid for
  CUDA 13, `ldd sllm_gpu.so` clean in the runtime image, cuBLAS runtime
  lib names match the copied globs.

## 2026-08-31 - sLLM: TP1/TP2 as selectable mode + full OpenAI-compatible serve output

### Summary
1. TP mode is now an option, not a hard recipe lock: `--tp|--nodes 1|2`
   (`serving/main.py`, `sllm`). Default = recipe `tp.size`; a LOWER value is
   accepted only when the recipe's new `defaults.weights_gb` fact fits one
   node (`weights_gb/nodes <= SLLM_NODE_WEIGHT_BUDGET_GB`, default 110 GB of
   the 128 GB coherent pool) and prints the arithmetic; otherwise the
   rejection names the numbers (27B tp1 allowed, qwen4_exp tp1 rejected at
   174/nodes > 110). No silent fallback in either direction.
2. Weights-size facts added to all recipe defaults: 174 (q4, docs/06),
   166 (DSv4, docs/07), 31 (27B, est. — comment says verify with du), 1
   (0.5B). q4 plan/recipe text corrected 173 -> 174 per audit.
3. `--mode serve`: OpenAI-compatible HTTP output finalized in
   `serving/server.py`: `GET /v1/models`, served `model` name (was
   hardcoded "dev-tiny"; requested model echoed back), `usage`
   {prompt/completion/total}_tokens, `finish_reason`, and `stream: true`
   explicitly rejected with 400 (never a fake non-SSE 200). Engine dispatch
   shared by run/serve via `_build_engine()` (qwen4_exp real weights and
   unimplemented arches fail loudly).
4. Latent bug fixed: `run_model` previously called `.chat()` on the tuple
   returned by `load_standard_engine` (would have crashed standard-arch
   one-shot runs); now takes element [0] via `_build_engine`.

### Files Changed
- `serving/main.py` (serve mode, `_build_engine`, TP-mode arithmetic
  validation, --tp alias), `serving/server.py` (OpenAI schema), `sllm`
  (--tp alias, usage text), four `recipes/*.yaml` (`weights_gb`), README,
  `tests/test_main_cli.py` (+6: TP1-fit rule, q4 tp1 rejection numbers,
  --tp alias, /v1/models, chat.completion usage/shape, stream-400)

### Validation
- Full suite 259 OK (skipped 33) incl. real HTTP round-trips against the
  dev engine; `bash -n` + `--dry` launcher smoke (`--tp 1 --mode serve`).
- Not validated (blocked): TP2 execution and serving real checkpoints
  (C phase); this entry ships plan/serve wiring + honest gates.

## 2026-08-31 - sLLM: launch-style recipes (named by model) + `sllm <recipe>` launcher

### Summary
Adopted the reference recipe format (user-supplied vLLM launcher example)
and the `[binary] [recipe]` interface:
1. Recipes are now FILES NAMED BY MODEL:
   `Qwen3.8-Flash-Next-FP8.yaml`, `Qwen3.8-27B-FP8.yaml`,
   `DeepSeek-V4-Flash-0731.yaml`, `Qwen2.5-Coder-0.5B.yaml`; all code/test
   references updated (docs/ and log.md keep historical names).
2. Recipe schema gained an optional launch section (validated, NOT meta):
   `recipe_version`, `name` (falls back to model_id), `description`,
   `container`, `defaults:` (host/port/nodes/tensor_parallel/max_model_len/
   max_num_seqs/max_num_batched_tokens/kv_utilization), `env:` (string map,
   injected as -e), `command:` template rendered via `render_command()`
   ({recipe} {nodes} {port} {host} {mode} {max_new} + all defaults).
   Consistency guards: defaults.tensor_parallel == tp.size; defaults.nodes
   >= tp.size. All old recipes/flows keep parsing (fields optional).
3. `sllm` (repo-root launcher, Docker-first): resolves CLI > recipe
   defaults > config.env; mounts the recipe's weights dir read-only; applies
   recipe env; runs the rendered command template inside `sllm-node:latest`
   (built by `deploy/run.sh build`, whose Dockerfile now calls build.sh so
   the image contains qwen4.cu); `--dry` prints the exact docker line.
   1-node vs 2-node drive = `--nodes N`, enforced against tp.size in
   `serving/main.py` (`python -m serving.main <recipe> [--nodes N]
   [--mode plan|run]` is also the native entry; plan prints rank table,
   cache bytes, weights presence; run is honestly gated: dev/tiny engines
   execute, real qwen4_exp weights raise until C, never silently degrade).
4. Data fix with evidence: `Qwen2.5-Coder-0.5B` tp.size 2 -> 1 (the standard
   engine is single-device and P0 ran it on one GPU; the new launcher would
   otherwise have forced TP2 on a 0.5B model).

### Files Changed
- `sllm` (new), `serving/main.py` (new this entry: recipe-driven entry),
  `recipes/schema.py`, all four `recipes/*.yaml` (renamed + launch section),
  references updated in 13 files, `tests/test_main_cli.py` (new, 9 tests)

### Validation
- Full suite 253 OK (skipped 33); launcher `bash -n` + `--dry` E2E for all
  four recipes (correct ports/env/mounts/template rendering; LF endings).

## 2026-08-31 - sLLM: config.env (cluster/toolchain common config) separated from recipes

### Summary
User-directed configuration split: `config.env` (repo root, tracked) holds
SLLM COMMON info only - head/worker external + pair IPs, pair NIC name,
serving port, `NVCC_ARCH`, shared tokenizer snapshot dir. Recipes keep MODEL
info including the weights location (`paths.local_dir` unchanged - recipes
are already model-only documents). Precedence everywhere: real env var >
config.env > code default (empty value = unset).
- `env_config.py` (root): dependency-free KEY=VALUE reader (`get`,
  `get_path` tilde-expanded, `get_int`, `SLLM_ENV_FILE` override).
- Consumers wired: `tp/topology.py` (SLLM_HEAD/WORKER[_PAIR]_IP replace the
  hard-coded constants, docs/08 values kept as fallbacks),
  `serving/dev_model.py` (Q27B_TOKENIZER_DIR: env > config.env > old dev
  default - Windows default preserved), `kernels/cuda/build.sh` (NVCC_ARCH,
  now accepts bare arch values like `native`/`sm_121` or a full flag),
  `deploy/entrypoint.sh` + `bench/probe_pair_link.sh` (source config.env;
  probe defaults PEER/IFACE from it).
- `bench/q4_subset_parity.py`: `--model-dir` now OPTIONAL, defaults to the
  recipe's `paths.local_dir` (expanduser) - run on head is just
  `python bench/q4_subset_parity.py`.

### Files Changed
- `config.env`, `env_config.py` (new), `tp/topology.py`,
  `serving/dev_model.py`, `kernels/cuda/build.sh`, `deploy/entrypoint.sh`,
  `bench/probe_pair_link.sh`, `bench/q4_subset_parity.py`,
  `recipes/qwen4_exp.yaml` (header comment only), `tests/test_envconfig.py`
  (new, 6 tests), README

### Reason
Transfer prep: head/worker settings must be edited in ONE obvious file
without touching model documents; per-node differences (NIC name, tokenizer
dir) must not require code changes.

### Validation
- Full suite 245 OK (skipped 33); new: parser/precedence/`SLLM_ENV_FILE`,
  recipes keep local_dir for all 4 models, config.env carries no model keys,
  topology precedence proven via SUBPROCESS (first in-process-reload version
  broke `test_tp_module` class identity - lesson: never module-reload shared
  classes inside the suite).
- `bash -n` clean on all three touched scripts; bench E2E smoke PASS.

## 2026-08-31 - sLLM: pre-deploy full-code inspection (head/worker transfer gate)

### Summary
Full static review before shipping to the cluster (no nvcc locally, so the
CUDA sources got line-by-line reading + structural checks instead of a
compile). Findings and fixes:
1. `qwen4.cu`: all 14 host wrappers re-read against `kernels/_q4_cuda.py`
   argtypes (14/14 match). Added input guards that were missing and could
   corrupt memory or produce negative-index writes: qsa_topk now rejects
   `end <= start` (padding loop could write before the row), qsa_pool_block
   rejects `end < ratio` (negative row index), gemma_rmsnorm / hc_mix_apply
   / hc_combine / sparse_attn / mqa got rows>0 / dim>0 / start>=0 guards.
   Dead-comment cleanup (gemv post-op note, E<=2048). No symbol collisions
   with kernels.cu (both compile into one .so; kernel names disjoint).
   Brace/paren balance verified programmatically.
2. `bench/probe_cublaslt_fp8.cu`: BUG FIX - heuristic requested a 32 MB
   workspace but Matmul was called with (nullptr, 0): algorithms needing
   workspace would fail at runtime on head. Now allocates and passes it.
   `#ifdef` branch brace artifact investigated and confirmed benign.
3. `deploy/Dockerfile`: STALE - still built `kernels.cu` only (no qwen4.cu,
   no cublas link) -> container .so would lack all sllm_q4_* symbols. Now
   calls `kernels/cuda/build.sh` (single source of truth).
4. `tests/test_loaders.py`: unguarded top-level `import ml_dtypes` would
   crash discovery on nodes without it -> module-level SkipTest.
5. `kernels/_q4_cuda.py`: host-side bounds guards (the C side cannot see
   host row counts): mqa/topk range validation, sparse_attn k/v shape and
   max-slot vs kcap checks, pool_block 0 < ratio <= end; pool_block now
   flattens (end,1,d) inputs so the C (end*d) contract can't be violated.
6. Hygiene verified: both .sh files + deploy scripts are LF (CRLF would
   break Linux), `bash -n` clean, `compileall` clean on every package, no
   TODO/FIXME in shipped code, no non-ASCII besides pre-existing em-dashes
   (valid UTF-8). Hard-coded `TOK_DIR` Windows dev paths exist but are all
   behind `skipUnless(isdir)` (they skip on the cluster); the runtime knob
   is `Q27B_TOKENIZER_DIR`. Runtime deps = numpy + pyyaml only
   (torch/ml_dtypes are test-oracle only; Dockerfile installs numpy pyyaml
   regex jinja2).

### Files Changed
- `kernels/cuda/qwen4.cu`, `kernels/_q4_cuda.py`,
  `bench/probe_cublaslt_fp8.cu`, `deploy/Dockerfile`,
  `tests/test_loaders.py`

### Reason
User gate: full inspection before transferring to head/worker. The two
real bugs (Dockerfile missing qwen4.cu; probe workspace nullptr) would
each have burned exclusive-cluster-window time.

### Validation
- Full suite 239 OK (skipped 33) after all edits; balance script clean;
  qwen4.cu semantic parity still pinned by the 9 GPU-gated tests (they
  exercise exactly the guarded paths once built).

### Residual risks (documented, accepted for v1)
- qwen4.cu error paths leak device allocations (fatal anyway; C1 rewrites
  the host layer).
- moe/topk v1 kernels are serial-on-thread0 (correctness-first).
- First real nvcc compile remains the known open gate at B.

## 2026-08-31 - sLLM: A7 milestone-B tooling (Q3 subset-parity bench + fp8 GEMM probe + pair-link probe)

### Summary
Built (and locally verified) the tools the cluster window will run, so B is
execute-only, not write-under-pressure:
1. `bench/q4_subset_parity.py` (Q3): loads the REAL qwen4_exp checkpoint
   through `loaders/streaming` (mmap + per-tensor fp8 dequant), picks a
   subset (default first linear_attention + first full_attention, PLE
   layers excluded, guard explicitly cleared and labeled), drives the
   numpy pipeline, and reports  (a) determinism (bit-identical reruns),
   (b) the noise floor fp32-dequant oracle vs bf16-cast weights (rel_std +
   argmax agreement — the floor future GEMM engines must beat), and saves
   logits .npz as the T1/T2 reference for the live sglang comparison.
   Prints fp32 footprint up front (real geometry needs vLLM stopped).
   Bug found while wiring: numpy native-dtype loads are memmap VIEWS, so
   `close()` did not release shard handles — bench now materializes before
   close (Windows surfaced it; same lifetime bug would exist on Linux).
2. `tests/_synth.write_q4_dev_fixture` + `tests/make_q4_fixture.py`: full
   dev-q4 weights as a real on-disk safetensors checkpoint (index.json),
   enabling end-to-end bench runs on any machine.
3. `bench/probe_cublaslt_fp8.cu` (B2): support matrix for FP8 E4M3 GEMM on
   GB10 — scalar device scales baseline + block-scale modes
   (VEC128_32F / BLK128x128_32F, #ifdef-guarded for older toolkits) using
   identity scales so support detection needs no layout assumptions;
   verify against host fp8 CPU reference. Compile+run in the window.
4. `bench/probe_pair_link.sh` (B3): ethtool + ping latency + iperf3 -P3 on
   the 10.100.25.x pair + documented nccl-tests invocation focused on the
   TP2 message size class (~10 KB/layer-step) — replaces the link-bandwidth
   ASSUMPTION with a measurement row in log.md.

### Files Changed
- `bench/q4_subset_parity.py`, `bench/probe_cublaslt_fp8.cu`,
  `bench/probe_pair_link.sh` (new), `tests/_synth.py`,
  `tests/make_q4_fixture.py` (new), `tests/test_q4_bench.py` (new, 2 tests)

### Reason
Phase B is gated on a cluster window; every minute there should be probe
execution and record-keeping, not development. Local verification via the
F32 fixture proves loader->driver identity (bench logits == in-memory
pipeline, bit-for-bit).

### Validation
- Full suite 239 OK (skipped 33). New gates: bench-on-fixture reproduces
  the in-memory pipeline EXACTLY (loader path adds zero error) + real
  recipe subset selection excludes PLE (12/36 layers, ple=[2]).
- E2E CLI smoke: make_q4_fixture -> q4_subset_parity --tiny: PASS
  (determinism true, noise-floor stats sane, npz written, exit 0).
- NOT verified (no local nvcc/bash target): probe_cublaslt_fp8.cu compile,
  probe_pair_link.sh execution — first cluster window.

## 2026-08-31 - sLLM: A6 qwen4_exp MTP oracle (hybrid 2-fc draft) + greedy-identity spec loop

### Summary
Ported the qwen4_exp MTP draft head (`oracle/upstream/sglang/qwen4_exp_mtp.py`)
to the numpy oracle stack (user question "mtp 지원은?"):
1. `ref/qwen4_exp_mtp.py`: the hc_count>1 HYBRID 2-fc input fusion
   (`fc_embedding(gemma_norm(embeds)) + fc_hidden(gemma_norm(hyper).view
   (S,hc,H))` broadcast back to hc*H) with the hc_count<=1 standard single-fc
   fallback; the draft model itself = one-layer Qwen4ExpModel (QSA+HC+MoE,
   no PLE) run through the existing pipeline driver via `mtp.*` -> driver
   key remapping; embed/lm_head shared with main (upstream parity).
   `generate_greedy_mtp`: v1 draft/verify loop (num_draft chain, draft rope
   positions CONTINUE the main context, fresh draft state per cycle),
   returns acceptance telemetry. By construction every emitted token is the
   main model's argmax -> output identical to plain greedy.
2. `ref/qwen4_exp_pipeline.py`: `_forward` gained `hyper_in` (single-token
   draft injection; S>1 guarded NotImplementedError = C-phase draft-extend)
   and `return_hyper` (the pre-final-combine hc*H tensor upstream feeds as
   `spec_info.hidden_states`); public `prefill(return_hyper)`,
   `decode_step_full`. Fixed a latent coordinate conflation in
   `_qsa_block_step`: rope position (absolute) vs QSA slot window (the
   layer's OWN cache rows) are now separate — equal for the main model
   (existing Q2/engine tests still bit-identical), required for a
   draft state seeded mid-context.
3. `serving/dev_model.py`: layer emission extracted to `_q4_emit_layer`
   (no key/shape change) + `tiny_qwen4_exp_mtp_weights`.

### Files Changed
- `ref/qwen4_exp_mtp.py` (new), `ref/qwen4_exp_pipeline.py`,
  `serving/dev_model.py`, `tests/test_q4_mtp.py` (new, 6 tests)

### Reason
MTP was the last unported oracle (recipe `mtp.enabled`/`spec.mtp_hybrid`);
its semantics (2-fc fusion, hyper handoff, position continuation, own-cache
slotting) must be pinned BEFORE the GPU C-phase so spec decode there is a
port, not an invention.

### Validation
- Full suite 237 OK (skipped 33). New: fusion formula (both variants),
  draft step shapes/position chaining, weights-map overwrite, and THE gate:
  `generate_greedy_mtp == generate_greedy` exactly across 3 seeds +
  num_draft=1, with the accept path exercised (accepted > 0).
- Existing greedy/prefix/parity gates re-prove the main path is
  bit-unchanged by the slot-coordinate refactor.

### Remaining
Executor/engine spec-decode wiring + batched multi-token verification
(the part that actually accelerates) = C-phase; PLE/ngram stays guarded.

## 2026-08-31 - sLLM: A5 qwen4_exp KV/state management (budget planner + amortized state growth)

### Summary
Closed the local-buildable half of qwen4_exp KV management (user question
"kv 관리 부분은?"):
1. Sizing/budget (`runtime/memory_planner.py`): `qwen4_exp_bytes_per_token`
   (per QSA layer: dense KV k+v + indexer token stream + 1/ratio compressed
   keys; GDN contributes 0 per-token), `qwen4_exp_seq_state_bytes` (GDN
   recurrent state is PER-SEQUENCE: real geometry = ~113 MB/seq fp32 x 36
   layers — admission that counts only tokens would badly over-admit),
   `qwen4_exp_plan` (max concurrent sequences x avg context under a byte
   budget). C2 scheduler admission will consume these.
2. Growth amortization (`ref/qwen4_exp_pipeline.py`): decode steps rebuilt
   k/v/tok_k/ck with `np.concatenate` every step (O(S) copy per token per
   layer — ~100 MB/token at S=2k). New `_append_axis` keeps a private
   capacity-doubling buffer with `st[key]` remaining an EXACT-length view
   (semantics of every reader/test unchanged). Prefill ck assembly switched
   from incremental concat to collect-then-stack (was O(nb^2) per layer).
   `Qwen4ExpState.state_bytes()` exposes live bytes (incl. capacity) for
   future admission/eviction accounting.

### Files Changed
- `runtime/memory_planner.py`, `ref/qwen4_exp_pipeline.py`,
  `tests/test_q4_memory.py` (new, 6 tests)

### Reason
qwen4_exp state had zero memory management: no budget arithmetic anywhere
(QSA per-token cost AND the large fixed per-sequence GDN state were
invisible to planning) and pathological per-step reallocation.

### Validation
- Full suite 231 OK (skipped 33). Decode-output correctness under the
  growth path is pinned by pre-existing gates: engine greedy == pipeline
  (bit-identical) and prefill/decode parity + indexer prefix stability.
- New tests: planner formulas (self-consistent), real-recipe magnitudes
  (fp8 KV+indexer 12x1184 B/tok; state 100-130 MB/seq fp32), plan counts
  state (token-only budgeting provably over-admits), append semantics
  (exact view contents/dtype, monotone capacity, amortized caps),
  state_bytes tracking.

### Remaining (C-phase, not local-buildable)
Device-resident KV for qwen4_exp (C1, mirrors kernels/device_decode.py
capacity-doubling on GPU), fp8 KV quantized storage (halves/quarters the
per-token term), scheduler admission wired to `qwen4_exp_plan` +
`state_bytes` (C2), eviction/preemption policy.

## 2026-08-31 - sLLM: A4 qwen4_exp CUDA kernel set (decode-path v1) + bindings + parity tests

### Summary
Ported the qwen4_exp decode-path math to CUDA as `kernels/cuda/qwen4.cu`,
every entry a direct port of a `ref/qwen4_exp.py` oracle (which cites
`oracle/upstream/sglang/*`): grouped GemmaRMSNorm, HC mix (GEMV rows with
fused silu(x/hc)/sigmoid postops + mean-of-branches apply) and HC combine
(inject dot + 2*sigmoid gated residual add), MoE router (softmax -> stable
top-k -> renorm eps 1e-20, matching the oracle's stable-descending tie
order), swiglu, per-row weighted expert accum (axpy) and sigmoid-gated
shared-expert accum, QSA indexer chain (average-pool block, MQA relu logits,
stable smem top-k with -1 padding) and sparse GQA decode attention over
selected slots (gather rows from the (kvh, cap, hd) caches). API style
matches the transfer-era host-pointer pattern of kernels.cu so each kernel
is parity-testable in isolation; device-resident composition (one sync/step,
typed operands, radix fast-topk for >2048-block rows, fp8 GEMM) is C1 and
reuses the same `__global__` bodies. GDN decode reuses the existing
`sllm_gated_delta_step` (same family, already GPU-verified).
- `kernels/_q4_cuda.py`: separate lazy bindings module so a pre-A4
  `sllm_gpu.so` keeps working (probes for `sllm_q4_*` symbols).
- `kernels/cuda/build.sh`: now compiles `kernels.cu qwen4.cu` into one .so.

### Files Changed
- `kernels/cuda/qwen4.cu` (new), `kernels/cuda/build.sh`, `kernels/_q4_cuda.py`
  (new), `tests/test_qwen4_kernels.py` (new, 9 tests)

### Reason
Operational-track Phase A4: get the qwen4_exp kernel semantics written and
parity-gated BEFORE the exclusive-GPU milestones, so C1 composes verified
pieces instead of inventing math under time pressure.

### Validation
- Full suite: 225 OK (skipped 33: no GPU/.so) — the 9 kernel parity tests
  (HC chain vs `hc_mix`/`hc_combine`, router vs `moe_route` incl. id
  equality, swiglu/axpy/shared-gate, gemma+rope vs `apply_rope_lastdim`,
  pool/mqa/topk vs `qsa_fast_topk`, sparse attention vs
  `qsa_sparse_attention`) auto-enable on a CUDA box once built.
- NOT verified: nvcc compilation of qwen4.cu and the kernel parity runs
  themselves (no nvcc on this dev box) — FIRST cluster build in milestone B
  is the compile gate; fix-forward expected there.

## 2026-08-31 - sLLM: A3 tp module (topology, collectives iface + sim, per-rank weight view)

### Summary
Delivered the TP2 programming surface the C2 dual-node run builds on, all
locally verifiable:
- `tp/topology.py`: the verified DGX Spark pair as data (head rank0
  192.168.0.250 / worker rank1 192.168.0.231, pair link 10.100.25.x from
  docs/design/08); `from_recipe` refuses any tp.size/world mismatch.
- `tp/collectives.py`: `Collectives` transport interface (+ explicit
  NotImplementedError `NcclCollectives` placeholder for C2's libnccl
  transport) and `SimCollectives`, a single-process world that performs the
  exact NCCL SUM/CONCAT semantics over tagged per-rank partials and errors on
  missing/duplicate/mis-shaped/out-of-world contributions.
- `tp/rank_table.py`: `RankWeightTable` = LazyWeightTable (A2) x
  Qwen4ExpSharding (A2) x rank: fp8 tensors stay RAW uint8 per rank; `dequant`
  shard-then-dequantizes ONLY the rank slice with co-sharded scales (the full
  tensor is never materialized — the 173 GB memory contract); experts raise
  `RankOwnership` for non-owners.
- Contract hardening: multi-segment (GDN q|k|v) fp8 scale slicing now REJECTS
  non-block-aligned segments explicitly (the real checkpoint geometry always
  satisfies it; sub-block toy cases fail loudly instead of silently).

### Files Changed
- `tp/topology.py`, `tp/collectives.py`, `tp/rank_table.py` (all new)
- `tests/_synth.py` (shared synthetic-checkpoint writer),
  `tests/test_tp_module.py` (new, 10 tests)

### Reason
Operational-track Phase A3: rank-logic bugs (wrong rank counts, gather order,
ownership, scale co-sharding on the real loader path) must be caught on the
dev machine before cluster windows are spent.

### Validation
- `tests.test_tp_module`: 10 OK — topology from the real qwen4_exp recipe +
  tp-mismatch rejection; sim all-reduce/all-gather == reference helpers,
  missing/double/bad-rank/shape errors; rank table over a synthetic 2-shard
  fp8 checkpoint: raw-uint8 rank slice, rank dequant == sliced full dequant
  (bit-exact; q_proj split_out, o_proj split_in, owned expert, replicated
  shared expert), expert ownership + owned_names, replicated router equality,
  vocab-row split bit-exact reassembly, sub-block multi-segment scale
  rejection.
- Full suite: 216 OK (skipped 24) — no regression vs 206.
- NOT verified: NCCL transport (C2), GB10 pair bandwidth (B2).

## 2026-08-31 - sLLM: A2 fp8-persistent streaming loader + scale-aware TP2 sharding

### Summary
Delivered the loader/sharding contracts the TP2 operational path (C2) builds
on, fully testable on the dev machine:
- `loaders/streaming.py`: `ShardFile` (mmap lazily; F8_E4M3 stays RAW uint8,
  BF16 -> float32 per tensor), `CheckpointIndex` (HF `*.safetensors.index.json`
  weight map or single-file), `LazyWeightTable` (per-tensor on-demand
  `dequant`, per-layer fetch folding `_scale_inv` away). No whole-model
  materialization anywhere — the 173 GB fp8 bytes are the persistent form.
- `loaders/tp_shard.py`: `Qwen4ExpSharding` derives name-driven TP plans from
  Qwen4ExpCfg geometry (attention at head granularity incl. the q|gate
  interleave, GDN at k/v-head granularity with q|k|v segment awareness, MoE
  expert partition, routers/HC/norms/indexer replicated). FP8 block-scale
  alignment is a PROVEN precondition (`validate_tensor`: every rank slice
  block-aligned or block-contained), scales slice with floor/ceil block rules
  and multi-segment plans scatter (not concat) rows back. Key geometry
  finding: the shared expert (intermediate 640) must REPLICATE — 640/2 = 320
  cuts fp8 block 2 (320 % 128 != 0), and ~236 MB replication is the cheap
  block-safe answer.

### Files Changed
- `loaders/streaming.py` (new), `loaders/tp_shard.py` (new)
- `tests/test_streaming_tp.py` (new, 13 tests)

### Reason
Operational-track Phase A2 (docs/design/09 as resequenced on 2026-08-31): the
engine cannot load 173 GB as dequantized tensors, and TP2 sharding must be
proven scale-exact before any device kernel or cluster run depends on it.

### Validation
- `tests.test_streaming_tp`: 13 OK —
  streaming: raw-uint8 F8 passthrough, dequant == reference blocked loop,
  BF16->f32 norms, layer fetch, index + single-file resolution, unknown-name
  error; TP: all real-geometry split tensors validate block-safe (incl. the
  48-row GDN a/b block-contained case), plan kinds (q split_out/o split_in/
  experts/shared replicated), rank ranges head-granular (q_proj 6144 rows =
  12 heads x 512; GDN qkv rank-1 rows [(1024,2048),(3072,4096),(7168,10240)]),
  real-size (10240x2560) qkv shard_pair: rank dequant == full-dequant slices
  bit-exact + scale reassembly bit-exact; tiny sub-block tensors: rank dequant
  == sliced full dequant (shared scale rows OK); column/row GEMM partition
  identities; MoE expert-partition + all-reduce == full MoE; expert cover
  256/256 exact; forced misaligned 640 split rejected.
- Full suite: 206 OK (skipped 24) — no regression vs 193.
- NOT verified: real checkpoint bytes (B1 on head), C1/C2 device paths.

## 2026-08-31 - sLLM: qwen4_exp engine integration (A1, operational-track phase A)

### Summary
Operational goal adopted (user): serve Qwen3.8-Flash-Next-FP8 (qwen4_exp,
FP8 173 GB) for real; vLLM/sglang containers will be stopped for the live
operating test, so TP2 dual-node is the deployment target. Roadmap reorganized
into Phase A (local CPU) / B (head CPU, cluster window) / C (exclusive GPU):
A1 engine integration, A2 fp8-persistent streaming loader + TP2 shard splitter,
A3 tp module (collectives iface + sim), A4 qwen4_exp CUDA kernel sources;
B = Q3 subset parity + cublasLt fp8 block-scale probe + pair-link bandwidth
measurement; C = device-resident fp8 decode -> TP2 full model -> T3 parity ->
MTP spec decode + ops artifacts.
A1 delivered: `qwen4_exp` now runs through the serving executor (CPU
incremental). `Qwen4ExpCfg.from_recipe` derives the pipeline cfg from a parsed
recipe (QSA knobs from the recipe `spec` meta; `qsa_attention` layer type maps
to the pipeline's full-attention block). `ReferenceModel` routes arch
`qwen4_exp` prefill/decode_step/logits to `ref/qwen4_exp_pipeline` (the real
recipe knobs keep tripping the PLE guard until Q5, by design). Dev CLI:
`python -m serving.cli --qwen4 ...` serves the tiny fixture with a stub
char tokenizer.

### Files Changed
- `ref/qwen4_exp_pipeline.py`: + `Qwen4ExpCfg.from_recipe(recipe)`
- `serving/executor.py`: `ReferenceModel(q4_cfg=...)`, `_is_q4`/`_q4cfg()`,
  qwen4 branches in `logits`/`supports_incremental`/`prefill`/`decode_step`
  (numpy-only until the Q4-GPU kernels; logit contract normalized to (1,S,V))
- `serving/dev_model.py`: + `tiny_qwen4_exp_recipe()` (recipe twin of the tiny
  cfg), `TinyCharTokenizer` (V=32 stub, lowercase roundtrip),
  `build_dev_qwen4_exp_engine()` (cfg-less model -> exercises from_recipe)
- `serving/cli.py`: + `--qwen4`
- `tests/test_qwen4_engine.py` (new, 8 tests)

### Reason
Milestone A1/Q4-CPU of docs/design/09 as resequenced for the operational
track: the Q2 pipeline must run through the real serving machinery before any
GPU/TP work.

### Validation
- `tests.test_qwen4_engine`: 8 OK — from_recipe(tiny)==tiny cfg; real recipe
  knobs (hidden 2560 / 512 experts / idx budget 2048 ratio 4 / rotary 64 /
  ple (2)) + PLE NotImplementedError; engine greedy == pipeline
  `generate_greedy` (bit-identical); logits oracle shape/last-row exact;
  context clamp ValueError; continuous batch == sequential greedy; stub
  tokenizer roundtrip.
- Full suite: 193 OK (skipped 24, no-GPU/local) — no regression vs 185.
- CLI smoke: `--qwen4 --chat` and `--qwen4 --batch` run end to end (garbled
  text expected with random tiny weights).
- NOT verified: real checkpoint weights (B1), GPU kernels (A4+), TP2 (C2).

## 2026-08-30 - sLLM: qwen4_exp Q2 - model pipeline (HC + GDN + QSA + MoE) with parity tests

### Summary
Milestone Q2 of docs/design/09: wired the Q1 component oracles into a
checkpoint-named single-sequence qwen4_exp forward (batched prefill +
incremental decode) with a tiny fixture, gated by T1-style parity tests.

### Files Changed
- `ref/qwen4_exp_pipeline.py` (new): `Qwen4ExpCfg` (knob-validated; PLE
  guarded), `Qwen4ExpState` (GDN state + conv window, KV, indexer pending
  token_k + compressed-key caches), `prefill/decode_step/generate_greedy`.
  Layer flow ports `Qwen4ExpLayerExtensionMixin`: hc_mix -> block ->
  hc_combine -> mlp hc_mix -> MoE -> combine; final
  `hyper_connection_mixer.mix` -> lm_head (the plain `norm` is deleted in
  upstream, so none here). Prefill QSA = batched compressed blocks +
  row-wise select; decode = ring-free incremental compression at
  `(pos+1) % ratio == 0` with block rope from the block's FIRST token
  (qsa_indexer.py:363-366). GDN reuse: `ref/qwen3_5.gated_delta_net_forward`
  (prefill) and `gated_delta_rule_recurrent` (decode), mirroring
  `ref/incremental.py` hybrid state handling.
- `serving/dev_model.py`: `tiny_qwen4_exp_cfg()/tiny_qwen4_exp_weights()` -
  checkpoint-named fixture (names/shapes verified against the real
  `model.safetensors.index.json` + safetensors headers on head: q_proj
  [nh*2*hd,H] q|gate interleaved, index_qk_proj [(idx_h+1)*idx_d,H],
  mix_down [lr,hc*H], block_inject [hc,hc*H]).
- `tests/test_qwen4_exp_pipeline.py` (new, 7 tests): prefill/decode logit
  parity (rtol 1e-4 / atol 2e-4), indexer prefix-stability across runs
  (atol 1e-6), incremental cache growth invariants, greedy loop, PLE guard.

### Reason
Q2 exit gate: T1 layer + T2-token parity on a synthetic model before any
real-checkpoint work (Q3).

### Validation
- `tests.test_qwen4_exp_pipeline`: 7 OK (prefill-vs-decode logits parity
  incl. chunked-vs-recurrent GDN and batched-vs-incremental QSA selection).
- Full suite: 185 OK (skipped 49) locally - no regression vs 178.
- Remaining tolerance evidence to carry to Q3: parity holds at atol 2e-4
  on the tiny fp32 model; real-weight noise floor to be measured on GPU.

## 2026-08-30 - sLLM: sequential roadmap (qwen4_exp -> deepseek_v4) + Q1 component oracle

### Summary
User decision: support the remaining arches sequentially, qwen4_exp first,
then deepseek_v4. Extracted the upstream oracle sources, wrote the roadmap
doc, and delivered milestone Q1 (component-level numpy oracles with T0 tests).

### Files Changed
- `oracle/upstream/sglang/` (new, READ-ONLY reference): qwen4_exp.py,
  qwen4_exp_mtp.py, qwen4_exp_config.py, hyperconnection.py,
  qwen_sparse_attn_backend.py, qsa/*.py (10), sglang_qwen3_5.py + README.md
  provenance note. Extracted via `docker cp` from the deployed image
  `lmsysorg/sglang:qwen38flashnext` (Apache-2.0) on the head node.
- `docs/design/09-roadmap-seq-qwen4exp-dsv4.md` (new): evidence base, reuse
  map, milestone tables (Q1-Q5, D1-D5), validation/regression rules.
- `ref/qwen4_exp.py` (new): numpy T0 oracles - GemmaRMSNorm (plain/grouped),
  GatedResidual hc_mix/hc_combine (per-branch norm), MoE (softmax -> top-k
  -> renorm w/ 1e-20, swiglu experts, sigmoid-gated shared expert, dense
  emulation), QSA indexer chain (MQA logits sum relu/sqrt(d), average pool,
  relative fast top-k, block->token expand with partial-block tail +
  compaction, sparse GQA reference), indexer q/k projection (gemma norm +
  partial NeoX rope). Each function cites its upstream file.
- `tests/test_ref_qwen4_exp.py` (new): 17 CPU tests, all hand-rolled
  independent recomputations.

### Reason
`oracle/upstream` rule of docs/design/04: reference math must come from code,
not guesses. The qwen4_exp checkpoint ships no modeling code; the deployed
serving image is the authoritative implementation (it is the T3 baseline).

### Validation
- `python -m unittest tests.test_ref_qwen4_exp`: 17 OK.
- Full suite: 178 OK (skipped 49) locally - no regression vs the prior 161.
- Knob values cross-checked against the real checkpoint config.json
  (num_experts 512 / top-10, hc_count 4 / hc_lowrank 320, indexer 4h/1kv/
  dim128/budget2048/ratio4, head_dim 256 partial_rotary 0.25).
- Documented assumption: torch.topk tie order (oracle uses stable ascending).

## 2026-08-30 - sLLM: general-purpose fused + dtype device decode (fp32/bf16, partial RoPE)

### Summary
Generalized the device-resident decode engine from "Qwen2.5 fp32 full-rotary
only" to an architecture/dtype-agnostic serving engine, then fixed three real
bugs the broader test matrix exposed. Fusion: each decoder layer now runs in
11 device ops (was ~20) by concatenating q|k|v and gate|up into single GEMMs
and folding q/k bias + RoPE, residual-add + RMSNorm, and silu*mul into fused
kernels. dtype: `fp32` or `bf16` model dtype for weights / embedding / KV /
GEMM operands (RNE-rounded, bit-exact for BF16 checkpoints) with the residual
stream, softmax and logits kept in fp32 (the mainstream split). Partial RoPE:
the standard path now honours `recipe.rotary_dim()` (was hard-coded to
head_dim), so partial-rotary checkpoints work. Root causes of three latent
bugs, each caught only after widening the test fixture: (1) `rope_bias` read
the q/k bias without the per-head row offset, so every head reused head-0's
bias (invisible until biases entered the fixture; the real Qwen2.5 q-bias
reaches ~80 so it diverges hard there); (2) the engine added the k_proj bias
TWICE for the K-cache row (once in rope_bias, again in the K append) — fixed
to write the already-biased K verbatim; (3) `rope_bias` only applied bias to
the rotated `rot` columns, dropping it on the pass-through tail `[rot,dim)`
that partial-rotary models keep — bias now covers all `dim`. Switched the
generic fused kernels to exact `expf`/`sqrtf` (they are memory-bound; keeps a
tight fp32 floor). Validation on the real 0.5B: fused **fp32 teacher-forced
max|Δlogit| = 2.5e-5** (was 8.9 pre-fix), greedy `generate()` **identical to
numpy** for fp32 and bf16; bench **fp32 58 tok/s / bf16 90 tok/s** vs 29
numpy, all greedy-identical. Full suite 161 OK on head + 161 OK local.

### Files Changed
- `kernels/cuda/kernels.cu`: + `sllm_buf_h2d_raw` (typed-byte upload),
  + dtype-tagged fused API `sllm_gemm_ex` (cublasGemmEx, fp32 accumulate),
  `sllm_gather_row_t`, `sllm_add_rms` (residual-add + RMSNorm in one, typed
  output), `sllm_rope_bias` (q/k bias on ALL dims + partial RoPE),
  `sllm_silu_mul`, `sllm_kv_write_t` (typed KV + optional v-bias + fp32
  staging), `sllm_kv_relayout_w` (dtype-agnostic word-granular growth),
  `sllm_attention_decode_t` (typed KV, fp32 q/math). All `t_store/t_load`
  templated over fp32/bf16. Transfer-era kernels left untouched.
- `kernels/_sllm_cuda.py`: `T_F32/T_BF16`, `to_bf16()` (RNE), `DeviceView`
  (non-owning fused-GEMM slices), `DeviceBuffer.upload_raw`, `alloc_n`,
  `kv_relayout_w` and wrappers for the fused/typed entries (None-able args).
- `kernels/device_decode.py`: `DeviceWeightTable` now stores per-layer
  `<p>.qkv_w` (q|k|v concat) and `<p>.gu_w` (gate|up concat) in the table
  dtype, norms/biases in fp32, free-memory guard unchanged. `DeviceDecodeState`
  rewritten to the 11-op fused schedule, dtype-aware KV + scratch, `rot` from
  `recipe.rotary_dim()`, geometry sanity vs concatenated widths.
- `ref/standard.py`, `ref/incremental.py`: standard path builds cos/sin over
  `recipe.rotary_dim()` (partial RoPE); `IncrementalCache` carries `rot`.
  No behaviour change for full-rotary checkpoints (factor 1.0).
- `kernels/standard_decode.py`: transfer path follows `cache.rot` too.
- `serving/executor.py`, `serving/serve_standard.py`: `gpu_dtype` wiring
  (`SLLM_GPU_DTYPE` / `--gpu-dtype fp32|bf16`; fp32 default).
- `serving/dev_model.py`: tiny standard fixture gains q/k/v biases at scale
  0.3 (per-head) so bias-index bugs are structurally caught.
- `bench/gpu_decode_timing.py`: `--dtype`/`--no-bf16`/`--no-transfer`, reports
  first greedy divergence index instead of a bare differs count.
- `tests/test_gpu_decode.py`: + `TestBf16Conversion` (RNE + bad-dtype
  fallback), `TestResidentDtype` (fp32/bf16 step==numpy, bf16 KV growth,
  bf16 generate identity), `TestPartialRotaryReference`/`TestPartialRotaryGpu`
  (rotary_dim=2 parity across oracle/transfer/resident).

### Root Cause
The pre-fusion engine had only ever been tested with bias-free, full-rotary
fp32 fixtures, so three correctness bugs (q/k bias row offset, K double-bias,
partial-rotary bias tail) were latent. Widening the fixture to include qkv
biases and adding partial-rotary + bf16 cases exposed them; a component-swap
bisect on the real 0.5B isolated the q_proj path, and an op-by-op device
replay pinpointed the exact diverging op.

### Validation
- Real 0.5B (head GPU): fused fp32 teacher-forced max|Δlogit|=2.5e-5,
  bf16=0.17 (argmax-stable); `generate()` greedy IDENTICAL to numpy for fp32
  and bf16 (32 steps), no silent fallback. Bench: numpy 29 / fp32 58 / bf16 90
  tok/s, all greedy-identical. fp32 resident build correctly skipped when free
  VRAM < weights+slack (guard), bf16 (half size) still builds under pressure.
- Full suite: 161 OK on head (skip 25 cluster-only), 161 OK local (skip 49
  no-GPU). No regressions; transfer/hybrid paths unchanged and still pass.

## 2026-08-30 - sLLM: device-resident GPU decode (weights + persistent on-device KV)

### Summary
Removed the per-op H2D/D2H + `cudaMalloc/free` + per-kernel-sync bottleneck of
the transfer-era GPU decode by making the standard-path decode
**device-resident**: all weights are uploaded ONCE (`DeviceWeightTable`), the
attention KV lives on the GPU and grows on demand (`DeviceDecodeState`,
capacity-doubling relayout kernel), and every activation stays in a reused
on-device scratch for the whole step. One `cudaDeviceSynchronize` + two small
D2H copies (logits + the rows appended this step) per token replace the old
per-op syncs. Added GQA-aware device attention (`sllm_attention_decode_dev`),
a device RoPE kernel, KV append/relayout kernels and row-major cuBLAS gemm
entries. The host `IncrementalCache` stays authoritative: each step mirrors
the appended rows back, so a runtime GPU failure degrades transparently to the
transfer/numpy path. Reached **89 tok/s on the real 0.5B (vs 29 numpy / 0.4
transfer)** with greedy output identical to numpy, single and continuous-batch.

### Files Changed
- `dgxspark-serve/kernels/cuda/kernels.cu`: + shared lazily-created cuBLAS
  handle; + device-resident entries `sllm_sync`, `sllm_gemm_dev`,
  `sllm_gemm_linear_dev`, `sllm_bias_add_dev`, `sllm_elwise_{add,mul}_dev`,
  `sllm_silu_dev`, `sllm_rms_norm_dev`, `sllm_gather_row`, `sllm_rope_dev`,
  `sllm_kv_write` (+ staging), `sllm_kv_relayout`,
  `sllm_attention_decode_dev` (GQA, capacity-stride). No host transfers /
  cudaMalloc / sync inside these.
- `dgxspark-serve/kernels/_sllm_cuda.py`: `DeviceBuffer.upload()` + ctypes
  bindings and thin wrappers for the device-resident entries.
- `dgxspark-serve/kernels/device_decode.py` (new): `DeviceWeightTable`
  (one-shot upload + free-memory guard) and `DeviceDecodeState` (on-device KV
  with capacity doubling, reused activation scratch, one-sync-per-step, host
  cache mirror).
- `dgxspark-serve/serving/executor.py`: `ReferenceModel` tries the
  device-resident step first for `standard_gqa` (`SLLM_GPU_RESIDENT=1` default;
  `=0` forces the transfer path), builds the table lazily, and disables the
  resident path once on any failure (host cache untouched on a failed step).
- `dgxspark-serve/tests/test_gpu_decode.py`: + `TestDeviceResidentStandard`
  (step==numpy + host mirror exact, KV capacity growth, engine `generate()`
  identity + resident-actually-used, env-disable -> transfer path, two
  sequences interleaved on one shared table).
- `dgxspark-serve/bench/gpu_decode_timing.py` (new): numpy / transfer / resident
  timing + greedy-agreement on the real 0.5B or the tiny model.
- `dgxspark-serve/README.md`.

### Reason
The transfer-era driver shipped every weight to the GPU per call and synced per
kernel, so the GPU path was ~40x slower than CPU on a busy GB10. This is the
device-residency + persistent-KV performance stage that design doc 03 already
scaffolded (KV placement / device residency).

### Root Cause (of the prior 0.12 tok/s)
`ck.gemm`/`attention_decode`/etc. did `cudaMalloc`+H2D+`cudaDeviceSynchronize`
+D2H+`cudaFree` on every call with host-resident weights; the standard decode
issues ~9 such ops per layer x L layers x per token.

### Validation
- Cluster GPU (head): rebuild `sllm_gpu.so` (nvcc CUDA 13). `tests.test_gpu_decode`
  = 9/9 OK (5 new resident tests). Full suite: 152 OK (skipped 25). Dev (no
  .so): 152 OK (skipped 43).
- Real 0.5B `bench/gpu_decode_timing.py --steps 24`: numpy 29.21 / transfer 0.37
  / **resident 89.23 tok/s**, all three greedy-identical. `free_bytes` guard let
  the ~1.98 GB fp32 table build with headroom; rejected when short.
- Real 0.5B E2E (`serve_standard`, palindrome prompt, 64 new): GPU resident ==
  numpy (diff clean); 3-prompt continuous batch GPU == CPU (no cross-talk).
  Confirmed the resident path actually ran (no silent fallback).
- NOT verified: 27B qwen3_5 real weights; hybrid device-resident (resident is
  `standard_gqa`-only this stage); TP2.

---
## 2026-08-30 - sLLM: GPU decode loop wired into serving (standard + hybrid)

### Summary
Wired the GPU kernel decode path into the serving loop end to end
("pipes-to-C": prefill stays numpy/host-KV, every `decode_step` can run on GPU
kernels, with transparent numpy fallback). Added a fused SiLU kernel and a
`gpu_standard_decode_step` driver; `ReferenceModel(use_gpu=...)` / env
`SLLM_USE_GPU=1` / `serve_standard --use-gpu` select it. `generate()` and
`BatchedInferenceEngine` both route through `model.decode_step`, so the whole
serving stack benefits without touching its structure.

### Files Changed
- `dgxspark-serve/kernels/cuda/kernels.cu`: + `sllm_silu` (fused SiLU).
- `dgxspark-serve/kernels/_sllm_cuda.py`: + `silu(x)`.
- `dgxspark-serve/kernels/standard_decode.py` (new): `gpu_standard_decode_step`
  (cuBLAS GEMMs + q/k/v biases + last-row attention + elwise + silu).
- `dgxspark-serve/serving/executor.py`: `ReferenceModel(use_gpu=None)` (env
  `SLLM_USE_GPU`), `_gpu_available()`, `decode_step` dispatches to the GPU
  driver (standard/hybrid) with numpy fallback on any failure.
- `dgxspark-serve/serving/serve_standard.py`: `--use-gpu`.
- `dgxspark-serve/tests/test_gpu_decode.py` (new; skip without .so): standard &
  hybrid GPU step == numpy (allclose + argmax); engine-level
  `generate()` identity (standard) / determinism (hybrid).
- `dgxspark-serve/README.md`.

### Reason
Deliver the GPU decode loop as a selectable serving path (correctness first),
so the engine can switch the per-token math to kernels without restructuring.

### Validation
- Cluster GPU (head): rebuild .so; `tests.test_gpu_decode` +
  `test_hybrid_gpu` + `test_cuda_kernels` = 13/13 OK. Full suite: 147 OK
  (skipped 25). Dev: 147 OK (skipped 34, no .so).
- Real-model E2E (Qwen2.5-Coder-0.5B, GPU): `use_gpu=True` decode steps ran on
  kernels and produced **identical output to the numpy path (0/24 token
  mismatches)** — correctness confirmed end to end.
- **Performance (honest):** this milestone is ~40x slower than the CPU numpy
  path on the busy cluster (0.12 vs 4.78 tok/s): per-op host<->device
  transfers (weights are host-resident; `ck.gemm` uploads each call),
  `cudaDeviceSynchronize` per kernel, and the GPU is saturated by vLLM
  (~2 GB free, volatile). Speed is deferred to the device-resident weights +
  persistent on-device KV + no-per-op-sync stage (ideally on a quiet GPU),
  which is exactly the KV-placement/device-residency design already scaffolded.
- NOT verified: 27B qwen3_5 real weights (needs quiet window + ~31 GB RAM);
  persistent device KV; TP2.

---
## 2026-08-30 - sLLM: qwen3_5 hybrid GPU kernels (GatedDeltaNet step + hybrid decode driver)

### Summary
First GPU kernels for the qwen3_5 hybrid path: a GatedDeltaNet single-step
(decode) recurrence kernel, plus a GPU hybrid decode driver that runs the
per-token decode with cuBLAS GEMMs, the last-row attention kernel, the delta
kernel and elementwise adds — mirroring `ref/incremental._decode_hybrid`.
Validated on the cluster GPU against the numpy incremental path (tiny hybrid
model).

### Files Changed
- `dgxspark-serve/kernels/cuda/kernels.cu`: + `sllm_gated_delta_step` (per
  value-head: decay -> kv_mem -> error delta -> state update -> out; state
  in place).
- `dgxspark-serve/kernels/_sllm_cuda.py`: binding + `gated_delta_step(q,k,v,
  g,beta,state)` helper.
- `dgxspark-serve/kernels/hybrid_decode.py` (new): `gpu_hybrid_decode_step`
  — dense proj/MLP/lm-head via cuBLAS `gemm`, linear layers via
  `gated_delta_step`, full-attention via `attention_decode`, residual via
  `elwise_add`; norms/rotary/silu/gating/conv-window stay on host.
- `dgxspark-serve/tests/test_hybrid_gpu.py` (new; skip without .so): delta
  kernel vs numpy recurrence; full hybrid GPU decode logits == numpy per step
  (argmax equal); determinism.
- `dgxspark-serve/README.md`.

### Reason
Next sLLM milestone: bring the hybrid (production) architecture onto the GPU.
The prior phases delivered CPU incremental decode + dense/standard GPU kernels;
this adds the GatedDeltaNet kernel that qwen3_5 (and qwen4_exp) need.

### Validation
- Cluster GPU (head): `kernels/cuda/build.sh` (nvcc 13.x + cuBLAS) -> .so.
  `tests.test_hybrid_gpu`: 3/3 OK — gated_delta_step matches numpy recurrence
  (state + out, rtol/atol 1e-3); full hybrid GPU decode_step logits match the
  numpy incremental decoder across 4 chained steps with argmax equality;
  deterministic.
- Full suite on head: 143 OK (skipped 25). Dev box: 143 OK (skipped 34 —
  includes GPU tests, no .so).
- Note: GPU device memory is shared with vLLM (~101 GB resident) and fluctuates;
  `cudaMalloc` may transiently fail "out of memory" when the unified pool is
  saturated — retry works once memory frees. Device KV placement already
  guards this (see previous entry).
- NOT verified: real 27B qwen3_5 weights/kernels (needs a quiet GPU window and
  ~31 GB RAM); fused conv/rope/norm kernels; decode-loop wiring into serving.

---
## 2026-08-29 - sLLM: KV memory placement option (device|host) + GPU tensor support

### Summary
Added a recipe-driven memory-placement option (`memory.kv_placement:
device|host`) so serving can choose between the conventional all-on-GPU KV
layout and the safer "KV in host RAM" layout. Added the matching GPU tensor
building blocks (cuBLAS GEMM, O(S) last-row decode-attention kernel, device
buffers) and a KV-backend abstraction with graceful CPU fallback. CPU mode
always uses host RAM, so the option is a no-op there (per user note).

### Files Changed
- `dgxspark-serve/recipes/schema.py`: `MemorySpec` (kv_placement in
  {device, host}, optional budget caps, utilization) + `Recipe.memory`;
  invalid placement/utilization raises `RecipeError`; `memory` excluded from
  meta passthrough.
- `dgxspark-serve/recipes/qwen2_5_coder_0_5b.yaml`: `memory.kv_placement:
  device` + docs.
- `dgxspark-serve/runtime/placement.py` (new): `KVMemoryPlan` (sizing, budgets),
  `KVBackend` (reserve/release + store/gather/free), `HostKVBackend` (numpy,
  usable RAM, OutOfCapacity on over-budget), `DeviceKVBackend` (GPU buffers),
  `build_kv_backend` (device -> host fallback with warning).
- `dgxspark-serve/runtime/memory_planner.py`: + `kv_bytes_per_token(recipe)`.
- `dgxspark-serve/kernels/cuda/kernels.cu`: + `sllm_gemm` (cuBLAS SGEMM,
  row-major c=a@b), `sllm_attention_decode` (per-head softmax(q.K*scale).V),
  `sllm_buf_new/free/h2d/d2h` device helpers; `build.sh` links
  `-lcublas -lcublasLt`.
- `dgxspark-serve/kernels/_sllm_cuda.py`: bindings + `DeviceBuffer`,
  `to_device`, `gemm`, `attention_decode`.
- `dgxspark-serve/serving/serve_standard.py`: `--kv-placement` override +
  resolved plan printed.
- `dgxspark-serve/tests/test_recipes.py` (+5 memory-spec), `tests/test_placement.py`
  (new, 8), `tests/test_cuda_kernels.py` (+3 GPU, skip without .so).
- `dgxspark-serve/README.md`.

### Reason
User requirement: keep a conventional VRAM mode AND a VRAM+RAM split mode as a
selectable option. Motivation: all-KV-in-VRAM OOM can hang the GB10 node until
power off (no recovery), whereas host-RAM KV OOM is recoverable (process exit /
swap). CPU mode needs none of this (RAM only), which the design honours.

### Validation
- `python -m unittest discover -s tests`: 140 OK (124 prior + 16 new; 31
  skipped: dev-cache tokenizer + CUDA .so absent). py_compile on all changed.
- Coder recipe resolves placement=device, bytes_per_token 12288
  (24 layers*2*2kvh*64hd*2B), 19660 blocks / 314560 max tokens at the default
  4 GiB device budget.
- On this CPU box, device placement degrades to the host backend with a
  RuntimeWarning (intended "CPU uses RAM only" + graceful fallback).
- Host backend: store/gather/free roundtrip; over-budget reserve raises
  OutOfCapacity (recoverable).
- **Cluster GPU (head, CUDA 13.x + cuBLAS) after `kernels/cuda/build.sh`**:
  - `sllm_gemm` vs numpy: pass; `sllm_attention_decode` vs numpy last-row:
    maxdiff 3.3e-7; device-buffer roundtrip: pass; 6/6 GPU kernel tests OK.
  - Full suite on head with kernels: 140 OK (skipped 25).
  - Device KV placement on the busy GPU (vLLM holds most of the ~128 GB): the
    new free-memory guard (cudaMemGetInfo) found 1.05 GB free < 4 GiB planned
    budget -> **automatically degraded to host-RAM KV with a warning**, i.e.
    the exact over-subscription case that can hang GB10 is avoided by design.
    Roundtrip/capacity/fallback all verified on hardware.
- NOT verified: a quiet-GPU run that actually exercises DeviceKVBackend
  storage (needs vLLM off / dedicated window); full GPU decode loop wiring.

---


## 2026-08-29 - sLLM: QKV-bias correctness fix + repetition penalty + 0.5B HTML game demo

### Summary
Fixed a real checkpoint-correctness bug: Qwen2.5-Coder-0.5B ships q/k/v
projection **biases** (72 tensors), which the standard path was silently
ignoring -> our logits were garbage (BPE soup) while transformers was sane.
Added optional q/k/v biases to the standard attention (recompute + incremental
paths), a HF-style `repetition_penalty` to the sampler, and generated a working
HTML5 snake game with the real 0.5B model through the engine.

### Files Changed
- `dgxspark-serve/ref/standard.py`: `standard_attention_forward` /
  `standard_model_forward` take optional `q_bias`/`k_bias`/`v_bias`
  (all-default-None keeps Llama bias-free behaviour).
- `dgxspark-serve/ref/incremental.py`: same biases in standard prefill + decode
  (K/V capture and per-step projections).
- `dgxspark-serve/runtime/sampler.py`: `apply_repetition_penalty` (logit/id /= p).
- `dgxspark-serve/serving/executor.py`: `generate`/`complete`/`chat`/
  `BatchedInferenceEngine.submit` accept `repetition_penalty` (default None).
- `dgxspark-serve/tests/test_incremental.py`: + qkv-bias parity test;
  + repetition-penalty tests.
- `dgxspark-serve/demo/snake_0.5b.html` (new): generated game artifact.
- `dgxspark-serve/README.md`.

### Reason / Root Cause
While trying to generate an HTML game with the 0.5B model, every prompt
(including "function add(a,b){...}") produced nonsense. Comparing against a
fresh transformers load showed weights exact (embed/q_proj/norm maxabsdiff
0.0) but logits cosine -0.13. The checkpoint safetensors header contains
72 q/k/v `.bias` tensors that our forward dropped, so softmax worked on wrong
q/k/v -> garbage. (The earlier "transformers hardcodes bias" note was about a
different model, Qwen3.8-27B-FP8.) After adding the biases our logits match
transformers: cosine 1.0, top-5 identical, maxabsdiff 7.7e-5.
Without biases the 0.5B also loops repetitively on long code; a HF-style
repetition penalty lets it finish a complete file.

### Validation
- `python -m unittest discover -s tests`: 124 OK (122 prior + 2 new; 28
  skipped dev-cache/CUDA). Same 124 OK on the cluster head (ARM64).
- Real-model logits (head, Qwen2.5-Coder-0.5B, "Write JavaScript: function
  add(a,b){return a+b;}"): after fix ours top5 == transformers top5
  [198, 729, 323, 311, 201], cosine 1.0000001, maxabsdiff 7.7e-5 (pre-fix:
  cosine -0.13, top1 "addAll").
- Game: scaffold completion + greedy produced real snake JS but looped in
  draw(); with repetition_penalty=1.15, temp 0.6, top_p 0.92 the first seed
  returned a complete file (<!DOCTYPE>..</html> with </script>, canvas,
  arrow-key controls, wall/self collision) in 42 s (~16.2 cont tokens/s).
  Saved to `demo/snake_0.5b.html` (0.5B quality: no food/scoring; "test game").
- NOT verified: GPU kernels; TP2; session history.

---


## 2026-08-29 - sLLM: batch incremental decode + first-decode off-by-one fix

### Summary
Wired the incremental (KV/recurrent-state) path into `BatchedInferenceEngine` so
concurrent sequences each keep their own loop-carried memory across decode
steps (O(context) last-row attention per request per step, no cross-talk).
Fixed an off-by-one in the first decode step that existed in the previous
Phase-A landing of `generate()` (prefill logits were discarded and
`decode_step(prompt_last)` re-embedded a duplicate token).

### Files Changed
- `dgxspark-serve/serving/executor.py`: `BatchedInferenceEngine` builds the
  per-sequence `IncrementalCache` on the first prefill action, keeps
  `prefill_L` (prefill last-position logits) for the first decode token then
  `decode_step(cache, last_id)` afterwards; non-incremental models fall back to
  the recompute path. `generate()` incremental branch now samples the first
  token from the prefill logits (`plogits[0,-1]`) instead of
  `decode_step(prompt_last)`.
- `dgxspark-serve/tests/test_incremental.py`: 4 batch tests (dev-runnable via a
  fake tokenizer): batch == recompute (standard), batch == sequential single
  generate, hybrid deterministic/bounded, long-prompt chunked prefill.
- `dgxspark-serve/README.md`.

### Reason / Root Cause
Extend the memory-continuity + speed win to multi-request continuous batching
(the explicit follow-up of the KV-cache phase). During validation a batch test
caught that `generate()`'s first decode step was wrong: with context `prompt`
(positions 0..S-1), the first token must be sampled from the prefill's
last-position logits; `decode_step(prompt_last)` instead embedded a duplicate
token at position S and predicted position S+1 (off by one). The tiny-model
unit test had passed by coincidence (stable repeated token), and the earlier
real-model "greedy mismatch / 15 of 38 tokens" was this same bug, not the fp32
GEMM seam.

### Validation
- `python -m unittest discover -s tests`: 121 OK (117 + 4 batch; 28 skipped
  dev-cache/CUDA), dev + cluster head (ARM64).
- Real-model parity (head, Qwen2.5-Coder-0.5B): after the fix
  **greedy identical True, 0/38 token mismatches** (was 15/38 before the fix),
  0 argmax flips over 12 chained steps, per-step logits maxdiff 2.5e-4 (fp32
  GEMM seam only; does not flip argmax here).
- Real-model batch (head): 3 concurrent prompts x 40 tokens = 120 tokens in
  4.69 s; batch output == sequential `generate()` for all three prompts;
  single-seq 40-token runs ~1.5-1.7 s each.
- `python -m py_compile` on changed files: OK.
- NOT verified: GPU kernels; session-history; TP2.

---


## 2026-08-29 - sLLM: incremental decode (loop-carried KV/recurrent-state memory)

### Summary
Added a KV-cached decode path for the reference engine so the runtime
"memory" (attention KV cache, GatedDeltaNet recurrent state, causal-conv
window) is continuous across decode steps. This turns the previous
recompute-every-step loop (superlinear; 640 tokens > 900s on the cluster)
into `prefill` once + `decode_step` per token (last-row attention over cached
KV, O(context) per step). Also enforced the context window:
`max_position_embeddings`.

### Files Changed
- `dgxspark-serve/ref/incremental.py` (new): `IncrementalCache`, `prefill`
  (exact existing forward + capture K/V / state / conv window),
  `decode_step` for the standard (Llama/Qwen2) and qwen3_5 hybrid paths.
- `dgxspark-serve/serving/executor.py`: `ReferenceModel` +
  `supports_incremental` / `max_context` / `prefill` / `decode_step`;
  `generate()` uses the cache path when supported and clamps at
  `max_position_embeddings`; `logits()` recompute path kept as oracle.
- `dgxspark-serve/tests/test_incremental.py` (new): standard bit-parity +
  token-identical greedy vs recompute; hybrid logits parity (rtol/atol) and
  consistent deterministic generation; state chaining; context clamp and
  over-long prompt error.
- `dgxspark-serve/bench/incremental_timing.py` (new): recompute vs
  incremental timing.
- `dgxspark-serve/README.md`: progress + container notes updated.

### Reason
The cluster operating test showed a single 16-token request at ~0.29 s/token,
but 640+ token requests did not finish in >900 s: the reference forward
recomputes the full context every decode step and `eager_attention` builds the
full SxS score matrix each call (O(T^3)-ish totals). Priority from user:
"work-memory continuity and speed improvement".

### Root Cause
- No incremental KV/state caching: every decode step re-ran the whole forward
  (`serving/executor.py` old `generate` -> `model.logits(full ids)`).
- `qwen3_5.eager_attention` computes scores/softmax for all S positions though
  decode only needs the last row (per-step O(S^2)).
- The runtime block/coordinator/planner code existed but was bookkeeping only
  (not wired to real attention state).

### Fix
- Incremental path: prefill once captures per-full-layer K/V (post-norm,
  post-rotary, pre-repeat) + per-linear-layer final state + last K-1 raw mixed
  vectors for the conv1d window. `decode_step` appends one token's K/V and
  runs an exact last-row attention (identical fp32 ops to `eager_attention`),
  and runs the existing `gated_delta_rule_recurrent` single step for linear
  layers (O(1)).
- Generation stops at `max_position_embeddings`; over-long prompts raise a
  clear ValueError instead of silently truncating (truncation policy stays for
  a later phase).
- `BatchedInferenceEngine` unchanged in this phase (batch-decode optimization
  explicitly out of scope); it still uses the recompute `logits()` path.

### Validation
- `python -m unittest discover -s tests`: 117 tests OK (109 previous + 8 new;
  28 skipped: dev-cache tokenizer + no CUDA .so on this box). Same 117 OK on
  the cluster head (ARM64).
- Standard parity: `decode_step` logits == full-recompute logits (rtol/atol,
  exact within fp noise) and greedy sequences identical; hybrid logits match
  within the chunked-prefill vs recurrent-decode seam (measured single-step
  max abs diff 2.98e-7 on the tiny model), generation deterministic.
- Context clamp: 253-token prompt + max_new 100 stops at 256; 256-token prompt
  raises ValueError.
- Bench (tiny standard model, dev machine): recompute vs incremental —
  50 tok 1.6x, 100 tok 2.4x, 200 tok 6.6x speedup, growing with context.
- Real-model operating test (cluster head, Qwen2.5-Coder-0.5B): incremental
  generated 200 tokens in 7.50 s (26.7 tok/s); recompute generated 20 tokens
  in 5.00 s (4.0 tok/s). Position-wise argmax parity (0 flips over 12 chained
  steps, per-step logits maxdiff 2.5e-4); a fresh 16-token greedy run diverged
  from recompute starting ~step 12-13 after a single near-tie argmax flip
  (fp32 GEMM prefill/decode seam — same class as vLLM; not a logic error).
- `python -m py_compile` on all changed/added Python: OK (dev + ARM64).
- NOT verified: cluster 64-layer qwen3_5, GPU kernels; session-history
  continuity (out of scope).

### Notes (fp32 prefill/decode seam)
The standard-path "last-row attention" uses an M=1 GEMM while the recompute
oracle uses an M=S GEMM; BLAS accumulation order differs, so logits agree to
~1e-4 absolute on the real model and greedy argmax can flip only on near-ties.
This mirrors the genuine prefill(chunked) vs decode(single-step) seam in vLLM.

---


## 2026-08-29 - sLLM: GPU kernels + containerised operation (sllm-node)

### Summary
Added the first self-contained CUDA kernels and wrapped the engine in a Docker
container for GPU operation, per user direction (container name `sllm-node`,
CUDA 13.0 baseline). Verified the container drives the GB10 GPU.

### Files Changed
- `dgxspark-serve/kernels/cuda/kernels.cu` (new): rms_norm, elwise_add,
  device_count (C ABI)
- `dgxspark-serve/kernels/cuda/build.sh` (new): nvcc build to sllm_gpu.so
  (CUDA 13.0 baseline, -arch=native)
- `dgxspark-serve/kernels/_sllm_cuda.py` (new): ctypes loader
- `dgxspark-serve/kernels/smoke.py` (new): GPU kernel smoke vs numpy reference
- `dgxspark-serve/tests/test_cuda_kernels.py` (new; skipped without the .so)
- `dgxspark-serve/deploy/Dockerfile` (new, multi-stage nvcc -> slim python),
  `deploy/entrypoint.sh`, `deploy/run.sh`, `deploy/.dockerignore`
- `dgxspark-serve/README.md`: container operation section

### Reason
Operate sLLM through a Docker container on the cluster and put real GPU kernels
behind it (the design's "self-made kernels" target; dense GEMM stays reserved
for cuBLAS per the design doc).

### Validation (cluster)
- Host: nvcc 13.3 (13.0 baseline honoring) built `sllm_gpu.so`; kernel smoke:
  device_count=1, rms_norm maxdiff 4.77e-7 vs numpy reference, elwise_add 0.
- `deploy/Dockerfile` built `sllm-node:latest` (230MB; CUDA 13.0.0-devel base
  resolved on arm64). `docker run --gpus all` ran the kernel smoke inside the
  container: device_count=1, SMOKE_OK.
- `nvidia-smi` during a looping kernel run showed the container's `python`
  process (PID 567405) using the GPU (11 MiB) alongside vLLM (101 GB).
- Local suite: 109 OK (3 cuda tests skipped on the dev box).
- Known dev fixes during build: Compress-Archive path issue (moved to python
  zipfile), COPY line must not carry shell redirects, entrypoint needs /bin/bash
  (no exec bit through zip).

### Next (planned)
- Full GPU forward of the 0.5B model: attention + rotary + GEMM (cuBLAS)
  kernels inside the container, verified against the numpy reference.

---

## 2026-08-29 - sLLM: temporary program name adopted

### Summary
The serving program is now (temporarily) named **sLLM**. The repository
directory `dgxspark-serve` and cluster deploy paths are unchanged (documented
as-is). Only documentation updated; no code changes.

### Files Changed
- `dgxspark-serve/README.md`: title + naming note
- `dgxspark-serve/docs/design/01-architecture.md`: title note

### Validation
- `python -m unittest discover -s tests`: 106 OK (no code changed).

---

## 2026-08-29 - dgxspark-serve: ~/models layout reflected in recipes + cluster continuous-batching operating test

### Summary
Recorded the cluster's shared model store (`$HOME/models/<model>` on both
head/worker) into the recipes as `paths.local_dir`, wrote
`docs/design/08-cluster-layout.md`, and ran a continuous-batching operating
test of the real Qwen2.5-Coder-0.5B on the cluster (3 concurrent requests).

### Files Changed
- `dgxspark-serve/recipes/schema.py`: + Recipe.paths / local_dir property
- `dgxspark-serve/recipes/*.yaml`: `paths.local_dir` added to all 4 recipes
- `dgxspark-serve/serving/serve_standard.py`: + `--batch` (BatchedInferenceEngine)
- `dgxspark-serve/docs/design/08-cluster-layout.md` (new)
- `dgxspark-serve/tests/test_recipes.py`: + local_dir assertion
- `dgxspark-serve/README.md`

### Reason
User: models live identically on both nodes at `~/models/<model>`; recipes
should carry that. Also continue development (continuous-batching operating
test on the real low-capacity model).

### Facts (audit)
- Head `~/models/` (13) vs worker (subset): big serving set identical
  (DeepSeek-V4-Flash 156G, Qwen3.8-27B-FP8 29G both, Flash-Next-FP8 173G
  both, ...). Head-only: Qwen2.5-Coder-0.5B (0.95G), Qwen3.5-122B, Qwen3.6-27B
  BF16/FP8. Worker hostname `aitopatom-b7ca`, plus a stray 0-size `sh` entry.
- Recipes now resolve checkpoints via `local_dir` (e.g. `~/models/Qwen3.8-27B-FP8`).

### Validation
- `python -m unittest discover -s tests`: 106 OK locally; 106 OK / 25 skipped
  on the cluster (ARM64).
- Cluster continuous-batch operating test on real Qwen2.5-Coder-0.5B:
  `--batch "def add(a,b):" "x = [1," "print(" --max-new 6` -> 3 prompts
  completed concurrently in 6.7s; outputs identical to the transformers
  baseline. One-shot process (no lingering server) so vLLM stays unaffected.

---

## 2026-08-29 - dgxspark-serve: standard-transformer support + real Qwen2.5-Coder-0.5B operating test on the cluster + transformers cross-validation

### Summary
Added standard dense-transformer (Llama/Qwen2 family) support and operated the
engine on a real low-capacity checkpoint on the DGX Spark cluster, then
cross-validated against transformers (WSL) — logits and greedy output match
exactly. Also discovered/isolated a transformers 5.16.1 attention-bias quirk.

### Files Changed
- `dgxspark-serve/ref/standard.py` (new): rms_norm_plain, standard_gqa
  attention (no gate/norm, full-dim RoPE), standard_model_forward (tied emb)
- `dgxspark-serve/recipes/qwen2_5_coder_0_5b.yaml` (new, status ready)
- `dgxspark-serve/recipes/schema.py`: text.prefix, tie_word_embeddings,
  optional head_dim (effective_head_dim)
- `dgxspark-serve/serving/tokenizer.py`: default `pretokenize_regex` fallback
  (Qwen2.5 tokenizer_config has none)
- `dgxspark-serve/serving/executor.py`: ReferenceModel routes kernel
  "standard_gqa"; `serving/serve_standard.py` (new, load+generate+HTTP);
  `serving/dev_model.py`: tiny standard fixture; `tests/test_standard.py` (new)
- `dgxspark-serve/tests/test_tokenizer.py`: Qwen2.5-Coder oracle parity
- `dgxspark-serve/README.md`

### Reason
"Low-capacity model operating test": bring up our own serving program on a real
small model on the actual DGX Spark node. Qwen2.5-Coder-0.5B (BF16, standard
dense GQA) is the target; the engine previously only executed the qwen3_5
hybrid numpy reference.

### Root Cause / Findings
- Cluster facts: head 192.168.0.250 aarch64 Python 3.12, Ubuntu; GB10;
  `/mnt/external/models` holds large checkpoints; HF reachable. Deployed the
  repo (used python zipfile after Compress-Archive produced broken path
  entries), built a venv (numpy/pyyaml/regex/jinja2/ml_dtypes/tokenizers,
  later torch+transformers for validation). Loaded real Qwen2.5-Coder-0.5B,
  one-shot generate 2.75s/8 tok; HTTP serve on :8002 with health + chat +
  text completions live from the dev machine.
- Initially our logits vs transformers differed hugely. Layer-by-layer +
  intra-attention diagnostics (WSL torch/transformers) showed projections were
  exact but q/k/v OUTPUTS differed by ~80/143 -> single suspect: attention
  biases. **transformers 5.16.1 Qwen2Attention hardcodes q/k/v bias=True**;
  the checkpoint declares `attention_bias=False` (no bias tensors), so random
  init biases survived -> the naive transformers baseline is NOT
  checkpoint-semantics. Zeroing those biases makes baselines agree.
- After zeroing: logits max diff <= 2e-4 and greedy 4-8 token generations are
  token-identical to transformers for many prompts. Our engine is
  checkpoint-correct; the "gibberish" output is this tiny model's true greedy
  continuation (both engines agree).
- Per user instruction, heavy torch comparisons run on this PC (WSL), not the
  cluster, so the shared vLLM container stays up.

### Validation
- `python -m unittest discover -s tests`: 105 OK (dev); 105 OK / 25 skipped on
  the cluster (ARM64).
- Real-model: cluster one-shot generation (2.75s/8 tok) + live HTTP smoke
  (health 200; chat 2.1s; text 0.6s) + prompt-conditioned deterministic.
- transformers cross-check (WSL): next-token argmax and greedy sequences match
  exactly after zeroing attention biases (72 biases).
- Impact: none of the "operating test" code changed vs cluster; only docs/log.
  Cluster server process stopped after the test; vLLM container unaffected.
- NOT verified: full 64-layer qwen3_5, TP2, kernels, live vLLM parity of the
  large models.

---

## 2026-08-29 - dgxspark-serve: qwen4_exp + deepseek_v4 audits and skeleton recipes

### Summary
Audited Qwen3.8-Flash-Next-FP8 (qwen4_exp) and DeepSeek-V4-Flash-0731
(deepseek_v4) from their live config.json + weight indices, wrote evidence
docs, extended the recipe schema (MoE MLP spec, qsa/mla block types,
status/meta passthrough), and added the two skeleton recipe YAMLs that parse
and validate.

### Files Changed
- `dgxspark-serve/docs/design/06-audit-qwen4-exp.md`, `07-audit-deepseek-v4.md` (new)
- `dgxspark-serve/recipes/schema.py`: MLPSpec moe fields; KNOWN_LAYER_TYPES +=
  qsa_attention/mla_attention; Recipe.status + meta passthrough
- `dgxspark-serve/recipes/qwen4_exp.yaml`, `recipes/deepseek_v4.yaml` (new, skeleton)
- `dgxspark-serve/tests/test_recipes.py`: + TestSkeletonRecipes

### Reason
The engine targets three recipes. Audits turn the two remaining architectures
into fact-based skeletons so P4/P5 scope and loader/kernel gaps are explicit.

### Key Findings
- qwen4_exp = 36 GatedDeltaNet (same family as qwen3_5 -> kernel reuse) + 12
  QSA sparse attention; NEW components: hyper-connection mixer (model + per
  attn/mlp), 512-expert MoE, PLE/ngram spec module, hybrid MTP (its own MoE,
  two fcs); FP8 per-128x128 (weight_scale_inv).
- deepseek_v4 = 43 dense MLA layers (no recurrent layers -> no hybrid KV
  bookkeeping needed); HDC sinkhorn chunking, sparse/hash windowed attention,
  FP4 experts (loader needs fp4 + ue8m0), DSPark Markov draft head in a
  3-block nextn stack; custom encoding_dsv4.py message encoder.

### Validation
- `python -m unittest discover -s tests`: 97 tests OK (added 3 skeleton-recipe
  tests). Skelton YAMLs parse (layer counts 48->36/12 and 43), status=skeleton,
  meta carries audited facts; ready qwen3_5 recipe unaffected.
- NOT verified: cluster GPU, real weights/kernels for these two models.

---

## 2026-08-29 - dgxspark-serve: MTP reference + speculative decoding (sound acceptance)

### Summary
Ported the Qwen3_5 MTP math from vLLM's `qwen3_5_mtp.py` into `ref/mtp.py`
(concat(pre_fc_norm_embed(tokens), pre_fc_norm_hidden(hidden)) -> fc (2H->H)
-> full-attention decoder layer -> norm -> lm_head; single MTP layer, reuses
main embedding + lm_head). Added `runtime/spec.py` MTP draft + main-model
greedy verification, and an invariant test: spec decode output is bit-identical
to plain greedy generation for num_draft 1..3.

### Files Changed
- `dgxspark-serve/ref/mtp.py` (new): mtp_forward, mtp_layer_weights, mtp_next_token
- `dgxspark-serve/runtime/spec.py` (new): spec_decode_greedy (draft+verify)
- `dgxspark-serve/ref/pipeline.py`: model_forward now can return pre-final-norm hidden
- `dgxspark-serve/serving/dev_model.py`: tiny MTP weights + recipe mtp enabled
- `dgxspark-serve/tests/test_mtp.py` (new)
- `dgxspark-serve/README.md`

### Reason
qwen3_5 ships with MTP; exact numerics come from vLLM (transformers ignores MTP
via _keys_to_ignore). This establishes the spec-decode path that later extends
to qwen4_exp (MTP3) and gives a GPU-free soundness invariant.

### Validation
- `python -m unittest discover -s tests`: 94 tests OK (5 mtp).
- MTP forward matches an independent manual re-implementation (rtol 1e-4).
- Invariant: spec_decode_greedy == greedy generate for num_draft 1/2/3; output
  length within [prompt, prompt+max_new].
- NOT verified: cluster GPU, real MTP weights, kernels, live vLLM parity.

---

## 2026-08-29 - dgxspark-serve: continuous-batching E2E (scheduler drives real generation)

### Summary
Wired the continuous-batching scheduler into the generation loop: a new
`BatchedInferenceEngine` (serving/executor.py) admits multiple requests, chunks
their prefills, interleaves decode steps across concurrently running sequences,
and frees resources on completion. Added a `--batch` CLI demo and integration
tests that prove real interleaving and queueing on the dev machine.

### Files Changed
- `dgxspark-serve/serving/executor.py`: + BatchedInferenceEngine, generate_batch
- `dgxspark-serve/serving/cli.py`: + --batch
- `dgxspark-serve/tests/test_batched.py` (new)
- `dgxspark-serve/README.md`

### Reason
Turn the RuntimeTask scheduler + reference model into a working single-node
continuous-batching server that P1 kernels can replace the numpy backend under.

### Decisions
- Prefill actions only advance bookkeeping (the reference recomputes full
  context every step); chunk boundaries will matter when real kernels arrive.
- Requests carry their own sampling config + seeded RNG.
- eos checked before append; capacity is strict so decode never OOMs.

### Validation
- `python -m unittest discover -s tests`: 89 tests OK (6 batched).
- Covered: order-preserving completion, generated-token counts, two sequences
  sharing a decode step (true interleaving), serialization under state cap 1
  with full resource recycling, chunked prefill producing exactly
  ceil(len/chunk) steps, seeded temperature determinism.
- `python -m serving.cli --batch "one two" "three four" --max-new 5 --seed 2`:
  two prompts served concurrently (garbled output as expected).
- NOT verified: cluster GPU, real 64-layer weights, kernels, live vLLM parity.

---

## 2026-08-29 - dgxspark-serve: runtime bookkeeping (allocators, memory planner, continuous-batching scheduler)

### Summary
Implemented the CPU-testable half of the serving runtime: paged KV block +
recurrent-state allocators with a hybrid coordinator, an FP8 KV memory planner
parameterized by the audited qwen3_5 geometry, and a continuous-batching
scheduler (admission control, chunked prefill, prefill->decode, finish->free
->re-admit). All pure bookkeeping, no GPU.

### Files Changed
- `dgxspark-serve/runtime/blocks.py` (new): KVBlockAllocator, StateAllocator,
  BlockTable, HybridKVCoordinator
- `dgxspark-serve/runtime/memory_planner.py` (new): fp8_kv_bytes_per_token,
  plan_block_count, qwen3_5_kv_profile
- `dgxspark-serve/runtime/scheduler.py` (new): Request/Action/Schedule/Scheduler
- `dgxspark-serve/tests/test_runtime.py` (new)
- `dgxspark-serve/README.md`

### Reason
Design doc 03 calls for these components before any kernel work; they are pure
bookkeeping and fully verifiable without an approved GPU window.

### Decisions
- Admission is strict (worst-case KV blocks for prompt_len+max_new allocated at
  admission), so decode never hits a capacity failure and preemption is
  deferred to a later phase.
- Recurrent state is one slot per sequence (per-linear-layer layout inside the
  slot is a future kernel concern).
- Scheduler returns a Schedule of Actions; an executor advances positions via
  `advance()` (tests emulate it).

### Validation
- `python -m unittest discover -s tests`: 83 tests OK (18 runtime).
- Covers: alloc/reuse/ownership/OOM, hybrid grow/free, planner math vs the
  audited 16-layer/4-KV/256 head geometry (32768 B/token FP8), admission
  (concurrency/state/KV caps, FIFO), chunked prefill bounds, prefill->decode,
  finish frees resources for the next request, early eos finish.
- NOT verified: cluster GPU, real weights, kernels, live vLLM parity.

---

## 2026-08-29 - dgxspark-serve: dev serving stub (E2E sampler/executor/http/cli)

### Summary
Added the first end-to-end serving slice that runs entirely on the dev machine:
post tokens (real Qwen tokenizer) through a tiny numpy model of the real vocab
size and back to generated text. Components are decoupled behind the
`InferenceEngine` interface so the kernel-backed model can replace the numpy
reference in P1 without touching the serving layer.

### Files Changed
- `dgxspark-serve/runtime/sampler.py` (new): greedy + temperature/top-k/top-p
- `dgxspark-serve/serving/executor.py` (new): ReferenceModel + generate +
  InferenceEngine (complete/chat)
- `dgxspark-serve/serving/server.py` (new): stdlib ThreadingHTTPServer,
  /health, /v1/chat/completions, /v1/completions
- `dgxspark-serve/serving/cli.py` (new): `python -m serving.cli --chat "hi"`
- `dgxspark-serve/serving/dev_model.py` (new): tiny recipe/weights with the
  real vocab size (248320) so real-tokenizer ids fit; build_dev_engine
- `dgxspark-serve/tests/test_sampler.py`, `test_executor.py`, `test_server.py`
  (new); `test_pipeline.py` refactored to reuse dev_model fixtures
- `dgxspark-serve/README.md`

### Reason
Make the whole request path (chat template -> tokenize -> forward -> sample ->
detokenize) testable and demonstrable without a GPU window, and give P1 a
stable Serving API to bind to.

### Findings / Decisions
- Tiny model vocab is set to the real 248320 so the real tokenizer ids are
  valid embedding row indices (logits/generation live in the real id space).
- `sampler.sample(..., temperature<=0)` == greedy (deterministic).
- Stop tokens are checked before append (a stopping token is never emitted).
- Server keeps the engine on the handler's `self.server` (avoids class-body
  closure pitfalls) and runs threaded for concurrent requests.

### Validation
- `python -m unittest discover -s tests`: 65 tests OK.
- `python -m serving.cli --chat "hello" --max-new 6 --seed 1`: end-to-end
  assistant text produced (garbled, as expected with random tiny weights;
  proves the full wiring incl. real tokenization/decode and chat template).
- HTTP smoke tests: /health, chat & text completions, bad-JSON and
  missing-messages -> 400.
- NOT verified: cluster GPU, 64-layer real weights, kernels, live vLLM parity.

---

## 2026-08-29 - dgxspark-serve: self-made byte-level BPE tokenizer with oracle parity

### Summary
Implemented a self-made byte-level BPE tokenizer (`serving/bpe.py`) and a
serving wrapper (`serving/tokenizer.py`: special-token splitting +
chat-template rendering). Encode output matches the official `tokenizers`
library on the real Qwen3.8-27B-FP8 tokenizer across unicode, Korean, numbers,
emoji, whitespace and special-token cases.

### Files Changed
- `dgxspark-serve/serving/bpe.py` (new): bytes_to_unicode, pre-tokenization,
  rank-based greedy BPE merge, encode/decode (byte-level lossless)
- `dgxspark-serve/serving/tokenizer.py` (new): loads vocab/merges/config,
  split_special_tokens=False handling, added-token id injection, jinja2 chat
  template
- `dgxspark-serve/tests/test_tokenizer.py` (new): synthetic BPE algorithm
  tests + real-file oracle tests (skipWithout cached files)
- `dgxspark-serve/README.md`

### Reason
Tokenizer is the next required piece before serving; byte-exact parity with HF
is mandatory for T2/T3 token identity. Implemented the merge algorithm
ourselves (only the Unicode regex engine `regex` is reused to run the
checkpoint's stored pre-tokenize pattern).

### Findings
- Qwen2Tokenizer special tokens (`<|im_start|>` etc.) are ADDED tokens with ids
  >= vocab size; they live only in `added_tokens_decoder`, not in vocab.json.
  They are injected into the id map so a special string becomes a single id.
- bytes_to_unicode must map to unicode chars (chr), not ints; the byte-fallback
  path in _bpe_to_ids only triggers for symbols missing from vocab (rare).

### Validation
- `python -m unittest discover -s tests`: 48 tests OK.
- Oracle parity on the real Qwen tokenizer (tokenizers 0.23 + cached files):
  encodes identical for all sampled inputs incl. "<|im_start|>user\nhi<|im_end|>"
  (special ids 248045/248046 match the oracle exactly); decode roundtrips;
  chat template renders <|im_start|>user/assistant markers.
- Dev deps added (test-only): `regex`, `tokenizers`, `jinja2`.
- NOT verified: cluster GPU runs, live parity vs vLLM (need cluster).

---

## 2026-08-29 - dgxspark-serve: real-weight layer T1 on checkpoint bytes

### Summary
Ran the reference forward on the actual Qwen3.8-27B-FP8 weights (fetched via
HTTP Range, no full download): layer-0 GatedDeltaNet and layer-3 full attention
with real FP8-dequantized weights. Added `load_tensors_from_url` to the
safetensors reader and `bench/audit_real_forward.py`.

### Files Changed
- `dgxspark-serve/loaders/safetensors_reader.py`: + `load_tensors_from_url`
  (HTTP Range based tensor loading)
- `dgxspark-serve/bench/audit_real_forward.py` (new, real-weight T1 tool)

### Reason
Validate the loader -> FP8 dequant -> reference-math integration on genuine
checkpoint bytes (not just synthetic data) before touching the cluster.

### Validation
- `python bench/audit_real_forward.py`: PASS
  - layer-0 GatedDeltaNet, real weights, seq 37: chunked-vs-recurrent
    max_abs_diff = **4.47e-8**; output RMS 0.00137 (input 0.05); state shape
    (1, 48, 128, 128) matches config.
  - layer-3 full attention (24 heads/4 KV, head_dim 256, partial M-RoPE):
    finite, rms 0.0395.
- Full suite: `python -m unittest discover -s tests` -> 35 tests OK.
- NOT verified: full-model/live parity vs vLLM and tokenizer (need cluster).

---

## 2026-08-29 - dgxspark-serve P1-prep: FP8/safetensors loaders, full-model reference pipeline, real-shard FP8 validation

### Summary
Extended the new engine (P0 -> P1 prep, still dev-machine only): FP8/BF16 decode
with blocked dequant, a minimal dependency-free safetensors reader, a
checkpoint weight loader, a full-model text reference forward (T2-ready), and a
HTTP-Range shard audit that proved the entire FP8 load path on real checkpoint
bytes.

### Files Changed
- `dgxspark-serve/loaders/fp8.py` (new): OCP E4M3FN decode (pinned to
  ml_dtypes), BF16 decode, per-128x128-block dequant + loop reference
- `dgxspark-serve/loaders/safetensors_reader.py` (new): minimal safetensors
  header/tensor reader
- `dgxspark-serve/loaders/weights.py` (new): dequant_tensors +
  load_recipe_weights (checkpoint-named weight dict, T2-ready)
- `dgxspark-serve/ref/pipeline.py` (new): embed -> 64 layers -> norm ->
  lm_head wiring over the checkpoint weight dict
- `dgxspark-serve/bench/audit_shard_fp8.py` (new): Range-based real-shard FP8
  validation tool
- `dgxspark-serve/tests/test_loaders.py`, `tests/test_pipeline.py` (new)
- `dgxspark-serve/README.md`, `ref/qwen3_5.py` (decoder_layer_forward fixes)

### Reason
P1 begins once components can be verified without a GPU window. This batch made
the weight pipeline (bytes -> FP8 decode -> blocked dequant -> reference
forward) real and testable on the dev machine, and validated it against actual
checkpoint bytes to retire the "FP8 layout is per-128x128 block" assumption
with evidence.

### Root Cause / Findings
- OCP E4M3FN semantics (verified vs ml_dtypes): subnormals supported
  (exp==0 -> mant*2^-9); exp==15 & mant==0..6 are normal (256..448); the only
  NaN is exp==15 & mant==0b111. Earlier draft treated all exp==15 as NaN and
  used a wrong subnormal exponent (2^-13) - corrected.
- Vectorized scale expansion in dequant used per-element indexing; fixed to
  repeat *block* indices (non-divisible shapes handled).
- Real shard evidence (`layers-0.safetensors`, HTTP Range): header parse OK
  (20 tensors), `in_proj_qkv.weight` F8_E4M3 (10240,5120) + scale (80,40);
  dequant range [-0.33, 0.44]; **fraction of 128x128 blocks with max abs fp8
  >= 400 is 1.000** (per-block saturated quantizer signature) - strong
  confirmation of decode + block layout.

### Validation
- `python -m unittest discover -s tests`: 35 tests OK (ref parity, recipe
  schema, fp8/bf16 vs ml_dtypes oracle, dequant vs loop, reader roundtrip,
  pipeline wiring vs manual composition).
- `python bench/audit_shard_fp8.py`: PASS on real checkpoint bytes.
- `python -m py_compile` on all changed Python: OK.
- Dev dependency added: `ml_dtypes` (test-only FP8 oracle).
- NOT verified: cluster GPU runs, tokenizer/chat, live parity vs vLLM.

---

## 2026-08-29 - dgxspark-serve: new model-serving engine project, P0 (design + audit + reference math)

### Summary
Started a from-scratch dual-node DGX Spark serving engine (`dgxspark-serve`)
for three models: Qwen3.8-27B-FP8, Qwen3.8-Flash-Next-FP8, DeepSeek-V4-Flash.
P0 delivered: design docs, a checkpoint audit that corrected two architecture
assumptions, a recipe schema v1, and a torch-free numpy reference math with
parity tests (20 pass). No GPU/cluster work performed.

### Files Changed
- `dgxspark-serve/README.md`, `docs/design/01-architecture.md` … `04-validation-phases.md` (new)
- `dgxspark-serve/docs/design/05-audit-qwen3-5.md` (new, evidence record)
- `dgxspark-serve/recipes/schema.py`, `recipes/qwen3_5.yaml` (new)
- `dgxspark-serve/ref/qwen3_5.py` (new, numpy reference of upstream math)
- `dgxspark-serve/tests/test_ref_qwen3_5.py`, `tests/test_recipes.py` (new)
- `dgxspark-serve/.gitignore` (new)

### Reason
User approved the PLAN for a homebrew serving engine (design-then-P0) targeting
dual DGX Spark. P0 goal: establish numeric-parity source of truth and recipe
abstraction before any kernel work.

### Root Cause / Key Findings (audit)
- The `qwen3_5` "linear attention" is a **GatedDeltaNet (gated delta rule)**,
  not a Mamba selective scan (corrected the design); both Qwen models share
  this kernel family.
- The `qwen3_5` MLP is **dense** (gate/up/down); the `mlp.gate` /
  `shared_expert_gate` names in `modules_to_not_convert` are template residue
  with no matching tensors.
- FP8 weights are F8_E4M3 with **per-128x128-block inverse scales**
  (`weight_scale_inv`, BF16), dynamic activation quantization.
- MTP = 1 full-attention layer + dense MLP + fc (BF16 fc).

### Validation
- `python -m unittest discover -s tests`: 20 tests OK (chunked-vs-recurrent
  delta-rule parity, state-chaining, norms/conv/rotary/attention/MLP reference
  checks, recipe schema load/validation).
- `python -m py_compile` on all new Python: OK.
- Audit facts verified against live HF artifacts (config.json, index, safetensors
  headers of shards 0/3).
- NOT verified: cluster GPU runs, real weight loading, tokenizer, live parity
  vs vLLM (deferred; needs approved GPU window).

---

## 2026-08-28 - Code review follow-up: fix all review findings

### Summary
Addressed every finding from the full-repo code review (C++ API misuse, double
free in the Phase 2 gate tools, manager counter-lock misuse, misleading
`free()`, unvalidated config values, residency attribute width, and stale
duplicate-tree drift). Added regression tests for the new behavior.

### Files Changed
- `vllm-docker-um/gb10um/um_ext/gb10_unified_memory_ext.cpp` (+ `prototype/um_ext/` copy)
  - `gb10_prefetch`: was calling `cudaMemPrefetchAsync(ptr, nbytes, cudaMemLocation, 0u, stream)`
    (struct where `int dstDevice` is expected + a 5th phantom arg that cannot
    match any overload) -> now `(ptr, nbytes, device, stream)` (standard 4-arg
    form). The old body would not compile on any CUDA runtime.
  - `loc_device()` now zero-initializes `cudaMemLocation` (had indeterminate
    fields); `read_mostly`/`dont_access_last` used C compound literals (invalid
    in C++) -> unified on `loc_device(device)`.
  - `gb10_range_residency`: `cudaMemRangeAttributeAccessedBy` fills an `int`,
    was read into `int64_t` (4 trailing garbage bytes); now read into `int`.
    `cudaMemLocation last` zero-initialized before the attribute read.
  - `probe_cuda_symbols()` now also requires `cudaMemAdvise` (we call it).
- `vllm-docker-um/gb10um/gb10_unified_memory.py` (+ identical `mods/`,
  `prototype/`, `overlay/` copies kept byte-identical)
  - `GB10UMConfig.from_dict`: coerces/validates types (int/float/bool/mode)
    and raises `GB10UnifiedMemoryError` on invalid values instead of silently
    assigning wrong-typed config (e.g. string limits from JSON).
  - `allocate()`/public `prefetch()`: counters now updated under the manager
    lock; `_prefetch_locked` renamed `_prefetch` (it never held the lock) and
    now takes the lock itself. `migration_time_s`/`um_kv_bytes` decremented
    consistently under lock.
  - `free()`: documented contract (memory is owned by the tensor's from_blob
    deleter -> the manager only drops bookkeeping/counters); removed the
    unconditional `torch.cuda.synchronize()` that ran even on unknown tensors.
  - `detect_capabilities`: exception fallback now writes `mem_free_gb`/`mem_total_gb`
    None (was the divergent `mem_info` key).
  - `_ExtBackend.get_info`/`get_memory_info` accept `device_idx` instead of
    hardcoding device 0.
- `vllm-docker-um/bench/gb10_um_benchmark.py` (+ synced root `bench/` copy)
  - `run_ctx`: removed the explicit `ext.free_managed(buf_m.data_ptr())` that
    double-freed a tensor whose from_blob deleter already calls `cudaFree`;
    release via `del buf_m, buf_c` + synchronize.
  - `parse_sizes`: accepts plain token counts and blanks (was `KeyError` on
    any suffix-less size such as `4096`).
- `vllm-docker-um/bench/flashinfer_direct_probe.py` (+ synced root `bench/` copy)
  - removed the same double-free (`ext.free_managed(km.data_ptr())` on a
    view whose storage deleter owns the pointer).
  - removed the dead `page_size` parameter and its unused `--page-size` CLI flag.
- `vllm-docker-um/tests/test_offline_cpu.py`: +2 tests (config coercion/
  rejection, plain-token + blank size parsing).
- Root `bench/` (drift fix): copied `gb10_um_benchmark.py`,
  `flashinfer_direct_probe.py`, `run_bench.sh`, `run_mem_limit_ab.sh` from the
  deployable `vllm-docker-um/bench/` (the root copy still carried the Phase-2
  pre-fix import path/units bug and the stale `vllm-node-qwen38-opt` tag).

### Reason
Review (2026-08-28) found: um_ext had never been build-verified and its
prefetch call could not compile; both Phase-2 gate scripts would double-free
managed backing; manager counters were updated outside the lock and `free()`
pretended to free memory it does not own; config values were unvalidated; the
root `bench/` tree diverged from the active one, so running it used the old
broken code.

### Validation
- `python -m py_compile` on all edited/ synced Python files: OK.
- `python tests/test_offline_cpu.py` (Windows python 3.14, torch stubbed,
  PyYAML 6.0.3): 23 tests OK (was 21; 2 new).
- SHA256 byte-identity re-checked for the 4 copies of `gb10_unified_memory.py`,
  the 2 copies of `gb10_unified_memory_ext.cpp`, and root `bench/` vs
  `vllm-docker-um/bench/` (all MATCH).
- NOT verified locally: um_ext JIT compile on aarch64 (no CUDA host here; the
  cudaMemAdvise `cudaMemLocation` overload availability remains ASSUMPTION to
  confirm at first real build), `bash -n` (no bash on this host; no .sh content
  changed - copies only), and any GPU/cluster run.

---

## 2026-08-28 - Offline CPU test suite added (GPU busy: code-only validation)

### Summary
Added and ran a hardware-free test suite covering all logic that can be verified
without GPU/cluster; one test-side tearDown bug found and fixed during the run.

### Files Changed
- `vllm-docker-um/tests/test_offline_cpu.py` (new): 21 tests
  - A: patcher apply/check/revert cycle, idempotency, ambiguous/unknown anchor
    rejection, CLI roundtrip
  - B: manager config defaults, canonical schema == `config_gb10_um.json.example`
    (drift guard), dtype map
  - C: env contract — disabled-by-default => `get_manager() None`;
    enabled without `um_ext` => hard fail (no unsafe fallback); config-file merge +
    unknown-key filtering; malformed JSON fails clearly; disabled allocate() is a no-op
  - D: recipe2env argv compile for all recipes + launcher heredoc
    (mp backend, --served-model-name, kv-dtype, enforce-eager, single --port,
    JSON braces, container tag)
  - E: launcher static invariants (no run-recipe.sh/Dockerfile.qwen38-opt/
    spark-vllm-docker residue; current REPO_DIR/IMAGE; guarded recipe write)
  - torch is stubbed when absent so the suite runs on plain python3

### Reason
Another model occupies the cluster; only code-level testing was permitted.

### Validation
- Windows python 3.11 (torch stubbed): `Ran 21 tests ... OK`
- WSL python3: `Ran 21 tests ... OK`
- `bash -n`: all repo shell scripts OK
- Not covered (needs GPU/cluster, unchanged from before): um_ext JIT build on
  aarch64, real `cudaMallocManaged` behavior, flashinfer 0.6.17 API surface,
  `tests/smoke.sh` inside the built image, and any serving run.

---

## 2026-08-28 - Phase 2 bench readiness fixes (cluster-independent prep)

### Summary
Fixed latent defects in the Phase 2 benchmark tooling and added the missing
FlashInfer direct-mode gate probe (WORK_ORDER_v2 Phase 2 gate #2).

### Files Changed
- `vllm-docker-um/bench/gb10_um_benchmark.py`
  - import path fix: hardcoded `prototype/` -> `_pkg_dir()` (resolves `gb10um/` in
    this repo, falls back to `prototype/`); `um_ext` path followed the same fix
  - unit bug: `--sizes` values were treated as raw BYTES (262K -> 268 KiB, smaller
    than one 196608 B block => meaningless/trivial runs); now context TOKEN counts
    with KV bytes = tokens * feat (feat=12288 matches ~12 KiB/token QSA FP8 KV)
  - added QSA-like sparse block-subset gather (64 blocks/step), pinned-host H2D
    reference (variant C), and `range_residency` reporting; default sizes include 4K
- `vllm-docker-um/bench/run_bench.sh`
  - added `--entrypoint bash` (base image entrypoint is the vLLM serve wrapper;
    `bash -c` as CMD would never execute - same reason start.sh overrides it)
  - extra CLI args passthrough; `--force` de-duplicated from passthrough
- `vllm-docker-um/bench/flashinfer_direct_probe.py` (new)
  - Phase 2 gate #2 probe: FlashInfer decode on managed-memory KV vs cudaMalloc KV,
    correctness (allclose + max abs diff) and latency (cold / prefetched-warmed /
    cuda); clear exit codes (2 busy, 3 flashinfer API unavailable, 4 mismatch)
- `vllm-docker-um/docs/gb10-unified-memory/WORK_ORDER_v2.md`: §5 tooling/units note

### Reason
Phase 2 is the sole remaining gate but its scripts had never run (latent bugs
unexercised): the import path would crash on this repo layout, the sizes unit bug
would produce invalid measurements, and run_bench.sh could not execute its payload
under the image entrypoint. The direct-vs-staged decision (v2 recommendation
deliverable) needs a FlashInfer correctness probe that did not exist.

### Validation
- `python -m py_compile` on both bench python files: OK
- `bash -n run_bench.sh`: OK
- API signatures cross-checked against `gb10um/um_ext/gb10_unified_memory_ext.cpp`
  (`create_managed_tensor/alloc_managed/free_managed/prefetch/mem_get_info/
  range_residency`) and `gb10um/gb10_unified_memory.detect_capabilities`
- NOT yet run on hardware (cluster 10.100.25.1 unreachable from this PC; flashinfer
  0.6.17 `single_decode_with_kv_cache` surface is ASSUMPTION, verified at probe
  runtime with exit-3 fallback): requires approved idle window.

---

## 2026-08-28 - GB10-UM work order revised (v2)

### Summary
Reviewed the chat-issued "GB10 Unified Memory KV Cache Manager" development work order
and issued a corrected v2.

### Files Changed
- `vllm-docker-um/docs/gb10-unified-memory/WORK_ORDER_v2.md` (new)

### Reason / Root Cause
The v1 work order predated the current repo state: it re-commanded finished phases
(0/1 and the authored integration patch), targeted the removed Ray multi-node setup and
the old `vllm-node-qwen38-um` image name, described UM on GB10 as separate slower memory
(it is one shared coherent pool; UM buys allocation/over-subscription semantics), used a
sequential N+1/N+2 prefetch model that does not match attention access (full-history /
QSA-indexer block-subset reads), left the pressure metric and
`determine_available_memory()` accounting coupling undefined, validated FlashInfer too
late (Phase 7), allowed an unsafe config fallback contradicting the implemented
hard-fail contract, and missed prefix-caching/chunked-prefill/multimodal test cases.

### Validation
Doc-only change. Cross-checked every v2 statement against
`kv-cache-architecture.md`, `gb10-um-analysis.md`, `gb10-um-benchmark.md`,
`gb10_unified_memory.py`, `config_gb10_um.json.example`, `apply_gb10_um_patch.py`,
and the current recipes/runner. Execution resumes at Phase 2 (benchmark gate).

---

## 2026-08-28 - Code review follow-up on the launch fix

### Summary
Self code review of the previous fix; three findings corrected.

### Files Changed
- `vllm_qwen38_flash_next_fp8.sh`
  - Image step: build solo (`./build.sh`) when only head misses the image;
    `./build.sh -c` (build-if-missing + worker copy) only when the worker misses it
    (restores the old "copy only when worker missing" reuse-first behavior; avoids a
    full `docker save` transfer when head-only was missing).
  - Recipe write is now guarded: the embedded heredoc is written only for
    `RECIPE=qwen3.8-flash-next-fp8-um.yaml`; any `RECIPE=` override must point at an
    existing recipe and is used as-is (prevents clobbering the sklee rollback recipe).
- `vllm-docker-um/recipes/qwen3.8-flash-next-fp8-um.yaml`: description referenced the
  stale tag `vllm-node-qwen38-um` -> `vllm-node-um` (matches its own `container:`).

### Known residual risks (not changed - pre-existing runner behavior)
- `scripts/start.sh` launches rank0 (head) before rank1; a worker launch failure exits
  via `set -e` and can leave an orphaned head `vllm_node` container.
- Launcher step 0 hardcodes container name `vllm_node` while start.sh honors
  `CONTAINER_NAME=`.
- `RUN_EXTRA_ARGS` is word-split (args containing spaces unsupported; pre-existing).
- Docs under `docs/gb10-unified-memory/` still describe the historical "ray executor"
  serving config (kept as historical analysis records).

### Validation
- `bash -n` on the launcher: OK.
- recipe2env parse of all 5 recipe variants: all checks pass (backend mp,
  --served-model-name, --kv-cache-dtype fp8, --enforce-eager, JSON args intact).

---

## 2026-08-28 - Fix vLLM dual-node launch: ray/mp backend conflict and repo/image drift

### Summary
Aligned all Qwen3.8-Flash-Next-FP8 serving recipes and the top-level launcher with the
`vllm-docker-um` runner (`./start.sh`), switched the distributed executor backend from
`ray` to `mp`, and unified the container image tag to the built `vllm-node-um:latest`.

### Files Changed
- `vllm_qwen38_flash_next_fp8.sh`
  - `REPO_DIR` default `/home/sklee/spark-vllm-docker` -> `/home/sklee/vllm-docker-um`
  - `IMAGE` default `vllm-node-qwen38-opt:latest` -> `vllm-node-um:latest`
  - `RECIPE` default -> `qwen3.8-flash-next-fp8-um.yaml` (sklee recipe left intact as the
    production/rollback recipe; it is no longer overwritten by this script)
  - image step now reuses `./build.sh -c` (build only if missing, then copy to worker)
  - launch via `./start.sh` instead of the missing `run-recipe.sh`
  - mount step dropped (runner `docker_opts` already mounts `/mnt/external/models:/models`)
  - embedded recipe: backend `mp`, added `--kv-cache-dtype fp8` and `--enforce-eager`
- `qwen3.8-flash-next-fp8-um.yaml` (root copy): container tag -> `vllm-node-um:latest`,
  backend `ray` -> `mp`, added `--kv-cache-dtype fp8` / `--enforce-eager`
- `qwen3.8-flash-next-fp8-sklee.yaml` (root copy): backend `ray` -> `mp`
- `vllm-docker-um/recipes/qwen3.8-flash-next-fp8-sklee.yaml`: backend `ray` -> `mp`
- `vllm-docker-um/recipes/qwen3.8-flash-next-fp8-um.yaml`: added
  `--kv-cache-dtype fp8` / `--enforce-eager` (match the validated serving config in
  `docs/gb10-unified-memory/kv-cache-architecture.md`)

### Reason / Root Cause
- `scripts/start.sh` always appends `--nnodes/--node-rank/--master-addr/--master-port`
  (mp external-launcher multi-node; documented in `docs/B12X-ARCH.md`), which conflicts
  with `--distributed-executor-backend ray` in the recipes -> vLLM argument/validation
  failure and rank0 container exit during the 120s survival poll.
- The launcher still targeted the old repo (`run-recipe.sh`, `Dockerfile.qwen38-opt`,
  `vllm-node-qwen38-opt:latest`) while the active runner/image are `start.sh` /
  `vllm-node-um:latest` -> preflight "repo dir not found" or "container image missing".
- A suspected corrupted heredoc line (`--served-model-name`) was checked at byte level
  and found intact (display artifact only); no change needed.

### Validation
- `bash -n`: launcher, `scripts/start.sh`, `scripts/build.sh` - syntax OK.
- `recipe2env.py` parse of all 5 recipe variants (root x2, repo x2, launcher heredoc):
  argv len 44, single `--port`, `--served-model-name qwen` present, backend `mp`,
  `--kv-cache-dtype fp8` and `--enforce-eager` present, JSON args single-braced intact.
- `./start.sh --dry-run recipes/qwen3.8-flash-next-fp8-um.yaml` (WSL): rank0/rank1
  docker commands serialize byte-for-byte with `--nnodes 2 --node-rank <r>
  --master-addr 10.100.25.1 --master-port 29501`; run dir removed afterwards.
- Not validated locally (cluster unreachable from this PC, 10.100.25.1:22 timeout):
  actual `vllm serve` acceptance of `--nnodes` with mp on the pinned vLLM build, and
  image presence of `vllm-node-um:latest` on both nodes. Validate on the head node:
  `./start.sh --dry-run recipes/...`, then real launch + `curl :8100/v1/models` +
  check `logs/run-*/head*.container.log`.
