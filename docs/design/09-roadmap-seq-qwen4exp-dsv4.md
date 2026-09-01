# 09 — Sequential Support Roadmap: qwen4_exp -> deepseek_v4

Decision (2026-08-30, user): support the remaining recipe arches sequentially,
`qwen4_exp` first, then `deepseek_v4`. This document supersedes the ordering
assumptions in `04-validation-phases.md` sections P4/P5 (content of the phases
is unchanged; sequencing and evidence sources are made concrete here).

## 1. Evidence base (verified 2026-08-30)

| Item | Path | Size / facts |
|---|---|---|
| qwen4_exp checkpoint | head `~/models/Qwen3.8-Flash-Next-FP8` | 173 GB, 131 shards, fp8 e4m3 + bf16 `weight_scale_inv` (loader already supports) |
| deepseek_v4 checkpoint | head `~/models/DeepSeek-V4-Flash-0731` | 156 GB, 48 shards, bf16 dense + fp4 (I8-packed e2m1) MoE with F8_E8M0 block-32 scales (loader gap) |
| qwen4_exp oracle code | extracted from image `lmsysorg/sglang:qwen38flashnext` into repo `oracle/upstream/sglang/` | models/qwen4_exp.py (2132 L), qwen4_exp_mtp.py (220 L), layers/hyperconnection.py (344 L), layers/attention/qwen_sparse_attn_backend.py (1912 L), layers/attention/qsa/* (10 files), configs/qwen4_exp.py (165 L), models/qwen3_5.py (2434 L, GDN cross-check) |
| deepseek_v4 oracle code | bundled in checkpoint `inference/` + `encoding/` | model.py, kernel.py, generate.py, convert.py, encoding_dsv4.py |
| live baselines on head | sglang image (qwen38), `skfine24/qwen38-flash-next-vllm`, `vllm-node-qwen38-um`, `aidendle94/sparkrun-vllm-ds4-gb10` containers | T3 parity targets |
| engine today | arches `qwen3_5`, `qwen3_5_moe`; kernels `standard_gqa`, `paged_flash`; fused bf16 device-resident decode (standard_gqa) done | full CPU oracle stack `ref/qwen3_5.py`, `ref/pipeline.py`, `ref/mtp.py` exists |

## 2. Reuse map

| qwen4_exp component | engine status | work |
|---|---|---|
| GatedDeltaNet linear attn (36/48 layers) | `ref/qwen3_5.py` full oracle | cross-check vs `oracle/upstream/sglang/sglang_qwen3_5.py`; wire HC around it |
| QSA full attention (12/48) | partial-RoPE + q/k-norm + GQA oracle exist | new: indexer + block-sparse selection (port `qsa/qsa_indexer.py`, `sparse_attn.py`) |
| hyper-connection mixer | none | port `hyperconnection.py` (`GatedResidual`, `HyperConnectionConfig`) |
| MoE 512/top-10 + 1 shared (fp8) | fp8 block loader exists | new: router/top-k/expert-sum oracle (dense emulation first, grouped GEMM later) |
| PLE / ngram / MTP-hybrid spec | `ref/mtp.py` (qwen3_5 single-fc) | deferred to last (new 2-fc variant, ngram embed 20M shards) |
| FP8 e4m3 + [128,128] scales | `loaders/fp8.py` | reuse as-is |
| vision | none | out of scope (text-only serving track) |
| full 173 GB model on 128 GB node | n/a | layer-subset validation on one node; full serving needs TP2 (`03-runtime-tp`) |

DeepSeek reuse (unchanged from the 2026-08-30 plan review): MLA fits
`IncrementalCache` as `(kvh=1, cap, 576)` with V = latent[0:512]; needs a
rope-offset extension (rotates the LAST 64 dims); fp4/ue8m0 loader (also
unlocks the three NVFP4 Qwen checkpoints); MoE oracle shares the Q3 design
(noaux_tc + sqrtsoftplus variant).

## 3. Milestones (sequential; each ends green with tests + log.md entry)

### Track Q — qwen4_exp

| # | Deliverable | Parity gate |
|---|---|---|
| Q1 | Component oracles in `ref/qwen4_exp.py` (numpy, CPU): hyper-connection mixer, QSA indexer + sparse attention selection, MoE routing/experts (dense emulation) — each ported from `oracle/upstream/sglang/`, unit tests on synthetic tensors | T0 per component |
| Q2 | Layer + pipeline wiring: `qwen4_exp` layer forward (GDN+HC / QSA+HC / MoE), tiny synthetic fixture in `serving/dev_model.py`, `IncrementalCache` extensions (HC stream, indexer state), decode step path | T1 layer, T2-token parity on synthetic model |
| Q3 | Real-checkpoint subset parity on head: load actual fp8 tensors for one linear + one QSA layer + MoE via existing loader, oracle forward, measured noise floor vs dequant-matmul reference | T1/T2 on real weights (single node) |
| Q4 | Engine integration: arch `qwen4_exp` through executor (CPU incremental first), then device kernels (reuse fused dtype machinery; GDN scan + QSA sparse attention kernels) | regression 161 tests green |
| Q5 | Full-model path: TP2 dual-node plan execution + T3 live parity vs deployed sglang/vLLM; MTP-hybrid + PLE/ngram spec decode last | T3 |

Track Q status (updated 2026-09-01; regression 357 OK / 34 cluster-skip):
- Q1/Q2/Q4 DONE (oracles, pipeline, tiny fixture, engine path).
- PLE/ngram + MTP-hybrid (Q3+ item) DONE locally: ngram oracle
  `ref/qwen4_exp_ple.py`, pipeline wiring `_ple_forward` (hyper `+= PLE`,
  shared ngram ctx + per-layer conv state), MTP fusion/draft
  `ref/qwen4_exp_mtp.py`, runtime entry `runtime/spec.py::
  spec_decode_greedy_qwen4exp` (incl. batched S>1 hyper-injection). The MoE
  router is a plain dense softmax-topk (ngram influence arrives via the PLE
  feature on the hyper stream) -- no router change needed.
- Q3 (real-subset noise floor): cluster-gated via `bench/q4_subset_parity.py`.
- Q5 TP2 plan: local sharding exists (`loaders.tp_shard.Qwen4ExpSharding` +
  `tp/`); dual-node T3 live parity = cluster. Vision tower (VP170, `out_hidden
  2560`) is the remaining unimplemented Q3+ item -- the vendored
  `modeling_qwen4_exp.py` rotary-length arithmetic is not locally
  torch-verifiable (rope_dim vs head_dim), so the port stays cluster-gated.

### Track D — deepseek_v4 (after Track Q)

| # | Deliverable | Parity gate |
|---|---|---|
| D1 | fp4/ue8m0 loader: e2m1 nibble unpack + F8_E8M0 decode + block-32 dequant in `loaders/` (numpy oracle, oracle code = checkpoint `inference/convert.py`) | T0; unblocks NVFP4 Qwen too |
| D2 | MLA oracle + rope-offset generalized kernel + device-resident MLA decode (`kvh=1, cap, 576`) | T0/T1 |
| D3 | MoE V4: noaux_tc routing + sqrtsoftplus + top-6/shared-1 oracle, grouped-GEMM CUDA port | T0/T1 |
| D4 | HDC (hc_mult 4, sinkhorn 20) + sparse index/hash components | T0/T1 |
| D5 | Real-model: layer-subset parity -> TP2 dual-node (156 GB) -> DSPark spec decode + message encoder | T2/T3 |

Track D status (updated 2026-09-01):
- D1-D4 DONE: e2m1/ue8m0 loader (`loaders/fp8.py`, `tp_shard.DeepseekV4Sharding`),
  MLA + rope-offset oracle with device-resident decode, noaux_tc/sqrtsoftplus/
  top-6/shared-1 MoE oracle, HDC hc_mult-4 (incl. DSPark spec draft model).
- D5 local parts DONE: `DeepseekV4SpecModel` + backing spec decode
  (`runtime/spec.py::spec_decode_greedy_deepseek`, greedy-identity invariant),
  vision tower + VL prefill/chat (`ref/vision_deepseek.py` bf16, torch-free),
  message encoder (`serving/encoding_dsv4.py`, tool merge).
- D5 cluster-gated: TP2 dual-node on the 156 GB checkpoint, real-checkpoint
  OpenAI-vision E2E (`tests/test_real_checkpoint_cluster.py`), fused fp4/MLA
  CUDA kernels (`kernels/_deepseek_cuda.py` + `cuda/deepseek.cu` parity stubs;
  `.so`-gated tests skip locally).

## 4. Validation and regression rules

- Existing 161-test suite must stay green at every milestone (CPU-runnable).
- Every oracle claim cites its upstream source file/lines in a comment.
- fp8/fp4 paths compare against BF16-dequant reference; tolerance is measured
  per layer, never assumed (rule of doc 04).
- GPU windows: layer-subset tests only; full-model runs only in Q5/D5 TP2.

## 5. Risks

- sglang oracle depends on fused/triton internals; port semantics from the
  pure-python reference paths in those files, flag any triton-only semantics
  with a probe test before committing to math.
- QSA decode-graph interplay (`get_qsa_indexer_metadata`) is engine-side in
  sglang; our CPU oracle defines the contract, engine re-implements.
- 173 GB (qwen4) and 156 GB (dsv4) exceed one GB10 node: subset parity
  validates math; throughput claims require TP2.
