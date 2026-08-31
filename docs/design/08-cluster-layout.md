# 08 — Cluster Deployment Layout (DGX Spark dual node)

Operating notes for the real serving target, as verified at test time.

## 1. Nodes

| role | external (dev machine view) | internal pair link | OS / arch |
|---|---|---|---|
| head (n1) | `192.168.0.250:22` (sklee) | `10.100.25.1` | Ubuntu, aarch64 (GB10) |
| worker (n2) | `192.168.0.231:22` (sklee) | `10.100.25.2` | Ubuntu, aarch64 (GB10) |

- The two nodes talk to each other over `10.100.25.x`; the dev machine reaches
  them over `192.168.0.x`.
- Access from the dev machine: SSH (password auth) via a transient paramiko
  helper; credentials are never stored in the repo.

## 2. Model store

Both nodes keep checkpoints in `$HOME/models/<model>`
(`/home/sklee/models/...`). Audit (head sizes shown; worker holds the same big
set):

| model dir | size (head) |
|---|---|
| DeepSeek-V4-Flash-0731 | 156 G |
| Qwen3.5-122B-A10B-FP8 | 119 G (head only) |
| Qwen3.6-27B (BF16) | 52 G (head only) |
| Qwen3.6-27B-FP8 | 29 G (head only) |
| Qwen3.6-27B-NVFP4 | 21 G |
| Qwen3.8-27B (BF16) | 59 G |
| Qwen3.8-27B-DFlash2 | 3.6 G |
| Qwen3.8-27B-FP8 | 29 G (both) |
| Qwen3.8-27B-NVFP4 | 22 G |
| Qwen3.8-Flash-Next-FP8 | 173 G (both) |
| Qwen3.8-Flash-Next-NVFP4 | 126 G |
| gemma-4-31B-it | 59 G |
| Qwen2.5-Coder-0.5B | 0.95 G (head only; added for the low-capacity test) |

Recipe local paths are recorded in each recipe as `paths.local_dir`
(`~/models/<model>`), so the engine can resolve the checkpoint on the node.
Note: not every model is mirrored on both nodes; Qwen2.5-Coder-0.5B and the
3.5/3.6 head-only sets would need copying for a TP2/two-node run.

## 3. Running services / constraints

- The head runs the shared **vLLM** container `vllm_node`
  (`eugr/spark-vllm-b12x:nightly-*`). It must not be disturbed.
- Host memory is GB10 unified (~128 GB/node, often ~7-8 GB free while vLLM
  runs). Our engine currently operates in **CPU/RAM-only** mode (numpy
  reference) and must stay light so it does not fight vLLM for host memory.
- Rule (user): heavy transformers/torch comparisons run on the dev machine
  (WSL), not on the cluster.

## 4. Our engine on the cluster

- Repo: `/home/sklee/dgxspark-serve` (head checkout as of 2026-08; the
  project directory is now `sLLM` — re-sync to `/home/sklee/sLLM` on the
  next deploy); venv `/home/sklee/dgsvenv`
  (numpy/pyyaml/regex/jinja2/ml_dtypes/tokenizers + torch/transformers for
  validation).
- Serve entrypoints:
  - `python -m serving.serve_standard --model-dir <dir> --prompt ...` (one-shot)
  - `... --serve --port 8002` (HTTP; the standard engine)
- Verified: full test suite (105 OK / 25 skipped) on ARM64; Qwen2.5-Coder-0.5B
  one-shot (2.75s/8 tokens) and HTTP smoke; logits/greedy output match
  transformers exactly.
- TP2 (the 27B/Flash-Next/DeepSeek recipes) is the P3 target on this layout:
  weights are identical on both nodes, internal link `10.100.25.x`.
