# sLLM

sLLM is a from-scratch **model serving engine for dual-node NVIDIA DGX Spark
(GB10)** that serves several models, one at a time, under a single engine. It
exposes an OpenAI-compatible HTTP API (chat / completions, SSE streaming) and
a small CLI, with a recipe-driven configuration model.

This README is the **user guide** — install, configure, run, and serve models.
Design/implementation details live under [`docs/design/`](#design-documents).

---

## 1. Supported models

Each model has a **recipe** (`recipes/<Model>.yaml`) that describes the model
identity, geometry, weight location and launch settings.

| Recipe | Model | Architecture | Weights |
|---|---|---|---|
| `Qwen3.8-27B-FP8.yaml` | Qwen/Qwen3.8-27B-FP8 | GDN linear-attention + dense MLP + MTP + vision | ~29 GiB |
| `Qwen3.8-Flash-Next-FP8.yaml` | Qwen/Qwen3.8-Flash-Next-FP8 | GDN + QSA sparse + MoE + MTP + PLE/ngram | ~173 GiB |
| `DeepSeek-V4-Flash-0731.yaml` | deepseek-ai/DeepSeek-V4-Flash-0731 | MLA + sparse + MoE (fp4) + DSPark spec decode | ~156 GiB |
| `Qwen2.5-Coder-0.5B.yaml` | Qwen/Qwen2.5-Coder-0.5B | standard dense transformer | ~1 GiB |

**Runnability today**

- **Local (dev machine, no GPU / no checkpoint):** tiny reference engines —
  `python -m serving.cli` and `python -m serving.server` (built-in weights).
- **Real standard model:** `Qwen2.5-Coder-0.5B` (and other Llama/Qwen2-family
  checkpoints) via `serving/serve_standard.py`.
- **Real qwen4_exp / deepseek_v4 checkpoints:** cluster-gated (full fp8/fp4
  GPU + TP2 milestones). Use `--mode plan` to see the run plan without a GPU.

---

## 2. Installation

Requirements: **Python 3.10+**, `numpy`, `pyyaml`, `regex`, `jinja2`.
Docker is needed for the container launcher `sllm`; CUDA tooling (nvcc 13.0)
is only needed on the cluster to build the GPU kernels.

```bash
# Dependencies are VERSION-PINNED in requirements.txt (runtime only; dev/
# validation packages like ml_dtypes/tokenizers/torch are not shipped).

# container image (recommended for deployment) -- pinned deps + runtime-only files
./build.sh

# native install on a node (venv + pinned deps + GPU kernels)
./build.sh --native
```

The container image ships the **deployment-only** source tree: dev/validation
resources (`tests/`, `bench/`, `oracle/`, `demo/`, `docs/`, vendored
`ref/hf_sources/`, dev entrypoints `serving/cli.py` / `serving/dev_model.py`)
are excluded via `.dockerignore` and the Dockerfile's explicit COPY list.

No install step is required for the Python paths — run `python -m ...` from
the repository root.

---

## 3. Configuration

Configuration is split in two places, intentionally:

- **Recipe = the model** (identity, geometry, `defaults:`, `env:`, `command:`
  template, weights `paths.local_dir`).
- **`config.env` = common cluster / serving settings** (node IPs, pair link,
  fallback bind address/port, toolchain, image).

| Key (`config.env`) | Meaning | Default |
|---|---|---|
| `SLLM_HEAD_IP` / `SLLM_WORKER_IP` | SSH/coordination address of each node | 192.168.0.250 / .231 |
| `SLLM_HEAD_PAIR_IP` / `SLLM_WORKER_PAIR_IP` | internal pair-link addresses (NCCL) | 10.100.25.1 / .2 |
| `SLLM_PAIR_IFACE` | pair-link NIC name on **this** node | (empty) |
| `SLLM_HOST` | serve bind address when the recipe has no `defaults.host` | 0.0.0.0 |
| `SLLM_PORT` | serve bind port when the recipe has no `defaults.port` | 8002 |
| `SLLM_IMAGE` | container image for `sllm` | sllm-node:latest |
| `SLLM_NODE_WEIGHT_BUDGET_GIB` | max weights GiB per node for a lower-TP override | 110 |
| `NVCC_ARCH` | nvcc target for `kernels/cuda/build.sh` | native |
| `Q27B_TOKENIZER_DIR` | tokenizer snapshot dir (tests/dev only) | (empty) |

**Precedence:** CLI flag > recipe `defaults:` > `config.env` > built-in
default. Real environment variables always win over `config.env`
(`config.env > code default` is the fallback chain for the rest).

- Serving host/port primarily come from the **recipe**
  (`defaults.host` / `defaults.port`); `SLLM_HOST` / `SLLM_PORT` are the
  fallback for paths without a recipe.
- Node-pair IPs come from **`config.env`** (`tp/topology.py`).
- The `sllm` launcher bind-mounts the host's **live `config.env`** into the
  container, so changing node IPs / port / host on the node applies without
  rebuilding the image.

---

## 4. Quick start

### 4.1 Plan (no GPU, safe anywhere)

```bash
sllm recipes/Qwen3.8-Flash-Next-FP8.yaml            # default mode = plan
sllm recipes/Qwen3.8-27B-FP8.yaml --tp 1            # force a 1-node plan
```

`plan` resolves everything and prints the run plan (rank table, weights,
cache budget, serving address) without executing.

### 4.2 Run (one-shot chat)

```bash
sllm recipes/Qwen2.5-Coder-0.5B.yaml --mode run --chat "say hi" --max-new 16
```

### 4.3 Serve (OpenAI-compatible HTTP API)

```bash
sllm recipes/Qwen2.5-Coder-0.5B.yaml --mode serve   # OpenAI API on :8002
```

The same entry runs natively from the repository root:

```bash
python -m serving.main recipes/Qwen2.5-Coder-0.5B.yaml --mode serve \
    --host 127.0.0.1 --port 8002
```

### 4.4 Dev / tiny engines (no weights, no GPU)

```bash
python -m serving.cli --chat "hello"                # tiny standard model
python -m serving.cli --qwen4 --chat "hi"           # tiny qwen4_exp (HC+GDN+QSA+MoE)
python -m serving.cli --batch "a b" "c d"           # continuous batching demo
python -m serving.server                            # tiny model HTTP stub ($SLLM_PORT / 8000)
```

---

## 5. CLI reference

### Docker launcher (`sllm <recipe>`)

```
sllm <recipe> [--tp|--nodes 1|2] [--mode plan|run|serve]
             [--host H] [--port P] [--chat TEXT] [--max-new N]
             [--model-dir D] [--dry]
```

| Flag | Meaning |
|---|---|
| `--tp / --nodes N` | drive shape (1 or 2); default from the recipe. A lower value is accepted only when `defaults.weights_gib` provably fits one node |
| `--mode` | `plan` (default) / `run` / `serve` |
| `--host / --port` | override the resolved bind address / port |
| `--chat TEXT / --max-new N` | one-shot run input / length |
| `--model-dir D` | override the recipe weights location |
| `--dry` | print the docker command without running it |

### Native entry (`python -m serving.main`)

Same flags plus:

| Flag | Meaning |
|---|---|
| `--log-level L` | `TRACE\|DEBUG\|INFO\|WARNING\|ERROR` (default `$SLLM_LOG_LEVEL`/INFO) |
| `--version` / `-V` | print `sllm <version> (<git rev>)` and exit |

---

## 6. HTTP API (OpenAI-compatible)

Endpoints served by `--mode serve`:

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "model": ...}` |
| `GET /v1/models` | list the served model |
| `POST /v1/chat/completions` | chat completion (OpenAI schema + `usage`) |
| `POST /v1/completions` | text completion |

```bash
curl http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-Coder-0.5B","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

`"stream": true` returns an **SSE** token stream (`data:` frames terminated by
`data: [DONE]`). Body fields: `messages` (chat) / `prompt` (completions),
`max_tokens` (or `max_new`), `temperature`, `top_k`, `top_p`, `seed`.

---

## 7. Serving standard checkpoints (`serve_standard`)

For real Llama/Qwen2-family checkpoints:

```bash
# one-shot
python -m serving.serve_standard --model-dir ~/models/Qwen2.5-Coder-0.5B \
    --prompt "def fib(n):" --max-new 16
# continuous-batch
python -m serving.serve_standard --model-dir ~/models/Qwen2.5-Coder-0.5B \
    --batch "p1" "p2"
# HTTP serve
python -m serving.serve_standard --model-dir ~/models/Qwen2.5-Coder-0.5B \
    --serve --host 0.0.0.0 --port 8002
```

Options: `--host` / `--port` (`$SLLM_HOST` / `$SLLM_PORT` fallback),
`--kv-placement device|host`, `--use-gpu`, `--gpu-dtype fp32|bf16`.

---

## 8. Diagnostics (vLLM-style)

Every engine prints a **startup banner** (version, model, backend, weights,
cache budget, arch details) and **per-request stats** (`prompt/out tokens`,
wall time, `tokens/s`, finish reason) as tag-prefixed lines on **stderr**:

```
13:07:30 INFO  [sllm] tiny/qwen4_exp (arch=qwen4_exp)
13:07:30 INFO  [sllm]   version       : 0.1.0 (a12545a)
13:07:30 INFO  [sllm]   architecture  : qwen4_exp
13:07:51 INFO  [gen] prompt=5 out=2 wall=0.006s (351.8 tokens/s) finish=length
13:07:51 INFO  [http] POST /v1/chat/completions -> 200 7.2ms
```

- Levels: `TRACE` (per-op) / `DEBUG` (per-token ids + arch detail) / `INFO`
  (banner + request stats, default) / `WARNING` / `ERROR`.
- Set via `--log-level L` or the `SLLM_LOG_LEVEL` env var.
- Per-token detail (token id + top-1) is at `DEBUG`.

---

## 9. Container operation

```bash
deploy/run.sh build                       # build sllm-node:latest
deploy/run.sh kernel                      # GPU kernel smoke
SLLM_BATCH="p1;p2" deploy/run.sh batch    # continuous-batch one-shot
deploy/run.sh serve                       # HTTP serving (MODEL_DIR=... override)
```

The `sllm` launcher bind-mounts the host's **live `config.env`** (and a recipe
from outside the repo) into the container, so editing cluster/serving settings
on the node does not require a rebuild.

---

## 10. KV memory placement

`memory.kv_placement` in the recipe (`device` default | `host`):

| Placement | Behaviour | OOM safety |
|---|---|---|
| `device` | weights + compute + KV on the GPU, admission rejects overflow | hard bound from the planned budget |
| `host` | KV / recurrent state in host RAM, gathered per decode step | recoverable failure / swap |

CPU/numpy mode always uses host RAM. CLI override:
`serve_standard --kv-placement device|host`.

---

## Design documents

- [`docs/design/01-architecture.md`](docs/design/01-architecture.md) — architecture, data flow, directory layout
- [`docs/design/02-recipes-kernels.md`](docs/design/02-recipes-kernels.md) — recipe schema, per-model kernel inventory
- [`docs/design/03-runtime-tp.md`](docs/design/03-runtime-tp.md) — batching/scheduling, KV/memory, tensor parallelism
- [`docs/design/04-validation-phases.md`](docs/design/04-validation-phases.md) — parity/perf methodology and phase plan
- [`docs/design/08-cluster-layout.md`](docs/design/08-cluster-layout.md) — GB10 pair, vLLM coexistence
- [`docs/design/09-roadmap-seq-qwen4exp-dsv4.md`](docs/design/09-roadmap-seq-qwen4exp-dsv4.md) — qwen4_exp → deepseek_v4 roadmap
- [`docs/design/10-deepseek-v4-vision-track.md`](docs/design/10-deepseek-v4-vision-track.md) — DeepSeek-V4 vision track
