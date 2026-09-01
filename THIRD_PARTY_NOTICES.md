# Third-party notices

sLLM is MIT-licensed for the code written in this repository (see LICENSE).
Some directories contain **vendored upstream sources and references that are
NOT our code** — they are kept for architecture analysis/audits and are
excluded from the deployment image (`.dockerignore` / `deploy/Dockerfile`).
The model **weights** referenced by the recipes are never shipped in this
repository.

## Vendored / referenced upstream source (analysis only)

| Path | Upstream | License (as published) |
|---|---|---|
| `ref/hf_sources/dsv4/` | DeepSeek model code | MIT (DeepSeek) |
| `ref/hf_sources/modeling_qwen4_exp.py` | Qwen model code | Apache-2.0 (Qwen) |
| `oracle/upstream/` (sglang, transformers) | sglang / HuggingFace transformers | Apache-2.0 |
| `ref/mtp.py` (derivative notes) | MTP draft math per vLLM model code | Apache-2.0 (vLLM) |

These files retain their respective upstream copyright/license notices.

## Model checkpoints (not distributed here)

Recipes describe weights by name/path; the checkpoints themselves are governed
by their publishers' licenses (e.g., Qwen models: Apache-2.0; DeepSeek models:
MIT). Users are responsible for obtaining them under the applicable terms.

## Binary/toolchain dependencies

Runtime Python deps are pinned in `requirements.txt` (MIT/BSD/Apache-2.0
licensed). The CUDA build uses NVIDIA CUDA/cuBLAS/cuBLASLt under the NVIDIA
Software License; NCCL under the NCCL license.
