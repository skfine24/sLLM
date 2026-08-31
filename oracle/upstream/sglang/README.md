# Upstream reference sources (READ-ONLY)

Extracted from the deployed serving images to serve as the numerical source of
truth for `ref/` ports (docs/design/04 rule: "code, not guess"). Do not import
these files at runtime; they depend on sglang internals.

Provenance (extracted 2026-08-30 from docker image `lmsysorg/sglang:qwen38flashnext`,
paths under `/sgl-workspace/sglang/python/sglang/srt/`, sglang is Apache-2.0):

| local file | upstream path | role |
|---|---|---|
| qwen4_exp.py | models/qwen4_exp.py | qwen4_exp model (HC, QSA wiring, MoE, embedding) |
| qwen4_exp_mtp.py | models/qwen4_exp_mtp.py | MTP (hybrid) draft path |
| qwen4_exp_config.py | configs/qwen4_exp.py | config classes / knob defaults |
| hyperconnection.py | layers/hyperconnection.py | GatedResidual / HyperConnectionConfig |
| qwen_sparse_attn_backend.py | layers/attention/qwen_sparse_attn_backend.py | QSA attention backend |
| qsa/*.py | layers/attention/qsa/ | QSA indexer (glue/qsa_indexer/dsa_indexer), sparse attn, config |
| sglang_qwen3_5.py | models/qwen3_5.py | GDN/linear-attention cross-check vs ref/qwen3_5.py |

DeepSeek-V4 oracle code is bundled with its checkpoint:
`~/models/DeepSeek-V4-Flash-0731/inference/` (+ `encoding/`).
