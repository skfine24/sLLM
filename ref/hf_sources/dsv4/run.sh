#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${CKPT_PATH:?Set CKPT_PATH to the converted tensor-parallel checkpoint directory}"

MP="${MP:-4}"
CONFIG="${CONFIG:-config.json}"
INPUT_FILE="${INPUT_FILE:-examples/example_vl.txt}"

torchrun --nproc-per-node "${MP}" generate.py \
  --ckpt-path "${CKPT_PATH}" \
  --config "${CONFIG}" \
  --input-file "${INPUT_FILE}"
