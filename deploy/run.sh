#!/usr/bin/env bash
# Build and run the sllm-node container (GPU).
# Recipe-driven serving goes through the repo-root `sllm <recipe.yaml>`
# launcher; run.sh keeps the image build + the raw container modes
# (kernel smoke / standard-engine serve|batch|oneshot).
# Usage:
#   deploy/run.sh build
#   deploy/run.sh kernel                # GPU kernel smoke
#   deploy/run.sh serve [model_dir]     # HTTP serving of the standard engine
#   MODEL_DIR=~/models/<dir> SLLM_BATCH="p1;p2" deploy/run.sh batch
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
. "$REPO/env_source.sh"           # config.env values, real env vars win
sllm_load_env "$REPO/config.env"
IMG="${SLLM_IMAGE:-sllm-node:latest}"
NAME="sllm-node"
PORT="${SLLM_PORT:-8002}"
MODELS="${SLLM_MODELS_DIR:-$HOME/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS/Qwen2.5-Coder-0.5B}"

cd "$REPO"
if [ "${1:-}" = "build" ]; then
  docker build -t "$IMG" -f deploy/Dockerfile .
  exit 0
fi
MODE="${1:-kernel}"
shift || true

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host \
  -e SLLM_MODE="$MODE" \
  -e MODEL_DIR="$MODEL_DIR" \
  -e SLLM_PORT="$PORT" \
  -e SLLM_BATCH="${SLLM_BATCH:-}" \
  -e SLLM_PROMPT="${SLLM_PROMPT:-def fib(n):}" \
  -e SLLM_MAX_NEW="${SLLM_MAX_NEW:-8}" \
  -v "$MODELS:$MODELS:ro" \
  -p "$PORT:$PORT" \
  "$IMG" >/dev/null
echo "$NAME started (mode=$MODE, port=$PORT, model=$MODEL_DIR)."
docker logs -f "$NAME"
