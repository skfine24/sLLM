#!/usr/bin/env bash
# Build the sllm-node CUDA kernels into kernels/cuda/sllm_gpu.so.
# Default toolchain: CUDA 13.0 (nvcc). Use -arch=native so the build targets
# the GB10 GPU at hand (SM12x). Override with NVCC_ARCH for cross arch.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../.. && pwd)"
if [ -f "$REPO/env_source.sh" ]; then
  . "$REPO/env_source.sh"
  sllm_load_env "$REPO/config.env"       # NVCC_ARCH etc.; real env vars win
elif [ -f "$REPO/config.env" ]; then     # minimal build stage without helper
  set -a; . "$REPO/config.env"; set +a
fi
NVCC="${NVCC:-nvcc}"
# NVCC_ARCH accepts either an arch value (native, sm_121) or a full flag
ARCH="${NVCC_ARCH:-native}"
[[ "$ARCH" == -* ]] || ARCH="-arch=$ARCH"
"$NVCC" -O2 "$ARCH" -shared -Xcompiler -fPIC -o sllm_gpu.so kernels.cu qwen4.cu deepseek.cu -lcublas -lcublasLt
echo "built: $(pwd)/sllm_gpu.so"
