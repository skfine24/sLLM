# sllm-node deploy: containerised operation of the sLLM serving program.
#
# Image / container name: sllm-node (overridable with SLLM_IMAGE / --image).
# Build:  ./build.sh            (or: docker build -f deploy/Dockerfile -t sllm-node .)
#
# Two stages:
#   build   : nvcc (CUDA 13.0) compiles the GB10 kernels -> sllm_gpu.so
#   runtime : python:3.12-slim + PINNED runtime deps (requirements.txt) +
#             DEPLOYMENT-ONLY source (dev/validation/audit resources are
#             excluded via .dockerignore + the explicit COPY list below).
# Streaming libs: the kernels link cudart + cuBLAS/cuBLASLt, shipped from the
# build stage (aarch64 runtime .so only).

# ---- stage 1: build the CUDA kernels (nvcc, CUDA 13.0) --------------------
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
WORKDIR /src
COPY kernels/cuda kernels/cuda
COPY env_source.sh config.env ./
# docker build has no GPU: "-arch=native" would fail, so the image defaults
# to the GB10 arch (sm_121, CUDA >= 12.9). Override:
#   docker build --build-arg NVCC_ARCH=sm_121 ...  (GB10 default)
ARG NVCC_ARCH=sm_121
ENV NVCC_ARCH=${NVCC_ARCH}
# build.sh is the single source of truth for sources/flags (kernels.cu +
# qwen4.cu + deepseek.cu, -lcublas -lcublasLt); do not duplicate the nvcc line.
RUN cd kernels/cuda && bash build.sh && ls -la sllm_gpu.so

# ---- stage 2: slim python runtime (deployment-only) ------------------------
FROM python:3.12-slim
WORKDIR /sllm
# The kernels link against cudart + cuBLAS/cuBLASLt; the slim image has none
# of them -- ship the runtime libs from the build stage.
COPY --from=build /usr/local/cuda/lib64/libcudart.so.13* /usr/lib/aarch64-linux-gnu/
COPY --from=build /usr/local/cuda/lib64/libcublas.so.13* /usr/lib/aarch64-linux-gnu/
COPY --from=build /usr/local/cuda/lib64/libcublasLt.so.13* /usr/lib/aarch64-linux-gnu/
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/lib

# pinned runtime deps (dev/validation packages intentionally absent)
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# --- deployment-only source tree (see .dockerignore for the exclusions) -----
COPY config.env env_source.sh env_config.py ./
COPY serving ./serving
COPY ref ./ref
COPY runtime ./runtime
COPY loaders ./loaders
COPY tp ./tp
COPY recipes ./recipes
COPY kernels ./kernels
COPY deploy/entrypoint.sh ./deploy/entrypoint.sh
COPY --from=build /src/kernels/cuda/sllm_gpu.so /sllm/kernels/cuda/sllm_gpu.so
# belt-and-suspenders: dev entrypoints / vendored audit refs never ship
# (redundant with .dockerignore, holds even if the context differs)
RUN rm -f /sllm/serving/cli.py /sllm/serving/dev_model.py \
 && rm -rf /sllm/ref/hf_sources

ENV PYTHONPATH=/sllm
WORKDIR /sllm
ENTRYPOINT ["/bin/bash", "/sllm/deploy/entrypoint.sh"]
