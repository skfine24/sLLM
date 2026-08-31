# sllm-node deploy: containerised operation of the sLLM serving program.
#
# Image / container name: sllm-node (defaults listed below).
# CUDA 13.0 is the baseline for the kernel build stage.

# ---- stage 1: build the CUDA kernels (nvcc, CUDA 13.0) --------------------
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
WORKDIR /src
COPY kernels/cuda kernels/cuda
COPY env_source.sh config.env ./
# docker build has no GPU: "-arch=native" would fail, so the image defaults
# to the GB10 arch (sm_121, CUDA >= 12.9). Override:
#   docker build --build-arg NVCC_ARCH=native ...   (requires --network host
#   style GPU-less builds NOT possible; keep sm_121 unless cross-building)
ARG NVCC_ARCH=sm_121
ENV NVCC_ARCH=${NVCC_ARCH}
# build.sh is the single source of truth for sources/flags (kernels.cu +
# qwen4.cu, -lcublas -lcublasLt); do not duplicate the nvcc line here.
RUN cd kernels/cuda && bash build.sh && ls -la sllm_gpu.so

# ---- stage 2: slim python runtime ------------------------------------------
FROM python:3.12-slim
WORKDIR /sllm
# The kernels link against cudart + cuBLAS/cuBLASLt; the slim image has none
# of them -- ship the runtime libs from the build stage.
COPY --from=build /usr/local/cuda/lib64/libcudart.so.13* /usr/lib/aarch64-linux-gnu/
COPY --from=build /usr/local/cuda/lib64/libcublas.so.13* /usr/lib/aarch64-linux-gnu/
COPY --from=build /usr/local/cuda/lib64/libcublasLt.so.13* /usr/lib/aarch64-linux-gnu/
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/lib
COPY . /sllm
COPY --from=build /src/kernels/cuda/sllm_gpu.so /sllm/kernels/cuda/sllm_gpu.so
RUN python -m pip install --no-cache-dir numpy pyyaml regex jinja2
ENV PYTHONPATH=/sllm
WORKDIR /sllm
ENTRYPOINT ["/bin/bash", "/sllm/deploy/entrypoint.sh"]
