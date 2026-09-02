// sLLM FP8 (E4M3, 128x128 block inverse scale) GEMM -- milestone C / Q4-GPU
// enabling kernel.
//
// Correct-by-construction naive parity stub first (mirrors the deepseek.cu
// approach: prove wiring/build before the fused tensor-core rewrite). The
// FULL model (173 GiB fp8) cannot fit a node as fp32/bf16, so the weights
// MUST stay fp8 on device and be dequantized inside the GEMM -- this kernel
// is the contract; the fused tensor-core (cublasLt fp8 / __nv_fp8_e4m3 mma)
// rewrite is the optimization on the same A/B/scale layout.
//
// dequant[i, j] = e4m3(B[i, j]) * scale_inv[i / block_h, j / block_w]
// C = A (M,K) @ dequant(B) (N,K)^T  -> (M,N) fp32, cuBLAS column-major free.
//
// The scale_inv is the BF16-inverse-scale block, pre-decoded to float32 on the
// host (loaders/fp8.decode_bf16_array) -- same contract as loaders/fp8.py
// dequant_weight_blocked.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static char g_fp8_err[512] = {0};
extern "C" const char* sllm_fp8_last_error(void) { return g_fp8_err; }

#define FP8_CHECK(e)                                                       \
  do {                                                                     \
    cudaError_t _e = (e);                                                  \
    if (_e != cudaSuccess) {                                               \
      snprintf(g_fp8_err, sizeof(g_fp8_err), "%s: %s", #e,                \
               cudaGetErrorString(_e));                                    \
      return -1;                                                           \
    }                                                                      \
  } while (0)

// E4M3FN -> float32 (bias 7, 3 mantissa bits, no subnormals).
__device__ __forceinline__ float fp8_e4m3_to_f32(unsigned char v) {
  int s = (v >> 7) & 1;
  int e = (v >> 3) & 0x0F;
  int m = v & 0x07;
  float f;
  if (e == 0x0F) {
    f = (m == 0) ? (s ? -INFINITY : INFINITY) : NAN;
  } else {
    f = (1.0f + (float)m / 8.0f) * exp2f((float)e - 7.0f);
  }
  return s ? -f : f;
}

// one thread per output element; B dequant applied on the fly via the block
// scale. O(M*N*K) -- correctness/parity first.
__global__ void fp8_gemm_kernel(const float* __restrict__ A,
                                const unsigned char* __restrict__ B,
                                const float* __restrict__ scale_inv,
                                float* __restrict__ C, int M, int N, int K,
                                int block_h, int block_w) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)M * N;
  if (idx >= total) return;
  int m = (int)(idx / N), n = (int)(idx % N);
  const float* a = A + (size_t)m * K;
  const unsigned char* b = B + (size_t)n * K;
  int bh = n / block_w;             // block col for this output row
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    int bhh = k / block_h;
    float s = scale_inv[(size_t)bhh * ((N + block_w - 1) / block_w) + bh];
    acc += a[k] * fp8_e4m3_to_f32(b[k]) * s;
  }
  C[idx] = acc;
}

extern "C" int sllm_fp8_gemm(const float* A, const unsigned char* B,
                             const float* scale_inv, float* C, int M, int N,
                             int K, int block_h, int block_w) {
  if (M <= 0 || N <= 0 || K <= 0 || block_h <= 0 || block_w <= 0) return -1;
  float *dA, *dC;
  unsigned char* dB;
  float* dS;
  size_t sa = (size_t)M * K * sizeof(float);
  size_t sb = (size_t)N * K;
  size_t sc = (size_t)M * N * sizeof(float);
  size_t ss = (size_t)((K + block_h - 1) / block_h) *
              ((N + block_w - 1) / block_w) * sizeof(float);
  FP8_CHECK(cudaMalloc((void**)&dA, sa));
  FP8_CHECK(cudaMalloc((void**)&dB, sb));
  FP8_CHECK(cudaMalloc((void**)&dS, ss));
  FP8_CHECK(cudaMalloc((void**)&dC, sc));
  FP8_CHECK(cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice));
  FP8_CHECK(cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice));
  FP8_CHECK(cudaMemcpy(dS, scale_inv, ss, cudaMemcpyHostToDevice));
  size_t total = (size_t)M * N;
  int threads = 256;
  fp8_gemm_kernel<<<(unsigned)((total + threads - 1) / threads), threads>>>(
      dA, dB, dS, dC, M, N, K, block_h, block_w);
  FP8_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dS); cudaFree(dC);
  return 0;
}

// device-resident variant: A/B/scale/C are DEVICE pointers (no host copies).
extern "C" int sllm_fp8_gemm_dev(const float* A, const unsigned char* B,
                                 const float* scale_inv, float* C, int M,
                                 int N, int K, int block_h, int block_w) {
  if (M <= 0 || N <= 0 || K <= 0 || block_h <= 0 || block_w <= 0) return -1;
  size_t total = (size_t)M * N;
  int threads = 256;
  fp8_gemm_kernel<<<(unsigned)((total + threads - 1) / threads), threads>>>(
      A, B, scale_inv, C, M, N, K, block_h, block_w);
  return 0;
}
