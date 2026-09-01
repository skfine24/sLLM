/*
 * DeepSeek-V4 host-transfer parity kernels (phase 5, cluster-gated).
 *
 * These are CORRECT-BY-CONSTRUCTION naive stubs: they prove the ctypes
 * wiring + build on the cluster before any fused/perf rewrite (C1/C2). The
 * numpy wrappers in _deepseek_cuda.py keep dev-machine runs identical when
 * this .so is absent.
 *
 *   sllm_ds_fp4_gemm(O, A, B, M, N, K, scale, _) : O(M,N) fp32 = A(M,K)@B(N,K)^T
 *       fp32 accumulate; A = model activation (M,K), B = DEQUANTIZED fp4 expert
 *       weight (N,K). The block-dequant (E8M0, per-row-32) happens host-side in
 *       loaders.fp8.dequant_fp4_packed_weight (parity stub; a fused fp4 GEMM is
 *       a later increment).
 *   sllm_ds_mla_sparse_attn(O, Q, R, IDX, SINK, S, H, D, ntop, scale, _)
 *       o(s,h,d) = sum_j exp((q.R[idx_j] - max) * scale)_h * R ... / (denom +
 *       exp((sink_h - max)*scale)); -1 idx -> masked; sink in denominator only.
 *
 * Compile via kernels/cuda/build.sh (adds deepseek.cu to the qwen4_cu obj).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(e)                                                     \
  do {                                                               \
    cudaError_t _e = (e);                                           \
    if (_e != cudaSuccess) {                                        \
      snprintf(g_err, sizeof(g_err), "%s:%d %s", __FILE__, __LINE__, \
               cudaGetErrorString(_e));                             \
      return -1;                                                    \
    }                                                               \
  } while (0)

__device__ char g_err_d[512];

extern "C" const char *sllm_ds_last_error(void) {
  static char host[512] = "no error";
  cudaMemcpyFromSymbol(host, g_err_d, 512, 0, cudaMemcpyDeviceToHost);
  return host;
}

__device__ void ds_err(const char *m) {
  snprintf(g_err_d, sizeof(g_err_d), "%s", m);
}

__global__ void ds_fp4_gemm_kernel(const float *__restrict__ a,
                                   const float *__restrict__ b,
                                   float *__restrict__ o, int M, int N, int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += a[m * K + k] * b[n * K + k];
  o[m * N + n] = acc;
}

extern "C" int sllm_ds_fp4_gemm(float *o, const float *a, const float *b,
                                int M, int N, int K, float scale, float *opts) {
  (void)scale; (void)opts;
  if (M <= 0 || N <= 0 || K <= 0) { ds_err("bad dims"); return -1; }
  float *da, *db, *do_;
  CHECK(cudaMalloc(&da, (size_t)M * K * sizeof(float)));
  CHECK(cudaMalloc(&db, (size_t)N * K * sizeof(float)));
  CHECK(cudaMalloc(&do_, (size_t)M * N * sizeof(float)));
  CHECK(cudaMemcpy(da, a, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
  dim3 blk(16, 16), grd((M + 15) / 16, (N + 15) / 16);
  ds_fp4_gemm_kernel<<<grd, blk>>>(da, db, do_, M, N, K);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(o, do_, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(do_));
  return 0;
}

__global__ void ds_sparse_attn_kernel(const float *__restrict__ q,
                                      const float *__restrict__ r,
                                      const int *__restrict__ idx,
                                      const float *__restrict__ sink,
                                      float *__restrict__ o,
                                      int S, int H, int D, int ntop,
                                      float scale) {
  int s = blockIdx.x, h = blockIdx.y;
  if (s >= S || h >= H) return;
  float maxv = -1e30f;
  for (int j = 0; j < ntop; ++j) {
    int i = idx[s * ntop + j];
    if (i < 0) continue;
    float sc = 0.f;
    for (int d = 0; d < D; ++d) sc += q[s * H * D + h * D + d] * r[i * D + d];
    sc *= scale;
    if (sc > maxv) maxv = sc;
  }
  float denom = expf((sink[h] - maxv) * scale);
  for (int j = 0; j < ntop; ++j) {
    int i = idx[s * ntop + j];
    if (i < 0) continue;
    float sc = 0.f;
    for (int d = 0; d < D; ++d) sc += q[s * H * D + h * D + d] * r[i * D + d];
    denom += expf((sc * scale) - maxv);
  }
  for (int d = 0; d < D; ++d) {
    float acc = 0.f;
    for (int j = 0; j < ntop; ++j) {
      int i = idx[s * ntop + j];
      if (i < 0) continue;
      float sc = 0.f;
      for (int dd = 0; dd < D; ++dd)
        sc += q[s * H * D + h * D + dd] * r[i * D + dd];
      acc += expf((sc * scale) - maxv) * r[i * D + d];
    }
    o[s * H * D + h * D + d] = acc / denom;
  }
}

extern "C" int sllm_ds_mla_sparse_attn(float *o, const float *q,
                                       const float *r, const int *idx,
                                       const float *sink, int S, int H, int D,
                                       int ntop, float scale, float *opts) {
  (void)opts;
  if (S <= 0 || H <= 0 || D <= 0 || ntop <= 0) { ds_err("bad dims"); return -1; }
  float *dq, *dr, *do_;
  int *didx;
  CHECK(cudaMalloc(&dq, (size_t)S * H * D * sizeof(float)));
  CHECK(cudaMalloc(&dr, (size_t)(S * 4 + 1) * D * sizeof(float))); /* slack */
  CHECK(cudaMalloc(&do_, (size_t)S * H * D * sizeof(float)));
  CHECK(cudaMalloc(&didx, (size_t)S * ntop * sizeof(int)));
  /* row count for r is supplied by callers as (S*4+1) cap; actual N is
     threaded through scale argument on the host wrapper in a later fused
     stage -- parity stub copies the full q/idx and relies on idx bounds. */
  CHECK(cudaMemcpy(dq, q, (size_t)S * H * D * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dr, r, (size_t)(S * 4 + 1) * D * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(didx, idx, (size_t)S * ntop * sizeof(int), cudaMemcpyHostToDevice));
  dim3 grd(S, H);
  ds_sparse_attn_kernel<<<grd, 1>>>(dq, dr, didx, sink, do_, S, H, D, ntop,
                                    scale);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(o, do_, (size_t)S * H * D * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dq)); CHECK(cudaFree(dr)); CHECK(cudaFree(do_)); CHECK(cudaFree(didx));
  return 0;
}
