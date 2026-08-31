// sllm-node GPU kernels (first set).
//
// Self-contained CUDA kernels (no cuBLAS dependency in this milestone):
//   - rms_norm        : Llama/Qwen2-style RMSNorm (out = x * rsqrt(mean(x^2)+eps) * w)
//   - elwise_add      : y = a + b
//   - sllm_device_count : probe how many CUDA devices are visible
//
// Host API is C-ABI for direct ctypes loading from Python
// (see kernels/_sllm_cuda.py).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <string.h>

#define CHECK(e)                                                      \
  do {                                                                \
    cudaError_t _e = (e);                                             \
    if (_e != cudaSuccess) {                                          \
      snprintf(g_last_error, sizeof(g_last_error), "%s: %s", #e,      \
               cudaGetErrorString(_e));                               \
      return -1;                                                      \
    }                                                                 \
  } while (0)

static char g_last_error[512] = {0};

extern "C" const char* sllm_last_error(void) { return g_last_error; }

extern "C" int sllm_device_count(void) {
  int n = -1;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    snprintf(g_last_error, sizeof(g_last_error), "cudaGetDeviceCount: %s",
             cudaGetErrorString(e));
    return -1;
  }
  return n;
}

__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ w,
                                float* __restrict__ y, int rows, int dim,
                                float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + (size_t)row * dim;
  float* yr = y + (size_t)row * dim;

  __shared__ float sh[256];
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  float acc = 0.0f;
  for (int i = tid; i < dim; i += nthreads) {
    float v = xr[i];
    acc += v * v;
  }
  sh[tid] = acc;
  __syncthreads();
  for (int s = nthreads / 2; s > 0; s >>= 1) {
    if (tid < s) sh[tid] += sh[tid + s];
    __syncthreads();
  }
  __syncthreads();

  float inv = rsqrtf(sh[0] / (float)dim + eps);
  for (int i = tid; i < dim; i += nthreads) {
    yr[i] = xr[i] * inv * w[i];
  }
}

extern "C" int sllm_rms_norm(const float* x, const float* w, float* y,
                             int rows, int dim, float eps) {
  if (rows <= 0 || dim <= 0) return -1;
  float *dx = NULL, *dw = NULL, *dy = NULL;
  CHECK(cudaMalloc((void**)&dx, (size_t)rows * dim * sizeof(float)));
  CHECK(cudaMalloc((void**)&dw, (size_t)dim * sizeof(float)));
  CHECK(cudaMalloc((void**)&dy, (size_t)rows * dim * sizeof(float)));
  CHECK(cudaMemcpy(dx, x, (size_t)rows * dim * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dw, w, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
  rms_norm_kernel<<<rows, 256>>>(dx, dw, dy, rows, dim, eps);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(y, dy, (size_t)rows * dim * sizeof(float),
                   cudaMemcpyDeviceToHost));
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dy);
  return 0;
}

__global__ void elwise_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

extern "C" int sllm_elwise_add(const float* a, const float* b, float* y,
                               long n) {
  if (n <= 0) return 0;
  float *da = NULL, *db = NULL, *dy = NULL;
  size_t nb = (size_t)n * sizeof(float);
  CHECK(cudaMalloc((void**)&da, nb));
  CHECK(cudaMalloc((void**)&db, nb));
  CHECK(cudaMalloc((void**)&dy, nb));
  CHECK(cudaMemcpy(da, a, nb, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b, nb, cudaMemcpyHostToDevice));
  long blocks = (n + 255) / 256;
  elwise_add_kernel<<<blocks, 256>>>(da, db, dy, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(y, dy, nb, cudaMemcpyDeviceToHost));
  cudaFree(da);
  cudaFree(db);
  cudaFree(dy);
  return 0;
}

__global__ void silu_kernel(const float* __restrict__ x,
                            float* __restrict__ y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * (1.0f / (1.0f + __expf(-x[i])));
}

extern "C" int sllm_silu(const float* x, float* y, long n) {
  if (n <= 0) return 0;
  float *dx = NULL, *dy = NULL;
  size_t nb = (size_t)n * sizeof(float);
  CHECK(cudaMalloc((void**)&dx, nb));
  CHECK(cudaMalloc((void**)&dy, nb));
  CHECK(cudaMemcpy(dx, x, nb, cudaMemcpyHostToDevice));
  long blocks = (n + 255) / 256;
  silu_kernel<<<blocks, 256>>>(dx, dy, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(y, dy, nb, cudaMemcpyDeviceToHost));
  cudaFree(dx);
  cudaFree(dy);
  return 0;
}

// ---------------------------------------------------------------------------
// Device tensor helpers (opaque GPU buffers for KV placement)
// ---------------------------------------------------------------------------

extern "C" void* sllm_buf_new(long nbytes) {
  void* p = NULL;
  if (nbytes <= 0) return NULL;
  cudaError_t e = cudaMalloc(&p, (size_t)nbytes);
  if (e != cudaSuccess) {
    snprintf(g_last_error, sizeof(g_last_error), "cudaMalloc(%ld): %s",
             nbytes, cudaGetErrorString(e));
    return NULL;
  }
  return p;
}

// Free device memory in bytes (cudaMemGetInfo); -1 on error.
extern "C" long sllm_mem_free_bytes(void) {
  size_t free_b = 0, total_b = 0;
  cudaError_t e = cudaMemGetInfo(&free_b, &total_b);
  if (e != cudaSuccess) {
    snprintf(g_last_error, sizeof(g_last_error), "cudaMemGetInfo: %s",
             cudaGetErrorString(e));
    return -1;
  }
  return (long)free_b;
}

extern "C" int sllm_buf_free(void* p) {
  if (!p) return 0;
  CHECK(cudaFree(p));
  return 0;
}

extern "C" int sllm_buf_h2d(void* dst, const float* src, long nbytes) {
  CHECK(cudaMemcpy(dst, src, (size_t)nbytes, cudaMemcpyHostToDevice));
  return 0;
}

// Raw-byte variant (bf16 tables etc.); dst is a device pointer.
extern "C" int sllm_buf_h2d_raw(void* dst, const void* src, long nbytes) {
  CHECK(cudaMemcpy(dst, src, (size_t)nbytes, cudaMemcpyHostToDevice));
  return 0;
}

extern "C" int sllm_buf_d2h(float* dst, const void* src, long nbytes) {
  CHECK(cudaMemcpy(dst, src, (size_t)nbytes, cudaMemcpyDeviceToHost));
  return 0;
}

// ---------------------------------------------------------------------------
// Dense GEMM via cuBLAS: c(m,n) = a(m,k) @ b(k,n).
// Buffers are COLUMN-MAJOR (cuBLAS natural): a lda=m, b ldb=k, c ldc=m.
// The ctypes layer (sllm_cuda.gemm) feeds Fortran-order arrays.
// ---------------------------------------------------------------------------

// Shared lazily-created cuBLAS handle (used by sllm_gemm and the device-side
// gemm entries below; no per-call create/destroy).
static cublasHandle_t g_handle = NULL;
static cublasHandle_t* get_handle(void) {
  if (!g_handle) {
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
      snprintf(g_last_error, sizeof(g_last_error), "cublasCreate failed");
      return NULL;
    }
  }
  return &g_handle;
}

extern "C" int sllm_gemm(const float* a, const float* b, float* c,
                         int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0) return -1;
  float *da = NULL, *db = NULL, *dc = NULL;
  size_t na = (size_t)m * k * sizeof(float);
  size_t nb = (size_t)k * n * sizeof(float);
  size_t nc = (size_t)m * n * sizeof(float);
  CHECK(cudaMalloc((void**)&da, na));
  CHECK(cudaMalloc((void**)&db, nb));
  CHECK(cudaMalloc((void**)&dc, nc));
  CHECK(cudaMemcpy(da, a, na, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b, nb, cudaMemcpyHostToDevice));

  cublasHandle_t* h = get_handle();
  if (!h) {
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return -1;
  }
  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasSgemm(*h, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k, &alpha, da, m, db, k, &beta, dc, m);
  if (st != CUBLAS_STATUS_SUCCESS) {
    snprintf(g_last_error, sizeof(g_last_error), "cublasSgemm: %d", (int)st);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return -1;
  }
  CHECK(cudaMemcpy(c, dc, nc, cudaMemcpyDeviceToHost));
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}

// ---------------------------------------------------------------------------
// Single-row (decode) attention over cached KV: out[h,:] =
// softmax_s( q[h] . K[h,s,:] * scale ) . V[h,s,:]
// One block per head; 256 threads; global `scores[heads*S]` scratch for the
// per-position scores (softmax max/sum reduced across threads).
// ---------------------------------------------------------------------------

__global__ void attention_decode_kernel(const float* __restrict__ q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ scores,
                                        float* __restrict__ out,
                                        int S, int D, float scale) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float red[256];
  const float* qh = q + (size_t)h * D;
  const float* kh = K + (size_t)h * (size_t)S * D;
  const float* vh = V + (size_t)h * (size_t)S * D;
  float* sh = scores + (size_t)h * S;

  // Phase 1: per-position scores (each thread handles its s-strided subset).
  for (int s = tid; s < S; s += blockDim.x) {
    float acc = 0.0f;
    const float* krow = kh + (size_t)s * D;
    for (int d = 0; d < D; ++d) acc += qh[d] * krow[d];
    sh[s] = acc * scale;
  }
  __syncthreads();

  // Phase 2: max over s.
  float m = -1e30f;
  for (int s = tid; s < S; s += blockDim.x) m = fmaxf(m, sh[s]);
  red[tid] = m;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  float mval = red[0];

  // Phase 3: sum(exp(scores - max)).
  float psum = 0.0f;
  for (int s = tid; s < S; s += blockDim.x) psum += __expf(sh[s] - mval);
  red[tid] = psum;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float inv = 1.0f / red[0];

  // Phase 4: weighted output per dimension d.
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
      acc += __expf(sh[s] - mval) * vh[(size_t)s * D + d];
    }
    out[(size_t)h * D + d] = acc * inv;
  }
}

extern "C" int sllm_attention_decode(const float* q, const float* K,
                                     const float* V, float* out,
                                     int heads, int S, int D, float scale) {
  if (heads <= 0 || S <= 0 || D <= 0) return -1;
  size_t qb = (size_t)heads * D;
  size_t kb = (size_t)heads * (size_t)S * D;
  size_t sb = (size_t)heads * (size_t)S;
  float *dq = NULL, *dK = NULL, *dV = NULL, *dscores = NULL, *dout = NULL;
  CHECK(cudaMalloc((void**)&dq, qb * sizeof(float)));
  CHECK(cudaMalloc((void**)&dK, kb * sizeof(float)));
  CHECK(cudaMalloc((void**)&dV, kb * sizeof(float)));
  CHECK(cudaMalloc((void**)&dscores, sb * sizeof(float)));
  CHECK(cudaMalloc((void**)&dout, qb * sizeof(float)));
  CHECK(cudaMemcpy(dq, q, qb * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dK, K, kb * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dV, V, kb * sizeof(float), cudaMemcpyHostToDevice));
  attention_decode_kernel<<<heads, 256>>>(dq, dK, dV, dscores, dout, S, D, scale);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out, dout, qb * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dq);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dscores);
  cudaFree(dout);
  return 0;
}

// ---------------------------------------------------------------------------
// GatedDeltaNet single-step (decode) recurrence:
//   state[h] *= exp(g[h]);  kv_mem[h] = state[h] . k[h]
//   delta[h] = (v[h] - kv_mem[h]) * beta[h];  state[h] += k[h] (x) delta[h]
//   out[h] = state[h] . q[h]
// Grid = value heads (Vh), block = Vd threads. state is updated IN PLACE
// (decay -> barrier -> delta/update) and mirrored to `out`.
// q/k are expected pre-normalized/scaled by the caller (L2 + kd^-0.5),
// replicated to one row per value head — matching ref/qwen3_5 recurrent path.
// ---------------------------------------------------------------------------

__global__ void gated_delta_step_kernel(const float* __restrict__ q,
                                        const float* __restrict__ k,
                                        const float* __restrict__ v,
                                        const float* __restrict__ g,
                                        const float* __restrict__ beta,
                                        float* __restrict__ state,
                                        float* __restrict__ out,
                                        int Kd, int Vd) {
  int h = blockIdx.x;
  int e = threadIdx.x;
  if (e >= Vd) return;
  const float* qh = q + (size_t)h * Kd;
  const float* kh = k + (size_t)h * Kd;
  float* sh = state + (size_t)h * (size_t)Kd * Vd;
  float* oh = out + (size_t)h * Vd;

  float dec = __expf(g[h]);
  float kv_mem = 0.0f;
  for (int d = 0; d < Kd; ++d) {
    float sde = sh[d * Vd + e] * dec;
    sh[d * Vd + e] = sde;
    kv_mem += sde * kh[d];
  }
  __syncthreads();
  float delta = (v[(size_t)h * Vd + e] - kv_mem) * beta[h];
  for (int d = 0; d < Kd; ++d) sh[d * Vd + e] += kh[d] * delta;

  float acc = 0.0f;
  for (int d = 0; d < Kd; ++d) acc += sh[d * Vd + e] * qh[d];
  oh[e] = acc;
}

extern "C" int sllm_gated_delta_step(const float* q, const float* k,
                                     const float* v, const float* g,
                                     const float* beta, float* state,
                                     float* out, int Vh, int Kd, int Vd) {
  if (Vh <= 0 || Kd <= 0 || Vd <= 0) return -1;
  size_t hq = (size_t)Vh * Kd;
  size_t hb = (size_t)Vh * Vd;
  size_t sb = (size_t)Vh * (size_t)Kd * Vd;
  float *dq = NULL, *dk = NULL, *dv = NULL, *dg = NULL, *dbeta = NULL;
  float *dst = NULL, *dout = NULL;
  CHECK(cudaMalloc((void**)&dq, hq * sizeof(float)));
  CHECK(cudaMalloc((void**)&dk, hq * sizeof(float)));
  CHECK(cudaMalloc((void**)&dv, hb * sizeof(float)));
  CHECK(cudaMalloc((void**)&dg, (size_t)Vh * sizeof(float)));
  CHECK(cudaMalloc((void**)&dbeta, (size_t)Vh * sizeof(float)));
  CHECK(cudaMalloc((void**)&dst, sb * sizeof(float)));
  CHECK(cudaMalloc((void**)&dout, hb * sizeof(float)));
  CHECK(cudaMemcpy(dq, q, hq * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dk, k, hq * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dv, v, hb * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dg, g, (size_t)Vh * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dbeta, beta, (size_t)Vh * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dst, state, sb * sizeof(float), cudaMemcpyHostToDevice));
  gated_delta_step_kernel<<<Vh, 256>>>(dq, dk, dv, dg, dbeta, dst, dout, Kd, Vd);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(state, dst, sb * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(out, dout, hb * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg); cudaFree(dbeta);
  cudaFree(dst); cudaFree(dout);
  return 0;
}

// ---------------------------------------------------------------------------
// Device-resident API (device-resident weights + persistent on-device KV).
//
// All buffers are DEVICE pointers owned by the caller (sllm_buf_new); the ops
// perform NO host transfers, NO cudaMalloc and NO cudaDeviceSynchronize. A
// decode step enqueues its whole graph on the default stream and synchronises
// ONCE at the end (sllm_sync + one logits D2H). GEMM buffers are ROW-MAJOR
// (the opposite convention from sllm_gemm's column-major host API): cuBLAS is
// called with the standard row-major-as-column-major transposition trick.
// ---------------------------------------------------------------------------

extern "C" int sllm_sync(void) {
  CHECK(cudaDeviceSynchronize());
  return 0;
}

// c(m,n) row-major = a(m,k) @ b(k,n), all device pointers.
extern "C" int sllm_gemm_dev(const float* a, const float* b, float* c,
                             int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0) return -1;
  cublasHandle_t* h = get_handle();
  if (!h) return -1;
  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasSgemm(*h, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k, &alpha, b, n, a, k, &beta, c, n);
  if (st != CUBLAS_STATUS_SUCCESS) {
    snprintf(g_last_error, sizeof(g_last_error), "cublasSgemm(gemm_dev): %d",
             (int)st);
    return -1;
  }
  return 0;
}

// c(m,n) row-major = h(m,k) @ w(n,k)^T  (linear-layer projection; the weight
// stays on device in its native (out,in) row-major layout, never transposed).
extern "C" int sllm_gemm_linear_dev(const float* hst, const float* w,
                                    float* c, int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0) return -1;
  cublasHandle_t* h = get_handle();
  if (!h) return -1;
  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasSgemm(*h, CUBLAS_OP_T, CUBLAS_OP_N,
                                  n, m, k, &alpha, w, k, hst, k, &beta, c, n);
  if (st != CUBLAS_STATUS_SUCCESS) {
    snprintf(g_last_error, sizeof(g_last_error), "cublasSgemm(linear_dev): %d",
             (int)st);
    return -1;
  }
  return 0;
}

__global__ void bias_add_kernel(float* __restrict__ y,
                                const float* __restrict__ bias,
                                int rows, int cols) {
  int r = blockIdx.x;
  if (r >= rows) return;
  float* yr = y + (size_t)r * cols;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] += bias[c];
}

// In-place y(r,cols) += bias(cols), device buffers.
extern "C" int sllm_bias_add_dev(float* y, const float* bias, int rows,
                                 int cols) {
  if (rows <= 0 || cols <= 0) return -1;
  bias_add_kernel<<<rows, 256>>>(y, bias, rows, cols);
  CHECK(cudaGetLastError());
  return 0;
}

__global__ void elwise_mul_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i];
}

// Device y = a * b (y may alias a or b).
extern "C" int sllm_elwise_mul_dev(const float* a, const float* b, float* y,
                                   long n) {
  if (n <= 0) return 0;
  long blocks = (n + 255) / 256;
  elwise_mul_kernel<<<blocks, 256>>>(a, b, y, n);
  CHECK(cudaGetLastError());
  return 0;
}

// Device y = a + b (y may alias a or b).
extern "C" int sllm_elwise_add_dev(const float* a, const float* b, float* y,
                                   long n) {
  if (n <= 0) return 0;
  long blocks = (n + 255) / 256;
  elwise_add_kernel<<<blocks, 256>>>(a, b, y, n);
  CHECK(cudaGetLastError());
  return 0;
}

// Device SiLU (y may alias x).
extern "C" int sllm_silu_dev(const float* x, float* y, long n) {
  if (n <= 0) return 0;
  long blocks = (n + 255) / 256;
  silu_kernel<<<blocks, 256>>>(x, y, n);
  CHECK(cudaGetLastError());
  return 0;
}

// Device RMSNorm (same kernel as the host-transfer entry, no transfers).
extern "C" int sllm_rms_norm_dev(const float* x, const float* w, float* y,
                                 int rows, int dim, float eps) {
  if (rows <= 0 || dim <= 0) return -1;
  rms_norm_kernel<<<rows, 256>>>(x, w, y, rows, dim, eps);
  CHECK(cudaGetLastError());
  return 0;
}

__global__ void gather_row_kernel(const float* __restrict__ table, long row,
                                  float* __restrict__ out, int dim) {
  const float* src = table + (size_t)row * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) out[d] = src[d];
}

// out(dim) = table(row, dim) — embedding lookup on device.
extern "C" int sllm_gather_row(const float* table, long row, float* out,
                               int dim) {
  if (row < 0 || dim <= 0) return -1;
  gather_row_kernel<<<1, 256>>>(table, row, out, dim);
  CHECK(cudaGetLastError());
  return 0;
}

__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k,
                            const float* __restrict__ cs, int qrows,
                            int krows, int dim, int rot) {
  // One thread per rotated PAIR element; in-place, pair handled by one thread
  // (no cross-thread partner reads). cs = cos(rot) ++ sin(rot).
  long total = ((long)(qrows + krows)) * (rot / 2);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int half = rot / 2;
  long row = i / half;
  int j = (int)(i % half);
  float* p;
  if (row < qrows) p = q + (size_t)row * dim;
  else p = k + (size_t)(row - qrows) * dim;
  float x1 = p[j], x2 = p[j + half];
  float c1 = cs[j], c2 = cs[half + j];   // cos[j], cos[j+half] (doubled table)
  float s1 = cs[rot + j], s2 = cs[rot + half + j];
  p[j] = x1 * c1 - x2 * s1;
  p[j + half] = x2 * c2 + x1 * s2;
}

// In-place RoPE on q(qrows,dim) + k(krows,dim); cs = device (cos||sin) row of
// length 2*rot for one position; dims >= rot keep pass-through (partial rope).
extern "C" int sllm_rope_dev(float* q, float* k, const float* cs, int qrows,
                             int krows, int dim, int rot) {
  if (qrows < 0 || krows < 0 || dim <= 0) return -1;
  if (rot <= 0 || rot % 2 || rot > dim) return -1;
  long total = (long)(qrows + krows) * (rot / 2);
  if (!total) return 0;
  long blocks = (total + 255) / 256;
  rope_kernel<<<blocks, 256>>>(q, k, cs, qrows, krows, dim, rot);
  CHECK(cudaGetLastError());
  return 0;
}

__global__ void kv_write_kernel(float* __restrict__ kv, int cap, int pos,
                                const float* __restrict__ row, int heads,
                                int dim, float* __restrict__ stage,
                                int stage_off) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= heads * dim) return;
  int h = t / dim, d = t % dim;
  float val = row[t];
  kv[((size_t)h * cap + pos) * dim + d] = val;
  if (stage) stage[stage_off + t] = val;
}

// Append one row (heads,dim) into kv(heads,cap,dim) at index pos; optionally
// mirror it to stage[stage_off .. +heads*dim) for the per-step host batch copy.
extern "C" int sllm_kv_write(float* kv, int cap, int pos, const float* row,
                             int heads, int dim, float* stage, int stage_off) {
  if (cap <= 0 || pos < 0 || pos >= cap || heads <= 0 || dim <= 0) return -1;
  long n = (long)heads * dim;
  long blocks = (n + 255) / 256;
  kv_write_kernel<<<blocks, 256>>>(kv, cap, pos, row, heads, dim, stage,
                                   stage_off);
  CHECK(cudaGetLastError());
  return 0;
}

__global__ void kv_relayout_kernel(float* __restrict__ dst,
                                   const float* __restrict__ src,
                                   int heads, int cap_old, int cap_new,
                                   int dim) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)heads * cap_old * dim) return;
  int d = (int)(i % dim);
  int s = (int)((i / dim) % cap_old);
  int h = (int)(i / ((long)cap_old * dim));
  dst[((size_t)h * cap_new + s) * dim + d] = src[i];
}

// Copy kv(heads,cap_old,dim) into a fresh (heads,cap_new,dim) buffer
// (capacity growth; positions are preserved, new rows are untouched).
extern "C" int sllm_kv_relayout(float* dst, const float* src, int heads,
                                int cap_old, int cap_new, int dim) {
  if (heads <= 0 || dim <= 0 || cap_new < cap_old) return -1;
  long n = (long)heads * cap_old * dim;
  if (!n) return 0;
  long blocks = (n + 255) / 256;
  kv_relayout_kernel<<<blocks, 256>>>(dst, src, heads, cap_old, cap_new, dim);
  CHECK(cudaGetLastError());
  return 0;
}

// dtype-agnostic capacity growth: copies kv(heads,cap_old,block) into a
// larger (heads,cap_new,block) buffer, where one (h,s) row is `blk_words`
// 32-bit words (dim*elem/4). Preserves positions; new rows untouched.
__global__ void kv_relayout_w_kernel(unsigned int* __restrict__ dst,
                                     const unsigned int* __restrict__ src,
                                     int heads, int cap_old, int cap_new,
                                     int blk_words) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)heads * cap_old * blk_words;
  if (i >= total) return;
  int w = (int)(i % blk_words);
  int s = (int)((i / blk_words) % cap_old);
  int h = (int)(i / ((long)cap_old * blk_words));
  dst[((size_t)h * cap_new + s) * blk_words + w] = src[i];
}

extern "C" int sllm_kv_relayout_w(void* dst, const void* src, int heads,
                                  int cap_old, int cap_new, int blk_words) {
  if (heads <= 0 || cap_new < cap_old || blk_words <= 0) return -1;
  long total = (long)heads * cap_old * blk_words;
  if (!total) return 0;
  long blocks = (total + 255) / 256;
  kv_relayout_w_kernel<<<blocks, 256>>>((unsigned int*)dst,
                                        (const unsigned int*)src, heads,
                                        cap_old, cap_new, blk_words);
  CHECK(cudaGetLastError());
  return 0;
}

// GQA last-row decode attention on device buffers; head h uses KV head
// h/n_rep; scores = caller scratch (heads*S). out(heads,D) row-major.
// K/V are (heads, stride, D) capacity-allocated; only the first S rows are
// read (stride >= S so the buffer can grow without re-copying attention).
__global__ void attention_decode_gqa_kernel(const float* __restrict__ q,
                                            const float* __restrict__ K,
                                            const float* __restrict__ V,
                                            float* __restrict__ scores,
                                            float* __restrict__ out,
                                            int S, int stride, int D,
                                            int n_rep, float scale) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  int kh = h / n_rep;
  __shared__ float red[256];
  const float* qh = q + (size_t)h * D;
  const float* kh0 = K + (size_t)kh * (size_t)stride * D;
  const float* vh0 = V + (size_t)kh * (size_t)stride * D;
  float* sh = scores + (size_t)h * S;

  for (int s = tid; s < S; s += blockDim.x) {
    float acc = 0.0f;
    const float* krow = kh0 + (size_t)s * D;
    for (int d = 0; d < D; ++d) acc += qh[d] * krow[d];
    sh[s] = acc * scale;
  }
  __syncthreads();
  float m = -1e30f;
  for (int s = tid; s < S; s += blockDim.x) m = fmaxf(m, sh[s]);
  red[tid] = m;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  float mval = red[0];
  float psum = 0.0f;
  for (int s = tid; s < S; s += blockDim.x) psum += __expf(sh[s] - mval);
  red[tid] = psum;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float inv = 1.0f / red[0];
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < S; ++s)
      acc += __expf(sh[s] - mval) * vh0[(size_t)s * D + d];
    out[(size_t)h * D + d] = acc * inv;
  }
}

extern "C" int sllm_attention_decode_dev(const float* q, const float* K,
                                         const float* V, float* scores,
                                         float* out, int heads, int kv_heads,
                                         int S, int stride, int D,
                                         float scale) {
  if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || S <= 0 || D <= 0 ||
      stride < S)
    return -1;
  attention_decode_gqa_kernel<<<heads, 256>>>(q, K, V, scores, out, S, stride,
                                              D, heads / kv_heads, scale);
  CHECK(cudaGetLastError());
  return 0;
}

// ===========================================================================
// GENERAL-PURPOSE FUSED + TYPED API (architecture/dtype agnostic serving)
//
// Design (mirrors mainstream engines): the residual stream and logits stay
// fp32; only GEMM operands, the KV cache and the embedding table use the
// model dtype (fp32 | bf16). Every fused kernel takes runtime dtype tags:
//   SLLM_T_F32 = 0, SLLM_T_BF16 = 1.
// Fusion set for a decoder layer (10 device ops instead of ~20):
//   add_rms  -> gemm(qkv) -> rope_bias -> kv_write x2 -> attention
//   -> gemm(o) -> add_rms -> gemm(gate|up) -> silu_mul -> gemm(down)
// ===========================================================================

#define SLLM_T_F32 0
#define SLLM_T_BF16 1

template <int T>
__device__ __forceinline__ float t_load(const void *p, long i) {
  return T == SLLM_T_F32 ? ((const float *)p)[i]
                         : __bfloat162float(((const __nv_bfloat16 *)p)[i]);
}

template <int T>
__device__ __forceinline__ void t_store(void *p, long i, float v) {
  if (T == SLLM_T_F32) ((float *)p)[i] = v;
  else ((__nv_bfloat16 *)p)[i] = __float2bfloat16_rn(v);
}

// c(m,n) fp32 = a(m,k)@w(n,k)^T with a,w in dtype T, fp32 accumulate.
extern "C" int sllm_gemm_ex(const void *a, int atype, const void *w, void *c,
                            int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0) return -1;
  cublasHandle_t *h = get_handle();
  if (!h) return -1;
  float alpha = 1.0f, beta = 0.0f;
  cudaDataType_t dt = (atype == SLLM_T_BF16) ? CUDA_R_16BF : CUDA_R_32F;
  cublasStatus_t st = cublasGemmEx(
      *h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, dt, k, a, dt, k,
      &beta, c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) {
    snprintf(g_last_error, sizeof(g_last_error), "cublasGemmEx: %d", (int)st);
    return -1;
  }
  return 0;
}

// out(row,dim) = table(row,dim) cast to fp32.
template <int T>
__global__ void gather_row_t_kernel(const void * __restrict__ table, long row,
                                    float * __restrict__ out, int dim) {
  const long base = row * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    out[d] = t_load<T>(table, base + d);
}

extern "C" int sllm_gather_row_t(const void *table, long row, float *out,
                                 int dim, int t) {
  if (row < 0 || dim <= 0) return -1;
  if (t == SLLM_T_BF16)
    gather_row_t_kernel<SLLM_T_BF16><<<1, 256>>>(table, row, out, dim);
  else
    gather_row_t_kernel<SLLM_T_F32><<<1, 256>>>(table, row, out, dim);
  CHECK(cudaGetLastError());
  return 0;
}

// y = rmsnorm(a + (b?0)) * w in ONE kernel; also writes the sum (residual).
// b may be NULL (pure norm of a; the sum store then re-writes `a` into sum).
template <int T>
__global__ void add_rms_kernel(const float *__restrict__ a,
                               const float *__restrict__ b,
                               float *__restrict__ sum_out,
                               void *__restrict__ norm_out,
                               const float *__restrict__ w, int rows, int dim,
                               float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const float *ar = a + (size_t)row * dim;
  const float *br = b ? b + (size_t)row * dim : NULL;
  float *sr = sum_out + (size_t)row * dim;
  __shared__ float sh[256];
  int tid = threadIdx.x, nt = blockDim.x;

  float acc = 0.0f;
  for (int i = tid; i < dim; i += nt) {
    float v = ar[i] + (br ? br[i] : 0.0f);
    sr[i] = v;
    acc += v * v;
  }
  sh[tid] = acc;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) sh[tid] += sh[tid + s];
    __syncthreads();
  }
  // exact 1/sqrt (not rsqrtf): decode-scale kernels are memory bound and the
  // serving engine keeps a tight fp32 error floor against the numpy oracle.
  float inv = 1.0f / sqrtf(sh[0] / (float)dim + eps);
  const int esz = (T == SLLM_T_F32) ? 4 : 2;
  void *nr = (char *)norm_out + (size_t)row * dim * esz;
  for (int i = tid; i < dim; i += nt)
    t_store<T>(nr, i, sr[i] * inv * w[i]);
}

extern "C" int sllm_add_rms(const float *a, const float *b, float *sum_out,
                            void *norm_out, const float *w, int rows, int dim,
                            float eps, int norm_type) {
  if (rows <= 0 || dim <= 0) return -1;
  if (norm_type == SLLM_T_BF16)
    add_rms_kernel<SLLM_T_BF16><<<rows, 256>>>(a, b, sum_out, norm_out, w,
                                               rows, dim, eps);
  else
    add_rms_kernel<SLLM_T_F32><<<rows, 256>>>(a, b, sum_out, norm_out, w, rows,
                                              dim, eps);
  CHECK(cudaGetLastError());
  return 0;
}

// RoPE with fused q/k bias add. Bias applies to ALL `dim` columns (matching
// HF Linear semantics); rotation applies in place to the first `rot` dims as
// pairs (j, j+rot/2). qb/kb may be NULL. Threads split into a pair-rotation
// range and a pass-through bias-only tail [rot, dim) (disjoint indices, so
// no inter-thread races).
__global__ void rope_bias_kernel(float * __restrict__ q,
                                 float * __restrict__ k,
                                 const float * __restrict__ qb,
                                 const float * __restrict__ kb,
                                 const float * __restrict__ cs, int qrows,
                                 int krows, int dim, int rot) {
  long half = rot / 2;
  long rows = qrows + krows;
  long nrot = rows * half;
  long ntail = (long)rows * (dim - rot);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nrot + ntail) return;
  float *p;
  const float *bias;
  long row, boff;
  if (i < nrot) {
    row = i / half;
    int j = (int)(i % half);
    if (row < qrows) { p = q + (size_t)row * dim; bias = qb; boff = (size_t)row * dim; }
    else { p = k + (size_t)(row - qrows) * dim; bias = kb;
           boff = (size_t)(row - qrows) * dim; }
    float x1 = p[j] + (bias ? bias[boff + j] : 0.0f);
    float x2 = p[j + half] + (bias ? bias[boff + half + j] : 0.0f);
    float c1 = cs[j], c2 = cs[half + j];
    float s1 = cs[rot + j], s2 = cs[rot + half + j];
    p[j] = x1 * c1 - x2 * s1;
    p[j + half] = x2 * c2 + x1 * s2;
  } else {
    long t = i - nrot;
    row = t / (dim - rot);
    int d = rot + (int)(t % (dim - rot));
    if (row < qrows) { p = q + (size_t)row * dim; bias = qb; boff = (size_t)row * dim; }
    else { p = k + (size_t)(row - qrows) * dim; bias = kb;
           boff = (size_t)(row - qrows) * dim; }
    p[d] += (bias ? bias[boff + d] : 0.0f);
  }
}

extern "C" int sllm_rope_bias(float *q, float *k, const float *qb,
                              const float *kb, const float *cs, int qrows,
                              int krows, int dim, int rot) {
  if (qrows < 0 || krows < 0 || dim <= 0) return -1;
  if (rot <= 0 || rot % 2 || rot > dim) return -1;
  long total = (long)(qrows + krows) * ((rot / 2) + (dim - rot));
  if (!total) return 0;
  rope_bias_kernel<<<(total + 255) / 256, 256>>>(q, k, qb, kb, cs, qrows,
                                                 krows, dim, rot);
  CHECK(cudaGetLastError());
  return 0;
}

// out = silu(g) * u, computed fp32, stored as dtype T.
template <int T>
__global__ void silu_mul_kernel(const float *__restrict__ g,
                                const float *__restrict__ u,
                                void *__restrict__ out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float gv = g[i];
    t_store<T>(out, i, gv * (1.0f / (1.0f + expf(-gv))) * u[i]);
  }
}

extern "C" int sllm_silu_mul(const float *g, const float *u, void *out,
                             long n, int out_type) {
  if (n <= 0) return 0;
  long blocks = (n + 255) / 256;
  if (out_type == SLLM_T_BF16)
    silu_mul_kernel<SLLM_T_BF16><<<blocks, 256>>>(g, u, out, n);
  else
    silu_mul_kernel<SLLM_T_F32><<<blocks, 256>>>(g, u, out, n);
  CHECK(cudaGetLastError());
  return 0;
}

// Append row(heads,dim) (fp32) + optional bias into typed kv(heads,cap,dim)
// at pos; fp32 staging mirror of the biased row for the host batch copy.
template <int T>
__global__ void kv_write_t_kernel(void *__restrict__ kv, int cap, int pos,
                                  const float *__restrict__ row,
                                  const float *__restrict__ bias, int heads,
                                  int dim, float *__restrict__ stage,
                                  int stage_off) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= heads * dim) return;
  int h = t / dim, d = t % dim;
  float val = row[t] + (bias ? bias[t] : 0.0f);
  t_store<T>(kv, ((size_t)h * cap + pos) * dim + d, val);
  if (stage) stage[stage_off + t] = val;
}

extern "C" int sllm_kv_write_t(void *kv, int cap, int pos, const float *row,
                               const float *bias, int heads, int dim,
                               float *stage, int stage_off, int kv_type) {
  if (cap <= 0 || pos < 0 || pos >= cap || heads <= 0 || dim <= 0) return -1;
  long n = (long)heads * dim;
  long blocks = (n + 255) / 256;
  if (kv_type == SLLM_T_BF16)
    kv_write_t_kernel<SLLM_T_BF16><<<blocks, 256>>>(kv, cap, pos, row, bias,
                                                    heads, dim, stage,
                                                    stage_off);
  else
    kv_write_t_kernel<SLLM_T_F32><<<blocks, 256>>>(kv, cap, pos, row, bias,
                                                   heads, dim, stage,
                                                   stage_off);
  CHECK(cudaGetLastError());
  return 0;
}

// GQA last-row attention: fp32 q (straight from the qkv GEMM), typed KV,
// fp32 math; output written in dtype T so it feeds the next typed GEMM.
template <int T>
__global__ void attention_decode_gqa_t_kernel(const float *__restrict__ q,
                                              const void *__restrict__ K,
                                              const void *__restrict__ V,
                                              float *__restrict__ scores,
                                              void *__restrict__ out, int S,
                                              int stride, int D, int n_rep,
                                              float scale) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  int kh = h / n_rep;
  __shared__ float red[256];
  const float *qh = q + (size_t)h * D;
  const long qbase = (size_t)h * D;
  const long kbase = (size_t)kh * stride * D;
  float *sh = scores + (size_t)h * S;

  for (int s = tid; s < S; s += blockDim.x) {
    float acc = 0.0f;
    const long kb = kbase + (size_t)s * D;
    for (int d = 0; d < D; ++d) acc += qh[d] * t_load<T>(K, kb + d);
    sh[s] = acc * scale;
  }
  __syncthreads();
  float m = -1e30f;
  for (int s = tid; s < S; s += blockDim.x) m = fmaxf(m, sh[s]);
  red[tid] = m;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
    __syncthreads();
  }
  float mval = red[0];
  float psum = 0.0f;
  // exact expf (not __expf): keep the generic engine's fp32 floor tight.
  for (int s = tid; s < S; s += blockDim.x) psum += expf(sh[s] - mval);
  red[tid] = psum;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  float inv = 1.0f / red[0];
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < S; ++s)
      acc += expf(sh[s] - mval) * t_load<T>(V, kbase + (size_t)s * D + d);
    t_store<T>(out, qbase + d, acc * inv);
  }
}

extern "C" int sllm_attention_decode_t(const float *q, const void *K,
                                       const void *V, float *scores,
                                       void *out, int heads, int kv_heads,
                                       int S, int stride, int D, float scale,
                                       int t) {
  if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || S <= 0 || D <= 0 ||
      stride < S)
    return -1;
  int n_rep = heads / kv_heads;
  if (t == SLLM_T_BF16)
    attention_decode_gqa_t_kernel<SLLM_T_BF16><<<heads, 256>>>(
        q, K, V, scores, out, S, stride, D, n_rep, scale);
  else
    attention_decode_gqa_t_kernel<SLLM_T_F32><<<heads, 256>>>(
        q, K, V, scores, out, S, stride, D, n_rep, scale);
  CHECK(cudaGetLastError());
  return 0;
}
