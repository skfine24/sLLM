// sLLM qwen4_exp kernel set (operational track A4, decode-path v1).
//
// Numpy-oracle parity targets (ref/qwen4_exp.py, which cites
// oracle/upstream/sglang/*): hyper-connection mix/combine, MoE router
// (softmax -> top-k stable -> renorm 1e-20), swiglu, QSA indexer chain
// (block pool, MQA relu logits, stable top-k) and sparse decode attention.
//
// v1 style matches the transfer-era API of kernels.cu (host pointers,
// per-call H2D/D2H): the priority here is a PARITY-VERIFIABLE port of the
// oracle semantics; device-resident composition (one sync per step, typed
// operands) is milestone C1 and reuses the same __global__ bodies.
//
// GDN prefill/decode reuses sllm_gated_delta_step from kernels.cu (same
// GatedDeltaNet family, verified in test_hybrid_gpu).

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// error reporting (separate symbol: kernels.cu owns sllm_last_error)
static char g_q4_err[512] = {0};
extern "C" const char* sllm_q4_last_error(void) { return g_q4_err; }

#define Q4_CHECK(e)                                                        \
  do {                                                                     \
    cudaError_t _e = (e);                                                  \
    if (_e != cudaSuccess) {                                               \
      snprintf(g_q4_err, sizeof(g_q4_err), "%s: %s", #e,                   \
               cudaGetErrorString(_e));                                    \
      return -1;                                                           \
    }                                                                      \
  } while (0)

#define Q4_MAX_SMEM_ROWS 2048

static __device__ __forceinline__ float q4_silu(float x) {
  return x / (1.0f + expf(-x));
}
static __device__ __forceinline__ float q4_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// hyper-connection (oracle: ref/qwen4_exp.hc_mix / hc_combine)
// ---------------------------------------------------------------------------

// out[r, i] = x[r,i] * rsqrt(mean(x[r, group]^2) + eps) * (1 + w[i])
// with groups of `hs` along the hc*hs feature axis (GroupedGemmaRMSNorm).
__global__ void grouped_gemma_rmsnorm_kernel(const float* __restrict__ x,
                                             const float* __restrict__ w,
                                             float* __restrict__ y, int rows,
                                             int hc, int hs, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float* xr = x + (size_t)r * hc * hs;
  float* yr = y + (size_t)r * hc * hs;
  __shared__ float sh[256];
  for (int c = 0; c < hc; ++c) {
    const float* xg = xr + (size_t)c * hs;
    float* yg = yr + (size_t)c * hs;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hs; i += blockDim.x) acc += xg[i] * xg[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
      __syncthreads();
    }
    float inv = rsqrtf(sh[0] / (float)hs + eps);
    for (int i = threadIdx.x; i < hs; i += blockDim.x)
      yg[i] = xg[i] * inv * (1.0f + w[i]);
    __syncthreads();
  }
}

extern "C" int sllm_q4_grouped_gemma_rmsnorm(const float* x, const float* w,
                                             float* y, int rows, int hc,
                                             int hs, float eps) {
  if (rows <= 0 || hc <= 0 || hs <= 0) return -1;
  size_t nb = (size_t)rows * hc * hs * sizeof(float);
  float *dx, *dw, *dy;
  Q4_CHECK(cudaMalloc((void**)&dx, nb));
  Q4_CHECK(cudaMalloc((void**)&dw, (size_t)hc * hs * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dy, nb));
  Q4_CHECK(cudaMemcpy(dx, x, nb, cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dw, w, (size_t)hc * hs * sizeof(float),
                      cudaMemcpyHostToDevice));
  grouped_gemma_rmsnorm_kernel<<<rows, 256>>>(dx, dw, dy, rows, hc, hs, eps);
  Q4_CHECK(cudaMemcpy(y, dy, nb, cudaMemcpyDeviceToHost));
  cudaFree(dx); cudaFree(dw); cudaFree(dy);
  return 0;
}

// one-output-per-thread dense GEMV: out[r,o] = sum_k A[r,k] * W[o,k]
// (W row-major, O rows of K)
__global__ void gemv_rows_kernel(const float* __restrict__ A,
                                 const float* __restrict__ W,
                                 float* __restrict__ out, int rows, int K,
                                 int O, int post, float scale) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)rows * O;
  if (idx >= total) return;
  int r = (int)(idx / O), o = (int)(idx % O);
  const float* a = A + (size_t)r * K;
  const float* w = W + (size_t)o * K;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += a[k] * w[k];
  if (post == 1) acc = q4_silu(acc / scale);
  else if (post == 2) acc = q4_sigmoid(acc / scale);
  out[idx] = acc;
}

// post: 0 none, 1 silu(x/scale), 2 sigmoid(x/scale)
extern "C" int sllm_q4_gemv_rows(const float* A, const float* W, float* out,
                                 int rows, int K, int O, int post,
                                 float scale) {
  if (rows <= 0 || K <= 0 || O <= 0) return -1;
  float *dA, *dW, *dout;
  Q4_CHECK(cudaMalloc((void**)&dA, (size_t)rows * K * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dW, (size_t)O * K * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dout, (size_t)rows * O * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dA, A, (size_t)rows * K * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dW, W, (size_t)O * K * sizeof(float),
                      cudaMemcpyHostToDevice));
  size_t total = (size_t)rows * O;
  int threads = 256;
  gemv_rows_kernel<<<(unsigned)((total + threads - 1) / threads), threads>>>(
      dA, dW, dout, rows, K, O, post, scale);
  Q4_CHECK(cudaMemcpy(out, dout, total * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dW); cudaFree(dout);
  return 0;
}

// mixed[r, j] = mean_c wgate[r, c*hs+j] * normed[r, c*hs+j]
__global__ void hc_mix_apply_kernel(const float* __restrict__ wgate,
                                    const float* __restrict__ normed,
                                   float* __restrict__ mixed, int rows,
                                   int hc, int hs) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)rows * hs) return;   // rows/hc/hs > 0 via wrapper
  int r = (int)(idx / hs), j = (int)(idx % hs);
  float acc = 0.0f;
  for (int c = 0; c < hc; ++c) {
    size_t o = (size_t)r * hc * hs + (size_t)c * hs + j;
    acc += wgate[o] * normed[o];
  }
  mixed[idx] = acc / (float)hc;
}

extern "C" int sllm_q4_hc_mix_apply(const float* wgate, const float* normed,
                                    float* mixed, int rows, int hc, int hs) {
  if (rows <= 0 || hc <= 0 || hs <= 0) return -1;
  size_t rb = (size_t)rows * hc * hs * sizeof(float);
  float *dw, *dn, *dm;
  Q4_CHECK(cudaMalloc((void**)&dw, rb));
  Q4_CHECK(cudaMalloc((void**)&dn, rb));
  Q4_CHECK(cudaMalloc((void**)&dm, (size_t)rows * hs * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dw, wgate, rb, cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dn, normed, rb, cudaMemcpyHostToDevice));
  size_t total = (size_t)rows * hs;
  hc_mix_apply_kernel<<<(unsigned)((total + 255) / 256), 256>>>(dw, dn, dm,
                                                                rows, hc, hs);
  Q4_CHECK(cudaMemcpy(mixed, dm, total * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dw); cudaFree(dn); cudaFree(dm);
  return 0;
}

// inj[r, c] = 2*sigmoid(dot(normed[r], inj_w[c]) / hc);
// hyper[r, c*hs+j] += inj[r, c] * block_out[r, j]
// (block per (row, c): dot then broadcast update; hc is tiny (<= 4)).
__global__ void hc_combine_kernel(float* __restrict__ hyper,
                                  const float* __restrict__ block_out,
                                  const float* __restrict__ normed,
                                  const float* __restrict__ inj_w, int rows,
                                  int hc, int hs) {
  int r = blockIdx.x, c = blockIdx.y;
  if (r >= rows || c >= hc) return;
  const float* nrow = normed + (size_t)r * hc * hs;
  const float* wrow = inj_w + (size_t)c * hc * hs;
  __shared__ float sh[256];
  float acc = 0.0f;
  for (int i = threadIdx.x; i < hc * hs; i += blockDim.x) acc += nrow[i] * wrow[i];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  float inj = 2.0f * q4_sigmoid(sh[0] / (float)hc);
  float* hrow = hyper + (size_t)r * hc * hs + (size_t)c * hs;
  const float* brow = block_out + (size_t)r * hs;
  for (int j = threadIdx.x; j < hs; j += blockDim.x) hrow[j] += inj * brow[j];
}

extern "C" int sllm_q4_hc_combine(float* hyper, const float* block_out,
                                  const float* normed, const float* inj_w,
                                  int rows, int hc, int hs) {
  if (rows <= 0 || hc <= 0 || hs <= 0) return -1;
  size_t rb = (size_t)rows * hc * hs * sizeof(float);
  float *dh, *dbo, *dn, *dw;
  Q4_CHECK(cudaMalloc((void**)&dh, rb));
  Q4_CHECK(cudaMalloc((void**)&dbo, (size_t)rows * hs * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dn, rb));
  Q4_CHECK(cudaMalloc((void**)&dw, (size_t)hc * hc * hs * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dh, hyper, rb, cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dbo, block_out, (size_t)rows * hs * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dn, normed, rb, cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dw, inj_w, (size_t)hc * hc * hs * sizeof(float),
                      cudaMemcpyHostToDevice));
  dim3 grid(rows, hc);
  hc_combine_kernel<<<grid, 256>>>(dh, dbo, dn, dw, rows, hc, hs);
  Q4_CHECK(cudaMemcpy(hyper, dh, rb, cudaMemcpyDeviceToHost));
  cudaFree(dh); cudaFree(dbo); cudaFree(dn); cudaFree(dw);
  return 0;
}

// ---------------------------------------------------------------------------
// MoE (oracle: ref/qwen4_exp.moe_route / swiglu_mlp / shared gate)
// ---------------------------------------------------------------------------

// per token: softmax -> top-k (desc value, ties by lower index) -> renorm
// (eps 1e-20). E <= 2048 (smem copy + selection marks).
__global__ void moe_router_kernel(const float* __restrict__ logits,
                                  float* __restrict__ w_out,
                                  int* __restrict__ id_out, int E, int topk) {
  int t = blockIdx.x;
  extern __shared__ float p[];                    // E floats (probs)
  const float* lg = logits + (size_t)t * E;
  __shared__ float shm;                           // row max
  __shared__ float shs;                           // softmax sum
  // v1 keeps the reduction order deliberately simple (E <= 2048): thread 0
  // computes the row max; parity with the numpy oracle is what matters.
  if (threadIdx.x == 0) {
    float mm = -INFINITY;
    for (int i = 0; i < E; ++i) mm = fmaxf(mm, lg[i]);
    shm = mm;
    shs = 0.0f;
  }
  __syncthreads();
  float local = 0.0f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    float e = expf(lg[i] - shm);
    p[i] = e;
    local += e;
  }
  if (local != 0.0f) atomicAdd(&shs, local);
  __syncthreads();
  float sum = shs;
  for (int i = threadIdx.x; i < E; i += blockDim.x) p[i] /= sum;
  __syncthreads();
  // stable selection + renorm (eps 1e-20): thread 0, topk <= 16 over <= 512
  if (threadIdx.x == 0) {
    float wsum = 1e-20f;
    float ws[16];
    int ids[16];
    for (int k = 0; k < topk; ++k) {
      int best = -1;
      float bv = -INFINITY;
      for (int i = 0; i < E; ++i) {
        if (p[i] > bv) { bv = p[i]; best = i; }   // strict > keeps leftmost
      }
      ws[k] = bv; ids[k] = best;
      p[best] = -INFINITY;
      wsum += bv;
    }
    for (int k = 0; k < topk; ++k) {
      w_out[(size_t)t * topk + k] = ws[k] / wsum;
      id_out[(size_t)t * topk + k] = ids[k];
    }
  }
}

extern "C" int sllm_q4_moe_router(const float* logits, float* w_out,
                                  int* id_out, int n, int E, int topk) {
  if (n <= 0 || E <= 0 || topk <= 0 || topk > E || E > Q4_MAX_SMEM_ROWS ||
      topk > 16)
    return -1;
  float *dl, *dw;
  int *di;
  Q4_CHECK(cudaMalloc((void**)&dl, (size_t)n * E * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dw, (size_t)n * topk * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&di, (size_t)n * topk * sizeof(int)));
  Q4_CHECK(cudaMemcpy(dl, logits, (size_t)n * E * sizeof(float),
                      cudaMemcpyHostToDevice));
  moe_router_kernel<<<n, 256, E * sizeof(float)>>>(dl, dw, di, E, topk);
  Q4_CHECK(cudaMemcpy(w_out, dw, (size_t)n * topk * sizeof(float),
                      cudaMemcpyDeviceToHost));
  Q4_CHECK(cudaMemcpy(id_out, di, (size_t)n * topk * sizeof(int),
                      cudaMemcpyDeviceToHost));
  cudaFree(dl); cudaFree(dw); cudaFree(di);
  return 0;
}

// out = silu(g) * u (elementwise, n floats)
__global__ void swiglu_kernel(const float* __restrict__ g,
                              const float* __restrict__ u,
                              float* __restrict__ out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = q4_silu(g[i]) * u[i];
}

extern "C" int sllm_q4_swiglu(const float* g, const float* u, float* out,
                              long n) {
  float *dg, *du, *do_;
  Q4_CHECK(cudaMalloc((void**)&dg, n * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&du, n * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&do_, n * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dg, g, n * sizeof(float), cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(du, u, n * sizeof(float), cudaMemcpyHostToDevice));
  swiglu_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dg, du, do_, n);
  Q4_CHECK(cudaMemcpy(out, do_, n * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dg); cudaFree(du); cudaFree(do_);
  return 0;
}

// out[t, :] += w[t] * y[t, :]  (per-row scalar weight from device array)
__global__ void axpy_rows_kernel(float* __restrict__ out,
                                 const float* __restrict__ y,
                                 const float* __restrict__ w, int n, int H) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)n * H) return;
  int t = (int)(i / H);
  out[i] += w[t] * y[i];
}

extern "C" int sllm_q4_axpy_rows(float* out, const float* y, const float* w,
                                 int n, int H) {
  float *dout, *dy, *dw;
  Q4_CHECK(cudaMalloc((void**)&dout, (size_t)n * H * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dy, (size_t)n * H * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dw, (size_t)n * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dout, out, (size_t)n * H * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dy, y, (size_t)n * H * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dw, w, (size_t)n * sizeof(float),
                      cudaMemcpyHostToDevice));
  long total = (long)n * H;
  axpy_rows_kernel<<<(unsigned)((total + 255) / 256), 256>>>(dout, dy, dw, n,
                                                             H);
  Q4_CHECK(cudaMemcpy(out, dout, total * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dout); cudaFree(dy); cudaFree(dw);
  return 0;
}

// out[t, :] += sigmoid(g[t]) * shared[t, :]  (shared-expert gate)
__global__ void shared_gate_kernel(float* __restrict__ out,
                                   const float* __restrict__ shared,
                                   const float* __restrict__ g, int n, int H) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)n * H) return;
  int t = (int)(i / H);
  out[i] += q4_sigmoid(g[t]) * shared[i];
}

extern "C" int sllm_q4_shared_gate_accum(float* out, const float* shared,
                                         const float* g, int n, int H) {
  float *dout, *dsh, *dg;
  Q4_CHECK(cudaMalloc((void**)&dout, (size_t)n * H * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dsh, (size_t)n * H * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dg, (size_t)n * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dout, out, (size_t)n * H * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dsh, shared, (size_t)n * H * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dg, g, (size_t)n * sizeof(float),
                      cudaMemcpyHostToDevice));
  long total = (long)n * H;
  shared_gate_kernel<<<(unsigned)((total + 255) / 256), 256>>>(dout, dsh, dg,
                                                               n, H);
  Q4_CHECK(cudaMemcpy(out, dout, total * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dout); cudaFree(dsh); cudaFree(dg);
  return 0;
}

// ---------------------------------------------------------------------------
// QSA indexer + sparse attention (oracle: ref/qwen4_exp qsa_* functions)
// ---------------------------------------------------------------------------

// GemmaRMSNorm rows: y = x * rsqrt(mean(x^2)+eps) * (1 + w)
__global__ void gemma_rmsnorm_kernel(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                      float* __restrict__ y, int rows, int d,
                                      float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  // d <= 256 threads-cover via strided loop; hs/d > 0 enforced by the wrapper
  const float* xr = x + (size_t)r * d;
  float* yr = y + (size_t)r * d;
  __shared__ float sh[256];
  float acc = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) acc += xr[i] * xr[i];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sh[0] / (float)d + eps);
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    yr[i] = xr[i] * inv * (1.0f + w[i]);
}

extern "C" int sllm_q4_gemma_rmsnorm(const float* x, const float* w,
                                     float* y, int rows, int d, float eps) {
  if (rows <= 0 || d <= 0) return -1;
  float *dx, *dw, *dy;
  Q4_CHECK(cudaMalloc((void**)&dx, (size_t)rows * d * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dw, (size_t)d * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dy, (size_t)rows * d * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dx, x, (size_t)rows * d * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dw, w, (size_t)d * sizeof(float),
                      cudaMemcpyHostToDevice));
  gemma_rmsnorm_kernel<<<rows, 256>>>(dx, dw, dy, rows, d, eps);
  Q4_CHECK(cudaMemcpy(y, dy, (size_t)rows * d * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dx); cudaFree(dw); cudaFree(dy);
  return 0;
}

// NeoX partial rope in place on rows of d: rotate [..., :rot] (cos/sin are
// length-rot doubled-half rows, one per row).
__global__ void rope_partial_kernel(float* __restrict__ x,
                                    const float* __restrict__ cosb,
                                    const float* __restrict__ sinb, int rows,
                                    int d, int rot) {
  int r = blockIdx.x;
  if (r >= rows) return;
  float* xr = x + (size_t)r * d;
  const float* c = cosb + (size_t)r * rot;
  const float* s = sinb + (size_t)r * rot;
  int half = rot / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float a = xr[i], b = xr[i + half];
    xr[i] = a * c[i] - b * s[i];
    xr[i + half] = b * c[i + half] + a * s[i + half];
  }
}

extern "C" int sllm_q4_rope_partial(float* x, const float* cosb,
                                    const float* sinb, int rows, int d,
                                    int rot) {
  if (rot <= 0 || rot % 2 || rot > d) return -1;
  float *dx, *dc, *ds;
  Q4_CHECK(cudaMalloc((void**)&dx, (size_t)rows * d * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dc, (size_t)rows * rot * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&ds, (size_t)rows * rot * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dx, x, (size_t)rows * d * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dc, cosb, (size_t)rows * rot * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(ds, sinb, (size_t)rows * rot * sizeof(float),
                      cudaMemcpyHostToDevice));
  rope_partial_kernel<<<rows, 256>>>(dx, dc, ds, rows, d, rot);
  Q4_CHECK(cudaMemcpy(x, dx, (size_t)rows * d * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dx); cudaFree(dc); cudaFree(ds);
  return 0;
}

// compressed-key block: out[j] = mean_i tok_k[end-ratio+i, j]
__global__ void pool_block_kernel(const float* __restrict__ tok_k,
                                  float* __restrict__ out, int end, int ratio,
                                  int d) {
  for (int j = threadIdx.x; j < d; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < ratio; ++i) acc += tok_k[(size_t)(end - ratio + i) * d + j];
    out[j] = acc / (float)ratio;
  }
}

extern "C" int sllm_q4_qsa_pool_block(const float* tok_k, float* out, int end,
                                      int ratio, int d) {
  if (d <= 0 || ratio <= 0 || end < ratio) return -1;
  float *dt, *do_;
  Q4_CHECK(cudaMalloc((void**)&dt, (size_t)end * d * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&do_, (size_t)d * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dt, tok_k, (size_t)end * d * sizeof(float),
                      cudaMemcpyHostToDevice));
  pool_block_kernel<<<1, 256>>>(dt, do_, end, ratio, d);
  Q4_CHECK(cudaMemcpy(out, do_, (size_t)d * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dt); cudaFree(do_);
  return 0;
}

// MQA relu logits for rows [start,end): logits[b] = sum_h relu(q[h,:] . ck[b,:]) / scale
__global__ void qsa_mqa_kernel(const float* __restrict__ q,
                               const float* __restrict__ ck,
                               float* __restrict__ logits, int nh, int d,
                               int start, float scale) {
  int b = blockIdx.x + start;
  __shared__ float sh[256];
  const float* kb = ck + (size_t)b * d;
  float acc = 0.0f;
  for (int h = 0; h < nh; ++h) {
    const float* qh = q + (size_t)h * d;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dot += qh[i] * kb[i];
    sh[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
      __syncthreads();
    }
    acc += fmaxf(sh[0], 0.0f);
    __syncthreads();
  }
  if (threadIdx.x == 0) logits[b] = acc / scale;
}

extern "C" int sllm_q4_qsa_mqa_logits(const float* q, const float* ck,
                                      float* logits, int nh, int d, int start,
                                      int end, float scale) {
  if (nh <= 0 || d <= 0 || start < 0) return -1;
  if (end <= start) return 0;
  float *dq, *dc, *dl;
  Q4_CHECK(cudaMalloc((void**)&dq, (size_t)nh * d * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dc, (size_t)end * d * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dl, (size_t)end * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dq, q, (size_t)nh * d * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dc, ck, (size_t)end * d * sizeof(float),
                      cudaMemcpyHostToDevice));
  qsa_mqa_kernel<<<end - start, 256>>>(dq, dc, dl, nh, d, start, scale);
  Q4_CHECK(cudaMemcpy(logits + start, dl + start,
                      (size_t)(end - start) * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dq); cudaFree(dc); cudaFree(dl);
  return 0;
}

// stable top-k over logits[start,end) (desc value, ties -> lower index),
// writing k RELATIVE indices; -1 padding. One block per row (m rows).
// v1: the row is copied to dynamic shared memory (fits up to
// Q4_MAX_SMEM_ROWS blocks) and thread 0 does the stable selection loop
// (topk <= 512). C1 replaces this with the radix-based fast-topk port of
// upstream qsa/kernel.py for long contexts.
__global__ void qsa_topk_sm_kernel(const float* __restrict__ logits,
                                   int* __restrict__ out, int stride,
                                   int start, int end, int topk) {
  int r = blockIdx.x;
  const float* row = logits + (size_t)r * stride;
  int len = end - start;
  extern __shared__ float sm[];
  for (int i = threadIdx.x; i < len; i += blockDim.x) sm[i] = row[start + i];
  __syncthreads();
  int width = len < topk ? len : topk;
  if (threadIdx.x == 0) {
    for (int k = 0; k < width; ++k) {
      int best = -1;
      float bv = -INFINITY;
      for (int i = 0; i < len; ++i) {
        if (sm[i] > bv) { bv = sm[i]; best = i; }
      }
      out[(size_t)r * topk + k] = best;
      sm[best] = -INFINITY;
    }
    for (int k = width; k < topk; ++k) out[(size_t)r * topk + k] = -1;
  }
}

extern "C" int sllm_q4_qsa_topk(const float* logits, int* out, int m,
                                int stride, int start, int end, int topk) {
  if (m <= 0 || topk <= 0 || start < 0 || end <= start) return -1;
  float* dl;
  int* di;
  Q4_CHECK(cudaMalloc((void**)&dl, (size_t)m * stride * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&di, (size_t)m * topk * sizeof(int)));
  Q4_CHECK(cudaMemcpy(dl, logits, (size_t)m * stride * sizeof(float),
                      cudaMemcpyHostToDevice));
  int len = end - start;
  if (len > Q4_MAX_SMEM_ROWS) return -2;  // C1 adds the radix fast-topk
  qsa_topk_sm_kernel<<<m, 256, (size_t)(len > 0 ? len : 1) * sizeof(float)>>>(
      dl, di, stride, start, end, topk);
  Q4_CHECK(cudaMemcpy(out, di, (size_t)m * topk * sizeof(int),
                      cudaMemcpyDeviceToHost));
  cudaFree(dl); cudaFree(di);
  return 0;
}

// sparse decode attention: one block per query head; softmax over the
// selected slots (in-sequence token ids, -1 padded); KV rows gathered from
// (kvh, cap, hd) caches.
__global__ void qsa_sparse_attn_kernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const int* __restrict__ slots,
                                       float* __restrict__ out, int nh,
                                       int kvh, int hd, long kcap, int W,
                                       float scale) {
  int h = blockIdx.x;
  if (h >= nh) return;
  int kv = (int)((long)h * kvh / nh);
  const float* qh = q + (size_t)h * hd;
  const float* kk = k + (size_t)kv * kcap * hd;
  const float* vv = v + (size_t)kv * kcap * hd;
  __shared__ float scores[4096];
  __shared__ int valid[4096];
  __shared__ float mred[256];
  __shared__ float sred[256];
  float m = -INFINITY;
  for (int t = threadIdx.x; t < W; t += blockDim.x) {
    int tok = slots[t];
    int ok = tok >= 0;
    valid[t] = ok;
    float sc = 0.0f;
    if (ok) {
      for (int j = 0; j < hd; ++j) sc += qh[j] * kk[(size_t)tok * hd + j];
      sc *= scale;
    }
    scores[t] = sc;
    if (ok && sc > m) m = sc;
  }
  // block max/min over the t-chunks was sequential per thread; reduce:
  mred[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) mred[threadIdx.x] = fmaxf(mred[threadIdx.x], mred[threadIdx.x + s]);
    __syncthreads();
  }
  m = mred[0];
  float lsum = 0.0f;
  for (int t = threadIdx.x; t < W; t += blockDim.x) {
    scores[t] = valid[t] ? expf(scores[t] - m) : 0.0f;
    lsum += scores[t];
  }
  sred[threadIdx.x] = lsum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
    __syncthreads();
  }
  float l = sred[0];
  float inv = l > 0.0f ? 1.0f / l : 0.0f;
  float* o = out + (size_t)h * hd;
  for (int j = threadIdx.x; j < hd; j += blockDim.x) {
    float acc = 0.0f;
    for (int t = 0; t < W; ++t) {
      if (valid[t] && scores[t] != 0.0f)
        acc += scores[t] * vv[(size_t)slots[t] * hd + j];
    }
    o[j] = acc * inv;
  }
}

extern "C" int sllm_q4_qsa_sparse_attn(const float* q, const float* k,
                                       const float* v, const int* slots,
                                       float* out, int nh, int kvh, int hd,
                                       long kcap, int W, float scale) {
  if (nh <= 0 || kvh <= 0 || hd <= 0 || kcap <= 0 || W < 0) return -1;
  if (W > 4096) return -2;  // v1 smem bound (budget+ratio-1 = 2051 fits)
  float *dq, *dk, *dv, *do_;
  int* dsl;
  Q4_CHECK(cudaMalloc((void**)&dq, (size_t)nh * hd * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dk, (size_t)kvh * kcap * hd * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dv, (size_t)kvh * kcap * hd * sizeof(float)));
  Q4_CHECK(cudaMalloc((void**)&dsl, (size_t)W * sizeof(int)));
  Q4_CHECK(cudaMalloc((void**)&do_, (size_t)nh * hd * sizeof(float)));
  Q4_CHECK(cudaMemcpy(dq, q, (size_t)nh * hd * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dk, k, (size_t)kvh * kcap * hd * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dv, v, (size_t)kvh * kcap * hd * sizeof(float),
                      cudaMemcpyHostToDevice));
  Q4_CHECK(cudaMemcpy(dsl, slots, (size_t)W * sizeof(int),
                      cudaMemcpyHostToDevice));
  qsa_sparse_attn_kernel<<<nh, 256>>>(dq, dk, dv, dsl, do_, nh, kvh, hd, kcap,
                                      W, scale);
  Q4_CHECK(cudaMemcpy(out, do_, (size_t)nh * hd * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dsl); cudaFree(do_);
  return 0;
}
