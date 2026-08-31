// B2 probe: does cublasLt on this GB10 support FP8 (E4M3) GEMM with
// 128-block scaling (the DeepSeek-style layout our qwen4_exp fp8 checkpoint
// ships: weight_scale_inv per 128x128 block)?
//
// Strategy: run the SAME 256x256x256 GEMM with scale tensors filled with
// 1.0. Support detection needs no layout assumptions (identity scales make
// every scale layout numerically identical); exact scale-layout semantics
// are pinned later by the parity kernel, NOT by this probe.
//
// Build (head):  nvcc -O2 -arch=native probe_cublaslt_fp8.cu -o probe_fp8 -lcublasLt
// Run:           ./probe_fp8
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define CK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  printf("  cublasLt error %d @%d\n", (int)s_, __LINE__); return -1; } } while (0)
#define CUDA_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("  cuda error %s @%d\n", cudaGetErrorString(e_), __LINE__); return -1; } } while (0)

static const int M = 256, N = 256, K = 256;

// fp8 e4m3 encode/decode on the host (probe-local, subnormal-ignoring is
// fine for values in [0.1, 1.0])
static float e4m3_decode(unsigned char b) {
  unsigned char s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
  if (e == 0xF && m == 7) return NAN;
  float v = e ? ldexpf(1.0f + m / 8.0f, (int)e - 7) : m / 512.0f;
  return s ? -v : v;
}
static unsigned char e4m3_encode(float v) {
  if (std::isnan(v)) return 0x7F;
  unsigned char s = v < 0 ? 0x80 : 0; v = fabsf(v);
  int e = 7; while (e > 0 && v < ldexpf(1.0f, e - 7)) --e;
  while (e < 15 && v >= ldexpf(2.0f, e - 7)) ++e;
  int m = (int)roundf((v / ldexpf(1.0f, e - 7) - 1.0f) * 8.0f);
  if (m > 7) { ++e; m = 0; }
  if (e == 0) return s | (unsigned char)(int)roundf(v * 512.0f);
  return s | (unsigned char)(e << 3) | (unsigned char)m;
}

struct Case { const char* name; bool use_a_mode; int a_mode; bool use_b_mode; int b_mode; };

static const size_t WS_BYTES = 32u << 20;

int run_case(cublasLtHandle_t lt, const Case& c,
             unsigned char* dA, unsigned char* dB, float* dC,
             float* alpha_dev, void* ws) {
  cublasLtMatmulDesc_t op;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;  // fp8 needs TN
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                    &alpha_dev, sizeof(alpha_dev)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                    &alpha_dev, sizeof(alpha_dev)));
#ifdef CUBLASLT_MATMUL_DESC_A_SCALE_MODE
  if (c.use_a_mode) {
    cublasLtMatmulMatrixScale_t m = (cublasLtMatmulMatrixScale_t)c.a_mode;
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &m, sizeof(m)));
  }
  if (c.use_b_mode) {
    cublasLtMatmulMatrixScale_t m = (cublasLtMatmulMatrixScale_t)c.b_mode;
    CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &m, sizeof(m)));
  }
#else
  if (c.use_a_mode || c.use_b_mode) { printf("  toolkit lacks SCALE_MODE attrs\n"); return -2; }
#endif

  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, M, K));  // col-major K x M (stored X^T)
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, N, K));  // col-major K x N (W row-major)
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, M, N, M));      // col-major M x N

  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &WS_BYTES, sizeof(WS_BYTES)));
  cublasLtMatmulHeuristicResult_t heur[4];
  int nres = 0;
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 4, heur, &nres);
  if (hs != CUBLAS_STATUS_SUCCESS || nres == 0) {
    printf("  %-42s NO ALGO (status=%d nres=%d)\n", c.name, (int)hs, nres);
    return 1;
  }
  float one = 1.0f, zero = 0.0f;
  CK(cublasLtMatmul(lt, op, &one, dA, la, dB, lb, &zero, dC, lc, dC, lc,
                    &heur[0].algo, ws, WS_BYTES, 0));
  CUDA_CK(cudaDeviceSynchronize());
  std::vector<float> h(M * N);
  CUDA_CK(cudaMemcpy(h.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  // CPU reference with identity scales (layout-agnostic correctness)
  double max_err = 0.0;
  std::vector<float> da(M * K), db(K * N);
  // host copies (A is X (M,K) row-major stored as dA; B is W (N,K) row-major)
  CUDA_CK(cudaMemcpy(da.data(), dA, M * K, cudaMemcpyDeviceToHost));
  CUDA_CK(cudaMemcpy(db.data(), dB, K * N, cudaMemcpyDeviceToHost));
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (double)e4m3_decode(da[i * K + k]) * e4m3_decode(db[j * K + k]);
      max_err = std::max(max_err, std::fabs(acc - h[j * M + i]));
    }
  printf("  %-42s OK  (nres=%d, max_err vs cpu ref = %.3e)\n", c.name, nres, max_err);
  cublasLtMatmulPreferenceDestroy(pref);
  return 0;
}

int main() {
  printf("cublasLt version: %zu\n", cublasLtGetVersion());
  cublasLtHandle_t lt;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) { printf("LtCreate failed\n"); return 1; }

  std::vector<unsigned char> hA(M * K), hB(K * N);
  srand(1234);
  for (auto& x : hA) x = e4m3_encode(0.1f + 0.9f * rand() / RAND_MAX);
  for (auto& x : hB) x = e4m3_encode(0.1f + 0.9f * rand() / RAND_MAX);
  unsigned char *dA, *dB; float *dC, *dS; void* ws;
  CUDA_CK(cudaMalloc(&dA, M * K)); CUDA_CK(cudaMalloc(&dB, K * N));
  CUDA_CK(cudaMalloc(&dC, sizeof(float) * M * N));
  CUDA_CK(cudaMalloc(&dS, sizeof(float)));
  CUDA_CK(cudaMalloc(&ws, WS_BYTES));
  float one = 1.0f;
  CUDA_CK(cudaMemcpy(dA, hA.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dB, hB.data(), K * N, cudaMemcpyHostToDevice));
  CUDA_CK(cudaMemcpy(dS, &one, sizeof(float), cudaMemcpyHostToDevice));

  int rc = 0;
  Case cases[] = {
    {"scalar fp8 (E4M3 x E4M3 -> F32, dev scales)", false, 0, false, 0},
#ifdef CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
    {"A=VEC128_32F, B=BLK128x128_32F (DeepSeek-like)", true,
     (int)CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, true,
#ifdef CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F
     (int)CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F},
    {"A=BLK128x128_32F, B=BLK128x128_32F", true,
     (int)CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F, true,
     (int)CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F},
#else
     0},
#endif
#endif
  };
  for (const auto& c : cases) {
    int r = run_case(lt, c, dA, dB, dC, dS, ws);
    if (r < 0) { rc = 1; break; }
  }
  printf("probe done (support matrix above; NO ALGO = unsupported on this build)\n");
  return rc;
}
