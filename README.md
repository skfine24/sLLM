# sLLM

sLLM은 듀얼노드 NVIDIA DGX Spark(GB10)를 위한 **자체 개발 모델 서빙 엔진**입니다.
하나의 엔진으로 여러 모델을 하나씩 로드해 서빙하며, OpenAI 호환 HTTP API와 CLI를
제공합니다.

| 레시피 | 모델 | 아키텍처 | 크기 |
|---|---|---|---|
| `Qwen3.8-27B-FP8.yaml` | Qwen/Qwen3.8-27B-FP8 | GDN 하이브리드 + MTP | ~29 GiB |
| `Qwen3.8-Flash-Next-FP8.yaml` | Qwen/Qwen3.8-Flash-Next-FP8 | GDN + QSA + MoE + PLE + MTP | ~173 GiB |
| `DeepSeek-V4-Flash-0731.yaml` | deepseek-ai/DeepSeek-V4-Flash-0731 | MLA + sparse + MoE(fp4) + DSPark | ~156 GiB |
| `Qwen2.5-Coder-0.5B.yaml` | Qwen/Qwen2.5-Coder-0.5B | 표준 dense transformer | ~1 GiB |

---

## 주요 기능

- **하나의 엔진, 여러 아키텍처** — `qwen3_5`(GDN), `qwen4_exp`(QSA/MoE/PLE/MTP),
  `deepseek_v4`(MLA/HDC/fp4/DSPark), 표준 Llama/Qwen2 계열을 한 서빙 스택으로 처리
- **OpenAI 호환 API** — `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
  `stream:true` SSE, 요청당 `usage` 통계
- **GPU-AUTO 기본** — CUDA+`.so`가 있으면 GPU 디코드, 없으면 CPU(numpy) 자동 폴백;
  `--cpu` 강제, `SLLM_GPU_DTYPE=bf16`로 공유 GPU에서도 device-resident(고속) 활성
- **vLLM 스타일 진단** — 시작 배너(버전/아키텍처/백엔드/가중치/캐시), 요청 통계
  (`prompt/out`, `tokens/s`, `finish`), `--log-level`/`SLLM_LOG_LEVEL`
- **레시피 기반 설정** — 모델 정보는 `recipes/*.yaml`, 클러스터/서빙 공통은
  `config.env`(노드 IP, 포트, 이미지…); 프리시던스 CLI > 레시피 > config.env
- **TP2 듀얼노드** — 2노드 텐서 병렬 계획/워커 기동(클러스터 실행 마일스톤 게이트)
- **자체 CUDA 커널** — attention/재귀/희소/FP8/fused 커널을 자체 작성,
  Dense GEMM(cuBLAS)과 통신(NCCL)은 툴체인 재사용

## 기술 설명

- **디렉터리**: `ref/`(모델 numpy 오라클), `serving/`(엔진·HTTP·LLM CLI·토크나이저),
  `runtime/`(샘플러·메모리 배치·스케줄러·스펙 디코드), `loaders/`(fp8/fp4 로더·샤딩),
  `kernels/`(CUDA), `tp/`(텐서 병렬), `recipes/`(모델 정의)
- **엔진별 구성**: qwen3_5 = GDN+full-attn 증분 디코드; qwen4_exp = HC+Gating+QSA+MoE+PLE
  (numpy 파이프라인 + MTP 스펙); deepseek_v4 = MLA+압축+인덱서+MoE+DSPark 스펙
  디코드; 표준 = GQA+RoPE 증분/GPU 디코드
- **검증 방식**: torch 없이 순수 numpy 오라클 — greedy 정체성(spec 디코드 출력 ==
  plain greedy) 및 실 체크포인트 부분 패리티, bf16 노이즈 플로어를 기준으로 삼음
- **설계 문서**: [`docs/design/01-architecture.md`](docs/design/01-architecture.md)
  등 설계/감사/로드맵 01–10 참조

---

## 설치

요구사항: Python 3.10+, `numpy`/`pyyaml`/`regex`/`jinja2`(버전은
`requirements.txt`에 고정), 컨테이너 운영에는 Docker + NVIDIA 컨테이너 툴킷,
GPU 커널 빌드는 CUDA 13(nvcc). GitHub에서:

```bash
git clone https://github.com/skfine24/sLLM.git && cd sLLM
```

### 단일 노드 (SOLO)

```bash
./build.sh                 # 컨테이너 이미지 (sllm-node:latest, pinned deps, 배포 전용 소스)
# 또는
./build.sh --native        # 노드 venv 설치(동일 고정 deps + GPU 커널 빌드)

# per-node config.env: 노드 IP/pair/포트 확인 및 수정
```

### 듀얼 노드 (TP2)

모든 노드는 **동일한 컨테이너**를 사용합니다.

```bash
# head(192.168.0.250)와 worker(192.168.0.231) 각각에서 동일하게
./build.sh -tp2            # 듀얼노드용 빌드(이미지는 동일)

# config.env(양쪽 노드): SLLM_HEAD_IP/WORKER_IP, PAIR IP, SLLM_PAIR_IFACE(각 노드 NIC)
# 모델 가중치는 양쪽 모두 $HOME/models/<model> (paths.local_dir)
```

빌드 확인: `python -m serving.main --version` → `sllm 0.1.0 (…rev)`.

---

## 구동 방법

### 계획 (GPU/가중치 없이 안전)

```bash
./sllm recipes/Qwen3.8-27B-FP8.yaml --mode plan
```

### 원샷 생성

```bash
./sllm recipes/Qwen2.5-Coder-0.5B.yaml --mode run --chat "hi" --max-new 16
```

### 서빙 (OpenAI API)

```bash
./sllm recipes/Qwen2.5-Coder-0.5B.yaml --mode serve        # :8002
curl http://127.0.0.1:8002/v1/models
curl http://127.0.0.1:8002/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

`./sllm <recipe> [--tp 1|2] [--mode plan|run|serve] [--port P] [--host H]
[--chat TEXT] [--max-new N] [--cpu] [--dry]` — 프리시던스: CLI > 레시피
`defaults:` > `config.env` > 내장.

### 트러블슈팅용 옵션

- GPU: 기본 AUTO — 시작 시 `GPU decode enabled (CUDA devices visible: N)` 또는
  `GPU unavailable -> CPU (numpy) decode` 로그로 확인
- `--cpu`(강제 CPU) / `SLLM_USE_GPU=0` / `SLLM_GPU_DTYPE=bf16`(공유 GPU에서
  device-resident 가속)
- `--log-level DEBUG` — 토큰 단위 디버그

---

## 라이선스

원본 코드는 MIT 라이선스입니다 (`LICENSE`). 일부 분석용 벤더드 소스
(`ref/hf_sources/`, `oracle/upstream/` — Qwen/DeepSeek/transformers/sglang)와
모델 가중치는 각각의 라이선스를 따르며 배포 이미지에는 포함되지 않습니다
(자세한 내용은 `THIRD_PARTY_NOTICES.md`).
