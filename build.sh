#!/usr/bin/env bash
# sLLM deployment build.
#
#   ./build.sh                 build the sllm-node container image (SOLO /
#                              single-node use). Pinned runtime deps in
#                              requirements.txt; dev/validation resources are
#                              excluded via .dockerignore + the Dockerfile's
#                              explicit COPY list.
#   ./build.sh -tp2            DUAL-NODE (TP2) build. The produced image is
#                              identical (all nodes run the same container);
#                              run the SAME `./build.sh -tp2` on the worker so
#                              both nodes carry the identical image.
#   ./build.sh --native        install into a venv on THIS node (same pinned
#                              deps + GPU kernel build) for head/worker runs
#   ./build.sh --image TAG     override the image tag (SLLM_IMAGE env too)
#
# Uses config.env via env_source.sh; real environment variables win.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env_source.sh"
sllm_load_env "$HERE/config.env"
PYBIN="${PYTHON:-python3}"
IMG="${SLLM_IMAGE:-sllm-node:latest}"

NATIVE=0
TP2=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --native) NATIVE=1; shift ;;
    -tp2|--tp2) TP2=1; shift ;;
    --image) IMG="${2:?--image needs a tag}"; shift 2 ;;
    -h|--help)
      sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "sllm: unknown arg: $1" >&2; echo "  (see ./build.sh --help)" >&2; exit 1 ;;
  esac
done

if [[ "$NATIVE" == 1 ]]; then
  VENV="${SLLM_VENV:-$HOME/dgsvenv}"
  "$PYBIN" -m venv "$VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  python -m pip install -U pip
  python -m pip install -r "$HERE/requirements.txt"
  (cd "$HERE/kernels/cuda" && bash build.sh)
  python -c "import serving.version as v; print('sllm', v.version_string())"
  echo "built (native): $VENV (activate with: source $VENV/bin/activate)"
  if [[ "$TP2" == 1 ]]; then
    echo "dual-node: repeat the same native install on the worker."
  fi
else
  docker build -f "$HERE/deploy/Dockerfile" -t "$IMG" "$HERE"
  if [[ "$TP2" == 1 ]]; then
    echo "built (DUAL-NODE): $IMG -- all nodes share this image;"
    echo "  run the identical: cd <repo> && ./build.sh -tp2   (on the worker)"
  else
    echo "built (SOLO): $IMG (run with: ./sllm <recipe> --mode serve)"
  fi
fi
