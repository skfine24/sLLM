#!/usr/bin/env bash
# sllm-node container entrypoint.
# Modes (SLLM_MODE env):
#   kernel  : run the GPU kernel smoke test (default if no args)
#   serve   : start the HTTP serving stub (standard engine) on $SLLM_PORT
#   batch   : run continuous-batch one-shot over $SLLM_BATCH (;-separated)
#   oneshot : complete a single prompt $SLLM_PROMPT
# Otherwise run the given command as-is.
set -e
PY=/usr/local/bin/python
# config.env = SLLM common defaults (real env vars still win)
. /sllm/env_source.sh
sllm_load_env /sllm/config.env
# pair NIC: recipe env may leave NCCL_SOCKET_IFNAME empty on purpose -- the
# per-node value comes from config.env (SLLM_PAIR_IFACE) instead
if [ -n "${SLLM_PAIR_IFACE:-}" ]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$SLLM_PAIR_IFACE}"
fi
case "${SLLM_MODE:-}" in
  kernel)
    exec "$PY" /sllm/kernels/smoke.py
    ;;
  serve)
    exec "$PY" -m serving.serve_standard --model-dir "${MODEL_DIR:-/models}" \
         --serve --port "${SLLM_PORT:-8002}"
    ;;
  batch)
    IFS=';' read -ra P <<< "${SLLM_BATCH:-}"
    exec "$PY" -m serving.serve_standard --model-dir "${MODEL_DIR:-/models}" \
         --batch "${P[@]}" --max-new "${SLLM_MAX_NEW:-8}"
    ;;
  oneshot)
    exec "$PY" -m serving.serve_standard --model-dir "${MODEL_DIR:-/models}" \
         --prompt "${SLLM_PROMPT:-hi}" --max-new "${SLLM_MAX_NEW:-8}"
    ;;
  *)
    if [ "$#" -gt 0 ]; then exec "$@"; fi
    exec "$PY" /sllm/kernels/smoke.py
    ;;
esac
