#!/usr/bin/env bash
# B3 probe: measured bandwidth/latency of the direct 10.100.25.x pair link
# (head 10.100.25.1 <-> worker 10.100.25.2). The TP2 all-reduce budget
# (docs/design/03, docs/design/08) is currently an ASSUMPTION; this measures.
#
# Usage:
#   head:    bash bench/probe_pair_link.sh server
#   worker:  PEER=10.100.25.1 bash bench/probe_pair_link.sh client
# Optional NCCL round (both nodes, needs nccl-tests + this build's env):
#   NCCL=1 PEER=... bash bench/probe_pair_link.sh client   # prints the command
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/env_source.sh"
sllm_load_env "$ROOT/config.env"   # SLLM_* common defaults (env vars win)
PEER="${PEER:-${SLLM_HEAD_PAIR_IP:-}}"    # client on worker -> probe toward head
IFACE="${IFACE:-${SLLM_PAIR_IFACE:-}}"    # pair NIC from config.env
ROLE="${1:-client}"

if [[ -n "$IFACE" ]]; then
  echo "[link] interface speed ($IFACE):"
  ethtool "$IFACE" 2>/dev/null | grep -i "speed\|duplex" || echo "  (ethtool unavailable)"
fi

if [[ "$ROLE" == "server" ]]; then
  echo "[link] iperf3 server on 0.0.0.0:5201 (Ctrl-C to stop)"
  exec iperf3 -s -p 5201
fi

[[ -n "$PEER" ]] || { echo "set PEER=<pair ip of the other node>"; exit 1; }
echo "[link] latency head<->${PEER}:"
ping -c 8 -W 1 "$PEER" | tail -n 2

echo "[link] tcp throughput (iperf3, 3 streams x 10 s):"
iperf3 -c "$PEER" -p 5201 -t 10 -P 3 | tail -n 2

if [[ "${NCCL:-0}" == "1" ]]; then
  cat <<'EOF'
[link] NCCL round (run nccl-tests on BOTH nodes; adjust paths):
  NCCL_SOCKET_IFNAME=<pair iface> NCCL_DEBUG=INFO \
  ./all_reduce_perf -b 8 -e 256M -f 2 -g 1 -t 1 \
      -H "10.100.25.1:1,10.100.25.2:1"
Record: latency at 4-64 KB messages (per-token TP2 all-reduce size class,
~10 KB fp32 per layer-step at hidden 2560) and sustained GB/s at 128 MB.
EOF
fi
echo "[link] done — paste output into log.md (replaces the ASSUMPTION row)"
