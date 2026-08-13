#!/bin/bash
# Orquestra uma medicao: arma o ctl_san no dl380, dispara o inject_san no
# no injetor assim que "ARMADO" aparece, para minimizar o atraso entre arme
# e primeiro pacote (que o cronometro do ctl_san inclui, subestimando o
# throughput real).
#
# uso: ./run_measurement.sh <n_samples> <n_points> <porta-fpga>
set -euo pipefail
N_SAMPLES="${1:-4000003}"
N_POINTS="${2:-7}"
FPGA_PORT="${3:-62781}"
INJ_HOST=root@10.100.100.1
LOG=/root/ctl_med.log

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
rm -f "$LOG"
setsid nohup /root/ctl_san /root/san_net.xclbin 10.100.100.1 50000 \
    "$FPGA_PORT" "$N_POINTS" "$N_SAMPLES" 24576 > "$LOG" 2>&1 < /dev/null &

for i in $(seq 1 60); do
    if grep -q "ARMADO" "$LOG" 2>/dev/null; then break; fi
    if grep -q "ERRO" "$LOG" 2>/dev/null; then cat "$LOG"; exit 1; fi
    sleep 0.5
done
grep -q "ARMADO" "$LOG" || { echo "nao armou a tempo"; cat "$LOG"; exit 1; }

ssh -o BatchMode=yes "$INJ_HOST" \
    "/root/inject_san 10.100.100.50 $FPGA_PORT $N_SAMPLES $N_POINTS 140 50000"

for i in $(seq 1 60); do
    if grep -qE "SAN_NET_|TIMEOUT|FALHA" "$LOG" 2>/dev/null; then break; fi
    sleep 0.5
done
cat "$LOG"
