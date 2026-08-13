#!/bin/bash
# Orquestra uma medicao: arma o ctl_san no dl380, aquece o ARP, dispara o
# inject_san no no injetor assim que "ARMADO" aparece (minimiza o atraso que
# o cronometro do ctl_san inclui), e limpa o processo armado se qualquer
# passo falhar no meio — um ctl_san orfao segura o contexto do dispositivo e
# faz a PROXIMA tentativa falhar com "failed to open cu context", mascarando
# a causa real (visto em 2026-08-13).
#
# uso: ./run_measurement.sh <n_samples> <n_points> <porta-fpga> [taxa-Gbit/s]
#
# taxa-Gbit/s: PACEIE sempre. 8 e 12 Gbit/s confirmados bit-exatos e
# reproduziveis; o "ritmo natural" do sendto sem pace variou entre corridas
# (12,6 a 17,2 Gbit/s) e a variante mais rapida travou o networklayer.
set -uo pipefail
N_SAMPLES="${1:-4000003}"
N_POINTS="${2:-7}"
FPGA_PORT="${3:-62781}"
TAXA="${4:-8}"
INJ_HOST=root@10.100.100.1
LOG=/root/ctl_med.log

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
rm -f "$LOG"
setsid nohup /root/ctl_san /root/san_net.xclbin 10.100.100.1 50000 \
    "$FPGA_PORT" "$N_POINTS" "$N_SAMPLES" 24576 > "$LOG" 2>&1 < /dev/null &
CTL_PID=$!

limpa() {
    if kill -0 "$CTL_PID" 2>/dev/null; then
        echo "limpando ctl_san orfao (pid $CTL_PID)" >&2
        kill "$CTL_PID" 2>/dev/null
    fi
}
trap limpa EXIT

armado=0
for i in $(seq 1 30); do
    if grep -q "ARMADO" "$LOG" 2>/dev/null; then armado=1; break; fi
    if grep -q "ERRO" "$LOG" 2>/dev/null; then break; fi
    sleep 0.5
done
if [ "$armado" != 1 ]; then echo "nao armou a tempo"; cat "$LOG"; exit 1; fi

# Aquece o ARP antes do UDP: `ip neigh` mostrou a entrada da FPGA em STALE
# apos qualquer hiato (edicao de script, git, etc). Ping isolado depois de um
# hiato perde ~1 em 4-5; um stream continuo de 40 nao perdeu nenhum. 3 pings
# forcam confirmacao de alcancabilidade (o ICMP echo-reply confirma o vizinho
# dos dois lados) antes da rajada de UDP de verdade. Best-effort: nao aborta
# a medicao se o ping falhar por algum motivo transitorio de SSH.
ssh -o BatchMode=yes -o ConnectTimeout=10 "$INJ_HOST" \
    'ping -c3 -i0.2 -q 10.100.100.50 >/dev/null 2>&1' || true

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$INJ_HOST" \
    "/root/inject_san 10.100.100.50 $FPGA_PORT $N_SAMPLES $N_POINTS 140 50000 $TAXA"
then
    echo "injetor falhou"; exit 1
fi

# ctl_san espera ate' 120s pelo kernel; a espera aqui tem de cobrir isso com
# folga (bug anterior: so' esperava 30s e cortava a leitura do resultado).
for i in $(seq 1 45); do
    if grep -qE "SAN_NET_|TIMEOUT|FALHA" "$LOG" 2>/dev/null; then break; fi
    sleep 3
done
cat "$LOG"
grep -q "SAN_NET_BIT_EXATO" "$LOG"
