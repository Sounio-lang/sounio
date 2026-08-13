#!/bin/bash
# Varre a taxa de injecao, do lento pro rapido, parando no primeiro sinal de
# perda ou de kernel que nao completa. Cada ponto usa a MESMA coorte (4M
# amostras) para o metodo ficar comparavel entre pontos.
#
# Ao primeiro fracasso: registra o ultimo ponto bom, tenta a recuperacao
# validada (systemctl restart u250-vnx.service + reprogramacao) e PARA — nao
# segue a varredura as cegas depois de um fracasso.
#
# uso: ./sweep_rate.sh [porta-base]
set -uo pipefail
PORTA_BASE="${1:-63000}"
INJ_HOST=root@10.100.100.1
N=4000003
NP=7
LOG=/root/ctl_sweep.log
XCLBIN=/root/san_net.xclbin

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

TAXAS=(5 10 20 30 40 50 60 70 80 90)
ULTIMA_BOA=""

for i in "${!TAXAS[@]}"; do
    TX="${TAXAS[$i]}"
    PORTA=$((PORTA_BASE + i))
    echo "=== ponto $((i+1))/${#TAXAS[@]}: alvo ${TX} Gbit/s (porta $PORTA) ==="

    rm -f "$LOG"
    setsid nohup /root/ctl_san "$XCLBIN" 10.100.100.1 50000 "$PORTA" "$NP" "$N" 24576 \
        > "$LOG" 2>&1 < /dev/null &

    armado=0
    for j in $(seq 1 30); do
        grep -q ARMADO "$LOG" 2>/dev/null && { armado=1; break; }
        grep -q ERRO "$LOG" 2>/dev/null && break
        sleep 0.5
    done
    if [ "$armado" != 1 ]; then
        echo "  nao armou — parando aqui"; cat "$LOG"; break
    fi

    ssh -o BatchMode=yes "$INJ_HOST" \
        "/root/inject_san 10.100.100.50 $PORTA $N $NP 140 50000 $TX"

    ok=0
    for j in $(seq 1 40); do
        grep -qE "SAN_NET_BIT_EXATO" "$LOG" 2>/dev/null && { ok=1; break; }
        grep -qE "TIMEOUT|FALHA|DIVERGENTE" "$LOG" 2>/dev/null && break
        sleep 3
    done

    if [ "$ok" = 1 ]; then
        echo "  OK bit-exato em ${TX} Gbit/s"
        ULTIMA_BOA="$TX"
    else
        echo "  FALHOU em ${TX} Gbit/s — ultimo log:"
        tail -8 "$LOG"
        echo "=== recuperando ==="
        systemctl restart u250-vnx.service
        sleep 2
        xrt-smi program --device 0000:d8:00.1 --user "$XCLBIN" 2>&1 | tail -3
        break
    fi
done

echo "=== RESUMO: ultima taxa confirmada bit-exata: ${ULTIMA_BOA:-nenhuma} Gbit/s ==="
