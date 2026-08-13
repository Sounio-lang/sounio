#!/bin/bash
# Roda a medicao (com aquecimento de ARP) 10 vezes seguidas, para confirmar
# que a correcao da intermitencia (ARP indo STALE) segura de verdade.
# Se alguma corrida falhar, recupera a placa (systemctl restart
# u250-vnx.service + reprogramacao) e CONTINUA para as proximas — o objetivo
# e' o placar completo, nao parar no primeiro problema.
set -uo pipefail
N=4000003
NP=7
TAXA=8
PORTA_BASE=63100
XCLBIN=/root/san_net.xclbin

ok=0
falhas=0
for i in $(seq 1 10); do
    PORTA=$((PORTA_BASE + i))
    echo "=== corrida $i/10 (porta $PORTA) ==="
    if /root/run_measurement.sh "$N" "$NP" "$PORTA" "$TAXA" > "/root/confirm_$i.log" 2>&1; then
        echo "  OK"
        ok=$((ok+1))
    else
        echo "  FALHOU — ultimas linhas:"
        tail -8 "/root/confirm_$i.log"
        falhas=$((falhas+1))
        echo "  recuperando..."
        systemctl restart u250-vnx.service
        sleep 2
        source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
        xrt-smi program --device 0000:d8:00.1 --user "$XCLBIN" >/dev/null 2>&1
        sleep 1
    fi
done

echo "=== RESUMO: $ok/10 OK, $falhas/10 falharam ==="
