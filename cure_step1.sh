#!/usr/bin/env bash
set -uo pipefail
cd /workspace/sounio-mut
ulimit -s 1048576
SRC=self-hosted/compiler/lean_single.sio
echo "[bootstrap new canonical fixed point with the enhanced warning]"
./bin/souc "$SRC" /tmp/c1 > /tmp/c1.log 2>&1; echo " stage1 rc=$? md5=$(md5sum /tmp/c1|cut -c1-12)"; chmod +x /tmp/c1
/tmp/c1 "$SRC" /tmp/c2 > /tmp/c2.log 2>&1; echo " stage2 rc=$? md5=$(md5sum /tmp/c2|cut -c1-12)"; chmod +x /tmp/c2
/tmp/c2 "$SRC" /tmp/c3 > /tmp/c3.log 2>&1; echo " stage3 rc=$? md5=$(md5sum /tmp/c3|cut -c1-12)"; chmod +x /tmp/c3
if cmp -s /tmp/c2 /tmp/c3; then echo " CONVERGED md5=$(md5sum /tmp/c2|cut -c1-12)"; cp /tmp/c2 bin/souc; chmod +x bin/souc; echo " installed as bin/souc"; else echo " NOT CONVERGED — abort"; exit 1; fi
echo "[canonical gate]"; bash scripts/ci/canonical_compiler_gate.sh 2>&1 | tail -1
echo "[compile main.sio with new canonical -> named frame warnings]"
T0=$(date +%s)
./bin/souc self-hosted/compiler/main.sio /tmp/mc_named.elf > /tmp/named.log 2>&1
echo " main.sio compile rc=$? elapsed=$(($(date +%s)-T0))s"
echo "[named huge-frame functions, sorted by size]"
grep -a 'stack frame too large' /tmp/named.log | tr -cd '[:print:]\n' | sed -E 's/warning: stack frame too large \(([0-9]+) bytes\) in (fn#[0-9]+) at (.*) - consider.*/\1 \2 \3/' | sort -rn | head -30
echo "[total]"; grep -ac 'stack frame too large' /tmp/named.log
