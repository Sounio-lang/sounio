#!/usr/bin/env bash
set -uo pipefail
cd /workspace/sounio-mut
ulimit -s 1048576
SRC=self-hosted/compiler/lean_single.sio
echo "[1] canonical(89af1391) compiles slot-reuse lean -> stage1"
./bin/souc "$SRC" /tmp/sr1 > /tmp/sr1.log 2>&1; echo "  rc=$? errs=$(grep -cE '^error:|^E[0-9]+' /tmp/sr1.log) md5=$(md5sum /tmp/sr1|cut -c1-12)"; chmod +x /tmp/sr1 2>/dev/null
[[ -s /tmp/sr1 ]] || { echo "  STAGE1 FAILED"; tail -6 /tmp/sr1.log; exit 1; }
echo "[2] stage1 self-compiles -> stage2"
/tmp/sr1 "$SRC" /tmp/sr2 > /tmp/sr2.log 2>&1; echo "  rc=$? errs=$(grep -cE '^error:|^E[0-9]+' /tmp/sr2.log) md5=$(md5sum /tmp/sr2|cut -c1-12)"; chmod +x /tmp/sr2 2>/dev/null
echo "[3] stage2 self-compiles -> stage3"
/tmp/sr2 "$SRC" /tmp/sr3 > /tmp/sr3.log 2>&1; echo "  rc=$? errs=$(grep -cE '^error:|^E[0-9]+' /tmp/sr3.log) md5=$(md5sum /tmp/sr3|cut -c1-12)"; chmod +x /tmp/sr3 2>/dev/null
if cmp -s /tmp/sr2 /tmp/sr3; then echo "  [ORACLE] CONVERGED stage2==stage3 -> slot-reuse souc self-reproduces (strong correctness signal)"; else echo "  [ORACLE] *** NOT CONVERGED *** stage2 != stage3 -> MISCOMPILE, will revert"; exit 2; fi
echo "[4] compile main.sio with slot-reuse souc -> count remaining huge frames"
T0=$(date +%s)
/tmp/sr2 self-hosted/compiler/main.sio /tmp/mc_sr.elf > /tmp/mc_sr.log 2>&1
echo "  rc=$? elapsed=$(($(date +%s)-T0))s errs=$(grep -cE '^error:|^E[0-9]+' /tmp/mc_sr.log)"
echo "  huge-frame warnings: BEFORE=86  AFTER=$(grep -ac 'stack frame too large' /tmp/mc_sr.log)"
echo "  largest remaining frame: $(grep -a 'stack frame too large' /tmp/mc_sr.log | grep -oE '\([0-9]+' | tr -d '(' | sort -rn | head -1 | awk '{printf "%.1f MB", $1/1048576}')"
[[ -s /tmp/mc_sr.elf ]] || { echo "  NO mc.elf"; exit 3; }
chmod +x /tmp/mc_sr.elf
echo "[5] --check examples/hello.sio x20 (THE crash test)"
crash=0; ok=0; other=0
for i in $(seq 1 20); do timeout 30 /tmp/mc_sr.elf --check examples/hello.sio >/dev/null 2>&1; r=$?; if [[ $r -eq 139 ]]; then crash=$((crash+1)); elif [[ $r -eq 0 ]]; then ok=$((ok+1)); else other=$((other+1)); fi; done
echo "  crash(139)=$crash  ok(0)=$ok  other=$((20-crash-ok))"
