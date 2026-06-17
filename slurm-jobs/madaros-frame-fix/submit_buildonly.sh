#!/usr/bin/env bash
# Minimal: JIT builds REF's main.sio -> M.elf (no variants, no assertions).
set -euo pipefail
REPO="${REPO:-/workspace/sounio}"; REF="${REF:-fix/cmp-operands-bool}"; NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"; ACCOUNT="${ACCOUNT:-omics}"; QOS="${QOS:-cpuops}"
NODELIST="${NODELIST:-cpuops-t560-proxmox}"; WORKER_POD="${WORKER_POD:-slurm-pilot-worker-cpuops-hqr6v}"; WORKER_CTR="${WORKER_CTR:-slurmd}"
JOB_MEM="${JOB_MEM:-24G}"; JOB_CPUS="${JOB_CPUS:-4}"; JOB_TIME="${JOB_TIME:-00:40:00}"
RUN_ID="${RUN_ID:-madaros-buildonly-$(date -u +%Y%m%dT%H%M%S)}"; JIT="${JIT:-$REPO/bin/souc-linux-x86_64}"
LOGIN_POD="$(${KUBECTL} -n "$NS" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
STAGE="/tmp/$RUN_ID.stage"; rm -rf "$STAGE"; mkdir -p "$STAGE"
git -C "$REPO" archive --format=tar "$REF" self-hosted stdlib | tar -x -C "$STAGE"; cp "$JIT" "$STAGE/jit"
TARBALL="/tmp/$RUN_ID.tgz"; tar -C "$STAGE" -czf "$TARBALL" self-hosted stdlib jit; rm -rf "$STAGE"
${KUBECTL} -n "$NS" cp "$TARBALL" "$WORKER_POD:/tmp/$RUN_ID.tgz" -c "$WORKER_CTR"
SBATCH="/tmp/$RUN_ID.sbatch"
cat > "$SBATCH" <<SB
#!/usr/bin/env bash
#SBATCH -J $RUN_ID
#SBATCH -p $PARTITION
#SBATCH -A $ACCOUNT
#SBATCH --qos=$QOS
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c $JOB_CPUS
#SBATCH --mem=$JOB_MEM
#SBATCH --time=$JOB_TIME
#SBATCH -w $NODELIST
#SBATCH -o /tmp/$RUN_ID.slurmout
#SBATCH -e /tmp/$RUN_ID.slurmout
set -uo pipefail
ROOT=/tmp/$RUN_ID-\${SLURM_JOB_ID:-manual}; RES=/tmp/$RUN_ID.result; rm -rf "\$ROOT" "\$RES"; mkdir -p "\$ROOT" "\$RES"
tar -xzf /tmp/$RUN_ID.tgz -C "\$ROOT"; chmod +x "\$ROOT/jit"; cd "\$ROOT"; export SOUNIO_STDLIB_PATH="\$ROOT/stdlib"
/usr/bin/timeout 1800 ./jit self-hosted/compiler/main.sio "\$ROOT/M.elf" > "\$RES/build.log" 2>&1
echo "rc=\$? hard_errors=\$(grep -c '^error:' "\$RES/build.log" 2>/dev/null) size=\$(stat -c%s "\$ROOT/M.elf" 2>/dev/null||echo 0)" > "\$RES/SUMMARY.txt"
grep '^error:' "\$RES/build.log" | head -8 >> "\$RES/SUMMARY.txt" || true
[ -s "\$ROOT/M.elf" ] && chmod +x "\$ROOT/M.elf"
SB
${KUBECTL} -n "$NS" cp "$SBATCH" "$LOGIN_POD:/tmp/$RUN_ID.sbatch"
${KUBECTL} -n "$NS" exec "$LOGIN_POD" -- bash -lc "sbatch --parsable /tmp/$RUN_ID.sbatch"
echo "RUN_ID=$RUN_ID"; rm -f "$TARBALL" "$SBATCH"
