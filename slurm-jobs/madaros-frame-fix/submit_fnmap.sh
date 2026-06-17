#!/usr/bin/env bash
# Symbol-mapping diagnostic: run the lean_single seed on main.sio and capture its
# "stack frame too large (BYTES) in fn#N <NAME> at <loc>" warnings (lean_single's
# warning prints the function name + source loc, unlike the JIT's name-less one).
# This maps the oversized-frame functions to source names — no source modification.
set -euo pipefail
REPO="${REPO:-/workspace/sounio}"; NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"; ACCOUNT="${ACCOUNT:-omics}"; QOS="${QOS:-cpuops}"
NODELIST="${NODELIST:-cpuops-t560-proxmox}"; WORKER_POD="${WORKER_POD:-slurm-pilot-worker-cpuops-hqr6v}"; WORKER_CTR="${WORKER_CTR:-slurmd}"
JOB_MEM="${JOB_MEM:-20G}"; JOB_CPUS="${JOB_CPUS:-4}"; JOB_TIME="${JOB_TIME:-00:40:00}"
RUN_ID="${RUN_ID:-madaros-fnmap-$(date -u +%Y%m%dT%H%M%S)}"; REF="${REF:-fix/cmp-operands-bool}"
LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
STAGE="/tmp/${RUN_ID}.stage"; rm -rf "$STAGE"; mkdir -p "$STAGE"
git -C "$REPO" archive --format=tar "$REF" self-hosted stdlib | tar -x -C "$STAGE"
cp "$REPO/bin/souc-lean-single-x86_64" "$STAGE/seed"
TARBALL="/tmp/${RUN_ID}.tgz"; tar -C "$STAGE" -czf "$TARBALL" self-hosted stdlib seed; rm -rf "$STAGE"
${KUBECTL} -n "$NS" cp "$TARBALL" "$WORKER_POD:/tmp/${RUN_ID}.tgz" -c "$WORKER_CTR"
SBATCH="/tmp/${RUN_ID}.sbatch"
cat > "$SBATCH" <<SB
#!/usr/bin/env bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c ${JOB_CPUS}
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH -w ${NODELIST}
#SBATCH -o /tmp/${RUN_ID}.slurmout
#SBATCH -e /tmp/${RUN_ID}.slurmout
set -uo pipefail
ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"; RES="/tmp/${RUN_ID}.result"; rm -rf "\$ROOT" "\$RES"; mkdir -p "\$ROOT" "\$RES"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\$ROOT"; chmod +x "\$ROOT/seed"; cd "\$ROOT"; export SOUNIO_STDLIB_PATH="\$ROOT/stdlib"
/usr/bin/timeout 1500 ./seed self-hosted/compiler/main.sio "\$ROOT/out.elf" > "\$RES/full.log" 2>&1
echo "rc=\$? size=\$(stat -c%s "\$ROOT/out.elf" 2>/dev/null||echo 0)" > "\$RES/SUMMARY.txt"
grep -i "stack frame too large" "\$RES/full.log" | tr '\n' ' ' | sed 's/in fn#/\nin fn#/g' > "\$RES/frames.txt" 2>/dev/null || true
echo "frame warnings: \$(grep -c 'stack frame too large' "\$RES/full.log" 2>/dev/null)" >> "\$RES/SUMMARY.txt"
SB
${KUBECTL} -n "$NS" cp "$SBATCH" "$LOGIN_POD:/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "$NS" exec "$LOGIN_POD" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch"
echo "RUN_ID=${RUN_ID}"; rm -f "$TARBALL" "$SBATCH"
