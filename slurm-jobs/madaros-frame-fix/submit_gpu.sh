#!/usr/bin/env bash
# Validate the native_v2_core_begin_function_from_ir_into dynamic-frame fix (commit 7fa3c3524).
#
# GPU-NODE VARIANT of submit.sh. The original targeted the cpu-ops partition, but on
# 2026-06-16 the cpuops-t560-proxmox node was de-registered (sinfo cpu-ops -> 0 nodes).
# This variant retargets to the gpu-orangefs path that slurm-jobs/kretikos already uses
# on this cluster (account=plruntime, qos=gpuorangefs). It is a pure CPU build — no GPU
# gres is requested; it just uses the idle GPU node's CPUs.
#
# Build inputs (self-hosted/, stdlib/, lean_single seed) come from a CLEAN `git archive main`
# so the dirty /workspace/sounio working tree (another agent's WIP in module_frontend.sio etc.)
# cannot skew the binary. The reproducer is a test input only and is taken from the working tree.
#
# This job:
#   1. Builds Madaros from FIXED main source via bin/souc-lean-single-x86_64.
#   2. Compiles reproducer_madaros_codegen_2026-06-16g.sio at N=1/2/4/5 emit_cube_assignment
#      calls and runs each binary.
#   3. Verifies every variant prints: pass=1 trail=5 conflict=1
#
# Before the fix: N=2 gave wrong trail, N=5 SIGSEGVed (hard-coded 512B frame, 64 slots).
# After the fix: align16(func.reg_count * 8) gives each function its real frame.
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpu-orangefs}"
ACCOUNT="${ACCOUNT:-plruntime}"
QOS="${QOS:-gpuorangefs}"
NODELIST="${NODELIST:-gpuorangefs-r770-proxmox}"
WORKER_POD="${WORKER_POD:-slurm-pilot-worker-gpuorangefs-zj2l4}"   # the pod ON ${NODELIST}
WORKER_CTR="${WORKER_CTR:-slurmd}"
JOB_MEM="${JOB_MEM:-20G}"
JOB_CPUS="${JOB_CPUS:-4}"
JOB_TIME="${JOB_TIME:-00:30:00}"
RUN_ID="${RUN_ID:-madaros-frame-fix-$(date -u +%Y%m%dT%H%M%S)}"

REPRO="examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"
test -f "${REPO}/${REPRO}" || { echo "missing reproducer ${REPO}/${REPRO}" >&2; exit 1; }
git -C "${REPO}" cat-file -e main:bin/souc-lean-single-x86_64 2>/dev/null \
  || { echo "lean_single seed not in main" >&2; exit 1; }

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
${KUBECTL} -n "${NS}" get pod "${WORKER_POD}" >/dev/null \
  || { echo "worker pod ${WORKER_POD} not found" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}   worker pod: ${WORKER_POD} (node ${NODELIST})"

TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH="/tmp/${RUN_ID}.sbatch"
echo "packaging CLEAN main build inputs + working-tree reproducer -> ${TARBALL}"
# Build inputs from committed main (faithful to fix); reproducer from working tree.
STAGE="/tmp/${RUN_ID}.stage"
rm -rf "${STAGE}"; mkdir -p "${STAGE}"
git -C "${REPO}" archive --format=tar main self-hosted stdlib bin/souc-lean-single-x86_64 \
  | tar -x -C "${STAGE}"
mkdir -p "${STAGE}/$(dirname "${REPRO}")"
cp "${REPO}/${REPRO}" "${STAGE}/${REPRO}"
tar -C "${STAGE}" -czf "${TARBALL}" self-hosted stdlib bin/souc-lean-single-x86_64 "${REPRO}"
rm -rf "${STAGE}"
echo "payload: $(stat -c%s "${TARBALL}" | awk '{printf "%.1f MB\n",$1/1048576}')"

echo "staging payload to ${WORKER_POD}:/tmp ..."
${KUBECTL} -n "${NS}" cp "${TARBALL}" "${WORKER_POD}:/tmp/${RUN_ID}.tgz" -c "${WORKER_CTR}"

cat > "${SBATCH}" <<EOF
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

ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
REPO="\${ROOT}/repo"
RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"

echo "host=\$(hostname) job=\${SLURM_JOB_ID:-manual} mem=${JOB_MEM} cpus=${JOB_CPUS}" | tee "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"

tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/souc-lean-single-x86_64"
cd "\${REPO}"
export SOUNIO_STDLIB_PATH="\${REPO}/stdlib"

# ── Step 1: rebuild Madaros from fixed main source ─────────────────────────────
echo "[build] Madaros from fixed source ..." | tee "\${RES}/build.log"
/usr/bin/timeout 1500 ./bin/souc-lean-single-x86_64 \
  self-hosted/compiler/main.sio \
  "\${ROOT}/madaros.elf" >> "\${RES}/build.log" 2>&1
BRC=\$?
echo "MADAROS_BUILD_RC=\${BRC}" | tee -a "\${RES}/build.log"
if [ \${BRC} -ne 0 ] || [ ! -s "\${ROOT}/madaros.elf" ]; then
  echo "RESULT: MADAROS_BUILD_FAILED rc=\${BRC}" | tee "\${RES}/SUMMARY.txt"
  tail -20 "\${RES}/build.log" | sed 's/^/  /' | tee -a "\${RES}/SUMMARY.txt"
  exit 0
fi
chmod +x "\${ROOT}/madaros.elf"
SIZE=\$(stat -c%s "\${ROOT}/madaros.elf" 2>/dev/null || echo "?")
echo "madaros.elf size=\${SIZE}" | tee -a "\${RES}/build.log"

# ── Helper: generate an N-call reproducer and run it ──────────────────────────
run_repro() {
  local N="\$1"
  local SRC="\${ROOT}/repro_n\${N}.sio"
  BASE="\${REPO}/${REPRO}"
  if [ "\${N}" -eq 1 ]; then
    cp "\${BASE}" "\${SRC}"
  else
    CALL='    emit_cube_assignment(0, 0, 0, k)'
    CALLS="\$(for i in \$(seq 1 "\${N}"); do echo "\${CALL}"; done)"
    sed "s|    emit_cube_assignment(0, 0, 0, k)|\${CALLS}|" "\${BASE}" > "\${SRC}"
  fi

  local ELF="\${ROOT}/repro_n\${N}.elf"
  echo "" | tee -a "\${RES}/SUMMARY.txt"
  echo "=== N=\${N}: compile ===>" | tee -a "\${RES}/SUMMARY.txt"
  /usr/bin/timeout 120 "\${ROOT}/madaros.elf" build "\${SRC}" -o "\${ELF}" \
    >> "\${RES}/compile_n\${N}.log" 2>&1
  CRC=\$?
  echo "  compile_rc=\${CRC}" | tee -a "\${RES}/SUMMARY.txt"
  if [ \${CRC} -ne 0 ] || [ ! -s "\${ELF}" ]; then
    echo "  COMPILE_FAILED" | tee -a "\${RES}/SUMMARY.txt"
    tail -5 "\${RES}/compile_n\${N}.log" | sed 's/^/  /' | tee -a "\${RES}/SUMMARY.txt"
    return
  fi
  chmod +x "\${ELF}"
  echo "=== N=\${N}: run ===>" | tee -a "\${RES}/SUMMARY.txt"
  OUTPUT=\$(/usr/bin/timeout 10 "\${ELF}" 2>&1 || echo "rc=\$?")
  echo "  output: \${OUTPUT}" | tee -a "\${RES}/SUMMARY.txt"
  if echo "\${OUTPUT}" | grep -q "pass=1 trail=5 conflict=1"; then
    echo "  VERDICT: PASS" | tee -a "\${RES}/SUMMARY.txt"
  else
    echo "  VERDICT: FAIL (expected 'pass=1 trail=5 conflict=1')" | tee -a "\${RES}/SUMMARY.txt"
  fi
}

echo "=== Frame-fix validation (gpu-orangefs variant) ===" | tee "\${RES}/SUMMARY.txt"
echo "Madaros size: \${SIZE}" | tee -a "\${RES}/SUMMARY.txt"

run_repro 1
run_repro 2
run_repro 4
run_repro 5

echo "" | tee -a "\${RES}/SUMMARY.txt"
echo "Done." | tee -a "\${RES}/SUMMARY.txt"
echo "RESULT_DIR=\${RES}"
EOF

${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
