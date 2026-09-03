#!/usr/bin/env bash
# Validate the native_v2_core_begin_function_from_ir_into dynamic-frame fix.
#
# This job:
#   1. Builds Madaros from the FIXED source using lean_single (bin/souc-lean-single-x86_64)
#   2. Compiles reproducer_madaros_codegen_2026-06-16g.sio with N=1/N=2/N=4/N=5 emit_cube_assignment
#      calls before the main loop, and runs each binary.
#   3. Verifies that all variants produce: pass=1 trail=5 conflict=1
#
# Before the fix, N=2 gave wrong trail and N=5 crashed (SIGSEGV). The fix makes the frame
# size dynamic: align16(func.reg_count * 8) instead of the hard-coded 512.
#
# Staging: tarball goes to OrangeFS (/orangefs/training/tmp/) which is visible from both
# the login pod and the cpuops-t560-proxmox compute node.
#
# MUST be submitted via SLURM to avoid crashing the k8s pod (see 2026-06-08 incident).
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-20G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:30:00}"
RUN_ID="${RUN_ID:-madaros-frame-fix-$(date -u +%Y%m%dT%H%M%S)}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"

test -x "${REPO}/bin/souc-lean-single-x86_64" || { echo "no lean_single at ${REPO}/bin/souc-lean-single-x86_64" >&2; exit 1; }

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}"

# Create OrangeFS tmp dir on the login pod
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${ORANGEFS_TMP}'"

# Package payload from committed HEAD (exclude WIP working-tree changes)
# The reproducer is untracked so we append it separately.
TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH="/tmp/${RUN_ID}.sbatch"
SNAP="/tmp/${RUN_ID}-snap"
echo "packaging payload from git HEAD -> ${TARBALL}"
rm -rf "${SNAP}" && mkdir -p "${SNAP}/examples/erdos"
git -C "${REPO}" archive HEAD -- self-hosted stdlib bin/souc-lean-single-x86_64 \
  | tar x -C "${SNAP}"
cp "${REPO}/examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio" \
   "${SNAP}/examples/erdos/"
tar -C "${SNAP}" -czf "${TARBALL}" .
rm -rf "${SNAP}"

# Stage tarball to login pod then move to OrangeFS
echo "staging payload to OrangeFS ${ORANGEFS_TMP}/${RUN_ID}.tgz ..."
${KUBECTL} -n "${NS}" cp "${TARBALL}" "${LOGIN_POD}:/tmp/${RUN_ID}.tgz"
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "mv '/tmp/${RUN_ID}.tgz' '${ORANGEFS_TMP}/${RUN_ID}.tgz'"
echo "staged."

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
#SBATCH -o ${ORANGEFS_TMP}/${RUN_ID}.slurmout
#SBATCH -e ${ORANGEFS_TMP}/${RUN_ID}.slurmout
set -uo pipefail

ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
REPO="\${ROOT}/repo"
RES="${ORANGEFS_TMP}/${RUN_ID}.result"
rm -rf "\${ROOT}"; mkdir -p "\${REPO}" "\${RES}"

echo "host=\$(hostname) job=\${SLURM_JOB_ID:-manual} mem=${JOB_MEM} cpus=${JOB_CPUS}" | tee "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"

echo "extracting payload from ${ORANGEFS_TMP}/${RUN_ID}.tgz ..."
tar -xzf "${ORANGEFS_TMP}/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/souc-lean-single-x86_64"
cd "\${REPO}"

# ── Step 1: rebuild Madaros from fixed source ──────────────────────────────────
echo "[build] Madaros from fixed source ..." | tee "\${RES}/build.log"
/usr/bin/timeout 1200 ./bin/souc-lean-single-x86_64 \
  self-hosted/compiler/main.sio \
  "\${ROOT}/madaros.elf" >> "\${RES}/build.log" 2>&1
BRC=\$?
echo "MADAROS_BUILD_RC=\${BRC}" | tee -a "\${RES}/build.log"
if [ \${BRC} -ne 0 ] || [ ! -s "\${ROOT}/madaros.elf" ]; then
  echo "RESULT: MADAROS_BUILD_FAILED rc=\${BRC}" | tee "\${RES}/SUMMARY.txt"
  tail -20 "\${RES}/build.log" | tee -a "\${RES}/SUMMARY.txt"
  exit 0
fi
chmod +x "\${ROOT}/madaros.elf"
SIZE=\$(stat -c%s "\${ROOT}/madaros.elf" 2>/dev/null || echo "?")
echo "madaros.elf size=\${SIZE}" | tee -a "\${RES}/build.log"

# ── Helper: generate an N-call reproducer and run it ──────────────────────────
run_repro() {
  local N="\$1"
  local SRC="\${ROOT}/repro_n\${N}.sio"
  BASE="\${REPO}/examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"
  if [ "\${N}" -eq 1 ]; then
    cp "\${BASE}" "\${SRC}"
  else
    python3 -c "
import sys
src = open(sys.argv[1]).read()
single = '    emit_cube_assignment(0, 0, 0, k)'
calls = '\n'.join(['    emit_cube_assignment(0, 0, 0, k)'] * int(sys.argv[2]))
open(sys.argv[3], 'w').write(src.replace(single, calls, 1))
" "\${BASE}" "\${N}" "\${SRC}"
  fi

  local ELF="\${ROOT}/repro_n\${N}.elf"
  echo "" | tee -a "\${RES}/SUMMARY.txt"
  echo "=== N=\${N}: compile ===>" | tee -a "\${RES}/SUMMARY.txt"
  ulimit -s unlimited
  /usr/bin/timeout 60 "\${ROOT}/madaros.elf" "\${SRC}" -o "\${ELF}" \
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

echo "=== Frame-fix validation ===" | tee "\${RES}/SUMMARY.txt"
echo "Commit: \$(cd "\${REPO}" && git log --oneline -1 2>/dev/null || echo unknown)" | tee -a "\${RES}/SUMMARY.txt"
echo "Madaros size: \${SIZE}" | tee -a "\${RES}/SUMMARY.txt"

run_repro 1
run_repro 2
run_repro 4
run_repro 5

echo "" | tee -a "\${RES}/SUMMARY.txt"
echo "Done." | tee -a "\${RES}/SUMMARY.txt"
echo "RESULT_DIR=\${RES}"
EOF

REMOTE_SBATCH="${ORANGEFS_TMP}/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:${REMOTE_SBATCH}"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
