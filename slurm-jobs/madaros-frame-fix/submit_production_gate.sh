#!/usr/bin/env bash
# Validate the production Madaros launcher contract on Slurm.
#
# This job intentionally uses bin/madaros, not the raw Madaros ELF. The launcher
# owns the production stack policy (`ulimit -s unlimited`) and resolves the
# checked-in prebuilt bin/madaros-linux-x86_64. The raw no-ulimit job remains a
# separate diagnostic for frame-size work.
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-20G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:20:00}"
RUN_ID="${RUN_ID:-madaros-production-gate-$(date -u +%Y%m%dT%H%M%S)}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"

test -x "${REPO}/bin/madaros" || { echo "no launcher at ${REPO}/bin/madaros" >&2; exit 1; }
test -x "${REPO}/bin/madaros-linux-x86_64" || { echo "no prebuilt at ${REPO}/bin/madaros-linux-x86_64" >&2; exit 1; }

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}"

${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${ORANGEFS_TMP}'"

TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH="/tmp/${RUN_ID}.sbatch"
SNAP="/tmp/${RUN_ID}-snap"
echo "packaging payload from git HEAD -> ${TARBALL}"
rm -rf "${SNAP}" && mkdir -p "${SNAP}/examples/erdos"
git -C "${REPO}" archive HEAD -- stdlib bin/madaros bin/madaros-linux-x86_64 \
  | tar x -C "${SNAP}"
cp "${REPO}/examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio" \
   "${SNAP}/examples/erdos/"
tar -C "${SNAP}" -czf "${TARBALL}" .
rm -rf "${SNAP}"

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
echo "initial_stack_limit=\$(ulimit -s) kB" | tee -a "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"

echo "extracting payload from ${ORANGEFS_TMP}/${RUN_ID}.tgz ..."
tar -xzf "${ORANGEFS_TMP}/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/madaros" "\${REPO}/bin/madaros-linux-x86_64"
cd "\${REPO}"
export SOUNIO_STDLIB_PATH="\${REPO}/stdlib"

run_repro() {
  local N="\$1"
  local SRC="\${ROOT}/repro_n\${N}.sio"
  local BASE="\${REPO}/examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"
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
  echo "=== N=\${N}: public launcher build ===>" | tee -a "\${RES}/SUMMARY.txt"
  /usr/bin/timeout 120 "\${REPO}/bin/madaros" build "\${SRC}" -o "\${ELF}" \
    >> "\${RES}/compile_n\${N}.log" 2>&1
  CRC=\$?
  echo "  compile_rc=\${CRC}" | tee -a "\${RES}/SUMMARY.txt"
  if [ \${CRC} -ne 0 ] || [ ! -s "\${ELF}" ]; then
    echo "  COMPILE_FAILED" | tee -a "\${RES}/SUMMARY.txt"
    tail -12 "\${RES}/compile_n\${N}.log" | sed 's/^/  /' | tee -a "\${RES}/SUMMARY.txt"
    return 1
  fi
  chmod +x "\${ELF}"
  echo "=== N=\${N}: run ===>" | tee -a "\${RES}/SUMMARY.txt"
  OUTPUT=\$(/usr/bin/timeout 10 "\${ELF}" 2>&1 || echo "rc=\$?")
  echo "  output: \${OUTPUT}" | tee -a "\${RES}/SUMMARY.txt"
  if echo "\${OUTPUT}" | grep -q "pass=1 trail=5 conflict=1"; then
    echo "  VERDICT: PASS" | tee -a "\${RES}/SUMMARY.txt"
    return 0
  fi
  echo "  VERDICT: FAIL (expected 'pass=1 trail=5 conflict=1')" | tee -a "\${RES}/SUMMARY.txt"
  return 1
}

echo "=== Madaros production launcher Slurm gate ===" | tee "\${RES}/SUMMARY.txt"
"\${REPO}/bin/madaros" info | tee -a "\${RES}/SUMMARY.txt"
echo "Default stack before launcher: \$(ulimit -s) kB" | tee -a "\${RES}/SUMMARY.txt"

FAIL=0
run_repro 1 || FAIL=1
run_repro 2 || FAIL=1
run_repro 4 || FAIL=1
run_repro 5 || FAIL=1

echo "" | tee -a "\${RES}/SUMMARY.txt"
if [ \${FAIL} -eq 0 ]; then
  echo "RESULT: PASS production launcher compiles and runs frame reproducer variants" | tee -a "\${RES}/SUMMARY.txt"
else
  echo "RESULT: FAIL production launcher failed one or more variants" | tee -a "\${RES}/SUMMARY.txt"
fi
echo "RESULT_DIR=\${RES}"
exit \${FAIL}
EOF

REMOTE_SBATCH="${ORANGEFS_TMP}/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:${REMOTE_SBATCH}"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
