#!/usr/bin/env bash
# Diagnostic no-ulimit validation for raw Madaros frame-size work.
#
# This intentionally bypasses bin/madaros and invokes the freshly rebuilt raw
# ELF directly without raising the stack limit. Production users should be gated
# by submit_production_gate.sh, because bin/madaros owns the current stack
# policy. A failure here is a compiler-frame blocker, not automatically a
# production launcher failure.
#
# Changes being validated (commit 5b42e985b):
#   - emit_and_bool_rax_rbx / emit_or_bool_rax_rbx: use emit_byte_mut internally
#     (4.6 MB frame → ~256 KB)
#   - lower_instr + 70 lower_* / emit_binop_code / emit_float_binop_code:
#     converted to &! NativeCompiler with void return (20+ MB dispatch frame → ~0)
#
# Reproducer: N=1/2/4/5 emit_cube_assignment calls (N=5 SIGSEGVed without fix).
# Expected all: pass=1 trail=5 conflict=1
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-20G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:40:00}"
RUN_ID="${RUN_ID:-madaros-stack-fix-$(date -u +%Y%m%dT%H%M%S)}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"
WORKER_POD="${WORKER_POD:-}"

test -x "${REPO}/bin/souc-lean-single-x86_64" || { echo "no lean_single at ${REPO}/bin/souc-lean-single-x86_64" >&2; exit 1; }

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
git -C "${REPO}" archive HEAD -- self-hosted stdlib bin/souc-lean-single-x86_64 \
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
# Show the cluster default stack limit. This diagnostic must not raise it.
echo "stack_limit=\$(ulimit -s) kB" | tee -a "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"

echo "extracting payload from ${ORANGEFS_TMP}/${RUN_ID}.tgz ..."
tar -xzf "${ORANGEFS_TMP}/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/souc-lean-single-x86_64"
cd "\${REPO}"

# ── Step 1: rebuild Madaros from fixed source ──────────────────────────────────
echo "[build] Madaros from fixed source ..." | tee "\${RES}/build.log"
# lean_single (the bootstrap binary) was NOT compiled with our &! refactor, so it
# still has large-frame functions that need unlimited stack. Run in a subshell so
# the ulimit does NOT propagate to the run_repro step (the actual test).
( ulimit -s unlimited && /usr/bin/timeout 1800 ./bin/souc-lean-single-x86_64 \
  self-hosted/compiler/main.sio \
  "\${ROOT}/madaros.elf" ) >> "\${RES}/build.log" 2>&1
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

# ── Helper: generate N-call reproducer and run it ─────────────────────────────
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
  # NO ulimit -s unlimited — this is the key test
  /usr/bin/timeout 60 "\${ROOT}/madaros.elf" "\${SRC}" -o "\${ELF}" \
    >> "\${RES}/compile_n\${N}.log" 2>&1
  CRC=\$?
  echo "  compile_rc=\${CRC}" | tee -a "\${RES}/SUMMARY.txt"
  if [ \${CRC} -ne 0 ] || [ ! -s "\${ELF}" ]; then
    echo "  COMPILE_FAILED" | tee -a "\${RES}/SUMMARY.txt"
    tail -8 "\${RES}/compile_n\${N}.log" | sed 's/^/  /' | tee -a "\${RES}/SUMMARY.txt"
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

echo "=== Stack-fix validation (NO ulimit) ===" | tee "\${RES}/SUMMARY.txt"
echo "Commit: \$(cd "\${REPO}" && git log --oneline -1 2>/dev/null || echo unknown)" | tee -a "\${RES}/SUMMARY.txt"
echo "Madaros size: \${SIZE}" | tee -a "\${RES}/SUMMARY.txt"
echo "Default stack limit: \$(ulimit -s) kB" | tee -a "\${RES}/SUMMARY.txt"

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

if [ -n "${WORKER_POD}" ]; then
  echo "submitting via WORKER_POD=${WORKER_POD} ..."
  ${KUBECTL} -n "${NS}" exec "${WORKER_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
else
  echo "submitting ..."
  ${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
fi
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
