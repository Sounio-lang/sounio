#!/usr/bin/env bash
# Validate the machine_ir.sio block-aliasing fix (copy-modify-writeback).
#
# This job:
#   1. Builds Madaros from HEAD (includes machine_ir.sio fix) using lean_single.
#   2. Runs the existing frame-fix reproducer (N=1/2/4/5) to confirm no regression.
#   3. Runs a println(var) smoke test — if the block aliasing was causing wrong
#      instruction emission, this should now produce correct output (or at least
#      not SIGSEGV for reasons attributable to aliasing).
#   4. Runs a struct-field-write round-trip smoke test:
#      fn main() -> i64 { var x: i64 = 41; x = x + 1; return x }  → expect 0 exit
#
# The fix changes machine_function_add_instr_mut, _emit_legal_byval, _add_instr_byval
# to use copy-modify-writeback instead of aliased field writes on blocks[0].
#
# MUST be submitted via SLURM (Madaros rebuild can exhaust stack on the k8s pod).
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
RUN_ID="${RUN_ID:-madaros-machine-ir-fix-$(date -u +%Y%m%dT%H%M%S)}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"

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

# Generate sbatch script
cat > "${SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --job-name=${RUN_ID}
#SBATCH --output=${ORANGEFS_TMP}/${RUN_ID}.slurmout
#SBATCH --mem=${JOB_MEM}
#SBATCH --cpus-per-task=${JOB_CPUS}
#SBATCH --time=${JOB_TIME}
set -euo pipefail
ulimit -s unlimited

ROOT="/tmp/${RUN_ID}"
RES="${ORANGEFS_TMP}/${RUN_ID}.result"
mkdir -p "\${ROOT}" "\${RES}"

echo "extracting payload..."
tar -xzf "${ORANGEFS_TMP}/${RUN_ID}.tgz" -C "\${ROOT}"
chmod +x "\${ROOT}/bin/souc-lean-single-x86_64"
cd "\${ROOT}"

# ── Step 1: rebuild Madaros ────────────────────────────────────────────────────
echo "[build] Madaros from HEAD (machine_ir aliasing fix)..." | tee "\${RES}/build.log"
/usr/bin/timeout 1200 ./bin/souc-lean-single-x86_64 \
  self-hosted/compiler/main.sio "\${ROOT}/madaros.elf" >> "\${RES}/build.log" 2>&1
BRC=\$?
echo "MADAROS_BUILD_RC=\${BRC}" | tee -a "\${RES}/build.log"
if [ \${BRC} -ne 0 ] || [ ! -s "\${ROOT}/madaros.elf" ]; then
  echo "RESULT: MADAROS_BUILD_FAILED rc=\${BRC}" | tee "\${RES}/SUMMARY.txt"
  tail -20 "\${RES}/build.log" | tee -a "\${RES}/SUMMARY.txt"
  exit 0
fi
chmod +x "\${ROOT}/madaros.elf"
SIZE=\$(stat -c%s "\${ROOT}/madaros.elf")
echo "madaros.elf size=\${SIZE}" | tee -a "\${RES}/build.log"

echo "=== machine_ir block-aliasing fix validation ===" | tee "\${RES}/SUMMARY.txt"
echo "Commit: \$(cd "\${ROOT}" && git log --oneline -1 2>/dev/null || echo unknown)" | tee -a "\${RES}/SUMMARY.txt"
echo "Madaros size: \${SIZE}" | tee -a "\${RES}/SUMMARY.txt"

# ── Step 2: frame-fix regression (N=1/2/4/5) ──────────────────────────────────
run_repro() {
  local N="\$1"
  local SRC="\${ROOT}/repro_n\${N}.sio"
  BASE="\${ROOT}/examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"
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
  echo "=== frame-fix N=\${N} ===" | tee -a "\${RES}/SUMMARY.txt"
  /usr/bin/timeout 60 "\${ROOT}/madaros.elf" "\${SRC}" -o "\${ELF}" >> "\${RES}/compile_n\${N}.log" 2>&1
  CRC=\$?
  echo "  compile_rc=\${CRC}" | tee -a "\${RES}/SUMMARY.txt"
  if [ \${CRC} -ne 0 ] || [ ! -s "\${ELF}" ]; then
    echo "  COMPILE_FAILED" | tee -a "\${RES}/SUMMARY.txt"; return
  fi
  chmod +x "\${ELF}"
  OUTPUT=\$(/usr/bin/timeout 10 "\${ELF}" 2>&1 || echo "exit=\$?")
  echo "  output: \${OUTPUT}" | tee -a "\${RES}/SUMMARY.txt"
  if echo "\${OUTPUT}" | grep -q "pass=1 trail=5 conflict=1"; then
    echo "  VERDICT: PASS" | tee -a "\${RES}/SUMMARY.txt"
  else
    echo "  VERDICT: FAIL" | tee -a "\${RES}/SUMMARY.txt"
  fi
}
run_repro 1; run_repro 2; run_repro 4; run_repro 5

# ── Step 3: println(var) smoke test ───────────────────────────────────────────
echo "" | tee -a "\${RES}/SUMMARY.txt"
echo "=== println(var) smoke ===" | tee -a "\${RES}/SUMMARY.txt"
cat > "\${ROOT}/println_var.sio" <<'SIOEOF'
fn main() -> i64 {
    var x: i64 = 42
    println(x)
    let msg = "hello"
    println(msg)
    return 0
}
SIOEOF
/usr/bin/timeout 30 "\${ROOT}/madaros.elf" "\${ROOT}/println_var.sio" -o "\${ROOT}/println_var.elf" >> "\${RES}/println_compile.log" 2>&1
PRC=\$?
echo "  compile_rc=\${PRC}" | tee -a "\${RES}/SUMMARY.txt"
if [ \${PRC} -eq 0 ] && [ -s "\${ROOT}/println_var.elf" ]; then
  chmod +x "\${ROOT}/println_var.elf"
  POUT=\$(/usr/bin/timeout 5 "\${ROOT}/println_var.elf" 2>&1; echo "exit=\$?")
  echo "  output: \${POUT}" | tee -a "\${RES}/SUMMARY.txt"
  if echo "\${POUT}" | grep -q "42" && echo "\${POUT}" | grep -q "hello" && echo "\${POUT}" | grep -q "exit=0"; then
    echo "  VERDICT: PASS (i64 var + string var work)" | tee -a "\${RES}/SUMMARY.txt"
  else
    echo "  VERDICT: FAIL or partial (check output above)" | tee -a "\${RES}/SUMMARY.txt"
  fi
else
  echo "  VERDICT: COMPILE_FAILED" | tee -a "\${RES}/SUMMARY.txt"
  tail -5 "\${RES}/println_compile.log" | sed 's/^/  /' | tee -a "\${RES}/SUMMARY.txt"
fi

# ── Step 4: struct round-trip smoke ───────────────────────────────────────────
echo "" | tee -a "\${RES}/SUMMARY.txt"
echo "=== struct round-trip smoke ===" | tee -a "\${RES}/SUMMARY.txt"
cat > "\${ROOT}/struct_rt.sio" <<'SIOEOF'
struct Point { x: i64, y: i64 }
fn move_point(p: Point, dx: i64, dy: i64) -> Point {
    var q = p
    q.x = q.x + dx
    q.y = q.y + dy
    q
}
fn main() -> i64 {
    let p = Point { x: 10, y: 20 }
    let q = move_point(p, 5, 3)
    if q.x != 15 { return 1 }
    if q.y != 23 { return 2 }
    return 0
}
SIOEOF
/usr/bin/timeout 30 "\${ROOT}/madaros.elf" "\${ROOT}/struct_rt.sio" -o "\${ROOT}/struct_rt.elf" >> "\${RES}/struct_rt_compile.log" 2>&1
SRC2=\$?
echo "  compile_rc=\${SRC2}" | tee -a "\${RES}/SUMMARY.txt"
if [ \${SRC2} -eq 0 ] && [ -s "\${ROOT}/struct_rt.elf" ]; then
  chmod +x "\${ROOT}/struct_rt.elf"
  /usr/bin/timeout 5 "\${ROOT}/struct_rt.elf" 2>&1; SRET=\$?
  echo "  exit_code=\${SRET}" | tee -a "\${RES}/SUMMARY.txt"
  [ \${SRET} -eq 0 ] && echo "  VERDICT: PASS" || echo "  VERDICT: FAIL (exit \${SRET})" | tee -a "\${RES}/SUMMARY.txt"
else
  echo "  VERDICT: COMPILE_FAILED" | tee -a "\${RES}/SUMMARY.txt"
fi

echo "" | tee -a "\${RES}/SUMMARY.txt"
echo "Done." | tee -a "\${RES}/SUMMARY.txt"
echo "RESULT_DIR=\${RES}"
EOF

REMOTE_SBATCH="${ORANGEFS_TMP}/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:${REMOTE_SBATCH}"
echo "submitting..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
