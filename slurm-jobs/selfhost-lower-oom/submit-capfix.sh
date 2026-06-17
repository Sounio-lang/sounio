#!/usr/bin/env bash
# Verify the IR_MAX_FUNCS lift (1400->8192) + named-call resolver index on a SLURM
# cpu-ops worker (bounded --mem, so an OOM kills the JOB, never the workspace pod —
# see the 2026-06-08 incident). Two phases, both on the --native-compile path that
# actually exercises load_multimodule_ir -> the iterative BFS merge ->
# ir_module_resolve_named_calls (the code these commits change; --native-v2-compile
# uses the source-concat bridge instead, so it would NOT cover this path):
#
#   PHASE 1  NO-REGRESSION (landing precondition): compile+run the v1..v7 multi-module
#            witnesses with `gen.elf --native-compile`. Their exits are fixed by program
#            semantics (42,87,42,42,42,42,30); v4 = transitive (BFS depth), v5 = diamond
#            (visited-set dedup). Any wrong/missing exit = a regression to normal compiles.
#   PHASE 2  SELF-HOST memory experiment: gen.elf --native-compile main.sio under an RSS
#            poller. Open question this settles: does body-lowering churn scale with total
#            fn_count? "Merged IR: N" should now exceed 1400 (real cross-module merge);
#            RUN_RC=137 at ~16G = evidence heap-indirect IrModule is mandatory (a RESULT,
#            not a failure); a clean finish + gen2.elf = the cap lift sufficed.
set -euo pipefail

WORKTREE="${WORKTREE:-/tmp/kw-demote}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-16G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:35:00}"
RUN_ID="${RUN_ID:-capfix-$(date -u +%Y%m%dT%H%M%S)}"

test -x "${WORKTREE}/bin/souc" || { echo "no bin/souc in ${WORKTREE}" >&2; exit 1; }
test -d "${WORKTREE}/tests/native_v2_multimodule_gate/witnesses" || { echo "no multimodule witnesses in ${WORKTREE}" >&2; exit 1; }

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)"
WORKER_CTR="${WORKER_CTR:-slurmd}"
test -n "${WORKER_POD}" || { echo "no cpuops worker pod" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}   worker pod: ${WORKER_POD}"

TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH="/tmp/${RUN_ID}.sbatch"
echo "packaging payload -> ${TARBALL}"
tar -C "${WORKTREE}" -czf "${TARBALL}" self-hosted stdlib bin/souc tests/native_v2_multimodule_gate
echo "staging payload to worker pod /tmp ..."
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
#SBATCH -o /tmp/${RUN_ID}.slurmout
#SBATCH -e /tmp/${RUN_ID}.slurmout
set -uo pipefail

ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
REPO="\${ROOT}/repo"
RES="/tmp/${RUN_ID}.result"          # stable path for fetch.sh
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"

echo "host=\$(hostname) job=\${SLURM_JOB_ID:-manual} mem=${JOB_MEM} cpus=${JOB_CPUS}" | tee "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"

tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/souc"
cd "\${REPO}"

# ---- BUILD gen-N (also the faithful parse/typecheck test of the edited source) ----
echo "[build] gen-N from main.sio ..." | tee "\${RES}/build.log"
/usr/bin/timeout 900 ./bin/souc self-hosted/compiler/main.sio "\${ROOT}/gen.elf" >> "\${RES}/build.log" 2>&1
BRC=\$?
echo "BUILD_RC=\${BRC}" | tee -a "\${RES}/build.log"
if [ \${BRC} -ne 0 ] || [ ! -s "\${ROOT}/gen.elf" ]; then
  echo "BUILD FAILED rc=\${BRC} (edited source did not compile under gen N-1)" | tee "\${RES}/SUMMARY.txt"
  tail -30 "\${RES}/build.log" | tee -a "\${RES}/SUMMARY.txt"
  scontrol update JobId=\${SLURM_JOB_ID} Comment="capfix build_fail rc=\${BRC}" 2>/dev/null || true
  exit 0
fi
chmod +x "\${ROOT}/gen.elf"

# ---- PHASE 1: NO-REGRESSION (--native-compile on multi-module programs) ----
# flatA: generated flat sibling-import program (guaranteed-simple main -> exercises the
# compact ELF writer directly, independent of the subdir-import resolution question).
echo "[noreg] --native-compile multi-module programs ..." | tee "\${RES}/NOREG.txt"
mkdir -p tests/_noreg_flatA
printf 'use lib::{add5}\nfn main() -> i64 { add5(37) }\n' > tests/_noreg_flatA/prog.sio
printf 'fn add5(x: i64) -> i64 { x + 5 }\n' > tests/_noreg_flatA/lib.sio
declare -A WANT=( [_noreg_flatA]=42 [v1_basic]=42 [v2_two_fns]=87 [v3_chain]=42 [v4_transitive]=42 [v5_diamond]=42 [v6_struct]=42 [v7_collision_twin]=30 )
NR_PASS=0; NR_TOT=0
for w in _noreg_flatA v1_basic v2_two_fns v3_chain v4_transitive v5_diamond v6_struct v7_collision_twin; do
  if [ "\${w}" = "_noreg_flatA" ]; then WD="tests/_noreg_flatA"; else WD="tests/native_v2_multimodule_gate/witnesses/\${w}"; fi
  [ -d "\${WD}" ] || { echo "MISS  \${w} (no witness dir)" | tee -a "\${RES}/NOREG.txt"; continue; }
  NR_TOT=\$((NR_TOT+1))
  ST="\$(mktemp -d /tmp/nr_\${w}.XXXXXX)"
  cp -a "\${WD}/." "\${ST}/"
  rm -f "\${ST}/out.elf"
  ( cd "\${ST}" && /usr/bin/timeout 120 "\${ROOT}/gen.elf" --native-compile prog.sio -o out.elf > compile.log 2>&1 )
  CRC=\$?
  if [ -s "\${ST}/out.elf" ]; then
    chmod +x "\${ST}/out.elf"
    ( cd "\${ST}" && ./out.elf >/dev/null 2>&1 ); RC=\$?
    if [ "\${RC}" = "\${WANT[\$w]}" ]; then
      echo "PASS  \${w} (--native-compile ELF exit \${RC})" | tee -a "\${RES}/NOREG.txt"; NR_PASS=\$((NR_PASS+1))
    else
      echo "FAIL  \${w} (exit \${RC} want \${WANT[\$w]}) -- cross-module MISCOMPILE" | tee -a "\${RES}/NOREG.txt"
      cp "\${ST}/compile.log" "\${RES}/noreg_\${w}_compile.log" 2>/dev/null || true
    fi
  else
    echo "FAIL  \${w} (no ELF; compile rc=\${CRC}) -- FALSE-REJECT of valid multi-module" | tee -a "\${RES}/NOREG.txt"
    cp "\${ST}/compile.log" "\${RES}/noreg_\${w}_compile.log" 2>/dev/null || true
  fi
  rm -rf "\${ST}"
done
echo "NOREG_RESULT=\${NR_PASS}/\${NR_TOT}" | tee -a "\${RES}/NOREG.txt"

# ---- PHASE 2: SELF-HOST memory experiment ----
if [ "${SKIP_SELFHOST:-0}" = "1" ]; then
  echo "SKIP_SELFHOST=1 -> phase 2 skipped (no-regression-only run)" | tee "\${RES}/SUMMARY.txt"
  cat "\${RES}/NOREG.txt" | tee -a "\${RES}/SUMMARY.txt"
  echo "BUILD_RC=\${BRC}" | tee -a "\${RES}/SUMMARY.txt"
  scontrol update JobId=\${SLURM_JOB_ID} Comment="capfix-noreg-only \${NR_PASS}/\${NR_TOT}" 2>/dev/null || true
  exit 0
fi
echo "[selfhost] gen-N --native-compile main.sio (RSS poller) ..."
export SOUNIO_FN_INSTR_TRACE=1
export SOUNIO_LOWER_ORDERED_TRACE=1
"\${ROOT}/gen.elf" --native-compile self-hosted/compiler/main.sio -o "\${ROOT}/gen2.elf" \
  > "\${RES}/run.log" 2>&1 &
GPID=\$!
PEAK=0
: > "\${RES}/rss.log"
while kill -0 \${GPID} 2>/dev/null; do
  HWM=\$(awk '/^VmHWM:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  if [ -n "\${HWM:-}" ]; then
    NF=\$(grep -c '^FNINSTR ' "\${RES}/run.log" 2>/dev/null || echo 0)
    echo "\$(date +%s) hwm_kb=\${HWM} fninstr_lines=\${NF}" >> "\${RES}/rss.log"
    [ "\${HWM}" -gt "\${PEAK}" ] 2>/dev/null && PEAK=\${HWM}
  fi
  sleep 0.3
done
wait \${GPID}; RRC=\$?

NF=\$(grep -c '^FNINSTR ' "\${RES}/run.log" 2>/dev/null || echo 0)
MERGED=\$(grep -E '^Merged IR: [0-9]+ functions' "\${RES}/run.log" 2>/dev/null | tail -1)
GEN2OK="no"; [ -s "\${ROOT}/gen2.elf" ] && GEN2OK="yes"
{
  echo "==== CAPFIX VERIFICATION (IR_MAX_FUNCS 1400->8192 + resolver index) ===="
  echo "BUILD_RC=\${BRC}  (gen.elf built from edited source -> parse/typecheck OK)"
  echo "NOREG_RESULT=\${NR_PASS}/\${NR_TOT}  (--native-compile cross-module no-regression)"
  echo "--- no-regression detail ---"; cat "\${RES}/NOREG.txt"
  echo "--- self-host experiment ---"
  echo "RUN_RC=\${RRC}   (137=SIGKILL/OOM, 139=SIGSEGV, 0=clean)"
  echo "PEAK_HWM_KB=\${PEAK} PEAK_HWM_GB=\$((PEAK/1024/1024))"
  echo "MERGED_IR=\${MERGED:-<none reached>}   (was capped at 1400 before this change)"
  echo "FNINSTR_LINES=\${NF}   (body functions lowered)"
  echo "GEN2_ELF_PRODUCED=\${GEN2OK}"
  echo "--- last 25 run.log lines ---"
  tail -25 "\${RES}/run.log" 2>/dev/null
} | tee "\${RES}/SUMMARY.txt"
scontrol update JobId=\${SLURM_JOB_ID} Comment="capfix noreg=\${NR_PASS}/\${NR_TOT} selfhost_rc=\${RRC} peakGB=\$((PEAK/1024/1024)) merged=\${MERGED:-na}" 2>/dev/null || true
echo "RESULT_DIR=\${RES}"
EOF

REMOTE_SBATCH="/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:${REMOTE_SBATCH}"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
