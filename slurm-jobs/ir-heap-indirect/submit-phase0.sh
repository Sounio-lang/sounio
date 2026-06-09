#!/usr/bin/env bash
# Phase 0 (B/C scope): instrumented baseline of the front-half OOM. Build mc with the
# A+D-fixed bin/souc, then run `mc --native-compile main.sio` under an RSS poller that
# correlates VmHWM with the per-import PHASE0_MERGE_STEP markers (added to the merge loop
# in module_frontend.sio). Answers: how many 526 MB per-import allocations accumulate, the
# merged fn_count, and the RSS shape (monotonic climb = no-free allocator confirmed).
# Bounded 16 GB worker — the run is EXPECTED to OOM (137) partway; how far it gets IS the data.
set -euo pipefail
WORKTREE="${WORKTREE:-/workspace/sounio-ir}"
NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"; ACCOUNT="${ACCOUNT:-omics}"; QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-16G}"; JOB_CPUS="${JOB_CPUS:-2}"; JOB_TIME="${JOB_TIME:-00:40:00}"
RUN_ID="${RUN_ID:-phase0-$(date -u +%Y%m%dT%H%M%S)}"
test -x "${WORKTREE}/bin/souc" || { echo "no bin/souc" >&2; exit 1; }
LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)"
WORKER_CTR="${WORKER_CTR:-slurmd}"
test -n "${WORKER_POD}" || { echo "no cpuops worker pod" >&2; exit 1; }
echo "login: ${LOGIN_POD}  worker: ${WORKER_POD}"
TARBALL="/tmp/${RUN_ID}.tgz"; SBATCH="/tmp/${RUN_ID}.sbatch"
tar -C "${WORKTREE}" -czf "${TARBALL}" self-hosted stdlib bin/souc bin/souc-linux-x86_64
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
ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"; REPO="\${ROOT}/repo"; RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"
echo "host=\$(hostname) mem=${JOB_MEM}" | tee "\${RES}/SUMMARY.txt"; free -g | head -2 | tee -a "\${RES}/SUMMARY.txt"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"; chmod +x "\${REPO}/bin/souc" "\${REPO}/bin/souc-linux-x86_64"; cd "\${REPO}"

echo "[build] mc = bin/souc main.sio (instrumented merge loop) ..." | tee -a "\${RES}/SUMMARY.txt"
/usr/bin/timeout 1200 ./bin/souc self-hosted/compiler/main.sio "\${ROOT}/mc.elf" > "\${RES}/mc_build.log" 2>&1; BRC=\$?
echo "MC_BUILD_RC=\${BRC}" | tee -a "\${RES}/SUMMARY.txt"
if [ \${BRC} -ne 0 ] || [ ! -s "\${ROOT}/mc.elf" ]; then echo "mc build failed" | tee -a "\${RES}/SUMMARY.txt"; tail -20 "\${RES}/mc_build.log" | tee -a "\${RES}/SUMMARY.txt"; exit 0; fi
chmod +x "\${ROOT}/mc.elf"

echo "[run] mc --native-compile main.sio under RSS poller ..." | tee -a "\${RES}/SUMMARY.txt"
"\${ROOT}/mc.elf" --native-compile self-hosted/compiler/main.sio -o "\${ROOT}/gen.elf" > "\${RES}/run.log" 2>&1 &
GPID=\$!; PEAK=0
: > "\${RES}/rss_vs_step.log"
while kill -0 \${GPID} 2>/dev/null; do
  HWM=\$(awk '/^VmHWM:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  STEP=\$(grep -c '^PHASE0_MERGE_STEP ' "\${RES}/run.log" 2>/dev/null || echo 0)
  if [ -n "\${HWM:-}" ]; then
    echo "\$(date +%s) hwm_kb=\${HWM} hwm_mb=\$((HWM/1024)) merge_steps=\${STEP}" >> "\${RES}/rss_vs_step.log"
    [ "\${HWM}" -gt "\${PEAK}" ] 2>/dev/null && PEAK=\${HWM}
  fi
  sleep 0.2
done
wait \${GPID}; RRC=\$?
{
  echo "=== PHASE 0 RESULT ==="
  echo "NATIVE_COMPILE_RC=\${RRC}  (137=SIGKILL/OOM, 139=SIGSEGV, 0=completed)"
  echo "PEAK_HWM_MB=\$((PEAK/1024))  (worker cap = 16384 MB)"
  LOADED=\$(grep -oE 'of [0-9]+' "\${RES}/run.log" | head -1 | awk '{print \$2}')
  STEPS=\$(grep -c '^PHASE0_MERGE_STEP ' "\${RES}/run.log" 2>/dev/null || echo 0)
  echo "IMPORTS_LOADED=\${LOADED:-?}   MERGE_STEPS_REACHED=\${STEPS}  (vs loaded => died after step \${STEPS})"
  echo "LAST_MERGE_STEP=\$(grep '^PHASE0_MERGE_STEP ' "\${RES}/run.log" 2>/dev/null | tail -1)"
  echo "MERGED_IR_LINE=\$(grep -E '^Merged IR:' "\${RES}/run.log" 2>/dev/null | head -1) \$(grep -A1 -E '^Merged IR:' "\${RES}/run.log" 2>/dev/null | tail -1)"
  echo "--- RSS vs merge-step (per-step delta ~= 526 MB if per-import leak; monotonic = no free) ---"
  awk 'NR==1||NR%10==0{print}' "\${RES}/rss_vs_step.log" | tail -25
  echo "--- per-step RSS (first hwm at each new merge_steps value) ---"
  awk '{if(\$NF!=last){print; last=\$NF}}' "\${RES}/rss_vs_step.log" | tail -30
  echo "--- last 15 run.log lines ---"; tail -15 "\${RES}/run.log"
} | tee -a "\${RES}/SUMMARY.txt"
scontrol update JobId=\${SLURM_JOB_ID} Comment="phase0 rc=\${RRC} peakMB=\$((PEAK/1024)) steps=\${STEPS}" 2>/dev/null || true
echo "RESULT_DIR=\${RES}"
EOF
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '/tmp/${RUN_ID}.sbatch'"
echo "RUN_ID=${RUN_ID}"; echo "${RUN_ID}" > /tmp/phase0.last_run_id
rm -f "${TARBALL}" "${SBATCH}"
