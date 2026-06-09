#!/usr/bin/env bash
# Phase 1 verification of the A1 single-accumulator refactor. One shared mc build, then:
#   A (primary): mc --native-compile main.sio under an RSS poller. SUCCESS = peak drops
#       from ~16 GB (job 2382) toward ~1-1.5 GB AND lowering COMPLETES with a sane
#       "Merged IR: N functions" (N ~= 6642 = the merge produced the whole program, a
#       correctness proxy) instead of OOMing in the recursive lowerer.
#   B (no-regression): native_v2_multimodule gate (must stay 27/27). The v2 path is NOT
#       the path I changed, but it shares module_frontend.sio — confirms no collateral break.
# Acceptance (advisor): "27/27 + peak dropped", not peak alone.
set -euo pipefail
WORKTREE="${WORKTREE:-/workspace/sounio-ir}"
NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"; ACCOUNT="${ACCOUNT:-omics}"; QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-16G}"; JOB_CPUS="${JOB_CPUS:-2}"; JOB_TIME="${JOB_TIME:-00:50:00}"
RUN_ID="${RUN_ID:-phase1-$(date -u +%Y%m%dT%H%M%S)}"
test -x "${WORKTREE}/bin/souc" || { echo "no bin/souc" >&2; exit 1; }
LOGIN_POD="${LOGIN_POD:-$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')}"
LOGIN_CTR="${LOGIN_CTR:-}"
LCFLAG=""; [ -n "${LOGIN_CTR}" ] && LCFLAG="-c ${LOGIN_CTR}"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)"
WORKER_CTR="${WORKER_CTR:-slurmd}"
test -n "${WORKER_POD}" || { echo "no worker" >&2; exit 1; }
echo "login: ${LOGIN_POD}  worker: ${WORKER_POD}"
TARBALL="/tmp/${RUN_ID}.tgz"; SBATCH="/tmp/${RUN_ID}.sbatch"
tar -C "${WORKTREE}" -czf "${TARBALL}" self-hosted stdlib bin/souc bin/souc-linux-x86_64 scripts tests
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
echo "host=\$(hostname) mem=${JOB_MEM}" | tee "\${RES}/SUMMARY.txt"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"; chmod +x "\${REPO}/bin/souc" "\${REPO}/bin/souc-linux-x86_64"; cd "\${REPO}"

echo "[build] mc = bin/souc main.sio (A1 single-accumulator) ..." | tee -a "\${RES}/SUMMARY.txt"
/usr/bin/timeout 1200 ./bin/souc self-hosted/compiler/main.sio "\${ROOT}/mc.elf" > "\${RES}/mc_build.log" 2>&1; BRC=\$?
echo "MC_BUILD_RC=\${BRC}" | tee -a "\${RES}/SUMMARY.txt"
if [ \${BRC} -ne 0 ] || [ ! -s "\${ROOT}/mc.elf" ]; then echo "mc build FAILED (A1 source error?)" | tee -a "\${RES}/SUMMARY.txt"; tail -30 "\${RES}/mc_build.log" | tee -a "\${RES}/SUMMARY.txt"; exit 0; fi
chmod +x "\${ROOT}/mc.elf"

echo "=== PHASE A: mc --native-compile main.sio under RSS poller ===" | tee -a "\${RES}/SUMMARY.txt"
"\${ROOT}/mc.elf" --native-compile self-hosted/compiler/main.sio -o "\${ROOT}/gen.elf" > "\${RES}/nc.log" 2>&1 &
GPID=\$!; PEAK=0
while kill -0 \${GPID} 2>/dev/null; do
  HWM=\$(awk '/^VmHWM:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  [ -n "\${HWM:-}" ] && [ "\${HWM}" -gt "\${PEAK}" ] 2>/dev/null && PEAK=\${HWM}
  sleep 0.2
done
wait \${GPID}; NCRC=\$?
{
  echo "NATIVE_COMPILE_RC=\${NCRC}  (137=OOM, 139=SIGSEGV, 0=completed)"
  echo "PEAK_HWM_MB=\$((PEAK/1024))   (Phase-0 baseline job 2382 = 16194 MB / OOM)"
  echo "MERGED_IR=\$(grep -E '^Merged IR:' "\${RES}/nc.log" | head -1) \$(grep -A1 -E '^Merged IR:' "\${RES}/nc.log" | tail -1)"
  echo "--- DBG_A1 localization (scratch=global direct read; dst=pointer read) ---"
  grep -E '^DBG_A1 ' "\${RES}/nc.log" | head -12
  echo "gen.elf size: \$(stat -c%s "\${ROOT}/gen.elf" 2>/dev/null || echo 0)"
  if [ -s "\${ROOT}/gen.elf" ]; then chmod +x "\${ROOT}/gen.elf"; echo "gen.elf --help: \$(/usr/bin/timeout 20 "\${ROOT}/gen.elf" --help 2>&1 | head -1)"; fi
  echo "--- last 12 nc.log lines ---"; tail -12 "\${RES}/nc.log"
} | tee -a "\${RES}/SUMMARY.txt"

echo "=== PHASE B: native_v2_multimodule gate (no-regression, expect 27/27) ===" | tee -a "\${RES}/SUMMARY.txt"
export SOUNIO_MODULAR_SOUC="\${ROOT}/mc.elf"
/usr/bin/timeout 600 bash tests/native_v2_multimodule_gate/run.sh "\${ROOT}/mc.elf" > "\${RES}/mm.log" 2>&1; MMRC=\$?
echo "MULTIMODULE_GATE_RC=\${MMRC}" | tee -a "\${RES}/SUMMARY.txt"
grep -iE "[0-9]+/[0-9]+|UNRESOLVED|PASS|FAIL" "\${RES}/mm.log" | tail -8 | tee -a "\${RES}/SUMMARY.txt"

scontrol update JobId=\${SLURM_JOB_ID} Comment="phase1 ncrc=\${NCRC} peakMB=\$((PEAK/1024)) mm=\${MMRC}" 2>/dev/null || true
echo "RESULT_DIR=\${RES}" | tee -a "\${RES}/SUMMARY.txt"
EOF
${KUBECTL} -n "${NS}" cp ${LCFLAG} "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" ${LCFLAG} -- bash -lc "sbatch '/tmp/${RUN_ID}.sbatch'"
echo "RUN_ID=${RUN_ID}"; echo "${RUN_ID}" > /tmp/phase1.last_run_id
rm -f "${TARBALL}" "${SBATCH}"
