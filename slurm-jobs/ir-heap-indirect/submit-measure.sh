#!/usr/bin/env bash
# Measure WHERE the ~13GB peak accumulates in the full-IR --native-compile path: RSS poller
# correlated with STAGE_RSS markers (before_load / after_load[=merge done] / after_codegen).
set -uo pipefail
WORKTREE="${WORKTREE:-/workspace/sounio-ir}"; NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION=cpu-ops; ACCOUNT=omics; QOS=cpuops; JOB_MEM="${JOB_MEM:-16G}"; JOB_CPUS=2; JOB_TIME="00:40:00"
RUN_ID="${RUN_ID:-measure-$(date -u +%Y%m%dT%H%M%S)}"
LOGIN_POD="${LOGIN_POD:-$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')}"
LOGIN_CTR="${LOGIN_CTR:-}"; LCFLAG=""; [ -n "${LOGIN_CTR}" ] && LCFLAG="-c ${LOGIN_CTR}"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)"
test -n "${WORKER_POD}" || { echo "no worker" >&2; exit 1; }
echo "login=${LOGIN_POD} worker=${WORKER_POD}"
TARBALL="/tmp/${RUN_ID}.tgz"; SBATCH="/tmp/${RUN_ID}.sbatch"
tar -C "${WORKTREE}" -czf "${TARBALL}" self-hosted stdlib bin/souc bin/souc-linux-x86_64
${KUBECTL} -n "${NS}" cp "${TARBALL}" "${WORKER_POD}:/tmp/${RUN_ID}.tgz" -c slurmd
cat > "${SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH -N 1 -n 1 -c ${JOB_CPUS} --mem=${JOB_MEM} --time=${JOB_TIME}
#SBATCH -o /tmp/${RUN_ID}.slurmout
#SBATCH -e /tmp/${RUN_ID}.slurmout
set -uo pipefail
ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-m}"; REPO="\${ROOT}/repo"; RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"; chmod +x "\${REPO}/bin/souc"; cd "\${REPO}"
echo "[build]" | tee "\${RES}/SUMMARY.txt"
/usr/bin/timeout 1200 ./bin/souc self-hosted/compiler/main.sio "\${ROOT}/mc.elf" > "\${RES}/mc_build.log" 2>&1
[ -s "\${ROOT}/mc.elf" ] || { echo "BUILD_FAIL" | tee -a "\${RES}/SUMMARY.txt"; tail -20 "\${RES}/mc_build.log" | tee -a "\${RES}/SUMMARY.txt"; exit 0; }
chmod +x "\${ROOT}/mc.elf"
"\${ROOT}/mc.elf" --native-compile self-hosted/compiler/main.sio -o "\${ROOT}/gen.elf" > "\${RES}/nc.log" 2>&1 &
GPID=\$!; : > "\${RES}/rss_stage.log"
while kill -0 \${GPID} 2>/dev/null; do
  HWM=\$(awk '/^VmHWM:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  RSS=\$(awk '/^VmRSS:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  STG=\$(grep '^STAGE_RSS ' "\${RES}/nc.log" 2>/dev/null | tail -1 | awk '{print \$2}')
  [ -n "\${HWM:-}" ] && echo "\$(date +%s) hwm_mb=\$((HWM/1024)) rss_mb=\$((RSS/1024)) stage=\${STG:-init}" >> "\${RES}/rss_stage.log"
  sleep 0.2
done
wait \${GPID}; RC=\$?
{
  echo "NC_RC=\${RC}"
  echo "PEAK_HWM_MB=\$(awk -F'hwm_mb=' '{print \$2}' "\${RES}/rss_stage.log" | awk '{print \$1}' | sort -n | tail -1)"
  echo "--- RSS at first sighting of each stage ---"
  awk '{s=\$NF; if(s!=last){print; last=s}}' "\${RES}/rss_stage.log"
  echo "--- peak rss_mb per stage ---"
  awk '{gsub(/hwm_mb=|rss_mb=|stage=/,""); st=\$4; if(\$2>mx[st])mx[st]=\$2} END{for(s in mx)print s": peak_hwm_mb="mx[s]}' "\${RES}/rss_stage.log"
  echo "--- STAGE_RSS markers seen ---"; grep '^STAGE_RSS ' "\${RES}/nc.log"
  echo "--- Merged IR / result ---"; grep -E 'Merged IR:|successful|failed' "\${RES}/nc.log" | head
} | tee -a "\${RES}/SUMMARY.txt"
echo "RESULT_DIR=\${RES}"
EOF
${KUBECTL} -n "${NS}" cp ${LCFLAG} "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" ${LCFLAG} -- bash -lc "sbatch '/tmp/${RUN_ID}.sbatch'"
echo "RUN_ID=${RUN_ID}"; echo "${RUN_ID}" > /tmp/measure.last_run_id
rm -f "${TARBALL}" "${SBATCH}"
