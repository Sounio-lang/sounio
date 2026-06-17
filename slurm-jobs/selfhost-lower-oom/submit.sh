#!/usr/bin/env bash
# Diagnose the self-host lower/codegen OOM wall on a SLURM cpu-ops worker (bounded mem),
# so the OOM kills the job — never the workspace pod (see the 2026-06-08 incident).
#
# Packages the kw-demote worktree (self-hosted/ + stdlib/ + bin/souc), submits via the
# slurm-pilot k8s login pod to partition cpu-ops with --mem bound, builds gen-N, then runs
# gen-N --native-compile main.sio under an RSS poller with the per-function instr trace
# enabled. Results land worker-local; fetch with fetch.sh.
set -euo pipefail

WORKTREE="${WORKTREE:-/tmp/kw-demote}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-16G}"   # cpu-ops hard ceiling: MaxMemPerCPU=2G x 8 effective CPUs = 16G
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:25:00}"
RUN_ID="${RUN_ID:-lower-oom-$(date -u +%Y%m%dT%H%M%S)}"

test -x "${WORKTREE}/bin/souc" || { echo "no bin/souc in ${WORKTREE}" >&2; exit 1; }

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
tar -C "${WORKTREE}" -czf "${TARBALL}" self-hosted stdlib bin/souc
echo "staging payload to worker pod /tmp (sbatch script must stay small) ..."
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
RES="/tmp/${RUN_ID}.result"          # stable worker-local path for fetch.sh
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"

echo "host=\$(hostname) job=\${SLURM_JOB_ID:-manual} mem=${JOB_MEM} cpus=${JOB_CPUS}" | tee "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"

tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/souc"
cd "\${REPO}"

echo "[build] gen-N from main.sio ..." | tee "\${RES}/build.log"
/usr/bin/timeout 900 ./bin/souc self-hosted/compiler/main.sio "\${ROOT}/gen.elf" >> "\${RES}/build.log" 2>&1
BRC=\$?
echo "BUILD_RC=\${BRC}" | tee -a "\${RES}/build.log"
if [ \${BRC} -ne 0 ] || [ ! -s "\${ROOT}/gen.elf" ]; then
  echo "build failed rc=\${BRC}" | tee "\${RES}/SUMMARY.txt"
  scontrol update JobId=\${SLURM_JOB_ID} Comment="lower-oom build_fail rc=\${BRC}" 2>/dev/null || true
  exit 0
fi
chmod +x "\${ROOT}/gen.elf"

echo "[run] gen-N --native-compile main.sio (FN_INSTR_TRACE on, RSS poller) ..."
export SOUNIO_FN_INSTR_TRACE=1
export SOUNIO_LOWER_ORDERED_TRACE=1
export SOUNIO_STAGE_TRACE=1
# STACK_ULIMIT discriminator (job 2364+): raise the stack soft limit before the run so a
# stack-overflow crash moves DEEPER while a bss-spill-aliasing crash stays at the same depth.
STACK_ULIMIT_VAL="${STACK_ULIMIT:-}"
if [ -n "\${STACK_ULIMIT_VAL}" ]; then
  ulimit -s \${STACK_ULIMIT_VAL} 2>/dev/null || ulimit -s unlimited 2>/dev/null || true
fi
echo "ulimit_s_before_run=\$(ulimit -s)" | tee -a "\${RES}/env.txt"
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
LASTFN=\$(grep '^FNINSTR ' "\${RES}/run.log" 2>/dev/null | tail -1)
{
  echo "RUN_RC=\${RRC}   (137=SIGKILL/OOM)"
  echo "PEAK_HWM_KB=\${PEAK} PEAK_HWM_MB=\$((PEAK/1024)) PEAK_HWM_GB=\$((PEAK/1024/1024))"
  echo "FNINSTR_LINES=\${NF}   (functions lowered before death)"
  echo "LAST_FNINSTR=\${LASTFN}"
  echo "--- STAGE markers seen (order of execution) ---"
  grep '^STAGE ' "\${RES}/run.log" 2>/dev/null
  echo "PRESEED_LINES=\$(grep -c '^PRESEED ' "\${RES}/run.log" 2>/dev/null || echo 0)"
  echo "LAST_PRESEED=\$(grep '^PRESEED ' "\${RES}/run.log" 2>/dev/null | tail -1)"
  echo "--- instr_count distribution (count by instr bucket) ---"
  grep '^FNINSTR ' "\${RES}/run.log" 2>/dev/null | sed -E 's/.* instrs=([0-9]+).*/\1/' \
    | awk '{n++; s+=\$1; if(\$1>mx)mx=\$1; a[n]=\$1} END{ if(n==0){print "no data"; exit} asort(a); print "n="n" max="mx" mean="int(s/n)" p50="a[int(n*0.5)]" p90="a[int(n*0.9)]" p99="a[int(n*0.99)]; c=0; for(i=1;i<=n;i++) if(a[i]>=256)c++; print "at_or_over_256_cap="c}'
  echo "--- last 25 stdout lines ---"
  tail -25 "\${RES}/run.log" 2>/dev/null
} | tee "\${RES}/SUMMARY.txt"
scontrol update JobId=\${SLURM_JOB_ID} Comment="lower-oom rc=\${RRC} peakGB=\$((PEAK/1024/1024)) fns=\${NF}" 2>/dev/null || true
echo "RESULT_DIR=\${RES}"
EOF

REMOTE_SBATCH="/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:${REMOTE_SBATCH}"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
