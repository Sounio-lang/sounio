#!/usr/bin/env bash
# DECISIVE TEST before the Lowerer rework (advisor): does --native-v2-compile main.sio
# (= bridge-combine + real Lowerer + native-v2 back-half) clear the front-half at 6681-fn
# scale, and does the ELF it produces RUN? This tells us if the rework premise holds.
set -uo pipefail
WORKTREE="${WORKTREE:-/workspace/sounio-ir}"; NS=slurm-pilot; KUBECTL=kubectl
RUN_ID="${RUN_ID:-v2test-$(date -u +%Y%m%dT%H%M%S)}"
LOGIN_POD="${LOGIN_POD:-$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')}"
LOGIN_CTR="${LOGIN_CTR:-}"; LCFLAG=""; [ -n "${LOGIN_CTR}" ] && LCFLAG="-c ${LOGIN_CTR}"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)"
echo "login=${LOGIN_POD} worker=${WORKER_POD}"
TARBALL="/tmp/${RUN_ID}.tgz"; SBATCH="/tmp/${RUN_ID}.sbatch"
tar -C "${WORKTREE}" -czf "${TARBALL}" self-hosted stdlib bin/souc bin/souc-linux-x86_64
${KUBECTL} -n "${NS}" cp "${TARBALL}" "${WORKER_POD}:/tmp/${RUN_ID}.tgz" -c slurmd
cat > "${SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J ${RUN_ID}
#SBATCH -p cpu-ops -A omics --qos=cpuops
#SBATCH -N 1 -n 1 -c 2 --mem=16G --time=00:40:00
#SBATCH -o /tmp/${RUN_ID}.slurmout
#SBATCH -e /tmp/${RUN_ID}.slurmout
set -uo pipefail
ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-m}"; REPO="\${ROOT}/repo"; RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"; chmod +x "\${REPO}/bin/souc"; cd "\${REPO}"
echo "[build mc]" | tee "\${RES}/SUMMARY.txt"
/usr/bin/timeout 1200 ./bin/souc self-hosted/compiler/main.sio "\${ROOT}/mc.elf" > "\${RES}/build.log" 2>&1
[ -s "\${ROOT}/mc.elf" ] || { echo "BUILD_FAIL" | tee -a "\${RES}/SUMMARY.txt"; tail -15 "\${RES}/build.log" | tee -a "\${RES}/SUMMARY.txt"; exit 0; }
chmod +x "\${ROOT}/mc.elf"
rm -f /tmp/nv2dbg_parse /tmp/nv2dbg_summary /tmp/nv2dbg_bodies /tmp/nv2_combined_multimodule.sio
echo "[v2] mc --native-v2-compile main.sio under RSS poller ..." | tee -a "\${RES}/SUMMARY.txt"
"\${ROOT}/mc.elf" --native-v2-compile self-hosted/compiler/main.sio -o "\${ROOT}/v2.elf" > "\${RES}/v2.log" 2>&1 &
GPID=\$!; PEAK=0
while kill -0 \${GPID} 2>/dev/null; do
  HWM=\$(awk '/^VmHWM:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  [ -n "\${HWM:-}" ] && [ "\${HWM}" -gt "\${PEAK}" ] 2>/dev/null && PEAK=\${HWM}
  sleep 0.3
done
wait \${GPID}; V2RC=\$?
{
  echo "V2_COMPILE_RC=\${V2RC}  (0=ok, 137=OOM, 139=SIGSEGV)"
  echo "STAGE_REACHED: combined=\$([ -f /tmp/nv2_combined_multimodule.sio ] && wc -c </tmp/nv2_combined_multimodule.sio || echo MISSING) parse=\$([ -f /tmp/nv2dbg_parse ] && echo Y || echo N) summary=\$([ -f /tmp/nv2dbg_summary ] && echo Y || echo N) bodies=\$([ -f /tmp/nv2dbg_bodies ] && echo Y || echo N)"
  echo "PEAK_HWM_MB=\$((PEAK/1024))"
  echo "v2.elf size: \$(stat -c%s "\${ROOT}/v2.elf" 2>/dev/null || echo 0)"
  echo "--- front-half: Merged IR / errors ---"; grep -iE "NV2DBG|Merged IR:|error|fail|successful|REJECT" "\${RES}/v2.log" | head -12
  if [ -s "\${ROOT}/v2.elf" ]; then
    chmod +x "\${ROOT}/v2.elf"
    echo "--- DOES v2.elf RUN? (the residual-2 disambiguator) ---"
    /usr/bin/timeout 30 "\${ROOT}/v2.elf" --help > "\${RES}/v2run.log" 2>&1; RR=\$?
    echo "v2.elf --help rc=\${RR}  (0=runs, 139=SIGSEGV/core)"
    head -2 "\${RES}/v2run.log"
  else
    echo "no v2.elf produced"
  fi
} | tee -a "\${RES}/SUMMARY.txt"
echo "RESULT_DIR=\${RES}"
EOF
${KUBECTL} -n "${NS}" cp ${LCFLAG} "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" ${LCFLAG} -- bash -lc "sbatch '/tmp/${RUN_ID}.sbatch'"
echo "RUN_ID=${RUN_ID}"; echo "${RUN_ID}" > /tmp/v2test.last_run_id
rm -f "${TARBALL}" "${SBATCH}"
