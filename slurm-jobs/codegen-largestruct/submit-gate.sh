#!/usr/bin/env bash
# Final landing gate for the A+D codegen fixes: rebootstrap the FIXED lean_single to a
# fixed point, then run the honest release_gate with that rebuilt compiler as the
# bootstrap (so mc is built from main.sio by the A+D-fixed bin/souc). On a SLURM cpu-ops
# worker (bounded --mem). If FIXED_POINT=YES and RELEASE_GATE=PASS, gen3 (in the result
# dir) is the install candidate for canonical bin/souc.
set -euo pipefail

WORKTREE="${WORKTREE:-/workspace/sounio-codegen}"
NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"; ACCOUNT="${ACCOUNT:-omics}"; QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-16G}"; JOB_CPUS="${JOB_CPUS:-2}"; JOB_TIME="${JOB_TIME:-00:55:00}"
RUN_ID="${RUN_ID:-cggate-$(date -u +%Y%m%dT%H%M%S)}"

test -x "${WORKTREE}/bin/souc-linux-x86_64" || { echo "no seed bin/souc-linux-x86_64" >&2; exit 1; }
LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no running login pod" >&2; exit 1; }
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)"
WORKER_CTR="${WORKER_CTR:-slurmd}"
test -n "${WORKER_POD}" || { echo "no cpuops worker pod" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}   worker pod: ${WORKER_POD}"

TARBALL="/tmp/${RUN_ID}.tgz"; SBATCH="/tmp/${RUN_ID}.sbatch"
echo "packaging payload -> ${TARBALL}"
tar -C "${WORKTREE}" -czf "${TARBALL}" \
  self-hosted stdlib bin/souc-linux-x86_64 scripts tests \
  slurm-jobs/codegen-largestruct/repro_boxstore.sio slurm-jobs/codegen-largestruct/repro_huge.sio
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
ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"; REPO="\${ROOT}/repo"; RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"
echo "host=\$(hostname) job=\${SLURM_JOB_ID:-manual} mem=${JOB_MEM}" | tee "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/souc-linux-x86_64"
cd "\${REPO}"
LS=self-hosted/compiler/lean_single.sio

echo "=== PHASE 1: REBOOTSTRAP ===" | tee "\${RES}/SUMMARY.txt"
/usr/bin/timeout 1200 ./bin/souc-linux-x86_64 "\${LS}" "\${ROOT}/gen1" > "\${RES}/rebootstrap.log" 2>&1; S1=\$?
[ \${S1} -eq 0 ] && [ -s "\${ROOT}/gen1" ] && chmod +x "\${ROOT}/gen1"
if [ \${S1} -ne 0 ]; then echo "REBOOTSTRAP FAIL stage1 rc=\${S1}" | tee -a "\${RES}/SUMMARY.txt"; tail -30 "\${RES}/rebootstrap.log" | tee -a "\${RES}/SUMMARY.txt"; exit 0; fi
/usr/bin/timeout 1200 "\${ROOT}/gen1" "\${LS}" "\${ROOT}/gen2" >> "\${RES}/rebootstrap.log" 2>&1; S2=\$?
[ \${S2} -eq 0 ] && [ -s "\${ROOT}/gen2" ] && chmod +x "\${ROOT}/gen2"
/usr/bin/timeout 1200 "\${ROOT}/gen2" "\${LS}" "\${ROOT}/gen3" >> "\${RES}/rebootstrap.log" 2>&1; S3=\$?
[ \${S3} -eq 0 ] && [ -s "\${ROOT}/gen3" ] && chmod +x "\${ROOT}/gen3"
M2=\$(md5sum "\${ROOT}/gen2" 2>/dev/null | awk '{print \$1}'); M3=\$(md5sum "\${ROOT}/gen3" 2>/dev/null | awk '{print \$1}')
FP=NO; [ -n "\${M2}" ] && [ "\${M2}" = "\${M3}" ] && FP=YES
echo "STAGE_RC s1=\${S1} s2=\${S2} s3=\${S3}  md5 gen2=\${M2} gen3=\${M3}  FIXED_POINT=\${FP}" | tee -a "\${RES}/SUMMARY.txt"
# stash the install candidate (gen3 = the fixed point) for fetch
cp -f "\${ROOT}/gen3" "\${RES}/gen3" 2>/dev/null && md5sum "\${RES}/gen3" | tee -a "\${RES}/SUMMARY.txt"

echo "=== PHASE 2: A-FIX REPROS (gen1) ===" | tee -a "\${RES}/SUMMARY.txt"
for r in repro_boxstore repro_huge; do
  /usr/bin/timeout 120 "\${ROOT}/gen1" "slurm-jobs/codegen-largestruct/\${r}.sio" "\${ROOT}/\${r}.elf" > "\${RES}/\${r}.build.log" 2>&1
  if [ -s "\${ROOT}/\${r}.elf" ]; then chmod +x "\${ROOT}/\${r}.elf"; echo "\${r}: \$(/usr/bin/timeout 60 "\${ROOT}/\${r}.elf" 2>&1 | tr '\n' '/')" | tee -a "\${RES}/SUMMARY.txt"; else echo "\${r}: BUILD_FAILED" | tee -a "\${RES}/SUMMARY.txt"; fi
done

echo "=== PHASE 3: RELEASE GATE (bootstrap = rebuilt gen3) ===" | tee -a "\${RES}/SUMMARY.txt"
export SOUNIO_BOOTSTRAP_SOUC="\${ROOT}/gen3"
export SOUNIO_REPO_HARD_NO_RUST=1
export SOUNIO_BUILD_LOCK="/tmp/${RUN_ID}.buildlock"
/usr/bin/timeout 2400 bash scripts/ci/release_gate.sh > "\${RES}/release_gate.log" 2>&1; GRC=\$?
echo "RELEASE_GATE_RC=\${GRC}" | tee -a "\${RES}/SUMMARY.txt"
grep -E "RELEASE GATE:|PASS|FAIL|building modular" "\${RES}/release_gate.log" | tail -30 | tee -a "\${RES}/SUMMARY.txt"
echo "--- last 25 lines release_gate.log ---" | tee -a "\${RES}/SUMMARY.txt"
tail -25 "\${RES}/release_gate.log" | tee -a "\${RES}/SUMMARY.txt"

scontrol update JobId=\${SLURM_JOB_ID} Comment="cggate fp=\${FP} gate_rc=\${GRC}" 2>/dev/null || true
echo "RESULT_DIR=\${RES}" | tee -a "\${RES}/SUMMARY.txt"
EOF

${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '/tmp/${RUN_ID}.sbatch'"
echo "RUN_ID=${RUN_ID}"; echo "${RUN_ID}" > /tmp/cgrebootstrap.last_run_id
rm -f "${TARBALL}" "${SBATCH}"
