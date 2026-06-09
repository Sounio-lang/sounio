#!/usr/bin/env bash
# Rebootstrap + verify the A (deref-store) + D (call-reloc) lean_single codegen fixes on a
# SLURM cpu-ops worker (bounded --mem, so an OOM kills the JOB not the workspace pod — see
# the 2026-06-08 incident). Routes via the slurm-pilot k8s login pod + cpuops worker scratch.
#
# PHASE 1  3-STAGE REBOOTSTRAP + FIXED POINT (primary acceptance):
#          seed(bin/souc-linux-x86_64) compiles the FIXED lean_single -> gen1 (this is also the
#          faithful parse/typecheck/codegen test of the edits); gen1 -> gen2; gen2 -> gen3.
#          md5(gen2)==md5(gen3) => the fixed compiler self-reproduces. (gen1!=gen2 is EXPECTED:
#          gen1 was built by the old buggy seed, gen2 by the fixed gen1.)
# PHASE 2  A-FIX VERIFICATION (primary): gen1 compiles+runs the deref-store repros; expect
#          repro_boxstore=42/42/55/55 and min=0/42 -> 42/42 (pre-fix was 0). Proves A works.
# PHASE 3  D-FIX / WALL OBSERVATION (observational): gen1 builds souc_new.elf from main.sio
#          (the fixed bin/souc building the modular compiler), then souc_new.elf --native-compile
#          main.sio under an RSS poller. Watch for: the D guard ("too many call relocations" =
#          need a bigger PATCH cap), and whether the front-half gets further than the old broken
#          souc.elf. B/C (96MB IrModule) are UNFIXED, so a front-half SIGSEGV/OOM here is expected
#          and is a RESULT, not a regression.
set -euo pipefail

WORKTREE="${WORKTREE:-/workspace/sounio-codegen}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-16G}"            # cpu-ops hard ceiling
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:55:00}"
RUN_ID="${RUN_ID:-cgrebootstrap-$(date -u +%Y%m%dT%H%M%S)}"

test -x "${WORKTREE}/bin/souc-linux-x86_64" || { echo "no seed bin/souc-linux-x86_64 in ${WORKTREE}" >&2; exit 1; }

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no running login pod" >&2; exit 1; }
WORKER_POD="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)"
WORKER_CTR="${WORKER_CTR:-slurmd}"
test -n "${WORKER_POD}" || { echo "no cpuops worker pod" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}   worker pod: ${WORKER_POD}"

TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH="/tmp/${RUN_ID}.sbatch"
echo "packaging payload -> ${TARBALL}"
tar -C "${WORKTREE}" -czf "${TARBALL}" \
  self-hosted stdlib bin/souc bin/souc-linux-x86_64 \
  slurm-jobs/codegen-largestruct/repro_boxstore.sio \
  slurm-jobs/codegen-largestruct/repro_huge.sio
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
RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${REPO}" "\${RES}"

echo "host=\$(hostname) job=\${SLURM_JOB_ID:-manual} mem=${JOB_MEM} cpus=${JOB_CPUS}" | tee "\${RES}/env.txt"
free -g | head -2 | tee -a "\${RES}/env.txt"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${REPO}"
chmod +x "\${REPO}/bin/souc" "\${REPO}/bin/souc-linux-x86_64"
cd "\${REPO}"
LS=self-hosted/compiler/lean_single.sio

# ---------- PHASE 1: 3-stage rebootstrap + fixed point ----------
echo "[stage1] seed compiles FIXED lean_single -> gen1 ..." | tee "\${RES}/rebootstrap.log"
/usr/bin/timeout 1200 ./bin/souc-linux-x86_64 "\${LS}" "\${ROOT}/gen1" >> "\${RES}/rebootstrap.log" 2>&1
S1=\$?; echo "STAGE1_RC=\${S1}" | tee -a "\${RES}/rebootstrap.log"
if [ \${S1} -ne 0 ] || [ ! -s "\${ROOT}/gen1" ]; then
  echo "REBOOTSTRAP: FAIL at stage1 (fixed lean_single did not compile rc=\${S1})" | tee "\${RES}/SUMMARY.txt"
  tail -40 "\${RES}/rebootstrap.log" | tee -a "\${RES}/SUMMARY.txt"
  scontrol update JobId=\${SLURM_JOB_ID} Comment="cgreboot stage1_fail rc=\${S1}" 2>/dev/null || true
  exit 0
fi
chmod +x "\${ROOT}/gen1"
echo "[stage2] gen1 -> gen2 ..." | tee -a "\${RES}/rebootstrap.log"
/usr/bin/timeout 1200 "\${ROOT}/gen1" "\${LS}" "\${ROOT}/gen2" >> "\${RES}/rebootstrap.log" 2>&1
S2=\$?; echo "STAGE2_RC=\${S2}" | tee -a "\${RES}/rebootstrap.log"
echo "[stage3] gen2 -> gen3 ..." | tee -a "\${RES}/rebootstrap.log"
if [ \${S2} -eq 0 ] && [ -s "\${ROOT}/gen2" ]; then chmod +x "\${ROOT}/gen2"; fi
/usr/bin/timeout 1200 "\${ROOT}/gen2" "\${LS}" "\${ROOT}/gen3" >> "\${RES}/rebootstrap.log" 2>&1
S3=\$?; echo "STAGE3_RC=\${S3}" | tee -a "\${RES}/rebootstrap.log"
M2=\$(md5sum "\${ROOT}/gen2" 2>/dev/null | awk '{print \$1}')
M3=\$(md5sum "\${ROOT}/gen3" 2>/dev/null | awk '{print \$1}')
FIXEDPOINT=NO; [ -n "\${M2}" ] && [ "\${M2}" = "\${M3}" ] && FIXEDPOINT=YES
{
  echo "=== PHASE 1: REBOOTSTRAP ==="
  echo "STAGE_RC s1=\${S1} s2=\${S2} s3=\${S3}"
  echo "md5 gen2=\${M2} gen3=\${M3}  FIXED_POINT=\${FIXEDPOINT}"
} | tee "\${RES}/SUMMARY.txt"

# ---------- PHASE 2: A-fix repro verification (use gen1, the fixed compiler) ----------
echo "=== PHASE 2: A-FIX REPROS (gen1) ===" | tee -a "\${RES}/SUMMARY.txt"
for r in repro_boxstore repro_huge; do
  SRC="slurm-jobs/codegen-largestruct/\${r}.sio"
  /usr/bin/timeout 120 "\${ROOT}/gen1" "\${SRC}" "\${ROOT}/\${r}.elf" > "\${RES}/\${r}.build.log" 2>&1
  if [ -s "\${ROOT}/\${r}.elf" ]; then
    chmod +x "\${ROOT}/\${r}.elf"
    OUT=\$(/usr/bin/timeout 60 "\${ROOT}/\${r}.elf" 2>&1 | tr '\n' '/')
    echo "\${r}: \${OUT}" | tee -a "\${RES}/SUMMARY.txt"
  else
    echo "\${r}: BUILD_FAILED (see \${r}.build.log)" | tee -a "\${RES}/SUMMARY.txt"
  fi
done
echo "(expect repro_boxstore=42/42/55/55/ ; repro_huge=11/99/100/200/)" | tee -a "\${RES}/SUMMARY.txt"

# ---------- PHASE 3: D-fix / wall observation ----------
if [ "${SKIP_PHASE3:-0}" = "1" ]; then
  echo "=== PHASE 3: SKIPPED (SKIP_PHASE3=1) ===" | tee -a "\${RES}/SUMMARY.txt"
else
echo "=== PHASE 3: build souc_new.elf = gen1 main.sio, then --native-compile (observational) ===" | tee -a "\${RES}/SUMMARY.txt"
echo "[build mc] gen1 self-hosted/compiler/main.sio -> souc_new.elf ..." | tee "\${RES}/mc_build.log"
/usr/bin/timeout 1200 "\${ROOT}/gen1" self-hosted/compiler/main.sio "\${ROOT}/souc_new.elf" >> "\${RES}/mc_build.log" 2>&1
MCRC=\$?; echo "MC_BUILD_RC=\${MCRC}" | tee -a "\${RES}/SUMMARY.txt"
grep -iE "too many call relocations|too many" "\${RES}/mc_build.log" | tee -a "\${RES}/SUMMARY.txt" || true
if [ \${MCRC} -eq 0 ] && [ -s "\${ROOT}/souc_new.elf" ]; then
  chmod +x "\${ROOT}/souc_new.elf"
  echo "[native-compile] souc_new.elf --native-compile main.sio (RSS poller) ..." | tee -a "\${RES}/SUMMARY.txt"
  "\${ROOT}/souc_new.elf" --native-compile self-hosted/compiler/main.sio -o "\${ROOT}/gen_main.elf" > "\${RES}/native_compile.log" 2>&1 &
  GPID=\$!; PEAK=0
  while kill -0 \${GPID} 2>/dev/null; do
    HWM=\$(awk '/^VmHWM:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
    [ -n "\${HWM:-}" ] && [ "\${HWM}" -gt "\${PEAK}" ] 2>/dev/null && PEAK=\${HWM}
    sleep 0.3
  done
  wait \${GPID}; NCRC=\$?
  {
    echo "NATIVE_COMPILE_RC=\${NCRC} (137=SIGKILL/OOM, 139=SIGSEGV)"
    echo "PEAK_HWM_MB=\$((PEAK/1024))"
    echo "gen_main.elf size: \$(stat -c%s "\${ROOT}/gen_main.elf" 2>/dev/null || echo 0)"
    if [ -s "\${ROOT}/gen_main.elf" ]; then
      chmod +x "\${ROOT}/gen_main.elf"
      echo "gen_main.elf RUNS (self-host candidate!): \$(/usr/bin/timeout 30 "\${ROOT}/gen_main.elf" --help 2>&1 | head -1)"
    fi
    grep -iE "too many call relocations|merged ir|missing main|imported_simple" "\${RES}/native_compile.log" | head -10
    echo "--- last 20 lines native_compile.log ---"; tail -20 "\${RES}/native_compile.log"
  } | tee -a "\${RES}/SUMMARY.txt"
else
  echo "souc_new.elf not built (mc_build rc=\${MCRC}); skipping native-compile probe" | tee -a "\${RES}/SUMMARY.txt"
  tail -30 "\${RES}/mc_build.log" | tee -a "\${RES}/SUMMARY.txt"
fi
fi  # end SKIP_PHASE3 guard

scontrol update JobId=\${SLURM_JOB_ID} Comment="cgreboot fp=\${FIXEDPOINT} s1=\${S1}" 2>/dev/null || true
echo "RESULT_DIR=\${RES}" | tee -a "\${RES}/SUMMARY.txt"
EOF

REMOTE_SBATCH="/tmp/${RUN_ID}.sbatch"
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:${REMOTE_SBATCH}"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'"
echo "RUN_ID=${RUN_ID}"
echo "${RUN_ID}" > /tmp/cgrebootstrap.last_run_id
rm -f "${TARBALL}" "${SBATCH}"
