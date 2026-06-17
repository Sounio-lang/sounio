#!/usr/bin/env bash
# FULL SELF-HOSTED validation via the Rust JIT (bin/souc-linux-x86_64) as the build compiler.
# The lean_single seed miscompiles Madaros; the Rust JIT (Stage-0 root) compiles faithfully
# (verified: it built+ran a trivial program correctly). So use it to build main.sio twice:
#   M_fix  = committed main as-is (dynamic frame at live sites 6190 & 7333 — fix present)
#   M_base = same with 6190 & 7333 reverted to `sub rsp,512`
# Then reproducer A/B. Source = git archive main (committed, clean). Reproducer from work tree.
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"; REF="${REF:-main}"
NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpu-orangefs}"; ACCOUNT="${ACCOUNT:-plruntime}"; QOS="${QOS:-gpuorangefs}"
NODELIST="${NODELIST:-gpuorangefs-r770-proxmox}"
WORKER_POD="${WORKER_POD:-slurm-pilot-worker-gpuorangefs-zj2l4}"; WORKER_CTR="${WORKER_CTR:-slurmd}"
JOB_MEM="${JOB_MEM:-96G}"; JOB_CPUS="${JOB_CPUS:-8}"; JOB_TIME="${JOB_TIME:-02:00:00}"
RUN_ID="${RUN_ID:-madaros-jit-build-$(date -u +%Y%m%dT%H%M%S)}"
REPRO="examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"
JIT="${JIT:-${REPO}/bin/souc-linux-x86_64}"

test -x "${JIT}" || { echo "no JIT ${JIT}" >&2; exit 1; }
test -f "${REPO}/${REPRO}" || { echo "no reproducer" >&2; exit 1; }
LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
echo "login=${LOGIN_POD} worker=${WORKER_POD} ref=${REF}"

STAGE="/tmp/${RUN_ID}.stage"; rm -rf "${STAGE}"; mkdir -p "${STAGE}"
git -C "${REPO}" archive --format=tar "${REF}" self-hosted stdlib | tar -x -C "${STAGE}"
cp "${JIT}" "${STAGE}/jit"
mkdir -p "${STAGE}/$(dirname "${REPRO}")"; cp "${REPO}/${REPRO}" "${STAGE}/${REPRO}"
TARBALL="/tmp/${RUN_ID}.tgz"
tar -C "${STAGE}" -czf "${TARBALL}" self-hosted stdlib jit "${REPRO}"; rm -rf "${STAGE}"
echo "payload $(stat -c%s "${TARBALL}" | awk '{printf "%.1f MB",$1/1048576}')"
${KUBECTL} -n "${NS}" cp "${TARBALL}" "${WORKER_POD}:/tmp/${RUN_ID}.tgz" -c "${WORKER_CTR}"

SBATCH="/tmp/${RUN_ID}.sbatch"
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
#SBATCH -w ${NODELIST}
#SBATCH -o /tmp/${RUN_ID}.slurmout
#SBATCH -e /tmp/${RUN_ID}.slurmout
set -uo pipefail
ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"; RES="/tmp/${RUN_ID}.result"; SUM="\${RES}/SUMMARY.txt"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${ROOT}" "\${RES}"; : > "\${SUM}"
log(){ echo "\$@" >> "\${SUM}"; }
CG="self-hosted/native/codegen_x86_linux.sio"

FIX="\${ROOT}/src_fix"; mkdir -p "\${FIX}"; tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${FIX}"
chmod +x "\${FIX}/jit"
# assert dynamic frame present at the two LIVE sites
if ! sed -n '6190p' "\${FIX}/\${CG}" | grep -q 'align16((\*func).reg_count' \
   || ! sed -n '7333p' "\${FIX}/\${CG}" | grep -q 'align16((\*func).reg_count'; then
  log "ABORT: dynamic frame not at 6190 & 7333."; sed -n '6190p;7333p' "\${FIX}/\${CG}" >> "\${SUM}"; exit 0; fi
# base = revert the 4-line block to sub 512 at both live sites
BASE="\${ROOT}/src_base"; cp -a "\${FIX}" "\${BASE}"
awk 'NR==6190{print "    nc_emit_sub_rsp_imm32(nc, 512)";skip=3;next} NR==7333{print "    nc_emit_sub_rsp_imm32(nc, 512)";skip=3;next} skip>0{skip--;next}1' \
  "\${FIX}/\${CG}" > "\${BASE}/\${CG}"

build() { # name srcdir
  local N="\$1" D="\$2" M="\${ROOT}/M_\$1.elf"
  ( cd "\${D}" && export SOUNIO_STDLIB_PATH="\${D}/stdlib" &&
    /usr/bin/timeout 6000 ./jit self-hosted/compiler/main.sio "\${M}" > "\${RES}/build_\$1.log" 2>&1 )
  local rc=\$? errs=\$(grep -c '^error:' "\${RES}/build_\$1.log" 2>/dev/null)
  log "build \$1: rc=\${rc} hard_errors=\${errs} size=\$(stat -c%s "\${M}" 2>/dev/null || echo 0)"
  [ \${errs} -gt 0 ] && grep '^error:' "\${RES}/build_\$1.log" | head -8 | sed 's/^/    /' >> "\${SUM}"
  tail -3 "\${RES}/build_\$1.log" | sed 's/^/    tail: /' >> "\${SUM}"
  [ -s "\${M}" ] && chmod +x "\${M}"
}
build fix  "\${FIX}"
build base "\${BASE}"
log "binaries: \${ROOT}/M_fix.elf \${ROOT}/M_base.elf  (reproducer at \${FIX}/${REPRO})"
log "Done."
EOF
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
