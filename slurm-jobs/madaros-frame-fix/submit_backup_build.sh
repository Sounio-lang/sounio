#!/usr/bin/env bash
# #2 validation via a KNOWN-GOOD compiler. The bootstrap seed cannot build a working
# Madaros from current source (deeper seed miscompiles beyond the E001), so we instead
# use the working backup Madaros (artifacts/self-hosted/madaros.backup.20260616 — which
# compiles+runs the reproducer and exhibits the bug: N=2/4 wrong trail, N=5 SIGSEGV) as
# the build compiler to self-compile main.sio twice:
#   M_fixed = main.sio with the live frame fix (working tree as-is: 6190 & 7333 dynamic)
#   M_base  = main.sio with the two live frame-fix blocks reverted to `sub rsp,512`
# Binaries are preserved; the reproducer A/B is run manually afterward.
# Source uses &programs[i] which the modern backup parses fine — no seed-compat rewrite needed.
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpu-orangefs}"; ACCOUNT="${ACCOUNT:-plruntime}"; QOS="${QOS:-gpuorangefs}"
NODELIST="${NODELIST:-gpuorangefs-r770-proxmox}"
WORKER_POD="${WORKER_POD:-slurm-pilot-worker-gpuorangefs-zj2l4}"; WORKER_CTR="${WORKER_CTR:-slurmd}"
JOB_MEM="${JOB_MEM:-32G}"; JOB_CPUS="${JOB_CPUS:-8}"; JOB_TIME="${JOB_TIME:-01:00:00}"
RUN_ID="${RUN_ID:-madaros-backup-build-$(date -u +%Y%m%dT%H%M%S)}"
BACKUP="${BACKUP:-${REPO}/artifacts/self-hosted/madaros.backup.20260616}"
REPRO="examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"

test -x "${BACKUP}" || { echo "missing backup madaros ${BACKUP}" >&2; exit 1; }
LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
echo "login=${LOGIN_POD} worker=${WORKER_POD}"

TARBALL="/tmp/${RUN_ID}.tgz"; SBATCH="/tmp/${RUN_ID}.sbatch"
tar -C "${REPO}" -czf "${TARBALL}" self-hosted stdlib "${REPRO}"
echo "payload $(stat -c%s "${TARBALL}" | awk '{printf "%.1f MB",$1/1048576}'); staging payload + backup compiler"
${KUBECTL} -n "${NS}" cp "${TARBALL}" "${WORKER_POD}:/tmp/${RUN_ID}.tgz" -c "${WORKER_CTR}"
${KUBECTL} -n "${NS}" cp "${BACKUP}" "${WORKER_POD}:/tmp/${RUN_ID}.backup.elf" -c "${WORKER_CTR}"

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
BK="/tmp/${RUN_ID}.backup.elf"; chmod +x "\${BK}"

FIX="\${ROOT}/src_fix"; mkdir -p "\${FIX}"; tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${FIX}"
# assert live fix present
if ! sed -n '6190p' "\${FIX}/\${CG}" | grep -q 'align16((\*func).reg_count' \
   || ! sed -n '7333p' "\${FIX}/\${CG}" | grep -q 'align16((\*func).reg_count'; then
  log "ABORT: live fix not at 6190 & 7333"; exit 0; fi
# base = fix reverted
BASE="\${ROOT}/src_base"; cp -a "\${FIX}" "\${BASE}"
awk 'NR==6190{print "    nc_emit_sub_rsp_imm32(nc, 512)";skip=3;next} NR==7333{print "    nc_emit_sub_rsp_imm32(nc, 512)";skip=3;next} skip>0{skip--;next}1' \
  "\${FIX}/\${CG}" > "\${BASE}/\${CG}"
log "backup compiler: \$(stat -c%s "\${BK}") bytes"

build() { # name srcdir
  local N="\$1" D="\$2" M="\${ROOT}/M_\$1.elf"
  ( cd "\${D}" && export SOUNIO_STDLIB_PATH="\${D}/stdlib" &&
    /usr/bin/timeout 2400 "\${BK}" build self-hosted/compiler/main.sio -o "\${M}" > "\${RES}/build_\$1.log" 2>&1 )
  local rc=\$? errs=\$(grep -c '^error:' "\${RES}/build_\$1.log" 2>/dev/null)
  log "build \$1: rc=\${rc} hard_errors=\${errs} size=\$(stat -c%s "\${M}" 2>/dev/null || echo 0)"
  [ \${errs} -gt 0 ] && grep '^error:' "\${RES}/build_\$1.log" | head -5 | sed 's/^/    /' >> "\${SUM}"
  [ -s "\${M}" ] && chmod +x "\${M}"
}
build fixed "\${FIX}"
build base  "\${BASE}"
log "binaries: \${ROOT}/M_fixed.elf \${ROOT}/M_base.elf"
log "Done."
EOF

${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
