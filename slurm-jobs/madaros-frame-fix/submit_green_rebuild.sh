#!/usr/bin/env bash
# #2 validation via GREEN REBUILD. Current source can't self-build a working Madaros, but
# Madaros was green at 17d1157be (2026-06-14) which PREDATES the &programs[i] seed-misparse
# regression. So the seed can build 17d1157be into a working Madaros. We build two:
#   M_base = 17d1157be as-is (live emitters reserve `sub rsp,512`)  -> expect the bug
#   M_fix  = 17d1157be + dynamic frame cherry-picked at the two LIVE sites:
#            compile_ir_function_v2_core_ir_into (line 5445) and
#            native_v2_core_begin_fn_spill_into  (line 6354)
#            -> `let frame_sz = align16((*func).reg_count*8); if >0 { sub rsp, frame_sz }`
# Source+stdlib are era-matched (git archive 17d1157be). Seed = current working-tree seed.
# Binaries preserved; reproducer A/B run manually afterward.
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"; GREEN="${GREEN:-17d1157be}"
NS="${NS:-slurm-pilot}"; KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpu-orangefs}"; ACCOUNT="${ACCOUNT:-plruntime}"; QOS="${QOS:-gpuorangefs}"
NODELIST="${NODELIST:-gpuorangefs-r770-proxmox}"
WORKER_POD="${WORKER_POD:-slurm-pilot-worker-gpuorangefs-zj2l4}"; WORKER_CTR="${WORKER_CTR:-slurmd}"
JOB_MEM="${JOB_MEM:-24G}"; JOB_CPUS="${JOB_CPUS:-6}"; JOB_TIME="${JOB_TIME:-00:50:00}"
RUN_ID="${RUN_ID:-madaros-green-rebuild-$(date -u +%Y%m%dT%H%M%S)}"
REPRO="examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"

test -x "${REPO}/bin/souc-lean-single-x86_64" || { echo "no seed" >&2; exit 1; }
test -f "${REPO}/${REPRO}" || { echo "no reproducer" >&2; exit 1; }
LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
echo "login=${LOGIN_POD} worker=${WORKER_POD} green=${GREEN}"

# era-matched committed source/stdlib from the green commit; seed + reproducer from current tree
STAGE="/tmp/${RUN_ID}.stage"; rm -rf "${STAGE}"; mkdir -p "${STAGE}"
git -C "${REPO}" archive --format=tar "${GREEN}" self-hosted stdlib | tar -x -C "${STAGE}"
cp "${SEED_BIN:-${REPO}/bin/souc-lean-single-x86_64}" "${STAGE}/seed"
mkdir -p "${STAGE}/$(dirname "${REPRO}")"; cp "${REPO}/${REPRO}" "${STAGE}/${REPRO}"
TARBALL="/tmp/${RUN_ID}.tgz"
tar -C "${STAGE}" -czf "${TARBALL}" self-hosted stdlib seed "${REPRO}"; rm -rf "${STAGE}"
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

BASE="\${ROOT}/src_base"; mkdir -p "\${BASE}"; tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${BASE}"
chmod +x "\${BASE}/seed"
# assert the two live sub-512 sites
if ! sed -n '5445p' "\${BASE}/\${CG}" | grep -q 'nc_emit_sub_rsp_imm32(nc, 512)' \
   || ! sed -n '6354p' "\${BASE}/\${CG}" | grep -q 'nc_emit_sub_rsp_imm32(nc, 512)'; then
  log "ABORT: 5445/6354 are not the expected sub-512 lines at ${GREEN}."
  sed -n '5445p;6354p' "\${BASE}/\${CG}" | sed 's/^/    /' >> "\${SUM}"; exit 0; fi

# M_fix source = dynamic frame cherry-picked at both live sites
FIX="\${ROOT}/src_fix"; cp -a "\${BASE}" "\${FIX}"
awk '
NR==5445 || NR==6354 {
  print "    let frame_sz = align16((*func).reg_count * 8)";
  print "    if frame_sz > 0 {";
  print "        nc_emit_sub_rsp_imm32(nc, frame_sz)";
  print "    }"; next }
1' "\${BASE}/\${CG}" > "\${FIX}/\${CG}"
log "cherry-pick applied at 5445 & 6354. 5445-5448 now:"
sed -n '5445,5448p' "\${FIX}/\${CG}" | sed 's/^/    /' >> "\${SUM}"

build() { # name srcdir
  local N="\$1" D="\$2" M="\${ROOT}/M_\$1.elf"
  ( cd "\${D}" && export SOUNIO_STDLIB_PATH="\${D}/stdlib" &&
    /usr/bin/timeout 1800 ./seed self-hosted/compiler/main.sio "\${M}" > "\${RES}/build_\$1.log" 2>&1 )
  local rc=\$? errs=\$(grep -c '^error:' "\${RES}/build_\$1.log" 2>/dev/null)
  log "build \$1: rc=\${rc} hard_errors=\${errs} size=\$(stat -c%s "\${M}" 2>/dev/null || echo 0)"
  [ \${errs} -gt 0 ] && grep '^error:' "\${RES}/build_\$1.log" | head -5 | sed 's/^/    /' >> "\${SUM}"
  [ -s "\${M}" ] && chmod +x "\${M}"
}
build base "\${BASE}"
build fix  "\${FIX}"
log "binaries: \${ROOT}/M_base.elf \${ROOT}/M_fix.elf  (src_base reproducer at \${BASE}/${REPRO})"
log "Done."
EOF
${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
