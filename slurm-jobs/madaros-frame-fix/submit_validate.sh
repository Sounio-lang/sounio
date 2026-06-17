#!/usr/bin/env bash
# A/B validation of the PROPOSED frame fix on a working-front-half substrate.
#
# Substrate = the operator's CURRENT WORKING TREE (main + uncommitted WIP incl. the
# module_frontend.sio front-half fix), packaged READ-ONLY (tar reads; never writes).
# This is the only substrate on which the reproducer compiles at all (clean main HEAD
# SIGSEGVs in the front-half — see docs/audit/FRAME_FIX_7fa3c3524_DEAD_CODE_2026-06-16.md).
#
# Two Madaros are built from the SAME packaged source:
#   V_base : as-is (live emitter compile_ir_function_v2_core_ir_into:6190 = `sub rsp,512`)
#   V_fix  : line 6190 patched to align16(((*func).reg_count+1)*8)  [the proposal]
# Only line 6190 is patched (isolates the change; other 512 sites left alone).
#
# For each variant: compile+RUN hello and the reproducer at N=1/2/4/5 (N-variants made
# with awk, NOT raw-newline sed — the original run's sed produced empty files). Grade on
# `pass=1 trail=5 conflict=1`. V_base hello.elf + repro_n1.elf are preserved for disasm
# (to trace which emitter the reproducer's main actually uses: 6190 vs driver path).
set -euo pipefail

REPO="${REPO:-/workspace/sounio}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpu-orangefs}"
ACCOUNT="${ACCOUNT:-plruntime}"
QOS="${QOS:-gpuorangefs}"
NODELIST="${NODELIST:-gpuorangefs-r770-proxmox}"
WORKER_POD="${WORKER_POD:-slurm-pilot-worker-gpuorangefs-zj2l4}"
WORKER_CTR="${WORKER_CTR:-slurmd}"
JOB_MEM="${JOB_MEM:-24G}"
JOB_CPUS="${JOB_CPUS:-4}"
JOB_TIME="${JOB_TIME:-00:45:00}"
RUN_ID="${RUN_ID:-madaros-frame-validate-$(date -u +%Y%m%dT%H%M%S)}"

REPRO="examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"
test -f "${REPO}/${REPRO}" || { echo "missing reproducer ${REPO}/${REPRO}" >&2; exit 1; }
test -x "${REPO}/bin/souc-lean-single-x86_64" || { echo "missing seed" >&2; exit 1; }

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
echo "login=${LOGIN_POD}  worker=${WORKER_POD} (node ${NODELIST})"

TARBALL="/tmp/${RUN_ID}.tgz"
SBATCH="/tmp/${RUN_ID}.sbatch"
echo "packaging WORKING-TREE snapshot (read-only) -> ${TARBALL}"
tar -C "${REPO}" -czf "${TARBALL}" \
  self-hosted stdlib bin/souc-lean-single-x86_64 "${REPRO}"
echo "payload: $(stat -c%s "${TARBALL}" | awk '{printf "%.1f MB\n",$1/1048576}')"
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
#SBATCH -w ${NODELIST}
#SBATCH -o /tmp/${RUN_ID}.slurmout
#SBATCH -e /tmp/${RUN_ID}.slurmout
set -uo pipefail

ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
RES="/tmp/${RUN_ID}.result"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${ROOT}" "\${RES}"
SUM="\${RES}/SUMMARY.txt"
: > "\${SUM}"

log(){ echo "\$@" | tee -a "\${SUM}"; }

log "host=\$(hostname) job=\${SLURM_JOB_ID:-manual}"; free -g | head -2 | tee -a "\${SUM}"

SRC="\${ROOT}/src"; mkdir -p "\${SRC}"
tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${SRC}"
chmod +x "\${SRC}/bin/souc-lean-single-x86_64"

CG="self-hosted/native/codegen_x86_linux.sio"

# The working tree ALREADY carries the live-site fix at 6190 (compile_ir_function_v2_core_ir_into)
# and 7333 (native_v2_core_begin_fn_spill_into). Assert that, then build:
#   V_fix  = working tree as-is (fix present)
#   V_base = copy with both live sites reverted to `sub rsp,512` (the pre-fix baseline)
L6190="\$(sed -n '6190p' "\${SRC}/\${CG}")"
L7333="\$(sed -n '7333p' "\${SRC}/\${CG}")"
log ""
log "line 6190: \${L6190}"
log "line 7333: \${L7333}"
if ! echo "\${L6190}" | grep -q "align16((\*func).reg_count" || ! echo "\${L7333}" | grep -q "align16((\*func).reg_count"; then
  log "ABORT: live fix not present at 6190 & 7333 as expected — source drifted; refusing to guess."
  exit 0
fi

build_variant() {
  local NAME="\$1"; local SRCDIR="\$2"
  local M="\${ROOT}/madaros_\${NAME}.elf"
  log ""
  log "=== build \${NAME} ==="
  ( cd "\${SRCDIR}" && export SOUNIO_STDLIB_PATH="\${SRCDIR}/stdlib" &&
    /usr/bin/timeout 1500 ./bin/souc-lean-single-x86_64 self-hosted/compiler/main.sio "\${M}" \
      > "\${RES}/build_\${NAME}.log" 2>&1 )
  local rc=\$?
  local errs=\$(grep -c "^error:" "\${RES}/build_\${NAME}.log" 2>/dev/null || echo "?")
  log "  build_rc=\${rc}  size=\$(stat -c%s "\${M}" 2>/dev/null || echo 0)  source_errors=\${errs}"
  [ -s "\${M}" ] && chmod +x "\${M}" && echo "\${M}" || echo ""
}

# V_fix = packaged working tree as-is (live fix present)
M_FIX=\$(build_variant fix "\${SRC}")

# V_base = copy with the 4-line dynamic-frame block reverted to a single sub rsp,512 at both sites.
# Each live site is exactly: let frame_sz=...(6190/7333) ; if frame_sz>0 { ; sub ; }  -> 4 lines.
BASEDIR="\${ROOT}/src_base"; cp -a "\${SRC}" "\${BASEDIR}"
awk '
NR==6190 { print "    nc_emit_sub_rsp_imm32(nc, 512)"; skip=3; next }
NR==7333 { print "    nc_emit_sub_rsp_imm32(nc, 512)"; skip=3; next }
skip>0   { skip--; next }
1' "\${SRC}/\${CG}" > "\${BASEDIR}/\${CG}"
log ""
log "V_base revert check — 6190 & 7333 region:"
sed -n '6188,6192p' "\${BASEDIR}/\${CG}" | sed 's/^/    /' | tee -a "\${SUM}"
# sanity: V_base must have NO align16 reg_count frame left at the two live sites and must still parse-balance
if grep -q "align16((\*func).reg_count" "\${BASEDIR}/\${CG}"; then
  log "  NOTE: align16 reg_count frame still present elsewhere (dead fn 6572) — expected; live sites reverted."
fi
M_BASE=\$(build_variant base "\${BASEDIR}")

# ── test harness ───────────────────────────────────────────────────────────────
printf 'fn main() -> i64 with IO {\n    println("hi")\n    0\n}\n' > "\${ROOT}/hello.sio"

test_variant() {
  local NAME="\$1"; local M="\$2"; local KEEP="\$3"
  log ""
  log "########## TEST \${NAME} ##########"
  if [ -z "\${M}" ] || [ ! -s "\${M}" ]; then log "  (no madaros built; skip)"; return; fi
  export SOUNIO_STDLIB_PATH="\${SRC}/stdlib"
  ( cd "\${SRC}"
    # hello
    /usr/bin/timeout 60 "\${M}" build "\${ROOT}/hello.sio" -o "\${ROOT}/hello_\${NAME}.elf" > "\${RES}/hello_\${NAME}_c.log" 2>&1
    log "  hello compile_rc=\$?"
    if [ -s "\${ROOT}/hello_\${NAME}.elf" ]; then
      chmod +x "\${ROOT}/hello_\${NAME}.elf"
      out=\$(/usr/bin/timeout 5 "\${ROOT}/hello_\${NAME}.elf" 2>&1); rc=\$?
      log "  hello run_rc=\${rc} out=[\${out}]"
    else log "  hello NO ELF"; fi
    # reproducer N variants
    for N in 1 2 4 5; do
      awk -v n="\${N}" '/emit_cube_assignment\(0, 0, 0, k\)/{for(i=0;i<n;i++)print;next}1' \
        "\${SRC}/${REPRO}" > "\${ROOT}/repro_n\${N}.sio"
      cc=\$(grep -c "emit_cube_assignment(0, 0, 0, k)" "\${ROOT}/repro_n\${N}.sio")
      /usr/bin/timeout 120 "\${M}" build "\${ROOT}/repro_n\${N}.sio" -o "\${ROOT}/repro_\${NAME}_n\${N}.elf" \
        > "\${RES}/repro_\${NAME}_n\${N}_c.log" 2>&1
      crc=\$?
      if [ -s "\${ROOT}/repro_\${NAME}_n\${N}.elf" ]; then
        chmod +x "\${ROOT}/repro_\${NAME}_n\${N}.elf"
        out=\$(/usr/bin/timeout 10 "\${ROOT}/repro_\${NAME}_n\${N}.elf" 2>&1); rrc=\$?
        verdict=FAIL; echo "\${out}" | grep -q "pass=1 trail=5 conflict=1" && verdict=PASS
        log "  N=\${N}(calls=\${cc}) compile_rc=\${crc} run_rc=\${rrc} VERDICT=\${verdict} out=[\${out}]"
      else
        tl=\$(tail -1 "\${RES}/repro_\${NAME}_n\${N}_c.log" 2>/dev/null)
        log "  N=\${N}(calls=\${cc}) compile_rc=\${crc} NO_ELF last=[\${tl}]"
      fi
    done )
  if [ -n "\${KEEP}" ]; then
    cp -f "\${ROOT}/hello_\${NAME}.elf" "\${RES}/hello_\${NAME}.elf" 2>/dev/null || true
    cp -f "\${ROOT}/repro_\${NAME}_n1.elf" "\${RES}/repro_\${NAME}_n1.elf" 2>/dev/null || true
    cp -f "\${ROOT}/repro_\${NAME}_n5.elf" "\${RES}/repro_\${NAME}_n5.elf" 2>/dev/null || true
  fi
}

test_variant base "\${M_BASE}" keep
test_variant fix  "\${M_FIX}"  keep

log ""
log "Done."
EOF

${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
