#!/usr/bin/env bash
# #1 repair the self-build regression + #2 validate the frame fix, in one job.
#
# Root cause of the broken self-build: the bootstrap seed (bin/souc-lean-single-x86_64)
# MISPARSES `&programs[i]` (address-of array element) as a slice-borrow `&arr[a..b]` at
# module_frontend.sio:3809 & :3829 -> 2 hard E001 errors -> miscompiled Madaros front-half.
# These are the ONLY 2 hard errors in the build. Source is valid; the seed is too old.
#
# #1 repair (seed-compatible, applied READ-ONLY in this stage; operator's files untouched):
#   rewrite the 2 calls to copy the element into a local first, so no `[index]` sits under
#   the `&`:   let p = programs[i] ; f(&p, ...)
#
# #2 validation A/B on the repaired substrate (operator's working tree, which already has
#   the live-site frame fix at 6190 & 7333):
#   V_fix  = working tree + #1 repair (live frame fix present)
#   V_base = same + the two live frame-fix blocks reverted to `sub rsp,512`
#   Each: build (report hard-error count), compile+RUN hello and reproducer N=1/2/4/5.
#   Expect (if the fix is real + #1 works): V_base reproduces wrong-trail/segv at N=2/4/5,
#   V_fix passes all (pass=1 trail=5 conflict=1), and hello runs under V_fix.
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
JOB_TIME="${JOB_TIME:-00:50:00}"
RUN_ID="${RUN_ID:-madaros-repair-validate-$(date -u +%Y%m%dT%H%M%S)}"

REPRO="examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio"
test -f "${REPO}/${REPRO}" || { echo "missing reproducer" >&2; exit 1; }
test -x "${REPO}/bin/souc-lean-single-x86_64" || { echo "missing seed" >&2; exit 1; }

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }
echo "login=${LOGIN_POD} worker=${WORKER_POD} node=${NODELIST}"

TARBALL="/tmp/${RUN_ID}.tgz"; SBATCH="/tmp/${RUN_ID}.sbatch"
echo "packaging working-tree snapshot (read-only)"
tar -C "${REPO}" -czf "${TARBALL}" self-hosted stdlib bin/souc-lean-single-x86_64 "${REPRO}"
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
RES="/tmp/${RUN_ID}.result"; SUM="\${RES}/SUMMARY.txt"
rm -rf "\${ROOT}" "\${RES}"; mkdir -p "\${ROOT}" "\${RES}"; : > "\${SUM}"
log(){ echo "\$@" >> "\${SUM}"; }

MF="self-hosted/compiler/module_frontend.sio"
CG="self-hosted/native/codegen_x86_linux.sio"

# ---- unpack base substrate (V_fix source) ----
FIX="\${ROOT}/src_fix"; mkdir -p "\${FIX}"; tar -xzf "/tmp/${RUN_ID}.tgz" -C "\${FIX}"
chmod +x "\${FIX}/bin/souc-lean-single-x86_64"

# ---- #1 repair: assert + rewrite the two &programs[i] sites (seed-compatible) ----
A3809="\$(sed -n '3809p' "\${FIX}/\${MF}")"
A3829="\$(sed -n '3829p' "\${FIX}/\${MF}")"
log "module_frontend 3809: \${A3809}"
log "module_frontend 3829: \${A3829}"
if ! echo "\${A3809}" | grep -q 'module_frontend_lower_program_box_traced(&programs\[0\]' \
   || ! echo "\${A3829}" | grep -q 'module_frontend_lower_program_box_traced(&programs\[i\]'; then
  log "ABORT: module_frontend call sites not as expected; refusing to patch."
  exit 0
fi
awk '
NR==3809 { print "    let seed_prog_copy = programs[0]";
           print "    let seed_lower_result = module_frontend_lower_program_box_traced(&seed_prog_copy, 0, trace)"; next }
NR==3829 { print "        let cur_prog_copy = programs[i]";
           print "        let lower_result = module_frontend_lower_program_box_traced(&cur_prog_copy, i, trace)"; next }
1' "\${FIX}/\${MF}" > "\${FIX}/\${MF}.new" && mv "\${FIX}/\${MF}.new" "\${FIX}/\${MF}"
log ""
log "#1 repair applied — 3809-3810 & 3830-3831 now:"
sed -n '3809,3810p;3830,3831p' "\${FIX}/\${MF}" | sed 's/^/    /' >> "\${SUM}"

# assert live frame fix present at 6190 & 7333 (V_fix)
if ! sed -n '6190p' "\${FIX}/\${CG}" | grep -q 'align16((\*func).reg_count' \
   || ! sed -n '7333p' "\${FIX}/\${CG}" | grep -q 'align16((\*func).reg_count'; then
  log "ABORT: live frame fix not at 6190 & 7333 as expected."
  exit 0
fi

# ---- V_base source = copy of repaired tree with the two live frame-fix blocks reverted ----
BASE="\${ROOT}/src_base"; cp -a "\${FIX}" "\${BASE}"
awk '
NR==6190 { print "    nc_emit_sub_rsp_imm32(nc, 512)"; skip=3; next }
NR==7333 { print "    nc_emit_sub_rsp_imm32(nc, 512)"; skip=3; next }
skip>0   { skip--; next }
1' "\${FIX}/\${CG}" > "\${BASE}/\${CG}"

build() { # \$1=name \$2=srcdir ; writes \${ROOT}/m_\$1.elf
  local N="\$1" D="\$2" M="\${ROOT}/m_\$1.elf"
  ( cd "\${D}" && export SOUNIO_STDLIB_PATH="\${D}/stdlib" &&
    /usr/bin/timeout 1500 ./bin/souc-lean-single-x86_64 self-hosted/compiler/main.sio "\${M}" \
      > "\${RES}/build_\$1.log" 2>&1 )
  local rc=\$? errs=\$(grep -c '^error:' "\${RES}/build_\$1.log" 2>/dev/null)
  log ""
  log "=== build \$1: rc=\${rc} hard_errors=\${errs} size=\$(stat -c%s "\${M}" 2>/dev/null || echo 0) ==="
  [ \${errs} -gt 0 ] && grep '^error:' "\${RES}/build_\$1.log" | sed 's/^/    /' >> "\${SUM}"
  [ -s "\${M}" ] && chmod +x "\${M}"
}

printf 'fn main() -> i64 with IO {\n    println("hi")\n    0\n}\n' > "\${ROOT}/hello.sio"
test_run() { # \$1=name
  local N="\$1" M="\${ROOT}/m_\$1.elf"
  log ""
  log "########## TEST \$1 ##########"
  [ -s "\${M}" ] || { log "  (no madaros)"; return; }
  ( cd "\${FIX}" && export SOUNIO_STDLIB_PATH="\${FIX}/stdlib"
    /usr/bin/timeout 60 "\${M}" build "\${ROOT}/hello.sio" -o "\${ROOT}/h_\$N.elf" >/dev/null 2>&1
    log "  hello compile_rc=\$?"
    if [ -s "\${ROOT}/h_\$N.elf" ]; then chmod +x "\${ROOT}/h_\$N.elf"
      o=\$(/usr/bin/timeout 5 "\${ROOT}/h_\$N.elf" 2>&1); log "  hello run_rc=\$? out=[\${o}]"; fi
    for K in 1 2 4 5; do
      awk -v n=\$K '/emit_cube_assignment\(0, 0, 0, k\)/{for(i=0;i<n;i++)print;next}1' \
        "\${FIX}/${REPRO}" > "\${ROOT}/r\$K.sio"
      /usr/bin/timeout 120 "\${M}" build "\${ROOT}/r\$K.sio" -o "\${ROOT}/r_\$N_\$K.elf" \
        > "\${RES}/repro_\$N_\$K.log" 2>&1; crc=\$?
      if [ -s "\${ROOT}/r_\$N_\$K.elf" ]; then chmod +x "\${ROOT}/r_\$N_\$K.elf"
        o=\$(/usr/bin/timeout 10 "\${ROOT}/r_\$N_\$K.elf" 2>&1); rrc=\$?
        v=FAIL; echo "\${o}" | grep -q "pass=1 trail=5 conflict=1" && v=PASS
        log "  N=\$K compile_rc=\${crc} run_rc=\${rrc} VERDICT=\${v} out=[\${o}]"
      else log "  N=\$K compile_rc=\${crc} NO_ELF last=[\$(tail -1 "\${RES}/repro_\$N_\$K.log")]"; fi
    done )
}

build fix  "\${FIX}"
build base "\${BASE}"
test_run base
test_run fix
log ""
log "Done."
EOF

${KUBECTL} -n "${NS}" cp "${SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"
echo "submitting ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch"
echo "RUN_ID=${RUN_ID}"
rm -f "${TARBALL}" "${SBATCH}"
