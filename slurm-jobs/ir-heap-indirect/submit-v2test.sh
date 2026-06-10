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
#SBATCH -N 1 -n 1 -c 2 --mem=38G --time=00:50:00
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
rm -f /tmp/nv2dbg_preflight /tmp/nv2dbg_combine /tmp/nv2dbg_parse /tmp/nv2dbg_summary /tmp/nv2dbg_bodies /tmp/nv2_combined_multimodule.sio /tmp/nv2dbg_afterlex /tmp/nv2dbg_tc_saturated /tmp/nv2dbg_afterparse /tmp/nv2dbg_beforereverse /tmp/nv2dbg_afterreverse
# ALLOCATOR HYPOTHESIS TEST: emit_heap_alloc_x86 does mmap-per-Box with NO MAP_FAILED check.
# If the parse SIGSEGV is VMA exhaustion (vm.max_map_count, default ~65530), raising the
# limit should let it through. Record before/after; try to raise (needs privilege).
MMC_BEFORE=\$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo "?")
echo 4000000 > /proc/sys/vm/max_map_count 2>/dev/null && MMC_AFTER=\$(cat /proc/sys/vm/max_map_count) || MMC_AFTER="RAISE_FAILED(\${MMC_BEFORE})"
echo "VM_MAX_MAP_COUNT: before=\${MMC_BEFORE} after=\${MMC_AFTER}" | tee -a "\${RES}/SUMMARY.txt"
echo "[v2] mc --native-v2-compile main.sio under RSS+VMA poller ..." | tee -a "\${RES}/SUMMARY.txt"
"\${ROOT}/mc.elf" --native-v2-compile self-hosted/compiler/main.sio -o "\${ROOT}/v2.elf" > "\${RES}/v2.log" 2>&1 &
GPID=\$!; PEAK=0; PEAKVMA=0; TICK=0; TRAJ="\${RES}/rss_traj.txt"; : > "\${TRAJ}"
while kill -0 \${GPID} 2>/dev/null; do
  HWM=\$(awk '/^VmHWM:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  RSS=\$(awk '/^VmRSS:/{print \$2}' /proc/\${GPID}/status 2>/dev/null)
  [ -n "\${HWM:-}" ] && [ "\${HWM}" -gt "\${PEAK}" ] 2>/dev/null && PEAK=\${HWM}
  VMA=\$(wc -l < /proc/\${GPID}/maps 2>/dev/null)
  [ -n "\${VMA:-}" ] && [ "\${VMA}" -gt "\${PEAKVMA}" ] 2>/dev/null && PEAKVMA=\${VMA}
  TICK=\$((TICK+1))
  if [ \$((TICK % 7)) -eq 0 ]; then
    echo "t=\${TICK} rss_mb=\$(( \${RSS:-0}/1024 )) hwm_mb=\$(( \${PEAK}/1024 )) parse=\$([ -f /tmp/nv2dbg_parse ]&&echo Y||echo N) summary=\$([ -f /tmp/nv2dbg_summary ]&&echo Y||echo N) bodies=\$([ -f /tmp/nv2dbg_bodies ]&&echo Y||echo N)" >> "\${TRAJ}"
  fi
  sleep 0.3
done
wait \${GPID}; V2RC=\$?
{
  echo "V2_COMPILE_RC=\${V2RC}  (0=ok, 137=OOM, 139=SIGSEGV)"
  echo "STAGE_REACHED: preflight=\$([ -f /tmp/nv2dbg_preflight ] && echo Y || echo N) combinemk=\$([ -f /tmp/nv2dbg_combine ] && echo Y || echo N) combined=\$([ -f /tmp/nv2_combined_multimodule.sio ] && wc -c </tmp/nv2_combined_multimodule.sio || echo MISSING) parse=\$([ -f /tmp/nv2dbg_parse ] && echo Y || echo N) summary=\$([ -f /tmp/nv2dbg_summary ] && echo Y || echo N) bodies=\$([ -f /tmp/nv2dbg_bodies ] && echo Y || echo N)"
  echo "PARSE_LADDER: afterlex=\$([ -f /tmp/nv2dbg_afterlex ] && echo Y || echo N) tc_saturated=\$([ -f /tmp/nv2dbg_tc_saturated ] && echo Y || echo N) beforereverse=\$([ -f /tmp/nv2dbg_beforereverse ] && echo Y || echo N) afterreverse=\$([ -f /tmp/nv2dbg_afterreverse ] && echo Y || echo N) afterparse=\$([ -f /tmp/nv2dbg_afterparse ] && echo Y || echo N)"
  if [ -f /tmp/nv2_combined_multimodule.sio ]; then
    echo "COMBINED_DIAG: residual_module_decls=\$(grep -cE '^(module|mod) ' /tmp/nv2_combined_multimodule.sio) fn_defs=\$(grep -cE '^(pub )?fn ' /tmp/nv2_combined_multimodule.sio)  (expect residual~0, fn~6642)"
  fi
  echo "PEAK_HWM_MB=\$((PEAK/1024))  PEAK_VMA_COUNT=\${PEAKVMA}  (vm.max_map_count=\${MMC_AFTER})"
  echo "--- RSS TRAJECTORY (every ~2s; watch climb vs plateau + stage) ---"
  tail -32 "\${TRAJ}" 2>/dev/null
  echo "v2.elf size: \$(stat -c%s "\${ROOT}/v2.elf" 2>/dev/null || echo 0)"
  echo "--- front-half: Merged IR / errors ---"; grep -iE "NV2DBG|Merged IR:|error|fail|successful|REJECT" "\${RES}/v2.log" | grep -v ITEMERR | head -12
  echo "--- ITEMERR distribution (top-level items with parse errors; cline=combined-file line) ---"
  grep -c ITEMERR "\${RES}/v2.log" | sed 's/^/ITEMERR_total_items=/'
  grep ITEMERR "\${RES}/v2.log" | head -50
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
