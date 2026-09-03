#!/usr/bin/env bash
# Submit the full EISA validation battery (tests + conformance gate) as a
# CPU Slurm job, keeping the workspace pod free of heavy validation load.
#
# Topology (measured 2026-07-06):
#   - sbatch/srun work directly from the workspace pod; partition `all`.
#   - Compute nodes do NOT mount /workspace.
#   - OrangeFS (/orangefs/training) is NOT trustworthy for this flow: files
#     written by sbatch jobs read back NUL-padded from later sessions, and
#     raw binary through the srun stdin pipe arrives with a different sha
#     (probes 2026-07-06, jobs 5119/5138; see
#     docs/audit/CI_GATE_PORTABILITY_2026-07-06.md §5 for the cp variant).
#   - Node-local /tmp IS stable and persists across jobs on the same node,
#     so the whole flow is pinned to one node and uses its /tmp:
#       stage in : base64 tarball | srun > /tmp (7-bit clean pipe)
#       run      : sbatch --nodelist=<node>, unpack + run in /tmp
#       fetch    : srun --nodelist=<node> cat /tmp results
#
# The battery: every tests/stdlib/eisa/*.sio + tests/stdlib/math/test_qd128*
# + test_dd64* on the lean_single lane, then the bridge conformance gate.
#
# Usage:
#   bash slurm-jobs/eisa/submit-eisa-battery.sh            # submit
#   bash slurm-jobs/eisa/submit-eisa-battery.sh <run-id>   # fetch results
#
# Env:
#   SBATCH_PARTITION (default: all)
#   SBATCH_NODELIST  (default: gpuorangefs-5860-proxmox; must be ONE node)
#   JOB_TIME         (default: 00:20:00)
#   JOB_MEM          (default: 2G)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_PARTITION="${SBATCH_PARTITION:-all}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-5860-proxmox}"
JOB_TIME="${JOB_TIME:-00:20:00}"
JOB_MEM="${JOB_MEM:-2G}"

# ---------- fetch mode ----------
if [[ $# -ge 1 ]]; then
  RUN_ID="$1"
  echo "=== fetching ${RUN_ID} (node ${SBATCH_NODELIST}) ==="
  srun --partition="${SBATCH_PARTITION}" --nodelist="${SBATCH_NODELIST}" --time=00:02:00 --mem=256M \
    bash -c "cat '/tmp/${RUN_ID}.out/SUMMARY.txt' 2>/dev/null || echo '(no SUMMARY yet)'; echo; tail -40 '/tmp/${RUN_ID}.out/battery.log' 2>/dev/null || true"
  exit 0
fi

RUN_ID="eisa-battery-$(date -u +%Y%m%dT%H%M%S)"
STAGE_TGZ="/tmp/${RUN_ID}.tgz"

echo "[1/3] packing workspace subset (${ROOT_DIR})"
tar -C "${ROOT_DIR}" -czf "${STAGE_TGZ}" \
  bin/souc bin/souc-lean-single-x86_64 scripts/lib \
  stdlib \
  tools/eisa \
  scripts/ci/eisa_bridge_conformance_gate.sh \
  tests/stdlib/eisa tests/stdlib/math
echo "  tarball: $(du -h "${STAGE_TGZ}" | cut -f1)"
TGZ_SHA="$(sha256sum "${STAGE_TGZ}" | cut -d' ' -f1)"

echo "[2/3] staging to ${SBATCH_NODELIST}:/tmp/${RUN_ID}.tgz (base64 in transit)"
base64 "${STAGE_TGZ}" | srun --partition="${SBATCH_PARTITION}" --nodelist="${SBATCH_NODELIST}" \
  --time=00:05:00 --mem=512M \
  bash -c "base64 -d > '/tmp/${RUN_ID}.tgz' && sha256sum '/tmp/${RUN_ID}.tgz' | cut -d' ' -f1"
rm -f "${STAGE_TGZ}"
echo "  local sha256: ${TGZ_SHA} (compare with staged sha above)"

echo "[3/3] submitting batch job"
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_ID}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --nodelist=${SBATCH_NODELIST}
#SBATCH --time=${JOB_TIME}
#SBATCH --mem=${JOB_MEM}
#SBATCH --output=/tmp/${RUN_ID}.slurmout
set -uo pipefail
WORK=/tmp/${RUN_ID}.work
OUT=/tmp/${RUN_ID}.out
mkdir -p "\${WORK}" "\${OUT}"

# Verify the staged tarball before unpacking (fail loud, not empty-run).
got_sha=\$(sha256sum /tmp/${RUN_ID}.tgz 2>/dev/null | cut -d' ' -f1)
if [[ "\${got_sha}" != "${TGZ_SHA}" ]]; then
  echo "run: ${RUN_ID}" > "\${OUT}/SUMMARY.txt"
  echo "FAIL staging: tarball sha mismatch (want ${TGZ_SHA} got \${got_sha})" >> "\${OUT}/SUMMARY.txt"
  cat "\${OUT}/SUMMARY.txt"
  exit 125
fi

tar -C "\${WORK}" -xzf /tmp/${RUN_ID}.tgz
cd "\${WORK}"
export SOUNIO_STDLIB_PATH="\${WORK}/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
export TMPDIR="\${WORK}/tmpdir"
mkdir -p "\${TMPDIR}"

# Preflight (fail-fast before burning compute): tools the battery/gate need.
# anti-vacuity needs grep -a, not strings — verified on gpuorangefs-* 2026-07-06.
for _need in bash tar diff grep sed sort head mktemp chmod; do
  command -v "\${_need}" >/dev/null || { echo "FAIL preflight: missing \${_need}"; exit 127; }
done

pass=0; fail=0
: > "\${OUT}/battery.log"
for t in tests/stdlib/eisa/*.sio tests/stdlib/math/test_qd128_core.sio tests/stdlib/math/test_qd128_rump.sio tests/stdlib/math/test_dd64_cancellation.sio tests/stdlib/math/test_dd64_eft_exact.sio tests/stdlib/math/test_dd64_algebra.sio; do
  [[ -f "\$t" ]] || continue
  ./bin/souc run "\$t" > /tmp/one.log 2>&1
  rc=\$?
  last=\$(grep -v receipt /tmp/one.log | tail -1)
  if [[ \$rc -eq 0 ]]; then pass=\$((pass+1)); st=PASS; else fail=\$((fail+1)); st=FAIL; fi
  echo "\${st} rc=\${rc} \${t} :: \${last}" >> "\${OUT}/battery.log"
done

bash scripts/ci/eisa_bridge_conformance_gate.sh > "\${OUT}/gate.log" 2>&1
grc=\$?
lanes=\$(grep -c '^PASS' "\${OUT}/gate.log")
if [[ \$grc -eq 0 ]]; then gst=PASS; else gst=FAIL; fail=\$((fail+1)); fi

{
  echo "run: ${RUN_ID}"
  echo "host: \$(hostname)  date: \$(date -u +%FT%TZ)"
  echo "tests: pass=\${pass} fail=\${fail}"
  echo "gate: \${gst} rc=\${grc} lanes=\${lanes}"
} > "\${OUT}/SUMMARY.txt"
cat "\${OUT}/SUMMARY.txt"
rm -rf "\${WORK}" /tmp/${RUN_ID}.tgz
[[ \$fail -eq 0 ]]
EOF
sbatch "${SBATCH_FILE}"
echo
echo "monitor:  sacct --name=${RUN_ID} --format=JobID,State,ExitCode"
echo "fetch:    bash slurm-jobs/eisa/submit-eisa-battery.sh ${RUN_ID}"
