#!/usr/bin/env bash
# One-command NUMERIC gate for the od256 octuple kernels on a real GPU.
#
# Pipeline:
#   1. host: gen fixtures (packed f64 inputs + mpmath truth) via the numeric gate
#   2. stage ptx/ + fixtures/ + kaxi_ptx_runner.c + run_numeric.sh to orangefs
#   3. worker: build the runner (dlopens libcuda), EXECUTE each kernel, dump the
#      output limbs back (results/<kernel>.out.f64)
#   4. host: fetch results, run the gate --check → bit-exact vs reference +
#      effective bits vs mpmath. Prints how to fetch + how to finish the check.
#
# Unlike submit-od256.sh (ptxas assembly acceptance), this runs the kernels and
# verifies the numbers. Kernels are tid-parallel (thread t at slot t*stride), so
# all CASES cases run in one launch with --threads CASES (keep CASES ≤ 1024).
#
# Usage (from the eisa worktree):
#   cd /workspace/sounio-eisa
#   bash slurm-jobs/kaxi-ptxas-accept/submit-od256-numeric.sh
#   CASES=128 bash slurm-jobs/kaxi-ptxas-accept/submit-od256-numeric.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CASES="${CASES:-64}"
[[ "$CASES" -le 1024 ]] || { echo "CASES=$CASES exceeds one block (1024)" >&2; exit 1; }

NS="${NS:-slurm-pilot}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_DEPLOY_NAME="${LOGIN_DEPLOY_NAME:-slurm-pilot-login-slinky}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
RUN_ID="${RUN_ID:-od256-numeric-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/kaxi-ptxas-accept/${RUN_ID}"
STAGE_LOCAL="${STAGE_LOCAL:-/tmp/od256_numeric_stage}"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:20:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"

# --- 1. stage payload locally ----------------------------------------------
rm -rf "$STAGE_LOCAL"; mkdir -p "$STAGE_LOCAL/ptx" "$STAGE_LOCAL/fixtures"
for p in od256_two_sum od256_two_prod od256_add od256_mul; do
  g="tests/golden/kaxi_ptx/od256/${p}.ptx"
  [[ -f "$g" ]] || { echo "missing golden $g — run the golden gate first" >&2; exit 1; }
  cp -f "$g" "$STAGE_LOCAL/ptx/${p}.ptx"
done
python3 scripts/ci/od256_numeric_gate.py --gen "$STAGE_LOCAL/fixtures" --cases "$CASES" ${ADVERSARIAL:+--adversarial}
cp -f scripts/gpu/kaxi_ptx_runner.c "$STAGE_LOCAL/kaxi_ptx_runner.c"
# gpu-orangefs nodes are driver-only (no gcc); ship a prebuilt binary. It dlopens
# libcuda at runtime, so it only needs a compatible glibc (worker: 2.39).
if command -v gcc >/dev/null 2>&1; then
  gcc -O2 -o "$STAGE_LOCAL/kaxi_ptx_runner" scripts/gpu/kaxi_ptx_runner.c -ldl -lm \
    && echo "prebuilt runner staged (glibc $(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1))" \
    || echo "warn: local runner build failed — worker will try to build from source" >&2
else
  echo "warn: no local gcc — worker must have a compiler or the run will fail" >&2
fi
cp -f slurm-jobs/kaxi-ptxas-accept/run_numeric.sh "$STAGE_LOCAL/run_numeric.sh"
chmod +x "$STAGE_LOCAL/run_numeric.sh"
echo "staged $CASES cases/kernel -> $STAGE_LOCAL"

command -v "${KUBECTL_BIN}" >/dev/null 2>&1 || { echo "kubectl not found — payload staged at $STAGE_LOCAL (submit manually)" >&2; exit 0; }

# --- 2. resolve login pod ---------------------------------------------------
LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods \
  -l "app.kubernetes.io/instance=${LOGIN_DEPLOY_NAME},app.kubernetes.io/name=login" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[[ -n "${LOGIN_POD}" ]] || LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods -l "${LOGIN_SELECTOR}" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[[ -n "${LOGIN_POD}" ]] || { echo "no live login pod in ${NS}" >&2; exit 1; }
echo "login pod: ${LOGIN_POD}"

# --- 3. ship payload --------------------------------------------------------
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'"
LOCAL_TARBALL="/tmp/${RUN_ID}.tgz"; rm -f "$LOCAL_TARBALL"
TAR_ITEMS=(ptx fixtures kaxi_ptx_runner.c run_numeric.sh)
[[ -x "$STAGE_LOCAL/kaxi_ptx_runner" ]] && TAR_ITEMS+=(kaxi_ptx_runner)
tar -C "${STAGE_LOCAL}" -czf "${LOCAL_TARBALL}" "${TAR_ITEMS[@]}"
if ! "${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_TARBALL}" "${LOGIN_POD}:${STAGE_ROOT}/payload.tgz" >/dev/null 2>&1; then
  cat "${LOCAL_TARBALL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${STAGE_ROOT}/payload.tgz'"
fi
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "tar -tzf '${STAGE_ROOT}/payload.tgz' >/dev/null"

# --- 4. build + submit sbatch ----------------------------------------------
SBATCH_LOCAL="/tmp/${RUN_ID}.sbatch"; rm -f "$SBATCH_LOCAL"
cat > "${SBATCH_LOCAL}" <<EOF
#!/usr/bin/env bash
#SBATCH -J od256-numeric
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -A ${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
EOF
[[ -n "${SBATCH_NODELIST}" ]] && echo "#SBATCH -w ${SBATCH_NODELIST}" >> "${SBATCH_LOCAL}"
cat >> "${SBATCH_LOCAL}" <<'SBATCH_BODY'
set -uo pipefail
RUN_ROOT='RUN_ROOT_PLACEHOLDER'
LOCAL_ROOT="${TMPDIR:-/tmp}/od256-numeric-${SLURM_JOB_ID}"
RESULTS_DIR="${RUN_ROOT}/results"; LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${LOCAL_ROOT}"
exec > >(tee "${LOG_DIR}/job-${SLURM_JOB_ID}.log") 2>&1
echo "=== od256 numeric — $(date) host=$(hostname) ==="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
tar -xzf "${RUN_ROOT}/payload.tgz" -C "${LOCAL_ROOT}"
chmod +x "${LOCAL_ROOT}/run_numeric.sh"
cd "${LOCAL_ROOT}"
bash "${LOCAL_ROOT}/run_numeric.sh" "${LOCAL_ROOT}/ptx" "${LOCAL_ROOT}/fixtures" "${RESULTS_DIR}" "${LOCAL_ROOT}/kaxi_ptx_runner.c"
exit $?
SBATCH_BODY
sed -i "s|RUN_ROOT_PLACEHOLDER|${STAGE_ROOT}|g" "${SBATCH_LOCAL}"

SBATCH_FILE="/tmp/${RUN_ID}.remote.sbatch"
"${KUBECTL_BIN}" -n "${NS}" cp "${SBATCH_LOCAL}" "${LOGIN_POD}:${SBATCH_FILE}" >/dev/null 2>&1 || \
  cat "${SBATCH_LOCAL}" | "${KUBECTL_BIN}" -n "${NS}" exec -i "${LOGIN_POD}" -- sh -lc "cat > '${SBATCH_FILE}'"
SBATCH_OUTPUT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${SBATCH_FILE}'; rm -f '${SBATCH_FILE}'")"
echo "${SBATCH_OUTPUT}"
JOB_ID="$(printf '%s\n' "${SBATCH_OUTPUT}" | awk '/Submitted batch job/ {print $4; exit}')"

cat <<EOF

RUN_ID=${RUN_ID}
JOB_ID=${JOB_ID:-?}
STAGE_ROOT=${STAGE_ROOT}
LOGIN_POD=${LOGIN_POD}

# when done, fetch outputs + finish the numeric check on the host:
D=/tmp/${RUN_ID}.results; mkdir -p "\$D"
kubectl -n ${NS} exec ${LOGIN_POD} -- bash -lc "cd ${STAGE_ROOT}/results && tar -cz ." | tar -xz -C "\$D"
python3 scripts/ci/od256_numeric_gate.py --check "\$D"
EOF
