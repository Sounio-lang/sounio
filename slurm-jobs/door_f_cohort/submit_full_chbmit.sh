#!/usr/bin/env bash
# Door F full CHB-MIT cohort — sedenion associator probe on 24 patients.
#
# Pattern: single sbatch on cpu-ops partition, runs 24 souc invocations
# in parallel via xargs -P. Generates per-patient .sio files from EDFs
# on /orangefs/training/sounio/chbmit/ using a pyedflib venv on the login
# pod before submission. Aggregator runs inside the same sbatch as the
# final phase.
#
# Usage (from Sounio workspace):
#   cd /workspace/sounio
#   bash slurm-jobs/door_f_cohort/submit_full_chbmit.sh
#
# Prerequisites (auto-checked):
#   - kubectl access to namespace slurm-pilot
#   - EDFs staged at /orangefs/training/sounio/chbmit/ (see fetch_edfs.sh)
#   - /opt/sounio-venv on login pod with pyedflib+numpy
#
# Output:
#   /orangefs/training/sounio/door-f-runs/<RUN_ID>/results/door_f_<pat>.stdout  (x24)
#   /orangefs/training/sounio/door-f-runs/<RUN_ID>/results/door_f_cohort.tsv
#   /orangefs/training/sounio/door-f-runs/<RUN_ID>/logs/job-<ID>.log

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
PYTHON_VENV="${PYTHON_VENV:-/opt/sounio-venv/bin/python}"
EDF_DIR="${EDF_DIR:-/orangefs/training/sounio/chbmit}"
RUN_ID="${RUN_ID:-door-f-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/door-f-runs/${RUN_ID}"
MANIFEST_REL="scripts/research/door_f_cohort/chbmit_manifest.tsv"
PARALLEL="${PARALLEL:-8}"

REQUIRED=(
    "bin/souc"
    "bin/souc-native"
    "artifacts/self-hosted/souc-self-hosted-x86_64"
    "scripts/research/door_f_cohort/generate.py"
    "scripts/research/door_f_cohort/header.sio.part"
    "scripts/research/door_f_cohort/template.sio.part"
    "scripts/research/door_f_cohort/aggregate.py"
    "${MANIFEST_REL}"
    "stdlib/math/sedenion.sio"
    "stdlib/math/octonion.sio"
)
for rel in "${REQUIRED[@]}"; do
    if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
        echo "missing: ${SOUNIO_DIR}/${rel}" >&2
        exit 2
    fi
done

LOGIN_POD="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=login \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "${LOGIN_POD}" ]]; then
    echo "no live login pod" >&2; exit 3
fi
echo "login pod:  ${LOGIN_POD}"
echo "run id:     ${RUN_ID}"
echo "stage root: ${STAGE_ROOT}"

echo
echo "[1/5] verifying EDF cohort on /orangefs ..."
kubectl -n "${NS}" cp "${SOUNIO_DIR}/${MANIFEST_REL}" "${LOGIN_POD}:/tmp/_manifest.tsv"
MISSING=$(kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    awk -F'\\t' '\$1!~/^#/ && \$1!=\"patient\" && \$1!=\"\" {print \$2}' /tmp/_manifest.tsv | while read fn; do
        test -f '${EDF_DIR}'/\"\$fn\" || echo \"\$fn\"
    done
" | tr -d '\r')
if [[ -n "${MISSING}" ]]; then
    echo "missing EDFs on ${EDF_DIR}:" >&2
    echo "${MISSING}" >&2
    echo "run the login-pod fetch first (see docs in slurm-jobs/door_f_cohort/README.md)" >&2
    exit 4
fi
echo "  all 24 EDFs present"

echo
echo "[2/5] staging repo snapshot to ${STAGE_ROOT}/repo ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${STAGE_ROOT}/repo' '${STAGE_ROOT}/sio' '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'"
( cd "${SOUNIO_DIR}" && tar cf - \
    bin/souc \
    bin/souc-native \
    artifacts/self-hosted/souc-self-hosted-x86_64 \
    stdlib \
    scripts/research/door_f_cohort \
) | kubectl -n "${NS}" exec -i "${LOGIN_POD}" -- bash -c "tar xf - -C '${STAGE_ROOT}/repo'"
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "chmod +x '${STAGE_ROOT}/repo/bin/souc' '${STAGE_ROOT}/repo/bin/souc-native' '${STAGE_ROOT}/repo/artifacts/self-hosted/souc-self-hosted-x86_64'"

echo
echo "[3/5] generating 24 patient .sio files on login pod ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -e
    cd '${STAGE_ROOT}/repo'
    awk -F'\\t' '\$1!~/^#/ && \$1!=\"patient\" && \$1!=\"\" {printf \"%s\\t%s\\t%s\\n\", \$1, \$2, \$3}' \
        '${STAGE_ROOT}/repo/${MANIFEST_REL}' | \
    while IFS=\$'\\t' read -r pid edf onset; do
        '${PYTHON_VENV}' scripts/research/door_f_cohort/generate.py \
            --patient \"\$pid\" \
            --edf     '${EDF_DIR}'/\"\$edf\" \
            --onset   \"\$onset\" \
            --out     '${STAGE_ROOT}/sio'/door_f_\"\$pid\".sio
    done
    ls -la '${STAGE_ROOT}/sio/' | tail -30
"

echo
echo "[4/5] submitting sbatch ..."
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH_FILE}" <<SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_ID}
#SBATCH -p cpu-ops
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=${STAGE_ROOT}/logs/job-%j.log

set -euo pipefail
cd ${STAGE_ROOT}/repo
export SOUNIO_STDLIB_PATH=\${PWD}/stdlib

MANIFEST=${STAGE_ROOT}/repo/${MANIFEST_REL}
RESULTS=${STAGE_ROOT}/results
SIO=${STAGE_ROOT}/sio

echo "[\$(date)] Door F cohort N=24 on \$(hostname) with ${PARALLEL}-way parallelism"

run_one() {
    local pid="\$1"
    local out="\${RESULTS}/door_f_\${pid}.stdout"
    local err="\${RESULTS}/door_f_\${pid}.stderr"
    local t0=\$(date +%s)
    ./bin/souc run "\${SIO}/door_f_\${pid}.sio" >"\${out}" 2>"\${err}" \\
        && echo "  OK   \$pid  (\$(( \$(date +%s) - t0 ))s, \$(stat -c %s \${out}) bytes)" \\
        || echo "  FAIL \$pid  (exit=\$?, log=\${err})"
}
export -f run_one
export RESULTS SIO

awk -F'\\t' '\$1!~/^#/ && \$1!="patient" && \$1!="" {print \$1}' "\${MANIFEST}" | \\
    xargs -P ${PARALLEL} -I{} bash -c 'run_one "\$@"' _ {}

echo
echo "[\$(date)] aggregation"
python3 scripts/research/door_f_cohort/aggregate.py \\
    --results-dir "\${RESULTS}" \\
    --manifest   "\${MANIFEST}" \\
    --out-tsv    "\${RESULTS}/door_f_cohort.tsv" 2>&1 || echo "(aggregator failed; raw stdouts are still in \${RESULTS})"

echo "[\$(date)] done. Results under \${RESULTS}/"
SBATCH_EOF

kubectl -n "${NS}" cp "${SBATCH_FILE}" "${LOGIN_POD}:${STAGE_ROOT}/job.sbatch"
JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD}" -- sbatch --parsable "${STAGE_ROOT}/job.sbatch")
echo "  submitted: job ${JOB_ID}"

echo
echo "[5/5] summary"
cat <<SUM
  RUN_ID:      ${RUN_ID}
  Stage:       ${STAGE_ROOT}
  Job:         ${JOB_ID}
  Results:     ${STAGE_ROOT}/results/door_f_*.stdout (x24)
               ${STAGE_ROOT}/results/door_f_cohort.tsv

Monitor:
  kubectl -n ${NS} exec ${LOGIN_POD} -- squeue -j ${JOB_ID}
  kubectl -n ${NS} exec ${LOGIN_POD} -- tail -f ${STAGE_ROOT}/logs/job-${JOB_ID}.log

Pull:
  kubectl -n ${NS} exec ${LOGIN_POD} -- \\
    tar -C ${STAGE_ROOT}/results -cf - . | \\
    tar -C artifacts/research/door_f_cohort_N24 -xf -
SUM
