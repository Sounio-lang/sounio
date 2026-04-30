#!/usr/bin/env bash
# High-gamma (20-40 Hz) bandpower baseline — matched window schedule to Door F.
#
# Runs on the login pod using /opt/sounio-venv (pyedflib+numpy+scipy).
# Very fast (~few seconds for 24 patients) — no Slurm scheduling needed.
# Outputs a cohort TSV that the aggregator merges with sedenion results
# for head-to-head Cohen's d + meta-analysis.
#
# Usage (from Sounio workspace):
#   cd /workspace/sounio
#   bash slurm-jobs/door_f_cohort/submit_high_gamma_baseline.sh
#
# Output:
#   /orangefs/training/sounio/door-f-runs/<RUN_ID>/baseline/baseline_<pat>.tsv (x24)
#   /orangefs/training/sounio/door-f-runs/<RUN_ID>/baseline_cohort.tsv

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
PYTHON_VENV="${PYTHON_VENV:-/opt/sounio-venv/bin/python}"
EDF_DIR="${EDF_DIR:-/orangefs/training/sounio/chbmit}"
RUN_ID="${RUN_ID:-high-gamma-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/door-f-runs/${RUN_ID}"
MANIFEST_REL="scripts/research/door_f_cohort/chbmit_manifest.tsv"
SCRIPT_REL="scripts/research/door_f_cohort/high_gamma_baseline.py"

for rel in "${MANIFEST_REL}" "${SCRIPT_REL}"; do
    if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
        echo "missing: ${SOUNIO_DIR}/${rel}" >&2; exit 2
    fi
done

LOGIN_POD="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=login \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
echo "login pod:  ${LOGIN_POD}"
echo "run id:     ${RUN_ID}"
echo "stage root: ${STAGE_ROOT}"

echo
echo "[1/3] staging manifest + script to login pod ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${STAGE_ROOT}/baseline'"
kubectl -n "${NS}" cp "${SOUNIO_DIR}/${MANIFEST_REL}" \
    "${LOGIN_POD}:${STAGE_ROOT}/chbmit_manifest.tsv"
kubectl -n "${NS}" cp "${SOUNIO_DIR}/${SCRIPT_REL}" \
    "${LOGIN_POD}:${STAGE_ROOT}/high_gamma_baseline.py"

echo
echo "[2/3] running high-gamma baseline on login pod ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -e
    cd '${STAGE_ROOT}'
    '${PYTHON_VENV}' high_gamma_baseline.py \
        --manifest   chbmit_manifest.tsv \
        --edf-dir    '${EDF_DIR}' \
        --out-dir    baseline \
        --cohort-tsv baseline_cohort.tsv
    echo
    echo '=== cohort head ==='
    head -6 baseline_cohort.tsv
    wc -l baseline_cohort.tsv
"

echo
echo "[3/3] pulling cohort TSV back to workspace ..."
mkdir -p "${SOUNIO_DIR}/artifacts/research/door_f_cohort_N24"
kubectl -n "${NS}" cp \
    "${LOGIN_POD}:${STAGE_ROOT}/baseline_cohort.tsv" \
    "${SOUNIO_DIR}/artifacts/research/door_f_cohort_N24/baseline_cohort.tsv"

echo
cat <<SUM
=== High-gamma baseline complete ===
  Cluster path: ${STAGE_ROOT}/baseline_cohort.tsv
  Workspace:    ${SOUNIO_DIR}/artifacts/research/door_f_cohort_N24/baseline_cohort.tsv

Head-to-head comparison:
  python3 scripts/research/door_f_cohort/head_to_head_analysis.py \\
    --sedenion artifacts/research/door_f_cohort_N24/door_f_cohort.tsv \\
    --baseline artifacts/research/door_f_cohort_N24/baseline_cohort.tsv
SUM
