#!/usr/bin/env bash
# LEMON full-cohort confirmatory submission — OSSM-168 v3.
#
# Modes:
#   MODE=full (default)         — per-subject preprocessing + feature extraction
#   MODE=preprocess-only        — populate cache-dir with preprocessed epochs.npy only
#   MODE=confirmatory-only      — feature extraction + aggregation, assumes cache ready
#
# Usage (from Sounio workspace, cluster-connected host):
#   cd /workspace/sounio
#   MODE=full bash slurm-jobs/lemon_confirmatory/submit_full_lemon.sh
#
# Prerequisites:
#   - kubectl access to namespace slurm-pilot
#   - LEMON raw EEG staged at /orangefs/training/sounio/lemon/raw_eeg/
#   - endpoints.csv and subjects_all.txt staged at /orangefs/training/sounio/lemon/
#   - /opt/sounio-venv on login pod with MNE+picard+pyEDFlib

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
MODE="${MODE:-full}"
RUN_ID="${RUN_ID:-lemon-ossm168-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/lemon-runs/${RUN_ID}"
RESULTS_ROOT="/orangefs/training/sounio/lemon/results/${RUN_ID}"
ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-75}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"

RAW_ROOT="${RAW_ROOT:-/orangefs/training/sounio/lemon/raw_eeg}"
ENDPOINTS="${ENDPOINTS:-/orangefs/training/sounio/lemon/endpoints.csv}"
SUBJECTS_LIST="${SUBJECTS_LIST:-/orangefs/training/sounio/lemon/subjects_all.txt}"
CACHE_DIR="${CACHE_DIR:-/orangefs/training/sounio/lemon/preprocessed}"
PYTHON_VENV="${PYTHON_VENV:-/opt/sounio-venv/bin/python}"

# ---------------------------------------------------------------------------
# Validate local repo files
REQUIRED=(
    "scripts/research/ossm_168_dryrun/run_lemon_confirmatory.py"
    "scripts/research/ossm_168_dryrun/lemon_endpoints.py"
    "scripts/research/ossm_168_dryrun/lemon_preprocess.py"
    "scripts/research/ossm_168_dryrun/_features_fast.py"
    "scripts/research/ossm_168_dryrun/features.py"
    "scripts/research/ossm_168_dryrun/ossm.py"
)
for rel in "${REQUIRED[@]}"; do
    if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
        echo "missing required file: ${SOUNIO_DIR}/${rel}" >&2
        exit 2
    fi
done

# Validate mode
if [[ "${MODE}" != "full" && "${MODE}" != "preprocess-only" && "${MODE}" != "confirmatory-only" ]]; then
    echo "invalid MODE=${MODE}; expected full, preprocess-only, or confirmatory-only" >&2
    exit 2
fi

# Resolve login pod
if [[ -z "${LOGIN_POD_NAME}" ]]; then
    LOGIN_POD_NAME="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=login \
        --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
if [[ -z "${LOGIN_POD_NAME}" ]]; then
    echo "could not resolve login pod in ${NS}" >&2
    exit 3
fi
echo "login pod:    ${LOGIN_POD_NAME}"
echo "run id:       ${RUN_ID}"
echo "stage root:   ${STAGE_ROOT}"
echo "results root: ${RESULTS_ROOT}"
echo "mode:         ${MODE}"

# ---------------------------------------------------------------------------
# Verify remote data presence
_verify_remote() {
    local path="$1" desc="$2"
    local sz
    sz=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
        bash -c "test -e '${path}' && stat -c %s '${path}' || echo 0" 2>/dev/null || echo 0)
    if [[ "${sz}" == "0" || "${sz}" == "" ]]; then
        echo "missing remote data: ${desc} at ${path}" >&2
        return 1
    fi
    echo "  ${desc}: ${sz} bytes"
    return 0
}

echo
echo "[1/5] verifying remote data on /orangefs ..."
_verify_remote "${ENDPOINTS}" "endpoints.csv" || exit 4
_verify_remote "${SUBJECTS_LIST}" "subjects_all.txt" || exit 4
_verify_remote "${RAW_ROOT}" "raw_eeg dir" || exit 4

# Count subjects
N_TOTAL=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    bash -lc "wc -l < ${SUBJECTS_LIST} | tr -d ' \r'")
ARRAY_END=$((N_TOTAL - 1))
echo "  subjects: ${N_TOTAL} (array 0..${ARRAY_END})"

# ---------------------------------------------------------------------------
# Stage repo snapshot
echo
echo "[2/5] staging repo snapshot to ${STAGE_ROOT}/repo ..."
kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    bash -lc "mkdir -p '${STAGE_ROOT}/repo' '${RESULTS_ROOT}/features' '${RESULTS_ROOT}/logs'"
(
    cd "${SOUNIO_DIR}"
    tar cf - \
        scripts/research/ossm_168_dryrun \
        2>/dev/null || true
) | kubectl -n "${NS}" exec -i "${LOGIN_POD_NAME}" -- \
    bash -c "tar xf - -C '${STAGE_ROOT}/repo'"

# ---------------------------------------------------------------------------
# Generate per-subject sbatch
echo
echo "[3/5] generating per-subject sbatch ..."
SBATCH="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH}" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=${RUN_ID}
#SBATCH --array=0-${ARRAY_END}%${ARRAY_PARALLELISM}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=${RESULTS_ROOT}/logs/%A_%a.out
#SBATCH --error=${RESULTS_ROOT}/logs/%A_%a.err

set -euo pipefail
cd ${STAGE_ROOT}/repo
export PYTHONPATH=${PWD}/scripts/research
export OMP_NUM_THREADS=${CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${CPUS_PER_TASK}
export MKL_NUM_THREADS=${CPUS_PER_TASK}

SID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SUBJECTS_LIST}")
echo "task ${SLURM_ARRAY_TASK_ID} -> subject ${SID}"

# Guard: skip if raw EEG not yet staged
VHDR="${RAW_ROOT}/${SID}/RSEEG/${SID}.vhdr"
if [[ ! -f "${VHDR}" ]]; then
    echo "SKIP ${SID}: raw EEG not found at ${VHDR} (download may be in progress)"
    exit 0
fi

# Write single-subject list to temp
TASK_SUBJ="/tmp/${RUN_ID}_${SLURM_ARRAY_TASK_ID}_subjects.txt"
echo "${SID}" > "${TASK_SUBJ}"

MODE=${MODE}

if [[ "${MODE}" == "preprocess-only" ]]; then
    # Fast path: only cache preprocessed epochs, no feature extraction
    ${PYTHON_VENV} -c "
import sys
sys.path.insert(0, '${STAGE_ROOT}/repo/scripts/research')
from ossm_168_dryrun import lemon_preprocess as lp
epochs, prov = lp.preprocess_or_load('${SID}', '${RAW_ROOT}', '${CACHE_DIR}')
print(f'${SID}: preprocessed {epochs.shape} epochs  retained={prov[\"n_retained_epochs\"]}/{prov[\"n_total_epochs\"]}')
"
else
    # full or confirmatory-only: extract features (preprocessing hits cache if present)
    TASK_OUT="/tmp/${RUN_ID}_${SLURM_ARRAY_TASK_ID}_out"
    mkdir -p "${TASK_OUT}"
    ${PYTHON_VENV} -m ossm_168_dryrun.run_lemon_confirmatory \
        --raw-root "${RAW_ROOT}" \
        --endpoints "${ENDPOINTS}" \
        --cache-dir "${CACHE_DIR}" \
        --out-dir "${TASK_OUT}" \
        --subjects-list "${TASK_SUBJ}" \
        --max-epochs "${MAX_EPOCHS}" \
        --skip-existing-features
    if [[ -f "${TASK_OUT}/features.csv" ]]; then
        cp "${TASK_OUT}/features.csv" "${RESULTS_ROOT}/features/${SID}_features.csv"
    fi
fi
SBATCH_EOF

# Substitute variables into the heredoc
sed -i \
    -e "s|\${RUN_ID}|${RUN_ID}|g" \
    -e "s|\${ARRAY_END}|${ARRAY_END}|g" \
    -e "s|\${ARRAY_PARALLELISM}|${ARRAY_PARALLELISM}|g" \
    -e "s|\${CPUS_PER_TASK}|${CPUS_PER_TASK}|g" \
    -e "s|\${RESULTS_ROOT}|${RESULTS_ROOT}|g" \
    -e "s|\${SUBJECTS_LIST}|${SUBJECTS_LIST}|g" \
    -e "s|\${MODE}|${MODE}|g" \
    -e "s|\${STAGE_ROOT}|${STAGE_ROOT}|g" \
    -e "s|\${RAW_ROOT}|${RAW_ROOT}|g" \
    -e "s|\${ENDPOINTS}|${ENDPOINTS}|g" \
    -e "s|\${CACHE_DIR}|${CACHE_DIR}|g" \
    -e "s|\${MAX_EPOCHS}|${MAX_EPOCHS}|g" \
    -e "s|\${PYTHON_VENV}|${PYTHON_VENV}|g" \
    "${SBATCH}"

# Copy + submit
kubectl -n "${NS}" cp "${SBATCH}" "${LOGIN_POD_NAME}:${STAGE_ROOT}/job.sbatch"
JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    sbatch --parsable "${STAGE_ROOT}/job.sbatch")
echo "submitted job ${JOB_ID} — array 0..${ARRAY_END} at parallelism ${ARRAY_PARALLELISM}"

# ---------------------------------------------------------------------------
# Generate aggregator (only for non-preprocess-only modes)
if [[ "${MODE}" == "preprocess-only" ]]; then
    cat <<SUMMARY

=== Preprocess-only array submitted ===
  RUN_ID:   ${RUN_ID}
  Job ID:   ${JOB_ID}
  Array:    0..${ARRAY_END} at parallelism ${ARRAY_PARALLELISM}
  Cache:    ${CACHE_DIR}

When complete, rerun with MODE=confirmatory-only to extract features and run stats.
SUMMARY
    exit 0
fi

echo
echo "[4/5] generating aggregator sbatch ..."
AGG_SBATCH="/tmp/${RUN_ID}_agg.sbatch"
cat > "${AGG_SBATCH}" <<'AGG_EOF'
#!/bin/bash
#SBATCH --job-name=${RUN_ID}-agg
#SBATCH --partition=cpu-ops
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:${JOB_ID}
#SBATCH --output=${RESULTS_ROOT}/logs/agg.out
#SBATCH --error=${RESULTS_ROOT}/logs/agg.err

set -euo pipefail
cd ${STAGE_ROOT}/repo
export PYTHONPATH=${PWD}/scripts/research

# Concatenate per-subject features.csv into one (keep one header)
FEAT_COMBINED="${RESULTS_ROOT}/features.csv"
FIRST=$(ls -1 "${RESULTS_ROOT}"/features/*_features.csv 2>/dev/null | head -1 || true)
if [[ -z "${FIRST}" ]]; then
    echo "WARN: no per-subject features found — aggregator will produce empty results"
    exit 0
fi
HEADER=$(head -1 "${FIRST}")
echo "${HEADER}" > "${FEAT_COMBINED}"
for f in "${RESULTS_ROOT}"/features/*_features.csv; do
    tail -n +2 "${f}" >> "${FEAT_COMBINED}"
done
echo "combined $(wc -l < <(tail -n +2 "${FEAT_COMBINED}")) subject rows into ${FEAT_COMBINED}"

# Run full harness in stats-only mode: all features cached, just merge + BCa/Holm
${PYTHON_VENV} -m ossm_168_dryrun.run_lemon_confirmatory \
    --raw-root "${RAW_ROOT}" \
    --endpoints "${ENDPOINTS}" \
    --cache-dir "${CACHE_DIR}" \
    --out-dir "${RESULTS_ROOT}" \
    --subjects-list "${SUBJECTS_LIST}" \
    --max-epochs "${MAX_EPOCHS}" \
    --skip-existing-features

# Move per-subject features into an archive to keep root tidy
mkdir -p "${RESULTS_ROOT}/features_archive"
mv "${RESULTS_ROOT}"/features/*_features.csv "${RESULTS_ROOT}/features_archive/" 2>/dev/null || true
AGG_EOF

sed -i \
    -e "s|\${RUN_ID}|${RUN_ID}|g" \
    -e "s|\${JOB_ID}|${JOB_ID}|g" \
    -e "s|\${RESULTS_ROOT}|${RESULTS_ROOT}|g" \
    -e "s|\${RAW_ROOT}|${RAW_ROOT}|g" \
    -e "s|\${ENDPOINTS}|${ENDPOINTS}|g" \
    -e "s|\${CACHE_DIR}|${CACHE_DIR}|g" \
    -e "s|\${SUBJECTS_LIST}|${SUBJECTS_LIST}|g" \
    -e "s|\${MAX_EPOCHS}|${MAX_EPOCHS}|g" \
    -e "s|\${PYTHON_VENV}|${PYTHON_VENV}|g" \
    "${AGG_SBATCH}"

kubectl -n "${NS}" cp "${AGG_SBATCH}" "${LOGIN_POD_NAME}:${STAGE_ROOT}/agg.sbatch"
AGG_JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    sbatch --parsable "${STAGE_ROOT}/agg.sbatch")
echo "submitted aggregator job ${AGG_JOB_ID} (depends on ${JOB_ID})"

# ---------------------------------------------------------------------------
cat <<SUMMARY

=== LEMON confirmatory array submitted ===
  RUN_ID:     ${RUN_ID}
  Mode:       ${MODE}
  Main job:   ${JOB_ID}   (per-subject array)
  Agg job:    ${AGG_JOB_ID} (depends on main)
  Results:    ${RESULTS_ROOT}/
  Array:      0..${ARRAY_END} at parallelism ${ARRAY_PARALLELISM}
  CPUs/task:  ${CPUS_PER_TASK}
  Max epochs: ${MAX_EPOCHS}

Monitor:
  kubectl -n ${NS} exec ${LOGIN_POD_NAME} -- squeue -j ${JOB_ID}

Pull results when complete:
  kubectl -n ${NS} exec ${LOGIN_POD_NAME} -- tar -C ${RESULTS_ROOT} -cf - . \\
    | tar -C /tmp/lemon_${RUN_ID} -xf -
SUMMARY
