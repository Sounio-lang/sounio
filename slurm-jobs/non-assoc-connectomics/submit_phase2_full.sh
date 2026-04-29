#!/usr/bin/env bash
# Phase 2 full-cohort submission — ABIDE non-associative connectomics.
#
# Pattern adapted from slurm-jobs/brain-ossm/submit-abide-gpu.sh but
# CPU-only (Experiment B is scalar arithmetic; GPU is future scale-up).
#
# Precondition:
# - frames.bin v2 (8 eigenvectors × 200 ROIs) at
#   /orangefs/training/sounio/abide-data/frames.bin
# - Phase 1 real pilot gate satisfied (|d| > 0.15 with CI excluding zero);
#   this script will refuse to submit if PHASE1_GATE=0.
#
# Usage (from Sounio workspace):
#   cd /workspace/sounio
#   PHASE1_GATE=1 bash slurm-jobs/non-assoc-connectomics/submit_phase2_full.sh
#
# Output:
#   /orangefs/training/sounio/phase2-results/<RUN_ID>/obs/<N>.csv
#   /orangefs/training/sounio/phase2-results/<RUN_ID>/null/<N>.csv
#   /orangefs/training/sounio/phase2-results/<RUN_ID>/zd/all.csv

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
PHASE1_GATE="${PHASE1_GATE:-0}"
RUN_ID="${RUN_ID:-nac-phase2-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/non-assoc-runs/${RUN_ID}"
RESULTS_ROOT="/orangefs/training/sounio/phase2-results/${RUN_ID}"
ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-128}"
N_PERMS="${N_PERMS:-1000}"

if [[ "${PHASE1_GATE}" != "1" ]]; then
    echo "refusing to submit: Phase 1 gate not set (|d|>0.15 + CI excludes zero)" >&2
    echo "per PROTOCOL_PHASE2.md § Precondition. Run Phase 1 pilot first." >&2
    exit 2
fi

REQUIRED=(
    "bin/souc"
    "artifacts/self-hosted/souc-self-hosted-x86_64"
    "artifacts/research/sedenion_zd_168.bin"
    "experiments/non_assoc_connectomics/associator_field_full.sio"
    "experiments/non_assoc_connectomics/null_permutations.sio"
    "experiments/non_assoc_connectomics/zero_divisor_full.sio"
    "experiments/non_assoc_connectomics/analysis_phase2.py"
    "stdlib"
)
for rel in "${REQUIRED[@]}"; do
    if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
        echo "missing required file/dir: ${SOUNIO_DIR}/${rel}" >&2
        exit 3
    fi
done

# Resolve login pod
if [[ -z "${LOGIN_POD_NAME}" ]]; then
    LOGIN_POD_NAME="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=login \
        --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
if [[ -z "${LOGIN_POD_NAME}" ]]; then
    echo "could not resolve login pod in ${NS}" >&2
    exit 4
fi
echo "using login pod: ${LOGIN_POD_NAME}"

# Verify frames.bin v2
FRAMES_BIN="/orangefs/training/sounio/abide-data/frames.bin"
FRAMES_SZ=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- stat -c %s "${FRAMES_BIN}" 2>/dev/null || echo 0)
if [[ "${FRAMES_SZ}" -lt 1000000 ]]; then
    echo "frames.bin missing or too small at ${FRAMES_BIN} (size=${FRAMES_SZ})" >&2
    echo "run scripts/research/abide_preprocess.py on cluster first" >&2
    exit 5
fi
echo "frames.bin: ${FRAMES_SZ} bytes"

# Read n_asd, n_td from header to compute array size (header = 2 × i64 LE)
HEADER_HEX=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    bash -c "od -An -tx8 -N16 ${FRAMES_BIN} | head -1 | tr -d ' '")
N_ASD=$((16#$(echo "${HEADER_HEX:0:16}" | tac -rs.. | tr -d '\n')))
N_TD=$((16#$(echo "${HEADER_HEX:16:16}" | tac -rs.. | tr -d '\n')))
N_TOTAL=$((N_ASD + N_TD))
echo "cohort: ${N_ASD} ASD + ${N_TD} TD = ${N_TOTAL}"
ARRAY_END=$((N_TOTAL - 1))

# Stage snapshot via kubectl cp of a tar
kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    bash -c "mkdir -p '${STAGE_ROOT}/repo' '${RESULTS_ROOT}/obs' '${RESULTS_ROOT}/null' '${RESULTS_ROOT}/zd' '${RESULTS_ROOT}/logs'"

echo "staging repo snapshot to ${STAGE_ROOT}/repo ..."
(
    cd "${SOUNIO_DIR}"
    tar cf - bin/souc artifacts/self-hosted/souc-self-hosted-x86_64 \
        artifacts/research/sedenion_zd_168.bin stdlib \
        experiments/non_assoc_connectomics \
        scripts/research/abide_preprocess.py 2>/dev/null || true
) | kubectl -n "${NS}" exec -i "${LOGIN_POD_NAME}" -- \
    bash -c "tar xf - -C '${STAGE_ROOT}/repo'"

# Generate sbatch
SBATCH="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_ID}
#SBATCH --array=0-${ARRAY_END}%${ARRAY_PARALLELISM}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=${RESULTS_ROOT}/logs/%A_%a.out
#SBATCH --error=${RESULTS_ROOT}/logs/%A_%a.err

set -euo pipefail
cd ${STAGE_ROOT}/repo
export SOUNIO_STDLIB_PATH=\${PWD}/stdlib

SID=\$SLURM_ARRAY_TASK_ID

# Symlink frames.bin into the expected in-repo location
mkdir -p artifacts/research/abide
ln -sfn ${FRAMES_BIN} artifacts/research/abide/frames.bin

# Observed p95
./bin/souc run experiments/non_assoc_connectomics/associator_field_full.sio \\
    -- --subject=\$SID > ${RESULTS_ROOT}/obs/\${SID}.csv

# Null distribution (1000 perms)
./bin/souc run experiments/non_assoc_connectomics/null_permutations.sio \\
    -- --subject=\$SID --n-perms=${N_PERMS} > ${RESULTS_ROOT}/null/\${SID}.csv
SBATCH_EOF

# Copy + submit
kubectl -n "${NS}" cp "${SBATCH}" "${LOGIN_POD_NAME}:${STAGE_ROOT}/job.sbatch"
JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    sbatch --parsable "${STAGE_ROOT}/job.sbatch")
echo "submitted job ${JOB_ID} — array 0..${ARRAY_END} at parallelism ${ARRAY_PARALLELISM}"

# Separate single-task job for Experiment C (ZD)
ZD_SBATCH="/tmp/${RUN_ID}-zd.sbatch"
cat > "${ZD_SBATCH}" <<ZD_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_ID}-zd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:${JOB_ID}
#SBATCH --output=${RESULTS_ROOT}/logs/zd.out
#SBATCH --error=${RESULTS_ROOT}/logs/zd.err

set -euo pipefail
cd ${STAGE_ROOT}/repo
export SOUNIO_STDLIB_PATH=\${PWD}/stdlib
ln -sfn ${FRAMES_BIN} artifacts/research/abide/frames.bin

# All-subjects ZD scan
echo "group,subject_id,d_ZD" > ${RESULTS_ROOT}/zd/all.csv
for S in \$(seq 0 ${ARRAY_END}); do
    ./bin/souc run experiments/non_assoc_connectomics/zero_divisor_full.sio \\
        -- --subject=\$S | tail -n 1 >> ${RESULTS_ROOT}/zd/all.csv
done
ZD_EOF

kubectl -n "${NS}" cp "${ZD_SBATCH}" "${LOGIN_POD_NAME}:${STAGE_ROOT}/zd.sbatch"
ZD_JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    sbatch --parsable "${STAGE_ROOT}/zd.sbatch")
echo "submitted ZD job ${ZD_JOB_ID} (depends on ${JOB_ID})"

cat <<SUMMARY

=== Phase 2 submitted ===
  RUN_ID:            ${RUN_ID}
  Main job ID:       ${JOB_ID}   (Experiment B: associator + null)
  ZD job ID:         ${ZD_JOB_ID} (Experiment C: depends on main)
  Results:           ${RESULTS_ROOT}/
  Array:             0..${ARRAY_END} at parallelism ${ARRAY_PARALLELISM}
  Perms per subject: ${N_PERMS}

Monitor:
  kubectl -n ${NS} exec ${LOGIN_POD_NAME} -- squeue -j ${JOB_ID}

Pull results when complete:
  kubectl -n ${NS} exec ${LOGIN_POD_NAME} -- tar -C ${RESULTS_ROOT} -cf - . \\
    | tar -C /tmp/phase2_\${RUN_ID} -xf -

Then run:
  python3 experiments/non_assoc_connectomics/analysis_phase2.py \\
    --obs /tmp/phase2_\${RUN_ID}/obs \\
    --null /tmp/phase2_\${RUN_ID}/null \\
    --zd /tmp/phase2_\${RUN_ID}/zd \\
    --pheno <path-to-phenotypic.csv> \\
    --out artifacts/research/non_assoc_connectomics_phase2
SUMMARY
