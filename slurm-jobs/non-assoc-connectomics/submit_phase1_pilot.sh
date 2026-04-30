#!/usr/bin/env bash
# Phase 1 real pilot submission — 10 subjects (5 ASD + 5 TD, first by
# deterministic FILE_ID sort per PROTOCOL.md).
#
# Runs Phase 1's associator_field.sio on real ABIDE data. Completes in
# ~5 min wall. Used to gate Phase 2 per PROTOCOL_PHASE2.md precondition.
#
# Usage (from Sounio workspace):
#   cd /workspace/sounio
#   bash slurm-jobs/non-assoc-connectomics/submit_phase1_pilot.sh
#
# After completion, pull results and run analysis:
#   kubectl -n slurm-pilot exec <login-pod> -- cat <RESULTS>/obs/*.csv \\
#     > /tmp/phase1_pilot.csv
#   python3 experiments/non_assoc_connectomics/analysis.py /tmp/phase1_pilot.csv

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
RUN_ID="${RUN_ID:-nac-phase1-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/non-assoc-runs/${RUN_ID}"
RESULTS_ROOT="/orangefs/training/sounio/phase1-pilot-results/${RUN_ID}"

REQUIRED=(
    "bin/souc"
    "artifacts/self-hosted/souc-self-hosted-x86_64"
    "experiments/non_assoc_connectomics/associator_field.sio"
    "stdlib"
)
for rel in "${REQUIRED[@]}"; do
    if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
        echo "missing: ${SOUNIO_DIR}/${rel}" >&2
        exit 1
    fi
done

if [[ -z "${LOGIN_POD_NAME}" ]]; then
    LOGIN_POD_NAME="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=login \
        --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
if [[ -z "${LOGIN_POD_NAME}" ]]; then
    echo "could not resolve login pod" >&2
    exit 2
fi
echo "login pod: ${LOGIN_POD_NAME}"

FRAMES_BIN="/orangefs/training/sounio/abide-data/frames.bin"
FRAMES_SZ=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- stat -c %s "${FRAMES_BIN}" 2>/dev/null || echo 0)
if [[ "${FRAMES_SZ}" -lt 1000000 ]]; then
    echo "frames.bin missing at ${FRAMES_BIN} (size=${FRAMES_SZ})" >&2
    echo "run scripts/research/abide_preprocess.py on cluster first" >&2
    exit 3
fi
echo "frames.bin: ${FRAMES_SZ} bytes"

kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    bash -c "mkdir -p '${STAGE_ROOT}/repo' '${RESULTS_ROOT}/obs' '${RESULTS_ROOT}/logs'"

(
    cd "${SOUNIO_DIR}"
    tar cf - bin/souc artifacts/self-hosted/souc-self-hosted-x86_64 stdlib \
        experiments/non_assoc_connectomics/associator_field.sio
) | kubectl -n "${NS}" exec -i "${LOGIN_POD_NAME}" -- \
    bash -c "mkdir -p '${STAGE_ROOT}/repo/experiments/non_assoc_connectomics' && \
             tar xf - -C '${STAGE_ROOT}/repo'"

# Phase 1 pilot: 5 ASD + 5 TD (subjects 0-4 and first 5 of TD section).
# associator_field.sio's existing real-ABIDE branch processes up to 5+5
# deterministically; a single array task covers the pilot.
SBATCH="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_ID}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=${RESULTS_ROOT}/logs/pilot.out
#SBATCH --error=${RESULTS_ROOT}/logs/pilot.err

set -euo pipefail
cd ${STAGE_ROOT}/repo
export SOUNIO_STDLIB_PATH=\${PWD}/stdlib
mkdir -p artifacts/research/abide
ln -sfn ${FRAMES_BIN} artifacts/research/abide/frames.bin

./bin/souc run experiments/non_assoc_connectomics/associator_field.sio \\
    > ${RESULTS_ROOT}/obs/pilot.csv
SBATCH_EOF

kubectl -n "${NS}" cp "${SBATCH}" "${LOGIN_POD_NAME}:${STAGE_ROOT}/job.sbatch"
JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD_NAME}" -- \
    sbatch --parsable "${STAGE_ROOT}/job.sbatch")

cat <<SUMMARY

=== Phase 1 real pilot submitted ===
  RUN_ID:  ${RUN_ID}
  Job ID:  ${JOB_ID}
  Results: ${RESULTS_ROOT}/obs/pilot.csv

Monitor:
  kubectl -n ${NS} exec ${LOGIN_POD_NAME} -- squeue -j ${JOB_ID}

Collect:
  kubectl -n ${NS} exec ${LOGIN_POD_NAME} -- cat ${RESULTS_ROOT}/obs/pilot.csv \\
    > /tmp/phase1_pilot.csv
  python3 experiments/non_assoc_connectomics/analysis.py /tmp/phase1_pilot.csv

Gate per PROTOCOL_PHASE2.md:
  If |Cohen's d| > 0.15 AND 95% CI excludes zero → proceed to Phase 2
    (set PHASE1_GATE=1 and run submit_phase2_full.sh).
  Else → abort, revise protocol per § Post-freeze amendments.
SUMMARY
