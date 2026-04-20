#!/usr/bin/env bash
# P1.3 seed-robustness sweep for the Hessian biomarker (CHB-MIT 9-patient cohort).
#
# Stages a snapshot to /orangefs, then submits a single sbatch on r770 that
# runs scripts/research/hessian_seed_sweep/run_sweep.py across 20 seeds.
# Each individual run is ~4 s so the whole job is ~90 s wall; using a full
# sbatch is for auditability, not throughput.
#
# Usage:
#   cd /workspace/sounio
#   bash slurm-jobs/hessian_seed_sweep/submit_seed_sweep.sh
#
# Output:
#   /orangefs/training/sounio/hess-seed-runs/<RUN_ID>/results/hess_seed<NNNNN>.stdout
#   /orangefs/training/sounio/hess-seed-runs/<RUN_ID>/results/summary.json
#   /orangefs/training/sounio/hess-seed-runs/<RUN_ID>/logs/job-<ID>.log

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
RUN_ID="${RUN_ID:-hess-seed-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/hess-seed-runs/${RUN_ID}"
PARTITION="${PARTITION:-gpu-orangefs}"
NODELIST="${NODELIST:-gpuorangefs-r770-proxmox}"
SEEDS="${SEEDS:-0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000}"

REQUIRED=(
    "bin/souc"
    "bin/souc-native"
    "artifacts/self-hosted/souc-self-hosted-x86_64"
    "examples/seizure_hessian_ossm.sio"
    "seizure_hessian_manifest.tsv"
    "scripts/research/hessian_seed_sweep/run_sweep.py"
    "scripts/research/hessian_seed_sweep/aggregate.py"
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
echo "partition:  ${PARTITION}"
echo "nodelist:   ${NODELIST}"

echo
echo "[1/3] staging snapshot to ${STAGE_ROOT}/repo ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "mkdir -p '${STAGE_ROOT}/repo' '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'"
( cd "${SOUNIO_DIR}" && tar cf - \
    bin/souc \
    bin/souc-native \
    artifacts/self-hosted/souc-self-hosted-x86_64 \
    stdlib \
    examples/seizure_hessian_ossm.sio \
    seizure_hessian_manifest.tsv \
    scripts/research/hessian_seed_sweep \
) | kubectl -n "${NS}" exec -i "${LOGIN_POD}" -- bash -c "tar xf - -C '${STAGE_ROOT}/repo'"
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "chmod +x '${STAGE_ROOT}/repo/bin/souc' '${STAGE_ROOT}/repo/bin/souc-native' '${STAGE_ROOT}/repo/artifacts/self-hosted/souc-self-hosted-x86_64'"

echo
echo "[2/3] submitting sbatch on ${PARTITION} (${NODELIST}) ..."
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH_FILE}" <<SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${PARTITION}
#SBATCH -w ${NODELIST}
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=${STAGE_ROOT}/logs/job-%j.log

set -euo pipefail
cd ${STAGE_ROOT}/repo
export SOUNIO_STDLIB_PATH=\${PWD}/stdlib

echo "[\$(date)] Hessian seed sweep on \$(hostname)"
echo "SEEDS=${SEEDS}"

python3 scripts/research/hessian_seed_sweep/run_sweep.py \\
    --source examples/seizure_hessian_ossm.sio \\
    --souc ./bin/souc \\
    --out-dir ${STAGE_ROOT}/results \\
    --seeds "${SEEDS}" \\
    --tmpdir /tmp/${RUN_ID}_seeds

echo
echo "[\$(date)] aggregation"
python3 scripts/research/hessian_seed_sweep/aggregate.py \\
    --sweep-dir ${STAGE_ROOT}/results 2>&1 || echo "(aggregator failed)"

echo "[\$(date)] done"
SBATCH_EOF

kubectl -n "${NS}" cp "${SBATCH_FILE}" "${LOGIN_POD}:${STAGE_ROOT}/job.sbatch"
JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD}" -- sbatch --parsable "${STAGE_ROOT}/job.sbatch")
echo "  submitted: job ${JOB_ID}"

echo
echo "[3/3] summary"
cat <<SUM
  RUN_ID:      ${RUN_ID}
  Stage:       ${STAGE_ROOT}
  Job:         ${JOB_ID}
  Results:     ${STAGE_ROOT}/results/hess_seed<NNNNN>.stdout (x20)
               ${STAGE_ROOT}/results/summary.json

Monitor:
  kubectl -n ${NS} exec ${LOGIN_POD} -- squeue -j ${JOB_ID}
  kubectl -n ${NS} exec ${LOGIN_POD} -- tail -f ${STAGE_ROOT}/logs/job-${JOB_ID}.log

Pull:
  kubectl -n ${NS} exec ${LOGIN_POD} -- \\
    tar -C ${STAGE_ROOT}/results -cf - . | \\
    tar -C artifacts/research/hessian_seed_sweep_r770 -xf -
SUM
