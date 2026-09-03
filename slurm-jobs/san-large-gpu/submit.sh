#!/usr/bin/env bash
# Submit SAN large architecture training to the Slurm gpu-orangefs partition.
#
# Usage:
#   bash slurm-jobs/san-large-gpu/submit.sh [resnet50|vitlarge|gpt]
#
# Env:
#   SAN_LARGE_ONLY   leg to run (default: resnet50)
#   NS               k8s namespace for Slurm (default: slurm-pilot)
#   SBATCH_PARTITION (default: gpu-orangefs)
#   JOB_TIME         (default: 04:00:00)
#   JOB_MEM          (default: 64G)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEG="${1:-${SAN_LARGE_ONLY:-resnet50}}"
NS="${NS:-slurm-pilot}"
PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
JOB_TIME="${JOB_TIME:-04:00:00}"
JOB_MEM="${JOB_MEM:-64G}"
JOB_NAME="san-large-gpu-${LEG}"

LOGIN_POD="$(kubectl -n "${NS}" get pods --field-selector=status.phase=Running -o name | grep slurm-pilot-login | head -1 | sed 's/^pod\///')"
if [[ -z "${LOGIN_POD}" ]]; then
  echo "ERROR: no running slurm login pod found" >&2
  exit 1
fi

PAYLOAD_DIR="/orangefs/training/sounio/san-large-source"
OUT_ROOT="/orangefs/training/sounio/kimi-runs/san-large-gpu"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
JOB_SCRIPT="${PAYLOAD_DIR}/job_${RUN_ID}_${LEG}.sh"

echo "Staging payload to ${PAYLOAD_DIR}..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- mkdir -p "${PAYLOAD_DIR}" "${OUT_DIR}"
kubectl cp "${ROOT_DIR}/scripts/research/suffering_aware_large_architecture.py" \
  "${NS}/${LOGIN_POD}:${PAYLOAD_DIR}/suffering_aware_large_architecture.py"
kubectl cp "${ROOT_DIR}/slurm-jobs/san-large-gpu/run_gpu_worker.sh" \
  "${NS}/${LOGIN_POD}:${PAYLOAD_DIR}/run_gpu_worker.sh"
if [[ -f "${ROOT_DIR}/artifacts/san_large/corpus_snapshot_v2000.npz" ]]; then
  kubectl cp "${ROOT_DIR}/artifacts/san_large/corpus_snapshot_v2000.npz" \
    "${NS}/${LOGIN_POD}:${PAYLOAD_DIR}/corpus_snapshot_v2000.npz"
fi
kubectl -n "${NS}" exec "${LOGIN_POD}" -- chmod +x "${PAYLOAD_DIR}/run_gpu_worker.sh"

cat > /tmp/san_large_job_script.sh <<EOF
#!/usr/bin/env bash
#SBATCH -p ${PARTITION}
#SBATCH -A plruntime
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${OUT_DIR}/${LEG}-%j.out
#SBATCH --time=${JOB_TIME}
#SBATCH --mem=${JOB_MEM}
export SAN_LARGE_ONLY=${LEG}
export SAN_LARGE_DATASET=${SAN_LARGE_DATASET:-cifar10}
export SAN_LARGE_OUT=${OUT_ROOT}
export RUN_ID=${RUN_ID}
export SAN_LARGE_DELTA_RESNET=${SAN_LARGE_DELTA_RESNET:-0.55}
export SAN_LARGE_DELTA_VIT=${SAN_LARGE_DELTA_VIT:-0.45}
export SAN_LARGE_DELTA_GPT=${SAN_LARGE_DELTA_GPT:-0.31}
export SAN_LARGE_TAU_RESNET=${SAN_LARGE_TAU_RESNET:-0.34}
export SAN_LARGE_TAU_VIT=${SAN_LARGE_TAU_VIT:-0.251}
export SAN_LARGE_TAU_GPT=${SAN_LARGE_TAU_GPT:-0.165}
bash ${PAYLOAD_DIR}/run_gpu_worker.sh
EOF

kubectl cp /tmp/san_large_job_script.sh "${NS}/${LOGIN_POD}:${JOB_SCRIPT}"
kubectl -n "${NS}" exec "${LOGIN_POD}" -- chmod +x "${JOB_SCRIPT}"

echo "Submitting ${LEG} to ${PARTITION}..."
job="$(kubectl -n "${NS}" exec "${LOGIN_POD}" -- sbatch --parsable "${JOB_SCRIPT}")"

echo "submitted Slurm job_id=${job}"
echo "stdout: ${OUT_DIR}/${LEG}-${job}.out"
echo "artifacts: ${OUT_DIR}"
echo "poll:   kubectl -n ${NS} exec ${LOGIN_POD} -- squeue -j ${job}"
echo "result: kubectl -n ${NS} exec ${LOGIN_POD} -- sacct -j ${job} --format=JobID,State,ExitCode,NodeList -P"
