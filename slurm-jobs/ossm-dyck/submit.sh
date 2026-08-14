#!/usr/bin/env bash
# Submit OSSM Dyck scaling experiment to the GPU cluster.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-slurm-pilot}"
PAYLOAD_DIR="/orangefs/training/sounio/ossm-dyck-source"
OUT_DIR="/orangefs/training/sounio/kimi-runs/ossm-dyck"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

echo "Finding Slurm login pod..."
LOGIN_POD="$(kubectl -n "${NS}" get pods --field-selector=status.phase=Running -o name | grep slurm-pilot-login | head -1 | sed 's/^pod\///')"
if [[ -z "${LOGIN_POD}" ]]; then
  echo "ERROR: no running slurm login pod found" >&2
  exit 1
fi
echo "Login pod: ${LOGIN_POD}"

echo "Creating directories..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- mkdir -p "${PAYLOAD_DIR}" "${OUT_DIR}"

echo "Staging payload..."
kubectl cp "${ROOT_DIR}/scripts/research/ossm_dyck_scaling.py" \
  "${NS}/${LOGIN_POD}:${PAYLOAD_DIR}/ossm_dyck_scaling.py"
kubectl cp "${ROOT_DIR}/slurm-jobs/ossm-dyck/job_ossm_dyck.sh" \
  "${NS}/${LOGIN_POD}:${PAYLOAD_DIR}/job_ossm_dyck.sh"

kubectl -n "${NS}" exec "${LOGIN_POD}" -- chmod +x "${PAYLOAD_DIR}/job_ossm_dyck.sh"

echo "Submitting Slurm job..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- \
  sbatch --job-name="ossm-dyck-${RUN_ID}" \
         --output="${OUT_DIR}/ossm-dyck-%j.out" \
         "${PAYLOAD_DIR}/job_ossm_dyck.sh"

echo ""
echo "Output will appear in ${OUT_DIR}/"
echo "Check status: squeue -u root"
