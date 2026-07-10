#!/usr/bin/env bash
# One-command ptxas-acceptance submission for the od256 octuple kernels.
#
# Stages the four committed od256 golden PTX kernels (two_sum/two_prod/add/mul)
# — which are byte-identical to the emitter output (verified by the golden gate)
# — and submits the ptxas-acceptance job to the GPU worker via the slurm-pilot
# login pod. The worker runs `ptxas -arch=sm_50` on each kernel (no GPU compute
# needed: ptxas is the CUDA assembler). Staging the goldens directly avoids
# recompiling the K-AXI→PTX driver per pattern.
#
# Usage (from the eisa worktree, where the od256 kernels live):
#   cd /workspace/sounio-eisa
#   bash slurm-jobs/kaxi-ptxas-accept/submit-od256.sh
#
# Requires: kubectl access to the slurm-pilot cluster (login pod
# slurm-pilot-login-slinky). Node/partition defaults target gpuorangefs-r770
# (NVIDIA L4, cc 8.9); override with SBATCH_NODELIST / SBATCH_PARTITION.
#
# Fetch results after the job finishes (RUN_ID is printed on submit):
#   kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- \
#     cat /orangefs/training/sounio/kaxi-ptxas-accept/<RUN_ID>/results/summary.txt
#   # expect: KAXI_PTXAS_ACCEPT_OK and one PASS line per default__od256_*.ptx
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Dedicated staging dir so we submit ONLY the od256 kernels (not a stale mix).
export STAGE_LOCAL="${STAGE_LOCAL:-/tmp/kaxi_ptx_stage_od256}"
rm -rf "${STAGE_LOCAL}/ptx"
mkdir -p "${STAGE_LOCAL}/ptx"

n=0
for p in od256_two_sum od256_two_prod od256_add od256_mul; do
    g="tests/golden/kaxi_ptx/od256/${p}.ptx"
    [[ -f "$g" ]] || { echo "missing golden $g — run the golden gate / re-emit first" >&2; exit 1; }
    # ptxas needs a `.target`; all od256 goldens declare `.target sm_50`.
    grep -q '\.target sm_50' "$g" || { echo "warn: $g has no .target sm_50" >&2; }
    cp -f "$g" "${STAGE_LOCAL}/ptx/default__${p}.ptx"
    n=$((n + 1))
done
echo "staged ${n} od256 PTX kernels -> ${STAGE_LOCAL}/ptx"
ls -1 "${STAGE_LOCAL}/ptx"

echo "submitting ptxas-acceptance to the GPU worker..."
exec bash "${ROOT_DIR}/slurm-jobs/kaxi-ptxas-accept/submit.sh"
