#!/usr/bin/env bash
#SBATCH --job-name=cs6-event-local
#SBATCH --partition=gpu-orangefs
#SBATCH --account=plruntime
#SBATCH --qos=gpuorangefs
#SBATCH --nodelist=gpuorangefs-multi-r740-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00

set -euo pipefail

: "${CS6_STAGE_ROOT:?CS6_STAGE_ROOT must name the staged OrangeFS directory}"
REPO="$CS6_STAGE_ROOT/repo"
DEPS="$CS6_STAGE_ROOT/deps"
RESULTS="$CS6_STAGE_ROOT/results"
PROVENANCE="$CS6_STAGE_ROOT/provenance"

mkdir -p "$RESULTS" "$PROVENANCE"
exec > >(tee "$RESULTS/slurm.stdout.txt") \
  2> >(tee "$RESULTS/slurm.stderr.txt" >&2)

echo "SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-event-local-slurm.v1"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:?}"
echo "SLURM_NODELIST=${SLURM_NODELIST:?}"
echo "CS6_STAGE_ROOT=$CS6_STAGE_ROOT"
echo "STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 --version
python3 -c 'import platform; print(platform.platform())'
nvidia-smi --query-gpu=name,uuid --format=csv,noheader || true

[[ "$SLURM_NODELIST" == "gpuorangefs-multi-r740-proxmox" ]]
[[ -d "$REPO/scripts/research" ]]
[[ -d "$DEPS/flint" ]]

find "$REPO/scripts/research" -maxdepth 1 \
  -name 'cs6_v7b_target23_arb_tm2r_*.py' -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$PROVENANCE/worker-sources.sha256"
sha256sum \
  "$REPO/scripts/research/cs6_v7b_target23_arb_tm2r_event_local_run.sh" \
  "$REPO/scripts/research/cs6_v7b_target23_arb_tm2r_event_local_contract_v1.txt" \
  > "$PROVENANCE/control-files.sha256"

CS6_PYTHONPATH="$DEPS" CS6_OUTPUT_DIR="$RESULTS" \
  bash "$REPO/scripts/research/cs6_v7b_target23_arb_tm2r_event_local_run.sh"

sha256sum "$RESULTS"/* > "$PROVENANCE/result-files.sha256"
echo "COMPLETED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CS6_EVENT_LOCAL_SLURM_COMPLETE=true"
