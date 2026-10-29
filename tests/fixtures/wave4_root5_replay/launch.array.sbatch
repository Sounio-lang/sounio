#!/bin/bash
#SBATCH --job-name=root5-history-replay
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --export=NIL
set -euo pipefail
D=${1:?prepared and root-authorized replay directory required}
exec bash "$D/harness/root5_regression_replay.sh" case "$D" "$SLURM_ARRAY_TASK_ID"
