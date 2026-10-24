#!/bin/bash
#SBATCH --job-name=materializer-exit-status
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/materializer_exit_status02/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/materializer_exit_status02/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/materializer_exit_status02
trap 'echo "$?" >"$r/STATUS"' EXIT
bash -n "$r/before/bin/souc" "$r/before/bin/madaros" "$r/after/bin/souc" "$r/after/bin/madaros" "$r/regression.sh"
bash "$r/regression.sh" "$r/before" "$r/after" "$r/results"
