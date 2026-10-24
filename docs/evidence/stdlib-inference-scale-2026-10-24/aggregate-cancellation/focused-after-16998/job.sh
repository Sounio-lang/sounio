#!/bin/bash
#SBATCH --job-name=aggregate-guard-focused
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_guard_focused01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_guard_focused01/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
r=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_guard_focused01
trap 'echo "$?" >"$r/JOB_STATUS"' EXIT
/bin/bash "$r/dispatch.sh" /scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/madaros.elf "$r/artifacts"
