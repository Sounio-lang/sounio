#!/bin/bash
#SBATCH --job-name=stdlib-advanced-compiler
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --chdir=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/source
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/logs/slurm-%j.err
set -euo pipefail
[[ $(id -u) == 1000 && $(id -g) == 1000 ]] || exit 65
exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SLURM_JOB_ID="$SLURM_JOB_ID" /bin/bash /scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/dispatch.sh
