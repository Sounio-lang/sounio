#!/bin/bash
#SBATCH --job-name=stdlib-deep-verified-compiler
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192M
#SBATCH --time=00:55:00
#SBATCH --export=NIL
#SBATCH --chdir=/scratch/sounio-stdlib-deep-20261023/final1/source
#SBATCH --output=/scratch/sounio-stdlib-deep-20261023/final1/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-deep-20261023/final1/logs/slurm-%j.err
set -euo pipefail
[[ $(id -u) == 1000 && $(id -g) == 1000 ]] || exit 65
exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SLURM_JOB_ID="$SLURM_JOB_ID" /bin/bash /scratch/sounio-stdlib-deep-20261023/final1/dispatch.sh
