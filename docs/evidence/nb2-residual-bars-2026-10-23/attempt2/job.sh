#!/usr/bin/env bash
#SBATCH --job-name=nb2-residual-bars
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16384M
#SBATCH --time=00:25:00
#SBATCH --export=NIL
#SBATCH --chdir=/scratch/sounio-nb2-residual-bars-20261023/run2
#SBATCH --output=/scratch/sounio-nb2-residual-bars-20261023/run2/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-nb2-residual-bars-20261023/run2/logs/slurm-%j.err
set -euo pipefail
[[ $(id -u) == 1000 && $(id -g) == 1000 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 65
exec /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C "SLURM_JOB_ID=$SLURM_JOB_ID" /bin/bash --noprofile --norc /scratch/sounio-nb2-residual-bars-20261023/run2/dispatch.sh
