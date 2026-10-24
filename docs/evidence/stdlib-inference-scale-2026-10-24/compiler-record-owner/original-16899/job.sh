#!/bin/bash
#SBATCH --job-name=trace-original-ownerfix
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --chdir=/scratch/sounio-stdlib-advanced-20261023/trace_corrected_original15
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/trace_corrected_original15/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/trace_corrected_original15/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ownerfix01/artifacts/madaros.elf
printf '%s  %s\n' 92d2108ef595c5382d582aa363df6e95dab907a554f039d5f8f008073e535c21 "$compiler" | sha256sum -c -
exec /bin/bash /scratch/sounio-stdlib-advanced-20261023/trace_corrected_original15/dispatch.sh "$compiler" /scratch/sounio-stdlib-advanced-20261023/trace_corrected_original15/artifacts
