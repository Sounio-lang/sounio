#!/bin/bash
#SBATCH --job-name=normalization-paired
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/normalization_perf01_scripts/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/normalization_perf01_scripts/logs/slurm-%j.err
exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SLURM_JOB_ID="$SLURM_JOB_ID" bash /scratch/sounio-stdlib-advanced-20261023/normalization_perf01_scripts/profile.sh /scratch/sounio-stdlib-advanced-20261023/normalization_into01/source /scratch/sounio-stdlib-advanced-20261023/normalization_perf01_scripts/before.sio /scratch/sounio-stdlib-advanced-20261023/compiler/artifacts/madaros.elf /scratch/sounio-stdlib-advanced-20261023/normalization_perf01
