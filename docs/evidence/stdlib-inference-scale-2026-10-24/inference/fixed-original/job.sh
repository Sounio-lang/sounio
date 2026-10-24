#!/bin/bash
#SBATCH --job-name=count-lr-fixed-cpp
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --time=00:12:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/lr_fixed_cpp01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/lr_fixed_cpp01/logs/slurm-%j.err
exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash /scratch/sounio-stdlib-advanced-20261023/lr_fixed_cpp01/dispatch.sh
