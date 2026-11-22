#!/bin/bash
#SBATCH --job-name=count-profile-final-calibration
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=22:00:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/calibration_final02/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/calibration_final02/logs/slurm-%j.err
exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SLURM_JOB_ID="$SLURM_JOB_ID" /bin/bash /scratch/sounio-stdlib-advanced-20261023/calibration_final02/dispatch.sh
