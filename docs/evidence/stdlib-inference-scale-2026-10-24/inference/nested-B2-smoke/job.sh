#!/bin/bash
#SBATCH --job-name=stdlib-adv-nested_profile_smoke01
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --chdir=/scratch/sounio-stdlib-advanced-20261023/nested_profile_smoke01/source
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/nested_profile_smoke01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/nested_profile_smoke01/logs/slurm-%j.err
exec env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C SLURM_JOB_ID="$SLURM_JOB_ID" /bin/bash /scratch/sounio-stdlib-advanced-20261023/nested_profile_smoke01/dispatch.sh
