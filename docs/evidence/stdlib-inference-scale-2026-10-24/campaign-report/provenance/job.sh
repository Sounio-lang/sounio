#!/bin/bash
#SBATCH --job-name=sized-campaign-report
#SBATCH --partition=cpu-ops
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-advanced-20261023/campaign_report_sized03/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-advanced-20261023/campaign_report_sized03/logs/slurm-%j.err
exec bash /scratch/sounio-stdlib-advanced-20261023/campaign_report_sized03/dispatch.sh
