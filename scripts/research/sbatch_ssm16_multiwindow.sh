#!/bin/bash
#SBATCH --job-name=ssm16-mw
#SBATCH --partition=gpu-orangefs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/orangefs/training/sounio/ssm16_multiwindow_%j.out
#SBATCH --error=/orangefs/training/sounio/ssm16_multiwindow_%j.err

set -e
cd /workspace/sounio

python3 scripts/research/ssm_16ch_multiwindow.py \
    --edf-dir data/eegmmidb \
    --stride 40 \
    --output /orangefs/training/sounio/ssm16_multiwindow_results.json

echo "DONE job $SLURM_JOB_ID"
