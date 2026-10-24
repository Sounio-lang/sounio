#!/bin/bash
#SBATCH --job-name=nested-qualification-ops
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/qualification_ops_final_traced/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/qualification_ops_final_traced/logs/slurm-%j.err
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/qualification_ops_final_traced
export COUNT_PROFILE_TEST_TRACED=1
exec bash "$r/source/tests/operational/count_profile_calibration_nested.sh" "$r/source" "$r/baseline.cpp" /scratch/sounio-stdlib-advanced-20261023/trace_focused02/source /scratch/sounio-stdlib-inference-scale-20261024/compiler_ownerfix01/artifacts/madaros.elf /scratch/sounio-stdlib-advanced-20261023/calibration_final02/receipts/full02/RESULTS.normalized.tsv "$r/results"
