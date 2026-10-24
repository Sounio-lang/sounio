#!/bin/bash
#SBATCH --job-name=aggregate-exact-analysis
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation_analysis02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation_analysis02/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation_analysis02
cd "$r";trap 'echo "$?" >STATUS' EXIT
a=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation01/artifacts
sha256sum reference.cpp "$a/ownerfix.native.stdout" "$a/effectbridge.native.stdout" "$a/ownerfix.elf" "$a/effectbridge.elf" >INPUTS.sha256
g++ -std=c++23 -O2 reference.cpp -o reference >compile.stdout 2>compile.stderr
for v in ownerfix effectbridge;do ./reference "$a/$v.native.stdout" >"$v.REPORT.tsv" 2>"$v.REPORT.stderr";done
sha256sum -c INPUTS.sha256 >INPUTS.after.verify
