#!/bin/bash
#SBATCH --job-name=stdlib-worker-zero-direct-exit
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-zero-runtime06/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-zero-runtime06/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
ulimit -c 0
R=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-zero-runtime06
O=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-zero-runtime03
D=$R/results
trap 'r=$?;echo "$r" > "$D/STATUS";exit "$r"' EXIT
sha256sum -c "$O/results/INPUTS.sha256" > "$D/ORIGINAL_INPUTS.verify"
sha256sum -c "$O/results/TOOLS.sha256" > "$D/ORIGINAL_TOOLS.verify"
sha256sum "$R/input/reference.cpp" "$O/results/runtime.o" "$O/results/leaf.elf" "$O/results/addresses.stdout" "$(command -v g++)" > "$D/INPUTS.sha256"
g++ -std=c++23 -O2 -fPIE -pie -pthread -Wall -Wextra -Werror -pedantic "$R/input/reference.cpp" "$O/results/runtime.o" -ldl -o "$D/reference"
timeout 30 "$D/reference" "$O/results/leaf.elf" "$O/results/addresses.stdout" > "$D/reference.stdout" 2>&1
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
echo ZERO_TOKEN_EXIT_DIRECT_GUARD_NATIVE_OK_NOT_NEW_WORKER_ADMISSION
