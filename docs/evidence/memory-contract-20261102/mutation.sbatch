#!/bin/bash
#SBATCH -J sounio-sib-negative
#SBATCH -p cpu-ops
#SBATCH -w cpuops-t560-proxmox
#SBATCH -n1
#SBATCH -c1
#SBATCH --mem=3G
#SBATCH -t 00:03:00
#SBATCH -o /scratch/sounio-memory-contract-20261102/mutation-%j.log
set -euo pipefail
ulimit -s 262144
cd /scratch/sounio-memory-contract-20261102
run=mutation-$SLURM_JOB_ID
mkdir -p "$run"
timeout --kill-after=5s 90s bin/souc-lean-single-x86_64 no-sib-mutant.sio "$run/mutant.elf" > "$run/build.log" 2>&1
chmod +x "$run/mutant.elf"
set +e
timeout --kill-after=5s 40s "$run/mutant.elf" > "$run/test.log"
rc=$?
set -e
test "$rc" -eq 1
grep -F 'FAIL: T73 disp32 SIB full bytes' "$run/test.log"
grep -F 'AVX-512 tests: 72/73 passed' "$run/test.log"
echo MUTATION_REJECTED_EXIT_1
