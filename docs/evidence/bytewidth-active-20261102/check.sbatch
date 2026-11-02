#!/bin/bash
#SBATCH -J sounio-bytewidth-negative
#SBATCH -p cpu-ops
#SBATCH -w cpuops-t560-proxmox
#SBATCH -n1
#SBATCH -c2
#SBATCH --mem=4G
#SBATCH -t 00:04:00
#SBATCH -o /scratch/sounio-bytewidth-20261102/check-%j.log
set -euo pipefail
ulimit -s 1048576
cd /scratch/sounio-bytewidth-20261102
run=$PWD/run-21221
out=$PWD/check-$SLURM_JOB_ID
mkdir -p "$out"
tc=/scratch/amrex6090-sm75-86-build-20261101/deps/toolchain
export PATH="$tc/usr/bin:$PATH"
export COMPILER_PATH="$tc/usr/libexec/gcc/x86_64-linux-gnu/13:$tc/usr/bin"
export LIBRARY_PATH="$tc/usr/lib/x86_64-linux-gnu:$tc/usr/lib/gcc/x86_64-linux-gnu/13"
export LD_LIBRARY_PATH="$tc/usr/lib/x86_64-linux-gnu"
export CPLUS_INCLUDE_PATH="$tc/usr/include/c++/13:$tc/usr/include/x86_64-linux-gnu/c++/13:$tc/usr/lib/gcc/x86_64-linux-gnu/13/include:$tc/usr/include:$tc/usr/include/x86_64-linux-gnu"
"$tc/usr/bin/g++" -std=c++23 -O2 -Wall -Wextra -Werror oracle.cpp -o "$out/oracle"
"$out/oracle" "$run/witness.log"
cd source
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
export SOUNIO_MADAROS_BIN="$run/madaros"
timeout --kill-after=5s 90s bin/souc compile ../word-mutant.sio -o "$out/mutant.elf" > "$out/compile.log" 2>&1
chmod +x "$out/mutant.elf"
set +e
timeout --kill-after=5s 30s "$out/mutant.elf" > "$out/mutant.log"
rc=$?
"$out/oracle" "$out/mutant.log"
orc=$?
set -e
test "$rc" -eq 2
test "$orc" -eq 1
cat "$out/mutant.log"
echo MUTANT_WORD_READ_REJECTED_EXIT_2
