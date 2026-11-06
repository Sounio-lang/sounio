#!/bin/bash
#SBATCH -J sounio-sib-contract
#SBATCH -p cpu-ops
#SBATCH -w cpuops-t560-proxmox
#SBATCH -n1
#SBATCH -c2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o /scratch/sounio-memory-contract-20261102/probe-%j.log
set -euo pipefail
ulimit -s 262144
cd /scratch/sounio-memory-contract-20261102
run=run-$SLURM_JOB_ID
mkdir -p "$run"
tc=/scratch/amrex6090-sm75-86-build-20261101/deps/toolchain
export PATH="$tc/usr/bin:$PATH"
export COMPILER_PATH="$tc/usr/libexec/gcc/x86_64-linux-gnu/13:$tc/usr/bin"
export LIBRARY_PATH="$tc/usr/lib/x86_64-linux-gnu:$tc/usr/lib/gcc/x86_64-linux-gnu/13"
export LD_LIBRARY_PATH="$tc/usr/lib/x86_64-linux-gnu"
export CPLUS_INCLUDE_PATH="$tc/usr/include/c++/13:$tc/usr/include/x86_64-linux-gnu/c++/13:$tc/usr/lib/gcc/x86_64-linux-gnu/13/include:$tc/usr/include:$tc/usr/include/x86_64-linux-gnu"
sha256sum bin/souc-lean-single-x86_64 baseline-probe.sio candidate-probe.sio
"$tc/usr/bin/g++" -std=c++23 -O2 -Wall -Wextra check.cpp -o "$run/check"
for variant in baseline candidate;do
 timeout --kill-after=5s 90s bin/souc-lean-single-x86_64 "$variant-probe.sio" "$run/$variant.elf" > "$run/$variant-build.log" 2>&1
 test -s "$run/$variant.elf"
 chmod +x "$run/$variant.elf"
 timeout --kill-after=5s 40s "$run/$variant.elf" > "$run/$variant-emitted.log"
 tail -3 "$run/$variant-emitted.log"
 set +e
 mode=bytes;test "$variant" = candidate && mode=native
 "$run/check" oracle.txt "$run/$variant-emitted.log" "$mode" > "$run/$variant-check.log" 2>&1
 rc=$?
 set -e
 cat "$run/$variant-check.log"
 if test "$variant" = baseline;then test "$rc" -eq 1;else test "$rc" -eq 0;fi
done
echo BASELINE_REJECTED_CANDIDATE_NATIVE_PASS
