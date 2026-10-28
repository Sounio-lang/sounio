#!/bin/bash
#SBATCH --job-name=array-cast-fault-registers
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/array-semantics10/fault-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/array-semantics10/fault-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/array-semantics10;D=$R/fault02
mkdir "$D"
trap 'r=$?;echo "$r" > "$D/COLLECTOR_STATUS";exit "$r"' EXIT
export LD_LIBRARY_PATH="$B/services/bigframe-abi07/gcc13-private/root/deps"
find "$R/input/tracer" -type f -print0|sort -z|xargs -0 sha256sum > "$D/TRACER.sha256"
for name in value4 value1024 value_param ref4 heap_f64 same_array_mut_write;do
 for mode in 0 1;do
  exe="$R/results/$name$mode.elf";sha256sum "$exe" >> "$D/EXECUTABLES.sha256"
  rc=0;timeout 30 "$R/input/tracer/trace" "$exe" > "$D/$name$mode.trace.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
 done
done
sha256sum -c "$D/TRACER.sha256" > "$D/TRACER.final.verify"
sha256sum -c "$D/EXECUTABLES.sha256" > "$D/EXECUTABLES.final.verify"
echo ARRAY_CAST_EXISTING_TRACER_RECORDED
