#!/bin/bash
#SBATCH --job-name=quantum-pair-compiler-trace
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-pair-trace19/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-pair-trace19/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-pair-trace19;D=$R/results;U=$B/root-integrated01
mkdir "$D"
trap 'r=$?;echo "$r" > "$D/COLLECTOR_STATUS";exit "$r"' EXIT
export LD_LIBRARY_PATH="$B/services/bigframe-abi07/gcc13-private/root/deps"
export SOUNIO_STDLIB_PATH="$U/source/stdlib" SOUNIO_LOOP_REGIONS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
for name in match let;do
 rc=0;timeout 180 "$R/input/tracer/trace" "$U/results/madaros" compile "$R/input/tests/fixtures/wave4_quantum_pair/$name.sio" -o "$D/$name.elf" > "$D/$name.trace.stdout" 2>&1||rc=$?
 printf '%s\t%s\n' "$name" "$rc" >> "$D/PHASES.tsv"
done
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.final.verify"
echo DIAGNOSTIC_COMPILER_SIGSEGV_RECORDED
