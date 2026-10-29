#!/bin/bash
#SBATCH --job-name=quantum-original-slow-mode0
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-slow-mode0-35/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-slow-mode0-35/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-slow-mode0-35;S=$R/source;D=$R/results;U=$B/root-integrated02
mkdir "$D"
finish(){ r=$?;trap - EXIT;for n in SOURCE COMPILER EXECUTABLE WITNESS;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" > "$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" > "$D/STATUS";exit "$r";};trap finish EXIT
find "$S" -type f -print0|sort -z|xargs -0 sha256sum > "$D/SOURCE.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
src="$S/tests/stdlib/qpu/test_qpu_chsh_overtake_slow.sio"
sha256sum "$src" > "$D/WITNESS.sha256"
cmp "$src" "$B/services/quantum-pair-library28/input/source/tests/stdlib/qpu/test_qpu_chsh_overtake_slow.sio"
printf '%s\n' 'UNCHANGED original slow workload and3000s deadline, quantum library328ce88; ONLY loop-region mode0 differs from failed19533 mode1. Does not close mode1 or physical-QPU acceptance.' > "$D/SCOPE"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_LOOP_REGIONS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
timeout 1200 "$U/results/madaros" --check "$src" > "$D/check.stdout" 2>&1
timeout 1200 "$U/results/madaros" compile "$src" -o "$D/original.elf" > "$D/compile.stdout" 2>&1
chmod +x "$D/original.elf";sha256sum "$D/original.elf" > "$D/EXECUTABLE.sha256"
rc=0;"$B/services/bigframe-integrated01/tools/time" -v -o "$D/native.resource" timeout 3000 "$D/original.elf" > "$D/native.stdout" 2>&1||rc=$?
echo "$rc" > "$D/NATIVE_STATUS"
if [[ "$rc" == 0 ]];then grep -Fxq ALL_PASS "$D/native.stdout";fi
exit "$rc"
