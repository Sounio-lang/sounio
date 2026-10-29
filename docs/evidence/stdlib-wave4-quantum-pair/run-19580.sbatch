#!/bin/bash
#SBATCH --job-name=wide-singlefile-projection
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/wide-projection36/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/wide-projection36/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/wide-projection36;D=$R/results;U=$B/root-integrated02
mkdir "$D"
finish(){ r=$?;trap - EXIT;for n in INPUTS COMPILER EXECUTABLES SDK IR;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" > "$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" > "$D/STATUS";exit "$r";};trap finish EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
T=$B/services/bigframe-abi07/gcc13-private/root
find "$T" -type f -print0|sort -z|xargs -0 sha256sum > "$D/SDK.sha256"
export PATH="$T/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/deps"
export COMPILER_PATH="$T/usr/libexec/gcc/x86_64-linux-gnu/13:$T/usr/bin"
export LIBRARY_PATH="$T/usr/lib/gcc/x86_64-linux-gnu/13:$T/usr/lib/x86_64-linux-gnu:$T/lib/x86_64-linux-gnu"
"$T/usr/bin/x86_64-linux-gnu-g++-13" --sysroot="$T" -B"$T/usr/libexec/gcc/x86_64-linux-gnu/13/" -B"$T/usr/bin/" -isystem "$T/usr/include/c++/13" -isystem "$T/usr/include/x86_64-linux-gnu/c++/13" -std=c++23 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc "$R/input/tests/fixtures/wave4_quantum_pair/wide_projection_reference.cpp" -o "$D/reference"

sha256sum "$D/reference" > "$D/EXECUTABLES.sha256"

export SOUNIO_STDLIB_PATH="$B/services/quantum-pair-library28/after/stdlib" SOUNIO_LOOP_REGIONS=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_NV2_IR_TRACE=1
ulimit -s 524288;ulimit -c 0
src="$R/input/tests/fixtures/wave4_quantum_pair/wide_counts_projection.sio"
timeout 120 "$U/results/madaros" --check "$src" > "$D/check.stdout" 2>&1
timeout 180 "$U/results/madaros" compile "$src" -o "$D/projection.elf" > "$D/compile.stdout" 2>&1
chmod +x "$D/projection.elf";sha256sum "$D/projection.elf" >> "$D/EXECUTABLES.sha256"
sha256sum "$D/compile.stdout" > "$D/IR.sha256"
timeout 120 "$D/projection.elf" > "$D/native.stdout" 2>&1
"$D/reference" "$D/native.stdout" > "$D/oracle.stdout"
echo SINGLEFILE_PROJECTION_NOT_IMPORTED_BFS
