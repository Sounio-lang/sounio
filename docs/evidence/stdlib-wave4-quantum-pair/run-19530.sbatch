#!/bin/bash
#SBATCH --job-name=quantum-observables-cpp
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-observable27/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-observable27/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-observable27;D=$R/results;U=$B/root-integrated02;L=$B/services/quantum-pair-library25
mkdir "$D"
finish(){ r=$?;trap - EXIT;for n in INPUTS COMPILER EXECUTABLES SDK LIBRARIES;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" > "$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" > "$D/STATUS";exit "$r";};trap finish EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
find "$L/before/stdlib" "$L/after/stdlib" -type f -print0|sort -z|xargs -0 sha256sum > "$D/LIBRARIES.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
T=$B/services/bigframe-abi07/gcc13-private/root
find "$T" -type f -print0|sort -z|xargs -0 sha256sum > "$D/SDK.sha256"
export PATH="$T/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/deps"
export COMPILER_PATH="$T/usr/libexec/gcc/x86_64-linux-gnu/13:$T/usr/bin"
export LIBRARY_PATH="$T/usr/lib/gcc/x86_64-linux-gnu/13:$T/usr/lib/x86_64-linux-gnu:$T/lib/x86_64-linux-gnu"
"$T/usr/bin/x86_64-linux-gnu-g++-13" --sysroot="$T" -B"$T/usr/libexec/gcc/x86_64-linux-gnu/13/" -B"$T/usr/bin/" -isystem "$T/usr/include/c++/13" -isystem "$T/usr/include/x86_64-linux-gnu/c++/13" -std=c++23 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc "$R/input/tests/fixtures/wave4_quantum_pair/observables_reference.cpp" -o "$D/reference"
"$D/reference" --self-test > "$D/ORACLE_CONTROLS.stdout"
sha256sum "$D/reference" > "$D/EXECUTABLES.sha256"

export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
phase(){ local label=$1 seconds=$2;shift 2;local r=0;timeout "$seconds" "$@" > "$D/$label.stdout" 2>&1||r=$?;printf '%s\t%s\n' "$label" "$r" >> "$D/PHASES.tsv";return "$r"; }
fail=0
for variant in before after;do
 export SOUNIO_STDLIB_PATH="$L/$variant/stdlib"
 for mode in 0 1;do
  export SOUNIO_LOOP_REGIONS="$mode"
  src="$R/input/tests/fixtures/wave4_quantum_pair/observables.sio"
  rc=0;phase "$variant.$mode.check" 1200 "$U/results/madaros" --check "$src"||rc=$?
  if [[ "$variant" == before ]];then [[ "$rc" == 1 ]]&&grep -Fq 'error[E039]' "$D/$variant.$mode.check.stdout"||fail=1;else [[ "$rc" == 0 ]]||fail=1;fi
  rc=0;phase "$variant.$mode.compile" 1200 "$U/results/madaros" compile "$src" -o "$D/$variant.$mode.elf"||rc=$?
  if [[ "$variant" == before ]];then [[ "$rc" == 1 && ! -e "$D/$variant.$mode.elf" ]]&&grep -Fq 'error[E039]' "$D/$variant.$mode.compile.stdout"||fail=1
  elif [[ "$rc" == 0 && -s "$D/$variant.$mode.elf" ]];then
   chmod +x "$D/$variant.$mode.elf";sha256sum "$D/$variant.$mode.elf" >> "$D/EXECUTABLES.sha256"
   rc=0;phase "$variant.$mode.native" 30 "$D/$variant.$mode.elf"||rc=$?;[[ "$rc" == 0 ]]||fail=1
   rc=0;phase "$variant.$mode.oracle" 30 "$D/reference" "$D/$variant.$mode.native.stdout"||rc=$?;[[ "$rc" == 0 ]]||fail=1
  else fail=1;fi
 done
done
[[ "$fail" == 0 ]]
echo QUANTUM_CPP_OBSERVABLES_GATE_OK
