#!/bin/bash
#SBATCH --job-name=quantum-linear-pair-roles
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-pair26/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-pair26/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-pair26;D=$R/results;U=$B/root-integrated02
mkdir "$D"
finish(){ r=$?;trap - EXIT;for n in INPUTS COMPILER EXECUTABLES SDK;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" > "$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" > "$D/STATUS";exit "$r";};trap finish EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
T=$B/services/bigframe-abi07/gcc13-private/root
find "$T" -type f -print0|sort -z|xargs -0 sha256sum > "$D/SDK.sha256"
export PATH="$T/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/deps"
export COMPILER_PATH="$T/usr/libexec/gcc/x86_64-linux-gnu/13:$T/usr/bin"
export LIBRARY_PATH="$T/usr/lib/gcc/x86_64-linux-gnu/13:$T/usr/lib/x86_64-linux-gnu:$T/lib/x86_64-linux-gnu"
"$T/usr/bin/x86_64-linux-gnu-g++-13" --sysroot="$T" -B"$T/usr/libexec/gcc/x86_64-linux-gnu/13/" -B"$T/usr/bin/" -isystem "$T/usr/include/c++/13" -isystem "$T/usr/include/x86_64-linux-gnu/c++/13" -std=c++23 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc "$R/input/tests/reference/wave4_quantum_pair_reference.cpp" -o "$D/reference"
"$D/reference" --self-test > "$D/ORACLE_CONTROLS.stdout"
sha256sum "$D/reference" > "$D/EXECUTABLES.sha256"
export SOUNIO_STDLIB_PATH="$B/services/quantum-pair-library25/after/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
printf 'case\tmode\tphase\tstatus\n' > "$D/PHASES.tsv"
fail=0
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS="$mode"
 for name in match let let_layers let_take_field repeated_projection;do
  src="$R/input/tests/fixtures/wave4_quantum_pair/$name.sio"
  rc=0;timeout 120 "$U/results/madaros" --check "$src" > "$D/$name.$mode.check.stdout" 2>&1||rc=$?
  printf '%s\t%s\tcheck\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
  if [[ "$name" == repeated_projection ]];then [[ "$rc" == 1 ]]&&grep -Fq 'error[E039]' "$D/$name.$mode.check.stdout"||fail=1
  else [[ "$rc" == 0 ]]||fail=1;fi
  rc=0;timeout 180 "$U/results/madaros" compile "$src" -o "$D/$name.$mode.elf" > "$D/$name.$mode.compile.stdout" 2>&1||rc=$?
  printf '%s\t%s\tcompile\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
  if [[ "$name" == repeated_projection ]];then
   [[ "$rc" == 1 && ! -e "$D/$name.$mode.elf" ]]&&grep -Fq 'error[E039]' "$D/$name.$mode.compile.stdout"||fail=1
  elif [[ "$rc" == 0 && -s "$D/$name.$mode.elf" ]];then
   chmod +x "$D/$name.$mode.elf";sha256sum "$D/$name.$mode.elf" >> "$D/EXECUTABLES.sha256"
   rc=0;timeout 30 "$D/$name.$mode.elf" > "$D/$name.$mode.native.stdout" 2>&1||rc=$?
   printf '%s\t%s\tnative\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
   [[ "$rc" == 0 ]]||fail=1
   rc=0;"$D/reference" "$(if [[ "$name" == match ]];then echo match;else echo let;fi)" "$D/$name.$mode.native.stdout" > "$D/$name.$mode.oracle.stdout" 2>&1||rc=$?
   printf '%s\t%s\toracle\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
   [[ "$rc" == 0 ]]||fail=1
  else fail=1;fi
 done
done
[[ "$fail" == 0 ]]
echo LINEAR_PAIR_ROLES_AND_NEGATIVE_OK
