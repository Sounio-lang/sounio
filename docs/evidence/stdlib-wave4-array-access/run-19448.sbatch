#!/bin/bash
#SBATCH --job-name=array-cast-ir-semantics
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/array-semantics10/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/array-semantics10/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/array-semantics10;D=$R/results
U=$B/root-integrated01;RAW=$U/results/madaros
mkdir "$D"
trap 'r=$?;echo "$r" > "$D/COLLECTOR_STATUS";exit "$r"' EXIT
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
cp "$B/services/bigframe-portable09/results/reference" "$D/reference"
sha256sum "$D/reference" > "$D/REFERENCE.sha256"
"$D/reference" --self-test > "$D/ORACLE_CONTROLS.stdout"
export SOUNIO_STDLIB_PATH="$U/source/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_NV2_IR_TRACE=1 SOUNIO_DUMP_MERGED_CALLS=1
ulimit -s 524288;ulimit -c 0
count=0
while IFS= read -r name;do
 [[ -n "$name" && -s "$R/input/controls/$name.sio" ]]
 count=$((count+1))
 for mode in 0 1;do
  export SOUNIO_LOOP_REGIONS="$mode"
  rc=0;timeout 120 "$RAW" --check "$R/input/controls/$name.sio" > "$D/$name$mode.check.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\tcheck\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
  rc=0;timeout 300 "$RAW" compile "$R/input/controls/$name.sio" -o "$D/$name$mode.elf" > "$D/$name$mode.compile.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\tcompile\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
  if [[ "$rc" == 0 && -s "$D/$name$mode.elf" ]];then
   chmod +x "$D/$name$mode.elf"
   sha256sum "$D/$name$mode.elf" >> "$D/EXECUTABLES.sha256"
   rc=0;timeout 30 "$D/$name$mode.elf" > "$D/$name$mode.native.stdout" 2>&1 ||rc=$?
   printf '%s\t%s\tnative\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
   rc=0;"$D/reference" --verify array "$D/$name$mode.native.stdout" > "$D/$name$mode.oracle.stdout" 2>&1 ||rc=$?
   printf '%s\t%s\toracle\t%s\n' "$name" "$mode" "$rc" >> "$D/PHASES.tsv"
  fi
 done
done < "$R/input/CASES.tsv"
[[ "$count" == 15 ]]
sha256sum -c "$D/INPUTS.sha256" > "$D/INPUTS.final.verify"
sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.final.verify"
sha256sum -c "$D/REFERENCE.sha256" > "$D/REFERENCE.final.verify"
echo 'DIAGNOSTIC_ONLY: all IR/native/oracle statuses retained; source contract proposal remains separate.' > "$D/SCOPE"
echo ARRAY_CAST_IR_SEMANTICS_COLLECTED
