#!/bin/bash
#SBATCH --job-name=qpu-p00-raw04-elision-pair
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/qpu-hoist56/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/qpu-hoist56/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
D=$B/services/qpu-hoist56
S=$B/services/qpu-hoist55/input
RAW=$R/results/madaros
C=$B/root-integrated04-capture
mkdir "$D/receipts" "$D/capture" "$D/inputs"
export NODE_BIN=$B/services-tools-recovered/tools/node
ulimit -s 524288
ulimit -c 0
finish(){ r=$?;trap - EXIT;for m in SOURCES TOOLS EXECUTABLES;do if test -s "$D/receipts/$m.sha256";then sha256sum -c "$D/receipts/$m.sha256" >"$D/receipts/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
find "$S" -type f -print0|sort -z|xargs -0 sha256sum >"$D/receipts/SOURCES.sha256"
(find "$R/private-tools-v3" "$R/pthread-installed-v3" "$R/private-runtime" "$G" -type f -print0;printf '%s\0' "$R/tools-env-v3.sh" "$NODE_BIN" "$RAW" "$0")|sort -zu|xargs -0 sha256sum >"$D/receipts/TOOLS.sha256"
printf 'a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333  %s\n' "$RAW" >"$D/receipts/RAW.sha256"
sha256sum -c "$D/receipts/RAW.sha256" >"$D/receipts/RAW.verify"
cp "$C/inputs/nv2_trace_full_reference.cpp" "$C/inputs/trace-compare.cpp" "$D/inputs/"
find "$D/inputs" -type f -print0|sort -z|xargs -0 sha256sum >>"$D/receipts/SOURCES.sha256"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$D/inputs/nv2_trace_full_reference.cpp" -o "$D/parser"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$D/inputs/trace-compare.cpp" -o "$D/trace-compare"
sha256sum "$D/parser" "$D/trace-compare" >"$D/receipts/EXECUTABLES.sha256"
for e in 0 1;do
 export SOUNIO_LOOP_CHECKPOINT_ELISION=$e
 bash "$S/scripts/stdlib/wave4_qpu_hoist_gate.sh" "$D/numeric-$e" "$RAW"
done
echo 0 >"$D/NUMERICAL_STATUS"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_LOOP_REGIONS=1
printf 'case\telision\tcompile\tschema\n' >"$D/capture/PHASES.tsv"
bad=0
for e in 0 1;do
 export SOUNIO_LOOP_CHECKPOINT_ELISION=$e
 for name in cores qpu_imported;do
  src=$D/numeric-0/driver.sio
  if test "$name" = qpu_imported;then src=$S/tests/stdlib/qpu/test_qpu_chsh.sio;fi
  rc=0
  env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE=source timeout 1200 "$RAW" compile "$src" -o "$D/capture/$name.$e.elf" >"$D/capture/$name.$e.stdout" 2>"$D/capture/$name.$e.stderr"||rc=$?
  schema=0
  "$D/parser" --check source "$D/capture/$name.$e.stdout" >"$D/capture/$name.$e.schema.stdout" 2>"$D/capture/$name.$e.schema.stderr"||schema=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$e" "$rc" "$schema" >>"$D/capture/PHASES.tsv"
  grep -E '^(loop_checkpoint_elision:|NV2_IR_FULL_REFUSED)' "$D/capture/$name.$e.stdout" >"$D/capture/$name.$e.audit"||true
  if test "$rc" = 0;then sha256sum "$D/capture/$name.$e.elf" >>"$D/receipts/EXECUTABLES.sha256";fi
  if test "$rc" != 0 || test "$schema" != 0;then bad=1;fi
 done
done
for name in cores qpu_imported;do
 rc=0;"$D/trace-compare" "$D/capture/$name.0.stdout" "$D/capture/$name.1.stdout" >"$D/capture/$name.compare.stdout" 2>"$D/capture/$name.compare.stderr"||rc=$?
 echo "$rc" >"$D/capture/$name.compare.status"
 if test "$rc" != 0;then bad=1;fi
done
echo "$bad" >"$D/CAPTURE_STATUS"
exit "$bad"
