#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/root-integrated04
D=$B/compiler-resumed/pbpk-tuple-print02
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
mkdir "$D/source" "$D/source/cases" "$D/results"
O=$D/results;S=$D/source
finish(){ r=$?;trap - EXIT;if test -s "$O/SOURCE.sha256";then sha256sum -c "$O/SOURCE.sha256" >"$O/SOURCE.final.verify" 2>&1||r=1;fi;sha256sum -c "$O/TOOLS.sha256" >"$O/TOOLS.final.verify" 2>&1||r=1;echo "$r" >"$O/COLLECTOR_STATUS";exit "$r"; }
trap finish EXIT
cp -a "$B/root-bidi-inputs05/source/stdlib" "$S/stdlib"
cp "$D/input/"*.sio "$S/cases/"
cp "$D/input/tuple_module.sio" "$S/stdlib/pbpk_tuple_probe.sio"
find "$S" "$D/input" -type f -print0|sort -z|xargs -0 sha256sum >"$O/SOURCE.sha256"
sha256sum "$R/results/madaros" "$CXX" "$P/tools/time" "$D/matrix.sbatch" >"$O/TOOLS.sha256"
sha256sum -c "$R/RAW_READY" >"$O/RAW.before.verify"
sha256sum "$B/root-bidi-inputs05/source/tests/stdlib/chemistry/test_pbpk_clearance_ode.sio" "$R/source/self-hosted/ir/lower.sio" >"$O/ORIGINALS.sha256"
"$CXX" -std=c++23 -O2 -fno-fast-math -Wall -Wextra -Werror -pedantic "$D/input/reference.cpp" -o "$O/reference" >"$O/reference.compile.stdout" 2>"$O/reference.compile.stderr"
sha256sum "$O/reference" >"$O/EXECUTABLES.sha256"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
printf 'case\tloop_regions\tphase\tstatus\n' >"$O/PHASES.tsv"
phase(){ local n=$1 m=$2 p=$3;shift 3;rc=0;"$P/tools/time" -v "$@" >"$O/$n.$m.$p.stdout" 2>"$O/$n.$m.$p.stderr"||rc=$?;printf '%s\t%s\t%s\t%s\n' "$n" "$m" "$p" "$rc" >>"$O/PHASES.tsv"; }
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS=$mode
 for name in nested nested_declared_inner nested_typed;do
  phase "$name" "$mode" check timeout 1200 "$R/results/madaros" check "$S/cases/$name.sio"
  phase "$name" "$mode" compile timeout 1200 "$R/results/madaros" compile "$S/cases/$name.sio" -o "$O/$name.$mode.elf"
  if test "$rc" = 0;then
   test -s "$O/$name.$mode.elf";chmod +x "$O/$name.$mode.elf";sha256sum "$O/$name.$mode.elf" >>"$O/EXECUTABLES.sha256"
   if test "$name" != refuse_aggregate;then
    phase "$name" "$mode" native timeout 30 "$O/$name.$mode.elf"
    phase "$name" "$mode" oracle "$O/reference" "$name" "$O/$name.$mode.native.stdout"
   fi
  else test ! -e "$O/$name.$mode.elf";fi
 done
done
sha256sum -c "$O/EXECUTABLES.sha256" >"$O/EXECUTABLES.final.verify"
sha256sum -c "$O/ORIGINALS.sha256" >"$O/ORIGINALS.final.verify"
echo PBPK_TUPLE_PRINT_MATRIX_CAPTURED_NOT_ALL_CASES_QUALIFIED
