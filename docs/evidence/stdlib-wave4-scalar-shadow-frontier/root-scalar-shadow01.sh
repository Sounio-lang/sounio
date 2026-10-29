#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
D=$B/root-scalar-shadow01
I=$B/root-scalar-shadow01-input
RAW=$B/root-integrated04/results/madaros
AUD=$B/compiler-resumed/wide-mustdef01/reference
mkdir "$D"
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
find "$I" -type f -print0|sort -z|xargs -0 sha256sum >"$D/INPUTS.sha256"
sha256sum "$RAW" "$AUD" "$R/tools-env-v3.sh" "$CXX" "$(g++ -print-prog-name=cc1plus)" >"$D/TOOLS.sha256"
cat "$D/TOOLS.sha256" >>"$D/INPUTS.sha256"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$I/reference.cpp" -o "$D/reference" >"$D/reference.build.stdout" 2>"$D/reference.build.stderr"
timeout 30 "$D/reference" >"$D/reference.stdout"
sha256sum "$D/reference" >"$D/ELFS.sha256"
export SOUNIO_STDLIB_PATH="$B/root-bidi-inputs05/source/stdlib" SOUNIO_ORDERED_PATH_TRACE=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE=source
export LD_LIBRARY_PATH="$B/root-bidi-inputs07/results/native:$LD_LIBRARY_PATH"
ulimit -s 524288;ulimit -c 0
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS=$mode
 for case in scalar assertion-mutant;do
  timeout 1200 "$RAW" check "$I/$case.sio" >"$D/$case-$mode.check.stdout" 2>"$D/$case-$mode.check.stderr"
  timeout 1200 "$RAW" compile "$I/$case.sio" -o "$D/$case-$mode.elf" >"$D/$case-$mode.compile.stdout" 2>"$D/$case-$mode.compile.stderr"
  chmod +x "$D/$case-$mode.elf"
  sha256sum "$D/$case-$mode.elf" >>"$D/ELFS.sha256"
  "$AUD" "$D/$case-$mode.compile.stdout" >"$D/$case-$mode.audit.stdout" 2>"$D/$case-$mode.audit.stderr"
  rc=0;timeout 30 "$D/$case-$mode.elf" >"$D/$case-$mode.native.stdout" 2>"$D/$case-$mode.native.stderr"||rc=$?
  echo "$rc" >"$D/$case-$mode.NATIVE_STATUS"
  if [[ "$case" == scalar ]];then [[ "$rc" == 0 ]];else [[ "$rc" == 1 ]];fi
 done
done
echo MINIMAL_SHADOW_NUMERICS_AND_ASSERTION_CONTROL_ONLY
