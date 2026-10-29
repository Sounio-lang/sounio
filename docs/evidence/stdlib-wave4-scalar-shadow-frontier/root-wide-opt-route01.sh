#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
D=$B/root-wide-opt-route01
RAW=$B/root-integrated04/results/madaros
CASE=$B/root-integrated04-capture/cases/wide_projection.sio
AUD=$B/compiler-resumed/wide-mustdef01/reference
mkdir "$D"
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
sha256sum "$RAW" "$CASE" "$AUD" "$R/tools-env-v3.sh" >"$D/INPUTS.sha256"
cp "$B/compiler-resumed/wide-mustdef01/INPUTS.sha256" "$D/AUDITOR_SOURCE.sha256"
sha256sum -c "$D/AUDITOR_SOURCE.sha256" >"$D/AUDITOR_SOURCE.before.verify"
export SOUNIO_STDLIB_PATH="$B/root-bidi-inputs05/source/stdlib" SOUNIO_ORDERED_PATH_TRACE=1 SOUNIO_LOOP_REGIONS=1 SOUNIO_LOOP_CHECKPOINT_ELISION=0 SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE=source SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
for skip in 0 1;do
 args=(compile "$CASE" -o "$D/program-$skip.elf")
 if [[ "$skip" == 1 ]];then args+=(-O);fi
 rc=0
 timeout 1200 "$RAW" "${args[@]}" >"$D/compile-$skip.stdout" 2>"$D/compile-$skip.stderr"||rc=$?
 echo "$rc" >"$D/compile-$skip.STATUS"
 if [[ "$skip" == 0 ]];then [[ "$rc" == 0 ]];fi
 if [[ "$rc" != 0 ]];then continue;fi
 sha256sum "$D/program-$skip.elf" >>"$D/ELFS.sha256"
 "$AUD" "$D/compile-$skip.stdout" >"$D/audit-$skip.stdout" 2>"$D/audit-$skip.stderr"
done
sha256sum -c "$D/AUDITOR_SOURCE.sha256" >"$D/AUDITOR_SOURCE.final.verify"
echo OPT_ROUTE_DIAGNOSTIC_ONLY
