#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
D=$B/compiler-resumed/trace-empty-component01
S=$D/source;O=$D/results
mkdir "$O"
source "$R/tools-env-v3.sh"
export TIME_BIN="$P/tools/time"
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;if test -s "$O/SOURCE.sha256";then (cd "$S";sha256sum -c "$O/SOURCE.sha256") >"$O/SOURCE.final.verify" 2>&1||r=1;fi;sha256sum -c "$O/INPUTS.sha256" >"$O/INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$O/STATUS";exit "$r"; }
trap finish EXIT
sha256sum "$D/input/"* "$D/component.sbatch" "$R/results/madaros" "$R/source/bin/souc-lean-single-x86_64" "$CXX" "$TIME_BIN" "$R/tools-env-v3.sh" >"$O/INPUTS.sha256"
(cd "$R/source";sha256sum -c "$R/results/SOURCE.sha256") >"$O/ROOT.before.verify"
cp -a "$R/source" "$S"
tar -xf "$D/input/overlay.tar" -C "$S"
"$P/tools/node" - "$S" "$D/input/helper.before.sio.in" <<'JS'
const fs=require('fs'),s=process.argv[2],before=process.argv[3],dir=s+'/tests/fixtures/wave4_loop_checkpoint_elision/';
function body(text){const i=text.indexOf('fn native_v2_full_trace_number');if(i<0)throw Error('missing helper');return text.slice(i).trimEnd();}
const old=body(fs.readFileSync(before,'utf8')),fresh=body(fs.readFileSync(dir+'nv2_trace_full_helper.sio.in','utf8')),probe=fs.readFileSync(dir+'nv2_trace_full_probe.sio','utf8');
if(probe.split(fresh).length!==2)throw Error('helper identity');
fs.writeFileSync(dir+'nv2_trace_full_before.sio',probe.replace(fresh,old));
const prod=fs.readFileSync(s+'/self-hosted/native/codegen_x86_linux.sio','utf8');if(!prod.includes(fresh))throw Error('production helper mismatch');
JS
cd "$S"
find self-hosted stdlib tests/fixtures/wave4_loop_checkpoint_elision -type f -print0|sort -z|xargs -0 sha256sum >"$O/SOURCE.sha256"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
RAW=$R/results/madaros;SEED=$R/source/bin/souc-lean-single-x86_64
"$TIME_BIN" -v timeout 300 "$RAW" check tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_probe.sio >"$O/check.stdout" 2>"$O/check.stderr"
"$TIME_BIN" -v timeout 300 "$SEED" tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_probe.sio "$O/component.elf" >"$O/compile.stdout" 2>"$O/compile.stderr"
! grep -E '^error:|^error\[|Error:|field initializer mismatch|would have dropped|cannot compile an assignment' "$O/compile.stdout" "$O/compile.stderr" >"$O/EXPLICIT_ERRORS"
"$TIME_BIN" -v timeout 300 "$SEED" tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_before.sio "$O/before.elf" >"$O/before.compile.stdout" 2>"$O/before.compile.stderr"
! grep -E '^error:|^error\[|Error:|field initializer mismatch|would have dropped|cannot compile an assignment' "$O/before.compile.stdout" "$O/before.compile.stderr" >"$O/BEFORE_EXPLICIT_ERRORS"
chmod +x "$O/component.elf" "$O/before.elf"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_reference.cpp -o "$O/reference" >"$O/cpp.stdout" 2>"$O/cpp.stderr"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$D/input/reference.before.cpp" -o "$O/reference.before" >"$O/cpp.before.stdout" 2>"$O/cpp.before.stderr"
sha256sum "$O/component.elf" "$O/before.elf" "$O/reference" "$O/reference.before" >"$O/ELFS.sha256"
mkdir "$O/before"
printf 'stage\tmode\tproducer\toracle\n' >"$O/before/PHASES.tsv"
for stage in source prepared;do
 for mode in extern0 extern1 extern2 global_base global_large stub_badreg stub_badlabels;do
  env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE="$stage" timeout 30 "$O/before.elf" "$mode" >"$O/before/$stage-$mode.stdout" 2>"$O/before/$stage-$mode.stderr"
  case "$mode" in extern0|stub_badreg|stub_badlabels) "$O/reference.before" --probe "$stage" "$O/before/$stage-$mode.stdout" >"$O/before/$stage-$mode.oracle";;*) "$O/reference" --refusal "$stage" 4 "$O/before/$stage-$mode.stdout" >"$O/before/$stage-$mode.oracle";;esac
  printf '%s\t%s\t0\t0\n' "$stage" "$mode" >>"$O/before/PHASES.tsv"
 done
done
bash tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_gate.sh "$O/component.elf" "$O/reference" "$B/compiler-resumed/loop-full-trace02/results/valid.stdout" "$O/gate" >"$O/gate.stdout" 2>"$O/gate.stderr"
sha256sum -c "$O/ELFS.sha256" >"$O/ELFS.final.verify"
(cd "$R/source";sha256sum -c "$R/results/SOURCE.sha256") >"$O/ROOT.final.verify"
echo DIAGNOSTIC_SCHEMA2_EMPTY_EXTERNS_AND_GLOBAL_DATA_COMPONENT_OK
