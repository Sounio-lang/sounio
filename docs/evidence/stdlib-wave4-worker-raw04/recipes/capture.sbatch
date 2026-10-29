#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04;S=$R/source;RAW=$R/results/madaros
SDK=$B/services/normal-integrated01/sealed-services04
D=$B/root-integrated04-capture
mkdir "$D";mkdir "$D/source" "$D/cases" "$D/results" "$D/inputs"
source "$R/tools-env-v2.sh"
ulimit -s 524288;ulimit -c 0
bad=0
finish(){ r=$?;trap - EXIT;for m in ROOT SDK SOURCE TOOLS EXECUTABLES;do if test -s "$D/$m.sha256";then sha256sum -c "$D/$m.sha256" >"$D/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
sha256sum -c "$R/results/COMPILER.sha256" >"$D/RAW.before.verify"
read -r expected rest <"$R/RAW_READY";test "$(sha256sum "$RAW"|cut -d' ' -f1)" = "$expected"
(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") >"$D/ROOT.before.verify"
(cd "$SDK";sha256sum -c SEALED_SOURCE.sha256) >"$D/SDK.before.verify"
(cd "$S";find self-hosted stdlib scripts tests runtime -type f -print0|sort -z|xargs -0 sha256sum)|sed "s@  @  $S/@" >"$D/ROOT.sha256"
(cd "$SDK";cat SEALED_SOURCE.sha256)|sed "s@  @  $SDK/@" >"$D/SDK.sha256"
cp -a "$S/stdlib" "$D/source/stdlib"
for p in stdlib/ffi/platform.sio stdlib/ffi/worker.sio stdlib/thread/native.sio stdlib/thread/native_pool.sio stdlib/thread/native_pool_f64.sio;do cp "$SDK/$p" "$D/source/$p";done
F=$S/tests/fixtures/wave4_worker_capability/loader_gate
mkdir "$D/source/stdlib/worker_gate"
ln "$D/source/stdlib/ffi/worker.sio" "$D/source/stdlib/worker_gate/worker_alias.sio"
cp "$D/source/stdlib/ffi/worker.sio" "$D/source/stdlib/worker_gate/worker_copy.sio"
for p in "$F/modules/"*.sio.in;do name=$(basename "$p" .sio.in);cp "$p" "$D/source/stdlib/worker_gate/$name.sio";done
cp "$F/cases/direct.sio.in" "$D/cases/direct.sio"
cp "$S/tests/fixtures/wave4_quantum_pair/wide_counts_projection.sio" "$D/cases/wide_projection.sio"
cp "$S/tests/stdlib/qpu/test_qpu_chsh.sio" "$D/cases/qpu_imported.sio"
cp "$S/tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_reference.cpp" "$D/inputs/"
cp "$S/tests/fixtures/wave4_quantum_pair/wide_counts_reference.cpp" "$S/tests/fixtures/wave4_quantum_pair/wide_projection_reference.cpp" "$D/inputs/"
cp "$R/trace-compare.cpp" "$D/inputs/"
find "$D/source" "$D/cases" "$D/inputs" -type f -print0|sort -z|xargs -0 sha256sum >"$D/SOURCE.sha256"
sha256sum "$RAW" "$R/RAW_READY" "$CXX" "$R/tools-env-v2.sh" "$P/tools/time" "$0" >"$D/TOOLS.sha256"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$D/inputs/nv2_trace_full_reference.cpp" -o "$D/parser"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$D/inputs/trace-compare.cpp" -o "$D/trace-compare"
"$CXX" -std=c++23 -O2 -fno-fast-math -frounding-math -Wall -Wextra -Werror "$D/inputs/wide_projection_reference.cpp" -o "$D/projection-reference"
sha256sum "$D/parser" "$D/trace-compare" "$D/projection-reference" >"$D/EXECUTABLES.sha256"
export SOUNIO_STDLIB_PATH="$D/source/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_LOOP_REGIONS=1
printf 'case\telision\tphase\tstatus\n' >"$D/PHASES.tsv"
phase(){ local name=$1 elision=$2 kind=$3;shift 3;last=0;"$P/tools/time" -v "$@" >"$D/results/$name.$elision.$kind.stdout" 2>"$D/results/$name.$elision.$kind.stderr"||last=$?;printf '%s\t%s\t%s\t%s\n' "$name" "$elision" "$kind" "$last" >>"$D/PHASES.tsv"; }
for e in 0 1;do
 export SOUNIO_LOOP_CHECKPOINT_ELISION=$e
 for n in direct wide_projection qpu_imported;do
  phase "$n" "$e" check timeout 1200 "$RAW" check "$D/cases/$n.sio"
  test "$last" = 0 || bad=1
  phase "$n" "$e" compile env SOUNIO_NV2_IR_TRACE_FULL=1 SOUNIO_NV2_IR_TRACE_FULL_STAGE=source timeout 1200 "$RAW" compile "$D/cases/$n.sio" -o "$D/$n.$e.elf"
  compile_status=$last
  if test "$compile_status" != 0;then bad=1;fi
  phase "$n" "$e" schema "$D/parser" --check source "$D/results/$n.$e.compile.stdout"
  test "$last" = 0 || bad=1
  grep '^loop_checkpoint_elision:' "$D/results/$n.$e.compile.stdout" >"$D/results/$n.$e.audit.txt" || true
  if test "$compile_status" = 0 && test "$n" != direct;then
   test -s "$D/$n.$e.elf";chmod +x "$D/$n.$e.elf";sha256sum "$D/$n.$e.elf" >>"$D/EXECUTABLES.sha256"
   seconds=30;if test "$n" = wide_projection;then seconds=60;fi
   phase "$n" "$e" native timeout "$seconds" "$D/$n.$e.elf"
   test "$last" = 0 || bad=1
   if test "$n" = wide_projection;then phase "$n" "$e" oracle "$D/projection-reference" "$D/results/$n.$e.native.stdout";test "$last" = 0 || bad=1
   else grep -Fxq ALL_PASS "$D/results/$n.$e.native.stdout" || bad=1;fi
  fi
 done
done
for n in direct wide_projection qpu_imported;do
 phase "$n" compare counts "$D/trace-compare" "$D/results/$n.0.compile.stdout" "$D/results/$n.1.compile.stdout"
 test "$last" = 0 || bad=1
done
exit "$bad"
