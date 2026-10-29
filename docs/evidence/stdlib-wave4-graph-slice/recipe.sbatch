#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-graph-slice01
D=$R/results
RAW=$B/root-integrated02/results/madaros
mkdir "$D"
finish(){ r=$?;trap - EXIT;for m in INPUTS RAW SOURCE EXECUTABLES;do if [[ -s "$D/$m.sha256" ]];then sha256sum -c "$D/$m.sha256" >"$D/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum >"$D/INPUTS.sha256"
cp "$B/root-integrated02/RAW_READY" "$D/RAW.sha256"
sha256sum -c "$D/RAW.sha256" >"$D/RAW.before.verify"
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$R/input/tests/fixtures/wave4_graph_slice/reference.cpp" -o "$D/reference"
sha256sum "$D/reference" >"$D/EXECUTABLES.sha256"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
for variant in before after;do
 mkdir "$R/$variant"
 cp -a "$B/root-integrated02/source/stdlib" "$R/$variant/stdlib"
 cp "$R/input/tests/fixtures/wave4_graph_slice/types.$variant.sio" "$R/$variant/stdlib/graph/types.sio"
 find "$R/$variant" -type f -print0|sort -z|xargs -0 sha256sum >>"$D/SOURCE.sha256"
 export SOUNIO_STDLIB_PATH="$R/$variant/stdlib"
 for mode in 0 1;do
  export SOUNIO_LOOP_REGIONS=$mode
  original="$R/input/tests/stdlib/graph/test_graph_types_e2e.sio"
  timeout 1200 "$RAW" check "$original" >"$D/$variant.$mode.check.stdout" 2>&1
  rc=0
  timeout 1200 "$RAW" compile "$original" -o "$D/$variant.$mode.elf" >"$D/$variant.$mode.compile.stdout" 2>&1||rc=$?
  printf '%s\t%s\tcompile\t%s\n' "$variant" "$mode" "$rc" >>"$D/PHASES.tsv"
  if [[ "$variant" == before ]];then
   [[ "$rc" == 1 && ! -s "$D/$variant.$mode.elf" ]]
   grep -Fq 'name=len reason=missing_lowered_body' "$D/$variant.$mode.compile.stdout"
   continue
  fi
  [[ "$rc" == 0 && -s "$D/$variant.$mode.elf" ]]
  chmod +x "$D/$variant.$mode.elf"
  sha256sum "$D/$variant.$mode.elf" >>"$D/EXECUTABLES.sha256"
  timeout 30 "$D/$variant.$mode.elf" >"$D/$variant.$mode.native.stdout" 2>&1
  grep -Fxq GRAPH_TYPES_OK "$D/$variant.$mode.native.stdout"
  timeout 1200 "$RAW" compile "$R/input/tests/fixtures/wave4_graph_slice/support_size.sio" -o "$D/support.$mode.elf" >"$D/support.$mode.compile.stdout" 2>&1
  chmod +x "$D/support.$mode.elf"
  sha256sum "$D/support.$mode.elf" >>"$D/EXECUTABLES.sha256"
  timeout 30 "$D/support.$mode.elf" >"$D/support.$mode.native.stdout" 2>&1
  "$D/reference" "$D/support.$mode.native.stdout" >"$D/support.$mode.oracle.stdout"
 done
done
echo GRAPH_FIELD_SLICE_LOCAL_AND_ORIGINALS_OK
