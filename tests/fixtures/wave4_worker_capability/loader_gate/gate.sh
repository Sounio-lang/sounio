#!/bin/bash
# Run on the same immutable opt-in raw as the whole-stdlib integration gates.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Slurm required' >&2;exit 2; }
[[ $# == 5 ]] || { echo 'usage: gate.sh ROOT_SNAPSHOT SEALED_SERVICES FRESH_OUT RAW_READY RAW' >&2;exit 2; }
BASE=$(realpath -e "$1")
SERVICES=$(realpath -e "$2")
OUT=$(realpath -m "$3")
READY=$(realpath -e "$4")
RAW=$(realpath -e "$5")
[[ ! -e "$OUT" && -s "$READY" && -x "$RAW" && -d "$BASE/stdlib" ]] || exit 2
read -r expected remainder < "$READY"
[[ $expected =~ ^[0-9a-f]{64}$ && $(sha256sum "$RAW"|cut -d ' ' -f1) == "$expected" ]] || { echo RAW_READY_SHA_MISMATCH >&2;exit 3; }
TEMPLATES=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/source" "$OUT/inputs" "$OUT/cases" "$OUT/results"
D=$OUT/results
finish(){
 local result=$? changed=0
 trap - EXIT
 for manifest in TOOLS SOURCE EXECUTABLES;do
  if [[ -s "$D/$manifest.sha256" ]];then sha256sum -c "$D/$manifest.sha256" > "$D/$manifest.final.verify" 2>&1 ||changed=1;fi
 done
 [[ $changed == 0 ]] ||result=1
 printf '%s\n' "$result" > "$D/STATUS"
 if [[ $result == 0 ]];then echo WORKER_LOADER_BFS_NATIVE_GATE_OK;fi
 exit "$result"
}
trap finish EXIT
ulimit -s 524288
ulimit -c 0
export PATH=/usr/bin:/bin
cp -a "$BASE/stdlib" "$OUT/source/stdlib"
cp -a "$TEMPLATES" "$OUT/inputs/fixtures"
for p in stdlib/ffi/platform.sio stdlib/ffi/worker.sio stdlib/thread/native.sio stdlib/thread/native_pool.sio stdlib/thread/native_pool_f64.sio;do cp "$SERVICES/$p" "$OUT/source/$p";done
cp "$SERVICES/runtime/stdlib_platform.cpp" "$OUT/inputs/stdlib_platform.cpp"
mkdir -p "$OUT/source/stdlib/worker_gate"
ln "$OUT/source/stdlib/ffi/worker.sio" "$OUT/source/stdlib/worker_gate/worker_alias.sio"
cp "$OUT/source/stdlib/ffi/worker.sio" "$OUT/source/stdlib/worker_gate/worker_copy.sio"
for p in "$OUT/inputs/fixtures/modules/"*.sio.in;do name=$(basename "$p" .sio.in);cp "$p" "$OUT/source/stdlib/worker_gate/$name.sio";done
for p in "$OUT/inputs/fixtures/cases/"*.sio.in;do name=$(basename "$p" .sio.in);cp "$p" "$OUT/cases/$name.sio";done
for name in refuse_forged_cap refuse_raw_extract refuse_wrong_cap_abi;do cp "$SERVICES/tests/stdlib/wave4-science-pipeline/$name.sio.in" "$OUT/cases/service_$name.sio";done
cp "$SERVICES/SOURCE_HEAD" "$SERVICES/SEALED_SOURCE.sha256" "$OUT/inputs/"
find "$OUT/source" "$OUT/inputs" "$OUT/cases" -type f -print0|sort -z|xargs -0 sha256sum > "$D/SOURCE.sha256"
sha256sum "$RAW" "$READY" "$(command -v g++)" > "$D/TOOLS.sha256"
g++ --version > "$D/CXX_VERSION.txt"
printf 'case\tphase\tstatus\n' > "$D/RESULTS.tsv"
: > "$D/EXECUTABLES.sha256"
export SOUNIO_STDLIB_PATH="$OUT/source/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export LD_LIBRARY_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
phase(){
 local name=$1 kind=$2 seconds=$3 result=0
 shift 3
 timeout "$seconds" "$@" > "$D/$name.$kind.stdout" 2> "$D/$name.$kind.stderr" ||result=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$result" >> "$D/RESULTS.tsv"
 return "$result"
}
phase runtime build 180 g++ -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$OUT/inputs/stdlib_platform.cpp" -o "$OUT/libstdlib_platform.so"
phase reference build 90 g++ -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$OUT/inputs/fixtures/reference.cpp" -o "$OUT/reference"
sha256sum "$OUT/libstdlib_platform.so" "$OUT/reference" >> "$D/EXECUTABLES.sha256"
compile_positive(){
 local name=$1 source=$2 mode=$3
 phase "$name" check 300 env SOUNIO_LOOP_REGIONS="$mode" "$RAW" check "$source"
 grep -Fxq 'check: OK' "$D/$name.check.stdout"
 phase "$name" compile 600 env SOUNIO_LOOP_REGIONS="$mode" "$RAW" compile "$source" -o "$OUT/$name.elf"
 [[ -s "$OUT/$name.elf" && $(od -An -tx1 -N4 "$OUT/$name.elf"|tr -d ' \n') == 7f454c46 ]]
 ! grep -Eq '^error:|^error\[|would have dropped|cannot compile an assignment' "$D/$name.compile.stdout" "$D/$name.compile.stderr"
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" >> "$D/EXECUTABLES.sha256"
 phase "$name" native 120 "$OUT/$name.elf"
}
for name in direct facade homonym alias_bridge;do
 compile_positive "$name" "$OUT/cases/$name.sio" 0
 marker=$(printf '%s' "$name"|tr '[:lower:]' '[:upper:]')
 grep -Fxq "WORKER_LOADER_${marker}_OK" "$D/$name.native.stdout"
done
# Check and compile refusals separately; no negative callback is executed.
count=0
for source in "$OUT/cases/"refuse_*.sio "$OUT/cases/"service_refuse_*.sio;do
 name=$(basename "$source" .sio)
 for kind in check compile;do
  result=0
  if [[ $kind == check ]];then phase "$name" "$kind" 300 "$RAW" check "$source" ||result=$?
  else phase "$name" "$kind" 600 "$RAW" compile "$source" -o "$OUT/$name.elf" ||result=$?;fi
  [[ ( $result == 1 || ( $kind == compile && $result == 26 ) ) && ! -e "$OUT/$name.elf" ]]
  grep -Eiq 'WorkerFn|worker_fn|worker.*refused' "$D/$name.$kind.stdout" "$D/$name.$kind.stderr"
  ! grep -Eq 'E137|error\[E001\]|parser error|parse failed|unreadable import|AST closure incomplete' "$D/$name.$kind.stdout" "$D/$name.$kind.stderr"
 done
 count=$((count+1))
done
[[ $count == 6 ]]
# All four canonical kernels and their scalar/outref helpers are imported from
# the sealed root snapshot, never extracted or reimplemented in this gate.
for mode in 0 1;do
 name=normal_regions$mode
 compile_positive "$name" "$OUT/cases/normal.sio" "$mode"
 grep -Fxq WORKER_LOADER_NORMAL_OK "$D/$name.native.stdout"
 phase "$name" oracle 60 "$OUT/reference" verify "$D/$name.native.stdout"
 grep -Fxq 'WORKER_IMPORTED_NORMAL_CXX23_OK rows=60 kernels=4 bit_concordance=exact probability_relative=2e-12 log_absolute=2e-12' "$D/$name.oracle.stdout"
 for mutation in mutate reorder;do
  phase "$name-$mutation" generate 30 "$OUT/reference" "$mutation" "$D/$name.native.stdout" "$OUT/$name-$mutation.txt"
  result=0;phase "$name-$mutation" oracle-refuse 30 "$OUT/reference" verify "$OUT/$name-$mutation.txt" ||result=$?
  [[ $result == 1 ]]
 done
done
printf 'positive_source_programs=5\npositive_compilation_cases=6\nnegative_source_programs=6\nnegative_check_and_compile_phases=12\ncanonical_rows_per_mode=60\nloop_region_modes=0,1\nmain_production_opt_in=false\n' > "$D/COUNTS.txt"
