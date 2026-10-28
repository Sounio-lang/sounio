#!/usr/bin/env bash
# Real bounded workers + exact ADC calculation + incremental Arrow IPC.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 64
OUT=$(realpath -m -- "$1")
[[ ! -e "$OUT" ]] || { echo "fresh output required" >&2;exit 64; }
if [[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]];then AUTHORITY="slurm:$SLURM_JOB_ID"
elif [[ ${GITHUB_ACTIONS:-} == true && ${GITHUB_RUN_ID:-} =~ ^[1-9][0-9]*$ && ${GITHUB_JOB:-} =~ ^[A-Za-z0-9_.-]+$ ]];then AUTHORITY="github-actions:$GITHUB_RUN_ID:$GITHUB_JOB"
else echo "scheduled execution required" >&2;exit 64;fi
: "${MADAROS_RAW_BIN:?qualified raw compiler}"
: "${ARROW_CPP_ROOT:?Arrow C++26 SDK usr prefix}"
RAW=$(realpath -e "$MADAROS_RAW_BIN")
SDK=$(realpath -e "$ARROW_CPP_ROOT")
[[ -x "$RAW" && -f "$SDK/include/arrow/api.h" ]] || exit 64
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
mkdir "$OUT/inputs" "$OUT/runs"
printf '%s\n' "$AUTHORITY" > "$OUT/EXECUTION_AUTHORITY"
CXX=${CXX:-g++}
WRAPPER="$ROOT/scripts/ci/souc-native-wrapper.sh"
export SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export LD_LIBRARY_PATH="$OUT:$SDK/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 524288
ulimit -c 0
finish(){
 local r=$? changed=0
 trap - EXIT
 for name in SOURCES TOOLS EXECUTABLES INPUTS DYNAMIC ARTIFACTS SDK_HEADERS;do
  if [[ -s "$OUT/$name.sha256" ]];then sha256sum -c "$OUT/$name.sha256" > "$OUT/$name.final.verify" 2>&1 || changed=1;fi
 done
 [[ $changed == 0 ]] || r=1
 printf '%s\n' "$r" > "$OUT/STATUS"
 [[ "$r" != 0 ]] || echo WAVE3_NATIVE_POOL_GATE_OK
 exit "$r"
}
trap finish EXIT
(
 cd "$ROOT"
 find stdlib tests/stdlib/wave3-native-pool -type f -print0
 printf '%s\0' tests/stdlib/massive/test_csv_stream.sio tests/stdlib/massive/test_deadline_socket.sio runtime/stdlib_platform.cpp scripts/stdlib/wave3_native_pool_gate.sh scripts/ci/souc-native-wrapper.sh tests/reference/wave3_native_pool_reference.cpp tests/reference/wave3_adc_pipeline_reference.cpp
) | sort -zu | while IFS= read -r -d '' p;do sha256sum "$ROOT/$p";done > "$OUT/SOURCES.sha256"
sha256sum "$RAW" "$(command -v "$CXX")" "$SDK/lib/x86_64-linux-gnu/libarrow.so.2600.0.0" > "$OUT/TOOLS.sha256"
"$CXX" --version > "$OUT/CXX_VERSION"
printf 'case\tphase\tstatus\n' > "$OUT/RESULTS.tsv"
: > "$OUT/EXECUTABLES.sha256"
: > "$OUT/INPUTS.sha256"
phase(){
 local name=$1 kind=$2 seconds=$3 r=0
 shift 3
 timeout "$seconds" "$@" > "$OUT/$name.$kind.stdout" 2> "$OUT/$name.$kind.stderr" || r=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$r" >> "$OUT/RESULTS.tsv"
 return "$r"
}
compile_sio(){
 local name=$1 source=$2
 sha256sum "$source" >> "$OUT/INPUTS.sha256"
 phase "$name" check 300 bash "$WRAPPER" check "$source"
 grep -Fxq 'check: OK' "$OUT/$name.check.stdout"
 phase "$name" compile 600 bash "$WRAPPER" compile "$source" -o "$OUT/$name.elf"
 [[ -s "$OUT/$name.elf" && $(od -An -tx1 -N4 "$OUT/$name.elf"|tr -d ' \n') == 7f454c46 ]]
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
}

# Seal the compiler driver/backend/linker and SDK headers before compiling.
for tool in "$(command -v "$CXX")" "$("$CXX" -print-prog-name=cc1plus)" "$("$CXX" -print-prog-name=collect2)" "$("$CXX" -print-prog-name=as)" "$("$CXX" -print-prog-name=ld)" "$("$CXX" -print-file-name=libstdc++.so)" "$("$CXX" -print-file-name=libgcc_s.so.1)";do
 if [[ "$tool" != */* ]];then tool=$(command -v "$tool");fi
 tool=$(realpath -e "$tool")
 sha256sum "$tool" >> "$OUT/TOOLS.sha256"
 ldd "$tool" >> "$OUT/CXX_TOOLS.ldd"
done
! grep -q 'not found' "$OUT/CXX_TOOLS.ldd"
awk '/=> \//{print $3} /^[[:space:]]*\//{print $1}' "$OUT/CXX_TOOLS.ldd" | sort -u | xargs -r sha256sum > "$OUT/DYNAMIC.sha256"
find "$SDK/include" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SDK_HEADERS.sha256"
dynamic_manifest(){
 local executable=$1 name=$2
 ldd "$executable" > "$OUT/$name.ldd"
 ! grep -q 'not found' "$OUT/$name.ldd"
 awk '/=> \//{print $3} /^[[:space:]]*\//{print $1}' "$OUT/$name.ldd" | sort -u | xargs -r sha256sum >> "$OUT/DYNAMIC.sha256"
}

phase runtime build 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$ROOT/runtime/stdlib_platform.cpp" -o "$OUT/libstdlib_platform.so"
sha256sum "$OUT/libstdlib_platform.so" >> "$OUT/EXECUTABLES.sha256"
for mode in normal sanitized;do
 flags=(-O2)
 if [[ $mode == sanitized ]];then flags=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined);fi
 phase "pool-reference-$mode" build 180 "$CXX" -std=c++23 "${flags[@]}" -Wall -Wextra -Werror -pthread "$ROOT/tests/reference/wave3_native_pool_reference.cpp" "$ROOT/runtime/stdlib_platform.cpp" -ldl -Wl,--wrap=pthread_create -o "$OUT/pool-reference-$mode"
 sha256sum "$OUT/pool-reference-$mode" >> "$OUT/EXECUTABLES.sha256"
 dynamic_manifest "$OUT/pool-reference-$mode" "pool-reference-$mode"
 phase "pool-reference-$mode" native 120 env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$OUT/pool-reference-$mode"
 grep -Fxq 'NATIVE_POOL_CPP_OK tasks=10000 peak=2 cleanup=0 partial_spawn=joined' "$OUT/pool-reference-$mode.native.stdout"
 grep -Fxq 'TCP_READER_CPP_OK lease=exclusive dup=independent deadline=absolute excess=peeked' "$OUT/pool-reference-$mode.native.stdout"
done
phase adc-reference build 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pthread -isystem "$SDK/include" "$ROOT/tests/reference/wave3_adc_pipeline_reference.cpp" -L"$SDK/lib/x86_64-linux-gnu" -larrow -o "$OUT/adc-reference"
sha256sum "$OUT/adc-reference" >> "$OUT/EXECUTABLES.sha256"
dynamic_manifest "$OUT/adc-reference" adc-reference
for name in pool csv_checked adc_csv_arrow;do
 compile_sio "$name" "$ROOT/tests/stdlib/wave3-native-pool/$name.sio"
 mkdir "$OUT/runs/$name"
 (cd "$OUT/runs/$name";phase "$name" native 120 "$OUT/$name.elf")
done
grep -Fxq NATIVE_THREAD_POOL_OK "$OUT/pool.native.stdout"
grep -Fxq CSV_ROW_I64_CHECKED_OK "$OUT/csv_checked.native.stdout"
grep -Fxq ADC_CSV_ARROW_OK "$OUT/adc_csv_arrow.native.stdout"
phase public-adc verify 120 "$OUT/adc-reference" verify "$OUT/runs/adc_csv_arrow/adc_fixture_input.csv" "$OUT/runs/adc_csv_arrow/adc_fixture_output.arrow"
cp "$ROOT/tests/stdlib/wave3-native-pool/adc_driver.sio.in" "$OUT/inputs/driver.original.sio"
printf 'variant\tworkers\tread_chunk_bytes\n' > "$OUT/SUBSTITUTIONS.tsv"
for variant in one two four scale;do
 workers=4;chunk=17
 case "$variant" in one) workers=1;chunk=1;;two) workers=2;chunk=3;;scale) chunk=4096;;esac
 sed -e "s/native_pool_reservation(\\&p,4,16)/native_pool_reservation(\\&p,$workers,16)/" -e "s/workers:4,task_capacity/workers:$workers,task_capacity/" -e "s/read_chunk_bytes:17/read_chunk_bytes:$chunk/" "$OUT/inputs/driver.original.sio" > "$OUT/inputs/$variant.sio"
 printf '%s\t%s\t%s\n' "$variant" "$workers" "$chunk" >> "$OUT/SUBSTITUTIONS.tsv"
 compile_sio "$variant" "$OUT/inputs/$variant.sio"
 mkdir "$OUT/runs/$variant"
 rows=257
 [[ $variant != scale ]] || rows=1000000
 phase "$variant" generate 60 "$OUT/adc-reference" generate "$OUT/runs/$variant" "$rows"
 sha256sum "$OUT/runs/$variant/input.csv" >> "$OUT/INPUTS.sha256"
 (cd "$OUT/runs/$variant";phase "$variant" native 600 /usr/bin/time -v -o "$OUT/$variant.resource" "$OUT/$variant.elf")
 grep -Fxq ADC_PIPELINE_RESOURCES_CLOSED "$OUT/$variant.native.stdout"
 grep -Fxq ADC_CSV_ARROW_OK "$OUT/$variant.native.stdout"
 phase "$variant" verify 120 "$OUT/adc-reference" verify "$OUT/runs/$variant/input.csv" "$OUT/runs/$variant/output.arrow"
done
cmp "$OUT/runs/one/input.csv" "$OUT/runs/two/input.csv"
cmp "$OUT/runs/two/input.csv" "$OUT/runs/four/input.csv"
cmp "$OUT/runs/one/output.arrow" "$OUT/runs/two/output.arrow"
cmp "$OUT/runs/two/output.arrow" "$OUT/runs/four/output.arrow"
for mutation in wrong-value wrong-order;do
 phase "$mutation" generate 60 "$OUT/adc-reference" mutate "$OUT/runs/four/output.arrow" "$OUT/$mutation.arrow" "$mutation"
 r=0
 phase "$mutation" judge-reject 120 "$OUT/adc-reference" verify "$OUT/runs/four/input.csv" "$OUT/$mutation.arrow" || r=$?
 [[ $r == 1 ]]
done
cp "$OUT/runs/four/output.arrow" "$OUT/truncated.arrow"
truncate -s -1 "$OUT/truncated.arrow"
cp "$OUT/runs/four/output.arrow" "$OUT/trailing.arrow"
printf x >> "$OUT/trailing.arrow"
cp "$OUT/runs/four/output.arrow" "$OUT/no-eos.arrow"
truncate -s -8 "$OUT/no-eos.arrow"
cp "$OUT/runs/four/output.arrow" "$OUT/extra-eos.arrow"
printf '\377\377\377\377\0\0\0\0' >> "$OUT/extra-eos.arrow"
for mutation in truncated trailing no-eos extra-eos;do
 r=0
 phase "$mutation" judge-reject 120 "$OUT/adc-reference" verify "$OUT/runs/four/input.csv" "$OUT/$mutation.arrow" || r=$?
 [[ $r == 1 ]]
done
negative_input(){
 local name=$1 expected=$2 contents=$3 r=0
 mkdir "$OUT/runs/$name"
 printf '%b' "$contents" > "$OUT/runs/$name/input.csv"
 sha256sum "$OUT/runs/$name/input.csv" >> "$OUT/INPUTS.sha256"
 (cd "$OUT/runs/$name";phase "$name" native-refusal 120 "$OUT/four.elf") || r=$?
 [[ $r == "$expected" ]]
 [[ $(head -1 "$OUT/$name.native-refusal.stdout") == "$expected" ]]
 grep -Fxq ADC_PIPELINE_RESOURCES_CLOSED "$OUT/$name.native-refusal.stdout"
 ! grep -Fxq ADC_CSV_ARROW_OK "$OUT/$name.native-refusal.stdout"
}
negative_input header 22 'id,amplitude_adc\n0,1\n'
negative_input duplicate 22 'sample_id,amplitude_adc\n0,1\n0,2\n'
negative_input missing-id 22 'sample_id,amplitude_adc\n1,1\n'
negative_input empty 22 'sample_id,amplitude_adc\n0,\n'
negative_input fraction 22 'sample_id,amplitude_adc\n0,1.2\n'
negative_input overflow 34 'sample_id,amplitude_adc\n0,9223372036854775808\n'
negative_input underflow 34 'sample_id,amplitude_adc\n0,-9223372036854775809\n'
negative_input domain 34 'sample_id,amplitude_adc\n0,1000001\n'
negative_input quote 22 'sample_id,amplitude_adc\n0,"12\n'
negative_input columns 22 'sample_id,amplitude_adc\n0,1,2\n'
negative_input utf8 84 'sample_id,amplitude_adc\n0,\377\n'
for control in max-samples row-budget pool-budget invalid-config;do
 case "$control" in
 max-samples) subst='s/max_samples:1000000/max_samples:16/';expected=75;;
 row-budget) subst='s/max_row_bytes:128/max_row_bytes:4/';expected=105;;
 pool-budget) subst='s/pool_budget_bytes:budget.value/pool_budget_bytes:0/';expected=105;;
 invalid-config) subst='s/read_chunk_bytes:17/read_chunk_bytes:0/';expected=22;;
 esac
 sed "$subst" "$OUT/inputs/driver.original.sio" > "$OUT/inputs/$control.sio"
 compile_sio "$control" "$OUT/inputs/$control.sio"
 mkdir "$OUT/runs/$control"
 cp "$OUT/runs/four/input.csv" "$OUT/runs/$control/input.csv"
 sha256sum "$OUT/runs/$control/input.csv" >> "$OUT/INPUTS.sha256"
 r=0
 (cd "$OUT/runs/$control";phase "$control" native-refusal 120 "$OUT/$control.elf") || r=$?
 [[ $r == "$expected" && $(head -1 "$OUT/$control.native-refusal.stdout") == "$expected" ]]
 grep -Fxq ADC_PIPELINE_RESOURCES_CLOSED "$OUT/$control.native-refusal.stdout"
done
r=0
phase double-close check-refusal 300 bash "$WRAPPER" check "$ROOT/tests/stdlib/wave3-native-pool/double_close.sio" || r=$?
[[ $r != 0 ]]
grep -Eq 'E039' "$OUT/double-close.check-refusal."{stdout,stderr}
cp "$ROOT/tests/stdlib/wave3-native-pool/captured_callback.sio.in" "$OUT/inputs/captured_callback.sio"
phase captured check 300 bash "$WRAPPER" check "$OUT/inputs/captured_callback.sio"
r=0
phase captured compile-refusal 600 bash "$WRAPPER" compile "$OUT/inputs/captured_callback.sio" -o "$OUT/captured.elf" || r=$?
[[ $r != 0 && ! -s "$OUT/captured.elf" ]]
grep -Eq '^error: a capturing closure' "$OUT/captured.compile-refusal."{stdout,stderr}


# Replay original strict CSV bytes/Unicode/quoting with path-only relocation.
mkdir "$OUT/runs/original-csv"
sed "s|/tmp/sounio_wave3_csv_stream_fixture.csv|$OUT/runs/original-csv/input.csv|g" "$ROOT/tests/stdlib/massive/test_csv_stream.sio" > "$OUT/inputs/original-csv.sio"
sed "s|$OUT/runs/original-csv/input.csv|/tmp/sounio_wave3_csv_stream_fixture.csv|g" "$OUT/inputs/original-csv.sio" > "$OUT/inputs/original-csv.reversed"
cmp "$ROOT/tests/stdlib/massive/test_csv_stream.sio" "$OUT/inputs/original-csv.reversed"
compile_sio original-csv "$OUT/inputs/original-csv.sio"
(cd "$OUT/runs/original-csv";phase original-csv native 120 "$OUT/original-csv.elf")
grep -Fxq CSV_STREAM_OK "$OUT/original-csv.native.stdout"
compile_sio original-deadline "$ROOT/tests/stdlib/massive/test_deadline_socket.sio"
phase original-deadline native 120 "$OUT/original-deadline.elf"
grep -Fxq DEADLINE_SOCKET_OK "$OUT/original-deadline.native.stdout"
r=0
phase workspace-private check-refusal 300 bash "$WRAPPER" check "$ROOT/tests/stdlib/wave3-native-pool/workspace_private.sio" || r=$?
[[ $r != 0 ]]
grep -Eq E259 "$OUT/workspace-private.check-refusal."{stdout,stderr}

# Real TCP ingress: dup owner + exclusive read lease + absolute deadline.
compile_sio adc_tcp_arrow "$ROOT/tests/stdlib/wave3-native-pool/adc_tcp_arrow.sio"
mkdir "$OUT/runs/adc_tcp_arrow"
(cd "$OUT/runs/adc_tcp_arrow";phase adc_tcp_arrow native 120 "$OUT/adc_tcp_arrow.elf")
grep -Fxq ADC_TCP_ARROW_OK "$OUT/adc_tcp_arrow.native.stdout"
phase public-tcp verify 120 "$OUT/adc-reference" verify "$OUT/runs/adc_csv_arrow/adc_fixture_input.csv" "$OUT/runs/adc_tcp_arrow/adc_tcp_fixture_output.arrow"
cmp "$OUT/runs/adc_csv_arrow/adc_fixture_output.arrow" "$OUT/runs/adc_tcp_arrow/adc_tcp_fixture_output.arrow"
cp "$ROOT/tests/stdlib/wave3-native-pool/adc_tcp_driver.sio.in" "$OUT/inputs/tcp.sio"
compile_sio tcp "$OUT/inputs/tcp.sio"
tcp_run(){
 local name=$1 elf=$2 input=$3 chunk=$4 delay=$5 mode=$6 expected=$7 port='' producer_rc=0 peer_rc=0
 mkdir "$OUT/runs/$name"
 cp "$input" "$OUT/runs/$name/input.csv"
 sha256sum "$OUT/runs/$name/input.csv" >> "$OUT/INPUTS.sha256"
 (cd "$OUT/runs/$name";phase "$name" native 180 /usr/bin/time -v -o "$OUT/$name.resource" "$elf") &
 local producer=$!
 for attempt in {1..250};do
  if grep -Fxq TCP_LISTENING "$OUT/$name.native.stdout" 2>/dev/null;then
   port=$(sed -n '2p' "$OUT/$name.native.stdout")
   [[ $port =~ ^[0-9]+$ ]] && break
  fi
  if ! kill -0 "$producer" 2>/dev/null;then break;fi
  sleep 0.02
 done
 if [[ ! $port =~ ^[0-9]+$ ]];then wait "$producer" || true;echo "missing TCP port" >&2;return 1;fi
 phase "$name" peer 180 "$OUT/adc-reference" peer "$port" "$OUT/runs/$name/input.csv" "$chunk" "$delay" "$mode" || peer_rc=$?
 wait "$producer" || producer_rc=$?
 if [[ $expected == 0 ]];then
  [[ $producer_rc == 0 && $peer_rc == 0 ]]
  grep -Fxq ADC_TCP_ARROW_OK "$OUT/$name.native.stdout"
  phase "$name" verify 120 "$OUT/adc-reference" verify "$OUT/runs/$name/input.csv" "$OUT/runs/$name/output.arrow"
 else
  [[ $producer_rc == "$expected" ]]
  [[ $(sed -n '3p' "$OUT/$name.native.stdout") == "$expected" ]]
  grep -Fxq ADC_TCP_RESOURCES_CLOSED "$OUT/$name.native.stdout"
  ! grep -Fxq ADC_TCP_ARROW_OK "$OUT/$name.native.stdout"
  if [[ $peer_rc != 0 ]];then [[ $peer_rc == 1 ]];grep -Eq 'errno=(32|104)' "$OUT/$name.peer.stderr";fi
 fi
}
tcp_run tcp-small "$OUT/tcp.elf" "$OUT/runs/four/input.csv" 4096 0 normal 0
tcp_run tcp-short "$OUT/tcp.elf" "$OUT/runs/four/input.csv" 1 1 slow 0
tcp_run tcp-scale "$OUT/tcp.elf" "$OUT/runs/scale/input.csv" 4096 0 normal 0
cmp "$OUT/runs/four/output.arrow" "$OUT/runs/tcp-small/output.arrow"
cmp "$OUT/runs/four/output.arrow" "$OUT/runs/tcp-short/output.arrow"
cmp "$OUT/runs/scale/output.arrow" "$OUT/runs/tcp-scale/output.arrow"
sed 's/value+120000000000/value+10000000/' "$OUT/inputs/tcp.sio" > "$OUT/inputs/tcp-deadline.sio"
compile_sio tcp-deadline "$OUT/inputs/tcp-deadline.sio"
tcp_run tcp-deadline "$OUT/tcp-deadline.elf" "$OUT/runs/adc_csv_arrow/adc_fixture_input.csv" 4096 250 hold 110
sed 's/deadline,33554432/deadline,16/' "$OUT/inputs/tcp.sio" > "$OUT/inputs/tcp-budget.sio"
compile_sio tcp-budget "$OUT/inputs/tcp-budget.sio"
tcp_run tcp-budget "$OUT/tcp-budget.elf" "$OUT/runs/four/input.csv" 4096 0 normal 105
printf 'sample_id,amplitude_adc\n0,"12' > "$OUT/inputs/tcp-truncated.csv"
tcp_run tcp-truncated "$OUT/tcp.elf" "$OUT/inputs/tcp-truncated.csv" 1 0 normal 22

phase resource-bound verify 60 "$OUT/adc-reference" resources "$OUT/four.resource" "$OUT/scale.resource" "$OUT/tcp-scale.resource"
printf 'Maximum resident set size (kbytes): 1048576\n' > "$OUT/fake-growth.resource"
r=0
phase resource-bound judge-reject 60 "$OUT/adc-reference" resources "$OUT/four.resource" "$OUT/fake-growth.resource" "$OUT/tcp-scale.resource" || r=$?
[[ $r == 1 ]]

find "$OUT/inputs" "$OUT/runs" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/ARTIFACTS.sha256"
sha256sum "$OUT/SUBSTITUTIONS.tsv" >> "$OUT/INPUTS.sha256"
printf '%s\n' WAVE3_NATIVE_POOL_COMPLETE > "$OUT/COMPLETE"
