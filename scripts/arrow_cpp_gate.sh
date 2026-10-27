#!/usr/bin/env bash
# Official Arrow C++26 judge; numerical/format acceptance never invokes Python.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
sdk=${ARROW_CPP_PREFIX:-}
if [[ -z "$sdk" || "$sdk" != /* || ! -f "$sdk/usr/include/arrow/api.h" || ! -f "$sdk/usr/lib/x86_64-linux-gnu/libarrow.so" || ! -f "$sdk/usr/lib/x86_64-linux-gnu/libarrow_flight.so" ]];then
 echo "ARROW_CPP_SDK_REQUIRED: absolute Arrow26 SDK with IPC/Flight/JSON/LZ4/Zstd" >&2
 exit 64
fi
if [[ "${1:-}" == --preflight ]];then echo ARROW_CPP_SDK_FILES_PRESENT;exit 0;fi
[[ $# == 0 ]] || exit 64
platform=${STDLIB_PLATFORM_LIBRARY:-}
if [[ "$platform" != /* || "${platform##*/}" != libstdlib_platform.so || ! -f "$platform" ]];then
 echo "ARROW_NATIVE_PLATFORM_REQUIRED: absolute qualified libstdlib_platform.so" >&2
 exit 64
fi
platform_dir=$(cd "$(dirname "$platform")" && pwd)
platform=$(readlink -f "$platform")
out=${STDLIB_ARROW_OUT:-${RUNNER_TEMP:-/tmp}/sounio-arrow-cpp-${GITHUB_RUN_ID:-$$}-${GITHUB_RUN_ATTEMPT:-1}}
[[ "$out" == /* && "$out" =~ ^[A-Za-z0-9_./-]+$ && ! -e "$out" ]] || exit 64
raw=$(readlink -f "${MADAROS_RAW_BIN:?identify the current-source raw compiler}")
[[ -x "$raw" ]] || exit 64
mkdir -p "$out"
trap 'r=$?;echo "$r" > "$out/STATUS"' EXIT
lib="$sdk/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$platform_dir:$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_SOUC_ENGINE=madaros SOUNIO_REQUIRE_COMMITTED_MADAROS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 MADAROS_RAW_BIN="$raw"
souc=${SOUC:-$root/bin/souc}
node=${NODE:-node}
cxx=${CXX:-g++}
ulimit -s 524288
ulimit -c 0
find stdlib/arrow tests/stdlib/arrow -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/reference/arrow_cpp_interop.cpp tests/reference/arrow_legacy_case_inventory.tsv scripts/arrow_cpp_gate.sh scripts/arrow_fixture_relocate.mjs scripts/arrow_ipc_gate.sh scripts/arrow_{ipc,spec,ext,flight}_check.py bin/souc bin/madaros >> "$out/INPUTS.sha256"
sha256sum stdlib/io/native_file.sio stdlib/io/buffered_reader.sio stdlib/collections/byte_buffer.sio stdlib/ffi/platform.sio >> "$out/INPUTS.sha256"
sha256sum "$raw" > "$out/COMPILER.sha256"
sha256sum "$platform" > "$out/PLATFORM.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
run(){ local n=$1;shift;local r=0;"$@" > "$out/$n.stdout" 2> "$out/$n.stderr" || r=$?;printf '%s\t%s\n' "$n" "$r" >> "$out/RESULTS.tsv";return "$r"; }
run cpp-build timeout 180 "$cxx" -std=c++23 -O2 -Wall -Wextra -Werror -isystem "$sdk/usr/include" tests/reference/arrow_cpp_interop.cpp -L"$lib" -Wl,-rpath,"$lib" -Wl,-rpath-link,"$lib" -larrow_flight -larrow -o "$out/judge"
run cpp-version "$out/judge" --version
grep -Fx 'ARROW_CPP_VERSION 26.0.0' "$out/cpp-version.stdout"
sha256sum "$out/judge" > "$out/EXECUTABLES.sha256"
ldd "$out/judge" > "$out/BOUND_LIBRARIES.txt"
! grep -q 'not found' "$out/BOUND_LIBRARIES.txt"
awk '{if(index($3,"/")==1)print $3;else if(index($1,"/")==1)print $1}' "$out/BOUND_LIBRARIES.txt" | sort -u | xargs -r sha256sum > "$out/BOUND_LIBRARIES.sha256"
r=0;ARROW_CPP_PREFIX="$out/missing-sdk" bash scripts/arrow_cpp_gate.sh --preflight > "$out/sdk-absent.stdout" 2> "$out/sdk-absent.stderr" || r=$?
[[ "$r" == 64 ]] && grep -q ARROW_CPP_SDK_REQUIRED "$out/sdk-absent.stderr"
echo ARROW_CPP_SDK_ABSENCE_REFUSED
run relocate "$node" scripts/arrow_fixture_relocate.mjs "$root" "$out"
for name in ipc codec spec ext flight;do
 src="$out/src/test_arrow_$name.sio"
 run "$name-check" timeout 300 "$souc" check "$src"
 run "$name-compile" timeout 300 "$souc" compile "$src" -o "$out/$name.elf"
 sha256sum "$out/$name.elf" >> "$out/EXECUTABLES.sha256"
 run "$name-native" timeout 300 "$out/$name.elf"
 case "$name" in ipc) marker=ARROW_IPC_OK;;codec) marker=CODEC_OK;;spec) marker=ARROW_SPEC_OK;;ext) marker=ARROW_EXT_OK;;flight) marker=FLIGHT_IPC_OK;;esac
 [[ $(grep -cx "$marker" "$out/$name-native.stdout") == 1 ]]
 cat "$out/$name-native.stdout"
done
run cpp-check timeout 180 "$out/judge" --check "$out/files"
grep -Fx ARROW_CPP_INTEROP_OK "$out/cpp-check.stdout"
awk -F '\t' 'NR>1 {print $1}' tests/reference/arrow_legacy_case_inventory.tsv | LC_ALL=C sort > "$out/EXPECTED_CASES.txt"
grep '^CASE_OK ' "$out/cpp-check.stdout" | cut -d ' ' -f2 | LC_ALL=C sort > "$out/OBSERVED_CASES.txt"
[[ $(wc -l < "$out/EXPECTED_CASES.txt") == 30 ]]
cmp "$out/EXPECTED_CASES.txt" "$out/OBSERVED_CASES.txt"
cat "$out/cpp-check.stdout"
run cpp-mutant-write "$out/judge" --write-id-mutant "$out/files/sounio_arrow_ipc.arrow" "$out/id-mutant.arrow"
r=0;"$out/judge" --check-basic-logical "$out/id-mutant.arrow" > "$out/id-mutant.stdout" 2> "$out/id-mutant.stderr" || r=$?
[[ "$r" == 1 ]] && grep -Fq 'id[0]' "$out/id-mutant.stderr"
echo ARROW_CPP_ID_MUTANT_REJECTED
# Mandatory incremental native composition: official SDK produces, Sounio reads,
# Sounio produces, official SDK reads. Every output path is job-private.
mkdir "$out/native"
run native-relocate "$node" scripts/arrow_fixture_relocate.mjs "$root" "$out/native" --native-stream
run native-cpp-write "$out/judge" --write-incremental "$out/native/files/sounio_wave3_arrow_cpp.stream"
native_src="$out/native/src/test_arrow_native_stream.sio"
run native-check timeout 300 "$souc" check "$native_src"
run native-compile timeout 300 "$souc" compile "$native_src" -o "$out/native.elf"
sha256sum "$out/native.elf" >> "$out/EXECUTABLES.sha256"
run native-execute timeout 300 "$out/native.elf"
[[ $(grep -cx ARROW_NATIVE_STREAM_OK "$out/native-execute.stdout") == 1 ]]
[[ $(grep -cx ARROW_NATIVE_CPP_READ_OK "$out/native-execute.stdout") == 1 ]]
cat "$out/native-execute.stdout"
run native-cpp-read "$out/judge" --check-incremental "$out/native/files/sounio_wave3_arrow_native.stream"
grep -Fx ARROW_INCREMENTAL_CPP_OK "$out/native-cpp-read.stdout"
run native-cpp-dictionary "$out/judge" --check-incremental-dictionary "$out/native/files/sounio_wave3_arrow_dictionary.stream"
grep -Fx ARROW_INCREMENTAL_DICTIONARY_CPP_OK "$out/native-cpp-dictionary.stdout"
echo ARROW_INCREMENTAL_BIDIRECTIONAL_OK
find "$out/files" "$out/src" "$out/native" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/ARTIFACTS.sha256"
sha256sum "$out/FIXTURE_RELOCATION.json" "$out/id-mutant.arrow" >> "$out/ARTIFACTS.sha256"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.final.verify"
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.final.verify"
sha256sum -c "$out/PLATFORM.sha256" > "$out/PLATFORM.final.verify"
sha256sum -c "$out/EXECUTABLES.sha256" > "$out/EXECUTABLES.final.verify"
sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify"
[[ -s "$out/BOUND_LIBRARIES.sha256" ]]
sha256sum -c "$out/BOUND_LIBRARIES.sha256" > "$out/BOUND_LIBRARIES.final.verify"
echo ARROW_CPP_GATE_OK

