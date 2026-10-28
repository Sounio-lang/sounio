#!/usr/bin/env bash
# General bounded native channel range protocol; no callback Sounio.
set -euo pipefail
CPP_ONLY=${RANGE_CPP_ONLY:-0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 64
OUT=$(realpath -m -- "$1")
[[ ! -e "$OUT" ]] || { echo "fresh output required" >&2;exit 64; }
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || { echo "Slurm required" >&2;exit 64; }
CXX=${CXX:-g++}

RAW=''
if [[ $CPP_ONLY != 1 ]];then
 : "$MADAROS_RAW_BIN"
 RAW=$(realpath -e "$MADAROS_RAW_BIN")
 [[ -x "$RAW" ]] || exit 64
fi
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
mkdir "$OUT/inputs" "$OUT/runs"
printf 'slurm:%s\n' "$SLURM_JOB_ID" > "$OUT/EXECUTION_AUTHORITY"
ulimit -s 524288
ulimit -c 0
finish(){
 local r=$? changed=0
 trap - EXIT
 for name in SOURCES TOOLS EXECUTABLES INPUTS DYNAMIC ARTIFACTS;do
  if [[ -s "$OUT/$name.sha256" ]];then sha256sum -c "$OUT/$name.sha256" > "$OUT/$name.final.verify" 2>&1 || changed=1;fi
 done
 [[ $changed == 0 ]] || r=1
 printf '%s\n' "$r" > "$OUT/STATUS"
 if [[ $r == 0 ]];then
  if [[ $CPP_ONLY == 1 ]];then echo WAVE4_CHANNEL_RANGE_CPP_ONLY_OK
  else echo WAVE4_CHANNEL_RANGE_GATE_OK;fi
 fi
 exit "$r"
}
trap finish EXIT
(
 cd "$ROOT"
 find stdlib -type f -print0
 printf '%s\0' runtime/stdlib_platform.cpp tests/reference/wave4_channel_range_reference.cpp scripts/stdlib/wave4_channel_range_gate.sh scripts/ci/souc-native-wrapper.sh tests/stdlib/massive/test_channel.sio tests/stdlib/wave4-science-pipeline/channel_range.sio tests/stdlib/wave4-science-pipeline/channel_deadline_max.cpp.in tests/stdlib/wave4-science-pipeline/legacy_io_channel.original.sio.in
) | sort -zu | while IFS= read -r -d '' p;do sha256sum "$ROOT/$p";done > "$OUT/SOURCES.sha256"
printf 'case\tphase\tstatus\n' > "$OUT/RESULTS.tsv"
: > "$OUT/TOOLS.sha256"
: > "$OUT/EXECUTABLES.sha256"
: > "$OUT/INPUTS.sha256"
: > "$OUT/DYNAMIC.sha256"
phase(){
 local name=$1 kind=$2 seconds=$3 r=0
 shift 3
 timeout "$seconds" "$@" > "$OUT/$name.$kind.stdout" 2> "$OUT/$name.$kind.stderr" || r=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$r" >> "$OUT/RESULTS.tsv"
 return "$r"
}
dynamic_manifest(){
 local exe=$1 name=$2
 ldd "$exe" > "$OUT/$name.ldd"
 ! grep -q 'not found' "$OUT/$name.ldd"
 awk '/=> \//{print $3} /^[[:space:]]*\//{print $1}' "$OUT/$name.ldd" | sort -u | xargs -r sha256sum >> "$OUT/DYNAMIC.sha256"
}
for tool in "$(command -v "$CXX")" "$("$CXX" -print-prog-name=cc1plus)" "$("$CXX" -print-prog-name=collect2)" "$("$CXX" -print-prog-name=as)" "$("$CXX" -print-prog-name=ld)" "$("$CXX" -print-file-name=libstdc++.so)" "$("$CXX" -print-file-name=libgcc_s.so.1)";do
 if [[ "$tool" != */* ]];then tool=$(command -v "$tool");fi
 tool=$(realpath -e "$tool")
 sha256sum "$tool" >> "$OUT/TOOLS.sha256"
 ldd "$tool" >> "$OUT/CXX_TOOLS.ldd"
done
"$CXX" --version > "$OUT/CXX_VERSION"
! grep -q 'not found' "$OUT/CXX_TOOLS.ldd"
awk '/=> \//{print $3} /^[[:space:]]*\//{print $1}' "$OUT/CXX_TOOLS.ldd" | sort -u | xargs -r sha256sum >> "$OUT/DYNAMIC.sha256"
for mode in normal sanitized;do
 flags='-O2'
 if [[ $mode == sanitized ]];then flags='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined';fi
 # The split words below are fixed flags chosen in this script, never user input.
 phase "range-$mode" build 180 "$CXX" -std=c++23 $flags -Wall -Wextra -Werror -pthread "$ROOT/tests/reference/wave4_channel_range_reference.cpp" "$ROOT/runtime/stdlib_platform.cpp" -ldl -Wl,--wrap=pthread_create -o "$OUT/range-$mode"
 sha256sum "$OUT/range-$mode" >> "$OUT/EXECUTABLES.sha256"
 dynamic_manifest "$OUT/range-$mode" "range-$mode"
 phase "range-$mode" native 60 env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$OUT/range-$mode"
 grep -Fxq 'NATIVE_CHANNEL_RANGE_CPP_OK zero=exact range=general lease=retained deadline=absolute cancel=linearized partial_spawn=clean cleanup=0' "$OUT/range-$mode.native.stdout"
done
cp "$ROOT/tests/stdlib/wave4-science-pipeline/channel_deadline_max.cpp.in" "$OUT/inputs/max-deadline.cpp"
sha256sum "$OUT/inputs/max-deadline.cpp" >> "$OUT/INPUTS.sha256"
for header in /usr/include/c++/*/condition_variable /usr/include/c++/*/bits/chrono.h;do
 if [[ -f "$header" ]];then sha256sum "$header" >> "$OUT/TOOLS.sha256";fi
done
phase max-deadline build 180 "$CXX" -std=c++23 -O1 -g -fno-omit-frame-pointer -fsanitize=undefined -Wall -Wextra -Werror -pthread "$OUT/inputs/max-deadline.cpp" "$ROOT/runtime/stdlib_platform.cpp" -ldl -o "$OUT/max-deadline"
sha256sum "$OUT/max-deadline" >> "$OUT/EXECUTABLES.sha256"
dynamic_manifest "$OUT/max-deadline" max-deadline
phase max-deadline native 30 env UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$OUT/max-deadline"
grep -Fxq 'MAX_DEADLINE_CANCEL_CPP_OK cleanup=0' "$OUT/max-deadline.native.stdout"
printf '5c6c3499ecd641d7fd9cf348d0d94b7cbcd3089be03b86214e57164e119ed112  %s\n' "$ROOT/tests/stdlib/wave4-science-pipeline/legacy_io_channel.original.sio.in" | sha256sum -c > "$OUT/LEGACY_ORIGINAL.verify"
if [[ $CPP_ONLY == 1 ]];then
 printf '%s\n' CPP_PROTOCOL_ONLY_NO_SOUNIO_WRAPPER_QUALIFICATION > "$OUT/PARTIAL"
 exit 0
fi
sha256sum "$RAW" >> "$OUT/TOOLS.sha256"
export SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export LD_LIBRARY_PATH="$OUT"
phase runtime build 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$ROOT/runtime/stdlib_platform.cpp" -o "$OUT/libstdlib_platform.so"
sha256sum "$OUT/libstdlib_platform.so" >> "$OUT/EXECUTABLES.sha256"
for name in channel range;do
 source="$ROOT/tests/stdlib/massive/test_channel.sio"
 [[ $name != range ]] || source="$ROOT/tests/stdlib/wave4-science-pipeline/channel_range.sio"
 sha256sum "$source" >> "$OUT/INPUTS.sha256"
 phase "$name" check 300 bash "$ROOT/scripts/ci/souc-native-wrapper.sh" check "$source"
 grep -Fxq 'check: OK' "$OUT/$name.check.stdout"
 phase "$name" compile 600 bash "$ROOT/scripts/ci/souc-native-wrapper.sh" compile "$source" -o "$OUT/$name.elf"
 [[ -s "$OUT/$name.elf" && $(od -An -tx1 -N4 "$OUT/$name.elf"|tr -d ' \n') == 7f454c46 ]]
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
 mkdir "$OUT/runs/$name"
 (cd "$OUT/runs/$name";phase "$name" native 120 "$OUT/$name.elf")
done
grep -Fxq NATIVE_CHANNEL_OK "$OUT/channel.native.stdout"
grep -Fxq NATIVE_CHANNEL_RANGE_SOUNIO_OK "$OUT/range.native.stdout"
printf '%s\n' RANGE_PROTOCOL_GENERAL_CPP_LEASE_NOT_GENERIC_IO_CALLBACK_OR_TLS > "$OUT/CONTRACT"
printf '%s\n' WAVE4_NATIVE_CHANNEL_RANGE_COMPLETE > "$OUT/COMPLETE"
