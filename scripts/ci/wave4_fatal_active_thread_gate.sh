#!/bin/bash
set -eu
export PATH=/usr/bin:/bin
ulimit -s 524288
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ $# == 1 && "$1" == /* && ! -e "$1" ]] || exit 64
: "${BEFORE_RAW_BIN:?before raw compiler required}"
: "${AFTER_RAW_BIN:?after raw compiler required}"
[[ "$BEFORE_RAW_BIN" == /* && -x "$BEFORE_RAW_BIN" && "$AFTER_RAW_BIN" == /* && -x "$AFTER_RAW_BIN" ]] || exit 64
B=$1
mkdir -p "$B/source/runtime"
cp -a "$ROOT/stdlib" "$B/source/"
cp "$ROOT/runtime/stdlib_platform.cpp" "$B/source/runtime/"
cp "$ROOT/tests/fixtures/wave4_scalar_abi/fatal_active_thread.sio.in" "$B/source/probe.sio"
cd "$B/source"
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
export LD_LIBRARY_PATH="$B"
find . -type f -print0|sort -z|xargs -0 sha256sum >"$B/INPUTS.sha256"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread runtime/stdlib_platform.cpp -o "$B/libstdlib_platform.so"
sha256sum "$B/libstdlib_platform.so" >"$B/RUNTIME.sha256"
for version in before after;do
 seed="$BEFORE_RAW_BIN"
 if [ "$version" = after ];then seed="$AFTER_RAW_BIN";fi
 sha256sum "$seed" >>"$B/COMPILERS.sha256"
 timeout 300 "$seed" --check probe.sio >"$B/$version.check" 2>&1
 timeout 300 "$seed" compile probe.sio -o "$B/$version.elf" >"$B/$version.compile" 2>&1
 chmod +x "$B/$version.elf"
 sha256sum "$B/$version.elf" >>"$B/ELFS.sha256"
 set +e
 timeout --kill-after=2 5 "$B/$version.elf" >"$B/$version.native" 2>&1
 result=$?
 set -e
 printf '%s\t%s\n' "$version" "$result" >>"$B/RESULTS.tsv"
 grep -qx 'FATAL_WITH_ACTIVE_THREAD_ARMED' "$B/$version.native"
 expected=124
 if [ "$version" = after ];then expected=1;fi
 [ "$result" = "$expected" ]
done
sha256sum -c "$B/INPUTS.sha256" >"$B/INPUTS.verify"
sha256sum -c "$B/RUNTIME.sha256" >"$B/RUNTIME.verify"
sha256sum -c "$B/COMPILERS.sha256" >"$B/COMPILERS.verify"
sha256sum -c "$B/ELFS.sha256" >"$B/ELFS.verify"
printf '0\n' >"$B/STATUS"
