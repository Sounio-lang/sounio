#!/usr/bin/env bash
# Run against a source-built Madaros. The main return status must win even
# though a dynamically loaded pthread is still blocked.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
compiler=${1:?absolute path to source-built Madaros}
work=${2:?absolute output directory}
mkdir -p "$work"
cd "$root"
"${CXX:-c++}" -std=c++23 -Wall -Wextra -Werror -O2 -shared -fPIC -pthread \
  tests/native-v2/exit_group_thread_probe.cpp -o "$work/libexit_group_probe.so"
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_MADAROS_BIN="$compiler"
./bin/souc check tests/native-v2/exit_group_thread_return.sio > "$work/check.log" 2>&1
"$compiler" build tests/native-v2/exit_group_thread_return.sio -o "$work/probe" > "$work/build.log" 2>&1
sha256sum "$compiler" "$work/probe" "$work/libexit_group_probe.so" > "$work/SHA256SUMS"
rc=0
LD_LIBRARY_PATH="$work${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" timeout -k 2s 5s "$work/probe" > "$work/run.log" 2>&1 || rc=$?
cat "$work/run.log"
printf 'EXIT_GROUP_PROBE_RC=%d EXPECTED=37\n' "$rc"
grep -qx THREAD_READY_RETURN_37 "$work/run.log"
[[ "$rc" -eq 37 ]]
printf '%s\n' EXIT_GROUP_THREAD_RETURN_PASS
