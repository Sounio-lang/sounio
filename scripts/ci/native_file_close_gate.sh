#!/usr/bin/env bash
# Linux C++23 fault-injection proof for checked native file destruction.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
[[ "$(uname -sm)" == "Linux x86_64" ]] || exit 64
out=${STDLIB_NATIVE_CLOSE_OUT:-${RUNNER_TEMP:-/tmp}/stdlib-native-close-${GITHUB_RUN_ID:-$$}-${GITHUB_RUN_ATTEMPT:-1}}
[[ "$out" == /* && ! -e "$out" ]] || exit 64
mkdir -p "$out"
trap 'rc=$?;echo "$rc" > "$out/STATUS"' EXIT
ulimit -c 0
sha256sum runtime/stdlib_platform.cpp tests/reference/stdlib_file_close_reference.cpp scripts/ci/native_file_close_gate.sh > "$out/INPUTS.sha256"
g++ --version > "$out/CXX.txt"
timeout 180 g++ -std=c++23 -O2 -Wall -Wextra -Werror -pthread tests/reference/stdlib_file_close_reference.cpp runtime/stdlib_platform.cpp -Wl,--wrap=close -o "$out/reference" > "$out/compile.stdout" 2> "$out/compile.stderr"
sha256sum "$out/reference" > "$out/EXECUTABLES.sha256"
(cd "$out";timeout 30 ./reference) > "$out/native.stdout" 2> "$out/native.stderr"
grep -Fx NATIVE_FILE_CLOSE_CPP_OK "$out/native.stdout"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.final.verify"
sha256sum -c "$out/EXECUTABLES.sha256" > "$out/EXECUTABLES.final.verify"
echo STDLIB_NATIVE_FILE_CLOSE_OK

