#!/usr/bin/env bash
# Real binary concordance on the source-built Linux native compiler and seed.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
raw=$(readlink -f "${MADAROS_RAW_BIN:?identify current-source raw compiler}")
[[ -x "$raw" && "$(uname -sm)" == "Linux x86_64" ]] || exit 64
out=${STDLIB_APPEND_OUT:-${RUNNER_TEMP:-/tmp}/stdlib-append-${GITHUB_RUN_ID:-$$}-${GITHUB_RUN_ATTEMPT:-1}}
[[ "$out" == /* && ! -e "$out" ]] || exit 64
mkdir -p "$out"
trap 'rc=$?;echo "$rc" > "$out/STATUS"' EXIT
unset SOUNIO_SOUC_BIN
export MADAROS_RAW_BIN="$raw" SOUNIO_SOUC_ENGINE=madaros SOUNIO_REQUIRE_COMMITTED_MADAROS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_STDLIB_PATH="$root/stdlib"
ulimit -s 524288
sha256sum "$raw" bin/souc bin/madaros bin/souc-lean-single-x86_64 tests/run-pass/stdlib_append_file_checked.sio tests/reference/stdlib_append_file_reference.cpp scripts/ci/append_file_builtin_gate.sh > "$out/INPUTS.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
run(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
run reference-build timeout 180 g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/stdlib_append_file_reference.cpp -o "$out/reference"
sha256sum "$out/reference" > "$out/EXECUTABLES.sha256"
run native-check timeout 180 bin/souc check tests/run-pass/stdlib_append_file_checked.sio
run native-compile timeout 180 bin/souc compile tests/run-pass/stdlib_append_file_checked.sio -o "$out/native.elf"
run bootstrap-compile timeout 180 bin/souc-lean-single-x86_64 tests/run-pass/stdlib_append_file_checked.sio "$out/bootstrap.elf"
for engine in native bootstrap;do
 chmod +x "$out/$engine.elf"
 sha256sum "$out/$engine.elf" >> "$out/EXECUTABLES.sha256"
 run "$engine-run" timeout 60 "$out/$engine.elf"
 run "$engine-marker" grep -Fx APPEND_FILE_CHECKED_OK "$out/$engine-run.stdout"
 run "$engine-bytes" "$out/reference"
done
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.final.verify"
sha256sum -c "$out/EXECUTABLES.sha256" > "$out/EXECUTABLES.final.verify"
echo APPEND_FILE_BUILTIN_GATE_OK
