#!/usr/bin/env bash
# Compiler/storage component only: no claim of a rebuilt BFS compiler.
set -euo pipefail
[[ $# == 2 && -n "${SLURM_JOB_ID:-}" ]] || { echo "usage (inside Slurm): gate_storage.sh RAW_COMPILER FRESH_RESULTS" >&2;exit 2; }
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
raw=$(realpath "$1")
out=$(realpath -m "$2")
[[ -x "$raw" && ! -e "$out" ]]
mkdir -p "$out"
cd "$repo"
ulimit -s 524288
ulimit -c 0
export SOUNIO_STDLIB_PATH="$repo/self-hosted"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
find self-hosted stdlib tests/fixtures/wave4_borrow_capacity -type f -print0 |sort -z|xargs -0 sha256sum > "$out/SOURCES.sha256"
sha256sum "$raw" > "$out/COMPILER.sha256"
"${CXX:-g++}" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_borrow_capacity/reference.cpp -o "$out/reference"
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS="$mode"
 for name in component resource_controls;do
  timeout 180 "$raw" --check "tests/fixtures/wave4_borrow_capacity/$name.sio" > "$out/$name$mode.check.stdout" 2>&1
  timeout 300 "$raw" "tests/fixtures/wave4_borrow_capacity/$name.sio" -o "$out/$name$mode.elf" > "$out/$name$mode.compile.stdout" 2>&1
  grep -Fq "Compilation successful!" "$out/$name$mode.compile.stdout"
  [[ -s "$out/$name$mode.elf" ]]
  chmod +x "$out/$name$mode.elf"
  timeout 180 /usr/bin/time -v -o "$out/$name$mode.resource" "$out/$name$mode.elf" > "$out/$name$mode.native.stdout" 2>&1
 done
 "$out/reference" "$out/component$mode.native.stdout" > "$out/reference$mode.stdout"
 grep -Fxq PRIVATE_COMPACTION_RELOCATION_OK "$out/component$mode.native.stdout"
 for marker in INVALID_INDEX_FAIL_CLOSED BYTE_OVERFLOW_FAIL_CLOSED CORRUPT_CELL_FAIL_CLOSED CORRUPT_PATH_FAIL_CLOSED CORRUPT_LOAN_LENGTH_FAIL_CLOSED SYNTHETIC_OOM_STICKY SOURCE_ERROR_PRESERVES_REASON DUPLICATE_ID_PUBLICATION_REFUSED BINDING_ID_REWRITE_REFUSED INVALID_NAME_PUBLICATION_REFUSED SEMANTIC_INVALID_LOAN_NOT_STORAGE_FAILURE BORROW_RESOURCE_CONTROLS_OK;do
  grep -Fxq "$marker" "$out/resource_controls$mode.native.stdout"
 done
done
sha256sum "$out/"*.elf "$out/reference" > "$out/EXECUTABLES.sha256"
sha256sum -c "$out/SOURCES.sha256" > "$out/SOURCES.final.verify"
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.final.verify"
echo BORROW_PAGED_STORAGE_AND_INJECTED_CONTROLS_OK
