#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PROCESS_WITNESS_PRINCIPAL_CELL_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PROCESS_WITNESS_PRINCIPAL_CELL_SOURCE:-$ROOT_DIR/tools/loom/src/loom_process_witness_principal_cell.cpp}"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_PRINCIPAL_CELL_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-principal-cell}"

fail() {
  printf 'build-loom-process-witness-principal-cell: FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail "source is absent or linked: $SOURCE"
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-process-witness-principal-cell.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-process-witness-principal-cell-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none "$SOURCE" -lcrypto -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_PRINCIPAL_CELL path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio execveat=true same_pid=true pidfd=true material_grant=true material_execution=false launch_open=false\n' \
  "$OUTPUT"
