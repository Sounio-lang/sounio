#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_SOURCE:-$ROOT_DIR/tools/loom/src/loom_host_exec_quorum_principal_cell.cpp}"
BARRIER="$ROOT_DIR/tools/loom/src/loom_principal_cell_barrier.cpp"
OUTPUT="${SOUNIO_LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-host-exec-quorum-principal-cell}"

fail() {
  printf 'build-loom-host-exec-quorum-principal-cell: FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
for input in "$SOURCE" "$BARRIER"; do
  [[ -f "$input" && ! -L "$input" ]] || fail "required source is absent or linked: $input"
done
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-host-exec-quorum-principal-cell.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-host-exec-quorum-principal-cell-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none "$SOURCE" -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio frozen_barrier_reused=true material_grant=false material_execution=false launch_open=false\n' \
  "$OUTPUT"
