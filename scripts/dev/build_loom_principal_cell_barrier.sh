#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_SOURCE:-$ROOT_DIR/tools/loom/src/loom_principal_cell_barrier.cpp}"
OUTPUT="${SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-principal-cell-barrier}"

fail() {
  printf 'build-loom-principal-cell-barrier: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail "barrier source is absent or linked: $SOURCE"
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-principal-cell-barrier.XXXXXX")"
trap 'rm -f "$stage"' EXIT
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-principal-cell-barrier-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PRINCIPAL_CELL_BARRIER path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio command_surface=false launch_open=false material_grant=false material_execution=false\n' \
  "$OUTPUT"

