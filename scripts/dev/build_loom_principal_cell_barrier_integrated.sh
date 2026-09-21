#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_CXX:-c++}"
SOURCE="$ROOT_DIR/tools/loom/src/loom_principal_cell_barrier_integrated.cpp"
FROZEN_BARRIER="$ROOT_DIR/tools/loom/src/loom_principal_cell_barrier.cpp"
OUTPUT="${SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_INTEGRATED_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-principal-cell-barrier-integrated}"

fail() {
  printf 'build-loom-principal-cell-barrier-integrated: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'integrated barrier source is absent or linked'
[[ -f "$FROZEN_BARRIER" && ! -L "$FROZEN_BARRIER" ]] || fail 'frozen barrier source is absent or linked'
[[ "$(sha256sum "$FROZEN_BARRIER" | cut -d ' ' -f 1)" == \
  9885c7a22d14baf0972b9edde00718cc19b590ec3f3bea4f1b859310a62a636c ]] ||
  fail 'frozen barrier implementation drifted'

stage="$(mktemp "${TMPDIR:-/tmp}/loom-principal-cell-barrier-integrated.XXXXXX")"
trap 'rm -f "$stage"' EXIT
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-principal-cell-barrier-integrated-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  -I "$ROOT_DIR/tools/loom/src" "$SOURCE" -o "$stage"
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PRINCIPAL_CELL_BARRIER_INTEGRATED path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio inherited_fds=3,4 command_surface=false material_grant=false material_execution=false\n' \
  "$OUTPUT"
