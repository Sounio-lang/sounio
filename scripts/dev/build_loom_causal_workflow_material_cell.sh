#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_CAUSAL_MATERIAL_CELL_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_CAUSAL_MATERIAL_CELL_SOURCE:-$ROOT_DIR/tools/loom/src/loom_causal_workflow_material_cell.cpp}"
OUTPUT="${SOUNIO_LOOM_CAUSAL_MATERIAL_CELL_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-causal-workflow-material-cell}"

fail() {
  printf 'build-loom-causal-workflow-material-cell: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail "material cell source is absent or linked: $SOURCE"
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-causal-material-cell.XXXXXX")"
trap 'rm -f "$stage"' EXIT
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-causal-workflow-material-cell-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_CAUSAL_WORKFLOW_MATERIAL_CELL path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9037 arbitrary_path=false\n' \
  "$OUTPUT"
