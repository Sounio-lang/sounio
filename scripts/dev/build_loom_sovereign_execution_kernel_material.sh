#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_SOVEREIGN_MATERIAL_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_SOVEREIGN_MATERIAL_SOURCE:-$ROOT_DIR/tools/loom/src/loom_sovereign_execution_kernel_material.cpp}"
OUTPUT="${SOUNIO_LOOM_SOVEREIGN_MATERIAL_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-sovereign-execution-kernel-material}"
EXTRA_FLAGS_RAW="${SOUNIO_LOOM_SOVEREIGN_MATERIAL_CPPFLAGS:-}"

fail() {
  printf 'build-loom-sovereign-execution-kernel-material: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail "material source is absent or linked: $SOURCE"
mkdir -p "$(dirname "$OUTPUT")"

extra_flags=()
if [[ -n "$EXTRA_FLAGS_RAW" ]]; then
  read -r -a extra_flags <<< "$EXTRA_FLAGS_RAW"
fi
stage="$(mktemp "${TMPDIR:-/tmp}/loom-sovereign-material.XXXXXX")"
trap 'rm -f "$stage"' EXIT
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-sovereign-execution-kernel-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "${extra_flags[@]}" "$SOURCE" -lcrypto -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_SOVEREIGN_EXECUTION_KERNEL_MATERIAL path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9042\n' \
  "$OUTPUT"
