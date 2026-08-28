#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_KERNEL_PRINCIPAL_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_KERNEL_PRINCIPAL_PROBE_SOURCE:-$ROOT_DIR/tools/loom/src/loom_kernel_principal_probe.cpp}"
OUTPUT="${SOUNIO_LOOM_KERNEL_PRINCIPAL_PROBE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-principal-probe}"

fail() {
  printf 'build-loom-kernel-principal-probe: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" ]] || fail "kernel-principal probe source is missing: $SOURCE"
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-kernel-principal-probe.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  "$SOURCE" -lcrypto -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_KERNEL_PRINCIPAL_PROBE path=%s language=C++ role=MATERIAL_PARITY\n' \
  "$OUTPUT"
