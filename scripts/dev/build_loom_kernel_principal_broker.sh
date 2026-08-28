#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_SOURCE:-$ROOT_DIR/tools/loom/src/loom_kernel_principal_broker.cpp}"
OUTPUT="${SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-principal-broker}"

fail() {
  printf 'build-loom-kernel-principal-broker: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" ]] || fail "kernel-principal broker source is missing: $SOURCE"
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-kernel-principal-broker.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-kernel-principal-broker-v1 \
  -Wl,--build-id=none "$SOURCE" -lcrypto -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_KERNEL_PRINCIPAL_BROKER path=%s language=C++20 role=MATERIAL_PARITY transitory=true launch_open=false recycle_open=false\n' \
  "$OUTPUT"
