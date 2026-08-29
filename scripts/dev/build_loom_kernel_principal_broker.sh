#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_SOURCE:-$ROOT_DIR/tools/loom/src/loom_kernel_principal_broker.cpp}"
QUORUM_LAB="$ROOT_DIR/tools/loom/src/loom_exec_quorum_lab.inc"
PROCESS_WITNESS_LAB="$ROOT_DIR/tools/loom/src/loom_process_witness_lab.inc"
PRODUCT_EXEC_INGRESS_HOST_CANARY="$ROOT_DIR/tools/loom/src/loom_product_exec_ingress_host_canary.inc"
OUTPUT="${SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-principal-broker}"

fail() {
  printf 'build-loom-kernel-principal-broker: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" ]] || fail "kernel-principal broker source is missing: $SOURCE"
[[ -f "$QUORUM_LAB" && ! -L "$QUORUM_LAB" ]] || fail 'ExecQuorum broker module is absent or linked'
[[ -f "$PROCESS_WITNESS_LAB" && ! -L "$PROCESS_WITNESS_LAB" ]] || fail 'ProcessWitness broker module is absent or linked'
[[ -f "$PRODUCT_EXEC_INGRESS_HOST_CANARY" && ! -L "$PRODUCT_EXEC_INGRESS_HOST_CANARY" ]] || fail 'product ExecIngress host-canary module is absent or linked'
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-kernel-principal-broker.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-kernel-principal-broker-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none "$SOURCE" -lcrypto -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_KERNEL_PRINCIPAL_BROKER path=%s language=C++20 role=MATERIAL_PARITY transitory=true launch_open=false recycle_open=false\n' \
  "$OUTPUT"
