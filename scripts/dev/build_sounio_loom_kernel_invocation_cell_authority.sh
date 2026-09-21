#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_KERNEL_INVOCATION_CELL_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_KERNEL_INVOCATION_CELL_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_KERNEL_INVOCATION_CELL_MODULE:-$ROOT_DIR/stdlib/coordination/loom_kernel_invocation_cell_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_KERNEL_INVOCATION_CELL_MAIN:-$ROOT_DIR/tools/loom/kernel_invocation_cell_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-kernel-invocation-cell-authority-runtime}"

fail() {
  printf 'build-sounio-loom-kernel-invocation-cell-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "InvocationCell module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "InvocationCell entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-invocation-cell-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_kernel_invocation_cell_authority_runtime.sio"
compiled="$work/sounio-loom-kernel-invocation-cell-authority-runtime"

# Mechanical assembly only. Sounio owns every decision and expected result.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native InvocationCell executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_KERNEL_INVOCATION_CELL_SELFTEST PASS cases=17' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_KERNEL_INVOCATION_CELL_AUTHORITY path=%s language=Sounio engine=%s cases=17\n' \
  "$OUTPUT" "$ENGINE"
