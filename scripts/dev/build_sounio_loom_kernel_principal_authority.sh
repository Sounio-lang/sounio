#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_KERNEL_PRINCIPAL_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_KERNEL_PRINCIPAL_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_KERNEL_PRINCIPAL_MODULE:-$ROOT_DIR/stdlib/coordination/loom_kernel_principal_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_KERNEL_PRINCIPAL_MAIN:-$ROOT_DIR/tools/loom/kernel_principal_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_KERNEL_PRINCIPAL_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-kernel-principal-authority-runtime}"

fail() {
  printf 'build-sounio-loom-kernel-principal-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "kernel-principal module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "kernel-principal entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-principal-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_kernel_principal_authority_runtime.sio"
compiled="$work/sounio-loom-kernel-principal-authority-runtime"

# Mechanical assembly only. Sounio owns every decision and expected result.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native kernel-principal executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_KERNEL_PRINCIPAL_SELFTEST PASS cases=17' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_KERNEL_PRINCIPAL_AUTHORITY path=%s language=Sounio engine=%s cases=17\n' \
  "$OUTPUT" "$ENGINE"
