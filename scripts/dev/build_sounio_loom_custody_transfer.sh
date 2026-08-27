#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_CUSTODY_TRANSFER_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_CUSTODY_TRANSFER_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_CUSTODY_TRANSFER_MODULE:-$ROOT_DIR/stdlib/coordination/loom_custody_transfer.sio}"
ENTRYPOINT="${SOUNIO_LOOM_CUSTODY_TRANSFER_MAIN:-$ROOT_DIR/tools/loom/custody_transfer_main.sio}"
OUTPUT="${SOUNIO_LOOM_CUSTODY_TRANSFER_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-custody-transfer-runtime}"

fail() {
  printf 'build-sounio-loom-custody-transfer: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "custody-transfer module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "custody-transfer entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-custody-transfer-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_custody_transfer_runtime.sio"
compiled="$work/sounio-loom-custody-transfer-runtime"

# Mechanical source assembly only. Decisions and expected cases originate in
# the Sounio module.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native custody-transfer executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_CUSTODY_TRANSFER_SELFTEST PASS cases=30' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_CUSTODY_TRANSFER path=%s language=Sounio engine=%s cases=30\n' \
  "$OUTPUT" "$ENGINE"
