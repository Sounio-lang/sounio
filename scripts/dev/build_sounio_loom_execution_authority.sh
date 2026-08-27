#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EXECUTION_AUTHORITY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EXECUTION_AUTHORITY_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_EXECUTION_AUTHORITY_MODULE:-$ROOT_DIR/stdlib/coordination/loom_execution_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_EXECUTION_AUTHORITY_MAIN:-$ROOT_DIR/tools/loom/execution_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_EXECUTION_AUTHORITY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-execution-authority-runtime}"

fail() {
  printf 'build-sounio-loom-execution-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "execution-authority module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "execution-authority entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-execution-authority-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_execution_authority_runtime.sio"
compiled="$work/sounio-loom-execution-authority-runtime"

# Mechanical source assembly only. Decisions and expected results live in the
# Sounio module, never in this launcher.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native execution-authority executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_EXECUTION_AUTHORITY_SELFTEST PASS cases=32' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_EXECUTION_AUTHORITY path=%s language=Sounio engine=%s cases=32\n' \
  "$OUTPUT" "$ENGINE"
