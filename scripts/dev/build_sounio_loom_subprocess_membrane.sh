#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_SUBPROCESS_MEMBRANE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_SUBPROCESS_MEMBRANE_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_SUBPROCESS_MEMBRANE_MODULE:-$ROOT_DIR/stdlib/coordination/loom_subprocess_membrane_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_SUBPROCESS_MEMBRANE_MAIN:-$ROOT_DIR/tools/loom/subprocess_membrane_main.sio}"
OUTPUT="${SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-subprocess-membrane-runtime}"

fail() {
  printf 'build-sounio-loom-subprocess-membrane: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "subprocess-membrane module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "subprocess-membrane entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-subprocess-membrane-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_subprocess_membrane_runtime.sio"
compiled="$work/sounio-loom-subprocess-membrane-runtime"

# Mechanical source assembly only. Decisions and expected results live in the
# Sounio module, never in this launcher.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native subprocess-membrane executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_SUBPROCESS_MEMBRANE_SELFTEST PASS cases=43' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_SUBPROCESS_MEMBRANE path=%s language=Sounio engine=%s cases=43\n' \
  "$OUTPUT" "$ENGINE"
