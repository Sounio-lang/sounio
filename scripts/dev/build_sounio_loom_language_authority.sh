#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_MODULE:-$ROOT_DIR/stdlib/coordination/loom_language_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_MAIN:-$ROOT_DIR/tools/loom/language_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-language-authority-runtime}"

fail() {
  printf 'build-sounio-loom-language-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "language-authority module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "language-authority entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-language-authority-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_language_authority_runtime.sio"
compiled="$work/sounio-loom-language-authority-runtime"

# Mechanical source assembly only. Semantic decisions and expected cases live
# in the Sounio module above, never in this launcher.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native language-authority executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_LANGUAGE_AUTHORITY path=%s language=Sounio engine=%s cases=33\n' \
  "$OUTPUT" "$ENGINE"
