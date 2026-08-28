#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EFFECT_CLOSURE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EFFECT_CLOSURE_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_EFFECT_CLOSURE_MODULE:-$ROOT_DIR/stdlib/coordination/loom_effect_closure_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_EFFECT_CLOSURE_MAIN:-$ROOT_DIR/tools/loom/effect_closure_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-effect-closure-authority-runtime}"

fail() {
  printf 'build-sounio-loom-effect-closure-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "effect-closure module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "effect-closure entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-closure-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_effect_closure_authority_runtime.sio"
compiled="$work/sounio-loom-effect-closure-authority-runtime"

# Mechanical source assembly only. Decisions and expected results live in the
# Sounio module, never in this launcher.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native effect-closure executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_EFFECT_CLOSURE_SELFTEST PASS cases=19' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_EFFECT_CLOSURE_AUTHORITY path=%s language=Sounio engine=%s cases=19\n' \
  "$OUTPUT" "$ENGINE"
