#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_CONTINUITY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_CONTINUITY_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_CONTINUITY_MODULE:-$ROOT_DIR/stdlib/coordination/loom_continuity.sio}"
ENTRYPOINT="${SOUNIO_LOOM_CONTINUITY_MAIN:-$ROOT_DIR/tools/loom/continuity_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_CONTINUITY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime}"
PREBUILT="${SOUNIO_LOOM_CONTINUITY_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-continuity: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "continuity module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "continuity entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-continuity-build.XXXXXX")"
  cleanup() {
    rm -rf "$work"
  }
  trap cleanup EXIT

  combined="$work/loom_continuity_adapter.sio"
  compiled="$work/sounio-loom-continuity-runtime"
  {
    cat "$MODULE"
    cat "$ENTRYPOINT"
  } > "$combined"

  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

probe="$(printf '101 111 201 301 401 501 0 0 0 0 1 0 0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v1' ]] || \
  fail "native adapter failed its initial-generation probe: $probe"

printf 'BUILT_CONTINUITY path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
