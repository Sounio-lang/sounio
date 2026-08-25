#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_OBLIGATION_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_OBLIGATION_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_OBLIGATION_MODULE:-$ROOT_DIR/stdlib/coordination/loom_obligation.sio}"
ENTRYPOINT="${SOUNIO_LOOM_OBLIGATION_MAIN:-$ROOT_DIR/tools/loom/obligation_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_OBLIGATION_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-obligation-runtime}"
PREBUILT="${SOUNIO_LOOM_OBLIGATION_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-obligation: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "obligation module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "obligation entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-obligation-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_obligation_adapter.sio"
  compiled="$work/sounio-loom-obligation-runtime"
  {
    cat "$MODULE"
    cat "$ENTRYPOINT"
  } > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

zeros='0 0 0 0 0 0 0 0'
message='1 2 3 4 5 6 7 8'
probe="$(printf '9007 1 0 1 101 0 0 0 0 %s %s %s\n' "$message" "$zeros" "$zeros" | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=open state=1' ]] || \
  fail "native adapter failed its open probe: $probe"

printf 'BUILT_OBLIGATION path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
