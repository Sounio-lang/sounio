#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EPISTEMIC_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EPISTEMIC_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_EPISTEMIC_MODULE:-$ROOT_DIR/stdlib/coordination/loom_epistemic_machine.sio}"
ENTRYPOINT="${SOUNIO_LOOM_EPISTEMIC_MAIN:-$ROOT_DIR/tools/loom/epistemic_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_EPISTEMIC_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-epistemic-runtime}"
PREBUILT="${SOUNIO_LOOM_EPISTEMIC_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-epistemic: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "epistemic module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "epistemic entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-epistemic-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_epistemic_adapter.sio"
  compiled="$work/sounio-loom-epistemic-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

zeros='0 0 0 0 0 0 0 0'
probe="$(printf '9008 1 0 1 101 0 0 0 0 0 %s %s %s %s %s %s %s\n' \
  "$zeros" "$zeros" "$zeros" "$zeros" "$zeros" "$zeros" "$zeros" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_EPISTEMIC_ACCEPT schema=loom-native-epistemic-v0 transition=create state=active' ]] || \
  fail "native adapter failed its create probe: $probe"

printf 'BUILT_EPISTEMIC path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
