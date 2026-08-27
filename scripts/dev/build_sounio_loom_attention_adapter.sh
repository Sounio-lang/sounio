#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_ATTENTION_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_ATTENTION_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_ATTENTION_MODULE:-$ROOT_DIR/stdlib/coordination/loom_attention_compiler.sio}"
ENTRYPOINT="${SOUNIO_LOOM_ATTENTION_MAIN:-$ROOT_DIR/tools/loom/attention_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_ATTENTION_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-attention-runtime}"
PREBUILT="${SOUNIO_LOOM_ATTENTION_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-attention: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "attention module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "attention entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-attention-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_attention_adapter.sio"
  compiled="$work/sounio-loom-attention-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

ones='1 1 1 1 1 1 1 1'
twos='2 2 2 2 2 2 2 2'
threes='3 3 3 3 3 3 3 3'
zeros='0 0 0 0 0 0 0 0'
probe="$(printf '9009 1 1 100 101 201 202 301 401 900 800 700 50 100 800 900 900 50 100 %s %s %s %s\n' \
  "$ones" "$twos" "$threes" "$zeros" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=information-first' ]] || \
  fail "native adapter failed its compile probe: $probe"

printf 'BUILT_ATTENTION path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
