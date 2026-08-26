#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_MODULE:-$ROOT_DIR/stdlib/coordination/loom_witness_epoch_handoff.sio}"
ENTRYPOINT="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_MAIN:-$ROOT_DIR/tools/loom/witness_epoch_handoff_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-epoch-handoff-runtime}"
PREBUILT="${SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-witness-epoch-handoff: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "witness epoch handoff module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "witness epoch handoff entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch-handoff-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_witness_epoch_handoff_adapter.sio"
  compiled="$work/sounio-loom-witness-epoch-handoff-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native witness epoch handoff adapter'
  install -m 0755 "$compiled" "$OUTPUT"
fi

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
three='3 3 3 3 3 3 3 3'
four='4 4 4 4 4 4 4 4'
five='5 5 5 5 5 5 5 5'
six='6 6 6 6 6 6 6 6'
seven='7 7 7 7 7 7 7 7'
zero='0 0 0 0 0 0 0 0'
probe="$(printf '9015 1 1 1 2 3 3 4 4 501 501 7 1 12 12 %s %s %s %s %s %s %s %s %s\n' \
  "$one" "$two" "$three" "$four" "$five" "$five" \
  "$six" "$seven" "$zero" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_WITNESS_EPOCH_HANDOFF_ACCEPT schema=loom-native-witness-epoch-handoff-v0 transition=joint-quorum state=prepared' ]] || \
  fail "native witness epoch handoff adapter failed its probe: $probe"

printf 'BUILT_WITNESS_EPOCH_HANDOFF path=%s language=Sounio engine=%s\n' \
  "$OUTPUT" "$ENGINE"
