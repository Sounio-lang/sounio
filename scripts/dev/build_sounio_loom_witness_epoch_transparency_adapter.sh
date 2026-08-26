#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_MODULE:-$ROOT_DIR/stdlib/coordination/loom_witness_epoch_transparency.sio}"
ENTRYPOINT="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_MAIN:-$ROOT_DIR/tools/loom/epoch_transparency_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-epoch-transparency-runtime}"
PREBUILT="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-witness-epoch-transparency: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "epoch transparency module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "epoch transparency entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch-transparency-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_witness_epoch_transparency_adapter.sio"
  compiled="$work/sounio-loom-witness-epoch-transparency-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native epoch transparency adapter'
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
probe="$(printf '9016 1 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 %s %s %s %s %s %s %s %s %s %s %s %s\n' \
  "$one" "$one" "$zero" "$zero" "$two" "$two" \
  "$three" "$three" "$four" "$five" "$six" "$seven" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_WITNESS_EPOCH_TRANSPARENCY_ACCEPT schema=loom-native-witness-epoch-transparency-v0 rollback_bound=latest-quorum-witnessed-epoch state=verified' ]] || \
  fail "native epoch transparency adapter failed its probe: $probe"

printf 'BUILT_WITNESS_EPOCH_TRANSPARENCY path=%s language=Sounio engine=%s\n' \
  "$OUTPUT" "$ENGINE"
