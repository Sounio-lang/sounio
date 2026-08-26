#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_OUTCOME_AUTHORITY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_OUTCOME_AUTHORITY_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_OUTCOME_AUTHORITY_MODULE:-$ROOT_DIR/stdlib/coordination/loom_outcome_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_OUTCOME_AUTHORITY_MAIN:-$ROOT_DIR/tools/loom/outcome_authority_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_OUTCOME_AUTHORITY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-outcome-authority-runtime}"
PREBUILT="${SOUNIO_LOOM_OUTCOME_AUTHORITY_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-outcome-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "outcome-authority module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "outcome-authority entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-outcome-authority-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_outcome_authority_adapter.sio"
  compiled="$work/sounio-loom-outcome-authority-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native outcome-authority adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
three='3 3 3 3 3 3 3 3'
four='4 4 4 4 4 4 4 4'
five='5 5 5 5 5 5 5 5'
six='6 6 6 6 6 6 6 6'
probe="$(printf '9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 %s %s %s %s %s %s %s %s\n' \
  "$one" "$two" "$two" "$three" "$three" "$four" "$four" "$five" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_OUTCOME_AUTHORITY_ACCEPT schema=loom-native-outcome-authority-v0 transition=consume state=verified' ]] || \
  fail "native outcome-authority adapter failed its consume probe: $probe"

printf 'BUILT_OUTCOME_AUTHORITY path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
