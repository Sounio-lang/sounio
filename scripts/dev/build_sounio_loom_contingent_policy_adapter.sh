#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_CONTINGENT_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_CONTINGENT_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_CONTINGENT_MODULE:-$ROOT_DIR/stdlib/coordination/loom_contingent_policy.sio}"
ENTRYPOINT="${SOUNIO_LOOM_CONTINGENT_MAIN:-$ROOT_DIR/tools/loom/contingent_policy_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_CONTINGENT_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-contingent-runtime}"
PREBUILT="${SOUNIO_LOOM_CONTINGENT_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-contingent: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "contingent-policy module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "contingent-policy entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-contingent-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_contingent_policy_adapter.sio"
  compiled="$work/sounio-loom-contingent-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native contingent-policy adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

ones='1 1 1 1 1 1 1 1'
twos='2 2 2 2 2 2 2 2'
threes='3 3 3 3 3 3 3 3'
fours='4 4 4 4 4 4 4 4'
fives='5 5 5 5 5 5 5 5'
sixes='6 6 6 6 6 6 6 6'
sevens='7 7 7 7 7 7 7 7'
zeros='0 0 0 0 0 0 0 0'
probe="$(printf '9011 1 1 0 100 100 10 10 101 201 202 301 0 0 0 401 501 900 500 400 40 50 50 5 5 800 900 900 50 50 50 5 5 %s %s %s %s %s %s %s %s\n' \
  "$ones" "$twos" "$threes" "$fours" "$fives" "$sixes" "$sevens" "$zeros" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=compile policy=information-first' ]] || \
  fail "native contingent-policy adapter failed its compile probe: $probe"

printf 'BUILT_CONTINGENT path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
