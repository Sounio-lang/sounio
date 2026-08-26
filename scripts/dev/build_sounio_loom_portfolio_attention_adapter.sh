#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PORTFOLIO_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PORTFOLIO_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_PORTFOLIO_MODULE:-$ROOT_DIR/stdlib/coordination/loom_portfolio_attention.sio}"
ENTRYPOINT="${SOUNIO_LOOM_PORTFOLIO_MAIN:-$ROOT_DIR/tools/loom/portfolio_attention_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_PORTFOLIO_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-portfolio-runtime}"
PREBUILT="${SOUNIO_LOOM_PORTFOLIO_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-portfolio: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "portfolio module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "portfolio entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-portfolio-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_portfolio_attention_adapter.sio"
  compiled="$work/sounio-loom-portfolio-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native portfolio adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

ones='1 1 1 1 1 1 1 1'
twos='2 2 2 2 2 2 2 2'
threes='3 3 3 3 3 3 3 3'
fours='4 4 4 4 4 4 4 4'
fives='5 5 5 5 5 5 5 5'
zeros='0 0 0 0 0 0 0 0'
probe="$(printf '9010 1 1 100 100 10 10 101 201 202 301 401 900 800 700 40 50 50 5 5 800 900 900 50 50 50 5 5 %s %s %s %s %s %s\n' \
  "$ones" "$twos" "$threes" "$fours" "$fives" "$zeros" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=compile policy=information-first' ]] || \
  fail "native portfolio adapter failed its compile probe: $probe"

printf 'BUILT_PORTFOLIO path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
