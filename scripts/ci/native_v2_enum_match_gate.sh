#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_ENUM_MATCH_GATE_DIR:-$(mktemp -d /tmp/sounio-native-v2-enum-match.XXXXXX)}"
BIN="$OUT_DIR/enum_match"
CHECK_LOG="$OUT_DIR/driver.check.log"
COMPILE_LOG="$OUT_DIR/enum_match.compile.log"
STDOUT_LOG="$OUT_DIR/enum_match.stdout"
EXPECTED_LOG="$OUT_DIR/enum_match.expected"

mkdir -p "$OUT_DIR"

printf '[native-v2-enum-match] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-enum-match] out=%s\n' "$OUT_DIR"

"$SOUC_BIN" check self-hosted/compiler/native_compile_driver.sio >"$CHECK_LOG" 2>&1

"$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- \
  examples/native/enum_match.sio -o "$BIN" >"$COMPILE_LOG" 2>&1

if [[ ! -x "$BIN" ]]; then
  echo "[native-v2-enum-match] FAIL: generated binary not executable: $BIN" >&2
  tail -n 40 "$COMPILE_LOG" >&2 || true
  exit 1
fi

"$BIN" >"$STDOUT_LOG" 2>/dev/null
printf '2\n' >"$EXPECTED_LOG"

if ! cmp -s "$EXPECTED_LOG" "$STDOUT_LOG"; then
  echo "[native-v2-enum-match] FAIL: output mismatch" >&2
  echo "[native-v2-enum-match] expected: $(cat "$EXPECTED_LOG")" >&2
  echo "[native-v2-enum-match] got:      $(cat "$STDOUT_LOG")" >&2
  exit 1
fi

echo "[native-v2-enum-match] PASS"
