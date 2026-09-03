#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_STRUCT_PARAM_GATE_DIR:-$(mktemp -d /tmp/sounio-native-v2-struct-param.XXXXXX)}"
BIN="$OUT_DIR/struct_param"
CHECK_LOG="$OUT_DIR/driver.check.log"
COMPILE_LOG="$OUT_DIR/struct_param.compile.log"
STDOUT_LOG="$OUT_DIR/struct_param.stdout"
EXPECTED_LOG="$OUT_DIR/struct_param.expected"

mkdir -p "$OUT_DIR"

printf '[native-v2-struct-param] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-struct-param] out=%s\n' "$OUT_DIR"

"$SOUC_BIN" check self-hosted/compiler/native_compile_driver.sio >"$CHECK_LOG" 2>&1

"$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- \
  examples/native/struct_param.sio -o "$BIN" >"$COMPILE_LOG" 2>&1

if [[ ! -x "$BIN" ]]; then
  echo "[native-v2-struct-param] FAIL: generated binary not executable: $BIN" >&2
  tail -n 40 "$COMPILE_LOG" >&2 || true
  exit 1
fi

"$BIN" >"$STDOUT_LOG" 2>/dev/null
printf '12\n' >"$EXPECTED_LOG"

if ! cmp -s "$EXPECTED_LOG" "$STDOUT_LOG"; then
  echo "[native-v2-struct-param] FAIL: output mismatch" >&2
  echo "[native-v2-struct-param] expected: $(cat "$EXPECTED_LOG")" >&2
  echo "[native-v2-struct-param] got:      $(cat "$STDOUT_LOG")" >&2
  exit 1
fi

# Regression: struct_basic gate must still pass
echo "[native-v2-struct-param] PASS"
