#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_STRUCT_MUTATION_GATE_DIR:-$(mktemp -d /tmp/sounio-native-v2-struct-mutation.XXXXXX)}"
BIN="$OUT_DIR/struct_mutation"
CHECK_LOG="$OUT_DIR/driver.check.log"
COMPILE_LOG="$OUT_DIR/struct_mutation.compile.log"
STDOUT_LOG="$OUT_DIR/struct_mutation.stdout"
EXPECTED_LOG="$OUT_DIR/struct_mutation.expected"

mkdir -p "$OUT_DIR"

printf '[native-v2-struct-mutation] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-struct-mutation] out=%s\n' "$OUT_DIR"

"$SOUC_BIN" check self-hosted/compiler/native_compile_driver.sio >"$CHECK_LOG" 2>&1

"$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- \
  examples/native/struct_mutation.sio -o "$BIN" >"$COMPILE_LOG" 2>&1

if [[ ! -x "$BIN" ]]; then
  echo "[native-v2-struct-mutation] FAIL: generated binary not executable: $BIN" >&2
  tail -n 40 "$COMPILE_LOG" >&2 || true
  exit 1
fi

"$BIN" >"$STDOUT_LOG" 2>/dev/null
printf '14\n' >"$EXPECTED_LOG"

if ! cmp -s "$EXPECTED_LOG" "$STDOUT_LOG"; then
  echo "[native-v2-struct-mutation] FAIL: output mismatch" >&2
  echo "[native-v2-struct-mutation] expected: $(cat "$EXPECTED_LOG")" >&2
  echo "[native-v2-struct-mutation] got:      $(cat "$STDOUT_LOG")" >&2
  exit 1
fi

# Regression: struct_basic gate must still pass
echo "[native-v2-struct-mutation] PASS"
