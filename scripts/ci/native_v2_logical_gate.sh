#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-logical] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-logical] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_LOGICAL_GATE_DIR:-$(mktemp -d /tmp/sounio-native-v2-logical.XXXXXX)}"
BIN="$OUT_DIR/logical_ops"
COMPILE_LOG="$OUT_DIR/logical_ops.compile.log"
STDOUT_LOG="$OUT_DIR/logical_ops.stdout"
EXPECTED_LOG="$OUT_DIR/logical_ops.expected"

mkdir -p "$OUT_DIR"

printf '[native-v2-logical] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-logical] out=%s\n' "$OUT_DIR"

"$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- \
  examples/native/logical_ops.sio -o "$BIN" >"$COMPILE_LOG" 2>&1

if [[ ! -x "$BIN" ]]; then
  echo "[native-v2-logical] FAIL: driver did not produce executable" >&2
  tail -n 40 "$COMPILE_LOG" >&2 || true
  exit 1
fi

"$BIN" >"$STDOUT_LOG" 2>/dev/null

# x=3, y=7: (x>0 && y>0)=1 → ok=1; (x>10||y>5)=1 → ok=3; (!0)=1 → ok=7
printf '7\n' >"$EXPECTED_LOG"

if ! cmp -s "$EXPECTED_LOG" "$STDOUT_LOG"; then
  echo "[native-v2-logical] FAIL: stdout mismatch" >&2
  diff -u "$EXPECTED_LOG" "$STDOUT_LOG" >&2 || true
  exit 1
fi

echo "[native-v2-logical] PASS: &&, ||, ! logical operators, result=7"
