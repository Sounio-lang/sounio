#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_GATE_DIR:-$(mktemp -d /tmp/sounio-native-v2-serious-track.XXXXXX)}"
BIN="$OUT_DIR/hello-native-v2"
CHECK_LOG="$OUT_DIR/native_compile_driver.check.log"
COMPILE_LOG="$OUT_DIR/native_compile_driver.compile.log"
STDOUT_LOG="$OUT_DIR/hello.stdout"
STDERR_LOG="$OUT_DIR/hello.stderr"
EXPECTED_LOG="$OUT_DIR/hello.expected"
SECTIONS_LOG="$OUT_DIR/hello.readelf.sections"
STRINGS_LOG="$OUT_DIR/hello.strings"

mkdir -p "$OUT_DIR"

printf '[native-v2] souc=%s\n' "$SOUC_BIN"
printf '[native-v2] out=%s\n' "$OUT_DIR"

"$SOUC_BIN" check self-hosted/compiler/native_compile_driver.sio >"$CHECK_LOG" 2>&1

"$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- \
  examples/native/hello.sio -o "$BIN" >"$COMPILE_LOG" 2>&1

chmod +x "$BIN" 2>/dev/null || true

if [[ ! -x "$BIN" ]]; then
  echo "[native-v2] FAIL: generated binary is not executable: $BIN" >&2
  tail -n 40 "$COMPILE_LOG" >&2 || true
  exit 1
fi

"$BIN" >"$STDOUT_LOG" 2>"$STDERR_LOG"
printf 'Hello from self-hosted Sounio!\n42\n' >"$EXPECTED_LOG"

if ! cmp -s "$EXPECTED_LOG" "$STDOUT_LOG"; then
  echo "[native-v2] FAIL: generated binary output mismatch" >&2
  echo "[native-v2] expected:" >&2
  cat "$EXPECTED_LOG" >&2
  echo "[native-v2] got:" >&2
  cat "$STDOUT_LOG" >&2
  exit 1
fi

if command -v readelf >/dev/null 2>&1; then
    readelf -S "$BIN" >"$SECTIONS_LOG"
    if ! grep -q '\.rodata' "$SECTIONS_LOG" || ! grep -q '\.data' "$SECTIONS_LOG"; then
      readelf -l "$BIN" >>"$SECTIONS_LOG"
      grep -q 'LOAD' "$SECTIONS_LOG"
      grep -q ' R ' "$SECTIONS_LOG"
      grep -q 'RW' "$SECTIONS_LOG"
    fi
fi

if command -v strings >/dev/null 2>&1; then
  strings "$BIN" >"$STRINGS_LOG"
  grep -q 'Hello from self-hosted Sounio!' "$STRINGS_LOG"
fi

echo "[native-v2] PASS: driver check, ELF build, .rodata/.data witness, and runtime output"
