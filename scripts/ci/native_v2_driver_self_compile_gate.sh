#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-driver-self] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-driver-self] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR:-$(mktemp -d /tmp/sounio-native-v2-driver-self.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
HELLO_SRC="examples/native/hello.sio"
BASELINE_HELLO_BIN="$OUT_DIR/hello.baseline"
STAGE1_DRIVER="$OUT_DIR/native_compile_driver.stage1"
STAGE1_HELLO_BIN="$OUT_DIR/hello.stage1"

CHECK_LOG="$LOG_DIR/native_compile_driver.check.log"
BASELINE_GATE_LOG="$LOG_DIR/native_v2_serious_track_gate.log"
BASELINE_COMPILE_LOG="$LOG_DIR/hello.baseline.compile.log"
STAGE1_COMPILE_LOG="$LOG_DIR/native_compile_driver.stage1.compile.log"
STAGE1_HELLO_COMPILE_LOG="$LOG_DIR/hello.stage1.compile.log"
BASELINE_STDOUT="$LOG_DIR/hello.baseline.stdout"
BASELINE_STDERR="$LOG_DIR/hello.baseline.stderr"
STAGE1_STDOUT="$LOG_DIR/hello.stage1.stdout"
STAGE1_STDERR="$LOG_DIR/hello.stage1.stderr"
EXPECTED_STDOUT="$LOG_DIR/hello.expected.stdout"
STAGE1_FILE_LOG="$LOG_DIR/native_compile_driver.stage1.file.txt"
HELLO_FILE_LOG="$LOG_DIR/hello.stage1.file.txt"
HELLO_SECTIONS_LOG="$LOG_DIR/hello.stage1.readelf.sections"
HELLO_STRINGS_LOG="$LOG_DIR/hello.stage1.strings"

printf '[native-v2-driver-self] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-driver-self] out=%s\n' "$OUT_DIR"

bash scripts/ci/native_v2_serious_track_gate.sh >"$BASELINE_GATE_LOG" 2>&1

"$SOUC_BIN" check "$DRIVER_SRC" >"$CHECK_LOG" 2>&1

"$SOUC_BIN" run "$DRIVER_SRC" -- "$HELLO_SRC" -o "$BASELINE_HELLO_BIN" >"$BASELINE_COMPILE_LOG" 2>&1
if [[ ! -x "$BASELINE_HELLO_BIN" ]]; then
  echo "[native-v2-driver-self] FAIL: baseline driver did not produce executable hello" >&2
  tail -n 80 "$BASELINE_COMPILE_LOG" >&2 || true
  exit 1
fi

"$BASELINE_HELLO_BIN" >"$BASELINE_STDOUT" 2>"$BASELINE_STDERR"
printf 'Hello from self-hosted Sounio!\n42\n' >"$EXPECTED_STDOUT"
if ! cmp -s "$EXPECTED_STDOUT" "$BASELINE_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: baseline hello stdout mismatch" >&2
  diff -u "$EXPECTED_STDOUT" "$BASELINE_STDOUT" >&2 || true
  exit 1
fi

if ! "$SOUC_BIN" run "$DRIVER_SRC" -- "$DRIVER_SRC" -o "$STAGE1_DRIVER" >"$STAGE1_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: native-v2 driver did not self-compile stage1" >&2
  echo "[native-v2-driver-self] compile log: $STAGE1_COMPILE_LOG" >&2
  tail -n 120 "$STAGE1_COMPILE_LOG" >&2 || true
  exit 1
fi

if [[ ! -x "$STAGE1_DRIVER" ]]; then
  echo "[native-v2-driver-self] FAIL: stage1 driver is not executable: $STAGE1_DRIVER" >&2
  tail -n 120 "$STAGE1_COMPILE_LOG" >&2 || true
  exit 1
fi

if command -v file >/dev/null 2>&1; then
  file "$STAGE1_DRIVER" >"$STAGE1_FILE_LOG"
  grep -q 'ELF 64-bit LSB executable, x86-64' "$STAGE1_FILE_LOG"
fi

if ! "$STAGE1_DRIVER" "$HELLO_SRC" -o "$STAGE1_HELLO_BIN" >"$STAGE1_HELLO_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: stage1 driver failed while compiling hello" >&2
  echo "[native-v2-driver-self] compile log: $STAGE1_HELLO_COMPILE_LOG" >&2
  tail -n 120 "$STAGE1_HELLO_COMPILE_LOG" >&2 || true
  exit 1
fi
if [[ ! -x "$STAGE1_HELLO_BIN" ]]; then
  echo "[native-v2-driver-self] FAIL: stage1 driver did not produce executable hello" >&2
  tail -n 120 "$STAGE1_HELLO_COMPILE_LOG" >&2 || true
  exit 1
fi

if command -v file >/dev/null 2>&1; then
  file "$STAGE1_HELLO_BIN" >"$HELLO_FILE_LOG"
  grep -q 'ELF 64-bit LSB executable, x86-64' "$HELLO_FILE_LOG"
fi

"$STAGE1_HELLO_BIN" >"$STAGE1_STDOUT" 2>"$STAGE1_STDERR"
if ! cmp -s "$EXPECTED_STDOUT" "$STAGE1_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: stage1 hello stdout mismatch" >&2
  diff -u "$EXPECTED_STDOUT" "$STAGE1_STDOUT" >&2 || true
  exit 1
fi

if ! cmp -s "$BASELINE_STDOUT" "$STAGE1_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: stage1 hello stdout differs from baseline driver output" >&2
  diff -u "$BASELINE_STDOUT" "$STAGE1_STDOUT" >&2 || true
  exit 1
fi

if command -v readelf >/dev/null 2>&1; then
  readelf -S "$STAGE1_HELLO_BIN" >"$HELLO_SECTIONS_LOG"
  grep -q '\.rodata' "$HELLO_SECTIONS_LOG"
  grep -q '\.data' "$HELLO_SECTIONS_LOG"
fi

if command -v strings >/dev/null 2>&1; then
  strings "$STAGE1_HELLO_BIN" >"$HELLO_STRINGS_LOG"
  grep -q 'Hello from self-hosted Sounio!' "$HELLO_STRINGS_LOG"
fi

echo "[native-v2-driver-self] PASS: baseline, stage1 driver, stage1 hello ELF, sections, strings, and stdout parity"
