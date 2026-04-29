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
STAGE2_DRIVER="$OUT_DIR/native_compile_driver.stage2"
STAGE3_DRIVER="$OUT_DIR/native_compile_driver.stage3"
STAGE2_COMPILE_LOG="$LOG_DIR/native_compile_driver.stage2.compile.log"
STAGE3_COMPILE_LOG="$LOG_DIR/native_compile_driver.stage3.compile.log"
STAGE2_HELLO_BIN="$OUT_DIR/hello.stage2"
STAGE2_HELLO_COMPILE_LOG="$LOG_DIR/hello.stage2.compile.log"
STAGE2_STDOUT="$LOG_DIR/hello.stage2.stdout"

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

# ── Stage2: stage1 native binary compiles the driver again ───────────────────
if ! "$STAGE1_DRIVER" "$DRIVER_SRC" -o "$STAGE2_DRIVER" >"$STAGE2_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: stage1 driver failed to compile stage2" >&2
  tail -n 120 "$STAGE2_COMPILE_LOG" >&2 || true
  exit 1
fi
if [[ ! -x "$STAGE2_DRIVER" ]]; then
  echo "[native-v2-driver-self] FAIL: stage2 driver is not executable" >&2
  exit 1
fi

# ── Stage2 hello parity ───────────────────────────────────────────────────────
if ! "$STAGE2_DRIVER" "$HELLO_SRC" -o "$STAGE2_HELLO_BIN" >"$STAGE2_HELLO_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: stage2 driver failed while compiling hello" >&2
  tail -n 120 "$STAGE2_HELLO_COMPILE_LOG" >&2 || true
  exit 1
fi
"$STAGE2_HELLO_BIN" >"$STAGE2_STDOUT" 2>/dev/null
if ! cmp -s "$EXPECTED_STDOUT" "$STAGE2_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: stage2 hello stdout mismatch" >&2
  diff -u "$EXPECTED_STDOUT" "$STAGE2_STDOUT" >&2 || true
  exit 1
fi

# ── Fixed-point: stage2 compiles stage3, stage3 == stage2 ────────────────────
if ! "$STAGE2_DRIVER" "$DRIVER_SRC" -o "$STAGE3_DRIVER" >"$STAGE3_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: stage2 driver failed to compile stage3" >&2
  tail -n 120 "$STAGE3_COMPILE_LOG" >&2 || true
  exit 1
fi
if ! cmp -s "$STAGE2_DRIVER" "$STAGE3_DRIVER"; then
  echo "[native-v2-driver-self] FAIL: fixed-point broken — stage2 != stage3" >&2
  python3 -c "
a=open('$STAGE2_DRIVER','rb').read(); b=open('$STAGE3_DRIVER','rb').read()
diffs=[(i,a[i],b[i]) for i in range(min(len(a),len(b))) if a[i]!=b[i]]
print(f'  {len(diffs)} byte(s) differ')
for i,x,y in diffs[:5]: print(f'  offset 0x{i:x}: {x:02x} vs {y:02x}')
" >&2 || true
  exit 1
fi
STAGE2_MD5="$(md5sum "$STAGE2_DRIVER" | cut -d' ' -f1)"
printf '[native-v2-driver-self] fixed-point md5=%s\n' "$STAGE2_MD5"

# ── Epistemic fixed-point: read .sounio.epistemic from stage1/2/3 ────────────
read_epistemic() {
  python3 - "$1" <<'PYEOF'
import struct, sys, os
path = sys.argv[1]
if not os.path.exists(path):
    print("absent")
    sys.exit(0)
data = open(path, 'rb').read()
idx = data.find(b'SIEP')
if idx < 0:
    print("absent")
    sys.exit(0)
chunk = data[idx:idx+24]
version = struct.unpack_from('<I', chunk, 4)[0]
instr_count = struct.unpack_from('<Q', chunk, 8)[0]
u_c_scaled = struct.unpack_from('<Q', chunk, 16)[0]
print(f"{version}:{instr_count}:{u_c_scaled}")
PYEOF
}

STAGE1_EP="$(read_epistemic "$STAGE1_DRIVER")"
STAGE2_EP="$(read_epistemic "$STAGE2_DRIVER")"
STAGE3_EP="$(read_epistemic "$STAGE3_DRIVER")"
printf '[native-v2-driver-self] epistemic stage1=%s stage2=%s stage3=%s\n' \
  "$STAGE1_EP" "$STAGE2_EP" "$STAGE3_EP"

if [[ "$STAGE2_EP" == "absent" || "$STAGE3_EP" == "absent" ]]; then
  echo "[native-v2-driver-self] FAIL: epistemic section missing from stage2 or stage3" >&2
  exit 1
fi
if [[ "$STAGE2_EP" != "$STAGE3_EP" ]]; then
  echo "[native-v2-driver-self] FAIL: epistemic fixed-point broken — stage2 != stage3 epistemic profile" >&2
  exit 1
fi
STAGE2_INSTR="$(echo "$STAGE2_EP" | cut -d: -f2)"
printf '[native-v2-driver-self] epistemic-fixed-point instr=%s u_c=0 (stage2==stage3 confidence-stable)\n' "$STAGE2_INSTR"

echo "[native-v2-driver-self] PASS: baseline, stage1 driver, stage2 driver, fixed-point (stage2==stage3), hello parity across all stages, epistemic-fixed-point verified"
