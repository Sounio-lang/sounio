#!/usr/bin/env bash
# native_v2_prebundle_gate.sh
#
# Gate: native_prebundle.sio correctly walks use imports and produces a flat
# bundle that the native_compile_driver can compile and run.
#
# Test program: two-file program (main + math_utils), expected output: 49\n55
set -euo pipefail

SOUC="${SOUC:-./bin/souc}"
TMPDIR_GATE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE"' EXIT

echo "[native-v2-prebundle] souc=$SOUC"

# ── Create test module tree ───────────────────────────────────────────────────
cat > "$TMPDIR_GATE/math_utils.sio" << 'SOUNIO'
module math_utils

pub fn square(x: i64) -> i64 { x * x }

pub fn sum_n(n: i64) -> i64 {
    var acc: i64 = 0
    var i: i64 = 1
    while i <= n {
        acc = acc + i
        i = i + 1
    }
    acc
}
SOUNIO

cat > "$TMPDIR_GATE/main.sio" << 'SOUNIO'
use math_utils::*

fn main() -> i64 with IO {
    let r = square(7)
    print_int(r)
    print("\n")
    let s = sum_n(10)
    print_int(s)
    print("\n")
    0
}
SOUNIO

# ── Phase 1: Bundle ───────────────────────────────────────────────────────────
BUNDLE="$TMPDIR_GATE/bundle.sio"
$SOUC run self-hosted/compiler/native_prebundle.sio \
    -- "$TMPDIR_GATE/main.sio" -o "$BUNDLE" --root "$TMPDIR_GATE" 2>&1 \
    | grep "native_prebundle:"

if [ ! -f "$BUNDLE" ] || [ ! -s "$BUNDLE" ]; then
    echo "[native-v2-prebundle] FAIL: bundle not produced"
    exit 1
fi

MODULES=$(grep -c "^// SOURCE:" "$BUNDLE" || true)
echo "[native-v2-prebundle] bundle_modules=$MODULES"
if [ "$MODULES" -ne 2 ]; then
    echo "[native-v2-prebundle] FAIL: expected 2 SOURCE markers, got $MODULES"
    exit 1
fi

USE_LINES=$(grep -c "^use \|^module " "$BUNDLE" || true)
if [ "$USE_LINES" -ne 0 ]; then
    echo "[native-v2-prebundle] FAIL: bundle contains $USE_LINES use/module lines"
    exit 1
fi

# ── Phase 2: Compile bundle → ELF ────────────────────────────────────────────
ELF="$TMPDIR_GATE/test.elf"
$SOUC run self-hosted/compiler/native_compile_driver.sio \
    -- "$BUNDLE" -o "$ELF" 2>&1 | grep "native_compile:"

if [ ! -f "$ELF" ]; then
    echo "[native-v2-prebundle] FAIL: ELF not produced"
    exit 1
fi
chmod +x "$ELF"

# ── Phase 3: Run ELF ─────────────────────────────────────────────────────────
ACTUAL=$("$ELF")
EXPECTED=$'49\n55'
if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "[native-v2-prebundle] PASS: bundle→compile→run ok (49, 55)"
    exit 0
else
    echo "[native-v2-prebundle] FAIL: expected '$EXPECTED' got '$ACTUAL'"
    exit 1
fi
