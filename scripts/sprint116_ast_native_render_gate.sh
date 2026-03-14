#!/bin/bash
# Gate script for Sprint 116: AST-direct native compilation of triangle_basic.sio
#
# Validates:
#   1. AST-direct compiler (render_native_compile_driver_lean.sio) passes souc check
#   2. Compiles triangle_basic.sio to native ELF binary
#   3. Binary is executable and exits cleanly
#   4. PPM output has correct header (P3, 128x128, 255)
#   5. PPM output has correct line count (16387 = 3 header + 128*128 data)
#   6. PPM output is pixel-perfect match with JIT reference
#   7. Triangle pixels present (non-zero RGB in interior)
#   8. Background pixels present (dark blue gradient)
#
# Usage:
#   bash scripts/sprint116_ast_native_render_gate.sh

set -eo pipefail

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"
PASS=0
FAIL=0
NOT_RUN=0
TMPDIR_GATE=$(mktemp -d /tmp/s116_gate.XXXXXX)

check() {
    local name="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo "  PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    local name="$1"
    echo "  NOT_RUN  $name"
    NOT_RUN=$((NOT_RUN + 1))
}

echo "=== Sprint 116: AST-Direct Native Render Gate ==="
echo "souc: $SOUC"
echo "stdlib: $SOUNIO_STDLIB_PATH"
echo ""

# --- T1: Source file exists ---
echo "[gate] T1: source files exist"
if [ -f "self-hosted/compiler/render_native_compile_driver_lean.sio" ] && \
   [ -f "examples/render/triangle_basic.sio" ]; then
    check "source_files_exist" 0
else
    check "source_files_exist" 1
fi

# --- T2: souc check passes ---
echo "[gate] T2: souc check driver"
if timeout 60 $SOUC check self-hosted/compiler/render_native_compile_driver_lean.sio > /dev/null 2>&1; then
    check "driver_check" 0
else
    check "driver_check" 1
fi

# --- T3: Compile triangle_basic.sio to native ELF ---
echo "[gate] T3: native compile triangle_basic.sio"
NATIVE_ELF="$TMPDIR_GATE/triangle.elf"
if timeout 120 $SOUC run self-hosted/compiler/render_native_compile_driver_lean.sio -- \
    examples/render/triangle_basic.sio "$NATIVE_ELF" > "$TMPDIR_GATE/compile.log" 2>&1; then
    check "native_compile" 0
else
    check "native_compile" 1
    echo "    compile log:"
    cat "$TMPDIR_GATE/compile.log" 2>/dev/null | head -10 | sed 's/^/    /'
fi

# --- T4: Binary exists and is executable ---
echo "[gate] T4: binary exists"
if [ -f "$NATIVE_ELF" ]; then
    chmod +x "$NATIVE_ELF"
    check "binary_exists" 0
else
    check "binary_exists" 1
fi

# --- T5: Binary executes and produces output ---
echo "[gate] T5: execute binary"
NATIVE_PPM="$TMPDIR_GATE/native.ppm"
if [ -f "$NATIVE_ELF" ]; then
    if timeout 30 "$NATIVE_ELF" > "$NATIVE_PPM" 2>/dev/null; then
        check "execute_binary" 0
    else
        check "execute_binary" 1
    fi
else
    skip "execute_binary"
fi

# --- T6: PPM header correct ---
echo "[gate] T6: PPM header"
if [ -f "$NATIVE_PPM" ]; then
    HEADER1=$(sed -n '1p' "$NATIVE_PPM")
    HEADER2=$(sed -n '2p' "$NATIVE_PPM")
    HEADER3=$(sed -n '3p' "$NATIVE_PPM")
    if [ "$HEADER1" = "P3" ] && [ "$HEADER2" = "128 128" ] && [ "$HEADER3" = "255" ]; then
        check "ppm_header" 0
    else
        check "ppm_header" 1
        echo "    got: '$HEADER1' '$HEADER2' '$HEADER3'"
    fi
else
    skip "ppm_header"
fi

# --- T7: PPM line count ---
echo "[gate] T7: PPM line count"
if [ -f "$NATIVE_PPM" ]; then
    LINES=$(wc -l < "$NATIVE_PPM")
    if [ "$LINES" -eq 16387 ]; then
        check "ppm_line_count" 0
    else
        check "ppm_line_count" 1
        echo "    expected 16387, got $LINES"
    fi
else
    skip "ppm_line_count"
fi

# --- T8: JIT reference output ---
echo "[gate] T8: JIT reference"
JIT_PPM="$TMPDIR_GATE/jit_ref.ppm"
if timeout 30 $SOUC run examples/render/triangle_basic.sio > "$JIT_PPM" 2>/dev/null; then
    check "jit_reference" 0
else
    check "jit_reference" 1
fi

# --- T9: Pixel-perfect match ---
echo "[gate] T9: pixel-perfect match (native vs JIT)"
if [ -f "$NATIVE_PPM" ] && [ -f "$JIT_PPM" ]; then
    DIFF_COUNT=$(diff "$JIT_PPM" "$NATIVE_PPM" | wc -l)
    if [ "$DIFF_COUNT" -eq 0 ]; then
        check "pixel_perfect_match" 0
    else
        check "pixel_perfect_match" 1
        echo "    diff lines: $DIFF_COUNT"
        diff "$JIT_PPM" "$NATIVE_PPM" | head -10 | sed 's/^/    /'
    fi
else
    skip "pixel_perfect_match"
fi

# --- T10: Triangle pixels present (non-zero interior) ---
echo "[gate] T10: triangle pixels present"
if [ -f "$NATIVE_PPM" ]; then
    # Center pixel (row 64, col 63) = line 3 + 64*128 + 63 + 1 = 8259
    HIGH_PIX=$(sed -n '8259p' "$NATIVE_PPM" | awk '{print ($1 > 50 || $2 > 50 || $3 > 50) ? 1 : 0}')
    if [ "$HIGH_PIX" -eq 1 ]; then
        check "triangle_pixels" 0
    else
        check "triangle_pixels" 1
    fi
else
    skip "triangle_pixels"
fi

# --- T11: Background pixels present (dark gradient) ---
echo "[gate] T11: background pixels"
if [ -f "$NATIVE_PPM" ]; then
    # Row 0 col 0 should be dark background
    BG_PIX=$(sed -n '4p' "$NATIVE_PPM" | awk '{print ($1 < 50 && $2 < 50 && $3 < 50) ? 1 : 0}')
    if [ "$BG_PIX" -eq 1 ]; then
        check "background_pixels" 0
    else
        check "background_pixels" 1
    fi
else
    skip "background_pixels"
fi

# --- T12: Binary size reasonable (5-20KB) ---
echo "[gate] T12: binary size"
if [ -f "$NATIVE_ELF" ]; then
    SZ=$(stat -c%s "$NATIVE_ELF")
    if [ "$SZ" -gt 4000 ] && [ "$SZ" -lt 30000 ]; then
        check "binary_size" 0
        echo "    size: $SZ bytes"
    else
        check "binary_size" 1
        echo "    size: $SZ bytes (expected 4000-30000)"
    fi
else
    skip "binary_size"
fi

# --- Summary ---
echo ""
echo "=== Results ==="
TOTAL=$((PASS + FAIL + NOT_RUN))
echo "  PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"

# Cleanup
rm -rf "$TMPDIR_GATE"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
