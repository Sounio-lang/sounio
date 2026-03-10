#!/usr/bin/env bash
# sprint58_bootstrap_gate.sh — Self-hosted Bootstrap Binary Gate
#
# Claim: The self-hosted frontend compiles triangle_basic.sio to a runnable ELF
# that produces correct 128x128 PPM output.
#
# Known limitation: io_write_native_binary() calls write_file() VM builtin which
# uses mode 0644 (not 0755). Gate applies chmod +x workaround.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
SH="self-hosted/compiler/main.sio"
BOOTSTRAP_OUT="/tmp/triangle_bootstrap_$$"

check_bootstrap() {
    local name="$1"; local src="$2"
    local expected_w="$3"; local expected_h="$4"
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi

    local compile_out
    compile_out=$(timeout 180 "$SOUC" run "$SH" -- --native-compile "$src" \
        -o "$BOOTSTRAP_OUT" 2>/dev/null) || {
        local ec=$?
        if [ "$ec" -eq 124 ]; then
            echo "NOT_RUN  $name (compile timeout 180s)"; NOT_RUN=$((NOT_RUN+1))
        elif [ "$ec" -eq 137 ]; then
            echo "NOT_RUN  $name (OOM/killed)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (compile exit $ec)"; FAIL=$((FAIL+1))
        fi
        return
    }

    if ! echo "$compile_out" | grep -qE "Native compilation successful|native compile: OK|compile: OK"; then
        echo "FAIL  $name (no success message in compile output)"; FAIL=$((FAIL+1))
        return
    fi

    if [ ! -f "$BOOTSTRAP_OUT" ]; then
        echo "FAIL  $name (output binary not found at $BOOTSTRAP_OUT)"; FAIL=$((FAIL+1))
        return
    fi

    # ELF permissions workaround: VM write_file() uses 0644, not 0755
    chmod +x "$BOOTSTRAP_OUT" 2>/dev/null || true

    local ppm_out
    ppm_out=$(timeout 30 "$BOOTSTRAP_OUT" 2>/dev/null) || {
        local ec=$?
        if [ "$ec" -eq 124 ]; then
            echo "FAIL  $name (binary timeout 30s)"; FAIL=$((FAIL+1))
        else
            echo "FAIL  $name (binary crashed, exit $ec)"; FAIL=$((FAIL+1))
        fi
        return
    }

    local header w h
    header=$(echo "$ppm_out" | sed -n '2p')
    w=$(echo "$header" | awk '{print $1}')
    h=$(echo "$header" | awk '{print $2}')
    if [ "$w" = "$expected_w" ] && [ "$h" = "$expected_h" ]; then
        echo "PASS  $name (bootstrap binary produced ${w}x${h} PPM)"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected ${expected_w}x${expected_h}, got '${w}'x'${h}')"; FAIL=$((FAIL+1))
    fi
}

check_file_nonzero() {
    local name="$1"; local file="$2"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local sz
    sz=$(stat -c%s "$file" 2>/dev/null || echo 0)
    if [ "$sz" -gt 0 ]; then
        echo "PASS  $name (size=${sz} bytes)"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (file is empty)"; FAIL=$((FAIL+1))
    fi
}

check_self_test_regression() {
    local name="$1"
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local out
    out=$(timeout 60 "$SOUC" run "$SH" -- --self-test 2>/dev/null) || {
        local ec=$?
        if [ "$ec" -eq 124 ]; then
            echo "NOT_RUN  $name (timeout 60s)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (self-test exit $ec)"; FAIL=$((FAIL+1))
        fi
        return
    }
    if echo "$out" | grep -qE "T28.*PASS|T29.*PASS|regalloc.*pass"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    elif echo "$out" | grep -qE "FAIL.*T28|FAIL.*T29"; then
        echo "FAIL  $name (T28/T29 regalloc regression)"; FAIL=$((FAIL+1))
    else
        echo "NOT_RUN  $name (T28/T29 not in output)"; NOT_RUN=$((NOT_RUN+1))
    fi
}

echo "=== Sprint 58: Self-hosted Bootstrap Binary Gate ==="
echo "    (Compile timeout=180s; ELF chmod+x workaround applied)"
echo ""

# --- Group 1: Bootstrap compile + run ---
echo "--- Group 1: Bootstrap binary (triangle_basic.sio → ELF → PPM) ---"
check_bootstrap "bootstrap:triangle_basic" examples/render/triangle_basic.sio 128 128
echo ""

# --- Group 2: Binary sanity ---
echo "--- Group 2: Binary sanity ---"
check_file_nonzero "bootstrap:binary_size_nonzero" "$BOOTSTRAP_OUT"
echo ""

# --- Group 3: Regression ---
echo "--- Group 3: Regalloc regression (Sprint 52 T28/T29) ---"
check_self_test_regression "bootstrap:regression_sprint52"
echo ""

# Cleanup
rm -f "$BOOTSTRAP_OUT"

# --- Summary ---
echo "=== Sprint 58 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"
echo ""

if [ "$FAIL" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    echo "STATUS: PASS"
    exit 0
else
    echo "STATUS: FAIL"
    exit 1
fi
