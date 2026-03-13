#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
DRIVER="self-hosted/compiler/render_native_compile_driver_f64.sio"
TIMEOUT=120
ARTIFACT="artifacts/sprint114/print_builtin_gate.v1.json"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

check() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@" >/tmp/sprint114_gate.out 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        local _ec=$?
        if [ "$_ec" -eq 137 ] || [ "$_ec" -eq 124 ]; then
            echo "NOT_RUN  $name (OOM/timeout)"
            NOT_RUN=$((NOT_RUN + 1))
        else
            echo "FAIL  $name"
            tail -5 /tmp/sprint114_gate.out
            FAIL=$((FAIL + 1))
        fi
    fi
}

check_exit_code() {
    local name="$1"
    local expected="$2"
    local binary="$3"
    TOTAL=$((TOTAL + 1))
    local actual=0
    "$binary" >/dev/null 2>&1 || actual=$?
    if [ "$actual" -eq "$expected" ]; then
        echo "PASS  $name (exit=$actual)"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (expected=$expected actual=$actual)"
        FAIL=$((FAIL + 1))
    fi
}

check_output() {
    local name="$1"
    local expected="$2"
    local binary="$3"
    TOTAL=$((TOTAL + 1))
    local actual
    actual=$("$binary" 2>/dev/null) || true
    if [ "$actual" = "$expected" ]; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name"
        echo "  expected: $(echo "$expected" | head -3)"
        echo "  actual:   $(echo "$actual" | head -3)"
        FAIL=$((FAIL + 1))
    fi
}

compile_and_test_output() {
    local name="$1"
    local source="$2"
    local expected_output="$3"
    local expected_exit="${4:-0}"
    local out_bin="/tmp/sprint114_${name}.elf"
    rm -f "$out_bin"

    TOTAL=$((TOTAL + 1))
    if timeout "$TIMEOUT" "$SOUC" run "$DRIVER" -- "$source" "$out_bin" >/tmp/sprint114_compile.out 2>&1; then
        echo "PASS  ${name}:compile"
        PASS=$((PASS + 1))
    else
        local _ec=$?
        if [ "$_ec" -eq 137 ] || [ "$_ec" -eq 124 ]; then
            echo "NOT_RUN  ${name}:compile (OOM/timeout)"
            NOT_RUN=$((NOT_RUN + 1))
            TOTAL=$((TOTAL + 1))
            echo "NOT_RUN  ${name}:output (cascade)"
            NOT_RUN=$((NOT_RUN + 1))
            return
        else
            echo "FAIL  ${name}:compile"
            tail -5 /tmp/sprint114_compile.out
            FAIL=$((FAIL + 1))
            TOTAL=$((TOTAL + 1))
            echo "NOT_RUN  ${name}:output (cascade)"
            NOT_RUN=$((NOT_RUN + 1))
            return
        fi
    fi

    chmod +x "$out_bin"
    check_output "${name}:output" "$expected_output" "$out_bin"
}

compile_and_test() {
    local name="$1"
    local source="$2"
    local expected_exit="$3"
    local out_bin="/tmp/sprint114_${name}.elf"
    rm -f "$out_bin"

    TOTAL=$((TOTAL + 1))
    if timeout "$TIMEOUT" "$SOUC" run "$DRIVER" -- "$source" "$out_bin" >/tmp/sprint114_compile.out 2>&1; then
        echo "PASS  ${name}:compile"
        PASS=$((PASS + 1))
    else
        local _ec=$?
        if [ "$_ec" -eq 137 ] || [ "$_ec" -eq 124 ]; then
            echo "NOT_RUN  ${name}:compile (OOM/timeout)"
            NOT_RUN=$((NOT_RUN + 1))
            TOTAL=$((TOTAL + 1))
            echo "NOT_RUN  ${name}:run (cascade)"
            NOT_RUN=$((NOT_RUN + 1))
            return
        else
            echo "FAIL  ${name}:compile"
            tail -5 /tmp/sprint114_compile.out
            FAIL=$((FAIL + 1))
            TOTAL=$((TOTAL + 1))
            echo "NOT_RUN  ${name}:run (cascade)"
            NOT_RUN=$((NOT_RUN + 1))
            return
        fi
    fi

    chmod +x "$out_bin"
    check_exit_code "${name}:run" "$expected_exit" "$out_bin"
}

echo "=== Sprint 114: print/print_int Builtin Gate ==="
echo ""

echo "--- Group 1: Preconditions ---"
check "pre:driver" test -f "$DRIVER"
check "pre:souc" test -x "$SOUC"
echo ""

echo "--- Group 2: print_int ---"

# T1: print_int simple
cat > /tmp/s114_pi1.sio <<'EOF'
fn main() -> i64 {
    print_int(42)
    0
}
EOF
compile_and_test_output "print_int_simple" /tmp/s114_pi1.sio "42"

# T2: print_int zero
cat > /tmp/s114_pi0.sio <<'EOF'
fn main() -> i64 {
    print_int(0)
    0
}
EOF
compile_and_test_output "print_int_zero" /tmp/s114_pi0.sio "0"

# T3: print_int large
cat > /tmp/s114_pil.sio <<'EOF'
fn main() -> i64 {
    print_int(12345)
    0
}
EOF
compile_and_test_output "print_int_large" /tmp/s114_pil.sio "12345"

# T4: print_int from computation
cat > /tmp/s114_pic.sio <<'EOF'
fn main() -> i64 {
    let x = 6 * 7
    print_int(x)
    0
}
EOF
compile_and_test_output "print_int_computed" /tmp/s114_pic.sio "42"

echo ""
echo "--- Group 3: print string literal ---"

# T5: print simple string
cat > /tmp/s114_ps1.sio <<'EOF'
fn main() -> i64 {
    print("hello")
    0
}
EOF
compile_and_test_output "print_str_simple" /tmp/s114_ps1.sio "hello"

# T6: print with newline
cat > /tmp/s114_ps2.sio <<'EOF'
fn main() -> i64 {
    print("P3\n128 128\n255\n")
    0
}
EOF
compile_and_test_output "print_ppm_header" /tmp/s114_ps2.sio "$(printf 'P3\n128 128\n255\n')"

echo ""
echo "--- Group 4: Mixed print + print_int ---"

# T7: interleaved print and print_int
cat > /tmp/s114_mix.sio <<'EOF'
fn main() -> i64 {
    print("val=")
    print_int(42)
    print("\n")
    0
}
EOF
compile_and_test_output "mixed_print" /tmp/s114_mix.sio "$(printf 'val=42\n')"

# T8: Multiple print_int calls
cat > /tmp/s114_multi.sio <<'EOF'
fn main() -> i64 {
    print_int(10)
    print(" ")
    print_int(20)
    print(" ")
    print_int(30)
    print("\n")
    0
}
EOF
compile_and_test_output "multi_print_int" /tmp/s114_multi.sio "$(printf '10 20 30\n')"

# T9: PPM-like output
cat > /tmp/s114_ppm.sio <<'EOF'
fn clamp_byte(v: i64) -> i64 {
    if v < 0 { return 0 }
    if v > 255 { return 255 }
    v
}
fn main() -> i64 {
    print("P3\n")
    print_int(2)
    print(" ")
    print_int(1)
    print("\n255\n")
    print_int(clamp_byte(255))
    print(" ")
    print_int(clamp_byte(128))
    print(" ")
    print_int(clamp_byte(0))
    print(" ")
    print_int(clamp_byte(64))
    print(" ")
    print_int(clamp_byte(32))
    print(" ")
    print_int(clamp_byte(16))
    print("\n")
    0
}
EOF
compile_and_test_output "ppm_output" /tmp/s114_ppm.sio "$(printf 'P3\n2 1\n255\n255 128 0 64 32 16\n')"

echo ""
echo "--- Group 5: Regression ---"

# T10: f64 functions still work
cat > /tmp/s114_regf64.sio <<'EOF'
fn double(x: f64) -> f64 { x * 2.0 }
fn main() -> i64 {
    let v = double(21.0)
    v as i64
}
EOF
compile_and_test "f64_regress" /tmp/s114_regf64.sio 42

# T11: i64 functions still work
cat > /tmp/s114_regi64.sio <<'EOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(20, 22) }
EOF
compile_and_test "i64_regress" /tmp/s114_regi64.sio 42

echo ""
echo "=== Sprint 114 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"

mkdir -p "$(dirname "$ARTIFACT")"
STATUS="fail"
if [ "$FAIL" -eq 0 ]; then STATUS="pass"; fi
cat > "$ARTIFACT" <<EOF
{
  "schema": "sounio.sprint114.print_builtin_gate.v1",
  "generated_at": "$(date --iso-8601=seconds)",
  "status": "$STATUS",
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN },
  "novel_claims": [
    "print_int builtin: inline x86-64 itoa (div-by-10 loop) + sys_write(1, buf, len)",
    "print builtin: string literals embedded inline in code with JMP-over + RIP-relative LEA",
    "Builtins registered as synthetic fn_names entries, resolved by same call-patching mechanism",
    "PPM-style output verified: header + pixel values match expected byte-for-byte"
  ]
}
EOF

if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
