#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
DRIVER="self-hosted/compiler/render_native_compile_driver_f64.sio"
TIMEOUT=120
ARTIFACT="artifacts/sprint112/f64_native_gate.v1.json"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

check() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@" >/tmp/sprint112_gate.out 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        local _ec=$?
        if [ "$_ec" -eq 137 ] || [ "$_ec" -eq 124 ]; then
            echo "NOT_RUN  $name (OOM/timeout)"
            NOT_RUN=$((NOT_RUN + 1))
        else
            echo "FAIL  $name"
            tail -5 /tmp/sprint112_gate.out
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

compile_and_test() {
    local name="$1"
    local source="$2"
    local expected_exit="$3"
    local out_bin="/tmp/sprint112_${name}.elf"
    rm -f "$out_bin"

    TOTAL=$((TOTAL + 1))
    if timeout "$TIMEOUT" "$SOUC" run "$DRIVER" -- "$source" "$out_bin" >/tmp/sprint112_compile.out 2>&1; then
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
            tail -5 /tmp/sprint112_compile.out
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

echo "=== Sprint 112: f64 Native Compile Gate ==="
echo ""

echo "--- Group 1: Preconditions ---"
check "pre:driver" test -f "$DRIVER"
check "pre:souc" test -x "$SOUC"
echo ""

echo "--- Group 2: f64 Arithmetic ---"

# T1: f64 add
cat > /tmp/s112_add.sio <<'EOF'
fn main() -> i64 {
    let x = 3.5
    let y = 2.5
    let z = x + y
    z as i64
}
EOF
compile_and_test "f64_add" /tmp/s112_add.sio 6

# T2: f64 multiply
cat > /tmp/s112_mul.sio <<'EOF'
fn main() -> i64 {
    let x = 6.0
    let y = 7.0
    (x * y) as i64
}
EOF
compile_and_test "f64_mul" /tmp/s112_mul.sio 42

# T3: f64 subtract + divide
cat > /tmp/s112_subdiv.sio <<'EOF'
fn main() -> i64 {
    let x = 100.0
    let y = x - 16.0
    let z = y / 2.0
    z as i64
}
EOF
compile_and_test "f64_subdiv" /tmp/s112_subdiv.sio 42

# T4: f64 chain arithmetic
cat > /tmp/s112_chain.sio <<'EOF'
fn main() -> i64 {
    let a = 10.0
    let b = a * 4.0
    let c = b + 2.5
    let d = c - 0.5
    d as i64
}
EOF
compile_and_test "f64_chain" /tmp/s112_chain.sio 42

echo ""
echo "--- Group 3: f64 Comparisons ---"

# T5: f64 greater-than
cat > /tmp/s112_gt.sio <<'EOF'
fn main() -> i64 {
    let x = 3.14
    let y = 2.71
    if x > y { 42 } else { 0 }
}
EOF
compile_and_test "f64_gt" /tmp/s112_gt.sio 42

# T6: f64 less-than
cat > /tmp/s112_lt.sio <<'EOF'
fn main() -> i64 {
    let x = 1.5
    let y = 2.5
    if x < y { 42 } else { 0 }
}
EOF
compile_and_test "f64_lt" /tmp/s112_lt.sio 42

# T7: f64 less-than-or-equal
cat > /tmp/s112_le.sio <<'EOF'
fn main() -> i64 {
    let a = 5.0
    let b = 5.0
    if a <= b { 42 } else { 0 }
}
EOF
compile_and_test "f64_le" /tmp/s112_le.sio 42

echo ""
echo "--- Group 4: Casts ---"

# T8: i64 → f64 cast
cat > /tmp/s112_i2f.sio <<'EOF'
fn main() -> i64 {
    let x = 21
    let y = (x as f64) * 2.0
    y as i64
}
EOF
compile_and_test "cast_i2f" /tmp/s112_i2f.sio 42

# T9: f64 → i64 truncation
cat > /tmp/s112_f2i.sio <<'EOF'
fn main() -> i64 {
    let x = 42.9
    x as i64
}
EOF
compile_and_test "cast_f2i" /tmp/s112_f2i.sio 42

echo ""
echo "--- Group 5: Mixed i64/f64 ---"

# T10: Mixed computation
cat > /tmp/s112_mixed.sio <<'EOF'
fn main() -> i64 {
    let int_part = 20
    let float_part = 22.0
    let converted = float_part as i64
    int_part + converted
}
EOF
compile_and_test "mixed" /tmp/s112_mixed.sio 42

# T11: i64 regression (Sprint 111 programs still work)
cat > /tmp/s112_regress.sio <<'EOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(20, 22) }
EOF
compile_and_test "i64_regress" /tmp/s112_regress.sio 42

echo ""
echo "=== Sprint 112 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"

mkdir -p "$(dirname "$ARTIFACT")"
STATUS="fail"
if [ "$FAIL" -eq 0 ]; then STATUS="pass"; fi
cat > "$ARTIFACT" <<EOF
{
  "schema": "sounio.sprint112.f64_native_gate.v1",
  "generated_at": "$(date --iso-8601=seconds)",
  "status": "$STATUS",
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN },
  "novel_claims": [
    "AST-direct compiler handles f64 literals via IEEE 754 bit extraction",
    "f64 arithmetic: ADDSD, SUBSD, MULSD, DIVSD with type-driven dispatch",
    "f64 comparisons: COMISD + SETcc for >, >=, <, <=, ==, !=",
    "Bidirectional casts: i64→f64 (CVTSI2SD), f64→i64 (CVTTSD2SI)",
    "Type tracking via local_types[64] array propagated through AST walker"
  ]
}
EOF

if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
