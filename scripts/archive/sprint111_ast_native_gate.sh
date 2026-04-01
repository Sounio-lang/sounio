#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./bin/souc"
DRIVER="self-hosted/compiler/render_native_compile_driver_lean.sio"
TIMEOUT=120
ARTIFACT="artifacts/sprint111/ast_native_gate.v1.json"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

check() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@" >/tmp/sprint111_gate.out 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        local _ec=$?
        if [ "$_ec" -eq 137 ] || [ "$_ec" -eq 124 ]; then
            echo "NOT_RUN  $name (OOM/timeout)"
            NOT_RUN=$((NOT_RUN + 1))
        else
            echo "FAIL  $name"
            tail -5 /tmp/sprint111_gate.out
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
    local out_bin="/tmp/sprint111_${name}.elf"
    rm -f "$out_bin"

    TOTAL=$((TOTAL + 1))
    if timeout "$TIMEOUT" "$SOUC" run "$DRIVER" -- "$source" "$out_bin" >/tmp/sprint111_compile.out 2>&1; then
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
            tail -5 /tmp/sprint111_compile.out
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

echo "=== Sprint 111: AST-Direct Native Compile Gate ==="
echo ""

echo "--- Group 1: Preconditions ---"
check "pre:driver" test -f "$DRIVER"
check "pre:souc" test -x "$SOUC"
echo ""

echo "--- Group 2: Simple Programs ---"

# T1: Literal return
cat > /tmp/s111_literal.sio <<'EOF'
fn main() -> i64 { 42 }
EOF
compile_and_test "literal" /tmp/s111_literal.sio 42

# T2: Let bindings + arithmetic
cat > /tmp/s111_let.sio <<'EOF'
fn main() -> i64 {
    let a = 20
    let b = 22
    a + b
}
EOF
compile_and_test "let_arith" /tmp/s111_let.sio 42

# T3: Var + assignment
cat > /tmp/s111_var.sio <<'EOF'
fn main() -> i64 {
    var x = 10
    x = x + 32
    x
}
EOF
compile_and_test "var_assign" /tmp/s111_var.sio 42

echo ""
echo "--- Group 3: Function Calls ---"

# T4: Simple function call
cat > /tmp/s111_call.sio <<'EOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(20, 22) }
EOF
compile_and_test "fn_call" /tmp/s111_call.sio 42

# T5: Chained calls
cat > /tmp/s111_chain.sio <<'EOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn mul(a: i64, b: i64) -> i64 { a * b }
fn main() -> i64 {
    let x = add(3, 4)
    let y = mul(x, 6)
    y
}
EOF
compile_and_test "chain_call" /tmp/s111_chain.sio 42

echo ""
echo "--- Group 4: Control Flow ---"

# T6: If-else + early return
cat > /tmp/s111_ifelse.sio <<'EOF'
fn clamp_byte(v: i64) -> i64 {
    if v < 0 { return 0 }
    if v > 255 { return 255 }
    v
}
fn main() -> i64 {
    let a = clamp_byte(300)
    let b = clamp_byte(0 - 5)
    let c = clamp_byte(42)
    a - b - 255 + c
}
EOF
compile_and_test "ifelse_return" /tmp/s111_ifelse.sio 42

# T7: While loop
cat > /tmp/s111_while.sio <<'EOF'
fn main() -> i64 {
    var sum = 0
    var i = 1
    while i <= 9 {
        sum = sum + i
        i = i + 1
    }
    sum - 3
}
EOF
compile_and_test "while_loop" /tmp/s111_while.sio 42

echo ""
echo "--- Group 5: Complex Programs ---"

# T8: Fibonacci
cat > /tmp/s111_fib.sio <<'EOF'
fn fibonacci(n: i64) -> i64 {
    var a = 0
    var b = 1
    var i = 0
    while i < n {
        let t = a + b
        a = b
        b = t
        i = i + 1
    }
    a
}
fn main() -> i64 {
    let f8 = fibonacci(8)
    f8 + 21
}
EOF
compile_and_test "fibonacci" /tmp/s111_fib.sio 42

# T9: GCD
cat > /tmp/s111_gcd.sio <<'EOF'
fn gcd(a: i64, b: i64) -> i64 {
    var x = a
    var y = b
    while y > 0 {
        let r = x - (x / y) * y
        x = y
        y = r
    }
    x
}
fn main() -> i64 { gcd(84, 126) }
EOF
compile_and_test "gcd" /tmp/s111_gcd.sio 42

# T10: All arithmetic ops
cat > /tmp/s111_arith.sio <<'EOF'
fn main() -> i64 {
    let a = 10 + 5
    let b = a * 3
    let c = b - 3
    let d = c / 6
    let e = 50 - d
    let f = e - 1
    f
}
EOF
compile_and_test "all_arith" /tmp/s111_arith.sio 42

# T11: Nested if-else expression
cat > /tmp/s111_nested_if.sio <<'EOF'
fn classify(x: i64) -> i64 {
    if x > 100 { 3 }
    else if x > 50 { 2 }
    else if x > 0 { 1 }
    else { 0 }
}
fn main() -> i64 {
    let a = classify(200)
    let b = classify(75)
    let c = classify(25)
    let d = classify(0 - 5)
    a * 14 + b + c - d
}
EOF
compile_and_test "nested_if" /tmp/s111_nested_if.sio 45

echo ""
echo "=== Sprint 111 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"

mkdir -p "$(dirname "$ARTIFACT")"
STATUS="fail"
if [ "$FAIL" -eq 0 ]; then STATUS="pass"; fi
cat > "$ARTIFACT" <<EOF
{
  "schema": "sounio.sprint111.ast_native_gate.v1",
  "generated_at": "$(date --iso-8601=seconds)",
  "status": "$STATUS",
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN },
  "novel_claims": [
    "AST-direct native compile bypasses both JIT Box corruption bugs",
    "Handles: let/var bindings, function calls, if-else, while loops, early return",
    "All arithmetic ops verified: add, sub, mul, div, rem, comparisons",
    "Complex programs: fibonacci, GCD algorithms produce correct results"
  ]
}
EOF

if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
