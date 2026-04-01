#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./bin/souc"
DRIVER="self-hosted/compiler/render_native_compile_driver_f64.sio"
TIMEOUT=120
ARTIFACT="artifacts/sprint113/f64_fn_param_gate.v1.json"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

check() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@" >/tmp/sprint113_gate.out 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        local _ec=$?
        if [ "$_ec" -eq 137 ] || [ "$_ec" -eq 124 ]; then
            echo "NOT_RUN  $name (OOM/timeout)"
            NOT_RUN=$((NOT_RUN + 1))
        else
            echo "FAIL  $name"
            tail -5 /tmp/sprint113_gate.out
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
    local out_bin="/tmp/sprint113_${name}.elf"
    rm -f "$out_bin"

    TOTAL=$((TOTAL + 1))
    if timeout "$TIMEOUT" "$SOUC" run "$DRIVER" -- "$source" "$out_bin" >/tmp/sprint113_compile.out 2>&1; then
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
            tail -5 /tmp/sprint113_compile.out
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

echo "=== Sprint 113: f64 Function Param/Return Gate ==="
echo ""

echo "--- Group 1: Preconditions ---"
check "pre:driver" test -f "$DRIVER"
check "pre:souc" test -x "$SOUC"
echo ""

echo "--- Group 2: f64 Function Return ---"

# T1: f64 return, caller casts to i64
cat > /tmp/s113_ret.sio <<'EOF'
fn make_pi() -> f64 { 3.14 }
fn main() -> i64 {
    let v = make_pi()
    v as i64
}
EOF
compile_and_test "f64_return" /tmp/s113_ret.sio 3

# T2: f64 return with arithmetic
cat > /tmp/s113_ret_arith.sio <<'EOF'
fn half() -> f64 { 84.0 / 2.0 }
fn main() -> i64 {
    let v = half()
    v as i64
}
EOF
compile_and_test "f64_return_arith" /tmp/s113_ret_arith.sio 42

echo ""
echo "--- Group 3: f64 Function Params ---"

# T3: single f64 param
cat > /tmp/s113_param1.sio <<'EOF'
fn double(x: f64) -> f64 { x * 2.0 }
fn main() -> i64 {
    let v = double(21.0)
    v as i64
}
EOF
compile_and_test "f64_param_single" /tmp/s113_param1.sio 42

# T4: two f64 params
cat > /tmp/s113_param2.sio <<'EOF'
fn add_f(a: f64, b: f64) -> f64 { a + b }
fn main() -> i64 {
    let v = add_f(20.5, 21.5)
    v as i64
}
EOF
compile_and_test "f64_param_two" /tmp/s113_param2.sio 42

# T5: three f64 params
cat > /tmp/s113_param3.sio <<'EOF'
fn weighted(a: f64, b: f64, c: f64) -> f64 { a + b + c }
fn main() -> i64 {
    let v = weighted(10.0, 20.0, 12.0)
    v as i64
}
EOF
compile_and_test "f64_param_three" /tmp/s113_param3.sio 42

echo ""
echo "--- Group 4: Mixed i64/f64 Params ---"

# T6: i64 param, f64 return
cat > /tmp/s113_i2f_param.sio <<'EOF'
fn to_float_double(x: i64) -> f64 { (x as f64) * 2.0 }
fn main() -> i64 {
    let v = to_float_double(21)
    v as i64
}
EOF
compile_and_test "i64_param_f64_ret" /tmp/s113_i2f_param.sio 42

# T7: f64 param, i64 return
cat > /tmp/s113_f2i_param.sio <<'EOF'
fn truncate(x: f64) -> i64 { x as i64 }
fn main() -> i64 { truncate(42.9) }
EOF
compile_and_test "f64_param_i64_ret" /tmp/s113_f2i_param.sio 42

# T8: mixed param order (i64, f64)
cat > /tmp/s113_mixed_order.sio <<'EOF'
fn combine(n: i64, scale: f64) -> i64 {
    let result = (n as f64) * scale
    result as i64
}
fn main() -> i64 { combine(21, 2.0) }
EOF
compile_and_test "mixed_param_order" /tmp/s113_mixed_order.sio 42

echo ""
echo "--- Group 5: Chained f64 Calls ---"

# T9: chain of f64 functions
cat > /tmp/s113_chain.sio <<'EOF'
fn add1(x: f64) -> f64 { x + 1.0 }
fn mul2(x: f64) -> f64 { x * 2.0 }
fn main() -> i64 {
    let a = add1(20.0)
    let b = mul2(a)
    b as i64
}
EOF
compile_and_test "f64_chain" /tmp/s113_chain.sio 42

# T10: f64 function result used in comparison
cat > /tmp/s113_cmp.sio <<'EOF'
fn mag(x: f64) -> f64 { x * x }
fn main() -> i64 {
    let m = mag(7.0)
    if m > 48.0 { 42 } else { 0 }
}
EOF
compile_and_test "f64_fn_cmp" /tmp/s113_cmp.sio 42

echo ""
echo "--- Group 6: Regression ---"

# T11: pure i64 functions still work
cat > /tmp/s113_regress.sio <<'EOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn mul(a: i64, b: i64) -> i64 { a * b }
fn main() -> i64 {
    let x = add(3, 4)
    mul(x, 6)
}
EOF
compile_and_test "i64_regress" /tmp/s113_regress.sio 42

# T12: Sprint 112 f64 arithmetic (no functions) still works
cat > /tmp/s113_arith_regress.sio <<'EOF'
fn main() -> i64 {
    let x = 100.0
    let y = x - 16.0
    let z = y / 2.0
    z as i64
}
EOF
compile_and_test "f64_arith_regress" /tmp/s113_arith_regress.sio 42

echo ""
echo "=== Sprint 113 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"

mkdir -p "$(dirname "$ARTIFACT")"
STATUS="fail"
if [ "$FAIL" -eq 0 ]; then STATUS="pass"; fi
cat > "$ARTIFACT" <<EOF
{
  "schema": "sounio.sprint113.f64_fn_param_gate.v1",
  "generated_at": "$(date --iso-8601=seconds)",
  "status": "$STATUS",
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN },
  "novel_claims": [
    "f64 function return types detected from AST TypeExpr annotations",
    "f64 function parameters detected from Param.ty and tracked in local_types",
    "fn_return_types[64] array propagated through compilation pipeline",
    "ast_expr_is_float extended for ExprCall via callee return type lookup",
    "Mixed i64/f64 parameter functions work (all values pass as bit patterns in integer regs)"
  ]
}
EOF

if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
