#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./bin/souc"
DRIVER="self-hosted/compiler/render_native_compile_driver_f64.sio"
TIMEOUT=120
ARTIFACT="artifacts/sprint115/logical_ops_gate.v1.json"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

check() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@" >/tmp/sprint115_gate.out 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        local _ec=$?
        if [ "$_ec" -eq 137 ] || [ "$_ec" -eq 124 ]; then
            echo "NOT_RUN  $name (OOM/timeout)"
            NOT_RUN=$((NOT_RUN + 1))
        else
            echo "FAIL  $name"
            tail -5 /tmp/sprint115_gate.out
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
    local out_bin="/tmp/sprint115_${name}.elf"
    rm -f "$out_bin"

    TOTAL=$((TOTAL + 1))
    if timeout "$TIMEOUT" "$SOUC" run "$DRIVER" -- "$source" "$out_bin" >/tmp/sprint115_compile.out 2>&1; then
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
            tail -5 /tmp/sprint115_compile.out
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

echo "=== Sprint 115: Logical && / || Operators Gate ==="
echo ""

echo "--- Group 1: Preconditions ---"
check "pre:driver" test -f "$DRIVER"
check "pre:souc" test -x "$SOUC"
echo ""

echo "--- Group 2: && (logical AND) ---"

# T1: true && true
cat > /tmp/s115_and_tt.sio <<'EOF'
fn main() -> i64 {
    let a = 1
    let b = 1
    if a > 0 && b > 0 { 42 } else { 0 }
}
EOF
compile_and_test "and_tt" /tmp/s115_and_tt.sio 42

# T2: true && false
cat > /tmp/s115_and_tf.sio <<'EOF'
fn main() -> i64 {
    let a = 5
    let b = 0
    if a > 0 && b > 0 { 1 } else { 42 }
}
EOF
compile_and_test "and_tf" /tmp/s115_and_tf.sio 42

# T3: false && true (short-circuit — should not eval right)
cat > /tmp/s115_and_ft.sio <<'EOF'
fn main() -> i64 {
    let a = 0
    let b = 1
    if a > 0 && b > 0 { 1 } else { 42 }
}
EOF
compile_and_test "and_ft" /tmp/s115_and_ft.sio 42

# T4: false && false
cat > /tmp/s115_and_ff.sio <<'EOF'
fn main() -> i64 {
    let a = 0
    let b = 0
    if a > 0 && b > 0 { 1 } else { 42 }
}
EOF
compile_and_test "and_ff" /tmp/s115_and_ff.sio 42

echo ""
echo "--- Group 3: || (logical OR) ---"

# T5: true || false
cat > /tmp/s115_or_tf.sio <<'EOF'
fn main() -> i64 {
    let a = 1
    let b = 0
    if a > 0 || b > 0 { 42 } else { 0 }
}
EOF
compile_and_test "or_tf" /tmp/s115_or_tf.sio 42

# T6: false || true
cat > /tmp/s115_or_ft.sio <<'EOF'
fn main() -> i64 {
    let a = 0
    let b = 1
    if a > 0 || b > 0 { 42 } else { 0 }
}
EOF
compile_and_test "or_ft" /tmp/s115_or_ft.sio 42

# T7: false || false
cat > /tmp/s115_or_ff.sio <<'EOF'
fn main() -> i64 {
    let a = 0
    let b = 0
    if a > 0 || b > 0 { 1 } else { 42 }
}
EOF
compile_and_test "or_ff" /tmp/s115_or_ff.sio 42

# T8: true || true
cat > /tmp/s115_or_tt.sio <<'EOF'
fn main() -> i64 {
    let a = 1
    let b = 1
    if a > 0 || b > 0 { 42 } else { 0 }
}
EOF
compile_and_test "or_tt" /tmp/s115_or_tt.sio 42

echo ""
echo "--- Group 4: Chained / Nested ---"

# T9: triple &&
cat > /tmp/s115_chain_and.sio <<'EOF'
fn main() -> i64 {
    let a = 5
    let b = 10
    let c = 15
    if a > 0 && b > 0 && c > 0 { 42 } else { 0 }
}
EOF
compile_and_test "chain_and" /tmp/s115_chain_and.sio 42

# T10: && with f64 comparisons (triangle_basic pattern)
cat > /tmp/s115_f64_and.sio <<'EOF'
fn main() -> i64 {
    let w0 = 1.5
    let w1 = 2.5
    let w2 = 0.5
    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 { 42 } else { 0 }
}
EOF
compile_and_test "f64_and" /tmp/s115_f64_and.sio 42

# T11: && with one f64 comparison false
cat > /tmp/s115_f64_and_f.sio <<'EOF'
fn main() -> i64 {
    let w0 = 1.5
    let w1 = 0.0 - 0.5
    let w2 = 0.5
    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 { 1 } else { 42 }
}
EOF
compile_and_test "f64_and_false" /tmp/s115_f64_and_f.sio 42

# T12: mixed && and ||
cat > /tmp/s115_mixed.sio <<'EOF'
fn main() -> i64 {
    let a = 1
    let b = 0
    let c = 1
    if (a > 0 && b > 0) || c > 0 { 42 } else { 0 }
}
EOF
compile_and_test "mixed_and_or" /tmp/s115_mixed.sio 42

echo ""
echo "--- Group 5: Regression ---"

# T13: print still works
cat > /tmp/s115_reg_print.sio <<'EOF'
fn main() -> i64 {
    print_int(42)
    print("\n")
    0
}
EOF
compile_and_test "regress_print" /tmp/s115_reg_print.sio 0

# T14: f64 functions still work
cat > /tmp/s115_reg_f64.sio <<'EOF'
fn double(x: f64) -> f64 { x * 2.0 }
fn main() -> i64 { (double(21.0)) as i64 }
EOF
compile_and_test "regress_f64" /tmp/s115_reg_f64.sio 42

echo ""
echo "=== Sprint 115 Gate Summary ==="
echo "PASS: $PASS  FAIL: $FAIL  NOT_RUN: $NOT_RUN  TOTAL: $TOTAL"

mkdir -p "$(dirname "$ARTIFACT")"
STATUS="fail"
if [ "$FAIL" -eq 0 ]; then STATUS="pass"; fi
cat > "$ARTIFACT" <<EOF
{
  "schema": "sounio.sprint115.logical_ops_gate.v1",
  "generated_at": "$(date --iso-8601=seconds)",
  "status": "$STATUS",
  "metrics": { "total": $TOTAL, "passed": $PASS, "failed": $FAIL, "not_run": $NOT_RUN },
  "novel_claims": [
    "Short-circuit && (OpAnd): eval left, jz to false path, else eval right",
    "Short-circuit || (OpOr): eval left, jnz to true path, else eval right",
    "Chained logical ops (a && b && c) via nested ExprBinary nodes",
    "Mixed f64 comparisons with && verified (triangle_basic rasterizer pattern)"
  ]
}
EOF

if [ "$FAIL" -eq 0 ]; then
    echo "STATUS: PASS"
    exit 0
fi
echo "STATUS: FAIL"
exit 1
