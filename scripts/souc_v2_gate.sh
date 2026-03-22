#!/bin/bash
# souc_v2_gate.sh — Gate test for lean_single.sio / souc_v2 bootstrap compiler
# Runs bootstrap fixed-point, souc_v2 split fixed-point, feature tests, regression.
set -e
PASS=0; FAIL=0; TOTAL=0
cd "$(dirname "$0")/.."
BOOT4=/tmp/final_boot4.elf

if [ ! -x "$BOOT4" ]; then
    echo "ERROR: $BOOT4 not found or not executable"
    exit 1
fi

echo "=== souc-v2 Gate ==="
echo ""

# --- Bootstrap fixed-point ---
echo "--- Bootstrap (boot4 -> s1 -> s2 -> s3) ---"
$BOOT4 self-hosted/compiler/lean_single.sio /tmp/gate_s1.elf 2>&1 | tail -1
chmod +x /tmp/gate_s1.elf
/tmp/gate_s1.elf self-hosted/compiler/lean_single.sio /tmp/gate_s2.elf 2>&1 | tail -1
chmod +x /tmp/gate_s2.elf
/tmp/gate_s2.elf self-hosted/compiler/lean_single.sio /tmp/gate_s3.elf 2>&1 | tail -1
TOTAL=$((TOTAL+1))
if cmp -s /tmp/gate_s2.elf /tmp/gate_s3.elf; then
    echo "PASS: Fixed-point (s2==s3)"
    PASS=$((PASS+1))
else
    echo "FAIL: Fixed-point (s2!=s3)"
    FAIL=$((FAIL+1))
fi

S=/tmp/gate_s2.elf

# --- souc_v2 split fixed-point ---
echo ""
echo "--- souc_v2 split ---"
TOTAL=$((TOTAL+1))
if $S self-hosted/compiler/souc_v2/main.sio /tmp/gate_split1.elf 2>&1 | tail -1 && \
   chmod +x /tmp/gate_split1.elf && \
   /tmp/gate_split1.elf self-hosted/compiler/souc_v2/main.sio /tmp/gate_split2.elf 2>&1 | tail -1 && \
   cmp -s /tmp/gate_split1.elf /tmp/gate_split2.elf; then
    echo "PASS: souc_v2 split fixed-point"
    PASS=$((PASS+1))
else
    echo "FAIL: souc_v2 split fixed-point"
    FAIL=$((FAIL+1))
fi

# --- Feature tests ---
echo ""
echo "--- Feature tests ---"

run_test() {
    local name=$1 src=$2 expected=$3
    TOTAL=$((TOTAL+1))
    if timeout 30 $S "$src" /tmp/gate_out.elf 2>/dev/null && \
       chmod +x /tmp/gate_out.elf 2>/dev/null; then
        local output
        output=$(timeout 10 /tmp/gate_out.elf 2>/dev/null) || true
        if echo "$output" | grep -q "$expected"; then
            echo "PASS: $name"
            PASS=$((PASS+1))
        else
            echo "FAIL: $name (got: $(echo "$output" | head -c 60))"
            FAIL=$((FAIL+1))
        fi
    else
        echo "FAIL: $name (compile error)"
        FAIL=$((FAIL+1))
    fi
}

# Test: complex_native_demo (regression)
run_test "complex_native_demo" examples/algorithms/complex_native_demo.sio "ALL PASS"

# Test: f64 infix arithmetic
cat > /tmp/gate_t1.sio << 'EOF'
fn main() -> i64 with IO { let p = 3.14 + 2.0; print_int(f64_to_i64(p)); print("\n"); return 0 }
EOF
run_test "f64_infix" /tmp/gate_t1.sio "5"

# Test: struct + impl method
cat > /tmp/gate_t2.sio << 'EOF'
struct P { x: i64, y: i64 }
impl P { fn sum(self: i64) -> i64 { return self.x + self.y } }
fn main() -> i64 with IO { let p = P { x: 3, y: 4 }; print_int(p.sum()); print("\n"); return 0 }
EOF
run_test "struct_impl" /tmp/gate_t2.sio "7"

# Test: closure (lambda)
cat > /tmp/gate_t3.sio << 'EOF'
fn main() -> i64 with IO { let f = |x| x * 3; print_int(f(7)); print("\n"); return 0 }
EOF
run_test "closure" /tmp/gate_t3.sio "21"

# Test: capturing closure (KNOWN: cross-function closure capture is limited)
cat > /tmp/gate_t4.sio << 'EOF'
fn main() -> i64 with IO { let n = 100; let a = |x| x + n; print_int(a(5)); print("\n"); return 0 }
EOF
run_test "capture_closure" /tmp/gate_t4.sio "105"

# Test: Option match (Some/None)
cat > /tmp/gate_t5.sio << 'EOF'
fn main() -> i64 with IO { let x = Some(42); match x { Some(v) => { print_int(v) } None => { print_int(0) } }; print("\n"); return 0 }
EOF
run_test "option_match" /tmp/gate_t5.sio "42"

# Test: generics (monomorphization)
cat > /tmp/gate_t6.sio << 'EOF'
fn id<T>(x: T) -> T { return x }
fn main() -> i64 with IO { print_int(id<i64>(99)); print("\n"); return 0 }
EOF
run_test "generics" /tmp/gate_t6.sio "99"

# Test: print_f64
cat > /tmp/gate_t7.sio << 'EOF'
fn main() -> i64 with IO { print_f64(3.14); print("\n"); print_f64(-0.5); print("\n"); print("PASS\n"); return 0 }
EOF
run_test "print_f64" /tmp/gate_t7.sio "PASS"

# Test: error reporting (compile-time warning)
cat > /tmp/gate_t8.sio << 'EOF'
fn main() -> i64 with IO { let x = unknown_var; return 0 }
EOF
TOTAL=$((TOTAL+1))
err_out=$($S /tmp/gate_t8.sio /tmp/gate_err.elf 2>&1)
if echo "$err_out" | grep -q "warning.*line"; then
    echo "PASS: error_line_numbers"
    PASS=$((PASS+1))
else
    echo "FAIL: error_line_numbers (got: $(echo "$err_out" | head -c 80))"
    FAIL=$((FAIL+1))
fi

# --- Summary ---
echo ""
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="
if [ $FAIL -eq 0 ]; then
    echo "ALL PASS"
    exit 0
else
    echo "SOME FAILED"
    exit 1
fi
