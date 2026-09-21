#!/bin/bash
# souc_v2_gate.sh -- Gate test for lean_single.sio / souc_v2 bootstrap compiler
# Runs bootstrap fixed-point, souc_v2 split fixed-point, feature tests, regression.
# pipefail: a pipeline's rc must be the producer's, not the reader's.
# Without it every `prog | tail -1` below read tail's rc (0 for a prog that
# crashed after printing). The early-exiting-reader shapes this file used to
# carry are gone too: assertions grep here-strings, not `echo | grep -q`.
set -eo pipefail
PASS=0; FAIL=0; TOTAL=0
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
BOOT4="${SOUC_BOOT4:-/tmp/final_boot4.elf}"

HOST_TARGET="${SOUNIO_HOST_TARGET:-x86_64-linux}"
case "$(uname -s):$(uname -m)" in
  Linux:x86_64|Linux:amd64) HOST_TARGET="${SOUNIO_HOST_TARGET:-x86_64-linux}" ;;
  Linux:arm64|Linux:aarch64) HOST_TARGET="${SOUNIO_HOST_TARGET:-aarch64-linux}" ;;
  Darwin:arm64|Darwin:aarch64) HOST_TARGET="${SOUNIO_HOST_TARGET:-aarch64-macos}" ;;
  Darwin:x86_64|Darwin:amd64) HOST_TARGET="${SOUNIO_HOST_TARGET:-x86_64-macos}" ;;
esac

# Try multiple bootstrap binary locations
if [ ! -x "$BOOT4" ]; then
    for candidate in \
        artifacts/bootstrap/boot4.elf \
        artifacts/bootstrap/final_boot4.elf \
        bootstrap/stage0; do
        if [ -x "$candidate" ]; then
            BOOT4="$candidate"
            break
        fi
    done
fi

if [ ! -x "$BOOT4" ]; then
    echo "ERROR: no bootstrap binary found"
    exit 1
fi

echo "=== souc-v2 Gate ==="
echo ""

# --- Bootstrap fixed-point ---
echo "--- Bootstrap (boot4 -> s1 -> s2 -> s3) ---"
rm -f /tmp/gate_s1.elf /tmp/gate_s2.elf /tmp/gate_s3.elf
$BOOT4 self-hosted/compiler/lean_single.sio /tmp/gate_s1.elf --target "$HOST_TARGET" >/tmp/gate_s1.log 2>&1
tail -1 /tmp/gate_s1.log
chmod +x /tmp/gate_s1.elf
/tmp/gate_s1.elf self-hosted/compiler/lean_single.sio /tmp/gate_s2.elf --target "$HOST_TARGET" >/tmp/gate_s2.log 2>&1
tail -1 /tmp/gate_s2.log
chmod +x /tmp/gate_s2.elf
/tmp/gate_s2.elf self-hosted/compiler/lean_single.sio /tmp/gate_s3.elf --target "$HOST_TARGET" >/tmp/gate_s3.log 2>&1
tail -1 /tmp/gate_s3.log
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
# Each stage's status must be the compiler's own. This chain used to pipe
# every stage through `tail -1`, so tail's rc=0 stood in for a stage that
# failed after emitting output, and all but the last line of each stage's
# evidence was destroyed in the same stroke. Log to files (the shape the
# bootstrap stages above already use) and let the chain read the real rc.
echo ""
echo "--- souc_v2 split ---"
TOTAL=$((TOTAL+1))
rm -f /tmp/gate_split1.elf /tmp/gate_split2.elf /tmp/gate_split3.elf
if $S self-hosted/compiler/souc_v2/main.sio /tmp/gate_split1.elf >/tmp/gate_split1.log 2>&1 && \
   chmod +x /tmp/gate_split1.elf && \
   /tmp/gate_split1.elf self-hosted/compiler/souc_v2/main.sio /tmp/gate_split2.elf >/tmp/gate_split2.log 2>&1 && \
   chmod +x /tmp/gate_split2.elf && \
   /tmp/gate_split2.elf self-hosted/compiler/souc_v2/main.sio /tmp/gate_split3.elf >/tmp/gate_split3.log 2>&1 && \
   cmp -s /tmp/gate_split2.elf /tmp/gate_split3.elf; then
    echo "PASS: souc_v2 split fixed-point"
    PASS=$((PASS+1))
else
    echo "FAIL: souc_v2 split fixed-point"
    for lg in 1 2 3; do
        tail -n 3 /tmp/gate_split$lg.log 2>/dev/null
    done
    FAIL=$((FAIL+1))
fi

# --- Feature tests ---
echo ""
echo "--- Feature tests ---"

run_test() {
    local name=$1 src=$2 expected=$3
    TOTAL=$((TOTAL+1))
    if timeout 30 $S "$src" /tmp/gate_out.elf >/tmp/gate_compile.log 2>&1 && \
       chmod +x /tmp/gate_out.elf 2>/dev/null; then
        local output rc
        # The program's exit status is part of the contract. This used to be
        # `$(...) || true`, which let a binary print the expected text and
        # then crash and still score PASS. Capture the rc; require 0.
        set +e
        output=$(timeout 10 /tmp/gate_out.elf 2>/dev/null)
        rc=$?
        set -e
        if [ $rc -eq 0 ] && grep -q "$expected" <<<"$output"; then
            echo "PASS: $name"
            PASS=$((PASS+1))
        else
            echo "FAIL: $name (rc=$rc, got: ${output:0:60})"
            FAIL=$((FAIL+1))
        fi
    else
        echo "FAIL: $name (compile error)"
        tail -n 20 /tmp/gate_compile.log 2>/dev/null || true
        FAIL=$((FAIL+1))
    fi
}

run_test_exact() {
    local name=$1 src=$2 expected=$3
    shift 3
    TOTAL=$((TOTAL+1))
    if timeout 30 $S "$src" /tmp/gate_out.elf >/tmp/gate_compile.log 2>&1 && \
       chmod +x /tmp/gate_out.elf 2>/dev/null; then
        local output rc
        # Same contract as run_test: the program's own exit status must be 0.
        set +e
        output=$(timeout 10 /tmp/gate_out.elf "$@" 2>/dev/null)
        rc=$?
        set -e
        if [ $rc -eq 0 ] && [ "$output" = "$expected" ]; then
            echo "PASS: $name"
            PASS=$((PASS+1))
        else
            echo "FAIL: $name (rc=$rc)"
            echo "  expected: $(printf '%q' "$expected")"
            echo "  got:      $(printf '%q' "$output")"
            FAIL=$((FAIL+1))
        fi
    else
        echo "FAIL: $name (compile error)"
        tail -n 20 /tmp/gate_compile.log 2>/dev/null || true
        FAIL=$((FAIL+1))
    fi
}

run_cross_compile_test() {
    local name=$1 src=$2 target=$3 expected_kind=$4
    TOTAL=$((TOTAL+1))
    if timeout 30 $S "$src" /tmp/gate_cross.out --target "$target" >/tmp/gate_cross.log 2>&1; then
        local kind
        kind=$(file /tmp/gate_cross.out 2>/dev/null || true)
        if grep -q "$expected_kind" <<<"$kind"; then
            echo "PASS: $name"
            PASS=$((PASS+1))
        else
            echo "FAIL: $name (unexpected artifact: $kind)"
            FAIL=$((FAIL+1))
        fi
    else
        echo "FAIL: $name (compile error)"
        tail -n 20 /tmp/gate_cross.log 2>/dev/null || true
        FAIL=$((FAIL+1))
    fi
}

# Assert a program is REJECTED at compile time: the compiler exits non-zero,
# emits no ELF, and prints an `error:` line containing $3. This is the contract for
# constructs this engine cannot compile correctly. The alternative -- a warning,
# exit 0, and a binary that dereferences a null pointer -- is a miscompile that
# only shows up at run time. Extra arguments go to the compiler (e.g. --target).
run_test_rejected() {
    local name=$1 src=$2 expected=$3
    shift 3
    TOTAL=$((TOTAL+1))
    rm -f /tmp/gate_rej.elf
    local rc
    set +e
    timeout 30 $S "$src" /tmp/gate_rej.elf "$@" >/tmp/gate_rej.log 2>&1
    rc=$?
    set -e
    if [ $rc -ne 0 ] && [ ! -s /tmp/gate_rej.elf ] \
       && grep "^error:" /tmp/gate_rej.log | grep -qF "$expected"; then
        echo "PASS: $name"
        PASS=$((PASS+1))
    else
        echo "FAIL: $name (rc=$rc, elf=$([ -s /tmp/gate_rej.elf ] && echo emitted || echo none), want error containing: $expected)"
        tail -n 6 /tmp/gate_rej.log 2>/dev/null | sed 's/^/  /' || true
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

# Test: epistemic::propagate transcendental calls shadow the f64 builtins.
run_test "propagate_epistemic_exp" tests/run-pass/propagate_epistemic_exp.sio "PROPAGATE_EPISTEMIC_EXP_OK"
run_test "propagate_epistemic_ln" tests/run-pass/propagate_epistemic_ln.sio "PROPAGATE_EPISTEMIC_LN_OK"
run_test "propagate_epistemic_pow" tests/run-pass/propagate_epistemic_pow.sio "PROPAGATE_EPISTEMIC_POW_OK"

# Test: extern "C" math declarations retain intrinsic lowering.
run_test "extern_c_math_intrinsics" tests/run-pass/extern_c_math_intrinsics.sio "EXTERN_C_MATH_INTRINSICS_OK"

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

# Test: fixed-format print_f64
run_test_exact \
    "print_f64_fixed6" \
    self-hosted/compiler/native_print_f64_smoke.sio \
    $'3.141590\n-0.500000\n2.000000'

# Test: read_file + read_f64/read_i64
# Write test binary: little-endian f64(3.14159) f64(-0.5) i64(42)
printf '\x6e\x86\x1b\xf0\xf9\x21\x09\x40\x00\x00\x00\x00\x00\x00\xe0\xbf\x2a\x00\x00\x00\x00\x00\x00\x00' > /tmp/gate_read64.bin
run_test_exact \
    "read64_smoke" \
    self-hosted/compiler/native_read64_smoke.sio \
    $'3.141590\n-0.500000\n42' \
    /tmp/gate_read64.bin

# ARM64 smoke coverage: compile-only
run_cross_compile_test \
    "print_f64_aarch64_linux" \
    self-hosted/compiler/native_print_f64_smoke.sio \
    aarch64-linux \
    "ARM aarch64"
run_cross_compile_test \
    "read64_aarch64_linux" \
    self-hosted/compiler/native_read64_smoke.sio \
    aarch64-linux \
    "ARM aarch64"
run_cross_compile_test \
    "print_f64_aarch64_macos" \
    self-hosted/compiler/native_print_f64_smoke.sio \
    aarch64-macos \
    "Mach-O 64-bit arm64"
run_cross_compile_test \
    "read64_aarch64_macos" \
    self-hosted/compiler/native_read64_smoke.sio \
    aarch64-macos \
    "Mach-O 64-bit arm64"
run_cross_compile_test \
    "extern_c_math_intrinsics_aarch64_linux" \
    tests/run-pass/extern_c_math_intrinsics.sio \
    aarch64-linux \
    "ARM aarch64"

# Test: error reporting (compile-time warning)
cat > /tmp/gate_t8.sio << 'EOF'
fn main() -> i64 with IO { let x = unknown_var; return 0 }
EOF
TOTAL=$((TOTAL+1))
err_out=$($S /tmp/gate_t8.sio /tmp/gate_err.elf 2>&1 || true)
# #1634: the assertion used to be `grep -q "line"`, which the old vague
# `E200 \`x\` at line N` satisfied while carrying no file and no `error:`
# prefix -- so `grep '^error'` on a failing build returned nothing. The
# contract this test is really for is "a diagnostic names a location and is
# greppable as an error", so assert that instead of the word "line".
if grep -q "^error" <<<"$err_out" && grep -qE ":[0-9]+" <<<"$err_out"; then
    echo "PASS: error_line_numbers"
    PASS=$((PASS+1))
else
    echo "FAIL: error_line_numbers (got: ${err_out:0:80})"
    FAIL=$((FAIL+1))
fi

# Tuple-field borrows must FAIL CLOSED. `&t.0` on a tuple used to warn "field borrow
# requires struct base", exit 0 and compile the borrow to a null pointer, so the
# program crashed with SIGSEGV at run time (tests/run-pass/effects_sandbox_handlers.sio,
# #2595). Five call sites carried that behaviour -- three in the x86 backend (borrow
# of a variable base, of a global base, through a deref) and two in the aarch64
# backend (variable base, through a deref) -- so there is one witness per site.
# Programs use the shapes that were measured to crash.
TB_HDR='struct Big { tag: i64, n: i64 }
fn mk() -> (Big, i64) with Mut { let b = Big { tag: 7, n: 5 }; return (b, 2) }
fn has(b: &Big) -> bool { return b.tag == 7 }'

cat > /tmp/gate_tb_var.sio << EOF
$TB_HDR
fn main() -> i64 with IO { let t = mk(); if has(&t.0) { return 0 }; return 1 }
EOF
run_test_rejected "tuple_field_borrow_var_x86" /tmp/gate_tb_var.sio "field borrow requires struct base"
run_test_rejected "tuple_field_borrow_var_aarch64" /tmp/gate_tb_var.sio "field borrow requires struct base" --target aarch64-linux

cat > /tmp/gate_tb_glob.sio << EOF
$TB_HDR
var GT: (Big, i64)
fn main() -> i64 with IO { GT = mk(); if has(&GT.0) { return 0 }; return 1 }
EOF
run_test_rejected "tuple_field_borrow_global_x86" /tmp/gate_tb_glob.sio "field borrow requires struct base"

cat > /tmp/gate_tb_deref.sio << EOF
$TB_HDR
fn viaptr(p: &(Big, i64)) -> bool { return has(&(*p).0) }
fn main() -> i64 with IO { let t = mk(); if viaptr(&t) { return 0 }; return 1 }
EOF
run_test_rejected "tuple_field_borrow_deref_x86" /tmp/gate_tb_deref.sio "field borrow through deref requires struct pointee"
run_test_rejected "tuple_field_borrow_deref_aarch64" /tmp/gate_tb_deref.sio "field borrow through deref requires struct pointee" --target aarch64-linux

# Control: the same borrow through a destructured binding must still compile and run,
# so the rejections above are the tuple-field case and not a blanket failure.
cat > /tmp/gate_tb_ok.sio << EOF
$TB_HDR
fn main() -> i64 with IO { let (b, n) = mk(); if has(&b) && n == 2 { print("TB_OK\n") }; return 0 }
EOF
run_test "tuple_destructure_then_borrow" /tmp/gate_tb_ok.sio "TB_OK"

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
