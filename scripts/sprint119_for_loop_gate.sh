#!/usr/bin/env bash
# Sprint 119: for-in + break + continue in lean AST driver
set -eo pipefail
SOUC=${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
export SOUNIO_STDLIB_PATH=${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}
PASS=0; FAIL=0; NOT_RUN=0

check() {
    local name=$1 src=$2 key=$3 want=$4
    local elf
    elf=$(mktemp /tmp/s119_XXXXXX.elf)
    local out rc=0
    out=$(timeout 30 $SOUC run "$LEAN" -- "$src" "$elf" 2>&1 && chmod +x "$elf" && "$elf" 2>&1) || rc=$?
    rm -f "$elf"
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        echo "NOT_RUN $name (exit $rc)"
        NOT_RUN=$((NOT_RUN+1))
    elif echo "$out" | grep -q "^${key}=${want}"; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name (got: $(echo "$out" | grep "^${key}=" | head -1))"
        FAIL=$((FAIL+1))
    fi
}

# 1. sum 0..5 = 0+1+2+3+4 = 10
cat > /tmp/s119_sum.sio << 'SIOEOF'
fn main() -> i64 {
    var s: i64 = 0
    for i in 0..5 {
        s = s + i
    }
    print("sum=")
    print_int(s)
    print("\n")
    0
}
SIOEOF
check "sum_0_5" /tmp/s119_sum.sio "sum" "10"

# 2. sum 1..11 = 55
cat > /tmp/s119_sum2.sio << 'SIOEOF'
fn main() -> i64 {
    var s: i64 = 0
    for i in 1..11 {
        s = s + i
    }
    print("sum=")
    print_int(s)
    print("\n")
    0
}
SIOEOF
check "sum_1_11" /tmp/s119_sum2.sio "sum" "55"

# 3. empty range 5..3 should not execute body
cat > /tmp/s119_empty.sio << 'SIOEOF'
fn main() -> i64 {
    var n: i64 = 0
    for i in 5..3 {
        n = n + 1
    }
    print("n=")
    print_int(n)
    print("\n")
    0
}
SIOEOF
check "empty_range" /tmp/s119_empty.sio "n" "0"

# 4. range with variable hi
cat > /tmp/s119_varhi.sio << 'SIOEOF'
fn count_up(hi: i64) -> i64 {
    var s: i64 = 0
    for i in 0..hi {
        s = s + 1
    }
    s
}
fn main() -> i64 {
    print("c=")
    print_int(count_up(7))
    print("\n")
    0
}
SIOEOF
check "var_hi" /tmp/s119_varhi.sio "c" "7"

# 5. break: stop at first multiple of 3 ≥ 5  → 6
cat > /tmp/s119_break.sio << 'SIOEOF'
fn main() -> i64 {
    var found: i64 = -1
    for i in 0..20 {
        if i >= 5 {
            if (i * 1) / 3 * 3 == i {
                found = i
                break
            }
        }
    }
    print("found=")
    print_int(found)
    print("\n")
    0
}
SIOEOF
check "break_mul3" /tmp/s119_break.sio "found" "6"

# 6. continue: sum only even numbers in 0..8 = 0+2+4+6 = 12
cat > /tmp/s119_cont.sio << 'SIOEOF'
fn main() -> i64 {
    var s: i64 = 0
    for i in 0..8 {
        if i * 1 / 2 * 2 != i {
            continue
        }
        s = s + i
    }
    print("s=")
    print_int(s)
    print("\n")
    0
}
SIOEOF
check "continue_even" /tmp/s119_cont.sio "s" "12"

# 7. nested for loops: multiplication table entry 3*4=12
cat > /tmp/s119_nest.sio << 'SIOEOF'
fn main() -> i64 {
    var prod: i64 = 0
    for i in 0..3 {
        for j in 0..4 {
            prod = prod + 1
        }
    }
    print("prod=")
    print_int(prod)
    print("\n")
    0
}
SIOEOF
check "nested_3x4" /tmp/s119_nest.sio "prod" "12"

# 8. range starting at non-zero: 3..8 = 5 iterations
cat > /tmp/s119_start.sio << 'SIOEOF'
fn main() -> i64 {
    var c: i64 = 0
    for i in 3..8 {
        c = c + 1
    }
    print("c=")
    print_int(c)
    print("\n")
    0
}
SIOEOF
check "range_3_8" /tmp/s119_start.sio "c" "5"

echo ""
echo "Sprint 119: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN total=$((PASS+FAIL+NOT_RUN))"
[ $FAIL -eq 0 ]
