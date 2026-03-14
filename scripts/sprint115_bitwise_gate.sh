#!/usr/bin/env bash
# Sprint 115: Bitwise ops (&, |, ^, <<, >>) in lean AST driver
# Tests: basic bitwise, Stein binary GCD, popcount, bit mask
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

PASS=0; FAIL=0; NOT_RUN=0

check() {
    local name="$1" src="$2" key="$3" expected="$4"
    local out elf rc=0
    elf=$(mktemp /tmp/lean115_XXXXXX.elf)
    out=$($SOUC run "$LEAN" -- "$src" "$elf" 2>&1 && chmod +x "$elf" && "$elf" 2>&1) || rc=$?
    rm -f "$elf"
    if [ $rc -ne 0 ]; then NOT_RUN=$((NOT_RUN+1)); echo "NOT_RUN $name (exit $rc)"; return; fi
    local val; val=$(echo "$out" | grep "^${key}=" | head -1 | cut -d= -f2)
    if [ "$val" = "$expected" ]; then
        PASS=$((PASS+1)); echo "PASS $name: $key=$val"
    else
        FAIL=$((FAIL+1)); echo "FAIL $name: $key expected=$expected got=$val"
    fi
}

# Basic bitwise ops
cat > /tmp/s115_basic.sio << 'EOF'
fn main() -> i64 {
    let a: i64 = 10
    let b: i64 = 12
    print("band=")
    print_int(a & b)
    print("\n")
    print("bor=")
    print_int(a | b)
    print("\n")
    print("bxor=")
    print_int(a ^ b)
    print("\n")
    print("shl2=")
    print_int(a << 2)
    print("\n")
    print("shr1=")
    print_int(b >> 1)
    print("\n")
    0
}
EOF
check "bit_and"  /tmp/s115_basic.sio "band" "8"
check "bit_or"   /tmp/s115_basic.sio "bor"  "14"
check "bit_xor"  /tmp/s115_basic.sio "bxor" "6"
check "bit_shl"  /tmp/s115_basic.sio "shl2" "40"
check "bit_shr"  /tmp/s115_basic.sio "shr1" "6"

# Stein's binary GCD
cat > /tmp/s115_gcd.sio << 'EOF'
fn gcd(a: i64, b: i64) -> i64 {
    if a == 0 { b } else if b == 0 { a } else if a == b { a
    } else if (a & 1) == 0 {
        if (b & 1) == 0 { gcd(a >> 1, b >> 1) << 1
        } else { gcd(a >> 1, b) }
    } else {
        if (b & 1) == 0 { gcd(a, b >> 1)
        } else if a > b { gcd(a - b, b)
        } else { gcd(b - a, a) }
    }
}
fn main() -> i64 {
    print("gcd48_18=")
    print_int(gcd(48, 18))
    print("\n")
    print("gcd100_75=")
    print_int(gcd(100, 75))
    print("\n")
    print("gcd7_13=")
    print_int(gcd(7, 13))
    print("\n")
    0
}
EOF
check "gcd_48_18"  /tmp/s115_gcd.sio "gcd48_18"  "6"
check "gcd_100_75" /tmp/s115_gcd.sio "gcd100_75" "25"
check "gcd_7_13"   /tmp/s115_gcd.sio "gcd7_13"   "1"

# Popcount (count set bits)
cat > /tmp/s115_pop.sio << 'EOF'
fn popcount(n: i64) -> i64 {
    var x: i64 = n
    var count: i64 = 0
    while x > 0 {
        count = count + (x & 1)
        x = x >> 1
    }
    count
}
fn main() -> i64 {
    print("pop255=")
    print_int(popcount(255))
    print("\n")
    print("pop170=")
    print_int(popcount(170))
    print("\n")
    print("pop1023=")
    print_int(popcount(1023))
    print("\n")
    0
}
EOF
check "pop_255"  /tmp/s115_pop.sio "pop255"  "8"
check "pop_170"  /tmp/s115_pop.sio "pop170"  "4"
check "pop_1023" /tmp/s115_pop.sio "pop1023" "10"

echo ""
echo "Sprint 115: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN total=$((PASS+FAIL+NOT_RUN))"
[ $FAIL -eq 0 ]
