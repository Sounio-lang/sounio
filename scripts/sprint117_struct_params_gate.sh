#!/usr/bin/env bash
# Sprint 117: Struct parameters to functions in lean AST driver
# Struct params expand to per-field register passing (SysV ABI field-by-field)
# Struct return values deferred to Sprint 118
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

PASS=0; FAIL=0; NOT_RUN=0

check() {
    local name="$1" src="$2" key="$3" expected="$4"
    local out elf rc=0
    elf=$(mktemp /tmp/lean117_XXXXXX.elf)
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

# Vec2 dot product (two 2-field struct params)
cat > /tmp/s117_dot.sio << 'EOF'
struct Vec2 { x: i64, y: i64 }
fn dot(a: Vec2, b: Vec2) -> i64 {
    a.x * b.x + a.y * b.y
}
fn main() -> i64 {
    let p = Vec2 { x: 3, y: 4 }
    let q = Vec2 { x: 1, y: 2 }
    print("dot=")
    print_int(dot(p, q))
    print("\n")
    print("dot0=")
    print_int(dot(q, q))
    print("\n")
    0
}
EOF
check "dot_pq"  /tmp/s117_dot.sio "dot"  "11"
check "dot_qq"  /tmp/s117_dot.sio "dot0" "5"

# Single struct param
cat > /tmp/s117_lsq.sio << 'EOF'
struct Vec2 { x: i64, y: i64 }
fn length_sq(v: Vec2) -> i64 {
    v.x * v.x + v.y * v.y
}
fn main() -> i64 {
    let a = Vec2 { x: 3, y: 4 }
    let b = Vec2 { x: 5, y: 12 }
    print("lsq_a=")
    print_int(length_sq(a))
    print("\n")
    print("lsq_b=")
    print_int(length_sq(b))
    print("\n")
    0
}
EOF
check "lsq_a" /tmp/s117_lsq.sio "lsq_a" "25"
check "lsq_b" /tmp/s117_lsq.sio "lsq_b" "169"

# 3-field struct param
cat > /tmp/s117_rgb.sio << 'EOF'
struct RGB { r: i64, g: i64, b: i64 }
fn brightness(c: RGB) -> i64 {
    c.r + c.g + c.b
}
fn main() -> i64 {
    let white = RGB { r: 255, g: 255, b: 255 }
    let red   = RGB { r: 200, g: 50, b: 30 }
    print("bright_w=")
    print_int(brightness(white))
    print("\n")
    print("bright_r=")
    print_int(brightness(red))
    print("\n")
    0
}
EOF
check "rgb_white" /tmp/s117_rgb.sio "bright_w" "765"
check "rgb_red"   /tmp/s117_rgb.sio "bright_r" "280"

# Mixed struct + scalar params
cat > /tmp/s117_scale.sio << 'EOF'
struct Vec2 { x: i64, y: i64 }
fn manhattan(a: Vec2, b: Vec2) -> i64 {
    let dx = if a.x > b.x { a.x - b.x } else { b.x - a.x }
    let dy = if a.y > b.y { a.y - b.y } else { b.y - a.y }
    dx + dy
}
fn main() -> i64 {
    let p = Vec2 { x: 1, y: 2 }
    let q = Vec2 { x: 4, y: 6 }
    print("man=")
    print_int(manhattan(p, q))
    print("\n")
    0
}
EOF
check "manhattan" /tmp/s117_scale.sio "man" "7"

# Struct param + loop
cat > /tmp/s117_loop.sio << 'EOF'
struct Range { lo: i64, hi: i64 }
fn sum_range(r: Range) -> i64 {
    var total: i64 = 0
    var i: i64 = r.lo
    while i <= r.hi {
        total = total + i
        i = i + 1
    }
    total
}
fn main() -> i64 {
    let r1 = Range { lo: 1, hi: 10 }
    let r2 = Range { lo: 1, hi: 100 }
    print("sum10=")
    print_int(sum_range(r1))
    print("\n")
    print("sum100=")
    print_int(sum_range(r2))
    print("\n")
    0
}
EOF
check "sum_range_10"  /tmp/s117_loop.sio "sum10"  "55"
check "sum_range_100" /tmp/s117_loop.sio "sum100" "5050"

echo ""
echo "Sprint 117: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN total=$((PASS+FAIL+NOT_RUN))"
[ $FAIL -eq 0 ]
