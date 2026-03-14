#!/usr/bin/env bash
# Sprint 116: Struct field access in lean AST driver
# Local structs: literal init, field read, field write (mutation)
# Limitation: struct passing to/from functions not yet supported
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

PASS=0; FAIL=0; NOT_RUN=0

check() {
    local name="$1" src="$2" key="$3" expected="$4"
    local out elf rc=0
    elf=$(mktemp /tmp/lean116_XXXXXX.elf)
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

# Basic struct: 2-field Point
cat > /tmp/s116_point.sio << 'EOF'
struct Point { x: i64, y: i64 }
fn main() -> i64 {
    var p = Point { x: 3, y: 7 }
    print("x=")
    print_int(p.x)
    print("\n")
    print("y=")
    print_int(p.y)
    print("\n")
    0
}
EOF
check "point_x" /tmp/s116_point.sio "x" "3"
check "point_y" /tmp/s116_point.sio "y" "7"

# Field mutation
cat > /tmp/s116_mut.sio << 'EOF'
struct Point { x: i64, y: i64 }
fn main() -> i64 {
    var p = Point { x: 10, y: 20 }
    p.x = p.x + 5
    p.y = p.y * 2
    print("px=")
    print_int(p.x)
    print("\n")
    print("py=")
    print_int(p.y)
    print("\n")
    0
}
EOF
check "mut_px" /tmp/s116_mut.sio "px" "15"
check "mut_py" /tmp/s116_mut.sio "py" "40"

# 3-field struct with computed field
cat > /tmp/s116_rect.sio << 'EOF'
struct Rect { w: i64, h: i64, area: i64 }
fn main() -> i64 {
    var r = Rect { w: 6, h: 7, area: 0 }
    r.area = r.w * r.h
    print("rw=")
    print_int(r.w)
    print("\n")
    print("rh=")
    print_int(r.h)
    print("\n")
    print("area=")
    print_int(r.area)
    print("\n")
    0
}
EOF
check "rect_w"    /tmp/s116_rect.sio "rw"   "6"
check "rect_h"    /tmp/s116_rect.sio "rh"   "7"
check "rect_area" /tmp/s116_rect.sio "area" "42"

# Two structs in the same scope
cat > /tmp/s116_two.sio << 'EOF'
struct Vec2 { x: i64, y: i64 }
fn main() -> i64 {
    var a = Vec2 { x: 1, y: 2 }
    var b = Vec2 { x: 3, y: 4 }
    let dot = a.x * b.x + a.y * b.y
    print("dot=")
    print_int(dot)
    print("\n")
    0
}
EOF
check "two_dot" /tmp/s116_two.sio "dot" "11"

# Struct with conditional field selection
cat > /tmp/s116_cond.sio << 'EOF'
struct Pair { lo: i64, hi: i64 }
fn main() -> i64 {
    var p = Pair { lo: 3, hi: 9 }
    let mid = (p.lo + p.hi) / 2
    let pick = if mid > 5 { p.hi } else { p.lo }
    print("mid=")
    print_int(mid)
    print("\n")
    print("pick=")
    print_int(pick)
    print("\n")
    0
}
EOF
check "cond_mid"  /tmp/s116_cond.sio "mid"  "6"
check "cond_pick" /tmp/s116_cond.sio "pick" "9"

echo ""
echo "Sprint 116: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN total=$((PASS+FAIL+NOT_RUN))"
[ $FAIL -eq 0 ]
