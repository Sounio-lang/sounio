#!/usr/bin/env bash
# Sprint 118: Struct return values in lean AST driver
# Two-register return convention: rax=field0, rdx=field1
# Callee: ExprStructLit in return position → field0→rax, field1→rdx
# Caller: let s = f(p, q) → s_field0 from rax, s_field1 from rdx
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

PASS=0; FAIL=0; NOT_RUN=0

check() {
    local name="$1" src="$2" key="$3" expected="$4"
    local out elf rc=0
    elf=$(mktemp /tmp/lean118_XXXXXX.elf)
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

# Complex number add (2-field struct return)
cat > /tmp/s118_complex.sio << 'EOF'
struct Complex { re: i64, im: i64 }
fn complex_add(a: Complex, b: Complex) -> Complex {
    Complex { re: a.re + b.re, im: a.im + b.im }
}
fn main() -> i64 {
    let p = Complex { re: 3, im: 4 }
    let q = Complex { re: 1, im: 2 }
    let s = complex_add(p, q)
    print("re=")
    print_int(s.re)
    print("\n")
    print("im=")
    print_int(s.im)
    print("\n")
    0
}
EOF
check "complex_re" /tmp/s118_complex.sio "re" "4"
check "complex_im" /tmp/s118_complex.sio "im" "6"

# Vec2 negate (2-field struct return from a single struct param)
cat > /tmp/s118_negate.sio << 'EOF'
struct Vec2 { x: i64, y: i64 }
fn negate(v: Vec2) -> Vec2 {
    Vec2 { x: 0 - v.x, y: 0 - v.y }
}
fn main() -> i64 {
    let a = Vec2 { x: 5, y: 3 }
    let b = negate(a)
    print("nx=")
    print_int(b.x)
    print("\n")
    print("ny=")
    print_int(b.y)
    print("\n")
    0
}
EOF
check "negate_x" /tmp/s118_negate.sio "nx" "-5"
check "negate_y" /tmp/s118_negate.sio "ny" "-3"

# Vec2 scale by scalar (struct param + scalar param, struct return)
cat > /tmp/s118_scale.sio << 'EOF'
struct Vec2 { x: i64, y: i64 }
fn scale(v: Vec2, k: i64) -> Vec2 {
    Vec2 { x: v.x * k, y: v.y * k }
}
fn main() -> i64 {
    let a = Vec2 { x: 2, y: 3 }
    let b = scale(a, 4)
    print("sx=")
    print_int(b.x)
    print("\n")
    print("sy=")
    print_int(b.y)
    print("\n")
    0
}
EOF
check "scale_x" /tmp/s118_scale.sio "sx" "8"
check "scale_y" /tmp/s118_scale.sio "sy" "12"

# Chained struct returns: negate(scale(v, k))
cat > /tmp/s118_chain.sio << 'EOF'
struct Vec2 { x: i64, y: i64 }
fn scale(v: Vec2, k: i64) -> Vec2 {
    Vec2 { x: v.x * k, y: v.y * k }
}
fn negate(v: Vec2) -> Vec2 {
    Vec2 { x: 0 - v.x, y: 0 - v.y }
}
fn main() -> i64 {
    let a = Vec2 { x: 3, y: 5 }
    let b = scale(a, 2)
    let c = negate(b)
    print("cx=")
    print_int(c.x)
    print("\n")
    print("cy=")
    print_int(c.y)
    print("\n")
    0
}
EOF
check "chain_x" /tmp/s118_chain.sio "cx" "-6"
check "chain_y" /tmp/s118_chain.sio "cy" "-10"

# midpoint (struct return used in arithmetic)
cat > /tmp/s118_mid.sio << 'EOF'
struct Point { x: i64, y: i64 }
fn midpoint(a: Point, b: Point) -> Point {
    Point { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
}
fn main() -> i64 {
    let p = Point { x: 0, y: 0 }
    let q = Point { x: 10, y: 6 }
    let m = midpoint(p, q)
    print("mx=")
    print_int(m.x)
    print("\n")
    print("my=")
    print_int(m.y)
    print("\n")
    0
}
EOF
check "mid_x" /tmp/s118_mid.sio "mx" "5"
check "mid_y" /tmp/s118_mid.sio "my" "3"

echo ""
echo "Sprint 118: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN total=$((PASS+FAIL+NOT_RUN))"
[ $FAIL -eq 0 ]
