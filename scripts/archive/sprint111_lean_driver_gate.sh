#!/usr/bin/env bash
# Sprint 111: Lean Driver Gate — AST-direct x86-64 native compiler
# Tests: hello_world, primes, fizzbuzz, fib, gcd, collatz, hanoi, ackermann, isqrt
set -eo pipefail

SOUC=./bin/souc
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
TMPDIR=/tmp/sprint111_gate
mkdir -p "$TMPDIR"

pass=0; fail=0; not_run=0

compile() {
    local src=$1 out=$2
    timeout 120 $SOUC run $LEAN -- "$src" "$out" 2>/dev/null
}

check() {
    local name=$1 elf=$2 expected=$3
    local actual
    actual=$("$elf" 2>/dev/null || true)
    if echo "$actual" | grep -qF "$expected"; then
        echo "PASS: $name"
        pass=$((pass+1))
    else
        echo "FAIL: $name  expected='$expected'"
        echo "  got: $(echo "$actual" | tail -3)"
        fail=$((fail+1))
    fi
}

_ec=0

# ── T1: hello_world ──────────────────────────────────────────────────────────
cat > "$TMPDIR/hello.sio" << 'EOF'
fn main() -> i64 {
    print("Hello, World!")
    print("\n")
    print("The answer is: ")
    print_int(42)
    print("\n")
    0
}
EOF
compile "$TMPDIR/hello.sio" "$TMPDIR/hello.elf" || _ec=$?
chmod +x "$TMPDIR/hello.elf" 2>/dev/null
check "hello_world" "$TMPDIR/hello.elf" "The answer is: 42"

# ── T2: primes to 100 ────────────────────────────────────────────────────────
cat > "$TMPDIR/primes.sio" << 'EOF'
fn is_prime(n: i64) -> i64 {
    if n < 2 { return 0 }
    var i: i64 = 2
    while i * i <= n {
        if n % i == 0 { return 0 }
        i = i + 1
    }
    1
}
fn main() -> i64 {
    var n: i64 = 2
    var count: i64 = 0
    while n <= 100 {
        if is_prime(n) { count = count + 1 }
        n = n + 1
    }
    print("prime_count=")
    print_int(count)
    print("\n")
    0
}
EOF
compile "$TMPDIR/primes.sio" "$TMPDIR/primes.elf" || _ec=$?
chmod +x "$TMPDIR/primes.elf" 2>/dev/null
check "primes_100" "$TMPDIR/primes.elf" "prime_count=25"

# ── T3: FizzBuzz ─────────────────────────────────────────────────────────────
cat > "$TMPDIR/fizzbuzz.sio" << 'EOF'
fn main() -> i64 {
    var i: i64 = 1
    while i <= 20 {
        let r3 = i % 3
        let r5 = i % 5
        if r3 == 0 {
            if r5 == 0 { print("FizzBuzz") } else { print("Fizz") }
        } else {
            if r5 == 0 { print("Buzz") } else { print_int(i) }
        }
        print("\n")
        i = i + 1
    }
    0
}
EOF
compile "$TMPDIR/fizzbuzz.sio" "$TMPDIR/fizzbuzz.elf" || _ec=$?
chmod +x "$TMPDIR/fizzbuzz.elf" 2>/dev/null
check "fizzbuzz" "$TMPDIR/fizzbuzz.elf" "FizzBuzz"

# ── T4: Fibonacci ────────────────────────────────────────────────────────────
cat > "$TMPDIR/fib.sio" << 'EOF'
fn fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n-1) + fib(n-2) }
}
fn main() -> i64 {
    print("fib(10)=")
    print_int(fib(10))
    print("\n")
    print("fib(5)=")
    print_int(fib(5))
    print("\n")
    0
}
EOF
compile "$TMPDIR/fib.sio" "$TMPDIR/fib.elf" || _ec=$?
chmod +x "$TMPDIR/fib.elf" 2>/dev/null
check "fibonacci" "$TMPDIR/fib.elf" "fib(10)=55"

# ── T5: GCD / LCM ───────────────────────────────────────────────────────────
cat > "$TMPDIR/gcd.sio" << 'EOF'
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}
fn lcm(a: i64, b: i64) -> i64 {
    a / gcd(a, b) * b
}
fn main() -> i64 {
    print("gcd(48,18)=")
    print_int(gcd(48, 18))
    print("\n")
    print("lcm(12,18)=")
    print_int(lcm(12, 18))
    print("\n")
    0
}
EOF
compile "$TMPDIR/gcd.sio" "$TMPDIR/gcd.elf" || _ec=$?
chmod +x "$TMPDIR/gcd.elf" 2>/dev/null
check "gcd_lcm" "$TMPDIR/gcd.elf" "lcm(12,18)=36"

# ── T6: Collatz ──────────────────────────────────────────────────────────────
cat > "$TMPDIR/collatz.sio" << 'EOF'
fn collatz_steps(n: i64) -> i64 {
    var x: i64 = n
    var steps: i64 = 0
    while x != 1 {
        if x % 2 == 0 { x = x / 2 } else { x = x * 3 + 1 }
        steps = steps + 1
    }
    steps
}
fn main() -> i64 {
    print("collatz(27)=")
    print_int(collatz_steps(27))
    print("\n")
    0
}
EOF
compile "$TMPDIR/collatz.sio" "$TMPDIR/collatz.elf" || _ec=$?
chmod +x "$TMPDIR/collatz.elf" 2>/dev/null
check "collatz" "$TMPDIR/collatz.elf" "collatz(27)=111"

# ── T7: Tower of Hanoi ───────────────────────────────────────────────────────
cat > "$TMPDIR/hanoi.sio" << 'EOF'
fn hanoi(n: i64) -> i64 {
    if n == 0 { 0 } else { 2 * hanoi(n-1) + 1 }
}
fn main() -> i64 {
    print("hanoi(10)=")
    print_int(hanoi(10))
    print("\n")
    0
}
EOF
compile "$TMPDIR/hanoi.sio" "$TMPDIR/hanoi.elf" || _ec=$?
chmod +x "$TMPDIR/hanoi.elf" 2>/dev/null
check "hanoi" "$TMPDIR/hanoi.elf" "hanoi(10)=1023"

# ── T8: Ackermann ────────────────────────────────────────────────────────────
cat > "$TMPDIR/ack.sio" << 'EOF'
fn ack(m: i64, n: i64) -> i64 {
    if m == 0 { n + 1 }
    else if n == 0 { ack(m-1, 1) }
    else { ack(m-1, ack(m, n-1)) }
}
fn main() -> i64 {
    print("A(3,4)=")
    print_int(ack(3, 4))
    print("\n")
    0
}
EOF
compile "$TMPDIR/ack.sio" "$TMPDIR/ack.elf" || _ec=$?
chmod +x "$TMPDIR/ack.elf" 2>/dev/null
check "ackermann" "$TMPDIR/ack.elf" "A(3,4)=125"

# ── T9: Integer sqrt ─────────────────────────────────────────────────────────
cat > "$TMPDIR/isqrt.sio" << 'EOF'
fn isqrt(n: i64) -> i64 {
    if n <= 0 { return 0 }
    var x: i64 = n
    var y: i64 = (x + 1) / 2
    while y < x {
        x = y
        y = (x + n / x) / 2
    }
    x
}
fn main() -> i64 {
    print("isqrt(144)=")
    print_int(isqrt(144))
    print("\n")
    print("isqrt(10000)=")
    print_int(isqrt(10000))
    print("\n")
    0
}
EOF
compile "$TMPDIR/isqrt.sio" "$TMPDIR/isqrt.elf" || _ec=$?
chmod +x "$TMPDIR/isqrt.elf" 2>/dev/null
check "isqrt" "$TMPDIR/isqrt.elf" "isqrt(10000)=100"

echo ""
echo "=== Sprint 111 Lean Driver Gate: $pass PASS, $fail FAIL, $not_run NOT_RUN ==="
[ $fail -eq 0 ]
