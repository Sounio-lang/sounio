#!/usr/bin/env bash
# Sprint 112: Array Indexing Gate — local [i64; N] read/write + sieve/sort/dp
set -eo pipefail

SOUC=./bin/souc
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
TMPDIR=/tmp/sprint112_gate
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

# ── T1: Sieve of Eratosthenes ────────────────────────────────────────────────
cat > "$TMPDIR/sieve.sio" << 'EOF'
fn main() -> i64 {
    var sieve: [i64; 300] = [1; 300]
    sieve[0] = 0
    sieve[1] = 0
    var i: i64 = 2
    while i * i <= 200 {
        if sieve[i] {
            var j: i64 = i * i
            while j <= 200 {
                sieve[j] = 0
                j = j + i
            }
        }
        i = i + 1
    }
    var count: i64 = 0
    var n: i64 = 2
    while n <= 200 {
        if sieve[n] { count = count + 1 }
        n = n + 1
    }
    print("sieve_primes_200=")
    print_int(count)
    print("\n")
    0
}
EOF
compile "$TMPDIR/sieve.sio" "$TMPDIR/sieve.elf" || _ec=$?
chmod +x "$TMPDIR/sieve.elf" 2>/dev/null
check "sieve_200" "$TMPDIR/sieve.elf" "sieve_primes_200=46"

# ── T2: Array sum 1..10 ──────────────────────────────────────────────────────
cat > "$TMPDIR/array_sum.sio" << 'EOF'
fn main() -> i64 {
    var a: [i64; 20] = [0; 20]
    var i: i64 = 0
    while i < 10 {
        a[i] = i + 1
        i = i + 1
    }
    var sum: i64 = 0
    i = 0
    while i < 10 {
        sum = sum + a[i]
        i = i + 1
    }
    print("array_sum_10=")
    print_int(sum)
    print("\n")
    0
}
EOF
compile "$TMPDIR/array_sum.sio" "$TMPDIR/array_sum.elf" || _ec=$?
chmod +x "$TMPDIR/array_sum.elf" 2>/dev/null
check "array_sum" "$TMPDIR/array_sum.elf" "array_sum_10=55"

# ── T3: Bubble sort ──────────────────────────────────────────────────────────
cat > "$TMPDIR/bubble_sort.sio" << 'EOF'
fn main() -> i64 {
    var a: [i64; 10] = [0; 10]
    a[0] = 5
    a[1] = 3
    a[2] = 8
    a[3] = 1
    a[4] = 9
    a[5] = 2
    a[6] = 7
    a[7] = 4
    a[8] = 6
    a[9] = 0
    var n: i64 = 10
    var ii: i64 = 0
    while ii < n {
        var jj: i64 = 0
        while jj < n - ii - 1 {
            if a[jj] > a[jj + 1] {
                var tmp: i64 = a[jj]
                a[jj] = a[jj + 1]
                a[jj + 1] = tmp
            }
            jj = jj + 1
        }
        ii = ii + 1
    }
    print("sorted_max=")
    print_int(a[9])
    print("\n")
    print("sorted_min=")
    print_int(a[0])
    print("\n")
    0
}
EOF
compile "$TMPDIR/bubble_sort.sio" "$TMPDIR/bubble_sort.elf" || _ec=$?
chmod +x "$TMPDIR/bubble_sort.elf" 2>/dev/null
check "bubble_sort" "$TMPDIR/bubble_sort.elf" "sorted_max=9"

# ── T4: Prefix sum ───────────────────────────────────────────────────────────
cat > "$TMPDIR/prefix_sum.sio" << 'EOF'
fn main() -> i64 {
    var a: [i64; 20] = [0; 20]
    var prefix: [i64; 20] = [0; 20]
    var i: i64 = 0
    while i < 10 {
        a[i] = i + 1
        i = i + 1
    }
    prefix[0] = a[0]
    i = 1
    while i < 10 {
        prefix[i] = prefix[i - 1] + a[i]
        i = i + 1
    }
    print("prefix_9=")
    print_int(prefix[9])
    print("\n")
    0
}
EOF
compile "$TMPDIR/prefix_sum.sio" "$TMPDIR/prefix_sum.elf" || _ec=$?
chmod +x "$TMPDIR/prefix_sum.elf" 2>/dev/null
check "prefix_sum" "$TMPDIR/prefix_sum.elf" "prefix_9=55"

# ── T5: DP Fibonacci ─────────────────────────────────────────────────────────
cat > "$TMPDIR/dp_fib.sio" << 'EOF'
fn main() -> i64 {
    var dp: [i64; 50] = [0; 50]
    dp[0] = 0
    dp[1] = 1
    var i: i64 = 2
    while i <= 30 {
        dp[i] = dp[i - 1] + dp[i - 2]
        i = i + 1
    }
    print("fib_30=")
    print_int(dp[30])
    print("\n")
    print("fib_20=")
    print_int(dp[20])
    print("\n")
    0
}
EOF
compile "$TMPDIR/dp_fib.sio" "$TMPDIR/dp_fib.elf" || _ec=$?
chmod +x "$TMPDIR/dp_fib.elf" 2>/dev/null
check "dp_fibonacci" "$TMPDIR/dp_fib.elf" "fib_30=832040"

echo ""
echo "=== Sprint 112 Array Gate: $pass PASS, $fail FAIL, $not_run NOT_RUN ==="
[ $fail -eq 0 ]
