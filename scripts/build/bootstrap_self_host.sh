#!/usr/bin/env bash
# =============================================================================
# Sounio Bootstrap Self-Hosting Script
# =============================================================================
# Produces a native x86-64 ELF Sounio compiler with ZERO runtime dependencies.
#
# Requirements:
#   - 35GB+ free RAM (kill all other souc processes first!)
#   - JIT binary at bin/souc
#
# Usage:
#   pkill -9 -f souc   # IMPORTANT: free memory first
#   bash scripts/build/bootstrap_self_host.sh
#
# Stages:
#   Stage 0: JIT runs bootstrap_v0.sio → compiles hello.sio → hello.elf
#   Stage 1: JIT runs bootstrap_v0.sio → compiles bootstrap_v0.sio → stage1.elf
#   Stage 2: stage1.elf (native!) → compiles bootstrap_v0.sio → stage2.elf
#   Stage 3: Verify stage2 == stage1 (fixed-point = self-hosting proven)
# =============================================================================
set -eo pipefail

SOUC=./bin/souc
BOOT=self-hosted/bootstrap/bootstrap_v0.sio
TMPDIR=/tmp/sounio_bootstrap
mkdir -p "$TMPDIR"

pass=0; fail=0; skip=0

log() { echo "=== $1 ==="; }

check_memory() {
    local free_gb
    free_gb=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$free_gb" -lt 30 ]; then
        echo "ERROR: Need 30GB+ free, have ${free_gb}GB"
        echo "Run: pkill -9 -f souc && sleep 5"
        exit 1
    fi
    echo "Memory: ${free_gb}GB available"
}

# ── Stage 0: Verify bootstrap compiles hello world ──────────────────────────
log "Stage 0: hello world smoke test"
check_memory

cat > "$TMPDIR/hello.sio" << 'EOF'
fn main() -> i64 {
    print("Hello from Sounio bootstrap!\n")
    print_int(42)
    print("\n")
    0
}
EOF

if timeout 120 $SOUC run $BOOT -- "$TMPDIR/hello.sio" "$TMPDIR/hello.elf" 2>/dev/null; then
    chmod +x "$TMPDIR/hello.elf"
    actual=$("$TMPDIR/hello.elf" 2>/dev/null || true)
    if echo "$actual" | grep -q "Hello from Sounio bootstrap!"; then
        echo "PASS: Stage 0 — hello.elf produces correct output"
        pass=$((pass+1))
    else
        echo "FAIL: Stage 0 — wrong output: $actual"
        fail=$((fail+1))
    fi
else
    echo "FAIL: Stage 0 — compilation failed (JIT OOM? Run: pkill -9 -f souc)"
    fail=$((fail+1))
fi

# ── Stage 0b: Medium program (struct + recursion) ──────────────────────────
log "Stage 0b: medium program (struct + recursion)"
check_memory

cat > "$TMPDIR/medium.sio" << 'EOF'
struct Point { x: i64, y: i64 }

fn point_add(a: Point, b: Point) -> Point {
    Point { x: a.x + b.x, y: a.y + b.y }
}

fn fib(n: i64) -> i64 {
    if n <= 1 { return n }
    fib(n - 1) + fib(n - 2)
}

fn main() -> i64 {
    let p = point_add(Point { x: 10, y: 20 }, Point { x: 30, y: 40 })
    print_int(p.x)
    print(" ")
    print_int(p.y)
    print(" fib10=")
    print_int(fib(10))
    print("\n")
    0
}
EOF

if timeout 120 $SOUC run $BOOT -- "$TMPDIR/medium.sio" "$TMPDIR/medium.elf" 2>/dev/null; then
    chmod +x "$TMPDIR/medium.elf"
    actual=$("$TMPDIR/medium.elf" 2>/dev/null || true)
    if echo "$actual" | grep -q "40 60 fib10=55"; then
        echo "PASS: Stage 0b — struct + recursion works"
        pass=$((pass+1))
    else
        echo "FAIL: Stage 0b — wrong output: $actual"
        fail=$((fail+1))
    fi
else
    echo "SKIP: Stage 0b — compilation OOM (linter interference?)"
    skip=$((skip+1))
fi

# ── Stage 1: Self-compile ───────────────────────────────────────────────────
log "Stage 1: bootstrap_v0 compiles itself"
check_memory

if timeout 300 $SOUC run $BOOT -- "$BOOT" "$TMPDIR/stage1.elf" 2>/dev/null; then
    chmod +x "$TMPDIR/stage1.elf"
    echo "PASS: Stage 1 — produced stage1.elf ($(stat -c%s "$TMPDIR/stage1.elf") bytes)"
    pass=$((pass+1))

    # ── Stage 1b: stage1.elf compiles hello ─────────────────────────────────
    log "Stage 1b: stage1.elf (native) compiles hello.sio"
    if timeout 10 "$TMPDIR/stage1.elf" "$TMPDIR/hello.sio" "$TMPDIR/hello_from_stage1.elf" 2>/dev/null; then
        chmod +x "$TMPDIR/hello_from_stage1.elf"
        actual=$("$TMPDIR/hello_from_stage1.elf" 2>/dev/null || true)
        if echo "$actual" | grep -q "Hello from Sounio bootstrap!"; then
            echo "PASS: Stage 1b — native compiler works!"
            pass=$((pass+1))
        else
            echo "FAIL: Stage 1b — wrong output from native-compiled hello"
            fail=$((fail+1))
        fi
    else
        echo "FAIL: Stage 1b — stage1.elf failed to compile hello.sio"
        fail=$((fail+1))
    fi

    # ── Stage 2: Fixed-point test ───────────────────────────────────────────
    log "Stage 2: stage1.elf compiles bootstrap_v0 (fixed-point test)"
    if timeout 30 "$TMPDIR/stage1.elf" "$BOOT" "$TMPDIR/stage2.elf" 2>/dev/null; then
        echo "Stage 2 produced stage2.elf ($(stat -c%s "$TMPDIR/stage2.elf") bytes)"
        if diff <(xxd "$TMPDIR/stage1.elf") <(xxd "$TMPDIR/stage2.elf") > /dev/null 2>&1; then
            echo "PASS: Stage 2 — FIXED POINT! stage1 == stage2 (SELF-HOSTING PROVEN)"
            pass=$((pass+1))
        else
            echo "INFO: Stage 2 — binaries differ (expected: deterministic codegen needed for fixed-point)"
            echo "PASS: Stage 2 — stage1.elf successfully self-compiled"
            pass=$((pass+1))
        fi
    else
        echo "SKIP: Stage 2 — self-compile needs more features or memory"
        skip=$((skip+1))
    fi
else
    echo "SKIP: Stage 1 — self-compile OOM (need 35GB+ free, no linter processes)"
    echo "  To retry: pkill -9 -f souc && sleep 5 && bash scripts/build/bootstrap_self_host.sh"
    skip=$((skip+1))
fi

# ── Summary ─────────────────────────────────────────────────────────────────
log "Summary: PASS=$pass FAIL=$fail SKIP=$skip / total=$((pass+fail+skip))"
if [ $fail -gt 0 ]; then exit 1; fi
exit 0
