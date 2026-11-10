#!/usr/bin/env bash
# madaros_heap_reserve_gate.sh — the runtime heap is reserved, not fixed at 2 GiB
#
# A Madaros-compiled Linux program maps its runtime heap once at start up with
# MAP_NORESERVE, sized min(1 TiB, RAM + swap) and never below 2 GiB; if the
# kernel refuses (RLIMIT_AS, strict overcommit) it halves and retries down to
# 256 MiB and exits 184 below that. heap_limit comes from the length actually
# mapped. This replaces the fixed 2 GiB mapping and any environment knob for
# its size (ADR-028 / ADR-026 amendment, 2026-11-09).
#
# Run against a current-source Madaros (MADAROS_RAW_BIN). Asserts:
#   1. a trivial program prints 42; its resident set stays small even though
#      its address space covers the reservation (lazy commit);
#   2. under `ulimit -v 3 GiB` it still prints 42 (halving fallback);
#   3. under `ulimit -v 200 MiB` it exits 184 (below the floor), not SIGSEGV;
#   4. a by-value loop needing ~9.8 GB of arena, under `ulimit -v 4 GiB`,
#      exits 181 with the arena-full diagnostic — the wall is the mapping,
#      checked, not a crash.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "madaros_heap_reserve_gate"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-}}"
[[ -x "$SOUC" ]] || gate_fail "souc is not executable: $SOUC"
[[ -n "$RAW" && -x "$RAW" ]] || gate_fail "MADAROS_RAW_BIN must name a current-source Madaros ELF"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/heap-reserve.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

cat >"$WORK/small.sio" <<'SIO'
struct P { a: i64, b: i64, c: i64 }
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    let p = P { a: 1, b: 2, c: 39 }
    println(p.a + p.b + p.c)
    var s: i64 = 0
    var i: i64 = 0
    while i < 400000000 { s = s + i  i = i + 1 }
    if s < 0 { println(s) }
    0
}
SIO

cat >"$WORK/big.sio" <<'SIO'
struct Big { data: [f64; 4096], n: i64 }

fn set0(b: Big, v: f64) -> Big with Mut, Panic {
    var r = b
    r.data[0] = v
    return r
}

fn main() -> i64 with IO, Mut, Panic, Div {
    var b = Big { data: [0.0; 4096], n: 0 }
    var i: i64 = 0
    while i < 300000 {
        b = set0(b, i as f64)
        i = i + 1
    }
    println(b.data[0])
    0
}
SIO

build() {
  ( ulimit -s unlimited 2>/dev/null || true
    MADAROS_RAW_BIN="$RAW" "$SOUC" compile "$WORK/$1.sio" -o "$WORK/$1.elf" ) >"$WORK/$1.build.log" 2>&1 \
    || { sed -n '1,30p' "$WORK/$1.build.log" >&2; gate_fail "$1: compile failed"; }
  chmod +x "$WORK/$1.elf"
}
build small
build big

# 1. normal run; sample address space and RSS while it loops.
"$WORK/small.elf" >"$WORK/small.out" 2>&1 &
pid=$!
sleep 0.3
vm_kb=0
rss_kb=0
if [[ -r "/proc/$pid/status" ]]; then
  vm_kb=$(awk '/^VmSize:/ {print $2}' "/proc/$pid/status")
  rss_kb=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status")
fi
wait "$pid" || gate_fail "small program exited nonzero"
grep -qx 42 "$WORK/small.out" || gate_fail "small program did not print 42"
[[ "$vm_kb" -ge 2097152 ]] || gate_fail "address space ${vm_kb} KiB is below the 2 GiB minimum reservation"
[[ "$rss_kb" -le 65536 ]] || gate_fail "resident set ${rss_kb} KiB: the reservation is being committed eagerly"
echo "normal: prints 42; VmSize=${vm_kb} KiB, VmRSS=${rss_kb} KiB"

# 2. RLIMIT_AS below RAM: the halving fallback still maps a heap.
set +e
( ulimit -v 3145728; "$WORK/small.elf" ) >"$WORK/rl3.out" 2>&1
rc=$?
set -e
[[ "$rc" -eq 0 ]] && grep -qx 42 "$WORK/rl3.out" || gate_fail "under ulimit -v 3 GiB: rc=$rc, want 42 and rc 0"
echo "ulimit -v 3 GiB: prints 42"

# 3. Below the floor: a defined exit, not a fault.
set +e
( ulimit -v 204800; "$WORK/small.elf" ) >"$WORK/rl200.out" 2>&1
rc=$?
set -e
[[ "$rc" -eq 184 ]] || gate_fail "under ulimit -v 200 MiB: rc=$rc, want 184"
echo "ulimit -v 200 MiB: exit 184"

# 4. The wall is checked: a 4 GiB mapping fills and reports arena full.
set +e
( ulimit -v 4194304; timeout 600 "$WORK/big.elf" ) >"$WORK/big.out" 2>"$WORK/big.err"
rc=$?
set -e
[[ "$rc" -eq 181 ]] || { sed -n '1,10p' "$WORK/big.err" >&2; gate_fail "by-value loop under ulimit -v 4 GiB: rc=$rc, want 181"; }
grep -q "madaros: arena full" "$WORK/big.err" || gate_fail "arena-full diagnostic missing"
echo "ulimit -v 4 GiB, ~9.8 GB by-value loop: exit 181 with the arena-full diagnostic"

gate_pass "heap reserved lazily (RSS ${rss_kb} KiB of ${vm_kb} KiB mapped), rlimit fallback, floor exit 184, checked wall 181"
echo "measured=yes"
exit 0
