#!/usr/bin/env bash
# G1 expr-checking runaway-recursion regression gate.
#
# WHAT IT GUARDS: `--check` on a function with a NON-EMPTY body must NOT recurse
# without bound. The minimal repro is `fn main() { 1 }` (a single bare expression
# statement). On a broken build it SIGSEGVs (rc=139) under a bounded stack and its
# VmStk climbs into the GBs; on a fixed build it returns fast with VmStk peaking at
# KB–MB.
#
# ROOT CAUSE (see docs/audit/G1_LET_SPINE_CRASH_ROOTCAUSE_2026-06-01.md):
# runaway recursion in the expression-checking spine — NOT a frame-size overflow
# and NOT the `let` path. The source path for an int literal is linear AND guarded
# (checker_check_expr_inplace has a MAX_EXPR_DEPTH guard at check.sio:2490, and `1`
# hits the inline ExprIntLit leaf at 2496), so a runaway here is a `bin/souc`
# CODEGEN miscompile of a linear *mut function into a self-call, not a source bug.
#
# SAFETY: never runs at `ulimit -s unlimited` — that lets the recursion balloon to
# 100+ GB and OOM the pod. Caps the stack at 1 GB and kills after a few seconds.
#
# USAGE:  scripts/ci/g1_expr_recursion_gate.sh [path-to-mc.elf]
#   mc.elf defaults to .dbg/mc.elf; build it with:
#     bin/souc self-hosted/compiler/main.sio .dbg/mc.elf
#
# EXIT: 0 = PASS (bounded), 1 = FAIL (runaway recursion still present), 2 = setup error.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MC="${1:-$ROOT/.dbg/mc.elf}"

if [ ! -x "$MC" ]; then
    echo "SETUP ERROR: mc.elf not found/executable at: $MC" >&2
    echo "  build it: bin/souc self-hosted/compiler/main.sio .dbg/mc.elf && chmod +x .dbg/mc.elf" >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
printf 'fn main() {\n}\n'        > "$WORK/empty.sio"      # control: must PASS (rc=0)
printf 'fn main() {\n    1\n}\n' > "$WORK/bareexpr.sio"   # minimal repro: must PASS once fixed

FAIL=0

# 1) control — empty body must type-check cleanly under an 8 MB stack.
( ulimit -s 8192; "$MC" --check "$WORK/empty.sio" >/dev/null 2>&1 )
rc_empty=$?
if [ "$rc_empty" -ne 0 ]; then
    echo "FAIL: control fn main(){} returned rc=$rc_empty (expected 0) — build is broken" >&2
    FAIL=1
fi

# 2) the guarded repro — bounded 8 MB stack, must NOT SIGSEGV.
( ulimit -s 8192; "$MC" --check "$WORK/bareexpr.sio" >/dev/null 2>&1 )
rc_bare=$?

# 3) VmStk probe at 1 GB (NEVER unlimited) — peak must stay sub-GB.
peak_kb=0
( ulimit -s 1048576; "$MC" --check "$WORK/bareexpr.sio" >/dev/null 2>&1 ) &
pid=$!
for _ in $(seq 1 40); do
    v=$(awk '/VmStk/{print $2}' "/proc/$pid/status" 2>/dev/null)
    [ -n "${v:-}" ] && [ "$v" -gt "$peak_kb" ] && peak_kb=$v
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
done
kill -9 "$pid" 2>/dev/null
wait "$pid" 2>/dev/null

# Threshold: a healthy check of `1` peaks at KB–MB. >256 MB stack = runaway recursion.
PEAK_LIMIT_KB=262144
if [ "$rc_bare" -eq 139 ] || [ "$peak_kb" -gt "$PEAK_LIMIT_KB" ]; then
    echo "FAIL: fn main(){1} --check runaway recursion present (rc=$rc_bare, VmStk peak=${peak_kb}kB)" >&2
    echo "      see docs/audit/G1_LET_SPINE_CRASH_ROOTCAUSE_2026-06-01.md" >&2
    FAIL=1
else
    echo "PASS: fn main(){1} --check bounded (rc=$rc_bare, VmStk peak=${peak_kb}kB)"
fi

exit $FAIL
