#!/usr/bin/env bash
# Phase B deep-dive -- sedenion ZD empirical foundation gate.
#
# Compiles and runs tests/math/test_zd_deep_dive.sio:
#   B.4 enumerate primitive ZDs in {1..15}
#   B.5 h-sweep partition stability
#   B.6 multi-invariant equivalence spectrum
#   B.7 iterated trajectory drift between different-z pairs
#
# The test asserts ALL PASS as long as it runs cleanly; that string alone is
# not evidence, so this gate additionally pins the headline B.4 census. The
# synthesis block at the end is the durable artifact downstream readers (and
# future butterfly-shapers) consume.
#
# Reference baseline (count corrected 2026-08-23; see ZD_EXPECTED below):
#   - 84 primitive ZDs enumerated
#   - trivial dedup (same-z): structural, 2 per z, h-stable
#   - cross-z dedup: h-coincidental; diverges ~2x per iterated step
#   - recommendation: z-canonical hash, not orbit-class metadata

ulimit -s unlimited 2>/dev/null || true
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="$ROOT_DIR/bin/souc-linux-x86_64"
[[ -x "$SOUC_BIN" ]] || { echo "FAIL: $SOUC_BIN missing" >&2; exit 2; }
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

TEST_SRC="tests/math/test_zd_deep_dive.sio"
[[ -f "$TEST_SRC" ]] || { echo "FAIL: $TEST_SRC missing" >&2; exit 2; }

OUT_ELF="$(mktemp -t zd-deep.XXXXXX.elf)"
OUT_LOG="$(mktemp -t zd-deep.XXXXXX.log)"
trap 'rm -f "$OUT_ELF" "$OUT_LOG"' EXIT INT TERM

if ! "$SOUC_BIN" "$TEST_SRC" "$OUT_ELF" >/dev/null 2>&1; then
    echo "FAIL: compile $TEST_SRC failed" >&2
    "$SOUC_BIN" "$TEST_SRC" "$OUT_ELF" >&2 || true
    exit 1
fi
chmod +x "$OUT_ELF"

set +e
"$OUT_ELF" >"$OUT_LOG" 2>&1
rc=$?
set -e

echo "=== Phase B deep-dive findings ==="
cat "$OUT_LOG"
echo "=== end ==="

if [[ "$rc" -ne 0 ]]; then echo "FAIL: test exited $rc" >&2; exit 1; fi
grep -q "ALL PASS" "$OUT_LOG" || { echo "FAIL: 'ALL PASS' missing" >&2; exit 1; }
grep -q "Synthesis" "$OUT_LOG" || { echo "FAIL: synthesis block missing" >&2; exit 1; }

# Assert the headline number, not just the "ALL PASS" string. B.4 enumerates
# z = e_i+e_j against w = e_k-e_l over i<j, k<l in 1..15 and must find exactly
# 84 zero divisors. 84 is not this run's output taken as the baseline: it is
# machine-checked in formal/lean4/SounioZeroDivisorBridge.lean
# (`theorem prim_count_84 : validPrims.length = 84 := by native_decide`), it is
# what tests/math/test_zd_deep_dive.sio's own header cites from the literature,
# and it factors as the 42 distinct z values x 2 w-companions that B.5 reports.
# The "92" this header carried from the gate's first commit (2c853c3088) matched
# none of those and was never produced by the test, whose source has not changed
# since that same commit -- it was a wrong comment, not a regression.
ZD_EXPECTED=84

zd_count="$(grep -E '^  ZDs found:' "$OUT_LOG" | awk '{print $NF}' | tr -d '\n')"
if [[ ! "$zd_count" =~ ^[0-9]+$ ]]; then
    echo "FAIL: could not extract the B.4 ZD count from the test output" >&2
    exit 1
fi
if [[ "$zd_count" -ne "$ZD_EXPECTED" ]]; then
    echo "FAIL: B.4 enumerated $zd_count zero divisors, expected $ZD_EXPECTED" >&2
    exit 1
fi
echo "Phase B deep-dive: ZDs=$zd_count, $(grep -m1 recommendation "$OUT_LOG" | sed 's/^[[:space:]]*//')"

exit 0
