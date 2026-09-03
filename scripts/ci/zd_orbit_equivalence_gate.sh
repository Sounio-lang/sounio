#!/usr/bin/env bash
# Phase B -- sedenion zero-divisor orbit-equivalence verification gate.
#
# Compiles and runs tests/math/test_zd_orbit_equivalence.sio, captures the
# findings block, and asserts the test exits 0 with "ALL PASS". The findings
# block (annihilating pair count, max delta, VERDICT) is preserved verbatim
# in CI output for downstream interpretation.
#
# This gate is *evidence-recording*, not pass/fail of the empirical claim
# -- disconfirmation of the "5 ZD pairs bit-identical MSE" memory note is
# itself a successful run. Only crashes / compile failures fail the gate.
#
# Reference: 2026-05-09 baseline finding -- 2/5 annihilate, max delta ~2 ULP.

ulimit -s unlimited 2>/dev/null || true
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="$ROOT_DIR/bin/souc-linux-x86_64"
if [[ ! -x "$SOUC_BIN" ]]; then
    echo "FAIL: compiler not found ($SOUC_BIN)" >&2
    exit 2
fi
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

TEST_SRC="tests/math/test_zd_orbit_equivalence.sio"
if [[ ! -f "$TEST_SRC" ]]; then
    echo "FAIL: $TEST_SRC missing" >&2
    exit 2
fi

OUT_ELF="$(mktemp -t zd-orbit.XXXXXX.elf)"
trap 'rm -f "$OUT_ELF"' EXIT INT TERM

# Raw-ELF form: souc <src> <out>. The wrapper subcommands (compile/check)
# are bash-only; we invoke the binary directly so this gate doesn't depend
# on the wrapper script's surface staying stable.
if ! "$SOUC_BIN" "$TEST_SRC" "$OUT_ELF" >/dev/null 2>&1; then
    echo "FAIL: compile $TEST_SRC failed" >&2
    "$SOUC_BIN" "$TEST_SRC" "$OUT_ELF" >&2 || true
    exit 1
fi
chmod +x "$OUT_ELF"

OUT_LOG="$(mktemp -t zd-orbit.XXXXXX.log)"
set +e
"$OUT_ELF" >"$OUT_LOG" 2>&1
rc=$?
set -e

echo "=== Phase B ZD orbit-equivalence findings ==="
cat "$OUT_LOG"
echo "=== end ==="

if [[ "$rc" -ne 0 ]]; then
    echo "FAIL: test exited $rc" >&2
    rm -f "$OUT_LOG"
    exit 1
fi
if ! grep -q "ALL PASS" "$OUT_LOG"; then
    echo "FAIL: 'ALL PASS' not in test output" >&2
    rm -f "$OUT_LOG"
    exit 1
fi
if ! grep -q "VERDICT:" "$OUT_LOG"; then
    echo "FAIL: 'VERDICT:' findings line missing" >&2
    rm -f "$OUT_LOG"
    exit 1
fi

# Surface VERDICT in gate summary for grep-ability in CI logs.
echo "Phase B verdict: $(grep -m1 VERDICT: "$OUT_LOG")"

rm -f "$OUT_LOG"
exit 0
