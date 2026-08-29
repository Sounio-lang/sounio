#!/usr/bin/env bash
# Lane 8b — Multi-drug confidence aggregation gate (Phase J extension).
#
# Phase J's per-kernel --min-conf gate operates on a single confidence;
# this gate verifies the three aggregation rules that combine N
# per-drug confidences into one number the existing --min-conf flag can
# act on:
#
#   1. worst_case   — min over drugs (most conservative)
#   2. rss          — root-sum-square of complements (ISO uncorrelated)
#   3. cov_weighted — covariance-aware (collapses to rss when ρ=I)
#
# Acceptance:
#   1. souc check on the aggregator stdlib module rc=0
#   2. compiled e2e test runs and produces exact stdout matching the
#      committed golden snapshot at
#      tests/golden/multi_drug_conf/aggregator_outputs.txt
#   3. e2e test internally exercises both ADMIT and REFUSE paths of
#      --min-conf gating against the aggregated confidence.
#
# Disjoint by design from Lane 1's tests/golden/kaxi_ptx/** ownership:
# this gate's golden lives under tests/golden/multi_drug_conf/ instead.
# This lane does NOT add any new K-AXI/PTX pattern (which would require
# editing self-hosted/gpu/kaxi_to_ptx.sio, claimed by Lane 1).  The
# aggregation rules are pure-Sounio CPU code that operate on the
# integer/float confidences already produced by Phase J.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
SRC_AGG="stdlib/darwin_pbpk/aggregate_confidence.sio"
SRC_TEST="tests/run-pass/multi_drug_aggregate_test.sio"
GOLDEN="tests/golden/multi_drug_conf/aggregator_outputs.txt"

PASS=0
FAIL=0
FIRST_FAILURES=()

note_fail() {
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("$1")
}

if [[ ! -x "$SOUC" ]]; then
    echo "FAIL: souc binary not found at $SOUC" >&2
    exit 2
fi
if [[ ! -f "$SRC_AGG" ]]; then
    echo "FAIL: aggregator source missing: $SRC_AGG" >&2
    exit 2
fi
if [[ ! -f "$SRC_TEST" ]]; then
    echo "FAIL: e2e test source missing: $SRC_TEST" >&2
    exit 2
fi
if [[ ! -f "$GOLDEN" ]]; then
    echo "FAIL: golden snapshot missing: $GOLDEN" >&2
    exit 2
fi

# (1) souc check on aggregator
if "$SOUC" check "$SRC_AGG" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc check failed: $SRC_AGG"
fi

# (2) souc check on e2e test
if "$SOUC" check "$SRC_TEST" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc check failed: $SRC_TEST"
fi

# (3) compile + run + diff against golden
TMP_BIN="$(mktemp -t multi_drug_aggregate.XXXXXX)"
TMP_OUT="$(mktemp -t multi_drug_aggregate.XXXXXX.out)"
trap 'rm -f "$TMP_BIN" "$TMP_OUT"' EXIT

if "$SOUC" compile "$SRC_TEST" -o "$TMP_BIN" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc compile failed: $SRC_TEST"
fi

if [[ -x "$TMP_BIN" ]]; then
    if "$TMP_BIN" >"$TMP_OUT" 2>&1; then
        PASS=$((PASS + 1))
    else
        note_fail "compiled test exited non-zero (rc=$?)"
    fi
else
    note_fail "compiled binary not executable: $TMP_BIN"
fi

if [[ -s "$TMP_OUT" ]]; then
    if diff -q "$GOLDEN" "$TMP_OUT" >/dev/null 2>&1; then
        PASS=$((PASS + 1))
    else
        note_fail "stdout differs from golden $GOLDEN"
        echo "  (diff -u $GOLDEN <runtime>):" >&2
        diff -u "$GOLDEN" "$TMP_OUT" | head -40 >&2 || true
    fi
fi

# (4) acceptance line check — the test script prints a single-line
# success marker as its very last line.  Catches regressions where the
# stdout snapshot drifts but the test silently still exits 0.
if tail -n1 "$TMP_OUT" 2>/dev/null | grep -qx "PASS multi_drug_aggregate"; then
    PASS=$((PASS + 1))
else
    note_fail "acceptance line 'PASS multi_drug_aggregate' not last in stdout"
fi

echo "=== Lane 8b multi-drug confidence aggregation gate ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [[ "${#FIRST_FAILURES[@]}" -gt 0 ]]; then
    echo "  first failures:"
    for f in "${FIRST_FAILURES[@]}"; do
        echo "    $f"
    done
fi

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
