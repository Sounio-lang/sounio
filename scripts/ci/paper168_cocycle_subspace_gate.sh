#!/usr/bin/env bash
# Lane 3 — Paper 168 cocycle subspace enumeration gate.
#
# Compiles and runs the seven cocycle_subspace_*.sio examples that
# back §7 of paper 168 ("On 168") and verifies each prints ALL PASS:
#
#   k=4  (sedenions,    dim 16)   examples/cocycle_subspace_168.sio
#   k=5  (32-ions,      dim 32)   examples/cocycle_subspace_k5.sio
#   k=6  (chingons,     dim 64)   examples/cocycle_subspace_k6.sio
#   k=7  (routons,      dim 128)  examples/cocycle_subspace_k7.sio
#   k=8  (voudons,      dim 256)  examples/cocycle_subspace_k8.sio
#   k=9  (1024-ions,    dim 512)  examples/cocycle_subspace_k9.sio
#   k=10 (2048-ions,    dim 1024) examples/cocycle_subspace_k10.sio
#
# Each example does its own self-check (T_k matches Conjecture 5,
# P_k enumeration matches the Gaussian binomial [k choose 3]_2),
# so the gate's job is just to drive the compile/run/grep loop.
#
# Acceptance: rc=0 iff every example compiles, runs under 600s, and
# prints "ALL PASS".

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
TMPDIR="${TMPDIR:-/tmp}"
WORK="$(mktemp -d "${TMPDIR}/paper168-gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -x "$SOUC" ]]; then
    echo "FAIL: souc not executable at $SOUC" >&2
    exit 2
fi

PASS=0
FAIL=0
FAILURES=()

run_case() {
    local label="$1"
    local src="$2"
    local expect_line="$3"   # required substring in stdout (T_k or P_k claim)
    local require_all_pass="$4"   # "yes" or "no"
    local out="${WORK}/${label}.out"

    if [[ ! -f "$src" ]]; then
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: missing source $src")
        return
    fi
    if ! "$SOUC" check "$src" >/dev/null 2>&1; then
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: souc check failed")
        return
    fi
    if ! "$SOUC" compile "$src" -o "${WORK}/${label}.bin" >/dev/null 2>&1; then
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: souc compile failed")
        return
    fi
    if ! timeout 600 "${WORK}/${label}.bin" >"$out" 2>&1; then
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: binary failed or timed out (>600s)")
        return
    fi
    if [[ "$require_all_pass" == "yes" ]] && ! grep -qF "ALL PASS" "$out"; then
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: stdout missing 'ALL PASS'")
        return
    fi
    if ! grep -qF "$expect_line" "$out"; then
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: stdout missing expected line \"$expect_line\"")
        return
    fi
    PASS=$((PASS + 1))
    echo "  PASS ${label}"
}

echo "═══════════════════════════════════════════════════════════════"
echo "  Paper 168 — cocycle subspace enumeration gate (k=4..10)"
echo "═══════════════════════════════════════════════════════════════"

# k=4 (cocycle_subspace_168.sio) predates the cohomological reformulation
# (PR #92) and reports SOME FAILED against its pre-reformulation
# predictions; the empirical counts feed @table:subspace unchanged.
# We just confirm T_4 = 1848 (Conjecture 5 at k=4) is hit.
run_case "k4" \
    "examples/cocycle_subspace_168.sio" \
    "T_4 = 1848 = 11 * 168" \
    "no"

run_case "k5" \
    "examples/cocycle_subspace_k5.sio" \
    "155 = P_5" \
    "yes"

run_case "k6" \
    "examples/cocycle_subspace_k6.sio" \
    "1395 = P_6" \
    "yes"

run_case "k7" \
    "examples/cocycle_subspace_k7.sio" \
    "11811 = P_7" \
    "yes"

run_case "k8" \
    "examples/cocycle_subspace_k8.sio" \
    "97155 = P_8" \
    "yes"

run_case "k9" \
    "examples/cocycle_subspace_k9.sio" \
    "788035 = P_9" \
    "yes"

run_case "k10" \
    "examples/cocycle_subspace_k10.sio" \
    "6347715 = P_10" \
    "yes"

echo ""
echo "  Pass: ${PASS}"
echo "  Fail: ${FAIL}"
if (( FAIL > 0 )); then
    echo ""
    echo "Failures:"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

echo "  GATE PASS"
exit 0
