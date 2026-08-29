#!/usr/bin/env bash
# Phase J epistemic-PTX confidence-gate -- drift oracle.
#
# Verifies the genuinely first compile-time confidence enforcement at GPU
# code-emission time: a kernel declares ; conf_budget=<N> in K-AXI; the
# K-AXI→PTX driver checks it against --min-conf=<N> policy; on failure
# the kernel body is replaced by a CONF_REJECT comment and ptxas is never
# invoked.
#
# Three byte-identity assertions:
#
#   1. conf_reject_demo + --min-conf 500  →  refusal-PTX (budget=300 < 500)
#   2. conf_pass_demo   + --min-conf 500  →  normal kernel (budget=950 ≥ 500)
#   3. conf_reject_demo + no flag         →  normal kernel (gate disabled,
#                                            body identical to conf_pass_demo)
#
# Exit 0 on PASS; nonzero on FAIL. Captured fixtures live at
# tests/golden/kaxi_ptx/f64_epistemic_gate/.

ulimit -s unlimited 2>/dev/null || true
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GOLDEN_DIR="tests/golden/kaxi_ptx/f64_epistemic_gate"

if [[ ! -d "$GOLDEN_DIR" ]]; then
    echo "FAIL: golden dir $GOLDEN_DIR missing" >&2
    exit 2
fi

# Each entry: <pattern>|<flags>|<golden basename>
CASES=(
    "conf_reject_demo|--min-conf 500|conf_reject_min500.ptx"
    "conf_pass_demo|--min-conf 500|conf_pass_min500.ptx"
    "conf_reject_demo||conf_reject_no_gate.ptx"
)

PASS=0
FAIL=0
FIRST_FAILURES=()

for entry in "${CASES[@]}"; do
    pattern="${entry%%|*}"
    rest="${entry#*|}"
    flags="${rest%%|*}"
    golden_name="${rest#*|}"

    golden_path="$GOLDEN_DIR/$golden_name"
    if [[ ! -f "$golden_path" ]]; then
        FAIL=$((FAIL + 1))
        FIRST_FAILURES+=("MISS  $pattern [$flags] (no golden $golden_name)")
        continue
    fi

    tmp_ptx="$(mktemp -t phase-j-gate.XXXXXX.ptx)"

    set +e
    if [[ -n "$flags" ]]; then
        ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$tmp_ptx" --no-ptxas $flags >/dev/null 2>&1
    else
        ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$tmp_ptx" --no-ptxas >/dev/null 2>&1
    fi
    rc=$?
    set -e

    if [[ "$rc" -ne 0 || ! -s "$tmp_ptx" ]]; then
        FAIL=$((FAIL + 1))
        FIRST_FAILURES+=("DROP  $pattern [$flags] (rc=$rc)")
        rm -f "$tmp_ptx"
        continue
    fi

    cur_sha="$(sha256sum "$tmp_ptx" | awk '{print $1}')"
    gold_sha="$(sha256sum "$golden_path" | awk '{print $1}')"

    if [[ "$cur_sha" == "$gold_sha" ]]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FIRST_FAILURES+=("DIFF  $pattern [$flags] (golden=$gold_sha current=$cur_sha)")
    fi

    rm -f "$tmp_ptx"
done

# Semantic invariants: refusal-PTX *must* contain the CONF_REJECT comment
# and *must not* contain a .visible .entry. Pass-PTX is the inverse.
reject_golden="$GOLDEN_DIR/conf_reject_min500.ptx"
if ! grep -q "// CONF_REJECT:" "$reject_golden"; then
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("INV   conf_reject_min500.ptx missing CONF_REJECT comment")
fi
if grep -q ".visible .entry" "$reject_golden"; then
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("INV   conf_reject_min500.ptx still contains .visible .entry -- gate did NOT suppress body")
fi
pass_golden="$GOLDEN_DIR/conf_pass_min500.ptx"
if ! grep -q ".visible .entry" "$pass_golden"; then
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("INV   conf_pass_min500.ptx missing .visible .entry -- pass case mis-emitted")
fi

echo "=== Phase J epistemic-PTX confidence-gate ==="
echo "  PASS:    $PASS"
echo "  FAIL:    $FAIL"
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
