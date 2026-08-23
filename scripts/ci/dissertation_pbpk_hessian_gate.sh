#!/usr/bin/env bash
# Lane 8a — Second-order Hessian GUM through PBPK14.
#
# Dissertation contribution #1 extension. Acceptance:
#
#   1. souc check on the Hessian module + on the unit test + on the
#      e2e test all rc=0.
#   2. The unit test (tests/stdlib/darwin_pbpk/hessian_correction_test.sio)
#      compiles, runs, and prints "PASS unit_quadratic_recovery" — proves
#      the FD math recovers analytic Hessian elements exactly for a
#      polynomial of degree 2.
#   3. The e2e test (tests/run-pass/dissertation_pbpk14_hessian.sio)
#      compiles, runs, and prints "PASS hessian_correction_reduces_residual"
#      — proves on a controlled non-linear synthetic endpoint that
#      |GUM_2nd − truth| < |GUM_1st − truth|.
#   4. The e2e test also emits a rapamycin AUC Hessian budget in CSV
#      form. print_f64's six fixed decimals floor the first-order
#      variance (~2.05e-7) to 0.000000, so the e2e test additionally
#      emits a tagged full-precision line (HESSBUD|var_o1_scaled_1e9=)
#      and the gate canonicalises that value into the CSV as %.6e
#      (pbpk28-parity pattern: raw values out of Sounio, CSV shaping in
#      the gate) before diffing bytewise against the committed golden
#      at benchmarks/pbpk/hessian_budget.csv — the Hessian propagator
#      is fully deterministic, so any drift, including sub-5e-7 drift
#      the fixed-point rows cannot show, means a real regression in the
#      FD math or the simulator. Refresh the golden from a real run
#      with HESSIAN_BUDGET_UPDATE_GOLDEN=1.
#
# Files this gate exercises:
#   stdlib/darwin_pbpk/epistemic_pbpk14_hessian.sio
#   tests/stdlib/darwin_pbpk/hessian_correction_test.sio
#   tests/run-pass/dissertation_pbpk14_hessian.sio
#   benchmarks/pbpk/hessian_budget.csv

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
SRC_HESS="stdlib/darwin_pbpk/epistemic_pbpk14_hessian.sio"
SRC_UNIT="tests/stdlib/darwin_pbpk/hessian_correction_test.sio"
SRC_E2E="tests/run-pass/dissertation_pbpk14_hessian.sio"
GOLDEN_CSV="benchmarks/pbpk/hessian_budget.csv"

PASS=0
FAIL=0
FIRST_FAILURES=()

note_fail() {
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("$1")
}

if [[ ! -x "$SOUC" ]]; then
    echo "FAIL: souc not executable at $SOUC" >&2
    exit 2
fi
for f in "$SRC_HESS" "$SRC_UNIT" "$SRC_E2E" "$GOLDEN_CSV"; do
    if [[ ! -f "$f" ]]; then
        echo "FAIL: required file missing: $f" >&2
        exit 2
    fi
done

# --- (a) souc check on unit test (transitively checks math::pure) ---
if "$SOUC" check "$SRC_UNIT" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc check failed: $SRC_UNIT"
fi

# --- (b) souc check on e2e test (transitively checks the Hessian module) ---
if "$SOUC" check "$SRC_E2E" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc check failed: $SRC_E2E"
fi

# --- (c) compile + run + grep PASS line of unit test ---
TMP_UNIT="$(mktemp -t hess_unit.XXXXXX)"
TMP_UNIT_OUT="$(mktemp -t hess_unit.XXXXXX.out)"
TMP_E2E="$(mktemp -t hess_e2e.XXXXXX)"
TMP_E2E_OUT="$(mktemp -t hess_e2e.XXXXXX.out)"
TMP_E2E_CSV="$(mktemp -t hess_e2e.XXXXXX.csv)"
TMP_E2E_SECTION="$(mktemp -t hess_e2e.XXXXXX.section)"
trap 'rm -f "$TMP_UNIT" "$TMP_UNIT_OUT" "$TMP_E2E" "$TMP_E2E_OUT" "$TMP_E2E_CSV" "$TMP_E2E_SECTION"' EXIT

if "$SOUC" compile "$SRC_UNIT" -o "$TMP_UNIT" >/dev/null 2>&1 && [[ -x "$TMP_UNIT" ]]; then
    if "$TMP_UNIT" >"$TMP_UNIT_OUT" 2>&1; then
        if tail -n1 "$TMP_UNIT_OUT" | grep -qx "PASS unit_quadratic_recovery"; then
            PASS=$((PASS + 1))
        else
            note_fail "unit test stdout last line != 'PASS unit_quadratic_recovery'"
        fi
    else
        note_fail "unit test exited non-zero"
    fi
else
    note_fail "souc compile failed: $SRC_UNIT"
fi

# --- (d) compile + run + grep PASS line of e2e test ---
if "$SOUC" compile "$SRC_E2E" -o "$TMP_E2E" >/dev/null 2>&1 && [[ -x "$TMP_E2E" ]]; then
    if "$TMP_E2E" >"$TMP_E2E_OUT" 2>&1; then
        if tail -n1 "$TMP_E2E_OUT" | grep -qx "PASS hessian_correction_reduces_residual"; then
            PASS=$((PASS + 1))
        else
            note_fail "e2e test stdout last line != 'PASS hessian_correction_reduces_residual'"
        fi
    else
        note_fail "e2e test exited non-zero"
    fi
else
    note_fail "souc compile failed: $SRC_E2E"
fi

# --- (e) extract budget section, canonicalise var_o1, diff against golden ---
# print_f64 emits six fixed decimals, so the budget row's var_o1 value
# (var_first_order, ~2.05e-7 (ng·h/mL)²) prints as 0.000000: the runtime's
# own row text can neither match a full-precision golden nor fail one. The
# e2e test therefore emits a tagged full-precision channel before the
# section header — HESSBUD|var_o1_scaled_1e9=<print_f64 of the value ×1e9>
# — and this step folds it back in: every budget row passes through
# unchanged except var_o1, re-emitted as %.6e of scaled/1e9. The golden is
# a mechanical capture of that canonical form; drift in var_first_order
# beyond ~0.5 ppm relative goes red.
awk '
    /^# rapamycin_hessian_budget_v1$/ { keep = 1 }
    keep && /^PASS / { keep = 0; next }
    keep { print }
' "$TMP_E2E_OUT" > "$TMP_E2E_SECTION"

RAW_SCALED="$(awk -F= '/^HESSBUD\|var_o1_scaled_1e9=/ { print $2 }' "$TMP_E2E_OUT")"
if [[ -z "$RAW_SCALED" ]]; then
    note_fail "e2e stdout carries no HESSBUD|var_o1_scaled_1e9 full-precision channel"
else
    if awk -F, -v OFS=, -v SCALED="$RAW_SCALED" '
        $1 == "var_o1" && NF >= 4 {
            if (SCALED + 0 <= 0.0) {
                printf "  var_o1 scaled channel is not positive: %s\n", SCALED > "/dev/stderr"
                exit 3
            }
            $4 = sprintf("%.6e", (SCALED + 0) / 1e9)
        }
        { print }
    ' "$TMP_E2E_SECTION" > "$TMP_E2E_CSV"; then
        if [[ "${HESSIAN_BUDGET_UPDATE_GOLDEN:-0}" == "1" ]]; then
            cp "$TMP_E2E_CSV" "$GOLDEN_CSV"
            echo "  golden refreshed from a real run: $GOLDEN_CSV" >&2
        fi
        if diff -q "$GOLDEN_CSV" "$TMP_E2E_CSV" >/dev/null 2>&1; then
            PASS=$((PASS + 1))
        else
            note_fail "rapamycin Hessian CSV differs from golden $GOLDEN_CSV"
            echo "  (diff -u $GOLDEN_CSV <runtime>):" >&2
            diff -u "$GOLDEN_CSV" "$TMP_E2E_CSV" | head -40 >&2 || true
        fi
    else
        note_fail "var_o1 canonicalisation failed (scaled channel: $RAW_SCALED)"
    fi
fi

echo "=== Lane 8a second-order Hessian GUM gate ==="
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
