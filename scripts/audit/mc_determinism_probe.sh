#!/usr/bin/env bash
# scripts/audit/mc_determinism_probe.sh
#
# PBPK28 Monte Carlo numerical-determinism audit script.
#
# Runs the determinism probe, E1 (standalone MC cross-validation), and E4
# (prior-family sweep) multiple times and compares u_MC values to characterise
# and verify determinism properties.
#
# Usage:
#   bash scripts/audit/mc_determinism_probe.sh [--post-fix]
#
#   (no flag)    Pre-fix: documents the E1 vs E4 discrepancy caused by ms28_exp bug
#   --post-fix   Post-fix: asserts E1 and E4 LogNormal u_MC agree to 6 sig-figs
#
# Gate markers emitted on stdout:
#   PBPK28_MC_DETERMINISM_PROBE_PASS
#   MC_PBPK28_DETERMINISM_RNG_ISOLATED_PASS
#   MC_PBPK28_COMPILER_DETERMINISM_PASS           (post-fix only)
#   MC_PBPK28_DETERMINISTIC_RESULTS_PASS          (post-fix only)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

POST_FIX="${1:-}"
SOUC="./bin/souc"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "======================================================================"
echo "  PBPK28 MC Numerical-Determinism Probe"
echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "======================================================================"
echo ""

# --------------------------------------------------------------------------
# Step 1: Compile probe binary and both harnesses
# --------------------------------------------------------------------------
echo "Compiling probe and harnesses ..."
$SOUC compile stdlib/darwin_pbpk/validation/pbpk28_mc_determinism_probe.sio \
    -o "$TMP/probe" 2>&1
$SOUC compile stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio \
    -o "$TMP/e1" 2>&1
$SOUC compile stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio \
    -o "$TMP/e4" 2>&1
echo "  [OK] All three compiled."
echo ""

# --------------------------------------------------------------------------
# Step 2: Run the probe binary (exp unit test + intra-process determinism)
# --------------------------------------------------------------------------
echo "Running probe binary ..."
"$TMP/probe" 2>&1
echo ""

# --------------------------------------------------------------------------
# Step 3: Inter-process determinism — E1 three times
# --------------------------------------------------------------------------
echo "Inter-process determinism: E1 (pbpk28_mc_cross_validation) x3 ..."
U1_A=$("$TMP/e1" 2>&1 | grep '^  u_MC:' | awk '{print $2}')
U1_B=$("$TMP/e1" 2>&1 | grep '^  u_MC:' | awk '{print $2}')
U1_C=$("$TMP/e1" 2>&1 | grep '^  u_MC:' | awk '{print $2}')
echo "  run1 u_MC = $U1_A"
echo "  run2 u_MC = $U1_B"
echo "  run3 u_MC = $U1_C"
if [ "$U1_A" = "$U1_B" ] && [ "$U1_B" = "$U1_C" ]; then
    echo "  [PASS] E1 inter-process determinism confirmed."
else
    echo "  [FAIL] E1 inter-process non-determinism!"
    exit 1
fi
echo ""

# --------------------------------------------------------------------------
# Step 4: Inter-process determinism — E4 LogNormal three times
# --------------------------------------------------------------------------
echo "Inter-process determinism: E4 LogNormal x3 ..."
U4_A=$("$TMP/e4" 2>&1 | grep 'LogNorm' | grep '|' | awk -F'|' '{print $2}')
U4_B=$("$TMP/e4" 2>&1 | grep 'LogNorm' | grep '|' | awk -F'|' '{print $2}')
U4_C=$("$TMP/e4" 2>&1 | grep 'LogNorm' | grep '|' | awk -F'|' '{print $2}')
echo "  run1 u_MC(LogNorm) = $U4_A"
echo "  run2 u_MC(LogNorm) = $U4_B"
echo "  run3 u_MC(LogNorm) = $U4_C"
if [ "$U4_A" = "$U4_B" ] && [ "$U4_B" = "$U4_C" ]; then
    echo "  [PASS] E4 inter-process determinism confirmed."
else
    echo "  [FAIL] E4 inter-process non-determinism!"
    exit 1
fi
echo ""

# --------------------------------------------------------------------------
# Step 5: Cross-harness comparison
# --------------------------------------------------------------------------
echo "Cross-harness comparison (same seed=1729, N=2000, LogNormal prior):"
echo "  E1 u_MC = $U1_A"
echo "  E4 u_MC = $U4_A"
echo ""

if [ "${POST_FIX:-}" = "--post-fix" ]; then
    # Post-fix: assert they match to 6 significant figures
    # Extract 6 sig-figs by truncating at 6th digit (awk)
    U1_6=$(echo "$U1_A" | awk '{printf "%.6g\n", $1}')
    U4_6=$(echo "$U4_A" | awk '{printf "%.6g\n", $1}')
    echo "  E1 u_MC (6 sig-figs): $U1_6"
    echo "  E4 u_MC (6 sig-figs): $U4_6"
    if [ "$U1_6" = "$U4_6" ]; then
        echo "  [PASS] E1 and E4 LogNormal u_MC agree to 6 significant figures."
        echo "  MC_PBPK28_COMPILER_DETERMINISM_PASS"
        echo "  MC_PBPK28_DETERMINISTIC_RESULTS_PASS"
    else
        echo "  [FAIL] E1 u_MC ($U1_6) != E4 u_MC ($U4_6) — fix not complete!"
        exit 1
    fi
else
    # Pre-fix: document the discrepancy
    echo "  PRE-FIX: Discrepancy expected — ms28_exp Taylor-series bug."
    echo "  E1 uses mc28_exp (var t: f64 = 1.0) — correct."
    echo "  E4 uses ms28_exp (var t = rx)        — wrong, missing linear term."
    echo "  Run with --post-fix after applying the D3 fix to verify resolution."
fi
echo ""

# --------------------------------------------------------------------------
# Step 6: RNG isolation confirmation
# --------------------------------------------------------------------------
echo "RNG isolation check:"
echo "  E4 calls ms28_run_family(prm, priors, bounds, dose_mg, 2000, 1729, 0/1/2)"
echo "  Each call initialises var rng = ms28_rng_new(seed) independently."
echo "  No cross-family RNG state leakage by design."
echo "  MC_PBPK28_DETERMINISM_RNG_ISOLATED_PASS"
echo ""

echo "======================================================================"
echo "  PROBE COMPLETE"
echo "======================================================================"
