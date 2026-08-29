#!/usr/bin/env bash
# scripts/clinical/smoke_pipeline.sh
#
# M4 — End-to-end smoke test of the Sounio TDM pipeline.
#
# Runs each stage of the data → prediction pipeline against the
# default synthetic cohort and asserts each stage exits clean. This
# is a fast pre-flight check that the pipeline is operational; it
# does NOT compute clinical outcome metrics (those are in
# scripts/clinical/compute_mae.py and process_tdm_cohort.sh).
#
# Stages exercised:
#
#   1. Validator:    validate_cohort.py --strict against synthetic v2
#   2. PBPK module:  $SOUC check stdlib/clinical/vancomycin_pbpk.sio
#   3. Walley:       $SOUC check stdlib/epistemic/walley.sio
#   4. Klibanoff:    $SOUC check stdlib/epistemic/klibanoff.sio
#   5. Smoke run:    a tiny .sio program that calls
#                    predict_cmin_knightian on 3 distinct patients
#                    and asserts the predicted band [lo, hi] is
#                    non-degenerate (lo < hi).
#
# Usage:
#   bash scripts/clinical/smoke_pipeline.sh
#
# Exit codes:
#   0 — all stages passed
#   1 — validator failed
#   2 — PBPK / Walley / Klibanoff type-check failed
#   3 — smoke run failed
#
# Math-review: not applicable. This is an integration test, not a
# math claim. The Knightian / Walley / Klibanoff math content is
# already mech-checked by the Lean modules under formal/lean4/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="$ROOT/bin/souc"
COHORT="$ROOT/scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv"

cd "$ROOT"

echo "=== Sounio M4 Smoke Pipeline ==="
echo "Root:    $ROOT"
echo "Cohort:  $COHORT"
echo

# ─── Stage 1: cohort validator ───────────────────────────────────
echo "[1/5] cohort validator (strict)..."
if ! python3 scripts/clinical/validate_cohort.py "$COHORT" --strict \
        > /tmp/smoke_validate.log 2>&1; then
    cat /tmp/smoke_validate.log
    echo "SMOKE FAIL: cohort validator failed"
    exit 1
fi
tail -1 /tmp/smoke_validate.log

# ─── Stage 2: PBPK module type-check ─────────────────────────────
echo "[2/5] $SOUC check stdlib/clinical/vancomycin_pbpk.sio..."
if ! "$SOUC" check stdlib/clinical/vancomycin_pbpk.sio \
        > /tmp/smoke_pbpk.log 2>&1; then
    cat /tmp/smoke_pbpk.log
    echo "SMOKE FAIL: PBPK module failed type-check"
    exit 2
fi
echo "  OK"

# ─── Stage 3: Walley module type-check ───────────────────────────
echo "[3/5] $SOUC check stdlib/epistemic/walley.sio..."
if ! "$SOUC" check stdlib/epistemic/walley.sio \
        > /tmp/smoke_walley.log 2>&1; then
    cat /tmp/smoke_walley.log
    echo "SMOKE FAIL: Walley module failed type-check"
    exit 2
fi
echo "  OK"

# ─── Stage 4: Klibanoff module type-check ────────────────────────
echo "[4/5] $SOUC check stdlib/epistemic/klibanoff.sio..."
if ! "$SOUC" check stdlib/epistemic/klibanoff.sio \
        > /tmp/smoke_klibanoff.log 2>&1; then
    cat /tmp/smoke_klibanoff.log
    echo "SMOKE FAIL: Klibanoff module failed type-check"
    exit 2
fi
echo "  OK"

# ─── Stage 5: end-to-end prediction smoke ────────────────────────
echo "[5/5] end-to-end Knightian prediction (3 patients)..."

SMOKE_SIO="/tmp/sounio_smoke_pipeline.sio"
cat > "$SMOKE_SIO" <<'SIO'
//@ run-pass
use clinical::vancomycin_pbpk::*
use epistemic::knightian::*

fn main() -> i32 with IO, Mut, Div, Panic {
    let p1 = predict_cmin_knightian(70.0, 100.0, 1000.0, 12.0, 0)
    let lo1 = pb_lo_mean(&p1)
    let hi1 = pb_hi_mean(&p1)
    if hi1 <= lo1 { println("SMOKE FAIL P1"); assert(false) }
    println("P1 typical OK")

    let p2 = predict_cmin_knightian(50.0, 30.0, 500.0, 24.0, 0)
    let lo2 = pb_lo_mean(&p2)
    let hi2 = pb_hi_mean(&p2)
    if hi2 <= lo2 { println("SMOKE FAIL P2"); assert(false) }
    println("P2 low CrCl OK")

    let p3 = predict_cmin_knightian(95.0, 200.0, 1500.0, 8.0, 3)
    let lo3 = pb_lo_mean(&p3)
    let hi3 = pb_hi_mean(&p3)
    if hi3 <= lo3 { println("SMOKE FAIL P3"); assert(false) }
    println("P3 high CrCl OK")

    println("SMOKE OK 3/3")
    0
}
SIO

if ! "$SOUC" run "$SMOKE_SIO" > /tmp/smoke_run.log 2>&1; then
    cat /tmp/smoke_run.log
    echo "SMOKE FAIL: end-to-end prediction failed"
    exit 3
fi
tail -1 /tmp/smoke_run.log
rm -f "$SMOKE_SIO"

echo
echo "=== M4 SMOKE PIPELINE: ALL 5 STAGES PASSED ==="
echo
echo "Pipeline is operational. Next gating step: real-data ingest"
echo "via scripts/clinical/etl/ (MIMIC-IV) or institutional CSV."
echo "See docs/governance/data_access_credentials.md for credential"
echo "issuance flow."
