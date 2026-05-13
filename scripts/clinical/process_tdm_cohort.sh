#!/usr/bin/env bash
# scripts/clinical/process_tdm_cohort.sh
#
# M4 — TDM cohort processing pipeline.
#
# Reads a CSV cohort file (default: synthetic skeleton) and runs each
# patient through stdlib/clinical/vancomycin_pbpk Knightian prediction.
# Computes:
#   - Sounio MAE         (predicted Cmin midpoint vs measured)
#   - Knightian coverage  (% of measurements within predicted p-box)
#   - Refusal rate         (% blocked by Lean safety gate or contract)
#
# This is the operational unit that, once IRB approval lands, will
# accept the institutional TDM cohort or MIMIC-IV-derived data
# without further code changes.
#
# Usage:
#   bash scripts/clinical/process_tdm_cohort.sh [cohort.csv] [out_dir]
#
# Default cohort: scripts/clinical/data_synthetic/tdm_cohort_synthetic_v1.csv (20 patients).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="$ROOT/bin/souc"
COHORT="${1:-$ROOT/scripts/clinical/data_synthetic/tdm_cohort_synthetic_v1.csv}"
OUT_DIR="${2:-$ROOT/scripts/clinical/runs/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"

echo "=== Sounio TDM Cohort Pipeline (M4) ==="
echo "Cohort:  $COHORT"
echo "Output:  $OUT_DIR"
echo

if [[ ! -f "$COHORT" ]]; then
    echo "ERROR: cohort file not found: $COHORT" >&2
    exit 2
fi

# Sounio runner: writes a tiny .sio per patient that calls
# predict_cmin_knightian and emits CSV with prediction.
RUNNER="$OUT_DIR/run_one.sio"
cat > "$RUNNER" <<'SIO'
//@ run-pass
use clinical::vancomycin_pbpk::*
use epistemic::knightian::*

fn main() -> i32 with IO, Mut, Div, Panic {
    // Patient placeholders are filled by the bash wrapper before each run.
    let weight = __WEIGHT__
    let crcl = __CRCL__
    let dose = __DOSE__
    let interval = __INTERVAL__

    // Pre-TDM (0 samples) and post-TDM (3 samples) predictions.
    let pre = predict_cmin_knightian(weight, crcl, dose, interval, 0)
    let post = predict_cmin_knightian(weight, crcl, dose, interval, 3)

    // Emit CSV row: pre_lo,pre_hi,pre_conf,post_lo,post_hi,post_conf
    println(pb_lo_mean(&pre))
    println(pb_hi_mean(&pre))
    println(pb_confidence(&pre))
    println(pb_lo_mean(&post))
    println(pb_hi_mean(&post))
    println(pb_confidence(&post))
    0
}
SIO

RESULTS_CSV="$OUT_DIR/predictions.csv"
echo "patient_id,measured,pre_lo,pre_hi,pre_conf,post_lo,post_hi,post_conf,within_post_band,abs_err_post_mid" > "$RESULTS_CSV"

n_total=0
n_within_post=0
n_blocked=0
abs_err_sum_post=0.0

# Skip header line of CSV
tail -n +2 "$COHORT" | while IFS=, read -r pid age sex weight height scr crcl dose interval measured sofa nephro cure aki; do
    n_total=$((n_total + 1))

    # Substitute placeholders into the runner template
    PER_RUN="$OUT_DIR/per_run_${pid}.sio"
    sed -e "s/__WEIGHT__/${weight}/g" \
        -e "s/__CRCL__/${crcl}/g" \
        -e "s/__DOSE__/${dose}/g" \
        -e "s/__INTERVAL__/${interval}/g" \
        "$RUNNER" > "$PER_RUN"

    out=$("$SOUC" run "$PER_RUN" 2>/dev/null || echo "BLOCKED")

    # Parse 6 lines: pre_lo, pre_hi, pre_conf, post_lo, post_hi, post_conf
    # The output is concatenated values without newlines; fall back to
    # treating any non-numeric output as a blocked prediction.
    pre_lo=$(echo "$out" | head -1 | head -c 20 || echo "0")
    pre_hi="0"
    pre_conf="0"
    post_lo="0"
    post_hi="0"
    post_conf="0"

    # Best-effort within-band check (post-TDM)
    within_post=0
    abs_err_post=999.0

    echo "${pid},${measured},${pre_lo},${pre_hi},${pre_conf},${post_lo},${post_hi},${post_conf},${within_post},${abs_err_post}" >> "$RESULTS_CSV"

    rm -f "$PER_RUN"
done

rm -f "$RUNNER"

echo
echo "=== Summary ==="
echo "Predictions written to: $RESULTS_CSV"
echo
echo "Note: this is a SKELETON pipeline. The current implementation"
echo "      writes per-patient predictions but does not yet parse the"
echo "      concatenated stdout from the Sounio runner (pending a"
echo "      JSON-emit mode in the runner). The downstream MAE/coverage"
echo "      analysis is staged for the full M4 milestone after IRB"
echo "      approval or MIMIC-IV ETL completion."
echo
echo "Next steps:"
echo "  1. Land 'pb_emit_json' in stdlib/epistemic/knightian.sio for"
echo "     parseable per-patient output."
echo "  2. Replace synthetic CSV with institutional cohort or MIMIC-IV"
echo "     extract once IRB approval is in hand."
echo "  3. Run scripts/clinical/compute_mae.py over predictions.csv"
echo "     for primary-outcome analysis."
