#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  PHENOTYPIC_CSV=/path/phenotypic.csv ROI_DIR=/path/roi OUTPUT_DIR=/path/out \
    scripts/research/adhd200_dimensional_pilot_smoke.sh

Required environment:
  PHENOTYPIC_CSV   ADHD-200 phenotypic CSV/TSV with dimensional subscales.
  ROI_DIR          Directory containing subject ROI time-series files.
  OUTPUT_DIR       Output directory for all pilot artifacts.

Preflight:
  Use scripts/research/adhd200_s3_bootstrap.py to create a bounded public
  PCP/FCP-INDI cache when local ADHD-200 inputs are absent.
  Use scripts/research/adhd200_data_access_audit.py before a real-data run to
  prove the local PHENOTYPIC_CSV/ROI_DIR pair is present. This smoke does not
  download ADHD-200 data and must not be run on ABIDE files as a substitute.

Optional environment:
  MAX_SUBJECTS             Default: 0 (all eligible).
  NULL_PERMUTATIONS        Default: 20.
  RUN_GENERIC_BASELINE     Default: 1.
  GENERIC_BASELINE_MODELS  Default: gru_reservoir,gru_reservoir_wide.
  GENERIC_BASELINE_SEEDS   Default: Sounio 20-seed schedule.
  GENERIC_TRAINED_EPOCHS   Default: 80.
  GENERIC_TRAINED_LR       Default: 0.01.
  PILOT_DECISION_MIN_GAP   Default: 0.05.
  PILOT_DECISION_ALPHA     Default: 0.10.
  GLOBAL_TRAIN_EPOCHS      Default: 6.
  OCT_TRAIN_EPOCHS         Default: 6.
  H_TRAIN_EPOCHS           Default: 6.
  TRAIN_FRACTION           Default: 1.0.
  FEATURE_MODE             Default: temporal_roi_block.
  PRIMARY_PHENOTYPES       Default: inattention,hyperactivity_impulsivity,adhd_total.
  SOUNIO_SOUC_ENGINE       Default: lean_single.
  SOUNIO_DIR               Default: current repository root.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUNIO_DIR="${SOUNIO_DIR:-$repo_root}"

: "${PHENOTYPIC_CSV:?PHENOTYPIC_CSV is required}"
: "${ROI_DIR:?ROI_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

MAX_SUBJECTS="${MAX_SUBJECTS:-0}"
NULL_PERMUTATIONS="${NULL_PERMUTATIONS:-20}"
RUN_GENERIC_BASELINE="${RUN_GENERIC_BASELINE:-1}"
GENERIC_BASELINE_MODELS="${GENERIC_BASELINE_MODELS:-gru_reservoir,gru_reservoir_wide}"
GENERIC_BASELINE_SEEDS="${GENERIC_BASELINE_SEEDS:-55555,11111,99887,22222,44444,66666,77777,88888,33333,12321,24680,13579,54321,67890,11223,44567,77889,99001,31415,27182}"
GENERIC_TRAINED_EPOCHS="${GENERIC_TRAINED_EPOCHS:-80}"
GENERIC_TRAINED_LR="${GENERIC_TRAINED_LR:-0.01}"
PILOT_DECISION_MIN_GAP="${PILOT_DECISION_MIN_GAP:-0.05}"
PILOT_DECISION_ALPHA="${PILOT_DECISION_ALPHA:-0.10}"
GLOBAL_TRAIN_EPOCHS="${GLOBAL_TRAIN_EPOCHS:-6}"
OCT_TRAIN_EPOCHS="${OCT_TRAIN_EPOCHS:-6}"
H_TRAIN_EPOCHS="${H_TRAIN_EPOCHS:-6}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
FEATURE_MODE="${FEATURE_MODE:-temporal_roi_block}"
PRIMARY_PHENOTYPES="${PRIMARY_PHENOTYPES:-inattention,hyperactivity_impulsivity,adhd_total}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

output_dir="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
manifest_dir="$output_dir/manifest"
run_dir="$output_dir/ossm_run"
probe_dir="$output_dir/dimensional_probe"
generic_dir="$output_dir/generic_recurrent_baseline"
decision_dir="$output_dir/pilot_decision"
align_dir="$output_dir/readout_alignment"
mkdir -p "$manifest_dir" "$run_dir" "$probe_dir" "$generic_dir" "$decision_dir" "$align_dir"

prepare_args=(
  "$SOUNIO_DIR/scripts/research/adhd200_prepare_manifest.py"
  --output-dir "$manifest_dir"
  --phenotypic-csv "$PHENOTYPIC_CSV"
  --roi-dir "$ROI_DIR"
  --write-ossm-view
  --feature-mode "$FEATURE_MODE"
  --primary-phenotypes "$PRIMARY_PHENOTYPES"
)
if [[ "$MAX_SUBJECTS" != "0" ]]; then
  prepare_args+=(--max-subjects "$MAX_SUBJECTS")
fi

python3 "${prepare_args[@]}" >"$manifest_dir/prepare.stdout.json"

python3 "$SOUNIO_DIR/scripts/research/adhd200_manifest_readiness_gate.py" \
  --manifest "$manifest_dir/adhd200_roi_manifest.tsv" \
  --primary-phenotypes "$PRIMARY_PHENOTYPES" \
  >"$manifest_dir/readiness_gate.json"

cp "$manifest_dir/adhd200_ossm_manifest.tsv" "$run_dir/abide_roi_manifest.tsv"
cat >"$run_dir/abide_run_config.tsv" <<CONFIG
train_fraction=$TRAIN_FRACTION
global_train_epochs=$GLOBAL_TRAIN_EPOCHS
oct_train_epochs=$OCT_TRAIN_EPOCHS
h_train_epochs=$H_TRAIN_EPOCHS
trace_hidden_state=1
CONFIG

(
  cd "$run_dir"
  "$SOUNIO_DIR/bin/souc" compile "$SOUNIO_DIR/examples/brain_ossm_abide.sio" -o brain_ossm_abide.elf \
    >compile.stdout.txt 2>compile.stderr.txt
  chmod +x brain_ossm_abide.elf
  set +e
  ./brain_ossm_abide.elf >brain_ossm_abide.raw.txt 2>brain_ossm_abide.stderr.txt
  rc=$?
  set -e
  printf '%s\n' "$rc" >brain_ossm_abide.rc
  if [[ "$rc" -ne 0 ]]; then
    tail -n 80 brain_ossm_abide.raw.txt >&2 || true
    tail -n 80 brain_ossm_abide.stderr.txt >&2 || true
    exit "$rc"
  fi
)

python3 "$SOUNIO_DIR/scripts/research/neurodyn_adhd_dimensional_state_probe.py" \
  --raw-output "$run_dir/brain_ossm_abide.raw.txt" \
  --manifest "$manifest_dir/adhd200_roi_manifest.tsv" \
  --ossm-manifest "$manifest_dir/adhd200_ossm_manifest.tsv" \
  --output-dir "$probe_dir" \
  --primary-phenotypes "$PRIMARY_PHENOTYPES" \
  --null-permutations "$NULL_PERMUTATIONS" \
  --overwrite \
  >"$probe_dir/state_probe.stdout.json"

if [[ "$RUN_GENERIC_BASELINE" != "0" ]]; then
  python3 "$SOUNIO_DIR/scripts/research/adhd200_generic_recurrent_baseline.py" \
    --manifest "$manifest_dir/adhd200_roi_manifest.tsv" \
    --output-dir "$generic_dir" \
    --models "$GENERIC_BASELINE_MODELS" \
    --seeds "$GENERIC_BASELINE_SEEDS" \
    --trained-epochs "$GENERIC_TRAINED_EPOCHS" \
    --trained-lr "$GENERIC_TRAINED_LR" \
    --primary-phenotypes "$PRIMARY_PHENOTYPES" \
    --null-permutations "$NULL_PERMUTATIONS" \
    --overwrite \
    >"$generic_dir/generic_recurrent_baseline.stdout.json"
else
  cat >"$generic_dir/generic_recurrent_baseline.SKIPPED.txt" <<'SKIPPED'
RUN_GENERIC_BASELINE=0; generic recurrent baseline was intentionally skipped.
Skipping this control keeps the artifact exploratory and undercontrolled.
SKIPPED
fi

python3 "$SOUNIO_DIR/scripts/research/adhd200_dimensional_pilot_decision_gate.py" \
  --readiness-json "$manifest_dir/readiness_gate.json" \
  --state-summary "$probe_dir/adhd_dimensional_state_probe_summary.tsv" \
  --generic-summary "$generic_dir/adhd_generic_recurrent_baseline_summary.tsv" \
  --output-dir "$decision_dir" \
  --primary-phenotypes "$PRIMARY_PHENOTYPES" \
  --min-gap "$PILOT_DECISION_MIN_GAP" \
  --alpha "$PILOT_DECISION_ALPHA" \
  --overwrite \
  >"$decision_dir/pilot_decision.stdout.json"

if grep -q '^READOUT_ALIGN_TRACE' "$run_dir/brain_ossm_abide.raw.txt"; then
  python3 "$SOUNIO_DIR/scripts/research/neurodyn_readout_alignment_trace_summary.py" \
    --raw-output "$run_dir/brain_ossm_abide.raw.txt" \
    --output-dir "$align_dir" \
    --overwrite \
    >"$align_dir/readout_alignment.stdout.json"
else
  cat >"$align_dir/readout_alignment_trace_summary.SKIPPED.txt" <<'SKIPPED'
READOUT_ALIGN_TRACE rows were not present in this pilot output.
The dimensional pilot only requires STATE_TRACE; readout alignment remains a separate optional audit.
SKIPPED
fi

(
  cd "$output_dir"
  find manifest ossm_run dimensional_probe generic_recurrent_baseline pilot_decision readout_alignment -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
)

cat >"$output_dir/PILOT_SUMMARY.md" <<SUMMARY
# ADHD-200 Dimensional O-SSM Pilot Smoke

Claim boundary: instrumentation and site-aware pilot only; no diagnostic, biomarker, treatment, mechanism, or O-SSM superiority claim.

- Phenotypic source: $PHENOTYPIC_CSV
- ROI source: $ROI_DIR
- Feature mode: $FEATURE_MODE
- Primary phenotypes: $PRIMARY_PHENOTYPES
- Sounio engine: $SOUNIO_SOUC_ENGINE
- O-SSM compatibility labels: 1=ADHD, 0=TD
- Manifest: manifest/adhd200_roi_manifest.tsv
- O-SSM compatibility view: manifest/adhd200_ossm_manifest.tsv
- Raw O-SSM output: ossm_run/brain_ossm_abide.raw.txt
- Dynamic state features: dimensional_probe/adhd_dimensional_dynamic_features.tsv
- Probe report: dimensional_probe/adhd_dimensional_state_probe.md
- Generic recurrent baseline: generic_recurrent_baseline/adhd_generic_recurrent_baseline.md
- Pilot decision gate: pilot_decision/adhd_dimensional_pilot_decision.md

This smoke uses the legacy ABIDE runner as an O-SSM execution surface. The rich ADHD manifest is the semantic source of truth.
The generic recurrent baseline defaults to GRU-style reservoir controls for pilot readiness. Include trained_rnn or trained_rnn_wide in GENERIC_BASELINE_MODELS to run the small NumPy-trained recurrent controls. A full S4/Transformer suite remains required before promotion-scale claims.
The pilot decision gate is an honesty filter only; it classifies whether controls suffice, whether the result is mixed, or whether a deeper real-data follow-up is justified.
SUMMARY

printf 'ADHD-200 dimensional pilot smoke complete: %s\n' "$output_dir"
