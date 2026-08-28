#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  OUTPUT_ROOT=/tmp/adhd200_pcp_pilot24_bootstrap scripts/research/adhd200_pcp_pilot24_reproduce.sh

This reproduces the bounded ADHD-200 PCP pilot-24 smoke used by the
computational-psychiatry framework handoff. It is intentionally small and
underpowered: its purpose is executable framework proof, not clinical evidence.

Optional environment:
  OUTPUT_ROOT              Default: /tmp/adhd200_pcp_pilot24_bootstrap
  MAX_SUBJECTS             Default: 24
  NULL_PERMUTATIONS        Default: 1
  GENERIC_BASELINE_MODELS  Default: gru_reservoir,trained_rnn
  GENERIC_BASELINE_SEEDS   Default: 55555,11111
  GENERIC_TRAINED_EPOCHS   Default: 4
  GLOBAL_TRAIN_EPOCHS      Default: 2
  OCT_TRAIN_EPOCHS         Default: 2
  H_TRAIN_EPOCHS           Default: 2
  SOUNIO_SOUC_ENGINE       Default: lean_single
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/adhd200_pcp_pilot24_bootstrap}"
MAX_SUBJECTS="${MAX_SUBJECTS:-24}"
NULL_PERMUTATIONS="${NULL_PERMUTATIONS:-1}"
GENERIC_BASELINE_MODELS="${GENERIC_BASELINE_MODELS:-gru_reservoir,trained_rnn}"
GENERIC_BASELINE_SEEDS="${GENERIC_BASELINE_SEEDS:-55555,11111}"
GENERIC_TRAINED_EPOCHS="${GENERIC_TRAINED_EPOCHS:-4}"
GLOBAL_TRAIN_EPOCHS="${GLOBAL_TRAIN_EPOCHS:-2}"
OCT_TRAIN_EPOCHS="${OCT_TRAIN_EPOCHS:-2}"
H_TRAIN_EPOCHS="${H_TRAIN_EPOCHS:-2}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

python3 "$repo_root/scripts/research/adhd200_s3_bootstrap.py" \
  --output-dir "$OUTPUT_ROOT" \
  --max-subjects "$MAX_SUBJECTS" \
  --overwrite

python3 "$repo_root/scripts/research/adhd200_data_access_audit.py" \
  --phenotypic-csv "$OUTPUT_ROOT/adhd200_phenotypic.csv" \
  --roi-dir "$OUTPUT_ROOT/rois" \
  --output-dir "$OUTPUT_ROOT/access_audit" \
  --overwrite

PHENOTYPIC_CSV="$OUTPUT_ROOT/adhd200_phenotypic.csv" \
ROI_DIR="$OUTPUT_ROOT/rois" \
OUTPUT_DIR="$OUTPUT_ROOT/pilot_real24" \
NULL_PERMUTATIONS="$NULL_PERMUTATIONS" \
GENERIC_BASELINE_MODELS="$GENERIC_BASELINE_MODELS" \
GENERIC_BASELINE_SEEDS="$GENERIC_BASELINE_SEEDS" \
GENERIC_TRAINED_EPOCHS="$GENERIC_TRAINED_EPOCHS" \
GLOBAL_TRAIN_EPOCHS="$GLOBAL_TRAIN_EPOCHS" \
OCT_TRAIN_EPOCHS="$OCT_TRAIN_EPOCHS" \
H_TRAIN_EPOCHS="$H_TRAIN_EPOCHS" \
SOUNIO_SOUC_ENGINE="$SOUNIO_SOUC_ENGINE" \
"$repo_root/scripts/research/adhd200_dimensional_pilot_smoke.sh"

printf 'ADHD-200 PCP pilot-24 reproduction complete: %s\n' "$OUTPUT_ROOT/pilot_real24"
