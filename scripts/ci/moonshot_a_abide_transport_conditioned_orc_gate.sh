#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MOONSHOT_A_ABIDE_TRANSPORT_CONDITIONED_DIR:-$(mktemp -d /tmp/moonshot-a-abide-transport-conditioned.XXXXXX)}"
mkdir -p "$OUT_DIR"

F32_SUMMARY="${SOUNIO_MOONSHOT_A_F32_COHORT_SUMMARY:?set SOUNIO_MOONSHOT_A_F32_COHORT_SUMMARY}"
F32_ARTIFACT="${SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT:?set SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT}"
SCALAR_SUMMARY="${SOUNIO_MOONSHOT_A_SCALAR_COHORT_SUMMARY:-artifacts/research/abide_orc/cohort_summary.tsv}"
CLASSES="${SOUNIO_MOONSHOT_A_TRANSPORT_CONDITIONED_CLASSES:-8}"

TRANSPORT_DIR="$OUT_DIR/transport_168"
ANALYSIS_DIR="$OUT_DIR/f32_analysis"
OUT_JSON="$OUT_DIR/moonshot_a_abide_transport_conditioned_orc.v1.json"
OUT_TSV="$OUT_DIR/moonshot_a_abide_transport_conditioned_orc.tsv"

python3 -m py_compile \
  scripts/research/moonshot_a_abide_transport_conditioned_orc.py \
  scripts/research/moonshot_a_abide_f32_cohort_analysis.py \
  scripts/research/transport_168_modulation.py
bash -n \
  scripts/ci/moonshot_a_abide_transport_conditioned_orc_gate.sh \
  scripts/ci/moonshot_a_abide_f32_cohort_analysis_gate.sh \
  scripts/ci/moonshot_a_abide_claim_discipline_gate.sh \
  scripts/ci/moonshot_a_transport_168_modulation_gate.sh

SOUNIO_MOONSHOT_A_TRANSPORT_168_DIR="$TRANSPORT_DIR" \
  bash scripts/ci/moonshot_a_transport_168_modulation_gate.sh

SOUNIO_MOONSHOT_A_ABIDE_F32_ANALYSIS_DIR="$ANALYSIS_DIR" \
SOUNIO_MOONSHOT_A_SCALAR_COHORT_SUMMARY="$SCALAR_SUMMARY" \
SOUNIO_MOONSHOT_A_F32_COHORT_SUMMARY="$F32_SUMMARY" \
SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT="$F32_ARTIFACT" \
  bash scripts/ci/moonshot_a_abide_f32_cohort_analysis_gate.sh

SOUNIO_MOONSHOT_A_ABIDE_ANALYSIS_JSON="$ANALYSIS_DIR/moonshot_a_abide_f32_cohort_analysis.v1.json" \
SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT="$F32_ARTIFACT" \
  bash scripts/ci/moonshot_a_abide_claim_discipline_gate.sh >/tmp/moonshot_a_abide_claim_discipline_gate.out

python3 scripts/research/moonshot_a_abide_transport_conditioned_orc.py \
  --scalar-summary "$SCALAR_SUMMARY" \
  --f32-summary "$F32_SUMMARY" \
  --f32-artifact "$F32_ARTIFACT" \
  --transport-artifact "$TRANSPORT_DIR/synthetic_subspace.json" \
  --classes "$CLASSES" \
  --out-json "$OUT_JSON" \
  --out-tsv "$OUT_TSV"

python3 - "$OUT_JSON" "$OUT_TSV" "$CLASSES" <<'PY'
import csv
import json
import sys

doc = json.load(open(sys.argv[1]))
rows = list(csv.DictReader(open(sys.argv[2]), delimiter="\t"))
classes = int(sys.argv[3])
assert doc["status"] == "PASS_TRANSPORT_CONDITIONED_ORC_READY"
assert doc["cohort"]["joined_subjects"] == 1034
assert doc["cohort"]["selected_classes"] == classes
assert doc["cohort"]["feature_rows"] == 1034 * classes
assert len(rows) == 1034 * classes
assert doc["transport_conditioned"]["delta_abs_max"] > 1e-5
assert doc["transport_conditioned"]["delta_abs_max_vs_scalar_f32_mean_abs_ratio"] > 1.0
assert doc["variance"]["interpretation"] == "finite_overflow_bound_not_calibrated_coverage"
for phrase in (
    "descriptive_transport_conditioned_feature_package_not_confirmatory_biomarker_claim",
    "uses_transport_168_runtime_prototype_not_final_zd_surgery_semantics",
    "cohort_scope_is_accepted_1034_subject_scalar_baseline_not_raw_1035_directory",
    "does_not_claim_clinical_utility_diagnosis_or_external_validation",
):
    assert phrase in doc["boundaries"]
print(
    "moonshot_a_abide_transport_conditioned_orc_gate: PASS "
    f"rows={len(rows)} classes={classes} "
    f"transport_ratio={doc['transport_conditioned']['delta_abs_max_vs_scalar_f32_mean_abs_ratio']:.6g} "
    f"artifact={sys.argv[1]}"
)
PY

echo "moonshot_a_abide_transport_conditioned_orc_gate: out=$OUT_DIR"
