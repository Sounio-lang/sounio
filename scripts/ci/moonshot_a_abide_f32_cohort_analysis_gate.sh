#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MOONSHOT_A_ABIDE_F32_ANALYSIS_DIR:-$(mktemp -d /tmp/moonshot-a-abide-f32-analysis.XXXXXX)}"
OUT_JSON="$OUT_DIR/moonshot_a_abide_f32_cohort_analysis.v1.json"
SCALAR_SUMMARY="${SOUNIO_MOONSHOT_A_SCALAR_COHORT_SUMMARY:-artifacts/research/abide_orc/cohort_summary.tsv}"
F32_SUMMARY="${SOUNIO_MOONSHOT_A_F32_COHORT_SUMMARY:?set SOUNIO_MOONSHOT_A_F32_COHORT_SUMMARY}"
F32_ARTIFACT="${SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT:?set SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT}"

mkdir -p "$OUT_DIR"
python3 -m py_compile scripts/research/moonshot_a_abide_f32_cohort_analysis.py
python3 scripts/research/moonshot_a_abide_f32_cohort_analysis.py \
  --scalar-summary "$SCALAR_SUMMARY" \
  --f32-summary "$F32_SUMMARY" \
  --f32-artifact "$F32_ARTIFACT" \
  --out "$OUT_JSON"

python3 - "$OUT_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
doc = json.loads(path.read_text())
assert doc["status"] == "PASS_COHORT_ANALYSIS_READY"
assert doc["cohort"]["joined_subjects"] == 1034
assert doc["cohort"]["scalar_only_subjects"] == []
assert doc["cohort"]["f32_only_subjects"] == []
assert doc["variance"]["inf_count"] == 0
assert doc["variance"]["nan_count"] == 0
assert doc["variance"]["negative_count"] == 0
assert doc["variance"]["bad_first_index_count"] == 0
print(
    "moonshot_a_abide_f32_cohort_analysis_gate: PASS "
    f"joined={doc['cohort']['joined_subjects']} "
    f"mean_delta_abs_max={doc['scalar_vs_f32']['mean_delta_abs_max']:.6g} "
    f"artifact={path}"
)
PY
