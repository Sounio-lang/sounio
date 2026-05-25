#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MOONSHOT_A_ABIDE_COHORT_MANIFEST_DIR:-$(mktemp -d /tmp/moonshot-a-abide-cohort-manifest.XXXXXX)}"
OUT_JSON="$OUT_DIR/moonshot_a_abide_cohort_manifest.v1.json"
PHASE_STATUS="${SOUNIO_MOONSHOT_A_PHASE_STATUS_ARTIFACT:-}"

args=(--out "$OUT_JSON")
if [[ -n "$PHASE_STATUS" ]]; then
  args+=(--phase-status "$PHASE_STATUS")
fi

python3 scripts/research/moonshot_a_abide_cohort_manifest.py "${args[@]}"
python3 - "$OUT_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
doc = json.loads(path.read_text())
if doc.get("status") != "PASS_COHORT_BASELINE_READY":
    raise SystemExit(f"moonshot_a_abide_cohort_manifest_gate: FAIL status={doc.get('status')}")
cohort = doc["cohort_baseline"]
runtime = doc["runtime_baseline"]
if cohort["summary_subjects"] < 1000:
    raise SystemExit("moonshot_a_abide_cohort_manifest_gate: FAIL expected cohort-scale summary")
if runtime["scalar_status"] != "pass" or runtime["f32_epistemic_status"] != "pass":
    raise SystemExit("moonshot_a_abide_cohort_manifest_gate: FAIL missing accepted runtime baseline")
print(
    "moonshot_a_abide_cohort_manifest_gate: PASS "
    f"subjects={cohort['summary_subjects']} "
    f"labelled={cohort['labelled_subjects']} "
    f"mean_orc_d={cohort['asd_td_screen']['mean_orc_cohen_d']:.6g} "
    f"artifact={path}"
)
PY
