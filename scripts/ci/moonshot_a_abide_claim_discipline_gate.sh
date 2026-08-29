#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ANALYSIS_JSON="${SOUNIO_MOONSHOT_A_ABIDE_ANALYSIS_JSON:?set SOUNIO_MOONSHOT_A_ABIDE_ANALYSIS_JSON}"
F32_ARTIFACT="${SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT:?set SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT}"

python3 - "$ANALYSIS_JSON" "$F32_ARTIFACT" <<'PY'
import json
import pathlib
import re
import sys

analysis = json.loads(pathlib.Path(sys.argv[1]).read_text())
f32 = json.loads(pathlib.Path(sys.argv[2]).read_text())
docs = [
    pathlib.Path("docs/research/moonshot-a-orc-implementation-decision.md"),
    pathlib.Path("docs/research/moonshot-a-slurm-blocker-handoff.md"),
    pathlib.Path("docs/research/moonshot-a-abide-f32-cohort-analysis.md"),
    pathlib.Path("docs/research/moonshot-a-abide-transport-conditioned-orc.md"),
]
required = {
    "analysis_status": analysis.get("status") == "PASS_COHORT_ANALYSIS_READY",
    "analysis_joined": analysis.get("cohort", {}).get("joined_subjects") == 1034,
    "analysis_no_subject_mismatch": not analysis.get("cohort", {}).get("scalar_only_subjects")
    and not analysis.get("cohort", {}).get("f32_only_subjects"),
    "analysis_variance_clean": analysis.get("variance", {}).get("inf_count") == 0
    and analysis.get("variance", {}).get("nan_count") == 0
    and analysis.get("variance", {}).get("negative_count") == 0
    and analysis.get("variance", {}).get("bad_first_index_count") == 0,
    "f32_status": f32.get("status") == "pass",
    "f32_limit_zero": f32.get("campaign", {}).get("limit") == "0",
    "f32_subjects": f32.get("campaign", {}).get("subjects_reported") == 1034
    and f32.get("campaign", {}).get("expected_subjects") == 1034,
}
failures = [k for k, ok in required.items() if not ok]

required_phrases = [
    "accepted 1034-subject scalar baseline",
    "not the raw 1035-file directory",
    "not a confirmatory biomarker claim",
    "not calibrated coverage",
    "does not claim clinical utility",
]
combined = "\n".join(p.read_text() for p in docs)
missing_phrases = [p for p in required_phrases if p not in combined]

forbidden_patterns = [
    (r"\b1035-subject\b", "raw_1035_subject_claim"),
    (r"\bis a raw 1035-file campaign\b", "raw_1035_campaign_claim"),
    (r"\bconfirmatory biomarker\b(?!\s+claim)", "confirmatory_biomarker_unqualified"),
    (r"\bclinical utility\b(?!,\s+diagnosis,\s+or\s+external validation)", "clinical_utility_unqualified"),
    (r"\bdiagnostic result\b(?!\.)", "diagnostic_result_unqualified"),
    (r"\bis calibrated coverage\b", "calibrated_coverage_unqualified"),
]
forbidden_hits = []
for path in docs:
    text = path.read_text()
    for pattern, name in forbidden_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            forbidden_hits.append({"file": str(path), "name": name, "pattern": pattern})

if failures or missing_phrases or forbidden_hits:
    print(json.dumps({
        "status": "FAIL",
        "failed_requirements": failures,
        "missing_required_phrases": missing_phrases,
        "forbidden_hits": forbidden_hits,
    }, indent=2, sort_keys=True))
    raise SystemExit(1)

print(json.dumps({
    "status": "PASS_CLAIM_DISCIPLINE",
    "checked_docs": [str(p) for p in docs],
    "required": required,
    "required_phrases": required_phrases,
}, indent=2, sort_keys=True))
PY
