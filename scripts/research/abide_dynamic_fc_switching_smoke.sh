#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$(mktemp -d /tmp/abide-dynamic-fc-smoke.XXXXXX)}"
CACHE_DIR="${OUT_ROOT}/cache"
RESULT_DIR="${OUT_ROOT}/results"

mkdir -p "${CACHE_DIR}"

python3 - "${CACHE_DIR}" <<'PY'
import csv
import math
import sys
from pathlib import Path

cache = Path(sys.argv[1])
subjects = []
sites = ["SITE_A", "SITE_B", "SITE_C"]
for site_idx, site in enumerate(sites):
    for local_idx in range(2):
        global_idx = site_idx * 2 + local_idx
        sub_id = f"{50000 + global_idx}"
        file_id = f"{site}_{sub_id}"
        dx = "1" if local_idx == 0 else "2"
        subjects.append({"SUB_ID": sub_id, "FILE_ID": file_id, "SITE_ID": site, "DX_GROUP": dx})
        rows = []
        for t in range(80):
            phase = (t // 20) % 2
            values = []
            for roi in range(8):
                base = math.sin(0.17 * t + 0.31 * roi + 0.13 * global_idx)
                if phase == 0:
                    coupled = base + 0.25 * math.sin(0.11 * t + site_idx)
                else:
                    coupled = ((-1) ** roi) * base + 0.25 * math.cos(0.07 * t + local_idx)
                values.append(coupled + 0.01 * (global_idx + 1) * (roi + 1))
            rows.append(values)
        with (cache / f"{file_id}_rois_cc200.1D").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(" ".join(f"{value:.8f}" for value in row) + "\n")

with (cache / "phenotypic.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["SUB_ID", "FILE_ID", "SITE_ID", "DX_GROUP"])
    writer.writeheader()
    for row in subjects:
        writer.writerow(row)
PY

python3 "${ROOT}/scripts/research/abide_dynamic_fc_switching_target.py" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${RESULT_DIR}" \
  --window-tr 20 \
  --step-tr 5 \
  --min-timepoints 60 \
  --min-windows 10 \
  --pca-components 4 \
  --k 3 \
  --sensitivity-k 2,4 \
  --min-subjects 6 \
  --min-sites 3 \
  --min-state-occupancy-frac 0.01 \
  --min-switch-event-frac 0.01 \
  --max-zero-switch-subject-frac 1.0 \
  --max-window-switch-abs-corr 1.0

python3 "${ROOT}/scripts/research/abide_dynamic_fc_switching_gate.py" \
  --window-table "${RESULT_DIR}/dynamic_fc_window_table.tsv" \
  --output-dir "${RESULT_DIR}/switching_gate" \
  --epochs 120 \
  --lr 0.04 \
  --l2 0.01 \
  --null-permutations 3 \
  --min-test-events 6 \
  --min-test-positive-events 1

python3 "${ROOT}/scripts/research/abide_dynamic_fc_switching_decision_gate.py" \
  --target-audit-json "${RESULT_DIR}/dynamic_fc_target_audit.json" \
  --gate-json "${RESULT_DIR}/switching_gate/dynamic_fc_switching_gate.json" \
  --gate-summary "${RESULT_DIR}/switching_gate/dynamic_fc_switching_gate_summary.tsv" \
  --output-dir "${RESULT_DIR}/switching_decision" \
  --min-decision-subjects 50 \
  --min-decision-null-permutations 20

test -s "${RESULT_DIR}/dynamic_fc_window_table.tsv"
test -s "${RESULT_DIR}/dynamic_fc_subject_summary.tsv"
test -s "${RESULT_DIR}/dynamic_fc_target_audit.json"
test -s "${RESULT_DIR}/dynamic_fc_target_audit.md"
test -s "${RESULT_DIR}/switching_gate/dynamic_fc_switching_gate.json"
test -s "${RESULT_DIR}/switching_gate/dynamic_fc_switching_gate_summary.tsv"
test -s "${RESULT_DIR}/switching_decision/dynamic_fc_switching_decision.json"

python3 - "${RESULT_DIR}/dynamic_fc_target_audit.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["status"] == "pass", payload
assert payload["subject_count"] == 6, payload
assert payload["site_count"] == 3, payload
assert payload["primary_test_switch_event_denominator"] > 0, payload
print("ABIDE_DYNAMIC_FC_SWITCHING_SMOKE_PASS")
PY

python3 - "${RESULT_DIR}/switching_gate/dynamic_fc_switching_gate.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["verdict"] == "TRAINED_O_SSM_GATE_EXECUTED_NO_PROMOTION", payload
assert payload["event_rows"] > 0, payload
assert payload["summary"], payload
models = {row["model"] for row in payload["summary"]}
assert {"base_rate", "gru_reservoir", "hssm_reservoir", "ossm_reservoir", "trained_hssm", "trained_ossm"} <= models, payload
ossm = next(row for row in payload["summary"] if row["model"] == "ossm_reservoir")
assert float(ossm["null_permutations_mean"]) == 3.0, payload
print("ABIDE_DYNAMIC_FC_SWITCHING_GATE_SMOKE_PASS")
PY

python3 - "${RESULT_DIR}/switching_decision/dynamic_fc_switching_decision.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["overall_verdict"] == "UNDERCONTROLLED_LOW_POWER_OR_SMOKE_SPLIT", payload
assert any("subject_count" in reason for reason in payload["reasons"]), payload
assert payload["target_summary"]["split_policy"] == "leave_one_site_out", payload
assert any("null_permutations" in reason for reason in payload["reasons"]), payload
print("ABIDE_DYNAMIC_FC_SWITCHING_DECISION_SMOKE_PASS")
PY

echo "outputs: ${RESULT_DIR}"
