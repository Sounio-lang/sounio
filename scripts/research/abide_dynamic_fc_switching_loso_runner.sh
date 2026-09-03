#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$(mktemp -d /tmp/abide-dynamic-fc-loso.XXXXXX)}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/artifacts/research/abide}"
RESULT_DIR="${OUT_ROOT}/results"

MAX_SUBJECTS="${MAX_SUBJECTS:-60}"
ROI_LIMIT="${ROI_LIMIT:-40}"
WINDOW_TR="${WINDOW_TR:-30}"
STEP_TR="${STEP_TR:-6}"
MIN_TIMEPOINTS="${MIN_TIMEPOINTS:-90}"
MIN_WINDOWS="${MIN_WINDOWS:-10}"
PCA_COMPONENTS="${PCA_COMPONENTS:-8}"
K="${K:-4}"
SENSITIVITY_K="${SENSITIVITY_K:-3,5}"
TARGET_MIN_SUBJECTS="${TARGET_MIN_SUBJECTS:-50}"
TARGET_MIN_SITES="${TARGET_MIN_SITES:-5}"
TARGET_MIN_STATE_OCCUPANCY="${TARGET_MIN_STATE_OCCUPANCY:-0.005}"
TARGET_MIN_SWITCH_FRAC="${TARGET_MIN_SWITCH_FRAC:-0.001}"
TARGET_MAX_ZERO_SWITCH_FRAC="${TARGET_MAX_ZERO_SWITCH_FRAC:-1.0}"
TARGET_MAX_WINDOW_SWITCH_CORR="${TARGET_MAX_WINDOW_SWITCH_CORR:-1.0}"

GATE_EPOCHS="${GATE_EPOCHS:-80}"
GATE_LR="${GATE_LR:-0.04}"
GATE_L2="${GATE_L2:-0.01}"
GATE_NULL_PERMUTATIONS="${GATE_NULL_PERMUTATIONS:-20}"
GATE_MIN_TEST_EVENTS="${GATE_MIN_TEST_EVENTS:-2}"
GATE_MIN_TEST_POSITIVE_EVENTS="${GATE_MIN_TEST_POSITIVE_EVENTS:-0}"

DECISION_MIN_SUBJECTS="${DECISION_MIN_SUBJECTS:-50}"
DECISION_MIN_SITES="${DECISION_MIN_SITES:-5}"
DECISION_MIN_NULL_PERMUTATIONS="${DECISION_MIN_NULL_PERMUTATIONS:-20}"

RUN_SOUNIO_SMOKE="${RUN_SOUNIO_SMOKE:-0}"
SOUNIO_SMOKE_MAX_EVENTS="${SOUNIO_SMOKE_MAX_EVENTS:-96}"
SOUNIO_SMOKE_GLOBAL_EPOCHS="${SOUNIO_SMOKE_GLOBAL_EPOCHS:-1}"
SOUNIO_SMOKE_OCT_EPOCHS="${SOUNIO_SMOKE_OCT_EPOCHS:-1}"
SOUNIO_SMOKE_H_EPOCHS="${SOUNIO_SMOKE_H_EPOCHS:-1}"

python3 "${ROOT}/scripts/research/abide_dynamic_fc_switching_target.py" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${RESULT_DIR}" \
  --max-subjects "${MAX_SUBJECTS}" \
  --roi-limit "${ROI_LIMIT}" \
  --split-policy leave_one_site_out \
  --window-tr "${WINDOW_TR}" \
  --step-tr "${STEP_TR}" \
  --min-timepoints "${MIN_TIMEPOINTS}" \
  --min-windows "${MIN_WINDOWS}" \
  --pca-components "${PCA_COMPONENTS}" \
  --k "${K}" \
  --sensitivity-k "${SENSITIVITY_K}" \
  --min-subjects "${TARGET_MIN_SUBJECTS}" \
  --min-sites "${TARGET_MIN_SITES}" \
  --min-state-occupancy-frac "${TARGET_MIN_STATE_OCCUPANCY}" \
  --min-switch-event-frac "${TARGET_MIN_SWITCH_FRAC}" \
  --max-zero-switch-subject-frac "${TARGET_MAX_ZERO_SWITCH_FRAC}" \
  --max-window-switch-abs-corr "${TARGET_MAX_WINDOW_SWITCH_CORR}"

python3 "${ROOT}/scripts/research/abide_dynamic_fc_switching_gate.py" \
  --window-table "${RESULT_DIR}/dynamic_fc_window_table.tsv" \
  --output-dir "${RESULT_DIR}/switching_gate" \
  --epochs "${GATE_EPOCHS}" \
  --lr "${GATE_LR}" \
  --l2 "${GATE_L2}" \
  --null-permutations "${GATE_NULL_PERMUTATIONS}" \
  --min-test-events "${GATE_MIN_TEST_EVENTS}" \
  --min-test-positive-events "${GATE_MIN_TEST_POSITIVE_EVENTS}"

python3 "${ROOT}/scripts/research/abide_dynamic_fc_switching_decision_gate.py" \
  --target-audit-json "${RESULT_DIR}/dynamic_fc_target_audit.json" \
  --gate-json "${RESULT_DIR}/switching_gate/dynamic_fc_switching_gate.json" \
  --gate-summary "${RESULT_DIR}/switching_gate/dynamic_fc_switching_gate_summary.tsv" \
  --output-dir "${RESULT_DIR}/switching_decision" \
  --min-decision-subjects "${DECISION_MIN_SUBJECTS}" \
  --min-decision-sites "${DECISION_MIN_SITES}" \
  --min-decision-null-permutations "${DECISION_MIN_NULL_PERMUTATIONS}"

python3 - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

result = Path(sys.argv[1])
target = json.loads((result / "dynamic_fc_target_audit.json").read_text(encoding="utf-8"))
gate = json.loads((result / "switching_gate" / "dynamic_fc_switching_gate.json").read_text(encoding="utf-8"))
decision = json.loads((result / "switching_decision" / "dynamic_fc_switching_decision.json").read_text(encoding="utf-8"))
assert target["status"] == "pass", target
assert target["parameters"]["split_policy"] == "leave_one_site_out", target
assert gate["verdict"] == "TRAINED_O_SSM_GATE_EXECUTED_NO_PROMOTION", gate
assert decision["overall_verdict"] in {
    "NO_PROMOTION_RESERVOIR_ONLY",
    "NO_PROMOTION_PYTHON_TRAINED_ONLY",
    "NEGATIVE_OR_UNDERCONTROLLED_NO_ROBUST_O_SSM_SIGNAL",
    "EXPLORATORY_O_SSM_SWITCHING_SIGNAL",
}, decision
print("ABIDE_DYNAMIC_FC_SWITCHING_LOSO_RUNNER_PASS")
print("target_subjects=", target["subject_count"], "sites=", target["site_count"])
print("decision=", decision["overall_verdict"])
PY

if [[ "${RUN_SOUNIO_SMOKE}" == "1" ]]; then
  WINDOW_TABLE="${RESULT_DIR}/dynamic_fc_window_table.tsv" \
  OUT_ROOT="${RESULT_DIR}/sounio_smoke" \
  MAX_EVENTS="${SOUNIO_SMOKE_MAX_EVENTS}" \
  GLOBAL_TRAIN_EPOCHS="${SOUNIO_SMOKE_GLOBAL_EPOCHS}" \
  OCT_TRAIN_EPOCHS="${SOUNIO_SMOKE_OCT_EPOCHS}" \
  H_TRAIN_EPOCHS="${SOUNIO_SMOKE_H_EPOCHS}" \
  SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}" \
    bash "${ROOT}/scripts/research/abide_dynamic_fc_switching_sounio_smoke.sh"
fi

echo "outputs: ${RESULT_DIR}"
