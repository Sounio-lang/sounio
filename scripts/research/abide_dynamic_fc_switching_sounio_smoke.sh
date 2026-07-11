#!/usr/bin/env bash
# Run the existing compiled Brain O-SSM ABIDE benchmark on dynamic-FC switch events.
#
# This is a bridge smoke for execution only. Labels are temporal switch events,
# not ASD/control diagnoses, and the output must not be promoted as a clinical
# or O-SSM superiority claim.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WINDOW_TABLE="${WINDOW_TABLE:-}"
OUT_ROOT="${OUT_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/abide-dynamic-fc-sounio.XXXXXX")}"
MAX_EVENTS="${MAX_EVENTS:-96}"
ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

if [[ -z "$WINDOW_TABLE" ]]; then
  echo "WINDOW_TABLE is required; pass dynamic_fc_window_table.tsv" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
MANIFEST_DIR="$OUT_ROOT/sounio_manifest"
RAW_OUT="$OUT_ROOT/brain_ossm_dynamic_fc_switching.raw.txt"
PARSED_DIR="$OUT_ROOT/brain_ossm_dynamic_fc_switching_parsed"

python3 "$ROOT_DIR/scripts/research/abide_dynamic_fc_switching_sounio_manifest.py" \
  --window-table "$WINDOW_TABLE" \
  --output-dir "$MANIFEST_DIR" \
  --max-events "$MAX_EVENTS" \
  --config global_train_epochs="${GLOBAL_TRAIN_EPOCHS:-3}" \
  --config oct_train_epochs="${OCT_TRAIN_EPOCHS:-3}" \
  --config h_train_epochs="${H_TRAIN_EPOCHS:-3}" \
  --config oct_warmup_epochs="${OCT_WARMUP_EPOCHS:-1}" \
  --config h_warmup_epochs="${H_WARMUP_EPOCHS:-1}" \
  --config trace_hidden_state="${TRACE_HIDDEN_STATE:-0}"

(
  cd "$MANIFEST_DIR"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$ROOT_DIR/bin/souc" run "$ROOT_DIR/examples/brain_ossm_abide.sio"
) >"$RAW_OUT" 2>&1

python3 "$ROOT_DIR/scripts/research/parse_brain_ossm_abide_output.py" \
  --input "$RAW_OUT" \
  --output-dir "$PARSED_DIR"

python3 - "$MANIFEST_DIR/dynamic_fc_sounio_manifest_audit.json" "$PARSED_DIR/overall_metrics.json" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = json.load(open(sys.argv[2], encoding="utf-8"))
models = {row["model"] for row in metrics.get("overall", [])}
if audit.get("verdict") != "PASS":
    raise SystemExit(f"manifest audit did not pass: {audit.get('verdict')}")
missing = {"O-SSM", "H-SSM"} - models
if missing:
    raise SystemExit(f"missing parsed Sounio models: {sorted(missing)}")
print(
    "ABIDE_DYNAMIC_FC_SOUNIO_SMOKE_PASS "
    f"events={audit['event_count']} sites={audit['site_count']} models={','.join(sorted(models))}"
)
PY

echo "output_root=$OUT_ROOT"
