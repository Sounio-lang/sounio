#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GATE_SCRIPT="${SOUNIO_SPRINT2_IR_LOWER_GATE_SCRIPT:-$ROOT_DIR/scripts/sprint2_ir_lower_timeout_gate.sh}"
GATE_JSON="${SOUNIO_SPRINT2_IR_LOWER_OUT:-$ROOT_DIR/artifacts/sprint2/ir_lower_timeout_gate.v1.json}"
HOTSPOT_JSON="${SOUNIO_SPRINT2_IR_LOWER_HOTSPOT_OUT:-$ROOT_DIR/artifacts/sprint2/ir_lower_hotspot_trace.v1.json}"
OUT_JSON="${SOUNIO_SPRINT2_IR_LOWER_REGRESSION_OUT:-$ROOT_DIR/artifacts/sprint2/ir_lower_regression_status.v1.json}"

mkdir -p "$(dirname "$OUT_JSON")"

set +e
bash "$GATE_SCRIPT" > /tmp/sprint2_ir_lower_regression_gate.log 2>&1
gate_rc=$?
set -e

python3 - "$OUT_JSON" "$GATE_JSON" "$HOTSPOT_JSON" "$gate_rc" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
gate_json_path = Path(sys.argv[2])
hotspot_json_path = Path(sys.argv[3])
gate_rc = int(sys.argv[4])

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

gate_obj = load_json(gate_json_path)
hotspot_obj = load_json(hotspot_json_path)

status = "fail"
reason = "unknown"
probe_status = "not_run"
probe_reason = "probe_missing"
selfhost_banner_seen = 0
selfhost_horizon_seen = 0
fixture_contract_status = "not_run"
fixture_contract_reason = "missing"
hotspot_last = "none"
hotspot_ok = False

if gate_obj is None:
    reason = "gate_json_missing_or_invalid"
else:
    cases = gate_obj.get("cases", [])
    probe = next((c for c in cases if c.get("name") == "ir_lower_native_probe"), None)
    fixture = next((c for c in cases if c.get("name") == "source_fixture_contract"), None)
    if fixture:
        fixture_contract_status = str(fixture.get("status", "not_run"))
        fixture_contract_reason = str(fixture.get("reason", "unknown"))
    if probe:
        probe_status = str(probe.get("status", "not_run"))
        probe_reason = str(probe.get("reason", "unknown"))
        markers = probe.get("markers", {}) or {}
        selfhost_banner_seen = int(markers.get("selfhost_banner_seen", 0))
        selfhost_horizon_seen = int(markers.get("selfhost_horizon_seen", 0))

if hotspot_obj is not None:
    hotspot_last = str(hotspot_obj.get("last_progress_marker", "none"))
    hotspot_ok = str(hotspot_obj.get("status", "not_run")) == "ok"

# Deterministic lane contract:
# - fixture contract must pass
# - selfhost lane markers must be seen
# - probe must not timeout/not_run
# - acceptable outcomes:
#   a) full pass
#   b) IR-lower complete pass variants for this sprint gate
#   c) known semantic blocker: fail:ir_lowering_failed
if fixture_contract_status != "pass":
    status = "fail"
    reason = f"fixture_contract_{fixture_contract_status}:{fixture_contract_reason}"
elif selfhost_banner_seen != 1 or selfhost_horizon_seen != 1:
    status = "fail"
    reason = "selfhost_markers_missing"
elif probe_status in {"timeout", "not_run"}:
    status = "fail"
    reason = f"probe_{probe_status}:{probe_reason}"
elif probe_status == "pass" and probe_reason in {
    "ok",
    "ok_with_native_dispatch",
    "ok_ir_lower_complete",
    "ok_ir_lower_complete_native_timeout",
}:
    status = "pass"
    reason = f"probe_passed:{probe_reason}"
elif probe_status == "fail" and probe_reason == "ir_lowering_failed":
    status = "pass"
    reason = "known_semantic_blocker_preserved"
else:
    status = "fail"
    reason = f"unexpected_probe_outcome:{probe_status}:{probe_reason}"

payload = {
    "schema": "sounio.sprint2.ir_lower_regression_status.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": reason,
    "gate_exit_code": gate_rc,
    "gate_artifact": str(gate_json_path),
    "hotspot_artifact": str(hotspot_json_path),
    "fixture_contract": {
        "status": fixture_contract_status,
        "reason": fixture_contract_reason,
    },
    "probe": {
        "status": probe_status,
        "reason": probe_reason,
        "selfhost_banner_seen": selfhost_banner_seen,
        "selfhost_horizon_seen": selfhost_horizon_seen,
    },
    "hotspot": {
        "status": "ok" if hotspot_ok else "not_ok",
        "last_progress_marker": hotspot_last,
    },
}

out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={status} reason={reason}")
PY

status="$(python3 - "$OUT_JSON" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("status", "fail"))
PY
)"

echo "ir_lower_regression: out_json=${OUT_JSON#"$ROOT_DIR"/} status=$status"
if [ "$status" = "pass" ]; then
  exit 0
fi
exit 2
