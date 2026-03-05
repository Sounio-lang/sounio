#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IR_GATE_SCRIPT="${SOUNIO_SPRINT2_IR_LOWER_GATE_SCRIPT:-$ROOT_DIR/scripts/sprint2_ir_lower_timeout_gate.sh}"
IR_GATE_JSON="${SOUNIO_SPRINT2_IR_LOWER_OUT:-$ROOT_DIR/artifacts/sprint2/ir_lower_timeout_gate.v1.json}"
OUT_JSON="${SOUNIO_SPRINT2_NATIVE_POST_DISPATCH_OUT:-$ROOT_DIR/artifacts/sprint2/native_post_dispatch_gate.v1.json}"

mkdir -p "$(dirname "$OUT_JSON")"

set +e
bash "$IR_GATE_SCRIPT" > /tmp/sprint2_native_post_dispatch_ir_gate.log 2>&1
ir_gate_rc=$?
set -e

python3 - "$OUT_JSON" "$IR_GATE_JSON" "$ir_gate_rc" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
ir_gate_json = Path(sys.argv[2])
ir_gate_rc = int(sys.argv[3])

status = "fail"
reason = "unknown"

probe_status = "not_run"
probe_reason = "missing"
probe_exit_code = -1
markers = {}
log_path = ""
native_output = ""
native_output_exists = False
native_output_size = 0

upstream_status = "not_run"
upstream_reason = "missing_or_invalid"

if not ir_gate_json.exists():
    status = "not_run"
    reason = "ir_lower_gate_artifact_missing"
else:
    try:
        gate = json.loads(ir_gate_json.read_text(encoding="utf-8"))
    except Exception:
        gate = None
        status = "not_run"
        reason = "ir_lower_gate_artifact_invalid"

    if gate is not None:
        upstream_status = str(gate.get("status", "not_run"))
        upstream_reason = str(gate.get("reason", "unknown"))
        cases = gate.get("cases", [])
        probe = next((c for c in cases if c.get("name") == "ir_lower_native_probe"), None)

        if probe is None:
            status = "not_run"
            reason = "ir_lower_probe_case_missing"
        else:
            probe_status = str(probe.get("status", "not_run"))
            probe_reason = str(probe.get("reason", "unknown"))
            probe_exit_code = int(probe.get("exit_code", -1))
            markers = probe.get("markers", {}) or {}
            log_path = str(probe.get("log_path", ""))
            native_output = str(probe.get("native_output", ""))

            if native_output:
                native_path = Path(native_output)
                native_output_exists = native_path.exists()
                if native_output_exists:
                    native_output_size = native_path.stat().st_size

            selfhost_banner = int(markers.get("selfhost_banner_seen", 0))
            selfhost_horizon = int(markers.get("selfhost_horizon_seen", 0))
            type_done = int(markers.get("type_done", 0))
            ir_done = int(markers.get("ir_done", 0))
            merged_seen = int(markers.get("merged_seen", 0))
            native_dispatch_seen = int(markers.get("native_dispatch_seen", 0))
            native_bin_seen = int(markers.get("native_bin_seen", 0))
            written_seen = int(markers.get("written_seen", 0))

            # This gate starts only after the upstream IR-lower stage is complete.
            if not (selfhost_banner == 1 and selfhost_horizon == 1 and type_done == 1 and ir_done == 1):
                status = "not_run"
                reason = "upstream_ir_lower_not_complete"
            elif merged_seen != 1 or native_dispatch_seen != 1:
                status = "fail"
                reason = "missing_post_dispatch_markers"
            elif native_bin_seen == 1 or written_seen == 1 or (native_output_exists and native_output_size > 0):
                status = "pass"
                reason = "native_output_observed"
            elif probe_exit_code == 124 or probe_reason in {
                "ok_ir_lower_complete_native_timeout",
                "timeout_post_ir_lower_native_codegen",
            }:
                status = "fail"
                reason = "timeout_post_native_dispatch"
            elif probe_status == "fail":
                status = "fail"
                reason = f"native_stage_failed:{probe_reason}"
            else:
                status = "fail"
                reason = f"native_output_not_observed:{probe_status}:{probe_reason}"

payload = {
    "schema": "sounio.sprint2.native_post_dispatch_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": reason,
    "upstream_ir_lower_gate": {
        "status": upstream_status,
        "reason": upstream_reason,
        "exit_code": ir_gate_rc,
        "artifact": str(ir_gate_json),
    },
    "probe": {
        "status": probe_status,
        "reason": probe_reason,
        "exit_code": probe_exit_code,
        "log_path": log_path,
        "markers": markers,
        "native_output": native_output,
        "native_output_exists": native_output_exists,
        "native_output_size": native_output_size,
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

echo "native_post_dispatch_gate: out_json=${OUT_JSON#"$ROOT_DIR"/} status=$status"
if [ "$status" = "pass" ]; then
  exit 0
fi
if [ "$status" = "not_run" ]; then
  exit 3
fi
exit 2
