#!/usr/bin/env bash
# sprint3_native_gate.sh — Sprint 3 selfhosted native dispatch gate
# Passes when the selfhosted compiler produces a native ELF binary.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IR_GATE_SCRIPT="${SOUNIO_SPRINT2_IR_LOWER_GATE_SCRIPT:-$ROOT_DIR/scripts/sprint2_ir_lower_timeout_gate.sh}"
NATIVE_GATE_SCRIPT="${SOUNIO_SPRINT2_NATIVE_GATE_SCRIPT:-$ROOT_DIR/scripts/sprint2_native_post_dispatch_gate.sh}"
OUT_JSON="${SOUNIO_SPRINT3_NATIVE_GATE_OUT:-$ROOT_DIR/artifacts/sprint3/native_gate.v1.json}"

mkdir -p "$(dirname "$OUT_JSON")" artifacts/sprint3/logs

to_rel() {
  local p="$1"
  [[ "$p" == "$ROOT_DIR/"* ]] && printf '%s' "${p#$ROOT_DIR/}" || printf '%s' "$p"
}

# Run upstream ir_lower gate first
set +e
bash "$IR_GATE_SCRIPT" > /tmp/sprint3_native_ir_gate.log 2>&1
ir_rc=$?
set -e

IR_GATE_JSON="${SOUNIO_SPRINT2_IR_LOWER_OUT:-$ROOT_DIR/artifacts/sprint2/ir_lower_timeout_gate.v1.json}"
ir_status="not_run"
if [ -f "$IR_GATE_JSON" ]; then
  ir_status="$(python3 -c "import json; d=json.load(open('$IR_GATE_JSON')); print(d.get('status','fail'))")"
fi

if [ "$ir_status" != "pass" ]; then
  python3 - "$OUT_JSON" "$ir_status" "$IR_GATE_JSON" << 'PY'
import json, datetime as dt, sys
out_json, ir_status, ir_gate_json = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
  "schema": "sounio.sprint3.native_gate.v1",
  "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
  "status": "not_run",
  "reason": f"upstream_ir_lower_{ir_status}",
  "upstream": {"ir_lower": ir_status, "artifact": ir_gate_json},
}
open(out_json, "w").write(json.dumps(payload, indent=2) + "\n")
print(f"wrote {out_json}")
print(f"status=not_run reason=upstream_ir_lower_{ir_status}")
PY
  echo "sprint3_native_gate: out_json=$(to_rel "$OUT_JSON") status=not_run"
  exit 3
fi

# Run native post-dispatch gate
set +e
bash "$NATIVE_GATE_SCRIPT" > /tmp/sprint3_native_post_gate.log 2>&1
native_rc=$?
set -e

NATIVE_GATE_JSON="${SOUNIO_SPRINT2_NATIVE_POST_DISPATCH_OUT:-$ROOT_DIR/artifacts/sprint2/native_post_dispatch_gate.v1.json}"
native_status="not_run"
native_reason="missing"
if [ -f "$NATIVE_GATE_JSON" ]; then
  native_status="$(python3 -c "import json; d=json.load(open('$NATIVE_GATE_JSON')); print(d.get('status','fail'))")"
  native_reason="$(python3 -c "import json; d=json.load(open('$NATIVE_GATE_JSON')); print(d.get('reason','unknown'))")"
fi

python3 - "$OUT_JSON" "$native_status" "$native_reason" "$IR_GATE_JSON" "$NATIVE_GATE_JSON" << 'PY'
import json, datetime as dt, sys
out_json = sys.argv[1]
native_status = sys.argv[2]
native_reason = sys.argv[3]
ir_gate_json = sys.argv[4]
native_gate_json = sys.argv[5]

overall_status = native_status if native_status in ("pass", "fail", "not_run") else "fail"
payload = {
  "schema": "sounio.sprint3.native_gate.v1",
  "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
  "status": overall_status,
  "reason": native_reason,
  "upstream": {
    "ir_lower": "pass",
    "artifact": ir_gate_json,
  },
  "native_dispatch": {
    "status": native_status,
    "reason": native_reason,
    "artifact": native_gate_json,
  },
}
open(out_json, "w").write(json.dumps(payload, indent=2) + "\n")
print(f"wrote {out_json}")
print(f"status={overall_status} reason={native_reason}")
PY

echo "sprint3_native_gate: out_json=$(to_rel "$OUT_JSON") status=$native_status"
if [ "$native_status" = "pass" ]; then exit 0; fi
if [ "$native_status" = "not_run" ]; then exit 3; fi
exit 2
