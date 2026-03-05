#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_SPRINT2_PACK_OUT_DIR:-$ROOT_DIR/artifacts/sprint2}"
OUT_JSON="${SOUNIO_SPRINT2_PACK_OUT_JSON:-$OUT_DIR/sprint2_123_status.v1.json}"
OUT_MD="${SOUNIO_SPRINT2_PACK_OUT_MD:-$OUT_DIR/sprint2_123_status.md}"

SPRINT1_CRITICAL_JSON="${SOUNIO_SPRINT1_CRITICAL_JSON:-$ROOT_DIR/artifacts/sprint1/critical_bug_fixes_gate.v1.json}"
SPRINT1_FRONTEND_JSON="${SOUNIO_SPRINT1_FRONTEND_JSON:-$ROOT_DIR/artifacts/sprint1/reports/frontend_test_status.v1.json}"
SPRINT2_PARSER_JSON="${SOUNIO_SPRINT2_PARSER_JSON:-$ROOT_DIR/artifacts/sprint2/parser_stability_gate.v1.json}"
SPRINT2_LEXER_JSON="${SOUNIO_SPRINT2_LEXER_JSON:-$ROOT_DIR/artifacts/sprint2/lexer_feature_gate.v1.json}"
SPRINT2_WASM_JSON="${SOUNIO_SPRINT2_WASM_JSON:-$ROOT_DIR/artifacts/sprint2/wasm_dispatch_gate.v1.json}"
SPRINT2_IR_LOWER_JSON="${SOUNIO_SPRINT2_IR_LOWER_JSON:-$ROOT_DIR/artifacts/sprint2/ir_lower_timeout_gate.v1.json}"

mkdir -p "$OUT_DIR"

python3 - "$OUT_JSON" "$OUT_MD" "$SPRINT1_CRITICAL_JSON" "$SPRINT1_FRONTEND_JSON" "$SPRINT2_PARSER_JSON" "$SPRINT2_LEXER_JSON" "$SPRINT2_WASM_JSON" "$SPRINT2_IR_LOWER_JSON" <<'PY'
import datetime as dt
import json
from pathlib import Path
import sys

out_json = Path(sys.argv[1])
out_md = Path(sys.argv[2])
critical_path = Path(sys.argv[3])
frontend_path = Path(sys.argv[4])
parser_path = Path(sys.argv[5])
lexer_path = Path(sys.argv[6])
wasm_path = Path(sys.argv[7])
ir_lower_path = Path(sys.argv[8])


def load(path: Path):
    if not path.exists():
        return None, "not_run", "missing_artifact"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "not_run", "invalid_json"
    status = str(obj.get("status", obj.get("overall_status", "not_run")))
    raw_reason = obj.get("reason", obj.get("overall_reason"))
    if raw_reason is None:
        if status == "pass":
            reason = "all_steps_passed"
        else:
            reason = "unknown"
    else:
        reason = str(raw_reason)
    return obj, status, reason

critical_obj, critical_status, critical_reason = load(critical_path)
frontend_obj, frontend_status, frontend_reason = load(frontend_path)
parser_obj, parser_status, parser_reason = load(parser_path)
lexer_obj, lexer_status, lexer_reason = load(lexer_path)
wasm_obj, wasm_status, wasm_reason = load(wasm_path)
ir_lower_obj, ir_lower_status, ir_lower_reason = load(ir_lower_path)

critical_normalized = "pass" if critical_status == "pass" else ("fail" if critical_status in {"fail", "timeout"} else "not_run")
frontend_normalized = "pass" if frontend_status == "pass" else ("fail" if frontend_status in {"fail", "timeout"} else "not_run")
parser_normalized = "pass" if parser_status == "pass" else ("fail" if parser_status in {"fail", "timeout"} else "not_run")
lexer_normalized = "pass" if lexer_status == "pass" else ("fail" if lexer_status in {"fail", "timeout"} else "not_run")
wasm_normalized = "pass" if wasm_status == "pass" else ("fail" if wasm_status in {"fail", "timeout"} else "not_run")
ir_lower_normalized = "pass" if ir_lower_status == "pass" else ("fail" if ir_lower_status in {"fail", "timeout"} else "not_run")

overall = "pass"
overall_reason = "all_lanes_passed"
blockers = []

for lane, status, reason in [
    ("sprint1_critical_bug_fixes", critical_normalized, critical_reason),
    ("sprint1_frontend", frontend_normalized, frontend_reason),
    ("sprint2_parser_stability", parser_normalized, parser_reason),
    ("sprint2_lexer_features", lexer_normalized, lexer_reason),
    ("sprint2_wasm_dispatch", wasm_normalized, wasm_reason),
    ("sprint2_ir_lower", ir_lower_normalized, ir_lower_reason),
]:
    if status != "pass":
        overall = "fail"
        blockers.append(f"{lane}:{status}:{reason}")

if blockers:
    overall_reason = "one_or_more_lanes_not_pass"

payload = {
    "schema": "sounio.sprint2.status_pack_123.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": overall_reason,
    "lanes": {
        "sprint1_critical_bug_fixes": {
            "status": critical_normalized,
            "reason": critical_reason,
            "artifact": str(critical_path),
        },
        "sprint1_frontend": {
            "status": frontend_normalized,
            "reason": frontend_reason,
            "artifact": str(frontend_path),
        },
        "sprint2_parser_stability": {
            "status": parser_normalized,
            "reason": parser_reason,
            "artifact": str(parser_path),
        },
        "sprint2_lexer_features": {
            "status": lexer_normalized,
            "reason": lexer_reason,
            "artifact": str(lexer_path),
        },
        "sprint2_wasm_dispatch": {
            "status": wasm_normalized,
            "reason": wasm_reason,
            "artifact": str(wasm_path),
        },
        "sprint2_ir_lower": {
            "status": ir_lower_normalized,
            "reason": ir_lower_reason,
            "artifact": str(ir_lower_path),
        },
    },
    "blockers": blockers,
}

out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

lines = []
lines.append("# Sprint 2 1-2-3 Status Pack")
lines.append("")
lines.append(f"- status: `{overall}`")
lines.append(f"- reason: `{overall_reason}`")
lines.append("")
lines.append("## Lanes")
for lane_name, lane in payload["lanes"].items():
    lines.append(f"- {lane_name}: `{lane['status']}` reason=`{lane['reason']}` artifact=`{lane['artifact']}`")
if blockers:
    lines.append("")
    lines.append("## Blockers")
    for b in blockers:
        lines.append(f"- {b}")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"wrote {out_json}")
print(f"wrote {out_md}")
print(f"status={overall} reason={overall_reason}")
PY
