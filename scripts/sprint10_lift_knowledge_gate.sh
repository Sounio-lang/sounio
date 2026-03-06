#!/usr/bin/env bash
# sprint10_lift_knowledge_gate.sh — lift_knowledge epistemic bridge gate (H5.4)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  # shellcheck source=/dev/null
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="$SOUC_BIN"
fi
OUT_JSON="${SOUNIO_SPRINT10_GATE_OUT:-$ROOT_DIR/artifacts/sprint10/lift_knowledge_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT10_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint10_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

# Case 1: souc check self-hosted/compiler/main.sio passes
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check self-hosted/compiler/main.sio > /tmp/sprint10_check_main.log 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  record "check_main_sio" "pass" "all_checks_passed"
elif [ $rc -eq 124 ]; then
  record "check_main_sio" "not_run" "timeout"
else
  reason="$(tail -1 /tmp/sprint10_check_main.log | tr -s ' ' | head -c 80 || echo error)"
  record "check_main_sio" "fail" "check_failed: $reason"
fi

# Case 2: ir_name_is_lift_knowledge present in ir/ir.sio
if grep -q "ir_name_is_lift_knowledge" self-hosted/ir/ir.sio 2>/dev/null; then
  record "ir_name_is_lift_knowledge_in_ir_sio" "pass" "found"
else
  record "ir_name_is_lift_knowledge_in_ir_sio" "fail" "not_found"
fi

# Case 3: IrLiftKnowledge opcode present in ir/ir.sio
if grep -q "IrLiftKnowledge" self-hosted/ir/ir.sio 2>/dev/null; then
  record "ir_lift_knowledge_opcode_in_ir_sio" "pass" "found"
else
  record "ir_lift_knowledge_opcode_in_ir_sio" "fail" "not_found"
fi

# Case 4: lower_lift_knowledge_call + ir_name_is_lift_knowledge present in ir/lower.sio
if grep -q "lower_lift_knowledge_call" self-hosted/ir/lower.sio 2>/dev/null \
   && grep -q "ir_name_is_lift_knowledge" self-hosted/ir/lower.sio 2>/dev/null; then
  record "lower_lift_knowledge_in_lower_sio" "pass" "found"
else
  record "lower_lift_knowledge_in_lower_sio" "fail" "not_found"
fi

# Case 5: IrLiftKnowledge handled in native codegen
if grep -q "IrLiftKnowledge" self-hosted/native/lower_ir.sio 2>/dev/null; then
  record "ir_lift_knowledge_in_native_codegen" "pass" "found"
else
  record "ir_lift_knowledge_in_native_codegen" "fail" "not_found_in_native_lower_ir_sio"
fi

# Case 6: IrLiftKnowledge handled in WASM lowering
if grep -q "IrLiftKnowledge" self-hosted/wasm/lower.sio 2>/dev/null; then
  record "ir_lift_knowledge_in_wasm_lower" "pass" "found"
else
  record "ir_lift_knowledge_in_wasm_lower" "fail" "not_found_in_wasm_lower_sio"
fi

# Case 7: check_lift_knowledge_expr and call_expr_is_builtin_lift_knowledge in check/check.sio
if grep -q "check_lift_knowledge_expr" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "call_expr_is_builtin_lift_knowledge" self-hosted/check/check.sio 2>/dev/null; then
  record "lift_knowledge_in_check_sio" "pass" "found"
else
  record "lift_knowledge_in_check_sio" "fail" "not_found"
fi

# Case 8: lift_knowledge intercept in hlir/lower.sio
if grep -q '"lift_knowledge"' self-hosted/hlir/lower.sio 2>/dev/null; then
  record "lift_knowledge_in_hlir_lower" "pass" "found"
else
  record "lift_knowledge_in_hlir_lower" "fail" "not_found_in_hlir_lower_sio"
fi

# Case 9: fixture exists with full measurement-to-validation chain
if [ -f "tests/frontend/lift_knowledge_basic.sio" ] \
   && grep -q "lift_knowledge" tests/frontend/lift_knowledge_basic.sio 2>/dev/null \
   && grep -q "Knowledge<i64>" tests/frontend/lift_knowledge_basic.sio 2>/dev/null \
   && grep -q "prove_robust" tests/frontend/lift_knowledge_basic.sio 2>/dev/null \
   && grep -q "validate_manifest" tests/frontend/lift_knowledge_basic.sio 2>/dev/null; then
  record "fixture_full_chain_present" "pass" "all_four_forms_in_fixture"
else
  record "fixture_full_chain_present" "fail" "fixture_incomplete_or_missing"
fi

python3 - "$OUT_JSON" "$CASES_TSV" "$SOUC" "$TIMEOUT_SECS" << 'PY'
import json, datetime as dt, sys
from pathlib import Path

out_json = Path(sys.argv[1])
cases_tsv = Path(sys.argv[2])
souc = sys.argv[3]
timeout_secs = int(sys.argv[4])

cases = []
counts = {"pass": 0, "fail": 0, "not_run": 0}

for raw in cases_tsv.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    parts = raw.split("\t", 2)
    name, status, reason = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
    if status not in counts:
        status, reason = "fail", f"invalid_{status}"
    counts[status] += 1
    cases.append({"name": name, "status": status, "reason": reason})

overall = "pass" if counts["fail"] == 0 and counts["not_run"] == 0 else \
          "not_run" if counts["pass"] == 0 and counts["fail"] == 0 else "fail"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint10.lift_knowledge_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall, "reason": reason,
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {"total": len(cases), "passed": counts["pass"],
                "failed": counts["fail"], "not_run": counts["not_run"]},
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint10_lift_knowledge_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
