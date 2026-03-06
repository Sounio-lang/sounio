#!/usr/bin/env bash
# sprint8_validate_manifest_gate.sh — validate_manifest IR Lowering gate (H5.2)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUNIO_SOUC:-$ROOT_DIR/target/debug/souc}"
OUT_JSON="${SOUNIO_SPRINT8_GATE_OUT:-$ROOT_DIR/artifacts/sprint8/validate_manifest_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT8_TIMEOUT_SECS:-60}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint8_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

# Case 1: souc check self-hosted/compiler/main.sio passes
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check self-hosted/compiler/main.sio > /tmp/sprint8_check_main.log 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  record "check_main_sio" "pass" "all_checks_passed"
elif [ $rc -eq 124 ]; then
  record "check_main_sio" "not_run" "timeout"
else
  reason="$(tail -1 /tmp/sprint8_check_main.log | tr -s ' ' | head -c 80 || echo error)"
  record "check_main_sio" "fail" "check_failed: $reason"
fi

# Case 2: call_expr_is_builtin_validate_manifest in check/check.sio
if grep -q "call_expr_is_builtin_validate_manifest" self-hosted/check/check.sio 2>/dev/null; then
  record "validate_manifest_in_check_sio" "pass" "found"
else
  record "validate_manifest_in_check_sio" "fail" "not_found"
fi

# Case 3: check_validate_manifest_expr in check/check.sio
if grep -q "check_validate_manifest_expr" self-hosted/check/check.sio 2>/dev/null; then
  record "check_validate_manifest_expr_in_check_sio" "pass" "found"
else
  record "check_validate_manifest_expr_in_check_sio" "fail" "not_found"
fi

# Case 4: IrValidated opcode in ir/ir.sio
if grep -q "IrValidated" self-hosted/ir/ir.sio 2>/dev/null; then
  record "ir_validated_opcode_in_ir_sio" "pass" "found"
else
  record "ir_validated_opcode_in_ir_sio" "fail" "not_found"
fi

# Case 5: lower_validate_manifest_call in ir/lower.sio
if grep -q "lower_validate_manifest_call" self-hosted/ir/lower.sio 2>/dev/null \
   && grep -q "ir_name_is_validate_manifest" self-hosted/ir/lower.sio 2>/dev/null; then
  record "lower_validate_manifest_in_lower_sio" "pass" "found"
else
  record "lower_validate_manifest_in_lower_sio" "fail" "not_found"
fi

# Case 6: IrValidated handled in native codegen
if grep -q "IrValidated" self-hosted/native/lower_ir.sio 2>/dev/null; then
  record "ir_validated_in_native_codegen" "pass" "found"
else
  record "ir_validated_in_native_codegen" "fail" "not_found"
fi

# Case 7: IrValidated handled in WASM lowering
if grep -q "IrValidated" self-hosted/wasm/lower.sio 2>/dev/null; then
  record "ir_validated_in_wasm_lower" "pass" "found"
else
  record "ir_validated_in_wasm_lower" "fail" "not_found"
fi

# Case 8: full chain fixture exists with validate_manifest + Validated<i64>
if [ -f "tests/frontend/validate_manifest_basic.sio" ] \
   && grep -q "validate_manifest" tests/frontend/validate_manifest_basic.sio 2>/dev/null \
   && grep -q "Validated<i64>" tests/frontend/validate_manifest_basic.sio 2>/dev/null \
   && grep -q "prove_robust" tests/frontend/validate_manifest_basic.sio 2>/dev/null; then
  record "fixture_full_epistemic_chain" "pass" "complete_chain_in_fixture"
else
  record "fixture_full_epistemic_chain" "fail" "fixture_missing_or_incomplete"
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
    "schema": "sounio.sprint8.validate_manifest_gate.v1",
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
echo "sprint8_validate_manifest_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
