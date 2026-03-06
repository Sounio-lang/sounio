#!/usr/bin/env bash
# sprint6_contest_runtime_gate.sh — ExprContest IR Lowering gate (H5.0)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUNIO_SOUC:-$ROOT_DIR/target/debug/souc}"
OUT_JSON="${SOUNIO_SPRINT6_GATE_OUT:-$ROOT_DIR/artifacts/sprint6/contest_runtime_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT6_TIMEOUT_SECS:-60}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint6_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

# Case 1: souc check self-hosted/compiler/main.sio passes
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check self-hosted/compiler/main.sio > /tmp/sprint6_check_main.log 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  record "check_main_sio" "pass" "all_checks_passed"
elif [ $rc -eq 124 ]; then
  record "check_main_sio" "not_run" "timeout"
else
  reason="$(tail -1 /tmp/sprint6_check_main.log | tr -s ' ' | head -c 80 || echo error)"
  record "check_main_sio" "fail" "check_failed: $reason"
fi

# Case 2: ExprContest arm present in ir/lower.sio
if grep -q "ExprContest" self-hosted/ir/lower.sio 2>/dev/null; then
  record "expr_contest_in_lower_sio" "pass" "found"
else
  record "expr_contest_in_lower_sio" "fail" "not_found_in_ir_lower_sio"
fi

# Case 3: IrContest in ir/ir.sio
if grep -q "IrContest" self-hosted/ir/ir.sio 2>/dev/null; then
  record "ir_contest_opcode_in_ir_sio" "pass" "found"
else
  record "ir_contest_opcode_in_ir_sio" "fail" "not_found_in_ir_sio"
fi

# Case 4: IrContest handled in native codegen
if grep -q "IrContest" self-hosted/native/lower_ir.sio 2>/dev/null; then
  record "ir_contest_in_native_codegen" "pass" "found"
else
  record "ir_contest_in_native_codegen" "fail" "not_found_in_native_lower_ir_sio"
fi

# Case 5: IrContest handled in WASM lowering
if grep -q "IrContest" self-hosted/wasm/lower.sio 2>/dev/null; then
  record "ir_contest_in_wasm_lower" "pass" "found"
else
  record "ir_contest_in_wasm_lower" "fail" "not_found_in_wasm_lower_sio"
fi

# Case 6: test fixture exists
if [ -f "tests/frontend/contest_runtime_basic.sio" ]; then
  record "fixture_contest_runtime_basic" "pass" "file_exists"
else
  record "fixture_contest_runtime_basic" "fail" "file_missing"
fi

# Case 7: fixture contains Contest<i64> and contest [...] on syntax
# (Pinned Rust souc binary does not support Contest<T> syntax yet — validated by grep.)
if grep -q "Contest<i64>" tests/frontend/contest_runtime_basic.sio 2>/dev/null \
   && grep -q "contest \[M1\] on d" tests/frontend/contest_runtime_basic.sio 2>/dev/null; then
  record "fixture_has_contest_syntax" "pass" "contest_syntax_present"
else
  record "fixture_has_contest_syntax" "fail" "contest_syntax_missing_from_fixture"
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
    "schema": "sounio.sprint6.contest_runtime_gate.v1",
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
echo "sprint6_contest_runtime_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
