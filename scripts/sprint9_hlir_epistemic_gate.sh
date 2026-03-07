#!/usr/bin/env bash
# sprint9_hlir_epistemic_gate.sh — HLIR Epistemic Pipeline gate (H5.3)
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
OUT_JSON="${SOUNIO_SPRINT9_GATE_OUT:-$ROOT_DIR/artifacts/sprint9/hlir_epistemic_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT9_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint9_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_selfhost_preflight_case() {
  local case_name="$1" source_file="$2" log_file="$3"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- --probe-frontend "$source_file" > "$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
  elif grep -q "probe_frontend: ok" "$log_file" 2>/dev/null; then
    record "$case_name" "pass" "selfhost_frontend_preflight_ok"
  elif grep -q "probe_frontend: fail" "$log_file" 2>/dev/null; then
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 120 || echo error)"
    record "$case_name" "fail" "compile_failed: $reason"
  else
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 120 || echo error)"
    record "$case_name" "fail" "ambiguous_probe_output: $reason"
  fi
}

# Case 1: self-hosted compiler driver self-test passes
set +e
timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- --self-test > /tmp/sprint9_check_main.log 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  record "selfhost_compiler_main_self_test" "pass" "all_checks_passed"
elif [ $rc -eq 124 ]; then
  record "selfhost_compiler_main_self_test" "not_run" "timeout"
else
  reason="$(tail -1 /tmp/sprint9_check_main.log | tr -s ' ' | head -c 80 || echo error)"
  record "selfhost_compiler_main_self_test" "fail" "check_failed: $reason"
fi

# Case 2: ExprContest arm in hlir/lower.sio
if grep -q "ExprKind::ExprContest" self-hosted/hlir/lower.sio 2>/dev/null; then
  record "expr_contest_in_hlir_lower" "pass" "found"
else
  record "expr_contest_in_hlir_lower" "fail" "not_found_in_hlir_lower_sio"
fi

# Case 3: prove_robust intercept in hlir/lower.sio
if grep -q '"prove_robust"' self-hosted/hlir/lower.sio 2>/dev/null; then
  record "prove_robust_in_hlir_lower" "pass" "found"
else
  record "prove_robust_in_hlir_lower" "fail" "not_found_in_hlir_lower_sio"
fi

# Case 4: validate_manifest intercept in hlir/lower.sio
if grep -q '"validate_manifest"' self-hosted/hlir/lower.sio 2>/dev/null; then
  record "validate_manifest_in_hlir_lower" "pass" "found"
else
  record "validate_manifest_in_hlir_lower" "fail" "not_found_in_hlir_lower_sio"
fi

# Case 5: all three intercepts are before the generic hlir_lower_call
line_contest=$(grep -n "ExprKind::ExprContest" self-hosted/hlir/lower.sio 2>/dev/null | tail -1 | cut -d: -f1)
line_prove=$(grep -n '"prove_robust"' self-hosted/hlir/lower.sio 2>/dev/null | tail -1 | cut -d: -f1)
line_validate=$(grep -n '"validate_manifest"' self-hosted/hlir/lower.sio 2>/dev/null | tail -1 | cut -d: -f1)
line_generic=$(grep -n "hlir_lower_call(s, callee," self-hosted/hlir/lower.sio 2>/dev/null | tail -1 | cut -d: -f1)
if [ -n "$line_contest" ] && [ -n "$line_prove" ] && [ -n "$line_validate" ] && [ -n "$line_generic" ] \
   && [ "$line_prove" -lt "$line_generic" ] && [ "$line_validate" -lt "$line_generic" ]; then
  record "intercepts_before_generic_call" "pass" "ordering_correct"
else
  record "intercepts_before_generic_call" "fail" "ordering_wrong_or_missing"
fi

# Case 6: HlirTypeKnowledge remains in hlir/ir.sio (no regression)
if grep -q "HlirTypeKnowledge" self-hosted/hlir/ir.sio 2>/dev/null; then
  record "hlir_type_knowledge_retained" "pass" "found"
else
  record "hlir_type_knowledge_retained" "fail" "missing"
fi

# Case 7: full chain fixture exists and contains the explicit full surface
if [ -f "tests/frontend/validate_manifest_basic.sio" ] \
   && grep -q "models DefaultModels = \[M1\]" tests/frontend/validate_manifest_basic.sio 2>/dev/null \
   && grep -q "policy DefaultPolicy for i64 level Stable scope InDistribution" tests/frontend/validate_manifest_basic.sio 2>/dev/null \
   && grep -q "validation DefaultManifest for i64" tests/frontend/validate_manifest_basic.sio 2>/dev/null \
   && grep -q "Contest<i64, DefaultModels, DefaultPolicy>" tests/frontend/validate_manifest_basic.sio 2>/dev/null \
   && grep -q "prove_robust" tests/frontend/validate_manifest_basic.sio 2>/dev/null \
   && grep -q "validate_manifest(r, DefaultManifest)" tests/frontend/validate_manifest_basic.sio 2>/dev/null; then
  record "full_chain_fixture_present" "pass" "full_surface_fixture_present"
else
  record "full_chain_fixture_present" "fail" "fixture_incomplete_or_missing"
fi

# Case 8: self-hosted frontend can parse/resolve/typecheck the HLIR-chain fixture
run_selfhost_preflight_case \
  "preflight_fixture_validate_manifest_basic" \
  "tests/frontend/validate_manifest_basic.sio" \
  "/tmp/sprint9_compile_fixture.log"

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
    "schema": "sounio.sprint9.hlir_epistemic_gate.v1",
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
echo "sprint9_hlir_epistemic_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
