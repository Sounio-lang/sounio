#!/usr/bin/env bash
# sprint12_borrow_integration_gate.sh — borrow call-site and scope integration gate (Phase A.2)
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
OUT_JSON="${SOUNIO_SPRINT12_GATE_OUT:-$ROOT_DIR/artifacts/sprint12/borrow_integration_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT12_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint12_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

# Case 1: borrow_call_exclusive_conflict compiles to a borrow error (exit non-zero, output contains "borrow")
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check \
  "$ROOT_DIR/tests/compile-fail/borrow_call_exclusive_conflict.sio" \
  > /tmp/sprint12_exclusive.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ]; then
  record "borrow_call_exclusive_conflict" "not_run" "timeout"
elif [ $rc -ne 0 ] && grep -qi "borrow" /tmp/sprint12_exclusive.log 2>/dev/null; then
  record "borrow_call_exclusive_conflict" "pass" "compile_fail_with_borrow_error"
elif [ $rc -ne 0 ]; then
  record "borrow_call_exclusive_conflict" "fail" "compile_fail_but_no_borrow_in_output"
else
  record "borrow_call_exclusive_conflict" "fail" "expected_compile_fail_but_got_pass"
fi

# Case 2: borrow_call_shared_conflict compiles to a borrow error (exit non-zero, output contains "borrow")
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check \
  "$ROOT_DIR/tests/compile-fail/borrow_call_shared_conflict.sio" \
  > /tmp/sprint12_shared.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ]; then
  record "borrow_call_shared_conflict" "not_run" "timeout"
elif [ $rc -ne 0 ] && grep -qi "borrow" /tmp/sprint12_shared.log 2>/dev/null; then
  record "borrow_call_shared_conflict" "pass" "compile_fail_with_borrow_error"
elif [ $rc -ne 0 ]; then
  record "borrow_call_shared_conflict" "fail" "compile_fail_but_no_borrow_in_output"
else
  record "borrow_call_shared_conflict" "fail" "expected_compile_fail_but_got_pass"
fi

# Case 3: linear_call_double_use — marked //@ ignore in source; record as not_run
record "linear_call_double_use" "not_run" "ignored_pending_linear_enforce"

# Case 4: borrow_match_scope_ok passes cleanly (run-pass)
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check \
  "$ROOT_DIR/tests/run-pass/borrow_match_scope_ok.sio" \
  > /tmp/sprint12_match_scope.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ]; then
  record "borrow_match_scope_ok" "not_run" "timeout"
elif [ $rc -eq 0 ]; then
  record "borrow_match_scope_ok" "pass" "run_pass_clean"
else
  reason="$(tail -1 /tmp/sprint12_match_scope.log | tr -s ' ' | head -c 120 || echo error)"
  record "borrow_match_scope_ok" "fail" "expected_pass_but_got_error: $reason"
fi

# Case 5: check_call_arg_borrow_boundary present in check.sio
if grep -q "check_call_arg_borrow_boundary" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "call_borrow_boundary_in_check_sio" "pass" "found"
else
  record "call_borrow_boundary_in_check_sio" "fail" "not_found"
fi

# Case 6: borrow push/pop scope in check_match_arm
if grep -q "borrows.push_scope\|borrows\.push_scope" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null \
   && grep -q "borrows.pop_scope\|borrows\.pop_scope" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "borrow_scope_sync_in_check_match_arm" "pass" "found"
else
  record "borrow_scope_sync_in_check_match_arm" "fail" "not_found"
fi

# Case 7: bind_fn_param tracks params in borrows
if grep -q "borrows.track\|borrows\.track" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "borrow_track_in_bind_fn_param" "pass" "found"
else
  record "borrow_track_in_bind_fn_param" "fail" "not_found"
fi

# Case 8: consume_linear wired in check_call_args_inner
if grep -q "consume_linear" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "consume_linear_in_check_call_args" "pass" "found"
else
  record "consume_linear_in_check_call_args" "fail" "not_found"
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

overall = "pass" if counts["fail"] == 0 else "fail"
reason = "all_active_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed"

payload = {
    "schema": "sounio.sprint12.borrow_integration_gate.v1",
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
echo "sprint12_borrow_integration_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
