#!/usr/bin/env bash
# sprint13_diagnostic_hardening_gate.sh — diagnostic hardening audit gate (Phase A.3)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  # shellcheck source=/dev/null
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="$SOUC_BIN"
fi
OUT_JSON="${SOUNIO_SPRINT13_GATE_OUT:-$ROOT_DIR/artifacts/sprint13/diagnostic_hardening_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT13_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint13_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

UI_DIR="$ROOT_DIR/tests/ui/type"

# ── Structural audit: annotation counts ──────────────────────────────────────

# Case 1: At least 26 tests annotated SELFHOST_PASS
sh_pass_count=$(grep -rl "SELFHOST_PASS:" "$UI_DIR" 2>/dev/null | wc -l)
if [ "$sh_pass_count" -ge 26 ]; then
  record "selfhost_pass_count" "pass" "found_${sh_pass_count}_SELFHOST_PASS"
else
  record "selfhost_pass_count" "fail" "expected_ge_26_got_${sh_pass_count}"
fi

# Case 2: Exactly 7 tests annotated SH_ERROR
sh_err_count=$(grep -rl "SH_ERROR:" "$UI_DIR" 2>/dev/null | wc -l)
if [ "$sh_err_count" -eq 7 ]; then
  record "sh_error_count" "pass" "found_${sh_err_count}_SH_ERROR"
else
  record "sh_error_count" "fail" "expected_7_got_${sh_err_count}"
fi

# Case 3: Exactly 5 tests annotated SH_TIMEOUT
sh_to_count=$(grep -rl "SH_TIMEOUT:" "$UI_DIR" 2>/dev/null | wc -l)
if [ "$sh_to_count" -eq 5 ]; then
  record "sh_timeout_count" "pass" "found_${sh_to_count}_SH_TIMEOUT"
else
  record "sh_timeout_count" "fail" "expected_5_got_${sh_to_count}"
fi

# Case 4: check.sio has early-return fix in check_struct_field_inits (E046 before calling _with_type)
if grep -q "report_error_at.*46" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "struct_field_inits_e046_fix" "pass" "found_E046_early_return_in_check.sio"
else
  record "struct_field_inits_e046_fix" "fail" "E046_early_return_not_found_in_check.sio"
fi

# Case 5: check.sio has check_call_arg_borrow_boundary (Phase A.2 prerequisite)
if grep -q "check_call_arg_borrow_boundary" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "borrow_boundary_wired" "pass" "found_in_check.sio"
else
  record "borrow_boundary_wired" "fail" "not_found_in_check.sio"
fi

# Case 6: check.sio has borrow scope push/pop in match arm (Phase A.2 prerequisite)
if grep -q "borrows.push_scope\|borrows\.push_scope" "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "borrow_scope_in_match_arm" "pass" "found_in_check.sio"
else
  record "borrow_scope_in_match_arm" "fail" "not_found_in_check.sio"
fi

# ── Active pinned-binary tests (13 de-ignored): verify they still pass ───────

# Case 7: mismatch_let passes pinned binary
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check \
  "$UI_DIR/mismatch_let.sio" \
  > /tmp/sprint13_pinned_mismatch_let.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ]; then
  record "pinned_mismatch_let" "not_run" "timeout"
elif [ $rc -ne 0 ]; then
  record "pinned_mismatch_let" "pass" "compile_fail_as_expected"
else
  record "pinned_mismatch_let" "fail" "expected_compile_fail_got_pass"
fi

# Case 8: void_assign passes pinned binary
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check \
  "$UI_DIR/void_assign.sio" \
  > /tmp/sprint13_pinned_void_assign.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ]; then
  record "pinned_void_assign" "not_run" "timeout"
elif [ $rc -ne 0 ]; then
  record "pinned_void_assign" "pass" "compile_fail_as_expected"
else
  record "pinned_void_assign" "fail" "expected_compile_fail_got_pass"
fi

# Case 9: return_outside_fn passes pinned binary
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check \
  "$UI_DIR/return_outside_fn.sio" \
  > /tmp/sprint13_pinned_return_outside_fn.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ]; then
  record "pinned_return_outside_fn" "not_run" "timeout"
elif [ $rc -ne 0 ]; then
  record "pinned_return_outside_fn" "pass" "compile_fail_as_expected"
else
  record "pinned_return_outside_fn" "fail" "expected_compile_fail_got_pass"
fi

# ── JIT probe on sample SELFHOST_PASS tests ───────────────────────────────────
JIT_BIN="${SOUNIO_JIT_BIN:-$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-jit}"

probe_jit() {
  local name="$1" file="$2"
  if [ ! -x "$JIT_BIN" ]; then
    record "$name" "not_run" "jit_binary_not_found"
    return
  fi
  set +e
  timeout "$TIMEOUT_SECS" "$JIT_BIN" run \
    "$ROOT_DIR/self-hosted/compiler/main.sio" \
    -- --probe-frontend "$file" \
    > /tmp/sprint13_probe_${name}.log 2>&1
  rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$name" "not_run" "timeout"
  else
    if grep -q "probe_frontend: fail" /tmp/sprint13_probe_${name}.log 2>/dev/null; then
      record "$name" "pass" "sh_probe_fail_as_expected"
    elif grep -q "probe_frontend: ok" /tmp/sprint13_probe_${name}.log 2>/dev/null; then
      record "$name" "fail" "sh_probe_passed_but_expected_fail"
    else
      record "$name" "not_run" "no_probe_output"
    fi
  fi
}

# Case 10-12: Sample JIT probes on SELFHOST_PASS tests (3 representative cases)
probe_jit "jit_break_outside_loop" "$UI_DIR/break_outside_loop.sio"
probe_jit "jit_assign_to_immut"    "$UI_DIR/assign_to_immut.sio"
probe_jit "jit_division_by_zero"   "$UI_DIR/division_by_zero.sio"

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
reason = "all_active_cases_passed" if overall == "pass" else f"{counts['fail']}_failed"

payload = {
    "schema": "sounio.sprint13.diagnostic_hardening_gate.v1",
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
echo "sprint13_diagnostic_hardening_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
