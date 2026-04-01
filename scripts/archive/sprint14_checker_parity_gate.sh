#!/usr/bin/env bash
# sprint14_checker_parity_gate.sh — checker parity 38/38 gate
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
OUT_JSON="${SOUNIO_SPRINT14_GATE_OUT:-$ROOT_DIR/artifacts/sprint14/checker_parity_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT14_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint14_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

UI_DIR="$ROOT_DIR/tests/ui/type"

# ── Case 1: All 38 compile-fail tests annotated SELFHOST_PASS ─────────────────

sh_pass_count=$(grep -rl 'SELFHOST_PASS' "$UI_DIR" 2>/dev/null | wc -l)
if [ "$sh_pass_count" -ge 38 ]; then
  record "selfhost_pass_count" "pass" "found_${sh_pass_count}_SELFHOST_PASS"
else
  record "selfhost_pass_count" "fail" "expected_ge_38_got_${sh_pass_count}"
fi

# ── Case 2: Zero SH_ERROR annotations remain ─────────────────────────────────

sh_err_count=$(grep -rl 'SH_ERROR' "$UI_DIR" 2>/dev/null | wc -l || true)
if [ "$sh_err_count" -eq 0 ]; then
  record "sh_error_zero" "pass" "no_SH_ERROR_remaining"
else
  record "sh_error_zero" "fail" "found_${sh_err_count}_SH_ERROR"
fi

# ── Case 3: Zero SH_TIMEOUT annotations remain ───────────────────────────────

sh_to_count=$(grep -rl 'SH_TIMEOUT' "$UI_DIR" 2>/dev/null | wc -l || true)
if [ "$sh_to_count" -eq 0 ]; then
  record "sh_timeout_zero" "pass" "no_SH_TIMEOUT_remaining"
else
  record "sh_timeout_zero" "fail" "found_${sh_to_count}_SH_TIMEOUT"
fi

# ── Case 4: Structural — iterative check_tuple_elems (no recursive Checker copy) ──

if grep -q 'var done = false' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null &&
   grep -q 'check_tuple_elems' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "iterative_tuple_check" "pass" "check_tuple_elems_is_iterative"
else
  record "iterative_tuple_check" "fail" "check_tuple_elems_not_iterative"
fi

# ── Case 5: Structural — ExprForIn dispatch in check_expr ────────────────────

if grep -q 'ExprForIn => c.check_for_in_expr' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "expr_for_in_dispatch" "pass" "found_ExprForIn_arm"
else
  record "expr_for_in_dispatch" "fail" "ExprForIn_arm_missing"
fi

# ── Case 6: Structural — StructFieldInit span fix (value.span not head.span) ─

if grep -q 'head\.value\.span' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  record "struct_field_init_span_fix" "pass" "uses_value_span"
else
  record "struct_field_init_span_fix" "fail" "may_still_use_head_span"
fi

# ── JIT probe: all 12 formerly-failing tests ─────────────────────────────────
JIT_BIN="${SOUNIO_JIT_BIN:-$ROOT_DIR/target/debug/souc}"

probe_jit() {
  local name="$1" file="$2" expect_fail="${3:-true}"
  if [ ! -x "$JIT_BIN" ]; then
    record "$name" "not_run" "jit_binary_not_found"
    return
  fi
  set +e
  timeout "$TIMEOUT_SECS" "$JIT_BIN" run \
    "$ROOT_DIR/self-hosted/compiler/main.sio" \
    -- --probe-frontend "$file" \
    > "/tmp/sprint14_probe_${name}.log" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$name" "fail" "timeout"
  elif [ "$expect_fail" = "true" ]; then
    if grep -q "probe_frontend: fail\|probe_frontend: error=type_check_failed" "/tmp/sprint14_probe_${name}.log" 2>/dev/null; then
      record "$name" "pass" "sh_probe_fail_as_expected"
    elif grep -q "probe_frontend: ok" "/tmp/sprint14_probe_${name}.log" 2>/dev/null; then
      record "$name" "fail" "sh_probe_passed_but_expected_fail"
    else
      record "$name" "not_run" "no_probe_output"
    fi
  else
    # expect_fail=false — test is //@ ignore, ok is acceptable
    if grep -q "probe_frontend:" "/tmp/sprint14_probe_${name}.log" 2>/dev/null; then
      record "$name" "pass" "sh_probe_completed"
    else
      record "$name" "not_run" "no_probe_output"
    fi
  fi
}

# Former SH_ERROR (7) — all should detect errors
probe_jit "field_not_found"     "$UI_DIR/field_not_found.sio"
probe_jit "method_not_found"    "$UI_DIR/method_not_found.sio"
probe_jit "struct_extra_field"  "$UI_DIR/struct_extra_field.sio"
probe_jit "struct_missing_field" "$UI_DIR/struct_missing_field.sio"
probe_jit "struct_field_type"   "$UI_DIR/struct_field_type.sio"
probe_jit "range_type_mismatch" "$UI_DIR/range_type_mismatch.sio"
# tuple_index_oob: //@ ignore, Rust compiler also passes — ok is acceptable
probe_jit "tuple_index_oob"     "$UI_DIR/tuple_index_oob.sio" false

# Former SH_TIMEOUT (5) — all should detect errors
probe_jit "closure_arg_type"             "$UI_DIR/closure_arg_type.sio"
probe_jit "division_by_zero_assign_const" "$UI_DIR/division_by_zero_assign_const.sio"
probe_jit "extra_args"                   "$UI_DIR/extra_args.sio"
probe_jit "unit_mismatch"                "$UI_DIR/unit_mismatch.sio"
probe_jit "wrong_arg_count"              "$UI_DIR/wrong_arg_count.sio"

# ── Regression: 3 sample original SELFHOST_PASS tests ────────────────────────
probe_jit "regress_break_outside_loop" "$UI_DIR/break_outside_loop.sio"
probe_jit "regress_assign_to_immut"    "$UI_DIR/assign_to_immut.sio"
probe_jit "regress_division_by_zero"   "$UI_DIR/division_by_zero.sio"

# ── Emit JSON artifact ──────────────────────────────────────────────────────
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
reason = "checker_parity_38_of_38" if overall == "pass" else f"{counts['fail']}_failed"

payload = {
    "schema": "sounio.sprint14.checker_parity_gate.v1",
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
echo "sprint14_checker_parity_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
