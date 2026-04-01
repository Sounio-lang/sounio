#!/usr/bin/env bash
# sprint5_contest_validated_gate.sh — Contest/Validated/Robust type-system gate
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
OUT_JSON="${SOUNIO_SPRINT5_GATE_OUT:-$ROOT_DIR/artifacts/sprint5/contest_validated_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT5_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint5_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

extract_error_patterns() {
  local source_file="$1"
  python3 - "$source_file" <<'PY'
import re, sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
for line in source.splitlines():
    m = re.match(r"\s*//@\s*error-pattern:\s*(.*)", line)
    if m:
        print(m.group(1))
PY
}

run_selfhost_preflight_ok_case() {
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

run_selfhost_expect_fail_case() {
  local case_name="$1" source_file="$2" log_file="$3"
  shift 3
  local patterns=("$@")
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- --probe-frontend "$source_file" > "$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
    return
  fi
  if grep -q "probe_frontend: ok" "$log_file" 2>/dev/null; then
    record "$case_name" "fail" "expected_compile_failure_but_probe_ok"
    return
  fi
  if ! grep -q "probe_frontend: fail" "$log_file" 2>/dev/null; then
    local reason
    reason="$(tail -1 "$log_file" | tr -s ' ' | head -c 120 || echo error)"
    record "$case_name" "fail" "ambiguous_probe_output: $reason"
    return
  fi
  local pattern
  for pattern in "${patterns[@]}"; do
    if ! grep -qiF "$pattern" "$log_file" 2>/dev/null; then
      record "$case_name" "fail" "missing_error_pattern: $pattern"
      return
    fi
  done
  record "$case_name" "pass" "selfhost_frontend_compile_fail_matched"
}

# Case 1: self-hosted compiler frontend probe for main.sio passes
set +e
timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- --self-test > /tmp/sprint5_check_main.log 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  record "selfhost_compiler_main_self_test" "pass" "all_checks_passed"
elif [ $rc -eq 124 ]; then
  record "selfhost_compiler_main_self_test" "not_run" "timeout"
else
  reason="$(tail -1 /tmp/sprint5_check_main.log | tr -s ' ' | head -c 80 || echo error)"
  record "selfhost_compiler_main_self_test" "fail" "check_failed: $reason"
fi

# Case 2: TyPolicy/TyContest/TyRobust in check/types.sio
for variant in TyPolicy TyContest TyRobust; do
  if grep -q "$variant" self-hosted/check/types.sio 2>/dev/null; then
    record "typekind_$variant" "pass" "found_in_types_sio"
  else
    record "typekind_$variant" "fail" "not_found_in_types_sio"
  fi
done

# Case 3: ty_policy/ty_contest/ty_robust constructors present
for ctor in ty_policy ty_contest ty_robust; do
  if grep -q "^fn $ctor(" self-hosted/check/types.sio 2>/dev/null; then
    record "ctor_$ctor" "pass" "found_in_types_sio"
  else
    record "ctor_$ctor" "fail" "not_found_in_types_sio"
  fi
done

# Case 4: compile-fail fixtures exist
for f in contest_no_silent_unwrap contest_requires_annotation robust_not_validated; do
  tgt="tests/compile-fail/${f}.sio"
  if [ -f "$tgt" ]; then
    record "compile_fail_stub_$f" "pass" "file_exists"
  else
    record "compile_fail_stub_$f" "fail" "file_missing"
  fi
done

# Case 5: check_contest_type/check_policy_type/check_robust_type in epistemic.sio
for fn in check_contest_type check_policy_type check_robust_type; do
  if grep -q "^fn $fn(" self-hosted/check/epistemic.sio 2>/dev/null; then
    record "epistemic_fn_$fn" "pass" "found_in_epistemic_sio"
  else
    record "epistemic_fn_$fn" "fail" "not_found_in_epistemic_sio"
  fi
done

# Case 6: lower_type_expr dispatches TypePolicy/TypeContest/TypeRobust
for kind in TypePolicy TypeContest TypeRobust; do
  if grep -q "$kind =>" self-hosted/check/check.sio 2>/dev/null; then
    record "dispatch_$kind" "pass" "found_in_check_sio"
  else
    record "dispatch_$kind" "fail" "not_found_in_check_sio"
  fi
done

# Case 7-9: self-hosted frontend enforces the compile-fail fixtures
for f in contest_no_silent_unwrap contest_requires_annotation robust_not_validated; do
  tgt="tests/compile-fail/${f}.sio"
  if [ -f "$tgt" ]; then
    mapfile -t patterns < <(extract_error_patterns "$tgt")
    run_selfhost_expect_fail_case \
      "compile_fail_fixture_${f}" \
      "$tgt" \
      "/tmp/${f}.selfhost_probe.log" \
      "${patterns[@]}"
  fi
done

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
    "schema": "sounio.sprint5.contest_validated_gate.v1",
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
echo "sprint5_contest_validated_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
