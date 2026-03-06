#!/usr/bin/env bash
# sprint5_contest_validated_gate.sh — Contest/Validated/Robust type-system gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUNIO_SOUC:-$ROOT_DIR/target/debug/souc}"
OUT_JSON="${SOUNIO_SPRINT5_GATE_OUT:-$ROOT_DIR/artifacts/sprint5/contest_validated_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT5_TIMEOUT_SECS:-60}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint5_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

# Case 1: souc check self-hosted/compiler/main.sio passes
set +e
timeout "$TIMEOUT_SECS" "$SOUC" check self-hosted/compiler/main.sio > /tmp/sprint5_check_main.log 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  record "check_main_sio" "pass" "all_checks_passed"
elif [ $rc -eq 124 ]; then
  record "check_main_sio" "not_run" "timeout"
else
  reason="$(tail -1 /tmp/sprint5_check_main.log | tr -s ' ' | head -c 80 || echo error)"
  record "check_main_sio" "fail" "check_failed: $reason"
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

# Case 4: compile-fail test stubs exist
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
