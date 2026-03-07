#!/usr/bin/env bash
# sprint19_epistemic_session_gate.sh — epistemic session types stdlib gate
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

OUT_JSON="${SOUNIO_SPRINT19_SESSION_GATE_OUT:-$ROOT_DIR/artifacts/sprint19/epistemic_session_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT19_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint19_session_cases_$$.tsv"
: > "$CASES_TSV"

to_rel() { [[ "$1" == "$ROOT_DIR/"* ]] && printf '%s' "${1#$ROOT_DIR/}" || printf '%s' "$1"; }

record() {
  local name="$1" status="$2" reason="$3"
  printf '%s\t%s\t%s\n' "$name" "$status" "$reason" >> "$CASES_TSV"
}

run_case_cmd() {
  local case_name="$1" pass_reason="$2"
  shift 2
  local log_file="/tmp/${case_name}_$$.log"
  rm -f "$log_file"
  set +e
  timeout "$TIMEOUT_SECS" "$@" >"$log_file" 2>&1
  local rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    record "$case_name" "pass" "$pass_reason"
  elif [ $rc -eq 124 ]; then
    record "$case_name" "not_run" "timeout"
  else
    local reason
    reason="$(tail -5 "$log_file" | tr '\n' ' ' | tr -s ' ' | head -c 160 || echo error)"
    record "$case_name" "fail" "$reason"
  fi
}

# --- File existence checks ---

if [ -f "$ROOT_DIR/stdlib/web/http.sio" ]; then
  record "stdlib_web_http_exists" "pass" "found"
else
  record "stdlib_web_http_exists" "fail" "missing"
fi

if [ -f "$ROOT_DIR/stdlib/web/epistemic_http.sio" ]; then
  record "stdlib_web_epistemic_http_exists" "pass" "found"
else
  record "stdlib_web_epistemic_http_exists" "fail" "missing"
fi

if [ -f "$ROOT_DIR/stdlib/web/websocket.sio" ]; then
  record "stdlib_web_websocket_exists" "pass" "found"
else
  record "stdlib_web_websocket_exists" "fail" "missing"
fi

if [ -f "$ROOT_DIR/stdlib/science/ncbi.sio" ]; then
  record "stdlib_science_ncbi_exists" "pass" "found"
else
  record "stdlib_science_ncbi_exists" "fail" "missing"
fi

if [ -f "$ROOT_DIR/stdlib/science/clintrials.sio" ]; then
  record "stdlib_science_clintrials_exists" "pass" "found"
else
  record "stdlib_science_clintrials_exists" "fail" "missing"
fi

if [ -f "$ROOT_DIR/paper/epistemic-types/session_types.tex" ]; then
  record "paper_session_types_tex_exists" "pass" "found"
else
  record "paper_session_types_tex_exists" "fail" "missing"
fi

# --- Content checks ---

if grep -q "provenance_quality" "$ROOT_DIR/stdlib/web/epistemic_http.sio" 2>/dev/null; then
  record "epistemic_http_provenance_quality_fn" "pass" "found"
else
  record "epistemic_http_provenance_quality_fn" "fail" "missing"
fi

if grep -q "expression_uncertainty" "$ROOT_DIR/stdlib/science/ncbi.sio" 2>/dev/null; then
  record "ncbi_expression_uncertainty_fn" "pass" "found"
else
  record "ncbi_expression_uncertainty_fn" "fail" "missing"
fi

if grep -q "trial_evidence_quality" "$ROOT_DIR/stdlib/science/clintrials.sio" 2>/dev/null; then
  record "clintrials_trial_evidence_quality_fn" "pass" "found"
else
  record "clintrials_trial_evidence_quality_fn" "fail" "missing"
fi

if grep -q "EpistemicResponse" "$ROOT_DIR/stdlib/web/epistemic_http.sio" 2>/dev/null; then
  record "epistemic_http_epistemicresponse_struct" "pass" "found"
else
  record "epistemic_http_epistemicresponse_struct" "fail" "missing"
fi

# --- souc type-check (if available) ---

if [ -n "$SOUC" ] && [ -x "$SOUC" ]; then
  run_case_cmd "souc_check_web_http" "typecheck_ok" \
    "$SOUC" check "$ROOT_DIR/stdlib/web/http.sio"
  run_case_cmd "souc_check_web_epistemic_http" "typecheck_ok" \
    "$SOUC" check "$ROOT_DIR/stdlib/web/epistemic_http.sio"
  run_case_cmd "souc_check_science_ncbi" "typecheck_ok" \
    "$SOUC" check "$ROOT_DIR/stdlib/science/ncbi.sio"
else
  record "souc_check_web_http" "not_run" "souc_unavailable"
  record "souc_check_web_epistemic_http" "not_run" "souc_unavailable"
  record "souc_check_science_ncbi" "not_run" "souc_unavailable"
fi

python3 - "$OUT_JSON" "$CASES_TSV" "${SOUC:-none}" "$TIMEOUT_SECS" << 'PY'
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
    name, status, reason = raw.split("\t", 2)
    if status not in counts:
        status, reason = "fail", f"invalid_{status}"
    counts[status] += 1
    cases.append({"name": name, "status": status, "reason": reason})

overall = "pass" if counts["fail"] == 0 and counts["not_run"] == 0 else \
          "not_run" if counts["pass"] == 0 and counts["fail"] == 0 else \
          "not_run" if counts["fail"] == 0 and counts["not_run"] > 0 else "fail"
reason = "all_cases_passed" if overall == "pass" else \
         f"{counts['fail']}_failed" if counts["fail"] else "one_or_more_not_run"

payload = {
    "schema": "sounio.sprint19.epistemic_session_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": overall,
    "reason": reason,
    "config": {"souc": souc, "timeout_seconds": timeout_secs},
    "metrics": {
        "total": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "not_run": counts["not_run"],
    },
    "cases": cases,
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_json}")
print(f"status={overall} reason={reason}")
PY

rm -f "$CASES_TSV"
status="$(python3 -c "import json; d=json.load(open('$OUT_JSON')); print(d.get('status','fail'))")"
echo "sprint19_epistemic_session_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
