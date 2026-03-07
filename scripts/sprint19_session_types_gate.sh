#!/usr/bin/env bash
# sprint19_session_types_gate.sh — session types + user-defined effects gate (Tracks S+E1)
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
OUT_JSON="${SOUNIO_SPRINT19_GATE_OUT:-$ROOT_DIR/artifacts/sprint19/session_types_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT19_TIMEOUT_SECS:-180}"

mkdir -p "$(dirname "$OUT_JSON")"

CASES_TSV="/tmp/sprint19_cases_$$.tsv"
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
timeout "$TIMEOUT_SECS" "$SOUC" run self-hosted/compiler/main.sio -- --self-test > /tmp/sprint19_check_main.log 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
  record "selfhost_compiler_main_self_test" "pass" "all_checks_passed"
elif [ $rc -eq 124 ]; then
  record "selfhost_compiler_main_self_test" "not_run" "timeout"
else
  reason="$(tail -1 /tmp/sprint19_check_main.log | tr -s ' ' | head -c 80 || echo error)"
  record "selfhost_compiler_main_self_test" "fail" "check_failed: $reason"
fi

# Case 2: Session/Chan/Send/Recv/Close/Dual tokens present in lexer/token.sio
if grep -q "Session," self-hosted/lexer/token.sio 2>/dev/null \
   && grep -q "Chan," self-hosted/lexer/token.sio 2>/dev/null \
   && grep -q "Send," self-hosted/lexer/token.sio 2>/dev/null \
   && grep -q "Recv," self-hosted/lexer/token.sio 2>/dev/null \
   && grep -q "Close," self-hosted/lexer/token.sio 2>/dev/null \
   && grep -q "Dual," self-hosted/lexer/token.sio 2>/dev/null; then
  record "session_tokens_in_token_sio" "pass" "found"
else
  record "session_tokens_in_token_sio" "fail" "missing_session_tokens"
fi

# Case 3: Session keyword table entries present in lexer/tables.sio
if grep -q "TokenKind::Session" self-hosted/lexer/tables.sio 2>/dev/null \
   && grep -q "TokenKind::Chan" self-hosted/lexer/tables.sio 2>/dev/null \
   && grep -q "TokenKind::Close" self-hosted/lexer/tables.sio 2>/dev/null; then
  record "session_keywords_in_tables_sio" "pass" "found"
else
  record "session_keywords_in_tables_sio" "fail" "missing_keyword_table_entries"
fi

# Case 4: AST nodes for session types present in parser/ast.sio
if grep -q "SessionDecl" self-hosted/parser/ast.sio 2>/dev/null \
   && grep -q "TypeChan" self-hosted/parser/ast.sio 2>/dev/null \
   && grep -q "ItemSession" self-hosted/parser/ast.sio 2>/dev/null \
   && grep -q "session_decl" self-hosted/parser/ast.sio 2>/dev/null; then
  record "session_ast_nodes_present" "pass" "found"
else
  record "session_ast_nodes_present" "fail" "missing_ast_nodes"
fi

# Case 5: parse_session_item and parse_chan_type implemented in parser
if grep -q "parse_session_item" self-hosted/parser/items.sio 2>/dev/null \
   && grep -q "parse_chan_type" self-hosted/parser/types.sio 2>/dev/null; then
  record "session_parser_impl_present" "pass" "found"
else
  record "session_parser_impl_present" "fail" "missing_parser_impl"
fi

# Case 6: TySession/TyChan/TySessionEnd TypeKind variants in check/types.sio
if grep -q "TySession," self-hosted/check/types.sio 2>/dev/null \
   && grep -q "TyChan," self-hosted/check/types.sio 2>/dev/null \
   && grep -q "TySessionEnd," self-hosted/check/types.sio 2>/dev/null; then
  record "session_typekind_variants_present" "pass" "found"
else
  record "session_typekind_variants_present" "fail" "missing_typekind_variants"
fi

# Case 7: ty_session / ty_chan / ty_session_end constructors in check/types.sio
if grep -q "fn ty_session(" self-hosted/check/types.sio 2>/dev/null \
   && grep -q "fn ty_chan(" self-hosted/check/types.sio 2>/dev/null \
   && grep -q "fn ty_session_end(" self-hosted/check/types.sio 2>/dev/null \
   && grep -q "fn is_session_type(" self-hosted/check/types.sio 2>/dev/null \
   && grep -q "fn session_dual(" self-hosted/check/types.sio 2>/dev/null; then
  record "session_type_constructors_present" "pass" "found"
else
  record "session_type_constructors_present" "fail" "missing_type_constructors"
fi

# Case 8: ItemSession dispatch arms in check/check.sio
if grep -q "ItemSession =>" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "TypeChan =>" self-hosted/check/check.sio 2>/dev/null \
   && grep -q "TypeSession =>" self-hosted/check/check.sio 2>/dev/null; then
  record "session_checker_dispatch_present" "pass" "found"
else
  record "session_checker_dispatch_present" "fail" "missing_checker_dispatch"
fi

# Case 9: session type compatibility rules in check/compat.sio
if grep -q "TyChan" self-hosted/check/compat.sio 2>/dev/null \
   && grep -q "TySession" self-hosted/check/compat.sio 2>/dev/null \
   && grep -q "TySessionEnd" self-hosted/check/compat.sio 2>/dev/null; then
  record "session_compat_rules_present" "pass" "found"
else
  record "session_compat_rules_present" "fail" "missing_compat_rules"
fi

# Case 10: E1 — user-defined effect fixture exists and has run-pass annotation
if [ -f "tests/frontend/effect_user_defined_basic.sio" ] \
   && grep -q "run-pass" tests/frontend/effect_user_defined_basic.sio 2>/dev/null \
   && grep -q "^effect " tests/frontend/effect_user_defined_basic.sio 2>/dev/null; then
  record "effect_user_defined_fixture_present" "pass" "found"
else
  record "effect_user_defined_fixture_present" "fail" "fixture_missing_or_incomplete"
fi

# Case 11: session_basic.sio fixture present
if [ -f "tests/frontend/session_basic.sio" ] \
   && grep -q "session " tests/frontend/session_basic.sio 2>/dev/null \
   && grep -q "send " tests/frontend/session_basic.sio 2>/dev/null \
   && grep -q "recv " tests/frontend/session_basic.sio 2>/dev/null \
   && grep -q "close" tests/frontend/session_basic.sio 2>/dev/null; then
  record "session_basic_fixture_present" "pass" "found"
else
  record "session_basic_fixture_present" "fail" "fixture_missing_or_incomplete"
fi

# Case 12: session_basic.sio passes self-hosted frontend preflight
run_selfhost_preflight_case \
  "preflight_session_basic" \
  "tests/frontend/session_basic.sio" \
  "/tmp/sprint19_preflight_session_basic.log"

# Case 13: session_http_basic.sio passes self-hosted frontend preflight
run_selfhost_preflight_case \
  "preflight_session_http_basic" \
  "tests/frontend/session_http_basic.sio" \
  "/tmp/sprint19_preflight_session_http_basic.log"

# Case 14: effect_user_defined_basic.sio passes self-hosted frontend preflight
run_selfhost_preflight_case \
  "preflight_effect_user_defined_basic" \
  "tests/frontend/effect_user_defined_basic.sio" \
  "/tmp/sprint19_preflight_effect_user_defined_basic.log"

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
    "schema": "sounio.sprint19.session_types_gate.v1",
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
echo "sprint19_session_types_gate: out_json=$(to_rel "$OUT_JSON") status=$status"
[ "$status" = "pass" ] && exit 0
[ "$status" = "not_run" ] && exit 3
exit 2
