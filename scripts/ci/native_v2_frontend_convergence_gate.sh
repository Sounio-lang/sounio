#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-frontend] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-frontend] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_FRONTEND_DIR:-$(mktemp -d /tmp/sounio-native-v2-frontend.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
SUMMARY_JSON="${SOUNIO_NATIVE_V2_FRONTEND_SUMMARY:-$ARTIFACT_DIR/native_v2_frontend_convergence.v1.json}"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
STRICT_MODE="${SOUNIO_NATIVE_V2_FRONTEND_STRICT:-0}"
RUN_CPU_UMBRELLA="${SOUNIO_NATIVE_V2_FRONTEND_RUN_CPU_UMBRELLA:-1}"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

cat >"$RESULTS_TSV" <<'EOF'
case_id	command	rc	log_path
EOF

append_result() {
  local case_id="$1"
  local command="$2"
  local rc="$3"
  local log_path="$4"
  printf '%s\t%s\t%s\t%s\n' "$case_id" "$command" "$rc" "$log_path" >>"$RESULTS_TSV"
}

run_case() {
  local case_id="$1"
  local log_path="$2"
  shift 2

  echo "[native-v2-frontend] running $case_id"
  set +e
  "$@" >"$log_path" 2>&1
  local rc=$?
  set -e
  append_result "$case_id" "$*" "$rc" "$log_path"
}

run_cpu_umbrella() {
  local log_path="$LOG_DIR/cpu_umbrella.log"
  local gate_dir="$OUT_DIR/cpu_umbrella"

  if [[ "$RUN_CPU_UMBRELLA" =~ ^(0|false|False|FALSE|no|No|NO|off|Off|OFF)$ ]]; then
    echo "[native-v2-frontend] skipping cpu_umbrella by env"
    append_result "cpu_umbrella" "skipped_by_env" "0" "$log_path"
    printf 'skipped_by_env\n' >"$log_path"
    return 0
  fi

  run_case "cpu_umbrella" "$log_path" env \
    "SOUNIO_NATIVE_V2_CPU_COMPILER_DIR=$gate_dir" \
    bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh
}

emit_summary_json() {
  python3 - "$SUMMARY_JSON" "$RESULTS_TSV" "$SOUC_BIN" "$OUT_DIR" "$STRICT_MODE" "$ROOT_DIR" <<'PY'
import csv
import json
import pathlib
import sys
from datetime import datetime, timezone

summary_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])
souc_bin = sys.argv[3]
out_dir = pathlib.Path(sys.argv[4])
strict_mode = sys.argv[5].lower() in {"1", "true", "yes", "on"}
root = pathlib.Path(sys.argv[6])

def read(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def excerpt(text, limit=1200):
    text = text.replace("\x00", "")
    if len(text) <= limit:
        return text
    return text[-limit:]

with open(results_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

cases = []
classifications = []
failure_classes = []

module_parse = read(root / "self-hosted/compiler/module_parse.sio")
lexer_mod = read(root / "self-hosted/lexer/mod.sio")
stale_sourcefile_api = (
    "sourcefile_load" in module_parse
    and "sourcefile_parse_program_loaded" in module_parse
    and "pub fn lex_file_parse_items" in lexer_mod
    and "pub fn sourcefile_load" not in lexer_mod
)
if stale_sourcefile_api:
    classifications.append({
        "class": "frontend_parse_surface",
        "reason": "module_parse_uses_removed_sourcefile_api",
        "detail": "module_parse.sio still imports sourcefile_* while lexer::mod exposes lex_file_parse_items",
    })

case_by_id = {}
for row in rows:
    text = read(row["log_path"])
    rc = int(row["rc"])
    entry = {
        "case_id": row["case_id"],
        "command": row["command"],
        "rc": rc,
        "log_path": row["log_path"],
        "log_excerpt": excerpt(text),
    }
    cases.append(entry)
    case_by_id[row["case_id"]] = entry

def log_of(case_id):
    entry = case_by_id.get(case_id)
    if not entry:
        return ""
    return read(entry["log_path"])

self_log = log_of("lean_frontend_self_test")
hello_log = log_of("lean_frontend_hello_check")
ir_log = log_of("lean_frontend_hello_ir_summary")
modular_ir_log = log_of("lean_modular_hello_ir_summary")
imported_ir_log = log_of("lean_frontend_imported_ir_summary")
imported_modular_ir_log = log_of("lean_modular_imported_ir_summary")
main_probe_plain_log = log_of("main_probe_load_ir")
main_probe_log = log_of("main_probe_load_ir_trace")
main_probe_imported_plain_log = log_of("main_probe_imported_load_ir")
main_probe_imported_log = log_of("main_probe_imported_load_ir_trace")
main_probe_imported_call_target_log = log_of("main_probe_imported_call_target_native_compile")
main_probe_imported_core_entry_log = log_of("main_probe_imported_core_entry_native_compile")
main_probe_imported_expr_eval_log = log_of("main_probe_imported_expr_eval_native_compile")
main_probe_imported_expr_calls_log = log_of("main_probe_imported_expr_calls_native_compile")
main_probe_imported_expr_lets_log = log_of("main_probe_imported_expr_lets_native_compile")
main_probe_imported_expr_multilet_log = log_of("main_probe_imported_expr_multilet_native_compile")
main_probe_imported_expr_sub_log = log_of("main_probe_imported_expr_sub_native_compile")
main_probe_imported_chain_log = log_of("main_probe_imported_chain_native_compile")
main_probe_imported_stdout_log = log_of("main_probe_imported_stdout_native_compile")

if case_by_id.get("cpu_umbrella", {}).get("rc", 0) != 0:
    failure_classes.append("cpu_umbrella")
    classifications.append({
        "class": "cpu_umbrella",
        "reason": "native_v2_cpu_compiler_umbrella_failed",
        "case_id": "cpu_umbrella",
    })

if case_by_id.get("lean_frontend_self_test", {}).get("rc", 0) != 0:
    reason = "parse_failed" if "parse failed" in self_log else "self_test_failed"
    failure_classes.append("frontend_parse_surface")
    classifications.append({
        "class": "frontend_parse_surface",
        "reason": reason,
        "case_id": "lean_frontend_self_test",
    })

if case_by_id.get("lean_frontend_hello_check", {}).get("rc", 0) != 0:
    if "Segmentation fault" in hello_log and "Type checking module 0" in hello_log:
        reason = "checker_runtime_segfault_after_typechecking_module_0"
        failure_classes.append("checker_runtime")
    elif "Type checking module 0" in hello_log and "Type check complete for module 0" not in hello_log and "Type check failed for module 0" not in hello_log:
        reason = "checker_runtime_exit_after_typechecking_module_0"
        failure_classes.append("checker_runtime")
    elif "Type check failed" in hello_log:
        reason = "type_check_rejected"
        failure_classes.append("ontology_kernel/checker")
    else:
        reason = "lean_frontend_hello_check_failed"
        failure_classes.append("frontend_convergence")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "lean_frontend_hello_check",
    })

if case_by_id.get("lean_frontend_hello_ir_summary", {}).get("rc", 0) != 0:
    if "Segmentation fault" in ir_log:
        reason = "ir_summary_artifact_path_segfault"
        failure_classes.append("checker_artifacts")
    elif "Type check failed" in ir_log:
        reason = "ir_summary_type_check_rejected"
        failure_classes.append("ontology_kernel/checker")
    elif "Merged IR:" not in ir_log:
        reason = "checker_runtime_exit_before_ir_summary"
        failure_classes.append("checker_runtime")
    else:
        reason = "ir_summary_failed"
        failure_classes.append("ir_lowering")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "lean_frontend_hello_ir_summary",
    })

if case_by_id.get("lean_modular_hello_ir_summary", {}).get("rc", 0) != 0:
    if "Segmentation fault" in modular_ir_log:
        reason = "lean_modular_ir_summary_segfault"
        failure_classes.append("ir_loader_runtime")
    elif "souc-lean ir-summary: functions=" not in modular_ir_log:
        reason = "lean_modular_ir_summary_missing_success_witness"
        failure_classes.append("ir_loader")
    else:
        reason = "lean_modular_ir_summary_failed"
        failure_classes.append("ir_loader")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "lean_modular_hello_ir_summary",
    })

if case_by_id.get("lean_frontend_imported_ir_summary", {}).get("rc", 0) != 0:
    if "souc-lean-frontend ir-summary: functions=" not in imported_ir_log:
        reason = "lean_frontend_imported_ir_summary_missing_success_witness"
        failure_classes.append("imported_ir_loader")
    else:
        reason = "lean_frontend_imported_ir_summary_failed"
        failure_classes.append("imported_ir_loader")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "lean_frontend_imported_ir_summary",
    })

if case_by_id.get("lean_modular_imported_ir_summary", {}).get("rc", 0) != 0:
    if "souc-lean ir-summary: functions=" not in imported_modular_ir_log:
        reason = "lean_modular_imported_ir_summary_missing_success_witness"
        failure_classes.append("imported_ir_loader")
    else:
        reason = "lean_modular_imported_ir_summary_failed"
        failure_classes.append("imported_ir_loader")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "lean_modular_imported_ir_summary",
    })

if case_by_id.get("main_probe_load_ir", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_plain_log:
        reason = "main_probe_load_ir_segfault"
        failure_classes.append("ir_loader_runtime")
    elif "probe_load_ir: ok" not in main_probe_plain_log:
        reason = "main_probe_load_ir_missing_ok"
        failure_classes.append("ir_loader")
    else:
        reason = "main_probe_load_ir_failed"
        failure_classes.append("ir_loader")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_load_ir",
    })

if case_by_id.get("main_probe_load_ir_trace", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_log:
        reason = "main_probe_load_ir_trace_segfault"
        failure_classes.append("ir_loader_runtime")
    elif "probe_load_ir_trace: ok" not in main_probe_log:
        reason = "main_probe_load_ir_trace_missing_ok"
        failure_classes.append("ir_loader")
    else:
        reason = "main_probe_load_ir_trace_failed"
        failure_classes.append("ir_loader")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_load_ir_trace",
    })
elif "body_lowered=1" not in main_probe_log:
    failure_classes.append("ir_body_lowering")
    classifications.append({
        "class": "ir_body_lowering",
        "reason": "main_probe_load_ir_trace_missing_body_lowered_witness",
        "case_id": "main_probe_load_ir_trace",
    })

if case_by_id.get("main_probe_imported_load_ir", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_plain_log:
        reason = "main_probe_imported_load_ir_segfault"
        failure_classes.append("imported_ir_loader_runtime")
    elif "probe_load_ir: ok" not in main_probe_imported_plain_log:
        reason = "main_probe_imported_load_ir_missing_ok"
        failure_classes.append("imported_ir_loader")
    else:
        reason = "main_probe_imported_load_ir_failed"
        failure_classes.append("imported_ir_loader")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_load_ir",
    })

if case_by_id.get("main_probe_imported_load_ir_trace", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_log:
        reason = "main_probe_imported_load_ir_trace_segfault"
        failure_classes.append("imported_ir_loader_runtime")
    elif "probe_load_ir_trace: ok" not in main_probe_imported_log:
        reason = "main_probe_imported_load_ir_trace_missing_ok"
        failure_classes.append("imported_ir_loader")
    else:
        reason = "main_probe_imported_load_ir_trace_failed"
        failure_classes.append("imported_ir_loader")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_load_ir_trace",
    })
elif "body_lowered=3" not in main_probe_imported_log:
    failure_classes.append("imported_ir_body_lowering")
    classifications.append({
        "class": "imported_ir_body_lowering",
        "reason": "main_probe_imported_load_ir_trace_missing_body_lowered_3_witness",
        "case_id": "main_probe_imported_load_ir_trace",
    })

if case_by_id.get("main_probe_imported_call_target_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_call_target_log:
        reason = "main_probe_imported_call_target_native_compile_segfault"
        failure_classes.append("imported_call_target_native_runtime")
    else:
        reason = "main_probe_imported_call_target_native_compile_failed"
        failure_classes.append("imported_call_target_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_call_target_native_compile",
    })

if case_by_id.get("main_probe_imported_core_entry_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_core_entry_log:
        reason = "main_probe_imported_core_entry_native_compile_segfault"
        failure_classes.append("imported_core_entry_native_runtime")
    else:
        reason = "main_probe_imported_core_entry_native_compile_failed"
        failure_classes.append("imported_core_entry_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_core_entry_native_compile",
    })

if case_by_id.get("main_probe_imported_expr_eval_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_expr_eval_log:
        reason = "main_probe_imported_expr_eval_native_compile_segfault"
        failure_classes.append("imported_expr_eval_native_runtime")
    else:
        reason = "main_probe_imported_expr_eval_native_compile_failed"
        failure_classes.append("imported_expr_eval_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_expr_eval_native_compile",
    })

if case_by_id.get("main_probe_imported_expr_calls_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_expr_calls_log:
        reason = "main_probe_imported_expr_calls_native_compile_segfault"
        failure_classes.append("imported_expr_calls_native_runtime")
    else:
        reason = "main_probe_imported_expr_calls_native_compile_failed"
        failure_classes.append("imported_expr_calls_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_expr_calls_native_compile",
    })

if case_by_id.get("main_probe_imported_expr_lets_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_expr_lets_log:
        reason = "main_probe_imported_expr_lets_native_compile_segfault"
        failure_classes.append("imported_expr_lets_native_runtime")
    else:
        reason = "main_probe_imported_expr_lets_native_compile_failed"
        failure_classes.append("imported_expr_lets_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_expr_lets_native_compile",
    })

if case_by_id.get("main_probe_imported_expr_multilet_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_expr_multilet_log:
        reason = "main_probe_imported_expr_multilet_native_compile_segfault"
        failure_classes.append("imported_expr_multilet_native_runtime")
    else:
        reason = "main_probe_imported_expr_multilet_native_compile_failed"
        failure_classes.append("imported_expr_multilet_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_expr_multilet_native_compile",
    })

if case_by_id.get("main_probe_imported_expr_sub_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_expr_sub_log:
        reason = "main_probe_imported_expr_sub_native_compile_segfault"
        failure_classes.append("imported_expr_sub_native_runtime")
    else:
        reason = "main_probe_imported_expr_sub_native_compile_failed"
        failure_classes.append("imported_expr_sub_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_expr_sub_native_compile",
    })

if case_by_id.get("main_probe_imported_chain_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_chain_log:
        reason = "main_probe_imported_chain_native_compile_segfault"
        failure_classes.append("imported_chain_native_runtime")
    else:
        reason = "main_probe_imported_chain_native_compile_failed"
        failure_classes.append("imported_chain_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_chain_native_compile",
    })

if case_by_id.get("main_probe_imported_stdout_native_compile", {}).get("rc", 0) != 0:
    if "Segmentation fault" in main_probe_imported_stdout_log:
        reason = "main_probe_imported_stdout_native_compile_segfault"
        failure_classes.append("imported_stdout_native_runtime")
    elif "stdout_mismatch" in main_probe_imported_stdout_log:
        reason = "main_probe_imported_stdout_native_stdout_mismatch"
        failure_classes.append("imported_stdout_native_parity")
    else:
        reason = "main_probe_imported_stdout_native_compile_failed"
        failure_classes.append("imported_stdout_native")
    classifications.append({
        "class": failure_classes[-1],
        "reason": reason,
        "case_id": "main_probe_imported_stdout_native_compile",
    })

required = [
    "cpu_umbrella",
    "lean_frontend_check",
    "lean_frontend_self_test",
    "lean_frontend_hello_check",
    "lean_frontend_hello_ir_summary",
    "lean_modular_hello_ir_summary",
    "lean_frontend_imported_ir_summary",
    "lean_modular_imported_ir_summary",
    "main_probe_load_ir",
    "main_probe_load_ir_trace",
    "main_probe_imported_load_ir",
    "main_probe_imported_load_ir_trace",
    "main_probe_imported_native_compile",
    "main_probe_imported_call_target_native_compile",
    "main_probe_imported_core_entry_native_compile",
    "main_probe_imported_expr_eval_native_compile",
    "main_probe_imported_expr_calls_native_compile",
    "main_probe_imported_expr_lets_native_compile",
    "main_probe_imported_expr_multilet_native_compile",
    "main_probe_imported_expr_sub_native_compile",
    "main_probe_imported_chain_native_compile",
    "main_probe_imported_stdout_native_compile",
]
all_required_present = all(case_id in case_by_id for case_id in required)
all_required_pass = all(case_by_id.get(case_id, {}).get("rc") == 0 for case_id in required)

if all_required_present and all_required_pass and not classifications:
    status = "pass"
elif classifications:
    status = "partial"
else:
    status = "fail"

payload = {
    "schema": "sounio.native_v2_frontend_convergence.v1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": status,
    "strict_mode": strict_mode,
    "compiler_resolved": souc_bin,
    "target": "x86_64-linux",
    "fallback_path": "none",
    "host_callback": "none",
    "out_dir": str(out_dir),
    "results_tsv": str(results_path),
    "case_count": len(cases),
    "pass_count": sum(1 for c in cases if c["rc"] == 0),
    "fail_count": sum(1 for c in cases if c["rc"] != 0),
    "classifications": classifications,
    "failure_classes": sorted(set(failure_classes)),
    "cases": cases,
    "acceptance": {
        "cpu_umbrella_passed": case_by_id.get("cpu_umbrella", {}).get("rc") == 0,
        "lean_frontend_check_passed": case_by_id.get("lean_frontend_check", {}).get("rc") == 0,
        "lean_frontend_self_test_passed": case_by_id.get("lean_frontend_self_test", {}).get("rc") == 0,
        "lean_frontend_hello_check_passed": case_by_id.get("lean_frontend_hello_check", {}).get("rc") == 0,
        "lean_frontend_hello_ir_summary_passed": case_by_id.get("lean_frontend_hello_ir_summary", {}).get("rc") == 0,
        "lean_modular_hello_ir_summary_passed": case_by_id.get("lean_modular_hello_ir_summary", {}).get("rc") == 0,
        "lean_frontend_imported_ir_summary_passed": case_by_id.get("lean_frontend_imported_ir_summary", {}).get("rc") == 0,
        "lean_modular_imported_ir_summary_passed": case_by_id.get("lean_modular_imported_ir_summary", {}).get("rc") == 0,
        "main_probe_load_ir_passed": case_by_id.get("main_probe_load_ir", {}).get("rc") == 0,
        "main_probe_load_ir_trace_passed": case_by_id.get("main_probe_load_ir_trace", {}).get("rc") == 0,
        "main_probe_imported_load_ir_passed": case_by_id.get("main_probe_imported_load_ir", {}).get("rc") == 0,
        "main_probe_imported_load_ir_trace_passed": case_by_id.get("main_probe_imported_load_ir_trace", {}).get("rc") == 0,
        "main_probe_imported_native_compile_passed": case_by_id.get("main_probe_imported_native_compile", {}).get("rc") == 0,
        "main_probe_imported_call_target_native_compile_passed": case_by_id.get("main_probe_imported_call_target_native_compile", {}).get("rc") == 0,
        "main_probe_imported_core_entry_native_compile_passed": case_by_id.get("main_probe_imported_core_entry_native_compile", {}).get("rc") == 0,
        "main_probe_imported_expr_eval_native_compile_passed": case_by_id.get("main_probe_imported_expr_eval_native_compile", {}).get("rc") == 0,
        "main_probe_imported_expr_calls_native_compile_passed": case_by_id.get("main_probe_imported_expr_calls_native_compile", {}).get("rc") == 0,
        "main_probe_imported_expr_lets_native_compile_passed": case_by_id.get("main_probe_imported_expr_lets_native_compile", {}).get("rc") == 0,
        "main_probe_imported_expr_multilet_native_compile_passed": case_by_id.get("main_probe_imported_expr_multilet_native_compile", {}).get("rc") == 0,
        "main_probe_imported_expr_sub_native_compile_passed": case_by_id.get("main_probe_imported_expr_sub_native_compile", {}).get("rc") == 0,
        "main_probe_imported_chain_native_compile_passed": case_by_id.get("main_probe_imported_chain_native_compile", {}).get("rc") == 0,
        "main_probe_imported_stdout_native_compile_passed": case_by_id.get("main_probe_imported_stdout_native_compile", {}).get("rc") == 0,
        "main_probe_body_lowered_passed": "body_lowered=1" in main_probe_log,
        "main_probe_imported_body_lowered_passed": "body_lowered=3" in main_probe_imported_log,
    },
    "next_action": "promote use-bearing imported modules from source-summary body recognition to full AST-lowered imported native codegen",
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if status == "fail" or (strict_mode and status != "pass"):
    raise SystemExit(1)
PY
}

echo "[native-v2-frontend] souc=$SOUC_BIN"
echo "[native-v2-frontend] out=$OUT_DIR"
echo "[native-v2-frontend] summary=$SUMMARY_JSON"

run_cpu_umbrella
run_case "lean_frontend_check" "$LOG_DIR/lean_frontend.check.log" \
  "$SOUC_BIN" check self-hosted/compiler/lean_frontend.sio
run_case "lean_frontend_self_test" "$LOG_DIR/lean_frontend.self_test.log" \
  "$SOUC_BIN" run self-hosted/compiler/lean_frontend.sio -- --self-test
run_case "lean_frontend_hello_check" "$LOG_DIR/lean_frontend.hello_check.log" \
  "$SOUC_BIN" run self-hosted/compiler/lean_frontend.sio -- --check examples/hello.sio
run_case "lean_frontend_hello_ir_summary" "$LOG_DIR/lean_frontend.hello_ir_summary.log" \
  "$SOUC_BIN" run self-hosted/compiler/lean_frontend.sio -- --ir-summary examples/hello.sio
run_case "lean_modular_hello_ir_summary" "$LOG_DIR/lean_modular.hello_ir_summary.log" \
  "$SOUC_BIN" run self-hosted/compiler/lean.sio -- --ir-summary examples/hello.sio
run_case "lean_frontend_imported_ir_summary" "$LOG_DIR/lean_frontend.imported_ir_summary.log" \
  "$SOUC_BIN" run self-hosted/compiler/lean_frontend.sio -- --ir-summary tests/selfhost/native_runtime/import_nested_main_42.sio
run_case "lean_modular_imported_ir_summary" "$LOG_DIR/lean_modular.imported_ir_summary.log" \
  "$SOUC_BIN" run self-hosted/compiler/lean.sio -- --ir-summary tests/selfhost/native_runtime/import_nested_main_42.sio
run_case "main_probe_load_ir" "$LOG_DIR/main.probe_load_ir.log" \
  "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-load-ir examples/hello.sio
run_case "main_probe_load_ir_trace" "$LOG_DIR/main.probe_load_ir_trace.log" \
  "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-load-ir-trace examples/hello.sio
run_case "main_probe_imported_load_ir" "$LOG_DIR/main.probe_imported_load_ir.log" \
  "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-load-ir tests/selfhost/native_runtime/import_nested_main_42.sio
run_case "main_probe_imported_load_ir_trace" "$LOG_DIR/main.probe_imported_load_ir_trace.log" \
  "$SOUC_BIN" run self-hosted/compiler/main.sio -- --probe-load-ir-trace tests/selfhost/native_runtime/import_nested_main_42.sio
run_case "main_probe_imported_native_compile" "$LOG_DIR/main.probe_imported_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_nested_main_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_nested_main_42.native"
run_case "main_probe_imported_call_target_native_compile" "$LOG_DIR/main.probe_imported_call_target_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_call_target_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_call_target_42.native"
run_case "main_probe_imported_core_entry_native_compile" "$LOG_DIR/main.probe_imported_core_entry_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_core_entry_22.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 22' \
  bash "$SOUC_BIN" "$OUT_DIR/import_core_entry_22.native"
run_case "main_probe_imported_expr_eval_native_compile" "$LOG_DIR/main.probe_imported_expr_eval_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_expr_eval_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_expr_eval_42.native"
run_case "main_probe_imported_expr_calls_native_compile" "$LOG_DIR/main.probe_imported_expr_calls_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_expr_calls_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_expr_calls_42.native"
run_case "main_probe_imported_expr_lets_native_compile" "$LOG_DIR/main.probe_imported_expr_lets_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_expr_lets_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_expr_lets_42.native"
run_case "main_probe_imported_expr_multilet_native_compile" "$LOG_DIR/main.probe_imported_expr_multilet_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_expr_multilet_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_expr_multilet_42.native"
run_case "main_probe_imported_expr_sub_native_compile" "$LOG_DIR/main.probe_imported_expr_sub_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_expr_sub_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_expr_sub_42.native"
run_case "main_probe_imported_chain_native_compile" "$LOG_DIR/main.probe_imported_chain_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_chain_42.sio -o "$2" && chmod +x "$2" && "$2"; rc=$?; test "$rc" -eq 42' \
  bash "$SOUC_BIN" "$OUT_DIR/import_chain_42.native"
run_case "main_probe_imported_stdout_native_compile" "$LOG_DIR/main.probe_imported_stdout_native_compile.log" \
  bash -c '"$1" run self-hosted/compiler/main.sio -- --native-compile tests/selfhost/native_runtime/import_nested_print_42.sio -o "$2" && chmod +x "$2" && "$2" > "$3"; rc=$?; test "$rc" -eq 0 || exit 1; cmp -s "$3" "$4" || { echo stdout_mismatch; od -An -tx1 "$3"; exit 1; }' \
  bash "$SOUC_BIN" "$OUT_DIR/import_nested_print_42.native" "$OUT_DIR/import_nested_print_42.stdout" "$ROOT_DIR/tests/selfhost/native_runtime/import_nested_print_42.expected"

if emit_summary_json; then
  status="$(python3 - "$SUMMARY_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
  echo "[native-v2-frontend] status=$status summary=$SUMMARY_JSON"
else
  echo "[native-v2-frontend] FAIL: see summary=$SUMMARY_JSON" >&2
  exit 1
fi
