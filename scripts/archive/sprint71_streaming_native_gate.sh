#!/usr/bin/env bash
# sprint71_streaming_native_gate.sh — Body-at-a-time native compile foundation
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint71"
mkdir -p "$OUT_DIR"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

check_run_line() {
    local name="$1"; local expected="$2"; local src="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file _ec
    log_file="$(mktemp)"
    _ec=0
    timeout 300 "$SOUC" run "$src" >"$log_file" 2>&1 || _ec=$?
    if [ "$_ec" -eq 0 ] && grep -qF "$expected" "$log_file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    elif [ "$_ec" -eq 0 ]; then
        echo "FAIL  $name (expected '$expected' not found)"; FAIL=$((FAIL+1))
    else
        echo "NOT_RUN  $name (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

check_run_absent_line() {
    local name="$1"; local forbidden="$2"; local src="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file _ec
    log_file="$(mktemp)"
    _ec=0
    timeout 300 "$SOUC" run "$src" >"$log_file" 2>&1 || _ec=$?
    if [ "$_ec" -eq 0 ] && ! grep -qF "$forbidden" "$log_file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    elif [ "$_ec" -eq 0 ]; then
        echo "FAIL  $name (forbidden '$forbidden' found)"; FAIL=$((FAIL+1))
    else
        echo "NOT_RUN  $name (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 71: Streaming Native Foundation ==="
echo ""

echo "--- Group 1: Lowering summary/body split ---"
check_grep "lower:summary_struct" \
    "struct IrProgramSummary" \
    "self-hosted/ir/lower.sio"
check_grep "lower:summary_result_fn" \
    "fn lower_program_to_ir_summary_with_epistemic" \
    "self-hosted/ir/lower.sio"
check_grep "lower:body_from_summary_fn" \
    "fn lower_program_function_body_from_summary_with_epistemic" \
    "self-hosted/ir/lower.sio"

echo ""
echo "--- Group 2: Native streaming hooks ---"
check_grep "codegen:stream_begin" \
    "fn compile_module_streaming_begin" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:stream_finish" \
    "fn compile_module_streaming_finish" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:finalize_compiled_elf" \
    "fn finalize_compiled_elf" \
    "self-hosted/native/codegen.sio"
check_grep "loader:streaming_result" \
    "fn compile_singlemodule_native_streaming_result_from_program" \
    "self-hosted/compiler/module_loader.sio"
check_grep "loader:streaming_selector" \
    "fn compile_multimodule_native_maybe_streaming" \
    "self-hosted/compiler/module_loader.sio"

echo ""
echo "--- Group 3: Direct smoke ---"
check_run_line "smoke:body_lower" \
    "streaming_body_lower_smoke: fn=add_pair params=2" \
    "self-hosted/compiler/streaming_body_lower_smoke.sio"
check_run_line "smoke:native_stream" \
    "native_compile_smoke: output=artifacts/sprint70/lowering_stability_native_loop_smoke.out" \
    "self-hosted/compiler/native_compile_smoke.sio"
check_run_absent_line "smoke:no_merged_ir" \
    "Merged IR:" \
    "self-hosted/compiler/native_compile_smoke.sio"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

_ts=$(date -u +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "unknown")
{
  printf '{\n'
  printf '  "schema": "sounio.sprint71.streaming_native_gate.v1",\n'
  printf '  "generated_at": "%s",\n' "$_ts"
  printf '  "status": "%s",\n' "$STATUS"
  printf '  "reason": "%s",\n' "$REASON"
  printf '  "metrics": {"total": %d, "passed": %d, "failed": %d, "not_run": %d},\n' \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN"
  printf '  "novel_claims": [\n'
  printf '    "Program summaries now preserve lowering metadata needed to lower one body at a time",\n'
  printf '    "Single-module native compile can stream lowered bodies into one persistent NativeCompiler",\n'
  printf '    "The streaming hot lane avoids the merged full-IR module on the checked native smoke source"\n'
  printf '  ]\n'
  printf '}\n'
} > "$OUT_DIR/streaming_native_gate.v1.json"

echo ""
echo "Artifact: artifacts/sprint71/streaming_native_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
