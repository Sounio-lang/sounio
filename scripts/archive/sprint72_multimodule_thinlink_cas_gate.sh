#!/usr/bin/env bash
# sprint72_multimodule_thinlink_cas_gate.sh — Cross-module streaming native with thin-link cache
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint72"
CACHE_DIR="$OUT_DIR/cache"
mkdir -p "$OUT_DIR" "$CACHE_DIR"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qE -- "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

native_compile_log() {
    local log_file="$1"; shift
    local out_file="$1"; shift
    local src="$1"; shift
    local _ec=0
    timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --native-compile "$src" -o "$out_file" "$@" >"$log_file" 2>&1 || _ec=$?
    return "$_ec"
}

check_native_compile_and_run() {
    local name="$1"; local src="$2"; local out_file="$3"; local expected_exit="$4"; shift 4
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file _ec run_ec
    log_file="$(mktemp)"
    _ec=0
    native_compile_log "$log_file" "$out_file" "$src" "$@" || _ec=$?
    if [ "$_ec" -ne 0 ]; then
        echo "FAIL  $name (compile exit $_ec)"; FAIL=$((FAIL+1))
        sed -n '1,80p' "$log_file" || true
        rm -f "$log_file"
        return
    fi
    if grep -qF "Merged IR:" "$log_file" 2>/dev/null; then
        echo "FAIL  $name (forbidden 'Merged IR:' found)"; FAIL=$((FAIL+1))
        rm -f "$log_file"
        return
    fi
    if [ ! -x "$out_file" ]; then
        chmod +x "$out_file" 2>/dev/null || true
    fi
    run_ec=0
    "$out_file" >/dev/null 2>&1 || run_ec=$?
    if [ "$run_ec" -eq "$expected_exit" ]; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected exit $expected_exit, got $run_ec)"; FAIL=$((FAIL+1))
    fi
    rm -f "$log_file"
}

check_log_pattern() {
    local name="$1"; local expected="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if grep -qE "$expected" "$log_file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$expected' not found)"; FAIL=$((FAIL+1))
    fi
}

check_trace_pattern() {
    local name="$1"; local expected="$2"
    TOTAL=$((TOTAL+1))
    local trace_file="artifacts/omega/native_decision_trace.v1.json"
    if [ ! -f "$trace_file" ]; then
        echo "NOT_RUN  $name (trace not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qE "$expected" "$trace_file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$expected' not found in trace)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 72: Multimodule Thin-Link CAS ==="
echo ""

echo "--- Group 1: Structural cutover ---"
check_grep "loader:variant_key" \
    "struct ModuleVariantKey" \
    "self-hosted/compiler/module_loader.sio"
check_grep "loader:summary_type" \
    "struct ModuleSummary" \
    "self-hosted/compiler/module_loader.sio"
check_grep "loader:combined_index" \
    "struct CombinedSummaryIndex" \
    "self-hosted/compiler/module_loader.sio"
check_grep "loader:advanced_entry" \
    "fn compile_multimodule_native_advanced" \
    "self-hosted/compiler/module_loader.sio"
check_grep "main:cache_dir_flag" \
    "--cache-dir <dir>" \
    "self-hosted/compiler/main.sio"
check_grep "main:cache_mode_flag" \
    "--cache-mode <mode>" \
    "self-hosted/compiler/main.sio"
check_grep "main:cache_stats_flag" \
    "--cache-stats" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 2: Imported native correctness ---"
check_native_compile_and_run \
    "native:single_hop_o0" \
    "tests/multimodule/thin_single_main.sio" \
    "$OUT_DIR/thin_single_main.elf" \
    7
check_native_compile_and_run \
    "native:transitive_o0" \
    "tests/multimodule/thin_chain_main.sio" \
    "$OUT_DIR/thin_chain_main.elf" \
    7
check_native_compile_and_run \
    "native:contest_o0" \
    "tests/multimodule/thin_contest_main.sio" \
    "$OUT_DIR/thin_contest_main.elf" \
    0

echo ""
echo "--- Group 3: Optimized thin-link import planning ---"
INLINE_LOG="$OUT_DIR/thin_inline_opt.log"
INLINE_OUT="$OUT_DIR/thin_inline_opt.elf"
rm -f "$INLINE_LOG" "$INLINE_OUT"
if [ -x "$SOUC" ]; then
    _ec=0
    native_compile_log "$INLINE_LOG" "$INLINE_OUT" "tests/multimodule/thin_inline_main.sio" -O --cache-dir "$CACHE_DIR" --cache-mode rw --cache-stats || _ec=$?
    TOTAL=$((TOTAL+1))
    if [ "$_ec" -eq 0 ]; then
        chmod +x "$INLINE_OUT" 2>/dev/null || true
        run_ec=0
        "$INLINE_OUT" >/dev/null 2>&1 || run_ec=$?
        if [ "$run_ec" -eq 5 ]; then
            echo "PASS  native:inline_opt_run"; PASS=$((PASS+1))
        else
            echo "FAIL  native:inline_opt_run (expected exit 5, got $run_ec)"; FAIL=$((FAIL+1))
        fi
    else
        echo "FAIL  native:inline_opt_run (compile exit $_ec)"; FAIL=$((FAIL+1))
    fi
else
    TOTAL=$((TOTAL+1))
    echo "NOT_RUN  native:inline_opt_run (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
fi
check_log_pattern "native:inline_opt_log" "\\[inline\\] pass complete" "$INLINE_LOG"
check_log_pattern "native:inline_opt_cache_log" "\\[cache\\] summary h/m=" "$INLINE_LOG"
check_trace_pattern "trace:pipeline" "\"pipeline\":\"thinlink-cas\""
check_trace_pattern "trace:module_count" "\"module_count\":[0-9]+"
check_trace_pattern "trace:imported_body_count" "\"imported_body_count\":[1-9][0-9]*"
check_trace_pattern "trace:cache_section" "\"cache\":\\{"

echo ""
echo "--- Group 4: Warm cache reuse ---"
WARM_LOG1="$OUT_DIR/warm1.log"
WARM_LOG2="$OUT_DIR/warm2.log"
WARM_OUT="$OUT_DIR/warm_chain.elf"
rm -f "$WARM_LOG1" "$WARM_LOG2" "$WARM_OUT"
if [ -x "$SOUC" ]; then
    _ec=0
    native_compile_log "$WARM_LOG1" "$WARM_OUT" "tests/multimodule/thin_chain_main.sio" -O --cache-dir "$CACHE_DIR" --cache-mode rw --cache-stats || _ec=$?
    native_compile_log "$WARM_LOG2" "$WARM_OUT" "tests/multimodule/thin_chain_main.sio" -O --cache-dir "$CACHE_DIR" --cache-mode rw --cache-stats || _ec=$?
    TOTAL=$((TOTAL+1))
    if [ "$_ec" -eq 0 ] && grep -qE "summary h/m=[1-9][0-9]*/0" "$WARM_LOG2" && \
       grep -qE "body h/m=[1-9][0-9]*/0" "$WARM_LOG2" && \
       grep -qE "object h/m=[1-9][0-9]*/0" "$WARM_LOG2" && \
       grep -qE "link h/m=[1-9][0-9]*/0" "$WARM_LOG2"; then
        echo "PASS  cache:warm_hits"; PASS=$((PASS+1))
    elif [ "$_ec" -eq 0 ]; then
        echo "FAIL  cache:warm_hits (expected nonzero warm hits)"; FAIL=$((FAIL+1))
    else
        echo "FAIL  cache:warm_hits (compile exit $_ec)"; FAIL=$((FAIL+1))
    fi
else
    TOTAL=$((TOTAL+1))
    echo "NOT_RUN  cache:warm_hits (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "--- Group 5: Leaf edit invalidation ---"
LEAF_TMP="$(mktemp -d)"
cp tests/multimodule/thin_chain_a.sio "$LEAF_TMP/"
cp tests/multimodule/thin_chain_b.sio "$LEAF_TMP/"
cp tests/multimodule/thin_chain_main.sio "$LEAF_TMP/"
LEAF_CACHE="$OUT_DIR/leaf-cache"
mkdir -p "$LEAF_CACHE"
LEAF_LOG1="$OUT_DIR/leaf1.log"
LEAF_LOG2="$OUT_DIR/leaf2.log"
LEAF_OUT="$OUT_DIR/leaf_chain.elf"
if [ -x "$SOUC" ]; then
    _ec=0
    native_compile_log "$LEAF_LOG1" "$LEAF_OUT" "$LEAF_TMP/thin_chain_main.sio" -O --cache-dir "$LEAF_CACHE" --cache-mode rw --cache-stats || _ec=$?
    perl -0pi -e 's/x \\+ 1/x + 2/' "$LEAF_TMP/thin_chain_a.sio"
    native_compile_log "$LEAF_LOG2" "$LEAF_OUT" "$LEAF_TMP/thin_chain_main.sio" -O --cache-dir "$LEAF_CACHE" --cache-mode rw --cache-stats || _ec=$?
    TOTAL=$((TOTAL+1))
    if [ "$_ec" -eq 0 ] && grep -qE "summary h/m=[1-9][0-9]*/[1-9][0-9]*" "$LEAF_LOG2" && \
       grep -qE "object h/m=[1-9][0-9]*/[1-9][0-9]*" "$LEAF_LOG2" && \
       grep -qE "link h/m=0/[1-9][0-9]*" "$LEAF_LOG2"; then
        chmod +x "$LEAF_OUT" 2>/dev/null || true
        run_ec=0
        "$LEAF_OUT" >/dev/null 2>&1 || run_ec=$?
        if [ "$run_ec" -eq 8 ]; then
            echo "PASS  cache:leaf_invalidation"; PASS=$((PASS+1))
        else
            echo "FAIL  cache:leaf_invalidation (expected exit 8, got $run_ec)"; FAIL=$((FAIL+1))
        fi
    elif [ "$_ec" -eq 0 ]; then
        echo "FAIL  cache:leaf_invalidation (expected mixed hit/miss profile)"; FAIL=$((FAIL+1))
    else
        echo "FAIL  cache:leaf_invalidation (compile exit $_ec)"; FAIL=$((FAIL+1))
    fi
else
    TOTAL=$((TOTAL+1))
    echo "NOT_RUN  cache:leaf_invalidation (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
fi
rm -rf "$LEAF_TMP"

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
  printf '  "schema": "sounio.sprint72.multimodule_thinlink_cas_gate.v1",\n'
  printf '  "generated_at": "%s",\n' "$_ts"
  printf '  "status": "%s",\n' "$STATUS"
  printf '  "reason": "%s",\n' "$REASON"
  printf '  "metrics": {"total": %d, "passed": %d, "failed": %d, "not_run": %d},\n' \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN"
  printf '  "novel_claims": [\n'
  printf '    "Imported x86_64-linux native compile hard-cuts to a summary-first thin-link pipeline",\n'
  printf '    "Per-module summary/body/object artifacts are reusable across warm multimodule builds",\n'
  printf '    "Cross-module imported calls resolve without merged whole-program IR on the native lane",\n'
  printf '    "Leaf edits preserve cache hits for unaffected modules while forcing a relink"\n'
  printf '  ]\n'
  printf '}\n'
} > "$OUT_DIR/multimodule_thinlink_cas_gate.v1.json"

echo ""
echo "Artifact: artifacts/sprint72/multimodule_thinlink_cas_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
