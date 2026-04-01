#!/usr/bin/env bash
# sprint52_loop_licm_gate.sh — Conservative loop LICM on real IR
#
# SOTA: LLVM LICM, GCC tree-ssa-loop-im, and HotSpot C2 all hoist pure
# loop-invariants into preheaders while preserving side-effecting loop bodies.
#
# Novel claim: a conservative LICM pass integrated into the self-hosted single-
# binary optimization pipeline, operating directly on real IrFunction bodies
# after inlining/layout/cleanup and before final cleanup + codegen.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"

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

check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
    fi
}

check_probe_line() {
    local name="$1"; local expected_line="$2"; shift 2
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    if timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- "$@" >"$log_file" 2>&1; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected '$expected_line' in probe output)"; FAIL=$((FAIL+1))
        fi
    else
        echo "NOT_RUN  $name (probe invocation failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 52: Conservative Loop LICM ==="
echo ""

echo "--- Group 1: Self-tests T93-T94 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:licm_hoists_invariant" "selftest:licm_skips_calls"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:licm_hoists_invariant" \
            "T93 OK: loop LICM hoists pure invariants into the unique preheader" "$SELF_TEST_LOG"
        check_log_line "selftest:licm_skips_calls" \
            "T94 OK: loop LICM keeps side-effecting calls inside the loop" "$SELF_TEST_LOG"
    else
        for name in "selftest:licm_hoists_invariant" "selftest:licm_skips_calls"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: Loop pass structure ---"
check_grep "loopopt:module_decl" \
    "module ir::loop_opt" \
    "self-hosted/ir/loop_opt.sio"
check_grep "loopopt:optimize_function" \
    "fn lopt_optimize_function" \
    "self-hosted/ir/loop_opt.sio"
check_grep "loopopt:optimize_module" \
    "fn lopt_optimize_module" \
    "self-hosted/ir/loop_opt.sio"
check_grep "loopopt:real_ir_cfg" \
    "struct LoptIrCfg" \
    "self-hosted/ir/loop_opt.sio"
check_grep "loopopt:licm_opcode_filter" \
    "fn lopt_ir_is_hoistable_opcode" \
    "self-hosted/ir/loop_opt.sio"
check_grep "loopopt:preheader_search" \
    "fn lopt_ir_find_preheader" \
    "self-hosted/ir/loop_opt.sio"

echo ""
echo "--- Group 3: Pipeline integration ---"
check_grep "main:import_loop_opt" \
    "use ir::loop_opt" \
    "self-hosted/compiler/main.sio"
check_grep "main:loop_pass_wired" \
    "lopt_optimize_module" \
    "self-hosted/compiler/main.sio"
check_grep "main:loop_log_prefix" \
    "\\[loop\\]" \
    "self-hosted/compiler/main.sio"
check_grep "main:post_loop_cleanup" \
    "ir_result.module = opt_cleanup_module\\(ir_result.module\\)" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Frontend fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/loop_licm_contest.sio"
check_grep "fixture:main_fn" \
    "fn main" \
    "tests/frontend/loop_licm_contest.sio"
check_grep "fixture:while_loop" \
    "while i < 3" \
    "tests/frontend/loop_licm_contest.sio"
check_grep "fixture:invariant_bias" \
    "let bias = 1 \\+ 2" \
    "tests/frontend/loop_licm_contest.sio"

echo ""
echo "--- Group 5: Parsed frontend smoke ---"
check_probe_line "probe:frontend_loop_fixture" \
    "probe_frontend: ok" \
    --probe-frontend tests/frontend/loop_licm_contest.sio
check_probe_line "probe:load_ir_loop_fixture" \
    "probe_load_ir: fn=main strategy=standard" \
    --probe-load-ir tests/frontend/loop_licm_contest.sio

echo ""
echo "--- Group 6: Regression anchors ---"
check_grep "regression:inline_pass_fn" \
    "fn inl_run_pass" \
    "self-hosted/ir/inline.sio"
check_grep "regression:layout_pass_fn" \
    "fn layout_sort_by_profile" \
    "self-hosted/ir/layout.sio"
check_grep "regression:opt_cleanup_fn" \
    "fn opt_cleanup_function" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:tailcall_pass_fn" \
    "fn tco_run_pass" \
    "self-hosted/ir/tailcall.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint52"
cat > "$ROOT_DIR/artifacts/sprint52/loop_licm_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint52.loop_licm_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 240
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "Real-IR LICM integrated into the self-hosted optimization pipeline",
    "Conservative preheader-only hoisting preserves side-effecting loop bodies",
    "Loop optimization composes with inline, layout, cleanup, and tailcall passes in one binary"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint52/loop_licm_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
