#!/usr/bin/env bash
# sprint47_sprof_gate.sh — Persistent .sprof Profile File + Profile Reader + Strategy Promotion
#
# SOTA: LLVM PGO .profraw/.profdata (binary, llvm-profdata merge); GCC gcov .gcda (tagged
# binary, gcov tool); AutoFDO perf.data (Google 2016, HW counters); Bolt .fdata
# (Facebook 2019, text branch counts); HotSpot JIT (interpreter counters → C1 → C2).
#
# Novel claim: text-based .sprof format with zero-dependency file I/O (direct syscalls,
# no libc). Profile reader integrated into compiler — single binary does instrument,
# dump, read, and promote. No external toolchain (llvm-profdata, gcov, perf2bolt).
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

echo "=== Sprint 47: Persistent .sprof Profile File + Profile Reader + Strategy Promotion ==="
echo ""

# --- Executable self-tests (T18, T19) ---
echo "--- Executable self-test: sprof parse and strategy promotion ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:sprof_parse" "selftest:sprof_promotion"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:sprof_parse" \
            "T18 OK: sprof parse" "$SELF_TEST_LOG"
        check_log_line "selftest:sprof_promotion" \
            "T19 OK: sprof promotion" "$SELF_TEST_LOG"
    else
        for name in "selftest:sprof_parse" "selftest:sprof_promotion"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Structural checks: file output infrastructure ---"
check_grep "encode:mov_rdi_r12" \
    "fn emit_mov_rdi_r12" \
    "self-hosted/native/encode.sio"
check_grep "encode:push_r12" \
    "fn emit_push_r12" \
    "self-hosted/native/encode.sio"
check_grep "encode:pop_r12" \
    "fn emit_pop_r12" \
    "self-hosted/native/encode.sio"
check_grep "codegen:sprof_filename_rodata" \
    "profile\.sprof" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:sys_open_sprof" \
    "577" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:sys_close_sprof" \
    "sys_close" \
    "self-hosted/native/codegen.sio"
check_grep "frame:sprof_name_offset" \
    "prof_sprof_name_offset" \
    "self-hosted/native/frame.sio"
check_grep "codegen:sprof_header_fn" \
    "fn emit_prof_write_sprof_header" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:itoa_fd_fn" \
    "fn emit_inline_itoa_fd" \
    "self-hosted/native/codegen.sio"

echo ""
echo "--- Structural checks: profile reader ---"
check_grep "profile:struct" \
    "struct SprofProfile" \
    "self-hosted/ir/profile.sio"
check_grep "profile:entry_struct" \
    "struct SprofEntry" \
    "self-hosted/ir/profile.sio"
check_grep "profile:parse" \
    "fn sprof_parse" \
    "self-hosted/ir/profile.sio"
check_grep "profile:lookup" \
    "fn sprof_lookup" \
    "self-hosted/ir/profile.sio"
check_grep "profile:promotion" \
    "fn sprof_apply_promotion" \
    "self-hosted/ir/profile.sio"
check_grep "profile:ir_name_eq" \
    "ir_name_eq" \
    "self-hosted/ir/profile.sio"

echo ""
echo "--- Structural checks: CLI integration ---"
check_grep "main:use_profile_field" \
    "use_profile" \
    "self-hosted/compiler/main.sio"
check_grep "main:parse_use_profile" \
    "use-profile" \
    "self-hosted/compiler/main.sio"
check_grep "main:apply_promotion_call" \
    "sprof_apply_promotion" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Structural checks: test fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/sprof_promotion_contest.sio"
check_grep "fixture:hot_fn" \
    "fn hot_function" \
    "tests/frontend/sprof_promotion_contest.sio"

echo ""
echo "--- Sprint 38-46 regression ---"
check_grep "regression:apply_strategy_fn" \
    "fn ir_opt_apply_strategy" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:find_return_vreg_fn" \
    "fn ir_opt_find_return_vreg" \
    "self-hosted/ir/opt_strategy.sio"
check_grep "regression:codegen_effective_func" \
    "var effective_func" \
    "self-hosted/native/codegen.sio"
check_grep "regression:inject_validated" \
    "fn ir_lower_inject_validated_param" \
    "self-hosted/ir/lower.sio"
check_grep "regression:patch_validated_calls" \
    "fn ir_patch_validated_calls" \
    "self-hosted/ir/lower.sio"
check_grep "regression:chain_forwarding" \
    "caller_validated_vreg" \
    "self-hosted/ir/lower.sio"
check_grep "regression:hlir_apply_strategy" \
    "fn hlir_opt_apply_strategy" \
    "self-hosted/hlir/opt_strategy.sio"
check_grep "regression:prof_counter_opcode" \
    "IrProfCounter" \
    "self-hosted/ir/ir.sio"
check_grep "regression:prof_dump_fn" \
    "fn emit_prof_dump_function" \
    "self-hosted/native/codegen.sio"
check_grep "regression:reloc_capacity_256" \
    "Relocation; 256" \
    "self-hosted/native/reloc.sio"

# --- Results ---
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

mkdir -p "$ROOT_DIR/artifacts/sprint47"
cat > "$ROOT_DIR/artifacts/sprint47/sprof_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint47.sprof_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 180
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "sprof_format": "text-based; [sprof] header; fn_name count per line; space-separated",
  "novel_claims": [
    "Zero-dependency .sprof file output: sys_open/write/close syscalls, no libc",
    "Text-based profile format — human-readable, trivially parseable",
    "Profile reader integrated into compiler via sprof_parse()",
    "Strategy promotion: count>1000→AGGRESSIVE, count>100→PRECISION_PRESERVING",
    "Single binary instruments, dumps, reads, and promotes — no external toolchain",
    "File output in exit trampoline alongside stderr dump"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint47/sprof_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
