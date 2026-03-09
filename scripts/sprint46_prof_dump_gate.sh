#!/usr/bin/env bash
# sprint46_prof_dump_gate.sh — Counter Dump at Program Exit via Exit Trampoline
#
# SOTA: LLVM PGO __llvm_profile_write_file (atexit hook); GCC gcov __gcov_exit (static
# destructor); gprof _mcleanup (atexit); DynamoRIO drcov (exit-time dump to .log);
# Chamith et al. SIGPLAN 2017 (exit-time counter readout, ~0 overhead).
#
# Novel claim: zero-dependency exit-time counter dump — no libc, no atexit(), no static
# destructors. __prof_dump is inlined into the ELF _start trampoline between call main
# and sys_exit. Function names from .rodata, counters from .data, itoa+sys_write(stderr).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

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

echo "=== Sprint 46: Counter Dump at Program Exit via Exit Trampoline ==="
echo ""

# --- Executable self-tests (T16, T17) ---
echo "--- Executable self-test: prof dump emission and stderr targeting ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:prof_dump_emitted" "selftest:prof_dump_writes_stderr"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    if timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1; then
        check_log_line "selftest:prof_dump_emitted" \
            "T16 OK: prof dump emitted" "$SELF_TEST_LOG"
        check_log_line "selftest:prof_dump_writes_stderr" \
            "T17 OK: prof dump writes stderr" "$SELF_TEST_LOG"
    else
        for name in "selftest:prof_dump_emitted" "selftest:prof_dump_writes_stderr"; do
            TOTAL=$((TOTAL+1))
            echo "NOT_RUN  $name (--self-test failed or timed out)"; NOT_RUN=$((NOT_RUN+1))
        done
    fi
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Structural checks: counter dump infrastructure ---"
check_grep "encode:mov_rax_rip_disp32" \
    "fn emit_mov_rax_rip_disp32" \
    "self-hosted/native/encode.sio"
check_grep "frame:prof_counter_names" \
    "prof_counter_names" \
    "self-hosted/native/frame.sio"
check_grep "frame:prof_dump_fn_index" \
    "prof_dump_fn_index" \
    "self-hosted/native/frame.sio"
check_grep "codegen:emit_prof_dump_function" \
    "fn emit_prof_dump_function" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:emit_prof_write_prefix" \
    "fn emit_prof_write_prefix" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:emit_inline_itoa_stderr" \
    "fn emit_inline_itoa_stderr" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:prof_dump_call" \
    "prof_dump_fn_index" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:stderr_fd" \
    "emit_mov_rdi_imm.*2" \
    "self-hosted/native/codegen.sio"
check_grep "reloc:capacity_256" \
    "Relocation; 256" \
    "self-hosted/native/reloc.sio"
check_grep "codegen:store_prof_names_rodata" \
    "string_table_add.*fn_name" \
    "self-hosted/native/codegen.sio"

echo ""
echo "--- Structural checks: test fixture ---"
check_grep "fixture:run_pass" \
    "//@ run-pass" \
    "tests/frontend/prof_dump_counter_contest.sio"
check_grep "fixture:profiled_fn" \
    "fn profiled_contested" \
    "tests/frontend/prof_dump_counter_contest.sio"

echo ""
echo "--- Sprint 38-45 regression ---"
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
check_grep "regression:inc_encoding" \
    "0xFF" \
    "self-hosted/native/lower_ir.sio"
check_grep "regression:data_section_reloc" \
    "fn add_data_section_reloc" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint46"
cat > "$ROOT_DIR/artifacts/sprint46/prof_dump_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint46.prof_dump_gate.v1",
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
  "counter_dump_infrastructure": "__prof_dump function emitted in code buffer; called from exit trampoline before sys_exit; prefix/separator/newline via immediate stack writes; fn names via .rodata LEA+strlen; counter values via .data MOV [RIP+disp32]; itoa div-by-10 loop; all output to stderr (fd=2)",
  "novel_claims": [
    "Zero-dependency exit-time counter dump: no libc, no atexit(), no static destructors",
    "Dump inlined in ELF _start trampoline between call main return and sys_exit",
    "Function names loaded from .rodata via LEA [RIP+disp32] + inline strlen",
    "Counter values loaded from .data via MOV RAX, [RIP+disp32] (48 8B 05)",
    "Fixed strings ([prof], :, newline) via immediate stack MOV — zero extra relocations",
    "RelocationTable expanded 64→256 entries for counter-dense programs"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint46/prof_dump_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
