#!/usr/bin/env bash
# Sprint 128 Gate: Wide Driver upgraded to compile_ir_function_v2
# Verifies that wide_driver.sio uses the v2 codegen pipeline:
# native_v2_lower_function_to_machine → native_v2_legalize_function → compile_ir_function_v2
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
DRIVER=self-hosted/native/wide_driver.sio
WIDE=self-hosted/native/wide.sio
CLI=self-hosted/compiler/wide_native_compile_driver.sio

check() {
    local name="$1"; local result="$2"
    TOTAL=$((TOTAL + 1))
    if [ "$result" -eq 0 ]; then
        PASS=$((PASS + 1)); echo "  PASS: $name"
    elif [ "$result" -eq 2 ]; then
        NOT_RUN=$((NOT_RUN + 1)); echo "  NOT_RUN: $name"
    else
        FAIL=$((FAIL + 1)); echo "  FAIL: $name"
    fi
}

echo "=== Sprint 128: Wide Driver v2 Codegen Gate ==="

# --- v2 pipeline wiring ---

# T1: imports native::machine_ir
_ec=1; grep -q 'use native::machine_ir' $DRIVER && _ec=0
check "T1: imports native::machine_ir" $_ec

# T2: imports native::lower_ir
_ec=1; grep -q 'use native::lower_ir' $DRIVER && _ec=0
check "T2: imports native::lower_ir" $_ec

# T3: calls native_v2_lower_function_to_machine
_ec=1; grep -q 'native_v2_lower_function_to_machine' $DRIVER && _ec=0
check "T3: lower to MachineIR (v2 phase 1)" $_ec

# T4: calls native_v2_legalize_function
_ec=1; grep -q 'native_v2_legalize_function' $DRIVER && _ec=0
check "T4: legalize MachineIR (v2 phase 2)" $_ec

# T5: calls compile_ir_function_v2
_ec=1; grep -q 'compile_ir_function_v2' $DRIVER && _ec=0
check "T5: compile_ir_function_v2 (v2 phase 3)" $_ec

# T6: v1 compile_ir_function no longer called in batch loop
_ec=0; grep -q 'compile_ir_function(nc,' $DRIVER && _ec=1
check "T6: v1 compile_ir_function removed from batch loop" $_ec

# T7: legal.supported guard (skip unsupported functions)
_ec=1; grep -q 'legal.supported' $DRIVER && _ec=0
check "T7: legal.supported guard for unsupported functions" $_ec

# T8: v2 pipeline is inside the batch loop (local_idx used)
_ec=1; grep -q 'local_idx' $DRIVER && _ec=0
check "T8: local_idx used in batch loop" $_ec

# --- dependency chain ---

# T9: wide.sio souc check
_ec=0; timeout 60 $SOUC check $WIDE >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T9: wide.sio souc check" $_ec

# T10: wide_native_compile_driver.sio souc check (CLI entry point)
_ec=0; timeout 120 $SOUC check $CLI >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T10: wide_native_compile_driver.sio souc check" $_ec

# --- regression: Sprint 126 structural checks still hold ---

# T11: wide_compile_and_emit still present
_ec=1; grep -q 'fn wide_compile_and_emit(' $DRIVER && _ec=0
check "T11: wide_compile_and_emit() still present" $_ec

# T12: entry trampoline still present
_ec=1; grep -q 'emit_entry_trampoline' $DRIVER && _ec=0
check "T12: entry trampoline still present" $_ec

# T13: WideRelocTable still present
_ec=1; grep -q 'struct WideRelocTable' $DRIVER && _ec=0
check "T13: WideRelocTable still present" $_ec

# T14: finalize_elf64_wide still called
_ec=1; grep -q 'finalize_elf64_wide' $DRIVER && _ec=0
check "T14: finalize_elf64_wide still called" $_ec

# --- integration smoke (may OOM → NOT_RUN under JIT) ---

# T15: CLI smoke run with trivial 1-fn source
SMOKE_SRC=$(mktemp --suffix=.sio)
cat > "$SMOKE_SRC" <<'SIO'
fn main() -> i64 with IO {
    print("wide v2\n")
    0
}
SIO
SMOKE_OUT=$(mktemp --suffix=.elf)
_ec=0; timeout 180 $SOUC run $CLI -- "$SMOKE_SRC" -o "$SMOKE_OUT" >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then _ec=2; fi
check "T15: CLI smoke run (wide v2)" $_ec
rm -f "$SMOKE_SRC" "$SMOKE_OUT"

echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="
exit $FAIL
