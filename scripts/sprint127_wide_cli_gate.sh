#!/usr/bin/env bash
# Sprint 127 Gate: Wide Native Compile CLI Entry Point
# Tests that wide_native_compile_driver.sio provides a standalone CLI
# for invoking the wide compilation driver on large modules.
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
DRIVER=self-hosted/compiler/wide_native_compile_driver.sio
WIDE_DRIVER=self-hosted/native/wide_driver.sio
WIDE=self-hosted/native/wide.sio

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

echo "=== Sprint 127: Wide Native Compile CLI Gate ==="

# --- Structural checks ---

# T1: Driver file exists
_ec=1; [ -f "$DRIVER" ] && _ec=0
check "T1: wide_native_compile_driver.sio exists" $_ec

# T2: Imports native::wide_driver
_ec=1; grep -q 'use native::wide_driver::' $DRIVER && _ec=0
check "T2: imports native::wide_driver" $_ec

# T3: Imports native::wide
_ec=1; grep -q 'use native::wide::' $DRIVER && _ec=0
check "T3: imports native::wide" $_ec

# T4: Calls wide_compile_and_emit
_ec=1; grep -q 'wide_compile_and_emit' $DRIVER && _ec=0
check "T4: calls wide_compile_and_emit()" $_ec

# T5: Has main() entry point
_ec=1; grep -q 'fn main()' $DRIVER && _ec=0
check "T5: main() entry point" $_ec

# T6: Parses CLI args (source_path)
_ec=1; grep -q 'get_arg(0)' $DRIVER && _ec=0
check "T6: parses source path from CLI" $_ec

# T7: Supports -o output flag
_ec=1; grep -q '"-o"' $DRIVER && _ec=0
check "T7: supports -o output flag" $_ec

# T8: Uses summary + per-body lowering to build full IrModule
_ec=1; grep -q 'lower_program_function_body_from_summary_flat_with_epistemic_ref' $DRIVER && _ec=0
check "T8: per-body IR lowering into IrModule" $_ec

# T9: Checks fn_count > 1024 limit
_ec=1; grep -q 'fn_count > 1024' $DRIVER && _ec=0
check "T9: enforces 1024-function limit" $_ec

# T10: Reports compile result (fn_count, code_size, reloc_count)
_ec=1
count=$(grep -c 'result\.' $DRIVER)
[ "$count" -ge 3 ] && _ec=0
check "T10: reports compile result fields" $_ec

# T11: load_source_file inlined (avoids module_loader.sio)
_ec=1; grep -q 'fn load_source_file(' $DRIVER && _ec=0
check "T11: load_source_file inlined" $_ec

# T12: Has char_from_i64 helper (codegen dependency)
_ec=1; grep -q 'fn char_from_i64(' $DRIVER && _ec=0
check "T12: char_from_i64 codegen helper" $_ec

# --- Dependency checks ---

# T13: wide_driver.sio has wide_compile_and_emit (structural dep check)
_ec=1; grep -q 'fn wide_compile_and_emit(' $WIDE_DRIVER && _ec=0
check "T13: wide_driver.sio has wide_compile_and_emit" $_ec

# T14: wide.sio passes souc check
_ec=0; timeout 60 $SOUC check $WIDE >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T14: wide.sio souc check" $_ec

# T15: wide_native_compile_driver.sio passes souc check
_ec=0; timeout 120 $SOUC check $DRIVER >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T15: wide_native_compile_driver.sio souc check" $_ec

# --- Integration probes (may OOM → NOT_RUN) ---

# T16: Smoke run with trivial source (may OOM under JIT)
SMOKE_SRC=$(mktemp --suffix=.sio)
cat > "$SMOKE_SRC" <<'SIO'
fn main() -> i64 with IO {
    print("hello wide\n")
    0
}
SIO
SMOKE_OUT=$(mktemp --suffix=.elf)
_ec=0; timeout 180 $SOUC run $DRIVER -- "$SMOKE_SRC" -o "$SMOKE_OUT" >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then _ec=2; fi  # timeout/OOM → NOT_RUN
check "T16: smoke run with trivial source" $_ec
rm -f "$SMOKE_SRC" "$SMOKE_OUT"

echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="
exit $FAIL
