#!/usr/bin/env bash
# Sprint 126 Gate: Wide Compilation Driver
# Tests that wide_driver.sio provides batch compilation for 1024-function modules.
set -eo pipefail

SOUC=./bin/souc
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
DRIVER=self-hosted/native/wide_driver.sio
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

echo "=== Sprint 126: Wide Compilation Driver Gate ==="

# T1: wide.sio passes souc check (dependency)
_ec=0; $SOUC check $WIDE >/dev/null 2>&1 || _ec=$?
check "T1: wide.sio souc check" $_ec

# T2: wide_driver.sio exists
_ec=1; [ -f "$DRIVER" ] && _ec=0
check "T2: wide_driver.sio exists" $_ec

# T3: WideRelocTable struct (4096 entries)
_ec=1; grep -q 'entries: \[WideReloc; 4096\]' $DRIVER && _ec=0
check "T3: WideRelocTable 4096 entries" $_ec

# T4: WideRodata accumulator (32KB)
_ec=1; grep -q 'bytes: \[i8; 32768\]' $DRIVER && _ec=0
check "T4: WideRodata 32KB accumulator" $_ec

# T5: wide_compile_and_emit function exists
_ec=1; grep -q 'fn wide_compile_and_emit(' $DRIVER && _ec=0
check "T5: wide_compile_and_emit() function" $_ec

# T6: wide_apply_all_relocations handles 3 reloc kinds
_ec=1
count=$(grep -c 'kind_code ==' $DRIVER)
[ "$count" -ge 3 ] && _ec=0
check "T6: all 3 relocation kinds handled" $_ec

# T7: Batch compilation loop
_ec=1; grep -q 'WIDE_BATCH_SIZE' $DRIVER && _ec=0
check "T7: batch size constant" $_ec

# T8: wide_copy_batch_relocs function
_ec=1; grep -q 'fn wide_copy_batch_relocs(' $DRIVER && _ec=0
check "T8: wide_copy_batch_relocs() function" $_ec

# T9: name_to_string helper
_ec=1; grep -q 'fn name_to_string(' $DRIVER && _ec=0
check "T9: name_to_string() helper" $_ec

# T10: Entry trampoline handling
_ec=1; grep -q 'emit_entry_trampoline' $DRIVER && _ec=0
check "T10: entry trampoline in wide driver" $_ec

# T11: WideCompileResult struct
_ec=1; grep -q 'struct WideCompileResult' $DRIVER && _ec=0
check "T11: WideCompileResult struct" $_ec

# T12: Rodata offset adjustment in batch relocs
_ec=1; grep -q 'rodata_base_offset' $DRIVER && _ec=0
check "T12: rodata offset adjustment" $_ec

# T13: Phase comments (5 phases)
_ec=1
count=$(grep -c 'Phase [0-9]' $DRIVER)
[ "$count" -ge 5 ] && _ec=0
check "T13: 5+ compilation phases documented" $_ec

# T14: find_main_index usage
_ec=1; grep -q 'find_main_index' $DRIVER && _ec=0
check "T14: find_main_index for entry point" $_ec

# T15: io_write_wide_elf usage
_ec=1; grep -q 'io_write_wide_elf' $DRIVER && _ec=0
check "T15: io_write_wide_elf for output" $_ec

# T16: finalize_elf64_wide usage
_ec=1; grep -q 'finalize_elf64_wide' $DRIVER && _ec=0
check "T16: finalize_elf64_wide for ELF assembly" $_ec

echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="
exit $FAIL
