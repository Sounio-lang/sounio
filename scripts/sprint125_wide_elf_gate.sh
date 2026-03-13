#!/usr/bin/env bash
# Sprint 125 Gate: Wide ELF Finalization Pipeline
# Tests that wide.sio provides 256KB ELF output, 1024-function support,
# relocation application, and finalize_elf64_wide() function.
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
WIDE=self-hosted/native/wide.sio
RUNNER=self-hosted/native/wide_self_test_runner.sio

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

echo "=== Sprint 125: Wide ELF Finalization Gate ==="

# T1: wide.sio passes souc check
_ec=0; $SOUC check $WIDE >/dev/null 2>&1 || _ec=$?
check "T1: wide.sio souc check" $_ec

# T2: runner passes souc check
_ec=0; $SOUC check $RUNNER >/dev/null 2>&1 || _ec=$?
check "T2: wide_self_test_runner.sio souc check" $_ec

# T3: WideElf64Binary struct exists (262144 bytes)
_ec=1; grep -q 'bytes: \[i8; 262144\]' $WIDE && _ec=0
check "T3: WideElf64Binary 262144-byte buffer" $_ec

# T4: WideRelocationTable struct (1024 entries)
_ec=1; grep -q 'entries: \[Relocation; 1024\]' $WIDE && _ec=0
check "T4: WideRelocationTable 1024 entries" $_ec

# T5: WideSymbolData struct (1024 functions)
_ec=1; grep -q 'name_offsets: \[i64; 1024\]' $WIDE && _ec=0
check "T5: WideSymbolData 1024 functions" $_ec

# T6: finalize_elf64_wide function exists
_ec=1; grep -q 'fn finalize_elf64_wide(' $WIDE && _ec=0
check "T6: finalize_elf64_wide() function" $_ec

# T7: wide_apply_call_relocs function exists
_ec=1; grep -q 'fn wide_apply_call_relocs(' $WIDE && _ec=0
check "T7: wide_apply_call_relocs() function" $_ec

# T8: Wide &! ref helpers exist
_ec=1; grep -q 'fn wide_emit_byte_ref(bin: &! WideElf64Binary' $WIDE && _ec=0
check "T8: wide_emit_byte_ref &! pattern" $_ec

# T9: ELF header emitter
_ec=1; grep -q 'fn wide_emit_elf_header_ref(' $WIDE && _ec=0
check "T9: wide_emit_elf_header_ref() exists" $_ec

# T10: Section header emitter
_ec=1; grep -q 'fn wide_emit_section_header_ref(' $WIDE && _ec=0
check "T10: wide_emit_section_header_ref() exists" $_ec

# T11: io_write_wide_elf function exists
_ec=1; grep -q 'fn io_write_wide_elf(' $WIDE && _ec=0
check "T11: io_write_wide_elf() function" $_ec

# T12: Self-test functions exist (5 tests)
_ec=1
count=$(grep -c 'fn wide_self_test_' $WIDE)
[ "$count" -ge 5 ] && _ec=0
check "T12: 5+ self-test functions" $_ec

# T13: run_wide_self_tests orchestrator
_ec=1; grep -q 'fn run_wide_self_tests(' $WIDE && _ec=0
check "T13: run_wide_self_tests() orchestrator" $_ec

# T14: Patch helpers for ELF header fixup
_ec=1; grep -q 'fn wide_patch_u64_ref(' $WIDE && _ec=0
check "T14: wide_patch_u64_ref() exists" $_ec

# T15: GNU stack program header
_ec=1; grep -q 'fn wide_emit_phdr_gnu_stack_ref(' $WIDE && _ec=0
check "T15: wide_emit_phdr_gnu_stack_ref() exists" $_ec

# T16: .shstrtab emitter
_ec=1; grep -q 'fn wide_emit_shstrtab(' $WIDE && _ec=0
check "T16: wide_emit_shstrtab() exists" $_ec

echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="
exit $FAIL
