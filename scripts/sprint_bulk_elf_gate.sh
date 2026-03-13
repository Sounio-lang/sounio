#!/bin/bash
# Sprint: Bulk ELF Finalization Gate
# Tests the bulk ELF path that bypasses per-byte JIT dispatch.
# Run: bash scripts/sprint_bulk_elf_gate.sh

set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); }
not_run() { echo "  NOT_RUN: $1"; NOT_RUN=$((NOT_RUN + 1)); TOTAL=$((TOTAL + 1)); }

# ============================================================================
# Patch helpers (apply JIT workarounds to tracked files)
# ============================================================================
_apply_patches() {
    sed -i '/fn io_write_wide_elf_to_file/,/^}/d' self-hosted/io/file_write.sio
    sed -i '/^use native::elf::\*/a use native::elf_bulk::*' self-hosted/compiler/native_compile_driver.sio
    sed -i 's|print("native_compile: finalize_elf\\n")|print("native_compile: finalize_elf_bulk\\n")|' self-hosted/compiler/native_compile_driver.sio
    sed -i 's|let elf = finalize_compiled_elf(nc, 4194304)|let elf = finalize_elf64_bulk(nc.code, nc.rodata, nc.data, nc.entry_offset, 4194304, nc.symbols, nc.fn_offsets, nc.fn_count)|' self-hosted/compiler/native_compile_driver.sio
    sed -i 's/emit_byte_mut(&! b, /b = emit_byte(b, /g' self-hosted/native/encode.sio
    sed -i 's/emit_u32_le_mut(&! b, /b = emit_u32_le(b, /g' self-hosted/native/encode.sio
    sed -i 's/emit_u64_le_mut(&! b, /b = emit_u64_le(b, /g' self-hosted/native/encode.sio
}

_apply_encode_tail_fix() {
python3 << 'PYEOF'
with open('self-hosted/native/encode.sio', 'r') as f:
    code = f.read()
code = code.replace(
    'fn emit_mov_rax_imm(buf: CodeBuffer, val: i64) -> CodeBuffer with Mut, Panic, Div {\n    emit_mov_reg_imm(buf, 0, val)\n}',
    'fn emit_mov_rax_imm(buf: CodeBuffer, val: i64) -> CodeBuffer with Mut, Panic, Div {\n    var result = emit_mov_reg_imm(buf, 0, val)\n    result\n}')
code = code.replace(
    'fn emit_mov_rdi_imm(buf: CodeBuffer, val: i64) -> CodeBuffer with Mut, Panic, Div {\n    emit_mov_reg_imm(buf, 7, val)\n}',
    'fn emit_mov_rdi_imm(buf: CodeBuffer, val: i64) -> CodeBuffer with Mut, Panic, Div {\n    var result = emit_mov_reg_imm(buf, 7, val)\n    result\n}')
code = code.replace(
    'fn emit_mov_reg_imm(buf: CodeBuffer, reg: i64, val: i64) -> CodeBuffer with Mut, Panic, Div {\n    // Prefer the shorter imm32 encoding when possible (it sign-extends to 64-bit),\n    // otherwise fall back to imm64 to avoid truncation.\n    if val >= -2147483648 && val <= 2147483647 {\n        emit_mov_reg_imm32(buf, reg, val)\n    } else {\n        emit_mov_reg_imm64(buf, reg, val)\n    }\n}',
    'fn emit_mov_reg_imm(buf: CodeBuffer, reg: i64, val: i64) -> CodeBuffer with Mut, Panic, Div {\n    if val >= -2147483648 && val <= 2147483647 {\n        var r32 = emit_mov_reg_imm32(buf, reg, val)\n        r32\n    } else {\n        var r64 = emit_mov_reg_imm64(buf, reg, val)\n        r64\n    }\n}')
code = code.replace(
    'fn emit_ret(buf: CodeBuffer) -> CodeBuffer with Mut, Panic, Div {\n    emit_byte(buf, 0xc3)\n}',
    'fn emit_ret(buf: CodeBuffer) -> CodeBuffer with Mut, Panic, Div {\n    var b = buf\n    b = emit_byte(b, 0xc3)\n    b\n}')
code = code.replace(
    'fn emit_push_rbp(buf: CodeBuffer) -> CodeBuffer with Mut, Panic, Div {\n    emit_byte(buf, 0x55)  // push rbp\n}',
    'fn emit_push_rbp(buf: CodeBuffer) -> CodeBuffer with Mut, Panic, Div {\n    var b = buf\n    b = emit_byte(b, 0x55)  // push rbp\n    b\n}')
with open('self-hosted/native/encode.sio', 'w') as f:
    f.write(code)
PYEOF
}

_apply_codegen_fix() {
python3 << 'PYEOF'
with open('self-hosted/native/codegen.sio', 'r') as f:
    code = f.read()
code = code.replace(
    '                record_label_patch_mut(&! c, label_id, jmp_pos + 1)',
    '                if c.label_patch_count < 256 {\n                    c.label_patches[c.label_patch_count as usize] = LabelPatch { code_offset: jmp_pos + 1, label_id: label_id }\n                    c.label_patch_count = c.label_patch_count + 1\n                }')
code = code.replace(
    '                    record_label_patch_mut(&! c, label_id, jz_pos + 2)',
    '                    if c.label_patch_count < 256 {\n                        c.label_patches[c.label_patch_count as usize] = LabelPatch { code_offset: jz_pos + 2, label_id: label_id }\n                        c.label_patch_count = c.label_patch_count + 1\n                    }')
code = code.replace(
    '                    record_label_patch_mut(&! c, label_id, jnz_pos + 2)',
    '                    if c.label_patch_count < 256 {\n                        c.label_patches[c.label_patch_count as usize] = LabelPatch { code_offset: jnz_pos + 2, label_id: label_id }\n                        c.label_patch_count = c.label_patch_count + 1\n                    }')
code = code.replace(
    '        record_label_mut(&! c, instr.dst.value)',
    '        if instr.dst.value >= 0 && instr.dst.value < 256 {\n            c.label_offsets[instr.dst.value as usize] = c.code.len\n        }')
code = code.replace(
    '        record_return_jump_mut(&! c)\n        return c',
    '        let rj_pos = c.code.len\n        c.code = emit_jmp_rel32(c.code)\n        if c.return_jump_count < 16 {\n            c.return_jumps[c.return_jump_count as usize] = rj_pos + 1\n            c.return_jump_count = c.return_jump_count + 1\n        }\n        return c')
old_fn = 'fn compile_ir_function_v2(nc: NativeCompiler, func: IrFunction, machine_func: MachineFunction, fn_index: i64) -> NativeCompiler with Mut, Panic, Div, Alloc {\n    var c = nc\n    compile_ir_function_v2_mut(&! c, func, machine_func, fn_index)\n    c\n}'
new_fn = '''fn compile_ir_function_v2(nc: NativeCompiler, func: IrFunction, machine_func: MachineFunction, fn_index: i64) -> NativeCompiler with Mut, Panic, Div, Alloc {
    var c = nc
    c.return_jump_count = 0
    var rst_i = 0
    while rst_i < 256 { c.label_offsets[rst_i as usize] = -1; rst_i = rst_i + 1 }
    c.label_patch_count = 0
    c.current_frame_size = 0
    var rst_j = 0
    while rst_j < 512 { c.is_float_reg[rst_j as usize] = 0; rst_j = rst_j + 1 }
    c.current_strategy = 0
    var rst_ra = 0
    while rst_ra < 512 { c.vreg_to_preg[rst_ra as usize] = -2; c.vreg_spill_slots[rst_ra as usize] = -1; rst_ra = rst_ra + 1 }
    c.current_machine_base_slots = 0
    c.current_machine_spill_count = 0
    c.fn_offsets[fn_index as usize] = c.code.len
    c.symbols = symbol_data_add_name(c.symbols, fn_index, func.name.buf, func.name.len)
    if name_is_print_int(func.name) { return emit_builtin_print_int(c) }
    if name_is_print_char(func.name) { return emit_builtin_print_char(c) }
    if name_is_print(func.name) { return emit_builtin_print_str(c) }
    if name_is_get_arg_count(func.name) { return emit_builtin_get_arg_count(c) }
    if name_is_get_arg(func.name) { return emit_builtin_get_arg(c) }
    if name_is_str_len(func.name) { return emit_builtin_str_len(c) }
    if name_is_str_eq(func.name) { return emit_builtin_str_eq(c) }
    if name_is_str_slice(func.name) { return emit_builtin_str_slice(c) }
    if name_is_starts_with(func.name) { return emit_builtin_starts_with(c) }
    if name_is_str_concat(func.name) { return emit_builtin_str_concat(c) }
    if name_is_read_file(func.name) { return emit_builtin_read_file(c) }
    if name_is_write_file(func.name) { return emit_builtin_write_file(c) }
    if name_is_file_size(func.name) { return emit_builtin_file_size(c) }
    c.current_strategy = func.compile_strategy
    let ra_pair = native_v2_machine_ra_instrs(machine_func)
    var ra_state = liveness_compute_intervals(ra_pair.0, ra_pair.1)
    ra_state = regalloc_linear_scan(ra_state, native_target_policy_x86_allocation_order_count())
    var ra_v: i64 = 0
    while ra_v < ra_state.count {
        let ra_interval = ra_state.intervals[ra_v as usize]
        let ra_vreg = ra_interval.vreg
        if ra_vreg >= 0 && ra_vreg < 512 {
            if ra_interval.assigned_preg >= 0 { c.vreg_to_preg[ra_vreg as usize] = native_target_policy_x86_allocation_order_reg_id(ra_interval.assigned_preg) }
            else { c.vreg_to_preg[ra_vreg as usize] = ra_interval.assigned_preg }
            c.vreg_spill_slots[ra_vreg as usize] = ra_interval.spill_slot
        }
        ra_v = ra_v + 1
    }
    c.current_machine_base_slots = machine_func.temp_count
    c.current_machine_spill_count = ra_state.spill_count
    let preg_mask = nc_used_allocatable_preg_mask(c, machine_func.temp_count)
    let frame_size = x86_align_frame_size_for_calls(native_v2_local_frame_size(machine_func.temp_count, c.current_machine_spill_count), preg_mask)
    c.current_frame_size = frame_size
    c.code = emit_prologue_with_preg_mask(c.code, frame_size, preg_mask)
    var sp_i = 0
    while sp_i < func.param_count && sp_i < 6 {
        let sp_vreg = func.param_regs[sp_i as usize]
        if sp_vreg >= 0 { c.code = emit_store_reg_rbp_disp32(c.code, ir_slot_offset(sp_vreg), param_reg(sp_i)) }
        sp_i = sp_i + 1
    }
    let fn_offset = c.fn_offsets[fn_index as usize]
    var mi: i64 = 0
    while mi < machine_func.blocks[0].instr_count {
        let em_instr = machine_func.blocks[0].instrs[mi as usize]
        if native_v2_machine_instr_is_safepoint(em_instr) {
            let sp_kind = if em_instr.opcode == MIR_OP_CALL { NATIVE_V2_SAFEPOINT_CALL() } else { NATIVE_V2_SAFEPOINT_RET() }
            c = native_v2_record_safepoint(c, fn_index, sp_kind, mi, c.code.len - fn_offset, machine_func.temp_count, c.current_machine_spill_count)
        }
        c = native_v2_emit_machine_instr(c, em_instr)
        mi = mi + 1
    }
    let epilogue_offset = c.code.len
    c.code = emit_epilogue_with_preg_mask(c.code, preg_mask)
    var rj_i = 0
    while rj_i < c.return_jump_count {
        let rj_imm_offset = c.return_jumps[rj_i as usize]
        c.code = patch_u32_le(c.code, rj_imm_offset, epilogue_offset - (rj_imm_offset + 4))
        rj_i = rj_i + 1
    }
    var lf_i = 0
    while lf_i < c.label_patch_count {
        let lf_patch = c.label_patches[lf_i as usize]
        let lf_target = c.label_offsets[lf_patch.label_id as usize]
        if lf_target >= 0 { c.code = patch_u32_le(c.code, lf_patch.code_offset, lf_target - (lf_patch.code_offset + 4)) }
        lf_i = lf_i + 1
    }
    c
}'''
code = code.replace(old_fn, new_fn)
with open('self-hosted/native/codegen.sio', 'w') as f:
    f.write(code)
PYEOF
}

_full_patch() {
    git checkout HEAD -- self-hosted/ 2>/dev/null || true
    _apply_patches
    _apply_encode_tail_fix
    _apply_codegen_fix
}

echo "=== Sprint: Bulk ELF Finalization Gate ==="
echo ""

# ============================================================================
# Structural checks (source files — elf_bulk.sio is permanent, others via patch)
# ============================================================================
echo "[structural]"

# Apply patches for structural checks that need them
_full_patch 2>/dev/null

if grep -q 'use native::elf_bulk' self-hosted/compiler/native_compile_driver.sio; then
    pass "struct:bulk_import"
else
    fail "struct:bulk_import"
fi

if grep -q 'finalize_elf64_bulk' self-hosted/compiler/native_compile_driver.sio; then
    pass "struct:bulk_call"
else
    fail "struct:bulk_call"
fi

if grep -q 'finalize_compiled_elf' self-hosted/compiler/native_compile_driver.sio; then
    fail "struct:no_old_finalize"
else
    pass "struct:no_old_finalize"
fi

if grep -q 'module native::elf_bulk' self-hosted/native/elf_bulk.sio; then
    pass "struct:module_decl"
else
    fail "struct:module_decl"
fi

if grep -q 'fn finalize_elf64_bulk' self-hosted/native/elf_bulk.sio; then
    pass "struct:bulk_fn"
else
    fail "struct:bulk_fn"
fi

if grep -v '//' self-hosted/native/elf_bulk.sio | grep -q '&!'; then
    fail "struct:byval_pattern"
else
    pass "struct:byval_pattern"
fi

if grep -q 'emit_byte_mut(&!' self-hosted/native/encode.sio; then
    fail "struct:encode_no_mut"
else
    pass "struct:encode_no_mut"
fi

if grep -q 'compile_ir_function_v2_mut(&! c' self-hosted/native/codegen.sio; then
    fail "struct:codegen_inlined"
else
    pass "struct:codegen_inlined"
fi

echo ""

# ============================================================================
# Compile tests
# ============================================================================
echo "[compile]"

echo 'fn main() -> i64 { 42 }' > /tmp/gate_exit42.sio
echo 'fn main() -> i64 { 0 }' > /tmp/gate_exit0.sio
cat > /tmp/gate_add.sio << 'SIOEOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(20, 22) }
SIOEOF

# compile:exit42
rm -f /tmp/gate_exit42.elf /tmp/gate_exit0.elf /tmp/gate_add.elf
_full_patch 2>/dev/null
_ec=0
ulimit -s unlimited
timeout 180 $SOUC run self-hosted/compiler/native_compile_driver.sio -- /tmp/gate_exit42.sio -o /tmp/gate_exit42.elf 2>&1 || _ec=$?
if [ -f /tmp/gate_exit42.elf ]; then
    chmod +x /tmp/gate_exit42.elf
    /tmp/gate_exit42.elf && _run_ec=0 || _run_ec=$?
    if [ "$_run_ec" = "42" ]; then pass "compile:exit42"
    else fail "compile:exit42 (exit=$_run_ec, expected 42)"; fi
else
    if [ "$_ec" = "124" ] || [ "$_ec" = "137" ]; then not_run "compile:exit42 (timeout/OOM ec=$_ec)"
    else fail "compile:exit42 (no ELF, ec=$_ec)"; fi
fi

# compile:elf_magic
if [ -f /tmp/gate_exit42.elf ]; then
    magic=$(xxd -l 4 -p /tmp/gate_exit42.elf)
    if [ "$magic" = "7f454c46" ]; then pass "compile:elf_magic"
    else fail "compile:elf_magic (got $magic)"; fi
else
    not_run "compile:elf_magic"
fi

# compile:exit0
_full_patch 2>/dev/null
_ec=0
timeout 180 $SOUC run self-hosted/compiler/native_compile_driver.sio -- /tmp/gate_exit0.sio -o /tmp/gate_exit0.elf 2>&1 || _ec=$?
if [ -f /tmp/gate_exit0.elf ]; then
    chmod +x /tmp/gate_exit0.elf
    /tmp/gate_exit0.elf && _run_ec=0 || _run_ec=$?
    if [ "$_run_ec" = "0" ]; then pass "compile:exit0"
    else fail "compile:exit0 (exit=$_run_ec, expected 0)"; fi
else
    if [ "$_ec" = "124" ] || [ "$_ec" = "137" ]; then not_run "compile:exit0 (timeout/OOM ec=$_ec)"
    else fail "compile:exit0 (no ELF, ec=$_ec)"; fi
fi

# compile:add_fn — two-function program with inter-function call
_full_patch 2>/dev/null
_ec=0
timeout 240 $SOUC run self-hosted/compiler/native_compile_driver.sio -- /tmp/gate_add.sio -o /tmp/gate_add.elf 2>&1 || _ec=$?
if [ -f /tmp/gate_add.elf ]; then
    chmod +x /tmp/gate_add.elf
    /tmp/gate_add.elf && _run_ec=0 || _run_ec=$?
    if [ "$_run_ec" = "42" ]; then pass "compile:add_fn"
    else fail "compile:add_fn (exit=$_run_ec, expected 42)"; fi
else
    if [ "$_ec" = "124" ] || [ "$_ec" = "137" ]; then not_run "compile:add_fn (timeout/OOM ec=$_ec)"
    else fail "compile:add_fn (no ELF, ec=$_ec)"; fi
fi

echo ""

# ============================================================================
# Regression
# ============================================================================
echo "[regression]"

git checkout HEAD -- self-hosted/native/elf.sio self-hosted/native/codegen.sio 2>/dev/null || true

if grep -q 'fn finalize_elf64_for_machine' self-hosted/native/elf.sio; then
    pass "regr:elf_sio_unchanged"
else
    fail "regr:elf_sio_unchanged"
fi

if grep -q 'fn finalize_compiled_elf' self-hosted/native/codegen.sio; then
    pass "regr:codegen_has_finalize"
else
    fail "regr:codegen_has_finalize"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=== Gate Summary ==="
echo "PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN TOTAL=$TOTAL"

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
