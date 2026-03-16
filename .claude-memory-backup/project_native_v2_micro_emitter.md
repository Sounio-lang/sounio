---
name: native-v2-micro-emitter-breakthrough
description: Micro-emitter architecture bypasses both JIT bugs to produce correct native x86-64 binaries
type: project
---

## Native v2 Live Path — Micro-emitter Architecture (2026-03-13)

### Root Cause: Two JIT Bugs Block Standard Codegen

1. **NativeCompiler by-value return corruption (~200KB struct)**: `compile_ir_function` returns garbled code. Prologue/epilogue bytes survive but instruction body is lost. Confirmed: `mov rax, 42` (from IrLoadImm) is completely absent from output, only the store/load register moves survive.

2. **&! reference mutation invisibility**: `compile_ir_function_v2_mut(nc: &! NativeCompiler, ...)` writes through `(*nc).code = emit_*(...)` but caller sees `nc.code.len=0`. Known JIT register caching bug.

3. **Nested CodeBuffer returns also corrupt**: `emit_mov_rax_imm → emit_mov_reg_imm → emit_byte` chain loses bytes. The 7-byte `mov rax, 42` is completely missing from output.

### What Works Under JIT

- **`emit_byte(code: CodeBuffer, byte)` → CodeBuffer**: Single-level 65KB by-value return WORKS
- **Direct `emit_byte` chains**: `c = emit_byte(c, 0x48); c = emit_byte(c, 0xb8); ...` — all bytes preserved
- **`micro_store_rax`/`micro_load_rax` helpers**: Single-level CodeBuffer returns work
- **Inline ELF writer**: Direct array manipulation works
- **`compile_module_streaming_finish`**: Entry trampoline (236 bytes) is generated but GARBLED (missing mmap setup)

### Micro-emitter Architecture

**File**: `self-hosted/compiler/render_native_compile_driver_lean.sio`

- Uses `emit_byte` chains directly — NO nested `emit_*` function calls
- Stack-slot model: vreg N → `[rbp - 8*(N+1)]`
- Handles: IrLoadImm, IrLoadBool, IrReturn, IrBinOp (all arithmetic + comparison), IrCall, IrCopy, IrLabel, IrJump, IrBranchTrue, IrBranchFalse, IrLoadString
- Inline trampoline: `call main; mov edi,eax; mov eax,60; syscall` (14 bytes)
- Inline ELF writer: direct array writes (no finalize_elf64 which hangs)

**Proven**: `fn main() -> i64 { 42 }` → EXIT=42 ✓
**Proven**: `fn main() -> i64 { 20 + 22 }` → EXIT=42 ✓

### Remaining Blockers for triangle_basic.sio

1. **IR lowering**: `lower_program_function_body_from_summary_flat_with_epistemic_ref` fails for `let` bindings and function calls with arguments (body_error). Only works for trivial single-expression functions.
2. **Missing IR opcodes**: f64 (IrLoadFloat, IrIntToFloat, IrFloatToInt), struct (IrAlloc, IrFieldGet, IrFieldSet), index (IrIndexGet, IrIndexSet)
3. **No builtin function dispatch**: print, print_int, clamp_byte etc. need hand-coded x86 implementations
4. **No data section**: String constants, f64 constants need .rodata

**Why**: `emit_mov_rax_imm` (7 bytes) through nested function calls → bytes lost
**How to apply**: Always use direct `emit_byte` chains; never call `emit_*` helper functions from encode.sio
