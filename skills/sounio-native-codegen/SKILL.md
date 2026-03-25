---
name: sounio-native-codegen
description: "Work on the native x86-64 code generation backend: register allocation, instruction encoding, peephole optimization, frame sizing, ELF output, and the compile_ir_function pipeline; use when editing any sprint 52–65+ native/*.sio files."
---

# Sounio Native Codegen Backend

## Overview

The native backend lowers `IrFunction` to x86-64 ELF machine code in a single-pass pipeline wired inside `compile_ir_function()` in `self-hosted/native/codegen.sio`:

```
IrFunction → regalloc → peephole → lower_ir → disp8 encoding → compact frame → emit prologue → lower instrs → epilogue → patch jumps
```

All passes live in `self-hosted/native/`. The pipeline is complete as of Sprint 65 (30/30 gate).

## Workflow

### 1) Understand which pass to touch

| Pass | File | Entry Function | Sprint |
|------|------|---------------|--------|
| Linear-scan regalloc (4-preg r12-r15) | `native/regalloc.sio` | `nc_run_regalloc()` | 52/62 |
| Peephole optimizer (IR-level) | `native/peephole.sio` | `ph_run_on_func()` | 65 |
| Adaptive disp8/disp32 encoding | `native/lower_ir.sio` | `ir_slot_fits_disp8()` | 63 |
| Compact frame sizing | `native/lower_ir.sio` | `nc_min_frame_size()` | 64 |
| Instruction lowering (IrInstr → x86) | `native/lower_ir.sio` | `lower_instr()` | ongoing |
| Byte emitters | `native/encode.sio` | `emit_*()` | ongoing |
| Prologue/epilogue | `native/codegen.sio` | `emit_prologue_with_preg_mask()` | 62 |
| ELF output | `native/elf.sio` | `emit_elf_header()` | M3 |
| Relocations | `native/reloc.sio` | `RelocationTable`, `add_call_reloc()` | M3 |
| NativeCompiler state | `native/frame.sio` | `NativeCompiler` struct | M3 |

### 2) Add a new native optimization pass

1. Create `self-hosted/native/your_pass.sio` with `module native::your_pass` declaration
2. Import deps: `use ir::ir::{IrFunction, IrInstr, ...}` as needed
3. Define entry: `fn your_pass_run(func: IrFunction) -> IrFunction with Mut, Panic`
4. In `self-hosted/native/codegen.sio`:
   - Add `use native::your_pass::{your_pass_run}` to imports
   - Call `effective_func = your_pass_run(effective_func)` at the right pipeline point
5. Add self-tests T(N+1)–T(N+k) in `self-hosted/compiler/main.sio` — update `let total: i64`
6. Write gate script following `scripts/sprint65_peephole_gate.sh` as template

**Pipeline order in `compile_ir_function()`** (codegen.sio ~line 1574):
```
nc_run_regalloc → ph_run_on_func → nc_min_frame_size → emit_prologue → lower instrs → emit_epilogue
```

### 3) Add an encode.sio helper

Pattern for a new `emit_*` byte emitter:
```sio
fn emit_mov_rax_r13(code: CodeBuffer) -> CodeBuffer {
    // REX.W + MOV r/m64, r64: 48 8B C5 = mov rax, r13
    let c0 = code_append_byte(code, 0x48)
    let c1 = code_append_byte(c0,   0x8B)
    code_append_byte(c1, 0xC5)
}
```
All helpers in `encode.sio` are pure (take + return `CodeBuffer`). No side effects.

### 4) Run the relevant gate

```bash
SOUC=./bin/souc

bash scripts/sprint65_peephole_gate.sh     # 30/30 — peephole optimizer wiring
bash scripts/sprint64_frame_trim_gate.sh   # 25/25 — compact frame sizing
bash scripts/sprint63_disp8_gate.sh        # 30/30 — adaptive disp8/disp32 encoding
bash scripts/sprint62_multipreg_gate.sh    # 35/35 — 4-preg linear-scan regalloc
bash scripts/sprint52_regalloc_gate.sh     # 33/33 — regalloc wiring baseline
```

New passes must add their own gate script following the sprint65 pattern.

## References

- Pipeline orchestrator: `self-hosted/native/codegen.sio` — `compile_ir_function()` (~line 1537)
- Register allocator: `self-hosted/native/regalloc.sio` — `nc_run_regalloc()`, Poletto-Sarkar linear scan, 4 pregs (r15/r14/r13/r12)
- Peephole optimizer: `self-hosted/native/peephole.sio` — `ph_run_on_func()`, `ir_instr_to_ph()`, `ph_opt_single()`, `ph_opt_pair()`
- Instruction lowering: `self-hosted/native/lower_ir.sio` — `lower_instr()`, `ir_slot_fits_disp8()`, `nc_min_frame_size()`
- Byte emitters: `self-hosted/native/encode.sio` — 300+ `emit_*` functions
- NativeCompiler state: `self-hosted/native/frame.sio` — `NativeCompiler` struct, `vreg_to_preg[]`, `vreg_spill_slots[]`, `current_frame_size`
- ELF output: `self-hosted/native/elf.sio` — `emit_elf_header()`, `.text`/`.data`/`.rodata` sections
- Relocations: `self-hosted/native/reloc.sio` — `RelocationTable`, `add_call_reloc()`, `add_data_section_reloc()`
- Self-tests: `self-hosted/compiler/main.sio` — T62–T80 cover regalloc, frame, disp8, peephole
- Sprint history: MEMORY.md §Project Status (sprints 52, 62–65)
