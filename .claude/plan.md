# Phase 6: Native Codegen Self-Hosting — Implementation Plan

## Goal

Take the self-hosted native backend from "compiles `fn main() { 42 }`" to "compiles real Sounio programs with structs, arrays, and I/O". This is the critical path to eliminating the Poseidon VM from production use.

## Current State

The native backend handles 12 of 20 IR opcodes:
- **Working:** IrLoadImm, IrLoadBool, IrLoadString, IrCopy, IrBinOp (all 12 ops), IrUnaryOp, IrCall, IrReturn, IrJump, IrBranchTrue, IrBranchFalse, IrLabel, IrNop
- **Missing:** IrAlloc, IrFieldGet, IrFieldSet, IrIndexGet, IrIndexSet, IrLoadFloat

The self-hosted code uses ALL missing opcodes extensively — structs, arrays, field access, and indexing are the backbone of every module.

## Architecture Decisions

### 1. Memory Model: Bump-Allocated Heap via mmap

**Decision:** Use r12 as heap base, r13 as bump cursor. Initialize at `_start` via `mmap(NULL, 4MB, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)`.

**Rationale:**
- Stack-allocated composites won't work because `field_idx_from_name` is a hash (`first_char % 64`), not a sequential layout — the allocation must be 512 bytes (64 slots * 8 bytes) for each object
- The VM already uses a pointer-based heap model — matching it maintains semantic equivalence
- r12/r13 are callee-saved per System V ABI — survive across function calls with zero overhead

### 2. Field Access: Hash-Indexed Slots

IrFieldGet/Set use `field_idx = name.buf[0] % 64`. The native backend treats each allocated block as a sparse 64-slot array. `IrFieldSet` stores at `base_ptr + field_idx * 8`. `IrFieldGet` loads from the same offset.

### 3. Builtins: Emitted as Compiled Functions

Emit builtin runtime functions (print, print_int, print_char, etc.) as real x86-64 function bodies in the code buffer, before user functions. Calls to builtins use the same relocation mechanism as user function calls.

### 4. Buffer Sizes: Increase Gradually

Start with current 65KB code buffer. Increase to 256KB only when tests demonstrate the need (Milestone 6). Don't prematurely over-allocate.

---

## Milestones

### M1: IrAlloc + IrFieldGet + IrFieldSet (~130 LOC)
**Files:** `encode.sio`, `lower_ir.sio`, `codegen.sio`

**encode.sio** — new instructions:
- `emit_mov_rax_r13()` — `4c 89 e8` (mov rax, r13)
- `emit_add_r13_imm32(imm)` — `49 81 c5 <imm32>` (add r13, imm32)
- `emit_load_rax_mem_rax_disp32(disp)` — `48 8b 80 <disp32>` (mov rax, [rax+disp32])
- `emit_store_rbx_mem_rax_disp32(disp)` — `48 89 98 <disp32>` (mov [rax+disp32], rbx)

**lower_ir.sio** — new lowering cases in `lower_instr`:
- `lower_alloc`: mov rax, r13; add r13, 512; store rax to slot(dst)
- `lower_field_set`: load base from slot(src1); load value to rbx from slot(src2); store rbx to [rax + field_idx*8]
- `lower_field_get`: load base from slot(src1); load [rax + field_idx*8] to rax; store to slot(dst)

**codegen.sio** — modify `emit_entry_trampoline`:
- Before calling main: emit mmap syscall (rax=9, rdi=0, rsi=4194304, rdx=3, r10=0x22, r8=-1, r9=0)
- `mov r12, rax` (heap base), `mov r13, rax` (bump cursor)
- New encoding helpers for setting r10, r8, r9

**Test:** Struct with two fields, access and add them, verify exit code = sum

### M2: IrIndexGet + IrIndexSet (~80 LOC)
**Files:** `encode.sio`, `lower_ir.sio`

**encode.sio** — new instructions:
- `emit_load_rbp_disp32_rbx(disp)` — load to rbx from stack slot
- `emit_load_rbp_disp32_rcx(disp)` — load to rcx from stack slot
- `emit_imul_rbx_rbx_imm8(8)` — `48 6b db 08` (index * 8)
- `emit_add_rax_rbx()` — `48 01 d8`
- `emit_store_rcx_mem_rax()` — `48 89 08` (mov [rax], rcx)
- `emit_load_rax_mem_rax()` — `48 8b 00` (mov rax, [rax])

**lower_ir.sio** — new lowering:
- `lower_index_get`: load base from slot(src1); load index to rbx from slot(src2); imul rbx, 8; add rax, rbx; load [rax]; store to slot(dst)
- `lower_index_set`: load base from slot(src1); load index to rbx from slot(src2); imul rbx, 8; add rax, rbx; load value to rcx from slot(imm_i64); store rcx to [rax]

**Test:** Array of 4 elements, access elements by index, sum and verify exit code

### M3: Builtin Runtime Functions (~250 LOC)
**Files:** `codegen.sio`, `encode.sio`

**Builtins to implement (priority order):**
1. `print_int(n)` — integer-to-decimal conversion + sys_write(1, buf, len)
2. `print_char(c)` — push to stack + sys_write(1, rsp, 1)
3. `print(str)` — strlen + sys_write(1, str, len)

**codegen.sio** changes:
- Add `builtin_offsets: [i64; 16]` and `builtin_count: i64` to NativeCompiler
- Add `emit_builtins(nc)` — called before user functions in `compile_module`
- Add `is_builtin_name(name)` — byte-level name matching (like `name_is_main`)
- Modify relocation resolution to handle builtin function indices

**encode.sio** — syscall helpers:
- `emit_mov_rdi_1()` — for fd = stdout
- Additional register moves for syscall argument setup

**Test:** Program that calls print_int(42), print_char(10), verify stdout = "42\n"

### M4: argv Access + String Ops (~100 LOC)
**Files:** `codegen.sio`, `encode.sio`

**_start modifications:**
- Before mmap: `mov r14, rsp` (save original stack pointer with argc/argv)
- After mmap: store argc and argv pointer at heap[0] and heap[8], bump r13 past them

**Builtins:**
- `get_arg_count()` — load from [r12], return argc
- `get_arg(n)` — load argv base from [r12+8], then load [argv + n*8]
- `str_len(s)` — byte-scan for null terminator
- `str_eq(a, b)` — byte-by-byte comparison

**Test:** Program that returns get_arg_count() as exit code

### M5: File I/O Builtins (~150 LOC)
**Files:** `codegen.sio`

**Builtins:**
- `read_file(path, buf, max_len)` — sys_open(path, O_RDONLY) + sys_read(fd, buf, len) + sys_close(fd)
- `write_file(path, buf, len)` — sys_open(path, O_WRONLY|O_CREAT|O_TRUNC, 0644) + sys_write(fd, buf, len) + sys_close(fd)
- `file_size(path)` — sys_stat(path, statbuf) + extract st_size

**Test:** Write a file, read it back, verify contents match

### M6: Capacity Scaling (~30 LOC)
**Files:** `encode.sio`, `elf.sio`, `codegen.sio`, `reloc.sio`

**Increases (only if needed based on testing):**
- `CodeBuffer.bytes`: 65536 → 262144
- `Elf64Binary.bytes`: 65536 → 524288
- `RelocationTable`: 256 → 1024 entries
- `NativeCompiler.fn_offsets`: 64 → 256

**Risk:** Larger structs passed by value may hit Rust interpreter stack limits. Test with `RUST_MIN_STACK=8388608` if needed.

---

## Dependency Graph

```
M1 (Structs) → M2 (Arrays) → M4 (argv) → M5 (File I/O)
                                ↑
M3 (print/IO) ─────────────────┘
                                        All → M6 (Scale)
```

M1 and M3 are independent and can be done in parallel.

## Testing Strategy

Each milestone adds tests at two levels:

1. **Self-hosted unit test** in `self-hosted/native/test_phase2.sio` — build IR manually, compile to ELF, verify structure
2. **Rust integration test** in `crates/souc/tests/native_phase2_execution.rs` — compile fixture .sio → ELF via interpreter, execute, verify exit code and stdout

## Estimated Total: ~740 LOC new Sounio code

## Non-Goals (This Phase)

- IrLoadFloat (not needed for self-hosting core pipeline)
- IrPhi (not emitted by lowerer)
- macOS Mach-O / Windows PE (Linux-only for now)
- Register allocation optimization (stack-slot model is sufficient)
- Multi-module compilation (separate Project Poseidon track)
