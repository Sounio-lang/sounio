# Madaros native-v2 backend: f64 + str_from_bytes miscompilation (minimal repros)

Found 2026-06-30 while porting the octonion mass-δ artifact
(`examples/physics/octonion_mass_delta.sio`) from `lean_single` to Madaros (v0.80.0,
default `bin/souc`).

## Good news first
Madaros has advanced a lot vs the prior "segfault during octonion lowering" state:
frontend, merged-IR, type-check, and **integer** native-v2 codegen all work and emit a
running ELF. Integer programs (println + print_int) run correctly.

## The boundary (3 distinct native-v2 defects)
All three **compile** ("Compilation successful!", ELF written) but produce wrong results
or crash **at runtime**. The identical files run correctly under `lean_single`
(`SOUNIO_SOUC_ENGINE=lean_single`), so these are Madaros-native-v2-specific.

| repro | program | lean_single | Madaros native-v2 |
|---|---|---|---|
| m2 | `let x:f64=3.0/8.0; print_int((x*1e6) as i64)` | `375000` | **`0`** |
| m3 | f64 Newton sqrt loop (`g=0.5*(g+0.375/g)`) | `612372` | **SIGFPE (floating point exception)** |
| m4 | `str_from_bytes(buf,2)` on `[i8;8]` | `Hi` | **Segmentation fault** |

### Repro m2 — f64→i64 cast yields 0 (f64 arithmetic likely lowered as integer)
```sounio
fn main() -> i32 with IO, Mut, Div, Panic {
    let x: f64 = 3.0 / 8.0
    print_int((x * 1000000.0) as i64); print("\n"); 0
}
```
Expected `375000`; Madaros prints `0`. Suggests f64 `/` and `*` (or the f64→i64 `as`
cast) are being emitted as integer ops, so `3/8 → 0`.

### Repro m3 — f64 division SIGFPEs at runtime
```sounio
fn main() -> i32 with IO, Mut, Div, Panic {
    var g: f64 = 1.0; var i: i64 = 0
    while i < 30 { g = 0.5*(g + 0.375/g); i = i+1 }
    print_int((g*1000000.0) as i64); print("\n"); 0
}
```
Expected `612372` (√(3/8)); Madaros raises a hardware **floating-point exception** —
consistent with an integer `idiv` being emitted where an f64 `divsd` is required.

### Repro m4 — str_from_bytes segfaults at runtime
```sounio
fn main() -> i32 with IO, Mut, Div, Panic {
    var b: [i8; 8] = [0; 8]
    b[0]=72 as i8; b[1]=105 as i8
    println(str_from_bytes(b, 2)); 0
}
```
Expected `Hi`; Madaros segfaults — likely a bad pointer/length to the str_from_bytes
builtin in native-v2.

## Impact
The octonion mass-δ scientific artifact is f64- and string-heavy, so it cannot run
correctly on Madaros until m2/m3 (f64 ALU lowering) and m4 (str_from_bytes) are fixed.
`lean_single` remains the engine that emits correct numeric binaries.
Multimodule note: the full artifact (with `use algebra::octonion`) fails earlier at
`multimodule native thin-link compilation failed` / `imported_simple_ir_emit_failed`
— a separate imported-IR emit path, also worth a look, but the f64 ALU bug is upstream
of it.

## Suggested first fix (SUPERSEDED — see corrected root cause below)
~~Audit the native-v2 instruction selection for f64~~ — this framing was wrong.

## CORRECTED ROOT CAUSE (2026-06-30, after disassembling the emitted ELF)
Scanned the m2 ELF for opcodes: **zero** `divsd/mulsd/addsd/subsd` (no SSE float ops at
all), but `idiv`+`imul` present. The program isn't being generally compiled on the modular
path at all:

- Madaros (Horizon 3) routes these files through the **"compact modular IR table" path**
  (`module_native_driver.sio` → `native_driver_write_imported_simple_ir_elf`).
- That path's emitter, `module_native_simple_driver.sio::simple_driver_emit_fn`, is a
  **bootstrap template stub**: it pattern-matches a fixed set of function "kinds"
  (kind 1=tail-call, kind 2 = `mov eax,imm32; ret` = *return a constant*, kind 3=print, …)
  and writes hand-crafted bytes. It is **not a general code generator**. m2 returned `0`
  because it matched a "return constant" template; arbitrary f64/general code is unhandled.
- The **f64-correct backend already exists**: `lower_ir.sio::lower_ir_binop` →
  `emit_float_binop_code` → `emit_divsd_xmm0_xmm1` (encode.sio:1825, opcode 0x5E), with
  `is_float_reg` tracking. It is reached only by `main.sio`'s direct path (`lower_instr`,
  main.sio:5590), which the modular path bypasses.

So the real gap is **backend routing**, not f64 instruction selection.

### Correct fix (architectural, not a localized patch)
Translate the compact modular IR records into `IrInstr` and drive `lower_instr` (the full
`lower_ir.sio` lowering + regalloc + encode pipeline, already f64-correct) instead of
`simple_driver_emit_fn`'s templates. Then `make build-madaros` (bootstrap rebuild) and
re-test m2/m3/m4 (expect 375000 / 612372 / "Hi") + the multimodule octonion artifact.
Substantial integration with bootstrap-fixed-point risk. Until then, f64-heavy code runs
correctly on `lean_single` (which uses the full backend).

---

## REBUILD VERIFIED (2026-06-30): f64 fix WORKS; two separate blocks remain

Rebuilt Madaros with the routing fix (lean_single seed) and tested:

| repro | before | after fix |
|---|---|---|
| m2 `(f64 3.0/8.0)*1e6 as i64` | 0 | **375000** ✓ FIXED |
| m3 f64 Newton division | SIGFPE | **612372** ✓ FIXED |
| m4 `str_from_bytes` | segfault | empty output, no crash (string still wrong) — separate builtin bug |
| `octonion_mass_delta.sio` (multimodule, octonion intrinsics) | thin-link failed | takes the full merged-IR path (fix active) but the v2 backend SIGSEGVs (139) on octonion intrinsics |

**Verdict:** the routing fix RESOLVES the f64 codegen bug for general code on Madaros
(m2/m3 correct, verified). Two independent blockers remain before the full octonion
artifact runs on Madaros, both in the full v2 backend (not the routing):
1. `str_from_bytes` returns empty in the v2 backend (no longer segfaults).
2. The v2 backend segfaults compiling the octonion intrinsics (`oct_mul`/`oct_associator`
   lowering, `IrHyperMulO`/`IrAssociator` paths in lower_ir.sio).
Both are out of scope of the f64 routing fix. f64-heavy *scalar* programs now compile and
run correctly on Madaros; the octonion artifact stays on lean_single until (1)+(2) land.
No regression: integer + f64 programs compile and run correctly with the fixed binary.

---

## (a) str_from_bytes — ROOT CAUSE LOCATED (2026-06-30)

`str_from_bytes` is **missing from the v2 backend's builtin table**. In
`self-hosted/native/codegen_x86_linux.sio`, `native_v2_builtin_id_for_func_ref`
(lines ~1065-1088) maps ~21 builtins (str_len=6, str_char_at=20, str_eq=7,
str_slice=8, starts_with=9, str_concat=10, read_file=11, …) but has **no entry for
str_from_bytes**. So a call to it returns id 0 (not-a-builtin) → treated as a call to a
bodiless function → returns garbage/empty (matches m4: empty output, no crash).

VM string representation (from `emit_builtin_str_concat`): null-terminated `char*` on a
bump heap (`RuntimeContext.heap_cursor`).

### Fix recipe (NOT applied — needs a working rebuild loop to verify; writing raw
machine-code emitters blind is unsafe, doubly so now the binary is shared/promoted):
1. Add recognizer `name_is_str_from_bytes(n)` ("str_from_bytes" = 14 bytes), mirroring
   `name_is_str_concat`.
2. Register an id (e.g. 22) in `native_v2_builtin_id_for_func_ref` and in the name-based
   dispatch near line 4752 (`if name_is_str_from_bytes(func.name) { return emit_builtin_str_from_bytes(c) }`).
3. Add to the id dispatch near line 5570:
   `if builtin_id == 22 { native_v2_persist_builtin_emit_into(nc, emit_builtin_str_from_bytes((*nc))); return }`.
4. Write `emit_builtin_str_from_bytes`: args rdi=buf, rsi=len → bump-allocate len+1 from
   heap_cursor, copy len bytes (rep movsb or a byte loop like starts_with), store NUL at
   end, return pointer in rax. Mirror `emit_builtin_str_concat`'s heap-alloc/copy/advance
   sequence exactly (it is the canonical string-producing builtin).
Verify: rebuild + m4 → "Hi".

## (b) octonion-intrinsic SIGSEGV in v2 — NOT yet localized
The full octonion artifact takes the (now-default) full merged-IR path but the v2 backend
SIGSEGVs (exit 139) compiling the octonion intrinsics (`oct_mul`/`oct_associator` →
`IrHyperMulO`/`IrAssociator`). Next step: determine whether the v2 path (`compile_ir_function_v2`)
handles these IR opcodes at all (lower_ir.sio has `lower_hyper_mul_o_fano`/`lower_associator`,
but the v2 machine-IR path may not route to them). Needs investigation + rebuild to verify.

---

## (a) str_from_bytes — wired into v2, but runtime ABI segfault (2026-06-30, NOT promoted)

Added `name_is_str_from_bytes` + builtin id 22 + dispatch + `emit_builtin_str_from_bytes`
(`mov rax, rdi; ret` — return the buffer pointer, matching lean_single's
"returns the address of the byte buffer" semantics). Rebuilt (scratch madaros-fix-a):
- m2/m3 still correct (375000 / 612372) — no f64 regression.
- m4 now **compiles** (builtin recognized; previously returned empty) but **SIGSEGVs at
  runtime** (exit 139).

So the missing-builtin was real but not the whole story: with the builtin wired, the fault
moves to the **array-argument / builtin-call ABI**. `str_from_bytes(b, 2)` passes an array
`[i8;8]` as arg0; the emitter assumes rdi = &b (as str_slice does for a string pointer), but
the runtime segfault indicates the array address is not arriving in rdi the way a string
pointer does (array-arg passing differs from pointer-arg passing in the v2 call path), or the
returned stack-array pointer is mishandled by the caller. Needs: disassemble the m4 call site
to confirm what's in rdi at the call, and/or inspect how the v2 backend lowers array
arguments to a call. **fix-a was NOT promoted** — it would regress str_from_bytes from
empty-output to crash. The promoted binary remains the f64-only fix (str_from_bytes returns
empty, no crash).

---

## (a) EXACT root cause found via disassembly (2026-06-30)

Disassembled the m4 ELF (raw binary code segment fed to `objdump -b binary -m i386:x86-64`,
since the ELF has no section headers). Found the call site:

```
mov rdi, [rbp-0x8]     ; rdi = local var "b" ([i8;8] array)
mov rsi, [rbp-0x38]    ; rsi = 2 (len)
call 0x40141b          ; str_from_bytes(rdi, rsi)   <- my new emitter: mov rax,rdi; ret
```

And the array-store code (for `b[0]=72; b[1]=105;`) reveals arrays in the v2 backend are
**boxed/indirect, not raw byte buffers**:

```
mov rax, [rbp-8]              ; rax = "b" = a SLOT INDEX (small int), NOT a pointer
imul rbx, rax, 0x30           ; rbx = slot_index * 48   (descriptor stride)
mov rax, [RuntimeContext+0x18]; rax = array-descriptor-table base
add rax, rbx
mov rax, [rax]                ; DEREFERENCE -> rax = the REAL data pointer (descriptor field 0)
...
mov [rax + (i+4)*8], value    ; element i stored at data_ptr + 32 (4-qword header) + i*8
                               ; (8 bytes PER i8 ELEMENT — uniform boxed representation)
```

So `rdi` at the `str_from_bytes` call site is a **slot index into an array-descriptor
table**, not a byte-buffer address. My emitter (`mov rax,rdi; ret`, mirroring how
`str_slice`/`str_concat` treat their args as raw `char*`) just echoes the slot index back —
the caller (`println`) then dereferences that small integer as if it were a string pointer
→ SIGSEGV. `str_slice`/`str_concat` never hit this because they only ever receive **strings**
(already raw null-terminated `char*`, from literals/heap), never a stack-declared `[i8;N]`
array — so this ABI mismatch was latent until `str_from_bytes` (the only builtin that takes
an array argument) was wired up.

### What a correct fix requires
`emit_builtin_str_from_bytes` must, given `rdi`=slot index and `rsi`=len:
1. Resolve the descriptor: `descriptor_ptr = [RuntimeContext+0x18] + rdi*48`.
2. Load the real data pointer: `data_ptr = [descriptor_ptr]`.
3. Bump-allocate `rsi+1` bytes from `RuntimeContext.heap_cursor` (mirror
   `emit_builtin_str_concat`'s allocation phase) for a **packed** result buffer.
4. Copy `rsi` bytes, reading each source byte from `[data_ptr + 32 + i*8]` (low byte of the
   8-byte boxed element) and writing packed to `[result + i]`.
5. NUL-terminate, return `result` in rax.
This is a real fix (not a one-liner) requiring the exact descriptor stride (0x30) and header
size (32 bytes / 4 qwords) confirmed above, plus a copy loop. Not applied this session —
needs a dedicated rebuild-test cycle to get the encoding right; the promoted shared binary
must not regress. Current promoted binary keeps the safe (non-crashing, wrong-output)
str_from_bytes state; `madaros-fix-a` (the crashing version) stays unpromoted.

---

## (a) Complete fix recipe — all pieces identified, NOT hand-assembled (2026-06-30)

Traced every constant and helper needed, all already exist and are used elsewhere (so
they are trustworthy, tested code paths):

- `native_v2_resolve_handle_to_object_base_rax(nc) -> nc` (codegen_x86_linux.sio:2691):
  takes a handle in `rax`, returns the object's data pointer in `rax`. This is the exact
  function `native_v2_core_emit_local_array_load_into` (line 7991, used for ordinary
  `arr[i]` reads) calls — confirms it's the right, tested primitive.
- `native_v2_handle_entry_size() = 48` (gc.sio:35) — matches the `imul rbx,rbx,0x30` seen
  in the disassembly.
- `native_v2_object_header_size() = 32` (gc.sio:27) — matches the `+4` qword offset (`(i+4)*8`)
  seen for element access.
- `nc_emit_load_rax_mem_rdx_rbx8` / `nc_emit_store_rax_mem_rdx_rbx8` (codegen_x86_linux.sio:1501):
  raw `mov rax,[rdx+rbx*8]` / `mov [rdx+rbx*8],rax` — the exact element-read/write instruction,
  confirmed byte-identical to the disassembled array-store code.

So `emit_builtin_str_from_bytes(nc)` [rdi=handle, rsi=len] should:
1. `rax=rdi; c = native_v2_resolve_handle_to_object_base_rax(c)` → `rax` = object_base.
2. Bump-allocate `len+1` bytes from `heap_cursor` (mirror `emit_builtin_str_concat`'s Phase 3)
   → result pointer.
3. Loop `i` in `0..len`: read element via `object_base`, index `(i + 32/8)`, using
   `nc_emit_load_rax_mem_rdx_rbx8` (rdx=object_base, rbx=i+4) → low byte is the packed byte;
   write to `[result+i]`.
4. NUL-terminate at `[result+len]`; return `result` in `rax`.

### Why this was NOT hand-assembled and shipped
Steps 3's loop requires a conditional branch (`jz`/`jnz` to a "done" label) whose relative
displacement must be the exact byte count of the intervening instructions — this codebase's
raw-builtin style (`emit_builtin_str_concat`, `emit_builtin_starts_with`) computes these by
**manual byte-counting** (see e.g. `starts_with`'s comments `// jz match (skip 2+2+3+3+2=12)`),
with no assembler/two-pass relocation available at this level (unlike full IR-compiled
functions, which get real label/reloc support — see `IrIndexGet`/`IrIndexSet`'s clean
label-based lowering at codegen_x86_linux.sio:6676, a SAFER model this builtin could follow
in a future rewrite: expand `str_from_bytes` as small generated IR code with `IrIndexGet` +
`IrBranchTrue`/`IrJump`, not a hand-written raw-bytes leaf function). A single miscounted
byte here produces a silently wrong jump target (crash or, worse, silent memory
corruption) that only surfaces after another ~10min rebuild — on a currently
contended shared build box, and on a binary that gets promoted to shared use. Per this
session's discipline (verify before promote, no blind machine-code), this is deliberately
left as a precise, actionable spec rather than shipped code.
