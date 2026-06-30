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
