# Wiring native-v2 to emit ELF from source — seam ready, blocked by a source→IR crash (2026-06-03)

Goal: make `mc --native-compile foo.sio -o foo` (modular compiler) produce a runnable ELF via
the real source→IR→native-v2→ELF path (today `--native-compile` runs a stub: prints
"source=fallback fallback=unresolved_default" and emits nothing).

## What is READY (verified)
- **The seam is correct and small.** `compile_native_v2_preview_to_file(module: &IrModule, spec,
  out) -> i32` (codegen_x86_linux.sio:7530) is a **real, general backend** — it iterates every
  function in the IrModule (`compile_ir_function_v2_from_ir_into`), finds main, emits the entry
  trampoline, applies relocations, writes a min ELF64. Not a 9-witness toy. Returns 12 if any
  function fails to compile.
- **The back-half works end-to-end.** `--native-v2-emit13` builds an `IrModule` by hand
  (`compiler_main_make_native_v2_scalar_module(ret) -> IrModule`) and emits a runnable ELF
  (exit 13). So IrModule→ELF is proven.
- **The IR is reachable in principle.** `load_multimodule_ir(src) -> MultiModuleIrResult { ok,
  module: IrModule, error_msg }` carries the IrModule directly in `.module`. The wiring would be:
  `let r = load_multimodule_ir(src); if r.ok { compile_native_v2_preview_to_file(&r.module, spec, out) }`.

## The BLOCKER (precisely pinned)
The **source→IR front** crashes (SIGSEGV) even on trivial input — `fn main(){}` and
`fn main()->i64{7}`. Both `--ir-dump` and `--emit-obj` (which call `load_multimodule_ir` /
`compiler_preflight_ir_load`) fault. `--check` on the same files is clean (rc 0).

Pinned by a marker build (mc rebuilt with `\n`-flushing print markers; `print` verified to flush
on `\n` before a crash via a print-then-stack-overflow probe):
- Markers fired: `A_before_load_mm` → `C_traced_start` → `D_before_lmf` → `E_after_lmf`, then crash.
- Crash is in `module_frontend::load_multimodule_ir_traced` (module_frontend.sio) at **lines
  3328–3334**, i.e. right after `let main_prog = load_module_file(main_path)` returns the large
  `Program` AST struct by value, before the line-3335 `module_frontend_full: main_loaded` trace.
- Fault instruction `mov 0x0(%rdx),%rax` with a smashed stack (regular `0x..X008` pointer
  pattern) — a **corrupted-pointer deref**, the signature of the large-struct-by-value (SRET)
  miscompile, NOT a stack-guard hit.

## Diagnosis: this is the cluster-C large-struct-by-value family
- `IrModule` is multi-MB (`[IrFunction;1400]` + `[Name;4096]` + epistemic/algebra/ontology); `Program`
  is also a large AST struct. The load path returns these by value through `load_module_file ->
  Program`, `load_multimodule_ir -> MultiModuleIrResult`, `compiler_preflight_ir_load ->
  CompilerIrLoadResult` (each wraps the big aggregate).
- **Layout-sensitive** (the [[project_modular_span_sensitive_crash]] / cluster-C property): the
  *same* `load_module_file -> Program` return is fine under `--check`'s caller frames but smashes in
  the IR-load entry's large frames (`run_ir_dump_mode` alone has a 128KB `serialized_buf` +
  the multi-MB result local). emit13 avoids it with a small-frame caller + a hand-built module.
- This is the same miscompile the whole `*mut`/move-codegen arc exists to avoid, documented as
  intractable-codegen-without-gdb (B-repro verdict) — reliable fix is the `*mut`/out-param
  migration.

## Why no cheap fix landed
The by-value `Program` return **completed** (marker E fired after the assignment), so it is not a
clean single return-boundary I could wrap with one out-param. The smash is inside the load chain
and layout-sensitive; converting `load_module_file -> Program` (used everywhere, incl. the working
`--check` path) to an out-param is invasive and may merely relocate the layout-sensitive fault.
A real MVP would be a new small-frame source→IR driver built on the existing `Box<IrModule>`
lowering (`module_frontend_lower_items_into`, which already heap-routes to dodge the SRET) plus a
Box/heap parse — a genuine path, but multi-step with rebuilds, not a thin wrapper.

## Honest status & options (escalated to user)
The native-v2 **back-half is ready and the seam is one call**; the block is entirely the front-half
source→IR large-struct-by-value crash (cluster-C family) in `load_multimodule_ir_traced`.
1. **Commit to the source→IR `*mut`/Box migration** — the reliable path (mirrors the 481→0 checker
   migration; multi-session).
2. **Targeted attempt** — Box the `Program` local / out-param `load_module_file` on the IR-load
   path only; risks relocating the layout-sensitive smash; needs iteration + rebuilds.
3. **Leave documented** (like cluster C) — the seam stays ready for when the front-half crash is
   fixed; no quiet slide into an unrequested migration.

Note: even after the crash is cleared, a value-returning `main` (needed for an observable exit code
to prove execution) likely hits the still-live spurious-E008 (the fn_sigs fix is banked, not
shipped) — so a genuine MVP is: crash-free source→IR + the banked fn_sigs patch + an observable exit.
