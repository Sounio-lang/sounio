<!-- docs:meta
topic_id: repo.docs.compiler.known-limitations
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.known-limitations
-->

# Known Language Limitations

This document tracks limitations in the Sounio language implementation.
**Authoritative source for maturity tiering is `docs/serious-language/public-claim-registry.v1.tsv`.**
This file is reconciled to that registry. If they disagree, the registry wins.

Last reconciled: 2026-05-27 (PL adoption audit). Earlier "all-green" claims have been corrected against live probes; see §"Reconciliation notes" below.

## Maturity Tiers

Tiers below mirror the public-claim registry's `claim_level`/`closure_status` columns. "Production" is reserved for rows the registry calls `stable / closed`. Anything the registry marks `prototype` or `stale_conflicting` is tiered accordingly here — even if the feature works on small fixtures.

### Production (registry: stable / closed)

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer/Parser/AST | Production | logos-based, error recovery, comprehensive |
| Type Checker (core) | Production | Bidirectional inference, generic monomorphization (1–2 params), unification |
| Epistemic Types — Knowledge<T> / GUM | Production* | Registry tier: `validated_research`. "Production" here = single-file emit + GUM propagation works; clinical use must still cite the named gate. |
| Effects System | Production | 9+ effects (IO, Mut, Panic, Div, Alloc, Session, Observe, Audit, Hypothesis; +GPU, Deterministic). Strict E035 subset check at call sites. |
| HIR + HLIR | Production | SSA generation, async transform |
| SIR | Production | Domain-specific IR, epistemic passes |
| Ownership/Borrowing | Production | Method receiver type resolved from declared signature; exclusive `&!Self` enforces borrow-conflict tracking; shared `&Self` is read-only. No heuristic string matching. |
| Native Backend (Linux x86-64) | Production | Registry: `platform.linux_x86_64 = stable`. Direct x86-64 ELF emission + epistemic runtime + continuations. (There is no Cranelift/JIT backend — the retired Rust Cranelift runner is gone; the default compiler is self-hosted Madaros.) |
| Interpreter | Production | Full eval, 100+ builtins |
| Module System (single-file import unit) | Production | 2-pass resolver, imports, hierarchical namespaces |
| CLI commands `check`/`compile`/`build`/`run`/`info`/`--version` | Production | Wired through `bin/souc`. **Exit-code contract repaired 2026-05-27 — typecheck failures now exit non-zero (G2).** |
| snn/ (sedenion NN) | Production | Training, backward, similarity, 8 scoring functions |

### Validated Research (registry: validated_research / closed-with-named-gate)

| Component | Status | Notes |
|-----------|--------|-------|
| Native Backend (macOS arm64 / x86_64) | Validated research | Registry: `platform.macos = validated_research`. Mach-O cross-compile lane; no Apple JIT, no native-v2 parity. |
| Self-hosted Compiler (single-file path) | Validated research | `lean_single.sio` self-hosts; gen2==gen3 fixed point. Registry: `selfhost = validated_research`. |
| LLVM Codegen | Validated research | LLVM 18 bridge `self-hosted/llvm/souc_emit_llvm.c` wired but disabled in the checked artifact; `--backend llvm` needs a feature-flag rebuild. Previously over-claimed as Production. |
| Refinement Types + SMT | Validated research | Static engine handles constants, condition narrowing, monotonicity; complex predicates fall back to runtime assertions with W040 diagnostic. |
| Module imports across files | Validated research | Registry: `modules.imports = validated_research`. |
| Ownership / borrowing examples | Validated research | Registry: `ownership.borrowing = validated_research`; avoid Rust equivalence. |
| Editor tooling preview | Validated research | Registry: `tooling.editor = validated_research/closed`. `scripts/ci/sounio_editor_tooling_support_gate.sh` proves public `bin/souc format`/`fmt`, file-backed `bin/souc repl`, preview `bin/souc lsp --stdio`, G5a/G5b, bash LSP smoke, initialize capability smoke, and VS Code/Helix/Neovim static wiring. This is a SOTA-preview support contract, not mature IDE support. |
| LSP pure-Sounio server rebuild | Prototype blocker | `self-hosted/lsp/server.sio` currently fails to rebuild under the active Madaros path; the checked preview LSP route is `tools/lsp/sounio-lsp.sh` via `bin/souc lsp --stdio`. Do not claim the pure-Sounio LSP rebuild until `tools/lsp/test_protocol.sh` or an equivalent gate is green. |
| GPU PTX backend | Validated research | Registry: `gpu.ptx = validated_research`. Named gate covers L4 fixtures; out-of-fixture behavior is research. |
| 168 / Cayley-Dickson algebra | Validated research | Registry: `algebra.168 = validated_research`. Algebraic/formal artifacts only — no biological or EEG advantage claims. |
| Ontology subsystem | Validated research | Registry: `ontology = validated_research`. Rebuilt ontology validation surfaces only. |

Editor-tooling details:

- Formatter: `souc format <file>` / `souc fmt` dispatches to
  `tools/fmt/sounio-fmt.sh`. Phase 1 is token-level and idempotent for the
  G5a corpus; it does not claim AST round-trip formatting or full style
  configuration.
- REPL: `souc repl` dispatches to `tools/repl.sh`, a file-backed eval loop
  that accumulates definitions and runs expressions through the active `souc`
  wrapper. A fully Sounio-native eval loop remains deferred until process-spawn
  primitives are available.
- LSP: `souc lsp --stdio` dispatches to the preview `tools/lsp/sounio-lsp.sh`
  server. It is smoke-tested for JSON-RPC framing, compiler-backed diagnostics,
  hover/definition roundtrips, multi-document isolation, timeouts, and failure
  diagnostics. No pure-Sounio LSP rebuild under current Madaros is claimed or
  demonstrated; that server source remains a separate rebuild blocker.

### Bounded validated research surfaces and prototypes

| Component | Registry row | Honest status |
|-----------|--------------|---------------|
| **Standard library support surface** | `stdlib.surface = validated_research` | Claim only the bounded support contract checked by `scripts/ci/sounio_stdlib_surface_support_gate.sh`: current inventory has 1316 `.sio` files, 0 disabled files, 0 stub-only `mod.sio` files, and 178 active module entrypoints; package-backed epistemic/GUM, units, formats, io-primitives, canonical PETAB, and PBPK/GUM workflows pass through `scripts/ci/package_pbpk_gum_gate.sh`. **NOT PROVED:** broad all-file stdlib callability, `scripts/ci/stdlib_evolution_gate.sh`, hyper native lanes, fMRI/PBPK science pipeline, external runtime dependencies, cryptographic security, clinical/regulatory validity, or API stability beyond the checked gate. |
| **Package manager / registry** | `tooling.package = validated_research` | Local `~/.sounio/registry/` only. No public registry. Local package manifests, local package imports, and `tools/sounio-pkg/sounio-pkg` build/check/test smoke are covered by `scripts/ci/sounio_package_support_gate.sh`. |
| **Generic structs/functions/traits** | `generics.* = prototype` | Multi-type-param generic functions work (incl. 3+ params, verified); do not claim a mature trait ecosystem. Trait bounds are parsed but not enforced at call sites. No trait objects. |
| **Linear closures** | `closures.lambdas = validated_research` | Regular closures (capture, HOF, escape) are **implemented and gate-tested** (16/17 `tests/run-pass/closure_*.sio`). **Linear closures** (capturing linear resources, `closure_linear.sio`) are the one open feature — marked `//@ ignore`, tracked separately. |
| **Units of measure** | `units.measure = prototype` | Fixture-backed prototype surface. |
| **Refinement types (general)** | `refinement.types = prototype` | Beta/prototype; runtime fallback dominates non-trivial predicates. |
| **Hypercomplex NN (broad)** | `hypercomplex.nn = prototype` | Research/prototype unless a named gate covers the exact behavior. |
| **Direct-driver support cohort** | `direct_driver.support = validated_research` | Claim only the bounded support cohort checked by `scripts/ci/sounio_direct_driver_support_gate.sh`: 24/24 `tests/selfhost-driver-output/*.sio` fixtures compile to ELF and execute with expected stdout/exit. **NOT PROVED:** large-surface direct-driver execution, ontology-sized semantic truth, wrapper-provenance replacement, native-v2 driver self-compile/fixed-point closure, direct-driver negative-truth restoration, or broad production readiness. |
| **Direct-driver execution at scale** | `direct_driver = prototype` | Large-surface direct-driver execution remains a maturity frontier. The bounded support cohort above does not promote direct-driver semantic authority on ontology-sized or compiler-sized surfaces. |
| **Windows target** | `platform.windows = prototype` | PE/COFF lane wired; not stable. |
| **`binary.source` (modular self-hosted tree)** | `validated_research` | The checked x86-64 Madaros prebuilt is built from `self-hosted/compiler/main.sio` and covered by the named Madaros source-to-ELF/full gates. This claim applies to `bin/madaros-linux-x86_64`, not the legacy `bin/souc-linux-x86_64`; `lean_single.sio` remains the bootstrap seed and escape hatch. |

### Active Known Bugs / Architectural Gaps

**Imported-module native path (D1–D4) — partial closeout (2026-07-14 → Wave10 2026-07-21).** On the default **native** engine, composing real modules historically failed or *silently miscompiled* in four distinct ways. Full catalogue + minimal repros + priority: [`docs/audit/MADAROS_IMPORTED_MODULE_NATIVE_PATH_ESCALATION_2026-07-14.md`](../audit/MADAROS_IMPORTED_MODULE_NATIVE_PATH_ESCALATION_2026-07-14.md). Which stdlib results survive native import: [`docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md`](../audit/EPISTEMIC_TRUST_MAP_2026-07-14.md). Live gate: `scripts/epistemic_trust_gate.sh`.

- **D1 — `f64 → i64/i32` cast on an f64 *parameter* was a bit reinterpretation, not a truncating convert.** ~~OPEN~~ **FIXED (2026-07-19 → Wave10 trust closeout 2026-07-21):** root-caused as general f64-param `scalar_kind` (#983); joint D5+D1 land (#1252 / `fix/madaros-d5-d1-f64-param-cast`); stdlib `dof_to_i64` arithmetic-source + half-up round (prescription-chain). Finite-dof `gum_k95` under native import returns **t95(ν)** (e.g. **2.776** at ν≈4), not the normal 1.960 collapse. Gate: `scripts/epistemic_trust_gate.sh` Section A (`GUM_TRUST_OK` + `witness_gum_k95` → `2776`). **NB:** the pre-Wave10 witness used a Type-B-dominant budget so k95=1.960 was *correct* and could never flip — fixed to Type-A-dominant. Dispatch: `docs/audit/MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST_2026-07-14.md`. Issues #932/#983 **CLOSED**.
- **D2 — `&local_array` passed to a builtin receives a wrong base pointer.** ~~OPEN~~ **FIXED for `write_file` / `str_from_bytes` (2026-07-19/20):** handle-by-value `write_file(path, buf, n)` (#1247) + call-site auto-unwrap of `&buf` / ref-typed slots (#933 residual, `fix/madaros-d2-ref-buf-builtin`) so both shapes share the GC-handle unpack path. `read_file` is 1-arg `path→string` (separate fix #1078) + packed `s[i]` (#1258). Gate: `scripts/native_d2_ref_buf_builtin_gate.sh` (needs current-source Madaros). Dispatch: `docs/audit/DATA_IO_TRILHA_B_BUILTIN_BUFPTR_DISPATCH_2026-07-14.md`. Issue #933.
- **D3 — multi-module native lowering fails** (segfault in `lower_array` dep-lowering, or thin-link `rc=12`). Native compile fails whenever the program's dependency closure has ≥2 modules or a module `use`s another — including `Knowledge<T>`/`epistemic::knowledge`, `epistemic::propagate`, `epistemic::order_spread_exact` (the CPC N=4 receipt). Extends `docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`, `docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md`. Issues #901, #921. *Highest leverage.*
- **D4 — named `use m::sym` + `print_f64` trip E137** in importing programs. `docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md`. Issue #862.

Workarounds for remaining residuals: inline logic into `main()`, keep modules self-contained (no stdlib `use` deps), or run under `lean_single` where multi-module still fragile. Recommended residual order after D1/D2 closeout: **D3 memory-wall / exclusive-ref chains → D4 ergonomics**.

**Multi-module bundle compile — RESOLVED 2026-05-29.** All three G1 architectural roots closed. Bundle: **0 errors** (arc 766 → 0, commits `fcce29dd3` through `8c4f619de`). The modular self-hosted tree (`self-hosted/compiler/main.sio`) now compiles clean. The checked x86-64 Madaros prebuilt is source-built from that modular tree and covered by `scripts/ci/madaros_full_gate.sh` plus `scripts/ci/madaros_source_to_elf_gate.sh`. This is a validated-research source-built Madaros lane, not a claim that `lean_single.sio` has been retired as the bootstrap seed.

Previously tracked G1 roots, all closed:

1. ~~**i64 type-hash overflow on 3-level pointer nesting.**~~ **FIXED** (`fcce29dd3`) — composite-type intern table; nested `&Option<Box<T>>` uses stable interned IDs. Regression: `tests/run-pass/type_hash_3level_nesting.sio` ✓.
2. ~~**SRET for ≥8-field struct returns — bundle path.**~~ **FIXED** — closed by composite-type interning + TUP_CACHE fixes. Also fixed in x86/a64 codegen: global struct-array subscript (`esz>8`) now emits pointer arithmetic (`8a3bfe636`). Regression: `tests/run-pass/sret_8_field_return.sio` ✓.
3. ~~**Nested `&!` struct-field mutation.**~~ **FIXED** (`a40989429`) — `ident.f1[i].f2[j] = v` pattern compiles correctly. Regression: `tests/run-pass/nested_ref_field_array_store.sio` ✓.

See `docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md` §5 / G1 and memory `[[project_closures_shipped_2026-05-27]]` for the full arc.

**Impl-method signature preseed under multi-module summary lowering — FIXED 2026-07-07 (WP-A10).** Importing a module that declares an `impl` (e.g. `impl ExactRing for i64`) SIGSEGV'd the Madaros compiler *at compile time* inside `lower_program_to_ir_summary_box_with_externs_ref` (trace: `lower_array: dep_begin 1` → `module_frontend_lower: summary_begin`, never `summary_done`). Root cause: `lowerer_preseed_fn_signature_mut` (in `self-hosted/ir/lower.sio`) — the *only* caller is the impl-method summary preseed, so it stayed latent until an imported module carried an `impl` — mutated the function slot through the direct nested lvalue `(*(*lo).module).functions[fn_id].field = X`. The 512-byte aggregate store `.param_regs = [IR_INVALID_REG; 64]` hits the documented lean_single two-level-nested-store miscompile: the aggregate lvalue base address is computed wrong and the write faults. Fix: extract the module `Box` + function slot to locals, mutate the local slot, write it back once — the same safe idiom already used by `lowerer_preseed_program_items_mut`'s `ItemFn` branch. Minimal repros (both build+run green post-fix): `tests/run-pass/gen_dep_summary_min.sio` (generic `[F;2048]` dep, no impl → was already OK) and `tests/run-pass/gen_dep_summary_min2.sio` (generic dep **with** trait + `impl for i64` + trait-bounded generics → reproduced the crash, now `GEN_DEP_SUMMARY2_OK`). Single-module + non-generic paths are byte-identical (the fixed function has no other caller). **Does NOT unblock `cd_exact_generic_i64` end-to-end** — see next entry.

**Unsplit full `oct_mul` core_ir fallthrough / main re-entry — FIXED (2026-07-20).** After #1292 (vreg tables 512→2048) a single-file unsplit 8-component exclusive-ref `oct_mul` still compiled but re-entered `main` (`ENTER`/`BEFORE_MUL` storm until stack death). Root cause was **not** wrong call-target/PLT: `IR_MAX_INSTRS=2048` silently dropped body tail + `IrReturn` once the exclusive-ref expansion hit the wall (measured: N=7 → 1910 IR ops PASS; N=8 → 2048 ops, no ret). `compile_ir_function_v2_core_ir_into` only emits epilogue on `IrReturn`, so control fell through into the next function. Fix: `IR_MAX_INSTRS` / `IrFunction.instrs` 2048→4096 (unsplit body ~2182 ops) + synthetic epilogue if no return was seen. Gate: `scripts/madaros_unsplit_oct_mul_gate.sh`. **#1274 lo/hi split kept** in `stdlib/algebra/octonion.sio` as defense-in-depth for older binaries / multi-module import. Residual: doubling per-function IR storage grows `IrModule` further (multi-module body-lowering memory wall still OPEN below); RA still caps at 2048 simple instrs (core_ir path does not use RA).

**Multi-module dependency body-lowering memory wall + downstream segfault — OPEN (blocks `cd_exact_generic_i64` / WP-A5).** With the impl-preseed fix above, `cd_exact_generic_i64` now clears module-1 (`cayley_dickson_exact`) *summary* lowering, but still fails to produce an ELF. Two coupled residuals, both PRE-EXISTING (identical on the base branch for `algebra_g2_invariants_import` and `associator_field_octonion`, neither of which contains an `impl`, so they are unrelated to the impl fix): (1) **Pathological memory.** `IrModule` is large (`functions: [IrFunction; 2048]`, each `instrs: [IrInstr; 4096]` as of the unsplit-oct fix); the multi-module lowerer makes many by-value module/summary copies, so compiling the 4-module cd_exact closure peaks at **~18 GB+ VM**. Under the `bin/madaros` wrapper's `ulimit -v 16 GiB` the allocator fails mid-`lower_program_bodies_ref` and SIGSEGVs while lowering the 5th generic `[F;2048]`-returning body (`cd_sub_exact`, structurally identical to the `cd_add_exact` that lowered fine one step earlier — i.e. accumulation, not a per-statement defect). (2) **Downstream crash.** Running the raw `madaros.elf` with NO ulimit (node has 93 GB free, so 18 GB is not an OS OOM) advances past module-1 body lowering + merge and then SIGSEGVs at `lower_array: dep_begin 2` (module-2 summary lowering) — a genuine fault, not memory pressure. Fixing this requires shrinking `IrModule`/eliminating by-value module copies in the multi-module path and/or resolving the module-2 summary fault; it is distinct from both A8 (SRET forwarding) and A10 (impl-preseed). `cd_exact_generic_i64` remains NOT verified on Madaros.

**`cd_exact_generic_i64` — GREEN on Madaros 2026-07-07 (WP-A5).** Build now succeeds (the memory wall above is cleared by building the raw madaros ELF under `ulimit -v unlimited` instead of the wrapper's `ulimit -v 16 GiB`; the 18 GB VM peak is data, not a compiler bug). With the build unblocked, the last *runtime* SIGSEGV was traced (Slurm + capstone disasm of the emitted ELF) to a **transitive cross-module call drop**: generic `cd_mul_exact` calls `cd_sigma` via `use algebra::cayley_dickson::{cd_sigma}`, and on the Madaros imported-lane a direct call whose target lives in a *transitively-imported* module is silently elided — the `IrCall` never emits, and its result vreg defaults to 0, clobbering parameter slot 0 (`a`), so the next `a.c[i]` dereferences a null handle inside the nested accumulation loop → SIGSEGV. Same-module callees (`cd_zero_exact`, the `er_*` trait methods) resolve correctly; only different-module targets drop. **Fix (stdlib workaround):** define `cd_sigma_x` same-module in `cayley_dickson_exact.sio` (verbatim copy) and call it — exactly the pattern the concrete sibling `cayley_dickson_exact_i64.sio` already uses (`cd_sigma_exact_i64`, whose header documents the `use cd_sigma` "HARD BLOCKER"). Result: `ZD PROVED` / `SQ PASS` / `NONZERO PASS` / 16×`COMP i 0`, rc=0. Stdlib-only; compiler untouched.

**Transitive cross-module call drop (A14) — FIXED (2026-07-07).** The residual compiler defect behind the A5 stdlib workaround is now root-caused and fixed in `module_frontend.sio`; the `cd_sigma_x` workaround has been **reverted** (`cd_mul_exact` again calls the transitively-imported `cd_sigma`). Root cause: the merged-IR call-target canonicalization (`ir_module_compact_duplicate_fn_refs` + the `ir_module_finalize_merged_calls` resolve loop) rebound a call's `fn_id` with a whole-`IrInstr` writeback (`var ins=slot; ins.fn_id=X; slot=ins`). `IrInstr` carries a `Box` (`call_args`); lean_single miscompiles that by-value copy and ZEROES the slot (`op/dst/fn → 0`), so a transitive call from a non-first function was elided (result vreg 0 aliases param slot 0 → null-handle deref → SIGSEGV, or a wrong result). Rewriting to a nested scalar store (`out.functions[fi].instrs[ii].fn_id=X`, direct or via `&!` pointer) does NOT work either — lean_single silently DROPS 3-level nested stores, leaving `fn_id` stale. Fix: rebind per WHOLE FUNCTION using the merge-append idiom (by-value `IrFunction` copy, 2-level owned `fn_id` write, 1-level array writeback), which both persists and avoids the Box-carrying `IrInstr` copy. Witnesses (Slurm, actual rc): `a13_crossmod_nonfirst_fn_drop_ctrlA/ctrlC` 139→0; new `a14_transitive_min` 0→115; `cd_exact_generic_i64` (now via transitive `cd_sigma`) 139→0 `ZD PROVED`/`SQ PASS`/`NONZERO PASS`/16×`COMP i 0`; `cd_exact_generic_vs_concrete` 139→0. Net `module_frontend.sio` +231 bytes (8MB import budget intact).

**Multi-module arena reset blocked by live `call_args` (memory-wall residual) — FIXED (2026-07-21).** Per-module `__arena_reset` after multi-mod merge (a096d1c / #719) could not reclaim dependency windows that carried live `IrInstr.call_args: Option<Box<IrRegList>>` (shallow pointer into the dep Box-arena). The intended BSS re-box path was disabled post-#832 (`dep_arena_can_reset = scan_ok && MF_AREBOX_COUNT == 0`) after heads appeared corrupted. Root cause: `mf_arebox_apply` wrote `(*module).functions[fi].instrs[ii].call_args = rebuilt` — a 3-level nested store through Box that lean_single **silently drops** (same family as A14), so post-reset the shallow (now dangling) pointers remained. **Fix:** re-enable re-box for live sites; extract via module indices (no Option-by-value helper); apply with the A14 whole-function idiom (owned `IrFunction`, 2-level `call_args` store, 1-level writeback). Do not reintroduce A8 by-value `IrInstr`. Loud metrics: `arena_reset_ok … rebox_sites N` / `arena_reset_totals ok=… skip=… sites_reclaimed=…`; skip only on scratch overflow. Measured (current Madaros): dual gum+knowledge 2→0 skips, **260** sites reclaimed, `DUAL_GUM_KNOWLEDGE_OK`; SPECIAL erf/gamma 1→0 skips, **20** sites each, `SPECIAL_*_GATE_OK`. Overflow skip path kept. Does not by itself close the larger `IrModule` footprint wall above.

**Array-field struct by-value return (SRET) — NOT a limitation; the residual was `println` dispatch.** 2026-07-06 investigation (WP-A4): returning a struct that contains an array field *by value* on the default Madaros engine works — structs are handle-based (one heap object; the array field is a separate handle stored in a slot), so the returned handle round-trips correctly at any width. Verified across `struct{c:[i64;2],x}`, `struct{c:[i64;4],bits}`, the generic `struct G4<F>{c:[F;4],bits}` @`i64`, a 3-arg callee (asymmetric args), and `[i64;256]`. The actual cause of `tests/run-pass/generic_struct_return.sio` rc=139 was the **`println` builtin dispatch**: `expr_result_scalar_kind_ref` (in `self-hosted/ir/lower.sio`) had no `ExprKind::ExprIndex` case, so an i64 array element such as `r.c[0]` classified as kind 0 → routed to the char\* printer, which dereferenced the integer value as a pointer and SIGSEGV'd. (Monomorphized generic structs register fields from the generic declaration `[F;N]`, so a declared-type i64 marker cannot fire for them.) Fix: classify a subscript's element positively as float (matching the existing `index_base_elem_is_float_ref` path → `print_f64`) and default the remaining numeric element to int (`print_int`); float-element index-println tests (`bdf_stiff`, `dissertation_pbpk28_*`, `hof_mut_struct_min`, `ode_generic_solver`) are unaffected. Regression tests: `tests/run-pass/generic_struct_return.sio`, `tests/run-pass/println_int_array_field.sio`, `tests/run-pass/sret_array4_return.sio`, `tests/run-pass/sret_array4_generic_return.sio`, `tests/run-pass/sret_array_args_return.sio`. Residual (tracked in the continuity new-gap ledger, NOT this fix): `println(<computed local>)` where the local is bound from a bare arithmetic/index initializer without an int scalar-kind marker still routes to the char\* printer.

**Cross-module large-struct SRET forwarding (A8) — FIXED (2026-07-06).** A dependency-module
function that forwards through an inner same-module struct-returning call — the shape
`var r = zero(k); r.c[i] = 1; return r` — used to RUN-segfault (rc=139) when reached across a
module boundary (`tests/run-pass/sret_forwarding_cross_module_min.sio`, `a8_diag_fwd/sizes/step.sio`),
while the byte-identical single-module source ran clean. **Root cause (corrects the earlier
body-lowering hypothesis):** the merged-IR body lowering is *correct* (verified pre-finalize:
`Call dst=2 arg=[0] fn=<zero>`). The corruption was introduced in `ir_module_finalize_merged_calls`
(`self-hosted/compiler/module_frontend.sio`): its final call-target-resolution pass called
`ir_module_resolve_one_call_target(&out, ins)` passing the whole `IrInstr` **by value**. `IrInstr`
carries a `Box` (`call_args`); under lean_single the by-value large-struct copy is miscompiled and
scrambles the *caller's* `ins` local (`dst 2→3`, `src1 0→2`, `call_args [0]→[2]`, `fn_id`→wrong),
which was then written back unconditionally — so `r.c[i]` / `return r` dereferenced a garbage handle.
(`ir_module_compact_duplicate_fn_refs` used the same read/writeback but never passed `ins` by value,
which is exactly why it was safe.) **Fix:** resolve from scalar fields only —
`ir_module_resolve_call_target_fields(module, old_id: i64, name: Name)` (Name is a small fixed struct,
passed by value safely throughout) — and write the slot back only when `fn_id` actually changes.
Witnesses (Slurm, actual rc): min → rc=0 `CROSS_SRET_MIN_OK`; cd_mul → rc=0 `CD_MUL_CROSS_SRET_OK`;
a8_diag_fwd/sizes/step/ctrl → rc=0; `fano_basics` FAIL→PASS. Zero regressions (base-vs-fix
differential across ~20 multi-module tests: identical-or-better). **Still open (separate, pre-existing —
NOT this fix):** `cd_exact_generic_i64` (A5 headline) SIGSEGVs at *compile time* in
`lower_program_to_ir_summary_box_with_externs_ref` for its generic dependency module (trace stops
between `module_frontend_lower: summary_begin` and `summary_done` on `lower_array: dep_begin 1`);
EISA `test_eisa_isa/evm` fail earlier at "multimodule native thin-link compilation failed" (brc=1).
Both reproduce identically on the base branch.

### Reconciliation notes (2026-05-27)

This file previously claimed several rows as "Production" that the public-claim registry already had downgraded. The PL adoption audit (`docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md` §2) catalogued the diff. Changes made here:

- **Removed:** the row claiming `Formatter Production`. No `souc format` subcommand exists; no `tools/fmt*` binary exists.
- **Removed:** the "No active known bugs" assertion. Bundle baseline is 766 errors with three named roots.
- **Removed:** `CLI: check/build/run/repl/format/doc` — was wrong about `repl` and `format` (both now wired 2026-05-28, see G5a/G5b rows above). `doc` remains unimplemented in this checkout.
- **Added 2026-05-28:** G5a formatter shipped (`tools/fmt/sounio-fmt.sh`, `souc format` subcommand). See formatter row above for Phase 1 scope and Phase 2 deferrals.
- **Added 2026-05-28:** G5b REPL shipped (`souc repl` → `tools/repl.sh`, souc-native eval loop).
- **Added 2026-05-28:** G6 public install path — `scripts/install.sh` + `scripts/release.sh` stage compiler + stdlib + launcher into arbitrary `--prefix`. Discriminator: `bash scripts/install.sh --prefix=/tmp/t && /tmp/t/bin/souc --version`.
- **Downgraded:** LLVM Codegen from `Production` to `Validated research` — the LLVM bridge is wired but disabled in the checked binary.
- **Reframed:** LSP, REPL, Package Manager to mirror the registry's `prototype` tier.
- **Added:** "Active Known Bugs / Architectural Gaps" section enumerating the three bundle-compile roots.
- **Fixed:** CLI exit-code contract — `souc check` now propagates typecheck failures (G2). Probed before/after in the audit. The wrapper `bin/souc` had `exit 0` in each dispatch arm; replaced with explicit `exit "$_rc"`.

### lambda-spec-reconciliation (RESOLVED 2026-05-27)

`closures.lambdas` was previously marked `stale_conflicting`. Probe on 2026-05-27 confirmed that lambda literals (capture, HOF, escape, arity-2) compile and run correctly — 16/17 `tests/run-pass/closure_*.sio` pass. The registry is updated to `validated_research/closed`. Spec §4.7 is now fully normative. The one open item is linear closures (`closure_linear.sio`, `//@ ignore`).

### Fixed in Self-Hosted Compiler — All Bugs Closed

**`extern "C"` integer FFI return register** (fixed): `strip_extern_blocks()` now emits Sounio stub functions (OS syscalls for integer-returning `getpid`/`getppid`, `heap_alloc`/`heap_free` for `malloc`/`free`, `__native_*_f64` intrinsics for math). Stubs use Sounio's internal calling convention (RAX), bypassing the XMM0/RAX confusion entirely. Unblocks `stdlib/os/`, `stdlib/mem/`, `stdlib/sync/`. Regression test: `tests/run-pass/ffi_integer_return.sio`.

**Observation boundary coverage** (fixed): `Observe` now enforced for comparison, IO-arg, FFI-arg, and pattern-match scrutinee in both x86-64 and ARM64 codepaths. Self-hosted compiler and multi-file checker are now aligned. Test: `tests/compile-fail/observe_io_boundary.sio`.

### Fixed in Self-Hosted Compiler (live in current binary)

The following bugs were fixed in `lean_single.sio` and are active in the current `bin/souc-native` (rebuilt 2026-04-20):

**Mixed-Hyper optimizer metadata** (fixed): When a function mixes Hyper algebras (2+ distinct algebra kinds in its type signature), `checker_infer_fn_hyper_algebra` now computes the most-restrictive algebra kind (intersection of rule sets) instead of bailing with -1. `ocp_configure_small_context` applies the appropriate conservative reassoc strategy for that kind: free(0) for Real/Complex/Quaternion, fano_selective(2) for Octonion, blocked(1) for Sedenion/Clifford. Additionally, when a function's `hyper_algebra_kind` is -1 (tag lost at lowering) but the compilation unit has a single unambiguous algebra declaration, `ocp_infer_algebra_from_table` re-infers the kind from the registry entry so homogeneous helper functions benefit from algebra-specific reassociation. Also fixed: Octonion (kind=3) incorrectly defaulted to strategy=1 (blocked) in the fallback path; now correctly uses strategy=2 (fano_selective). Multi-algebra intersection remains a TODO (`// TODO: mixed algebra intersection` in `ocp_infer_algebra_from_table`).

**`&![T; N]` mutable ref mutation — bare array index** (fixed): When passing a bare array variable by `&!` reference, mutations via `arr[i] = v` (bare index, without explicit deref) are now correctly written back through the pointer for all element sizes. Root cause: the parameter registration in the codegen did not set `VAR_ESIZ` for `&![T; N]` fixed-size array ref parameters, so the element stride defaulted to 8 regardless of the actual element type. For `&![i64; N]` this happened to work (stride-8 is correct), but for `&![i8; N]` the stride was wrong, causing memory corruption. Fix: after `var_add` registers the parameter slot, a new branch detects `SCAN_TY == 10` with inner type `8` and sets `VAR_ESIZ = arr_hash_esiz(ref_hash_inner_hash(SCAN_TY_HASH))`. Regression test: `tests/run-pass/array_mut_ref_bare.sio`.

**Implicit `var`/`let` with `i32` type** (fixed): Integer literal narrowing now allows `var x: i32 = 5` without "expected I32, found I64" errors. Literals are compatible with annotated smaller integer types (i32, i8).

**`Option::None` type inference** (fixed): Bidirectional type inference now propagates the expected type for enum variant paths. `let x: Option<i32> = Option::None` correctly infers `Option<i32>`.

**Unit type declarations** (fixed): The resolver now registers `unit` declarations as `SymUnit` (was incorrectly using `SymTypeAlias`).

**String methods** (fixed): `.as_bytes()` returns the string as a byte array (works). `.len()` on `string` now emits a runtime null-terminated byte count (x86-64 and ARM64); previously the condition missed `EXPR_TY == 3` and leaked the string pointer as the length. Regression test: `tests/run-pass/string_len.sio`.

**Turbofish + generic monomorphization** (working): Multi-type-parameter generic functions are monomorphised and execute correctly, including **3+ type parameters** — `func::<T>(args)`, `func::<T, U>(args)`, and `func::<A, B, C>(args)` are all supported (verified: `fn trip<A,B,C>(a,b,c)->C` with `trip::<i64,i64,i64>(1,2,7)` compiles and returns 7). The `<TPARAMS>` section is stripped from the specialised token copy, every type parameter is substituted, and the specialised function is compiled as an ordinary function.

**Range slice half-open syntax** (fixed): `&arr[..n]` (start omitted, defaults to 0) now correctly compiles. Previously `compile_primary()` consumed the `..` token as an unrecognised primary, causing both the range-check and base-check to fail. Fix: detect `..`/`..=` at the start of the slice index and emit start=0 directly.

**String `.as_bytes()`** (fixed): `.as_bytes()` on a `string` is now a recognised builtin — it passes through as a no-op (string pointer unchanged, type stays `string`), making `&bytes[..n]` range slices work on the result. Previously the method fell through to field-access dispatch, producing type 0 and causing the slice borrow to segfault.

**Trait definitions** (added): `trait Name { fn method(); ... }` syntax is now parsed and trait definitions are collected into the `TraitRegistry`. Builtin trait implementations (Copy, Drop, Eq, Ord, Hash, Add, Sub, Mul, Div, Display, Debug) are pre-registered for primitive types.

**`&string[..n]` slice borrow** (fixed): String variables are now accepted as slice borrow bases in `&bytes[..n]`. Element size is 1 byte, runtime length is computed via `strlen`. Result type is `&[i8]`. Previously produced "slice borrow requires array or slice base" warning and a null-pointer segfault.

**Borrow release at call boundaries** (fixed): Borrows taken for function call arguments are now unconditionally released after the call returns, fixing false positive errors on consecutive calls borrowing the same variable.

**`(*ptr).field = value` store through explicit deref** (fixed): Explicit pointer dereference field assignment (`(*c).field = v` where `c: &! S`) was silently a no-op in the JIT — mutations were lost. The LHS deref-then-field store path was only recognising raw pointer type (ty==11) and rejecting `&!T` exclusive references (ty==10). Fix: both type 10 (`&!T`) and type 11 (`*T`) are now accepted; inner type and field offset lookup uses the shared `ptr_hash_inner_ty`/`ptr_hash_inner_hash` helpers which work identically for both. Test: `tests/run-pass/explicit_deref_field.sio`.

**Ownership state machine** (wired): The `OwnContext` ownership tracker (2836 lines, 72+ functions) is now integrated into the `Checker` — linear variable registration, ownership transfer on use, and linear-at-end checking at function exit.

**Effect propagation** (verified): Call-site effect checking (`check_callee_effects`) validates that callee effects are a subset of the caller's declared effects, reporting E035 on violations.

### Strict Numerical Regressions and Mathematical Rigor Policy

Numerical regressions in pharmacokinetics, GUM uncertainty propagation, and clinical pathways are strictly monitored. It is an absolute policy that regression tolerances must never be loosened or "afrouxadas" through artificial modifications simply to make tests pass. If a physical model test or mathematical/clinical verification fails, the underlying compiler code or the physical model itself must be fixed honestly.

### Pruned/Experimental Modules

The following stdlib modules are stubs or incomplete:

- `stdlib/gpu/` - requires CUDA runtime (behind `--features gpu`)
- `stdlib/crypto/` - pure-Sounio sha256/hmac/rng are active; random.sio.disabled and hash.sio.disabled require additional algorithm work
- `stdlib/compress/` - gzip.sio requires libz at link time; zstd.sio requires libzstd at link time (external runtime libraries, not an FFI limitation)
- `stdlib/ffi/` - stub
- `stdlib/autodiff/` - framework only
- `stdlib/interop/` - stub
- `stdlib/text/*.sio.disabled`, `stdlib/time/*.sio.disabled` - old Rust-style stubs (use `u32`/`u64`/closures/`for..in`); superseded by pure-Sounio rewrites already active as `.sio` files

### Recently Activated Modules

- `stdlib/text/format.sio` - `format_int(i64) → string`, `format_f64(f64) → string` (4 decimal places); uses str_concat+str_slice, no heap. Smoke test: `tests/run-pass/stdlib_time_basic.sio`.
- `stdlib/text/case.sio` - char/string case conversion (uppercase, lowercase, titlecase, snake_case, camelCase, PascalCase, kebab-case); pure Sounio, no FFI.
- `stdlib/text/unicode.sio` - Unicode character classification (alphabetic, numeric, whitespace, punctuation, control, ASCII variants); pure Sounio.
- `stdlib/time/duration.sio` - `Duration` struct with nanosecond precision; arithmetic: dur_add, dur_sub, dur_from_millis, dur_to_millis; pure Sounio, no FFI.
- `stdlib/time/datetime.sio` - `DateTime` struct with full calendar arithmetic (leap year, days-in-month, unix epoch roundtrip, year rollover); pure Sounio, no FFI. Smoke test: `tests/run-pass/stdlib_time_basic.sio`.
- `stdlib/time/instant.sio` - Monotonic clock via `clock_gettime` syscall; uses integer FFI (now working).
- `stdlib/os/process.sio` - getpid/getppid/exit/abort via extern "C" stubs (integer FFI now works)
- `stdlib/mem/` - heap_alloc/heap_free (malloc/free stubs), arena bump allocator, box/rc/arc wrappers — all active
- `stdlib/sync/mutex.sio` - pthread_mutex_{init,lock,trylock,unlock,destroy} via extern "C" stubs
- `stdlib/prob/` - Beta, Normal, MCMC, random distributions (4 modules activated)
- `stdlib/onn/` - Octonion neural network: activation, attention, conv, linear, loss, normalization, optimizer, training (8 modules)
- `stdlib/ontology/` - LOINC, biomedical module, namespaces (3 modules)
- `stdlib/compress/deflate.sio` - stored-block DEFLATE only (RFC 1951 BTYPE=00, no compression); gzip/zstd modules still require integer FFI
- `stdlib/heliobiology/units.sio` - space weather units
- `stdlib/ode/tsit5_multicomp.sio` - multi-compartment adaptive Tsit5 solver
- `stdlib/medlang/` - full MedLang DSL (lexer, parser, AST, codegen, PK models, population, dosing) — all active

### Optional External Dependencies

| Feature | Dependency | Effect if Missing |
|---------|------------|-------------------|
| `--features llvm` | LLVM 18 (`libLLVM-18.so`) | `--backend llvm` and `--emit-llvm` active; install `llvm-18-dev` + `clang-18` |
| `--features smt` | Z3 + cmake | Without Z3: static engine handles constants/narrowing/monotonicity; QF_LIA Fourier-Motzkin tier (`smt_qflia.sio`) sits between static analysis and runtime fallback; complex predicates beyond FM fall back to runtime checks with W040 |
| `--features gpu` | CUDA toolkit | GPU codegen works, execution requires runtime |

### Platform Support

- **Linux x86-64**: Primary supported platform (default)
- **Linux aarch64**: Supported via `--target aarch64-linux`
- **macOS x86-64**: Mach-O backend (2,512 lines) wired; cross-compile via `--target x86_64-macos`
- **macOS ARM64**: Mach-O ARM64 backend wired; cross-compile via `--target aarch64-macos`
- **Windows x86-64**: PE/COFF backend (3,508 lines) wired; cross-compile via `--target x86_64-windows`. No pre-built .exe shipped in this checkout.

Cross-compiled binaries must be executed on the target OS. The compiler runs on Linux and emits the correct binary format for each target.

---

## Legacy bootstrap seed (`lean_single.sio`)

**Status:** bootstrap seed and escape hatch. Not a bug; a maturity-stage reality that contributors must know about before editing compiler logic.

### What the situation actually is

The preserved bootstrap compiler binary (`bin/souc-linux-x86_64`, also available through `SOUNIO_SOUC_ENGINE=lean_single`) is produced from a **single self-hosted source file**:

- `self-hosted/compiler/lean_single.sio`

The modular directory layout most readers expect —

- `self-hosted/lexer/`
- `self-hosted/parser/`
- `self-hosted/check/`
- `self-hosted/types/`
- `self-hosted/ir/`
- `self-hosted/native/`

— is now the source for the checked x86-64 Madaros prebuilt (`bin/madaros-linux-x86_64`). The 2-stage bootstrap recipe below remains the legacy seed/escape-hatch path and uses `lean_single.sio` exclusively:

```bash
./bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio /tmp/souc-stage1
/tmp/souc-stage1 self-hosted/compiler/lean_single.sio /tmp/souc-stage2
cp /tmp/souc-stage2 bin/souc-linux-x86_64
```

### Implication for contributors

Changes to the default user-facing compiler path must land in the modular tree and be proven through the Madaros gates before refreshing `bin/madaros-linux-x86_64`. Changes needed for the legacy seed or explicit `SOUNIO_SOUC_ENGINE=lean_single` path still need the corresponding `lean_single.sio` update.

Examples of this pattern in recent history:

- 2026-04-20 — surgical type gates (`ExactlyPrivate`, `Editable`, `CapabilityGated`) and error codes `E201`–`E203` added to `lean_single.sio`; modular files updated in parallel.
- 2026-04-29 — extended surgical type gates (`Composable`, `Audited`, `Revivable`, `Interpretable`), new effect bit-flags (`Witness=32768`, `Temporal=65536`, `Learn=131072`), and error codes `E204`–`E207` added to `lean_single.sio`; 2-stage bootstrap executed; `bin/souc-linux-x86_64` rebuilt.

### Risk of silent divergence

Because the modular compiler and legacy seed are no longer the same source file, reviewers should check which lane a PR affects. Default Madaros changes need the modular-source build plus named Madaros gates; legacy-seed changes still need a `lean_single.sio` bootstrap proof.

### Planned resolution

1. **Parity harness (`tests/parity/`, planned near-term).** For a fixed set of `.sio` programs drawn from `examples/` and `tests/compile-fail/`, compile via both paths and diff the stdout/stderr and exit codes (not the binaries — timestamps and symbol ordering make binary-equality unreliable). Divergence flips CI red.
2. **Bootstrap retirement (roadmap, long term).** Retire `lean_single.sio` as an escape hatch once the modular compiler has enough fixed-point and parity evidence.

Until that lands, treat the modular tree as the source for the default Madaros prebuilt and `lean_single.sio` as the seed/escape-hatch source.

---

## Syntax Limitations - All Resolved

This section documents previously-resolved limitations for historical context.

## Syntax - All Resolved

### Module System
- **Status**: Resolved (v0.99.0)
- **Resolution**: Full `module`/`use` support with file-based module loading and hierarchical namespace resolution.

### Visibility Modifiers
- **Status**: Resolved (v0.99.0)
- **Resolution**: `pub` visibility supported and enforced across module boundaries.

### Logical Operators
- **Status**: Resolved (v0.66.0)
- **Resolution**: `&&` and `||` implemented with short-circuit evaluation and boolean type checking.
```sio
if a > 0 && b > 0 { ... }
if is_empty || is_null { ... }
```

### Documentation Comments
- **Status**: Resolved (v0.99.0)
- **Resolution**: `///` outer docs and `//!` inner docs are parsed and preserved through AST → HIR.

### Numeric Literals
- **Status**: Resolved (v0.99.0)
- **Resolution**: Scientific notation supported in the lexer (e.g., `1e10`, `1.5e-3`).

## Type System - All Resolved

### Type Aliases
- **Status**: Resolved (v0.99.0)
- **Resolution**: `type` aliases are supported, including generic aliases; aliases expand transparently during type checking.
```sio
type Vec2 = (f64, f64)
```

### Unit Definitions
- **Status**: Resolved (v0.99.0)
- **Resolution**: User-defined units are supported and integrate with unit checking.
```sio
unit kg;
unit mg = 0.001 * kg;
unit velocity = m / s;
```

## Reserved Keywords

The following identifiers are reserved and used by the language:
- `var` - mutable binding
- `effect` - effect declaration
- `type` - type alias definition
- `module` - module declaration
- `use` - module import
- `pub` - public visibility modifier
- `unit` - unit definition

## Scoping Behavior - All Resolved

### Variable Shadowing
- **Status**: Resolved (v0.99.0)
- **Resolution**: Shadowing works correctly across nested scopes.

### Forward Declarations
- **Status**: Resolved (v0.99.0)
- **Resolution**: 2-pass resolver enables forward references and mutual recursion.

## Feature Resolution Summary

All previously planned features are implemented as of v0.99.0:

| Feature | Resolved In | Resolution |
|---------|------------|------------|
| Module system | v0.99.0 | File-based module loading with `module`/`use` |
| `&&` / `\|\|` operators | v0.66.0 | Short-circuit logical operators |
| `pub` visibility | v0.99.0 | Visibility enforcement across modules |
| Scientific notation | v0.99.0 | Lexer supports `1e10`, `1.5e-3` |
| Type aliases | v0.99.0 | `type Name = Type;` with generics |
| Doc comments | v0.99.0 | `///` + `//!` parsed and preserved |
| Variable shadowing | v0.99.0 | Correct scoping rules |
| Forward declarations | v0.99.0 | 2-pass resolver |
| Unit definitions | v0.99.0 | User-defined units + checking |

## Hessian AD Capabilities and Architectural Limits (β⁷)

`hessian_of(expr, j, k)` computes ∂²expr/∂xⱼ∂xₖ via second-order forward-mode AD.

### What Works

- **8 function inputs** (channels 0–7): indices 0–7 from `measure()` calls, 36 upper-triangular pairs
- **Arithmetic**: `+`, `−`, `*`, `/` propagate full Hessian and first-order sensitivities
- **Transcendentals (unary)**: `sqrt`, `exp`, `ln`/`log`, `sin`, `cos`, `tan`, `atan`, `tanh`, `asin`, `acos` — full chain rule f′ and f″ in all 8 channels
- **Two-arg builtins**: `atan2(y,x)` and `pow(x,y)` — full Hessian propagation for channels 0–3 and 10 pairs

### Architectural Limitations (Tier 4 — Not Planned for Near-Term)

- **Inter-procedural**: Hessian shadows do not cross user-defined function call boundaries. Workaround: inline the computation.
- **Loop accumulation**: Hessian state resets between loop iterations; only the final body is live.
- **Branch merging**: `if/else` branches do not merge Hessian state (no phi nodes for shadow slots).
- **Channels 4–7 in transcendentals**: Transcendental chain rule only propagates channels 0–3. Channels 4–7 are zero for transcendental outputs even if the input has active sensitivity there.
- **Two-arg builtins (channels 4–7)**: `atan2`/`pow` handlers propagate channels 0–3 only.

### Channel-at-`.value` semantics (resolves former "Butterfly #3")

Phase 5 re-evaluation: the MEAS_KNOW_IDX counter at `lean_single.sio:393` is incremented on every `.value` access to a Knowledge variable.  Channels are assigned **at `.value` extraction time, not at `measure()` time**.  A Knowledge struct at rest has no channel identity; it acquires one only when the user extracts `.value`.

This means the KAS-1 pattern (extract `.value` first, do scalar arithmetic) is **not a workaround** for a compiler limitation — it is the direct expression of the channel-assignment semantics.  Formalised in `formal/ChannelAssignmentSemantics.lean` (Phase 5 Lean file).

`compile_knowledge_muldiv_x86` at `lean_single.sio:5766` correctly does not touch `MEAS_KNOW_IDX`; Knowledge multiplication is channel-silent.  Attempting `hessian_of((k1 * k2).value, 0, 1)` asks for `∂²/∂x_0∂x_1` of a one-input function (the single `.value` access seeds only channel 0); the result is zero by correctness of the channel-at-`.value` model, not by any bug.

**The KAS-1 pattern (formalised in `formal/KnowledgeArithmeticSoundness.lean` + `formal/ChannelAssignmentSemantics.lean`)** expresses a multi-input Hessian function directly under the channel-at-`.value` semantics:

```sio
// Two-input Hessian function f(x, y) = x * y:
let k1: Knowledge<f64> = measure(2.0, uncertainty: 0.1)
let k2: Knowledge<f64> = measure(3.0, uncertainty: 0.1)
let x = k1.value          // seeds channel 0 with 1.0, channel 1 with 0.0
let y = k2.value          // seeds channel 1 with 1.0, channel 0 with 0.0
let z = x * y             // scalar; shadows propagate via product rule
let j: [f64; 8] = [sensitivity_of(z, 0), sensitivity_of(z, 1), ...]
let h: [f64; 36] = [hessian_of(z, 0, 0), hessian_of(z, 0, 1), ...]
let v2 = gum_second_order_variance(j, h, &sigma)
```

Phase 5 attempted to "close the butterfly" at the compiler level (commit reverted — `self-hosted/compiler/lean_single.sio` unchanged).  The attempt added 44 cross-function shadow-bridging globals and product-rule emission inside `compile_knowledge_muldiv_x86`.  It correctly set `EXPR_SSHADOW` before the function returned, but the downstream `.value` access re-seeded channel 0 via MEAS_KNOW_IDX — overwriting the propagated shadow.  The lesson: under channel-at-`.value` semantics, there is no butterfly to close.  `tests/run-pass/knowledge_kas1_policy.sio` remains as a demonstration of the two paths; the "butterfly" path correctly returns zero under the model.

## Native linking / FFI (open)

- **Dynamic `.so` linking / GOT–PLT**: The native toolchain emits statically linked executables on the default bring-up path. Relocation metadata records `R_X86_64_PLT32` for ET_REL objects (`self-hosted/native/reloc.sio`), but wiring arbitrary shared-library symbols end-to-end (libc-style `-lfoo`, full GOT/PLT for external calls) remains future work. FFI tests that require `libzstd` stay behind `//@ ignore` until dynamic link and stable stubs land consistently.
- **`*const u8` in callee position**: Passing `expr as *const u8` can still produce arity/type mismatch diagnostics versus `*mut u8` in some FFI-heavy shapes; stdlib wrappers prefer `*mut u8` until call lowering treats `*const T` and `*mut T` uniformly at the invocation site. See the header comment in `tests/stdlib/compress/test_zstd_e2e.sio`.

## Zero-event native-v2 frontier (open)

The zero-event receipt layer is checkable, its receipt witness now executes on
default Madaros, and constructor opacity is enforced by both `check` and
`compile`. Two neighboring native limitations remain:

1. **Minimal `qd128` import fails closed during native emission.**
   `tests/known_failures/qd128_import_native_v2_probe.sio` now completes
   imported-module lowering without a segmentation fault, then reports backend
   `rc=12`. The neighboring `dd64` control executes successfully.
2. **Minimal sedenion import executes but does not prove its semantic marker.**
   `tests/known_failures/sedenion_import_native_v2_probe.sio` reaches
   `Compilation successful!`; the generated executable exits `1`, without a
   segmentation fault and without `SEDENION_IMPORT_NATIVE_V2 PASS`.

Constructor privacy was closed by running the same visibility preflight used
by `check` before the canonical native `compile` path. The receipt and erased
constructors are compile-fail tests with E176. Direct reads of private fields
remain a separate language-wide visibility limitation and are not treated as
constructor authority.

The classified compiler handoff, root-cause boundary, and required acceptance
matrix are recorded in
`docs/handoff/zero_event_native_compile_privacy_2026-07-11.md`. In particular,
globally enabling the merged checker's visibility bit is not an accepted fix:
the native path must preserve legitimate import edges and cover the generic
specialization branch as well as the ordinary fallback.

Reproduce the classified matrix with:

```bash
bash scripts/ci/zero_event_native_v2_matrix.sh
bash scripts/ci/zero_event_gate.sh
```

Do not promote the known-failure probes to `run-pass` until the default engine
executes the same markers without `SOUNIO_SOUC_ENGINE=lean_single`. Do not alter
the semantic oracles while repairing compiler lowering, aggregate return, or
privacy enforcement.

## Reporting Issues

If you encounter any new issues, please report them at:
https://github.com/sounio-lang/sounio/issues
