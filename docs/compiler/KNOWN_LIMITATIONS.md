<!-- docs:meta
topic_id: repo.docs.compiler.known-limitations
authority: repo_only
audience: contributors
last_validated: 2026-09-11
validated_by: codex-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.known-limitations
-->

# Known Limitations — open-item ledger

This file is a **ledger of open defects** in the Sounio language
implementation. One entry per open item; each entry names the engine it
applies to, a reproduction, the gate or ratchet that pins the current
behaviour, and the rung of the closure program (`KL-N`) that owns it. When a
rung closes an item its row is deleted here and its reproduction moves to
`tests/run-pass/` or `tests/compile-fail/` under a named gate.

Rewritten as a ledger on 2026-09-11 (KL-0) against `origin/main` `3868c1805`.
What was here before — the closed-item narrative, D1–D6, G1, A8/A14, the
Hessian and zero-event histories, the f128 ladder log — is archived verbatim
in `docs/audit/KNOWN_LIMITATIONS_HISTORY_2026-09.md`. Maturity tiers moved to
`docs/compiler/MATURITY.md` (registry-reconciled; the token gates read that
file now). The bootstrap-seed policy and the #1494 record moved to
`docs/compiler/BOOTSTRAP_SEED.md`.

Conventions. **Engine** is `madaros` (default, `bin/madaros-linux-x86_64`
built from `self-hosted/compiler/main.sio`), `lean_single` (seed,
`self-hosted/compiler/lean_single.sio`, `SOUNIO_SOUC_ENGINE=lean_single`) or
`both`. An unqualified claim anywhere in the tree must hold for both. A row
whose pin is `none` has no executable pin yet; the owning rung adds one before
it fixes anything. Line numbers are as measured at `3868c1805`.

## Rung map

| Rung | Scope | Engine |
|---|---|---|
| KL-9 | seed: #1494 imported-module typecheck errors non-fatal | lean_single |
| KL-11 | #1792 first-order / variance across user calls (pow FO closed) | madaros |
| KL-14 | FFI: 14a–14d3 CLOSED | madaros |
| KL-15 | `f256` surface (15a softfloat add/sub partial), `Knowledge<f128>`/GUM | madaros |
| KL-16 | Hessian Tier-4 (16a–16f CLOSED; residual H-multi/non-H00 if/a64 atan2) | lean_single |
| KL-17 | generic `impl` blocks: associated-call and literal `T` inference, raw-word `HeapVec<T>` | madaros |

## Ledger



### KL-9 — seed: #1494 imported-module typecheck errors

- **`f128` greenwash on lean_single — CLOSED (KL-9 partial, 2026-09-12; #2387).**
  `f128` lowers as real binary128 (kind 13, libgcc `__*tf*`); the probe
  `examples/numerics/f128_is_f64_probe.sio` reports 113 halvings. `f256` has no
  lowering and is refused fail-closed (`tc_wide_float_refused`) instead of being
  lowered as f64 via the lowercase-unknown-type rule.
  Pins: `scripts/ci/language_gap_ratchet_gate.sh` (`f128 halvings on lean_single`,
  `f256 refused by lean_single`);
  witness: `tests/compile-fail/f256_refused_on_lean_single.sio`.
- **Imported-module typecheck errors are non-fatal (#1494).** Engine:
  `lean_single`. `CONVERGENCE FIX` block stubs an imported fn only above 10
  errors; below that the partial codegen ships. The current build tolerates
  three such errors (`lower.sio`, `imports.sio`, `opt_cleanup.sio`). Record and
  options: `docs/compiler/BOOTSTRAP_SEED.md`. Pin: none (the build log carries
  the errors inline). Closing this needs an explicit policy choice among the
  three options in that doc — not a silent threshold tweak.
- Any further seed change re-runs the 2-stage bootstrap,
  `canonical_compiler_gate.sh`, `verify_lean_seed.sh` + `SeedReceipt.json`,
  `engine_parity_gate.sh`, and rebuilds Madaros from the new seed.


### KL-11 — #1792: first-order channels do not cross user calls

- Engine: `madaros` prints `var=0.000000` where `lean_single` shows ~1e-5 on
  the dissertation adaptive witnesses (`tests/run-pass/rapamycin_epistemic_adaptive.sio`,
  `stdlib/darwin_pbpk/epistemic_pbpk28.sio`), plus an ep28 confidence
  bit-pattern fabrication. `tests/run-pass/gum_fo_across_call.sio` documents
  that FO/variance channels stop at `ir_call`.
- Pin: `scripts/ci/epistemic_fabrication_detect_gate.sh` (detect-only).
- Locus: `ir/lower.sio` Knowledge layout / `variance_*_regs` / `pending_variance_reg`.
  Audit: `docs/audit/EPISTEMIC_FABRICATION_DETECT_2026-08-17.md`,
  `docs/audit/MADAROS_FO_CALL_BOUNDARY_DISPATCH_2026-08-18.md`.
- **`pow` FO / Hessian transfer — CLOSED (KL-11 partial, 2026-09-12).**
  `fo_xfer_seed_transcendentals` registers `pow` as kind 9; `fo_apply_transfer_kind`
  emits `∂/∂x`, `∂²/∂x²`, and first-order `∂/∂y` when the exponent carries
  sensitivity. Witness: `tests/run-pass/kl11_pow_fo_hessian.sio`
  (`hessian_of(pow(x, 3.0), 0, 0)` at `x=0.5` → `3.0`). Pin:
  `scripts/ci/madaros_kl11_pow_fo_gate.sh`. Mixed Hessian in the exponent, and
  FO across arbitrary user `fn` bodies, remain open.


### KL-13 — derived units and unit loss (#2388) — CLOSED

- **Call-boundary loss — CLOSED.** Unit-typed args no longer enter bare
  `f64` parameters unchecked. Pins:
  `tests/compile-fail/unit_lost_at_call_boundary.sio`,
  `tests/run-pass/unit_call_cast_strips_brand.sio`.
- **Derived declarations — CLOSED.** `unit velocity = m / s` registers the
  composed dimension on Madaros (`collect_unit_decl` honours
  `type_alias_ty`; ItemUnit is collected on the `*mut` spine). lean_single
  already had the Pass-0a path.
- **Derived annotations — CLOSED (KL-13b).** Bare `mol/cm3` / `cal/mol`
  parse as unit type expressions (TypeReference=/ TypeRefMut=* encoding,
  same as `parse_unit_item`). `f64<m/s>` accepts the same chain inside
  generics. Pins:
  `tests/run-pass/unit_derived_annotation_mol_per_cm3.sio`,
  `tests/compile-fail/unit_derived_annotation_refuse_add.sio`,
  `scripts/ci/language_gap_ratchet_gate.sh`.
- **Return-boundary loss — CLOSED (KL-13 partial).** A returned value is
  checked against the unit the signature declares, for an explicit `return`
  and for the body's tail expression alike. lean_single: the `return` branch
  of `compile_stmt` and the implicit-return epilogue of `compile_all` /
  `compile_all_arm64` both call
  `unit_call_arg_mismatch(CURRENT_RET_HASH, EXPR_UNIT, EXPR_UNIT_DIM)` and
  report `unit mismatch in return value`, on x86-64 and `--target
  aarch64-linux`. The helper and its f64-kind guard are the call-boundary
  ones, so the same three
  shapes are refused: a bare number into `-> molal`, `molar` into `-> molal`,
  and a unit-typed value into `-> f64`. Explicit `as f64` remains the escape
  hatch. Madaros refuses the same shapes with E008 (return value does not
  match the declared return type). Pins:
  `tests/compile-fail/unit_return_bare_number.sio`,
  `tests/compile-fail/unit_return_bare_number_tail.sio`,
  `tests/compile-fail/unit_return_wrong_unit.sio`,
  `tests/run-pass/unit_return_same_unit_tail.sio`.
- **Engine divergence, open (measured 2026-09-15 on the merged tree, after
  KL-13b).** Madaros accepts a bare number returned as a built-in unit
  (`fn f() -> mg { 250.0 }`, with or without `return`), which lean_single
  refuses; a declared unit (`-> molal`) is refused on both.
- Audit: `docs/audit/DIMENSIONAL_TYPING_GAP_2026-09-02.md`.

### KL-14 — FFI

- **Aggregate-reference arguments through the signatureless `ffi_` path —
  CLOSED (KL-14a).** Engine: `madaros`. `extern "C"` names are rewritten to
  `ffi_<name>` builtins (`parser/items.sio`, allowlist
  `extern_name_has_ffi_intrinsic`). A `&[i8; N]` argument used to forward a
  GC-handle / empty pointer; the call site now packs via `str_from_bytes`
  (Madaros arrays are 8-byte boxed slots, not contiguous C bytes) and passes
  the resulting `char*` to `ffi_system`. The `string` binding is unchanged.
  Pin: `tests/run-pass/ffi_system_array_arg.sio`. Doc:
  `docs/audit/MADAROS_EXTERN_C_BUILTIN_PORT_DISPATCH_2026-08-16.md`.
- **Dynamic linking MVP — CLOSED (KL-14b).** Engine: `madaros`. One
  non-builtin extern (`kl14b_add`) resolves via `PT_INTERP` + `PT_DYNAMIC` +
  `DT_NEEDED` (`libkl14b_probe.so`) + GOT/`R_X86_64_GLOB_DAT`, plus a
  `PT_LOAD` (R) of the ELF header page at `base_addr` so `ld.so` can see
  phdrs. Empty-stub body is `call [rip+got]; ret`. Dyn metadata is appended
  after the runtime-context data payload. Pin:
  `tests/run-pass/kl14b_dynlink_one_symbol.sio`,
  `scripts/ci/madaros_kl14b_dynlink_gate.sh`.
- **N-symbol dynlink — CLOSED (KL-14c).** Engine: `madaros`. Unique symbols
  from `extern_relocs` (cap 8) each get a dynsym + GOT slot +
  `R_X86_64_GLOB_DAT`; SysV hash chains them under `nbucket=1`. Pin:
  `tests/run-pass/kl14c_dynlink_n_symbols.sio` (`kl14c_add`/`mul`/`neg` via
  `libkl14c_probe.so`), `scripts/ci/madaros_kl14c_dynlink_gate.sh`.
- **Multi-`DT_NEEDED` — CLOSED (KL-14d1).** Engine: `madaros`. Symbol→soname
  allowlist emits one `DT_NEEDED` per unique library (cap 4). Pin:
  `tests/run-pass/kl14d_multi_needed.sio` (`kl14d_a`/`kl14d_b` via
  `libkl14d_a.so` + `libkl14d_b.so`),
  `scripts/ci/madaros_kl14d_multi_needed_gate.sh`.
- **libzstd e2e — CLOSED (KL-14d2).** Engine: `madaros`. `ZSTD_compress` /
  `ZSTD_decompress` / `ZSTD_isError` resolve via `DT_NEEDED libzstd.so.1`.
  `stdlib/compress/zstd.sio` wrappers fill `ZstdResult`. Pin:
  `tests/run-pass/kl14d_zstd_e2e.sio`,
  `scripts/ci/madaros_kl14d_zstd_gate.sh`. Dynlink GOT stubs tail-`jmp` (not
  `call; ret`) so SysV stack alignment holds for SIMD callees.
- **dlopen + call-through — CLOSED (KL-14d3).** Engine: `madaros`.
  `dlopen` / `dlsym` / `dlclose` / `dlerror` via `DT_NEEDED libdl.so.2`.
  Pin: open + `dlsym` → `as fn(i64) -> i64` → `f(35) == 42` + close
  (`tests/run-pass/kl14d_dlopen.sio`,
  `scripts/ci/madaros_kl14d_dlopen_gate.sh`).

### KL-15 — `f256` surface and epistemic `f128`

- Engine: `madaros`. **KL-15a partial CLOSED**: IEEE binary256 add/sub over
  `F256Bits` in `stdlib/math/softfloat_f256.sio` (ladder `--stage v0f5`).
  Residual: language `f256` arithmetic stays V0-E.4.1 fail-closed; fields,
  params, arrays, printing, `Knowledge<f128>`, GUM over `f128`, and
  `MeasuredF256` are not implemented. Consequence:
  `benchmarks/chemistry/RESULTS.md` §7.7 stays blocked on a genuine reference
  integration path.
- Pin: `scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0f5` (add/sub);
  `--stage v0e57` pins the `[f256; N]` refusal; `--stage v0e41` pins the
  no-greenwash rule.

### KL-16 — Hessian Tier-4 on the seed

- Engine: `lean_single`. `hessian_of(expr, j, k)` works for 8-channel
  arithmetic; unary transcendentals and `atan2`/`pow` on channels 0–7
  (x86 seed). a64 unary already loops 8 channels; a64 `atan2`/`pow`
  remain value-only (no AD shadow).
- **KL-16a — CLOSED (pin only).** Gate + Tier 1–3 pins without seed edit.
- **KL-16b — CLOSED (seed).** Fixes `VAR_HSHADOW` leak across Knowledge
  locals (H[4,5] of a product was `1+f`, observed as `7.0`); extends
  x86 unary/`atan2`/`pow` FO+Hessian to channels 4–7; pins
  `epistemic_hessian_8inputs.sio` at analytic `1.0` and
  `epistemic_hessian_ch47.sio`. Seed refresh + SeedReceipt required.
- **KL-16c — CLOSED (seed).** Inter-procedural FO ch0 (`EXPR_SSHADOW`)
  + `H[0,0]` (`EXPR_HSHADOW_00`) across user `f64 → f64` fns via BSS
  ARG/RET slots mirroring β⁵ variance. Pin:
  `tests/run-pass/kl16c_fo_across_user_fn.sio`,
  `scripts/ci/lean_single_kl16c_interproc_shadow_gate.sh`.
- **KL-16d — CLOSED (seed).** Extends inter-procedural FO to channels
  1–7 (`EXPR_SSHADOW_1..7` / `VAR_SSHADOW_1..7`) across user
  `f64 → f64` fns; HSHADOW multi-pair across calls remains OPEN.
  Pin: `tests/run-pass/kl16d_fo_multich_across_user_fn.sio`,
  `scripts/ci/lean_single_kl16d_interproc_multich_gate.sh`.
- **KL-16e — CLOSED (seed).** If/else join merges FO `EXPR_SSHADOW(_1..7)`
  and `EXPR_HSHADOW_00` via path-local spill into shared join slots
  (phi-like select by execution). Pin:
  `tests/run-pass/kl16e_ifelse_shadow_merge.sio`,
  `scripts/ci/lean_single_kl16e_ifelse_shadow_merge_gate.sh`.
- **KL-16f — CLOSED (seed).** Loop accumulation of FO
  `VAR_SSHADOW(_1..7)` and `VAR_HSHADOW_00` on mutable `f64`. Declaration
  allocates nine fixed slots and spills the RHS; reassignment spills into
  those slots and does not retarget the metadata pointers (ephemeral
  `EXPR_*` no longer alias the variable). Pin:
  `tests/run-pass/kl16f_loop_accum.sio`,
  `scripts/ci/lean_single_kl16f_loop_accum_gate.sh`.
- **Residual (OPEN).** HSHADOW multi-pair interproc; HSHADOW pairs other
  than `[0,0]` through if/else; a64 `atan2`/`pow` AD.
- Channel-at-`.value` semantics (`MEAS_KNOW_IDX`,
  `formal/ChannelAssignmentSemantics.lean`) are a model, not a defect —
  see the history snapshot for the KAS-1 rationale.

### KL-17 — generic `impl` blocks: residuals

- Engine: `madaros`. `impl S<T>` / `impl<T> S<T>` on a generic struct is
  monomorphized per struct instance: the specializer clones the block as
  `impl S__args` with every method substituted, rewrites `S<A>` types and
  `S::<A>::f(..)` paths in concrete code to `S__A`, and places the clones
  ahead of every other item (`check/specializer.sio`,
  `spec_emit_generic_impl_instances`). Witnesses:
  `tests/run-pass/madaros_generic_impl_instances.sio` (`Cell<i64>` and
  `Cell<f64>` in one unit), `tests/stdlib/collections/test_heap_vec_generic.sio`.
- **Associated calls need the turbofish.** `G::new()` with no type arguments
  is refused (`an associated function of a generic struct must be called with
  explicit type arguments`); the `let` annotation does not select the
  instance. Pin: `tests/compile-fail/madaros_generic_impl_static_no_targs.sio`.
- **Literals that cannot infer `T`.** `var g: G<i64> = G { n: 0 }`, where no
  field mentions `T`, is typed `G__T` by the checker and rejected with E001:
  the annotation is rewritten, the literal is not. Use `G::<i64>::new(..)` or
  a field that fixes `T`. Pin: none.
- **Method type params inside a generic impl.** `fn pair_with<U>(self: &G<T>,
  u: U) -> U` is refused even when called as `g.pair_with::<f64>(1.5)`
  (`generic method specialization requires one unambiguous explicit
  turbofish`): the impl clone substitutes `T` only, and the method no longer
  enters the generic-fn table. Pin: none.
- **Header shape.** Header arguments must be single uppercase letters, one per
  struct parameter. Struct literals inside the impl are rewritten only when the
  header reuses the struct's own parameter names (`impl G<U>` on `struct G<T>`
  leaves `G { .. }` to the checker's inference).
- **Raw-word storage in `collections::heap_vec::HeapVec<T>`.** Slots hold the
  64-bit pattern of `T` via `write_i64`/`read_i64`, which is sound only for
  8-byte scalars. Reads must not cast: `read_i64(..) as T` converts the number
  (2.5 read back as 4612811918334230528.0). `sum` returns the i64 sum of the
  patterns.
- Caps: 64 generic impl blocks (specializer code 6); 1,024 emitted
  generic-struct specializations, shared by generic-impl instances (code 4);
  1,024 distinct generic-function instantiations (code 5); a generated `__sp_`
  symbol that collides with a source item is refused (code 7). The
  type-parameter registry holds 16,384 declarations, 65,536 names and 1 MiB of
  name text; a full registry refuses the compile.

## Registry-governed, not rungs

These are maturity statements, owned by
`docs/serious-language/public-claim-registry.v1.tsv` and mirrored in
`docs/compiler/MATURITY.md`; a rung does not close them.

- `closures.lambdas = stale_conflicting`: captured closures as first-class
  values and native cross-engine parity unresolved; `closure_linear.sio`
  ignored.
- `generics.* = prototype`: trait bounds parsed, not enforced at call sites;
  no trait objects.
- LSP pure-Sounio rebuild (`self-hosted/lsp/server.sio`) does not rebuild
  under current Madaros; the checked route is `tools/lsp/sounio-lsp.sh`.
- Compact imported IR (`SOUNIO_ENABLE_COMPACT_IMPORTED_IR=1`) still reports
  `imported_simple_ir_emit_failed` and falls back to full IR; not default.
- Multi-module exclusive-ref shapes outside the gated corpus may still be
  fragile; the gated ones (`madaros_d3_exclref_shipped_gate.sh`,
  `madaros_trait_i64_cd_exact_gate.sh`) are green.

## Reading a rejection: E035 and E137 are not language limitations

Measured 2026-09-03 over `stdlib/`, `examples/` and `tests/run-pass/` — 4539
files, 78.4% accepted by `souc check`
(`scripts/dev/language_limitation_sweep.sh`,
`artifacts/audit/language_limitation_sweep_20260903.tsv`).

- **E035** (`effect not declared in function signature`): 199 files, 140 in
  `stdlib/`; 57 shared one cause (`Epistemic::measured` over-declared `with
  Mut, Div, Panic`, removed in `35b92be4d1`, stdlib 140 → 36 files). The
  effect checker was correct throughout; read the callee named by `required
  by` before blaming the caller.
- **E137** (`use of undeclared variable`, also fires on unresolved function
  names): 323 files, 3623 occurrences. 191 files import nothing at all; of
  the 676 occurrences in files that do import, 529 name a function defined
  nowhere, 147 name one that exists but is not imported, and **0** name one
  that exists and is imported. That zero is the number that clears name
  resolution.
- Measure with the compiler and stdlib pinned together (`SOUNIO_MADAROS_BIN`
  inside one tree; `SOUNIO_STDLIB_PATH` changes verdicts on its own). Full
  numbers and caveats: history snapshot, section "Reading a rejection".
