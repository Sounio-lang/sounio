<!-- docs:meta
topic_id: repo.docs.compiler.known-limitations
authority: repo_only
audience: contributors
last_validated: 2026-09-22
validated_by: claude
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
| KL-20 | enum variants with payloads (`Circle(f64)`, `Rect { w: f64 }`): no runtime representation | both |
| KL-21 | user fns named like compiler builtins: `pub` residual on Madaros; seed hijacks or rejects | both |
| KL-18 | `Hyper<…>` CPU values: Madaros fail-closed; lean_single prints a wrong value (seed) | both |

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

- #1792 (filed 2026-08-17) named two distinct witnesses. **F1**:
  `tests/run-pass/rapamycin_epistemic_adaptive.sio` — `madaros` originally
  printed `var(blood)=0.000000` where `lean_single` shows ~1e-5. **F2**:
  `stdlib/darwin_pbpk/epistemic_pbpk28.sio` TEST 6 — Madaros printed an IEEE
  bit-pattern (~4.6e18) as the AUC confidence, not a variance collapse.
  **F2 is CLOSED**, fixed by PR #1882
  (`d33cf5856b57f3341db9392d263045d090d88ae7`, merged 2026-08-18):
  `Knowledge.confidence` was tagged `is_float: 3` at IR layout instead of `1`,
  so `sitofp` on the raw bits produced the huge value;
  `ir_register_knowledge_layout` now tags it `1`. **F1, and this rung in
  general, stay OPEN**: `tests/run-pass/gum_fo_across_call.sio` and
  `tests/run-pass/fo_call_boundary_arity3.sio` still document, with a live
  `//@ known-failure`, that FO/variance channels stop at `ir_call` for the
  general case. Re-measured 2026-09-22: the specific
  `rapamycin_epistemic_adaptive` witness (F1) now reports non-zero variance
  too, and the accompanying test change is `b2df0727` (2026-09-18) — but that
  commit touches only the test fixture and its recorded gate/dataset
  artefacts (`git show --stat b2df0727`: no `self-hosted/` file), relaxing
  `ok_mech` from requiring `epist_active > 0` to `epist_active > 0 ||
  ok_var`. It documents that the lookbehind mechanism no longer needs to
  fire for the test to pass; it does **not** touch variance computation, so
  it cannot be the cause of the witness's variance moving off zero. That
  compiler-side cause is **unidentified**. Do not read the healthy witness,
  or `b2df0727`, as KL-11 closing.
- Pin: `scripts/ci/epistemic_fabrication_detect_gate.sh` (detect-only; as of
  2026-09-22 both its F1 and F2 checks take the "engine healthy" branch on
  the two named witnesses, which is expected given the above and is not by
  itself evidence this rung is closed).
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

### KL-20 — enum payload variants (P1.3)

- Engine: `both`. Repro (measured 2026-10-06, `origin/main` `fa695abaa`):

  ```sounio
  enum Shape { Circle(f64), Rect(f64, f64) }
  fn area(s: Shape) -> f64 {
      match s { Shape::Circle(r) => 3.0 * r * r, Shape::Rect(w, h) => w * h }
  }
  fn main() with IO { println(area(Shape::Circle(2.0))) }
  ```

  | engine | before | after (this branch) |
  |---|---|---|
  | Madaros (shipped ELF and built from `main`) | `parse error: expected token at line 1:20 expected=132 actual=-8894744987059046268` (raw token ids) | **E264** at each construction and each payload pattern |
  | lean_single (seed, unchanged) | declaration accepted; ``error[E200]: undefined identifier `r` `` in the arm; constructing `Shape::Rect(3, 4)` is `E006 arity mismatch ... expected 1 got 2` | same |

- **Struct variants were a silent miscompile on Madaros.** `enum Shape {
  Circle { r: i64 }, Rect { w: i64, h: i64 }, Empty }` built and ran:
  `area(Shape::Rect { w: 3, h: 4 })` returned `0`, and a tag-only match put
  both `Circle` and `Rect` values on no arm (result `0`); only the unit
  variant `Empty` was right. Cause: `ir/lower.sio` represents every enum
  value as a bare discriminant (`lower_match_arm_ref` compares the scrutinee
  to the variant index, binds no sub-pattern, and skips `PatStruct` arms
  outright, "Unsupported pattern — skip"). This branch refuses it with
  **E264** (construction in struct-literal or call form, payload
  sub-patterns, struct patterns over a user enum). Declaring a payload
  variant is still accepted (`tests/selfhost/native_runtime/
  enum_payload_then_plain_match_42.sio` declares one and matches only unit
  variants), and tuple variants now parse (fields `_0`.._7) so the refusal
  can name the construction site.
- Not done here, and why: closing this needs a heap representation for
  payload-bearing enums (tag word + fields), lowering at every construction
  site and every match arm, per-binding float/struct metadata in the lowerer
  (`LowerLocalStack.scalar_kind` etc.), payload field typing in the checker
  (sub-patterns are bound to the enum type itself today, `checker_bind_
  pattern_inplace`), a refusal of `==` on such enums, and the same on
  lean_single. That is a feature across parser, checker and the 28k-line
  lowerer, not a focused change.
- Workaround: a payload-free tag enum plus a struct (`struct Shape { kind:
  ShapeKind, a: f64, b: f64 }`), or `Option<T>` for one optional value.
- Pins: `tests/compile-fail/enum_payload_tuple_variant_refused.sio`,
  `enum_payload_struct_variant_refused.sio`, `enum_payload_pattern_refused.sio`.
- `if let` (`tests/run-pass/if_let_pattern.sio`, previously E006/E137 on
  Madaros because the parser dropped the pattern) is **CLOSED** on this
  branch: it desugars to `match`, like `while let`. Pin:
  `tests/run-pass/if_let_desugar_forms.sio`. Found on the way: a bare `None`
  pattern parsed as a *binding*, so `match v { None => a, Some(x) => b }`
  took the `None` arm for `Some(42)`; fixed in the same branch. Residual,
  unchanged: Madaros lowers `Option<i64>` as a nullable word, so `Some(0)`
  is indistinguishable from `None`.

### KL-21 — user functions named like compiler builtins (P1.4)

- Engine: `both`. The checker (`checker_check_call_expr_inplace`) and the
  lowerer (`lower_call_expr_ref` → `call_expr_uses_special_a_ref` / `_b_ref`)
  recognise builtins by the bare identifier before any user signature.
  Measured 2026-10-06 with `fn NAME(x: i64) -> i64 { x + 1000 }` and
  `NAME(5)` for 65 names (one program per name):
  - **Madaros, shipped and built from `main`: 43 of 65 wrong.** Rejected at
    the call site: `measure`, `acknowledge`, `uncertainty_of`,
    `require_confidence` (E008); `Knowledge`, `print_int`, `print_char`,
    `seq_len`, `seq_set`, `second_order_mean`, `gpu_barrier` (E001);
    `variance_of`-family, `correlate`, `seq_new`/`seq_push`/`seq_get` (E010);
    `assert` (E174); the decision/transition builtins (E058–E133). Silently
    wrong: user `print` and `println` were replaced by the builtin. The
    reported `fn measure(...) -> i32` gives `E001 expected i32, found
    Knowledge<i64>` at the binding (E008 when returned).
  - **After this branch: 64 of 65 call the user function.** A module's own
    private fn shadows the builtin in that module: after parsing,
    `builtin_shadow_apply_items` (`compiler/private_fn_identity.sio`) renames
    it and every reference in the module to `<name>__user`. When the rewrite
    cannot be proven (a local, parameter or pattern spelled the same; a bare
    value use in a match-arm body or struct-literal field) the compile stops
    with ``error[builtin_shadow]: ... `measure` is a builtin; rename your
    function.`` Off switch: `SOUNIO_DISABLE_BUILTIN_SHADOW=1`.
  - **Residual (OPEN), Madaros:** `pub fn` with a builtin name is not
    renamed (importers spell it), so the builtin still wins at bare-name call
    sites; `f128_from_limbs` / `f128_to_lo` / `f128_to_hi` are deliberately
    excluded because `stdlib/math/softfloat_f128.sio` defines them as the
    implementation the intercept stands for (a user one fails to build).
    The renamed spelling `<name>__user` appears in diagnostics.
  - **Residual (OPEN), lean_single (seed, not changed — a fix needs a seed
    refresh):** 26 of 65 wrong. Silently wrong value (builtin ran, printed
    the argument `5` or `0`): `abs`, `f64_to_bits`, `lift_knowledge`,
    `prove_robust`, `validate_manifest`, `gpu_thread_id_x`, `slice_len`.
    Rejected: `print_int`, `print_char`, `print_f64`, `get_arg`, `str_eq`,
    `seq_new`, `seq_push`, `variance_of`, `sensitivity_of`, `gpu_barrier`
    (E001), `hessian_of`, `seq_set` (E200), `seq_len` (P0003); no output:
    `seq_count`, `seq_get`, `str_len`; failed to compile: `acknowledge`,
    `require_confidence`, `f128_to_lo`.
- Pins: `tests/run-pass/user_fn_shadows_builtin_measure.sio`,
  `tests/run-pass/user_fn_shadows_builtin_names.sio`,
  `tests/compile-fail/builtin_shadow_local_binding_refused.sio`.
### KL-18 — `Hyper<Algebra, T>` values on the CPU path (P0.8.2)

- Engine: both. There is no CPU value lowering for `Hyper<Octonion, f64>`
  (or any `Hyper<…>`); the only implemented CPU octonion product is
  `algebra::octonion::oct_mul` over `[f64; 8]` (Fano convention, e1·e2 = e3).
  Reference: `(2 + e1)(3 + e2) = [6, 3, 2, 1, 0, 0, 0, 0]`, which `oct_mul`
  returns on both engines.
- **Madaros — fail-closed.** `[..] as Hyper<…>` is refused in CPU lowering
  ("Hyper<...> values ... are not implemented on the native CPU path"). Before
  the refusal (measured 2026-10-06 on the shipped ELF): `.e1` of the cast
  printed `0.000000`, `a + b` segfaulted, `a * b` failed in the backend with
  rc 12. Pin: `tests/compile-fail/hyper_octonion_mul_cpu_refused.sio`.
  The `--backend gpu` path lowers `Hyper<…>` through HLIR and is unaffected.
- **lean_single — OPEN, silent wrong value.** The engine does not know the
  type: `println(([2.0, 1.0, 0.0, ..] as Hyper<Octonion, f64>) * ..)` prints
  `8843176242182119936` and exits 0; routing the product back through
  `as [f64; 8]` segfaults (rc 139). Closing it requires a lean_single source
  change and a seed refresh (`scripts/dev/refresh_lean_seed.sh`), which is a
  founder-run step. Until then: do not use `Hyper<…>` values under
  `SOUNIO_SOUC_ENGINE=lean_single`.

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
- **Builtin effects (#2760).** Madaros now attributes the effect row of the
  implicit builtins the way lean_single does: `IO` for `print`, `println`,
  `read_byte`, `read_line`, `read_file`, `file_size`, `write_file`,
  `write_bytes`, `append_file` (`print_int`, `print_char` and `syscall6` already
  did); `Panic` for `assert` and `panic`; `Alloc` for `malloc`, `heap_alloc`,
  `heap_realloc` and `heap_free`. A caller that does not declare the effect is
  rejected with E035 on both engines. A user function with the same name as a
  builtin keeps its own signature. Still unguarded on **both** engines:
  `print_f64`, `read_i64`, `write_i64`, `read_f64`, `write_f64` -- lean_single
  does not check them either, so they are a shared gap, not a divergence.
  `print_int` is guarded by Madaros only.
- **Two E035 divergences from lean_single remain, both older than #2760.**
  (1) lean_single grants `main` every basic effect; Madaros requires
  `fn main() with IO` even when `main` only calls a `with IO` user function.
  (2) Madaros checks a closure body against an empty effect row: it neither
  inherits the enclosing function's effects nor honours `|x| -> T with IO`, so a
  closure that prints, or calls a `with IO` function, is E035 even inside
  `main with IO`. After #2760 this includes the builtins above;
  `tests/run-pass/closure_effect_transparent_hof.sio` carries `known-failure`
  for it.
