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
| KL-1 | small parity items: `~`, `Epistemic(-N)`, `*const`/`*mut` at call sites, `println` of computed local | madaros |
| KL-2 | IEEE 754 special values (#2389) | both |
| KL-3 | specializer caps, method turbofish | madaros |
| KL-4 | private field reads unchecked | both |
| KL-5 | ε polarity fork | madaros |
| KL-6 | Hessian quotient / composite chain on Madaros | madaros |
| KL-7 | `i256`/`i512` wide-local `print_int` | madaros |
| KL-8 | `f128` surface residuals | madaros |
| KL-9 | seed: `f128` greenwash, #1494 tolerated errors | lean_single |
| KL-11 | #1792 first-order / variance across calls | madaros |
| KL-12 | thin-link `rc=12` probes | madaros |
| KL-13 | derived units, unit loss at call boundary (#2388) | both |
| KL-14 | FFI: aggregate-ref args, dynamic linking | madaros |
| KL-15 | `f256` surface, `Knowledge<f128>`/GUM | madaros |
| KL-16 | Hessian Tier-4 on the seed | lean_single |

## Ledger

### KL-1 — small parity items

- **Unary `~` refused by Madaros.** Engine: `madaros` (E005; `lean_single`
  accepts). Repro: `tests/run-pass/bitwise_not_bootstrap_regression.sio`. Pin:
  `tests/engine_parity_baseline.txt` row `LEAN-ONLY` for that file. Locus:
  `self-hosted/parser/exprs.sio:211` maps `Tilde` to `UnaryOp::OpNot`;
  `check/compat.sio:1233` makes `OpNot` bool-only; `native/encode.sio:1576`
  `emit_not_rax` exists and is unused.
- **`Epistemic(-N)` collides with "no payload".** Engine: `both` accept the
  syntax; Madaros erases `-1` to the no-payload sentinel
  (`parser/ast.sio:140` `EffectRef.payload`, `parser/types.sio:940`).
  Positive floors are enforced (E215). Receipt:
  `docs/audit/MADAROS_EPISTEMIC_PAYLOAD_GATE_2026-08-20.md`. Pin: none.
- **`*const T` vs `*mut T` at a call site.** Engine: `madaros`. Passing
  `expr as *const u8` where the callee takes `*mut u8` (or the reverse) can
  produce arity/type diagnostics; stdlib wrappers prefer `*mut u8`. Locus:
  `check.sio:17310` `checker_call_arg_types_compatible_table`,
  `check/compat.sio:226`. Pin: none (the fixture the old text cited,
  `tests/stdlib/compress/test_zstd_e2e.sio`, is now a constants-only stub and
  does not exercise this).
- **`println(<computed local>)` routes to the `char*` printer.** Engine:
  `madaros`. A local bound from a bare index/field-access initializer without
  an int scalar-kind marker (`let v = r.c[0]; println(v)`) reaches
  `println_dispatch_name` (`ir/lower.sio:12514`) with kind 0. Locus:
  `lower_let_stmt_ref` unannotated path `lower.sio:16922-16991` only marks
  int for `ExprCall`/`ExprMethodCall`/`Binary`/`Unary`. Pin: none.

### KL-2 — IEEE 754 special values (#2389)

- Engine: `both`. `nan == nan` is `1`, `nan != nan` is `0`, `nan < 1.0` is
  `1` (IEEE: `0 / 1 / 0`); `x != x` cannot detect NaN. `lean_single`:
  `println(inf)` never returns, `println(nan)` prints non-numeric bytes.
  `madaros`: `println(inf)` prints `9223372036854775808.000000`,
  `println(nan)` prints `-9223372036854775808.000000`.
- Repro: `tests/known-gaps/numerics/{nan_compare_is_not_ieee,print_inf_never_returns,print_nan_is_garbage}.sio`.
- Pin: `scripts/ci/language_gap_ratchet_gate.sh` (workflow
  `language-gap-ratchet.yml`) pins the wrong values; the fix flips those
  lines deliberately.
- Locus: `native/codegen_x86_linux.sio:8150-8199` (`sete/setne/setb/…` after
  `ucomisd`, no PF handling), `native/lower_ir.sio:689-727`, seed
  `lean_single.sio:21203-21214`; `print_f64` at `codegen_x86_linux.sio:6185`
  (`cvttsd2si` without inf/nan guard) and seed `__native_print_f64_n`
  (`lean_single.sio:41729`).
- Working rule until fixed: bound every loop that exits on a float
  comparison and range-check after it (`ulp()` in
  `examples/chemistry/rep_stagnation.sio`).

### KL-3 — specializer caps and method turbofish

- **Silent caps.** Engine: `madaros`. At most 4 type parameters per generic
  function (`tp_h0..tp_h3`), 256 generic functions, 256 generic structs and
  64 emitted hashes per unit (`check/specializer.sio:74-143`, `:455`, `:486`,
  `:1045`). Exceeding a cap degrades silently. Pin: none.
- **Method-call turbofish is inert.** Engine: `madaros`. `x.m::<T>()` parses
  and stores `e.type_args` (`parser/exprs.sio:1536-1579`) but
  `checker_check_method_call_with_base_ty_inplace` (`check.sio:8122`) never
  reads it and the specializer only walks `ExprCall`
  (`specializer.sio:1202`). The annotation is discarded, not rejected. Pin:
  none.
- Context (closed, kept for orientation): one-instantiation-per-template
  monomorphiser; non-scalar multi-instantiation is refused
  (`scripts/ci/madaros_specializer_nested_targ_gate.sh`); recursive name
  rendering with `_Lb_`/`_Cm_`/`_Rb_`;
  `docs/handoff/BLK-20260903-specializer-nested-targ-collision.md`.

### KL-4 — private field reads are not checked

- Engine: `both` for direct reads; on Madaros the parser drops the field's
  `pub` (`parser/items.sio:1861-1863`, also `:1966`, `:2072`), so
  `FieldDef`/`FieldInfo` carry no visibility (`parser/ast.sio:1159`,
  `check/defs.sio:16`). Struct-literal construction of a private struct is
  refused (E176 via `checker_struct_visible_inplace`); field access
  (`check.sio:7301`) is not.
- Repro/pin: none (the zero-event constructor-privacy fixtures cover the
  struct-literal half only). Blast radius over `stdlib/` is measured before
  enforcement; a warning-first ratchet is acceptable for one rung.

### KL-5 — ε has opposite polarities in the two engines

- Engine: `madaros` accepts `tests/compile-fail/vancomycin_low_conf.sio`
  (`check: OK`, rc=0); `lean_single` refuses it with P0003. Madaros reads ε as
  an error bound (`epsilon_subsumes` is `a <= b`,
  `check/epistemic.sio:607`; `epsilon_subsumes_call_boundary` `:613` ignores
  `EpsilonBound.op`), the corpus reads it as confidence (`ε >= 0.82`: 15
  uses, `ε =`: 10, `ε <`: 0). Seed: `ty_eq` `lean_single.sio:4276-4295`
  honours the operator.
- Pin: the three covering gates (`clinical_vanco_tdm_e2e`,
  `epistemic_prescription_chain_e2e`, `ousadia_epistemic_method_rx`) pin
  `SOUNIO_SOUC_ENGINE=lean_single` and are not workflow-reachable;
  `scripts/ci/epsilon_engine_parity_gate.sh` (ci.yml) covers a narrower set.
- Audit: `docs/audit/EPSILON_POLARITY_FORK_2026-08-19.md`. Do not state the
  vancomycin ε guarantee without naming the engine.

### KL-6 — Hessian on Madaros: quotient and composite chain rule

- Engine: `madaros`. `tests/run-pass/madaros_hessian_quotient.sio` prints
  `h_aa=h_ab=h_bb=h_rec=0.000000` and `MADAROS_HESSIAN_QUOTIENT_FAIL` on the
  committed ELF and on source. Composite inner expressions (`exp(x*x)`,
  `exp(sin(x))`, `sin(sin(x))`) are deliberately refused by clearing the
  Hessian and returning `0.0` (`ir/lower.sio:11338-11347`, `:11423`); the
  `f'(g)·H(g)` term is not implemented.
- Locus: `lower_expr_variance_ref` guard `lower.sio:12277`
  (`fo_expr_or_snap_has_sens(sL)` only — a Hessian carried by the right
  operand alone is dropped), `so_combine_hess_div` `:9782`,
  `so_combine_hess_mul` `:9871`.
- Pin: the transcendental witness
  `tests/run-pass/madaros_hessian_transcendental.sio` is green and must stay
  so; the quotient witness is the failing repro. Do not generalise the
  transcendental result to Hessian AD as a whole.

### KL-7 — `i256`/`i512` wide-local `print_int`

- Engine: `madaros`. Wide values are consecutive virtual registers plus an
  immediate pool (`ir/lower.sio:48-49`, `fresh_wide_reg :12788`,
  `ir/numeric_payload.sio:165`); `print_int` of a wide local
  (`lower.sio:19661-19679`) prints the low limb. Negative wide literals rely
  on low-limb sign extension that is not asserted.
- Repro: `tests/run-pass/r1_i256_lorenz_peak.sio` (green — proves multiply
  and shift, not printing). Pin: `scripts/ci/madaros_wide_int_gate.sh`
  (emitter only, no `print`). Receipt:
  `docs/audit/R1_I256_I512_LIMBS_2026-08-20.md`.
- Out of scope for this ledger and for any rung: the Lorenz certificate
  conclusions in `stdlib/systems/` remain unaudited; do not state that any
  certificate conclusion is proved or wrong from these receipts (spec
  `docs/spec/S12_NUMERIC_TOWER.md` §12.2.6, §12.4-6).

### KL-8 — `f128` surface residuals (after V0-E.5.10)

- Engine: `madaros`. Ladder record:
  `docs/architecture/F128_F256_LADDER.md`; every closed stage is pinned by
  `scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0b … v0e510`.
- A float literal as the tail of a **nested** block used as the fn tail
  (`fn f() -> f128 { if c { 1.0 } else { 2.0 } }`) is E008 — only the body
  block and `return` are routed (`LOWER_F128_FN_BODY_PENDING`,
  `ir/lower.sio`).
- `%` and `+=` on `f128` are refused (no `soft_rem` desugar; compound
  assignment not routed).
- Builtin `print`/`println` of an `f128` is refused at lowering (`cannot
  safely lower print/println argument with unresolved scalar kind`);
  `println_f128` needs an explicit `use math::softfloat_f128_fmt` because
  the implicit import (V0-E.5.10) covers `math/softfloat_f128.sio` only.
- The legacy `native_compile_driver.sio` lexer (`driver_lex_source_to_globals`,
  `:8872-8945`) does not take `_` separators; no longer reachable from any gate (the `native_v2_*` leaf gates were
  deprecated in KL-10, `scripts/ci/deprecated/`).
- Pin: the v0e510 gate pins the closed half; the residuals above have no
  negative fixture yet.

### KL-9 — seed: `f128` greenwash and #1494

- **`lean_single` lowers `f128` as f64.** Engine: `lean_single`. Repro:
  `examples/numerics/f128_is_f64_probe.sio` (53 halvings). Pin:
  `scripts/ci/language_gap_ratchet_gate.sh` line for `f128 halvings`.
  Locus: lowercase-unknown-type rule `lean_single.sio:24359-24361`,
  `:17204-17206`, `type_name_kind :4191`. Target: refuse `f128`/`f256`
  spellings in the seed (fail closed), never widen.
- **Imported-module typecheck errors are non-fatal (#1494).** Engine:
  `lean_single`. `CONVERGENCE FIX` block `lean_single.sio:31416-31443` stubs
  an imported fn only above 10 errors; below that the partial codegen ships.
  The current build tolerates three such errors (`lower.sio`, `imports.sio`,
  `opt_cleanup.sio`). Record and options: `docs/compiler/BOOTSTRAP_SEED.md`.
  Pin: none (the build log carries the errors inline).
- Any seed change re-runs the 2-stage bootstrap, `canonical_compiler_gate.sh`,
  `verify_lean_seed.sh` + `SeedReceipt.json`, `engine_parity_gate.sh`, and
  rebuilds Madaros from the new seed.

### KL-11 — #1792: first-order channels do not cross user calls

- Engine: `madaros` prints `var=0.000000` where `lean_single` shows ~1e-5 on
  the dissertation adaptive witnesses (`tests/run-pass/rapamycin_epistemic_adaptive.sio`,
  `stdlib/darwin_pbpk/epistemic_pbpk28.sio`), plus an ep28 confidence
  bit-pattern fabrication. `tests/run-pass/gum_fo_across_call.sio` documents
  that FO/variance channels stop at `ir_call`.
- Pin: `scripts/ci/epistemic_fabrication_detect_gate.sh` (detect-only).
- Locus: `ir/lower.sio:1120-1185` Knowledge layout,
  `variance_base_regs/variance_value_regs [1024]` `:975-977`,
  `pending_variance_reg :988`. Audit:
  `docs/audit/EPISTEMIC_FABRICATION_DETECT_2026-08-17.md`,
  `docs/audit/MADAROS_FO_CALL_BOUNDARY_DISPATCH_2026-08-18.md`.

### KL-12 — thin-link `rc=12`

- Engine: `madaros` native-v2. Two fail-closed probes:
  `tests/known_failures/thinlink_bool_cmp_field_probe.sio` (`Pair { a: 2.0 >
  0.0, b: 3.0 > 0.0 }`, ~3 fn) and
  `tests/known_failures/zero_provenance_native_v2_probe.sio`
  (sedenion + `eisa::core_v2`, ~111 fn). Both stop in
  `compile_ir_function_v2_from_ir_into` (`codegen_x86_linux.sio:12515-12541`)
  with `NV2_IR unsupported fn=` and no opcode named.
- Pin: `scripts/ci/madaros_thinlink_bool_cmp_field_gate.sh`,
  `scripts/ci/madaros_zero_provenance_failclosed_gate.sh`. BLKs:
  `docs/handoff/BLK-20260805-thinlink-ir-threshold.md`,
  `docs/handoff/BLK-20260805-p0b-zero-provenance.md`. The compact
  zero-provenance smoke (`zero_provenance_native_v2_smoke.sio`) is a
  distinct, smaller CU — do not cite it for the combined import.

### KL-13 — derived units and unit loss (#2388)

- Engine: `both`. `mol/cm3` does not parse on either engine
  (`parser/items.sio:4117-4172` `parse_unit_item` has no unit-expression
  grammar); `unit velocity = m / s` declares a dimensionless unit
  (`check.sio:20169` `collect_unit_decl` ignores the expression); a
  quotient of unit-typed values loses its dimension on `lean_single`; a `K`
  value passes into an `f64` parameter unchecked on both
  (`check.sio:26765` `check_call_arg_unit_boundary`, seed
  `lean_single.sio:7416-7437`). Direct `mol + K` is rejected on both (E041).
- Repro: `tests/known-gaps/units/{derived_unit_annotation_unparsed,direct_unit_mismatch_is_caught,unit_lost_at_call_boundary}.sio`.
- Pin: `scripts/ci/language_gap_ratchet_gate.sh`. Audit:
  `docs/audit/DIMENSIONAL_TYPING_GAP_2026-09-02.md`.

### KL-14 — FFI

- **Aggregate-reference arguments through the signatureless `ffi_` path.**
  Engine: `madaros`. `extern "C"` names are rewritten to `ffi_<name>`
  builtins (`parser/items.sio:1072-1124`, allowlist
  `extern_name_has_ffi_intrinsic`); a `&[i8; N]` argument forwards an empty
  pointer. Repro: `tests/run-pass/ffi_system_array_arg.sio`
  (`//@ known-failure`). Non-allowlisted externs fail closed with E250
  (`check.sio:9478`). Doc:
  `docs/audit/MADAROS_EXTERN_C_BUILTIN_PORT_DISPATCH_2026-08-16.md`.
- **No dynamic linking.** Engine: `madaros`. The ELF writer emits static
  executables; `native/reloc.sio:271-291` records `R_X86_64_PLT32` for
  ET_REL only; there is no `PT_INTERP`/`PT_DYNAMIC`/`.dynsym`/`.rela.plt`/
  `DT_NEEDED` anywhere. `-lfoo`-style shared-library calls are not possible;
  `tests/stdlib/compress/test_zstd_e2e.sio` is a constants-only stub because
  no libzstd call can be linked. Pin: none.

### KL-15 — `f256` surface and epistemic `f128`

- Engine: `madaros`. `f256` has type spellings, exact literals (V0-B/V0-E.5.9)
  and the V0-E.4.1 fail-closed refusal for arithmetic; fields, params,
  arrays, printing and any `softfloat_f256` are not implemented.
  `Knowledge<f128>`, GUM over `f128` and `MeasuredF256` are out of scope of
  the V0-E ladder. Consequence: `benchmarks/chemistry/RESULTS.md` §7.7 stays
  blocked on a genuine reference integration path.
- Pin: `scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0e57` pins the
  `[f256; N]` refusal; `--stage v0e41` pins the no-greenwash rule.

### KL-16 — Hessian Tier-4 on the seed

- Engine: `lean_single`. `hessian_of(expr, j, k)` works for 8 channels,
  arithmetic, unary transcendentals and `atan2`/`pow` on channels 0–3.
  Not implemented: inter-procedural shadows across user fn calls, loop
  accumulation (state resets per iteration), `if/else` merge of shadow
  slots, channels 4–7 in transcendentals and two-arg builtins.
- Pin: none beyond the positive witnesses. Channel-at-`.value` semantics
  (`MEAS_KNOW_IDX`, `formal/ChannelAssignmentSemantics.lean`) are a model,
  not a defect — see the history snapshot for the KAS-1 rationale.

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
