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
| KL-9 | seed: `f128` greenwash, #1494 tolerated errors | lean_single |
| KL-11 | #1792 first-order / variance across calls | madaros |
| KL-12 | thin-link `rc=12` probes | madaros |
| KL-13 | derived units, unit loss at call boundary (#2388) | both |
| KL-14 | FFI: aggregate-ref args, dynamic linking | madaros |
| KL-15 | `f256` surface, `Knowledge<f128>`/GUM | madaros |
| KL-16 | Hessian Tier-4 on the seed | lean_single |

## Ledger



### KL-9 — seed: #1494 imported-module typecheck errors

- **`f128` greenwash on lean_single — CLOSED (KL-9 partial, 2026-09-12).**
  The seed refuses `f128`/`f256` spellings fail-closed (`tc_wide_float_refused`)
  instead of lowering them as f64 via the lowercase-unknown-type rule.
  Pin: `scripts/ci/language_gap_ratchet_gate.sh` (`f128 refused by lean_single`);
  witness: `tests/compile-fail/f128_refused_on_lean_single.sio`.
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
- Locus: `ir/lower.sio:1120-1185` Knowledge layout,
  `variance_base_regs/variance_value_regs [1024]` `:975-977`,
  `pending_variance_reg :988`. Audit:
  `docs/audit/EPISTEMIC_FABRICATION_DETECT_2026-08-17.md`,
  `docs/audit/MADAROS_FO_CALL_BOUNDARY_DISPATCH_2026-08-18.md`.
- **`pow` has no FO transfer entry on Madaros.** Measured at KL-6 close:
  `hessian_of(pow(x, 3.0), 0, 0)` prints `0.000000` (true `6x = 3.0`;
  `lean_single` prints `3.000000`). The call is opaque to
  `fo_apply_call_transfer` (`ir/lower.sio`, `fo_xfer_seed_transcendentals`),
  so sensitivity and Hessian are both cleared — a structural zero, not a
  diagnostic. Every unary builtin (`sin cos exp atan asin acos tan tanh log
  sqrt`) has a first- and second-derivative entry; `pow` is the only math
  builtin left out.

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
