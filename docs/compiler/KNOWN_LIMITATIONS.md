<!-- docs:meta
topic_id: repo.docs.compiler.known-limitations
authority: repo_only
audience: contributors
last_validated: 2026-10-06
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
| KL-20 | enum variants with payloads: Madaros CLOSED for non-generic enums (tagged cells, 2026-10-30); generic enums do not parse; lean_single unchanged | lean_single (+ generic enums) |
| KL-21 | user fns named like compiler builtins: `pub` residual on Madaros; seed hijacks or rejects | both |
| KL-18 | `Hyper<…>` CPU values: Madaros fail-closed; lean_single prints a wrong value (seed) | both |
| KL-17 | #2773 same-named items across modules: fns resolved per module on both engines; differing same-named structs refused instead of kept apart | both (types) |

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

- **Narrowed (#1706, 2026-10-13).** A call to a *pure helper* (an `f64`
  return, or a struct return when a field of the call is read; `f64`, integer
  or bool parameters, struct parameters by value or by reference whose
  argument names a struct local (`p`, `&p`), plus an optional `self`; no effect beyond
  Div/Panic/Epistemic/Mut; body = `let`s with re-lower-safe right-hand sides
  plus a tail; no loop/match/closure) carries first-order channels: the
  variance lowering re-lowers the helper's body inline in the caller with each
  parameter bound to its argument's value, variance and sensitivities
  (`fo_inline_call_variance_ex` in `self-hosted/ir/lower.sio`). Methods of a
  struct local bind `self` to the receiver (`pk.clearance(eta)` reads `pk.cl0`'s
  field variance); `make(x).cl0` is projected out of the helper's tail struct
  literal. An `if` whose condition is decided at run time merges the two arms'
  variance and sensitivities on that condition (`lower_if_variance_runtime_ref`):
  the result follows the arm actually taken, where the rule it replaces picked
  the then-arm at compile time -- also outside helpers
  (`let y = if n < 0.5 { x } else { 2.0 * x }`). Recursion is unrolled through
  that merge, up to 8 nested inline levels and 96 inlines per outermost call
  site. A struct parameter is bound, like `self`, to the caller's local, so
  its fields keep their variance (`twice(p)`, `twice_ref(&p)`, `p.prod()` with
  no f64 parameter at all). With second order on (a function that calls
  `hessian_of`), the re-lowering carries each argument's Hessian row too, so
  `sensitivity_of(sq(x), 0)` = 2x and `hessian_of(sq(x), 0, 0)` = 2 across the
  call (`kl16c_fo_across_user_fn` now runs on both engines; Madaros read 0.0).
  Witnesses passing on Madaros: every
  `madaros_gum_fo_*` in tests/run-pass (madaros_gum_fo_knowledge_ops, in
  tests/known_failures/, is refused by E245 -- Knowledge (x) Knowledge has no
  Madaros lowering),
  `gum_fo_across_call`, `fo_call_boundary_arity3`, `gum_fo_arity3_boundary`,
  `gum_fo_inline_helper_1706`, `gum_fo_runtime_if_1706` and
  `gum_fo_struct_helpers_1706` (values against C++ oracles). A linear recurrence `if n < 1 { x } else { self(x, n - 1) + x }` carries the channel without unrolling (pin `tests/run-pass/gum_fo_recur_linear.sio`, measured n0 0.090000, n3 1.440000, n12 15.210000, nneg 0.090000). a geometric recurrence `if n < 1 { x } else { 2.0 * self(x, n - 1) }` carries 2^n without unrolling (pin `tests/run-pass/gum_fo_recur_scale.sio`, measured s0 0.090000, s3 5.760000, s12 1509949.440000, sneg 0.090000). a two-channel recurrence `if n < 1 { x } else { self(x, y, n - 1) + y }` carries x + n*y without unrolling (pin `tests/run-pass/gum_fo_recur_mix.sio`, measured m0 0.090000, m3 0.180000, m12 1.530000, mneg 0.090000). the same 2^n written as `self(x, n - 1) + self(x, n - 1)` carries the channel without unrolling (pin `tests/run-pass/gum_fo_recur_twin.sio`, measured t0 0.090000, t1 0.360000, t2 1.440000, t3 5.760000, t12 1509949.440000). the factor is the literal in the body, not only 2 (pin `tests/run-pass/gum_fo_recur_coef.sio`, measured g0 0.090000, g2 7.290000, g12 25418658283.290001). a product recurrence `if n < 1 { x } else { y * self(x, y, n - 1) }` carries x*y^n and both derivatives without unrolling (pin `tests/run-pass/gum_fo_recur_ymul.sio`, measured y0 0.090000, y1 0.610000, y2 5.440000, y12 152504893.440000). a same-channel product `if n < 1 { x } else { x * self(x, n - 1) }` carries (n+1) x^n without unrolling (pin `tests/run-pass/gum_fo_recur_xmul.sio`, measured p0 0.090000, p1 9.000000, p2 506.250000, p12 906586647033691392.000000). an affine recurrence `if n < 1 { x } else { c * self(x, n - 1) + x }` carries the factor (c^(n+1)-1)/(c-1) without unrolling, and n+1 when c is 1 (pin `tests/run-pass/gum_fo_recur_affine.sio`, measured a0 0.090000, a1 0.810000, a2 4.410000, a12 6038323.290000, and c=3 at n=2 is 15.210000). a sum of k identical calls carries k^n (pin `tests/run-pass/gum_fo_recur_ksum.sio`, measured k1 0.810000, k2 7.290000, k12 25418658283.290001, and four calls at n=1 is 1.440000). `if n < 1 { x } else { c * self(x, y, n - 1) + y }` carries both channels (pin `tests/run-pass/gum_fo_recur_affy.sio`, measured y1 0.370000, y12 1677639.690000, and c=3 at n=2 is 7.450000). `if n < 1 { x } else { y * self(x, y, n - 1) + x }` carries the product rule on y (pin `tests/run-pass/gum_fo_recur_chcoef.sio`, measured c1 1.060000, c2 10.660000, c12 513571635.540000). a power whose base is a numeric literal carries b*n*x^(n-1) (pin `tests/run-pass/gum_fo_recur_powbase.sio`, measured b0 0.000000, b2 9.000000, b12 30899047851562496.000000, and base 2 at n=2 is 36.000000). `n <= 0` is the same stop as `n < 1` (pin `tests/run-pass/gum_fo_recur_guard.sio`, measured g8 7.290000, g12 15.210000, and a factor of 2 at n=12 is 1509949.440000). division by a numeric literal is the geometric step 1/c (pin `tests/run-pass/gum_fo_recur_div.sio`, measured d2 0.005625; n=12 is 0.09/4096^2, which print_f64 renders as 0.000000). an else-expression in the recursive value and the parameters is differentiated and repeated n times (pin `tests/run-pass/gum_fo_recur_step.sio`, measured rw12 25418658283.290001, sb12 10.890000, sq2 22500.000000, dv1 0.038125, dv2 0.021250). a block of lets whose recursive call is the previous step is lowered by the ordinary let lowerer inside the loop (pin `tests/run-pass/gum_fo_recur_let.sio`, measured l3 1.440000, l12 15.210000, two lets at n=12 is 6038323.290000, and t*t at n=2 is 22500.000000). the loop carries the Hessian the lowerer already computed (pin `tests/run-pass/gum_fo_recur_hess.sio`, measured h1 2.000000, h2 300.000000, and the same square through a let). exp(L/2)*exp(L/2) keeps the value 12.180705 and the Taylor centre 14.122522, which is p*(1 + s/2); the one-shot exp keeps the lognormal centre 14.285865 and variance 76.639194, while the product keeps the linear variance 47.305391 (pin `tests/run-pass/gum_fo_exp_square_mean.sio`). Four steps of exp(ln P / 4) land on that same Taylor centre, inside the one-shot interval (pin `tests/run-pass/gum_fo_recur_vanthoff.sio`). A step with two delays, past the unroll cap, still drops the missing steps;
  an `if` whose condition calls a function carries the arm taken (the condition is lowered once; pin `tests/run-pass/gum_fo_if_call_cond.sio`, measured zero 0.090000, other 0.360000, direct 0.090000); a `while` carries the loop-carried channel across the call (pin `tests/run-pass/gum_fo_while_call.sio`, measured three 0.810000 and four 1.440000); `for` and `loop`/`break` carry the same channel across the call, and a `let`-bound closure inside the callee is re-lowered at its call (pin `tests/run-pass/gum_fo_for_loop_call.sio`, measured for3 0.810000, loop3 0.810000, closure 0.360000); a `match` on integer, bool or wildcard arms carries the arm taken
  (pin `tests/run-pass/gum_fo_match_call.sio`, measured arm0 0.090000, arm1 0.360000, arm2 9.000000); a `let`-bound closure carries the body's variance (`let f = |y: f64| 2.0 * y; variance_of(f(a))`; pin `tests/run-pass/gum_fo_closure_call.sio`, measured scale 0.360000, square 9.000000, capture 0.090000); a struct reached through a reference local carries the borrowed field (`let r = &p; variance_of(r.cl0)`, `let s = r`, `(*r).cl0`; pin `tests/run-pass/gum_fo_ref_local_field.sio`).
- #1792 (filed 2026-08-17) named two distinct witnesses. **F1**:
  `tests/run-pass/rapamycin_epistemic_adaptive.sio` — `madaros` originally
  printed `var(blood)=0.000000` where `lean_single` shows ~1e-5. **F2**:
  `stdlib/darwin_pbpk/epistemic_pbpk28.sio` TEST 6 — Madaros printed an IEEE
  bit-pattern (~4.6e18) as the AUC confidence, not a variance collapse.
  **F2 is CLOSED**, fixed by PR #1882
  (`d33cf5856b57f3341db9392d263045d090d88ae7`, merged 2026-08-18):
  `Knowledge.confidence` was tagged `is_float: 3` at IR layout instead of `1`,
  so `sitofp` on the raw bits produced the huge value;
  `ir_register_knowledge_layout` now tags it `1`. The two call-boundary pins pass on Madaros (measured 2026-10-15):
  `tests/run-pass/gum_fo_across_call.sio` prints `CALL_var 0.002117` against
  its oracle, and `tests/run-pass/fo_call_boundary_arity3.sio` prints
  `id3_var 1.000000`. Neither file carries `//@ known-failure`. What stays
  open is the list above. Re-measured 2026-09-22: the specific
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
- **`pow` FO / Hessian transfer — CLOSED (KL-11 partial, 2026-09-12; mixed exponent 2026-10-31).**
  `fo_xfer_seed_transcendentals` registers `pow` as kind 9; `fo_apply_transfer_kind`
  emits `∂/∂x`, `∂²/∂x²`, first-order `∂/∂y`, `∂²/∂y²`, and the mixed `∂²/∂x∂y`.
  Witnesses: `tests/run-pass/kl11_pow_fo_hessian.sio`
  (`hessian_of(pow(x, 3.0), 0, 0)` at `x=0.5` → `3.0`) and
  `tests/run-pass/kl11_pow_mixed_hessian.sio` (at `x=2`, `y=3`:
  `Hxx=12`, `Hxy=x^(y-1)*(1+y*ln x)`, `Hyy=x^y*(ln x)^2`). Pin:
  `scripts/ci/madaros_kl11_pow_fo_gate.sh`. FO across arbitrary user `fn`
  bodies remains open.


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
- **KL-16g — CLOSED (seed, x86, #2776 part 2).** First-order variance
  through struct fields (literal, nested, rebind, alias), struct-returning
  calls, by-value and by-reference struct parameters, methods, constructor
  lets and a returned `Knowledge` value: a field-shadow table keyed by the
  struct's region carries each f64 field's variance and sensitivities and
  crosses calls in tagged BSS blocks. Channels 8..31 give the 9th..32nd
  measured input of a function its own first-order channel; when the
  program asks for an uncertainty and has nine or more `measure(` sites
  (`SX_XCALL`) they also cross direct calls, method calls and struct fields.
  A call nested in a later argument no longer overwrites the earlier
  arguments' shadows, and a closure literal no longer wipes the shadows of
  the locals declared before it. Pin: `tests/run-pass/gum_fo_struct_fields_channels_2776.sio`
  (analytic values: `tests/oracles/gum_fo_struct_fields_channels_2776.cpp`).
- **Residual (OPEN).** HSHADOW multi-pair interproc; HSHADOW pairs other
  than `[0,0]` through if/else; a64 `atan2`/`pow` AD. Calls through a
  function value or a closure carry shadows like direct calls (since #2776
  part 2; they used to keep the LAST ARGUMENT's), and a closure's env carries
  each captured f64's variance and channels 0-7 (not channels 8..31).
  Calls with more than 16
  arguments carry shadows for the first 16. The #2776 part 2 work is x86
  only; the a64 mirror is unchanged.
- Channel-at-`.value` semantics (`MEAS_KNOW_IDX`,
  `formal/ChannelAssignmentSemantics.lean`) are a model, not a defect —
  see the history snapshot for the KAS-1 rationale.

### KL-20 — enum payload variants (P1.3)

- **Madaros: CLOSED for non-generic enums (2026-10-30, `fix/kl20-20261030`).**
  A payload enum value is a tagged cell `[tag, payload words..]` sized for the
  largest variant; layout and the alternatives not taken are in
  [ENUM_PAYLOAD_LAYOUT.md](ENUM_PAYLOAD_LAYOUT.md). Construction (call and
  struct-literal form), unit variants, `match` with payload binding (by value
  and through `&`), nested variant / literal / `Some(..)` sub-patterns,
  or-patterns, guards, struct and array and `Box<Self>` payloads, passing,
  returning and methods all lower. The checker types each payload binding with
  its declared field type, checks constructor arity (E277) and argument types
  (E009), pattern arity (E278), refuses `==` on a payload enum (E279) and a
  match that misses a variant (E280). E264 now only refuses a payload field that
  is not one word (tuple, f128/f256, Knowledge<T>). Pins:
  `tests/run-pass/enum_payload_{scalar,aggregate,nested}.sio`,
  `tests/compile-fail/enum_payload_*.sio`. The repro below prints `12`.
- **Still open:** generic enums (`enum Opt<T> { S(T), N }` is a parse error on
  Madaros: the enum header takes no type parameters); builtin `Result<T, E>`
  (`Ok(..)`/`Err(..)` are untyped in the checker and lower to a trapping stub);
  lean_single (seed) has no payload representation. History of the refusal
  follows.

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
  took the `None` arm for `Some(42)`; fixed in the same branch. `Some(0)` is not `None` (#2787; pin
  `tests/run-pass/option_scalar_match_issue2787.sio`, measured
  `ISSUE_2787_OK`).

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
  - **`pub fn` fixed for Madaros (2026-10-23), single definer.**
    `builtin_shadow_apply_items` now collects `visibility.kind == 0 || 1` and
    renames an intercepted `pub fn` to `<name>__user` like a private one;
    `private_fn_identity_apply` records each (base name, definer) pair and the
    #2773 pub phase rebinds every importer that spells the base name in a
    `use` item -- by name or through a glob -- through the same table
    (`PFI_T_USER` marks the entries that spell `__user` instead of
    `__m<idx>`). Witness: leaf `pub fn measure(x: i64) -> i64 { x + 1000 }`,
    main `use ...::{measure}` + `measure(5) == 1005` → before: `check` rc=1
    with `expected Knowledge<i64>` (E004/E036); after: `check` rc=0 and `run`
    exit 0. Glob form (`use ...::*`) also exit 0. Single module `pub fn
    measure` + own call: before rc=1 `use of undeclared variable
    measure__user`, after rc=0.
  - **Two defects found while doing that, both fixed here.**
    (1) **#2773 broke the private half it landed beside.** Its guards
    (`PFI_T_DEF == 1`, `PFI_T_TGT == PFI_CUR_MOD`) in `pfi_rename_item_def`
    were never matched by the shadow's table entries, so only the
    *references* were renamed and the definition kept the builtin spelling:
    every module with a private builtin-named fn failed with `use of
    undeclared variable <name>__user`. Measured on the pristine `3ef3dbaa6e`
    binary: `user_fn_shadows_builtin_measure.sio` rc=1 with that message;
    after this change rc=0 with both PASS lines. The shadow now marks its
    entries as definitions of module 0 carrying the `__user` spelling.
    (2) **`run_check_mode` never ran the identity pass** (`compiler/main.sio`):
    it loaded the modules, ran only `module_struct_identity_check`, and
    checked -- so the checker saw the un-rebound `measure` and intercepted the
    builtin before compilation could rewrite it. It now calls
    `module_frontend_identity_pass` (use map + struct check + identity), the
    same order the compile path uses.
  - **Residual (OPEN), Madaros:** two *loaded* modules that each export a
    `pub fn` with the same builtin name, plus a module that imports it -- the
    rename cannot pick a target at a bare call site, so the compile stops with
    ``error[builtin_shadow]`` naming the cause (it used to type the call as
    `Knowledge`). Merely loading both modules is fine: the pub phase renames
    the second `<name>__user` apart. `f128_from_limbs` / `f128_to_lo` /
    `f128_to_hi` stay deliberately excluded because
    `stdlib/math/softfloat_f128.sio` defines them as the implementation the
    intercept stands for (a user one fails to build). The renamed spelling
    `<name>__user` appears in diagnostics. Trace:
    `SOUNIO_TRACE_BUILTIN_SHADOW=1` prints the census and the rebind
    decisions.
  - **Residual (OPEN), lean_single (seed, not changed — a fix needs a seed
    refresh):** 26 of 65 wrong. Silently wrong value (builtin ran, printed
    the argument `5` or `0`): `abs`, `f64_to_bits`, `lift_knowledge`,
    `prove_robust`, `validate_manifest`, `gpu_thread_id_x`, `slice_len`.
    Rejected: `print_int`, `print_char`, `print_f64`, `get_arg`, `str_eq`,
    `seq_new`, `seq_push`, `variance_of`, `sensitivity_of`, `gpu_barrier`
    (E001), `hessian_of`, `seq_set` (E200), `seq_len` (P0003); no output:
    `seq_count`, `seq_get`, `str_len`; failed to compile: `acknowledge`,
    `require_confidence`, `f128_to_lo`.
  - **Runtime callees named by the lowerer (2026-10-30).** The lowerer also
    reaches runtime functions by bare name for operators, not only calls:
    `println(f64)` -> `print_f64`, and `string == string` / `!=` / a string
    literal `match` arm -> `str_eq`. A user fn of that name was bound instead
    (`stdlib/str/lib.sio`'s `pub fn str_eq(a: &Str, b: &Str)` received two
    string handles). Both names are on the shadow list now, so the user fn is
    `<name>__user` and the operator keeps the runtime.
- Pins: `tests/run-pass/user_fn_shadows_builtin_measure.sio`,
  `tests/run-pass/user_print_f64_does_not_capture_println.sio`,
  `tests/run-pass/user_str_eq_does_not_capture_string_eq.sio`,
  `tests/run-pass/user_fn_shadows_builtin_names.sio`,
  `tests/run-pass/user_fn_shadows_builtin_pub_measure.sio`,
  `tests/multimodule/user_fn_shadows_builtin_pub_main.sio` (+ `_leaf`),
  `tests/multimodule/user_fn_shadows_builtin_pub_glob_main.sio` (+ `_leaf`),
  `tests/compile-fail/builtin_shadow_local_binding_refused.sio`.
  A `requires: madaros` pin is skipped unless `SOUNIO_MADAROS_AVAILABLE` is
  set, and the Madaros CI job runs only the tests a diff touches -- which is
  why the #2773 regression above went unseen. The four `run` cases are
  therefore also listed in `tests/madaros/multimodule_witness/manifest.tsv`,
  run unconditionally by `scripts/ci/madaros_multimodule_witness.sh`
  (16/16 witnesses, measured 2026-10-23).
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

### KL-17 — #2773: same-named items in different modules

- **Madaros, functions — CLOSED for plain `pub` and private fns (#2773).** A
  module's unqualified call to its own function used to bind to a same-named
  `pub fn` of another, first-loaded module (the merged IR keys functions by
  bare name): measured `-14.5` expected, `3.0` printed, on both engines.
  `self-hosted/compiler/private_fn_identity.sio` (pub phase) now renames
  every non-first plain-`pub` definer to `N__m<idx>` and rebinds each
  importer's unqualified, module-qualified and import-list references through
  that importer's own `use` items (explicit import beats glob; own definition
  beats both). Two explicit imports of different definers, or two globs of
  different definers neither of which is the first-loaded one, are refused
  with `error[module_name_resolution]`. Pins:
  `tests/run-pass/madaros_module_local_name_resolution.sio`,
  `tests/run-pass/madaros_module_local_name_import_rebind.sio`,
  `tests/run-pass/madaros_module_local_name_glob.sio`,
  `tests/compile-fail/madaros_module_name_ambiguous_import.sio`.
- **Madaros, structs — refused, not resolved.** Two loaded modules that
  define one struct name with *different* definitions are refused with
  `error[module_struct_identity]` (struct layouts and the checker's struct
  table are bare-name keyed, so they would be one type). Identical copies are
  accepted. Pins: `tests/compile-fail/madaros_module_same_struct_differs.sio`,
  `tests/run-pass/madaros_module_same_struct_identical_copies.sio`. Residual:
  keeping two different same-named structs apart (module-qualified type
  identity) is not implemented; same-named enums, type aliases and globals are
  not checked at all.
- **Madaros, residual fn shapes.** A name referenced by a module that does not
  import it from any definer keeps the first-loaded binding (unchanged); a
  re-export (`pub use`) is not followed; restricted-public (`pub(crate)` …)
  collisions are refused as before.
- **lean_single — CLOSED for the same shapes (#2773).** `lmr_resolve` in
  `self-hosted/compiler/lean_single.sio` runs on the token stream right after
  lexing and applies the same rules: a free fn defined at top level in two or
  more modules keeps its name in the first-loaded definer and is renamed
  `N__m<k>` in every other definer k; each module's references bind through
  its own definition, then its explicit `use` imports, then its globs (only
  `pub` definers count as import sources); two explicit imports of different
  definers, or globs of two non-first definers, are refused with
  `error[module_name_resolution]`; `q::N(..)` resolves through the `use` whose
  last segment is `q`. `lmr_struct_identity` refuses same-named structs with
  different definitions (`error[module_struct_identity]`). The pins above now
  run on both engines. Same residuals as Madaros (no import from any definer
  keeps the first-loaded binding; `pub use` is not followed); a bare fn value
  is rebound only in a module that declares no local of that name.

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
