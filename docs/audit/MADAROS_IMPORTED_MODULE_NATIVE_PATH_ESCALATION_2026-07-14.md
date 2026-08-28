<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-module-native-path-escalation-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-imported-module-native-path-escalation-2026-07-14
-->

# Escalation — the imported-module native path is the single blocker gating real-world Sounio

**Date:** 2026-07-14
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default **native** engine)
**Owner:** CODEX-2 (`self-hosted/` — imported-module IR lowering / merge / native emission)
**Type:** consolidated escalation (references existing dispatches; does not duplicate them)

## TL;DR

Every real Sounio program imports the stdlib. On the default **native** engine, the
imported-module path has a small number of distinct, reproducible defects that
together make composing real modules **fail or silently miscompile**. Building one
modest I/O vertical surfaced **four** distinct defects on this path (D1–D4 below;
a fifth, `str_slice` ignoring its length argument, is a general builtin bug that
also reproduces single-file, so it is out of scope here). They are not four
verticals' worth of work — they are one subsystem (imported-module IR lowering →
merge → native emit), and
that subsystem is exactly what the active `codex/*-ir-*` rebuild (SOIR core, Place
IR, `IrModuleArena` v2, ref-field projections) is reconstructing. **This is the
one fix that unblocks I/O readers, cross-module reuse, correct GUM coverage
factors, and the entire `Knowledge<T>` / uncertainty-propagation core at once.**

## Defect catalogue (distinct roots, all on the imported-module native path)

| # | Defect | Symptom | Repro / dispatch | Blocks |
|---|---|---|---|---|
| D1 | **`f64 → i64/i32` cast whose operand is an f64 *parameter* is a bitcast**, not a truncating convert — **general, reproduces single-file** (not imported-only; see 2026-07-16 update below) | ~~silent wrong numbers~~ **CLOSED** (#983 + #1252; Wave10 trust gate) — `f(4.172)→4`; finite-dof `gum_k95≈2.776` | `MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST_2026-07-14`; **root-caused in #983**; **joint land #1252** | ~~correct GUM coverage~~ **unblocked** — gated by `scripts/epistemic_trust_gate.sh` Section A |
| D2 | **`&local_array` passed to a builtin gets a wrong base pointer** | SIGSEGV / garbage bytes | `DATA_IO_TRILHA_B_BUILTIN_BUFPTR_DISPATCH_2026-07-14` | `read_file`, `write_file`, `str_from_bytes` → all Data I/O **readers** and the file sink |
| D3 | **Multi-module native lowering fails** — segfault in `lower_array` dep-lowering, or thin-link `rc=12` | native compile fails whenever the program's dep closure has ≥2 modules or a module `use`s another | this doc's witnesses (`knowledge`, `propagate`, `order_spread_exact`, `uncertain_eq`); extends `MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30`, `MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14`, `MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22` | cross-module reuse (`data::csv` + `epistemic::gum`); `Knowledge<T>`, `propagate`, and any module with `use` deps |
| D4 | **named `use m::sym` E137 + `print_f64` E137** in importing programs | type-check rejects valid code | `MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13` | selective imports; float printing in importing programs |

**Tracking issues:** D1 → #932 (narrow) + **#983 (general root cause)**, D2 → #933,
D3 → #901 (+ thin-link variant #921), D4 → #862. **D5 (new) → #986** — see update.

D2 is a silent/local mis-lowering; D3 is the structural multi-module path; D4 is
type-check-level. D2/`&buf`→builtin and long `if`-chains do not reproduce single-file.
**Correction (was wrong for D1):** D1 *does* reproduce single-file — the earlier
"`f64 as i64` works single-file" check cast a **local**/literal, which is fine; the
bug fires only when the cast operand is a **parameter**.

## Update 2026-07-16 — D1 root-caused (#983), and its fix is BLOCKED by a new defect (#986)

**Root cause of D1 (#983):** `lower_cast_expr_ref` (`self-hosted/ir/lower.sio` ~L9281)
emits `IrFloatToInt` only when the operand is a detected float source
(`lookup_local_scalar_kind == 2`). f64 **parameters carry `scalar_kind 0`**: path A
(`lowerer_lower_fn_params_mut`) recorded the kind via `(*lo) = (*lo).bind_local_scalar_kind(...)`
— a by-value `Lowerer` RMW through a `&!` pointer, dropped by the by-value-aggregate-store
miscompile (the file already carries `_mut` analogues for exactly this). Locals reassign a
`var lo`, so they survive → locals work, params don't.

**⚠️ The one-line fix is UNSAFE on its own → new defect D5 (#986).** Giving params
`scalar_kind = 2` also activates the *other* scalar_kind consumers — **println-dispatch**
and **variance-shadow tracking** — never exercised for parameters before. A deterministic
clean-vs-fix full-suite diff showed +10 pass (all `dissertation_pbpk28_parity_ref_*`) but
**−20 fail with ZERO casts** (`autodiff_tape_basic`→HANG, `test_fem`/`matnm_test`→SEGFAULT,
`arima_levinson_ar2`→HANG; +3 thin-link `rc=12`). So **f64 parameters are systemically
under-supported (cast + println + variance-shadow)**, long masked by the `scalar_kind=0` bug.

**Fix ordering:** land **#986** (println/variance-shadow of f64-param values, + thin-link
headroom) *before/with* the D1/#983 `scalar_kind` param fix. A **safe stdlib workaround**
(no compiler change) exists for the GUM-critical site: arithmetic-route the cast
(`let d = dof + 0.0; d as i64`) — verified on the clean compiler. Per owner decision this is
**dispatch-only**; no compiler/stdlib code shipped.

**Aside:** the D3 "multimodule compose" symptom for `data::csv` + `epistemic::gum` is **stale**
on current source (compiles + runs); `Knowledge<T>`/`propagate` import still fail (D3 stands).

## Impact map — what each fix unblocks

- **D3 (multi-module lowering)** — highest leverage. Unblocks *importing the stdlib
  at all* from a native program: cross-module reuse, `Knowledge<T>` (the flagship
  type, currently segfaults on import), the `propagate` uncertainty layer, the CPC
  `order_spread_exact` N=4 receipt on native, and every real multi-module program.
- **D1 (f64→i64 bitcast)** — highest *risk*: silent, not a crash. Unblocks correct
  finite-sample coverage factors and any imported numeric routine that casts to int.
- **D2 (`&buf`→builtin)** — the Data I/O **file** gate (readers + file writers);
  the stdout writers already shipped around it (PR #918).
- **D4 (named-import / print_f64)** — quality-of-life for importing programs.

Recommended attack order: **D3 → (D5+D1) → D2 → D4** (structural enabler first, then
the silent-corruption root — but **D5/#986 must land with D1/#983**, else the corrected
float-marking hangs/segfaults the 20 zero-cast programs — then the I/O sink, then
ergonomics). D1 and D3 both bottom out in the imported-module lowering/merge stage the
`codex/*-ir-*` rebuild owns.

## Trust boundary this currently imposes

`docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md` (Wave10 update): full GUM
(value + `u_c` + finite-dof `k95`/`U95`/`U99`), free+method `Epistemic`,
`propagate` delta-method + value-style MC, correlation/p-box are usable under
native import. Residual fragile forms: generic `monte_carlo` fn-ptr, exclusive-ref
xoshiro, language `Knowledge<T>` generics. A real PBPK/GUM pipeline can use
default Madaros for those promoted surfaces — not lean_single-only.

## Verification hooks (how to confirm each fix)

These gates already exist and will flip the moment a defect is fixed — no manual
re-checking:

- **D1:** **CLOSED Wave10.** `scripts/epistemic_trust_gate.sh` Section A now
  **gates** finite-dof `k95i=2776` via Type-A-dominant `witness_gum_k95.sio`
  (`gum_combine2(98.3, gum_type_a(0.30,5), gum_type_b_uniform(0.001))`). The
  pre-Wave10 Section B trip-wire expected 1960 on a Type-B-dominant budget and
  could never flip (k95=1.960 was correct there). Issues #932/#983 closed;
  joint land #1252.
- **D5 (#986):** the −20 zero-cast HANG/SEGFAULT programs above (`test_fem`, `matnm_test`,
  `arima_levinson_ar2`, `autodiff_tape_basic`) must run correctly once params are float-marked.
- **D2:** the byte-exact round-trip in the D2 dispatch (`str_from_bytes`/`write_file`
  + `read_file`); re-enables Data I/O readers.
- **D3:** `scripts/epistemic_trust_gate.sh` Section C prints "now COMPILES" when
  `knowledge` / `order_spread_exact` become native-importable.
- **Regression:** `scripts/data_io_csv_gate.sh`, `scripts/data_io_json_gate.sh`
  (byte-exact writers) must stay green across the IR rebuild.

## Ask

Fold D1–D5 into the `codex/*-ir-*` imported-module lowering rebuild as explicit
acceptance criteria, in the D3→(D5+D1)→D2→D4 order, and wire the verification hooks
above into that lane's CI so the trust map updates automatically as fixes land.
**D1 (#983) and D5 (#986) must land together** — the D1 float-marking fix is unsafe
until D5's println/variance-shadow handling of f64-param values is fixed.

## AI disclosure

Consolidation, repros, and impact analysis by AI agent (Claude) under human
direction, on Madaros v0.80.0. Cited dispatches carry their own math-review
offloads where applicable. No `self-hosted/` sources were modified. GAIDeT-ICMJE 2025.
