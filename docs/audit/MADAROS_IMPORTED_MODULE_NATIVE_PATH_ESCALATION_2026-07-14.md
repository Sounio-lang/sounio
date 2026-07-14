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
modest I/O vertical hit five of them. They are not five verticals' worth of work —
they are one subsystem (imported-module IR lowering → merge → native emit), and
that subsystem is exactly what the active `codex/*-ir-*` rebuild (SOIR core, Place
IR, `IrModuleArena` v2, ref-field projections) is reconstructing. **This is the
one fix that unblocks I/O readers, cross-module reuse, correct GUM coverage
factors, and the entire `Knowledge<T>` / uncertainty-propagation core at once.**

## Defect catalogue (distinct roots, all on the imported-module native path)

| # | Defect | Symptom | Repro / dispatch | Blocks |
|---|---|---|---|---|
| D1 | **`f64 → i64/i32` cast in an imported-module body is a bitcast**, not a truncating convert | silent wrong numbers (e.g. `4.172 as i64` → `4616383272838735331`) | `MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST_2026-07-14` | correct GUM coverage `k95`/`U95`/`U99`; any imported fn casting f64→int (confidence scaling, indices) |
| D2 | **`&local_array` passed to a builtin gets a wrong base pointer** | SIGSEGV / garbage bytes | `DATA_IO_TRILHA_B_BUILTIN_BUFPTR_DISPATCH_2026-07-14` | `read_file`, `write_file`, `str_from_bytes` → all Data I/O **readers** and the file sink |
| D3 | **Multi-module native lowering fails** — segfault in `lower_array` dep-lowering, or thin-link `rc=12` | native compile fails whenever the program's dep closure has ≥2 modules or a module `use`s another | this doc's witnesses (`knowledge`, `propagate`, `order_spread_exact`, `uncertain_eq`); extends `MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30`, `MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14`, `MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22` | cross-module reuse (`data::csv` + `epistemic::gum`); `Knowledge<T>`, `propagate`, and any module with `use` deps |
| D4 | **named `use m::sym` E137 + `print_f64` E137** in importing programs | type-check rejects valid code | `MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13` | selective imports; float printing in importing programs |

D1–D2 are silent/local mis-lowerings; D3 is the structural multi-module path; D4 is
type-check-level. All four are the *imported-module* path — none reproduce in a
single-file `main()` (verified: the same `f64 as i64`, `&buf`→builtin, and long
`if`-chains all work correctly single-file).

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

Recommended attack order: **D3 → D1 → D2 → D4** (structural enabler first, then the
silent-corruption root, then the I/O sink, then ergonomics). D1 and D3 both bottom
out in the imported-module lowering/merge stage the `codex/*-ir-*` rebuild owns.

## Trust boundary this currently imposes

`docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md` classifies the fallout: only
self-contained, cast-free epistemic modules (GUM point+`u_c`, correlation,
p-box, covariance) are usable under native import today; `Knowledge<T>`, the
propagation layer, and GUM coverage intervals are not. A real PBPK/GUM pipeline
must therefore run under **lean_single**, not the default native engine.

## Verification hooks (how to confirm each fix)

These gates already exist and will flip the moment a defect is fixed — no manual
re-checking:

- **D1:** `scripts/epistemic_trust_gate.sh` Section B prints "coverage factor may be
  FIXED" when `gum_k95` stops returning 1960; the `f64→i64` minimal repro in the
  D1 dispatch.
- **D2:** the byte-exact round-trip in the D2 dispatch (`str_from_bytes`/`write_file`
  + `read_file`); re-enables Data I/O readers.
- **D3:** `scripts/epistemic_trust_gate.sh` Section C prints "now COMPILES" when
  `knowledge` / `order_spread_exact` become native-importable.
- **Regression:** `scripts/data_io_csv_gate.sh`, `scripts/data_io_json_gate.sh`
  (byte-exact writers) must stay green across the IR rebuild.

## Ask

Fold D1–D4 into the `codex/*-ir-*` imported-module lowering rebuild as explicit
acceptance criteria, in the D3→D1→D2→D4 order, and wire the four verification hooks
above into that lane's CI so the trust map updates automatically as fixes land.

## AI disclosure

Consolidation, repros, and impact analysis by AI agent (Claude) under human
direction, on Madaros v0.80.0. Cited dispatches carry their own math-review
offloads where applicable. No `self-hosted/` sources were modified. GAIDeT-ICMJE 2025.
