<!--
  Forensic dispatch (CLAUDE.md §8): evidence + proposed fix recorded BEFORE any
  self-hosted/ edit. No compiler patch is applied by this document.
-->

# DISPATCH EFF.2 — Make per-module target-check clean (the EFF.1 coverage-gate prerequisite)

**Opened.** 2026-06-08.
**Predecessor.** EFF.1 Phase B (`PHASE_B_RESULTS.md`, `phase_b_raw.json`).
**Class.** Checker seeding + diagnosis; `self-hosted/check/{mod.sio,check.sio}` +
`self-hosted/compiler/module_frontend.sio`. Soundness-sensitive.
**Priority.** P3 — prerequisite for the EFF.1 coverage gate; itself partly downstream of the
271-wall (see §4). Not on any current critical path.
**Branch.** `claude/effects-enforcement`.
**Status.** OPEN — spec only (diagnosis-first).

---

## §0 — Why this exists

EFF.1's coverage gate (check every module body once with imports seeded, under `toggle = 2`) needs
each module to **check cleanly as a target**. Phase B found only **16 / 107** do: **66 TYPEFAIL,
25 PARSEFAIL**. EFF.2 is about turning the 66 TYPEFAIL into clean checks (or genuine, attributable
errors) so the gate measures effects, not seeding artefacts.

---

## §1 — Evidence (probes, fixed binary `souc-eff-warn.elf`, warn-mode)

**Correction to `PHASE_B_RESULTS.md` §"dominant failure modes":** that doc said the TYPEFAILs were
because `checker_boot4_seed_imported` "seeds function signatures but not imported type definitions."
**That is wrong** — disproven below. Imported structs and enums *do* seed. The real decomposition is
three distinct causes, two of which are not a seeding bug at all.

### 1.1 Imported structs/enums seed correctly; type **aliases** do not

2-module probe (`main` imports `lib` defining a struct `Foo`, enum `Col`, alias `Bar`):

| Imported type used by target | Result |
|---|---|
| `struct Foo` | **`check: OK`** — resolves |
| `enum Col` | **`check: OK`** — resolves |
| `type Bar = i64` (alias) | **`error[E008]`, type checking failed** — NOT resolved |

Locus: `checker_collect_item_inplace` (`check.sio:2726`) wires `ItemFn`/`ItemStruct`/`ItemEnum`/
`ItemImpl`, but its `else` branch is an explicit no-op for `ItemUse` / **type alias** / session /
policy ("*mut collectors pending"). So imported **type aliases are never seeded** → E008. This is a
genuine, narrow seeding gap.

### 1.2 The 271-wall poisons importers transitively (confirmed)

Probe: a clean `main` importing a `lib` that contains a 271-wall trigger (`let module = 1`, i.e.
`module` as an identifier):

| Check | Result |
|---|---|
| `lib3` alone | `Parse failed: 2 errors` (271-wall) |
| clean `main3` importing `lib3` | **`Type check failed`** (poisoned) |
| clean `main4` importing a clean `lib4` | `check: OK` |

Mechanism: `module_frontend_import_typecheck_main` (`module_frontend.sio:3551`) BFS-seeds imports and
has a **C0 guard** — `if parser_last_error_count() != 0 { return 1 }` — so if **any** transitively
imported module fails to parse, the *importer's* check returns TYPEFAIL (correctly: you can't seed an
unparseable module's signatures). Since the 25 PARSEFAIL modules include `ir/ir.sio`,
`check/check.sio`, `check/mod.sio`, all of `ir/*`, and `lexer/{cursor,span,tables}` — which almost
every other module imports — **this is very likely the dominant cause of the 66 TYPEFAILs.** Such
TYPEFAILs show `sample_error = ""` (C0 returns before body-checking; 7 in Phase B) or fail before the
real effect gap is visible.

### 1.3 Residual unexplained: E015 "unknown struct type" (×16), E016, some E001

These modules' closures parsed (no C0 trip) yet a referenced struct is still unknown. **Not yet
pinned.** §1.1 shows direct imported structs resolve, so candidates are: structs beyond the
256-entry BFS cap; generic/parameterised types; name-keying mismatches; or struct fields whose types
are themselves unseeded aliases (§1.1). Must be diagnosed, not guessed.

---

## §2 — Fix directions (by cause)

| Cause | Share (est.) | Owner |
|---|---|---|
| (a) transitive 271-wall poisoning (§1.2) | likely majority | **NOT EFF.2** — the 271-wall (keyword demotion, concurrent session). EFF.2 inherits this dependency. |
| (b) imported type aliases not seeded (§1.1) | the E008 class | **EFF.2** — add a `*mut` type-alias collector to `checker_collect_item_inplace`'s `else` branch (and the by-value mirror), registering imported aliases into the type-alias table. |
| (c) residual E015/E016 (§1.3) | the E015×16 / E016×8 classes | **EFF.2, diagnosis-first** — attribute precisely before any fix; the "structs not seeded" hypothesis is already disproven. |

**Out of bounds:** recursive import body-checking (rejected in EFF.1 §2-Fa); the 271-wall itself;
M4 effectful-HOF.

---

## §3 — Attack plan

- **Phase 1 — attribute the 66 (no compiler edit).** For each TYPEFAIL module, pre-scan its
  transitive import closure for any PARSEFAIL (271-wall) member. Split the 66 into
  **271-poisoned** (closure contains a PARSEFAIL → not EFF.2's bug) vs **genuine** (closure parses
  clean → real seeding/resolution gap). This sizes (b)+(c) honestly and tells us how much of EFF.1
  is actually gated only on the 271-wall.
- **Phase 2 — type-alias seeding (b).** Add the `*mut` alias collector at `check.sio:2726` `else`;
  verify with the §1.1 probe (imported `type Bar` resolves) and that `release_gate.sh` stays green.
- **Phase 3 — pin + fix residual (c).** Diagnose the genuine-bucket E015/E016 (cap? generics?
  field-type order?) and fix or defer with the exact gap. No silent miscompile, no false-reject.
- **Phase 4 — re-measure.** Re-run EFF.1 Phase B; the OK count should rise to the
  271-wall-free module set; remaining TYPEFAILs should be exactly the 271-poisoned ones.

---

## §4 — Dependencies / risks

- **Most of the 66 is probably the 271-wall, not EFF.2.** Phase 1 confirms the split; if (a)
  dominates, EFF.2's own surface (b)+(c) is small and the real blocker remains the 271-wall.
- **Soundness:** alias/type seeding must register the *interface* only (no body check); must not
  introduce false-accepts of ill-typed cross-module use (the #2 design property). The by-value
  Checker mirror is historically lossy (SRET/double-deref) — prefer the `*mut` spine.
- EFF.1's gate cannot be completed until EFF.2 (b)+(c) **and** the 271-wall are both done.

---

## §5 — Deliverables

1. Phase-1 attribution table (271-poisoned vs genuine) — sizes the real EFF.2 surface.
2. `*mut` type-alias collector (Phase 2).
3. Residual diagnosis + fix-or-defer (Phase 3).
4. Re-run Phase-B coverage delta (Phase 4); SYNTHESIS.md on resolution.

**END OF DISPATCH (spec only — no `self-hosted/` edit applied).**
