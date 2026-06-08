<!--
  Forensic dispatch (CLAUDE.md §8): evidence + proposed fix recorded BEFORE any
  self-hosted/ edit. No compiler patch is applied by this document.
-->

# DISPATCH EFF.1 — Per-module effect-coverage gate for the mandatory-effects flip

**Opened.** 2026-06-08.
**Predecessor evidence.** `artifacts/effects/measure_reverify_2026-06-08/REVERIFY_REPORT.md`
(MEASURE re-verification + toggle=2 rebuild) and memory
`project_effects_enforcement_baseline_2026-06-07.md`.
**Class.** Build/gate tooling + a small, contained generalisation in
`self-hosted/compiler/module_frontend.sio`. **Not** a checker-semantics change.
**Priority.** P3 — required before the mandatory-effects flip (`effects_enforcement_mode() -> 2`)
can be trusted for the compiler's own source; **not** on any current critical path (see §4).
**Branch.** `claude/effects-enforcement` (M2 toggle lives here; parser prerequisite already landed
as `526b0091f`).
**Status.** OPEN — spec only.

---

## §0 — Constraint / framing

M2 landed source-tagged effect enforcement behind a free-function toggle
`effects_enforcement_mode()` (`self-hosted/check/check.sio:3341`): `0`=dark, `1`=warn (`W035`,
shipped), `2`=error (`E035`, the mandatory flip). The flip's value proposition is that the compiler
*itself* obeys its own effect discipline. That requires every function in the compiler's source to
be effect-checked **somewhere** in the validation pipeline. This dispatch is about achieving that
**coverage** — it is **not** about changing how the checker treats imports.

---

## §1 — What was located (evidence)

### 1.1 The mechanism (code)

- `preflight_multimodule_frontend` (`module_frontend.sio:3602`) is the `--check` entry. For a file
  with imports it calls `module_frontend_import_typecheck_main` then a count-only summary.
- `module_frontend_import_typecheck_main` (`module_frontend.sio:3551`):
  `checker_boot4_alloc_seed_main(main.items)` → BFS over transitively-imported files calling
  `checker_boot4_seed_imported(checker, dep.items)` (signatures only) → checks the **target (main)**
  body. **Imported module bodies are seeded as interfaces, never re-checked by the importer.**
- This is **separate compilation, working as designed** — a module is enforced when it is the
  *target*; importers trust its validated interface. It is **not** "under-enforcement" and must
  **not** be "fixed" by recursive import body-checking (see §2, rejected direction).

### 1.2 Behaviour confirmed by the toggle=2 rebuild (2026-06-08)

Built `effects_enforcement_mode() -> 2` via `souc-build-lock.sh ./bin/souc main.sio` →
`artifacts/effects/souc-eff-error.elf` (retained; toggle reverted to 1 after):

| Probe (error-mode binary) | Result | Interpretation |
|---|---|---|
| `println` w/o `with IO` at file top | `error[E035]`, check fails | a **target's** body **is** enforced |
| synthetic 200 clean fns + end-violation | `error[E035]`, check fails | enforcement reaches the whole target body |
| `main` (no `module` header) imports a `module`-headed `lib` w/ a print-no-IO fn; `lib_ok` resolves | `check: OK`, 0×`E035` | imported body seeded-as-interface, not re-checked (separate compilation) |

> **Retracted data point:** an earlier `--native-v2-compile` run of `main`+`lib` reporting
> "E035 = 0" is **void** — that build died with `ir_summary_failed` *before* any check fired. It is
> not evidence about enforcement. The basis for §1.1 is the code, not that run.

### 1.3 Prerequisite already landed

`parse_module_item` previously swallowed the rest of any `module`-headed file (looped on a
`Newline` token the lexer never emits) → a module checked as a *target* parsed empty. **Fixed in
`526b0091f`** (consume `module` + path via `parse_type_path`). With that fix, `--check <module>`
now parses and checks each module's body (e.g. `knowledge_context.sio` → 158 items, real `W035`
surfaced). This is what makes a per-module coverage gate feasible.

---

## §2 — Fix directions

### (Rejected) Fa — recursive import body-check

Make `import_typecheck_main` recursively *body*-check every imported module. **Rejected:** re-checks
shared dependencies once per importer (e.g. `ir::ir` imported by ~dozens of modules → checked
dozens of times), conflates *using* a module with *checking* it, and reverses the deliberate #2
import-typecheck design. Wrong model, not merely expensive.

### (Recommended) Fb — generalise seed-and-check-target + a coverage gate

Two contained pieces:

1. **Generalise the target.** `import_typecheck_main(main_path)` already does exactly the sound
   per-module operation — *seed this file's transitive imports as signatures, then body-check this
   file*. It is only hardwired to `main`. Expose
   `module_frontend_import_typecheck_target(target_path)` (same body; `main_path` → `target_path`),
   leaving `import_typecheck_main` as a thin wrapper. **No checker-semantics change.**
2. **Coverage gate.** A script/driver that enumerates the module set (the transitive `use` closure
   from `main.sio` — the 107-file live set, list at
   `artifacts/effects/measure_reverify_2026-06-08/live_module_set_107.txt`) and runs the target-check
   on **each** module under `toggle = 2`, failing if any module emits `E035`. Each module body is
   checked exactly once (as its own target), with its imports seeded — so no module is re-checked
   per-importer, and none TYPEFAILs on unresolved imports (the failure mode of naive
   `--check <file>`, e.g. `knowledge_context` → 184 errors standalone-without-seeding).

**Out of bounds for EFF.1:** changing effect inference, row-polymorphism / effectful-HOF support
(M4 — not needed for the first-order compiler per the MEASURE finding), or annotating the gap.

---

## §3 — Attack plan

- **Phase A — generalise (15 min).** Rename/wrap `import_typecheck_main` → `_target`; confirm
  `import_typecheck_main(p)` behaviour is byte-identical. Rebuild via
  `souc-build-lock.sh ./bin/souc main.sio`; `release_gate.sh` must stay 20/20.
- **Phase B — measure coverage cleanly (toggle=1, warn).** Run the target-check over all 107
  modules and tabulate: per module → parses? checks-as-target cleanly (no spurious TYPEFAIL)? `W035`
  count. This is the **first trustworthy per-module effect-gap measurement** — it supersedes the
  unmeasurable standalone-`--check` sweep from the MEASURE re-verification. Expect some modules to
  surface real `W035` (the annotation gap) and some to TYPEFAIL if they depend on symbols not
  reachable by downward import seeding (record these — they bound the gate's validity).
- **Phase C — annotate the gap (toggle=1).** Use the Phase-B `W035` list with
  `scripts/dev/effects_annotate.sh` (untracked, UNVERIFIED — verify first); iterate rebuild rounds
  (~2.7 min each; the corpus is heavily over-declared so rounds should be few).
- **Phase D — flip + gate (toggle=2).** Set `effects_enforcement_mode() -> 2`; the coverage gate
  must pass (0 `E035` across all modules) and `release_gate.sh` stay green.

---

## §4 — Dependencies, risks, scope

- **Coverage of `main.sio` itself is gated behind the 271-wall.** `main.sio` cannot be parsed by
  gen-N today (271 keyword-collision errors — see `project_parser_selfhost_gap_2026-06-08`, `module`
  = 81 %). So EFF.1's gate can cover the 107 imported modules now, but `main.sio`'s own ~1,418
  functions need the 271-wall (`module`/`effect`/`is`/`study` keyword demotion) fixed first.
  Likewise true `gen2 == gen3` self-host verification of the flip needs the 271-wall. **EFF.1 is
  therefore downstream of the parser self-host gap and not on the critical path.**
- **False-reject risk.** A module checked as target only sees its *downward* imports, not symbols
  from modules that import it. Well-formed modules do not depend upward; Phase B measures whether any
  compiler module violates this (would TYPEFAIL) before the flip relies on the gate.
- **Performance.** 107 seed-and-check passes; bounded, acceptable for a gate (not per-edit).
- **No silent miscompile / no false-reject** is the cardinal rule; Phase B gates Phase D.

---

## §5 — Deliverables

1. `module_frontend_import_typecheck_target` + thin `_main` wrapper (Phase A).
2. A coverage-gate script (Phase B/D), wired into `release_gate.sh` once green.
3. A Phase-B coverage table = the first trustworthy per-module effect-gap measurement.
4. SYNTHESIS.md on resolution (patch, validation, LOC, self-host status vs the 271-wall).

**END OF DISPATCH (spec only — no `self-hosted/` edit applied).**
