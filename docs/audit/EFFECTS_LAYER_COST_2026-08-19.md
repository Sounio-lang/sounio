<!-- docs:meta
topic_id: repo.docs.audit.effects-layer-cost-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effects-layer-cost-2026-08-19
-->

# Effects layer cost — measurement, not reconnection

Written **before** any `souc check`, import census, or git-history command
that this document will quote. Criteria first; numbers after.

```
Semantic-Lane-ID: effects-layer-cost-20260819
Owner: grok-cli5
Concept-IDs: none
Intent-Preserved: ENIR, MIR, HLIR and effects are layers to live in.
  This lane measures cost. It does not reconnect anything.
Transformation: none. Observation only.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a per-file check rc, an importer count from the
  validated `use <dir>::` instrument, and a yes/no on whether
  handlers.sio is what PR #1926 reimplements.
Claims-Forbidden: "the layer compiles" from an aggregate rc; "zero
  importers" from an unvalidated grep; "on CI"; that anything was
  reconnected; that any file under self-hosted/effects/ was edited
  (none were); "fable-1 is rewriting 99 KB" without a surface
  comparison.
Assumptions: origin/main at worktree creation is 515d93a8e3.
  Madaros is the engine. Inherited SOUC_BIN is poison and is unset.
  souc check is cheap and does not take the build lock. Any rebuild
  goes through Slurm; scripts/ci/build_modular_madaros.sh must not
  be wrapped in souc-build-lock.sh.
Write-Set: docs/audit/EFFECTS_LAYER_COST_2026-08-19.md
  docs/audit/EFFECTS_LAYER_COST_2026-08-19.tsv
Read-Set: self-hosted/effects/**
  self-hosted/check/effects.sio
  PR #1926 (lane/fable-1/cei-p0-handler-lowering)
Positive-Witness: the import instrument counts `use parser::` ≈ 93
  and `use ir::` ≈ 48 on this SHA (founder's validated control).
Negative-Witness: a `use|import|from|mod` grep that also matches
  the word "model" is a lying ruler and is not used.
Acceptance-Gate: five questions answered with per-file evidence;
  Q5 sent to fable-1 before this document is finished.
Integration-Target: origin/main (docs-only receipt; no reconnection).
Authoritative-Only-If: every number names its instrument, its
  positive control, and the refutation that would have killed it.
```

## Refutation criteria (written first)

A number in this document is dead if any of the following is observed.
These were written before the commands ran.

### R1 — recoverability of the four files

The layer is **not recoverable as living code** if, on a per-file
`souc check` (not an aggregate):

- the diagnostic is a parse / lexer failure (missing token, unexpected
  token, old syntax the current language no longer accepts), **and**
- the same file cannot be made to parse by a local, named syntax fix
  that does not change the declared meaning.

A type / name / effect diagnostic (E001, E035, undeclared identifier,
unknown effect) is **semantic**, not syntactic. Semantic red means the
file is still in today's language and can be argued with. Syntactic red
means the language moved out from under it.

If `souc check` exits 0 on a file, that file **exists as syntax**. It
does not exist as a pipeline layer. Existence ≠ connection.

### R2 — importer instrument

The importer count for directory `D` is the number of `.sio` files
under the repository that contain the literal substring `use D::`.

Refuted if:

- a file that the founder named as a known importer of `parser` does
  not match `use parser::`, or
- the parser count is not near 93, or the ir count is not near 48, or
- the instrument is a grep for `use|import|from|mod` (that matches
  `model` and lies).

A zero from an instrument that failed its positive control is not a
zero. It is a broken ruler.

### R3 — "has an enum, therefore more advanced than production"

Refuted if `self-hosted/effects/types.sio` does not contain a real
`enum EffectKind` whose variants are compared as enum values, **or**
if the live checker (`self-hosted/check/effects.sio`) already has the
same enum. An enum that is never constructed by the live pipeline is
not "in production". It is a designed type with no caller.

### R4 — history as scaffold-vs-craft

"Written in one large batch" is the reading if creation and last
meaningful touch are the same commit, or a burst of sibling commits
on the same day that also added the other disconnected layers.
Refuted if the four files accrue distinct authors / months of
substantive edits (not whitespace, not header comments).

### R5 — #1926 reimplements handlers.sio

**Yes** only if the #1926 surface names the same types or functions
(`EfhContext`, `efh_perform_effect`, `efh_resume`, `efh_cps_*`) or
imports `use effects::`. Shared English ("handler", "perform",
"clause") is not enough: the live compiler already uses those words
for AST-dispatch helpers.

**No** if #1926 is a compile-time tail-resumptive inliner in
`check.sio` / `ir/lower.sio` with zero `efh_*` symbols.

## Measurement SHA

`515d93a8e344` (`origin/main` at worktree creation, includes #1950).
Worktree: `/workspace/.wt/effects-cost`.
Engine: staged copy of this SHA's `bin/madaros-linux-x86_64` + `bin/souc` +
`bin/madaros` on OrangeFS, invoked via `scripts/dev/slurm_srun_minimal.sh`
on host `cpuops-t560-proxmox` as uid `sounio`.
The pod cannot see `/orangefs`; Slurm cannot see `/workspace`. Staging
was `kubectl cp` to `slurm-pilot-login-slinky` then `srun`.

## Import instrument (validated before effects=0 is believed)

**Ruler:** git-tracked `self-hosted/**/*.sio`, file **outside**
`self-hosted/<dir>/`, line-start `^use <dir>::`.

**Lying ruler (rejected):** `use|import|from|mod` matches 5381 tracked
`.sio` files (it hits the word `model`). Not used.

| dir | this SHA | founder (today, earlier SHA) | control |
|---|---:|---:|---|
| parser | **93** | 93 | **hits** |
| check | **49** | 49 | **hits** |
| ir | **48** | 48 | **hits** |
| native | 20 | 19 | near; SHA moved |
| wasm | 17 | 15 | near; SHA moved |
| hlir | **2** | 2 | **hits** |
| gpu | 2 | 1 | near; SHA moved |
| enir | **0** | 0 | **hits** |
| mli | **0** | 0 | **hits** |
| llvm | **0** | 0 | **hits** |
| vm | **0** | 0 | **hits** |
| effects | **0** | 0 | believed only because parser/ir/check fired |

Positive-control sample (a file the instrument must accept):
`self-hosted/compiler/main.sio` contains `use parser::` and `use ir::`
at line start and is outside those directories.

If parser had not been 93, effects=0 would have been discarded (R2).

## Q1 — per-file `souc check` (Slurm)

Positive control of the check instrument: staged
`positive/i64_pass.sio` (construct `i64`, add, compare) → **rc=0**,
`check: OK`, Madaros v0.80.0.

| file | bytes | lines | rc | class | first diagnostic |
|---|---:|---:|---:|---|---|
| `self-hosted/effects/mod.sio` | 594 | 15 | **0** | syntax+types exist | `check: OK` |
| `self-hosted/effects/types.sio` | 10895 | 375 | **1** | **semantic**, not parse | `error[E015] … effect_io … unknown struct type` (`Name`) |
| `self-hosted/effects/checker.sio` | 70331 | 2056 | **0** | syntax+types exist | `check: OK` |
| `self-hosted/effects/handlers.sio` | 101678 | 2991 | **0** | syntax+types exist | `check: OK` |

`types.sio` log: 39 `error[…]` total, **8× E015**, **31× E137**,
**0 parse**. Missing `Name` (defined in `self-hosted/parser/ast.sio`,
never imported), plus `name_eq`, `ty_unit`, `ty_string`. The file is
written as a sibling of a module that was never wired; the language
did not move out from under it.

**Recoverability (R1):** three of four files check clean on today's
Madaros. The red one is module-boundary semantics, not aged syntax.
This layer is disconnected, not archaeological syntax. Existence ≠
connection. An aggregate "the directory compiles" would have hidden
the dead `types.sio`.

Slurm logs: `/orangefs/training/effects-cost-20260819/receipt/`.

## Q2 — what they declare

### `types.sio` — a real enum

```
enum EffectKind {
    EffIO, EffMut, EffAlloc, EffPanic, EffDiv, EffGPU, EffAsync,
    EffProb, EffEpistemic, EffCausal, EffNetwork, EffSensor,
    EffExn, EffVar, EffUnknown
}
```

15 variants. Compared as enum values (`eff.kind == EffectKind::EffVar`).
`Effect` + `EffectSet` (max 16, row-polymorphic `EffVar`) +
`RuntimeEffectDef` / `EffectOperation`.

### `checker.sio` — i64 bit positions, not bytes

`EFF_IO()=1` … `EFF_EXN()=13`, `EFF_MAX()=14`. `EffBitSet` is an `i64`.
Names are `[i8;16]` tables for printing, not for recognition.

### Live production — `self-hosted/check/effects.sio`

`effect_name_to_id` is 23 handwritten byte comparisons on
`name_buf[i] == N as i8`. Instrument: 51 occurrences of `name_buf[`
and 51 of `as i8` in that file (founder's "51 comparacoes de bytes").
IDs 0–22:

`IO Mut Alloc Panic Div GPU Async Prob Epistemic Causal Network
Sensor Render Observe NonAssoc Audit Hypothesis MultiTest ZD
Witness Temporal Learn Chaotic`.

No `enum`.

### The inversion, with the caveat that kills the slogan

The disconnected layer **does** have a real enum. Production **does
not**. Representation-wise the orphan is ahead.

Vocabulary-wise it is **behind**. Missing from the enum vs live:
Render, Observe, NonAssoc, Audit, Hypothesis, MultiTest, ZD, Witness,
Temporal, Learn, Chaotic (11). Extra in the enum: `EffExn`, `EffVar`,
`EffUnknown`. Wiring the enum as-is would *delete* eleven live
effects. That is cost, not a free upgrade.

## Q3 — creation and last touch

| file | created | last touch | n commits | shape |
|---|---|---|---:|---|
| `mod.sio` | `d3d2715b54` 2026-02-16 Phase 1.1 | same commit | 1 | 15-line stub, never touched |
| `types.sio` | `d3d2715b54` 2026-02-16 (370 lines) | `a7dcd4abed` 2026-07-26 soft-keyword rename | 5 | small maintenance over 5 months |
| `checker.sio` | `c0cc456f63` 2026-02-28 wave 5 mega-profile | same commit | 1 | **2056 lines in one commit**, never touched |
| `handlers.sio` | `b0a9df2202` 2026-03-01 wave 8 (12 new modules, 236 files) | same commit | 1 | **2991 lines in one commit**, never touched |

`mod.sio` still says "this is a stub module … not yet integrated".

R4: checker and handlers are the scaffold smell — one large batch,
sibling of a 200-file / 236-file bootstrap wave, zero subsequent
edits. types.sio is the exception (overflow fix 2026-03-03, codegen
pass 2026-03-19, two one-line-class edits in July). The 99 KB that
bothers the founder is the batch that never grew a caller.

## Q4 — tests / fixtures / examples

Inside the files, yes. Outside, no living caller.

- `checker.sio`: 50 `effc_test_*` functions. No `fn main`.
- `handlers.sio`: 65 `test_efh_*` functions. No `fn main`.
- `git grep -l 'use effects::' -- '*.sio'`: empty.
- No file under `tests/` or `examples/` names these paths.
- `scripts/bootstrap/bootstrap_concat.sh` lists the four paths.
- `bootstrap/bootstrap_stage1.sio` inlines the source under
  `// SOURCE: self-hosted/effects/handlers.sio` — concatenation, not
  a call. Same text lives in `archive/build-legacy/`.

The in-file tests would only run if someone compiled the file as a
program and called them. `souc check` succeeding on `handlers.sio`
does not run them. A test that nothing invokes is a document.

## Q5 — does `handlers.sio` implement what #1926 is reimplementing?

**No.** Sent to fable-1 / `cei-p0-handler-lowering` as
`msg-1787134355-1982491-29927` before this section was filled.

| | `self-hosted/effects/handlers.sio` | PR #1926 |
|---|---|---|
| kind | runtime CPS machine | compile-time tail-resumptive inliner |
| types | `EfhContext`, `EfhContinuation`, `EfhHandler`, `EfhClause`, stack segments | no `Efh*` |
| verbs | `efh_perform_effect`, `efh_capture_continuation`, `efh_resume`, `efh_abort`, `efh_shift`, `efh_cps_transform_function` | `checker_expr_is_handler_perform`, `CHECK_HANDLER_DEPTH`, `lower_handle_expr_ref`, `LOWER_HANDLER_DEPTH` |
| resume | Once / Multi / Tail / Never | Tail only, by inlining the clause; no `resume()` |
| import | none | `use effects::` = 0 on the #1926 branch; `efh_*` = 0 in `check.sio` and `ir/lower.sio` |
| own words | header: "CPS-based algebraic effect handler dispatch" | CEI spec L58–60: P0 does **not** wire the orphaned 2991-line CPS runtime; it is a parallel minimal mechanism |

Overlap is the problem class (handle / perform / clauses), not the
artefact. `ResumeTail` is one of four strategies in the 99 KB; #1926
implements only that one, in the lowerer. Wiring the 99 KB would be a
different project. fable-1 is not rewriting those kilobytes.

## Cost, not a promise

What a reconnection would have to buy, if someone later promises a
date:

1. Give `types.sio` a `Name` (or stop using `parser/ast.Name` without
   `use parser::`). Today it is the only red file.
2. Decide the vocabulary: extend `EffectKind` by eleven live names, or
   admit the enum is a 2026-02 snapshot.
3. Pick an architecture: CPS runtime (`efh_*`) **or** #1926 inliner.
   They do not compose by concatenation.
4. Write a caller. Zero `use effects::` in the history of the
   repository. The bootstrap concat is not a caller.
5. Do not lower the criterion. `souc check` green on three files is
   not a pipeline.

No file in `self-hosted/effects/` was edited this turn. No PR.

