<!-- docs:meta
topic_id: repo.docs.spec.e2e-specification-frame
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.e2e-specification-frame
-->

# End-to-End Specification — Frame

Concept-ID: `SOUNIO-E2E-SPEC-FRAME`

Status: **Garden.** This is **not** the specification. It is the frame the
specification must fill, the method for filling it, and an honest inventory of
what is already decided, what is contested between the two engines, and what
does not exist.

## Why a specification, and why now

Founder, 2026-08-19: *the seed of Madaros being lean_single is the principal
culprit; things will only move with an end-to-end spec.*

The diagnosis is supported by the tree. `CLAUDE.md` states it directly:

> **Fixed-point scope:** `make build` verifies the fixed point over
> `lean_single.sio` (the seed), **not** over `main.sio`/Madaros. Do not describe
> Madaros itself as fixed-point-verified.

So the chain is:

- **`lean_single`** is the seed, is fixed-point verified, and is what CI runs
  (measured 2026-08-19, PR #1964: the Full Test Suite uses `souc-stage2`)
- **Madaros** is built *by* `lean_single`, is canonical for users, and has **no
  fixed point**

**What is verified is not what is used, and what is used is built by what is
verified.** Madaros inherits lean_single's semantics wherever it does not
override them, and diverges silently wherever it does — because there is nothing
both must satisfy.

The consequence is precise and it is the reason this document exists:

> A divergence measurement can establish **where** two engines differ. It cannot
> establish **which is correct**. Without a specification, "divergence" has no
> referent — one can only say *they differ*, and the choice of which to follow
> becomes preference.

A specification is what makes one of two engines **wrong** rather than merely
different.

## What this frame is not

The fourteen concepts registered on 2026-08-19 are specification fragments, and
every one of them is about a **failure mode**: what may not be lost, what may
not be confused, what must refuse. Not one says what the language *does*.

The difference is between a code of conduct and a grammar. The corpus has the
first. This frame is the outline of the second.

## Required sections, with measured status

Each section is `undefined` until it has a normative statement **and** a
conformance test that both engines are run against. Status vocabulary is the
registry's; the ladder is monotone (`MATURITY_LADDER`).

| § | Section | What exists today | Status |
|---|---|---|---|
| 1 | Lexical structure | `self-hosted/lexer/` (8 files); token kinds have **two numberings** and comparing raw code to an enum cast silently never matches | contested |
| 2 | Grammar | `self-hosted/parser/` (9 files), `TypeExprKind` 54 variants; no grammar document | undefined |
| 3 | Type system — kinds | **eight** type-kind enums, 238 variants, 57 multi-enum stems, **0 homonyms** (#1949) | measured, unspecified |
| 4 | Type system — rules | bidirectional inference in `check/`; no written rules | undefined |
| 5 | Effects — vocabulary | 29 ids after #1963; `EffectKind` enum now in production | measured |
| 6 | Effects — rows and subtyping | **written**: `S06_EFFECTS_ROWS.md`. Rows are implemented in the *live* tree (`check/effects_row.sio`) with **zero external callers**; the set is `[i64; 8]` against **23** named effects, and four of them gate no decision | Hypothesis — §6.0 ruled |
| 7 | Effects — handlers | **written**: `S07_EFFECT_HANDLERS.md`. Measured stronger than the earlier reading — **no** path has execution semantics: `ExprHandle` occurs 0× in `ir/`, `native/`, `enir/`; Madaros erases the expression, lean_single refuses by ignorance | undefined — measured |
| 8 | Epistemic values | **written**: `S08_EPISTEMIC_VALUES.md`. The struct in the earlier cell **does not exist** — three incompatible declarations do, plus a two-line fixture; the compiler type holds *predicates*, not numbers | Hypothesis — §8.0 and the 1000-endpoint ruled |
| 9 | Uncertainty propagation | machine-level emitters in `ir/lower.sio`; `emit_variance_independent_product` assumes independence with nothing checking it | contested |
| 10 | Units and dimensions | `stdlib/units/` 9 files incl. QUDT; whether `mg` is an alias, a nominal type or a primitive is **under measurement** | undefined |
| 11 | Ontological validation | ChEBI, GO, HPO, LOINC bundles + reasoner; TBox present, **ABox absent** | measured, incomplete |
| 12 | Numeric tower | **written**: `S12_NUMERIC_TOWER.md`. The row below was measured against the enums; measured against the *compiler*, `i<n>`/`u<n>` are accepted for any n≥1 (`i999999` typechecks) and **no width has semantics** — `i8` gives 200 for 100+100, and `i256` wraps at `i64`. `f128`/`f256` **Reserved** (`E218`) on Madaros only. | written |
| 13 | Memory and linearity | `linear struct`, `&`/`&!`; `OwnTypeKind` has 4 variants | undefined |
| 14 | Lowering pipeline | live: `parser → check → ir → native`; HLIR is the **GPU frontend**; ENIR is a parallel verified pipeline with 0 production importers | contested |
| 15 | Backends | x86-64 ELF, arm64, GPU/PTX, wasm | undefined |
| 16 | Diagnostics | error codes `E001`–`E230+`; no catalogue with normative meaning | undefined |
| 17 | Conformance suite | **written**: `S17_CONFORMANCE_SUITE.md`. 437 of 1,527 CI greens fail under source-built Madaros (28.6%), and part of the gap is **absence, not disagreement**: the seed has no `Knowledge<…>` annotation machinery at all | undefined — measured |

`contested` means the two engines are known or suspected to disagree there, and
**the specification's job is to say which is right** — not to describe both.

## The recurring shape

Six subsystems were measured on 2026-08-19 by different agents with different
instruments. Every one has the same shape, and it is not the shape people expect
of an unfinished language.

| subsystem | designed | built | connected |
|---|---|---|---|
| ENIR verified pipeline | yes | yes | **no** — imports neither `hlir::` nor `ir::`, and nothing imports it |
| Effect rows (`check/effects_row.sio`) | yes — rule stated in its header | yes, 84 lines | **no** — three functions call one another, zero external callers; `check_handler_coverage` has none at all |
| Effect handlers (`handle`) | yes | token, parser, checker | **no** — `ExprHandle` occurs 0 times in `ir/`, `native/`, `enir/` (control: `ExprCall` 23) |
| `Knowledge<T>` (compiler) | yes — ε, validity, provenance, proof constraints | yes, in the AST | **no** — the stdlib declares its own three-field struct instead |
| Provenance vocabulary | six kinds | six declared | **no** — three have no surface syntax; unknown ones are dropped without a diagnostic |
| Refinement ↔ totality | yes | `pred_implies` works, path narrowing works | **no** — `/` generates no obligation, so nothing ever asks |

The exception proves the reading. `self-hosted/ir/egraph.sio` is 3,526 lines and
286 functions implementing equality saturation, and it is **imported by
`self-hosted/compiler/main.sio` and `ir/opt_cleanup.sio`**. It is the one large
piece examined that is both designed and connected — and it is the most
technically demanding of them.

**So the failure mode is not ambition, and not execution.** In each case the
design and a working implementation exist; what was never written is the call
site. And in each case the design and the partial implementation were committed
**together** — `AstProvenanceKind`'s six cases and the parser's three branches
entered in the same commit, `f9da2142f4`, 2026-02-27. Nothing rotted. The
connection was never there.

Two consequences for this specification.

**First, `undefined` is usually the wrong status for these.** A subsystem whose
semantics are written down in a header comment and implemented in code that
nothing calls is not undefined — it is *unreachable*. The distinction matters
because the remedy differs: an undefined section needs a ruling, an unreachable
one needs a caller.

**Second, this is what the specification is for.** None of the six was found by
someone using the language and hitting a wall — they were found by measurement,
months later, by agents grepping. A specification that states what must reach
what turns each of these from an archaeological discovery into a failing gate.

## Method

1. **Derive, do not invent.** Every normative statement starts as a measurement
   of what the compiler does, with the receipt cited. A statement no receipt
   supports is marked `undefined`, never guessed.
2. **Where the engines disagree, the founder rules.** The measurement names the
   divergence; the ruling picks the normative behaviour; the losing engine
   acquires a defect with an owner. This is the only step a measurement cannot
   perform.
3. **A section is `Executable` when a conformance test exists and both engines
   are run against it** — a correct programme that must pass and an incorrect
   one that must be refused, with the negative control showing the refusal fires
   for the stated reason (`SOUNIO-NO-VERSUS-UNKNOWN`).
4. **Nothing is `Claim-ready` on one engine.** A section passing only on
   lean_single is exactly the defect that produced this document.
5. **The frame's own status is enforced.** `scripts/ci/concept_status_gate.sh`
   (`ci.yml:68`) refuses a status claim that reality has passed; this document
   is subject to it like any other.

## Required Invariants

- The specification is the referent for "correct". An engine is not a
  specification; two engines are not a specification; a passing test on one
  engine is not evidence about the language.
- Every normative statement carries its receipt. `SOUNIO-EFFORT-LOCATION`: a
  number without its measurement conditions is not evidence.
- `undefined` is a legitimate and mandatory status. A section nobody has ruled
  on must say so — filling it with plausible prose is the failure this whole
  corpus exists to prevent.
- Fixed-point scope is a specification concern. A specification that the
  canonical compiler is not verified against describes a language nobody runs.

## Claims Forbidden

- Do not cite this as the specification. It is the frame; every section is
  `undefined` or worse until filled by the method above.
- Do not read the status column as coverage. It records what a receipt exists
  for, not what is correct.
- Do not treat the seventeen sections as complete. They are what one session's
  measurements surfaced.
- Do not read "contested" as "both are acceptable". It means a ruling is owed.

## Related

- `SOUNIO-GATING-ENGINE` — which engine's verdict counts while the spec is being
  written
- `SOUNIO-NO-VERSUS-UNKNOWN` — why `undefined` must be sayable
- `MATURITY_LADDER` — the ladder each section climbs
- `SOUNIO-EFFORT-LOCATION` — why each section needs a conformance test, not a
  paragraph
