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
| 6 | Effects — rows and subtyping | `EffVar` row polymorphism exists in `effects/types.sio`, **not** in production | contested |
| 7 | Effects — handlers | fast path (#1926) exists; **the CPS path has no execution semantics** (`EFFECTS_JUNCTION_ROUTING_2026-08-19`) | undefined |
| 8 | Epistemic values | `Knowledge<T>` = `{value, variance, confidence}`; **not linear**, no provenance field; `.value` is an unmarked projection | measured, contested |
| 9 | Uncertainty propagation | machine-level emitters in `ir/lower.sio`; `emit_variance_independent_product` assumes independence with nothing checking it | contested |
| 10 | Units and dimensions | `stdlib/units/` 9 files incl. QUDT; whether `mg` is an alias, a nominal type or a primitive is **under measurement** | undefined |
| 11 | Ontological validation | ChEBI, GO, HPO, LOINC bundles + reasoner; TBox present, **ABox absent** | measured, incomplete |
| 12 | Numeric tower | `i8..i128`, `u8..u128`, `f32/f64`; `f128`/`f256` **Reserved** (`E218`); `i256`/`i512`/`u256`/`u512` **absent from every enum** | measured |
| 13 | Memory and linearity | `linear struct`, `&`/`&!`; `OwnTypeKind` has 4 variants | undefined |
| 14 | Lowering pipeline | live: `parser → check → ir → native`; HLIR is the **GPU frontend**; ENIR is a parallel verified pipeline with 0 production importers | contested |
| 15 | Backends | x86-64 ELF, arm64, GPU/PTX, wasm | undefined |
| 16 | Diagnostics | error codes `E001`–`E230+`; no catalogue with normative meaning | undefined |
| 17 | Conformance suite | the corpus runs on **lean_single**, not the canonical compiler (#1964) | contested |

`contested` means the two engines are known or suspected to disagree there, and
**the specification's job is to say which is right** — not to describe both.

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
