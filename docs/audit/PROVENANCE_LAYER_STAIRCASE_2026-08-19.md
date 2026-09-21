<!-- docs:meta
topic_id: repo.docs.audit.provenance-layer-staircase-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.provenance-layer-staircase-2026-08-19
-->

# Provenance: the layers disagree about which cases exist

## Why this exists

Two independent measurements of the same object reached opposite verdicts.

- One (claude-1) read the commit history and concluded **the enum is the design
  and the parser is behind**.
- The other (glm-cli2, dispatched blind) read usage and concluded **the enum grew
  by anticipation** — six cases entered together with three branches, and nothing
  ever used the missing three.

Neither verdict survives a layer-by-layer count. This records what does.

## Measured

`origin/main`, 2026-08-19. For each of the six `AstProvenanceKind` cases, whether
the parser can produce it and whether `check/epistemic.sio` handles it.

| case | `parser/types.sio` | `check/epistemic.sio` | end to end |
|---|---|---|---|
| `AstProvDerived` | **produces** | no explicit branch — falls through | via the default |
| `AstProvSource` | — | explicit branch, `PROVENANCE_KIND_SOURCE` | **no** |
| `AstProvComputed` | **produces** | explicit branch | **yes** |
| `AstProvLiterature` | — | explicit branch, `PROVENANCE_KIND_LITERATURE` | **no** |
| `AstProvMeasured` | **produces** | explicit branch | **yes** |
| `AstProvInput` | — | explicit branch, `PROVENANCE_KIND_INPUT` | **no** |

Layer totals: declaration 6, checker 5 explicit + 1 default, parser 3.
**Two of six are wired end to end.**

## This settles anticipation versus lag, and it is lag

`Source`, `Literature` and `Input` are not bare enum names. Each has a **runtime
constant** (`PROVENANCE_KIND_*`) and an **explicit branch** in
`provenance_from_ast`. A speculative enum does not acquire runtime constants and
consumer branches; it sits in its declaration. The checker was built six-wide and
works. The parser was built three-wide.

The blind measurement's usage count is not wrong — it excluded declarations and
consumer match arms by design, which is exactly where the evidence lives. Two
correct measurements of different things, and the disagreement is the finding.

## A third defect, previously unrecorded

`provenance_from_ast` ends:

    provenance_new(PROVENANCE_KIND_DERIVED)      // fall-through
    None => provenance_new(PROVENANCE_KIND_DERIVED)

**An absent provenance and an explicit `derived` produce the same value.** A
reader downstream cannot distinguish *"the author said this is derived"* from
*"the author said nothing"*. `SOUNIO-NO-VERSUS-UNKNOWN`, in the field whose whole
purpose is to record where a value came from.

This is independent of the keyword question. Giving `Input` a keyword does not
fix it; separating the two states does.

## What this does to the founder's ruling

`asserted → Input` (2026-08-19) is not blocked by a design question. `Input`
already has a runtime constant and a checker branch. What it lacks is a token and
a parser branch — measured elsewhere at 2–4 lines each, with **zero** collisions
in versioned `.sio` (`docs/audit/PROVENANCE_KEYWORD_COVERAGE_DISPATCH_2026-08-19.md`).

## Claims forbidden

- That the enum grew by anticipation. Three of the unreachable cases carry
  runtime constants and consumer branches.
- That the checker is incomplete. It handles all six; one via a deliberate
  default that happens to collide with absence.
