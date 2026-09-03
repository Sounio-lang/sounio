<!-- docs:meta
topic_id: repo.docs.internal.concepts.semantic-lane-contract
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.semantic-lane-contract
-->

# Semantic Lane Contract

Status: active
Authority: required companion for lanes that alter a Sounio concept, type
meaning, effect meaning, IR semantic field, scientific interpretation, or claim
boundary.

This extends `.claude/PARALLEL_BLOCKER_CONTRACT.md`. It does not replace file
ownership, blocker classification, offload policy, or executable gates.

## Core Rule

Agents implement transformations over founder-authorized concepts. They may
propose a new concept or distinction, but must not silently redefine an existing
one to fit a local compiler, backend, library, or publication constraint.

## Required Lane Declaration

```text
Semantic-Lane-ID:
Owner:
Concept-IDs:
Intent-Preserved:
Transformation:
Types-Changed:
Effects-Changed:
IR-Changed:
Claims-Introduced:
Claims-Forbidden:
Assumptions:
Write-Set:
Read-Set:
Positive-Witness:
Negative-Witness:
Acceptance-Gate:
Integration-Target:
Authoritative-Only-If:
```

Use `none` explicitly. An omitted field is not an implicit `none`.

## Field Semantics

- `Concept-IDs`: rows in `docs/internal/concepts/registry.tsv`, or a proposed
  new ID with a draft contract.
- `Intent-Preserved`: founder-level invariant that must survive the lane.
- `Transformation`: exact semantic change, not the implementation plan.
- `Claims-Introduced`: statements newly supportable if the gate passes.
- `Claims-Forbidden`: attractive interpretations that remain unsupported.
- `Authoritative-Only-If`: evidence required to replace an older path.

## Mandatory Distinctions

```text
value                    != arithmetic error
arithmetic error         != uncertainty
uncertainty              != ignorance
computational provenance != physical causality
IEEE zero class          != zero provenance
effect annotation        != physical mechanism
compile success          != runtime parity
formal model             != empirical claim
analogy                  != ontology
```

## Integration Receipt

At handoff or review-ready state, append:

```text
Semantic-Outcome:
Concept-Status-Before:
Concept-Status-After:
Distinctions-Added:
Distinctions-Preserved:
Distinctions-Erased:
Evidence-Run:
Fallback-Path:
Legacy-Kept:
Conflicting-Lanes:
Next-Semantic-Interface:
```

`Distinctions-Erased` must say `none` or name the discharge or founder waiver
authorizing the erasure.

## Stop Conditions

Stop and request founder review when:

1. two concepts assign different meanings to the same field or effect;
2. a local limitation requires weakening founder intent;
3. analogy is being promoted to a physical or clinical claim;
4. a backend proposes silent precision, provenance, uncertainty, or status
   erasure;
5. two active lanes write the same semantic surface; or
6. the lane cannot state what would falsify its new claim.

## Scanner

```bash
bash scripts/dev/sounio_semantic_status.sh
```

The scanner is observational. It reports dirty semantic writers and overlaps,
but does not infer ownership, kill processes, modify worktrees, or resolve
conceptual conflicts automatically.
