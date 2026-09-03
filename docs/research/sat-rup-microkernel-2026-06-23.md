<!-- docs:meta
topic_id: repo.docs.research.sat-rup-microkernel-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sat-rup-microkernel-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SAT RUP Microkernel

Date: 2026-06-23

This note records a bounded SAT/RUP proof-checker microkernel for the solver
and proof-checker upgrade lane. The kernel checks unit-propagation conflict for
three scalar clauses over variables `1..3`. It is an executable seed for the
SAT/LRAT/FRAT family, not a full LRAT parser and not a SAT solver.

This microkernel contains no Hadwiger-Nelson result and no Lorenz runtime
evidence. It does not claim imported/native runtime evidence, public theorem
promotion, finite-cover evidence, boundary-gluing evidence, or global Lorenz
flowpipe evidence.

## Gate Record

- Module: `stdlib/theorem/sat_rup_microkernel.sio`
- Tiny runtime test: `tests/run-pass/sat_rup_microkernel_tiny.sio`
- Imported smoke test: `tests/run-pass/sat_rup_microkernel_imported.sio`
- Artifact fingerprint: `715284039`
- Audit fingerprint: `304918672`
- Instance fingerprint: `806157423`
- Certificate fingerprint: `129640587`
- Status code: `97`
- Chain artifact fingerprint: `428619705`
- Chain audit fingerprint: `917264038`
- Chain instance fingerprint: `572803941`
- Chain certificate fingerprint: `248619370`
- Chain status code: `98`

## Checked Microkernel Shape

The tiny positive case verifies that the clauses:

- `(x)`
- `(~x OR y)`
- `(~y)`

derive a conflict by unit propagation. Therefore the empty clause is accepted
as RUP for this three-clause scalar formula.

The tiny negative cases reject:

- a wrong third antecedent that prevents the conflict
- a tautological candidate clause

## Checked Chain Shape

The chain layer checks two bounded RUP steps:

1. derive the unit clause `(y)` by RUP from `(x)` and `(~x OR y)`
2. derive the empty clause by RUP from the derived `(y)` and the antecedent
   `(~y)`

It rejects a wrong derived unit and a wrong conflict clause. This is still a
bounded scalar chain; it is not an LRAT hint parser or a proof-log parser.

## Scope Boundary

This is a bounded scalar RUP kernel:

- no clause database parser
- no LRAT hint parser
- no FRAT elaborator
- no deletion handling
- no imported/native runtime promotion
- no public theorem promotion

The imported smoke test is typecheck-only while the current Madaros
imported/native ABI blocker remains active.

## Claim Boundary

This microkernel preserves these nonclaims:

- `current_runtime_family_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

This is only a bounded scalar RUP microkernel and chain. It makes no imported
runtime, public theorem, graph-colouring, Lorenz, finite-cover, or
boundary-gluing claim. Its job is to put one small, runnable RUP rule into the
Sounio proof-checker family.
