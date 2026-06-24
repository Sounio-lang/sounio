<!-- docs:meta
topic_id: repo.docs.research.smt-farkas-microkernel-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.smt-farkas-microkernel-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SMT Farkas Microkernel

Date: 2026-06-23

This note records a bounded scalar Farkas proof-checker seed for the SMT/QF_LRA
side of the solver/proof-checker family. The executable kernel checks one
two-inequality, one-variable integer-scaled certificate.

## Gate Record

- Module: `stdlib/theorem/smt_farkas_microkernel.sio`
- Tiny runtime test: `tests/run-pass/smt_farkas_microkernel_tiny.sio`
- Imported smoke test: `tests/run-pass/smt_farkas_microkernel_imported.sio`
- Artifact fingerprint: `583104927`
- Audit fingerprint: `719460238`
- Instance fingerprint: `284916705`
- Certificate fingerprint: `640271893`
- Status code: `100`

## Checked Certificate

The checked inequalities are interpreted as:

```text
x <= 0
-x <= -1
```

With Farkas multipliers `y1=1`, `y2=1`, their nonnegative linear combination
has:

```text
1*x + 1*(-x) = 0
1*0 + 1*(-1) = -1
```

So the combined row is `0 <= -1`, a contradiction.

The self-contained and imported tests check:

- the positive certificate above is accepted
- a wrong multiplier vector is rejected because the variable coefficient does
  not cancel to zero
- a nonnegative right-hand side is rejected
- a negative multiplier is rejected
- the expected scalar left and right sums are pinned

## Claim Boundary

This is a bounded scalar Farkas kernel only. It is not a full SMT solver, not an
Alethe parser, not Carcara/LFSC, not a general Fourier-Motzkin proof replay,
not imported/native runtime evidence, and not a public theorem. It gives the
Sounio solver lane a small executable SMT/Farkas proof-checker seed next to the
SAT/RUP and PB/VeriPB-shaped seeds.
