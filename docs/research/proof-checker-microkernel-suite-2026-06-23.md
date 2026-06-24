<!-- docs:meta
topic_id: repo.docs.research.proof-checker-microkernel-suite-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-checker-microkernel-suite-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Checker Microkernel Suite

Date: 2026-06-23

This note records a local suite manifest for the first executable
proof-checker microkernels in the solver/proof-checker upgrade lane. It ties
together the SAT/RUP chain, PB high-width row kernel, and SMT/Farkas scalar
kernel without promoting them to imported/native runtime evidence.

## Gate Record

- Module: `stdlib/safety/proof_checker_microkernel_suite.sio`
- Tiny runtime test: `tests/run-pass/proof_checker_microkernel_suite_tiny.sio`
- Imported smoke test: `tests/run-pass/proof_checker_microkernel_suite_imported.sio`
- Artifact fingerprint: `857420631`
- Audit fingerprint: `460917285`
- Instance fingerprint: `738291604`
- Certificate fingerprint: `159640872`
- Status code: `101`

## Anchors

The suite anchors three reviewed microkernels:

- SAT/RUP chain: artifact/audit `428619705` / `917264038`, status `98`
- PB high-width row kernel: artifact/audit `671902843` / `238570416`, status
  `99`
- SMT/Farkas scalar kernel: artifact/audit `583104927` / `719460238`, status
  `100`

The imported smoke test calls the public receipt and audit functions from all
three theorem modules, then feeds those results into the suite manifest. That
is frontend/API integration evidence only while the imported/native runtime ABI
blocker remains active.

## Masks

- `anchor_mask = 63`
- `self_contained_runtime_mask = 7`
- `imported_runtime_evidence_mask = 0`
- `covered_family_mask = 7`
- `covered_proof_format_mask = 7`
- `suite_next_action_mask = 15`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`
- `ok_mask = 1023`

The family/format masks mean local microkernel coverage for three small
families: SAT/RUP, PB/VeriPB-shaped, and SMT/Farkas. They do not mean complete
support for LRAT/FRAT, VeriPB/CakePB, Alethe/Carcara, or general SMT.

## Claim Boundary

This suite is integration bookkeeping over local microkernels. It is not a SAT,
SMT, or PB solver result; not an imported/native runtime proof; not a full
proof-format checker; not a Lorenz theorem; and not a public theorem promotion.
Its job is to make the current proof-checker seeds visible as one gated unit
without erasing the remaining runtime and format-completeness gaps.
