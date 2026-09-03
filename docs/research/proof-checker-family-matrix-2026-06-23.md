<!-- docs:meta
topic_id: repo.docs.research.proof-checker-family-matrix-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-checker-family-matrix-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Checker Family Runtime Matrix

Date: 2026-06-23

This note records a proof-checker family runtime matrix for the solver and
Lorenz i128/i256 upgrade lane. The matrix turns the proof-profile and imported
runtime lift contract into five explicit runtime-evidence slots. It does not
implement any checker and does not assert current imported/native runtime
evidence.

This matrix contains no solver output and no Lorenz runtime evidence. Numeric
IDs named here are non-substantive traceability tokens only. They are not
outputs of this matrix and are not reasserted as SAT, SMT, PB, or Lorenz proof
checking; replay execution; replay verification; imported runtime evidence;
finite-cover evidence; boundary-gluing evidence; or theorem evidence.

## Gate Record

- Module: `stdlib/safety/proof_checker_family_matrix.sio`
- Tiny runtime test: `tests/run-pass/proof_checker_family_matrix_tiny.sio`
- Imported smoke test: `tests/run-pass/proof_checker_family_matrix_imported.sio`
- Artifact fingerprint: `193746582`
- Audit fingerprint: `682540913`
- Instance fingerprint: `471829306`
- Certificate fingerprint: `902174635`
- Status code: `96`

## Non-Substantive Traceability Tokens

- Solver proof-profile token pair: `964210753` / `526184309`
- Imported runtime lift-contract token pair: `642810357` / `508176294`
- Private evidence envelope token pair: `834620917` / `276591483`
- Solver proof-profile status: `88`
- Imported runtime lift-contract status: `95`
- Private evidence envelope status: `94`

## Family Runtime Slots

- `planned_family_mask = 31`
- `sat_lrat_runtime_mask = 0`
- `sat_frat_runtime_mask = 0`
- `pb_veripb_runtime_mask = 0`
- `smt_farkas_runtime_mask = 0`
- `lorenz_i256_runtime_mask = 0`
- `current_runtime_family_mask = 0`
- `missing_runtime_family_mask = 31`
- `imported_runtime_promotion_mask = 0`
- `portable_runtime_evidence_mask = 0`
- `family_matrix_next_action_mask = 31`
- `family_anchor_mask = 63`
- `ok_mask = 1023`

Here, `planned_family_mask = 31` names five desired runtime-evidence families:
SAT/LRAT, SAT/FRAT, PB/VeriPB, SMT/Farkas, and Lorenz i128/i256 numeric
receipts. `missing_runtime_family_mask = 31` records that all five are still
missing in imported/native runtime mode. It is not success evidence.

## Promotion Boundary

Every family-specific runtime mask is zero in this matrix. A future positive
matrix must replace this blocked matrix only after the selected compiler
artifact can run the relevant imported checker receipt at native runtime. A
frontend/typecheck-only imported smoke is not enough.

The current matrix therefore keeps:

- `current_runtime_family_mask = 0`
- `imported_runtime_promotion_mask = 0`
- `portable_runtime_evidence_mask = 0`

## Claim Boundary

This matrix preserves these nonclaims:

- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `finite_cover_certificate_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `formal_theorem_ready = 0`

This is not a SAT, SMT, PB, or Lorenz proof checker result; not imported runtime
evidence; not replay execution; not replay verification; not a Hadwiger-Nelson
result; not a public theorem promotion; not boundary gluing; not a finite-cover
certificate; and not a global Lorenz flowpipe theorem. Its job is to keep the
upgrade target concrete: five named proof-checker runtime families, all still
blocked until imported/native runtime evidence exists.
