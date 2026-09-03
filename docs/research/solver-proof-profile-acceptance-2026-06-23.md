<!-- docs:meta
topic_id: repo.docs.research.solver-proof-profile-acceptance-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-proof-profile-acceptance-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver Proof Profile Acceptance

Date: 2026-06-23

This note records a small Sounio proof-profile gate shared by the SAT, SMT/PB,
and Lorenz i256 lanes. It is deliberately not a new SAT solver, SMT solver, PB
checker, or Lorenz theorem. It is a reusable acceptance profile that blocks
producer-only claims from being promoted as checked results.

## Receipt

- Module: `stdlib/theorem/solver_proof_profile.sio`
- Tiny runtime test: `tests/run-pass/solver_proof_profile_tiny.sio`
- Imported smoke test: `tests/run-pass/solver_proof_profile_imported.sio`
- Artifact fingerprint: `964210753`
- Audit fingerprint: `526184309`
- Instance fingerprint: `710284936`
- Certificate fingerprint: `295748103`
- Status code: `88`

## Proof Formats

- `1`: DRAT-family SAT proof
- `2`: LRAT-family SAT proof
- `3`: FRAT-family SAT proof
- `4`: VeriPB / CakePB-style pseudo-Boolean proof
- `5`: Farkas / QF_LRA proof
- `6`: Sounio numeric receipt, used by Lorenz i128/i256 lanes

## Domains

- `1`: SAT / graph-colouring UNSAT
- `2`: pseudo-Boolean / cardinality
- `3`: SMT QF_LRA / Farkas
- `4`: Sounio numeric receipt
- `5`: hybrid profile that may accept any currently enumerated family

## Acceptance Rule

A solver-produced claim is accepted only when all of these gates pass:

- solver result is UNSAT/valid: `solver_result = 0`
- solver domain accepts the proof format
- instance digest is checked
- proof object or numeric receipt is present
- producer trace is present
- independent checker passes
- Sounio kernel replay passes
- scope gate passes

Public theorem promotion has an extra rule: if `public_claim_mask != 0`, then
`formal_theorem_ready` must be `1`. This keeps solver results, external
checker results, and public mathematical theorem claims in separate layers.
The receipt keeps `formal_theorem_ready = 0`, `public_claim_mask = 0`, and
`global_flowpipe_claim_mask = 0`.

The receipt also separates acceptance and rejection masks:

- `accepted_profile_mask = 15` records the four accepted private profiles
  (SAT/LRAT, SMT/Farkas, PB/VeriPB, Lorenz numeric receipt).
- `rejected_profile_mask = 48` records the two rejected promotion patterns in
  non-overlapping high bits: solver-only and public-without-formal.

## Claim Boundary

This profile is an acceptance gate only. It does not claim a new `chi(R^2) >= 6`
result, does not certify a new Hadwiger-Nelson graph, does not formalize the
2026 Erdős #90 disproof, and does not prove a global Lorenz flowpipe theorem.

For the Lorenz i256 lane, proof format `6` is accepted only as a private numeric
receipt when `global_flowpipe_claim_mask = 0`. A global Lorenz claim still needs
the later invariant/shadowing/global-cover proof layer.
