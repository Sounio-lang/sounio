<!-- docs:meta
topic_id: repo.docs.research.private-evidence-envelope-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.private-evidence-envelope-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Private Evidence Envelope

Date: 2026-06-23

This note records a private evidence envelope over the runtime ABI blocker and
the kernel replay evidence router. The envelope is deliberately conservative:
it aggregates local bookkeeping tokens while preserving the active imported
runtime blocker and every public/global nonclaim.

This envelope contains no solver output and no Lorenz runtime evidence. Numeric
IDs named here are non-substantive traceability tokens only. They are not
outputs of this envelope and are not reasserted as solver proof, replay
execution, replay verification, imported runtime evidence, finite-cover
evidence, boundary-gluing evidence, or theorem evidence.

## Gate Record

- Module: `stdlib/safety/private_evidence_envelope.sio`
- Tiny runtime test: `tests/run-pass/private_evidence_envelope_tiny.sio`
- Imported smoke test: `tests/run-pass/private_evidence_envelope_imported.sio`
- Artifact fingerprint: `834620917`
- Audit fingerprint: `276591483`
- Instance fingerprint: `590147326`
- Certificate fingerprint: `803714269`
- Status code: `94`

## Non-Substantive Traceability Tokens

- Runtime ABI blocker token pair: `584291376` / `936740152`
- Kernel replay router token pair: `709284613` / `418905276`
- Prior profile token pair: `964210753` / `526184309`
- Prior Lorenz replay-verifier preflight token pair: `391742608` / `650219347`
- Runtime ABI blocker status: `92`
- Kernel replay router status: `93`
- Profile status: `88`
- Verifier preflight status: `91`

## Private Envelope Observations

- `private_envelope_mask = 15`
- `local_evidence_mask = 7`
- `imported_evidence_mask = 0`
- `runtime_promotion_mask = 0`
- `public_promotion_mask = 0`
- `global_promotion_mask = 0`
- `finite_cover_promotion_mask = 0`
- `boundary_gluing_promotion_mask = 0`
- `abi_blocker_active_mask = 1`
- `private_acceptance_level = 2`
- `envelope_next_action_mask = 31`
- `envelope_anchor_mask = 255`
- `ok_mask = 1023`

Here, `private_envelope_mask = 15` records that four local bookkeeping anchors
are bundled into one private envelope. `local_evidence_mask = 7` is local-only
bookkeeping inherited from the router. It does not encode portable imported
runtime evidence. Every imported, runtime-promotion, public, global,
finite-cover, and boundary-gluing promotion mask must remain zero while the
runtime ABI blocker is active.

## Claim Boundary

This envelope preserves these nonclaims:

- `public_claim_mask = 0`
- `global_flowpipe_claim_mask = 0`
- `finite_cover_certificate_mask = 0`
- `boundary_gluing_proof_mask = 0`
- `formal_theorem_ready = 0`

This is not a SAT, SMT, PB, or Lorenz proof checker result; not imported runtime
evidence; not replay execution; not replay verification; not a Hadwiger-Nelson
result; not a public theorem promotion; not boundary gluing; not a finite-cover
certificate; and not a global Lorenz flowpipe theorem. Its job is to preserve a
private/local evidence envelope without letting that envelope become portable
runtime evidence or a public mathematical claim.
