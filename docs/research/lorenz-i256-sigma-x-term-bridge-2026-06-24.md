<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-sigma-x-term-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-sigma-x-term-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Sigma X-Term Bridge

Date: 2026-06-24

This note records a portable composition bridge for the Lorenz x-derivative
term `sigma*(y-x)`. It composes two existing high-width limb bridges:

- signed delta: sign plus absolute `y-x` over four base-`1_000_000_000` limbs
- scalar multiplication: multiplying the absolute delta by `sigma = 10`

## Gate Record

- Module: `stdlib/systems/lorenz_i256_sigma_x_term_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_sigma_x_term_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_sigma_x_term_bridge_imported.sio`
- Artifact fingerprint: `318704692`
- Audit fingerprint: `729516083`
- Instance fingerprint: `406827195`
- Certificate fingerprint: `982641307`
- Status code: `105`

## Checked Term Cases

The imported smoke checks:

- positive term:
  `y=[150000000,1,0,0]`, `x=[950000000,0,0,0]`
  gives `y-x=+[200000000,0,0,0]`, so `10*(y-x)=+[0,2,0,0]`
- negative term:
  reversing `x` and `y` gives sign `-1` with the same magnitude `[0,2,0,0]`
- zero term:
  equal `x` and `y` gives sign `0` and zero magnitude

The imported smoke also anchors the signed-delta bridge
`492681735` / `837260419` and the scale bridge
`734815269` / `216903584`.

## Boundary

This bridge records:

- `target_integer_width = 256`
- `limb_base = 1000000000`
- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

The imported smoke remains frontend/typecheck evidence only while the current
imported/native runtime ABI blocker remains active.

## Claim Boundary

This is not a complete Lorenz stepper, not signed interval integration, not
replay execution, not replay verification, not a finite-cover certificate, not
a boundary-gluing proof, not a global flowpipe theorem, not native `i256`
execution, and not imported/native runtime evidence. It only checks the
bounded `sigma*(y-x)` composition over decimal limbs.
