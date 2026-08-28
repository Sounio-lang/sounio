<!-- docs:meta
topic_id: repo.docs.research.lorenz-i256-trajectory2-certificate-bounded-bridge-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lorenz-i256-trajectory2-certificate-bounded-bridge-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lorenz i256 Trajectory2 Certificate Bounded Bridge

Date: 2026-06-24

This note records compact certificate fingerprinting for the bounded two-step
Lorenz replay verifier. A certificate fingerprint is emitted only after the
supplied `state_0`, `state_1`, and `state_2` replay exactly through the status
`117` verifier.

## Gate Record

- Module: `stdlib/systems/lorenz_i256_trajectory2_certificate_bounded_bridge.sio`
- Tiny runtime test: `tests/run-pass/lorenz_i256_trajectory2_certificate_bounded_bridge_tiny.sio`
- Imported smoke test: `tests/run-pass/lorenz_i256_trajectory2_certificate_bounded_bridge_imported.sio`
- Artifact fingerprint: `918274650`
- Audit fingerprint: `341902786`
- Instance fingerprint: `675230419`
- Certificate fingerprint: `208746531`
- Status code: `118`

## Checked Certificate

The accepted replay certificate is:

```text
state_0 = (0.500000000, 1.625000000, 0.375000000)
state_1 = (11.750000000, 13.812500000, 0.187500000)
state_2 = (32.375000000, 326.796875000, 161.984375000)
```

The local state fingerprints are:

- `state_0`: `250000085`
- `state_1`: `626001669`
- `state_2`: `2048507`

The full compact certificate fingerprint is `657108232`.

The imported smoke also checks:

- replay mismatch rejection before certificate emission
- invalid limb rejection in state fingerprinting

## Fingerprint Rule

For one axis with four base-`1000000000` limbs:

```text
axis_fp = l0 + 17*l1 + 31*l2 + 43*l3 mod 1000000000
```

For one state:

```text
state_fp = 1000003*state_index + 3*x_fp + 5*y_fp + 7*z_fp mod 1000000000
```

For the replayed certificate:

```text
certificate_fp = 734260981 + 11*state0_fp + 13*state1_fp + 17*state2_fp mod 1000000000
```

The leading `734260981` is the status `117` replay-verifier artifact anchor.

## Anchors

This bridge anchors bounded replay artifact/audit
`734260981` / `605913274`, status `117`.

Status lineage: `117` is the local replay-verifier receipt and `118` is this
local compact-certificate receipt. These are local audit status codes, not
theorem numbers, not older portfolio version numbers, and not public
mathematical milestones.

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

This is not a cryptographic hash, not a collision-resistant commitment, not a
complete Lorenz integrator, not a stability or accuracy theorem, not arbitrary
signed-state coverage, not a general four-limb i256 product, not adaptive
stepping, not interval integration, not a finite-cover certificate, not a
boundary-gluing proof, not a global flowpipe theorem, not native `i256`
execution, and not imported/native runtime evidence. It only checks compact
audit fingerprinting for a bounded exact two-step replay certificate under the
explicit restrictions inherited from the replay verifier.
