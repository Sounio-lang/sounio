<!-- docs:meta
topic_id: repo.docs.internal.concepts.proof-carrying-deployment-validity-revocable-authority
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.proof-carrying-deployment-validity-revocable-authority
-->

> Metadata note: `last_validated` is generated from the repository governance
> baseline. The D10 evidence date is 2026-07-19; executable gates and review
> receipts, not that metadata field, determine the status of this lane.

# Proof-Carrying Deployment Validity And Revocable Authority

Concept-ID: `SOUNIO-PROOF-CARRYING-DEPLOYMENT-VALIDITY-REVOCABLE-AUTHORITY`

Status: executable bounded fixture typestate with immutable local spending and
epoch-transition traces.

Canonical surface:
`stdlib/epistemic/proof_carrying_deployment_validity_revocable_authority.sio`

## Meaning

D10 treats deployment warrant as a scoped, ordered, temporal resource. A
statistical receipt does not become a permanent permission merely because it
once passed a threshold. Model, feature pipeline, calibrator, policy, runtime,
population, site, workflow, evidence window, monitoring epoch, and deferral
path remain part of the warrant identity.

In this lane, `proof-carrying` remains a project-local term for values carrying
bounded executable receipts. D10 is not Necula-style proof-carrying code, a
formal proof term, a cryptographic certificate, or a theorem about arbitrary
deployment systems.

The central state transition is deliberately narrower than clinical use:

```text
frozen D9 fixture boundary
  -> coverage-checked fixture evidence
  -> provenance-bound fixture evidence
  -> bounded sequential-validity evidence
  -> locally spent fixture-warrant trace
  -> site-scoped fixture canary lease
```

That chain can produce only a synthetic canary lease. It cannot produce
external validation, production deployment authority, patient state, or
clinical action authority.

## Sequential Validity

The exact two-look contest has path masses `9,3,3,1` over `CC,CM,MC,MM`.
Each look separately covers `12/16 = 3/4`. A stopping rule that stops at the
first miss and otherwise at look two covers only on `CC`, hence `9/16`.

A separate bounded time-uniform fixture covers at both looks exactly on paths
whose first symbol is `C`, so its simultaneous and stopped coverage are both
`12/16`. This is a finite two-look construction, not a general confidence
sequence or conformal theorem.

The e-process control uses a fair two-step tree:

```text
E0 = 1
E1 = 2 on H, 0 on T
E2 = 4 on HH, 0 otherwise
```

The frozen stopping rule stops after first-step `H` and otherwise continues to
step two, giving stopped path values `[2,2,0,0]` on `[HH,HT,TH,TT]`.
Expectations at each fixed time and at that stopping rule are exactly one. A
fixed-time e-value and an e-process remain nominally different.

## Metric Collision

For outcomes `[0,0,1,1]`, two permille score vectors are frozen:

```text
A = [100,200,800,900]
B = [400,450,550,600]
```

Both have perfect rank discrimination and identical decisions at threshold
`500`. Their exact Brier scores differ:

```text
A = 1/40
B = 29/160
```

Discrimination and threshold accuracy therefore cannot substitute for a
calibration profile. This is one finite arithmetic collision, not a claim that
Brier score alone establishes clinical calibration.

## Deployment Scope

The fixture keeps these categories non-substitutable:

```text
ManufacturerSafetyCase != LocalDeploymentSafetyCase
ResearchValidation      != ExternalValidation
ModelAbstention         != SafeDeferral
HumanPresent            != EffectiveOversight
PCCPPlan                != AuthorizedModification
NoDetectedShift         != NoShift
InputShift              != PerformanceDrift != CalibrationDrift
FixtureCanaryLease      != ProductionDeploymentAuthority
```

A site-A deferral fixture requires destination, capacity, acknowledgement,
response-time compliance, and an unresolved-case policy. A numerically
identical site-B abstention lacks capacity and acknowledgement and must remain
unsafe. An abstention is not itself a mitigation.

An out-of-protocol model change is quarantined even when its frozen performance
metrics match the prior version. Regulatory observations carry jurisdiction,
instrument, version, and effective window; D10 has no generic
`RegulatoryCompliant=true` value.

## Opaque Tokens

Positive typestate tokens, the canary lease, and all authority-bearing types
are module-private. Imported callers may pipe private tokens by type inference,
but cannot construct their literals. Public structs are observations or
refusals only and carry no authority.

This is classic nominal typestate enforced by compiler-rejected private stage
types. It is not indexed, dependent, affine, linear, or a proof-term system.
Any caller may invoke the frozen synthetic fixture producers and then pipe the
inferred private values. Constructor privacy therefore establishes a nominal
literal-fabrication wall, but proves no caller authenticity, external origin,
cryptographic provenance, custody, or theorem-level invariant.

## Immutable Transition Limits

The private tokens preserve the exact local budget, spend amounts, nonces,
ledger epoch, and lease epochs across an immutable runtime transition trace.
Current ordinary Sounio values are copyable, so passing an unspent token by
value does not statically destroy aliases or create a shared transactional
ledger. Likewise, an immutable old registry facet can be retained by a caller.
D10 therefore does not claim static no-double-spend, global uniqueness, or
post-revocation non-use.

Those stronger properties require affine or linear receipts, indexed identity
constraints, quantitative effects, a trusted live reference monitor, and a
formal operational semantics.

## Supported Claim

For frozen synthetic metric, two-look, e-process, site, deferral, change, and
local transition fixtures, Sounio can preserve nominal stage order, reject
category promotion, compute exact collision arithmetic, emit explicit
quarantine and reuse refusals, and keep production and clinical authority
unconstructible.

## Unsupported Claims

D10 establishes no real external validation, transportability, device safety,
regulatory conformity, clinical utility, patient benefit, production
deployment permission, static resource consumption, actual live revocation,
general anytime-valid inference, novelty, or priority.

## Pending Interface

`affine-warrant-consumption-and-trusted-live-institutional-authority-adapter`
remains pending. A future adapter must be case-scoped, site-scoped,
version-scoped, jurisdiction-scoped, time-bounded, revocable, auditable, and
issued by a competent external institution. Sounio may check that capability;
the standard library must not invent it.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
