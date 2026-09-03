<!-- docs:meta
topic_id: repo.docs.internal.concepts.proof-carrying-shift-robust-risk-transport
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.proof-carrying-shift-robust-risk-transport
-->

> Metadata note: `last_validated` is generated from the repository governance
> baseline. D11 evidence and review receipts are dated 2026-07-19.

# Proof-Carrying Shift-Robust Risk Transport

Concept-ID: `SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT`

Status: executable bounded finite-fixture target-risk arithmetic and nominal
authority attenuation.

Canonical surface:
`stdlib/epistemic/proof_carrying_shift_robust_risk_transport.sio`

## Meaning

D11 demonstrates, on frozen synthetic fixtures, how to ask whether a canary
warrant that was valid for a source population remains usable for a declared
target population. It does not assume that source validity travels with a
model binary. Model, feature pipeline, calibrator, policy, site, workflow,
source and target population, joint fixture, shift model, loss, subgroup plan,
evidence window, weight provenance, and epoch remain part of the transport
identity.

`Proof-carrying` remains project-local language for values carrying bounded
executable receipts. D11 is not a proof of transportability for arbitrary
populations, a clinical validation, or an implementation of the general
results cited in the research note.

## Nominal Shift Algebra

The lane keeps these assumptions and evidence classes non-substitutable:

```text
CovariateShift       : P_T(Y | X) = P_S(Y | X)
LabelShift           : P_T(X | Y) = P_S(X | Y)
ConceptStability     : declared target conditional remains stable
ConceptShiftFailure  : source conditional cannot be transported
MarginalTargetRisk   != PrespecifiedSubgroupRisk
SourceCalibration    != TargetLocalCalibration
SourceConformalRisk  != ShiftRobustTargetRisk
NoDetectedShift      != NoShift
```

These categories need not be mutually exclusive in real data. D11 makes no
claim that an unlabeled detector can identify their causal decomposition.
The positive frozen fixture uses deterministic coarse `X=Y`, so its covariate-
and label-conditional invariances are jointly compatible. Every positive
private evidence token derives from one exact labeled joint law and carries its
fixture run, noncryptographic joint-law fingerprint, model, population, window,
loss, calibrator, label-probe, and subgroup-plan identities where applicable.

## Exact Fixtures

One private exact-law token fixes source atom counts `(3,3,3,3)`, target atom
counts `(6,2,2,2)`, common denominator `12`, and bounded loss `(1,0,0,0)`.
The same private token freezes the coarse atom map, deterministic labels,
separate probe confusion matrix, diagnostic and active score vectors, and the
positive subgroup allocation. Each positive certifier checks its observation
payload against those fields before emitting an opaque token. The mass-and-loss
fingerprint is the deterministic base-31 composition `92352734845403155`.
It is a fixture-integrity checksum, not a cryptographic digest or
external-origin authentication.

Coarsening atom 0 into `A` and atoms 1..3 into `B` changes mass from `(3,9)/12`
to `(6,6)/12`. With conditional losses `(1,0)`, exact weights `(2,2/3)`
transport source risk `1/4` to target risk `1/2`. A separate support-failure
fixture leaves target risk only partially identified in `[0,1/2]`.

The same coarsening gives label priors `(3,9)/12 -> (6,6)/12`. Label-shift
identification uses a separate perfect probe `31311`; the evaluated loss
`31711` is never repurposed as its confusion matrix. A singular-probe fixture
produces the same unlabeled prediction histogram under target risks `1/4` and
`3/4`, so it cannot issue an identified label-shift token.

The concept contest holds the target inputs and served scores fixed. The
stable world has risk `2/4`; a later labeled snapshot has risk `4/4`. Without
those later labels the worlds are observationally equivalent, so the public
ambiguity record is diagnostic rather than a state-transition authority.

Two twelve-case target arrangements have the same marginal error `6/12`.
Their prespecified subgroup errors are `(3/6,3/6)` versus `(0/6,6/6)`, with
worst-group risks `1/2` and `1`. Marginal risk therefore cannot replace
worst-subgroup risk.

A constant diagnostic score `1/4` has source residual `0` under coarse mass
`(3,9)/12` and target residual `1/4` under `(6,6)/12`. The integer witness
computes the target residual as
`[6(4*1-1) + 6(4*0-1)] / (12*4) = 12/48 = 1/4`; the factor `4`
is the score denominator. This diagnostic record explicitly authorizes no
state transition. The active target-local calibrator `(1,0)` is exact on the
baseline target law; a private later labeled snapshot with outcomes `(0,0)`
has residual `-1/2` and supplies the distinct degradation trigger.

The conformal-risk contest uses the same four-atom law and loss. Source risk is
`1/4`, target risk is `1/2`, and L1 distance is `1/2`, hence joint total
variation is exactly `1/4`. Thus the bounded inequality
`R_T <= R_S + TV` is attained. This is one finite identity, not a
conformal-risk theorem.

## Non-Expansive Authority

The frozen local trace is:

```text
source canary rank 3
  -> target continuation rank 3
  -> degraded rank 2
  -> suspended rank 1
  -> revoked rank 0
```

The positive path binds exact source members `3114101..3114108` and target
members `3114101..3114104`; equality of those four member identities proves the
fixture-local subset relation. A smaller disjoint scope is rejected even when
its count is four. D10 supplies the source canary boundary but explicitly does
not attest this D11 subset fact.

Within one nominal trace, every transition preserves the fixture run, warrant,
site, model, workflow, loss, group plan, calibrator, and scope identities. The
private triggers bind successive windows and epochs:

```text
31121 / epoch 1 --active calibrator drift--> 31122 / epoch 2
31122 / epoch 2 --target ambiguity--------> 31123 / epoch 3
31123 / epoch 3 --concept shift-----------> 31124 / epoch 4
```

The states also retain labeled-snapshot lineage: baseline fingerprint
`92352734845403155`, active-calibrator drift snapshot `3162401`, the same last
labeled snapshot through the unlabeled ambiguity step, and final concept-shift
snapshot `3142401`.

There is no inverse, join, widening, or promotion edge from a lower token. The
final token is terminal only inside this nominal trace. It is not globally
absorbing, does not disable a runtime canary, and does not prove a unique
single execution chain.

The target continuation is still the same synthetic canary grade, not a new
authority. Production deployment authority, patient state, clinical action
authority, external transport validation, `NoShift`, and competent
institutional revocation authority are private reserved types with acceptors
and no producers. In the parallel ontology, `ReservedAuthorityArtifact`
descends directly from `EpistemicArtifact`; the reserved authority classes
descend from it, never from `WarrantStateArtifact`.

## Opaque Tokens And Limits

Positive evidence and attenuation states are module-private. Imported callers
may pipe them by inference but cannot construct their literals. Public records
are observations or refusals and carry no authority.

Frozen fixture producers remain callable and authenticate no external origin.
They can replay the same fixture, including an earlier continuation, after a
nominal revoked token has been produced. Ordinary Sounio values are copyable.
D11 therefore demonstrates an immutable local rank trace, not affine
consumption, alias invalidation, global freshness, or a live reference monitor.
A copied stale continuation is not statically destroyed by a later suspension
or revocation. D11 does not statically invalidate copied stale tokens and
provides zero runtime protection against replay of such a copy.

## Executable Evidence

- kernel and private typestate:
  `stdlib/epistemic/proof_carrying_shift_robust_risk_transport.sio`;
- exact standalone runtime:
  `tests/run-pass/clinical_shift_robust_risk_transport_native_witness.sio`;
- imported private-token flow:
  `tests/run-pass/clinical_shift_robust_risk_transport_witness.sio`;
- independent exact arithmetic:
  `scripts/research/proof_carrying_shift_robust_risk_transport_oracle.py`;
- nominal refusal matrices: `tests/compile-fail/clinical_d11_*_d11.sio` and
  `tests/compile-fail/ontology_d11_*_d11.sio`;
- recursive acceptance gate:
  `scripts/ci/proof_carrying_shift_robust_risk_transport_gate.sh`.

The gate, not this prose file in isolation, determines executable status.

## Supported Claim

For one frozen finite labeled joint law and its declared projections, Sounio can
separate six shift and risk evidence classes, compute exact transport and
impossibility collisions, require a member-wise fixture scope proof and all
target-bound evidence before preserving a synthetic canary rank, enforce
nominal downward-only state substitution, and keep production and clinical
authority unconstructible.

## Unsupported Claims

D11 establishes no real transportability, target-population validation,
distribution-free guarantee, general conformal risk control, causal diagnosis
of shift, universal subgroup guarantee, clinical utility, patient benefit,
production permission, affine use, live revocation, novelty, or priority.

## Pending Interface

`compiler-enforced-affine-attenuation-and-trusted-target-monitor` remains
pending. A real adapter must bind detector custody, target labels, overlap,
weight uncertainty, loss identity, subgroup plan, institutional competence,
and a live epoch reference monitor. The standard library must not invent those
facts.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
