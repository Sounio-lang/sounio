<!-- docs:meta
topic_id: repo.docs.internal.concepts.proof-carrying-statistical-coverage-empirical-binding
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.proof-carrying-statistical-coverage-empirical-binding
-->

# Proof-Carrying Statistical Coverage And Empirical Binding

Concept-ID: `SOUNIO-PROOF-CARRYING-STATISTICAL-COVERAGE-EMPIRICAL-BINDING`

Status: executable bounded finite model with an external candidate-data
criticism layer.

Canonical surface:
`stdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio`

The generated `docs:meta` validation date is a repository governance baseline.
It is not a D9 scientific sign-off, dataset custody receipt, validation date,
or clinical authorization.

## Meaning

D9 makes coverage and empirical applicability proof obligations carried by
nominal receipts. Its first rule is that coverage belongs to a procedure,
sampling law, target, and conditioning scope. It does not belong to the bytes
or endpoints of one realized region.

Here, `proof-carrying` means that nominal values carry exact finite executable
witness receipts that subsequent functions require. It does not mean
proof-carrying code in the formal-methods sense, a Lean or Coq proof term, or a
machine-checked general coverage theorem. The contract is version-bound to the
linked Sounio programs, fixtures, compiler, and negative tests.

The following categories are deliberately non-substitutable:

```text
ExactIdentifiedSet
ConfidenceRegion
PredictiveSet
DeclaredContextBinding
EmpiricalBinding
PatientState
ClinicalActionAuthority
```

This is not a single promotion ladder. A value may happen to carry the same
mask or numeric endpoints while answering a different question under a
different evidence-generating process.

## Exact Finite Coverage

D9 consumes the D8 exact identified set `{A, B}`, mask `3`, only as a declared
target. The D8 receipt remains an identified set under a frozen model and
assumption set. It is neither a confidence region nor a patient state.

The frozen D9 procedure returns masks `3, 1, 2, 7` for four possible outcomes.
Two positive integer-mass designs with denominator `12` produce:

```text
Design A weights                  5,1,1,5
Design A whole-set coverage       10/12
Design A memberwise minimum       11/12

Design B weights                  1,5,5,1
Design B whole-set coverage        2/12
Design B memberwise minimum        7/12
```

Mask `7` is a strict synthetic superset of the target `{A,B}`. The external
binary-label fixture uses only masks `1`, `2`, and `3`; it is a separate
integration path, and the synthetic enumeration does not validate it.

Both designs realize mask `3` at outcome zero. Their procedure coverage is
nevertheless different. The `3/4` threshold is an arbitrary frozen fixture
requirement, not a statistically or clinically derived boundary. D9 compares
exact integers by cross multiplication. Permille floor and remainder are
diagnostics and never replace the fraction.

The exhaustive oracle enumerates all 455 nonnegative four-outcome mass vectors
summing to 12. Exactly 165 have positive mass on every outcome, 290 fail that
fixture's sampling-support gate, and 25 positive designs satisfy the threshold.
These are finite combinatorial facts, not an asymptotic or distribution-free
coverage theorem. The fixture is illustrative only and carries no scaling,
power, or efficiency claim.

## Coverage Scope

Whole-set coverage and memberwise coverage have distinct types. Likewise,
marginal, subgroup, and selection-conditioned coverage have distinct types.

The finite selection control has marginal coverage `9/10`, common-group
coverage `9/9`, and rare-group coverage `0/1`. A rule that selects the rare
group therefore has selected coverage `0/1`. Good marginal coverage is not a
warrant for a selected case or pointwise individual truth.

`D9PredictiveSetReceipt` targets a future response category under one declared
predictive procedure. It is neither a confidence region over an identified set
nor a treatment-effect estimate.

## Context And Provenance

A declared context binding carries explicit identities for:

- confidence procedure and realized region;
- target population and time window;
- sampling-support positivity;
- statistical coverage calibration;
- metrological instrument calibration;
- instrument-population compatibility;
- matched declared provenance.

Statistical calibration and metrological calibration are separate nominal
obligations. Sampling positivity is not treatment positivity, exchangeability,
or a causal effect.

Declared lineage records source and transformation identities. It is not by
itself integrity, authenticity, measurement validity, or external custody.
Two provenance graphs in the synthetic fixture deliberately carry equal table
fingerprints and region masks but incompatible dataset, instrument,
population, and time-window identities.

Base-10 counts, IDs, masks, and printed fingerprints are audit diagnostics.
Only the host gate's SHA-256 comparison binds the checked public fixture bytes,
and even that byte integrity does not establish authenticity or original
custody.

## External Candidate Layer

D9 freezes the public UCI Drug Consumption (Quantified) dataset as a candidate
for model criticism, not as patient-state evidence. The repository records the
official source, CC BY 4.0 declaration, archive and data hashes, schema, and a
protocol hash.

The protocol was locally byte-frozen before the full-data result was calculated.
It uses deterministic development, calibration, and evaluation partitions and
a fixed, unfitted score. The development partition is deliberately unused and
fits no parameter. Calibration coverage is `286/377`, while held-out evaluation
coverage is `265/375`; the latter fails the arbitrary declared `3/4` gate.

That temporal ordering is recorded only by the local D9 workflow; it has no
independent timestamp, public preregistration, or pre-analysis Git commit. The
hash detects later changes to committed bytes, not outcome-dependent choices
made before commit.

The external result is therefore only a software-integration and refusal-path
fixture on public health-adjacent data. It is not clinical evaluation or
predictive-model validation. It still lacks verified collection windows,
metrological instrument calibration, original custody, and sealed validation.
It does not bind a patient state and cannot authorize a clinical action.

The supplied outcome is self-reported benzodiazepine recency. It is not a
clonazepam dose-response observation, diagnosis, symptom trajectory, benefit,
harm, or prescribing outcome.

## Mandatory Abstention

Calibration failure, sampling-support failure, and instrument-population
incompatibility each have a dedicated failure type and abstention constructor.
The failed artifact is preserved; no fallback state is selected.

The primary three-gate truth table has eight combinations. Only the all-pass
combination is eligible for a declared fixture binding; the other seven must
abstain.

`D9DeclaredContextBindingReceipt` means only that declared identities and the
bounded synthetic fixture obligations match. It is not external validation or
authority: its own `external_empirical_binding`, `patient_state`, and
`clinical_action_authority` fields are false.

For the external candidate fixture, reason mask `230` records:

```text
2    evaluation or model-criticism failure
4    metrological instrument calibration absent
32   verified collection window absent
64   external custody absent
128  sealed validation absent
```

The arbitrary `3/4` threshold controls only reason bit `2`. Even if another
threshold cleared that bit, the four absent external obligations yield exact
mask `4 + 32 + 64 + 128 = 228`, so abstention remains mandatory.

An abstention is not a negative prediction, patient escalation, diagnosis,
treatment, or action authorization.

## Authority Boundary

No positive constructor exists in bounded D9 for:

```text
D9ExternalDataCustodyReceipt
D9SealedValidationReceipt
D9EmpiricalBindingReceipt
D9PatientStateReceipt
D9ClinicalActionAuthorityReceipt
D9NegativePredictionReceipt
D9ClinicalEscalationReceipt
D9IntegrityReceipt
D9AuthenticityReceipt
D9MeasurementValidityReceipt
D9CausalTreatmentEffectReceipt
```

Those structs are module-private. Public consumer functions expose their
nominal walls so that wrong promotions produce exact expected/found compiler
diagnostics, while direct struct literals are rejected with `E176`.

The current D9 evidence establishes no route from prediction to treatment.
An observational-equivalence control explicitly carries opposite treatment
effect signs under the same observed-distribution fingerprint and emits causal
action ambiguity.

## Runtime And Ontology Boundary

The reusable kernel and imported API witness are current-source check evidence.
Imported execution remains excluded under
`BLK-20260718-D6-MULTIMODULE-RUNTIME`. Runtime evidence comes from a standalone
Sounio executor and an independent Python oracle.

The D8 import uses the established wildcard form because the current module
loader does not preserve this new cross-module nominal wall under the selective
single-type form. The D9 gate fixes and tests that integration shape. This lane
changes no compiler or resolver.

The ontology is a parallel nominal classification. Sibling types demonstrate
current-source non-substitution; they are not OWL disjointness axioms and do
not transport runtime receipts.

## Acceptance Surface

Acceptance requires:

- canonical Madaros, with no legacy-engine fallback;
- kernel, ontology, and imported-witness checks;
- native execution of all frozen exact arithmetic;
- independent exhaustive enumeration of 455 synthetic designs;
- exact replay of the external candidate protocol and results;
- SHA-256 refusals after dataset or protocol byte tampering;
- 51 exact negative programs, including three private-constructor refusals;
- concept registry and binding-manifest checks;
- default and rebuilt current-source ontology validation;
- recursive D8-D0 gates;
- independent math and clinical-authority review.

## Supported Claim

For one frozen D8 target, two finite synthetic sampling designs, and one
hash-bound public candidate dataset protocol, Sounio can preserve coverage
target, scope, design, context, and provenance; calculate exact finite coverage;
record a held-out software-fixture failure; and force abstention when declared
validity obligations fail.

The current type surface rejects promotion into external empirical binding,
patient state, causal treatment effect, and clinical action authority.

## Unsupported Claims

D9 establishes no estimator consistency, asymptotic validity, universal
conditional coverage, causal identification, transportability, measurement
validity, original consent, deidentification audit, data authenticity, custody,
diagnosis, prognosis, treatment benefit, patient truth, clinical utility,
clinical action, predictive-model validation, external preregistration,
certified temporal priority, novelty, or priority.

## Pending Interface

`external-custody-and-sealed-validation` remains pending. A future positive
interface would need real source custody, authenticity and integrity evidence,
instrument and measurement validation, population and temporal applicability,
sealed out-of-sample validation, causal and benefit-harm evidence where action
is contemplated, consent, and explicit clinical governance. None may be
inferred from D9's current receipts.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
