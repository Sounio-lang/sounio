<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-19-proof-carrying-statistical-coverage-empirical-binding-d9-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-19-proof-carrying-statistical-coverage-empirical-binding-d9-design
-->

# D9 Design: Proof-Carrying Statistical Coverage and Empirical Binding

Status: implemented and independently reviewed bounded design, pending stacked
publication and current-main compiler integration.

Date: 2026-07-19.

Base: D0-D8 head `8e5cab45f`, stacked above PR #1155 until that PR is
integrated.

Constraint: no compiler, resolver, or existing D0-D8 semantic changes.

## 1. Question

Can Sounio attach exact coverage semantics, sampling design, conditioning
scope, target population, time window, calibration status, support, and
provenance to a statistical set while refusing to promote that set into an
external empirical binding, an individual patient state, or clinical action
authority?

D9 answers in two layers: exact finite synthetic fixtures and a hash-bound
public candidate-dataset software-integration refusal layer. The latter is
deliberately not an empirical binding and does not validate a model, patient,
instrument, population, treatment, or clinical act.

`Proof-carrying` is used here in the project-local sense of typed values carrying
exact finite executable witness receipts. It is not Necula-style proof-carrying
code, a theorem-prover proof term, or a machine-checked general statistical
theorem.

## 2. Required Distinctions

The following objects are many-sorted. They are not one promotion ladder:

```text
ExactIdentifiedSet<Model, Assumptions>
!= ConfidenceProcedure<SamplingLaw, Target, CoverageScope>
!= RealizedConfidenceRegion<Procedure, SampleOutcome>
!= PredictiveSet<Response, ExchangeabilityScope>
!= DeclaredContextBinding<Population, Window, Instrument, Provenance>
!= EmpiricalBinding<ExternalCustody, SealedValidation>
!= PatientState
!= ClinicalActionAuthority
```

In particular:

1. coverage belongs to a procedure under a sampling law, not to one realized
   numerical region;
2. coverage of every member separately is not simultaneous coverage of a set;
3. marginal coverage is not subgroup, selection-conditional, or pointwise
   individual coverage;
4. equal masks, endpoints, or table bytes do not erase design or lineage;
5. sampling support is not treatment positivity or causal exchangeability;
6. statistical calibration is not metrological instrument calibration;
7. lineage is not integrity, authenticity, or measurement validity;
8. abstention is not a negative prediction, escalation, or treatment;
9. no statistical or empirical receipt is a patient state or clinical
   authority receipt.

## 3. Authority Structure

The bounded positive path is:

```text
D8 exact identified set
+ frozen sampling design
+ declared set-coverage semantics
    -> exact finite whole-set coverage

exact finite whole-set coverage
+ threshold comparison by exact cross multiplication
    -> adequate or insufficient coverage

adequate coverage
+ realized confidence region
+ target population and window
+ sampling positivity
+ statistical coverage calibration
+ metrological calibration
+ instrument-population compatibility
+ matched declared provenance
    -> DeclaredContextBindingReceipt
```

The real-world path remains intentionally incomplete:

```text
DeclaredContextBindingReceipt
+ ExternalDataCustodyReceipt
+ SealedValidationReceipt
    -> EmpiricalBindingReceipt
```

The bounded D9 fixture has no positive constructor for either receipt on that
last path, for `EmpiricalBindingReceipt`, or for patient-state and clinical
authority receipts. Synthetic provenance, public checksums, and a downloaded
dataset cannot satisfy them.

## 4. Principal Exact Finite Witness

The fixed target is the D8 `AB` exact identified set `{A, B}`, mask `3`.
One set-valued procedure maps four possible sample outcomes to:

```text
outcome       0  1  2  3
region mask   3  1  2  7
```

Mask `7` is a strict synthetic superset containing `{A,B}` and one additional
member. It tests whole-target containment by a superset. The independent UCI
binary-label path uses only masks `1`, `2`, and `3`; the finite synthetic
enumeration neither validates nor statistically binds that external replay.

Two designs assign exact integer masses with common denominator `12`:

```text
Design A weights  5, 1, 1, 5
Design B weights  1, 5, 5, 1
```

For simultaneous whole-set coverage, only masks `3` and `7` contain the whole
target. Therefore:

```text
Design A whole-set coverage = 10/12
Design B whole-set coverage =  2/12
```

Both designs have positive mass on every frozen outcome. Both realize the
same numerical region, mask `3`, at outcome zero. The realized region alone
therefore cannot determine its procedure coverage.

Memberwise coverage is a separate target:

```text
Design A minimum memberwise coverage = 11/12
Design B minimum memberwise coverage =  7/12
```

At threshold `3/4`, exact cross multiplication gives:

```text
Design A: 10 * 4 >= 3 * 12  -> adequate
Design B:  2 * 4 <  3 * 12  -> insufficient
```

Permille is stored as a diagnostic floor plus remainder, never as the
authoritative comparison:

```text
10/12 -> 833 remainder 4
 2/12 -> 166 remainder 8
```

The `3/4` threshold is an arbitrary frozen fixture requirement. It has no
statistical or clinical optimality claim.

Any exact threshold in `(1/6, 5/6]` preserves the synthetic A-pass/B-fail
separation; `3/4` is one frozen member of that interval.

## 5. Exhaustive Oracle Domain

The independent oracle enumerates every nonnegative four-outcome design with
total mass `12`:

```text
total designs                          455
all four outcomes positive             165
sampling-positivity failures            290
support-size histogram              4,66,220,165
```

For the 165 positive designs, whole-set coverage numerators have histogram:

```text
2:9, 3:16, 4:21, 5:24, 6:25,
7:24, 8:21, 9:16, 10:9
```

Exactly 25 of the 165 positive designs meet the `3/4` threshold. This is exact
enumeration of one artificial mass family, not uniform, asymptotic, Bayesian,
conformal, or external-population coverage.

## 6. External Candidate-Data Control

D9 freezes the official UCI
[Drug Consumption (Quantified)](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)
dataset, DOI [10.24432/C5TC7S](https://doi.org/10.24432/C5TC7S), as a model-
criticism candidate under its recorded CC BY 4.0 license.

```text
rows                         1885
eligible                     1877
Semeron-excluded                8
development                  1125
calibration                   377
evaluation                    375
```

The evaluation protocol was byte-frozen before full-data calculation. It uses
a deterministic ID-modulo-five partition and a fixed, unfitted score over
`Nscore + Impulsive + SS` to predict a binary self-reported Benzos recency
category.

The development partition is deliberately unused: it fits no coefficient and
selects no threshold. The local before-calculation ordering has no independent
timestamp, public preregistration, or pre-analysis commit. Hash checking detects
later mutation of committed bytes, not outcome-dependent choices before commit.

```text
calibration covered  286/377  -> adequate at 3/4
evaluation covered   265/375  -> insufficient at 3/4
support bands        3/3      -> compatible
```

The dataset SHA-256 is
`90b8cf500b07ad455baf9fe1dc519998c75a1df6d87f6bd7069176f0826ea8c1`.
The protocol SHA-256 is
`b443191335bda3eb0eaa3bd8fee47a30cebc16080ad9e0862d85d9734fee4a1e`.
The gate rejects a one-byte change to either input.

This layer proves reproducible byte and arithmetic agreement with the committed
candidate fixture. It does not establish original custody, authenticity,
consent, deidentification, metrological validity, temporal applicability,
predictive-model validity, patient state, or treatment relevance. It is a
software-integration and refusal-path fixture, not clinical evaluation. In
particular, a Benzos recency label is not a clonazepam response or psychiatric-
state observation.

## 7. Conditioning And Selection Control

A second frozen control has ten prediction occasions:

```text
common subgroup: 9/9 covered
rare subgroup:   0/1 covered
marginal:        9/10 covered
selection rule:  selects the rare subgroup
selected:        0/1 covered
```

This establishes only the finite arithmetic fact that good marginal coverage
can coexist with zero coverage in a selected subgroup. D9 therefore provides
distinct nominal receipts for marginal, subgroup, and selection-conditioned
coverage and rejects erasure between them.

## 8. Empirical Context And Provenance

Two declared provenance graphs carry the same table fingerprint and the same
realized numeric region but distinct:

- dataset and source IDs;
- sampling-design IDs;
- instrument versions;
- target-population versions;
- collection and applicability windows;
- transformation and calibration roles.

Their lineage mismatch produces a refusal receipt. It does not choose one
lineage, infer authenticity, or establish custody. W3C PROV motivates explicit
entity/activity/agent lineage, but provenance alone does not prove trust or
measurement validity.

## 9. Validation And Mandatory Abstention

The declared binding constructor requires nominal pass receipts for all of:

- exact adequate whole-set coverage;
- sampling positivity for the declared target support;
- finite statistical coverage calibration;
- current metrological instrument calibration;
- instrument-population compatibility;
- matched provenance, target population, and time window.

Failure receipts cannot enter the constructor. They produce abstention with a
reason mask:

```text
1    sampling positivity failure
2    statistical coverage or calibration failure
4    metrological calibration or validity-window failure
8    instrument-population incompatibility
16   provenance mismatch
32   target or time-window mismatch
64   external custody absent
128  sealed validation absent
```

Every abstention preserves the input region and states:

```text
binding_issued = false
patient_state_issued = false
clinical_authority_issued = false
```

The oracle also enumerates the three primary binary validity gates
(calibration, positivity, instrument-population compatibility): eight
combinations, one eligible combination, and seven abstentions.

The external candidate emits reason mask `230`: evaluation/model criticism
failure (`2`), absent metrological calibration (`4`), unverified collection
window (`32`), absent external custody (`64`), and absent sealed validation
(`128`).

Threshold choice controls only bit `2`. Even if that bit cleared, the remaining
external-obligation mask is exactly `228`, so no threshold change can create an
empirical binding in this fixture.

## 10. Predictive And Clinical Boundary

`D9PredictiveSetReceipt` targets a future response under a declared predictive
procedure. It cannot substitute for a confidence region over an identified
set, an empirical binding, or a patient state.

A small observational-equivalence control records two synthetic causal worlds
with the same predictive artifact and opposite treatment-effect signs. It
issues `D9CausalActionAmbiguityReceipt`, never clinical authority. A future
positive action path would additionally require a causal comparator,
benefit-harm or utility model, contraindication and safety context, consent,
and protocol authority.

## 11. Runtime And Compiler Boundary

- The reusable D9 module and imported API witness are current-source
  check-only evidence under `BLK-20260718-D6-MULTIMODULE-RUNTIME`.
- A standalone scalar witness and an independent Python oracle provide runtime
  evidence.
- The ontology is a parallel nominal surface, not runtime transport and not an
  OWL disjointness theorem.
- Canonical Madaros is mandatory. No lean_single fallback is accepted.
- No compiler, resolver, IR, Contest, or rebracketing authority is changed.
- The established wildcard D8 import is fixed by the gate because the current
  loader does not preserve this new nominal wall under a selective single-type
  import.

## 12. Literature Compass

- Identified sets and confidence regions have different targets and coverage
  semantics: [Imbens and Manski](https://doi.org/10.1111/j.1468-0262.2004.00555.x),
  [Chernozhukov, Hong, and Tamer](https://doi.org/10.1111/j.1468-0262.2007.00794.x).
- Distribution-free marginal predictive coverage is not unrestricted
  pointwise conditional coverage:
  [Barber, Candes, Ramdas, and Tibshirani](https://arxiv.org/abs/1903.04684).
- Conditional guarantees must be indexed by a declared class or group scope:
  [Gibbs, Cherian, and Candes](https://doi.org/10.1093/jrsssb/qkaf008).
- Selection changes the validity target:
  [Jin and Ren](https://doi.org/10.1093/jrsssb/qkaf016).
- Transport requires explicit source/target assumptions:
  [Pearl and Bareinboim](https://doi.org/10.1214/14-STS486),
  [Dahabreh et al.](https://doi.org/10.1111/biom.13009).
- Positivity and overlap determine the population for which an estimand is
  identified; trimming changes that target:
  [Crump et al.](https://doi.org/10.1093/biomet/asn055).
- Calibration is a joint frequency property of forecasts and outcomes and is
  distinct from sharpness:
  [Gneiting, Balabdaoui, and Raftery](https://doi.org/10.1111/j.1467-9868.2007.00587.x).
- Provenance records derivation but does not itself grant trust:
  [W3C PROV-O](https://www.w3.org/TR/prov-o/).
- Abstention carries its own risk-coverage semantics:
  [El-Yaniv and Wiener](https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf).
- Prediction is not treatment benefit or clinical utility:
  [Vickers and Elkin](https://doi.org/10.1177/0272989X06295361).
- The candidate dataset's source, schema, license, and DOI come from the
  [official UCI repository](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified).

These sources motivate the distinctions. They do not validate the D9 fixture
or establish novelty for Sounio.

## 13. Supported Claim

For two frozen finite synthetic designs, declared context fixtures, and one
hash-bound public candidate-data protocol, Sounio can carry exact rational
whole-set coverage, distinguish it from memberwise, marginal, subgroup,
selection-conditioned, and predictive coverage, preserve provenance and target
identity despite numeric collisions, record held-out model criticism, and
force abstention when declared validity obligations fail.

Here, `model criticism` means only failure of the frozen software-fixture gate;
it is not validation or scientific assessment of a fitted predictive model.

The type system rejects promotion of these receipts into external empirical
binding, patient state, and clinical authority categories.

## 14. Claims Forbidden

D9 does not establish real data custody, authenticity, consent or
deidentification audit, metrological or external calibration, estimator
consistency, asymptotic or distribution-free coverage, universal conditional
coverage, causal identification, transportability, diagnosis, prognosis,
treatment benefit, clinical utility, patient truth, clinical action,
predictive-model validation, certified preregistration or temporal priority,
cryptographic sealing, novelty, or priority.

## 15. Falsifiers

The bounded claim fails if any of the following occurs:

- the D8 target set substitutes for a confidence region or predictive set;
- designs A and B do not produce `10/12` and `2/12` whole-set coverage;
- exact threshold comparison depends on truncated permille;
- memberwise coverage substitutes for simultaneous whole-set coverage;
- marginal coverage substitutes for subgroup or selected coverage;
- equal region masks or table fingerprints erase provenance identity;
- any failed validation path emits a binding instead of abstention;
- a declared context binding becomes an external empirical binding;
- an abstention becomes a negative prediction or clinical escalation;
- a statistical, predictive, binding, or patient-state artifact becomes
  clinical action authority;
- the oracle counts differ from the frozen enumerations;
- dataset or protocol byte tampering is not rejected before analysis;
- the external accounting or `286/377` and `265/375` receipts drift;
- the lane changes a compiler, resolver, D0-D8 kernel, or legacy ontology path.
