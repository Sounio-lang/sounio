<!-- docs:meta
topic_id: repo.docs.research.psychiatric-counterfactual-authority-abstention-contract-2026-07-21
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.psychiatric-counterfactual-authority-abstention-contract-2026-07-21
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# A Counterfactual Is Not Authority To Act

Status: historical, source-backed research boundary and library-first roadmap

Date: 2026-07-21

Claim boundary: this document specifies a research representation for evidence
and abstention. It does not diagnose a person, estimate an effect for a real
patient, recommend a treatment, validate an intervention, or establish that a
named Sounio construct has clinical authority.

## The Missing Distinction

The preceding psychiatric-state contract separates an observation, a model
projection, a functional surrogate, a trajectory, a causal effect, and a
decision proposal. The next boundary is inside the last three categories:

```text
counterfactual-shaped value
!= identified causal estimand
!= transportable result for this target context
!= calibrated prediction in its declared domain
!= authority to make a clinical decision
```

This is not pedantry. A latent-state model may be useful while a treatment
effect remains unidentified. A carefully identified effect may not transport to
the person, setting, measurement protocol, or action support at hand. A
calibrated predictor can honestly abstain without identifying a causal effect.
None of those steps is replaced by a high posterior probability, a narrow
interval, a successful compiler run, or a type named `Counterfactual`.

The practical Sounio proposition is therefore modest and testable: preserve
these authority transitions as distinct receipts, and make abstention a normal
typed result rather than a silent default, exception, or missing branch.

## What The Literature Licenses

### A Target Trial Clarifies The Question, Not The Data

The target-trial framework first specifies the protocol of the pragmatic trial
whose causal question is intended, then asks whether observational data can
emulate it. Hernan and colleagues explicitly distinguish design errors that an
emulation can prevent from data limitations that it cannot remove.

- Hernan et al. (2025), [target-trial framework](https://pubmed.ncbi.nlm.nih.gov/39961105/).
- Chakraborty and Moodie (2013), [dynamic treatment regimes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4231831/).
- Orellana et al. (2013), [comparative effectiveness with the parametric g-formula](https://pmc.ncbi.nlm.nih.gov/articles/PMC3769803/).

For longitudinal dynamic regimes, consistency, sequential exchangeability, and
positivity remain substantive assumptions. They are not facts that a type
checker can derive from a trajectory. The language can require an explicit
receipt naming them, retain a sensitivity plan, and refuse a promotion when an
action is outside the declared support. It cannot turn an unmeasured
time-varying confounder into an observed variable.

### Transport Is A Separate Permission

Transportability formalizes when causal information learned in one environment
may be transferred to another. It requires an explicit account of what differs
between source and target environments; it is not implied by a model's internal
fit or by generic external validation.

- Pearl and Bareinboim (2011), [formal transportability framework](https://escholarship.org/uc/item/3tv1b3bg).
- Dahabreh et al. (2023), [generalizing and transporting clinical trial findings](https://pmc.ncbi.nlm.nih.gov/articles/PMC10392887/).
- Van Calster et al. (2022), [targeted validation for clinical prediction models](https://pmc.ncbi.nlm.nih.gov/articles/PMC9773429/).

This supports a separate `TransportReceipt`: eligibility, measurement mapping,
effect modifiers available in both environments, target setting, and the
assumptions under which a source result is used in the target. It does not
license transfer because two records share a diagnostic label or a scalar risk
score.

### Calibration And Abstention Are Not Causal Identification

Selective prediction and risk-controlling prediction sets give a principled way
to return a set, defer, or abstain under a stated loss and calibration regime.
Their guarantees are valuable, but they are guarantees about the declared
prediction risk under their assumptions, not proof of a treatment effect or a
clinical benefit.

- Bates et al. (2021), [distribution-free risk-controlling prediction sets](https://www.gsb.stanford.edu/faculty-research/publications/distribution-free-risk-controlling-prediction-sets).
- Angelopoulos et al. (2022), [Learn then Test risk control](https://www.gsb.stanford.edu/faculty-research/working-papers/learn-then-test-calibrating-predictive-algorithms-achieve-risk).
- Kompa et al. (2021), [uncertainty and abstention in medical ML](https://pmc.ncbi.nlm.nih.gov/articles/PMC7785732/).

A `SelectiveRiskReceipt` must therefore name the calibration population,
exchangeability or shift assumption, loss, threshold, coverage or risk target,
and time of calibration. It cannot be used as an `IdentificationReceipt`.
Conversely, identification alone cannot be used as a selective-prediction
guarantee.

### Reporting Is Necessary But Not A Validation Token

TRIPOD+AI provides reporting guidance for prediction models using regression or
machine learning. It helps another party inspect what was done; it does not
make a model clinically ready, establish transportability, or substitute for an
impact study.

- Collins et al. (2024), [TRIPOD+AI statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/).
- Moons et al. (2015), [TRIPOD statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC4297220/).

Accordingly, a reproducibility or reporting receipt is useful input to an
audit. It is not a `ClinicalValidationReceipt`.

### Psychiatric Dynamics Increase The Need For This Split

Longitudinal computational psychiatry can model latent trajectories and
individual heterogeneity. Those models make a richer state representation
plausible; they do not by themselves identify an intervention or authorize a
personalized regimen.

- Insel et al. (2025), [dynamical systems framework for precision psychiatry](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484574/).
- Gokce et al. (2024), [dynamic causal modeling of psychotic trajectories](https://pmc.ncbi.nlm.nih.gov/articles/PMC11104383/).
- Marquand et al. (2021), [normative modeling in computational psychiatry](https://pmc.ncbi.nlm.nih.gov/articles/PMC7613648/).

Their value for Sounio is a representation demand: a latent trajectory may be
typed as a model-dependent projection, while causal authority, calibration, and
clinical validation remain separately earned.

## Authority Is A Directed Evidence Path

The following path is intentionally not commutative. Each arrow consumes a
different kind of evidence; changing their order changes what has been shown.

```text
ObservationReceipt + ModelProjection
  -> CounterfactualQuestionReceipt

CounterfactualQuestionReceipt + TargetTrialProtocolReceipt
  + IdentificationReceipt + SensitivityReceipt
  -> CausalEffectEstimate

CausalEffectEstimate + TransportReceipt
  -> TargetContextEffectEstimate

ModelProjection + SelectiveRiskReceipt
  -> SelectivePredictionReceipt | AbstentionReceipt

TargetContextEffectEstimate + SelectivePredictionReceipt
  + EligibilityReceipt + HarmConstraintReceipt
  -> ResearchDecisionCandidate

ResearchDecisionCandidate + independently governed clinical validation
  -> outside the compiler and outside this contract
```

The two middle paths deliberately do not collapse. A selective prediction can
be appropriate for monitoring or research triage even when the causal path is
unidentified. A causal estimate can be valuable even when the prediction
surface is not calibrated for a particular target context. Neither result is a
clinical directive.

## Proposed Receipt Taxonomy

These are proposed library-level names, not parser syntax, standard-library
APIs, or a claim that the current compiler enforces them.

| Receipt | May state | Must not silently become |
| --- | --- | --- |
| `CounterfactualQuestionReceipt` | factual context, intervention definition, outcome, horizon, estimand question. | an identified effect. |
| `TargetTrialProtocolReceipt` | eligibility, assignment procedure, time zero, follow-up, censoring, outcome, analysis plan. | an emulation result. |
| `IdentificationReceipt` | explicit graph or design, consistency, sequential exchangeability, positivity/support, estimator and sensitivity plan. | a guarantee that assumptions hold. |
| `TransportReceipt` | source and target populations, selection assumptions, common measurements, effect-modifier handling. | target-population validation. |
| `SelectiveRiskReceipt` | loss, calibration cohort, risk or coverage target, calibration algorithm, domain/shift assumption. | causal identification. |
| `AbstentionReceipt` | reason, failed precondition, evidence still needed, review/acquisition route. | an error that callers may ignore. |
| `ResearchDecisionCandidate` | a bounded research comparison with explicit harms, constraints, eligibility, and provenance. | a treatment instruction. |
| `ClinicalValidationReceipt` | independently appropriate empirical and governance evidence. | a compiler pass, a model fit, or a reporting checklist. |

The critical constructor rule is negative: no public conversion from
`CounterfactualQuestionReceipt`, `CausalEffectEstimate`, a model score, or a
`SelectiveRiskReceipt` alone may construct `ResearchDecisionCandidate`.

## Existing Sounio Surfaces And Their Boundary

Sounio already contains useful workflow-shaped syntax and test fixtures:

```text
Counterfactual<T>
Contest<T, Models, Policy>
Deferred<T>
AlternativeSet<T>
AcquisitionPlan<T>
RecoursePlan<T>
ObservedTransition<T>
```

The fixtures under `tests/frontend/*counterfactual*` demonstrate that these
objects can participate in contest, deferral, acquisition, recourse, and
transition protocols. `stdlib/causal/` also provides DAG-oriented utilities.
Those are language and library facts. They do **not** prove that a
`Counterfactual<T>` is identified, a `Deferred<T>` is calibrated abstention,
or an `AlternativeSet<T>` is clinically safe.

The research contribution is to bind the missing evidence explicitly rather
than rename existing workflow objects. In a first library implementation,
ordinary nominal records and constructors are sufficient. No new grammar,
effect annotation, IR field, or clinical data is required.

## Abstention Is A First-Class Outcome

An abstention result should carry a reason whose semantics are inspectable:

| Reason | Required missing evidence | Permitted next route |
| --- | --- | --- |
| `UnidentifiedEstimand` | target-trial protocol, graph/design, or identification assumptions. | causal review or protocol specification. |
| `SupportViolation` | demonstrated action support/positivity for the history. | restrict estimand or acquire suitable evidence. |
| `TransportUnresolved` | target-context mapping or transport assumption. | target validation or transport analysis. |
| `CalibrationOutOfScope` | compatible calibration population or bounded shift argument. | recalibration/evaluation; not causal promotion. |
| `HarmConstraintUnresolved` | explicit utility, harm, or eligibility constraint. | human/governance review. |
| `ClinicalValidationAbsent` | independently governed validation. | no automated clinical action. |

This makes abstention constructive. The model can still emit an observation,
projection, uncertainty description, or research question. It cannot silently
use a missing receipt as a zero, a high-confidence default, or a recommendation.

## First Synthetic Collision Gate

After the imported-native #901 gate is accepted, the first executable bridge
should be synthetic and should prove only receipt separation:

| Collision | Hold fixed | Vary | Required result |
| --- | --- | --- | --- |
| Question versus authority | Same named `Counterfactual` question. | Presence of target-trial and identification receipts. | Question cannot construct causal-effect receipt without them. |
| Identification versus transport | Same identified source estimand. | Target context and transport receipt. | Source estimate cannot stand in for target-context estimate. |
| Calibration versus causality | Same model projection and scalar uncertainty. | Calibration cohort/domain receipt. | Selective prediction cannot construct causal authority. |
| Positivity boundary | Same trajectory and candidate action. | Support certificate. | Missing support yields `AbstentionReceipt`, never extrapolated action. |
| Abstention routing | Same failed threshold. | Acquisition/review route. | Different missing evidence yields distinct abstention provenance. |

The gate must use synthetic constants, reject category substitutions at compile
time, and execute natively only to show that the typed distinction survives the
selected source-to-IR path. It must not ingest patient data, output a dose, or
claim clinical performance.

## What Types Can And Cannot Enforce

Types can enforce that a programmer states a protocol identifier, refuses a
known missing prerequisite, preserves a reason for abstaining, and cannot pass
one receipt category where another is required. Tests can verify that selected
identities survive current-source compilation.

Types cannot establish biological plausibility, no unmeasured confounding,
positivity in a real population, exchangeability under time-varying treatment,
transportability, calibration after deployment shift, clinical utility, consent,
or governance approval. Those remain empirical and institutional questions.
The purpose of the type boundary is to make their absence impossible to hide.

## Falsifiers

This direction should be revised or demoted if:

- the receipt distinction cannot produce a synthetic compile-fail collision
  without inventing a new language feature;
- abstention reasons cannot be reconstructed from their stated prerequisites;
- a proposed transport receipt adds no information beyond the stated source
  protocol in a preregistered target-context study;
- selective-risk evidence is presented as causal identification in the
  implementation or its documentation;
- a research decision candidate can reach a clinical-action API without an
  independently governed validation boundary.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: psychiatric-counterfactual-authority-research-v0
Owner: codex-root
Concept-IDs: SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-PHYSICAL-OBSERVATION; SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: counterfactual representation, causal identification, transportability, calibration, abstention, and clinical validation remain distinct claims
Transformation: none; literature-backed research boundary mapped to existing workflow syntax and prospective library receipts
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: future Sounio psychiatric fixtures should treat abstention as explicit evidence-bearing output and require separate authority receipts
Claims-Forbidden: patient-level effect estimation, treatment recommendation, automated action, causal identification from fit, transport from label matching, and clinical validation from compiler success
Assumptions: cited work supplies design constraints, not a biological model or validation of any future Sounio package
Write-Set: docs/research/psychiatric_counterfactual_authority_abstention_contract_2026-07-21.md; docs/governance/topic-registry.v1.json; docs/governance/DOCS_AUTHORITY_MATRIX.md
Read-Set: docs/research/psychiatric_state_inference_contract_2026-07-21.md; tests/frontend/defer_action_counterfactual_basic.sio; tests/frontend/plan_acquisition_counterfactual_basic.sio; tests/frontend/propose_alternatives_counterfactual_basic.sio; stdlib/causal/README.md
Positive-Witness: existing frontend counterfactual/deferral fixtures demonstrate workflow-shaped objects; the synthetic authority collision gate is prospective
Negative-Witness: no existing counterfactual or deferred workflow object is accepted as clinical authority by this document
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Integration-Target: internal research planning only
Authoritative-Only-If: the cited sources, current Sounio surfaces, and forbidden claims remain aligned
```
