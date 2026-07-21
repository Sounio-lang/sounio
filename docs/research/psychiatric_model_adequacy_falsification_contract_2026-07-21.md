<!-- docs:meta
topic_id: repo.docs.research.psychiatric-model-adequacy-falsification-contract-2026-07-21
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.psychiatric-model-adequacy-falsification-contract-2026-07-21
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# A Fit Is Not A Falsification

Status: historical, source-backed research boundary and library-first roadmap

Date: 2026-07-21

Claim boundary: this document proposes a way to keep model specification,
inference computation, simulation recovery, posterior-predictive adequacy, and
empirical interpretation distinct in future research software. It does not fit
a model to patient data, establish a psychiatric mechanism, diagnose a person,
estimate a treatment effect, recommend an intervention, or validate a clinical
system.

## The Missing Boundary

The current psychiatric contracts already keep observation, measurement model,
functional surrogate, trajectory, causal effect, transport, calibration,
assurance, and clinical validation distinct. A gap remains inside the model
layer itself:

```text
same fit statistic != recoverable parameter
same recovered parameter != recoverable generating model
same posterior-predictive similarity != adequate explanation of every feature
same simulation-based calibration result != adequate empirical model
same adequate model check != identified causal effect
same fitted or checked model != clinical authority
```

This distinction matters especially for complex psychiatric systems. Two
mechanistic stories can make similarly good predictions for a selected
endpoint, yet imply different latent states, different responses to a changed
context, and different counterfactual interpretations. Conversely, an
inference implementation can be wrong even when a single fit looks plausible.

Sounio need not decide which model is true. Its useful, narrower role is to
make a program carry the scope and failure mode of the check it actually ran,
and to prevent that receipt from being silently promoted into a biological,
causal, or clinical conclusion.

## What The Literature Licenses

### Recovery Is A Question Of Distinguishability

Parameter recovery asks whether known synthetic parameter values can be
recovered after the model generates data and is fit back to those data. Model
recovery asks whether a data-generating model can be distinguished from its
candidate alternatives. These are tests of a specified model space, task
design, parameter range, and inference procedure. They do not establish that
the model space contains the process that generated empirical data.

- Hess et al. (2025), [Bayesian workflow for generative modeling in computational psychiatry](https://pmc.ncbi.nlm.nih.gov/articles/PMC11951975/), separates model and parameter recovery from later model evaluation.
- Wilson and Collins (2019), [ten simple rules for computational modeling of behavioral data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6879303/), treats parameter recovery as a crucial part of a model-based analysis.
- Karvelis, Paulus, and Diaconescu (2023), [individual differences in computational psychiatry](https://doi.org/10.1016/j.neubiorev.2023.105137), reviews reliability and recoverability limitations in computational measures.

Thus a `ParameterRecoveryReceipt` may say that a declared synthetic regime was
tested and summarize its recovery criterion. It cannot state that an empirical
participant's inferred parameter is stable, valid, causal, or a property of
that person outside the declared task and model.

### Computation Calibration Is Not Model Validation

Simulation-based calibration (SBC) checks an inference algorithm against
replicated simulations drawn from its declared Bayesian joint model. Under its
assumptions it can expose computational bias or inconsistency in posterior
inference. It says nothing by itself about whether the joint model is an
adequate description of an observed psychiatric system.

- Talts et al. (2020), [validating Bayesian inference algorithms with simulation-based calibration](https://arxiv.org/abs/1804.06788), gives SBC as a validation procedure for inference algorithms within a specified generative model.

Therefore an `InferenceCalibrationReceipt` belongs to the computation path. It
is neither a `ModelAdequacyReceipt` nor an `IdentificationReceipt`.

### Posterior Predictive Checking Tests Declared Features

Posterior predictive checking compares features of observed data with data
generated from fitted posterior predictions. It evaluates absolute adequacy for
the features and domain that were actually checked, not an unrestricted claim
that a model explains the system. In a psychiatric model, a check of a mean
response or aggregate score can miss individual heterogeneity, transitions,
time dependence, or a context-sensitive feature that motivated the model.

Hess et al. explicitly distinguish posterior-predictive model evaluation from
relative model comparison and pre-specify the features used for their checks.
That supports a receipt which names the discrepancy features, data scope,
simulation procedure, and failure criterion rather than storing only a scalar
"fit passed" value.

## Ordered Evidence Paths

The following paths are intentionally non-associative. Each check consumes a
different input and answers a different question.

```text
ModelSpecificationReceipt + InferenceProcedureReceipt
  -> FittedModelProjection

ModelSpecificationReceipt + SyntheticRegimeReceipt
  + InferenceProcedureReceipt
  -> ParameterRecoveryReceipt | ParameterRecoveryAbstentionReceipt

CandidateModelSpaceReceipt + SyntheticRegimeReceipt
  + InferenceProcedureReceipt
  -> ModelRecoveryReceipt | ModelRecoveryAbstentionReceipt

ModelSpecificationReceipt + InferenceProcedureReceipt
  + SimulationCalibrationPlanReceipt
  -> InferenceCalibrationReceipt | InferenceCalibrationAbstentionReceipt

FittedModelProjection + PosteriorPredictivePlanReceipt
  -> ModelAdequacyReceipt | ModelDiscrepancyReceipt
```

None of these arrows can be replaced by another:

```text
ParameterRecoveryReceipt != ModelRecoveryReceipt
InferenceCalibrationReceipt != ModelAdequacyReceipt
ModelAdequacyReceipt != MeasurementInvarianceReceipt
ModelAdequacyReceipt != IdentificationReceipt
ModelAdequacyReceipt != ClinicalValidationReceipt
```

The first two paths ask whether a model and its parameters can be distinguished
inside a declared synthetic world. The third asks whether the computation
faithfully implements inference under that same declared world. The last asks
whether the fitted model reproduces declared features of the observed world.
Even their conjunction is still scoped research evidence, not a causal or
clinical conclusion.

## Proposed Receipt Taxonomy

These are names for a future library protocol. They are not parser syntax,
standard-library APIs, numerical methods, or evidence that the current
compiler enforces an epistemic property.

| Receipt | It may state | It must carry or reference | It must not silently become |
| --- | --- | --- | --- |
| `ModelSpecificationReceipt` | model family, state/update assumptions, observation model, priors or constraints, and version. | candidate-model boundary and known exclusions. | proof that the model represents the world. |
| `InferenceProcedureReceipt` | algorithm, initialization/seed policy, stopping rules, diagnostics, and implementation identity. | declared numerical/computational scope. | evidence that inference is correct. |
| `SyntheticRegimeReceipt` | generated parameter/model range, task or observation design, replication plan, and scoring rule. | simulation domain and limits. | empirical-population coverage. |
| `ParameterRecoveryReceipt` | recovery result for stated parameters under a stated synthetic regime. | model, inference procedure, metric, failures, and range. | construct validity, test-retest reliability, or causal identity. |
| `ModelRecoveryReceipt` | discrimination result among a declared candidate model space. | confusion/error profile, alternatives considered, and synthetic regime. | evidence that no omitted alternative explains the data. |
| `InferenceCalibrationReceipt` | result of a stated algorithm-calibration check such as SBC under a declared joint model. | simulation count, quantities, diagnostic criterion, and failure limits. | model adequacy or real-world calibration. |
| `PosteriorPredictivePlanReceipt` | predeclared predictive features, discrepancy criteria, evaluation domain, and simulation procedure. | feature-level test plan and scope. | a global adequacy claim. |
| `ModelAdequacyReceipt` | which posterior-predictive checks passed in the declared data domain. | model identity, plan, observed-data scope, failures, and unresolved features. | causal identification, mechanism truth, or clinical validation. |
| `ModelDiscrepancyReceipt` | a failed or out-of-scope predictive feature, conflict, or untested critical feature. | affected model, feature, severity, and revision/abstention route. | a warning that downstream code may ignore. |
| `ModelAdequacyAbstentionReceipt` | why adequacy cannot be claimed for a requested use. | missing plan, failed discrepancy, out-of-domain input, or untested feature. | a zero effect or an implicit accept. |

## Synthetic Collision Matrix

The first executable bridge should remain synthetic and import-bearing only
after #901 has a real checker-level negative gate. It should test receipt
separation, not perform statistical inference or ingest clinical data.

| Collision | Hold fixed | Vary | Required refusal |
| --- | --- | --- | --- |
| Fit versus parameter recovery | Same scalar fit summary. | Synthetic regime and recovery evidence. | Fit cannot satisfy `ParameterRecoveryReceipt`. |
| Parameter versus model recovery | Same parameter recovery result. | Candidate alternatives and model-confusion evidence. | Parameter recovery cannot satisfy `ModelRecoveryReceipt`. |
| SBC versus adequacy | Same model and inference identity. | Algorithm calibration evidence versus predictive feature evidence. | `InferenceCalibrationReceipt` cannot satisfy `ModelAdequacyReceipt`. |
| Predictive scope | Same favorable predictive summary. | Checked features and data domain. | A check of one feature cannot construct a global adequacy receipt. |
| Discrepancy routing | Same fitted projection. | One failed or untested critical feature. | `ModelDiscrepancyReceipt` routes to abstention/revision, never a positive adequacy type. |
| Adequacy versus causality | Same adequate predictive checks. | Identification and target-trial receipts. | Adequacy cannot satisfy `CausalEffectEstimate` or `ResearchDecisionCandidate`. |

The leaf module should define only ordinary nominal evidence records and
constructors. An importing main should attempt the invalid promotions so the
compile-fail surface crosses the module boundary. A positive native run, if
the compiler path supports it, proves only that the selected receipt distinction
survives source-to-IR compilation. It does not validate the mathematical
diagnostic, the model, or a psychiatric use case.

## Relationship To The Existing Psychiatric Contracts

This contract belongs between `ModelProjection` and any attempt to interpret a
model-derived state. It reinforces rather than replaces earlier distinctions:

```text
ObservationReceipt != ModelProjection
ModelProjection + fit summary != FunctionalStateSurrogate
ModelAdequacyReceipt != FunctionalContextReceipt
ModelAdequacyReceipt != MeasurementInvarianceReceipt
ModelAdequacyReceipt != CausalEffectEstimate
ModelAdequacyReceipt != ClinicalValidationReceipt
```

For example, a model might reproduce a selected symptom trajectory while its
latent-state parameters are non-recoverable, its observation model changes
across reporters, or its treatment mechanism is unidentified. Sounio should
preserve all three facts, including their failure routes, instead of allowing a
good-looking summary scalar to erase the distinctions.

## Falsifiers And Demotions

This direction should be narrowed or rejected if:

- the proposed receipts cannot produce a synthetic compile-time substitution
  failure beyond a renamed scalar or boolean;
- recovery, computation calibration, and predictive checks do not require
  materially distinct inputs or produce distinct abstention routes;
- a reviewer finds that a proposed constructor lets a fit or calibration
  receipt silently create a causal, clinical, or authority-bearing type;
- the types are presented as proving identifiability, biological truth,
  empirical validity, or patient benefit; or
- a future real-data evaluation does not predeclare its model space, synthetic
  regimes, predictive features, and interpretation scope.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: PSYCHIATRIC-MODEL-ADEQUACY-RESEARCH-20260721
Owner: Codex
Concept-IDs: SOUNIO-PSYCHIATRIC-STATE-INFERENCE; SOUNIO-COUNTERFACTUAL-AUTHORITY-ABSTENTION
Intent-Preserved: complex-system models remain expressive while fit, synthetic recoverability, computation calibration, predictive adequacy, causal interpretation, and clinical authority remain separate facts
Transformation: source-backed receipt taxonomy and collision matrix for a future library-first protocol; no language change
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a future synthetic fixture may prove selected nominal receipt substitutions are rejected across an import boundary
Claims-Forbidden: model truth, parameter validity in a person, reliability, causal identification, treatment effect, clinical utility, clinical authorization, or validation from compilation or any receipt alone
Assumptions: cited workflow literature establishes methodological distinctions; it does not provide a psychiatric mechanism, model result, or clinical recommendation
Write-Set: docs/research/psychiatric_model_adequacy_falsification_contract_2026-07-21.md
Read-Set: docs/research/psychiatric_state_inference_contract_2026-07-21.md; docs/research/psychiatric_temporal_authority_receipt_matrix_2026-07-21.md; docs/research/psychiatric_counterfactual_authority_abstention_contract_2026-07-21.md
Positive-Witness: future imported synthetic receipt chain, only after #901 checker-level acceptance
Negative-Witness: fit, recovery, inference-calibration, predictive-adequacy, causal, and clinical receipt substitutions refuse at compile time
Acceptance-Gate: future focused import-bearing compile-fail and native control on one declared source-fresh compiler artifact
Integration-Target: research documentation branch, then reviewed integration if a future executable fixture is accepted
Authoritative-Only-If: no receipt in this document is authority; any future checker result proves only the selected program boundary
```

## Integration Receipt

```text
Semantic-Outcome: model fit, synthetic recovery, inference computation calibration, and posterior-predictive adequacy are separated before any causal or clinical interpretation
Concept-Status-Before: the psychiatric contracts required explicit model identity but did not name a distinct recovery and adequacy receipt boundary
Concept-Status-After: a future library protocol has a source-backed refusal map for model-space, inference, recovery, discrepancy, and abstention states
Distinctions-Added: fit != parameter recovery; parameter recovery != model recovery; computation calibration != adequacy; adequacy != causality
Distinctions-Preserved: observation != model projection; model projection != functional state; state trajectory != causal effect; research evidence != clinical authority
Distinctions-Erased: none
Evidence-Run: source review of cited primary workflow and computational-psychiatry literature; documentation consistency checks pending
Fallback-Path: no scalar fit, calibration result, compiler pass, or documentation status is a fallback source of model or clinical authority
Legacy-Kept: existing direct psychiatric authority controls and temporal receipt research remain unchanged
```
