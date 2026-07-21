<!-- docs:meta
topic_id: repo.docs.research.psychiatric-temporal-authority-receipt-matrix-2026-07-21
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.psychiatric-temporal-authority-receipt-matrix-2026-07-21
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Time Zero Is Evidence, Not Just A Timestamp

Status: historical, source-backed research matrix and library-first roadmap

Date: 2026-07-21

Claim boundary: this document maps temporal and evidentiary distinctions for
future research software. It does not infer a causal effect, diagnose a person,
recommend a treatment, validate a clinical decision-support system, or assert
that a future Sounio receipt makes an assumption true.

## Why The Temporal Layer Matters

The psychiatric-state and counterfactual-authority contracts already separate
an observation, a state projection, a counterfactual question, an identified
effect, transport, selective prediction, abstention, and clinical validation.
Longitudinal work adds a necessary preceding boundary:

```text
same recorded time != aligned causal time zero
same symptom trajectory != same observation process
same repeated score != comparable latent construct
same action label != same dynamic treatment strategy
same fitted state transition != identified effect of that strategy
same source estimate != transported target-context estimate
same calibrated score != authority to act
```

This is especially important for state-dependent systems. A later observation
can be a consequence of earlier treatment, a reason treatment was changed, an
artifact of who was measured when, or several of these at once. A state-space
model may represent those facts. It cannot decide their causal role merely by
fitting a trajectory.

Sounio's proposed contribution is narrow: make the temporal commitments,
measurement-process commitments, and missing-evidence routes explicit enough
that a program cannot silently substitute one for another.

## What The Literature Licenses

### Time Zero Is A Design Commitment

Target-trial work treats eligibility, treatment-strategy assignment, and the
start of follow-up as a coordinated design choice. Misalignment can create
design-induced bias; a timestamp in an event log does not by itself establish
the causal origin of follow-up.

- Fu et al. (2026), [aligning eligibility and assignment at time zero](https://www.bmj.com/content/392/bmj-2025-084909.short).
- Cashin et al. (2025), [TARGET reporting statement](https://www.bmj.com/content/390/bmj-2025-087179).
- Matthews et al. (2022), [target-trial emulation principles](https://www.bmj.com/content/378/bmj-2022-071108).

This supports a `TemporalOriginReceipt` that names the eligibility event,
assignment window, follow-up start, outcome horizon, and mapping caveats. It
does not show that the chosen trial is feasible, that confounding is controlled,
or that a patient encountered a clinical decision at that time.

### A Dynamic Regime Is A Sequence Of Rules, Not A Trajectory Label

Dynamic treatment-regime literature defines a regime as a sequence of
decision rules using evolving treatment and covariate history. In observational
settings, causal interpretation depends on substantive assumptions such as
consistency, sequential exchangeability, and positivity or feasible support.

- Chakraborty and Moodie (2013), [dynamic treatment regimes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4231831/).
- Orellana et al. (2013), [dynamic regimes and the parametric g-formula](https://pmc.ncbi.nlm.nih.gov/articles/PMC3769803/).
- Coulombe et al. (2023), [covariate-driven observation times in individualized treatment rules](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248307/).

The third source makes an important refinement for Sounio: observation times
and treatment mechanisms can both depend on patient characteristics. Thus a
regularly sampled model trace and an irregular, covariate-driven clinical trace
must not receive the same observation-process authority simply because their
values are numerically aligned.

### Repeated Measurement Needs A Comparability Claim

Longitudinal measurement invariance asks whether an instrument represents the
same construct in the same metric across time, groups, reporters, or other
declared conditions. Without that evidence, a change in observed score may be
a change in item functioning, response process, language, reporting context,
or measurement model rather than a change in the intended latent construct.

- Olino (2020), [clinical applications of measurement invariance](https://pmc.ncbi.nlm.nih.gov/articles/PMC7895483/).
- Liu et al. (2017), [longitudinal invariance for ordered-categorical measures](https://pmc.ncbi.nlm.nih.gov/articles/PMC5121102/).
- Karcher et al. (2022), [measurement noninvariance in biological psychiatry](https://pmc.ncbi.nlm.nih.gov/articles/PMC9106809/).
- Horvath et al. (2025), [longitudinal PHQ-9 invariance during pharmacotherapy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11915754/).

These sources do not make every longitudinal score invalid. They instead
license a condition on its interpretation: a model must name the measurement
model and the tested comparability scope before it treats a score difference as
evidence of comparable construct change. Failed, partial, or untested
invariance should remain an explicit route to qualification, sensitivity
analysis, or abstention.

### Shift, Calibration, And Live Evaluation Remain Separate

Clinical distribution shift can involve more than observed covariate shift, and
the requirements for transportable causal effects differ from those for
predictive calibration. Selective prediction and conformal methods can provide
bounded prediction uncertainty under declared conditions; they do not identify
the effect of an intervention.

- Han (2025), [distribution shift, prediction, and causal inference in clinical AI](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2834887).
- Collins et al. (2024), [TRIPOD+AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/).
- Vasey et al. (2022), [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9).

TRIPOD+AI concerns transparent reporting of prediction-model research. DECIDE-AI
concerns reporting of early-stage live clinical evaluation. Neither is a
certificate of causal identification, utility, safety, approval, or authority
to issue a treatment instruction. That distinction is the reason the proposed
`ClinicalEvaluationReceipt` below is intentionally descriptive and
independently governed.

## Proposed Temporal Authority Matrix

The names below are proposed library vocabulary, not current parser syntax or
standard-library APIs.

| Receipt | It may state | It must carry or reference | It must not silently become |
| --- | --- | --- | --- |
| `TemporalOriginReceipt` | eligibility event, assignment window, time zero, follow-up horizon, outcome clock. | protocol identifier and data-mapping caveats. | evidence that time zero is unbiased or clinically appropriate. |
| `RegimeDefinitionReceipt` | stage-specific action rule, admissible history, action semantics, target outcome. | a well-defined strategy and decision-stage ordering. | evidence that the strategy is feasible, identified, or beneficial. |
| `ObservationProcessReceipt` | measurement schedule, trigger, missingness/visit mechanism, source-system identity. | how observations entered the record and which process is assumed. | a claim that irregular observation is ignorable. |
| `MeasurementModelReceipt` | instrument/items, response coding, reporter, language, scoring/latent-model version, and intended construct. | the measurement function and its known limits. | evidence that scores are comparable across time or context. |
| `MeasurementInvarianceReceipt` | tested comparability level and the time/group/reporter scope under a declared method. | model specification, fit/sensitivity criteria, failures, and partial-invariance decisions. | proof that the latent construct is complete, causally identified, or universally comparable. |
| `MeasurementNoninvarianceAbstentionReceipt` | failed, absent, or out-of-scope comparability evidence and its affected comparisons. | the measurement model, failure mode, and review/sensitivity route. | a zero change, a compatible trajectory, or an ignorable warning. |
| `IdentificationReceipt` | stated estimand, graph/design, consistency, exchangeability, positivity/support, sensitivity plan. | explicit assumptions and an analysis family. | proof that its assumptions hold in a real population. |
| `TransportReceipt` | source/target definition, measurement map, effect-modifier treatment, shift assumptions. | target-context evidence and a transport analysis. | target validation from a shared label or feature schema. |
| `SelectiveRiskReceipt` | loss, calibration cohort, coverage/risk target, shift assumption, expiry or review window. | calibration procedure and deployment domain. | a causal effect or action recommendation. |
| `ClinicalEvaluationReceipt` | independently governed study, workflow setting, monitoring plan, human oversight, study status. | the actual external study/governance evidence. | a compiler pass, a reporting checklist, or authorization to treat. |
| `AbstentionReceipt` | failed precondition, evidence gap, review or acquisition route. | a named missing receipt and provenance. | a silent default, zero effect, or low-confidence recommendation. |

The matrix deliberately separates two often conflated facts. A
`TemporalOriginReceipt` can make a causal question well specified while an
`ObservationProcessReceipt` can make its data-generating limitations visible.
Neither replaces the other, and neither converts a state trajectory into an
intervention effect.

Likewise, an `ObservationProcessReceipt` can say how a value reached the record
while a `MeasurementModelReceipt` says what the value was intended to measure.
Neither is a `MeasurementInvarianceReceipt`. A regular schedule with identical
numeric scores does not establish that the measurement function is stable.

## The Evidence Path Is Ordered

For an intended dynamic-regime study, the prospective authority path is:

```text
ObservationReceipt + ObservationProcessReceipt
  + TemporalOriginReceipt + RegimeDefinitionReceipt
  -> CounterfactualQuestionReceipt

CounterfactualQuestionReceipt + IdentificationReceipt
  -> SourceContextEffectEstimate

SourceContextEffectEstimate + TransportReceipt
  -> TargetContextEffectEstimate

ModelProjection + SelectiveRiskReceipt
  -> SelectivePredictionReceipt | AbstentionReceipt

TargetContextEffectEstimate + SelectivePredictionReceipt
  + ClinicalEvaluationReceipt
  -> outside the compiler and outside this research contract
```

This diagram is not a clinical workflow. It is a refusal map. In particular:

```text
TemporalOriginReceipt != IdentificationReceipt
ObservationProcessReceipt != positivity evidence
SelectiveRiskReceipt != TransportReceipt
ClinicalEvaluationReceipt != clinical authorization
```

The arrows are intentionally non-commutative. Reordering them changes the
claim. For example, applying a calibration result before deciding whether the
target setting is comparable can yield a reliable-looking score without a
transportable causal result. Choosing a time zero after inspecting an outcome
can yield a polished trajectory while changing the estimand itself.

Before comparing a longitudinal projection, a distinct measurement path is
required:

```text
ObservationReceipt + ObservationProcessReceipt + MeasurementModelReceipt
  -> RecordedMeasurementReceipt

RecordedMeasurementReceipt + MeasurementInvarianceReceipt
  -> ComparableMeasurementProjection | MeasurementNoninvarianceAbstentionReceipt
```

This is not a claim that a type checker can establish psychometric invariance.
It makes the study/model evidence and its boundary explicit, so a repeated
score cannot silently become a comparable latent-state trajectory.

## Functional Path State Is Not A Scalar

The same discipline applies one level below a clinical trajectory. A receptor
occupancy observation or an aggregate activation proxy is not a complete
functional state. At minimum, a research model that needs to distinguish
mechanisms must preserve the measurement system, pathway vector, time window,
and prior regulatory state instead of collapsing them into one total.

```text
same occupancy != same G-protein and beta-arrestin pathway vector
same activation proxy != same desensitization or internalization state
same pathway endpoint != same assay time window or measurement system
same cumulative exposure != same ordered exposure history
same functional model state != a patient-specific effect or treatment decision
```

This is a modeling boundary, not a claim about the effect of any drug in any
person. It follows the literature's warning that GPCR efficacy is
multidimensional, that time and assay context can alter an apparent bias, and
that receptor regulation can alter later responsiveness.

- Urban et al. (2007), [functional selectivity across D2-mediated effectors](https://pubmed.ncbi.nlm.nih.gov/16554739/).
- Gundry et al. (2017), [assay, cell-system, and kinetic confounding in bias assessment](https://pubmed.ncbi.nlm.nih.gov/28174517/).
- Hoare et al. (2020), [kinetic measurement of efficacy and ligand bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC7000712/).
- Kolb et al. (2020), [IUPHAR community guidance on time- and state-dependent GPCR bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC7612872/).
- Grundmann et al. (2015), [G-protein, arrestin, desensitization, and internalization as distinct GPCR processes](https://pmc.ncbi.nlm.nih.gov/articles/PMC5595354/).

The 2020 IUPHAR guidance is particularly useful here: the reported bias can be
cell-phenotype and physiological-state dependent, signaling efficacy can
change over time, and an appropriate time point may differ across pathways.
Therefore a single endpoint must not silently stand in for the complete time
course or for another assay system.

### Proposed Functional-Path Receipts

These are proposed library-level names only. They are not current Sounio
syntax, receptor measurements, clinical biomarkers, or validated
pharmacodynamic estimands.

| Receipt | It may state | It must carry or reference | It must not silently become |
| --- | --- | --- | --- |
| `FunctionalPathwayObservation` | measured pathway-specific readouts under a declared assay. | receptor, ligand/exposure fixture, cell/assay system, time window, and raw observation provenance. | a pathway-independent efficacy value. |
| `FunctionalPathStateReceipt` | a model's declared vector of G-protein, beta-arrestin, and other chosen pathway coordinates. | the observation/model mapping and coordinate definitions. | a clinical state, a causal effect, or proof that the vector is complete. |
| `KineticWindowReceipt` | onset, peak, duration, sample window, and aggregation rule. | clock origin, sampling rule, and assay context. | a timeless property of the ligand or receptor. |
| `DesensitizationStateReceipt` | the modelled regulatory state relevant to a later response. | preceding exposure/order and the modelled regulatory mechanism. | evidence of in-vivo tolerance, global receptor loss, or a treatment instruction. |
| `InternalizationStateReceipt` | declared compartment/trafficking state in a model or assay. | measurement method and temporal window. | a substitute for functional-path or clinical-outcome evidence. |
| `FunctionalPathDivergenceReceipt` | that two declared vectors or trajectories differ despite a selected shared scalar. | the shared projection, both path identities, and the comparison domain. | superiority, safety, or clinical relevance. |

The crucial refusal is:

```text
ActivationProxyReceipt != FunctionalPathStateReceipt
FunctionalPathStateReceipt != DesensitizationStateReceipt
KineticWindowReceipt != assay-independent ligand property
FunctionalPathDivergenceReceipt != clinical recommendation
```

Thus a program may preserve an equal scalar as a comparison fact while still
requiring explicit evidence before it calls the underlying functional paths
equivalent. The distinction encodes equifinality: an observed endpoint can be
shared while the upstream routes and later reachable states differ.

## First Import-Bearing Collision Matrix

After #901 has an accepted current-source imported-native gate, the next
fixture should use the existing ordered-path provenance surface and prove only
these substitutions fail:

| Collision | Same surface | Difference that must remain typed | Required result |
| --- | --- | --- | --- |
| Time-zero collision | Same recorded timestamp and trajectory values. | Eligibility/assignment alignment. | No conversion from raw timestamp to `TemporalOriginReceipt`. |
| Observation-process collision | Same measured values. | Scheduled versus covariate-driven observation provenance. | A scheduled-process receipt cannot satisfy an irregular-process requirement. |
| Regime collision | Same endpoint scalar. | Ordered action/history rule. | A trajectory projection cannot satisfy `RegimeDefinitionReceipt`. |
| Measurement-invariance collision | Same instrument label and observed score change. | Measurement model, time/group/reporter scope, and tested comparability. | A score difference cannot satisfy a comparable-trajectory API. |
| Identification collision | Same model fit and question. | Assumption/sensitivity receipt. | Question cannot form source effect without identification. |
| Transport collision | Same source estimate. | Target context and measurement map. | Source effect cannot satisfy a target-context API. |
| Abstention collision | Same unresolved score. | Missing time origin, support, transport, or calibration. | Each gap emits distinct abstention provenance. |
| Functional-path collision | Same occupancy or chosen activation scalar. | G-protein/beta-arrestin pathway vector and assay domain. | An activation proxy cannot satisfy `FunctionalPathStateReceipt`. |
| Kinetic-window collision | Same selected pathway endpoint. | Time-zero, sample window, and aggregation rule. | A single endpoint cannot satisfy a time-course requirement. |
| Regulatory-history collision | Same cumulative synthetic exposure. | Ordered prior exposure and modelled desensitization state. | A cumulative total cannot satisfy `DesensitizationStateReceipt`. |

The fixture must use synthetic constants only. It should have an imported leaf
and a main module so that the selected receipt identities cross the source,
resolver, checker, IR, and native path. A direct single-module pass is a useful
control; it cannot discharge this imported-native requirement.

## What A Type System Can Actually Do

The language can verify that a program carries a receipt, preserves a reason
for abstaining, and fails when a caller passes a temporal label where an
identification or transport receipt is demanded. The language cannot inspect a
future population, detect all unmeasured confounders, validate a causal graph,
prove a visit process ignorable, establish clinical utility, obtain consent, or
approve a medical action.

That is not a limitation to conceal. It is the purpose of the construction:
place the empirical and institutional questions at explicit interfaces rather
than allowing them to disappear behind a model score or successful compilation.

## Falsifiers And Demotions

This proposal should be revised, narrowed, or rejected if:

- the temporal-origin and observation-process distinctions cannot create a
  compile-fail collision without a new language feature;
- an imported native fixture erases a selected receipt identity despite a
  passing direct control;
- equal scalar functional proxies can substitute for unequal pathway vectors,
  kinetic windows, or regulatory-history receipts;
- a repeated score or a shared instrument label can satisfy a comparable
  latent-trajectory requirement without a declared measurement model and
  invariance scope;
- the proposed `ObservationProcessReceipt` merely duplicates values already in
  `TemporalOriginReceipt` and adds no discriminating test;
- a documentation or API path promotes a selective-risk receipt into a causal
  or transport claim;
- a clinical-evaluation label is used as a substitute for independently
  governed evidence; or
- the receipt names cannot state the result that would demonstrate their
  assumptions were inadequate.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: psychiatric-temporal-authority-research-v0
Owner: codex-root
Concept-IDs: SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-PHYSICAL-OBSERVATION; SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: temporal order, observation provenance, causal assumptions, uncertainty, and clinical authority remain distinct facts
Transformation: literature-backed temporal authority matrix mapped to existing ordered-path and receipt work; no parser, effect, IR, or ontology change
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a future library fixture should represent time origin, observation process, measurement model/invariance, functional-path state, kinetic window, and regulatory history as evidence-bearing prerequisites rather than infer them from a trajectory or scalar proxy
Claims-Forbidden: causal identification from timestamps or model fit; comparable latent change from a repeated score; pathway equivalence from a scalar proxy; clinical authority from a receipt, a score, a reporting checklist, or compilation
Assumptions: cited work supplies methodological, psychometric, and assay-context constraints, not a psychiatric mechanism, a patient-specific effect, or a validated intervention
Write-Set: docs/research/psychiatric_temporal_authority_receipt_matrix_2026-07-21.md; docs/governance/topic-registry.v1.json; docs/governance/DOCS_AUTHORITY_MATRIX.md
Read-Set: docs/research/psychiatric_state_inference_contract_2026-07-21.md; docs/research/psychiatric_counterfactual_authority_abstention_contract_2026-07-21.md; tests/compiler/ordered_path_provenance_imported_leaf.sio; tests/compiler/ordered_path_provenance_imported_main.sio
Positive-Witness: direct psychiatric authority receipt control 8e8d4ccce; existing ordered-path imported provenance surface
Negative-Witness: psychiatric_counterfactual_question_cannot_authorize.sio; future imported temporal collision matrix
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Integration-Target: internal research planning, then #901-gated imported synthetic fixture
Authoritative-Only-If: cited constraints, exact receipt boundaries, and no-clinical-authority wording remain aligned
```

## Integration Receipt

```text
Semantic-Outcome: temporal-origin and observation-process evidence requirements are made explicit for the future authority fixture
Concept-Status-Before: order provenance and authority receipts were specified without a dedicated target-trial time-zero and observation-process matrix
Concept-Status-After: time origin, dynamic regime, observation process, measurement model/invariance, functional-path state, kinetic window, identification, transport, selective risk, evaluation, and abstention have separate stated authority boundaries
Distinctions-Added: recorded time != causal time zero; observed trajectory != observation process; repeated score != comparable latent construct; equal activation proxy != equal functional-path state; reporting != clinical validation
Distinctions-Preserved: ordered path != commutative endpoint; model projection != causal effect; compilation != clinical authority
Distinctions-Erased: none
Evidence-Run: node scripts/docs/sync_governance_metadata.mjs; bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Fallback-Path: no fallback compiler, model, or reporting path is evidence for clinical authority
Legacy-Kept: existing counterfactual, ordered-path provenance, and direct psychiatric authority fixtures remain unchanged
Conflicting-Lanes: #901 owns imported-native acceptance plumbing; this lane does not modify it
Next-Semantic-Interface: a #901-gated imported synthetic temporal collision fixture with explicit leaf-to-main receipt identities
```
