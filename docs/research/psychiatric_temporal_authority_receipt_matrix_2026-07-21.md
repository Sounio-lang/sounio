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

## First Import-Bearing Collision Matrix

After #901 has an accepted current-source imported-native gate, the next
fixture should use the existing ordered-path provenance surface and prove only
these substitutions fail:

| Collision | Same surface | Difference that must remain typed | Required result |
| --- | --- | --- | --- |
| Time-zero collision | Same recorded timestamp and trajectory values. | Eligibility/assignment alignment. | No conversion from raw timestamp to `TemporalOriginReceipt`. |
| Observation-process collision | Same measured values. | Scheduled versus covariate-driven observation provenance. | A scheduled-process receipt cannot satisfy an irregular-process requirement. |
| Regime collision | Same endpoint scalar. | Ordered action/history rule. | A trajectory projection cannot satisfy `RegimeDefinitionReceipt`. |
| Identification collision | Same model fit and question. | Assumption/sensitivity receipt. | Question cannot form source effect without identification. |
| Transport collision | Same source estimate. | Target context and measurement map. | Source effect cannot satisfy a target-context API. |
| Abstention collision | Same unresolved score. | Missing time origin, support, transport, or calibration. | Each gap emits distinct abstention provenance. |

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
Claims-Introduced: a future library fixture should represent time origin and observation process as evidence-bearing prerequisites rather than infer them from a trajectory
Claims-Forbidden: causal identification from timestamps or model fit; clinical authority from a receipt, a score, a reporting checklist, or compilation
Assumptions: cited work supplies methodological constraints and reporting boundaries, not a psychiatric mechanism or a validated intervention
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
Concept-Status-After: time origin, dynamic regime, observation process, identification, transport, selective risk, evaluation, and abstention have separate stated authority boundaries
Distinctions-Added: recorded time != causal time zero; observed trajectory != observation process; reporting != clinical validation
Distinctions-Preserved: ordered path != commutative endpoint; model projection != causal effect; compilation != clinical authority
Distinctions-Erased: none
Evidence-Run: node scripts/docs/sync_governance_metadata.mjs; bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Fallback-Path: no fallback compiler, model, or reporting path is evidence for clinical authority
Legacy-Kept: existing counterfactual, ordered-path provenance, and direct psychiatric authority fixtures remain unchanged
Conflicting-Lanes: #901 owns imported-native acceptance plumbing; this lane does not modify it
Next-Semantic-Interface: a #901-gated imported synthetic temporal collision fixture with explicit leaf-to-main receipt identities
```
