# Psychiatric Decision Boundaries: Nonregularity Is Not A Tie

**Status:** research architecture and falsifiable representation contract.

**Not clinical guidance.** Nothing in this document selects, ranks, starts,
stops, or doses a treatment for a person. It specifies what a future Sounio
model must retain before it can honestly say that its *research comparison* is
ambiguous, that its inference is nonregular, or that it declines to represent a
single best action.

This contract extends the psychiatric state-inference, temporal-authority, and
nonassociativity contracts:

- [state inference](psychiatric_state_inference_contract_2026-07-21.md)
- [temporal authority and measurement](psychiatric_temporal_authority_receipt_matrix_2026-07-21.md)
- [counterfactual authority and abstention](psychiatric_counterfactual_authority_abstention_contract_2026-07-21.md)
- [order and nonassociativity](psychiatric_nonassociativity_representation_contract_2026-07-21.md)

Its central refusal is simple:

```text
largest fitted score != uniquely best action
confidence interval containing zero != equality of actions
nonregular inference != invalid causal estimand
decision candidate set != clinical recommendation
causal estimand != authority to act for an individual
decision ambiguity != evidence of nonassociativity
```

The distinction is not pedantry. A system that collapses these statements can
turn statistical fragility into unwarranted decisiveness exactly where a person
may need a careful, plural, contextual conversation instead.

## 1. The Boundary We Need

At a decision time `t`, let a history `H_t` be the information declared
available at that point, `A_t` a *feasible research-comparison option*, and
`V(a, H_t)` an explicitly declared value estimand. For two options, a decision
contrast is:

```text
Delta(a, b | H_t) = V(a, H_t) - V(b, H_t)
```

The line above is deliberately incomplete until the program carries:

1. the causal or predictive interpretation of `V`;
2. the time-zero and history construction;
3. the outcome horizon and missing-data/observation process;
4. action feasibility and support;
5. the outcome aggregation or preference assumptions;
6. an uncertainty procedure that is appropriate for the estimator's regime.

Without these, a positive fitted `Delta` is a model output, not a basis for a
unique decision claim.

### 1.1 Four Different Reasons A Singleton Can Be Unjustified

The same visible output, such as `score(a) > score(b)`, can hide four different
problems. Sounio should keep them separate.

| Boundary | What is missing or unstable | What must *not* be concluded |
|---|---|---|
| Identification | A causal estimand is not identified under stated assumptions, or support/feasibility is absent | `a` causes a better outcome |
| Statistical decision margin | The contrast is near zero relative to uncertainty | `a` is uniquely best |
| Nonregular inference | The sampling behavior near an exceptional law invalidates a routine approximation | a nominal Wald interval settles the choice |
| Competing outcomes or unknown preference | The scalar value hides a trade-off | one option dominates for this person |

These can coexist, but none implies the others. In particular, inference can be
nonregular even when an estimand is meaningful; conversely, regular inference
does not repair a missing causal identification assumption.

## 2. What DTR Literature Actually Supplies

Dynamic treatment-regime (DTR) methods formalize a sequence of history-indexed
decision rules. That is useful for Sounio because it makes the decision point,
history, and future outcome horizon explicit. It does **not** turn a fitted
rule into a clinical command.

The technical literature gives two particularly valuable warnings.

### 2.1 Nonregularity Is A Sampling-Inference Problem

For optimal DTR estimation, treatment-effect parameters at earlier stages can
be nonregular under some data-generating laws. In that region, the limiting
behavior of an estimator is sensitive to local perturbations, and routine
Wald-style intervals may have poor coverage or produce misleading certainty.
The practical danger is strongest when a treatment contrast is zero or small
relative to data noise.

- Chakraborty et al. (2010), [inference for nonregular parameters in optimal DTRs](https://pmc.ncbi.nlm.nih.gov/articles/PMC2891316/), explains that nonregularity can yield biased estimates and invalid conventional confidence intervals.
- Laber et al. (2014), [technical challenges and applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC4209714/), emphasizes that simulation and inferential evaluation must include zero, small, and large treatment effects; it also treats a confidence set spanning zero as insufficient evidence for a uniquely best treatment.
- Chakraborty and Moodie (2014), [DTR review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4231831/), separates confidence for regime parameters and values from the broader causal-design assumptions a regime requires.

This is not a license to announce an empirical tie. A wide or zero-spanning
interval means that the declared procedure does not establish a unique choice
at its stated resolution; it does not prove equality of two treatment effects.

### 2.2 More Than One Outcome Can Legitimately Preserve More Than One Option

Set-valued DTR work addresses a separate issue: benefit and burden can be
competing outcomes, and preference trade-offs may not be available or stable.
Rather than smuggling a scalar trade-off into the model, a set-valued regime can
retain the non-inferior options under the declared outcome ordering.

- Laber, Lizotte, and Ferguson (2014), [set-valued dynamic treatment regimes for competing outcomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC3954452/), proposes decision rules that output a set of treatments rather than a singleton when outcomes conflict or preferences are heterogeneous.

This literature motivates an *epistemic representation*: the model may emit a
set of research candidates and the reason the set was preserved. It does not
grant an autonomous system authority to recommend or carry out a treatment.

## 3. Proposed Receipt Vocabulary

The following receipts are a design vocabulary. They are intentionally not yet
native Sounio syntax. They should become typed constructors only after the
source-fresh compiler path accepts the import-bearing psychiatric suite.

```text
DecisionEstimandReceipt
    outcome, horizon, intervention semantics, target population, value scale

DecisionHistoryReceipt
    time_zero, information cut, observation-process scope, available history

ActionFeasibilityReceipt
    candidate action label, availability, support, contraindication/eligibility scope

DecisionContrastReceipt
    estimand, action_pair, history, estimated_delta, estimation procedure

ContrastUncertaintyReceipt
    interval or posterior summary, resampling/design details, finite-sample scope

InferenceRegularityReceipt
    regularity assessment, exceptional-law sensitivity, procedure adequacy scope

CompetingOutcomeReceipt
    outcome vector, ordering relation, declared or unresolved preference input

DecisionCandidateSetReceipt
    candidate set, basis, retained uncertainty/preference rationale

UniqueActionAbstentionReceipt
    why uniqueness was not established, missing receipt links, permitted next research step

ExternalClinicalAuthorizationBoundary
    external, role-bound authorization boundary; deliberately not a Stage 3
    constructible nominal library record
```

`ActionFeasibilityReceipt` is intentionally not a pharmacological eligibility
engine. It only records what feasibility/support statement a research model
assumed. A clinical feasibility or safety conclusion requires a separate,
validated clinical pathway and appropriate human authority.

### 3.1 Authority Is One-Way

The intended construction order is:

```text
DecisionEstimandReceipt
  + DecisionHistoryReceipt
  + ActionFeasibilityReceipt
  + DecisionContrastReceipt
  + ContrastUncertaintyReceipt
  + InferenceRegularityReceipt
  -> DecisionCandidateSetReceipt | UniqueActionAbstentionReceipt
```

But the reverse direction is prohibited:

```text
point estimate                   != DecisionCandidateSetReceipt
confidence interval              != InferenceRegularityReceipt
nonregularity warning            != DecisionCandidateSetReceipt
DecisionCandidateSetReceipt      != external clinical authorization
UniqueActionAbstentionReceipt    != causal effect failure
```

The first forbidden arrow matters: a warning alone does not decide how a set
should be formed. A set requires its contrast basis, uncertainty rule,
feasibility scope, and, for multi-outcome cases, an explicit ordering or a
record that preference remains unresolved.

## 4. A Falsifiable Decision Contest

Sounio should not make `CandidateSet` the default merely because it feels
cautious. It should be earned by a predeclared contest between representations.

### 4.1 Competing Models

```text
PointChoiceModel
    output: argmax_a estimated V(a, H_t)

RegularityAwareChoiceModel
    output: singleton only when its declared uncertainty and regularity criteria are met

SetValuedComparisonModel
    output: declared non-inferior candidate set and its outcome/preference basis

AbstainingComparisonModel
    output: no candidate set when identification, support, or inference prerequisites fail
```

No model gets to borrow authority from another:

```text
PointChoiceModel output != RegularityAwareChoiceModel output
Candidate set cardinality > 1 != evidence all candidates are equal
Abstention != proof no beneficial action exists
```

### 4.2 Synthetic Discriminators

Before any empirical use, a future suite should include synthetic worlds with
known construction. The suite should distinguish the models above rather than
reward them for returning the same formatted answer.

| Case | Declared construction | Required representation result |
|---|---|---|
| Stable separated contrast | A contrast far from zero, supported actions, and a validated regular procedure | a singleton *research comparison* is allowed; clinical authority is still absent |
| Near-zero contrast | Contrast is zero or small relative to uncertainty | no unsupported unique-best claim; set or abstention depends on its predeclared rule |
| Nonregular exceptional-law neighborhood | A procedure whose ordinary approximation is known to be inadequate in the constructed neighborhood | `InferenceRegularityReceipt` records inadequacy; routine interval cannot mint uniqueness |
| Competing benefit/burden outcomes | No scalar preference trade-off is supplied | preserve a candidate set or explicit preference abstention |
| Unsupported or unidentified comparison | Positivity/feasibility/causal prerequisites fail | `UniqueActionAbstentionReceipt`, not a candidate-ranking shortcut |

Every synthetic world needs a deliberately wrong implementation control. For
example, replacing `RegularityAwareChoiceModel` with an unqualified argmax must
fail the near-zero and nonregular cases. Otherwise the test only checks that an
output exists.

### 4.3 Constructive Abstention Is Not An Automatic Data Request

When uniqueness is not established, a humane research system should be able to
say *why* the comparison is unresolved. It must not infer that any additional
measurement is therefore warranted. A request for information is itself a
decision with a target, a timing, an implementation path, and a burden.

Health decision-analysis literature calls this distinction value of information
(VoI). The expected value of sample information is decision-relative: it asks
whether potential data could improve a **declared future decision** under a
declared model, rather than rewarding uncertainty reduction for its own sake.
Importantly, implementation-aware VoI work warns that the usual idealization of
instant, complete uptake after learning is not realistic in health settings.

- Ades et al. (2004), [expected value of sample information in medical decision modeling](https://pubmed.ncbi.nlm.nih.gov/15090106/), frames further research as a way to reduce decision uncertainty, with the relevant population and decision context made explicit.
- Andronis and Barton (2015), [adjusting value of information for implementation](https://pubmed.ncbi.nlm.nih.gov/26566775/), shows why an information-value calculation that assumes immediate, complete implementation can overstate what further research changes in practice.
- Chakraborty and Moodie (2014), [dynamic treatment regimes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4231831/), grounds the sequential setting: later action and observation depend on the evolving history rather than an isolated score.

The resulting refusal map is:

```text
high predictive entropy != high decision value of information
DecisionUncertaintyReceipt != InformationValueAssessmentReceipt
InformationValueAssessmentReceipt != ResearchAcquisitionCandidateReceipt
ResearchAcquisitionCandidateReceipt != clinical test order
implementation assumption != observed uptake
more observations != more valid observations
```

The last line connects this contract to the temporal/measurement layer. A new
record can still be non-comparable, selectively observed, mistimed, or outside
the support required by the question it was meant to reduce.

#### Receipt Vocabulary For The Acquisition Boundary

```text
DecisionUncertaintyReceipt
    contrast uncertainty, regularity status, unresolved candidate basis

InformationValueQuestionReceipt
    declared decision target, candidate action set, estimand, possible observation,
    analysis rule, and decision rule that the observation could change

InformationValueAssessmentReceipt
    method, uncertainty model, assumed observation accuracy, declared value/burden
    scale, sensitivity domain, and implementation assumptions

ObservationValidityScopeReceipt
    timing, observation process, measurement/invariance scope, and known failures

AcquisitionBurdenScopeReceipt
    research-side burden, delay, accessibility, consent/governance boundary, and
    which of these inputs remain externally unresolved

ResearchAcquisitionCandidateReceipt
    bounded research proposal with its value-of-information rationale and all
    stated unresolved implementation/burden limits

AcquisitionAbstentionReceipt
    the exact missing decision, validity, burden, implementation, or authority
    prerequisite that prevents even a research acquisition candidate

ExternalAcquisitionAuthorizationBoundary
    external, role-bound authorization boundary; deliberately not a Stage 3
    constructible nominal library record
```

An `InformationValueAssessmentReceipt` cannot be constructed merely from an
uncertainty scalar. It has to name the decision it is supposed to inform and a
counterfactual rule for how that decision would differ after information. A
candidate cannot hide the social and operational bridge between a study result
and later use by treating uptake as a fact.

#### A Small Discriminator Suite

The first synthetic contest should be deliberately simple enough to falsify
the category distinctions:

| Case | Hold fixed | Vary | Required result |
|---|---|---|---|
| Same entropy, different decision threshold | uncertainty magnitude | declared action rule and consequence scale | distinct VoI assessments; entropy cannot stand in for decision value |
| Same information model, different uptake assumption | signal quality and candidate action set | implementation path | distinct implementation-scoped assessment; assumed uptake is not observed uptake |
| Same proposed observation, invalid measurement scope | observation label and nominal accuracy | timing, visit process, or invariance evidence | `AcquisitionAbstentionReceipt`, not a value shortcut |
| Same estimated information value, unresolved burden/authority | model estimate | burden scope or external authorization | no clinical order and, if prerequisites are missing, no research candidate |

These are not patient simulations. They are category-collision controls. An
implementation that maps `uncertainty -> acquire data` must fail them, because
it erases the very decision, measurement, burden, and authority distinctions
the contract exists to preserve.

## 5. The Nonassociativity Boundary

Decision ambiguity is not evidence that a history carrier has a nonzero
associator. A near-tie can arise with an entirely associative transition model;
a competing-outcome set can arise at one decision point with no temporal
aggregation at all; and a nonregular estimator is a property of an inferential
functional, not automatically a property of the state algebra.

```text
nonregular decision inference != parenthesization sensitivity
candidate set != AggregationBoundaryReceipt
near-zero contrast != nonzero associator
```

If a future model claims bracket sensitivity, it must still pass the separate
ordered-versus-bracketed representation contest in the nonassociativity
contract. `InferenceRegularityReceipt` cannot discharge that obligation.

## 6. Import-Bearing Type Collisions (Deferred Until #901)

Once the #901 compiler blocker has a source-fresh artifact that compiles and
executes the imported D11/D12 witnesses with no fallback, add a focused
psychiatric suite. The fixtures must import receipt definitions from a library;
same-file aliases are not a substitute for an import-boundary proof.

```text
run-pass/psychiatric_nonregular_candidate_set_witness.sio
    constructs a DecisionCandidateSetReceipt only from declared estimand,
    history, feasibility, contrast, uncertainty, regularity, and outcome inputs.

compile-fail/psychiatric_argmax_cannot_authorize_unique_action.sio
    expected UniqueActionAuthorityReceipt
    found PointEstimateReceipt

compile-fail/psychiatric_nonregular_warning_cannot_prove_tie.sio
    expected EqualityOfEffectsReceipt
    found InferenceRegularityReceipt

capability-only/psychiatric_candidate_set_cannot_authorize_clinical_action.sio
    reserved for a separately justified opaque/capability design; it must not
    be implemented as a public nominal authorization record in Stage 3

compile-fail/psychiatric_associator_probe_cannot_explain_decision_ambiguity.sio
    expected ParenthesizationSensitivityReceipt
    found InferenceRegularityReceipt

compile-fail/psychiatric_uncertainty_cannot_authorize_acquisition.sio
    expected ResearchAcquisitionCandidateReceipt
    found DecisionUncertaintyReceipt

capability-only/psychiatric_information_value_cannot_authorize_clinical_test.sio
    reserved for the same capability-only acceptance surface, never an
    ordinary nominal Stage 3 fixture
```

The associator negative is deliberately conceptual. It prevents a mathematically
interesting carrier-level observation from being smuggled in as an explanation
for an uncertainty boundary that has not been tested.

The two acquisition negatives close a different escape hatch. They prevent an
otherwise well-formed uncertainty or information-value calculation from
becoming an instruction to obtain a measurement. A future positive control must
construct `ResearchAcquisitionCandidateReceipt` only from the declared
decision, information-value, observation-validity, burden, and external-limit
receipts.

The two `capability-only` entries are design constraints, not #901-gated Stage
3 compile fixtures. Any public nominal authorization record would prove only a
selected type mismatch while leaving the named record forgeable. They become
executable acceptance cases only if a separately reviewed opaque/capability
mechanism supplies an external authority token, a threat model, and attacker
fixtures. Until then, the research package has no authorization constructor at
all.

## 7. Acceptance Standard

This contract is satisfied only when all of the following are true:

1. The model declares the estimand, history boundary, and feasibility/support
   scope before comparing options.
2. A unique research comparison is withheld when its own regularity and
   uncertainty criteria fail.
3. A candidate set records whether it comes from uncertainty, competing
   outcomes, unresolved preference, or another declared rule.
4. The implementation never equates a zero-spanning interval with equality.
5. The implementation never equates a candidate set with clinical authority.
6. Synthetic controls show that an unqualified argmax cannot pass the
   near-zero/nonregular cases.
7. A research acquisition candidate requires a decision-relative information
   question, observation-validity scope, burden/implementation scope, and does
   not authorize a clinical measurement.
8. Any claim about nonassociativity separately passes the representation
   contest; decision ambiguity alone does not count.

Failure of any item produces an abstention or an explicit model discrepancy,
not a silently confident single action.

## 8. Honest Scope

The present work contributes a research-language boundary and a future typed
receipt design. It does not yet provide:

- a validated psychiatric outcome model;
- a causal estimate for any intervention;
- a calibrated individual treatment rule;
- an implementation of nonregular confidence procedures;
- a clinical decision-support authorization path;
- evidence that a patient-specific action is safe, beneficial, or indicated.

That limitation is part of the design. The useful thing Sounio can do here,
before it knows enough to act, is preserve the difference between a score, an
uncertain comparison, a plural candidate representation, and a decision that
properly remains with accountable human care.

## 9. Semantic-Lane Declaration

This is a research-boundary lane. It composes existing concepts and introduces
no compiler, IR, or standard-library semantics.

```text
Semantic-Lane-ID: psychiatric-nonregular-decision-boundary-research-v0
Owner: Codex psychiatric state-inference research lane
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-EPISTEMIC-NUMERIC-VALUE; SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: a research system may preserve decision ambiguity, uncertainty, and unresolved preference without converting a fitted score, a causal estimand, an associator observation, or a candidate set into clinical authority
Transformation: literature-backed DTR nonregularity and set-valued comparison constraints are represented as prospective receipt boundaries and synthetic contest requirements
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: future synthetic psychiatric fixtures should distinguish a declared decision contrast, inference regularity, candidate-set basis, decision-relative information value, research acquisition candidate, unique-action abstention, and an external clinical-authorization boundary
Claims-Forbidden: equality of effects from a zero-spanning interval; unique patient action from argmax or a candidate set; research acquisition from uncertainty alone; clinical test order from information value; causal identification from regular inference; nonassociativity from decision ambiguity; clinical recommendation, treatment selection, dosing, safety, or validation
Assumptions: cited DTR sources constrain statistical representation and evaluation; they do not validate a psychiatric mechanism, an individual action, a clinical workflow, or a future Sounio package
Write-Set: docs/research/psychiatric_nonregular_decision_boundary_contract_2026-07-21.md; docs/governance/topic-registry.v1.json; docs/governance/DOCS_ACCEPTANCE_REPORT.md
Read-Set: FOUNDER_INTENT.md; AGENTS.md; docs/internal/concepts/{science-research-boundary,epistemic-numeric-value,ordered-path-provenance,nonassociative-order}.md; docs/research/psychiatric_{state_inference,temporal_authority_receipt_matrix,counterfactual_authority_abstention,nonassociativity_representation}_contract_2026-07-21.md
Positive-Witness: future import-bearing candidate-set receipt construction that requires declared estimand, history, feasibility, contrast, uncertainty, and regularity inputs
Negative-Witness: future imported argmax-to-authority, nonregular-warning-to-equality, candidate-set-to-clinical-authority, and associator-to-decision-ambiguity substitutions
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh; after #901, the focused imported psychiatric receipt suite on one source-fresh ELF with no fallback
Integration-Target: internal research planning, then the #901-gated imported synthetic fixture
Authoritative-Only-If: the sources, stated limitations, current Concept-ID contracts, and source-fresh test evidence remain aligned
```

```text
Semantic-Outcome: nonregular statistical inference, near-zero decision contrast, competing-outcome ambiguity, decision-relative information value, research acquisition, and clinical authority are represented as distinct research boundaries
Concept-Status-Before: psychiatric research contracts retained abstention and authority boundaries but had no dedicated nonregular-versus-unique-action contract
Concept-Status-After: a point estimate, inferential regularity assessment, candidate set, information-value assessment, research acquisition candidate, equality claim, and external clinical-authorization boundary have separate prospective roles
Distinctions-Added: point estimate != unique action; nonregularity != equality; uncertainty != information value; information value != acquisition authorization; candidate set != recommendation; decision ambiguity != nonassociativity
Distinctions-Preserved: research model != empirical result; causal estimand != authority; compiler success != clinical validation
Distinctions-Erased: none
Evidence-Run: literature review; git show --check; bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Fallback-Path: no fallback path is evidence for a unique action or clinical authority
Legacy-Kept: existing counterfactual, deferral, acquisition, and authority-boundary documentation remains unchanged
Conflicting-Lanes: generated governance metadata is owned by the active PBPK compiler lane; no shared generated artifact was edited here
Next-Semantic-Interface: imported synthetic receipt constructors only after #901 source-fresh D11/D12 compile-and-execute acceptance
```
