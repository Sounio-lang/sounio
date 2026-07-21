<!-- docs:meta
topic_id: repo.docs.research.psychiatric-informative-observation-contract-2026-07-21
authority: historical
audience: researchers
last_validated: 2026-07-21
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.psychiatric-informative-observation-contract-2026-07-21
-->

<!-- docs:status-note:start -->
> Docs status: historical
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# An Observation Is Not A Neutral State Readout

Status: historical, source-backed research boundary and prospective
library-first contract

Date: 2026-07-21

Claim boundary: this document proposes a representation for research evidence
about how an observation entered a record. It does not diagnose a person,
interpret an individual's absence from care, establish that an assessment
changed anyone's state, estimate a treatment effect, or authorize a clinical
action. The proposed names are not current Sounio syntax or standard-library
APIs.

## The Deeper Problem

The psychiatric-state and temporal-authority contracts already require an
**ObservationProcessReceipt**. This document refines, rather than replaces, that
umbrella. An observed score is not only a value at a time. It has at least four
separable histories:

~~~
opportunity to be observed
!= presence in a care or research channel
!= selection of a particular measure
!= the measurement act and recorded response
~~~

Those histories can be associated with an evolving state, prior actions,
clinician or participant choices, protocol rules, access, a digital channel,
and unrecorded causes. The act of assessment can also be a possible event in a
trajectory rather than merely a transparent window onto one. It must never be
assumed to be state-changing; whether it is reactive is a design- and
evidence-bound question.

The resulting distinctions are intentionally sharper than a single missingness
flag:

~~~
same recorded symptom value != same observation route
same unobserved interval != evidence of stability
same visit count != same measurement-selection mechanism
same regularized time grid != same actual observation history
same predictive utility != causal identifiability or transportability
assessment act != established intervention effect
~~~

This matters directly for the founder's non-associative intuition. In an
ordered trajectory, an encounter, a prompt, a response, a clinical action, and
a later encounter are not interchangeable tokens. They can select different
states, disclose different information, alter future monitoring, or possibly
participate in the subject-system interaction itself.

## What The Literature Licenses

### Presence, Observation, And Missingness Are Not One Variable

Routine-health-data literature distinguishes informative presence from
informative observation. Presence concerns whether there is data for a person
at a time; observation concerns the timing, frequency, or intensity of
longitudinal measurements. In opportunistic records, patient and clinician
decisions can determine whether a value is collected and which value is
recorded. That is not the same situation as a value missing from a prospectively
scheduled measurement.

- Ryan et al. (2021), [informative presence and observation in clinical risk prediction](https://academic.oup.com/jamia/article/28/1/155/5961436).
- Goldstein et al. (2019), [when informative visit processes bias EHR inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC6857502/).
- Du, Shi, and Mukherjee (2024), [clinically informative visiting processes](https://arxiv.org/abs/2410.13113).

The distinction is useful in two different ways, which must not be collapsed.
For descriptive or predictive work, a monitoring pattern may contain
information available at prediction time. For causal or explanatory work, the
same pattern can be a selection mechanism or a time-varying process that must
be modeled, bounded, or placed in an abstention route. Neither use turns a
recorded pattern into a direct observation of an unmeasured state.

### The Measurement Act Can Be a Protocol Event

Repeated self-report and ambulatory-assessment designs vary in prompts,
response windows, instruments, channels, and compliance. The literature treats
measurement reactivity as a question to report and, where possible, test; it is
not sound to assume either universal reactivity or universal neutrality.

- Wrzus and Neubauer (2023), [EMA design, compliance, and measurement-reactivity reporting](https://journals.sagepub.com/doi/10.1177/10731911211067538).
- French et al. (2020), [recommendations on reactions to measurement in trials](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614249/).
- Kirtley et al. (2021), [momentary-assessment designs for mood and anxiety](https://pubmed.ncbi.nlm.nih.gov/34079492/).

Therefore, a model may represent a prompt, assessment channel, contact, or
feedback loop as part of its history. It may not infer from that representation
that the measurement changed a state, improved a condition, or has no effect.
Those are separate empirical questions with their own design, comparator,
outcome, and uncertainty requirements.

### Observation Features Are Context-Bound Predictors

When a model uses visit intensity or the presence of a measurement as a
predictor, it may be using care behavior as well as a biological or psychiatric
signal. The same literature warns that these associations may fail to travel if
monitoring rules, care access, clinical practice, or deployment feedback alter
the observation process. A model that changes monitoring can also change the
input process it later consumes.

This is especially relevant to psychiatric systems: a self-report prompt, a
remote-monitoring notification, a clinician review, and an emergency contact
are not interchangeable manifestations of one latent severity variable. Their
meaning must name the channel and the decision process that placed them in the
history.

## A Minimal Generative Separation

The following is a modeling aid, not an identified causal graph or a claim that
all deployments have these variables.

~~~
state history             S(<=t)
prior clinical/system acts T(<t)
context and access         C(<=t)
protocol and observer      R(<=t)

opportunity                O(t)
presence/contact           P(t)
measurement selection      M(t)
measurement act            A(t)
recorded value             Y(t)
~~~

One possible ordered relation is:

~~~
S, T, C, R -> O -> P -> M -> A -> Y
                         |         |
                         +-> future monitoring and state history
~~~

The arrows are not a universal scientific fact. They state what a future model
must be able to distinguish when the study's design says a dependency is
plausible. In particular:

~~~
P(t) = 0 does not encode a zero symptom value
M(t) = 0 does not imply a measure was planned and missed
A(t) is not an intervention merely because it is an event
Y(t) is not a state value merely because it is numerically dense
~~~

This separation keeps a useful possibility without overclaiming it: a model
can learn that its observation process is informative for a narrowly declared
prediction task, while the receipt makes explicit that this information can
depend on a care system rather than travel as a person-level property.

## The Dyadic Boundary

The relevant object is not a solitary patient vector and not an omniscient
observer. It is an interactional record whose provenance includes at least the
subject-facing channel, the observing or care system, and the protocol that
linked them. A future Sounio model should be able to preserve this relation
without inventing a psychological interpretation for it.

~~~
subject state + observer-system rule + channel + ordered event history
  -> recorded observation

recorded observation
  != subject state alone
  != observer intent alone
  != causal effect of the act of observing
~~~

This is a productive use of the dyadic construct. It says that a data point is
an event at an interface, with a route through both a lived context and a
measurement/care system. The language need not choose a metaphysical theory of
mind to retain that interface as provenance.

## Proposed Receipt Taxonomy

These are prospective library vocabulary. **ObservationProcessReceipt** remains
the umbrella receipt in the temporal-authority matrix; the receipts below make
its internal claims discriminable.

| Receipt | May state | Must carry or reference | Must not silently become |
| --- | --- | --- | --- |
| **ObservationOpportunityReceipt** | scheduled or unscheduled opportunity, eligibility, time basis, and protocol window. | protocol rule, source-system scope, and known opportunity gaps. | proof that a non-observation is missing-at-random. |
| **EncounterPresenceReceipt** | factual encounter, channel presence, or record-presence event. | channel/system identity and event provenance. | a severity judgment, a symptom value, or a causal covariate by default. |
| **MeasurementSelectionReceipt** | which instrument/item/lab/prompt was selected conditional on the encounter or opportunity. | selection rule, actor/system, trigger, and selection scope. | a generic missingness flag or an ignorable sampling mechanism. |
| **MeasurementActReceipt** | prompt/assessment/contact event, respondent, channel, timing, and feedback exposure. | instrument version, response window, and interaction protocol. | a neutral readout or a demonstrated intervention. |
| **ReactivityStatusReceipt** | whether reactivity was untested, examined, bounded, or unresolved for a declared comparison. | design, comparator if any, outcome scope, and limitations. | proof of no reactivity or proof that assessment caused change. |
| **ObservationProcessModelReceipt** | declared model of visit intensity, monitoring, missingness, or selection. | estimand/use, assumptions, covariates, fit/sensitivity evidence, and deployment context. | identification, transport, or an empirical truth claim. |
| **ObservationFeatureUseReceipt** | an intended predictive/descriptive use of presence or intensity features. | prediction time, care-system scope, update rule, and revalidation trigger. | a causal effect, a stable individual trait, or a portable clinical signal. |
| **ObservationProcessAbstentionReceipt** | which observation distinction is absent or unresolved. | affected inference, missing evidence, and acquisition/review route. | interpolation, imputation, or a default claim of stability. |

The critical constructor boundaries are negative:

~~~
EncounterPresenceReceipt != RecordedMeasurementReceipt
MeasurementActReceipt != ReactivityStatusReceipt
ObservationFeatureUseReceipt != IdentificationReceipt
ObservationProcessModelReceipt != TransportReceipt
ObservationProcessAbstentionReceipt != a zero-valued observation
~~~

## Ordered History Is Not A Resampled Table

Many analysis systems make an irregular record usable by interpolation,
aggregation, imputation, windowing, or grid alignment. Those transformations
can be valuable, but their result is a transformed representation, not the
original observation history.

~~~
raw encounter/selection history + transformation receipt
  -> regularized analysis representation

regularized analysis representation
  != raw observation-process receipt
~~~

The non-associativity here is concrete. Consider two ordered paths:

~~~
path A: assessment -> recorded response -> clinical contact -> later assessment
path B: clinical contact -> assessment -> recorded response -> later assessment
~~~

They may produce the same final count of assessments and the same vector of
recorded scores. They do not have the same interaction history. A model may
choose to abstract that difference away for a named task, but it must do so by
an explicit transformation with an applicability claim, not by treating the two
paths as identical evidence.

Likewise, the sequence

~~~
observe -> predict -> change monitoring -> observe
~~~

is not equivalent to one in which monitoring remained fixed. The prediction can
become part of the future observation process. This is a feedback-risk receipt,
not an assertion that every monitoring system produces harmful or useful
feedback.

## Synthetic Collision Matrix

The first implementation must be synthetic. It must never consume patient data,
infer an individual's state, or emit a clinical recommendation.

| Collision | Hold fixed | Vary | Required result |
| --- | --- | --- | --- |
| Presence versus value | Same interval and source system. | An encounter is present versus absent. | Presence cannot construct a symptom/value receipt. |
| Selection versus missingness | Same unrecorded item. | No planned opportunity versus planned-but-unanswered measurement. | The two routes retain distinct provenance. |
| Act versus response | Same instrument and numeric response. | Prompt/response timing and feedback channel. | A recorded value cannot erase the measurement-act identity. |
| Reactivity boundary | Same observed trajectory. | Reactivity evidence absent versus scoped. | Neither status becomes a treatment or causal-effect receipt. |
| Prediction versus explanation | Same visit-intensity feature. | Declared prediction task versus causal estimand. | Feature-use receipt cannot construct identification evidence. |
| Care-system shift | Same measured values and model parameters. | Monitoring policy or access/channel context. | A source-system feature cannot silently become transport evidence. |
| Resampling collision | Same grid-aligned values. | Original encounter and selection histories. | The regularized table retains a transformation receipt and cannot replace raw provenance. |
| Feedback collision | Same initial score. | Prediction does versus does not change later monitoring. | The later observation process has distinct ordered provenance. |

The collision matrix has a deliberately modest proof target: preserve the
receipt distinction through a selected source-to-IR path. It does not prove a
valid measurement model, identify a causal graph, quantify assessment
reactivity, or establish a clinical effect.

## Future Executable Boundary

Only after the source-fresh imported-native #901 acceptance path exists, a
future import-bearing fixture may introduce nominal records for the receipt
taxonomy and test the following conditions:

1. **EncounterPresenceReceipt** is rejected where a
   **RecordedMeasurementReceipt** is required.
2. **MeasurementActReceipt** is rejected where a **ReactivityStatusReceipt** is
   required.
3. **ObservationFeatureUseReceipt** is rejected where an
   **IdentificationReceipt** or **TransportReceipt** is required.
4. A **RegularizedObservationRepresentation** cannot substitute for an
   **ObservationProcessReceipt** without a declared transformation receipt.
5. **ObservationProcessAbstentionReceipt** remains a normal, inspectable result
   and cannot be coerced into a zero observation or a decision candidate.

The positive fixture should use distinct synthetic constants and execute with a
source-fresh compiler (fallback=0) only to show that selected distinctions
survive the bounded lowering/runtime path. The negative fixtures must reject
category substitutions at compile time. None of these checks licenses a
clinical workflow or requires a new parser feature.

## Falsifiers And Stop Conditions

This distinction earns a first-class representation only if a bounded fixture
shows a category collision that an existing **ObservationProcessReceipt** cannot
make inspectable. The proposal should stop or remain documentation-only if:

- the required information is fully and consistently carried by existing
  temporal, measurement, and protocol receipts;
- no synthetic collision can distinguish opportunity, presence, selection, and
  measurement act without duplicating their fields under different names;
- a prospective API would imply that care presence has a universal clinical
  interpretation;
- an implementation upgrades a reactivity status into evidence that assessment
  changed, improved, or harmed a person's state; or
- a model uses care-system behavior as a predictor without declaring the
  deployment context and revalidation boundary.

The language contribution is therefore not a claim to solve informative
observation statistically. It is to prevent an implementation from erasing the
assumptions under which a statistical method or prediction is being used.

## Semantic Lane Declaration

~~~
Semantic-Lane-ID: psychiatric-informative-observation-research-v0
Owner: codex-root-psychiatric-state-inference-20260721
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE
Intent-Preserved: model complex psychiatric and medical systems without promoting records, analogies, or model outputs into unearned empirical or clinical authority
Transformation: refine the research-only ObservationProcessReceipt into separately auditable opportunity, presence, selection, measurement-act, reactivity, model, feature-use, and abstention boundaries
Types-Changed: none; proposed future library vocabulary only
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a future library fixture can preserve distinct observation-process provenance categories through a bounded source-to-IR path
Claims-Forbidden: that absence from care indicates stability; that assessment is always neutral or always an intervention; that informative observation identifies a causal effect; that a care-system feature transports to a new setting; that this contract authorizes a clinical action
Assumptions: the study or deployment can state relevant opportunity, channel, selection, measurement-act, and use context; any reactivity claim has separately scoped empirical evidence
Write-Set: docs/research/psychiatric_informative_observation_contract_2026-07-21.md
Read-Set: docs/research/psychiatric_temporal_authority_receipt_matrix_2026-07-21.md; docs/research/psychiatric_counterfactual_authority_abstention_contract_2026-07-21.md; docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md
Positive-Witness: future synthetic import-bearing receipt fixture after #901 source-fresh imported-native acceptance
Negative-Witness: future compile-fail substitutions from presence/act/feature-use/regularized representations into measurements, reactivity, identification, transport, or decision categories
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; source-fresh imported fixture only after #901, with fallback=0
Integration-Target: research documentation branch; future library-first package after owner acceptance
Authoritative-Only-If: a source-fresh imported-native witness proves the claimed receipt distinction without fallback, while all empirical and clinical claims remain independently governed
~~~

## Integration Receipt

~~~
Semantic-Outcome: informative observation is represented as a structured and ordered research provenance boundary rather than a missingness flag or neutral state readout
Concept-Status-Before: ObservationProcessReceipt names the general observation mechanism but does not distinguish opportunity, presence, selection, assessment act, reactivity status, and task-specific feature use
Concept-Status-After: those proposed distinctions have an explicit no-promotion matrix, collision design, and source-fresh future fixture boundary
Distinctions-Added: opportunity != presence != selection != measurement act != recorded response; reactivity status != intervention effect; prediction use != causal identification; regularization != original observation history
Distinctions-Preserved: ordered path != commutative endpoint; model representation != empirical claim; compilation != clinical authority
Distinctions-Erased: none
Evidence-Run: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh; git diff --check
Fallback-Path: none; documentation-only research contract
Legacy-Kept: the umbrella ObservationProcessReceipt remains the prior temporal-matrix vocabulary
Conflicting-Lanes: none observed at claim time; imported-native #901 work remains owned by the compiler/PBPK lane
Next-Semantic-Interface: consider a library-first synthetic receipt fixture only after #901 source-fresh imported-native acceptance
~~~
