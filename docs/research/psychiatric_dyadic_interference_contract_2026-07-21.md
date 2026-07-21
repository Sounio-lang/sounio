# Dyadic Psychiatric State: Interdependence Is Not Interference

**Status:** research architecture and falsifiable representation contract.

**Not clinical guidance.** This document does not diagnose a person, infer a
relationship problem, select an intervention, recommend a family/couple/peer
action, or authorize linkage of anybody's records. It specifies distinctions
that a future Sounio research model must preserve when interaction, roles, or
shared history matter to a scientific question.

This contract composes [psychiatric state inference](psychiatric_state_inference_contract_2026-07-21.md), [temporal authority](psychiatric_temporal_authority_receipt_matrix_2026-07-21.md), [counterfactual authority](psychiatric_counterfactual_authority_abstention_contract_2026-07-21.md), [model adequacy](psychiatric_model_adequacy_falsification_contract_2026-07-21.md), and [nonassociativity](psychiatric_nonassociativity_representation_contract_2026-07-21.md).

```text
two individual records != one dyadic history
co-occurrence != interaction observation
actor-partner association != partner causal effect
declared interference scope != identified spillover effect
dyadic effect estimate != relationship-intervention authority
ordered dyadic history != nonassociative carrier
```

## 1. The Scientific Boundary

Social context, interpersonal beliefs, communicative exchange, and the way two
people adapt to one another can matter to an explanatory model. Current work in
computational psychiatry calls for social features to be represented rather than
erased by default, but the evidence base remains methodologically uneven.

- Rhoads, Gu, and Barnby (2024), [advancing computational psychiatry through a social lens](https://www.nature.com/articles/s44220-024-00343-w), argues that social factors are often neglected in computational models despite their relevance to psychiatric contexts.
- Zavlis et al. (2025), [systematic review of computational modeling of interpersonal dynamics in psychopathology](https://www.nature.com/articles/s44220-025-00465-9.pdf), finds promising but uneven work that still needs performance, transparency, and validity evidence.

This does **not** mean every psychiatric process is relational, that a dyad
explains an individual outcome, or that an observed interaction is a mechanism.
It means that a model whose question is relational needs a representation that
does not silently discard its roles and dependencies.

## 2. Three Levels That Must Not Collapse

### Paired Records

Two records may be paired by time, household, conversation, care relationship,
or a study-assigned dyad. That pairing is an observation/design fact. It does
not establish effect, comparability, legitimate linkage beyond scope, or social
authority over either member.

### Statistical Interdependence

Actor-Partner Interdependence Models (APIM) represent non-independent dyadic
data and distinguish actor from partner pathways. The model is useful precisely
because it preserves the pair rather than averaging it away. Yet Cook and Kenny
explicitly note that a path called "influence" can still be predictive rather
than causal.

- Cook and Kenny (2005), [the actor-partner interdependence model](https://journals.sagepub.com/doi/10.1080/01650250444000405), defines APIM as a longitudinal model of dyadic interdependence and warns against reading every actor/partner path as causal.

### Causal Interference

Interference is stronger: one unit's potential outcome may depend on another
unit's assignment or exposure. The outcome is then indexed by an allocation
vector or a bounded exposure mapping, not by an individual's treatment alone.
Direct, indirect, total, and overall effects are different estimands.

- Hudgens and Halloran (2008), [toward causal inference with interference](https://www.treatment-effects.com/Hudgens-Halloran-2008.pdf), defines these estimands for groups where outcomes can depend on others' assignments and shows why design/assumptions are needed for their estimation.

```text
PairedObservationReceipt
  -> DyadicHistoryReceipt

DyadicHistoryReceipt + DependenceModelReceipt
  -> ActorPartnerAssociationReceipt | DyadicModelDiscrepancyReceipt

DyadicHistoryReceipt + InterferenceScopeReceipt + SpilloverEstimandReceipt
  + DyadicIdentificationReceipt
  -> DyadicEffectEstimate | DyadicInterferenceAbstentionReceipt
```

An APIM association can be useful where causal interference is unidentified.
Conversely, an interference claim needs an estimand, exposure mapping, design
or assumptions, and a valid analysis path; it cannot borrow authority from a
flexible association model.

## 3. The State Representation

For a time-indexed dyad `D = (r_0, r_1)`, a research model can declare:

```text
H_D(t) = (
    role map,
    actor observation history,
    partner observation history,
    interaction-event history,
    observation-process history,
    membership and context scope
)
```

This is a bookkeeping boundary, not a claim that every term is measured
accurately or causally sufficient. A dyad can be distinguishable under a
declared protocol or deliberately indistinguishable. A program must not swap
members silently and preserve a directional interpretation by accident.

## 4. Receipt Vocabulary

These are prospective research receipts, not native syntax. They are intended
for an import-bearing synthetic library only after the #901 source-fresh
compiler path accepts the psychiatric runtime suite.

| Receipt | It may state | It must not silently become |
|---|---|---|
| `DyadMembershipScopeReceipt` | study-defined relation, membership window, linkage/consent boundary, and de-identification scope. | permission to link additional records or a causal relation. |
| `RoleMapReceipt` | distinguishable/indistinguishable role policy and directional labels. | evidence that a role has causal priority or social authority. |
| `PairedObservationReceipt` | observed values attached to role and time. | interaction event, shared latent state, or comparable construct. |
| `InteractionEventReceipt` | protocol-defined interaction/coding event. | subjective intention, mechanism, or interpersonal cause. |
| `DyadicHistoryReceipt` | ordered role, observation, and interaction histories. | causal exposure map or nonassociative carrier. |
| `DependenceModelReceipt` | APIM or another declared association model. | a partner causal effect. |
| `ActorPartnerAssociationReceipt` | scoped actor/partner association result. | spillover, mediation, or relation-level causal claim. |
| `InterferenceScopeReceipt` | units that can affect outcomes and a bounded group/exposure mapping. | proof the scope is true. |
| `SpilloverEstimandReceipt` | direct, indirect, total, or overall causal quantity. | estimate, identification proof, or recommendation. |
| `DyadicIdentificationReceipt` | design/assumptions for a named spillover estimand. | evidence that assumptions hold. |
| `DyadicEffectEstimate` | bounded estimate for the named dyadic/interference estimand. | authority to alter another person's relationship, environment, or care. |
| `DyadicInterferenceAbstentionReceipt` | unresolved membership, role, measurement, scope, support, or identification. | no effect, no relationship, or permission to ignore a gap. |
| external relationship-intervention authorization | role-bound authority outside the library. | a constructible result of statistical receipts or a Stage 3 nominal record. |

Every receipt must carry the relevant protocol, temporal, observation-process,
uncertainty, and scope references from the preceding psychiatric contracts.

## 5. Falsifiable Representation Contest

Dyadic representation earns its complexity only when it preserves a
predeclared feature that an individual-only model loses.

```text
IndividualTrajectoryModel
    one person's ordered observations; no partner or interaction input

PairedAssociationModel
    role-indexed paired observations with a declared dependence/APIM model

InterferenceCausalModel
    role-indexed histories plus exposure mapping, spillover estimand,
    identification/design receipt, and uncertainty procedure
```

1. Use individual trajectories when the research feature is preserved and the
   paired terms add no declared information.
2. Use a paired association model only when it better captures a declared
   dependence feature under a model-adequacy contest.
3. Use causal interference only when the question is a named spillover estimand
   and its design/assumptions are represented.
4. Otherwise emit `DyadicInterferenceAbstentionReceipt`, not a partner effect.

| Case | Hold fixed | Vary | Required result |
|---|---|---|---|
| Same actor path, different partner/event history | actor observations and endpoint | role-indexed partner/interactions | individual-only receipt cannot satisfy a dyadic-history API |
| Same paired values, role swap | numeric values and timestamps | `RoleMapReceipt` | directional association refuses silent member exchange unless policy declares symmetry |
| Same APIM fit, no interference design | association/model result | scope, estimand, identification | association cannot form `DyadicEffectEstimate` |
| Same spillover estimand, support failure | estimand label and histories | allocation/support evidence | `DyadicInterferenceAbstentionReceipt`, not extrapolated effect |
| Same dyadic estimate, absent authority | statistical receipts | external authorization | estimate cannot form relationship-intervention authority |

Controls use synthetic constants and labels, never private dyadic data. They
prove only that a selected API keeps categories apart.

## 6. Order, Direction, And Nonassociativity

Dyadic interaction is naturally order-sensitive: the same reports can have a
different interpretation when role-bearing events arrive in another order. But
ordinary state transitions are functions over `H_D`, and their composition is
associative.

```text
ordered role-taking != noncommutative carrier
directed partner pathway != nonzero associator
dyadic history != parenthesization-sensitive aggregation
```

A genuinely nonassociative carrier must pass the separate
ordered-versus-bracketed representation contest. Neither APIM nor a spillover
estimand supplies an `AggregationBoundaryReceipt` or
`ParenthesizationSensitivityReceipt`.

This contract is intentionally distinct from historical O-CSSM and
relational-annihilation materials. It makes no algebraic homology claim, no
zero-divisor interpretation of a relationship, and no claim that interaction is
a rupture mechanism.

## 7. Import-Bearing Collision Suite (Deferred Until #901)

After one source-fresh artifact compiles and executes imported psychiatric
D11/D12 witnesses with no fallback, add:

```text
run-pass/psychiatric_dyadic_history_receipt_witness.sio
    constructs an ordered DyadicHistoryReceipt from membership, roles, paired
    observations, interaction events, time origin, and observation process.

compile-fail/psychiatric_paired_observation_cannot_claim_interaction.sio
    expected InteractionEventReceipt
    found PairedObservationReceipt

compile-fail/psychiatric_actor_partner_association_cannot_claim_spillover.sio
    expected DyadicEffectEstimate
    found ActorPartnerAssociationReceipt

compile-fail/psychiatric_interference_scope_cannot_claim_identification.sio
    expected DyadicIdentificationReceipt
    found InterferenceScopeReceipt

capability-only/psychiatric_dyadic_effect_cannot_authorize_relationship_intervention.sio
    reserved for a separately justified opaque/capability design; it must not
    be implemented as a public nominal authorization record in Stage 3

compile-fail/psychiatric_dyadic_history_cannot_claim_nonassociativity.sio
    expected ParenthesizationSensitivityReceipt
    found DyadicHistoryReceipt
```

The positive is an API/protocol witness only. It cannot make linkage valid,
prove a measurement model, identify a spillover effect, or authorize action.
The `capability-only` entry is not an imported nominal fixture: a public
nominal relationship-intervention authorization record would be forgeable and
would therefore test only a selected category mismatch. It can become an
executable acceptance case only after a separate opaque/capability design
supplies an external authority token, threat model, and attacker fixtures.
Until then, the dyadic adapter has no relationship-intervention authorization
constructor.

## 8. Acceptance Standard

This boundary is satisfied only when membership, role policy, observation
process, and time scope are explicit; pairing cannot masquerade as interaction;
association cannot masquerade as partner causality; causal interference carries
an estimand, exposure mapping, identification/design, and uncertainty; and
linkage, consent, and relationship authority remain external. Any
nonassociativity claim separately passes its own representation contest.

Failure is a named abstention or model discrepancy, never a default statement
that a dyad is independent, causal, symmetric, or available for intervention.

## 9. Semantic-Lane Declaration

```text
Semantic-Lane-ID: psychiatric-dyadic-interference-research-v0
Owner: Codex psychiatric state-inference research lane
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-EPISTEMIC-NUMERIC-VALUE; SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: social context may be represented without turning paired observations, association models, algebraic metaphors, or compiler success into causal, clinical, or relational authority
Transformation: dyadic interdependence and causal-interference distinctions are prospective receipts, synthetic contests, and explicit abstention routes
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: future synthetic fixtures distinguish membership, role mapping, paired observation, interaction event, actor-partner association, interference scope, spillover estimand, identification, dyadic effect, and an external authorization boundary
Claims-Forbidden: inferred linkage; interaction from co-occurrence; causal partner effect from APIM association; identified spillover from an interference label; relationship intervention from an estimate; nonassociativity from directed history; psychiatric or clinical mechanism, recommendation, safety, or validation
Assumptions: cited sources motivate social/dyadic modeling and causal-interference boundaries; they do not establish a psychiatric mechanism, valid consent/linkage, causal effect, or future Sounio package
Write-Set: docs/research/psychiatric_dyadic_interference_contract_2026-07-21.md; docs/governance/topic-registry.v1.json; docs/governance/DOCS_ACCEPTANCE_REPORT.md
Read-Set: FOUNDER_INTENT.md; AGENTS.md; docs/internal/concepts/{science-research-boundary,epistemic-numeric-value,ordered-path-provenance,nonassociative-order}.md; existing psychiatric research contracts
Positive-Witness: future imported synthetic dyadic-history receipt construction
Negative-Witness: future imported pairing-to-interaction, association-to-spillover, scope-to-identification, effect-to-authority, and history-to-nonassociativity substitutions
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh; after #901, focused imported dyadic receipt suite on one source-fresh ELF with no fallback
Integration-Target: internal research planning, then #901-gated imported synthetic fixture
Authoritative-Only-If: sources, limitations, current Concept-ID contracts, and source-fresh evidence remain aligned
```

```text
Semantic-Outcome: observations, dyadic association, causal interference, relationship authority, and nonassociative representation are separate research categories
Concept-Status-Before: psychiatric contracts represented ordered context and abstention but had no dedicated dyadic association-versus-interference boundary
Concept-Status-After: membership, role, pairing, interaction coding, association, spillover estimand, identification, effect, and the external authorization boundary have distinct prospective roles
Distinctions-Added: paired records != dyadic history; association != spillover; scope != identification; dyadic effect != relationship authority; direction != nonassociativity
Distinctions-Preserved: research model != empirical result; causal estimand != authority; compiler success != clinical validation
Distinctions-Erased: none
Evidence-Run: literature review; git diff --check; bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Fallback-Path: no fallback path is evidence for a relational causal claim or relationship intervention authority
Legacy-Kept: existing dialogue, O-CSSM, counterfactual, deferral, acquisition, and authority-boundary surfaces remain unchanged
Conflicting-Lanes: generated governance metadata is owned by the active PBPK compiler lane; no shared generated artifact was edited here
Next-Semantic-Interface: imported synthetic dyadic receipt constructors only after #901 source-fresh D11/D12 compile-and-execute acceptance
```
