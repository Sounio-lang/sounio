<!-- docs:meta
topic_id: repo.docs.research.psychiatric-nonassociativity-representation-contract-2026-07-21
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.psychiatric-nonassociativity-representation-contract-2026-07-21
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Order Alone Is Not The Associator

Status: historical, source-backed research boundary and falsifiable roadmap

Date: 2026-07-21

Claim boundary: this document specifies when a future psychiatric or PK/PD
research model may need explicit parenthesization sensitivity. It does not
claim that a brain, receptor system, patient, medication response, or clinical
decision process is octonionic or nonassociative. It does not fit or validate a
model, estimate a treatment effect, or recommend treatment.

## The Distinction That Keeps The Idea Honest

The founder's intuition is right to protect history, grouping, and order from
being flattened. But three different properties are often merged under the
single phrase "path dependent":

```text
same event multiset != same event order
same event order != same parenthesization
same history-sensitive state model != nonassociative carrier
nonzero algebraic associator != biological mechanism
```

An order-insensitive model treats events as a commutative bag. An ordered
state-transition model can retain the full sequence and still be associative:
if each event is represented by a state transformation, ordinary composition
of those transformations is associative even when it is noncommutative. In
that case, `A then B` can differ from `B then A`, while `(A then B) then C` and
`A then (B then C)` are only two descriptions of the same sequential
composition.

Genuine parenthesization sensitivity demands something stronger. The model
must name an interaction or aggregation operation for which grouping has an
operational meaning and for which the two groupings have different declared
outputs. A time window, a co-administration episode, a state-estimation reset,
or a coarse-graining map may be such a boundary. Mere arbitrary brackets in a
program are not.

Sounio's contribution is therefore not to force nonassociativity into every
history-sensitive system. It is to make the alternative representations
explicit, testable, and impossible to substitute silently:

```text
unordered summary
!= ordered associative transition model
!= parenthesization-sensitive aggregate model
```

## What The Literature Licenses

### Hysteresis Supports History Dependence, Not An Associator

PK/PD hysteresis shows that the measured concentration-effect relation need
not be a single timeless curve. Distributional delay, feedback, tolerance,
metabolites, target kinetics, and changing target state can make the same
measured concentration correspond to different effects at different points in
time. That licenses an explicit history or state representation.

- Louizos et al. (2014), [understanding PK/PD hysteresis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4332569/), reviews multiple mechanisms producing a time-dependent concentration-effect relationship.
- de Witte et al. (2020), [PK/PD models with drug-target binding kinetics](https://pmc.ncbi.nlm.nih.gov/articles/PMC7050630/), distinguishes several mechanistic explanations from a generic effect-compartment representation.

Neither result says that the appropriate state-update carrier is
nonassociative. A model that retains an effect-site state, receptor-regulation
state, or delayed feedback state may account for the history while preserving
ordinary associative composition of state transformations. That is a valid
scientific outcome and must remain available in Sounio.

### Receptor Regulation Makes The State Material

Agonist exposure can change subsequent signaling through desensitization,
arrestin-related processes, internalization, and other regulatory mechanisms.
Those mechanisms motivate a state variable that survives an individual event;
they do not determine a unique algebraic carrier for aggregating events.

- Carman and Benovic (1998), [GPCR desensitization](https://pubmed.ncbi.nlm.nih.gov/8701085/), reviews agonist-dependent loss of signaling responsiveness.
- Grundmann et al. (2015), [G-protein, arrestin, desensitization, and internalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC5595354/), separates these processes in GPCR analysis.
- Kolb et al. (2020), [IUPHAR guidance on ligand bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC7612872/), emphasizes time, system, and pathway context in interpreting signaling bias.

The immediate Sounio implication is modest: a cumulative dose or a receptor
occupancy scalar must not erase the declared regulatory history. Whether that
history is adequately represented by an associative state transition or
requires parenthesization-sensitive aggregation remains an empirical modeling
contest.

### Psychiatric Time And Context Do Not License A Shortcut

Computational psychiatry needs models that retain state variation, temporal
dynamics, and context. Dynamic treatment-regime methods likewise make each
stage depend on the information available from prior history. Both fields
support ordered state representations, but neither establishes that real-world
treatment history is nonassociative.

- Hitchcock, Fried, and Frank (2022), [computational psychiatry needs time and context](https://pmc.ncbi.nlm.nih.gov/articles/PMC8822328/), argues for models capable of retaining temporal and contextual dynamics.
- Laber et al. (2010), [statistical inference in dynamic treatment regimes](https://arxiv.org/abs/1006.5831), defines a regime as stage-specific decision rules over evolving history and exposes nonregularity in their inference.

This matters because a sequence of decision rules is a composition of functions
over a history state. It may be difficult, nonlinear, and noncommutative in
event order without being nonassociative. A nonassociative representation earns
its cost only if it preserves a discriminating fact that the ordered
associative representation demonstrably loses.

## The Mathematical Test

Let `S` be a declared state space and let every event `e` have a state update
`T_e : S -> S`. For a concrete sequence `a, b, c`, the sequential history is:

```text
T_c o T_b o T_a
```

Function composition is associative. Thus, keeping an exact state and each
event update supports order-sensitive dynamics without needing an associator.
The sequence can still be highly nonlinear and can retain all prior exposure
information that the state representation carries.

To claim parenthesization sensitivity, a model must instead identify an
explicit binary carrier `diamond` and a bracket-to-operation map, for example:

```text
(a diamond b) diamond c
a diamond (b diamond c)
```

The two expressions are scientifically distinct only when their brackets map
to distinct declared interventions, time-scale boundaries, aggregation rules,
or mechanistic interactions. The relevant empirical model claim is then not
"the system is octonionic." It is the much narrower statement:

```text
within the declared carrier and experiment,
the associative reduction loses a predeclared discriminating observation
that the parenthesization-sensitive model retains.
```

An algebraic octonion associator remains a precise mathematical probe:

```text
[a,b,c] = (a*b)*c - a*(b*c)
```

Its nonzero value can record that the chosen representation distinguishes the
two grouped products. It cannot identify a receptor mechanism, establish a
psychiatric phenotype, or validate a treatment model without an independent
representation binding and discriminating experiment.

## The Representation Contest

Before a biological or psychiatric model is called nonassociative, it should
be compared against three explicit classes over the same predeclared evidence:

| Class | Carrier | It preserves | It must not claim |
| --- | --- | --- | --- |
| `BagModel` | commutative aggregate of events. | totals or order-free summaries. | exposure order, regulatory history, or grouping. |
| `OrderedTransitionModel` | exact state plus ordered event transformations. | event order, declared latent state, and sequential feedback. | parenthesization sensitivity merely because order matters. |
| `BracketSensitiveModel` | declared binary aggregate plus bracket-to-experiment map. | a predeclared grouping distinction not retained by the competing classes. | physical nonassociativity, mechanism identity, or clinical benefit without evidence. |

The contest must hold the event identities, observation protocol, target
feature, and evaluation horizon fixed. It must predeclare which experimental
fact gives a bracket its meaning. A useful design might vary a rapid combined
perturbation, an explicitly separated sequence, and a declared intermediate
state-reset or coarse-graining boundary. It must not invent two bracketings
after examining the same undifferentiated time series.

The decision rule is intentionally demanding:

```text
if BagModel loses but OrderedTransitionModel retains the target feature,
  preserve ordered associativity

if BracketSensitiveModel retains a predeclared feature that the ordered model
cannot retain under matched assumptions and validation,
  preserve the bracket-sensitive representation as a scoped hypothesis

otherwise,
  emit nonassociativity abstention rather than an algebraic story
```

## Proposed Receipt Taxonomy

These are future library-level names, not syntax, a mechanism ontology, or a
clinical workflow.

| Receipt | It may state | It must carry or reference | It must not silently become |
| --- | --- | --- | --- |
| `PerturbationEventReceipt` | event identity, magnitude/exposure fixture, time, and protocol provenance. | observation/process scope and uncertainty. | an effect or state update. |
| `OrderedHistoryReceipt` | an ordered sequence of event identities. | event order, clock, and declared gaps. | a parenthesization-sensitive history. |
| `StateTransitionCarrierReceipt` | chosen state space, update family, and state-retention policy. | all state variables retained and deliberate coarse-graining. | evidence that the chosen state is sufficient. |
| `AggregationBoundaryReceipt` | why a grouping boundary corresponds to a distinct intervention, time-scale, reset, or aggregation rule. | experiment/protocol mapping and the boundary's scope. | a decorative set of parentheses. |
| `BracketingDesignReceipt` | predeclared grouped conditions, target feature, and comparison plan. | event identities, boundary receipt, and falsification criterion. | empirical support before the contest runs. |
| `AssociatorProbeReceipt` | mathematical result of a declared nonassociative carrier. | carrier convention, operands, grouping, arithmetic path, and precision status. | biological mechanism or empirical model fit. |
| `AssociativityContestReceipt` | matched comparison of bag, ordered, and bracket-sensitive representations. | model identities, held-fixed evidence, evaluation result, and limitations. | universal superiority of one carrier. |
| `ParenthesizationSensitivityReceipt` | bounded evidence that a declared grouping predicts or preserves a predeclared discriminating fact beyond the stated ordered comparator. | design scope, effect/model assumptions, uncertainty, and counterevidence. | a claim that the system itself is octonionic. |
| `NonassociativityAbstentionReceipt` | why a parenthesization-sensitive conclusion is not supported. | missing boundary semantics, indistinguishable models, failed contest, or unavailable observation. | evidence that order is irrelevant. |

The critical constructor refusals are:

```text
OrderedHistoryReceipt != AggregationBoundaryReceipt
AggregationBoundaryReceipt != ParenthesizationSensitivityReceipt
AssociatorProbeReceipt != FunctionalPathStateReceipt
AssociativityContestReceipt != CausalEffectEstimate
ParenthesizationSensitivityReceipt != ClinicalValidationReceipt
```

## Future Import-Bearing Collision Suite

After #901 has a checker-level imported-native acceptance gate, Sounio can add
a synthetic fixture that proves only the receipt boundary. It should contain no
patient data, receptor claim, dose calculation, or empirical fit.

| Collision | Hold fixed | Vary | Required result |
| --- | --- | --- | --- |
| Order versus grouping | Same ordered event IDs. | Absent versus declared aggregation boundary. | Ordered history cannot construct a bracketing design. |
| Transition versus associator | Same event history and state carrier. | Associative state transition versus nonassociative probe. | State carrier cannot construct an associator result. |
| Probe versus interpretation | Same nonzero synthetic algebraic probe. | Functional/causal/clinical receipts absent. | Probe cannot construct functional state, effect, or authority. |
| Failed contest | Same grouped synthetic inputs. | Comparator retains the feature or bracket map is absent. | Emit `NonassociativityAbstentionReceipt`. |
| Scoped success | Same comparator set. | Declared bracket map and discriminating synthetic observation. | Construct only `ParenthesizationSensitivityReceipt`, never clinical authority. |

The positive control may use ordinary nominal records in an imported leaf and
an importing main. The paired compile-fail controls should pass a receipt from
one column where another is required. A native run, when the compiler path is
accepted, shows only that the selected types survive the source-to-IR path.

## Relation To Existing Sounio Concepts

This proposal refines rather than broadens `SOUNIO-NONASSOCIATIVE-ORDER`.
That concept already requires parenthesization preservation and explicitly
forbids treating a `NonAssoc` effect as physical ontology. The new psychiatric
boundary supplies a test for when a state-history model should request that
effect at all.

It also composes with `SOUNIO-ORDERED-PATH-PROVENANCE` and the existing
psychiatric functional-path receipts:

```text
OrderedHistoryReceipt + StateTransitionCarrierReceipt
  -> ordered model input

OrderedHistoryReceipt + AggregationBoundaryReceipt
  + BracketingDesignReceipt + AssociativityContestReceipt
  -> ParenthesizationSensitivityReceipt
    | NonassociativityAbstentionReceipt
```

This keeps the representation ladder visible. A history can be ordered without
being bracket-sensitive; a bracket-sensitive model can be mathematically
precise without being biologically identified; an identified mechanism would
still not be a causal treatment effect or clinical decision.

## Falsifiers And Demotions

This direction should be narrowed or rejected for a particular model if:

- no physical, protocol, or time-scale operation gives brackets distinct
  experimental meaning;
- an exact ordered state-transition model preserves every predeclared feature
  attributed to the nonassociative carrier;
- the bracket-sensitive model wins only after its grouping or target feature is
  chosen post hoc;
- a nonzero algebraic associator is presented as biological or clinical evidence
  without an independent empirical representation binding;
- the proposed types cannot distinguish ordered history, aggregation boundary,
  associator probe, and abstention in a synthetic compile-fail collision; or
- a model compresses history but cannot disclose which state, events, or
  aggregation boundaries were discarded.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: PSYCHIATRIC-NONASSOCIATIVITY-REPRESENTATION-RESEARCH-20260721
Owner: Codex
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER; SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-PHYSICAL-OBSERVATION
Intent-Preserved: order and grouping remain available when they carry scientific information, while nonassociativity is never promoted from metaphor or chronology alone
Transformation: source-backed distinction and contest between order-free, ordered-associative, and bracket-sensitive state representations; no language change
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a future synthetic library fixture may prove selected receipt substitutions across an import boundary are rejected
Claims-Forbidden: physical octonion ontology, psychiatric mechanism, treatment effect, clinical utility, or authority from a nonzero associator, a history label, or compilation
Assumptions: cited PK/PD, GPCR, computational-psychiatry, and dynamic-regime work supports history/state requirements, not a nonassociative biological carrier
Write-Set: docs/research/psychiatric_nonassociativity_representation_contract_2026-07-21.md
Read-Set: docs/internal/concepts/nonassociative-order.md; docs/internal/concepts/ordered-path-provenance.md; docs/research/psychiatric_state_inference_contract_2026-07-21.md; docs/research/psychiatric_temporal_authority_receipt_matrix_2026-07-21.md
Positive-Witness: future imported synthetic representation contest, after #901 checker-level acceptance
Negative-Witness: ordered-history, aggregation-boundary, associator-probe, causal, and clinical receipt substitutions refuse at compile time
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh; future focused import-bearing compile-fail plus native control
Integration-Target: internal research planning, then the #901-gated psychiatric collision suite
Authoritative-Only-If: no receipt in this contract establishes physical, psychiatric, causal, or clinical authority; any future native gate proves only the selected program boundary
```

## Integration Receipt

```text
Semantic-Outcome: history-sensitive psychiatric modeling now has an explicit decision rule separating order retention from genuine parenthesization sensitivity
Concept-Status-Before: Sounio preserved nonassociative parenthesization but the psychiatric contracts did not specify when state history justified requesting a nonassociative carrier
Concept-Status-After: a future research fixture can require a declared aggregation boundary and representation contest before a bracket-sensitive receipt is constructible
Distinctions-Added: event order != grouping; ordered state model != nonassociative carrier; associator probe != biological mechanism; representation contest != causal effect
Distinctions-Preserved: nonassociative order; functional pathway state; temporal origin; measurement comparability; research evidence != clinical authority
Distinctions-Erased: none
Evidence-Run: source review of cited PK/PD, GPCR, computational-psychiatry, and dynamic-treatment-regime literature; documentation consistency checks pending
Fallback-Path: an associative state model is an admissible result when it preserves the predeclared information; no algebraic fallback makes a biological or clinical claim
Legacy-Kept: existing nonassociative-order concept, rebracketing authority, direct psychiatric authority controls, and temporal receipt research remain unchanged
Conflicting-Lanes: #901 owns imported-native acceptance plumbing and global governed documentation outputs; this lane does not modify them
Next-Semantic-Interface: #901-gated imported synthetic contest between ordered and bracket-sensitive psychiatric receipts
```
