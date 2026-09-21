<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-reflexive-inquiry-d3-2026-07-15
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-reflexive-inquiry-d3-2026-07-15
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Carrying Reflexive Inquiry D3

Status: frozen bounded synthetic specification, 2026-07-15.

## Thesis

D3 tests a programming-language proposition:

> When acquiring evidence may change the observed process, the instrument, or
> only the observer's knowledge, a program should retain those as distinct
> effect loci, preserve inquiry order, require exact evidence before declaring
> non-commutation, and refuse to infer the locus from an order effect alone.

This is not a model of a patient or a claim that psychiatric measurement is
always reactive. It is an exact finite counterexample to an unsafe software
assumption: that every value called an observation may be freely reordered as
if acquiring it were a read-only operation.

Here, **proof-carrying** means that each private transition receipt retains
enough bounded data for its consumer to replay the frozen transition table,
mask update, burden delta, and provenance recurrence at runtime. D3 claims no
Lean theorem, compiler metatheorem, cryptographic authenticity, or
module-sealed construction. The public false claim flags are negative metadata,
not capabilities; clinical and mechanism authority remain separate types that
no D3 constructor returns. Because sealing is not assumed, every public pair,
footprint, provenance link, probe, and observation is revalidated before any
state or fingerprint arithmetic; private fields are defense in depth only.
The base-31 fingerprint is a bounded audit convenience and is explicitly not
collision-free identity or authentication evidence.

## Semantic lane declaration

```text
Semantic-Lane-ID: psychiatric-regimes-d3-proof-carrying-reflexive-inquiry
Owner: Codex implementation under founder direction
Concept-IDs: SOUNIO-REFLEXIVE-INQUIRY; SOUNIO-PROOF-CARRYING-INFERENCE; SOUNIO-DYADIC-NONREDUCTION; SOUNIO-RELATIONAL-ASSOCIATOR
Intent-Preserved: order, grouping, effect locus, provenance, ambiguity, and claim level cannot disappear silently
Transformation: D2 passive model contest becomes a layered inquiry-action contest with exact commutation controls
Types-Changed: new stdlib and ontology types only
Effects-Changed: none in compiler IR; effect footprints are library-level executable receipts
IR-Changed: none
Claims-Introduced: one bounded synthetic fixture distinguishes passive commutation, state-only commutation, exact non-commutation, locus ambiguity, and within-family locus identification
Claims-Forbidden: universal measurement reactivity; non-associativity; biological mechanism; physical observation; individual counterfactual; suffering; diagnosis; prognosis; treatment; clinical authority; historical priority
Assumptions: frozen three-hypothesis family; exact integer transitions; two matched synthetic schedules; declared layer audit; exact provenance; no stochastic noise
Write-Set: D3 kernel, ontology, witnesses, negative fixtures, oracle, gate, concept contract, registry, this specification, offload log
Read-Set: D0/D1/D2 surfaces; evidence custody; state-dependent inference; reset commutator; effectful grouping associator
Positive-Witness: P then Q versus Q then P removes the passive model but leaves relational and instrument loci; relational-layer audit selects the relational model within-family
Negative-Witness: footprints cannot become non-commutation; non-commutation cannot become non-associativity; ambiguity cannot become mechanism; state-only commutation cannot authorize full trace reorder
Acceptance-Gate: scripts/ci/proof_carrying_reflexive_inquiry_gate.sh
Integration-Target: research/psychiatric-regime-contest-20260712
Authoritative-Only-If: canonical Madaros checks reusable surfaces, native and ontology witnesses execute, independent exhaustive oracle agrees, negatives reject, and D2/D1 regressions remain green
```

The ontology portion of this acceptance boundary is category-level and
parallel. D3's executable kernel returns ordinary receipt structs, while the
ontology module and focused fixtures independently encode corresponding nominal
non-subsumptions. No kernel-produced D3 value is currently carried into IR as
an ontology-typed result, and the gate does not imply such transport.

## Literature compass

The lane sits at the intersection of established fields. Its evidence is their
typed composition, not historical priority over any one of them.

| Literature | What it already establishes | What D3 tests instead |
|---|---|---|
| Dynamic epistemic logic and epistemic actions | Information exchange can be represented as an action that transforms epistemic state; resource-sensitive formulations model epistemic actions algebraically. <https://arxiv.org/abs/math/0608166> | A bounded executable receipt carries both epistemic update and possible writes to non-epistemic layers, with claim boundaries checked by types. |
| Algebraic effects and handlers | Effect systems can prove contextual equivalences, including commutation for computations using non-interfering references. <https://doi.org/10.2168/LMCS-10(4:9)2014> | Footprint overlap is only a potential-interference receipt; exact traces decide state commutation and evidence-trace commutation separately. |
| Interactive POMDPs | Multi-agent state can include physical state and models of other agents; beliefs are updated from action and observation histories, with finite nesting used for computability. <https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/gmytrasiewicz05a.pdf> | D3 is deterministic and finite, keeps a relational-process layer without recursive probability, and returns proof-carrying ambiguity rather than an approximately optimal action. |
| Participatory sense-making | Dyadic interaction can have dynamics not reducible to methodologically individual descriptions. <https://doi.org/10.1007/s11097-007-9076-9> | D3 represents a relational locus as one rival in a contest; it does not promote the enactive theory to an empirical psychiatric mechanism. |
| Measurement reactivity | A randomized EMA-frequency experiment found changes in several reported subjective experiences but not smoking abstinence, showing both outcome specificity and the danger of treating assessment as uniformly passive. <https://pubmed.ncbi.nlm.nih.gov/26011583/> | D3 treats reactivity as a hypothesis, not a default, and includes a passive control that must remain viable unless the trace eliminates it. |
| Null or limited reactivity evidence | A 12-month prospective EMA study did not find an association between completed prompts and self-reported eating behavior, so reactivity cannot be universalized. <https://doi.org/10.1007/s40519-023-01556-1> | The passive hypothesis is first-class and exact; D3 is falsified if the implementation assumes every inquiry writes the target. |
| Performative prediction | Predictions used for decisions can change the distribution of later outcomes. <https://proceedings.mlr.press/v119/perdomo20a.html> | D3 concerns acquisition actions and separates relational-process change from instrument adaptation; it does not solve performative risk minimization. |
| Observation versus intervention | Causal inference distinguishes passive observation from intervention; predictive difference alone cannot cross that boundary. <https://ftp.cs.ucla.edu/pub/stat_ser/bareinboim-etal-ch27-acm-2021.pdf> | Every D3 action, observation, audit, and conclusion explicitly refuses physical-intervention and causal-mechanism authority. |

The narrow opening is therefore not a new theory of mind. It is a PL object
that can say all of the following at once:

```text
these operations may interfere
!= they were observed to be non-commuting
!= the changed layer has been identified
!= the identified declared model is physically causal
!= the inquiry is clinically authorized
```

## Frozen layered model

Every hypothesis begins from the same exact full state:

```text
relational process r = 1
instrument offset   i = 0
observable projection y = r + i = 1
```

The matched-pair constructor rejects every other root. D3 is therefore a
fixture-specific witness, not a state-parametric commutation theorem.

There are three hypotheses:

| mode | action writes | P | Q | R |
|---|---|---|---|---|
| passive | neither layer | identity | identity | identity |
| relational | `r` | `r := 2r` | `r := r+1` | `r := r+2` |
| instrument | `i` | `i := 2i+1` | `i := i+1` | `i := i+2` |

The observer always receives `y` after the action. The action also appends an
ordered provenance identifier to the epistemic trace. These integer equations
are a fixture, not a psychometric, pharmacological, or instrument model.

## Exact controls

### Passive control

For the passive hypothesis:

```text
P;Q observations = 1,1; final state = (1,0)
Q;P observations = 1,1; final state = (1,0)
```

Both process/instrument layers and emitted value sequences agree, so D3 may
issue a bounded `InquiryReorderAuthorizationReceipt`. The complete ordered
state does not become equal: action order and provenance remain retained, and
the receipt explicitly records that it claims no complete-state equivalence.

### Overlap-is-not-proof control

Under the relational hypothesis, Q and R both write `r`:

```text
Q;R observations = 2,4; final state = (4,0)
R;Q observations = 3,4; final state = (4,0)
```

The process/instrument-layer commutator is zero, even though footprints
overlap. The trace still differs, so full inquiry reordering remains forbidden.
This falsifies the shortcut `write overlap => exact non-commutation` and also
the shortcut `equal process layers => evidence reorder is harmless`.

## Order-effect collision

For the relational hypothesis:

```text
P;Q observations = 2,3; final state = (3,0)
Q;P observations = 2,4; final state = (4,0)
projected commutator = 3 - 4 = -1
```

For the instrument hypothesis:

```text
P;Q observations = 2,3; final state = (1,2)
Q;P observations = 2,4; final state = (1,3)
projected commutator = 3 - 4 = -1
```

The full hidden states differ, but every projected observation in the paired
schedule is identical. Therefore the admissible order trace `2,3 | 2,4`
updates the version space as:

```text
{passive, relational, instrument}
    -> {relational, instrument}
```

It is evidence for an order effect inside this family, not evidence for its
locus. D3 must emit `ReactiveLocusAmbiguityReceipt` at this point.

## Layer audit

After `P;Q`, a synthetic relational-layer audit predicts:

```text
passive=1, relational=3, instrument=1
```

An admissible value `3`, linked to the prior order-trace provenance, updates:

```text
{relational, instrument} -> {relational}
```

The result is `DeclaredLocusIdentificationReceipt(mode=relational)`. It says
only that the relational hypothesis is the sole survivor of this frozen
family. A symmetric instrument audit with value `2` would select the instrument
hypothesis.

The order trace costs four declared fixture units and the audit costs three.
The total seven units are not patient burden, harm, discomfort, or suffering.

## Non-commutativity is not non-associativity

D3 compares reversed order:

```text
P;Q != Q;P
```

That is non-commutativity on the bounded transition operators. It does not
compare parenthesizations of three operands:

```text
(P;Q);R ?= P;(Q;R)
```

and therefore does not establish non-associativity. D1 remains the executable
grouping-sensitive surface. D3 adds a negative type gate preventing an exact
non-commutation receipt from masquerading as a non-associativity receipt.

## Expanded-state rival

The full state `(relational, instrument, action history)` makes every frozen
transition deterministic. This does not erase the order effect; it locates it
in the composition of state-changing actions. The observable scalar `y` alone
is not Markov sufficient because relational and instrument modes collide on
the complete P/Q trace while ending in different layered states.

This rival demotes any claim that the residue is irreducible to state
augmentation. The stronger surviving claim is that software must retain the
augmentation and must not silently reorder or collapse it.

## Falsification and demotion

D3 fails if any of the following occurs:

- missing or unaudited order evidence removes a hypothesis;
- hypothesis identifiers influence predictions or partitions;
- passive P/Q fails to commute in both state and trace;
- footprint overlap alone issues an exact non-commutation receipt;
- Q/R overlap is treated as sufficient to forbid state commutation;
- equal process/instrument layers authorize reordering despite a different
  evidence trace;
- relational and instrument paths fail to produce the same projected P/Q
  order trace;
- the order trace identifies a locus before a layer audit;
- the audit uses disconnected provenance;
- within-family identification becomes truth, mechanism, counterfactual, or
  clinical action;
- non-commutativity becomes non-associativity;
- the independent exhaustive oracle disagrees;
- D2 or D1 regresses.

Even a green gate establishes no empirical measurement reactivity, real dyadic
state, causal effect, psychiatric interpretation, or clinical utility. Those
would require calibrated physical observation types, an ethically approved
design, real provenance, explicit missingness assumptions, and external
validation. It also establishes no runtime kernel-to-ontology result identity.

## Frontier revealed: endogenous observability

D3 treats missing and unaudited values conservatively, but assumes that their
occurrence carries no model-discriminating information. The literature shows
why that is the next assumption to relax:

- irregular clinical visits can depend on clinical history, making the
  observation process informative and potentially biasing longitudinal
  estimates when it is ignored;
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6919310/>
- in a study comparing EMA responses with continuous activity monitors,
  prompts received during higher activity were more likely to be answered
  late, after activity had fallen;
  <https://pubmed.ncbi.nlm.nih.gov/39830851/>
- a 2025 state-space simulation study found materially worse recovery under
  autoregressive time-dependent missingness and MNAR than under MCAR, MAR, or
  simpler time-dependent MAR;
  <https://pubmed.ncbi.nlm.nih.gov/40091737/>

This suggests D4: **proof-carrying endogenous observability**. A future kernel
would type the sequence

```text
scheduled probe
-> delivery opportunity
-> response opportunity
-> observed value | typed nonresponse | delayed response
```

and jointly contest target-state, response-process, and measurement-policy
hypotheses. Nonresponse could update only a model that explicitly predicts the
response process; it could never be silently coerced into a symptom value.
Likewise, an adaptive prompt policy would have to remain in provenance because
it changes which moments can enter the dataset. This is a literature-grounded
next lane, not a result established by D3.
