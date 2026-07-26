<!-- docs:meta
topic_id: repo.docs.research.dyadic-relational-associator-ontology-spec-2026-07-14
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.dyadic-relational-associator-ontology-spec-2026-07-14
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Dyadic Relational Associator D1 and Ontology Binding

Status: frozen bounded synthetic specification, 2026-07-14.

## Research question

Can Sounio represent a relational process whose result depends on explicit
grouping, while preventing a bounded mathematical witness from being promoted
to causal, subjective, or clinical knowledge?

The D1 answer is deliberately narrow: yes, for one frozen exact operation and
one finite witness. It is not evidence that real dyads obey this operation.

## Literature-grounded commitments

The ontology uses an n-ary relational artifact instead of assigning the
relation to either participant. This follows the modeling pattern in the W3C
n-ary relations note, where a relation with its own attributes or context is
reified: <https://www.w3.org/TR/swbp-n-aryRelations/>.

Time and provenance are separate axes. OWL-Time supplies temporal entities and
ordering relations, while PROV-O separates entities, activities, agents, and
influence or derivation chains:
<https://www.w3.org/TR/owl-time/> and <https://www.w3.org/TR/prov-o/>.
Observation is also not collapsed into process state; SOSA/SSN distinguishes an
observation, its procedure, feature of interest, and result:
<https://www.w3.org/TR/vocab-ssn-2023/>.

The process/participant distinction is compatible with the OBO Relations
Ontology pattern in which an occurrent has a participant:
<https://oborel.github.io/obo-relations/process-relations/>. This is a modeling
alignment, not a claim that the experimental Sounio classes are imported BFO
universals.

Clinical temporal-abstraction work has long separated parameter, event,
pattern, context, and goal ontologies, including contexts induced by other
abstractions: <https://pmc.ncbi.nlm.nih.gov/articles/PMC61392/>. That separation
supports making `MediatedHistory`, `GroupingStructure`, and
`ObservationReceipt` different classes.

Predictive-state research gives the essential rival hypothesis. Computational
mechanics treats causal states as minimal predictive representations
(<https://arxiv.org/abs/cond-mat/9907176/>), and predictive state
representations encode state through predictions sufficient for control
(<https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf>).
Therefore a history-sensitive witness must always be challenged by an expanded
sufficient state. D1 records that rival rather than claiming irreducible memory.

Operational work on non-Markovian processes motivates a falsifiable question:
whether future observations are conditionally independent of the past given
the declared present state. It does not imply that dyads are quantum systems:
<https://arxiv.org/abs/1801.09811/> and
<https://arxiv.org/abs/1811.03448/>.

Psychotherapy research supports studying bidirectional, context-sensitive
dynamics, but not the D1 equation. Reviews describe interpersonal synchrony as
dynamic and multilayered (<https://pmc.ncbi.nlm.nih.gov/articles/PMC4907088/>),
and recent dyadic-system work emphasizes bidirectional influence and attractor
or repeller behavior (<https://www.nature.com/articles/s44220-025-00465-9>).
These sources motivate the question only. The operation below remains a
synthetic construction.

## Frozen D1 operation

For nonnegative exact rationals, define asymmetric mediation:

```text
x odot y = (2*x + y) / 3
```

The left operand represents already mediated context and receives weight two;
the incoming event receives weight one. With ordered leaves
`a=3/10`, `b=3/5`, `c=9/10`:

```text
(a odot b) odot c = 17/30
a odot (b odot c) = 13/30
[a,b,c]_odot       =  2/15
```

The implementation retains unreduced numerators and denominators, checks
multiplication bounds before cross-products, and independently replays the
fixture with Python `Fraction`.

## Controls and rival

The associative null uses exact rational addition on the same ordered leaves:

```text
(a + b) + c = a + (b + c) = 9/5
```

The state-expansion rival promotes the grouping tree into the declared state.
The two expanded states are then distinct. A total two-entry transition table,
indexed by grouping code, exactly replays `17/30` and `13/30`. Under the
declared finite-state convention, this permits a Markov representation of the
fixture. That sentence is an inference from the total transition table, not a
separate compiler receipt. It does not make `odot` associative or establish a
sufficient state for a real system; it shows that the same-leaf description
had projected away grouping.

## Ontology competency questions

1. Is every `GroupingSensitiveProcess` a `RelationalProcess`? Yes, by subclass
   closure.
2. Is every `AssociativeControlProcess` a `RelationalProcess`? Yes.
3. Can `ParticipantState` substitute for `DyadicRelationalState`? No.
4. Can `AssociativeControlProcess` substitute for
   `GroupingSensitiveProcess`? No.
5. Can `BoundedAssociatorWitness` substitute for `CausalMechanismReceipt` or
   `ClinicalAuthorityReceipt`? No.
6. Does promoting `GroupingStructure` distinguish the two D1 states? Yes for
   this finite fixture.

The current ontology evidence is deliberately category-level and parallel.
The executable kernel uses ordinary receipt structs, while the ontology module
and focused fixtures independently encode the corresponding nominal
non-subsumptions. No kernel-produced value is carried into IR as an
ontology-typed result by D1; the ordered-path compiler interface currently
preserves checked parameter identity only. A future result-identity bridge must
have its own source-to-IR witness and cannot be inferred from these tests.

## Claim boundary

D1 establishes an executable nonassociative model and parallel
compiler-checked ontology boundaries. It does not establish runtime
kernel-to-ontology transport, empirical calibration, or a law of mental states.
Any future clinical binding requires measurement definitions, units,
prospective data, identifiable alternatives, uncertainty, consent, and an
independently governed decision boundary.
