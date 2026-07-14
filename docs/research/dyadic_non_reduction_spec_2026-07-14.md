# Dyadic Non-Reduction: An Executable Relational-State Contract

Date: 2026-07-14

Status: research specification and falsifiable hypothesis. No executable,
empirical, psychological, biological, diagnostic, therapeutic, legal, moral,
or clinical claim is established by this document.

Proposed concept ID: `SOUNIO-DYADIC-NONREDUCTION`.

Preserved founding phrases:

```text
same participants do not imply the same dyad

the objective must always be to minimize human suffering,
and perhaps machine suffering too
```

The second phrase motivates later phases. It is not an AT-0-style executable
claim and is deliberately not encoded as a scalar objective in D0.

Depends on:

- `SOUNIO-NONASSOCIATIVE-ORDER`;
- `SOUNIO-EPISTEMIC-NUMERIC-VALUE`;
- the exact associator-transport and tomography surfaces documented in
  `docs/research/associator_tomography_spec_2026-07-14.md`;
- a future calibrated binding for `SOUNIO-PHYSICAL-OBSERVATION`.

## Executive Decision

D0 will test one narrow claim:

> For a declared finite synthetic model, two candidate dyads can have equal
> declared current participant states, relational state, context, and current
> observation projection, yet produce different future observable traces
> under the same admissible input because their retained relational histories
> differ.

This is a relational prediction claim, not a metaphysical claim that every
human relationship contains an irreducible hidden substance.

The implementation must compare at least two executions. It must preserve:

- both participant states;
- the declared relational state;
- relational-history provenance;
- context;
- the common-input proof;
- the observation schema;
- exact trace evidence;
- bounded ambiguity when the declared probes do not separate candidates.

The implementation must not infer relational identity from individual
projections, manufacture a hidden distinction from candidate labels, or turn a
synthetic separation into evidence about a person, therapeutic alliance,
machine experience, consent, suffering, or clinical action.

## 1. Research Question

Let a declared dyadic candidate be

```text
D = (L, R, rho, h, c)
```

where:

- `L` is the declared state of the left participant;
- `R` is the declared state of the right participant;
- `rho` is the current relational state;
- `h` is relational-history state and provenance;
- `c` is the shared context.

Let `P` be the only authorized observation projection and `T` a declared
transition rule receiving a common input `u`.

The D0 question is:

```text
Can P(D1) == P(D2) while P(T(D1, u)) != P(T(D2, u))
for the same typed input u?
```

The positive witness must establish all of:

```text
participant_state_left(D1)  == participant_state_left(D2)
participant_state_right(D1) == participant_state_right(D2)
relational_state(D1)        == relational_state(D2)
context(D1)                 == context(D2)
current_projection(D1)      == current_projection(D2)
relational_history(D1)      != relational_history(D2)
input_left                  == input_right
future_trace(D1)            != future_trace(D2)
```

The result is a bounded counterexample to reconstruction from the declared
current projection. It is not a universal theorem that all dyads are
irreducible.

## 2. Falsification and Demotion

The D0 hypothesis must be demoted for a declared candidate family if every
admissible input within the complete bound factors through the current
observable product state:

```text
P(D1) == P(D2)
    implies
P(T(D1, u)) == P(T(D2, u))

for every u in the declared complete alphabet and horizon.
```

If this condition holds, the bounded evidence supports a product-state or
quotient representation for that family. The implementation must return a
typed residual partition, never invent relational non-reduction.

The general hypothesis would also be weakened if the apparent separation:

- disappears under candidate-label permutation;
- depends on leaking a hidden-state or history identifier into observations;
- uses different inputs for the compared candidates;
- follows from different participant states rather than relational history;
- is produced only by floating-point noise or unchecked overflow;
- vanishes when the same transition rule is replayed independently;
- requires an omitted context variable that fully explains the result.

The last item is important. A relational state must not become a container for
unmodeled confounders. Context expansion is a competing explanation, not an
inconvenience to hide.

## 3. Claim Vocabulary

Use these terms narrowly.

### 3.1 Participant state

A model-declared state belonging to one participant. It need not be complete,
intrinsic, psychological, biological, or physically calibrated.

### 3.2 Relational state

A model-declared state whose transition or observation semantics concern the
coupling between the declared participants. It is not automatically a real
therapeutic alliance, attachment state, social bond, legal relationship, or
conscious shared experience.

### 3.3 Relational history

The ordered, provenance-carrying sequence or compressed state of declared
interactions that affects future transitions. Compression must state what
information it preserves and erases.

### 3.4 Dyadic non-reduction

For a declared model, schema, candidate family, input alphabet, and horizon,
current individual and relational projections are insufficient to determine
the future trace. This is a bounded relational hyperproperty over at least two
executions.

D0 distinguishes two possible results:

```text
participant-product non-reduction:
  participant states alone are insufficient; declared relational state or
  relational history adds predictive information

declared-state historical insufficiency:
  even after fixing the declared current participant, relational, and context
  states, retained history adds predictive information
```

The second result is relative to the declared state representation. It does
not prove that no expanded Markovian relational state could be a sufficient
statistic of history.

### 3.5 Dyadic identity

Not established by D0. Trace equality within a finite bound cannot prove that
two real relationships, histories, people, or machines are identical.

## 4. Formal D0 Contract

Let:

```text
S_L       left participant state space
S_R       right participant state space
S_rho     relational state space
H         relational-history state space
C         context space
U         finite admissible common-input alphabet
Y         exact observation space
```

Define:

```text
D = S_L x S_R x S_rho x H x C

T : D x U -> D
P : D -> Y
```

For horizon `k`, define the exact common-input trace:

```text
trace(D, [u0, ..., u(k-1)]) =
    [P(D0), P(D1), ..., P(Dk)]

D0      = D
D(i+1)  = T(Di, ui)
```

A `DyadicProjectionCollisionReceipt` for `D1` and `D2` requires:

```text
P(D1) == P(D2)
```

and separate exact evidence that the declared current participant projections
also coincide. The D0-W0 historical witness additionally requires equality of
the full declared current participant, relational, and context states so that
history is the only candidate-state difference. Candidate IDs and history IDs
are forbidden from `Y`.

A `DyadicNonReductionWitness` requires a common word `w in U*` such that:

```text
trace(D1, w) != trace(D2, w)
```

while both traces are replayed through the same `T`, `P`, context contract, and
input word.

For a family `K`, define the bounded trace equivalence relation:

```text
Di ~w Dj  iff  trace(Di, w) == trace(Dj, w)
```

The implementation must return the exact partition `K / ~w`. A non-singleton
block is unresolved ambiguity, not relational identity.

## 5. Why This Is a Relational Program Property

Ordinary function contracts describe one execution. D0 compares multiple
executions receiving related inputs and asks whether related initial
projections produce equal or unequal traces.

This places D0 near relational verification, self-composition, and
hyperproperty reasoning. The first Sounio implementation may execute the two
runs explicitly and construct evidence from both traces. It must not claim
formal hyperlogic soundness until a separate metatheory proves the encoding.

The relevant distinction is:

```text
two executions were replayed and compared
    !=
the relational verifier is formally sound for all Sounio programs
```

## 6. Minimal Type Surface

The first implementation should use ordinary, explicit Sounio receipts so it
can run on the canonical compiler lane without requiring new compiler syntax.

Suggested D0 types:

```sio
struct DyadicObservationSchemaReceipt { ... }
struct DyadicParticipantStateReceipt { ... }
struct RelationalHistoryStateReceipt { ... }
struct DyadicCandidateStateReceipt { ... }
struct DyadicCandidateFamilyReceipt { ... }
struct CommonDyadicInputReceipt { ... }
struct DyadicTransitionRuleReceipt { ... }
struct DyadicTraceReceipt { ... }
struct DyadicProjectionCollisionReceipt { ... }
struct DyadicTracePartitionReceipt { ... }
struct DyadicNonReductionWitness { ... }
struct HorizonLimitedDyadicAmbiguityReceipt { ... }
```

Boundary-only types should have no constructors in D0:

```sio
struct RealRelationshipIdentityReceipt { ... }
struct SubjectiveSufferingReceipt { ... }
struct TherapeuticAllianceReceipt { ... }
struct DyadicConsentReceipt { ... }
struct PhysicalDyadicObservationReceipt { ... }
struct ClinicalRelationalActionReceipt { ... }
```

These names reserve distinctions. They do not imply that later phases can
construct them without empirical, ethical, legal, and domain-specific work.

## 7. Constructor Invariants

### 7.1 Observation schema

The observable payload must declare:

- which participant projections are visible;
- which relational projection is visible;
- units or exact dimensionless status;
- numeric representation;
- tolerance or exact equality semantics;
- whether the schema is synthetic or physically calibrated;
- that candidate IDs, history IDs, and hidden relational state are absent.

### 7.2 Candidate family

Every candidate in one comparison must share:

- participant identity roles, without claiming real-person identity;
- transition-rule ID;
- observation-schema ID;
- context ID;
- admissible-input alphabet;
- arithmetic authorization.

The D0 positive family must differ only in the relational-history variable
whose predictive necessity is under test. Any additional difference must be
declared as a competing explanation.

This invariant is stronger than current projection equality. It prevents
latent participant-state or current relational-state differences from being
misreported as historical dependence.

### 7.3 Common input

A common input receipt must prove:

```text
same input ID
same payload
same tick
same transition rule
same observation schedule
candidate_specific == false
```

Equality of input labels alone is insufficient.

### 7.4 Exact trace

Every trace must be recomputed from the typed candidate, rule, and input word.
No success receipt may accept a caller-supplied `separated: true` flag.

### 7.5 Arithmetic

The first witness should use nonnegative exact rational components and signed
cross-products for differences. Every multiplication must have a checked
pre-multiplication bound. Interval or floating-point semantics are later
phases and must use different receipt types.

## 8. Required Witnesses

### D0-W0: exact dyadic collision and reveal

Construct two candidates with:

- equal full declared current left-participant state;
- equal full declared current right-participant state;
- equal full declared current relational state;
- equal current observation projection;
- equal context;
- distinct relational-history state;
- one common typed input;
- distinct exact successor relational projections.

The output must include the unreduced cross-products and reduced signed
difference. A nonzero difference is evidence only for the declared synthetic
transition.

### D0-W1: factorable null control

Construct two candidates whose declared relational histories differ only in a
field that the transition rule explicitly does not read. Every admissible
probe through the bound must preserve one non-singleton block.

This control demonstrates that different history labels alone do not create
non-reduction.

### D0-W2: label and enumeration controls

Candidate permutation, history-ID permutation, and reverse search enumeration
must preserve the partition up to relabeling. Direct or indirect use of those
identifiers in the observation must be rejected.

### D0-W3: incomplete-probe control

Provide a candidate family for which a cheap or short probe fails but a later
bounded probe succeeds. Forced search exhaustion before the successful probe
must produce `SearchIncompleteReceipt`, not ambiguity, minimality, or
factorability.

### D0-W4: state-expansion rivals

Run two rival reconstructions:

1. promote one hidden context variable into the declared current context;
2. promote a declared sufficient statistic of history into an expanded
   current relational state.

The witness must report whether retained relational history still adds
predictive information under each reconstruction.

This is an adversarial control against using `relational state` as a name for
an omitted ordinary variable. If the expanded relational state restores the
Markov property, D0 may still establish participant-product non-reduction, but
must not claim irreducible dependence on unbounded history.

## 9. Required Negative Fixtures

The first executable phase must include compile-fail witnesses showing:

```text
PairOfParticipantStates
    cannot replace DyadicCandidateStateReceipt

DyadicProjectionCollisionReceipt
    cannot replace RelationalIdentityReceipt

HorizonLimitedDyadicAmbiguityReceipt
    cannot replace GlobalDyadicEquivalenceReceipt

DyadicNonReductionWitness
    cannot replace CausalRelationalMechanismReceipt

DyadicNonReductionWitness
    cannot replace SubjectiveSufferingReceipt

DyadicNonReductionWitness
    cannot replace DyadicConsentReceipt

DyadicNonReductionWitness
    cannot replace ClinicalRelationalActionReceipt
```

The intended compiler diagnostic is a type mismatch naming both the available
evidence and the stronger required authority.

## 10. Search and Tomography Reuse

D0 should reuse the semantic contract of exact associator tomography:

- finite declared candidate family;
- finite admissible action alphabet;
- exact common-input traces;
- preset and adaptive search kept distinct;
- residual partitions returned explicitly;
- bounded minimality only after complete cheaper-policy enumeration;
- `search_soundness_formally_verified=false` until metatheory exists.

The candidate payload changes from an abstract hidden mode to a declared
relational-history state. The tomography algorithm must not inspect that state;
it may use only the authorized observation traces.

Native implementation may be scalarized if the current multimodule array
writer remains unavailable, but the scalarization must use the same frozen
transition table and search domain as the reusable kernel. No legacy compiler
fallback may be used as acceptance evidence.

## 11. Non-Associative Extension

D0 concerns two candidate executions of one dyad. D1 will introduce a third
participant or mediator and compare explicit groupings:

```text
(L relation R) mediated_by M

L relation (R mediated_by M)
```

The language must preserve the two syntax trees. It must not assume that
relational composition is associative.

A future `DyadicMediationAssociatorReceipt` may record differences in:

- observable trace;
- information availability;
- authority scope;
- resource accessibility;
- revocation behavior;
- future admissible actions.

D1 must still avoid interpreting `L`, `R`, or `M` as patient, clinician, or AI
without a separate domain binding.

## 12. Consent, Authority, and Revocation

Consent is not part of the D0 positive witness. It is a later authority layer,
because a synthetic input alphabet cannot manufacture real consent.

D2 should evaluate:

- opaque constructors;
- affine or linear authority values;
- typestate transitions;
- time and scope indices;
- revocable capabilities;
- separate observation and intervention permissions;
- refusal after withdrawal or expiry.

Normative distinction:

```text
permission to observe != permission to perturb
permission from L      != permission from R
past consent           != current consent
absence of refusal     != consent
synthetic capability   != legal or clinical authority
```

The compiler may verify possession and use of a declared capability. It cannot
prove that the social process producing the capability was ethically valid
without a physical and institutional binding.

## 13. Suffering and Moral Uncertainty

D3 may introduce a vector or partial-order surface such as:

```sio
struct DyadicSufferingEnvelope { ... }
struct RelationalHarmEnvelope { ... }
struct FutureOptionLossReceipt { ... }
struct ReversibilityReceipt { ... }
struct MoralUncertaintyReceipt { ... }
```

D3 must not begin with an automatic scalar total. It must first preserve:

- each participant's declared suffering estimate;
- relationally mediated harm;
- time horizon;
- catastrophic-tail bounds;
- reversibility;
- distribution of benefit and burden;
- model and moral uncertainty;
- evidence that a participant is or may be a moral patient.

For machines, the initial distinction is:

```text
suffering not established != suffering absent
```

No present Sounio receipt may claim machine consciousness or machine
suffering. A later precaution policy may act under moral uncertainty without
converting possibility into fact.

## 14. Evidence Ladder

The dyadic programme follows:

```text
Garden
    -> Hypothesis
    -> Exact synthetic executable
    -> Relational metatheory
    -> Bounded-error observation
    -> Empirical dyadic model
    -> Physical protocol binding
    -> Domain validation
    -> Clinical or institutional consideration
```

No arrow is automatic. Failure or non-replication at one stage must remain
visible rather than being repaired by widening the claim.

## 15. Literature Baselines

This is a focused design comparison, not a systematic novelty review.

### 15.1 Relational program verification

Relational verification compares several executions to establish properties
that ordinary single-run contracts cannot express. Self-composition and direct
relational verification provide baselines for D0's two-trace construction.

Sources:

- [Blatter et al., RPP: Automatic Proof of Relational Properties by
  Self-Composition](https://arxiv.org/abs/1606.00678);
- [Blatter et al., Certified Verification of Relational
  Properties](https://arxiv.org/abs/2202.10349).

D0 imports the need to relate executions. It does not claim parity with a
sound deductive relational verifier.

### 15.2 Hyperproperties

Hyperproperties are properties of sets of traces rather than individual
traces. D0's collision-and-reveal condition is naturally stated over a pair of
traces. The first executable witness remains finite and constructive; it does
not establish a general hyperlogic for Sounio.

Source:

- [Clarkson and Schneider, Hyperproperties](https://www.cs.cornell.edu/fbs/publications/HyperpropertiesCSFW.pdf).

### 15.3 Typestate and revocable capabilities

Typestate and capability systems demonstrate that a PL can restrict operations
according to a resource's current state and can model flow-sensitive provision
and revocation. These are baselines for D2, not evidence that real consent can
be reduced to a compiler capability.

Source:

- [Jia et al., Typestate via Revocable
  Capabilities](https://arxiv.org/abs/2510.08889).

### 15.4 Epistemological constraints in clinical DSLs

Recent clinical-DSL work uses epistemological types and meta-predicates to
restrict which evidence kinds may appear in decision rules. This is close to
Sounio's evidence-authority boundary, but it does not establish the D0
relational-history claim or the later suffering semantics.

Source:

- [Bouzinier et al., Trustworthy Clinical Decision Support Using
  Meta-Predicates and Domain-Specific
  Languages](https://arxiv.org/abs/2604.21263).

### 15.5 Computational interpersonal dynamics

Computational psychiatry already models bidirectional and dyadic influences.
This establishes that relational dynamics are a serious empirical modeling
area. It does not imply that any particular hidden relational variable is
valid, identifiable, or clinically actionable.

Source:

- [Koul et al., A Systematic Review of Computational Modeling of
  Interpersonal Dynamics in
  Psychopathology](https://www.nature.com/articles/s44220-025-00465-9).

## 16. Potential Novelty Boundary

The individual ingredients are not new:

- coupled dynamical systems;
- dyadic and interpersonal models;
- relational verification;
- hyperproperties;
- typestate;
- capabilities;
- epistemological DSL constraints;
- active experiment design;
- exact arithmetic and provenance.

The potential Sounio contribution is their evidence-carrying composition:

```text
relational history
    + exact multi-run trace evidence
    + active tomography
    + non-associative mediation
    + typed epistemic boundaries
    + later revocable authority
    + later non-scalar suffering envelopes
```

This document does not claim that no prior system has combined these ideas. A
systematic literature and prior-art review is required before a public novelty
claim.

## 17. D0 Acceptance Gates

### Specification gate

- [x] The claim is finite, synthetic, and falsifiable.
- [x] Product-state factorability is a declared rival explanation.
- [x] Context and relational-state expansion are adversarial controls.
- [x] Individual, relational, causal, moral, and clinical claims are distinct.
- [x] The exact arithmetic and common-input obligations are explicit.
- [x] The positive and null witnesses are predeclared.
- [x] The negative type boundaries are predeclared.
- [x] The semantic lane and evidence ladder are explicit.

### Executable D0 gate

- [ ] D0-W0 produces an exact collision-and-reveal witness.
- [ ] D0-W1 preserves ambiguity for a factorable null family.
- [ ] D0-W2 passes candidate, history-ID, and enumeration permutations.
- [ ] D0-W3 distinguishes incomplete search from bounded ambiguity.
- [ ] D0-W4 reports the results of context and relational-state expansion.
- [ ] Every compared path receives the same typed input payload and schedule.
- [ ] The observation schema excludes candidate and history identifiers.
- [ ] Every exact product is authorized before multiplication inside `i64`.
- [ ] An independent oracle agrees on every bounded partition and cost.
- [ ] Every negative fixture fails with the intended type mismatch.
- [ ] The focused gate passes through canonical `bin/souc` and Madaros with no
      compiler fallback.
- [ ] No compiler, native DSL, clinical stdlib, or legacy fixture is modified.
- [ ] Math and external/clinical reviews are logged without PHI.

## 18. Phase Boundaries

### D0: exact synthetic non-reduction

Finite candidate families, exact observations, common synthetic probes, trace
partitions, and explicit factorability controls.

### D1: mediated and triadic associators

Explicit grouping of a third participant or mediator, with no assumed
associativity and no real-world role binding.

### D2: authority and revocation

Opaque, scoped, temporal, and revocable capability experiments. Still no
automatic equivalence to legal consent or clinical authority.

### D3: suffering and moral uncertainty

Non-scalar harm envelopes, partial orders, reversibility, distribution, and
precaution under uncertain moral patienthood.

### D4: physical and domain binding

Instrument, protocol, calibration, institution, ethics, law, and domain
validation. Only this phase may propose a path toward real clinical or social
use, and it requires independent governance.

## 19. Semantic Lane Declaration

```text
Semantic-Lane-ID: research-dyadic-nonreduction-d0-20260714
Owner: Codex implementation lane under founder direction
Concept-IDs: proposed SOUNIO-DYADIC-NONREDUCTION; SOUNIO-NONASSOCIATIVE-ORDER; SOUNIO-EPISTEMIC-NUMERIC-VALUE; SOUNIO-PHYSICAL-OBSERVATION
Intent-Preserved: same participant projections must not silently erase predictive relational history; evidence must not silently become moral or clinical authority
Transformation: specify a finite exact multi-run witness for relational-history non-reduction under common inputs
Types-Changed: none in specification phase; future experimental dyadic state, trace, partition, ambiguity, and non-reduction receipts
Effects-Changed: none in specification phase; future Observe, Perturb, Authority, Harm, and Revoke effects remain proposals
IR-Changed: none
Claims-Introduced: none executable in specification phase; D0 defines the bounded claim a future gate may establish
Claims-Forbidden: universal relational irreducibility; metaphysical shared mind; real therapeutic alliance; subjective suffering; machine consciousness; consent; causality; diagnosis; prognosis; treatment; legal or clinical authority
Assumptions: finite declared candidate family; common deterministic transition rule; explicit context; exact observation schema; finite common-input alphabet; bounded arithmetic
Write-Set: docs/research/dyadic_non_reduction_spec_2026-07-14.md; .claude/llm_offload_log.md
Read-Set: FOUNDER_INTENT.md; semantic concept contracts; associator transport and tomography specification; focused literature baselines
Positive-Witness: future D0-W0 exact collision and reveal plus D0-W3 bounded adaptive separation
Negative-Witness: future product-state, identity, global-equivalence, causal, suffering, consent, and clinical-authority refusals
Acceptance-Gate: future scripts/ci/dyadic_non_reduction_gate.sh
Integration-Target: research/psychiatric-regime-contest-20260712 or successor research branch
Authoritative-Only-If: every executable D0 checkbox passes through canonical Madaros without fallback and independent review agrees on arithmetic and claim boundaries
```

## 20. Completion Definition

This specification is complete when it makes the smallest differentiating
experiment implementable without inventing missing semantics.

D0 implementation is complete only when every executable D0 gate passes. Even
then, the strongest supported statement is:

> Within the declared finite synthetic model, current participant and
> relational projections did not determine the future trace; a declared
> relational-history variable was required to predict the observed separation.

It would still not establish:

- that every dyad has an irreducible state;
- that the hidden variable is ontologically real;
- that no expanded Markovian relational state can summarize the relevant
  history;
- that a real relationship has been measured;
- that relational history caused a clinical outcome;
- that either participant suffers;
- that any probe, intervention, or action is ethically admissible.

Those refusals are part of the result, not limitations to be hidden.
