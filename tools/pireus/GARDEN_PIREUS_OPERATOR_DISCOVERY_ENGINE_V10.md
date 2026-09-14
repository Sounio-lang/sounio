# Garden: Pireus Operator Discovery Engine V10

Status: GARDEN
Concept-ID: SOUNIO-PIREUS-OPERATOR-DISCOVERY-ENGINE
Semantic lane: pireus-operator-genome-v3-20260829
Founder direction: Pireus must generate operator novelty, not merely lower a
catalogue supplied from outside.

## The Seed

Pireus should become a discovery instrument for operators.

The strong form of the idea is not:

> enumerate syntax, score it, and label the unfamiliar output novel.

It is:

> synthesize a typed operator while constructing the evidence that separates
> it from every operator class in a declared frozen universe.

The candidate and its separation certificate are born together. Generation is
therefore constrained proof search, not post-hoc naming.

## Parent Evidence

V9 is the exact parent of this Garden:

- frozen Sounio semantics:
  `8fad13c09d2b17ea1adcce0a6b89612964e80ddb4a7a576916c2c1da30286df6`;
- seven generated genomes;
- six typed executable genomes and one typed rejection;
- four basis-distinct child maps;
- one residual twist that collapses to the parent under complete basis
  equality;
- an identity-only equivalence contract;
- no algorithmic, material, historical, priority, or claim-ready novelty.

V10 must not reinterpret those facts. In particular, V10 cannot promote the
four basis-distinct maps into historical novelty.

The complete V9 transcript is a semantic input, not merely a convenient table.
V10 admits it only after Sounio computes its frozen SHA-256 directly over every
byte. It then independently reconstructs the three bilinear atlas classes from
the V9 source definitions `oag_rotl4`, `oag_affine_address`, and
`oag_genome_coefficient`. Neither witness is allowed to stand in for the other.

## Novelty Is Indexed

There is no unqualified `Novel` value. The intended type is conceptually:

```text
Novel<UniverseHash, EquivalenceHash, PropertyContractHash, EvidenceLevel>
```

Every novelty certificate names:

1. the finite universe of known representatives it excludes;
2. the equivalence relation or action quotient used for comparison;
3. the semantic carrier and operator fragment;
4. the property contract the candidate satisfies;
5. a separating witness for every excluded class;
6. the evidence level actually reached.

Changing the universe, equivalence, carrier, or property contract creates a
different proposition. A certificate cannot survive that change by wording.

## The Five Novelty Levels

V10 keeps five levels distinct.

### N0: Syntax Distinction

The genome encoding differs. This is useful for lineage and mutation, but it
is not semantic novelty.

### N1: Extensional Distinction

The complete declared pure-tensor basis map differs under the identity
relation. For an arity-`r` multilinear map on `Z^16`, that map contains all
`16^r` evaluations `(e_i1, ..., e_ir)`, and every evaluation is a complete
16-component vector. Equivalently, the representation contains `16^(r+1)`
integer structure coefficients. Multilinearity makes those values determine
the full map. This is the level V9 can already witness for four child maps.

### N2: Quotient Separation

The candidate is not equivalent to any frozen atlas class under a declared
finite group action and normalization quotient. This is the first level called
`relative_operator_novelty`.

### N3: Law-Spectrum Separation

The candidate has a checked law or counterlaw spectrum not represented in the
frozen atlas. The spectrum can include associativity, commutativity,
alternativity, flexibility, Moufang fragments, bounded zero-divisor and rank
profiles unless separately closed by theorem, associator support, commutator
support, and declared domain-specific invariants.

N3 is stronger relative-atlas evidence about law behavior, not proof of
mathematical interest. It remains indexed by the atlas and checked model.

### N4: Material Distinction

At least one admitted lowering has a target-observed cost, dependency,
precision, stability, or throughput profile outside the frozen material atlas.
N4 requires C++ material parity and real canonical target receipts for Xeon,
Apple Silicon, DGX, and the dual AMD Alveo U250 surface.

### H: Historical Priority

Historical novelty and priority require prior-art search, attribution review,
and founder promotion. They are not computable from repository absence and
cannot be confirmed by an LLM receipt. Pireus may produce a review package; it
may not emit `historically_novel=true` autonomously.

## Novelty By Construction

The V10 generator receives a `DiscoveryIntent`:

```text
DiscoveryIntent {
  carrier
  arity
  grammar_hash
  required_laws
  forbidden_laws
  atlas_hash
  equivalence_hash
  target_envelope_hash
  search_budget
}
```

For every atlas class `A[k]`, synthesis also creates a separator obligation:

```text
exists basis_input x . candidate(x) != action(A[k], x)
```

where `action` ranges over the exact finite equivalence contract. A candidate
that lacks a separator for even one class is classified, archived, and used as
a counterexample to refine the search. It is not admitted as N2.

This turns novelty into constrained orbit-avoiding synthesis: construct an
operator that satisfies its positive laws while escaping every forbidden
equivalence class. It is not classical anti-unification or least-general
generalization.

## Discovery Loop

The first executable loop is bounded and deterministic:

```text
SEED
-> MUTATE_TYPED_GENOME
-> REJECT_ILL_TYPED
-> EVALUATE_COMPLETE_BASIS
-> CANONICALIZE_DECLARED_QUOTIENT
-> REFUTE_AGAINST_ATLAS
-> CHECK_LAW_SPECTRUM
-> EMIT_CERTIFICATE_OR_COUNTEREXAMPLE
-> FEED_COUNTEREXAMPLE_TO_NEXT_GENERATION
```

The feedback is semantic. It does not ask an LLM to invent the expected
result. A mutation survives because Sounio can replay its certificate.

## Genome Grammar

The first V10 fragment is finite and deterministically enumerable. Its first
constructor family changes an operand-independent integer coefficient program
at one tensor coordinate by one unit. Both input indices are nonzero, so the
frozen `e0` left/right unit equations are preserved by construction. The
coordinate and sign are derived from the candidate id, never from an expected
result. Later families may compose:

- affine address transforms over `F2^4` that materialize operand-independent
  integer structure tensors and therefore preserve linearity in every tracked
  input slot;
- frozen sign/cocycle cells;
- associator and commutator residual cells;
- operand-independent integer coefficient programs;
- the two defined ternary binary-tree parenthesizations;
- bounded sums, differences, compositions, and tensor contractions accepted by
  a slot-linearity typing rule that tracks which original argument feeds every
  use and rejects duplication or nonlinear reuse of a linear slot;
- explicit zero coefficients;
- target-neutral layout annotations that cannot change semantics.

The first fragment must not use a basis certificate for arbitrary nonlinear
maps over `Z^16`. A future finite value carrier may open full function equality
under a separate hash and contract.

## Canonicalization Strategy

Exact quotient search can be expensive. V10 uses a proof-preserving cascade:

1. cheap invariant partitioning;
2. law-spectrum partitioning;
3. canonical basis digest under the declared action;
4. exhaustive equivalence search only inside the surviving partition;
5. a concrete separating basis input for every rejected equivalence.

Cheap invariants may accelerate exclusion, but they do not replace the final
separator unless the invariant itself has a checked implication theorem in the
declared model.

## Exact Action Convention

The first quotient action is not an unspecified set of rewrites. V10 begins
with the exact group `C2_diag`: identity and the unsigned lane permutation that
swaps basis lanes 0 and 1, applied diagonally to the output and both bilinear
input slots. Thus every sign is `+1`; this is not advertised as a nontrivial
sign gauge or as the full basis-change group. For this involutive group the
input permutation and inverse-output permutation share the same numerical
table. A future non-involutive group must represent the inverse separately.
For an arity-`r` map `F : (Z^16)^r -> Z^16`:

```text
apply(q, F)(x1, ..., xr)
  = Pout(q)^-1 F(P1(q)x1, ..., Pr(q)xr)
```

Larger affine `F2^4` address groups and sign gauges are admitted only after they
materialize such matrices. The frozen action domain must carry a complete
finite group table and executable witnesses for identity, closure, inverses,
associativity, and the action laws. Without those witnesses, Pireus may report
separation under a finite action set, but it may not call the result a quotient
or an equivalence class.

## Interestingness Is Not Novelty

Pireus should search for founder-relevant differences, not random distance. A
separate heuristic `InterestVector` may rank admitted N2 candidates by:

- rarity of law spectrum;
- associator or commutator sparsity and structure;
- zero-divisor or rank behavior;
- compatibility with XOR-convolution lowering;
- numerical stability under declared precision;
- target feasibility and estimated information movement;
- relevance to a founder-declared scientific question.

These features are search heuristics, not evidence that an operator is
mathematically interesting. The ranking never changes the novelty certificate.
A low-ranked candidate can remain novel relative to the atlas, and a fast
candidate can remain equivalent to an existing class.

## Proof-Carrying Candidate

An admitted V10 candidate must carry:

```text
OperatorDiscoveryCertificate {
  parent_semantics_hash
  intent_hash
  genome_hash
  semantic_basis_hash
  atlas_hash
  equivalence_hash
  canonical_representative_hash
  class_separator_table
  law_spectrum
  counterexample_lineage
  evidence_level
}
```

The separator table is content, not a count. Every entry identifies the atlas
class, attempted action domain, and first deterministic witness.

## Counterexamples Are Productive Output

The engine records why a candidate failed:

- type refusal;
- basis collision;
- quotient collision and matched representative;
- required-law counterexample;
- forbidden-law witness absent;
- target envelope impossible;
- budget exhausted before exhaustive separation.

A failed candidate changes the next generation only through a deterministic
Sounio mutation policy frozen in the executable. Failure data may guide a later
Garden, but it cannot retrospectively alter the current expected result.

## Canonical Material Targets

The discovery engine is target-neutral at semantic authority and target-aware
at ranking and material parity.

Canonical material classes are:

1. Xeon CPU family;
2. Apple Silicon;
3. DGX;
4. dual AMD Alveo U250 FPGA.

No target can define operator semantics. A target may refute feasibility,
provide a material fingerprint, or distinguish lowerings after the Sounio
artifact is frozen.

## Acceptance Boundary For The First V10 Executable

The first executable is accepted only if it:

1. consumes the exact frozen V9 parent hash;
2. freezes a finite atlas and equivalence contract in Sounio;
3. generates candidates by enumerating a typed constructor family rather than
   selecting a hand-written result table;
4. emits at least one quotient collision as a negative control;
5. emits a full separator table for every N2-admitted candidate;
6. recomputes every basis map and separator in an independent Sounio matcher;
7. remains deterministic under replay;
8. refuses missing policy, timeout, error, Python, Rust, authority promotion by
   LLM, and semantic writes by parity languages before dispatch;
9. keeps N3, N4, historical novelty, priority, and claim readiness false unless
   their distinct gates are later discharged.

The Garden does not require the first executable to discover a historically
new algebra. It requires Pireus to possess the machinery that could generate
and honestly certify a bounded mathematical difference.

## Falsifiers

The V10 direction is demoted if any of these holds:

- the generator is only a disguised hard-coded candidate table;
- two admitted representatives collide under the declared quotient;
- a separator cannot be replayed from the frozen Sounio source;
- an invariant filter excludes a true equivalence without a checked theorem;
- changing an atlas hash leaves a novelty certificate valid;
- material ranking changes semantic admission;
- an external LLM result becomes expected-result authority;
- repository absence is promoted to historical novelty.

## First Concrete Experiment

Freeze an arity-indexed atlas containing exactly the three V9 bilinear classes
(genomes 0, 2, and 3). The two V9 ternary classes are outside this proposition,
not silently counted as separated. Expand each sparse V9 basis row into the
complete 4,096-coefficient bilinear tensor, enumerate a bounded typed
coefficient-program neighborhood of the frozen parent class, and require a
complete separator table under the exact `C2_diag` lane-swap group. Later
Gardens may add a multi-seed schedule without changing what this first search
proved. The differentiating result is one of:

The `lean_single` bootstrap directly re-seals the 1,270,431-byte V9 transcript
in V10 with Sounio's SHA-256 block compression. It also verifies the exact V9
source, freeze, parity-open, and semantic hashes; parses all 768 bilinear rows;
and independently reconstructs the three complete tensors from the parent
multiplication and frozen residual formulas. Any failure in either the direct
hash or the reconstruction rejects the atlas before search.

- `QUOTIENT_COLLISION`, with the exact matched class and action;
- `N2_RELATIVE_NOVELTY`, with complete separator evidence;
- `SEARCH_INCOMPLETE`, with no novelty promotion.

That is the first point where Pireus stops merely carrying operators and begins
to discover them.
