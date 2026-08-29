# Garden: Pireus Quotient Novelty Forge v5

Status: `GARDEN`

Concept-ID: `SOUNIO-PIREUS-QUOTIENT-NOVELTY-FORGE`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

Founder direction preserved:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

Pireus Cubic Operator Forge v4 generated 48 executable operator children
outside the complete bilinear phase grammar in the declared coordinates. It
also stopped at the correct boundary: pairwise-distinct mutation tables are
not automatically 48 genuinely distinct operator directions after basis
relabeling, basis-sign gauge, and operand exchange.

v5 makes novelty a typed, replayable object instead of an adjective. A novelty
statement must name the population being compared, the equivalence being
quotiented, the frozen parent, the completeness witness, and the evidence
stage:

```text
Novelty<Population, Equivalence, Parent, Witness, Stage>
```

This notation is the semantic schema for v5. It does not claim that Sounio
already has dependent type syntax with this spelling.

The v5 question is fixed before an executable, automorphism count, class count,
representative, or witness exists:

> Given the frozen 48-child v4 population, can Sounio enumerate the complete
> finite linear/swap action universe, discover exactly which actions preserve
> the frozen parent modulo basis-sign gauge, quotient all 48 children under
> those parent-relative actions, and emit a canonical witness for every class
> membership without ranking or selecting a child?

This is the first exact novelty quotient for generated Pireus operators. It is
not the last. The type boundary is designed so later equivalence profiles can
add nonlinear permutations, isotopy, algebra isomorphism, program equivalence,
material realization, or scientific behavior without retroactively widening
the v5 result.

## Authority order

The only admissible order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The first Sounio executable must contain no expected parent stabilizer size,
gauge-class count, parent-relative class count, class partition,
representative child, witness matrix, witness gauge, class digest, atlas
digest, selected child, ranking, material result, or frozen result matcher.
Sounio produces all first results.

Lean 4, Koka, C++, Haskell, Xeon, Apple Silicon, DGX, U250, and external LLMs
must not run on v5 before a hash-frozen Sounio result exists. Python and Rust
are forbidden as generators, oracles, validators, freeze producers, or parity
legs. Node, Ruby, shell, awk, `bc`, or another disposable language may not be
substituted as a semantic oracle.

## Frozen parent population

The immutable semantic parent is Pireus Cubic Operator Forge v4:

```text
parent source
  2c295c48bcd2de0f43a42787dcc612f78c7d40d528641e4fec890858d881c974
parent semantics
  e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff
parent freeze
  1da425c1ff53273825a71b46850e0cd9e7d4cd5b77aa79eb65ef269aadd5a87b
parent transcript
  d27915015cabda1d11211968e0bde5655757599d8dc3313fbfc0506877e49694
parent parity-open receipt
  82cdb8875783d34a903b7b599aeeb9501d73eba9ed0a2426040bd708aaf2665a
```

The v4 population remains exactly:

```text
f_0(i,j) = sign_bit(cd_sigma(i,j,4)) XOR b_1128(i,j)
f_m(i,j) = f_0(i,j) XOR m(i,j)
mutation IDs = 0..47 in the frozen v4 order
```

Here `f_0` is the complete frozen parent sign-bit table, not only its bilinear
phase, and `f_m` is the complete sign-bit table of child `m`. v5 must obtain
the parent and all children by executing the canonical frozen Sounio modules.
It may not restate, regenerate in another language, edit, backfill, or reorder
the v4 population.

The inherited v4 facts remain parent facts:

- the population has 48 pairwise-distinct mutation truth tables;
- every mutation lies outside all 65536 bilinear phases in the declared
  coordinates;
- every child has immutable lineage and a complete 256-cell sign table;
- no child is selected;
- all 1920 target obligations remain unresolved.

v5 does not promote coordinate-wise distinction into quotient novelty without
executing the quotient below.

## Declared coordinate universe

Let `V=F2^4`, serialized as integers `0..15`. A packed matrix `M` ranges over
all 65536 four-by-four matrices over `F2`. Exactly the invertible matrices form
`GL(4,2)`. v5 inherits the already frozen v2 constants and must independently
recount the complete universe:

```text
matrix encodings scanned = 65536
invertible matrices       = 20160
operand-swap choices      = 2
candidate actions         = 40320
```

An action `A=(M,s)` on a sign-bit table `f:V x V -> F2` is

```text
A.f(i,j) = f(Mi,Mj)  when s=0
A.f(i,j) = f(Mj,Mi)  when s=1.
```

The matrix acts on both operands and, because it is linear over `F2`, preserves
the XOR product skeleton:

```text
M(i XOR j) = Mi XOR Mj.
```

Direct substitution gives

```text
(M,s)^-1 = (M^-1,s).
```

Operand exchange commutes with the diagonal matrix action, so the declared
action group has the direct-product law `GL(4,2) x C2`.

The action universe covers linear XOR-coordinate changes and optional operand
exchange. It does not cover nonlinear permutations, arbitrary real basis
changes, isotopy, anti-isotopy beyond the declared exchange, or a general
algebra-isomorphism search.

## Exact basis-sign gauge

A basis-sign gauge is a function `q:V->F2` with `q(0)=0`. Its coboundary is

```text
dq(i,j) = q(i) XOR q(j) XOR q(i XOR j).
```

The gauge domain has 15 bits. Its four-dimensional kernel consists of linear
characters, so its coboundary image has dimension 11.

v5 uses the exact tree normalizer already established by Operator Genesis v1
and v2, but it must expose a replayable witness rather than only a Boolean
membership result. For any table `h`, the normalizer returns:

```text
normalize(h) = (remainder, canonical_q)
```

with the canonical kernel choice

```text
canonical_q(0)   = 0
canonical_q(e_0) = 0
canonical_q(e_1) = 0
canonical_q(e_2) = 0
canonical_q(e_3) = 0.
```

The other 11 values are solved in the frozen tree order. For each non-basis
vector `x`, let `e` be its highest set basis bit and `p=x XOR e`. The pivot
cell is `(p,e)`, and `p<x`. With the four basis values fixed to zero,

```text
canonical_q(x) = h(p,e) XOR canonical_q(p).
```

The 11 pivot equations are therefore triangular with diagonal one. v5 must
not treat this as inherited prose alone. The executable must construct the 11
canonical gauge basis functions, verify that the 11-by-11 pivot map has full
rank, enumerate all 2048 canonical gauge words, and require a unique
zero-remainder round trip for each word.

For every normalization request, the executable must replay all 256 cells and
establish

```text
h = remainder XOR d(canonical_q).
```

Consequently

```text
h is an exact coboundary iff remainder is the zero table.
```

The triangular pivot certificate and complete 2048-word round trip make this a
complete decision procedure for the declared basis-sign gauge. The 256-cell
reconstruction is retained as implementation-integrity evidence, but it is not
misrepresented as an independent theorem: once `remainder = h XOR dq` is
defined, reconstructing `h` is algebraically tautological. This procedure is
not a numerical approximation, hash-only comparison, SAT answer, or external
theorem-prover result.

## Parent-relative action group

The relevant group is not all transformations that preserve the v4 cubic
grammar. That would make the grammar, rather than the frozen operator, the
reference object. v5 instead discovers the gauge stabilizer of the exact
parent sign table:

```text
displacement_A = A.f_0 XOR f_0

GaugeAut(f_0) = {
  A in GL(4,2) x C2
  | normalize(displacement_A).remainder = 0
}.
```

An admitted action therefore preserves the frozen parent modulo a replayable
basis-sign gauge. `GaugeAut(f_0)` is a parent-relative coordinate/gauge
stabilizer. v5 must not relabel it as the complete algebra automorphism group.

The action must also normalize the gauge image. For every `q`, linearity of
`M` and symmetry of `dq` give the exact identity

```text
A_(M,s).dq = d(q compose M).
```

The swap bit does not change the right side because `dq(i,j)=dq(j,i)`. The
pullback `q compose M` need not satisfy the canonical zero-on-basis section, so
the executable removes its linear character and serializes

```text
q*_M(v) = q(Mv) XOR sum_(r=0..3) v_r * q(M e_r).
```

Because the removed term is linear, `d q*_M = d(q compose M)`. v5 must certify
this transport independently of the 48-child relation: for every admitted
action, every one of the 11 canonical gauge basis functions, and every one of
the 256 cells, Sounio compares the pulled-back coboundary with `d q*_M` and
requires zero failures.

Sounio must scan all 40320 actions, emit a nonzero digest over the admitted
action list, require the identity, and for every admitted action:

1. encode `(M,s)` canonically as `2*M+s`, where `0 <= M < 65536` is the
   nonnegative 16-bit row-packed matrix encoding, decode it back to the same
   invertible matrix and `C2` swap bit, and emit or absorb the canonical parent
   gauge;
2. replay the parent displacement on all 256 cells;
3. compute the packed inverse matrix;
4. require the inverse action to be admitted;
5. replay the inverse parent displacement on all 256 cells;
6. require the composition of every ordered pair of admitted actions to be
   present in the admitted census, and check its packed matrix action on all
   16 vectors.

No expected admitted cardinality or swap distribution is fixed in this
Garden. Algebraically, closure follows from the gauge-equivariance identity:
composing two parent displacements then remains a coboundary. The executable
certificate re-establishes that identity for the admitted finite census; it is
evidence for the implementation, not a premise that makes the algebra true.
The executable also checks all ordered action pairs extensionally so a bug in
the implemented admission or composition cannot hide behind that derivation.
Parent replay and gauge equivariance together are the executable action axioms
for the projected quotient; the packed-pair group law alone is not promoted to
a faithfulness theorem.
The emitted 48-child relation is additionally checked extensionally for
reflexivity, symmetry, and transitivity; those finite relation checks do not
replace gauge equivariance.

This is the projected action group on sign tables modulo gauge. v5 does not
choose or claim a lifted group law on triples `(M,swap,q)`: the canonical
gauge attached to each action is an equality witness, not an additional group
coordinate with a separately certified cocycle law.

## Three nested novelty quotients

Novelty is not a scalar. v5 emits a finite spectrum with three nested
equivalence profiles over the same frozen population.

### Q0: extensional identity

```text
f_a ~Q0 f_b iff f_a(i,j) = f_b(i,j) for all 256 cells.
```

v4 already supplies the pairwise-inequality witness. v5 rebinds it to the
parent digest and population order.

### Q1: basis-sign gauge

```text
f_a ~Q1 f_b iff normalize(f_a XOR f_b).remainder = 0.
```

Every admitted pair must carry the canonical 11-bit gauge witness, replayed on
all 256 cells.

### Q2: parent-relative linear/swap/gauge equivalence

```text
f_a ~Q2 f_b iff there exists A in GaugeAut(f_0) such that
  normalize(A.f_a XOR f_b).remainder = 0.
```

Every admitted pair must carry an action witness `(matrix,swap)` and a
canonical gauge witness. The witness is replayed by checking every cell:

```text
A.f_a(i,j) XOR f_b(i,j) = dq(i,j).
```

Because the identity belongs to `GaugeAut(f_0)`, the partitions must refine in
the declared direction:

```text
Q0 refines Q1 refines Q2.
```

The Q2 relation is restricted to the 48 frozen candidates, but every action in
the complete parent stabilizer is considered. An action image may leave the
mixed-cubic grammar; that does not invalidate the quotient. Two candidates
share a Q2 class exactly when some complete-group action and gauge witness
connects their full sign tables.

## Canonical novelty atlas

For each quotient profile, Sounio constructs the complete 48-by-48 Boolean
relation, verifies it is reflexive, symmetric, and transitive, and partitions
all child IDs exactly once.

The canonical representative of a class is the smallest v4 child ID in that
class. This is a serialization rule, not a quality score. For each class the
first executable emits:

- quotient profile ID;
- canonical representative child ID;
- exact 48-bit member mask;
- member count;
- inherited v4 child semantic digests;
- a replayable witness from every member to the representative;
- a class digest bound to the parent, population, profile, members, and
  witnesses.

For each quotient profile it also emits:

- class count discovered by Sounio;
- sum of member counts, which must equal 48;
- minimum and maximum class size;
- relation checks and failures;
- pair-separation checks between distinct classes;
- ordered class digests;
- a nonzero profile digest.

The three profile digests form a `NoveltyAtlas` digest. Its identity changes if
the parent, population, action universe, gauge convention, equivalence
profile, partition, representative rule, or any witness changes.

No count, mask, representative, witness, class digest, profile digest, or atlas
digest is frozen in this Garden. The first Sounio transcript is their only
semantic birthplace.

## The novelty certificate

For a Q2 class `C`, v5 may emit a bounded certificate with the semantic shape:

```text
NoveltyCertificate {
  producer_language = SOUNIO
  producer_role = SEMANTIC_AUTHORITY
  stage = SOUNIO_EXECUTABLE
  parent_semantics_sha256
  population_semantics_sha256
  equivalence_profile = PARENT_LINEAR_SWAP_GAUGE
  action_universe_complete
  parent_stabilizer_complete
  quotient_relation_complete
  canonical_representative
  member_mask
  witness_digest
  class_digest
}
```

The certificate means:

> Within the frozen 48-child v4 population, and under every linear
> XOR-coordinate change and operand exchange that preserves the frozen parent
> modulo basis-sign gauge, this class is distinct from every other emitted Q2
> class.

It does not mean globally new mathematics, a new algebra up to arbitrary
isomorphism, a useful algorithm, a realizable instruction, improved
performance, scientific utility, historical priority, or patentability.

The certificate is evidence-carrying and monotone. A broader later quotient
may merge v5 classes, but may not rewrite what Q2 meant or alter the v5
population. The later result must name a new equivalence profile and retain the
v5 certificate as lineage.

## Generation pipeline after v5

Pireus operator novelty becomes an explicit pipeline:

```text
generate population
-> prove local grammar escape
-> quotient by declared semantic equivalence
-> emit canonical novelty atlas
-> open target materialization envelopes
-> measure target-local realization
-> compare algorithms and scientific workloads
-> promote only the claims separately discharged
```

Generation and quotienting remain semantic authority in Sounio. Processor
ontologies may propose realizations after freeze. They may not change the
population, equivalence profile, class partition, or expected result.

No class is selected in v5. Search objectives, material cost, performance,
scientific behavior, and founder judgment belong to later typed stages. This
prevents quotient size or representative order from becoming a hidden fitness
function.

## Four canonical target envelopes

All inherited child envelopes remain unresolved for exactly:

```text
701200 Darwin Xeon
701201 Apple Silicon
701202 DGX Spark
711001 dual AMD Alveo U250
```

Quotient membership does not merge material receipts. Two semantically
equivalent children can still require different concrete schedules when a
target representation fixes a basis or layout. Conversely, two inequivalent
semantic classes may lower to the same instruction sequence under a restricted
workload. Those are later material observations, not reasons to alter Q2.

v5 creates zero target observations, discharges, lowerings, cost records,
performance records, and material receipts. It inherits all 1920 unresolved
v4 obligations without satisfying any of them.

## Novelty vocabulary

If every census, normalizer, relation, partition, and witness check passes, v5
may emit only the bounded fields:

```text
declared_action_universe_complete=true
parent_gauge_stabilizer_complete=true
extensional_quotient_exact=true
basis_sign_gauge_quotient_exact=true
parent_linear_swap_gauge_quotient_exact=true
canonical_novelty_atlas_complete=true
parent_relative_operator_novelty_typed=true
no_child_selected=true
```

`parent_relative_operator_novelty_typed` has no meaning without its frozen
population, equivalence profile, parent, witnesses, and stage. A serializer or
consumer that drops any of those parameters must fail closed.

The following remain false or open:

```text
global_linear_swap_gauge_quotient_complete
nonlinear_permutation_quotient_complete
isotopy_quotient_complete
algebra_isomorphism_complete
program_equivalence_complete
relative_algebraic_novelty
algorithmic_novelty
material_novelty
scientific_novelty
global_novelty
historical_novelty
priority_claim
external_prior_art_complete
target_lowering_admitted
target_cost_admitted
target_performance_admitted
formal_parity_open
effect_parity_open
material_parity_open
claim_ready
```

The v4 parent being `PARITY_OPEN` does not make v5 parity-open. Every new atlas
identity restarts the ordered authority sequence.

## Executable identity checks

The first Sounio executable must prove extensionally:

1. the exact v4 source, semantics, freeze, transcript, and parity receipt bind;
2. the live v4 population matches its frozen semantics and contains 48 ordered
   children;
3. all 65536 packed matrices are classified and exactly 20160 are invertible;
4. all 40320 matrix/swap actions are considered exactly once;
5. the 11 gauge pivots have full rank and all 2048 canonical gauge words make
   unique zero-remainder round trips;
6. every admitted parent action has a 256-cell canonical gauge replay;
7. every admitted action preserves all 11 gauge basis coboundaries on all 256
   cells;
8. packed matrix `33825` acts as identity on all 16 vectors and the identity
   parent action is admitted;
9. every admitted action makes a lossless `2*M+s` encode/decode round trip and
   every admitted parent action has an admitted packed inverse;
10. every ordered pair of admitted actions composes to an admitted action and
    its packed composition agrees with nested `matrix_apply` on all 16 vectors;
11. Q0, Q1, and Q2 each cover all 48 children exactly once;
12. every emitted membership witness replays on all 256 cells;
13. each relation is reflexive, symmetric, and transitive;
14. Q0 refines Q1 and Q1 refines Q2;
15. distinct classes in a profile have no admitted relation edge;
16. each representative is the smallest member ID, not a ranked winner;
17. every class and profile digest is nonzero and lineage-bound;
18. all target facts remain unresolved and no child is selected.

The executable may use digests for identity after it has performed the exact
cell comparisons. A digest collision assumption may not replace extensional
equality or a replayable gauge/action witness.

## Fail-closed refusals

The Sounio executable and outer Guardian gate must refuse:

- a parent hash, receipt, population order, or matcher mismatch;
- a matrix scan other than 65536 encodings, 20160 invertible matrices, and
  40320 actions;
- an admitted parent action with a nonzero normalized remainder;
- a rank-deficient gauge pivot map, non-unique gauge word, or failed
  gauge-equivariance cell;
- a missing identity or inverse action;
- a gauge witness that fails any of its 256 cells;
- a relation that is not reflexive, symmetric, or transitive;
- a partition that omits or duplicates a child;
- a Q0/Q1/Q2 refinement failure;
- two separate classes connected by an admitted witness;
- a non-minimal canonical representative;
- a count or representative used as ranking or selection;
- any target observation or discharge in v5;
- any nonlinear, isotopy, algebra-isomorphism, algorithmic, material,
  scientific, global, historical, priority, or claim-ready promotion;
- parity before a hash-frozen v5 Sounio result;
- C++, Lean, Koka, Haskell, hardware, or an LLM promoted to semantic producer;
- Python or Rust before process launch;
- policy absence, policy timeout, malformed receipt, or invalid waiver.

## First execution contract

The chronology is mandatory:

1. obtain math review of this Garden as `REVIEW_ONLY`;
2. commit the reviewed Garden;
3. create and commit a matcher-free Sounio executable and structural test;
4. obtain Guardian `ALLOW` for `SOUNIO_EXECUTABLE`;
5. run only `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc ...`;
6. preserve the first transcript and receipt in Git;
7. derive the frozen matcher only from that transcript;
8. replay under Guardian `ALLOW` for `SEMANTICS_FROZEN`;
9. run deliberate negative and transcript-tamper tests;
10. only then consider v5 parity.

The raw compiler ELF must never be invoked directly.

## Falsifiers

The v5 result is demoted if:

- an expected stabilizer size, quotient count, partition, representative, or
  witness was present before the first Sounio run;
- any of the 40320 declared actions was skipped or duplicated;
- an admitted parent action is not an exact gauge stabilizer;
- a class witness fails direct 256-cell replay;
- the computed Q2 relation is not an equivalence relation;
- a child appears in zero or multiple classes of one profile;
- two declared Q2 classes are connected by a parent-stabilizing action;
- a broader equivalence is silently treated as already discharged;
- a canonical representative is interpreted as a best operator;
- a target fact, cost, performance result, or lowering appears without a
  separately authorized target-bound receipt;
- a parity language or target ran before the v5 freeze;
- an external reviewer supplied or confirmed a golden;
- Python, Rust, or a disposable replacement produced semantic evidence;
- any forbidden novelty field becomes true.

## Success sentence

The strongest permitted v5 statement is:

> Pireus constructed, in Sounio, a canonical novelty atlas for its frozen
> 48-child cubic population. It exhaustively enumerated the declared
> `GL(4,2) x C2` universe, discovered the exact actions preserving the frozen
> parent modulo basis-sign gauge, and partitioned the children under
> extensional identity, gauge, and parent-relative linear/swap/gauge
> equivalence with replayable witnesses. The resulting novelty is exact only
> for that frozen population and declared equivalence. No child is selected,
> no target is claimed, and no algebraic, algorithmic, material, scientific,
> historical, or global novelty claim is made.
