# Garden: Pireus Cubic Operator Forge v4

Status: `GARDEN`

Concept-ID: `SOUNIO-PIREUS-CUBIC-OPERATOR-FORGE`

Semantic-Lane-ID: `pireus-cubic-operator-forge-v4-20260829`

Founder direction preserved:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

Operator Genesis v2 exhausted every bilinear phase on `F2^4`. Operator Genome
v3 turned its selected operator into an executable, content-addressed semantic
normal form and opened parity without running a parity language or target.

v4 crosses the next boundary. Pireus must produce a population of new operator
genomes outside the exhausted bilinear grammar, not merely select another
matrix from the old space. It must do so with a grammar fixed before execution,
an exact finite population, executable witnesses, immutable parentage, and no
hardware or literature claim smuggled into semantic novelty.

The v4 question is fixed before an executable or child digest exists:

> Starting from the frozen v3 genome, can Sounio generate every single mixed
> cubic phase mutation, prove each mutation lies outside the full bilinear
> phase grammar in the declared coordinates, and emit 48 independently
> addressable child genomes without selecting a winner or inventing material
> evidence?

## Authority order

The only admissible order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The first Sounio executable must not contain expected child sign counts,
associator counts, group masks, child digests, population digest, selected
child, target schedule, cost, performance, or result matcher. Sounio produces
all first results.

Lean 4, Koka, C++, Haskell, Xeon, Apple Silicon, DGX, U250, and external LLMs
must not run on v4 before a hash-frozen Sounio result exists. Python and Rust
are forbidden as generators, oracles, validators, freeze producers, or parity
legs. A disposable replacement language may not supply a golden.

## Frozen parent

The immutable semantic parent is Pireus Operator Genome v3:

```text
parent source
  92765416ad8854376a779ef452f89497e2df77f225bf5a4eb5f74f4cd9004a6d
parent semantics
  99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926
parent freeze
  0b4486ae3c7d0034ffb82208f19330b710ed7d7e92115e93a6f411b354dd03f6
parent transcript
  3e79844d3dbd9034e0d8706bf0c3055cba9a7dda0fcfb2daae959e9dbf0c1905
parent parity-open receipt
  b2100377695575e024e333a4519687a0ff727989198f7ee0213d0f78c36bc7eb
```

The parent genotype remains exactly:

```text
bits=4
dimension=16
class_id=26
quadratic_code=198
packed_matrix=1128
matrix_rows=(8,6,4,0)
```

v4 may read and re-execute the parent only through the canonical Sounio
module. It may not edit, backfill, or reinterpret the v3 genome.

## Phase-polynomial grammar

Let `V=F2^4`. Write the left index as `x=(x_0,...,x_3)` and the right
index as `y=(y_0,...,y_3)`. The frozen parent phase is

```text
b_B(x,y) = x^T B y mod 2, with B=1128.
```

v4 admits exactly the mixed cubic monomials that contain variables from both
operands:

```text
L2R1(r,s,t) = x_r * x_s * y_t,  0 <= r < s < 4, 0 <= t < 4
L1R2(r,s,t) = x_r * y_s * y_t,  0 <= r < 4, 0 <= s < t < 4
```

There are `6*4=24` members of each kind and 48 members total. Mutation IDs are
canonical:

1. IDs `0..23` enumerate `L2R1` by increasing `r`, then `s`, then `t`;
2. IDs `24..47` enumerate `L1R2` by increasing `r`, then `s`, then `t`.

For mutation `m`, the child phase and sign are:

```text
p_m(x,y)     = b_B(x,y) XOR m(x,y)
sigma_m(x,y) = cd_sigma(x,y,4) * (-1)^p_m(x,y)
r_m[d]       = sum_(i=0..15) sigma_m(i,i XOR d)
               * a[i] * b[i XOR d].
```

Every monomial contains at least one left and one right variable. Therefore
`m(x,0)=m(0,y)=0`, so the mutation preserves the declared unit cells. This is
a property of the mutation grammar, not a claim about norm, alternativity, or
associativity.

The grammar is the radius-one cubic shell around the frozen parent. It is not
all Boolean phases: it excludes multiple simultaneous mutations, pure-left
terms, pure-right terms, constants, linear terms, quartic and higher terms,
arbitrary sign tables, and changes to the XOR partner law.

## Exact grammar-extension witness

Algebraic normal form over Boolean functions is unique. A mixed cubic monomial
cannot equal any bilinear phase. v4 must not rely on that sentence alone; it
must execute a witness for every child.

For any phase `m`, define its 2-cocycle failure, namely the group-cohomology
differential of the 2-cochain:

```text
delta_m(i,j,k) = m(i,j)
  XOR m(i XOR j,k)
  XOR m(j,k)
  XOR m(i,j XOR k).
```

Every bilinear phase has `delta=0`. For `L2R1(r,s,t)`, the executable uses

```text
i=e_r, j=e_s, k=e_t
```

and must obtain `delta_m=1`. For `L1R2(r,s,t)`, it uses the same coordinate
assignment and must also obtain `delta_m=1`.

The expanded identities are:

```text
delta_L2R1(i,j,k) = (i_r*j_s XOR j_r*i_s) * k_t
delta_L1R2(i,j,k) = i_r * (j_s*k_t XOR k_s*j_t).
```

Thus every child phase lies outside the entire 65536-member bilinear phase
grammar in the declared coordinates. This is an exact finite grammar result.
It is not yet a complete quotient under arbitrary basis changes, arbitrary
basis-sign gauges, nonlinear permutations, isotopy, or algebra isomorphism.

Each monomial is one on exactly 32 of the 256 ordered input pairs because
three Boolean coordinates are fixed and five are free. The executable must
recount that support rather than trust the formula.

## Population, not winner

v4 emits all 48 children. It has no ranking function and no selected child.
This prevents an observed structural statistic from becoming a retrospective
objective.

For every mutation, Sounio must emit a `CubicChildGenome` containing:

- the exact v3 parent digest;
- mutation ID, kind, and three coordinate IDs;
- the 256 derived sign cells in displacement-major execution order;
- the 32 derived eight-cell comparison masks;
- unit-preservation checks;
- monomial support recount;
- the explicit nonzero `delta` witness;
- square-negative, commutator-defect, and associator-defect counts;
- four unresolved target envelopes;
- child semantic digest and child lineage digest.

The 48 child digests form an ordered content-addressed population. The
population digest absorbs the grammar identifier, parent digest, all mutation
descriptors, and all child semantic digests.

No child digest, mask, defect count, or population digest is fixed in this
Garden. Those values must first be produced by Sounio.

The three structural counts are diagnostics of the complete child sign table
`cd_sigma XOR b_B XOR m`, not invariants of the mutation monomial alone. The
fixture is smoke-only because both evaluators consume the same already
extensionally checked child sign bits; it is not an independent proof.

## Executable identity checks

The first executable must prove extensionally:

1. exactly 48 mutation descriptors were emitted in canonical order;
2. every descriptor contains three in-range, distinct-within-side coordinates;
3. every mutation truth table has 256 cells and support 32;
4. all 48 explicit `delta` witnesses evaluate to one;
5. no mutation truth table equals any bilinear truth table in the declared
   coordinates;
6. every pair of mutation truth tables differs in at least one cell;
7. every child sign is exactly `-1` or `+1`;
8. every child differs from the parent sign table in exactly the mutation
   support cells;
9. every child preserves all 256 XOR partner, destination, and ordinal fields;
10. every child produces exactly 32 comparison groups;
11. the parent source, semantics, freeze, transcript, and parity receipt match;
12. no child is selected and no target obligation is discharged.

The direct reference evaluator and child-microprogram evaluator must preserve
the same ascending-`i` addition spine. A bit-exact integer fixture may compare
the two paths for every child, but it is a smoke check after extensional cell
identity, not a universal floating-point theorem.

## Merkle lineage

The lineage is an append-only semantic DAG:

```text
v2 selected operator
-> v3 executable parent genome
-> v4 mutation descriptor
-> v4 child genome
```

A child identity must change if its parent digest, mutation coordinates,
truth table, microprogram, target-envelope state, or serialization changes.
Two children may share a material phenotype later, but they may not share a
semantic identity unless their complete serialized genomes are equal.

Deletion, in-place mutation, parent rebinding, and digest reuse are refused.

## Four canonical target envelopes

Every child inherits unresolved envelopes for exactly:

```text
701200 Darwin Xeon
701201 Apple Silicon
701202 DGX Spark
711001 dual AMD Alveo U250
```

Each child therefore starts with 40 unresolved obligations and zero
observations, discharges, lowerings, cost records, performance records, and
material receipts. Across 48 children, v4 creates 1920 unresolved obligations.

This is deliberate. Generating an operator is not evidence that any processor
can execute it efficiently or at all. The dual U250 declaration keeps two
cards and two engine slots without inventing a partition.

Processor ontologies may later generate phenotype candidates against each
child. They may not change a child sign, partner, destination, ordinal,
reduction barrier, or expected result.

## Novelty vocabulary

If all grammar and population certificates pass, v4 may emit only:

```text
relative_bilinear_grammar_novelty=true
mixed_cubic_population_complete=true
noncocycle_witness_complete=true
executable_child_genomes=true
```

`relative_bilinear_grammar_novelty` means exact inequality from every pure
bilinear phase in the declared coordinates, witnessed by a nonzero cochain
defect. It is a bounded semantic-grammar statement.

The following remain false:

```text
declared_gl4_gauge_inequivalence
relative_algebraic_novelty
algebra_isomorphism_complete
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

The v3 parent being `PARITY_OPEN` does not make its v4 children parity-open.
Every new semantic identity restarts the ordered authority sequence.

## Fail-closed refusals

The Sounio executable and outer Guardian gate must refuse:

- a parent hash or parent matcher mismatch;
- any mutation count other than 48;
- duplicate, missing, reordered, or malformed descriptors;
- a pure-left, pure-right, repeated-within-side, or out-of-range coordinate;
- monomial support other than 32;
- a zero declared defect witness;
- equality to a bilinear phase promoted as grammar novelty;
- two identical child truth tables;
- a child that changes a partner, destination, ordinal, or reduction barrier;
- a selected winner or ranking objective in v4;
- a target observation without a material receipt;
- any discharged target obligation in the first executable;
- GL/gauge, algebraic, algorithmic, material, scientific, global, historical,
  priority, or claim-ready promotion;
- parity before a hash-frozen v4 Sounio result;
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
10. only then consider v4 parity.

The raw compiler ELF must never be invoked directly.

## Falsifiers

The result is demoted if:

- any child was present as an expected result before the first Sounio run;
- an emitted mutation has support other than 32;
- an emitted mutation has zero defect at its declared witness;
- a pure bilinear phase reproduces an emitted mutation truth table;
- two mutation IDs produce the same truth table;
- a child changes anything outside its phase/sign layer;
- the population contains fewer or more than 48 children;
- a target fact, cost, performance result, or lowering appears without a
  target-bound receipt;
- a parity language or target ran before the v4 freeze;
- any external reviewer supplied or confirmed a golden;
- any forbidden novelty field becomes true.

## Success sentence

The strongest permitted v4 statement is:

> Pireus generated, in Sounio, the complete 48-member single-mixed-cubic shell
> around its frozen operator genome. Every child has an executable nonzero
> cochain-defect witness that places its phase outside the full bilinear grammar
> in the declared coordinates, an immutable content-addressed lineage, and four
> unresolved materialization envelopes. No child is selected, no processor is
> claimed, and no broad novelty claim is made.
