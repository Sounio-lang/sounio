# Garden: Pireus Operator Genesis GL4 v1

Status: `GARDEN`

Founder direction preserved:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

This campaign strengthens the first Operator Genesis search. The v0 winner
survived 48 coordinate-permutation and operand-exchange actions. v1 asks a
harder question before any parity language is allowed to run:

> Does a generated twisted-XOR operator remain outside the frozen corpus after
> every linear XOR-basis change in `GL(4,2)`, every basis-sign gauge, and
> optional operand exchange?

The answer must be produced first by Sounio. The Garden fixes the question,
the exact finite universe, the canonicalization, the search objective, and the
permitted vocabulary before the first v1 executable exists.

## Authority order

The only admissible order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Sounio owns the first executable equivalence relation, candidate selection,
canonical representative, witness, and expected result. Lean, Koka, C++,
Haskell, Xeon, DGX, Apple Silicon, U250, and external LLMs may compare, prove,
or measure only after the Sounio artifact is frozen by hash.

Python and Rust are forbidden as generators, oracles, validators, freeze
producers, or parity legs. Existing historical Python results are not inputs to
this campaign. No disposable replacement language may supply expected values.

## Operator grammar

v1 preserves the exhaustible 16-member grammar from v0. For `m` in `[0,15]`,
indices `i,j` in `V = F2^4`, and live Sounio `cd_sigma`:

```text
phase_m(i,j) = (-1)^parity(m AND i AND j)
sigma_m(i,j) = cd_sigma(i,j,4) * phase_m(i,j)
r_m[d] = sum_i sigma_m(i,i XOR d) * a[i] * b[i XOR d]
```

The generation seed may reorder candidates but may not add, remove, or mutate
a grammar member. The source digest must absorb all 4096 candidate sign cells
and the imported `cd_sigma` source hash.

## Exact sign-gauge quotient

Write a sign table as a bit table `f: V x V -> F2`, with `0` for `+1` and `1`
for `-1`. A basis-sign gauge is a function `q: V -> F2` with `q(0)=0`. It acts
by the order-two coboundary

```text
f^q(i,j) = f(i,j) XOR q(i) XOR q(j) XOR q(i XOR j).
```

There are `2^15 = 32768` gauge functions. The kernel consists exactly of the
16 linear characters `V -> F2`, so there are `32768 / 16 = 2048` distinct
coboundary actions.

The search must quotient all gauges exactly without scanning 32768 gauges for
every table. It uses this deterministic tree normalization:

1. set `q(0)=q(1)=q(2)=q(4)=q(8)=0`;
2. visit every other `x` in increasing numeric order;
3. let `e` be the highest set bit of `x` and `p=x XOR e`;
4. set `q(x)=f(p,e) XOR q(p)`;
5. emit `N(f)=f^q`.

This forces the 11 tree cells `N(f)(p,e)` to zero. Every gauge orbit has one
and only one such normalized table: any residual gauge preserving the 11 tree
cells differs only by a linear character, whose coboundary is zero.

The executable must independently count the 16-element kernel, the 2048
distinct actions implied by the quotient, and exhaustively verify on the
selected candidate that all 32768 gauges normalize to the same table.

## Complete `GL(4,2)` action

A matrix is encoded by four 4-bit rows in one 16-bit integer. Sounio scans all
65536 encodings in ascending order, accepts exactly the matrices whose induced
map on all 16 vectors is bijective, and therefore enumerates all

```text
|GL(4,2)| = (16-1)(16-2)(16-4)(16-8) = 20160
```

maps. Matrix action on a table is

```text
(M.f)(i,j) = f(Mi,Mj).
```

Optional operand exchange adds

```text
(M.swap.f)(i,j) = f(Mj,Mi).
```

The executable must not trust the count alone. Starting at the identity, a BFS
under adjacent row swaps plus one elementary row transvection must reach the
same 20160 encoded matrices, and every reached encoding must be bijective.
Because all invertible linear maps are enumerated extensionally, closure under
composition and the existence of inverses follow inside the declared set.

The combined declared action universe contains

```text
20160 linear maps * 2 operand orders * 2048 distinct gauges
  = 82575360 potential transformations per table.
```

Orbit sizes may be smaller because stabilizers are allowed.

## Canonical representative

For each of the 40320 linear-map and operand-order actions, Sounio first gauge
normalizes the transformed table. The canonical orbit key is the least tuple

```text
(diagonal_vector, commutator_vector, normalized_table)
```

under row-major lexicographic order with `0 < 1`. The diagonal and commutator
are gauge invariants, so the executable may use them as exact staged pruning:

```text
diagonal_vector[i] = f(i,i)                         for i=0..15
commutator_vector[16*i+j] = f(i,j) XOR f(j,i)      for i,j=0..15
```

Both vectors use the displayed ascending row-major order, including the zero
diagonal and both orientations of every unordered pair. The staged search is:

1. retain actions with least 16-cell diagonal vector;
2. among those, retain actions with least 256-cell commutator vector;
3. among those, choose the least normalized 256-cell table;
4. break identical-table ties by lowest matrix encoding, then no-swap before
   swap.

This hierarchical key is the v1 canonical definition. It is not claimed to be
the same ordering as a table-only lexicographic minimum.

The selected representative must be invariant under a generating set for the
full declared action: adjacent coordinate swaps, an elementary transvection,
operand exchange, and all 32768 gauges.

## Corpus and search objective

The corpus remains executable and ordered:

1. untwisted XOR, all coefficients `+1`;
2. live Cayley-Dickson-16 `cd_sigma`;
3. diagonal bicharacter `(-1)^parity(i AND j)`.

Sounio canonicalizes all 16 candidates and all three corpus members under the
same v1 equivalence universe. Candidate equivalence is exact equality of the
canonical orbit keys.

For ranking only, v1 defines `canonical_quotient_distance` as the Hamming
distance between the normalized-table components of two canonical keys. This
is a deterministic distance between quotient representatives. It is not the
minimum raw-table Hamming distance over the combined orbit and must never be
reported as such.

The search maximizes the minimum `canonical_quotient_distance` to the three
corpus representatives. Ties are broken by the seeded exhaustive generation
order, then by mask. The first executable contains no exact expected winner,
score, action, fingerprint, or digest.

## Permitted novelty statement

If and only if all candidates, matrices, operand orders, gauge checks, and
corpus representatives complete and the winner has no equal canonical key,
the executable may emit:

```text
relative_semantic_novelty=true
declared_gl4_gauge_inequivalence=true
relative_monomial_gauge_novelty=true
relative_algebraic_novelty=false
algebra_isomorphism_complete=false
```

The stronger bounded field means inequivalence under all linear XOR-basis
changes, all basis-sign gauges, and operand exchange relative to this corpus.
It does not cover nonlinear permutations, arbitrary real linear basis changes,
isotopy, anti-isotopy beyond operand exchange, continuous automorphisms,
extensions of the corpus, or historical prior art.

Global, historical, priority, algorithmic, material, scientific, and complete
algebraic novelty remain false. `CLAIM_READY` remains closed.

## Required executable certificates

The first Sounio execution must emit and hash at least:

- all 65536 matrix encodings considered;
- exactly 20160 invertible matrices in ascending encoding order;
- BFS generator closure count 20160;
- 32768 gauge functions, kernel size 16, quotient-action count 2048;
- exhaustive gauge-normalization invariance for the selected candidate;
- generator-invariance checks for the selected canonical key;
- canonical keys for all 16 candidates and all three corpus members;
- exact equality/inequality status against each corpus member;
- the selected mask, canonical matrix, operand order, quotient distance,
  differing-cell witness, and structural fingerprint;
- nonzero digests for the grammar, matrix universe, gauge quotient, corpus,
  search, selected candidate, canonical key, witness, and receipt;
- exact novelty scope and every forbidden promotion flag.

## Fail-closed refusals

The executable and final gate must refuse:

- a matrix scan other than 65536 encodings or a `GL(4,2)` count other than
  20160;
- generator closure that does not reach exactly the enumerated matrix set;
- gauge count, kernel count, or quotient count drift;
- a failed tree-cell normalization or gauge-invariance check;
- incomplete candidate or corpus canonicalization;
- canonical-key equality promoted as novelty;
- a fingerprint or positive distance used without exact key inequality;
- `canonical_quotient_distance` relabeled as minimum orbit Hamming distance;
- complete algebraic, historical, scientific, material, or global novelty;
- parity before a hash-frozen Sounio result;
- C++, Lean, Koka, Haskell, hardware, or an LLM promoted to semantic producer;
- a Python or Rust oracle before process launch;
- policy absence, policy timeout, malformed receipts, or an unscoped waiver.

## Falsifiers

The v1 result is demoted if any of these occurs:

- two gauge-related tables normalize differently;
- two declared-equivalent tables have different canonical keys;
- the matrix scan and generator closure enumerate different sets;
- replay selects a different candidate from identical frozen inputs;
- the winner equals a corpus canonical key;
- a serialized semantic input can change without changing its owning digest;
- a later parity implementation supplies or rewrites an expected value;
- the Guardian launches a forbidden oracle before denial.

## Material boundary

After `SEMANTICS_FROZEN`, the unchanged selected operator may be lowered to the
canonical Xeon, DGX, Apple Silicon, and dual-U250 targets. Those targets vote on
cost, resource fit, and realized execution. They cannot select a different
operator or amend the Sounio meaning.

## Exit criteria

`SOUNIO_EXECUTABLE` opens only after Sounio emits the complete unanticipated
v1 result and all positive/negative self-checks pass without frozen matchers.

`SEMANTICS_FROZEN` opens only afterward, when source, imported base twist,
Garden, matrix/gauge universes, corpus, result, command, toolchain, hardware,
Guardian decisions, and transcript are hash-bound.

`PARITY_OPEN` requires a separately admitted Guardian action naming that exact
semantics hash. `CLAIM_READY` remains closed in v1.
