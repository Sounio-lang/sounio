# Pireus Operator Genesis GL4 v1

Status: `SEMANTICS_FROZEN`

Pireus now has an executable first form of operator genesis: Sounio generates
an exhaustible family of twisted-XOR products, canonicalizes every member under
the declared equivalence action, partitions the family into semantic classes,
and selects a candidate by a frozen objective. This is a generator with a
certificate, not a novelty label attached after the fact.

## Authority chain

The admitted order is:

```text
GARDEN
  -> SOUNIO_EXECUTABLE
  -> SEMANTICS_FROZEN
  -> PARITY_OPEN
  -> CLAIM_READY
```

The first result was produced by Sounio before any expected winner, class
count, score, or digest existed in executable source. The immutable chronology
is:

- Garden: commit `47a1e8a3ef`;
- first Sounio executable without a matcher: commit `fd2ff88871`;
- per-candidate semantic-key exposure: commit `c62eb1120e`;
- exact key/digest alignment after the first class transcript exposed a
  certificate mismatch: commit `01d8645fb8`;
- Sounio frozen matcher: commit `2915618ce5`.

Lean, Koka, C++, Haskell, Xeon, DGX, Apple Silicon, and AMD Alveo U250 may not
rewrite this result. Their roles remain formal parity, effect parity,
material parity, optional denotational baseline, or review.

## Generated family

For `m` in `[0,15]` and `i,j` in `F2^4`, Sounio exhausts:

```text
sigma_m(i,j) = cd_sigma(i,j,4) * (-1)^parity(m AND i AND j)
r_m[d]       = sum_i sigma_m(i,i XOR d) * a[i] * b[i XOR d]
```

The family contains exactly 16 candidates. The grammar is exhaustive relative
to that declaration; it is not an exhaustive search over all `2^256` sign
tables, and it assumes the imported Cayley-Dickson twist.

## Declared equivalence

Each sign table is quotiented by:

- all 20160 elements of `GL(4,2)`;
- operand exchange;
- all 32768 basis-sign gauges, with a 16-element kernel and therefore 2048
  distinct coboundary actions.

A gauge is a function `q:F2^4 -> F2` with `q(0)=0`, acting on sign bits by
`f^q(i,j)=f(i,j) XOR q(i) XOR q(j) XOR q(i XOR j)`. This is the domain whose
cardinality is `2^15=32768`.

The declared potential action universe is `20160 * 2 * 2048 = 82575360` per
table. Canonical semantic identity is exact equality of:

```text
(diagonal_vector, commutator_vector, normalized_table)
```

The matrix, swap choice, and pruning counts are provenance for obtaining that
identity. They are frozen in the receipt but are not included in the semantic
identity digest.

## Sounio result

The 16-member grammar collapses into four declared semantic classes:

```text
generation ordinals: 0 0 0 0 0 0 0 | 1 | 2 2 2 2 2 2 2 | 3
phase masks:         9 10 11 12 13 14 15 | 0 | 1 2 3 4 5 6 7 | 8
```

The selected operator is:

```text
generation_ordinal             = 15
phase_mask                     = 8
canonical_matrix               = 62024
canonical_swap                 = false
canonical_quotient_distance    = 112
nearest_corpus                 = untwisted XOR
declared_equivalent_to_corpus  = false
```

The frozen corpus has exactly three named members: untwisted XOR,
Cayley-Dickson-16, and the diagonal bicharacter `(-1)^parity(i AND j)`.

The structural fingerprint is 144 positive and 112 negative signs, 210
commutator defects, 1848 associator defects, and exactly seven negative signs
for every XOR displacement. The inequivalence witness against the nearest
corpus is cell `(1,6)`, with candidate sign `-1` and corpus sign `+1`.

`canonical_quotient_distance=112` is the Hamming distance between canonical
normalized representatives. It is deliberately not called minimum orbit
Hamming distance.

## Admitted novelty

The frozen receipt permits only:

```text
relative_semantic_novelty=true
declared_gl4_gauge_inequivalence=true
relative_monomial_gauge_novelty=true
```

It fixes all stronger statements false:

```text
relative_algebraic_novelty=false
algebra_isomorphism_complete=false
algorithmic_novelty=false
material_novelty=false
scientific_novelty=false
global_novelty=false
historical_novelty=false
priority_claim=false
claim_ready=false
```

The equivalence search does not cover nonlinear permutations, arbitrary real
linear basis changes, general isotopy, anti-isotopy beyond operand exchange,
continuous automorphisms, a larger corpus, or prior-art search.

## Enforcement

`scripts/ci/pireus_operator_genesis_gl4.sh` is the admission gate. It:

- proves the Garden preceded the first matcher-free Sounio executable;
- pins all source, toolchain, hardware, command, result, and transcript hashes;
- replays the Sounio authority and dedicated test byte-identically;
- obtains Guardian `ALLOW` for pre-execution and freeze stages;
- obtains Guardian `DENY` before a Python oracle can launch;
- fails closed on missing policy, timeout, C++ semantic authority, or promotion
  of an LLM review;
- rejects tampered winner and digest transcripts.

The frozen stage does not open parity automatically. A separate Guardian
admission bound to the exact semantics hash is required.

## Canonical material targets

After `PARITY_OPEN`, the unchanged operator may be lowered and measured on:

- the Xeon fleet;
- DGX targets;
- Apple Silicon;
- both AMD Alveo U250 cards.

Those targets may produce cost models, schedules, kernels, and material
receipts. They cannot select a replacement operator or amend its Sounio
meaning. That separation is what allows Pireus to become an operator foundry
without turning hardware accidents into semantics.
