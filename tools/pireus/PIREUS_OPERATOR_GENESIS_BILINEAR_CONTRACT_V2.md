# Pireus Bilinear Operator Genesis v2

Status: `SEMANTICS_FROZEN`

Pireus v2 is the first Sounio authority artifact that exhausts the complete
16-bit bilinear phase grammar over `F2^4`, proves its declared gauge quotient,
forms exact classes under every admitted linear/operand-exchange symmetry, and
selects a new operator by a structural objective.

It is an operator foundry result. It is not a claim that all possible
16-dimensional algebras, sign tables, algorithms, or prior art were exhausted.

## Authority chronology

The required order is preserved:

```text
GARDEN
  -> SOUNIO_EXECUTABLE
  -> SEMANTICS_FROZEN
  -> PARITY_OPEN
  -> CLAIM_READY
```

The immutable chronology is:

- mathematical Garden: commit `eb6e7997dc`;
- first matcher-free Sounio executable: commit `34619031a9`;
- Sounio frozen matcher: commit `bdfb27cb3c`.

The first execution occurred only after `34619031a9` existed. That source
contained no expected stabilizer size, class count, selected matrix, signature,
witness, or digest. The values below were produced by Sounio and then admitted
as goldens by the later matcher.

Lean, Koka, C++, Haskell, external LLMs, Xeon, DGX, Apple Silicon, and AMD Alveo
U250 do not define or amend these semantics. Their possible roles remain formal
parity, effect parity, material parity, optional denotational baseline, review,
or hardware realization.

## Generated grammar

For each packed binary matrix `B` in `[0,65535]`, Sounio defines

```text
b_B(i,j)     = i^T B j                       in F2
sigma_B(i,j) = cd_sigma(i,j,4) * (-1)^b_B(i,j)
r_B[d]       = sum_i sigma_B(i,i XOR d) * a[i] * b[i XOR d]
```

The destination is the fixed XOR displacement `d`; execution therefore has a
horizontal-reduction shape. The base twist is imported from the authoritative
Sounio `cd_sigma` implementation rather than restated in the contract.

Sounio scanned all 65536 matrices. Every induced phase is a bilinear 2-cocycle.
This says that the bilinear phase does not add an associator obstruction to the
base twist; it does not say that the resulting product is associative.

## Gauge quotient

A basis-sign gauge is `q:F2^4 -> F2` with `q(0)=0`, acting by

```text
delta q(i,j) = q(i) XOR q(j) XOR q(i XOR j).
```

The gauge domain has dimension 15, `ker delta` (the linear functionals) has dimension 4, and
its full coboundary image has dimension 11. The full image is not contained in
the bilinear grammar. Its intersection with bilinear forms is precisely the
six-dimensional space of alternating matrices.

Consequently the 65536 bilinear matrices form 1024 declared gauge classes of
64 matrices. The complete class invariant is the ten-bit quadratic form

```text
Q_B(x) = x^T B x.
```

Sounio checked all 65536 fibers: each matrix has the same `Q_B` as its
upper-triangular representative and their difference normalizes to zero under
the gauge normalizer. All 1024 buckets have size 64.

This is an exact quotient inside the declared bilinear family. It is not a
quotient of all `2^256` sign tables.

## Declared equivalence

Sounio enumerated all 20160 elements of `GL(4,2)` independently by both a packed
invertibility scan and generator BFS. With operand exchange this gives 40320
candidate actions.

An action is admitted only when its displacement of the Cayley-Dickson base is
a bilinear phase modulo gauge. Exactly 336 actions are admitted:

```text
without operand exchange = 168
with operand exchange    = 168
total                    = 336
```

Every admitted inverse was checked, and every affine action/inverse pair was
checked on all 1024 quadratic codes, for 344064 successful round trips.

The induced affine action partitions the 1024 codes into exactly 32 classes.
Their quadratic sizes sum to 1024 and their raw sizes sum to 65536. The four
classes reached by the diagonal v1 grammar are recorded separately.

The declared equivalence covers linear XOR-coordinate changes, operand
exchange, and basis-sign gauge inside this family. It does not cover nonlinear
permutations, arbitrary real basis changes, general isotopy, or an unbounded
algebra-isomorphism problem.

## Frozen corpus

The corpus contains exactly three named tables:

1. untwisted XOR;
2. Cayley-Dickson-16;
3. the diagonal bicharacter `(-1)^parity(i AND j)`.

Exact incidence under all 40320 actions was computed for each corpus member.
Only Cayley-Dickson-16 lies in the declared family, at quadratic code zero and
class zero. Incidence masks are constant on every declared class, with zero
failures.

This corpus is deliberately small and frozen. Inequivalence to it is relative
semantic novelty, not a prior-art conclusion.

## Selected operator

Pireus excludes every class represented in the frozen corpus or in the v1
diagonal grammar. For each remaining class it computes

```text
(absolute associator delta,
 absolute commutator delta,
 absolute square-negative delta)
```

to each corpus table, takes the lexicographically nearest corpus tuple, then
maximizes that nearest tuple. Ties choose the smallest packed matrix. This is a
structural objective; it is not Hamming distance between orbit representatives.

The frozen winner is an execution golden from the exhaustive Sounio census:

```text
class_id                    = 26
quadratic_code              = 198
minimum_packed_matrix       = 1128
quadratic_codes_in_class    = 28
raw_matrices_in_class       = 1792
square_negative_count       = 5
commutator_defects          = 90
associator_defects          = 1848
nearest_corpus              = Cayley-Dickson-16
structural_delta            = (0,120,10)
represented_in_v1           = false
equivalent_to_frozen_corpus = false
```

Packed matrix 1128 has rows `(8,6,4,0)` in the module's low-nibble-first row
encoding. Its quadratic form is

```text
Q(x) = x_1 XOR x_2 XOR (x_0 x_3) XOR (x_1 x_2).
```

The ten-bit quadratic encoding orders coefficients as diagonal monomials
`x_0,x_1,x_2,x_3` in bits `0..3`, followed by
`x_0x_1,x_0x_2,x_0x_3,x_1x_2,x_1x_3,x_2x_3` in bits `4..9`.
Thus bits `1,2,6,7` are set and the packed code is
`2 + 4 + 64 + 128 = 198`.

The witness is the commutator-defect component: 90 for the selected class
against 210 for Cayley-Dickson-16. The unchanged associator count 1848 is an
expected consequence of multiplying the base twist by a bilinear 2-cocycle.

## Admitted claims

The frozen receipt permits exactly these positive novelty statements:

```text
expanded_bilinear_grammar_exhausted=true
bilinear_gauge_quotient_exact=true
declared_family_equivalence_exact=true
corpus_incidence_exact=true
relative_semantic_novelty=true
relative_grammar_extension_novelty=true
```

The receipt fixes the stronger boundaries false:

```text
relative_algebraic_novelty=false
algebra_isomorphism_complete=false
all_sign_tables_exhausted=false
orbit_hamming_distance=false
algorithmic_novelty=false
material_novelty=false
scientific_novelty=false
global_novelty=false
historical_novelty=false
priority_claim=false
external_prior_art_complete=false
claim_ready=false
```

In particular, v2 does not answer the open question of whether this twisted XOR
product admits a subquadratic algorithm. It discovers and classifies operators;
it does not yet synthesize a lowering or cost proof for matrix 1128.

## Enforcement

`scripts/ci/pireus_operator_genesis_bilinear.sh` is the dedicated admission
gate. It pins the Garden, base twist, first executable, frozen sources,
toolchain, Xeon hardware receipt, commands, result contract, full transcript,
and Sounio digests. It proves the commit chronology, replays the transcript and
test byte-identically, and fails closed through the native Loom Guardian.

Negative gates include missing policy, policy timeout, Python and Rust oracles,
C++ semantic authority, LLM review promotion, parity before freeze, and
transcript tampering. Python and Rust are denied before any such process can be
launched.

The stage remains `SEMANTICS_FROZEN`. Formal, effect, and material parity are
not opened by this contract. A later admission must bind every parity receipt
to the exact frozen semantics hash without promoting the parity producer to
semantic authority.

## Canonical material targets

After `PARITY_OPEN`, the unchanged selected operator may be lowered and measured
on:

- the Xeon fleet;
- DGX targets;
- Apple Silicon;
- both AMD Alveo U250 cards.

Those targets may discover schedules, vector permutations, FPGA dataflows,
resource bounds, and cost models. They cannot retrospectively choose another
operator or alter `B=1128`. That is the boundary between Pireus as semantic
foundry and Pireus as hardware ontology.
