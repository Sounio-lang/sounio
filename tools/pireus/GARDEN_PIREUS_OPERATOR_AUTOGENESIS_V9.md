# Garden: Pireus Operator Autogenesis v9

Status: GARDEN
Concept-ID candidate: `SOUNIO-PIREUS-OPERATOR-AUTOGENESIS`
Semantic lane: `pireus-operator-genome-v3-20260829`
Claim ready: no

## Founding Phrase

Pireus must not only search a space of operators. It must learn how to generate
the next space from the exact structure that the previous space failed to
absorb.

## Parent

The first lawful parent is Pireus Operator Seed Kernel v8:

- Sounio source SHA-256:
  `f90f7fb48fb2e8f79142c43d69a30623289f88ffca516157423ae071539249ac`
- frozen semantics SHA-256:
  `db63ada1b919dbf869bf3e74163a64acaf7c6d2ec496fb964b6ca0c689c5f508`
- freeze receipt SHA-256:
  `871544da76207ff1ea8c3e6c92af2efaaa80ee0dcf01e1f728d24522045b578a`
- frozen process evidence SHA-256:
  `6c58170ce4712b1944892cdd488781df29cdeb483712c9b28bf1621214f8f958`
- value carrier: `Z^16`
- operator type: `Z^16 x Z^16 -> Z^16`
- address space: the `F2` vector space `F2^4`
- algebra laws asserted: none
- historical or priority claim: none

v8 is important because it closes one feedback loop: a residual witness from
the v7 quotient search became an executable operator instead of being thrown
away as a failed match.

## Thesis

A failed equivalence witness can contain a typed grammar delta.

The old loop is:

```text
declare grammar -> enumerate -> quotient -> select or reject
```

The v9 loop is:

```text
frozen parent
-> generate typed mutations from residual witnesses
-> execute every generated operator on a complete finite basis
-> quotient under an explicit equivalence contract
-> preserve separating witnesses
-> emit a new frozen parent or a diagnostic residual
```

The system is autogenetic only in this bounded sense. It does not invent truth,
declare novelty, or rewrite its authority policy. It generates new executable
operator descriptions whose semantics still originate in Sounio.

## The Bold Difference

Existing Pireus forges exhaust declared finite families. v9 must also generate
the family declaration itself.

The generative object is not a table alone. It is a typed operator genome with:

1. arity, value carrier, and operator type;
2. linearity profile for every input slot;
3. index group and address program;
4. coefficient program;
5. reduction topology;
6. parenthesization policy;
7. declared equivalence group;
8. preserved and intentionally unpreserved laws;
9. evidence stage and authority role;
10. lowering obligations, still empty before material parity.

The first v9 fragment contains multilinear maps with integer structure
constants. It does not claim that basis evaluation determines an arbitrary
function `Z^16 x Z^16 -> Z^16`.

## Mutation Algebra

The first v9 executable should support a deliberately small mutation algebra.
No mutation is random and no winner is predeclared.

### Residual twist

Given `S(i,j), R(i,j) in F2`, generate:

```text
S'(i,j) = S(i,j) xor R(i,j)
```

The integer coefficient is then lifted explicitly as `1 - 2*S'(i,j)`. The
result is a new sign program, not automatically a new algebra.

### Address deformation

Replace the right input address with a typed affine program over `F2^4`:

```text
j = A*i xor B*d xor c
```

For fixed `i`, the map in `d` is bijective exactly when `B` is in
`GL(4,F2)`. For fixed `d`, the map in `i` is bijective exactly when `A` is in
`GL(4,F2)`. Each genome names its traversal. A `(d,i)` traversal must prove the
induced map `(d,i) -> (i,j)` is a bijection over all 256 input basis pairs and
that every destination has 16 terms. A dual `(d,j)` traversal has the analogous
obligation for `(d,j) -> (i,j)`. Coordinate bijectivity alone is not called a
complete coverage certificate. An invalid address program is a negative
witness, not a fallback.

### Coefficient lift

Allow coefficient programs assembled from frozen sign bits, declared integer
constants, and content-addressed lookup cells. In the first executable these
coefficients cannot depend on operand values, so the declared multilinearity is
preserved. A future nonlinear coefficient program requires a different
certificate or a declared finite value domain; basis testing alone is refused.
The first executable remains in exact integer arithmetic. Floating point and
target instructions are not part of semantic generation.

### Arity lift

Generate distinct ternary programs by explicit composition:

```text
left  = T(T(a,b),c)
right = T(a,T(b,c))
```

These programs must remain distinct unless all 4096 basis triples prove the
trilinear associator is zero for the frozen multilinear operator.
Parenthesization is semantic.

### Reduction mutation

Generate different reduction trees only as explicitly typed programs. A tree
mutation cannot silently claim equality in non-associative or finite-precision
settings.

## Equivalence Is Part Of The Genome

Pireus must never say "new" without saying "new modulo what".

Every generated family declares its quotient transformations, selected from:

- input and output basis changes;
- operand permutation;
- sign gauge or coboundary action;
- index relabeling;
- coefficient renaming that preserves the declared carrier;
- parenthesization equivalence only when proven, never assumed.

Two candidates that differ outside the declared quotient remain distinct.

## Novelty Lattice

v9 records separate booleans and witnesses for:

- `program_distinct`: different typed genome;
- `basis_map_distinct`: different exhaustive structure constants within the
  declared multilinear fragment;
- `finite_function_distinct`: reserved for a later explicitly finite value
  carrier with complete function evaluation;
- `quotient_relative_distinct`: outside a declared equivalence orbit;
- `law_profile_distinct`: different checked associator, commutator, square, or
  other declared invariant profile;
- `algorithmically_distinct`: requires a separately frozen complexity witness;
- `materially_distinct`: requires target receipts from canonical hardware;
- `historically_novel`: requires prior-art review and cannot be produced by an
  LLM verdict;
- `claim_ready`: requires all applicable parity and evidence gates.

Only `program_distinct`, `basis_map_distinct`, `quotient_relative_distinct`,
and `law_profile_distinct` are candidates for the initial Sounio executable,
and none is prefilled.

## First Differentiating Experiment

The matcher-free v9 executable must:

1. ingest v8 only through its frozen source, freeze receipt, and frozen process
   evidence hashes;
2. reconstruct the complete 256-cell v8 seed in Sounio;
3. construct a bounded frontier from the mutation algebra above;
4. assign every genome a canonical content digest before evaluation;
5. refuse any candidate whose declared multilinearity is not preserved by its
   address, coefficient, and reduction programs;
6. evaluate multilinear bilinear candidates on all 256 basis products;
7. evaluate multilinear ternary candidates on all 4096 basis triples when
   arity lift is enabled;
8. compute separating witnesses without target cost information;
9. quotient only under transformations named by the genome;
10. emit every admitted candidate and every rejection reason;
11. keep all novelty, parity, material, historical, priority, and claim-ready
    fields false unless a later stage supplies the required witness.

The executable must contain no expected frontier size, selected candidate,
class count, digest, or output transcript.

## Falsifiers

Demote or reject the v9 thesis if:

- generated genomes cannot be reconstructed from their mutation lineage;
- two supposedly basis-distinct multilinear candidates have the same complete
  structure constants under the declared equivalence contract;
- a nonlinear candidate is admitted by a basis-only certificate;
- a mutation changes carrier, arity, or parenthesization without recording it;
- candidate identity depends on Xeon, Apple, DGX, or U250 cost data;
- a target lowering retrospectively changes semantic selection;
- Python, Rust, a parity language, or an LLM supplies an expected result;
- the system promotes program distinctness into historical novelty.

## Canonical Material Fan-Out

Only after `SEMANTICS_FROZEN`, material parity may open for:

- Intel Xeon;
- Apple Silicon;
- NVIDIA DGX;
- dual AMD Alveo U250.

These targets may measure or realize frozen candidates. They do not select the
semantic winner and cannot create the expected result.

## Evidence Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The immediate next artifact is a matcher-free Sounio executable contract, not
a novelty claim.
