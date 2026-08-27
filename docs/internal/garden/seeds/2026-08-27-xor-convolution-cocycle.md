# XorConvolution: Make The Twist A Semantic Operand

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

The current Cayley-Dickson multiplication exposes 256 scalar products and
asks a general lowering pipeline to rediscover an XOR-indexed algebra. The
founder supplied the reindexing that the language should see directly:

```text
r[i XOR j] += sigma(i, j) * a[i] * b[j]

d = i XOR j
j = i XOR d

r[d] = SUM_i sigma(i, i XOR d) * a[i] * b[i XOR d]
```

For a fixed displacement `d`, every admitted pair writes to the same output.
The semantic operation is therefore a family of displacement-indexed
reductions. It is not a collection of unrelated scalar accumulations.

This Garden opens the contract `XorConvolution(bits, twist)`. The founder's
original spelling `XorConvolution(bits, cocycle)` remains the intended surface
syntax, subject to the vocabulary distinction below. This seed creates no
lowering, instruction choice, cost, or performance claim.

## Founder Evidence

The founder research note observed at
`docs/research/cayley_dickson_vector_lowering_20260826.md`, SHA-256
`533b8aa9e407f16848e2e554da45d1111b90a18927884bda4be03da3c3461bbe`,
records:

- a numerical comparison between direct and displacement-indexed forms;
- a sign census for the 16-dimensional Cayley-Dickson table;
- a non-separability probe;
- a regularity in the nonzero displacement sign rows;
- emitted-code measurements from LLVM and Madaros;
- discarded measurements and the reasons they were discarded;
- open questions about ISA permutation coverage and subquadratic algorithms.

Those observations motivate the executable. They are not expected values for
it. The first authoritative counts, witnesses, error bits, and classification
must be produced by Sounio after this Garden is admitted.

## Vocabulary Boundary

Three predicates must not be collapsed:

```text
twist:
  a pure coefficient function sigma(i, j, bits)

left-square law:
  sigma(i, j) * sigma(i, i XOR j) = -1 for admitted nonzero i

group 2-cocycle law:
  sigma(i, j) * sigma(i XOR j, k)
    = sigma(j, k) * sigma(i, j XOR k)
```

The repository historically calls the left-square law a Cayley-Dickson
"cocycle" identity. In standard twisted-group-algebra terminology, the third
predicate is the associativity condition. The first Sounio executable must
evaluate and report these predicates separately. It must not promote the local
name of one identity into proof of another.

The public contract may retain the founder's `cocycle` parameter name only if
the semantic record carries an explicit classification. A coefficient function
that fails the group 2-cocycle law remains a valid `twist` for an XOR reduction,
but cannot be marked `AssociativeCocycle`.

## First Executable Contract

The first Sounio executable must create the operation and its result without an
external oracle. It must:

1. define a bounded `XorConvolutionContract` carrying `bits`, dimension,
   coefficient classification, accumulation order, and validation state;
2. admit only dimensions exactly derived as `2^bits`, with explicit bounds and
   capacity failure;
3. treat the twist as a pure function of `(i, j, bits)` and verify that each
   returned coefficient is admitted by the declared coefficient domain;
4. implement the direct pair form and displacement-indexed horizontal form as
   distinct Sounio functions;
5. derive `j = i XOR d` inside the horizontal form and never accept a caller-
   supplied destination table as semantic authority;
6. freeze the scalar accumulation order used by both forms so floating-point
   comparison is meaningful and reproducible;
7. generate its deterministic input vectors in Sounio;
8. compare the complete output vectors, report exact result bits, maximum
   absolute difference, mismatch count, and first mismatch witness;
9. enumerate the complete twist table for the admitted dimension and create
   the first sign census in Sounio;
10. decide whether the twist is zero-free, displacement-only, rank-one sign
    separable, normalized, left-square, and a group 2-cocycle;
11. emit the first counterexample for each failed predicate, including all
    indices and both compared sides;
12. enumerate the negative count of every displacement sign row without a
    preloaded expected count;
13. compute deterministic contract, input, direct-result, horizontal-result,
    twist-table, property, and witness digests in Sounio;
14. state which facts are operation semantics and which are only properties of
    the selected Cayley-Dickson twist at `bits = 4`;
15. emit no lowering, ISA, hardware, instruction-count, or speed claim.

The executable may duplicate the recursive Cayley-Dickson sign definition in
its first isolated witness only when it also checks complete table agreement
with the canonical Sounio implementation. A later read-path switch must remove
that duplication or make one definition explicitly derived from the other.

## Semantic Shape

The operation boundary is:

```text
XorConvolutionContract {
  bits
  dimension
  twist_identity
  coefficient_domain
  accumulation_order
  classification
}

DirectPairForm(contract, a, b) -> r
DisplacementReductionForm(contract, a, b) -> r
TwistClassification(contract) -> evidence
```

The equivalence obligation is pointwise:

```text
for every d in [0, 2^bits):
  DirectPairForm(a, b)[d]
    = SUM_i twist(i, i XOR d) * a[i] * b[i XOR d]
```

The equality above is an indexing identity over a fixed accumulation order.
It does not imply that the twist is separable, associative, transformable by a
Walsh-Hadamard transform, or lowerable at any particular cost.

## Required Negative Surface

At minimum, Sounio must deliberately reject or falsify:

- negative `bits`, an overflowing dimension, or capacity exhaustion;
- a vector length inconsistent with `2^bits`;
- a twist value outside its declared coefficient domain;
- a caller-supplied XOR destination inconsistent with `i XOR j`;
- a displacement row with a caller-supplied partner inconsistent with
  `i XOR d`;
- a comparison that changes accumulation order between forms;
- a missing table cell or a duplicate `(i, j)` cell;
- a zero-free claim when any admitted coefficient is zero;
- a displacement-only claim without equality across every row fiber;
- a separability claim without a complete factor witness;
- a group 2-cocycle claim without the full triple law;
- an associative classification when any associator-defect witness exists;
- a Walsh-Hadamard eligibility claim inferred only from XOR indexing;
- a Fano interpretation inferred only from a count containing the number seven;
- a vector-lowering or instruction-cost claim without a material receipt;
- a parity result promoted into the Sounio authority stream;
- a result produced by Python, Rust, Node, Ruby, shell, `awk`, `bc`, an RDF
  engine, or an external model.

Shell may launch the canonical Sounio wrapper and hash frozen files. It cannot
compute an expected mathematical result.

## Pireus Connection

After the Sounio semantics are frozen, Pireus may consume the contract as an
operation node with explicit dataflow:

```text
input a -----------+
                   |
input b -> XOR permutation -> twist application -> horizontal reduction -> r[d]
                   |
contract ---------+
```

Pireus may then ask its frozen target ontologies which material realization is
legal on Darwin Xeons, Apple Silicon, or DGX. Canonical target status is not an
observation, and ontology membership is not a performance result.

## Claim Boundary

The first frozen result may establish only:

- the exact Sounio operation contract;
- equivalence of the two Sounio indexing forms for Sounio-born inputs;
- the Sounio-produced finite classification of the selected twist;
- deterministic negative witnesses and digests;
- the distinction between twist, left-square law, and group 2-cocycle law.

It may not establish:

- the estimated `~112` instruction cost;
- that `vpermps` or `vpermi2ps` realizes any row;
- AVX-512, NEON, SVE, SME, Metal, PTX, or SASS lowering quality;
- a Fano-plane explanation of row counts;
- a subquadratic algorithm for a nonseparable twist;
- cross-language parity;
- a hardware or performance claim.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN` for
`SOUNIO-XOR-CONVOLUTION-COCYCLE`.
