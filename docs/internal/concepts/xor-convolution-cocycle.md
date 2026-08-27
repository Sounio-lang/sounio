# XorConvolution Twist Contract

> **Status**: Executable | **Concept-ID**: `SOUNIO-XOR-CONVOLUTION-COCYCLE`

## Authority

The canonical executable is `stdlib/algebra/xor_convolution.sio`. The frozen
semantic result is recorded in
`docs/research/xor_convolution_cocycle_semantics.md`.

Sounio is `SEMANTIC_AUTHORITY`. Lean 4, Koka, C++, Haskell, external models,
and hardware measurements cannot create or repair this operation's expected
result. Python and Rust are prohibited.

## Operation

For dimension `N = 2^bits`, a pure twist `sigma(i,j)` and vectors `a,b`, the
contract admits the bilinear product

```text
r[i XOR j] += sigma(i,j) * a[i] * b[j].
```

Its displacement-indexed form is

```text
r[d] = SUM_{i=0}^{N-1}
         sigma(i, i XOR d) * a[i] * b[i XOR d].
```

Both forms use ascending `i` accumulation for each output. This order is part
of the executable contract, not a lowering preference.

## Classification

The contract distinguishes:

- a general pure twist;
- normalization on zero indices;
- dependence on XOR displacement alone;
- rank-one sign separability;
- the left-square law for every admitted `i != 0` and every admitted `j`;
- the standard group 2-cocycle law for every admitted triple `(i,j,k)`.

The last law is exactly the associativity condition for the basis product
`e_i e_j = sigma(i,j)e_(i XOR j)`. The historical repository name "Cayley-
Dickson cocycle" for the left-square identity does not imply this standard
group-cohomological law.

## Evidence Boundary

The frozen Sounio result establishes the indexing equivalence, the selected
`bits=4` Cayley-Dickson twist table classification, deterministic witnesses,
and exact digests. It establishes no ISA lowering, instruction count,
performance result, Fano-plane explanation, transform algorithm, or
subquadratic complexity claim.

The intended later Pireus node is therefore parameterized by `twist`, not by
an unproved associative classification. A future surface spelling with
`cocycle = cd_sigma` must retain the explicit classification in its semantic
record.

## Remaining Work

The next admissible layers are:

- a Pireus operation node over the frozen contract;
- Lean 4 formal parity for the finite definitions and reindexing theorem;
- Koka effect parity for purity, allocation, and material-observation effects;
- C++ material parity after the Sounio semantics hash is bound;
- target-specific lowerings and measurements for Darwin Xeons, Apple Silicon,
  and DGX;
- research into structured transforms for nonseparable, nonassociative twists.

Parity remains closed until an explicit Loom transition opens it. Claim-ready
status remains false.
