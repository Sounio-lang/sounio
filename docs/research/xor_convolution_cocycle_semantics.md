# Frozen Semantics: XorConvolution With A Cayley-Dickson Twist

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Chain

The byte-exact Garden parent is
`docs/internal/garden/seeds/2026-08-27-xor-convolution-cocycle.md`, SHA-256
`246651b5804f8f24ddc6a8292d898db5483a7495f355c53fb1c4b50b7fb62e80`.

The first executable was committed separately after that Garden and contained
no expected count, result bit pattern, witness, or digest. The values below are
the output of that Sounio executable. The freeze matcher was added only after
the first result existed.

## Exact Contract

The admitted contract is:

```text
bits                 = 4
dimension            = 16
vector_length        = 16
twist_identity       = Cayley-Dickson cd_sigma
coefficient_domain   = {-1,+1}
accumulation_order   = ascending i for each output d
```

The direct implementation loops over every `(i,j)` pair and updates
`r[i XOR j]`. The independent horizontal implementation loops over each
displacement `d`, derives `j = i XOR d`, and reduces in ascending `i` order.

For every admitted `d`, the frozen obligation is

```text
DirectPairForm(a,b)[d]
  = SUM_{i=0}^{15} sigma(i, i XOR d) * a[i] * b[i XOR d].
```

The deterministic input vectors are generated in Sounio from bounded integer
formulas and converted to `f64`. No external input file or expected vector is
read.

## Exact Equivalence Result

Sounio produced:

```text
mismatch_count          = 0
first_mismatch          = -1
max_abs_difference_bits = 0
```

Because both implementations add the same summands in the same ascending-`i`
order, the complete results are bit-identical. The earlier research error
`1.78e-15` remains valid evidence for its own implementation and accumulation
order; it is not the frozen result of this contract.

The exact IEEE-754 `f64` result bits for `d=0..15` are:

```text
 4613118981945187609
-4611510553506841004
-4614727410383534212
-4608966312158910915
 4604258003457569030
-4604813642372634225
 4612475610569848966
 4616423571282154268
-4600193066131565794
 4615458514219146306
-4606699890268513426
-4603058993167165201
-4605968786432901330
-4614201015621893512
-4608907823852061948
-4604608933298662840
```

## Twist Classification

Sounio enumerated all 256 table cells:

```text
plus_count               = 136
minus_count              = 120
zero_count               = 0
other_count              = 0
zero_free                = true
normalized               = true
displacement_only        = false
rank_one_separable       = false
left_square              = true
group_two_cocycle        = false
associator_defect_count  = 1848
```

Here `associator_defect_count` is exactly

```text
#{(i,j,k) in [0,15]^3 :
    sigma(i,j) * sigma(i XOR j,k)
      != sigma(j,k) * sigma(i,j XOR k)}.
```

The quantifiers are explicit:

```text
left_square:
  for all i in 1..15 and all j in 0..15,
  sigma(i,j) * sigma(i, i XOR j) = -1

group_two_cocycle:
  for all i,j,k in 0..15,
  sigma(i,j) * sigma(i XOR j,k)
    = sigma(j,k) * sigma(i,j XOR k)
```

The first predicate holds. The second fails. Therefore the selected sign
function is an order-two-valued twist satisfying the repository's left-square
"cocycle" identity, but it is not a group 2-cocycle and does not define an
associative twisted group algebra at `bits=4`.

This corrects the motivating research phrase "twisted group algebra with sigma
as an order-2 cocycle." The safe mathematical description frozen here is:

```text
a nonassociative XOR-graded algebra over (Z/2Z)^4,
with a normalized {-1,+1}-valued twist and explicit associator defect.
```

## First Witnesses

The first Sounio witnesses in enumeration order are:

```text
not displacement-only:
  i=1 j=1 d=0 actual=-1 displacement_reference=+1

not rank-one separable:
  i=1 j=1 d=0 actual=-1 factor_product=+1

not a group 2-cocycle:
  i=1 j=2 k=4 output_index=7 lhs=+1 rhs=-1
```

No witness exists for a zero coefficient, failed normalization, or failed
left-square law in this finite result.

## Displacement Rows

The Sounio-produced negative counts for `d=0..15` are:

```text
15 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
```

This freezes the finite count only. It does not freeze a Fano-plane
interpretation, a theorem for other bit widths, or a lowering strategy.

## Digests

All digests below were computed inside Sounio with SHA-256 over explicit
canonical integer and `f64`-bit fields:

| Object | SHA-256 |
| --- | --- |
| contract | `486ef520df7669f360be3c531ea5fad28e0e70ea793e6d53ef6b9d34c1090856` |
| inputs | `4d4152b488cb59c4e451af1d3255d077bd5c708c88aabf46fc334a47522c1039` |
| direct result | `2aad13bb99d7f04fcc1116036ccdd2b47abee50aba7c4d8eb2801c40e0c07b6c` |
| horizontal result | `2aad13bb99d7f04fcc1116036ccdd2b47abee50aba7c4d8eb2801c40e0c07b6c` |
| twist table | `1c09a640d55cc98cfb9c51a5144dd28d1a0c5dd4ff9e8fc4a7e9d7b189cdc014` |
| properties | `f24a6f2f6c8d0c3c77440f54d6c2e683ba413c9e56d81a7f0aa08f1d84edf928` |
| witnesses | `42155cb1da7c3564bda5a83fc98d02040200095ee54902bbe5c87f96e9e3e18e` |

The direct and horizontal digests are identical, independently confirming the
per-lane bit comparison inside the same Sounio result.

## Negative Surface

Sounio passed all 20 deliberate negatives:

- invalid bits and dimension capacity;
- vector-length mismatch;
- unknown twist and coefficient domain;
- accumulation-order mismatch;
- caller-supplied destination and partner mismatch;
- missing and duplicate table cells;
- false zero-free, displacement, separability, cocycle, and associative claims;
- WHT, Fano, and material claims without their required witnesses;
- parity promotion;
- Python as a prohibited producer.

The Loom guardian also refused a pre-execution Python authority frame with
`E110 forbidden-language` while retaining stage `GARDEN`. No Python
interpreter was launched.

## Review-Only Boundary

After Sounio produced the result, xAI reviewed the indexing derivation and
predicate definitions. It found no false identity and requested explicit
quantifiers for the two displayed laws; those full domains are recorded above
and were already the loops executed by Sounio. The review produced no expected
value and is not authority evidence.

## What This Does Not Claim

The frozen result does not claim:

- `~112` emitted instructions or any other cost;
- coverage by `vpermps`, `vpermi2ps`, NEON, SVE, SME, Metal, PTX, or SASS;
- AVX-512 use;
- a Fano explanation of the row counts;
- a WHT diagonalization;
- a subquadratic algorithm;
- parity in Lean, Koka, C++, or Haskell;
- observation on Darwin, Apple Silicon, or DGX.

The next legal stage remains `SEMANTICS_FROZEN`. `PARITY_OPEN=false` and
`CLAIM_READY=false` until separately admitted by Loom.
