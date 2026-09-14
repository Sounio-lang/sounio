# Garden: Pireus Operator Genesis Bilinear v2

Status: `GARDEN`

Founder direction preserved:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

Operator Genesis GL4 v1 exhausted 16 diagonal bilinear phases and discovered
four declared semantic classes. v2 expands that grammar to every bilinear
phase on `F2^4`, but it must not pay for the expansion with a weaker
equivalence test or a brute-force compatibility theater.

The v2 question is fixed before an executable or expected result exists:

> Among all 65536 bilinear phases, does Sounio generate a declared semantic
> class that is outside the three-member corpus and outside every class already
> represented by the 16-member v1 grammar, after all `GL(4,2)` basis changes,
> basis-sign gauges, and optional operand exchange?

Sounio must answer first. This Garden fixes the finite grammar, the exact gauge
quotient, the affine stabilizer reduction, the class construction, the corpus
test, the selection objective, and the permitted vocabulary.

## Authority order

The only admissible order remains:

```text
GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY
```

The first Sounio executable contains no expected stabilizer size, orbit count,
class partition, selected matrix, structural signature, witness, or digest.
Lean, Koka, C++, Haskell, Xeon, DGX, Apple Silicon, U250, and external LLMs may
compare, prove, or measure only after a hash-frozen Sounio result exists.

Python and Rust are forbidden as generators, oracles, validators, freeze
producers, or parity legs. No disposable language may supply a golden.

## Expanded operator grammar

Let `V=F2^4`. Encode every binary `4x4` matrix `B` by 16 row-major bits. For
`i,j in V`, define

```text
b_B(i,j)     = i^T B j                         in F2
sigma_B(i,j) = cd_sigma(i,j,4) * (-1)^b_B(i,j)
r_B[d]       = sum_i sigma_B(i,i XOR d) * a[i] * b[i XOR d].
```

Every 16-bit encoding is a grammar member, whether or not `B` is invertible.
The grammar therefore contains exactly `2^16=65536` operators. Sounio scans
all encodings in ascending order.

The v1 family is the diagonal subgrammar

```text
B = diag(mask), mask in [0,15].
```

v2 strictly contains v1 and must identify which v2 classes already contain a
v1 diagonal member.

The imported Cayley-Dickson sign table remains an assumed base twist. The
grammar is exhaustive only over bilinear phases on that base, not over all
`2^256` sign tables.

## Gauge quotient of bilinear phases

A basis-sign gauge is a function `q:V->F2` with `q(0)=0`, acting on a sign-bit
table `f` by

```text
f^q(i,j) = f(i,j) XOR q(i) XOR q(j) XOR q(i XOR j).
```

There are `2^15=32768` such functions. Their coboundary image has dimension
11 because the kernel is the 4-dimensional space of linear characters.

Let `L` be the 16-dimensional space of bilinear tables `b_B`. Its intersection
with the coboundary image consists exactly of alternating matrices over `F2`,
meaning symmetric matrices with zero diagonal. This intersection has dimension
6:

```text
L intersect Coboundary = { b_A : A=A^T and diag(A)=0 }.
```

Indeed, the coboundary of a Boolean quadratic function is alternating
bilinear, and every alternating bilinear form is obtained this way. Linear
terms are precisely the coboundary kernel.

Therefore the 65536 matrices form

```text
2^(16-6) = 1024
```

exact gauge classes, each containing exactly `2^6=64` matrices.

## Quadratic class code

The complete gauge-class invariant of `B` is

```text
Q_B(x) = b_B(x,x) = x^T B x.
```

It has four linear coefficients `B_rr` and six quadratic coefficients
`B_rs XOR B_sr` for `r<s`, hence 10 bits. The kernel of `B -> Q_B` is exactly
the six-dimensional alternating subspace above.

v2 serializes a quadratic code in this order:

```text
bits 0..3: B_00, B_11, B_22, B_33
bits 4..9: (01), (02), (03), (12), (13), (23)
```

Sounio must scan all 65536 `B`, map each to a code in `[0,1023]`, prove every
bucket contains 64 matrices, and retain the smallest `B` encoding in every
bucket.

A deterministic representative `R(Q)` sets `B_rr` from bits 0..3, sets
`B_rs` from the corresponding pair bit for `r<s`, and sets `B_sr=0`. Sounio
must verify `Q_R(Q)=Q` for all 1024 codes.

## Exact affine stabilizer of the family

Let `c` be the Cayley-Dickson sign-bit table. Let an action `A=(M,s)` consist
of `M in GL(4,2)` and optional operand swap `s`:

```text
A.f(i,j) = f(Mi,Mj)       when s=0
A.f(i,j) = f(Mj,Mi)       when s=1.
```

There are `20160*2=40320` such actions. Define the base displacement

```text
h_A = A.c XOR c.
```

An action maps some bilinear-family member to another bilinear-family member
if and only if

```text
h_A belongs to L + Coboundary.
```

This condition is independent of the chosen `B`. Necessity follows because
the difference of two family members and the transformed bilinear phase are
in `L`; sufficiency follows by absorbing the coboundary remainder into a basis
sign gauge.

Sounio decides membership without an external solver:

1. extract the 10-bit diagonal code `D_A(x)=h_A(x,x)`;
2. construct the deterministic bilinear representative `R(D_A)`;
3. form `h_A XOR b_R(D_A)`;
4. apply the exact 11-tree-cell gauge normalizer from v1;
5. admit the action exactly when the normalized remainder is the zero table.

For every admitted action, the induced affine map on the 1024 gauge classes is

```text
Q -> pullback_M(Q) XOR D_A
pullback_M(Q)(x) = Q(Mx).
```

Operand swap does not alter `Q_B(x)=b_B(x,x)`; its effect is already present
in `D_A` through the base displacement.

The admitted actions are the stabilizer of the affine family `c+L+Coboundary`
inside `GL(4,2) x C2`. Sounio must enumerate them extensionally, require the
identity, verify inverse admission, and verify on every one of the 1024 codes
that the computed inverse affine action restores the input.

No expected stabilizer cardinality is fixed in this Garden.

## Exact semantic classes

Sounio initializes 1024 singleton quadratic codes and unions

```text
Q ~ pullback_M(Q) XOR D_A
```

for every admitted action and every `Q`. The resulting connected components
are exactly the declared semantic classes of the full 65536-member family
under `GL(4,2)`, all basis-sign gauges, and operand exchange.

Each quadratic code represents 64 raw matrices. For every class Sounio emits:

- the smallest quadratic code;
- the smallest raw `B` encoding across the class;
- the number of quadratic codes;
- the number of raw operators, equal to `64 * quadratic_codes`;
- the invariant structural signature defined below;
- whether it intersects the v1 diagonal grammar;
- whether it intersects each frozen corpus orbit.

The quadratic-code class sizes must sum to 1024 and the raw-operator class
sizes must sum to 65536.

## Exact corpus incidence

The corpus is ordered and unchanged:

1. untwisted XOR, all signs positive;
2. the live Cayley-Dickson-16 table `c`;
3. the diagonal bicharacter `(-1)^parity(i AND j)`.

For corpus table `g`, action `A=(M,s)`, and displacement

```text
h_(A,g) = A.c XOR g,
```

Sounio uses the same exact `L+Coboundary` membership test. When membership
holds with quadratic code `D_(A,g)`, the unique source family class that the
action sends to `g` is

```text
Q = pullback_(M^-1)(D_(A,g)).
```

Sounio enumerates all 40320 actions for every corpus member, marks every
incident quadratic code, and then requires incidence to be constant on each
semantic class. Fingerprints or positive distances may not substitute for
this exact equality test.

## Structural invariant and selection

For a sign-bit table `f`, define the declared invariant signature

```text
square_negative_count = count_i f(i,i)
commutator_defects     = count_(i,j) [f(i,j) != f(j,i)]
associator_defects     = count_(i,j,k)
  [f(i,j) XOR f(i XOR j,k) != f(j,k) XOR f(i,j XOR k)].
```

All three counts are invariant under basis relabeling, basis-sign gauge, and
operand exchange. Sounio must verify that the signature is constant across
every quadratic code in each computed class.

Every bilinear phase is a group 2-cocycle:

```text
b_B(i,j) XOR b_B(i XOR j,k)
XOR b_B(j,k) XOR b_B(i,j XOR k) = 0.
```

Consequently the associator obstruction of `c XOR b_B` is the obstruction of
`c`. Sounio must verify the cocycle identity for all 1024 deterministic
representatives and verify that all candidate associator-defect counts equal
the base count. This is a property of the declared grammar, not a claim that
the resulting product is associative.

For ranking only, define the distance from a candidate signature to a corpus
signature as the lexicographic tuple

```text
(abs(associator_delta), abs(commutator_delta), abs(square_delta)).
```

The candidate's nearest-corpus tuple is the lexicographic minimum across the
three corpus members. The selected class must:

1. have no exact corpus incidence;
2. contain no v1 diagonal member;
3. maximize its nearest-corpus tuple lexicographically;
4. break ties by the smallest raw `B` encoding.

This tuple is an invariant structural separation objective. It is not an orbit
Hamming distance, an algebra-isomorphism classifier, or a prior-art score.

## Permitted novelty statement

If and only if the 65536-member grammar, 1024-class gauge quotient, full
40320-action stabilizer scan, affine inverse checks, semantic partition,
corpus incidence, v1 incidence, invariant checks, and search all complete, the
selected result may emit:

```text
expanded_bilinear_grammar_exhausted=true
bilinear_gauge_quotient_exact=true
declared_family_equivalence_exact=true
relative_semantic_novelty=true
relative_grammar_extension_novelty=true
relative_algebraic_novelty=false
algebra_isomorphism_complete=false
```

`relative_grammar_extension_novelty` means only that the selected declared
semantic class is absent from the three-member corpus and from all classes
represented by the v1 diagonal grammar under the declared equivalence.

Global, historical, priority, algorithmic, material, scientific, and complete
algebraic novelty remain false. `CLAIM_READY` remains closed.

## Required first-execution certificates

The first Sounio executable must emit and hash at least:

- all 65536 `B` encodings scanned exactly once;
- all 1024 quadratic codes with bucket size 64;
- reconstruction and bilinear-cocycle failure counts;
- all 65536 matrix encodings and 20160 invertible `GL(4,2)` matrices;
- all 40320 linear/swap actions considered;
- the unanticipated admitted stabilizer size and swap distribution;
- identity, inverse-admission, and affine-inverse failure counts;
- the complete class count, class sizes, representatives, and total sums;
- exact v1-diagonal and corpus incidence per class;
- invariant-signature consistency per class;
- the selected `B`, quadratic code, class, signature, nearest corpus tuple,
  and a structural differing-field witness;
- nonzero digests for grammar, quotient, matrix universe, stabilizer, class
  partition, corpus incidence, search, selected operator, and receipt;
- every bounded novelty flag and every forbidden promotion flag.

## Fail-closed refusals

The executable and final gate must refuse:

- fewer or more than 65536 bilinear matrices;
- a quadratic bucket other than 64 or a quotient count other than 1024;
- a failed representative reconstruction or bilinear cocycle identity;
- a `GL(4,2)` scan other than 65536 encodings and 20160 matrices;
- an admitted action whose remainder is not an exact coboundary;
- a missing identity, inverse action, or failed affine inverse;
- class totals other than 1024 quadratic codes and 65536 raw operators;
- class-varying structural signatures or corpus incidence;
- a selected class already present in v1 or the corpus;
- the ranking tuple relabeled as orbit Hamming or algebraic distance;
- complete algebraic, historical, scientific, material, algorithmic, global,
  or priority novelty;
- parity before a hash-frozen Sounio result;
- C++, Lean, Koka, Haskell, hardware, or an LLM promoted to semantic producer;
- a Python or Rust oracle before process launch;
- policy absence, policy timeout, malformed receipts, or an unscoped waiver.

## Falsifiers

The v2 reduction and every downstream claim are demoted if any of these occurs:

- two `B` matrices with the same `Q_B` are not gauge-equivalent;
- a quadratic bucket has cardinality other than 64;
- an equivalence between family members uses an action rejected by the
  `L+Coboundary` membership test;
- an admitted affine map or its inverse disagrees with direct table action;
- two quadratic codes in one class have different invariant signatures;
- corpus incidence is not constant on a computed class;
- a v1 diagonal member reaches the selected class;
- replay changes the class partition or selected `B`;
- a serialized semantic input changes without its owning digest changing;
- any parity leg supplies or rewrites an expected value;
- the Guardian launches a forbidden oracle before denial.

## Material boundary

Only after `SEMANTICS_FROZEN` may the unchanged selected operator be lowered to
the canonical Xeon, DGX, Apple Silicon, and dual-U250 targets. Those targets
may measure cost and realization; they cannot change the class partition,
selection objective, or Sounio meaning.

## Exit criteria

`SOUNIO_EXECUTABLE` opens only after Sounio emits the complete unanticipated
v2 result and every positive/negative self-check passes without a frozen
matcher.

`SEMANTICS_FROZEN` opens only afterward, when source, imported base twist,
Garden, v1 parent semantics, result, command, toolchain, hardware, Guardian
decisions, and transcript are hash-bound.

`PARITY_OPEN` requires a separate Guardian admission naming that exact
semantics hash. `CLAIM_READY` remains closed in v2.
