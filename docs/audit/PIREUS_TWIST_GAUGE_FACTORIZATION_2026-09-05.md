<!-- docs:meta
topic_id: repo.docs.audit.pireus-twist-gauge-factorization-2026-09-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pireus-twist-gauge-factorization-2026-09-05
-->

# Pireus twist gauge factorization audit

```text
Semantic-Lane-ID: pireus-twist-gauge-search-20260905
Owner: codex
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER, SOUNIO-SECOND-ORDER-COMPILATION
Intent-Preserved: the Cayley-Dickson sign and multiplication order remain part
  of the operator rather than being erased to obtain a familiar transform
Transformation: test whether three independent sign gauges can turn the
  Cayley-Dickson twisted XOR convolution into pure XOR convolution
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: for bits 1 through 4 under Convention X, the tested
  three-gauge factorization is inconsistent over GF(2)
Claims-Forbidden: no general lower bound; no refusal of every sub-quadratic
  algorithm; no claim that every possible transform factorization was tested
Assumptions: signs are restricted to plus or minus one; gauges depend only on
  i, j, and i xor j respectively; cd_sign implements Convention X
Write-Set: Garden seed, Sounio authority, frozen values, receipt, gate, audit
Read-Set: existing Pireus operator contracts and cross-architecture candidates
Positive-Witness: deterministic elimination ranks plus a four-sign XOR
  rectangle whose product is negative at every tested bit width
Negative-Witness: changing the bits=1 Cayley-Dickson base sign removes the
  required contradiction and makes the authority executable refuse itself
Acceptance-Gate: scripts/ci/pireus_twist_gauge_factorization_gate.sh
Integration-Target: origin/main
Authoritative-Only-If: Sounio produces the first result and frozen semantics;
  reviews and other languages remain non-authoritative
```

## Result

For sign bits `A_i`, `B_j`, and `C_d`, the proposed factorization requires

```text
s(i,j) = A_i xor B_j xor C_(i xor j).
```

Incremental Gaussian elimination over GF(2) rejects this system for dimensions
2, 4, 8, and 16. At dimension 16 the coefficient rank is 42; processing all
256 equations produces 120 inconsistent reductions.

The smaller certificate does not depend on trusting those aggregate counts.
For any `i`, `j`, and nonzero delta, a valid gauge forces the product of signs
at `(i,j)`, `(i xor delta,j xor delta)`, `(i,j xor delta)`, and
`(i xor delta,j)` to be positive because every gauge factor occurs twice. The
Sounio search finds `(i,j,delta)=(0,0,1)` with product `-1` at every tested bit
width. Therefore this gauge-to-WHT construction is impossible under the stated
contract.

As a positive control, the executable also constructs a nontrivial sign family
from explicit `A`, `B`, and `C` gauges and checks every corresponding rectangle.
It observes zero violations at all four dimensions.

This is a useful negative novelty result, not a complexity lower bound. Pireus
must next search a richer class in which the transform carries matrix-valued or
block-valued twist state rather than only diagonal sign gauges.
