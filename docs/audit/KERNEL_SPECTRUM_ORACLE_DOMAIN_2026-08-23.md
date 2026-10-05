<!-- docs:meta
topic_id: repo.docs.audit.kernel-spectrum-oracle-domain-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.kernel-spectrum-oracle-domain-2026-08-23
-->

# The numbers in #1466 are right. The reason given for them is not.

**Date:** 2026-08-23
**Status:** corrective. #1466 stays merged; no published number changes.

## The objection

`docs/research/cayley_dickson_zd_kernel_spectrum_2026-07-26.md:155` says the
oracle

> computes `dim ker(L_u)` by rank over a prime field (entries are `0, ±1`, so
> the rank equals the rational rank)

and `scripts/research/cd_zd_kernel_spectrum.py:70` implements exactly that,
with `pow(R[r][c], P - 2, P)` and `P = 2^31 - 1`.

**The parenthetical is false.** An integer matrix with entries `0, ±1` can have
a minor that is nonzero over Q and divisible by P; the rank then drops mod P.
Entry magnitude bounds nothing about the determinant.

Nor does the size rescue it. For a 16×16 matrix with `|a_ij| ≤ 1` the Hadamard
bound is `16^8 = 2^32 = 4,294,967,296` against `P = 2,147,483,647` — a factor of
exactly **2.00**. A single minor can be twice P, so nothing forbids one being an
exact multiple.

The doc also describes this path as "exact rational arithmetic". It is modular.

## The recomputation

`scripts/research/cd_zd_kernel_spectrum_exact_check.py` redoes the whole n = 16
structure with `fractions.Fraction` — no modulus in the rank, none in the kernel
identity, none in the independence search — and diffs it against the modular
oracle.

| claim | modular F_P | exact Q |
|---|---:|---:|
| index pairs with `dim ker = 4` | 42 | **42** |
| distinct kernels | 42 | **42** |
| maximum linearly independent kernels | 3 | **3** |

**Every published number survives.** No claim in #1466 changes.

## What this does and does not settle

It settles the numbers. It does not settle the justification, and the two are
not the same thing.

The implication "entries are `0, ±1`, therefore rank over F_P equals rank over
Q" is false, and it would still be false if the numbers had agreed in every
dimension. What actually happened is narrower and luckier: none of the 56
relevant minors at n = 16 happened to be a multiple of this particular prime.
That is a fact about `2^31 - 1` and these 56 matrices, not a theorem about
0,±1 matrices.

So the correct form of the claim is:

- the spectrum, its maximum and the degeneracy rule stand on the combinatorial
  derivation, and never depended on the oracle's arithmetic;
- the three structural numbers at n = 16 are now verified **over Q**, and that
  is what should be cited;
- the modular path stays, as a fast check, and is no longer the authority.

If some property of these operators does guarantee the modular rank equals the
rational rank, it is a lemma and has to be proved. It cannot be inferred from
the alphabet of the entries.

## The architectural point

An oracle's arithmetic domain is a claim-bearing property:

```
mathematical claim -> oracle arithmetic domain -> proved correspondence -> evidence
```

A modular computation does not acquire the identity of a rational computation by
being described as one. This repository already refuses a green that has never
been red; it should equally refuse an "exact" that was computed mod P.

That belongs in the same family as the inert-surface ratchet landed today: a
thing that claims a property must be able to be refuted on it.

## Addendum, same day: n ≤ 128 certified exactly

The gap above is closed. `scripts/research/cd_zd_kernel_spectrum_exact_certificate.py`
certifies every kernel dimension for n = 8, 16, 32, 64, 128 — all 5,208 index
pairs — with integer arithmetic only.

Redoing the elimination over Q was not the way. At n = 128 there are 4,032
pairs and Fraction numerators explode; and no single-prime result can be made
safe by a bound, since the Hadamard bound for a 128×128 matrix with
`|a_ij| ≤ 2` is astronomically beyond any word-sized prime. The certificate
uses a fact that holds for EVERY prime instead:

```
rank over F_P  ≤  rank over Q        (a Q-dependency stays dependent mod P;
                                      the converse can fail)
so   dim ker over Q  ≤  dim ker over F_P
```

The modular dimension is therefore an **upper bound for free, with no
assumption about the entries at all** — which is precisely what the false
parenthetical was trying and failing to supply. Pinning the value then needs
only a lower bound, and that is exhibited rather than computed:

1. take the modular kernel basis,
2. lift each vector to integers in the symmetric range,
3. verify `M · v == 0` **exactly over Z**, no modulus,
4. verify the lifted vectors are independent over Q by integer-only Bareiss
   elimination.

Step 3 gives `dim ker_Q ≥ k`; the inequality above gives `≤ k`. Equality
follows, and every step is integer arithmetic.

Result — the published spectrum, certified:

| n | dim ker multiset | pairs | certified |
|---:|---|---:|---|
| 8 | {0: 12} | 12 | yes |
| 16 | {0: 14, 4: 42} | 56 | yes |
| 32 | {0: 30, 4: 84, 12: 126} | 240 | yes |
| 64 | {0: 62, 4: 252, 12: 168, 20: 168, 28: 342} | 992 | yes |
| 128 | {0: 126, 4: 684, 12: 504, 20: 336, 28: 336, 36: 336, 44: 336, 52: 504, 60: 870} | 4032 | yes |

Not one pair failed certification. The spectrum `{0} ∪ {4 + 8i}` and its
maximum `n/2 − 4` now rest on integer arithmetic across the whole published
range, not on a claim about the alphabet of the entries.

## Second addendum: the STRUCTURE, exactly, for n = 32, 64 and 128

`scripts/research/cd_zd_kernel_structure_exact.py`. Each of the three structure
claims fails in a DIFFERENT direction under a modular computation, and two of
them turn out to need no extra work at all:

**Distinct kernels — the modular count is a lower bound, and it is tight.**
Equal over ℚ implies equal modular key, because the lifted modular basis is (by
the certificate above) an exact basis of `ker_ℚ`, so equal ℚ-spaces reduce to
equal `F_P`-spaces. Distinct keys therefore mean distinct over ℚ, and the only
thing left to check is whether any key bucket secretly holds two different
ℚ-subspaces. Rank test, integers only. **Zero non-uniform buckets in every
dimension.**

**Clique — already exact, and nobody had noticed.** `zero_pair` sums four ±1
table entries, so every component satisfies `|v_k| ≤ 4 < P`, and `v % P == 0`
is *equivalent* to `v == 0` over ℤ. The modular test was never approximate here.

**Maximum independent set — the modular value is a lower bound.** Independent
mod `P` implies independent over ℚ, so confirming a maximum requires showing no
set of size `max+1` is independent over ℚ. That is genuine combinatorial search
and is the one claim with no shortcut; it is still running for `n = 32` and is
not reported here.

| n | pairs | `ker = 0` | `ker > 0` | distinct kernels | non-uniform buckets | clique |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 56 | 14 | 42 | **42** | 0 | 2 |
| 32 | 240 | 30 | 210 | **210** | 0 | 3 |
| 64 | 992 | 62 | 930 | **930** | 0 | 3 |
| 128 | 4032 | 126 | 3906 | **3906** | 0 | 3 |

Three facts the published document does not state:

1. **The map pair → kernel is injective** on non-degenerate pairs, in every
   dimension measured. Two *primitives* share a subspace (`e_a + e_b` and
   `e_a − e_b`); two *pairs* never do. The document observes this at `n = 16`
   and treats it as a remark; it holds across all 5,208 pairs.
2. **The clique rises from 2 to 3 and stops.** `n = 16` gives 2; `n = 32`, 64
   and 128 all give 3. The document only speaks of `n = 16`, so this is new
   ground rather than verification.
3. **`ker = 0` count is exactly `n − 2`** in every dimension, which confirms the
   degeneracy rule `c₊ = 0 ⟺ b′ = 0 ∨ a = b′` by an independent count.

## Not claimed

- The maximum-independent-set value at `n ≥ 32` is NOT established. The modular
  search gives a lower bound and the exact confirmation is still running.
- The certificate covers the kernel DIMENSIONS. The n = 16 structure numbers —
  42 distinct kernels, clique 2, max 3 independent — were verified separately
  over Q by `cd_zd_kernel_spectrum_exact_check.py`; the analogous structure at
  n ≥ 32 was not computed and is not claimed.
- The false parenthetical is still in the merged document text and still needs
  removing. Certifying the numbers does not repair the sentence.
- The exact check is Python, matching the oracle it verifies. Porting the
  science path to Sounio is a separate and larger piece of work; introducing a
  second language here would have been the drift, not the check.
