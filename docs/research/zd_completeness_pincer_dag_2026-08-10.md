<!-- docs:meta
topic_id: repo.docs.research.zd-completeness-pincer-dag-2026-08-10
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zd-completeness-pincer-dag-2026-08-10
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The completeness pincer: every edge marked PROVED or MEASURED

Step 0 of the current plan. The claim the paper wants is: **the pair (tr A², tr A³) is a complete
invariant of the ZD-fibre geometry, ∀n.** This file is the dependency DAG, with each edge's real
status, so that nothing downstream is built on a fit.

## Edges

| # | edge | status | where |
|---|---|---|---|
| E1 | `tr(A²)` injective in the fibre invariant `g` | **PROVED ∀n** | 2-adic valuation argument (2026-08-04) |
| E2 | `A_σ = JᵀMJ`, `JJᵀ = 2I`, exact spectral halving, `rank ≤ 2^{n−2}−1` | **PROVED ∀n** | fibre antisymmetry (2026-07-31) |
| E3 | the within-fibre deviation of `tr(A³)` IGNORES `g` | **MEASURED** | §31/(III), 2026-08-04 — never proved |
| E4 | the transfer recursion `s3(m+1) = 8·s3 + 24·cp2 + …`, `cp2(m+1) = 4·cp2 + …` | **MEASURED** | §57.49 — 75 in-sample + 92 out-of-sample transitions, 0 failures |
| E5 | the base case `Δs3(j) = 1728·[j,3]₂`, `Δcp2(j) = 0` | **MEASURED** | §57.49 — exact at `m = j`, `j = 3..7` |
| E6 | obligation (ii): `resB` holds off the six lines, for `g(W) = 0` | **PROVED ∀n** | Tiers 100–108, both reference families |
| E7 | the maximal-seam exception is a mask artifact | **MEASURED** (twice recomputed) | §57.49 |
| E8 | the 168-orbit theorem (PSL(2,7) acting on fibres) | **PROVED ∀n** | 2026-07-11 |

## What that means

Three measured edges sit in the chain: **E3, E4, E5**. Fable's count is confirmed against the tree.

- **E3 is the dangerous one.** If the within-fibre deviation does not ignore `g`, the reference pairs
  do not represent their classes and E4/E5 are about the wrong objects. It must be closed or
  declared. Open question, recorded here rather than assumed: is E3 a consequence of E6 (which
  covers exactly `g(W) = 0`) plus the orbit action of E8, or is it the still-open V1 in disguise?
- **E4 and E5 are the two lemmas that make the headline true**, and Fable's reduction applies to
  both: since the inhomogeneity is label-independent it cancels in every within-fibre difference, so
  what must be proved is only the HOMOGENEOUS pair
  `Δcp2(m+1) = 4·Δcp2(m)` and `Δs3(m+1) = 8·Δs3(m) + 24·Δcp2(m)`,
  plus the base. The 7-dimensional affine system, the closed forms in `H`, and `cp3` are discovery
  scaffolding and are NOT proof targets.

## The `8 = 2³` remark, recorded so it is not lost

A cubic trace under a Cayley–Dickson doubling whose folding map satisfies `JJᵀ = 2I` acquires `2³`.
That is where the proof of the `s3` line should come from, and it routes through E2 — which promotes
E2 from infrastructure to load-bearing.

## Status of the deviation law itself

NOT proved. `D[tri3] = 1728·8^(m−j)·[j,3]₂` rests on E4 + E5, both measured. The Lean development
proves its supporting obligation (E6), not the law.

## Step 1, first result — and a null control I had to run against myself

The homogeneous pair verified 206/206 on within-fibre pairs. **That number is weaker than it looks,
and the check that showed it is the kind this lane exists to force.**

Within a fibre, `Δcp2 = 0` in 95/95 pairs tested — `cp2` is fibre-constant. So:

- the line `Δcp2(m+1) = 4·Δcp2(m)` is **VACUOUS** on within-fibre pairs: it reads `0 = 0`;
- the `24·Δcp2` term **never fires** within a fibre, so the second line is not
  `Δs3(m+1) = 8·Δs3(m) + 24·Δcp2(m)` there but simply `Δs3(m+1) = 8·Δs3(m)`.

So E4 was one edge and is really two, of very different kinds:

| | content | status |
|---|---|---|
| **E4a** | `cp2` is fibre-constant, hence `Δcp2 = 0` | **assembly-reachable now** |
| **E4b** | `Δs3(m+1) = 8·Δs3(m)` within a fibre | **the open lemma** |

**E4a is closer than the DAG said.** `cp2_count` (Tier 95) proves `cp2 = −(H−2)(H−6)` on the `g = 0`
class CONDITIONAL on the four-sign law for its summand — and that law is exactly what Tiers 96–108
established for the reference labels: the interior through `cp2_summand_core` + `P1_mul_P3_mask` +
Bridge 1 + `starP_all_{octonion,pow2}_labels`, and the borders directly in Tiers 97–98. Discharging
`cp2_count`'s hypothesis is assembly, not new mathematics, and it turns `Δcp2 = 0` on the reference
class from measurement into a theorem.

**E4b is where the mathematics is**, and Fable's structural remark is the lead: `8 = 2³` on a cubic
trace under a doubling whose folding map satisfies `JJᵀ = 2I`. That routes through E2.

## E4b is not a new lemma — it is §57.50's obligation (i), on differences

Chasing E4b to its floor: `tri3_level_transfer` is already a THEOREM (Tier 90) —

    tri3 at level m+1  =  tri3 at level m  +  3·T1 + 3·T2 + T3

with `T1, T2, T3` the three ε-weighted orthant sums, themselves theorems. Taking within-fibre
differences kills the label-independent inhomogeneity, so what E4b needs is exactly

    3·ΔT1 + 3·ΔT2 + ΔT3  =  7·Δs3 + 24·Δcp2          [95/95 within-fibre pairs, m = 3,4,5]

which is §57.50's obligation (i) restricted to differences. So the DAG collapses further than the
last revision said: **E4b is not an independent lemma**, it is the evaluation of the ε-sums'
combination, and Tier 90 already supplies everything except that evaluation.

### The DAG, current

    E1, E2, E6, E8                    PROVED
    E4a  (cp2 fibre-constant)          assembly of Tiers 95–108 — not yet written
    E4b  = obligation (i)              THE open lemma: evaluate 3T1 + 3T2 + T3
    E5   (base case at m = j)          open, and where the q-binomial's combinatorial content sits
    E3   (deviation ignores g)         open, and the dangerous one

Three open, two of them (E4b, E5) being the two Fable named, and E3 being the one that decides
whether the other two are about the right objects.
