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
