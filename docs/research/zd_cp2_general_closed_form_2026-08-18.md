<!-- docs:meta
topic_id: repo.docs.research.zd-cp2-general-closed-form-2026-08-18
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zd-cp2-general-closed-form-2026-08-18
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# `cp2`'s general closed form — the fibre coordinate identified

**Date:** 2026-08-18
**Status:** the scaling half is a THEOREM; the base case is MEASURED, not proved.
**Lane:** exact algebra / ZD fibers (`formal/lean4/SounioZDFiberAntisym.lean`)

## The result

For every level `m` and every label `1 ≤ W < 2^(m+1)`, with `H = 2^(m+1)`:

    cp2(m, W) = −(H−2)(H−6) + 192 · c(m, W)

    c(m, W) = Σ_{i=1}^{k−1} (−1)^(i+1) · 4^(m − p_i) · [p_i − 1, 2]₂

where `p₁ > p₂ > … > p_k` are the positions of the set bits of `W`, and the sum runs over all
of them **except the lowest**. `[n,2]₂ = (2ⁿ−1)(2^(n−1)−1)/3` is the Gaussian binomial, `= 0`
for `n < 2`.

Three immediate consequences:

* **`W` a power of two ⟹ `c = 0`.** One set bit, and it is the lowest, so the sum is empty.
  That is exactly the class on which Tier 108 discharges the four-sign law, which is why
  Tier 95's `cp2 = −(H−2)(H−6)` is correct there — and why it is incomplete elsewhere. The
  old closed form was never wrong; it was the `c = 0` stratum of this one.
* **Only bits at position ≥ 2 contribute**, since `[p−1,2]₂ = 0` for `p ≤ 1`. Hence `c ≡ 0` at
  level 2 — measured, and now explained.
* **The `192·[m−1,2]₂` of the 2026-08-10 seam note is the leading term**: when the top bit sits
  at `p₁ = m`, the first summand is `4⁰·[m−1,2]₂`.

## What is proved and what is not

**PROVED: the level scaling `c(m+1, W) = 4·c(m, W)`.** It follows from `cp2_level_recursion`
(Tier 167, `57b6cb9918`) with no new work. Substituting the ansatz into
`cp2(m+1) = 4·cp2(m) + 36 − 16H`:

    −(2H−2)(2H−6) + 192·c(m+1,W)  =  4·[−(H−2)(H−6) + 192·c(m,W)] + 36 − 16H
    −4H² + 16H − 12 + 192·c(m+1,W) =  −4H² + 16H − 12 + 768·c(m,W)

so `c(m+1,W) = 4·c(m,W)`, which is precisely the `4^(m−p_i)` factor in the formula. The whole
`m`-dependence of `c` is therefore a consequence of an already-proved theorem.

**MEASURED, NOT PROVED: the base case.** A label `W` with top bit at `p₁` first exists at level
`m = p₁`. Everything above that level is forced by the scaling, so the entire remaining content is

    c(p, 2^p + r) = [p−1, 2]₂ − 4 · c(p−1, r)        for 1 ≤ r < 2^p
    c(p, 2^p)     = 0

i.e. a transfer in the LABEL's top bit at the level where that bit first appears — a different
move from the level transfer, which is what `cp2Split` and Tier 167 handle.

## Evidence

`#eval` of the raw double sum from `P3`'s definition, against the formula, for **every** label at
levels 2, 3, 4 and 5 — 7 + 15 + 31 + 63 = 116 labels. **Zero mismatches.**

Measured `c` tables (`192·c` is the deviation from `−(H−2)(H−6)`):

| level | blocks of `W` → `c` |
|---|---|
| 2 | `1..7` → 0 |
| 3 | `1..8` → 0; `9..15` → 1 |
| 4 | `1..8` → 0; `9..15` → 4; `16` → 0; `17..24` → 7; `25..31` → 3 |
| 5 | `1..8` → 0; `9..15` → 16; `16` → 0; `17..24` → 28; `25..31` → 12; `32` → 0; `33..40` → 35; `41..47` → 19; `48` → 35; `49..56` → 7; `57..63` → 23 |

Spot checks against the formula: `W = 25` at `m = 4` has bits `{4,3,0}`, so
`4⁰·[3,2]₂ − 4¹·[2,2]₂ = 7 − 4 = 3` ✓. `W = 57` at `m = 5` has bits `{5,4,3,0}`, so
`4⁰·[4,2]₂ − 4¹·[3,2]₂ + 4²·[2,2]₂ = 35 − 28 + 16 = 23` ✓.

## Why this matters to the lane

`cp2` is one of the two coordinates of the proved transfer matrix `[[8,24],[0,4]]`. Its
RECURSION was already unconditional in the label (Tier 167); its VALUE was known only on the
powers of two. This closes the gap in measurement and reduces the proof obligation to a single
label-transfer identity.

It also makes "`cp2` is fibre-constant" (§57.49) precise: `c` is constant on blocks, and the
block structure is the binary one above — not constancy across all labels.

## What this note does NOT claim

* The base-case identity is not proved. Nothing here is a Lean theorem except the scaling, and
  that was already `cp2_level_recursion`.
* The 116-label sweep is exhaustive only for `m ≤ 5`. The formula's form is fitted to that range.
* No claim is made about `cp3`, whose transfer row remains fitted-only.
