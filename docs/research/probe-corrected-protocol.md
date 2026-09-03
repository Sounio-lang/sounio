<!-- docs:meta
topic_id: repo.docs.research.probe-corrected-protocol
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.probe-corrected-protocol
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The probe, corrected: the gap alone is a false positive — the mechanism is subspace alignment

*The first probe was built to read a POSITIVE and could not read a NEGATIVE (OPUS-4.8-EXTRA revised
protocol). The product-spectrum gap is refuted as a discriminant; the valid one is the principal-angle
alignment of consecutive dying subspaces. And the S-SSM is the worst target, not the best — its 4/8/4 is
put there by algebra. Corrected here.*

## The refutation (`mechanism_analysis.py`)
A stack of matrices each with a low-multiplicity tail (4/8/4 per factor) but with the dead directions
**rotating** between factors is the decisive control: it has the algebraic structure per-factor but the
dead subspaces do **not** compose. Measured over depth 32:

| stack | mean cos(principal angle), dying 4-subspaces | gap_dominance (T=16) | gap_dominance null P(>1) |
|---|---|---|---|
| **aligned** (same zero divisor — genuine composing annihilation) | **0.988** | 5.71 | — |
| **rotating** (different z per layer — 4/8/4 per factor, no composition) | 0.530 | **99.4** | **97%** |
| real Gaussian | 0.415 | 0.33 | 1% |

Two things this settles: (1) **`gap_dominance` is not a valid discriminant** — the rotating control's gap is
*larger* than genuine structure (99 vs 6) and exceeds 1 in 97% of nulls; against a Gaussian null it looked
significant (1% FP), against the *right* control it is noise. (2) **the principal-angle alignment separates
them cleanly** — `0.988` (genuine) vs `~0.5` baseline (rotating, real). The gap only means "composing
annihilation" if the dead subspaces are **aligned**; the spectrum alone cannot tell composition from
rotation, which are opposite conclusions. So the corrected probe measures **`subspace_alignment` first**.

## gap(T): a curve, not a point
`gap(T)` for the genuine aligned stack stays high and roughly flat (the gap *survives* composition) while
the real Gaussian decays (`0.93→0.33` — the gap dies under composition); the rotating control spuriously
inflates. So "the gap survives/grows with T" separates aligned+rotating from real, but only alignment
separates aligned from rotating. Report the **gap(T) curve alongside the alignment**, never a single-T
label.

## Corrected protocol (in order)
1. **Principal angles first** (`subspace_alignment`) — the mechanism; it disambiguates a UNIFORM-SLIDE
   (no dead subspace *vs* dead subspaces that rotate — opposite meanings the spectrum conflates).
2. **Nulls, reported as a distribution** of the discriminant, not a label: random Gaussian, shuffled
   weights, untrained init, and the **matched-4/8/4-but-rotating** control (the one that breaks the naive
   gap). If the signature is already at init, it is architecture, not learning.
3. **Primary target: a NON-sedenion model** (LSTM, S4/Mamba, linear attention, vanilla RNN) — where 4/8/4
   has no reason to appear, so its appearance would be discovery. The **S-SSM is only a declared positive
   control** (its 4/8/4 is architectural; reading it back is the compiler working, not evidence).
4. **`gap(T)` over hundreds of input sequences** (∂h_T/∂h_0 depends on the whole path) — report consistency
   and the scaling curve; a verdict over "a few paths" is an uncharacterized small sample.
5. **Only then, the link to performance**: does the alignment/gap predict a loss plateau, long-sequence
   degradation, or specific unlearned examples? Without that link it is a spectrum shape, not a diagnostic.

## Status
The instrument now measures the mechanism (alignment), carries a null (the rotating control), reads the
gap as a curve in `T`, and is aimed at the informative target. `make_forward` should be written for the
LSTM / S4 first; the S-SSM second, labeled calibration. Harnesses `mechanism_analysis.py` (the refutation)
and `probe_jacobian_spectrum.py` (`subspace_alignment`, `gap_vs_T`).
