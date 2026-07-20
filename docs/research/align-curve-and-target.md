<!-- docs:meta
topic_id: repo.docs.research.align-curve-and-target
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.align-curve-and-target
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# align(k) as a curve resolves the low-rank confounder; the target is a dense RNN, not S4/Mamba

*Recorded verdict first: the product-spectrum gap is **dead** as a discriminant — the rotating control
(4/8/4 per factor, dead subspaces rotating) killed it (gap_dominance 99 > genuine 5.7, null P(>1)=97%). The
**principal-angle alignment** is what survived. This adds the confounder the three nulls missed and the
target correction. Implements OPUS-4.8-EXTRA.*

## The missing null: low effective rank
Alignment ≈ 1 has two readings: (A) a shared **dead** subspace (annihilation), or (B) **low effective
rank** — signal in a few directions, the weak complement shared trivially, so any layer's bottom subspace
lies in it. Low rank is the most documented property of trained nets, and the existing nulls miss it:
shuffled weights and untrained init are both **full-rank** → low alignment → exactly the contrast one would
misread as structure.

## The fix needs no new null: read align(k) as a curve, find the shoulder (`align_curve.py`)
Sweep `k` and look for the **shoulder** (the drop in `align(k)` above the `√(k/d)` baseline). Validated on
three depth-12 stacks:

| stack | shoulder at k | reading |
|---|---|---|
| aligned near-ZD sedenions (dead≈4) | **k=4** (align 0.99→0.85, peak ≫ baseline 0.50) | **annihilation** — small dead subspace, healthy bulk above |
| shared-complement low rank (r=6, dead≈10) | k=10 | **low effective rank** — shoulder only at large k |
| real Gaussian | none (tracks baseline) | **nothing** |

The **shoulder position is the dead-subspace dimension** — a small-k shoulder with a healthy bulk is
annihilation; a large-k shoulder is low rank; a flat baseline is magnitude. This also dispenses with
choosing `k`: 𝕊's "4" does not translate to width 512, and the datum becomes *is there a small-k shoulder*
and *where*, not a preset `k`.

## The target: LSTM / GRU / dense RNN — not S4/Mamba
S4 and Mamba use a **diagonal** (or diagonal-plus-low-rank) state matrix for the associative scan: S4's
`∂h_{t+1}/∂h_t = Ā` is the *same* matrix each step; Mamba's `Ā_t = exp(Δ_t A)` with `A` diagonal changes
eigenvalues but **keeps eigenvectors**. So the singular subspaces coincide at every step and **alignment = 1
by architecture, not learning** — the same circularity as the S-SSM, subtler. **Point at a dense RNN
(LSTM/GRU):** input-dependent dense transition, per-step Jacobians genuinely distinct, nothing forcing
alignment — and LSTM vanishing gradient is known but conceptually unresolved ("magnitude, or subspace
death?"), so the question has real content and an unknown answer there. `lstm_probe.py` extracts the
per-step state Jacobians `∂[h;c]_{t+1}/∂[h;c]_t` for `align_curve`.

**Feedforward caveat:** for ResNet/transformer, `J_l = I + F'_l`; the residual anchors every layer in the
same basis and inflates alignment by architecture — measure the **branches** `F'_l = J_l − I`, not the full
`J_l`.

## Final protocol
1. `align(k)` as a curve — find the shoulder, not a value.
2. Nulls as a distribution: rotating, shuffled, untrained-init, **and planted-low-rank** (the decisive one).
3. Primary target: **dense RNN (LSTM/GRU)**. S4/Mamba **disqualified** (diagonal). S-SSM only as declared
   positive control.
4. `gap(T)` as a curve over hundreds of sequences (`∂h_T/∂h_0` depends on the whole path).
5. The performance link last — does the shoulder predict a loss plateau, long-sequence degradation, or
   specific unlearned examples? Only then is it a diagnostic, not a spectrum shape.

Start with an LSTM trained on a long-dependency task, where vanishing gradient is known to operate — if
subspace death exists anywhere, it is there. Harnesses `align_curve.py`, `lstm_probe.py`.
