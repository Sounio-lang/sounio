<!-- docs:meta
topic_id: repo.docs.research.probe-result-deep-ffn
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.probe-result-deep-ffn
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Probe result — deep residual FFN, the confound-free target

**Status: NEGATIVE.** On the one target free of the shared-backbone confound — a deep residual MLP with
**distinct weights per layer**, trained to 96 % on a genuinely nonlinear task — the branch-Jacobian
subspace-alignment signature of composition annihilation is **absent**: trained is indistinguishable from
untrained init *and* from the orientation-scramble null at every rank `k`.

This is the terminal result of the Jacobian-probe line. It closes the empirical question the H=256 LSTM
left open.

## Why this target, and why it is the decisive one

The probe asks whether a trained network's per-step (or per-layer) Jacobians share a *dying subspace* —
consecutive maps annihilating the *same* low-dim directions, the finite-dimensional shadow of a sedenion
zero-divisor `x·y = 0`. Measured as principal-angle alignment of the bottom-`k` right singular subspaces
of consecutive Jacobians, as a **curve** `align(k)` (a small-`k` shoulder with a healthy bulk = genuine
subspace death; high-to-large-`k` = mere low rank; flat `√(k/d)` = nothing).

Every earlier target was confounded — the alignment was written into the architecture before any training:

- **S4 / Mamba** — diagonal `Ā` shares eigenvectors across steps by construction → `align ≡ 1`. Disqualified.
- **LSTM / dense RNN** — shares one recurrent matrix `W_hh` every step. The H=256 run
  ([`PROBE-RESULT-h256-scale.md`](PROBE-RESULT-h256-scale.md)) found `align(k) ≈ 1.0` at *all* `k`, but the
  init/untrained control was equally ≈ 1.0 → **architectural**, not learned. The shared backbone forces
  consecutive Jacobians to share singular subspaces regardless of training.

The only clean target has **distinct weights at every layer**, so there is no shared matrix to manufacture
alignment. A residual net `z_{l+1} = z_l + F_l(z_l)` fits: its layer Jacobian is `J_l = I + F'_l`, and the
residual identity `I` anchors *all* `J_l` to the same basis — so we must probe the **branch**
`F'_l = J_l − I`, never the full `J_l` (probing `J_l` would recover the identity's trivial alignment).

## The clean-target proof (`deep_ffn_probe.py`)

Untrained deep residual FFN, distinct random weights per layer, branch `F'_l` alignment:

```
  k              1     2     4     8    16    32
  baseline    0.12  0.18  0.25  0.35  0.50  0.71     √(k/d)
  INIT F'_l   0.10  0.15  0.21  0.30  0.43  0.64
  max excess over baseline: −0.02
```

Init sits **at (slightly below) baseline** at every `k` — distinct weights ⇒ **no architectural alignment**.
Contrast the LSTM init (≈ 1.0 everywhere). The target is clean: any alignment above baseline in a *trained*
net would be a genuine learned signal, not the architecture.

## Trained result (`deep_ffn_train.py`)

ResMLP (`W=96`, `L=8`, embed→8 residual `tanh` blocks→readout), distinct weights per layer, trained with
Adam to **96 % test accuracy** on `y = sign(x₀x₁ + x₂x₃ − x₄x₅)` (a genuinely nonlinear 6-dim signal in
`d=64` that rewards depth — a linear readout scores chance, confirmed).

```
ResMLP branch F'_l align(k)   W=96, L=8, acc=0.96:
  k              1     2     4     6     8    16    32    48
  baseline    0.10  0.14  0.20  0.25  0.29  0.41  0.58  0.71
  TRAINED     0.09  0.12  0.17  0.22  0.25  0.36  0.53  0.66
  INIT        0.08  0.12  0.17  0.21  0.25  0.35  0.51  0.64
  SCRAMBLE    0.08  0.12  0.17  0.21  0.24  0.35  0.51  0.64
  trained max excess over baseline: −0.00
```

- **TRAINED ≈ INIT** at every `k` — learning to 96 % adds **no** shared dying subspace.
- **TRAINED ≈ SCRAMBLE** — the orientation-scramble null (`F'_l → O_{l+1} F'_l O_lᵀ`, spectrum-preserving,
  alignment-destroying) is *not beaten*: whatever alignment exists is the residual measure-concentration
  baseline `√(k/W)`, not structure.
- No small-`k` shoulder, no high-`k` low-rank plateau, no architectural inflation.

**Verdict: NEGATIVE.** A genuinely trained standard deep net, once the shared-backbone confound is removed,
does **not** develop the composition-annihilation signature.

## What this establishes (and what it does not)

- **The instrument works and is honest.** It survived ~12 successive refutations (gap-dominance →
  subspace alignment → `align(k)` curve → QR/Lyapunov de-censoring → target disqualification), caught its own
  H=40 `d=56` false positive via controls, correctly attributed the H=256 LSTM `align≈1` to architecture, and
  now — pointed at the one target where the answer is *not* pre-written — returns a clean negative that beats
  no null. This is what a working instrument returning "absent" looks like.
- **The hypothesis it tested is falsified on standard nets.** "Training deep networks on ordinary tasks
  leaves a sedenion-zero-divisor fingerprint (a shared dying Jacobian subspace)" is **not** supported. Vanishing
  gradients in these nets are magnitude decay and/or ordinary low rank, distributed across *unaligned*
  directions — not annihilation by composition onto a common subspace.
- **What stands is untouched.** The rupture algebra (associator = graded composition-failure signature;
  sedenion ZD variety ≅ `G₂/V₂(ℝ⁷)`), the compiler/𝕊-arithmetic, and the Spinoza/conatus geometry of the
  `σ_min` field (norm-preservation `‖xy‖=‖x‖‖y‖`, connectivity = "no condemnation") are theory and stand on
  their own. This negative concerns only the narrower empirical claim that *ordinary gradient-trained nets*
  carry the signature — they do not.

The connection between non-associative annihilation and learning dynamics, if it exists, is **not** visible in
the Jacobian spectra of standard trained networks. It would have to be engineered into an architecture or a
loss, not discovered as a fingerprint of vanilla training. That is a sharper, more honest place to stand than
a claimed positive would have been.

## Reproduce

```
python3 docs/research/deep_ffn_probe.py    # clean-target proof: init branch align at baseline
python3 docs/research/deep_ffn_train.py    # train ResMLP→96%, TRAINED vs INIT vs SCRAMBLE branch align(k)
```

Analytic branch Jacobian `F'_l = B_l·diag(1−tanh²(A_l z))·A_l` (numpy, exact); alignment via bottom-`k`
right-singular-subspace principal angles; orientation-scramble null as the spectrum-preserving control.
See also [`probe-corrected-protocol.md`](probe-corrected-protocol.md),
[`lyapunov-repositioning.md`](lyapunov-repositioning.md).
