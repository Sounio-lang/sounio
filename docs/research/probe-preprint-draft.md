<!-- docs:meta
topic_id: repo.docs.research.probe-preprint-draft
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.probe-preprint-draft
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Separating structural subspace annihilation from magnitude decay in products of recurrent Jacobians

**Demetrios Chiuratto Agourakis**
*Independent researcher*

**Status:** draft v0.3 — arXiv target: cs.LG (cross-list math.DS)
**Markers:** `[FILL]` = still open (only Zenodo deposit). Run numbers from `PROBE-RESULT-*.md`, multi-seed panels in `PROBE-RESULT-multiseed.md` / `artifacts/multiseed_*.json`, and the harnesses they cite.

---

## Abstract

When gradients vanish across a deep or recurrent composition, at least three structurally distinct mechanisms can be responsible, and they call for different remedies: (i) *magnitude decay*, in which the whole singular spectrum of the composed Jacobian slides downward; (ii) *rank collapse*, in which the representation degenerates towards rank one; and (iii) *structural subspace annihilation*, in which a small subspace is extinguished while the bulk of the spectrum remains healthy. Existing diagnostics — dynamical isometry, Lyapunov spectra, rank-collapse measures — characterise bulk properties and do not separate the third case from the first two. We introduce a discriminative probe consisting of a depth-resolved gap statistic, two principal-angle alignment measures (consecutive-factor and prefix-to-next-factor), and an `align(k)` curve whose *shape* distinguishes a small dead subspace from generic low effective rank. The central methodological contribution is a conditional null — the **orientation scramble** — which inserts random orthogonal matrices between factors, exactly preserving every local singular value, local multiplicity and depth while destroying only the geometric correspondence between the output of one factor and the input of the next. On synthetic chains with known ground truth we show that a gap statistic alone is *not* evidence of selective composition: matched chains with rotated singular bases produce gaps exceeding two decades at depth. Applied to a trained LSTM on the long-dependency *adding* problem, the probe returns a negative: an apparent positive (Cohen $d=+56$ at a frozen $m^\dagger=4$ against the orientation-scramble null alone) is eliminated by the `align(k)` shape, which follows the low-rank profile (trained h→h stays $0.9\to0.97$ out to $k=12$), and by the untrained-initialisation control, in which alignment is *higher* ($0.99$–$1.00$) than in the trained network ($0.92$ at $k=4$). Vanishing gradients in this setting are magnitude and rank, not structural annihilation.

**Keywords:** vanishing gradients, Jacobian spectra, Lyapunov exponents, principal angles, recurrent networks, negative results, null models

---

## 1. Introduction

The composed Jacobian of a recurrent or deep feedforward map,

$$P_T = J_{T-1}J_{T-2}\cdots J_0,$$

governs how a perturbation of the initial state propagates to the final state, and its singular spectrum is the standard object through which trainability is analysed. The observation that this spectrum degrades with depth is old and the remedies are well established: normalisation, residual paths, orthogonal initialisation, gating.

What is less often asked is *which* degradation has occurred. At least three qualitatively different spectra are compatible with "the gradient vanished":

| regime | signature | remedy |
|---|---|---|
| magnitude decay | whole spectrum slides down; relative shape preserved | scaling, normalisation, orthogonal init |
| rank collapse | spectrum degenerates towards a single surviving direction | architectural (skip paths, attention modifications) |
| **structural annihilation** | a small subspace is extinguished; bulk remains near its original scale | none obvious — norm-preserving fixes do not act on it |

The third regime is the one that existing diagnostics are least equipped to detect, and for a specific reason. Dynamical isometry asks whether the *entire* spectrum concentrates near unity; Lyapunov spectra characterise exponential growth rates of the composition; rank-collapse measures describe degeneration towards rank one. All three are bulk statistics. A composition in which four directions out of five hundred die while the remaining spectrum stays near its initial scale registers as approximately isometric under every one of them.

Whether that regime *occurs* in trained networks is an empirical question that, to our knowledge, has not been asked in this form. This paper builds the instrument to ask it and then asks it once.

**Contributions.**

1. A depth-resolved gap statistic $G_m(T)$ with an explicit treatment of the selection over $m$.
2. Two principal-angle alignment measures — consecutive ($A^{\mathrm{local}}$) and prefix-to-next-factor ($A^{\mathrm{carry}}$) — the second of which is the direct mechanistic quantity.
3. The `align(k)` curve, whose *shape* (shoulder versus plateau) separates a small dead subspace from generic low effective rank without requiring an additional control.
4. The **orientation scramble** null, which isolates the mechanism by preserving all local spectral content and destroying only inter-factor geometric correspondence.
5. A demonstration, on synthetic chains with known ground truth, that a gap statistic *alone* is not evidence: rotated-basis controls produce multi-decade gaps.
6. An application to a trained LSTM returning a clean negative, with the false positive caught by the controls rather than by inspection.

We regard (4) and (6) as the substantive contributions. (6) in particular: the instrument's value is demonstrated by its refusal of a result it was built to find.

---

## 2. Related work

**Dynamical isometry.** Saxe et al. and Pennington et al. established the programme of controlling the entire singular spectrum of the input–output Jacobian near unity, and the mean-field machinery for predicting when this is achievable. The criterion is by construction a bulk criterion: it is satisfied by a composition in which a low-dimensional subspace has been extinguished, provided the remaining directions are well conditioned. [1,2]

**Lyapunov spectra of recurrent networks.** The spectrum of $P_T$ as $T$ grows is, up to normalisation, the Lyapunov spectrum of the recurrent map, and this has been computed for RNNs — including in relation to trainability and chaos. Our $G_m(T)$ growth coefficient $\beta$ is a difference of Lyapunov exponents, $\lambda_{m+1}-\lambda_m$. We therefore make no claim of novelty for the spectral measurement itself; the contribution lies in the discriminant and the null. [3,4]

**Covariant Lyapunov vectors.** The question of whether the contracting directions of successive factors align is, in dynamical-systems language, a question about the Oseledets filtration and its covariant vectors. Our $A^{\mathrm{carry}}$ is a finite-$T$, network-adapted version of this measurement. [5]

**Rank collapse.** Dong et al. showed that pure attention degenerates doubly exponentially towards rank one. This is the opposite limiting case to the one we probe: total collapse of the bulk, rather than extinction of a small subspace with a healthy bulk. Our classifier is designed to separate the two. [6]

**Low intrinsic dimensionality.** Trained networks are widely reported to have low effective rank. This is the principal confounder for any alignment-based measurement, because a shared low-rank signal subspace forces the *complementary* subspaces to align trivially. Section 3.4 addresses it. [9]

---

## 3. Method

### 3.1 Setup

Let $s_t \in \mathbb{R}^{D}$ be the full recurrent state and $x_{0:T-1}$ a fixed input sequence, treated as conditioning rather than as a differentiation variable. Define

$$J_t = \frac{\partial s_{t+1}}{\partial s_t},\qquad P_T = \frac{\partial s_T}{\partial s_0}=J_{T-1}\cdots J_0.$$

For an LSTM, $s_t = (h_t, c_t) \in \mathbb{R}^{2H}$. Both components must be included: a probe of $\partial h_T/\partial h_0$ with $c$ excluded from the state does not describe the closed dynamical system and can fabricate rank deficiency by partial projection. Blocks of the full $P_T$ may of course be analysed afterwards, and Section 3.6 explains why they must be.

Implementation uses a pure closure `s0 -> sT` with `torch.func.jacrev`. Chain consistency was verified as

$$\frac{\|P_T^{\mathrm{direct}} - J_{T-1}\cdots J_0\|_F}{\|P_T^{\mathrm{direct}}\|_F} = 3.27\times10^{-16}\ (T=8),\qquad 3.30\times10^{-16}\ (T=32).$$

### 3.2 Gap statistic

Order the singular values of $P_T$ increasingly, $\sigma_1 \le \cdots \le \sigma_D$, and define

$$G_m(T) = \log_{10}\frac{\sigma_{m+1}(P_T)}{\sigma_m(P_T)},\qquad G^*(T)=\max_m G_m(T),\qquad m^*(T)=\arg\max_m G_m(T).$$

The maximisation over $m$ is a selection, and every null sample performs the same maximisation, so that the cost of searching for the best gap is absorbed into the null distribution. We sweep the full range of $m$; earlier versions of this work capped $m \le D/4$, a value inherited from an unrelated algebraic motivation and without empirical justification.

No fixed threshold on $G$ is used. Any threshold, when eventually reported, is a quantile of the null distribution conditional on dimension, depth, sequence and architecture.

### 3.3 Alignment

With $J_t = U_t \Sigma_t V_t^{\top}$ and $U^-, V^-$ denoting the $m$ lowest singular subspaces, define

$$A^{\mathrm{local}}_t(m) = \frac{1}{m}\big\|(U_t^-)^{\top}V_{t+1}^-\big\|_F^2, \qquad
A^{\mathrm{carry}}_t(m) = \frac{1}{m}\big\|(U_{P_t}^-)^{\top}V_t^-\big\|_F^2 .$$

$A^{\mathrm{local}}$ asks whether the contracted output of one factor enters the contracting direction of the next. $A^{\mathrm{carry}}$ — the primary mechanistic quantity — asks whether the subspace *already contracted by the prefix* is contracted again by the next factor.

Because singular vectors are not unique under repeated or nearly repeated singular values, all comparisons are between *subspaces* via principal angles, never between individual columns.

The random baseline for $m$-dimensional subspaces in dimension $D$ is $\sqrt{m/D}$; all alignments are reported against it.

### 3.4 The `align(k)` curve

A single alignment value at a single $k$ cannot distinguish structural annihilation from low effective rank, because a shared low-rank signal subspace forces the complementary subspaces to align. The *shape* of $k \mapsto \mathrm{align}(k)$ does:

| hypothesis | shape |
|---|---|
| annihilation, dead subspace of dimension $\mu$, healthy bulk | high for $k \le \mu$, falling for $k > \mu$ — a **shoulder** at small $k$ |
| low effective rank, deficit $D-r$ | high up to **large** $k$ — a plateau, no small-$k$ shoulder |
| neither | flat at the $\sqrt{k/D}$ baseline |

This also removes an otherwise arbitrary choice: no privileged $k$ need be specified in advance, and the shoulder position becomes data rather than a parameter.

### 3.5 Numerics: do not form the product

Forming $P_T$ explicitly destroys the tail. Once the condition number of the product exceeds machine precision, the small singular values are no longer resolved and $G(T)$ censors at roughly twelve decades — precisely where the signature should be strongest.

We therefore use the discrete QR (Benettin / Dieci–Van Vleck) method standard for Lyapunov spectra:

$$Q_0 = I,\qquad J_tQ_{t-1}=Q_tR_t,\qquad \log\sigma_i(P_T)\approx\sum_t \log\big|R_t[i,i]\big|.$$

Reorthonormalising at each step keeps the conditioning bounded, extends the dynamic range essentially without limit, costs one QR per step, and supplies the $U_{P_t}$ required by $A^{\mathrm{carry}}$ without an SVD of the product.

### 3.6 Block decomposition in gated architectures

In a standard LSTM the gates depend on $h_{t-1}$ and $x_t$ but never on $c_{t-1}$. Hence

$$\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t),\qquad
\frac{\partial h_t}{\partial c_{t-1}} = \operatorname{diag}\!\big(o_t\odot\tanh'(c_t)\odot f_t\big).$$

The entire $c$-column of $J_t$ is diagonal, with the *same* index structure at every step. This is the constant error carousel functioning as designed — and it means that half of the state Jacobian is coordinate-aligned by architecture and can carry an alignment measurement on its own.

We therefore report alignment by block:

| block | status |
|---|---|
| $h \to h$ | dense; the only place a claim of structural annihilation can be made |
| $c \to c$ | diagonal by construction; **an internal positive control for architectural alignment**, free of extra cost |
| full state | interpretable only in the light of the two above |

If $A^{\mathrm{carry}}$ is near unity on the $c$ block and at null level on the $h$ block, the full-state alignment is entirely architectural. This diagnosis is available within the same computation. The same reasoning disqualifies S4 and Mamba as primary targets: their state matrices are diagonal (or diagonal-plus-low-rank) by design, so successive Jacobians share eigenvectors exactly and alignment equals unity by architecture rather than by learning.

---

## 4. Null models

### 4.1 Orientation scramble (primary)

Insert a random orthogonal $Q_t$ between consecutive factors:

$$P_T^{\mathrm{scr}} = J_{T-1}Q_{T-2}J_{T-2}\cdots Q_0 J_0 .$$

This preserves **exactly**: every local singular value, every local multiplicity, the local contraction magnitudes, the depth, and the sequence of marginal spectra. It destroys **only** the geometric correspondence between the output of $J_t$ and the input of $J_{t+1}$.

It is therefore the conditional null of the mechanism: any excess of the observed statistic over this null is attributable to inter-factor geometry and to nothing else.

**As executed in the runs of Section 6.** One orientation scramble per sequence, with each $Q_t$ drawn by QR factorisation of a standard-normal matrix (Haar on $O(n)$ up to the usual sign convention of the thin QR), applied independently to each of $n=40$ sequences for the full control curves (`run_probe_full.py`) and $n=200$ sequences for the Cohen-$d$ readout (`train_and_probe_lstm.py`). The ResMLP clean-target curves average $16$ input draws and $16$ independent scrambles (`deep_ffn_train.py`). Identical maximisation over $m$ (or the frozen discovery/confirmation split for $m^\dagger$); identical censoring treatment.

> **Protocol debt.** A heavier design — $64$ signed-permutation scrambles per sequence plus $16$ Haar scrambles on a fixed subset of $32$ sequences, with the paired distribution
>
> $$\Delta_i(T) = G_i^*(T) - \operatorname{median}_b G_{i,b}^{*,\mathrm{scr}}(T)$$
>
> and empirical $p = (1+\#\{G_{\mathrm{null}}\ge G_{\mathrm{obs}}\})/(B+1)$ — is the right report form for a camera-ready submission. It is **not** what the committed artefacts used. Until that re-run is archived, the numbers below are means over sequences / draws, not paired $\Delta$ distributions.

### 4.2 Untrained initialisation

Same architecture, same input vectors, freshly drawn parameters. The decisive control curves of Section 6.1 used a single untrained seed against the trained seed (`run_probe_full.py`). A multi-seed panel (`multiseed_lstm_init.py`; $n{=}16$ seeds × $16$ sequences, pure-numpy analytic Jacobians, $H{=}40$, $T{=}30$) confirms that INIT h→h at $k{=}4$ is $0.992\pm 0.005$ (min $0.981$, max $0.997$; every seed $>0.95$). Asks whether the pattern was acquired or was present in the parameterisation from the start. **In our application this null was decisive, and the multi-seed panel shows it is not a one-seed fluke.**

### 4.3 Gate-wise weight shuffle

Parameters permuted independently within each gate block, preserving the empirical multiset of weights. Destroys learned wiring but does not preserve row norms or local spectra, and is therefore secondary to the orientation scramble.

### 4.4 Matched synthetic controls

Chains with identical local spectral proportions, in two regimes: common singular bases (positive control) and independently rotated bases (matched negative control). This separates *local multiplicity* from *multiplicative persistence*.

### 4.5 Planted low rank

A random chain with injected low-rank structure. This is the null that fires in real networks and that neither the shuffle nor the initialisation control captures — both of those produce more nearly full-rank matrices and would leave the low-rank confound intact.

---

## 5. Synthetic validation

### 5.1 The three regimes are separable

Calibration on depth-$T{=}16$ products of local $4/8/4$ spectra (`mechanism_analysis.py`; `probe-corrected-protocol.md`):

| stack | mean cos (dying $4$-subspaces) | `gap_dominance` ($T{=}16$) | $P(\mathrm{gap\_dominance}>1)$ under null |
|---|---:|---:|---:|
| **aligned** (common zero-divisor basis — genuine composing annihilation) | **0.988** | 5.71 | — |
| **rotating** (matched $4/8/4$ per factor, bases re-drawn) | 0.530 | **99.4** | **97%** |
| real Gaussian | 0.415 | 0.33 | 1% |

Reference alignment against a baseline of $\sqrt{k/D}$: aligned $0.988$, rotated $0.530$, Gaussian $0.415$. The `align(k)` shoulder (`align_curve.py`, depth-$12$ stacks) sits at $k=4$ for the annihilation construction (align $0.99\to0.85$, peak $\gg$ baseline $0.50$), at $k=10$ for planted low rank (shared-complement rank $r{=}6$, dead $\approx 10$), and tracks the baseline for the Gaussian null.

### 5.2 A gap alone is not evidence

Chains with matched local spectral proportions, in aligned and rotated regimes, at increasing depth:

| $T$ | aligned bases (decades) | rotated bases (decades) |
|---:|---:|---:|
| 1 | 1.10 | 1.10 |
| 2 | 2.19 | 0.78 |
| 4 | 4.39 | 1.20 |
| 8 | 8.78 | 1.20 |
| 16 | 12.11 *(censored)* | 3.28 |
| 32 | 11.77 *(censored)* | 2.47 |

The informative entry is not the growth of the aligned chain, which was constructed. It is that the **rotated** chain — with no common contracting subspace whatsoever — produces gaps exceeding two decades at depth. Finite random products develop a hierarchy of exponents and sample gaps.

Consequently:

> $G > 1$ or $G > 2$ is not, by itself, evidence of selective composition.

A companion observation from an earlier iteration of this work makes the same point from the other side: a gap-dominance statistic gave $99.4$ for the rotated control against $5.71$ for genuine aligned structure, passing a fixed threshold in $97\%$ of null samples. Against a Gaussian null the same statistic appeared significant at a $1\%$ false-positive rate. **The choice of null, not the statistic, determined the conclusion.**

Censoring beyond $T=8$ in the table above is the numerical ceiling of Section 3.5 and is removed by the QR method in the current implementation.

---

## 6. Application: two trained targets

We apply the probe to two architectures. The first, an LSTM, produced an apparent positive that the controls eliminated; the second, a deep residual MLP, is a cleaner target and returns a negative against the empirical null.

### 6.1 Target 1 — LSTM, and an architectural confound

**Primary run** (`run_probe_full.py`, `PROBE-RESULT-lstm-adding.md`): LSTMCell, hidden size $H=40$, trained on the *adding* problem (two marked positions in a length-$T{=}30$ sequence; target $=$ sum of the marked values) for $2500$ Adam steps, test MSE $8\times 10^{-4}$ against chance variance $\approx 0.17$. Control curves average $n=40$ sequences. A naive readout — orientation-scramble null alone, frozen $m^\dagger=4$ — reported trained h→h alignment $0.92$ versus scramble $0.27$, **Cohen $d=+56$**, and would have been labelled subspace death.

An apparent signature was observed at Cohen $d=+56$. It did not survive:

1. **`align(k)` shape.** The curve followed the low-rank profile — trained h→h $0.76,0.84,0.90,0.92,0.95,0.96,0.97$ at $k=1,2,3,4,6,8,12$ (baseline at $k{=}12$ is $0.55$) — rather than a small-$k$ shoulder with a healthy bulk.
2. **Untrained-initialisation control (decisive).** Alignment in the untrained network was $0.99$–$1.00$ at $k=4$–$12$, *higher* than the trained $0.92$. An equal value would have been ambiguous, since learned structure could be superimposed on architectural structure; a value that *decreases* with training cannot be read as acquisition. Multi-seed ($n{=}16$, §4.2): INIT@$k{=}4$ mean $0.992\pm 0.005$, min $0.981$ — every seed above the trained value.

**Scale check** (`probe_h256_init.py`, `PROBE-RESULT-h256-scale.md`): untrained $H{=}256$, $T{=}200$, pure-numpy analytic LSTM Jacobian (validated to $7\times 10^{-8}$ against autograd). INIT h→h $\approx 1.00$ at every $k\in\{1,\ldots,63\}$ against baseline $\sqrt{k/(2H)}\in[0.04,0.35]$. The architectural confound is sharper at scale, not weaker.

The mechanism of the confound is now identified and is worth stating, because it generalises: **in an LSTM the same recurrent weight matrix $W_{hh}$ appears at every time step.** Successive $J_t$ are therefore the same operator modulated by gates, and they share singular structure by construction. Any recurrence with a shared backbone will exhibit near-unit alignment independently of learning. This is the same disqualification that excludes S4 and Mamba (Section 3.6), arriving through a different route: there, a diagonal state matrix; here, a repeated dense one.

### 6.2 Target 2 — deep residual MLP, a target without a backbone

A feedforward network with **distinct weights per layer** has no shared backbone, and therefore nothing that fabricates alignment. This is confirmed at initialisation: branch Jacobians $F'_l = J_l - I$ in an untrained network sit at or marginally below the analytic baseline.

**Initialisation cleanliness check** (`deep_ffn_probe.py`): untrained residual FFN with *distinct* random weights per layer, depth $L{=}24$, ambient dimension $d{=}64$, branch width $h{=}128$; curves averaged over $12$ input draws. The tabulated baseline uses $d{=}64$ (this check), not the trained width of Section 6.3.

| $k$ | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| baseline $\sqrt{k/d}$ | 0.12 | 0.18 | 0.25 | 0.35 | 0.50 | 0.71 |
| untrained $F'_l$ | 0.10 | 0.15 | 0.21 | 0.30 | 0.43 | 0.64 |

Excess: $-0.02$. **Any signal above the null in a trained network of this class would therefore be real.** (The trained ResMLP of §6.3 uses width $W{=}96$; its own init row is reported there and likewise sits at the scramble baseline.)

### 6.3 Result on the clean target

ResMLP, width $W=96$, depth $L=8$. **Single-seed reference** (`deep_ffn_train.py`): **96%** test accuracy (Adam, $5000$ steps, BCE-with-logits, test $n{=}4000$). **Multi-seed panel** (`multiseed_resmlp.py`; $n{=}16$ seeds, early-stop at acc $\ge 0.90$, mean acc $0.941\pm 0.006$):

| $k$ | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline $\sqrt{k/W}$ | 0.10 | 0.14 | 0.20 | 0.29 | 0.41 | 0.58 | 0.71 |
| trained (16 seeds) | 0.083 | 0.121 | 0.174 | 0.250 | 0.359 | 0.519 | 0.649 |
| untrained | 0.080 | 0.114 | 0.168 | 0.243 | 0.349 | 0.507 | 0.635 |
| orientation scramble | 0.081 | 0.118 | 0.169 | 0.244 | 0.350 | 0.507 | 0.635 |

Trained $\approx$ untrained $\approx$ orientation-scramble null at every $k$ (absolute levels). No shoulder at small $k$; no plateau at large $k$; no architectural inflation.

**Paired $\Delta$ (multi-seed).** For each seed and input, $\Delta_i(k)=\mathrm{align}_i^{\mathrm{tr}}(k)-\operatorname{median}_{b=1\ldots 8}\mathrm{align}_{i,b}^{\mathrm{scr}}(k)$; pooled $n{=}16\times 16{=}256$. One-sided sign-flip $p$ under $H_0$: $\Delta$ symmetric about 0 ($B{=}9999$):

| $k$ | mean $\Delta$ | sd | $p_{\mathrm{signflip}}$ |
|---:|---:|---:|---:|
| 1 | $+0.0025$ | 0.025 | 0.053 |
| 4 | $+0.0043$ | 0.013 | 0.0001 |
| 16 | $+0.0090$ | 0.006 | 0.0001 |
| 48 | $+0.0134$ | 0.003 | 0.0001 |

A mean $\Delta$ of order $0.01$ is *detectable* at this $n$ for $k\ge 2$, but is **not** a subspace-annihilation signal: (i) no $k$ meets a substantive threshold mean$\Delta>0.05$ with $p<0.05$; (ii) mean $\Delta$ *rises* with $k$ (opposite of a small-$k$ shoulder); (iii) absolute curves remain on the scramble / init floor. The single-seed “trained exceeds untrained by $0.01$–$0.02$ at large $k$” eyeball is a tiny offset, not a signature. Full tables: `PROBE-RESULT-multiseed.md`.

> ⚠️ **Baseline caveat.** All three conditions sit systematically below the analytic $\sqrt{k/W}$, by a margin too consistent to be noise. The likely cause is a mismatch between the analytic expression (derived for the mean squared cosine between uniformly random subspaces) and the statistic actually computed. The **orientation-scramble null is the correct comparator** and should lead; $\sqrt{k/W}$ belongs in a footnote with this caveat.

### 6.4 Scope

Vanishing gradients in these networks are magnitude decay and low effective rank in *non-aligned* directions — not annihilation over a common subspace.

We state the scope without softening. The clean target is a residual network of moderate depth ($L=8$) that trains successfully to $96\%$. Residual connections exist precisely to prevent composition failure, and a network that trains well may simply have no composition failure to detect. This is therefore a negative obtained under conditions favourable to absence. The strongest remaining test in this class would be a plain (non-residual) network deep enough to exhibit genuine optimisation difficulty; we did not run it.

What the result does establish: **the signature is not a fingerprint left by ordinary training.** If a link between non-associative composition and learning exists, it must be engineered into an architecture or an objective, not discovered in networks that compose additively.

This is, notably, what the framework predicts. Under additive composition, annihilation is structurally inaccessible: $g_1+g_2=0$ only by opposition, which is a negative product, not a zero one. A residual network composes additively along its skip path. The negative is therefore coherent with the necessary condition rather than merely disappointing — it confirms, in the empirical regime, a constraint derived analytically.

---

## 7. Discussion

**What the instrument is for.** The probe's value is demonstrated by the fact that it eliminated a positive it had been constructed to find. Twelve successive design revisions were required before it could do so, and each was prompted by identifying a way in which an earlier version could only read a positive: a classifier calibrated on three hand-written spectra with no false-positive rate; a Gaussian null too weak to separate structure from any structure; a target architecture in which the sought signature is written in by construction; a gap statistic blind to the difference between composition and rotation. Each of these would have produced a publishable-looking positive.

**A by-product.** Alignment in the untrained network exceeded that in the trained network, which is to say that training *reduces* architecturally-imposed subspace alignment. We record this as an observation rather than a finding: $n=1$ architecture, one task, and a mundane explanation (weights growing and diversifying, effective rank rising) is available and untested.

**What the method cannot do.** It measures geometry, not consequence. Even a clean positive against all nulls would establish the existence of a spectral phenomenon, not that it matters. The functional link — whether $A^{\mathrm{carry}}$, $\beta_G$ and $m^*/D$ predict performance beyond conventional spectral magnitude ($\log\sigma_{\min}$, median $\log\sigma_i$, $\|P_T\|_F$, gate saturation, current loss) — is a second hypothesis, to be tested only after a mechanistic positive, and out of sample.

**Heterogeneity.** If the effect were to appear only on a subset of input sequences, this would not be automatic artefact, but it would change the object of the claim from a global property of the model to a state- or sequence-conditioned geometry. Under the global hypothesis as formulated, such heterogeneity counts as a negative.

---

## 8. Limitations

- Single architecture family (LSTM); S4/Mamba excluded by construction (Section 3.6), transformers not tested.
- Single task family.
- The discovery/confirmation split controls selection of $m$ but not selection of architecture or task.
- The performance link is not tested; no claim of a failure mode is made.
- Product-formation censoring of $G(T)$ (ceiling $\sim 12$–$16$ decades for direct SVD of $P_T$) is removed by the discrete QR / Lyapunov method (`lyapunov_qr.py`): at $T{=}256$ the QR spectrum reaches $\min\log_{10}\sigma\approx -312$ with a gap of $34.6$ decades, with no residual ceiling in the recorded range. Remaining numerical risk is ordinary floating-point noise in the per-step QR, not product-formation censoring.

---

## 9. Code and data availability

Harnesses and result notes live in the Sounio repository under `docs/research/`:

- Protocol / analysis: `train_and_probe_lstm.py`, `run_probe_full.py`, `probe_h256_init.py`, `deep_ffn_probe.py`, `deep_ffn_train.py`, `multiseed_lstm_init.py`, `multiseed_resmlp.py`, `mechanism_analysis.py`, `align_curve.py`, `lyapunov_qr.py`
- Frozen result notes: `PROBE-RESULT-lstm-adding.md`, `PROBE-RESULT-h256-scale.md`, `PROBE-RESULT-deep-ffn.md`, `PROBE-RESULT-multiseed.md`, `probe-corrected-protocol.md`, `align-curve-and-target.md`, `lyapunov-repositioning.md`
- Multi-seed JSON: `artifacts/multiseed_lstm_init.json`, `artifacts/multiseed_resmlp.json`
- Repository: [https://github.com/Sounio-lang/sounio](https://github.com/Sounio-lang/sounio) (paths above on the commit that lands this draft)

[FILL: frozen artefact tarball + DOI — Zenodo not yet deposited.] The multi-seed JSONs above are the minimum deposit payload for a camera-ready re-run.

---

## Acknowledgements

None beyond the AI tools listed below.

## AI contribution disclosure (GAIDeT / ICMJE 2025)

The following generative AI tools were used, with the tasks delegated to each:

- **Claude Opus 4.8 (Claude Code)** — critical review of experimental design and null-model specification across the successive protocol revisions; drafting assistance on method sections.
- **Claude / Codex agents (Claude Code, OpenAI Codex)** — implementation support for the probe harnesses and control scripts under `docs/research/`.
- **Claude Opus 4.8 / Grok (xAI)** — drafting and editorial passes on this manuscript; bibliographic verification of references (PR #1367); filling of run numbers from committed artefacts (this draft).

The author reviewed, verified and takes full responsibility for all content, including all numerical results and their interpretation.

## References

> **Verification note (2026-07-21).** All entries below were verified by independent
> title+author search (bibliographic details cross-checked against the publisher/arXiv
> record, not against a page summary — the appendix pattern of the program registry).
> The scite MCP was the intended verifier but was quota-blocked; WebSearch was used
> instead, so **Smart-Citation tallies (supporting/contrasting) are not attached** — only
> existence, authorship, venue and DOI are confirmed. Ref 9 (Ansuini et al.) is newly added
> to close the §2 "low intrinsic dimensionality" [CHECK]. A subsequent scite pass (2026-07-21)
> completed the mandatory retraction/correction check **clean** for refs 3 (Engelken), 7
> (Benettin, Part 1) and 8 (Dieci–Van Vleck) — `retraction_notices` absent; tallies still not
> attached (scite backend was intermittent).

1. Saxe AM, McClelland JL, Ganguli S. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In: 2nd International Conference on Learning Representations (ICLR); 2014. arXiv:1312.6120.
2. Pennington J, Schoenholz SS, Ganguli S. Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. In: Advances in Neural Information Processing Systems 30 (NIPS 2017); 2017. p. 4785–95.
3. Engelken R, Wolf F, Abbott LF. Lyapunov spectra of chaotic recurrent neural networks. Phys Rev Res. 2023;5(4):043044. doi:10.1103/PhysRevResearch.5.043044.
4. Vogt R, Puelma Touzel M, Shlizerman E, Lajoie G. On Lyapunov exponents for RNNs: understanding information propagation using dynamical systems tools. Front Appl Math Stat. 2022;8:818799. doi:10.3389/fams.2022.818799.
5. Ginelli F, Poggi P, Turchi A, Chaté H, Livi R, Politi A. Characterizing dynamics with covariant Lyapunov vectors. Phys Rev Lett. 2007;99(13):130601. doi:10.1103/PhysRevLett.99.130601.
6. Dong Y, Cordonnier J-B, Loukas A. Attention is not all you need: pure attention loses rank doubly exponentially with depth. In: Proceedings of the 38th International Conference on Machine Learning (ICML); PMLR 139; 2021. p. 2793–803.
7. Benettin G, Galgani L, Giorgilli A, Strelcyn J-M. Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory. Meccanica. 1980;15(1):9–20. doi:10.1007/BF02128236. [Part 2: Numerical application. Meccanica. 1980;15(1):21–30. doi:10.1007/BF02128237.]
8. Dieci L, Van Vleck ES. Computation of a few Lyapunov exponents for continuous and discrete dynamical systems. Appl Numer Math. 1995;17(3):275–91. doi:10.1016/0168-9274(95)00033-Q.
9. Ansuini A, Laio A, Macke JH, Zoccolan D. Intrinsic dimension of data representations in deep neural networks. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019); 2019. p. 6109–19. arXiv:1905.12784.
