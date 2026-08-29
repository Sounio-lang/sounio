# KEC-α on the DCT — two-wave result (HONEST NULL for the structural metric)

**Date:** 2026-05-27 · **Pipeline:** `scripts/research/dct_kec_fixture.py` (I/O) +
`examples/semantic_orc/dct_kec_compute.sio` (all math, native Sounio).

## Question
Does the **semantic coherence of a subject's egocentric-distance assignment** track depression?
We treat the Demonstrative Choice Task (DCT, Kruse, Rocca & Wallentin) proximal/distal choices as a
binary field `s∈{−1,+1}` over a *shared* GloVe-6B-300d k-NN semantic graph (k=10, cosine), and define

> **C = Σ_{(i,j)∈E} w_ij (s_i − s_j)² / Σ_i s_i² = 4·(weight of semantic edges cut by the
> proximal/distal boundary)/N.**  Low C = coherent (boundary respects semantic clusters);
> high C = fragmented.

This is the structural quantity the original paper's per-item PCA→logistic model **cannot** see (it
treats words independently). Non-redundancy test: does C predict PHQ-9 **after** controlling for
overall proximal fraction and per-subject valence-bias?

## Discipline
Wave 1 (n=775) = DISCOVERY, Wave 2 (n=879) = CONFIRMATION. Metric, covariates, k, embedding, and the
test were frozen in the plan before touching PHQ-9. Primary = partial Spearman ρ(C, PHQ9 | prox, vbias),
two-tailed, with a 500-draw label-permutation null. C, E(=prox/entropy), and graph all GloVe-100%-covered.

## Result

| statistic | Wave 1 (n=775) | Wave 2 (n=879) |
|---|---|---|
| Spearman ρ(C, PHQ9) | **0.085** | **0.044** |
| partial ρ(C, PHQ9 \| prox, vbias) | **0.053** | **0.010** |
| permutation p (two-sided, B=500) | **0.144** | **0.762** |
| AUC (C → PHQ9_class≥10) | 0.532 | 0.509 |
| ρ(valence_bias, PHQ9) | 0.193 | 0.117 |
| ρ(C, valence_bias) | 0.231 | 0.312 |
| ρ(C_vad, PHQ9) | 0.104 | 0.075 |
| ρ(prox_frac, PHQ9) | −0.060 | −0.012 |
| C mean / prox_frac mean / vbias mean | 4.61 / 0.475 / −0.095 | 4.67 / 0.474 / −0.099 |

## Verdict — clean null, replicated
- **The structural coherence C carries no depression signal.** ρ is small in wave 1 (0.085) and
  *weaker* in the independent confirmation wave (0.044); the partial correlation collapses to ~0
  (0.053 → 0.010) and the permutation test is non-significant in both (p=0.14, 0.76). C-as-classifier
  AUC ≈ chance (0.53, 0.51), far below the paper's per-item model (.67/.62).
- **The only real signal is `valence_bias`** (ρ=+0.19→+0.12). `vbias = corr(s, valence)` with
  s=+1 distal/−1 proximal; the population mean is negative (−0.095/−0.099), i.e. the *normative*
  pattern is distal↔low-valence (negative/fearful words pushed "away"). The **positive** correlation
  with PHQ9 means depression goes with a **weaker, flatter** valence-driven sorting — the normative
  negative-valence→distal coupling **attenuates with severity** (affective flattening / blunted
  valence–distance coupling), rather than intensifying. This re-finds the paper's NRC-VAD valence
  effect (and its modest cross-wave replication), now read in the correct direction.
- **C is moderately correlated with valence_bias** (ρ=0.23→0.31), so C's already-tiny raw relation to
  PHQ9 is largely the shadow of the valence effect; controlling vbias collapses it to ~0. Mediation
  language is therefore warranted.
- **Interpretation:** depression's footprint in the DCT is a *valence* effect on per-word choice (and
  a blunting of it), not a *structural/topological* property of how the proximal/distal boundary sits
  in semantic space. The egocentric-distance field is no more (or less) semantically organized in
  depression. The KEC-α coherence reframing is well-posed and machine-checked, but the phenomenon it
  targets is absent here.

## Robustness to embedding (the plan's "original" primary)
Repeated with **fastText `wiki-news-300d-1M`** (the plan's frozen-primary embedding; GloVe-6B above was
the convenience substitute). 100% coverage; result essentially identical → the null is not a GloVe
artifact.

| statistic | W1 GloVe → fastText | W2 GloVe → fastText |
|---|---|---|
| Spearman ρ(C, PHQ9) | 0.085 → 0.076 | 0.044 → 0.044 |
| partial ρ(C, PHQ9 \| prox, vbias) | 0.053 → 0.052 | 0.010 → 0.013 |
| permutation p | 0.144 → 0.150 | 0.762 → 0.691 |
| AUC | 0.532 → 0.529 | 0.509 → 0.517 |
| ρ(C, vbias) | 0.231 → 0.215 | 0.312 → 0.296 |

(On-disk `dct_kec_wave{1,2}.sio` + manifests now reflect fastText; rerun with
`python3 scripts/research/dct_kec_fixture.py --wave N --embedding glove` to restore GloVe.)

## Original KEC (spectral E, C) on the per-subject proximal-induced subgraph
Second instrument, same dataset: the *canonical* KEC spectral metrics (`kec_spectral.sio`) —
**E = von Neumann spectral entropy**, **C = λ₂ algebraic connectivity** — computed per subject on the
**proximal-induced subgraph** of the shared embedding graph (nodes = words chosen "this", original kNN
edges among them). Jacobi eigensolver generalized to N≤288, self-verified each run on K₁₂/C₈/K₂₀ closed
forms (λ₂ = 12 / 0.5858 / 20 ✓). ~264s (wave1) / ~300s (wave2) for 775/879 eigensolves. File:
`examples/semantic_orc/dct_kec_spectral.sio`; runnable `dct_kec_spec_wave{1,2}.sio`.

| statistic | Wave 1 (n=775) | Wave 2 (n=879) |
|---|---|---|
| ρ(E, PHQ9) raw | −0.106 | −0.015 |
| ρ(C=λ₂, PHQ9) raw | −0.108 | −0.020 |
| ρ(E, \|P\|) (size confound) | 0.908 | 0.886 |
| ρ(C, \|P\|) (size confound) | 0.777 | 0.741 |
| partial ρ(E \| prox, vbias) | −0.061 | +0.024 |
| partial ρ(C \| prox, vbias) | −0.023 | +0.023 |
| permutation p (on E) | 0.084 | 0.459 |

**Verdict — null, confirmed.** E and C are dominated by proximal-set size (ρ≈0.9/0.74): a bigger induced
graph mechanically has higher entropy and connectivity. The raw weak-negative wave-1 signal (−0.11) did
**not** replicate (−0.02), the partial correlation **flipped sign** across waves, and permutation p went
0.084→0.459 — noise. Controlling proximal fraction (=|P|/N) removes it. Two independent instruments
(KEC-α cut metric + original spectral E/C), two waves each, all null: depression's DCT footprint is the
valence effect on per-word choice, not any graph-structural property of the proximal/distal field.

## Validity / limits
- C metric hand-verified on tiny graphs (`dct_kec_core.sio`: all-same→0, checkerboard→3.0, block→1.0).
- Sign of the ±1 coding is irrelevant (squared difference); polarity-free.
- Permutation shuffles PHQ labels (omnibus null); a residual-permutation null would be marginally more
  exact for the *partial* statistic but cannot rescue a p=0.76.
- E and K (coarse-ORC) were planned as exploratory secondaries; given the primary null they were not
  pursued — C alone settles the structural question. C_vad (VAD-space coherence) is likewise null.
- A null is the result: no fabrication-risk inflation, no forking. Both waves reproducible:
  `bin/souc run examples/semantic_orc/dct_kec_wave{1,2}.sio` (~1.3s each).

## Provenance
Data: OSF `bqhyg` (df_wide1/2, post-exclusion n=775/879), `df_wide{1,2}.csv` SHA-256 in
`dct_kec_wave{1,2}_manifest.json`. Embeddings: GloVe-6B-300d, NRC-VAD-Lexicon. Encoding from
Kruse `dct_preprocessing_vol1.Rmd` (this=proximal, that=distal).
