<!-- docs:meta
topic_id: repo.docs.research.ocssm-preregistration-v0
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ocssm-preregistration-v0
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# O-CSSM / Tapestry — Pre-Registration Document v0

**Project title:** Octonionic and Sedenionic Conversational State-Space Models (O-CSSM): pre-registered structural correspondences between Cayley–Dickson algebras and dyadic dialogue dynamics.

**Principal investigator:** Demetrios Chiuratto Agourakis (ORCID 0009-0001-8671-8878).

**Document status:** Draft v0, 2026-04-23. Not yet deposited. Target repository: OSF (Open Science Framework) with SHA-256 content hash committed to git repository `sounio` under `docs/research/ocssm_preregistration_YYYYMMDD.md` *before* any training run on the final corpus.

**Binding scope:** This document specifies the confirmatory analyses and decision rules that will be used to evaluate the three structural correspondences advanced in the O-CSSM homology thesis. Any analysis performed before deposit or deviating from this specification will be reported as *exploratory*, not confirmatory.

---

## 1. Background (brief)

The O-CSSM thesis claims that the non-associative algebras of Cayley–Dickson — the octonions 𝕆 and their ambient sedenions 𝕊 — are the natural algebraic structure of meaning-composition, affect, and rupture in human dyadic dialogue. The claim is *homological* (same algebra), not *analogical*. Three structural correspondences are advanced simultaneously:

- **(i) Associator = context-dependent re-parenthesization of meaning.**
- **(ii) L/R multiplicative asymmetry = speaker-directional influence.**
- **(iii) Sedenion zero-divisor pairs = dissociative affect configurations.**

Full theoretical skeleton: `docs/research/ocssm_preprint_skeleton.md`.

Formal G₂-invariance argument underlying (iii): `docs/research/ocssm_g2_invariance_proof.md`.

---

## 2. Hypotheses

**H1 (Moufang equivariance).** A trained encoder `e : U → S^{15} ⊂ 𝕊` satisfying the algebraic restrictions {R1..R7} (see skeleton §4) will satisfy, on a held-out corpus, both:
- **H1a (alternativity):** `[e(u), e(u), e(v)] ≈ 0` for self-repetition patterns `u·u·v`.
- **H1b (middle Moufang):** `(e(a)e(b))(e(c)e(a)) ≈ e(a)((e(b)e(c))e(a))` for 4-turn speaker-return patterns `(a, b, c, a)`.

A generic-capacity unrestricted encoder will not.

**H2 (conjugation-equivariance under speaker reversal).** For pairs `(u, u')` where `u'` is the same semantic content produced by the opposite speaker, `e(u') ≈ \overline{e(u)}` (sedenionic conjugation). For random pairs, this equivalence fails.

**H3a (zero-divisor arithmetic survival).** The 84 primitive sedenionic zero-divisor pairs satisfy `‖a·b‖ < 10^{-18}` in Sounio bit-identical arithmetic; in PyTorch f64 arithmetic, `‖a·b‖` exceeds this threshold due to accumulated floating-point error through the CD doubling.

**H3b (downstream ZD detection is bit-identical-arithmetic-dependent).** A detector trained on the Sounio backend using ZD-proximity as discriminator achieves substantially higher rupture-detection AUC than the same detector trained/run on PyTorch f64.

---

## 3. Predictions with pre-registered numerical thresholds

### 3.1 Instruments

Let `e` denote the trained algebraically-restricted encoder. Let `e_gen` denote the trained unrestricted-capacity encoder (matched architecture minus {R1..R7}). All norms are sedenionic Euclidean `‖·‖ = √⟨·,·⟩` computed in bit-identical Sounio arithmetic.

```
M_alt      := ⟨ ‖[e(u), e(u), e(v)]‖ / (‖e(u)‖² · ‖e(v)‖) ⟩          over repetition instances
M_mou      := ⟨ ‖(e(a)e(b))(e(c)e(a)) − e(a)((e(b)e(c))e(a))‖ / 𝒩 ⟩  over 4-turn-return instances
             where 𝒩 = ‖e(a)‖² · ‖e(b)‖ · ‖e(c)‖
M_conj     := ⟨ ‖e(u') − \overline{e(u)}‖ / ‖e(u)‖ ⟩                  over speaker-paired instances
M_conj_rnd := ⟨ ‖e(u') − \overline{e(u)}‖ / ‖e(u)‖ ⟩                  over random pairs
M_ZD(B)    := (1/84) · #{ (a,b) ∈ ZD_primitive : ‖a·b‖_B < 10^{-18} }  for backend B
AUC_det(B) := rupture detection AUC for detector trained on backend B
OP(B)      := ⟨ d_min(h, Z) ⟩ over anno-rupture turns                 orbit proximity on backend B
```

### 3.2 Pre-registered thresholds

```
τ_alt      = 1.0 × 10^{-12}   (~4 orders of headroom over f64 epsilon after ~40 CD mult ops)
τ_mou      = 1.0 × 10^{-11}   (proportionally less headroom due to longer product chain)
τ_conj     = 0.10             (10% of encoder norm; matched to inter-annotator paraphrase distance, cf. MRPC)
ratio_conj = 5                (M_conj_rnd / M_conj must be ≥ 5 for speaker-direction to be captured)
τ_ZD_pass  = 1.0 × 10^{-18}   (exact ZD threshold in bit-identical arithmetic)
τ_gen_ratio = 10^3            (M_alt on generic encoder must be ≥ 10^3 × τ_alt for algebraic work to be demonstrable)
AUC_hi     = 0.75             (Sounio detector must clear this)
AUC_lo     = 0.55             (PyTorch detector must be below this)
OP_ratio   = 10               (orbit proximity on PyTorch / Sounio must be ≥ 10× at rupture turns)
```

### 3.3 Pre-registered predictions

| Quantity | Predicted range | Justification |
|----------|-----------------|---------------|
| `M_alt(e)` | `< τ_alt` | R5 alternativity; {R1..R7} force algebraic constraint |
| `M_mou(e)` | `< τ_mou` | R6 Moufang |
| `M_alt(e_gen)` | `≥ τ_gen_ratio · τ_alt` | generic encoder has no algebraic prior |
| `M_conj` | `< τ_conj` | R4 conjugation-equivariance |
| `M_conj_rnd / M_conj` | `≥ ratio_conj` | random pairs lack structural correspondence |
| `M_ZD(Sounio-f64)` | `= 1.00` | bit-identical CD arithmetic preserves exact ZDs |
| `M_ZD(PyTorch-f64)` | `≤ 0.05` | f64 rounding accumulates through CD doubling |
| `M_ZD(PyTorch-f32)` | `= 0.00` | single-precision rounding destroys exactness |
| `M_ZD(PyTorch-mixed)` | `= 0.00` | mixed precision destroys exactness a fortiori |
| `AUC_det(Sounio)` | `≥ AUC_hi = 0.75` | ZD-proximity is meaningful signal in bit-identical |
| `AUC_det(PyTorch-f64)` | `≤ AUC_lo = 0.55` | ZD-proximity signal degraded by rounding |
| `OP(PyTorch) / OP(Sounio)` | `≥ OP_ratio = 10` | at rupture turns, bit-identical arithmetic places h nearer G₂·Z |

---

## 4. Corpus specification

**Domain:** medical doctor–patient consultation (decided 2026-04-24). Includes ambulatory consultation, psychiatric anamnesis, medication-adherence consultation, counseling.

**Annotation manual (binding companion document):** `docs/research/ocssm_annotation_manual_v0.md`. This manual specifies operational definitions, inclusion/exclusion criteria, subtype schemas, Cohen's κ targets, and pilot protocol for the four annotation categories (a)–(d) used in falsifications F1, F2, F3b.

### 4.1 Annotation categories (formalized in companion manual)

| Category | Used in | κ target (binary) | Detail in manual |
|----------|---------|-------------------|------------------|
| (a) Self-repetition (same speaker, W_5, ≥ 80% proposition overlap) | F1a (R5 alternativity) | ≥ 0.70 | §2 |
| (b) 4-turn speaker-return (4-turn window, same speaker returns to topic) | F1b (R6 Moufang) | ≥ 0.70 | §3 |
| (c) σ-pairs (opposite-speaker reformulation, primarily natural via reflective listening) | F2 (R4 conjugation-equivariance) | ≥ 0.70 | §4 |
| (d) Rupture (contradiction / withdrawal / collapse / cascade) | F3b (detection AUC) | ≥ 0.65 | §5 |

### 4.2 Corpus choice (finalized)

**Primary:** Alexander Street Counseling & Therapy Corpus (≈ 10,000 transcriptions), conditional on institutional license obtainable through PUC-SP or São Leopoldo Mandic. Ideal because it yields high-density examples of all four categories (consultation transcripts with reflective listening, rupture incidents, repetition, and topic-return patterns).

**Secondary (fallback if primary unavailable):** hybrid corpus constructed from:
- IEMOCAP transcribed component (categories a, b; general dialog with structure).
- OSCE simulated-patient transcripts (category d; annotatable rupture patterns).
- C2-synthetic σ-pairs created by the PI (category c; documented as synthetic and reported as exploratory, not confirmatory).

**Baseline parametric reference:** MRPC / PAWS for justifying `τ_conj = 0.10` via inter-annotator paraphrase distance in general domain. Does not enter confirmatory corpus.

### 4.3 Corpus size and splits

- Pilot (reliability): 40 conversations annotated under full protocol (10 each from: ambulatory, psychiatric, adherence, counseling). Not part of confirmatory corpus.
- Confirmatory: 500 conversations anticipated (manual §7.4). Power calculation: with predicted `ΔAUC = 0.20` and expected rupture rate ~5%, ≈750 rupture-positive turns anticipated — sufficient for α = 0.05, power 0.80 on AUC difference detection.
- Splits: 70% train / 15% validation / 15% test. Test set locked (SHA-256 hashed and logged) before any model training. Each split stratified by corpus subtype (therapy / OSCE / adherence / counseling) and by rupture prevalence.

### 4.4 Annotation protocol (per manual §6)

- Two annotators + one adjudicator; trained on pilot 20 conversations; pilot κ-target ≥ 0.60 across categories before formal pilot starts.
- Independent first pass, blind to co-annotator.
- κ recomputed every 20 conversations; calibration meetings if below target.
- Adjudicator final for disagreements. Items without adjudication excluded from confirmatory dataset.

### 4.5 Limitations documented in paper

- Medical domain restriction may limit generalization of the homology claim; O-CSSM thesis is about dyadic dialogue in general, not medical specifically. Paper explicitly frames medical corpus as "first test domain, where rupture annotation is best-established in the literature."
- If primary corpus inaccessible and fallback used, the heterogeneity (therapy + simulated + counseling) weakens claims of naturalism; paper reclassifies as pre-registered case-series under fallback.
- Bilingual corpus (EN + PT-BR likely): if mixed, annotators must be bilingual; else scope restricted to one language. Decision documented before pilot.

### 4.6 Substitutes if required category unavailable

If any annotation category cannot achieve κ-target even after manual revision and annotator retraining, the corresponding falsification test (F1a, F1b, F2, or F3b) is **deferred to v2** of the paper with written justification in the limitations section. Affirmation-specific falsifications are not substituted by weaker proxies.

---

## 5. Analysis pipeline

### 5.1 Training protocol (identical across backends for F3b)

1. Fix random seed (`seed = 42`; explicitly specified in registration).
2. Fix architecture: transformer encoder → projection head producing `e(u) ∈ S^{15}` with explicit Re/Im and 𝕆/𝕆·ℓ splits.
3. Fix loss: sum of task loss (rupture classification or dialog act prediction, per sub-study) plus algebraic-constraint penalties on R4, R5, R6 (G₂-invariant penalties per R7).
4. Fix training duration: `N_epochs = 50` on training split; early stopping forbidden for confirmatory run (but logged as auxiliary). Fixed optimizer: AdamW, `lr = 1e-4`, weight decay `1e-5`.
5. Fix hardware path: Sounio backend = self-hosted compiler, `./bin/souc`, CPU f64 only. PyTorch backend = CPU only, deterministic mode enabled.

### 5.2 Metric computation

- All metrics computed on locked held-out test set.
- Each metric reported with 95% bootstrap CI over test instances (1000 resamples).
- No hyperparameter tuning on test set.

### 5.3 F3a execution

Standalone. Does not require corpus, training, or encoder. Runs deterministically on enumerated ZD pairs. Compute once in each backend; record exact residual norms.

---

## 6. Decision rules (pre-registered)

### 6.1 Confirmatory decisions

Each hypothesis is independently evaluated against its threshold. Decisions are recorded in a fixed table; no peeking between analyses.

| Hypothesis | PASS condition | Outcome if PASS | Outcome if FAIL |
|-----------|----------------|----------------|-----------------|
| H1a | `M_alt < τ_alt` AND contrast `≥ τ_gen_ratio` | (i) supported via alternativity | (i) retracted or retracted-partial |
| H1b | `M_mou < τ_mou` AND contrast `≥ τ_gen_ratio` | (i) supported via Moufang | (i) retracted or retracted-partial |
| H2 | `M_conj < τ_conj` AND `M_conj_rnd/M_conj ≥ ratio_conj` | (ii) supported | (ii) retracted |
| H3a | all four `M_ZD(·)` predictions within ±0.05 | Sounio-as-evidence supported | bit-identity rhetoric weakens or Sounio arithmetic bug |
| H3b | AUC and OP thresholds all met | (iii) supported in strong form | (iii) retracted OR arithmetic-independence established |

### 6.2 Partial-failure reporting

The paper commits *in advance* that partial failure of the hypothesis set is published as a **structural delimitation** of the homology program, not as refutation. Specifically:
- Failure of H1 alone → paper claims (ii) and (iii); revisits (i) as open question.
- Failure of H2 alone → paper claims (i) and (iii); speaker direction is non-conjugation-structural.
- Failure of H3 alone → paper retains (i) and (ii) as structural observations; retracts the "algebraic not numerical" rhetoric.
- Failure of all three → paper is published as a **pre-registered null for sedenionic rupture detection in dyadic dialogue**, with full reporting of negative results and structural post-mortem.

### 6.3 Forbidden post-hoc adjustments

The following are explicitly forbidden under this registration:
- Changing any `τ_*`, `AUC_*`, `OP_*`, or `ratio_*` threshold after results inspection.
- Replacing the test corpus after training.
- Adding additional falsification tests and reporting them as confirmatory.
- Selecting best-of-N training runs; one confirmatory run per backend, seed-fixed.
- Dropping low-`κ` annotation subsets after inspection.

Any such action renders the associated result exploratory and must be marked as such in the paper.

---

## 7. Deviation policy

If, before deposit but after drafting, this pre-registration document is modified, a new version (v1, v2, ...) is created with:
- A changelog section documenting what changed and why.
- A SHA-256 hash of the previous version archived in the changelog.
- No modification to a deposited version; only forward versioning.

If, after deposit, a change is scientifically necessary (e.g., a bug is found in the threshold calculation before training starts), a deviation-registered amendment is filed with OSF and the paper reports the change transparently.

---

## 8. Registration logistics checklist

Before any training run on the final corpus, the following must be complete:

- [ ] §4 corpus specification finalized (names, sizes, annotation schemas, κ targets, splits).
- [ ] Annotation reliability pilot completed; κ above target.
- [ ] F3a executed (standalone, requires no corpus); results recorded but paper-blinded until main analysis.
- [ ] `τ_*` thresholds re-verified via numerical propagation analysis on target hardware.
- [ ] Document hashed (SHA-256) and deposited to OSF with embargo until paper publication.
- [ ] Commit hash of this file in `sounio` repository recorded alongside OSF DOI in paper methods.
- [ ] Training scripts frozen; seed, architecture, optimizer, and loss code committed to `sounio` with tag `ocssm-preregistration-locked`.

---

## 9. Outcomes-of-publication commitments

Regardless of outcome:
- Raw metrics reported.
- Code + arithmetic logs released under the same license as the Sounio compiler.
- Corpus (if permissible under IRB/licensing) released; otherwise, enough processing metadata released for replication on equivalent corpora.
- Residual norms from F3a released as reference dataset for the field.

---

## 10. Out-of-scope (explicitly)

- Neural-architecture search, model-size scaling, or capacity-matched comparisons beyond the single specified `e_gen` baseline.
- Multi-party (non-dyadic) dialogue.
- Affect-valence or discrete-emotion classification as primary metric (treated as secondary exploration only; the paper's claim is about structural algebraic correspondence, not emotion prediction accuracy).
- Comparison against other non-Euclidean geometries of dialogue (hyperbolic, Riemannian, etc.) — noted as future work.

---

## 11. Appendices (to include before deposit)

- **Appendix A:** Full enumeration of the 84 primitive ZD pairs with explicit Baez-convention Fano labeling.
- **Appendix B:** Derivation of `τ_alt`, `τ_mou`, `τ_conj` from numerical-propagation analysis (currently values stated with justification sketches in §3.2).
- **Appendix C:** Exact code for metric computation (inline listings, language-agnostic pseudocode + Sounio reference implementation).
- **Appendix D:** Signed timestamp from OSF/arxiv/chaintime.

---

**End of v0 draft. §4 closed 2026-04-24 via companion annotation manual (`ocssm_annotation_manual_v0.md`). Remaining blocking items for v1 deposit: (i) licensing decision on Alexander Street corpus; (ii) pilot annotation of 40 conversations with reliability verification; (iii) Appendix A (84 primitive ZD enumeration) and Appendix B (τ numerical derivation) — see §11.**
