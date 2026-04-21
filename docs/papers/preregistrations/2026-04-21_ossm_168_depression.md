<!-- docs:meta
topic_id: repo.preregistration.ossm-168-depression
authority: repo_only
audience: researchers
registration_date: 2026-04-21
status: FROZEN_PENDING_COMMIT_HASH
source_of_truth: this file
-->

# Pre-Registration: O-SSM-Derived Hypotheses on LEMON + MODMA

**Registration date (frozen prior to any dataset access):** 2026-04-21
**Authors:** Demetrios C. Agourakis, Marli Gerenutti
**Correspondence:** demetrios@agourakis.med.br
**ORCID (D.C.A.):** 0009-0001-8671-8878
**Affiliations:** PUC-SP (Biomaterials and Regenerative Medicine); Faculdade São Leopoldo Mandic
**Repository:** github.com/agourakis82/sounio
**Registration branch:** `integration/sounio-dev-ready-base`
**Commit hash at registration:** `<FILLED BY git rev-parse HEAD AT FREEZE COMMIT>`
**SHA-256 of this document at freeze (commit 999035d8, pre-hash-insertion):** `63c2f60223657491829c38a57a2745fb079d48259bed306538ef32223ae4c93d`

## 0. Binding statement

This document is a pre-registration in the OSF sense: it specifies every analytic choice before the authors have access to any of the datasets listed in Section 3. Any deviation from this protocol that occurs during execution must be recorded in an appended "Deviations log" with justification and timestamp; deviations invalidate the confirmatory status of the affected test, which is then reported as exploratory.

The authors commit to publishing the result of every registered test, including null results.

## 1. Background

The 168 Theorem (Agourakis & Gerenutti 2026, `docs/papers/main/168-theorem.typ`) establishes three structural facts about the octonion $\mathbb{O}$ and sedenion $\mathbb{S}_{16}$ algebras:

- **T1.** The number of ordered triples $(i,j,k) \in \{1,\dots,7\}^3$ of imaginary octonion basis elements with nonzero associator is exactly $168 = |\mathrm{PSL}(2,7)|$, via the regular action of $\mathrm{Aut}(\mathcal{F}) \cong \mathrm{PGL}(3,\mathbb{F}_2)$ on ordered non-collinear triples. This count is *invariant under any relabelling* of the seven imaginary basis elements.
- **T2.** $\|[e_i, e_j, e_k]\| \in \{0, 2\}$ exactly for all basis triples, proved via $\mathbb{Z}_2^k$ grading and associativity of bitwise XOR. Basis-level non-associativity is a discrete parity phenomenon.
- **T3.** The sedenion $\mathbb{S}_{16}$ admits exactly 336 = 2×168 primitive unit zero-divisor pairs (de Marrais 2000; Cawagas 2004), localised on the $\mathbb{Z}_2^4$ upper half by the structural argument in §4.2 of the 168 Theorem paper.

Each of the three registered hypotheses below is a direct operational consequence of one of T1/T2/T3. The pre-registration is falsifiable at two levels: *each hypothesis* can fail against its clinical endpoint, and *the algebraic load-bearing claim* can fail against a base-permutation invariance control (Section 7.1) that the previous sedenion work did not include.

## 2. Hypotheses

Let the O-SSM of `.claude/prompts/door3_octonion_ssm.md` define a hidden-state sequence $\{h_t\}_{t=1}^T$ with $h_t \in \mathbb{O}^{d_\text{state}}$, $d_\text{state} = 2$, driven by $d_\text{input} = 7$ regional-average EEG signals (regions defined in Section 5). The architecture is fixed verbatim in Section 6; no hyperparameter tuning is permitted between fixation and test.

### H1 — Rumination ↔ Orbit-of-168 associator mass

**Clinical substrate.** Rumination (Nolen-Hoeksema 1991; Ehring & Watkins 2008 PTQ) is characterised by path-dependent recursion on self-referential content. Path-dependence is algebraically non-associativity.

**Feature $F_1$.** For each subject and each epoch $e$, compute
$$
F_1^{(e)} \;=\; \frac{1}{T \cdot 168} \sum_{t=1}^{T}\; \sum_{(i,j,k)\in\mathcal{O}_{168}} \big\|\,[\,h_t[i],\, h_t[j],\, h_t[k]\,]\,\big\|,
$$
where $h_t[i]$ denotes the $i$-th component of the first octonion of $h_t$ (scalar), and $\mathcal{O}_{168}$ is the fixed orbit of ordered non-collinear basis triples in $\mathrm{PG}(2,2)$ (explicitly enumerated in `formal/OctonionAlgebra.lean`). The subject-level feature is the median of $F_1^{(e)}$ across epochs.

**Primary endpoint.** Spearman rank correlation $\rho_1 = \rho(F_1, \mathrm{PTQ}_{\mathrm{total}})$ at subject level.

**Sign prediction.** $\rho_1 > 0$ (higher orbit-168 mass → higher rumination).

**Psychometric fallback hierarchy (pre-declared, decided by dataset release availability checked *before* any signal access):**
1. PTQ (Perseverative Thinking Questionnaire, Ehring et al. 2011) — total score
2. RRS (Ruminative Responses Scale, Treynor et al. 2003) — total score
3. PSWQ (Penn State Worry Questionnaire, Meyer et al. 1990) — total score

If none of the three are present in the released LEMON metadata as of registration date, H1 is not testable on LEMON; fall through to MODMA if available, else H1 is reported as untestable for reasons pre-documented in Deviations.

### H2 — Anhedonia ↔ Proximity to sedenion zero-divisor manifold

**Clinical substrate.** Anhedonia is the attenuation of expected reward signal: stimulus and reward circuitry both present, but their product vanishes. In the Cayley-Dickson tower, the first algebra admitting zero divisors is $\mathbb{S}_{16}$; the 168 Theorem §5 enumerates 336 primitive zero-divisor pairs.

**Feature $F_2$.** Extend the octonion state $h_t \in \mathbb{O}^2$ to a sedenion $\tilde{h}_t \in \mathbb{S}_{16}$ by Cayley-Dickson doubling ($\tilde{h}_t = (h_t[0], h_t[1])$ in standard pairing). Let $\mathcal{Z}_{336}$ denote the fixed set of 336 primitive sedenion zero-divisor pairs (enumerated in `stdlib/math/sedenion.sio`). Define the subject-level feature as the median across epochs of
$$
F_2^{(e)} \;=\; \min_{(z,w)\in\mathcal{Z}_{336}} \;\frac{\| \tilde{h}^{(e)} \cdot z - 0\|^2 + \| \tilde{h}^{(e)} \cdot w - 0\|^2}{\|\tilde{h}^{(e)}\|^2 + \varepsilon},
$$
with $\tilde{h}^{(e)}$ the epoch-mean sedenion state and $\varepsilon = 10^{-12}$. Lower $F_2$ = closer to the zero-divisor variety.

**Primary endpoint.** Spearman $\rho_2 = \rho(-F_2,\, \mathrm{Anhedonia\_score})$.

**Sign prediction.** $\rho_2 > 0$ (closer to zero-divisor variety → higher anhedonia).

**Anhedonia psychometric hierarchy:**
1. SHAPS (Snaith-Hamilton Pleasure Scale) — total score
2. BDI-II items 4+12+15+21 (anhedonia subset, Olino et al. 2012) — sum
3. MASQ-AD (Mood and Anxiety Symptom Questionnaire, Anhedonic Depression) — total

### H3 — Negative-valence bias ↔ Associator parity asymmetry

**Clinical substrate.** By Lemma 2 of the 168 Theorem, basis-level associator norms take only the values $\{0, 2\}$: non-associativity at the base is a $\mathbb{Z}_2$ sign flip, not a continuous rotation. At the general-octonion level this induces a parity asymmetry $\|[a,b,c]\| \neq \|[c,b,a]\|$ whose magnitude we hypothesise tracks affective polarisation.

**Feature $F_3$.** For each epoch,
$$
F_3^{(e)} \;=\; \frac{1}{T \cdot 168} \sum_{t=1}^{T}\; \sum_{(i,j,k)\in\mathcal{O}_{168}} \Big|\,\|[h_t[i], h_t[j], h_t[k]]\| - \|[h_t[k], h_t[j], h_t[i]]\|\,\Big|.
$$
Subject-level feature = median of $F_3^{(e)}$.

**Primary endpoint.** Spearman $\rho_3 = \rho(F_3,\, \mathrm{PANAS\text{-}Negative})$.

**Sign prediction.** $\rho_3 > 0$ (higher parity asymmetry → higher negative-valence bias).

**Valence psychometric hierarchy:**
1. PANAS-Negative subscale (Watson et al. 1988) — sum
2. NEO-FFI Neuroticism facet — sum
3. TAS-20 total (alexithymia, as indirect valence-processing proxy) — sum

## 3. Datasets

### 3.1 Primary: MPI Leipzig Mind-Body-Emotion (LEMON)

- **Release:** Babayan et al. 2019, *Sci Data* 6:180308. GIN repository `https://www.gin.g-node.org/juh/MPILMBB` or OpenNeuro `ds000221`, whichever is the canonical release indexed on the LEMON landing page on registration date.
- **Version pinned:** the release available at the GIN/OpenNeuro DOI snapshot dated ≤ 2026-04-21. If two versions exist, the one with higher minor-revision number is used; record which.
- **N expected:** 227 subjects (LEMON nominal).
- **Modality used:** resting-state 62-channel EEG, eyes-closed (primary) and eyes-open (sensitivity analysis only).
- **Psychometric metadata used:** PTQ/RRS/PSWQ (H1), SHAPS/BDI-II/MASQ-AD (H2), PANAS/NEO-FFI/TAS-20 (H3), BDI-II total (covariate).

### 3.2 Replication: MODMA

- **Release:** Cai et al. 2020, *Scientific Data* 7:203, HUSM MODMA dataset, 128-channel version.
- **N expected:** 53 MDD + 55 HC.
- **Modality used:** resting-state 128-channel EEG.
- **Psychometric metadata used:** PHQ-9 total (case-control primary), PHQ-9 item 1+4 (anhedonia proxy for H2 replication only).
- **Usage constraint:** MODMA is accessed *only* for replication of hypotheses that survive the primary LEMON analysis under the multiple-comparisons correction of Section 8. Hypotheses that fail on LEMON are not re-tested on MODMA.

### 3.3 Exclusion: any dataset previously accessed by either author

CHB-MIT, Siena Scalp, TUSZ, and any dataset used in `artifacts/research/door_f_cohort_N24/` are explicitly excluded from this registration. Sedenion-associator biomarker work on CHB-MIT (Phase A, April 2026) is declared invalidated and not reused.

## 4. Inclusion / exclusion at the subject level

**Included:** subjects with (a) all primary psychometric scales of at least one of H1/H2/H3 present and non-missing, (b) ≥ 4 minutes of usable eyes-closed resting-state EEG after preprocessing.

**Excluded:** subjects with documented neurological disorder other than unipolar depression (LEMON metadata flag), active psychotropic medication at scan (where available), BDI-II > 29 plus clinical diagnosis of bipolar or psychotic spectrum, or <75% retained epochs after automated rejection (Section 5).

**Sample size justification.** LEMON N ≈ 227 yields detectable Spearman effect size $|\rho| \geq 0.19$ at two-sided $\alpha = 0.05/3$ (Holm) with 80% power; this defines the minimum effect of clinical interest. Effects smaller than this are declared uninterpretable regardless of statistical significance.

## 5. Preprocessing (frozen)

Identical pipeline for both datasets, parameters pinned:

1. Re-reference: common average reference.
2. Bandpass 1–45 Hz, zero-phase 4th-order Butterworth.
3. Notch 50 Hz.
4. Downsample to 250 Hz.
5. Automated artefact rejection: amplitude threshold ±150 µV, gradient threshold 50 µV/ms; rejected segments discarded, not interpolated.
6. ICA (Infomax, fixed random seed = 20260421), component rejection by ICLabel threshold P(brain) < 0.30.
7. Epoch segmentation: non-overlapping 4-second windows.
8. Per-epoch z-score normalisation per channel.

**Regional aggregation to 7 octonion basis inputs.** Channels are averaged within seven canonical scalp regions, pre-declared as:

| Basis index | Region | Canonical channels (10-20 superset) |
|---|---|---|
| $e_1$ | Left frontal | Fp1, F3, F7, FC5 |
| $e_2$ | Right frontal | Fp2, F4, F8, FC6 |
| $e_3$ | Left temporal | T7, TP9, CP5 |
| $e_4$ | Right temporal | T8, TP10, CP6 |
| $e_5$ | Central | Fz, Cz, FC1, FC2 |
| $e_6$ | Left parieto-occipital | P3, P7, O1, PO9 |
| $e_7$ | Right parieto-occipital | P4, P8, O2, PO10 |

This regional assignment is the one "labelling choice" that the invariance control (Section 7.1) explicitly tests.

## 6. O-SSM architecture (frozen)

Architecture specified by `.claude/prompts/door3_octonion_ssm.md`, with the following pinned hyperparameters:

- $d_\text{input} = 7$ (regional averages), $d_\text{state} = 2$ (two octonion units = 16 real values), $d_\text{output} = 1$ (subject-level scalar; not used for classification, only for forward dynamics).
- State transition: $h_t = \sigma(A \otimes h_{t-1} + B \otimes x_t)$ with $A \in \mathbb{O}^2$ diagonal, $B \in \mathbb{O}^{2 \times 7}$, $\sigma$ component-wise sigmoid.
- $\otimes$ = Cayley-Dickson octonion multiplication (non-associative, non-commutative), verbatim as `stdlib/algebra/octonion.sio`.
- Parameter initialisation: $A$ and $B$ components drawn i.i.d. from $\mathcal{N}(0, \sigma_0^2)$ with $\sigma_0 = 0.1$, seed = 20260421 + subject_index. Seeds are deterministic per subject; no parameter learning is performed for this pre-registration (features are read out from untrained dynamics, identical across subjects modulo per-subject seed offset).
- Uncertainty: every octonion component carries $\mathrm{Knowledge}\langle f64 \rangle$ (GUM, JCGM 100:2008) per `docs/papers/main/168-epistemic-preprint.typ`; feature values are reported with propagated $u_c$.

**No training.** Features $F_1, F_2, F_3$ are computed from untrained forward dynamics. This removes optimiser-driven researcher degrees of freedom. A follow-up confirmatory registration for trained O-SSM is out of scope here.

## 7. Control conditions (run once, reported regardless of outcome)

Each control below is run in parallel with the confirmatory test. All three controls must be passed for the respective hypothesis to be reported as confirmed; failure of any single control demotes the hypothesis to exploratory in the final manuscript.

### 7.1 Base-permutation invariance (labelling control)

For each subject, repeat feature extraction $K = 30$ times, each time applying one of the 30 inequivalent Fano labellings of the 7 imaginary basis elements (enumerated in `formal/OctonionAlgebra.lean` under `theorem fano_labelling_orbits_count`). Each labelling is a permutation of $\{e_1,\dots,e_7\}$ that preserves Fano incidence; each of the 30 classes gives one representative.

- **Pass criterion H1, H3:** distribution of subject-level $\rho$ across the 30 labellings has interquartile range $\leq 0.05$ and mean within $\pm 0.05$ of the canonical labelling value. (Features built on the full 168-orbit should be invariant by T1; non-invariance signals an implementation leak.)
- **Pass criterion H2:** analogous invariance on sedenion basis labellings (336 primitive pairs form 2 orbits of size 168 under $\mathrm{PGL}(3,\mathbb{F}_2)$; we sample 30 labellings at random with fixed seed 20260421).

### 7.2 Algebra-substitution control

For each hypothesis, re-compute the feature with $\otimes$ replaced by:

(a) quaternion multiplication (associative, $\dim = 4$; zero-pad 7 inputs to 3 active channels, rest masked) — null prediction: $\rho$ collapses toward 0 for H1, H3 (no associator structure); H2 untestable (no zero divisors).
(b) real-valued matrix multiplication of matched parameter count — null prediction: all three $\rho$ collapse.
(c) random fixed tensor product with same sparsity pattern as $\otimes$, seed 20260421 — null prediction: all three $\rho$ collapse.

**Pass criterion:** canonical octonion $|\rho|$ exceeds each of (a), (b), (c) by $\geq 0.10$ after Spearman Fisher-z transform, on the same subjects.

### 7.3 Subject-level bootstrap

Cluster bootstrap at the subject level, $B = 10{,}000$ resamples, seed 20260421. Report 95% BCa confidence interval for each $\rho_k$. Window-level bootstraps are explicitly prohibited as primary inference.

## 8. Multiple-comparisons correction and decision rule

Three primary tests (H1, H2, H3) on LEMON, corrected by Holm–Bonferroni at family-wise $\alpha = 0.05$. Replication on MODMA is conditional and tests only those hypotheses that reach corrected significance on LEMON; the replication family is corrected independently.

**Decision rule for each hypothesis $H_k$:**

- **Confirmed:** $\rho_k$ significant under Holm on LEMON, sign matches pre-declared direction, clinical effect $|\rho_k| \geq 0.19$, all three controls of Section 7 pass, and the same sign replicates directionally on MODMA (nominal $p < 0.05$ uncorrected is sufficient for directional replication).
- **Directional but not confirmed:** significance on LEMON with correct sign but failure of any control or failure of MODMA replication.
- **Null:** does not reach corrected significance on LEMON.
- **Anomalous:** significance on LEMON but wrong sign — reported as failed prediction of the theoretical model.

All four outcomes are publishable under this protocol.

## 9. Covariates and nuisance controls

Subject-level partial Spearman correlations are additionally reported, controlling separately (not jointly) for:

1. Age (LEMON: continuous; MODMA: binned per release).
2. Sex at birth (binary, as released).
3. BDI-II total (for H1 and H3 only; controlling for overall depression severity isolates the specific dimension).
4. Head-motion / rejected-epoch fraction (as a data-quality covariate).

Partial correlations are reported alongside raw $\rho_k$ but the raw value is the primary endpoint; partials are decision-relevant only to distinguish "specific to dimension" from "generic severity" in the Discussion.

## 10. Deviation reporting

A `DEVIATIONS.md` file will be appended in the same directory as this protocol. Every deviation must record: timestamp, commit hash at time of deviation, description, justification, and explicit reclassification of the affected test to exploratory. No deviation is retrospective.

## 11. Freeze protocol

This file is frozen by the following sequence, executed before any dataset is downloaded:

1. Final edit committed to branch `integration/sounio-dev-ready-base`.
2. `sha256sum docs/papers/preregistrations/2026-04-21_ossm_168_depression.md` is computed and pasted into the `SHA-256` field at the top of this document in a follow-up commit whose message contains the hash.
3. The commit SHA containing the hash is pasted into the `Commit hash at registration` field in a third commit, which is then tagged `prereg/ossm-168-depression-v1`.
4. The tag is pushed to the public remote (with the user's explicit authorisation).
5. Only after the tag is public does any member of the team execute `git clone` / `datalad get` on LEMON or MODMA.

Any access to LEMON or MODMA signal data prior to step 4 voids the confirmatory status of the entire registration.

## 12. Publication commitment

The authors commit to submitting a manuscript reporting the outcome of all three hypotheses (and all three controls) regardless of whether any, all, or none are confirmed. Preferred venue order: *Imaging Neuroscience*, *NeuroImage*, *Scientific Reports*; registered-report track where available.

## 13. Key references

- Agourakis, D. C. & Gerenutti, M. (2026). *The 168 Theorem: a PSL(2,7)-orbit count of non-associative octonion basis triples.* `docs/papers/main/168-theorem.typ`.
- Babayan, A. *et al.* (2019). A mind-brain-body dataset of MRI, EEG, cognitive, emotional, and peripheral physiological data from young and old adults. *Sci Data* 6:180308.
- Cai, H. *et al.* (2020). A multi-modal open dataset for mental-disorder analysis. *Sci Data* 7:203.
- Cawagas, R. E. (2004). On the structure and zero divisors of the Cayley–Dickson sedenion algebra. *Discuss. Math. Gen. Algebra Appl.* 24:251–265.
- de Marrais, R. P. A. (2000). The 42 assessors and the box-kites they fly: diagonal axis-pair systems of zero divisors in the sedenions' 16 dimensions. arXiv:math/0011260.
- Ehring, T. *et al.* (2011). The Perseverative Thinking Questionnaire (PTQ). *J Behav Ther Exp Psychiatry* 42:225–232.
- JCGM 100:2008. *Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM).*
- Nolen-Hoeksema, S. (1991). Responses to depression and their effects on the duration of depressive episodes. *J Abnorm Psychol* 100:569–582.
- Pion-Tonachini, L. *et al.* (2019). ICLabel: an automated electroencephalographic independent component classifier. *NeuroImage* 198:181–197.
- Snaith, R. P. *et al.* (1995). A scale for the assessment of hedonic tone the Snaith–Hamilton Pleasure Scale. *Br J Psychiatry* 167:99–103.
- Watson, D., Clark, L. A., & Tellegen, A. (1988). Development and validation of brief measures of positive and negative affect: the PANAS scales. *J Pers Soc Psychol* 54:1063–1070.

---

*End of pre-registration. Any text below this line is outside the registered protocol.*
