# P5 Pre-registration — Rigorous defence of the sedenion-associator observable

**Authors**: Demetrios Chiuratto Agourakis (ORCID 0009-0001-8671-8878,
PUC-SP & São Leopoldo Mandic), Marli Gerenutti (ORCID 0000-0001-7165-646X,
São Leopoldo Mandic).

**Date frozen**: this document commits *before* any of the pre-registered
tests are executed. The git SHA of the commit that first adds this file
is the timestamp of the pre-registration.

**Subject of the tests**: the cohort-level dip-then-spike pattern in the
sedenion-associator norm `‖[a, b, c]‖` computed over 16-channel EEG
windows of the CHB-MIT public dataset, as reported in
`artifacts/research/door_f_cohort_N24/door_f_cohort.tsv`
(commit `344bc214`, N=24 patients).

---

## 1. Hypotheses

- **H1 (dip).** The minimum associator norm over the pre-ictal window set
  `{PRE30, PRE10, PRE5}` is **strictly less** than the associator norm at the
  far-baseline window `FAR` (60 s before onset). Per-patient statistic:
  `dip = (FAR_assoc − min(PRE30, PRE10, PRE5)_assoc) / FAR_assoc`.

- **H2 (spike).** The associator norm at the ictal window `IC` is
  **strictly greater** than at `PRE5`. Per-patient statistic:
  `spike = (IC_assoc − PRE5_assoc) / PRE5_assoc`.

- **H3 (co-occurrence).** Patients exhibit **both** dip and spike jointly
  at a rate exceeding the product of marginal rates under independence.

- **H4 (classification).** A two-feature classifier (`dip`, `spike`)
  trained on N−1 patients distinguishes the single ictal window from the
  six non-ictal windows of the held-out patient at **AUC > 0.5**
  (chance), with the lower bound of the bootstrap 95 % CI strictly above
  0.5.

All four hypotheses are one-sided in the directions specified. Two-sided
tests are not used.

## 2. Pre-registered tests

| Test | Hypothesis | Statistic | Null | Decision rule |
|---|---|---|---|---|
| T1 Permutation null, dip | H1 | Per-patient sign-flipped empirical p from 10 000 shuffles of the epoch-level associator trajectory; cohort p via Fisher's method. | Temporal labels uniform over the ±90 s window. | Reject H0 at combined p ≤ 0.05 after BH. |
| T2 Permutation null, spike | H2 | Same design, spike statistic. | Same as T1. | Same as T1. |
| T3 Channel-subset robustness | H1, H2 | For each patient, 100 random 16-channel draws from the ≥23 available channels; per-draw recomputation of dip and spike; report median and 95 % percentile CI. Robustness is declared only if ≥ 95 % of draws preserve the sign of the cohort-median dip and spike. | N/A (descriptive). | Must pass for both dip and spike. |
| T4 LOO classification | H4 | LeaveOnePatientOut on N=24 with a logistic regression on z-scored `(dip, spike)` features; per-window labels are `IC = 1`, all others = 0; pooled across folds. | AUC = 0.5 (DeLong bootstrap). | Lower bound of the 95 % bootstrap AUC CI strictly above 0.5. |
| T5 Co-occurrence | H3 | Observed `#(dip>0 AND spike>0)` vs Monte-Carlo under patient-level permutation of dip/spike columns independently (10 000 draws). | Factorial independence. | One-sided p ≤ 0.05. |

## 3. Corrections

- **Primary family:** T1, T2, T4. Benjamini–Hochberg with q = 0.05.
- **Secondary:** T3 and T5 are declared supporting, not primary; no
  BH correction is applied to them, but their p-values are reported.
- **Two-tailed / one-tailed:** all tests are one-tailed in the direction
  pre-specified in §1. No test is ever reported as two-tailed.
- **Stopping rule:** none — all 24 patients already have their per-window
  associator values in `door_f_cohort.tsv`; no new patients will be
  added, no patients will be excluded, and no analysis choices depend on
  the observed statistics beyond what is written above.

## 4. Exact parameters (frozen)

- **Cohort:** 24 patients in `scripts/research/door_f_cohort/chbmit_manifest.tsv`.
- **Windows:** II, FAR, PRE30, PRE10, PRE5, IC, POST, each 80 samples at
  256 Hz (312 ms), placed relative to the first seizure onset per patient.
- **Channels:** first 16 EDF channels by index (`CH_MAP = range(16)`)
  for all primary statistics. T3 draws 16-subsets uniformly at random
  *without* replacement from the available channels.
- **Normalisation:** II-train (first 64 of 80 samples) mean/std/max-abs;
  clip ±5 on probe windows; identical to the production generator.
- **Permutation resolution for T1/T2:** a ±90 s window around each
  seizure onset is split into non-overlapping 80-sample epochs
  (≈ 460 epochs per patient). Within-patient epoch labels are permuted;
  no cross-patient permutation is performed.
- **Bootstrap resolution for T4:** 2 000 stratified bootstrap resamples
  of the N=24 × 7-window table for the DeLong-equivalent AUC CI.
- **Seeds:** the random-number generator for T3 channel draws and T4/T5
  bootstraps uses NumPy `default_rng(20260420)`. Any re-run with this
  seed reproduces the full analysis bit-for-bit.

## 5. Outputs (committed to git)

- `artifacts/research/door_f_cohort_N24/permutation_nulls.tsv` (T1, T2)
- `artifacts/research/door_f_cohort_N24/channel_robustness.tsv` (T3)
- `artifacts/research/door_f_cohort_N24/loo_classification.json` (T4)
- `artifacts/research/door_f_cohort_N24/cooccurrence_null.json` (T5)
- `artifacts/research/door_f_cohort_N24/rigorous_analysis.json` (summary)
- `artifacts/research/door_f_cohort_N24/rigorous_analysis.md` (human table)

## 6. What this pre-registration does NOT do

- It does not prevent honest exploratory analyses, which will be reported
  separately and clearly labelled exploratory.
- It does not pre-register the preprint's narrative framing — only the
  inferential claims.
- It does not cover the Hessian biomarker (O-SSM) pipeline, which was
  already subjected to a seed-robustness sweep (commit `634f8316`) and
  is being reported with its honest seed-induced uncertainty.
