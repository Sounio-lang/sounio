<!-- docs:meta
topic_id: repo.docs.research.neurodyn-abide-dynamic-fc-switching-prereg-2026-07-08
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.neurodyn-abide-dynamic-fc-switching-prereg-2026-07-08
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NeuroDyn ABIDE Dynamic-FC Switching Preregistration

Date: 2026-07-08

Status: preregistration roadmap and execution gate. No real-data result is
claimed in this document.

## Claim Boundary

This document defines a computational-psychiatry feasibility study for temporal
brain-state modeling in resting-state fMRI. It is not a diagnostic,
biomarker, treatment-response, biological-mechanism, clinical-decision, ASD
detection, or "the brain is octonionic" claim.

The strongest possible positive claim after a clean pass is bounded:

> Under preregistered controls, O-SSM improved prediction or calibration of
> dynamic functional-connectivity state-switching events in ABIDE-I relative to
> matched baselines.

The strongest possible negative claim after a clean fail is also bounded:

> Under this operationalization, generic recurrent or state-space baselines
> were sufficient for ABIDE-I dynamic-FC switching, and O-SSM showed no
> incremental advantage.

Diagnosis, ADOS scores, and clinical strata are secondary descriptive surfaces
only. They must not become the primary target in this lane.

## Literature Rationale

Computational psychiatry has repeatedly struggled to translate static,
context-free markers into clinically useful models. A temporal-dynamics lane is
therefore justified only if the target is itself temporal, reproducible, and
not just a diagnosis label reintroduced through a different door.

This preregistration follows four established lines of work:

1. Computational psychiatry needs time and context, not only static
   subject-level predictors.
2. Dynamic functional connectivity provides standard machinery for recurring
   resting-state brain states, including sliding windows, CAPs, HMM/HSMM
   states, dwell time, transition probability, and switching metrics.
3. ADHD and ASD dynamic-FC studies already report altered dwell, transition,
   variability, and occupancy patterns, but the literature remains sensitive to
   site, preprocessing, motion, windowing, and state-definition choices.
4. Metastability and energy-landscape models frame resting fMRI as movement
   through recurring basins or regimes. This makes state switching a more
   natural modeling target than case-control classification.

The O-SSM-specific modeling conjecture is deliberately narrower than an
anatomical, biological octonion, or mathematically derived advantage claim:

> If psychiatric-relevant brain dynamics are partly path-dependent, then a
> non-associative recurrent inductive bias may help model the order of
> transitions among metastable dynamic-FC states. This is testable only through
> held-out switching-event prediction against matched generic controls.

This is a preregistered empirical conjecture, not a theorem or derivation from
octonion algebra.

## Primary Research Question

In ABIDE-I resting-state fMRI, can O-SSM predict dynamic-FC state-switching
events better than persistence, shallow predictive controls, and matched
generic recurrent or state-space baselines?

The primary endpoint is not ASD versus control classification.

## Dataset

Primary dataset: ABIDE-I / Preprocessed Connectomes Project derivatives.

Initial derivative target:

- pipeline: CPAC;
- preprocessing variant: `filt_noglobal`;
- atlas: CC200 ROI time series;
- file surface: `*_rois_cc200.1D`;
- phenotype table: ABIDE-I phenotypic CSV with `DX_GROUP`, `SITE_ID`,
  `SUB_ID`, `FILE_ID`, age, sex, and available symptom columns.

The implementation should reuse the existing ABIDE cache and manifest
infrastructure when available, especially:

- `scripts/research/abide_prepare_manifest.py`;
- `scripts/research/build_abide_temporal_manifest.py`;
- `scripts/research/abide_manifest_quality_gate.py`;
- `examples/brain_ossm_abide.sio`;
- existing ABIDE pilot artifacts under `artifacts/research/brain_ossm_*`.

However, the current temporal manifest builder is not sufficient by itself:
it creates pooled or PCA-compressed ROI time-series windows, not dynamic-FC
state labels. This lane requires a new target-builder that computes
window-level functional connectivity, assigns external states, and derives
switching events before any O-SSM training claim is allowed.

## Subject Inclusion

Required for the confirmatory gate:

- valid phenotypic row with `DX_GROUP` in the ABIDE-I coding for ASD/control;
- valid `FILE_ID`;
- cached or downloadable CC200 `.1D` ROI time series;
- at least `min_timepoints` usable time points after parsing;
- finite ROI values after sanitization;
- sufficient windows under the selected `window_tr` and `step_tr` parameters.

Initial defaults:

- `min_timepoints = 90`;
- `window_tr = 30`;
- `step_tr = 3`;
- `min_windows = 20`.

With these defaults, a subject at exactly `min_timepoints = 90` yields
`floor((90 - 30) / 3) + 1 = 21` windows, so `min_windows = 20` is a
non-binding guardrail. It becomes binding only when smoke or sensitivity runs
change `window_tr`, `step_tr`, or `min_timepoints`.

If TR or scan length differs strongly by site, the target-builder must report
the number of time points and windows per subject and per site. A result cannot
be promoted if the switching target is mostly a proxy for site-specific scan
length.

## Dynamic-FC State Target

For each subject:

1. Load CC200 ROI time series as a matrix `T x R`.
2. Sanitize malformed rows only by documented truncation or padding.
3. Do not normalize a full subject run using future or held-out windows for a
   predictive feature. For window-level correlations, standardize ROI values
   within the current window only. Any cross-window scaler used for model
   inputs must be fit on training windows inside the split and then applied to
   held-out windows.
4. Split the time axis into sliding windows.
5. For each window, compute ROI-by-ROI Pearson correlation.
6. Apply Fisher-z transform after clipping correlations to a stable interval.
7. Vectorize the upper triangle.
8. Fit PCA on training windows only inside each split.
9. Project train and held-out windows using the training PCA.
10. Fit k-means states on training windows only.
11. Assign held-out windows to the nearest training centroids.

Default state count:

- primary: `k = 4`;
- sensitivity audit: `k in {3, 5, 6, 7}`.

Primary event target:

```text
switch_event_t = 1[state_t != state_{t-1}]
```

Primary subject-level descriptive statistic:

```text
switching_rate_subject = mean(switch_event_t)
```

Secondary descriptive statistics:

- dwell-time distribution;
- state occupancy;
- transition entropy;
- directional transition matrix;
- transition asymmetry;
- site-stratified switching-rate distribution;
- motion and scan-length sensitivity when covariates are present.

## Target-Builder Acceptance Gate

The first implementation phase ends at the target-builder audit. No model
advantage claim is allowed until this gate passes.

Required target-builder outputs:

- `dynamic_fc_window_table.tsv`: one row per subject-window with subject, site,
  label, window index, state, and split/fold metadata.
- `dynamic_fc_subject_summary.tsv`: one row per subject with window count,
  switching rate, dwell summaries, state occupancy, and missingness flags.
- `dynamic_fc_target_audit.json`: machine-readable audit.
- `dynamic_fc_target_audit.md`: reader-facing audit.
- optional PCA basis and k-means centroids per fold, saved under an artifact
  directory, not committed as source.

Executable baseline/model-gate outputs, once the target-builder passes:

- `dynamic_fc_switching_gate_predictions.tsv`: one held-out switch-event
  prediction per model, fold, subject, and event window.
- `dynamic_fc_switching_gate_summary.tsv`: fold-aggregated scoring metrics,
  including proper scoring rules and ranking metrics, for persistence,
  base-rate, logistic, generic recurrent, associative H-SSM-reservoir, and
  O-SSM-reservoir controls.
- `dynamic_fc_switching_gate.json`: machine-readable gate verdict and reasons.
- `dynamic_fc_switching_gate.md`: reader-facing gate summary.
- `dynamic_fc_switching_decision.json`: machine-readable decision over target
  readiness, model metrics, required controls, split policy, null support, and
  claim boundary.
- `dynamic_fc_switching_decision.md`: reader-facing decision summary.
- optional Sounio bridge artifacts:
  - `abide_roi_manifest.tsv`: an event-level manifest consumable by
    `examples/brain_ossm_abide.sio`;
  - `abide_run_config.tsv`: a bounded local smoke configuration;
  - `dynamic_fc_sounio_event_map.tsv`: mapping from exported event rows back
    to fold, subject, site, and window index;
  - `dynamic_fc_sounio_manifest_audit.json` and `.md`: machine-readable and
    reader-facing bridge audits.

The first executable gate is intentionally a smoke/control gate. It may return
`TRAINED_O_SSM_GATE_EXECUTED_NO_PROMOTION`, which means the target table can be
consumed by temporal predictors, generic controls, associative H-SSM controls,
lightweight NumPy O-SSM reservoir surfaces, and a deterministic candidate-trained
NumPy O-SSM surface. This is still not the full Sounio-trained O-SSM campaign
and cannot promote any O-SSM advantage claim.

`trained_ossm` and `trained_hssm` select a deterministic algebra parameter from
a bounded candidate set by training-set readout loss, then refit the same
logistic readout surface used by the other hidden-state controls. They are
optimization/plumbing controls, not substitutes for the compiled Sounio O-SSM
training path.

The compiled Sounio bridge is intentionally narrower than the Python model
gate. `scripts/research/abide_dynamic_fc_switching_sounio_manifest.py` exports
held-out dynamic-FC switch events into the existing Brain O-SSM ABIDE manifest
shape: each event row is an 8x8 sequence built only from prior state history,
with label `1` for a dynamic-FC switch and `0` for no switch. The labels are
not ASD/control diagnoses. `scripts/research/abide_dynamic_fc_switching_sounio_smoke.sh`
then runs `examples/brain_ossm_abide.sio` against that local manifest and parses
the emitted `PRED` rows. A pass here proves compiled Sounio path compatibility
and bounded execution, not a performance claim.

The gate also supports a bounded readout-retrain null via `--null-permutations`.
For reservoir surfaces, this permutes training labels and refits the logistic
readout on fixed hidden states. It is a smoke-level null for plumbing and
calibration, not a replacement for full model retraining nulls required for
promotion-scale claims.

The decision gate is:

- `scripts/research/abide_dynamic_fc_switching_decision_gate.py`.

It must downgrade smoke splits, low-subject runs, insufficient null support,
missing controls, reservoir-only O-SSM surfaces, or failed target audits to
blocked/undercontrolled/no-promotion verdicts. It may emit an exploratory
follow-up verdict only after the target audit passes, the required control
surfaces exist, the split policy matches the preregistered grouped-site gate,
null support meets the threshold, and O-SSM clears AUPRC plus at least one
proper-scoring-rule margin.

Required audit fields:

- subject count;
- site count;
- windows per subject distribution;
- windows per site distribution;
- state occupancy per split;
- switch-event prevalence overall and per site;
- number of subjects with zero switches;
- PCA variance explained;
- sensitivity to `k`;
- correlation of switching rate with number of windows;
- label balance by site;
- warnings for site dominance or degenerate states.

Target-builder blockers:

- fewer than 50 usable subjects;
- fewer than 5 usable sites;
- fewer than 20 usable windows for more than 20% of subjects;
- any state with less than 5% occupancy in the primary training folds;
- switch-event prevalence below 2% or above 80%;
- more than 25% of subjects with zero switches;
- switching rate strongly explained by window count or site before modeling;
- PCA/k-means fitted on held-out windows.
- any predictive feature normalization fitted on held-out windows or future
  windows.

If blocked, the lane may report a plumbing artifact only.

## Model Task

The model input is the sequence of dynamic-FC embeddings or state-related
features available up to time `t`. The primary prediction target is:

```text
switch_event_{t+1}
```

The model must not define the state labels it is scored against. States are
external targets built by the target-builder.

Primary metrics:

- AUPRC, because switching events may be imbalanced;
- Brier score, because calibrated switching probability matters;
- log-loss;
- AUROC as secondary only;
- hierarchical site-then-subject bootstrap confidence intervals, or a mixed
  model variance estimate, for grouped-site gates;
- site-stratified metrics.

Optional subject-level metric:

- predicted versus observed switching-rate calibration by subject.

## Required Model Surfaces

All surfaces must use the same subjects, folds, windows, state targets, seeds,
and train/test partitions.

Required before any promotion beyond readiness:

1. Persistence baseline: predicts no switch, previous event rate, or a
   site/subject base-rate variant explicitly declared before the run.
2. Logistic or small MLP baseline over the previous dynamic-FC embedding.
3. Matched GRU baseline.
4. Matched S4, Mamba-like, or Transformer baseline if available without
   introducing a separate infrastructure blocker.
5. O-SSM.

Parameter counts, hidden widths, sequence lengths, seeds, training epochs, and
early-stopping policy must be reported for each trainable model. If parameter
matching fails, the report must mark the comparison as undercontrolled.

## Nulls and Negative Controls

Required nulls:

1. Circular shift of state sequences within subject.
2. Block permutation preserving local runs.
3. Label-preserving feature noise or window-order shuffle that keeps marginal
   state occupancy but breaks temporal order.

Required rule:

- Null runs must retrain the model, not score a frozen model, unless the report
  explicitly labels the null as frozen-score exploratory.

Required negative controls:

- nuisance-only model using site, window count, age/sex when available, and
  motion if available;
- scan-length-only or window-count-only probe;
- state-frequency-only baseline with no temporal history.

Promotion is blocked if O-SSM beats the primary target but also succeeds on a
negative control that should be non-temporal.

## Splits

Recommended execution schedule:

1. Smoke: stratified subject split with site balance, for plumbing only.
2. Main gate: leave-one-site-out or grouped site-held-out split.
3. Sensitivity: repeated grouped splits if LOSO produces unstable tiny
   held-out sites.

No scientific claim can be based on the smoke split.

## Primary Decision Rule

The O-SSM dynamic-FC switching lane passes only if all of the following hold:

1. The target-builder gate passes.
2. O-SSM improves over persistence and the logistic/MLP baseline on AUPRC and
   at least one proper scoring rule.
3. O-SSM improves over the matched GRU, and over the matched S4/Transformer
   surface if that surface is included.
4. The improvement survives hierarchical site-then-subject bootstrap confidence
   intervals or a mixed-model variance estimate.
5. The improvement is not explained by site, window count, scan length, or
   nuisance-only probes.
6. Retrained nulls fail to reproduce the observed improvement.
7. Sensitivity over `k` does not reverse the interpretation.

If any of these fail, the correct status is negative or undercontrolled, not a
weaker positive claim.

## Secondary Analyses

Allowed secondary analyses after the primary target is built:

- ASD/control descriptive difference in observed switching rate;
- ASD/control descriptive difference in dwell time or transition entropy;
- site-adjusted exploratory regression of switching metrics on diagnosis;
- age and sex stratification;
- ADOS or symptom correlations when available and sufficiently supported.

These are exploratory unless separately preregistered with sample support,
covariates, missingness rules, and correction policy.

## Implementation Plan Sketch

Phase 1: target builder and audit.

- Target builder:
  `scripts/research/abide_dynamic_fc_switching_target.py`.
- Local smoke runner:
  `scripts/research/abide_dynamic_fc_switching_smoke.sh`.
- Input: ABIDE phenotypic CSV and CC200 ROI cache.
- Output: window table, subject summary, target audit, fold artifacts.
- Add a small synthetic or cached-fixture test that proves switch-event
  calculation and held-out PCA/k-means discipline.

Phase 2: readiness smoke.

- Run on a bounded ABIDE cache, initially `n >= 50` if locally available.
- Produce audit artifacts only.
- Do not train O-SSM if the target audit blocks.

Phase 3: model gate.

- Wire the target table into matched baselines and O-SSM.
- Baseline-control gate:
  `scripts/research/abide_dynamic_fc_switching_gate.py`.
- Decision gate:
  `scripts/research/abide_dynamic_fc_switching_decision_gate.py`.
- Bounded LOSO runner:
  `scripts/research/abide_dynamic_fc_switching_loso_runner.sh`.
- Compiled Sounio bridge exporter:
  `scripts/research/abide_dynamic_fc_switching_sounio_manifest.py`.
- Compiled Sounio bridge smoke:
  `scripts/research/abide_dynamic_fc_switching_sounio_smoke.sh`.
- The first O-SSM surfaces in this script are lightweight `ossm_reservoir` and
  `trained_ossm` smoke paths paired with `hssm_reservoir` and `trained_hssm`;
  promotion still requires the full grouped-site Sounio O-SSM campaign, matched
  generic controls, and retrained nulls.
- The LOSO runner can execute the compiled bridge with `RUN_SOUNIO_SMOKE=1`.
  This uses `SOUNIO_SOUC_ENGINE=lean_single` by default for the known-good
  bootstrap surface. It remains an execution bridge, not a confirmatory
  campaign result.
- Use a small `--null-permutations` value in smoke runs only; promotion requires
  pre-specified full retraining nulls at the campaign scale.
- The bounded LOSO runner is an executable local proof path, not the full
  confirmatory campaign. Its defaults cap subjects and ROI dimension for
  workspace safety while preserving grouped-site splitting, target audit, model
  gate, readout nulls, and decision gate.
- Run smoke split first.
- Run grouped site-held-out gate only after the smoke reproduces.

Phase 4: review.

- Run docs/governance sync.
- Run LLM offload for math/stats/clinical-claim boundary.
- Treat offload findings as hypotheses requiring numerical or textual
  verification before changes.

## References

- Hitchcock, P. F., Fried, E. I., & Frank, M. J. "Computational Psychiatry
  Needs Time and Context." Annual Review of Psychology.
  https://ski.clps.brown.edu/papers/Hitchcock_AnnRev.pdf
- Review of dynamic resting-state methods in neuroimaging.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12532080/
- Current methods and new directions in resting-state fMRI.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7365764/
- ADHD dynamic-FC HSMM study of dwell time, sojourn time, and transition
  probabilities.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7979530/
- ASD time-varying functional connectivity study.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5025431/
- ASD dynamic functional connectivity variability and hyper-connected pattern.
  https://pubmed.ncbi.nlm.nih.gov/31614075/
- Resting-state metastability and network switching.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5465179/
- Individual-level maximum entropy model for resting-state fMRI energy
  distributions in psychiatry.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10849576/
- Maximum entropy model and frequent brain-state switching in psychotic
  disorders.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12871292/
- Brain network dynamics and transdiagnostic symptom profiles.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12154617/
