<!-- docs:meta
topic_id: repo.docs.research.neurodyn-ossm-adhd-dimensional-prereg-2026-07-07
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.neurodyn-ossm-adhd-dimensional-prereg-2026-07-07
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NeuroDyn O-SSM ADHD Dimensional State Probe Preregistration Roadmap

Date: 2026-07-07

Claim boundary: this is a computational-psychiatry research plan and
readiness contract. It does not make a diagnostic, biomarker, treatment
response, clinical-decision, biological-mechanism, or "the brain is
octonionic" claim. A clean result may only support a bounded inductive-bias
statement about whether an O-SSM hidden trajectory exposes preregistered
ADHD-dimensional structure beyond matched controls.

## Locked Research Question

Do O-SSM hidden-state trajectories from resting-state fMRI contain
site-held-out structure aligned with dimensional ADHD symptom subscales,
above matched non-hypercomplex and associative controls?

The first endpoint is not diagnostic classification. Diagnosis is retained
only for stratification, sanity checks, and secondary descriptive summaries.

## Primary Dataset

Primary source: ADHD-200 / Preprocessed Connectomes Project resting-state
fMRI derivatives.

Rationale: the official ADHD-200 materials describe multi-site resting-state
fMRI, anatomical data, diagnostic status, dimensional ADHD symptom measures,
age, sex, IQ, medication status, and quality-control annotations. That matches
the selected first target better than smaller task-fMRI ADHD datasets.

Access rule: if phenotypic metadata or derivatives require NITRC login or a
manual download step, the pipeline must accept `--phenotypic-csv` and
`--roi-dir` inputs and record that access condition. It must not silently
replace missing symptom subscales with diagnosis labels or synthetic values.
Before a real-data pilot, run
`scripts/research/adhd200_data_access_audit.py` to prove that the local input
pair is ADHD-200 phenotypic metadata plus matching ROI time series. The audit
is not a downloader and must report `blocked` rather than falling back to ABIDE
or synthetic smoke data.

Bounded public-cache bootstrap: `scripts/research/adhd200_s3_bootstrap.py`
can fetch official PCP/FCP-INDI ADHD-200 phenotypic CSVs plus a deliberately
limited CC200 ROI subset from the public `fcp-indi` S3 bucket. It defaults to a
small pilot cohort and requires `--allow-full-download` before larger pulls.
The bootstrap is an access convenience only; the audit and readiness gates
remain authoritative before the O-SSM pilot.

Fallback source: OpenNeuro ADHD task datasets may be used only as a pipeline
pilot or external sanity check. They do not replace ADHD-200 for this v1
dimensional state-probe lane.

Source anchors:

- ADHD-200 / INDI landing page:
  `https://fcon_1000.projects.nitrc.org/indi/adhd200/`
- ADHD-200 phenotypic key:
  `https://fcon_1000.projects.nitrc.org/indi/adhd200/general/ADHD-200_PhenotypicKey.pdf`
- PCP ADHD-200 preprocessed derivatives:
  `https://preprocessed-connectomes-project.org/adhd200/`
- PCP download instructions:
  `https://preprocessed-connectomes-project.org/adhd200/download.html`
- Competition cautionary baseline: personal characteristics alone were strong
  enough to be competitive, so site, age, sex, IQ, medication, and motion/QC
  must be explicit nuisance/control surfaces, not afterthoughts.

## Primary Phenotypes

Primary dimensional endpoints, in order:

1. Inattention subscale.
2. Hyperactivity/impulsivity subscale.
3. ADHD total or index score.

If the selected ADHD-200 phenotypic CSV lacks a required subscale, the
confirmatory path is blocked. The run may continue only as a readiness or
pipeline-smoke artifact with an explicit `missing_primary_phenotype` failure.

Required covariates for readiness:

- age;
- sex;
- IQ when available;
- medication status when available;
- site;
- motion/QC summary when available.

## Model Surfaces

All model surfaces must use the same subjects, feature layout, leave-site
splits, seeds, train/test policy, and phenotype table.

Required before any promotion beyond readiness:

1. O-SSM hidden trajectory.
2. H-SSM or another associative hypercomplex control already exposed by the
   current benchmark surface.
3. A real-valued generic capacity baseline, initially a GRU with matched
   subject features, split policy, and seed policy.
4. A wide GRU warning control with doubled hidden width. If absent, the result
   is undercontrolled and exploratory.
5. Raw-feature and nuisance-only probes for leakage and shortcut auditing.

While `BLK-20260707-madaros-f64-arg-abi-oct-mul` remains open, O-SSM proof
runs should use `SOUNIO_SOUC_ENGINE=lean_single` and record that compiler path.

## Primary Endpoint

Primary endpoint: site-held-out hidden-state geometry aligned with ADHD
subscale structure.

The first operational statistic is a frozen hidden-state geometry probe:
within each model and seed, evaluate whether final hidden states preserve
phenotype-relevant structure under leave-one-site-out splits. The report must
include:

- fold count and site count;
- rows per held-out site;
- per-subscale target support and missingness;
- hidden-state nearest-centroid or distance-based summaries;
- permutation/null behavior with retrain-versus-frozen status explicitly
  recorded.

This endpoint is chosen because it asks whether the recurrent state learned a
stable representation before asking whether a clinical label can be predicted.

Operational export surface: `scripts/research/neurodyn_adhd_dimensional_state_probe.py`
joins O-SSM `STATE_TRACE` rows to the rich ADHD-200 manifest, writes
`adhd_dimensional_dynamic_features.tsv`, and evaluates site-held-out ridge
probes for hidden-state, covariate-only, and static input-summary surfaces. The
exported dynamic features are the first reusable feature-extractor artifact;
they are not a diagnostic model.

Data-access preflight: `scripts/research/adhd200_data_access_audit.py` scans
explicit or local candidate paths, optionally probes public source pages, and
writes `adhd200_data_access_audit.{json,md}` with a `ready`, `partial`, or
`blocked` status plus the exact smoke command when a usable input pair is
present.

Public S3 cache bootstrap: `scripts/research/adhd200_s3_bootstrap.py` writes a
bounded cache containing `adhd200_phenotypic.csv`, `rois/*_rois_cc200.1D`, and
`adhd200_s3_bootstrap_summary.json`. The intended sequence for a small real
pilot is:

```bash
python3 scripts/research/adhd200_s3_bootstrap.py \
  --output-dir /tmp/adhd200_pcp_pilot --max-subjects 24 --overwrite
python3 scripts/research/adhd200_data_access_audit.py \
  --phenotypic-csv /tmp/adhd200_pcp_pilot/adhd200_phenotypic.csv \
  --roi-dir /tmp/adhd200_pcp_pilot/rois \
  --output-dir /tmp/adhd200_pcp_pilot/access_audit --overwrite
```

Small pilot runner: `scripts/research/adhd200_dimensional_pilot_smoke.sh`
chains manifest preparation, readiness gating, legacy O-SSM execution, dynamic
state probing, generic recurrent baseline execution, optional readout-alignment
summarization, and artifact hashing. It writes only under `OUTPUT_DIR` and
defaults to `SOUNIO_SOUC_ENGINE=lean_single` while the Madaros f64 ABI blocker
remains open.

Generic pilot baseline surface:
`scripts/research/adhd200_generic_recurrent_baseline.py` reads the same rich
manifest and evaluates `gru_reservoir` plus `gru_reservoir_wide` under the same
leave-site-out dimensional ridge-readout protocol. This is a lightweight
GRU-style recurrent reservoir control for small pilot readiness. The same
script can also run `trained_rnn` and `trained_rnn_wide`, small NumPy-trained
real-valued recurrent regressors fit separately by phenotype and site-held-out
fold. Those trained controls are stronger than the frozen reservoir check, but
still do not replace the required full trained GRU/S4/Transformer baseline
suite before promotion-scale claims.

Pilot decision surface:
`scripts/research/adhd200_dimensional_pilot_decision_gate.py` reads the
readiness JSON, O-SSM/H-SSM dimensional state summary, and generic recurrent
baseline summary. It emits one of the bounded pilot outcomes:

- `PILOT_EXPLORATORY_O_SSM_SIGNAL_ALL_PRIMARY`;
- `PILOT_MIXED_EXPLORATORY`;
- `PILOT_NEGATIVE_CONTROLS_SUFFICE`;
- `PILOT_NO_ROBUST_SIGNAL`;
- `UNDERCONTROLLED`;
- `BLOCKED` or `BLOCKED_READINESS_GATE_FAILED`.

This is the required answer to the planning question "only then decide if
there is a clinical story." A positive gate is still only an invitation to
deeper follow-up; a negative or mixed gate is equally publishable as evidence
that generic/nuisance dynamics suffice in the tested setting.

Required join and null safeguards:

- the rich manifest and O-SSM compatibility view must share the same
  `subject_universe_sha256`;
- `STATE_TRACE` label fields must match the manifest row addressed by
  `subject_index`;
- `STATE_TRACE` site fields must either match the site text or the same DJB2
  site hash used by the Sounio manifest loader;
- every compared model must expose the same `(seed, subject_id)` universe;
- the dimensional probe must report phenotype-permutation nulls over frozen
  recurrent states and must label the null as non-retrained;
- the default null summary is one-sided in the positive-association direction.
  `null_spearman_p_ge_mean` is the mean of per-seed permutation p-values, not a
  combined across-seed p-value; per-seed rows remain the canonical audit record.

## Mandatory Mechanistic Audit

Every primary run must emit and summarize readout/alignment traces. The audit
must report:

- pre/post training alignment;
- train versus holdout behavior;
- score margin or calibration drift where available;
- whether shortcut readouts or nuisance covariates explain the effect.

If hidden-state geometry looks positive but alignment traces collapse,
overfit, or track only site/motion nuisance, the run is blocked from promotion.

## Secondary Analyses

Secondary analyses are allowed only after the primary hidden-state and
alignment audits complete:

- frozen hidden-state linear readout for each ADHD subscale;
- diagnosis-balanced sanity check;
- symptom total/index regression;
- medication-status and motion sensitivity;
- cross-site heterogeneity summary.

None of these secondary analyses can promote a clinical claim by themselves.

## Data Contract

The ADHD-200 preparer writes a rich manifest. The default feature layout is
`8x8_temporal_roi_block`: split each subject's ROI time series into eight
contiguous temporal windows, then average normalized ROI activity inside eight
ROI blocks for each window. This preserves a temporal sequence for the O-SSM
smoke path. The older `8x8_laplacian_eigenblock` projection is allowed only as
an explicit ablation because it is closer to a static connectome summary.

Rich manifest:

```text
subject_id  label  site  inattention  hyperactivity_impulsivity
adhd_total  age  sex  iq  medication_status  qc_status  mean_fd
source_file_id  roi_path  n_rois  n_timepoints  f0 ... f63
```

It may also write an O-SSM compatibility view:

```text
subject_id  label  site  f0 ... f63
```

The compatibility view is for current Sounio model execution only and encodes
`label` as `1=ADHD, 0=TD` because the legacy `brain_ossm_abide.sio` loader
accepts numeric binary labels but not the literal string `ADHD`. The rich
manifest and readiness JSON are the source of truth for phenotype validity.

## Readiness Gate

A dataset package is ready for smoke only if:

- at least two sites survive QC;
- at least one ADHD and one control subject survive globally;
- each requested primary subscale is present and non-constant among available
  rows;
- missingness for every primary subscale is below the configured threshold;
- all feature values are finite;
- feature variance and nonzero fraction pass thresholds;
- the package records source URLs or manual-access notes.

Failure of this gate blocks confirmatory O-SSM runs but may still produce a
readiness artifact for handoff.

## Promotion Rule

The strongest possible v1 claim after a clean pass is:

> In the preregistered ADHD-200 dimensional state-probe setting, O-SSM hidden
> trajectories preserve held-out ADHD subscale-aligned structure better than
> matched controls under the stated split, null, and leakage audits.

Forbidden claims:

- O-SSM diagnoses ADHD;
- O-SSM is a biomarker;
- ADHD is octonionic or non-associative;
- the result is treatment-response evidence;
- the result establishes a biological mechanism.

## External Review Requirement

Before committing or promoting this lane, run the repository LLM-offload
policy for external-facing clinical/research artifacts. At minimum, request a
clinical-pathway/research-claim review and record the provider, target, and
outcome in `.claude/llm_offload_log.md`.
