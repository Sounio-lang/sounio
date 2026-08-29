<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-algebra-b-null-retrain-audit-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-algebra-b-null-retrain-audit-2026-07-07
-->

# NeuroDyn Algebra-B Null Retrain Audit

Date: 2026-07-07
Author: Codex
Worktree: `/workspace/sounio`
Branch: `coord/lane-8c-dossier`

## Decision

`ALGEBRA_B_NULLS_WERE_FULL_PIPELINE_RETRAIN`

The Algebra-B pair-label permutation nulls were not frozen-score nulls. Each
null run built a fresh Slurm payload, compiled `examples/brain_ossm_abide.sio`,
initialized the model inside every held-out fold, trained on the permuted labels,
and then parsed fresh prediction output. Therefore `null_08` beating the true
run is not explained by scoring a frozen true-label checkpoint against permuted
labels.

This closes only item 1 of the Opus Algebra-C blocker acceptance gate. The
Algebra-C lane remains blocked by the continuous-target, generic-capacity-control,
associative-projection-specification, sign-AUC, and circularity-ceiling items in
`BLK-20260707-neurodyn-algebra-c-undercontrolled`.

## Reproducible command surface

The null runner requires `RUN_ID`, `MANIFEST_PATH`, and `OUTPUT_DIR`, then
constructs a worker-local snapshot and executes Slurm:

```bash
RUN_ID=neurodyn-algebra-b-null-08-20260706T231500Z \
MANIFEST_PATH=artifacts/research/neurodyn/synthetic/algebra_b_reformulation2_null_manifests_20260706T231300Z/pairpermnull_08_seed2026080810/pair_label_permutation_manifest.tsv \
OUTPUT_DIR=artifacts/research/neurodyn/synthetic/algebra_b_reformulation2_null_o_runs_20260706T231500Z/null_08 \
PAIRS_EXPECTED=28 \
NODE=gpuorangefs-5860-proxmox \
CPUS=2 \
MEM=4G \
READOUT_ASSOC_SCALE=0 \
READOUT_MEAN_SCALE=0 \
READOUT_DELTA_SCALE=0 \
READOUT_FLAT_SCALE=0 \
TRACE_HIDDEN_STATE=1 \
TRACE_READOUT_ALL_FOLDS=1 \
MIN_SUBJECTS=56 \
OCT_ASSOCIATIVE_PROJECTION=0 \
OCT_ASSOCIATOR_FIXED_DIM=6 \
OCT_ASSOCIATOR_SIGN_AUX=0 \
OCT_ASSOCIATOR_FIXED_DIM_AUX=0 \
OCT_ASSOCIATOR_FIXED_DIM_READOUT_AUX=0 \
OCT_PAIR_CONTRAST_AUX=0.08 \
H_TRAIN_EPOCHS=36 \
H_TRAIN_LR=0.012 \
H_CORE_LR_SCALE=0.65 \
H_WARMUP_EPOCHS=6 \
H_INIT_PRESET=1 \
H_INPUT_PROJ_MODE=4 \
H_PROJ_LR_SCALE=0.6 \
bash scripts/research/neurodyn_direct_slurm_smoke.sh
```

The emitted `abide_run_config.tsv` for `null_08` records the corresponding run
surface, including `oct_train_epochs=36`, `oct_train_lr=0.012`,
`oct_associative_projection=0`, `oct_associator_fixed_dim=6`,
`readout_assoc_scale=0`, `readout_mean_scale=0`, `readout_delta_scale=0`,
`readout_flat_scale=0`, and `min_subjects=56`.

## Evidence

### Runner executes a fresh train/test pipeline

`scripts/research/neurodyn_direct_slurm_smoke.sh` performs the following in a
new Slurm payload for each run:

- creates a fresh snapshot/output bundle from the supplied manifest;
- normalizes the manifest and runs `abide_manifest_quality_gate.py`;
- compiles `examples/brain_ossm_abide.sio` into `brain_ossm_abide.elf`;
- executes `./brain_ossm_abide.elf`;
- parses the new raw output into `results/overall_metrics.tsv`;
- records hashes in `SHA256SUMS.output`.

The runner does not load a persisted model or checkpoint before scoring.

Inside `examples/brain_ossm_abide.sio`, `run_fold(...)` calls
`init_model(is_oct)` after fold-specific mask/standardization/dropout setup and
before the epoch loop. The epoch loop then samples training subjects and calls
`update_binary(LABELS[sampled_subj], ...)`. Since the null manifest supplies
permuted `LABELS`, every null is a retrain of the full binary pipeline under
that permutation.

### Null_08 artifact integrity

Artifact:
`artifacts/research/neurodyn/synthetic/algebra_b_reformulation2_null_o_runs_20260706T231500Z/null_08`

- `run.rc`: `0`
- `sha256sum -c SHA256SUMS.output`: all checked files OK
- `brain_ossm_abide.raw.txt`: `Subjects: 56`, `ASD=28`, `Control=28`,
  `Sites: 7 grouped holdouts`
- parsed `results/overall_metrics.tsv`: O-SSM balanced accuracy `57.857143`,
  AUROC `57.372449`

### Null_08 manifest integrity

Generator summary:
`artifacts/research/neurodyn/synthetic/algebra_b_reformulation2_null_manifests_20260706T231300Z/pairpermnull_08_seed2026080810/pair_label_permutation_summary.json`

- schema: `neurodyn.pair_label_permutation_manifest.v1`
- seed: `2026080810`
- pair count: `28`
- row count: `56`
- flipped pairs: `14`
- kept pairs: `14`
- global label counts: `28/28`
- every pseudo-site retains `4/4` label balance

Direct comparison against the true manifest found:

- same 56 `subject_id` values;
- `label` changed in 28 rows;
- zero non-label column changes;
- per-site label balance retained for all seven pseudo-sites.

This is the expected pair-label exchangeability null: features, sites, row
structure, and pair balance are preserved while the orientation-label relation is
broken.

### Null envelope outcome

Final gate artifact:
`artifacts/research/neurodyn/synthetic/algebra_b_null_earlystop_decision_20260706T232906Z/algebra_b_decision_gate.json`

True O-SSM:

- balanced accuracy: `55.892857`
- AUROC: `57.621173`

Completed pair-label retrain nulls:

- count: `23`
- balanced accuracy min/mean/max: `39.821429` / `49.386646` / `57.857143`
- AUROC min/mean/max: `40.229592` / `49.199479` / `57.372449`
- nulls >= true balanced accuracy: `1/23`, plus-one p `0.083333`
- nulls >= true AUROC: `0/23`, plus-one p `0.041667`

The early-stop decision is therefore not a frozen-score artifact. The binary
Algebra-B target passed the attribution controls but failed the null envelope:

`ALGEBRA_B_ROUTE1_ATTRIBUTION_POSITIVE_BUT_NULLS_FAIL`

## Harness debt found during the audit

The Sounio raw run reports `Sites: 7 grouped holdouts`, but parsed prediction
rows for `null_08` contain only `UNKNOWN_SITE`, and parsed
`results/overall_metrics.tsv` reports `site_count=1`. This appears to be a
site-name reporting debt: `site_name_for_key(...)` knows ABIDE real-site hashes
but not the synthetic `pseudo_site_*` keys, so fold predictions fall through to
`UNKNOWN_SITE`.

This does not change the global fold-aggregated O-SSM balanced accuracy used by
the Algebra-B gate, but it blocks any site-wise or per-site claim from these
parsed tables until the synthetic-site mapping is fixed.

## Blocker status

```text
Blocker-ID: BLK-20260707-neurodyn-algebra-c-undercontrolled
Status: classified
Severity: B1
Class: evidence-gap
Owner: Codex (execution) / Opus (critique authored)
Lane: NeuroDyn Algebra-C continuous associator fidelity
Worktree: /workspace/sounio
Branch: coord/lane-8c-dossier
Files-Owned: docs/handoff/neurodyn_algebra_b_null_retrain_audit_2026-07-07.md
Files-Read-Only: docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md,
  scripts/research/neurodyn_direct_slurm_smoke.sh,
  examples/brain_ossm_abide.sio,
  scripts/research/neurodyn_pair_label_permutation_manifest.py,
  artifacts/research/neurodyn/synthetic/algebra_b_reformulation2_null_o_runs_20260706T231500Z/null_08,
  artifacts/research/neurodyn/synthetic/algebra_b_null_earlystop_decision_20260706T232906Z
Do-Not-Touch: examples/brain_ossm_abide.sio,
  scripts/gpu/prepare_abide_campaign_snapshot.sh,
  scripts/research/neurodyn_direct_slurm_smoke.sh,
  docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md
Repro: run the null_08 command surface above, then inspect `run.rc`,
  `SHA256SUMS.output`, `abide_run_config.tsv`, and `results/overall_metrics.tsv`.
Observed: Algebra-B null_08 was a full retrain and exceeded true O-SSM balanced
  accuracy: `57.857143` vs `55.892857`.
Expected: If Algebra-B promoted, all 99 retrain nulls would remain below the
  true O-SSM BA and AUROC.
Acceptance-Gate: Opus blocker item 1 is answered; items 2-6 remain required
  before any Algebra-C smoke unless the human author waives the B1 blocker.
Evidence-Level: E3
Evidence: this audit; null_08 artifact; early-stop decision gate artifact.
Fallback-Path: none
Legacy-Kept: n/a
LLM-Offload: logged:.claude/llm_offload_log.md
  outcome=NO-MATH-CONTENT non-approval raw=/tmp/llm-offload-uepfGT/
Next-Action: implement a genuinely continuous per-sequence associator target
  and support/tie audit in the prereg/script surfaces, or request Opus re-review
  of this item-1 audit before touching Algebra-C implementation.
```

## Claim boundary

This audit is synthetic-only and does not support clinical, biomarker,
biological-mechanism, treatment-response, MDD, ADHD, or broad O-SSM superiority
claims. It supports only the narrow conclusion that the Algebra-B null failure
was observed under full-pipeline retrain nulls, not under frozen scoring.
