# Non-Associative Epistemic Connectomics — Phase 2 Preregistered Protocol

**Version**: 1 (2026-04-12)
**Frozen before Phase 2 begins**: YES (this file committed before any full-cohort computation is run)
**Supersedes**: nothing. Extends [PROTOCOL.md](PROTOCOL.md) (Phase 1).
**Precondition**: Phase 1 real pilot reports |Cohen's d| > 0.15 with 95% bootstrap CI excluding zero on the `p95(A)` statistic. If not met, Phase 2 is *not* run; instead the design is revised and this document is either amended with a logged change or replaced.

## Purpose

Phase 1 de-risked the pipeline on synthetic data (α-recovery monotone) and gated the infra (beagle-sounio image + `cockpit_submit_job`). Phase 2 runs the full-cohort hypothesis test on real ABIDE-I data under a null model that controls for the obvious alternative explanations.

The Phase 1 pilot operates on a 30-ROI working subset and 10 subjects. Phase 2 uses the full 200-ROI CC200 parcellation and every ABIDE-I subject that survived the preprocessor, balanced and matched at the analysis stage rather than the sampling stage. Phase 2 also runs Experiment C (sedenion zero-divisor proximity) at full scale; Phase 1's stub is replaced with the real 168-class support table.

## Scope

| Axis | Phase 1 | Phase 2 |
|------|---------|---------|
| Subjects | 10 (5 ASD + 5 TD, deterministic by `FILE_ID` sort) | Full ABIDE-I (n ≈ 1,034; `DX_GROUP == 1` vs `2`, survivors only) |
| Nodes | First 30 ROIs by CC200 order | Full 200 ROIs |
| Triples per subject | C(30, 3) = 4,060 | C(200, 3) = 1,313,400 |
| Null permutations per subject | 0 (no null in Phase 1) | 1,000 (within-subject octonion channel shuffles) |
| Experiment C | Stub, 6 supports, synthetic input | Full 168 projective zero-divisor classes, per-subject 16D features |
| Covariates | None | Site, age, sex, mean FD (scan-quality) — regressed out at subject statistic level |
| Compute | Local `./bin/souc run` | `cockpit_submit_job` indexed array on `beagle-sounio:<sha>` |

**Total Phase 2 compute budget** (upper bound):
```
Experiment B:
  1,034 subjects × 1,313,400 triples × 120 FLOP (octonion product twice) ≈ 1.6e14 FLOP  (pilot run)
  + 1,034 × 1,000 null perms × 1,313,400 × 120 ≈ 1.6e17 FLOP  (null model)
  → ~100 CPU-hours on a modern core; parallelize via cockpit array (parallelism=128)
Experiment C:
  1,034 × 168 × ~16 ops per support distance ≈ 3e6 FLOP → seconds on one core
```

## Hypotheses (restated and fixed for Phase 2)

**H1 (primary)**: ASD subjects exhibit a higher subject-level associator-field 95th percentile (`p95(A)`, defined identically to Phase 1 but over full 200-ROI triples) than TD subjects, after (a) within-subject whitening against the octonion-channel permutation null and (b) site/age/sex/FD covariate regression.

**H2 (secondary)**: ASD subjects exhibit a smaller minimum Euclidean distance from their 16D sedenion feature point to the union of the 168 projective zero-divisor support classes than TD subjects, after the same covariate regression.

**H0**: no group difference after nulls + covariates.

The primary reduction remains `p95(A)`. The `mean(A)` reduction is reported as a secondary sanity check but is not part of the test family.

## Null model (Experiment B, full specification)

For each subject `s` with octonion-labeled nodes `L^s_1, ..., L^s_200` (component 0 = 1.0, components 1..7 = first 7 Laplacian eigenvectors at that ROI, as in PROTOCOL.md § Design):

1. Compute the observed per-triple norm² vector `A^s = { ‖[L^s_i, L^s_j, L^s_k]‖² : i<j<k }`. Reduce to observed `p95(A^s)`.
2. Draw 1,000 random permutations `σ_1, ..., σ_1000 ∈ S_8 \ {id}`. For each `σ_m`:
   - Build permuted labels `L^{s,m}_i = (L^s_i[σ_m(0)], ..., L^s_i[σ_m(7)])`. Fano product is recomputed under the original convention — permuting slot labels rather than relabeling generators *per se* yields a null that preserves the marginal distribution of each eigenvector's magnitude but destroys the specific slot assignment.
   - Compute `A^{s,m}`, reduce to `p95(A^{s,m})`.
3. Subject-level null-whitened statistic:
   ```
   z^s_B = ( p95(A^s) − mean_m(p95(A^{s,m})) ) / std_m(p95(A^{s,m}))
   ```
4. Group-level test: Cohen's d on `{z^s_B : s ∈ ASD}` vs `{z^s_B : s ∈ TD}`, with 10,000-bootstrap 95% CI and two-sample KS.

**Rationale for the chosen null**: octonion channel permutation destroys the specific alignment of eigenvector-to-slot while preserving (a) each eigenvector's marginal distribution, (b) the graph topology, (c) the total number of triples, (d) the overall norm of each node label. Anything the observed statistic captures *beyond* this null is attributable to the specific octonion-slot alignment — which is the effect the hypothesis is about.

## Covariate regression

Subject-level statistic `z^s_B` is regressed on `{site_id (categorical), age, sex, mean_FD}`. Residuals `ẑ^s_B` enter the group test. Phase 1 skipped this because the pilot's n=10 is too small; Phase 2's n≈1,034 permits it.

Site, age, sex, mean-FD are pulled from `artifacts/research/abide/phenotypic.csv` (already present per `abide_preprocess.py`).

## Experiment C (sedenion zero-divisor proximity) — full spec

Per subject `s`, compute 16D feature:
```
F^s = (μ(v1), σ(v1), μ(v2), σ(v2), ..., μ(v8), σ(v8))
```
where `v1..v8` are the first 8 non-trivial Laplacian eigenvectors (Phase 1 exported 7; Phase 2 exports 8, requires a minor edit to `abide_preprocess.py` or a Phase-2-specific `abide_fetch.py --eigenvectors=8`).

Distance:
```
d_ZD(F^s) = min_{(v,w) ∈ Z168} ‖F^s − supp(v,w)‖
```
where `Z168` is the set of primitive projective zero-divisor pair supports from `artifacts/research/sedenion_zero_divisor_geometry.v1.json` (already generated by `scripts/research/generate_sedenion_zero_divisor_geometry.py`).

Group test: Cohen's d on `d_ZD(F^s)` ASD vs TD, same bootstrap + KS pipeline.

## Multiple comparison

Family: {H1, H2}. Holm-Bonferroni at family α = 0.05:
- Smallest p ≤ 0.025 → first rejection
- Next p ≤ 0.050 → second rejection
- Otherwise no rejection

Report uncorrected and Holm-corrected p-values for both.

## Subject inclusion

Match Phase 1 plus:
- Subject has `mean_FD ≤ 0.5` (standard motion scrubbing threshold).
- Subject has complete phenotypic data for `{site, age, sex, mean_FD}`; no imputation.
- Subject's preprocessed frame was successfully extracted by `abide_preprocess.py` (no exceptions).

Report n in each group after filtering; expect ~450 ASD and ~500 TD based on prior G₂ bridge run (memory `project_g2_bridge.md`).

## Analysis code

Phase 2 replaces Phase 1's hand-rolled reductions with three Sounio + Python files:

| File | Role |
|------|------|
| `experiments/non_assoc_connectomics/associator_field_full.sio` | Phase 1's `associator_field.sio` with N_WORKING=200 (not 30), emits CSV per subject. |
| `experiments/non_assoc_connectomics/null_permutations.sio` | New: for each subject, runs 1,000 channel-permuted passes, emits one CSV row per (subject, perm) pair. |
| `experiments/non_assoc_connectomics/zero_divisor_full.sio` | Phase 1's stub replaced with full 168-class support loader + per-subject distance. |
| `experiments/non_assoc_connectomics/analysis_phase2.py` | Covariate regression + z-score + group test + Holm correction + publication figure. |

Full 200-ROI N_TRIPLES = 1,313,400 will exceed the 4,060-element insertion-sort used in Phase 1. Phase 2 switches to either (a) heap-based p95 (O(n log k) where k = top-5%), or (b) partial quicksort. Default: heap. Fall back to (b) only if heap implementation in Sounio proves problematic.

## Compute orchestration

All Phase 2 jobs go through the `cockpit_submit_job` MCP tool (see `reference_cockpit_mcp.md`):

```
cockpit_submit_array({
  campaign: "non-assoc-connectomics-phase2",
  image: "ttl.sh/beagle-sounio-<sha>:24h",
  array: { size: 1034, parallelism: 128 },
  command: [
    "bash", "-c",
    "python /app/experiments/non_assoc_connectomics/abide_fetch.py && " +
    "$SOUC_BIN run /app/experiments/non_assoc_connectomics/associator_field_full.sio " +
      "-- --subject=$SLURM_ARRAY_TASK_ID > /orangefs/training/sounio/phase2/obs/$SLURM_ARRAY_TASK_ID.csv && " +
    "$SOUC_BIN run /app/experiments/non_assoc_connectomics/null_permutations.sio " +
      "-- --subject=$SLURM_ARRAY_TASK_ID --n-perms=1000 > /orangefs/training/sounio/phase2/null/$SLURM_ARRAY_TASK_ID.csv"
  ],
  resources: { cpu: "4", memory: "8Gi", gpu: 0 },
  data_mounts: ["orangefs-training"],
  timeout_seconds: 3600
})
```

If `cockpit_submit_array` does not exist at launch time, fall back to 1,034 individual `cockpit_submit_job` calls — slower to submit but equivalent semantics.

## Stopping rule

Phase 2 is a one-shot run. No peeking at intermediate group statistics before all 1,034 subjects complete.

After Phase 2:
- **Publish (Phase 3 = writeup)** if ≥1 hypothesis is rejected at Holm-corrected α = 0.05.
- **Publish null result** (also a Phase 3 task) if neither is rejected. This was the outcome of the G₂ bridge (memory `project_g2_bridge.md`) — null results are scientifically valuable and will be written up as such.
- **Do not re-run with different settings** to chase significance. Any analysis beyond this protocol is exploratory and must be labeled as such in the paper.

## Phase 3 gate

Phase 3 (writeup) proceeds only after Phase 2 completes AND PILOT_NOTES.md has been updated with the observed effect sizes + CIs for H1 and H2. The protocol's decision rules mechanically determine which of {publish positive, publish null, revise+rerun} Phase 3 executes.

## Post-freeze amendments

Any change to this document after commit must be logged in a dated section below, with justification and explicit statement of whether the change was made before or after any Phase 2 data was seen. Changes made after seeing any group-level statistic are exploratory and must be labeled as such in the paper.
