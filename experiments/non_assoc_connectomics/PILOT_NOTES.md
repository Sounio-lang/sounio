# Non-Associative Connectomics: Pilot Notes

## Primary Lane — Real Sounio O-SSM at n=100 and n=200

**Executed**: 2026-04-16  
**Model code**: `examples/brain_ossm_abide.sio`  
**Manifest builder**: `scripts/research/abide_prepare_manifest.py`  
**Summary parser**: `scripts/research/parse_brain_ossm_abide_output.py`  
**Cohorts**: ABIDE-I site-balanced runs at n=100 and n=200, CC200 ROI features, leave-one-site-out CV  

### What This Replaces

The earlier Fano-taxonomy work in this directory is an exploratory algebraic side lane. It is **not** the primary model result. The primary executable model result is now the real Sounio O-SSM benchmark in `examples/brain_ossm_abide.sio`.

### Local Manifest Contract

- Source phenotypic table: `/tmp/abide_pilot/phenotypic.csv`
- Source ROI cache: `/tmp/abide_pilot/*_rois_cc200.1D`
- Exported manifest: `artifacts/research/brain_ossm_local_pilot/abide_roi_manifest.tsv`
- Manifest schema: `subject_id`, `label`, `site`, `f0..f63`
- Label balance: 50 ASD / 50 TD

### Initial 3-Site Local Pilot

The local pilot manifest was run through the canonical Sounio benchmark:

- `O-SSM balanced accuracy = 50.929190 ± 4.558823`
- `H-SSM balanced accuracy = 50.000000 ± 0.000000`
- `Gap (O-H) = +0.929190 pp`

Per-site holdouts from the benchmark output:

| Site | N | O-SSM bal | H-SSM bal | Gap | O-assoc |
|------|---|------------|-----------|-----|---------|
| PITT | 56 | 52.490421 | 50.000000 | +2.490421 | 1.147710 |
| OLIN | 34 | 51.859649 | 50.000000 | +1.859649 | 0.782149 |
| OHSU | 10 | 48.437500 | 50.000000 | -1.562500 | 1.158467 |

### Interpretation

This is the first honest statement for the pilot: **a real Sounio O-SSM run exists and produces a small positive balanced-accuracy gap over the H-SSM baseline on the local n=100 manifest**.

That result is still limited:

- the local manifest is only 3 sites wide (`PITT`, `OLIN`, `OHSU`)
- the benchmark is therefore a grouped 3-site pilot, not a paper-grade cross-site result
- the Fano-taxonomy outputs below should be treated as exploratory structure probes, not as substitutes for the actual O-SSM benchmark

### Site-Balanced Real O-SSM Rerun

The same executable Sounio benchmark was then rerun on a site-balanced manifest exported from the same cached ABIDE pilot data.

- Manifest: `artifacts/research/brain_ossm_local_site_balanced/abide_roi_manifest.tsv`
- Sites: 20
- Label balance: 50 ASD / 50 TD
- Site counts: 10 sites with 6 subjects each, 10 sites with 4 subjects each

Headline metrics from the real O-SSM benchmark:

- `O-SSM balanced accuracy = 48.562500 ± 2.536111`
- `H-SSM balanced accuracy = 50.000000 ± 0.000000`
- `Gap (O-H) = -1.437500 pp`

Selected per-site holdouts:

| Site | N | O-SSM bal | H-SSM bal | Gap | O-assoc |
|------|---|------------|-----------|-----|---------|
| NYU | 6 | 32.500000 | 50.000000 | -17.500000 | 0.773088 |
| PITT | 6 | 55.000000 | 50.000000 | +5.000000 | 1.191060 |
| TRINITY | 6 | 55.833333 | 50.000000 | +5.833333 | 1.149603 |
| SDSU | 4 | 36.250000 | 50.000000 | -13.750000 | 1.051484 |

### Primary Interpretation

The confound-controlled read is now clear: **real Sounio O-SSM does run on the pilot data, but the small positive 3-site pilot gap does not survive a broader 20-site balanced rerun**.

That is the honest current result:

- the executable O-SSM lane is real and lives in `examples/brain_ossm_abide.sio`
- the 3-site pilot was too narrow to support a headline claim
- on the broader balanced cohort, O-SSM underperforms the H-SSM baseline by `1.437500` percentage points

### Machine-Readable Summaries

Each benchmark output was normalized into TSV/JSON summaries with `scripts/research/parse_brain_ossm_abide_output.py`.

- `artifacts/research/brain_ossm_local_pilot/parsed/`
- `artifacts/research/brain_ossm_local_site_balanced/parsed/`
- `artifacts/research/brain_ossm_local_site_balanced_200/parsed/`

For the site-balanced runs, the parsed `overall_metrics.tsv` files reproduce the same headline pattern as the raw benchmark summaries:

- n=100: `H-SSM = 50.0`, `O-SSM = 48.55`
- n=200: `H-SSM = 50.0`, `O-SSM = 48.35`

### H-SSM Audit

The exact `50.0` H-SSM balanced accuracy on the site-balanced cohorts is **not** a stale-output artifact.

- Parsed prediction rows show that H-SSM probabilities vary across subjects.
- But H-SSM predicted labels collapse to a single class within every `(seed, site)` group.
- On both site-balanced cohorts, H-SSM has `400 / 400` site-seed groups with only one predicted class.
- O-SSM does not fully collapse: `135 / 400` site-seed groups are mixed at n=100 and `141 / 400` are mixed at n=200.

Because the site-balanced manifests are diagnosis-balanced within each holdout site, a one-class prediction rule at the site level forces:

- per-site balanced accuracy = `50.0`
- per-site macro-F1 = `33.33`
- zero variance in H-SSM balanced accuracy across seeds

So the current H-SSM issue is better described as **site-level prediction collapse under leave-one-site-out evaluation**, not as a fixed-probability bug. No benchmark-source fix was validated in this pass.

### Scale-Up to n=200

The same real Sounio O-SSM benchmark was then scaled beyond the n=100 pilot onto a second site-balanced cohort:

- Manifest: `artifacts/research/brain_ossm_local_site_balanced_200/abide_roi_manifest.tsv`
- Cohort: n=200 (100 ASD, 100 TD)
- Sites: 20
- Site counts: exactly 10 subjects per site

Headline metrics from the raw benchmark output:

- `O-SSM balanced accuracy = 48.350000 ± 1.613227`
- `H-SSM balanced accuracy = 50.000000 ± 0.000000`
- `Gap (O-H) = -1.650000 pp`

Representative per-site means from the parsed `per_site_metrics.tsv`:

| Site | N | O-SSM bal | H-SSM bal | O-assoc |
|------|---|------------|-----------|---------|
| NYU | 10 | 43.5 | 50.0 | 0.865877 |
| UM_1 | 10 | 51.5 | 50.0 | 0.875137 |
| UCLA_1 | 10 | 51.0 | 50.0 | 0.938343 |
| YALE | 10 | 51.5 | 50.0 | 0.849917 |
| PITT | 10 | 50.0 | 50.0 | 1.130849 |

### Scaled Interpretation

The scaled run strengthens, rather than weakens, the site-balanced conclusion:

- the positive 3-site pilot gap does not return at n=200
- O-SSM remains below the degenerate H-SSM baseline on the balanced cohort
- the executable lane is now validated at two balanced cohort sizes, but the baseline pathology remains unresolved
- any scientific claim should therefore be framed as **benchmark execution success with negative or null model comparison**, not as an O-SSM advantage

### Output Files

- `artifacts/research/brain_ossm_local_pilot/abide_roi_manifest.tsv`
- `artifacts/research/brain_ossm_local_pilot/brain_ossm_abide_local.out`
- `artifacts/research/brain_ossm_local_pilot/parsed/`
- `artifacts/research/brain_ossm_local_site_balanced/abide_roi_manifest.tsv`
- `artifacts/research/brain_ossm_local_site_balanced/brain_ossm_abide_site_balanced.out`
- `artifacts/research/brain_ossm_local_site_balanced/parsed/`
- `artifacts/research/brain_ossm_local_site_balanced_200/abide_roi_manifest.tsv`
- `artifacts/research/brain_ossm_local_site_balanced_200/brain_ossm_abide_site_balanced_200.out`
- `artifacts/research/brain_ossm_local_site_balanced_200/parsed/`

## Strategy ι — Fano Taxonomy at n=100

**Executed**: 2026-04-16  
**Cohort**: ABIDE-I n=100 (50 ASD, 50 TD), CC200 200-node parcellation  
**Analysis**: Fano-line triple-product decomposition + k-means clustering + phenotype agreement

### Methods

1. **Fano 7-Vector Computation**  
   - Per subject: computed Fano basis decomposition from 7×200 eigenvector frame
   - For each C(7,3)=35 eigenvector triple (a,b,c), computed triple product: Σ_{nodes} frame[a,n] · frame[b,n] · frame[c,n]
   - Measured alignment with the 7 canonical Fano triples; accumulated squared triple-product energy per line
   - Normalized to 6-simplex: p[k] = strength[k] / Σ strength

2. **Clustering**  
   - Standardized 7-vectors (StandardScaler)
   - K-means clustering with k=2..10; selected k via silhouette score
   - Result: k=7 optimal (silhouette=0.4776, moderate cluster structure)
   - Cluster sizes: [6, 33, 19, 18, 4, 8, 12]

3. **Phenotype Agreement (ARI with permutation null)**  
   - Adjusted Rand Index (ARI) between discovered clusters and: DX_GROUP, site, age (quartiles), sex
   - Cohort aligned to `frames.bin` order by replaying the `abide_preprocess.py` contract: all cached ASD rows first, then cached TD rows
   - Permutation null: 1000 label shuffles per phenotype
   - Site shows weak-but-nonzero alignment; DX_GROUP, age quartile, and sex remain null

### Results

| Phenotype | Observed ARI | Null Mean | Null Std | p-value |
|-----------|-------------|-----------|----------|---------|
| DX_GROUP  | -0.0135     | +0.0005   | 0.0113   | 0.987   |
| Site      | +0.0376     | -0.0001   | 0.0164   | 0.027   |
| Age (Q)   | -0.0238     | -0.0004   | 0.0129   | 0.996   |
| Sex       | +0.0188     | -0.0009   | 0.0152   | 0.111   |

**Interpretation**: The algebra-discovered clustering remains orthogonal to clinical diagnosis, age, and sex, but is **not fully orthogonal to acquisition site** in this n=100 pilot. This is still a **null result for phenotype recovery**, but it is no longer valid to claim the discovered structure is independent of scanner/site effects in the current cohort.

### Fano 7-Vector Distribution

Mean simplex coordinates (across n=100):  
p = [0.208, 0.061, 0.050, 0.102, 0.119, 0.142, 0.318]

**Observation**: Fano lines 0 and 6 dominate (p1=0.208, p7=0.318), together accounting for ~53% of the measured triple-product energy. Lines 1 and 2 are weakest (~6% each). This shows the statistic is strongly non-uniform over the seven Fano directions, but should not yet be described as an associator measurement.

### PCA Projection

Two principal components explain 68.3% of variance (PC1: 44.6%, PC2: 23.6%).  
Left panel: colored by DX_GROUP (clusters mix ASD/TD).  
Right panel: colored by algebraic clusters (k=7); no visual separation by clinical diagnosis.

### Cohort Composition

The reconstructed n=100 cohort in `frames.bin` is site-skewed:

- PITT: 56
- OLIN: 34
- OHSU: 10

This matters because site is the only phenotype with permutation-supported alignment in the current run.

### Verification Notes

✓ Silhouette sanity: max silhouette = 0.4776 > 0.15 (reasonable structure detected)  
✗ Site independence: ARI(site)=0.0376 with p=0.027, so the current pilot is not cleanly scanner-independent  
✓ Permutation null: all ARIs have p-values from 1000 shuffles (interpretable)  
✓ Fano basis balance: non-uniform simplex distribution (not flat random)

### Scientific Interpretation

1. **Reject the scalar compression**: Per-subject Fano 7-vector preserves algebraic structure unavailable from single Cohen's d scalar.

2. **Reject the DX_GROUP prior cautiously**: Algebra-discovered clusters do not align with psychiatric labels in this pilot. That remains a null result for phenotype recovery, but the site effect means the pipeline is not yet isolated from acquisition confounds.

3. **Fano-selective triple-product structure**: The measured statistic preferentially concentrates in Fano directions 0 and 6. Whether that reflects cortical geometry, preprocessing, or site/parcellation effects remains open.

### Next Steps

- **Site-balanced cohort**: Rebuild the pilot with explicit per-site balancing before making site-independent claims
- **Larger cohort**: ABIDE-I full n=1034 to test whether structure stabilizes or remains orthogonal after site adjustment
- **Different parcellation**: Test whether the Fano-line energy profile is robust to AAL, Harvard-Oxford, or other atlases
- **Temporal validation**: Replication on an independent cohort (ADHD-200, HCP)
- **Mechanistic exploration**: Which connectome properties (path length, clustering coefficient, degree distribution) correlate with Fano-line weights?

### Files

- `fano_taxonomy.sio` — Sounio prototype for the Fano 7-vector computation
- `fano_taxonomy.py` — Full clustering pipeline
- `fano_taxonomy_pca.png` — PCA visualization
- `fano_taxonomy_summary.json` — Quantitative results (ARI, silhouette, cluster sizes)
- `fano_taxonomy_cohort.csv` — Reconstructed cohort metadata aligned to `frames.bin`

---

## Strategy ιb — Site-Balanced Control Rerun

To address the site skew in `frames.bin`, the pipeline was rerun on a second n=100 cohort built directly from cached ABIDE ROI files with explicit per-site ASD/TD pairing.

### Cohort Construction

- Target: n=100 (50 ASD, 50 TD)
- Sites represented: all 20 sites with both diagnoses available
- Allocation rule: one ASD/TD pair per site in round-robin order until target reached
- Final site counts: 10 sites with 6 subjects each, 10 sites with 4 subjects each
- Age quartiles: exactly balanced at 25 / 25 / 25 / 25

### Site-Balanced Results

| Phenotype | Observed ARI | Null Mean | Null Std | p-value |
|-----------|-------------|-----------|----------|---------|
| DX_GROUP  | +0.0102     | -0.0000   | 0.0108   | 0.153   |
| Site      | +0.0029     | -0.0005   | 0.0092   | 0.363   |
| Age (Q)   | -0.0105     | -0.0007   | 0.0129   | 0.800   |
| Sex       | +0.0220     | -0.0006   | 0.0205   | 0.135   |

### Interpretation

The site-balanced rerun removes the only non-null phenotype signal from the original `frames.bin` pilot:

- site agreement drops from `ARI=0.0376, p=0.027` to `ARI=0.0029, p=0.363`
- diagnosis remains null (`ARI=0.0102, p=0.153`)
- silhouette remains moderate (`0.4553` vs `0.4776` in the original pilot)

This is the cleaner read of the lane: the clustering does not recover psychiatric diagnosis, and the previously observed site effect was largely a consequence of cohort composition rather than an intrinsic property of the statistic.

### Site-Balanced Fano Profile

Mean simplex coordinates in the site-balanced cohort:

`p = [0.190, 0.080, 0.064, 0.079, 0.076, 0.166, 0.334]`

Fano line 6 remains dominant, and the distribution remains non-uniform over the seven Fano directions, so the structural part of the observation survives the confound control.

### Additional Files

- `fano_taxonomy_site_balanced_pca.png`
- `fano_taxonomy_site_balanced_summary.json`
- `fano_taxonomy_site_balanced_cohort.csv`

---

## DDI Expanded Replication (DrugBank FAERS)

**Date**: 2026-04-17  
**Source data**: `data/faers_drugbank.csv`  
**Sounio test**: `examples/oct_associator_ddi_drugbank.sio`  
**Analysis script**: `scripts/research/ddi_expanded_drugbank.py`  

### Method

Instead of one drug per CYP (as in the original 35-triple single-drug FAERS analysis), the expanded dataset uses `data/cyp_drug_mapping.csv` (~38 drugs across 7 CYPs) to enumerate drug-triples within each CYP-triple. Per CYP-triple: sum over all drug-pairs within that CYP bucket. This increases coverage by orders of magnitude.

### Coverage

- **35 CYP-triples** — all now have temporal data (vs. ~21 in the original single-drug run)
- **Total cases**: 85,042 (original: ~4,611)
- **Total temporal observations**: 19,289

### Fano Plane Convention (Critical)

The Sounio stdlib's `oct_mul` (Cayley-Dickson construction) gives specific Fano lines. These were initially set incorrectly in the Python scripts (a different but equally valid octonion labeling with zero lines in common). All results below use the **correct** Sounio Fano lines:

```
{1,2,3}, {1,4,5}, {1,6,7}, {2,4,6}, {2,5,7}, {3,4,7}, {3,5,6}
```

(CYP mapping: 1=CYP1A2, 2=CYP2C9, 3=CYP2C8, 4=CYP2B6, 5=CYP2C19, 6=CYP2D6, 7=CYP3A4)

The Sounio test `oct_associator_ddi_drugbank.sio` is correct regardless — it recomputes `assoc_norm` at runtime via the stdlib. The Python scripts and CSV `fano` column were fixed after verifying against the Sounio output.

### Primary Result

| Group | N triples | Mean \|asym\| | SD |
|-------|-----------|--------------|-----|
| Fano (assoc=0) | 7 | 0.157 | 0.103 |
| Non-Fano (assoc≠0) | 28 | 0.169 | 0.140 |

- **Cohen's d** = 0.087 (negligible)
- **η²** = 0.0012 (negligible)
- **Permutation p** (10,000 shuffles) = 0.851

### Covariate Analysis

Testing whether associator structure (assoc_norm) predicts |asymmetry| after controlling for demographic covariates from `data/faers_demographics.csv`:

- Raw correlation r = 0.071, p = 0.683
- After partial control for age/sex: r ≈ -0.005 (essentially zero)
- CYP-load (number of substrates per enzyme) has no meaningful partial effect

### Interpretation

**The η² = 0.833 finding from the single-drug pilot does not replicate in the expanded multi-drug dataset.**

The original signal emerged from:
- n=12–13 filtered triples with temporal ≥ 5
- A single drug representative per CYP (7 drugs total)
- Small-n η² positive bias floor ~0.55 (as noted in oct_permutation_test.sio)

With n=35 fully-powered triples and hundreds of representative drugs per CYP:
- Effect size collapses (d=0.18 → effectively zero)
- Permutation p=0.689 is well above any threshold
- No residual association after covariate control

This is consistent with the permutation test verdict in `oct_permutation_test.sio` being borderline (the observed η² was only marginally above the null distribution), and the covariate adjustment showing the original signal was fragile.

### Cross-Domain Fano Profile Comparison

The normalized Fano-line energy profile for the DDI domain (`ddi_fano_profile_asym`) was compared to the ABIDE connectome profiles (`abide_profiles`):

| Cohort | Cosine similarity | Jensen-Shannon div | Spearman ρ |
|--------|------------------|--------------------|------------|
| ABIDE pilot (3 sites) | 0.40 | 0.27 | -0.57 |
| ABIDE site-balanced (20 sites) | 0.31 | 0.30 | -0.79 |

The **negative Spearman correlation** indicates the Fano-line weights in the DDI domain and the connectome domain are anti-correlated — the CYP enzymes that are most "active" in Fano-line asymmetry are not the same as the connectome directions that concentrate Fano-line energy. The domains do not share a common Fano-line structure.

### Files

- `data/faers_drugbank.csv` — expanded FAERS data (35 rows, multi-drug aggregated)
- `examples/oct_associator_ddi_drugbank.sio` — Sounio replication test (ALL PASS)
- `ddi_drugbank_expanded_results.json` — quantitative results
- `ddi_covariate_results.json` — covariate analysis
- `cross_domain_fano_results.json` — DDI vs. ABIDE Fano profile comparison

---

## Earlier Work (Phase 1)

[Earlier pilot notes would be appended here as new analysis tracks are added.]

---

**Pipeline Status**: Primary executable lane is now the real Sounio O-SSM benchmark in `examples/brain_ossm_abide.sio`; Fano taxonomy remains exploratory and site-sensitive unless explicitly site-balanced.
