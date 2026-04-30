# DDI DrugBank Benchmark Notes

## Purpose

This note records the first standardized benchmark run for the DrugBank FAERS
DDI lane after unifying the CYP/Fano mapping and wiring a canonical manifest +
benchmark surface.

## Executed Surface

- Manifest builder: `scripts/research/prepare_ddi_manifest.py`
- Benchmark runner: `scripts/research/ddi_ossm_benchmark.py`
- Shared contract: `scripts/research/ddi_campaign_lib.py`
- Source data:
  - `data/faers_drugbank.csv`
  - `data/faers_demographics.csv`
  - `data/faers_concomitants.csv`

## Dataset Contract

- Samples: 35 CYP triples
- Fano triples: 7
- Non-Fano triples: 28
- Sequence: 3 timesteps × 8 dims (CYP basis triplet)
- Target: continuous `asymmetry`
- Split: deterministic stratified 5-fold with cyclic validation fold

The canonical manifest is written to:

- `artifacts/research/ddi_drugbank_benchmark/ddi_drugbank_manifest.tsv`

## Models

- `symmetric_linear`
  - bag-of-CYP identity + scalar covariates
- `ordered_linear`
  - ordered 3-step CYP sequence + scalar covariates
- `h_ssm`
  - quaternion-block control representation + scalar covariates
- `o_ssm`
  - octonion representation + associator feature + scalar covariates
- `o_ssm_no_assoc`
  - octonion representation without the explicit associator readout

## Headline Result

From `artifacts/research/ddi_drugbank_benchmark/overall_metrics.tsv`:

| Model | Weighted MAE | Pearson r |
|-------|--------------|-----------|
| ordered_linear | 0.0787 | 0.5050 |
| symmetric_linear | 0.0891 | 0.4543 |
| h_ssm | 0.0907 | 0.3549 |
| o_ssm_no_assoc | 0.0908 | 0.2982 |
| o_ssm | 0.0936 | 0.2909 |

## Interpretation

This is an honest first benchmark result:

1. The DDI lane is now executable as a reproducible benchmark rather than a set
   of disconnected one-off scripts.
2. The best current model is the simple ordered linear baseline, not O-SSM.
3. The explicit associator feature does not help in the current setup:
   `o_ssm_no_assoc` slightly outperforms `o_ssm`.
4. The current result does **not** support a headline claim that octonion
   non-associativity improves DrugBank FAERS asymmetry prediction.

That negative result is still scientifically useful because it narrows the
question correctly:

- order information matters
- the present hypercomplex representation is not yet extracting value beyond
  simpler ordered encodings

## Covariate Robustness

From `artifacts/research/ddi_drugbank_benchmark/covariate_robustness.tsv`:

- `ordered_linear` has the strongest raw and demographic-adjusted correlation
  with the target, but only at marginal strength (`r ≈ 0.31-0.33`).
- O-SSM-family prediction/target correlations remain weak after covariate
  adjustment.

This is consistent with the older DDI summary scripts:

- `ddi_drugbank_expanded_results.json` shows only a tiny Fano/non-Fano mean
  asymmetry gap on the DrugBank-expanded dataset.
- `ddi_covariate_results.json` shows the associator/asymmetry correlation is
  weak even before covariate adjustment.

## Repo-Level Correction

The benchmark work also exposed a mapping inconsistency:

- `scripts/research/ddi_expanded_drugbank.py`
- `scripts/research/ddi_covariate_analysis.py`

were not using the same Fano triple set.

They now share the canonical mapping from `ddi_campaign_lib.py`, and their JSON
outputs were regenerated after the fix.

## Next Useful Step

Do not over-tune the current 35-row benchmark.

The next method-first step should be one of:

1. richer order-aware non-hypercomplex baselines
2. better DDI target construction with external validation
3. a mechanistic O-SSM redesign where the algebra acts on clinically meaningful
   drug features instead of only the CYP basis triplet
