<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-ab-fixed-octmul-reaudit-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-ab-fixed-octmul-reaudit-2026-07-07
-->

# NeuroDyn A/B Fixed-Octonion Re-Audit

Date: 2026-07-07
Owner: Codex
Branch: `coord/lane-8c-dossier`
Inputs:

- `BLK-20260707-neurodyn-oct-mul-not-normed` fixed and independently verified.
- `BLK-20260707-madaros-f64-arg-abi-oct-mul` remains open for default Madaros
  runtime proof, so Slurm smokes use the sanctioned `lean_single` worker path.

## Decision

The corrected octonion table changes the NeuroDyn synthetic A/B status to
negative.

- Algebra-B fixed-dim6 line is now `ALGEBRA_B_ROUTE4_TERMINAL_FIXEDDIM6_NEGATIVE`.
- The Fano/noncommutative-temporal candidate corresponding to the earlier
  "Algebra-A" lineage is also negative on the corrected model path.
- No Algebra-C smoke is authorized by this re-audit; Algebra-C remains blocked
  on the previously recorded controls.

## Algebra-B Re-Audit

Corrected package:

`artifacts/research/neurodyn/synthetic/algebra_b_fixed_octmul_reaudit_20260707T000000Z`

Parameters matched the prior reformulation-2 line:

- seed: `2026070802`
- pairs/sites/seq_len: `28 / 7 / 32`
- triple source: `scaled_unit`
- target associator component/sign: `6 / +1`
- scale jitter: `0.12`
- noise std: `0.0`
- reformulation attempt/max: `2 / 2`

Manifest changes relative to the broken table:

- non-associative triples available: `122 -> 168`
- target values: `56 / 56` distinct
- target tie fraction: `0.0`
- label and target signs balanced globally and per pseudo-site
- data audit `bad_pair_count: 0`
- raw-flat shortcut BA: `32.142857`

The legacy multi-dim balance gate still returns
`ASSOCIATOR_MANIFEST_BALANCE_GATE_NOT_READY` because this B design is fixed to
target component 6. That is a gate/assay mismatch, not a failure of the sign
fix.

Corrected Slurm true smoke:

`artifacts/research/neurodyn/synthetic/algebra_b_fixed_octmul_true_20260707T211500Z`

Run path:

- `scripts/research/neurodyn_direct_slurm_smoke.sh`
- worker engine: `SOUNIO_SOUC_ENGINE=lean_single`
- run rc: `0`
- subjects: `56`
- input TSV preserves seven pseudo-sites, eight rows each

Metrics:

| model | BA | AUROC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| O-SSM | 53.482143 | 52.110969 | 0.463579 | 49.827700 |
| H-SSM | 55.267857 | 54.413265 | 0.445783 | 49.827700 |

Decision gate:

`artifacts/research/neurodyn/synthetic/algebra_b_fixed_octmul_decision_20260707T211800Z`

```text
Decision: ALGEBRA_B_ROUTE4_TERMINAL_FIXEDDIM6_NEGATIVE
Route: 4
Interpretation: O-SSM is below 55% and the preregistered reformulation budget is exhausted.
```

Because O-SSM is subthreshold after attempt 2/2, the four-route contract does
not require A8, associative projection, or 99 null expansion to close this
fixed-dim6 B line.

## Algebra-A / Fano Candidate Re-Audit

No canonical repo file was found with the literal label `Algebra-A`. The closest
lineage is the Phase-1 Fano/noncommutative-temporal synthetic order family
referenced in the SOTA note and older artifacts. This report re-audits that
candidate because it uses the O-SSM `do_oct_mul` runtime path and includes the
affected Fano line `(2,5,7)`.

Corrected candidate package:

`artifacts/research/neurodyn/synthetic/fano_relation_counterbalanced_noise0005_pairs56_seed2026070820_20260707T191500Z`

Manifest facts:

- pairs/sites/seq_len: `56 / 7 / 32`
- records: `112`
- noise std: `0.0005`
- Fano lines include `(2,5,7)`
- paired invariants pass: mean/delta/start/end/energy/multiset unchanged
- label balance: `56 / 56`, balanced per Fano-line pseudo-site

Corrected Slurm true smoke:

`artifacts/research/neurodyn/synthetic/fano_relation_fixed_octmul_true_20260707T212500Z`

Run path:

- `scripts/research/neurodyn_direct_slurm_smoke.sh`
- worker engine: `SOUNIO_SOUC_ENGINE=lean_single`
- run rc: `0`
- subjects: `112`
- input TSV preserves seven Fano-line pseudo-sites, sixteen rows each
- run config used the strongest prior relation/readout-correction candidate
  surface: `oct_relation_target_aux=0.04`,
  `oct_relation_margin_aux=0.02`,
  `oct_relation_readout_correct_steps=56`,
  `oct_binary_lr_post_scale=0.5`

Metrics:

| model | BA | AUROC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| O-SSM | 36.875000 | 37.236926 | 0.629078 | 49.827700 |
| H-SSM | 26.785714 | 29.017857 | 0.729623 | 49.827551 |

Conclusion: the corrected Fano/noncommutative-temporal candidate remains
negative and is far below the synthetic promotion threshold.

## Site-Reporting Caveat

Both corrected Slurm runs preserved the intended site labels in
`abide_roi_manifest.tsv`, but the model output parser summarized per-site rows
as `UNKNOWN_SITE` and `site_count=1`.

Observed input counts:

- Algebra-B: seven pseudo-sites, eight rows each
- Fano candidate: seven Fano-line pseudo-sites, sixteen rows each

Impact:

- The global subject-level BA/AUROC values above are still usable for the
  preregistered threshold decisions.
- No site-wise or leave-site stability claim should be made from these corrected
  runs until the site reporting regression is fixed.

Suggested follow-up blocker if site-wise claims are needed:

```text
Blocker-ID: BLK-20260707-neurodyn-fixed-octmul-site-reporting-unknown
Severity: B2
Class: harness-routing / parser-output
Evidence-Level: E2 local Slurm outputs
Owner: Codex / NeuroDyn harness lane
Observed: input manifests preserve seven sites, but corrected run summaries emit UNKNOWN_SITE/site_count=1.
Expected: per-site metrics preserve pseudo_site_* / fano_line_* labels.
Acceptance-Gate: rerun one corrected B or Fano smoke and show per_site_metrics.tsv with seven sites.
```

## Hash Evidence

Algebra-B corrected true smoke:

```text
run.rc = 0
b95a6d44917d988ea92f88290b7d09674a2e73eeca147e82a7e2bf5ec0b917eb  results/overall_metrics.tsv
922da08c7e6881ee0cc2f599ae8682a47d8c6dcfa9a649a74d0250d9d8926e2a  abide_roi_manifest.tsv
e96021d316d557654bab8526dd64f2d2557c2d6edb1381fce287a2bf36e87960  brain_ossm_abide.raw.txt
```

Fano corrected true smoke:

```text
run.rc = 0
d80ace690f2d1098192fae680aed2f390ab22472b75736f40e29358acf87bc3f  results/overall_metrics.tsv
d57429754431ca75e57f5ba6fd50fa0c24681375c7ebbc91fda4584e535ec4c2  abide_roi_manifest.tsv
d517328334da6715b722364218bbe3fa3a7b3cad3642ab724ffc4b9e02e979ff  brain_ossm_abide.raw.txt
```

## Claim Boundary

This is synthetic, non-clinical model-assay evidence only. It supports a
negative conclusion for the current fixed-dim6 Algebra-B line and the re-run
Fano/noncommutative-temporal candidate. It does not support clinical,
biomarker, mechanistic, MDD/ADHD, or broad O-SSM claims.

## Algebra-C Status

Algebra-C remains blocked on:

1. genuinely continuous target controls and tie/support audit;
2. generic capacity controls, including the higher-capacity `gru_wide` warning
   surface;
3. associative-projection confound repair for targets with `k in {4,5,6,7}`;
4. retrain-null requirements;
5. `BLK-20260707-madaros-f64-arg-abi-oct-mul` or an explicitly waived
   `lean_single` proof path for compiled runtime evidence.
