<!-- docs:meta
topic_id: repo.docs.research.subptx-abide-cohort-orc
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.subptx-abide-cohort-orc
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sub-PTX B2: full ABIDE-I cohort ORC sweep (1035 subjects)

**Date:** 2026-05-11
**Hardware:** NVIDIA RTX 4000 Ada (sm_89), habitat-0
**Toolchain:** ptxas 12.0.140 / cuobjdump 12.0.140 / driver 595.58.03
**Branch:** `research/subptx-rounding-mode-step0`
**Builds on:** `subptx_abide_orc_sweep.md` (per-subject pipeline);
`subptx_phase_h_complete.md` (16-iter kernel infrastructure)

## What this delivers

A full-cohort Ollivier-Ricci curvature (ORC) matrix for every
ABIDE-I subject in the local CC200 collection (1035 .1D files,
of which 1034 parse to a 200-column time series). Each subject
yields a 200×200 directed ORC matrix over a k=15-NN connectome
(3000 edges per subject) computed by the Sounio 16-iter
Sinkhorn-LSE GPU kernel.

This is the cohort-scale conclusion of the B2 sub-PTX plan
locked on 2026-05-11.

## Run

```
scripts/research/abide_cohort_orc_sweep.py
  └─ enumerates artifacts/research/abide/*_rois_cc200.1D (1035 files)
  └─ skips subjects whose <subject>_orc.npy already exists (resume)
  └─ per subject: full per-edge pipeline as in abide_orc_sweep.py
       (k=15 kNN graph → 3000 edges → 16-elt neighborhoods →
        per-edge K matrix → 12 chunked launches of 256 threads
        on the Sinkhorn-LSE PTX kernel → transport plan → ORC)
  └─ writes artifacts/research/abide_orc/<subject>_orc.npy
  └─ joins per-subject summary with the cohort phenotype manifest
     (artifacts/research/brain_ossm_full_cohort/abide_roi_manifest.tsv)
  └─ writes artifacts/research/abide_orc/cohort_summary.tsv
```

## Cohort-scale outcome

| Metric | Value |
|---|---|
| Subjects total                  | 1035 |
| Subjects with valid 200-col .1D | 1034 (one malformed: `UM_1_0050284`) |
| Subjects labelled by manifest   | 499 (ASD 249 / TD 250) |
| Sites                           | 20 |
| Edges per subject               | 3000 (200 nodes × directed k=15) |
| Total directed ORC values       | 3.1 M (1034 × 3000) |
| Wall time                       | 48 min 32 s on RTX 4000 Ada |
| Per-subject mean                | ~2.85 s |

## Per-subject ORC distributional summary (labelled n=499)

```
mean of per-subject mean ORC : +0.5495
sd   of per-subject mean ORC :  0.0804
range of per-subject mean    : [+0.2667, +0.8265]
fraction of edges positively curved (κ > 0): 1.0000 ± 0.0002
                                                  (i.e. essentially 100 %
                                                   on every subject)
```

100 % positive curvature on every subject's direct k-NN edges
matches the per-subject finding from the 5-subject sub-cohort:
the k-NN graph connects strong functional partners, their
neighborhoods overlap heavily, Wasserstein-1 distances stay
small, and so κ = 1 − W stays positive on these direct edges.
Negative-curvature edges would emerge only if the graph were
extended beyond direct k-NN (2-hop, 3-hop, full-graph), where
neighborhood overlap drops.

## Per-site mean ORC (labelled subjects only)

| Site      | n  | mean(per-subject mean)  | sd(per-subject mean)  | mean(per-subject std) |
|-----------|----|-------------------------|-----------------------|------------------------|
| CALTECH   | 24 | +0.5319                 | 0.0840                | 0.1127                 |
| CMU       | 24 | +0.5903                 | 0.0443                | 0.0998                 |
| KKI       | 26 | +0.5227                 | 0.0943                | 0.1027                 |
| LEUVEN_1  | 24 | +0.5892                 | 0.0675                | 0.0972                 |
| LEUVEN_2  | 24 | +0.5648                 | 0.0621                | 0.0933                 |
| MAX_MUN   | 26 | +0.5260                 | 0.0781                | 0.1105                 |
| NYU       | 26 | +0.5113                 | 0.0731                | 0.0977                 |
| OHSU      | 24 | +0.5591                 | 0.0507                | 0.1158                 |
| OLIN      | 24 | +0.5758                 | 0.0709                | 0.0857                 |
| PITT      | 26 | +0.5777                 | 0.0833                | 0.0930                 |
| SBL       | 24 | +0.5879                 | 0.0766                | 0.1045                 |
| SDSU      | 24 | +0.5258                 | 0.0792                | 0.1000                 |
| STANFORD  | 26 | +0.5485                 | 0.0809                | 0.0785                 |
| TRINITY   | 26 | +0.5870                 | 0.0943                | 0.0978                 |
| UCLA_1    | 26 | +0.5644                 | 0.0802                | 0.0916                 |
| UCLA_2    | 24 | +0.5350                 | 0.0760                | 0.0961                 |
| UM_1      | 25 | +0.5053                 | 0.0759                | 0.0995                 |
| UM_2      | 24 | +0.5093                 | 0.0657                | 0.0899                 |
| USM       | 26 | +0.5509                 | 0.0799                | 0.1013                 |
| YALE      | 26 | +0.5316                 | 0.0625                | 0.1082                 |

Inter-site spread of site-mean ORC: range [+0.5053, +0.5903] = 0.085;
typical within-site sd of per-subject mean ≈ 0.075.

**The site effect and the within-site biological spread are
the same order of magnitude.** This is the canonical multi-site
fMRI biomarker problem — and it is exactly what the B2 plan
identified as the live frontier: bit-reproducible curvature
*permits* a clean disentanglement of "site = pipeline" from
"site = biology"; without bit-reproducibility you cannot rule
out that the inter-site spread is partly tooling drift.

The sub-PTX-locked kernel produces **bit-identical** ORC for
the same input on a given GPU (validated as a property of the
LSE-8 / Sinkhorn-16 emitters; the 3-run determinism gate
`kretikos_kaxi_sinkhorn16_gate.sh` passes 7/7). Cross-GPU
bit-equality (RTX 4000 Ada vs A5000 vs L4) is the explicit
next experiment.

## ASD vs TD on per-subject ORC features (n=499)

| Feature                       | ASD (n=249)         | TD (n=250)          | Cohen's d |
|-------------------------------|---------------------|---------------------|-----------|
| per-subject mean ORC          | +0.55000 ± 0.07864  | +0.54907 ± 0.08222  | +0.0112   |
| per-subject std ORC           |  0.09923 ± 0.02221  |  0.09829 ± 0.02448  | +0.0402   |

Both ASD/TD comparisons are essentially null. The mean-ORC d of
+0.011 is **even more nullward** than the G₂ bridge analysis
of CC200 eigenmodes (`project_g2_bridge.md`: d = 0.06).

No inferential p-values are claimed here. The ASD/TD table is a
descriptive effect-size screen over two predeclared per-subject
features, without covariate adjustment, site harmonisation, or
multiple-comparison correction. It is useful as a null-direction
engineering result for this direct-kNN ORC operator, not as a
confirmatory biomarker analysis.

## What this confirms — and what it doesn't

**Confirms.** The G₂ bridge null was not an artefact of the
prior pipeline; it generalises to a different geometric
biomarker (ORC on directed k-NN connectomes) computed under a
bit-reproducible sub-PTX-locked kernel. *Mean ORC on direct
k-NN edges of CC200 connectomes carries no ASD/TD signal.*

The mechanism is now well-supported: direct k-NN edges all
have positive curvature (κ ≈ 0.5–0.6 with little variation),
so any biological signal would have to live in the *higher
moments* of the κ distribution, not its mean. The cohort
std ORC d = +0.040 is also null, ruling out one obvious
moment.

**Does not yet rule out.** Negatively-curved edges, which
appear only at 2-hop or 3-hop extension of the graph (where
neighborhood overlap drops), are not in this sweep — that
would be a different ORC operator. Edge-class summary
features (e.g. "fraction of edges in the bottom decile of
κ", which on this sweep is essentially noise on the lower
tail of a tight unimodal distribution but might separate at
2-hop) are also not tested.

**Cross-center reproducibility.** Not yet demonstrated. All
1034 subjects ran on the same RTX 4000 Ada. The natural next
step is to re-emit on A5000 / L4 via the cluster path and
diff `<subject>_orc.npy` bytewise — the kernel-level 3-run
determinism gate already confirms run-to-run on the same
GPU; cross-GPU is the cluster experiment.

**GPU evidence boundary.** The committed TSV is an output artefact from
the recorded RTX 4000 Ada sweep. CPU-only checkouts can inspect and
reuse the TSV, but cannot regenerate the GPU `.npy` matrices or the
Sinkhorn execution evidence without a CUDA-capable runner.

## Output artefacts

```
artifacts/research/abide_orc/
├── <subject>_orc.npy           (1034 files; 200×200 float64; ~317 MB total)
├── cohort_summary.tsv          (per-subject feature table; 1034 rows, 12 cols)
└── logs/cohort_sweep.log       (run log, including the 1 parse-failure case)
```

The .npy files are gitignored (large + reproducible); the TSV
and log are committed for downstream analysis without
re-running the kernel.

## Reproduction

```bash
# Prerequisites.
sudo apt-get install -y nvidia-cuda-toolkit
pip3 install --user --break-system-packages numpy

# Emit kernel + build runner (one-time, ~3 min for the 5.26 MB PTX).
./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o /tmp/sinkhorn16iter.ptx
cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -lm -o /tmp/kaxi_runner

# Full cohort sweep (~50 min, resumes on existing .npy files).
python3 scripts/research/abide_cohort_orc_sweep.py

# Inspect one subject's ORC matrix.
python3 -c "
import numpy as np
o = np.load('artifacts/research/abide_orc/CMU_a_0050642_orc.npy')
print(o.shape, 'NaN-padded; n_edges =', np.isfinite(o).sum())
print('mean=', np.nanmean(o), 'range=', np.nanmin(o), np.nanmax(o))
"
```

## Full B2 stack on PR #128 (in shipped order)

| Commit | What |
|---|---|
| `e611776b` | Step 0: K-AXI `round=rN` attribute for f64 ops |
| `5a2be3eb` | B1: Cayley-Dickson FMA-fusion invariance proof |
| `6d014471` | B2-3: LSE-8 primitive validated on RTX 4000 Ada |
| `cb36f70b` | B2-4: Sinkhorn-16 finding (research note, source as text) |
| `59221a35` | B2-4: Sinkhorn-16 source edits applied |
| `bb7968c7` | B2-4: Sinkhorn-16 gate + extended FMA invariance |
| `b6141bc6` | Buffer-refactor investigation: original target was wrong |
| `5aa1c7b4` | Phase H: streaming PTX + 2 MB KaxiAsmBuf + 16-iter |
| `f10ed4b0` | B2: Sinkhorn-16 on real ABIDE-I single-edge K matrices |
| `298875e2` | B2: per-subject all-edge ORC sweep, 5 subjects |
| this commit | B2 closure: full 1035-subject cohort ORC, ASD/TD null at d=0.011 |

## Natural next steps

Two are independent and could run in either order:

1. **Cross-arch reproducibility.** Re-emit `sinkhorn16` on A5000
   and on L4 (via the cluster path); pick ~5 subjects from the
   cohort; compare `<subject>_orc.npy` bytewise across the three
   architectures. This is the cross-center claim the B2 plan
   committed to and is the strongest "what sub-PTX gives the
   field" demonstration. *Expected: bit-identical for the same
   sm_89 family, ULP-equivalent across families.*

2. **Beyond-direct-kNN ORC sweep.** Extend the graph to 2-hop
   (paths of length 2) and 3-hop; re-run the cohort sweep; check
   whether the *negative-curvature* tail appears and whether it
   carries an ASD/TD signal. The B2 plan flagged this as the
   biologically informative regime (where neighborhood overlap
   drops, so curvature can swing strongly negative on
   information-bottleneck edges).

The first lands the *infrastructure* claim. The second is the
*biology* claim. The d = 0.011 number here means direct-kNN ORC
is essentially settled-as-null; the live biological question
has moved one hop out.
