<!-- docs:meta
topic_id: repo.docs.research.subptx-abide-orc-sweep
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.subptx-abide-orc-sweep
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sub-PTX B2: per-subject all-edge ORC sweep on ABIDE-I

**Date:** 2026-05-11
**Hardware:** NVIDIA RTX 4000 Ada (sm_89), habitat-0
**Toolchain:** ptxas 12.0.140 / cuobjdump 12.0.140 / driver 595.58.03
**Branch:** `research/subptx-rounding-mode-step0`
**Builds on:** `subptx_abide_sinkhorn_probe.md` (single-edge probe);
`subptx_phase_h_complete.md` (16-iter kernel infrastructure)

## What this delivers

A per-subject Ollivier-Ricci curvature (ORC) computation over a full
3000-edge directed k=15-NN functional connectome, derived from real
ABIDE-I CC200 time-series, using the Sounio 16-iter Sinkhorn-LSE
GPU kernel as the inner OT solver. Cross-validated edge-by-edge
against a NumPy reference Sinkhorn.

## Pipeline

`scripts/research/abide_orc_sweep.py`:

```
ABIDE-I subject .1D  →  200x200 Pearson correlation  →  k-NN graph
  (k=15 directed edges per node = 3000 edges total)  →  for each node,
  16-element neighborhood (node + top-15 functional partners)  →
  for each edge (u, v):  cost matrix C[i, j] = 1 - corr(N_u[i], N_v[j])
                         log-kernel K = -C / (lambda · ln 2)
                         pack 320 f32 (la 16 + lb 16 + K 256 + zeros 32)
  →  concatenate into 3000 × 320 batch
  →  chunk into 12 launches of ≤ 256 edges each
     (256 = sm_89 register-budget cap: 255 regs/thread × 256 threads/block
      = 65280 < 65536 max regs per block)
  →  kernel reads each thread's 320-element slice, writes (u_out, v_out)
  →  transport plan P[e, i, j] = ex2(u_out[e, i] + K[e, i, j] + v_out[e, j])
  →  Wasserstein-1 W[e] = sum_{i,j} P[e, i, j] · C[e, i, j]
  →  ORC: kappa(u, v) = 1 - W   (for direct k-NN edges with graph d = 1)
  →  spot-check N random edges against NumPy reference Sinkhorn
```

## 5-subject cross-validation (RTX 4000 Ada / sm_89, λ=1.0, k=15)

| Subject | edges | ORC range | mean ± std | % positive | spot-check | wall |
|---|---|---|---|---|---|---|
| CMU_a_0050642 | 3000 | [0.300, 0.765] | 0.619 ± 0.081 | 100% | 3/3 PASS | 2.9s |
| CMU_a_0050646 | 3000 | [0.232, 0.745] | 0.551 ± 0.095 | 100% | 3/3 PASS | 2.8s |
| CMU_b_0050643 | 3000 | [0.116, 0.795] | 0.595 ± 0.096 | 100% | 3/3 PASS | 2.9s |
| CMU_a_0050647 | 3000 | [0.211, 0.849] | 0.592 ± 0.123 | 100% | 3/3 PASS | 3.0s |
| CMU_a_0050649 | 3000 | [0.134, 0.771] | 0.561 ± 0.115 | 100% | 3/3 PASS | 3.1s |

15 random-edge spot-checks total (3 per subject), all PASS at tolerance
5×10⁻³. Typical |Δκ| ≈ 10⁻⁷ — even tighter than the single-edge
probe's 5×10⁻⁶, because kappa is a 256-element sum and the per-element
`.approx` noise averages out.

## Observations

**100% positive curvature on k-NN edges.** Biologically plausible: the
direct k-NN edges connect strong functional partners; their
neighborhoods overlap heavily; Wasserstein-1 distance between similar
neighborhood distributions is small; therefore κ = 1 - W is positive.
For full diagnostic-biomarker work the interesting signal would be
in *negatively* curved edges — those would emerge when the graph is
extended beyond direct k-NN (e.g. 2-hop, 3-hop edges where neighborhood
overlap drops).

**Inter-subject variability is real.** Mean ORC across the 5 subjects
ranges from 0.551 (CMU_a_0050646) to 0.619 (CMU_a_0050642) — a ~10%
spread that's well outside the ULP-level numerical noise floor of the
kernel (≤ 10⁻⁶). This is the geometric signal the cohort-scale analysis
would test against ASD/TD labels, age, head-motion, etc.

**Throughput is comfortable.** ~3 seconds per subject including I/O.
At this rate the full 1035-subject ABIDE-I cohort sweep takes
~1 hour on this single RTX 4000 Ada — well within an overnight run.

## What this commit does NOT yet claim

- **Cross-center reproducibility.** All five subjects ran on the same
  RTX 4000 Ada. Cross-arch test (RTX 4000 Ada vs A5000 vs L4)
  requires either cluster access or local multi-GPU; that's the
  next concrete step.
- **A bit-exact-across-centers ORC matrix.** The transport-plan
  reconstruction `P = ex2(u + K + v)` and the Wasserstein sum are
  computed in NumPy float64 after the kernel returns. To claim
  bit-exact ORC across centers, the post-process needs to either
  (a) run on the GPU too, or (b) be deterministic in NumPy
  (currently is, but worth documenting).
- **An ORC algorithm choice that matches published ORC implementations.**
  Standard ORC uses *graph distance* between neighborhood vertices,
  not correlation-derived 1-corr. Our cost is a similarity-derived
  surrogate. The kernel is correct for the cost we feed it; whether
  this is the right cost for the analytic claim is a separate
  question that depends on the downstream use.
- **ASD/TD or any clinical claim.** This is infrastructure
  validation. Statistical comparison against phenotype labels is
  the cohort-scale follow-on.

## Output artefacts

For each subject swept:

```
artifacts/research/abide_orc/<subject>_orc.npy
```

A 200×200 float64 NumPy array. `orc[u, v]` is the directed ORC from
ROI u → its k-NN neighbour v if (u, v) is an edge in the k-NN graph;
NaN otherwise.

## Reproduction

```bash
# Prerequisites.
sudo apt-get install -y nvidia-cuda-toolkit
pip3 install --user --break-system-packages numpy

# Emit kernel + build runner (one-time).
./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o /tmp/sinkhorn16iter.ptx
cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -lm -o /tmp/kaxi_runner

# Single subject sweep.
python3 scripts/research/abide_orc_sweep.py --subject CMU_a_0050642 \
    --k 15 --lambda 1.0 --check 5

# Full 5-subject smoke.
for subj in CMU_a_0050642 CMU_a_0050646 CMU_b_0050643 \
            CMU_a_0050647 CMU_a_0050649; do
    python3 scripts/research/abide_orc_sweep.py --subject "$subj" \
        --k 15 --check 3 \
        | grep -E '^(subject|  range|  mean|  positive|spot-check:|total wall)'
done
```

## The full B2 stack on PR #128 (in shipped order)

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
| this commit | B2 closure: per-subject all-edge ORC sweep, 3000 edges, 5 subjects |

## Natural next step

Cohort-scale sweep: parallel-launch all 1035 ABIDE-I subjects through
this pipeline (~17 hours single GPU; minutes on cluster). Then:

1. Distributional ORC statistics per site (CMU_a vs CMU_b vs the
   other 17 ABIDE-I sites) — does the inter-site spread exceed the
   inter-subject spread?
2. ASD/TD comparison on the per-subject mean / variance / specific-
   edge-class ORC features — does the sub-PTX-locked kernel resolve
   any biomarker that the prior G₂ bridge null (CC200 eigenmodes,
   d = 0.06) couldn't?
3. Cross-arch reproducibility: same 5-subject sweep on A5000 / L4
   via the cluster path, diff the `<subject>_orc.npy` bytewise.
