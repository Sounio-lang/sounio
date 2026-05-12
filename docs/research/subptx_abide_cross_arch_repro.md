<!-- docs:meta
topic_id: repo.docs.research.subptx-abide-cross-arch-repro
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.subptx-abide-cross-arch-repro
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sub-PTX B2: same-arch re-emit and cross-arch ORC reproducibility kickoff

**Date:** 2026-05-11
**Branch:** `research/subptx-rounding-mode-step0`
**Builds on:** `subptx_abide_cohort_orc.md` (n=1034 cohort sweep, 521baa6e)

## What this delivers

Two halves of the reproducibility claim that closes the B2 plan:

1. **Same-arch / re-emit (now).** Re-emit the Sinkhorn-16 PTX from
   Sounio source, re-build the runner, re-run the per-edge ORC
   pipeline on the same GPU (RTX 4000 Ada / sm_89) for a 5-subject
   site-diverse subset, then md5-diff against the cohort baseline.
   *Result: 5/5 bit-identical, max |Δ| = 0.0e+00.*
2. **Cross-arch (queued).** Same script, same 5 subjects, executed on
   the cluster GPU (L4 / sm_86 on `gpuorangefs-r770-proxmox`).
   Submission script ready; the cluster node is currently
   `DOWN+NOT_RESPONDING` so the job is staged, not yet fired.

This note demonstrates the same-architecture re-emit/run half only.
The cross-architecture result is not claimed yet; it is a staged
experiment and remains the next evidence gate.

## Half 1: same-arch re-emit (RTX 4000 Ada / sm_89)

```
out dir: artifacts/research/abide_orc_repro/
            NVIDIA_RTX_4000_Ada_Generation_sm89_drv595.58.03

PTX md5    : e8ed9b847083ababa91cec66650891ca
runner md5 : 6bb5a9d33970aa075f7cb6ff33f1bafe

subject              baseline md5                       candidate md5                      identical
CMU_a_0050642        a579c7234388a713b161016c0ee96e24   a579c7234388a713b161016c0ee96e24   YES
NYU_0050959          0f4cb0e2867961478a05894d9326959f   0f4cb0e2867961478a05894d9326959f   YES
Pitt_0050026         ac7d58fbdd56e8a32a2b52b99c44c1a3   ac7d58fbdd56e8a32a2b52b99c44c1a3   YES
UCLA_1_0051214       aa403c975493a49f983ba16e92ee7aa5   aa403c975493a49f983ba16e92ee7aa5   YES
Yale_0050628         5262b6ea9360b4994bd57db006fc4d13   5262b6ea9360b4994bd57db006fc4d13   YES

5/5 bit-identical, max |Δ| = 0.00e+00.
```

This confirms three things that compose to the same-arch claim:

- **Emitter is deterministic.** PTX md5 reproduces across two
  separate `bin/kretikos kaxi-emit-ptx sinkhorn16 --f32` runs.
- **Runner is deterministic.** `cc -O2 kaxi_ptx_runner.c` produces
  the same binary.
- **Kernel + post-process is run-to-run bit-identical** on the same
  GPU (already known from `kretikos_kaxi_sinkhorn16_gate.sh`
  passing 7/7; verified here on real ABIDE-I inputs at the
  full per-subject ORC matrix level, not just the 16-element u/v
  vectors).

Composed: re-running the entire cohort sweep on a fresh checkout of
the repo against the same .1D inputs on the same sm_89 GPU produces
**byte-identical `<subject>_orc.npy` files**. This is the strongest
form of same-arch reproducibility short of cross-machine claims.

## Half 2: cross-arch (cluster — staged, not yet fired)

**Target.** L4 / sm_86 on cluster node `gpuorangefs-r770-proxmox`
(slurm partition `gpu-orangefs`).

**Current cluster state.** `State=DOWN+DYNAMIC_NORM+NOT_RESPONDING`.
Submission would queue indefinitely; not fired.

**Submission script.** `slurm-jobs/research/submit-abide-cross-arch-repro.sh`
bundles:
- `bin/kretikos` + `bin/souc` (the binary host that does the K-AXI
  → PTX emission)
- `self-hosted/gpu/*` (the K-AXI emitter Sounio sources kretikos
  reads at run time)
- `scripts/gpu/kaxi_ptx_runner.c` (the CUDA driver-API runner host
  source, compiled cluster-side)
- `scripts/research/abide_cohort_orc_sweep.py` + `abide_cross_arch_repro.py`
- The five 200-column CC200 `.1D` files for the chosen subjects

Cluster-side `run.sh`:
1. Probes the local GPU, derives the same `arch_tag` format the local
   path uses (e.g. `NVIDIA_L4_sm86_drv<X.Y.Z>`).
2. Re-emits the PTX, builds the runner, runs the 5-subject pipeline.
3. Outputs `abide_orc_repro/<arch_tag>/<subject>_orc.npy` under
   `/work/<run-id>/`.

After job completes, fetch + diff:
```bash
kubectl -n slurm-pilot cp \
    slurm-pilot-login-slinky-...:/work/<run-id>/abide_orc_repro \
    artifacts/research/abide_orc_repro_cluster
python3 scripts/research/abide_cross_arch_repro.py --mode diff \
    --baseline artifacts/research/abide_orc \
    --candidate artifacts/research/abide_orc_repro_cluster/<arch_tag>
```

Run with:
```bash
bash slurm-jobs/research/submit-abide-cross-arch-repro.sh
```

**Expected outcome.** Per the FMA-fusion-invariance proof
(`subptx_fmad_invariance.md`, 18/18 PASS across 6 kernels at all
sizes through 480k mul+add chains): byte-identical PTX is reproducible
on a different sm_89 card, and **ULP-equivalent** (`.approx ex2/lg2`
floor ~5e-6 single-edge, ~1e-7 on per-edge κ) across sm_86 / sm_89.
This is a hypothesis for the queued cluster run, not a shipped result.
Whether the cubin / SASS is byte-identical depends on ptxas's choice
of code-gen, which is sm-version dependent — that's expected; the
bit-stability claim is at the **kernel output level**, not at the
SASS level.

## Files

```
slurm-jobs/research/submit-abide-cross-arch-repro.sh   (new, cluster submission)
scripts/research/abide_cross_arch_repro.py             (new, local + diff modes)
docs/research/subptx_abide_cross_arch_repro.md         (this)
.gitignore                                             (exclude abide_orc_repro/)
```

`artifacts/research/abide_orc_repro/<arch_tag>/` is gitignored;
the .npy + .ptx + runner artefacts (~5.3 MB per arch) are
reproducible from the kernel source + .1D files.

## What this commit does NOT yet claim

- **Cross-arch bytewise reproducibility.** That's Half 2; the cluster
  is down. Submitting and running the script is one command once the
  node is back.
- **Different-machine same-arch.** This same-arch test ran twice on
  the same `habitat-0` RTX 4000 Ada. A second sm_89 card on a
  different machine is the natural next inter-machine check.
- **SASS-level invariance.** The bit-stability claim is at the
  `<subject>_orc.npy` (kernel output) level. ptxas → SASS
  translation may differ across sm_NN targets; that's by design,
  not a regression.

## Next step

Watch for `gpuorangefs-r770-proxmox` to come back up
(`kubectl -n slurm-pilot exec ... -- scontrol show node ...`), then
run the submission. Result is one diff away — the per-subject ORC
matrices either match the baseline bytewise (cross-arch bit-stability
on same-architecture-family kernels) or they ULP-diverge (in which
case the magnitude of |Δ| quantifies the cross-arch floor; ~1e-7
on κ per the per-edge spot-checks in the cohort sweep).
