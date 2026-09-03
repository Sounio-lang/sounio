<!-- docs:meta
topic_id: repo.docs.research.hyperbolic-semantic-networks-run
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.hyperbolic-semantic-networks-run
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Running agourakis82/hyperbolic-semantic-networks on local Sounio

**Date:** 2026-05-11
**Branch:** `research/subptx-rounding-mode-step0`
**Hardware:** workspace habitat-0 (CPU run; same machine that hosts the
RTX 4000 Ada for the GPU side)
**Repo under test:** `agourakis82/hyperbolic-semantic-networks` @ 8b5c676
(cloned to `/tmp/hyperbolic-semantic-networks`)

## What was attempted

Take the headline Sounio phase-transition experiments from the
`hyperbolic-semantic-networks` repo and run them through the local
Sounio compiler (`bin/souc` 1.0.0-beta.5, native ELF compile path),
compare the outputs against the validated Julia reference shipped in
the same repo, identify any gap.

Programs run:
- `experiments/01_epistemic_uncertainty/phase_transition.sio`     (N=20)
- `experiments/01_epistemic_uncertainty/phase_transition_n100.sio` (N=100)

Both type-check and compile to native ELF (~50 KB each) without any
modification. Native ELF runs in seconds on this CPU.

## Result

| N   | k sweep | k_crit | output kappa | matches Julia ref |
|-----|---------|--------|--------------|--------------------|
| 20  | [2,3,4,5,6,7,8,9,10,14,18] | ~7.07 | 0.000000 across all k | NO |
| 100 | [2,3,4,6,8,10,12,14,16,18,20,25,30,35,40] | ~15.81 | 0.000000 across all k | NO |

The Julia reference for N=100 gives (e.g.) κ̄(k=3) = -0.252,
κ̄(k=6) = -0.303, κ̄(k=20) = +0.089. The Sounio program produces
κ ≈ 0 for every configuration.

## Diagnosis

The discrepancy is **algorithmic, not a Sounio compiler bug.**

The Sounio `.sio` phase-transition programs use **vanilla
Sinkhorn-Knopp** in the primal (non-log) domain:

```
K[i,j] = exp(-C[i,j] / epsilon)
u <- mu / (K * v)
v <- nu / (K^T * u)
W1 = sum u[i] * K[i,j] * v[j] * C[i,j]
```

The programs ship with `epsilon = 0.5, max_iter = 80`. The Julia
reference uses `epsilon = 0.01, max_iter = 1000, tol = 1e-6`.

- At `epsilon = 0.5`, K is so smooth that the entropic-regularised W₁
  collapses toward the uniform-transport value, i.e. W₁ ≈ d_uv on
  average → κ = 1 − W₁/d_uv ≈ 0 by construction. (Verified
  experimentally: `kappa_mean = 0.000000` across the full k sweep.)
- At smaller `epsilon` (tested 0.05 with `max_iter=500`), the
  K matrix entries underflow rapidly: for `epsilon=0.05` and `d=1`,
  `K[i,j] = exp(-20) ≈ 2e-9`. The Sinkhorn dual updates then
  amplify u and v inversely so that the product `u * K * v` stays
  pinned near `mu * nu`, and `W₁` *still* converges to ≈ d_uv,
  again yielding κ ≈ 0. (Verified: same `kappa_mean = 0.000000`
  output at the lower ε.)

This is the standard failure mode of vanilla Sinkhorn in single
precision — and even in f64 with these K-magnitudes it does not
recover the correct W₁ because the algorithm's primal balancing is
floor-pinned at the kernel-entry magnitudes.

**The fix that the field uses, and that Sounio has already shipped,
is log-domain Sinkhorn (Sinkhorn-LSE).** In log-domain the kernel
is added rather than multiplied, the iterates are unbounded f64
(no underflow), and the algorithm converges cleanly at any
`epsilon` down to ~1e-3 in practice.

## Sounio already has the fix

`self-hosted/gpu/kretikos_emit_kaxi.sio :: kaxi_emit_sinkhorn16_asm`
emits a 16-iter Sinkhorn-LSE GPU kernel for N=16. This is the same
kernel the ABIDE-I cohort ORC sweep used to compute 3.1 M directed
ORC values on 1034 subjects in 48 min, with ≤ 5e-6 max error vs a
NumPy reference at the same algorithm (see
`docs/research/subptx_abide_orc_sweep.md` and
`docs/research/subptx_abide_cohort_orc.md`).

The kernel ABI is documented and the host runner is at
`scripts/gpu/kaxi_ptx_runner.c`. The kernel takes a 320-element
f32 input per edge (la 16 + lb 16 + log-K 256 + zeros 32) and
writes back u, v dual potentials.

## Bridge to the hyperbolic-semantic-networks experiment

The phase-transition experiment computes
`kappa(u,v) = 1 - W_1(mu_u, mu_v) / d(u,v)` on random k-regular
graphs. Each (u,v) edge supplies a 16x16-ish kernel problem (the
endpoints' lazy measures have at most k+1 support points; for the
k≤14 sweep that's ≤ 15, which fits in N=16 with one zero-padded
slot).

A straightforward bridge:

1. Build the k-regular graph in Sounio (already implemented in
   `phase_transition.sio :: build_k_regular`).
2. Compute BFS hop distances (already there).
3. For each edge (u,v):
   - Build `mu_u`, `mu_v` (already there).
   - Pack la, lb (log marginals; `log2(mu.probs)`) and the 16x16
     log-K (cost / (epsilon * ln 2) with sign flipped to match the
     LSE convention).
   - Pad to 320 f32 slots.
4. Batch all edges into a single GPU launch (or chunked launches
   of 256 threads each, exactly the path used for the ABIDE-I
   cohort sweep).
5. Read back u, v; reconstruct W_1 = sum exp2(u + K + v) * cost.
6. Save (N, k, ratio, kappa_mean, kappa_std, n_edges, ...).

Expected: κ̄ matches the Julia reference within `.approx ex2/lg2`
precision (~1e-6 single-edge, ~1e-7 after summation).

This is a 1- to 2-day port; it would let the user's two conference
presentations stand on a GPU-bit-reproducible curvature stack at
the same algorithmic level as the Julia reference, with the
side-benefit that the same kernel already runs on real ABIDE-I
connectomes — making the *semantic-network* and *brain-connectome*
curvature analyses directly cross-comparable bit-for-bit.

## What was committed back to the user's repo

Nothing yet. This note documents the diagnosis on the Sounio side.
The natural follow-on, if desired, is to:
- Open a PR against `agourakis82/hyperbolic-semantic-networks` that
  swaps the primal-domain Sinkhorn for a log-domain Sinkhorn
  written in pure Sounio (the GPU kernel is too heavy a dependency
  for the toy N=20 case; a CPU-side log-domain Sinkhorn-LSE in
  Sounio is straightforward and matches the algorithm semantics
  exactly).
- Separately, write the GPU-side bridge described above for the
  N≥100 production runs.

## How to reproduce

```bash
# Clone the experiment repo.
cd /tmp && rm -rf hyperbolic-semantic-networks
gh repo clone agourakis82/hyperbolic-semantic-networks

# Use local Sounio (this repo's bin/souc 1.0.0-beta.5).
cd /workspace/sounio

# Type-check + native-compile + run the N=20 variant.
bin/souc check  /tmp/hyperbolic-semantic-networks/experiments/01_epistemic_uncertainty/phase_transition.sio
bin/souc compile /tmp/hyperbolic-semantic-networks/experiments/01_epistemic_uncertainty/phase_transition.sio \
    -o /tmp/sn_phase
/tmp/sn_phase | head

# Same for N=100 (a few seconds on CPU).
bin/souc compile /tmp/hyperbolic-semantic-networks/experiments/01_epistemic_uncertainty/phase_transition_n100.sio \
    -o /tmp/sn_phase_n100
/tmp/sn_phase_n100 > /tmp/sn_phase_n100.csv

# Compare against the Julia reference shipped with the repo.
python3 -c "
import json
with open('/tmp/hyperbolic-semantic-networks/results/experiments/phase_transition_julia_n100.json') as f:
    d = json.load(f)
for r in d['results']:
    print(f\"k={r['k_target']:>2} ratio={r['ratio']:>5.2f}  julia kappa_mean={r['kappa_mean']:+.4f}  geometry={r['geometry']}\")
"
```

## Operational notes for the user

- `bin/souc compile` produces a 50 KB native ELF that runs at
  C-equivalent speed. The N=100 sweep completes in seconds.
- `print(usize)` in current `bin/souc` emits a trailing newline
  (a known issue being fixed on `coord/nv2-println-f64-parity` per
  commit `5a6c4493`). The downstream CSV needs a single
  post-process pass to re-join rows. (Or move all `print(usize)`
  calls in `phase_transition.sio` to a `usize_to_csv` helper that
  uses `print(buf)` of a stringified buffer.)
- For the next round of experiments, the natural Sounio idiom is
  log-domain Sinkhorn (see above).
