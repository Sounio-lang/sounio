<!-- docs:meta
topic_id: repo.docs.gpu.assoc-e2e-training
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.assoc-e2e-training
-->

# End-to-end training through the associator

The synthetic benchmark (`NONASSOC_BENCHMARK.md`) showed the associator is a useful **frozen** feature
(non-associativity is required, associative models are blind). With the associator's full VJP now in
hand (`ASSOC_VJP_COMPLETE.md`), this closes the loop: a model that **uses** the associator, trained
**end-to-end**, with the associator's forward and backward both running on compiler-lowered tensor-core
kernels.

## Model & training
    y = w · [a, b, c]          learnable octonion weights a, b and a linear readout w
- **forward**: `z = oct_assoc(a, b, C)` (the associator tensor-core kernel); `y_i = w·z_i`.
- **backward**: `dz_i = dy_i·w` → upstream `dD`; `dw += Σ dy_i·z_i`; and `da, db` via the merged
  tensor-core decomposition (`ossm_oct_bwd` + `oct_batch_mul` + the `a⊗b` product VJP).
- **optimizer**: Adam on `a, b, w`.
- **targets**: teacher-generated (reachable) — 8 batches × 16 = 128 samples.

## Result (DGX Spark GB10, 300 iters)
    iter   1  loss 1.10e+01
    iter  50  loss 2.68e-03
    iter 150  loss 1.15e-05
    iter 300  loss 1.06e-05          →  100% reduction

**The associator is a trainable tensor-core layer** — the loss falls through its VJP. This is the
stronger claim on top of the frozen-feature benchmark: not only does the associator carry signal
associative models are blind to, a model can be trained to exploit it end-to-end, with every heavy op
(the associator forward, the backward `L(·)ᵀ` tiles and `da`-accumulations) emitted by the compiler.

Harness: `run_assoc_train.cu`. Honest scope: teacher-target (reachable) — it validates that the
gradients flow and the model converges through the associator, not a benchmark-beating claim. The
non-associativity-required claim is the benchmark's (#1204); this is the trainability claim. Next: a
task combining both — where a model must *learn* to use non-associativity to solve something an
associative model cannot — and ultimately real data.
