<!-- docs:meta
topic_id: repo.docs.research.probe-result-h256-scale
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.probe-result-h256-scale
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The negative holds at scale (H=256, T=200) — and it exposes an architectural confound in the LSTM target

*Follow-up to `PROBE-RESULT-lstm-adding.md` (the H=40 negative). The decisive control — is the h→h
alignment architectural (present at init) — needs no training and no torch: the analytic LSTM Jacobian
(validated to 7e-8 vs autograd) is computed on a random H=256 net in pure numpy. It confirms the negative,
and reveals why the LSTM is not the confounder-free target it was taken to be.*

## Result — untrained H=256, T=200, align(k) (init/architectural control)
| k | 1 | 2 | 4 | 8 | 16 | 32 | 63 |
|---|---|---|---|---|---|---|---|
| baseline √(k/2H) | 0.04 | 0.06 | 0.09 | 0.12 | 0.18 | 0.25 | 0.35 |
| **INIT h→h** | 0.99 | 0.99 | **1.00** | 1.00 | 1.00 | 1.00 | **1.00** |
| INIT c→c | 0.78 | 0.83 | 0.83 | 0.82 | 0.81 | 0.85 | 0.90 |

In a **random** LSTM the h→h Jacobian alignment is **≈1.0 at every k** (1…63) while the baseline is 0.04–0.35.
No small-k shoulder, no fall to baseline — the low-rank/architectural shape, pushed to its extreme. The
negative is not only intact at scale, it is sharper.

## Why — a shared recurrent backbone (refines the target)
The LSTM applies the **same** recurrent matrix `W_hh` at every step; the gates modulate it, but the backbone
is shared, so consecutive Jacobians `∂h_{t+1}/∂h_t ≈ diag(gate_t)·(structure of W_hh)` share their singular
subspaces **by construction**, before any training. This is a *partial* version of the S4/Mamba
disqualification (shared eigenvectors): the "dense, input-dependent, genuinely distinct" argument for the
LSTM underweighted the shared `W_hh`. A truly confound-free target needs **distinct weights per step/layer**
— a deep feedforward net probed on its **branches** `F'_l = J_l − I`. The LSTM is not clean.

## Verdict at scale
The negative confirms: the trained-LSTM h→h alignment (the d=56 "positive" at H=40) is architectural and
low-rank, and at H=256 it is architecturally ≈1 everywhere — the sedenion-predicted small-k shoulder with a
healthy bulk is absent. No training can invert an alignment that is already ≈1 at init (verified at H=40,
where training slightly *reduced* it). The structural annihilation signature is not present in these RNN
Jacobians; the vanishing gradient is magnitude + rank + a shared-weight architectural alignment.

## On the compute path (recorded honestly)
The SLURM GPU nodes (RTX 4000 Ada) are bare: no pip/ensurepip, no internet, no shared filesystem — PyTorch
cannot be installed there, and the canonical `beagle-sounio` image is the Sounio compiler, not an ML image.
So the GPU-via-SLURM route does not run torch without a purpose-built container. It did not matter: the
decisive control (init/architectural) is training-free and torch-free via the analytic Jacobian, and it
answered the question. Harness `probe_h256_init.py` (pure numpy, analytic LSTM Jacobian).
