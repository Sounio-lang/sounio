<!-- docs:meta
topic_id: repo.docs.gpu.nonassoc-benchmark
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.nonassoc-benchmark
-->

# Synthetic non-associativity benchmark — the empirical payoff

The novelty map (`HYPERCOMPLEX_SSM_NOVELTY.md`) identified **one** gap between a systems/PL artifact and
an ML-result claim: a controlled, ablatable task where **non-associativity measurably helps**. This is
the first such result.

## The task (non-associativity required by construction)
From an octonion triple `(a, b, c)`, predict a fixed random linear projection `y = w·[a,b,c]` of the
**associator** `[a,b,c] = (a⊗b)⊗c − a⊗(b⊗c)`. The associator is **identically zero for every associative
algebra** (ℝ, ℂ, ℍ/quaternion) — so the task is *invisible by construction* to any associative model,
not merely hard. The octonion associator, computed by our compiler-lowered `oct_assoc` tensor-core
kernel, is the exact feature that carries the signal.

## Result (DGX Spark GB10, 8192 samples, 6144 train / 2048 held-out)
Linear-probe held-out R² on four feature sets:

| feature set | R² |
|---|---|
| **(1) octonion associator `[a,b,c]` — our `oct_assoc` tensor-core kernel** | **1.0000** |
| (2) quaternion associator (≡ 0 — the associative-algebra blind spot) | −0.0005 |
| (3) raw input `(a,b,c)`, 24-d (the associator is trilinear, not linear) | −0.0021 |
| (4) associative pairwise products `[a⊗b, b⊗c]`, 16-d | +0.0016 |

`oct_assoc` feature error vs the exact host associator ≈ 5.2e-4 (f16 tile precision — small enough that
the noisy kernel feature still linearizes the task perfectly).

**Only the octonion-associator feature linearizes the task; every associative or linear feature is at
chance (R² ≈ 0).** This is a structural separation, not a capacity gap: an associative-algebra model's
associator is *identically zero*, so it cannot represent the signal at any width or depth. It ties the
ML claim directly to the systems contribution — the useful feature is the one our kernel makes cheap.

## Honest scope
- Synthetic and, by design, favorable: the target *is* a projection of the associator. The claim is not
  "octonion models beat SOTA," it is "there exists a task, controlled and ablatable, where the
  associator is necessary and associative models are provably blind — and our kernel supplies it."
- Next: (a) a non-linear task (associator norm / a downstream decision) trained end-to-end through the
  `oct_matvec` real-capacity model; (b) an unstructured-MLP sample-efficiency curve (how much data an
  associative-agnostic MLP needs to approximate the associator from raw inputs); (c) ultimately a real
  dataset (the connectome associator-field pilot).

Harness: `run_nonassoc_bench.cu` (uses the merged `oct_assoc` kernel; least-squares probes on host).
