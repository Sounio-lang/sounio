<!-- docs:meta
topic_id: repo.docs.gpu.nonassoc-headtohead
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.nonassoc-headtohead
-->

# The combined result: a task that *needs* non-associativity, solved by a *trained* model

Two earlier results stood apart:
- `NONASSOC_BENCHMARK.md` (#1204) — the octonion associator carries signal that associative-algebra
  features are **blind** to (non-associativity is *required*), shown with a *frozen* feature.
- `ASSOC_E2E_TRAINING.md` (#1209) — the associator is a *trainable* tensor-core layer (loss falls
  through its VJP), shown on a *reachable teacher* target.

This experiment fuses them into one head-to-head: a task that **structurally requires
non-associativity**, with **every model trained**.

## The task
    y_i = ‖[a*, b*, c_i]‖²          the squared norm of the octonion associator of a fixed (unknown)
                                    teacher pair (a*, b*) with input octonion c_i
It is deliberately built to defeat associative models two ways at once:
- **non-associative** — it is an associator, which is identically zero for *any* associative algebra;
- **non-linear** — the squared norm is quadratic in `c`, so linear readouts cannot represent it.

## Models (all trained with Adam, 400 iters; 192 train / 96 test samples)
| Model | test R² | why |
|---|---|---|
| **A — octonion**, learns `a,b` via the associator VJP, `ŷ=‖[a,b,c]‖²` | **+1.0000** | right inductive bias; **forward + `da/db` on the compiler-lowered tensor-core kernels** |
| **B — quaternion associator ≡ 0** | −3.57 | **structurally blind** — the associator vanishes; no training or data can help |
| **C — linear on raw `c`** | −0.09 | fails — the target is quadratic |
| **D — MLP 8→16→1 on raw `c`** | −1.11 | unstructured baseline at this budget — needs far more data/capacity |

## Reading of the result
Only the non-associative model solves it, and — the point — it **trained** to, with the associator's
forward and its weight gradients `da, db` both running on the tensor-core kernels emitted by the
compiler. The quaternion model's failure is *structural* (its associator is identically zero), not a
matter of tuning; the linear model's failure is representational (quadratic target); the MLP is the
honest unstructured baseline — with only 192 samples and 16 hidden units it does not recover the
quadratic-over-8-dim form, which is exactly the sample-efficiency gap a correct inductive bias buys.

**Honest scope.** The MLP result is at *this* budget — a larger MLP with far more data would eventually
fit the quadratic form; the claim is not "no associative model can ever fit," it is "the associative
*algebra* model is structurally blind, and the non-associative model solves the task exactly and
sample-efficiently by construction, having *learned* its weights through the associator's VJP." This is
the artifact's ML spine: compile a non-associative operation (and its exact gradient) to tensor cores,
then *train a model that needs it*. Harness `run_nonassoc_headtohead.cu`.
