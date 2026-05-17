<!-- docs:meta
topic_id: repo.docs.dissertation.results.d4-optimizers-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d4-optimizers-v1
-->

# D.4 Tensor Optimizers V1

Branch: `codex/sounio-optimizers`

Gate marker: `D4_OPTIMIZER_INTEGRATION_PASS`

## Scope

D.4 implements tensor-level optimizer primitives over the D.1 dynamic tensor
stack on current `origin/main`. Current main includes D.1 tensors but does not
include the D.2 tensor tape module, so this lane deliberately avoids importing
the local D.2 bootstrap commit. `ParameterStore` stores direct gradient tensors
and preserves a `tape_id` field so a later D.2-aware sync adapter can be added
without changing optimizer state semantics.

The implementation is in `stdlib/nn/optimizer.sio` and uses fixed-capacity
parallel arrays for the parameter registry and optimizer moment buffers. This
matches the stability pattern used by the D.2 tape bootstrap and avoids
array-of-struct initialization brittleness in the current compiler.

## Implemented API

- `ParameterStore`
  - `param_store_new`
  - `param_store_register`
  - `param_store_zero_grad`
  - `param_store_set_grad`
  - `param_store_set_grad_value`
  - `param_store_grad_value`
  - `param_store_param_value`
  - `param_store_sync_grads_manual`
  - `param_store_free`
- Adam
  - `AdamState`
  - `adam_new`
  - `adam_new_full`
  - `adam_step`
- AdamW
  - `adamw_step`
- SGD with momentum
  - `SGDState`
  - `sgd_new`
  - `sgd_new_with_momentum`
  - `sgd_step`
- Gradient clipping
  - `clip_grad_global_norm`
  - `clip_grad_value`

## Validation

Focused validation lives in `tests/run-pass/d4_optimizer_integration.sio`.

The test covers:

- Parameter registration and `zero_grad`.
- Adam convergence on `f(w) = (w - 3)^2`.
- AdamW convergence with decoupled weight decay.
- Adam convergence on noiseless scalar linear regression with
  `W_true = 2.0`, `b_true = -1.0`, `n = 50`.
- SGD with momentum convergence on the same linear regression fixture.
- Global-norm and value gradient clipping.
- Deterministic Adam: two independent 100-step runs produce bit-identical
  parameter values.

Representative deterministic values from the implemented update rules:

| Check | Result |
| --- | ---: |
| Adam quadratic, 1000 steps | `w = 2.9991083973` |
| AdamW quadratic, 1200 steps | `w = 2.9977135747` |
| Adam linear regression | `W = 2.0`, `b = -1.0` |
| SGD+momentum linear regression | `W = 2.0`, `b = -1.0` |
| Clip fixture pre-norm | `50.0` |
| Clip fixture post-norm | `1.0` |

Focused test command:

```bash
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/run-pass/d4_optimizer_integration.sio
```

Observed marker:

```text
D4_OPTIMIZER_INTEGRATION_PASS
```

## D.2 Boundary

No D.2 APIs were incorporated. This is intentional: D.4 reads gradients and
updates parameters; it does not require autograd to prove optimizer semantics.
The current `ParameterStore` provides manual gradient setters and a stable
`tape_id` field. When the hardened D.2 tape lands on main, a narrow
`param_store_sync_grads` adapter can copy tape gradients into the existing
store without changing Adam, AdamW, SGD, or clipping behavior.

## Remaining Work

- Add the D.2-backed `param_store_sync_grads` adapter after the hardened tape is
  merged.
- Replace fixed-capacity backing arrays with `Vec<Tensor>` once the compiler
  reliably handles tensor-bearing dynamic collections in this path.
- Reuse this optimizer surface in the D.6 PINN loop after D.3 and D.5 tensor
  gates close.
