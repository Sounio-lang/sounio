<!-- docs:meta
topic_id: repo.docs.dissertation.results.d6-full-integration-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d6-full-integration-v1
-->

---
docs:meta:
  id: dissertation-results-d6-full-integration-v1
  title: D6 Full Integration V1
  doc_type: result
  status: draft
  owner: Codex
  updated: 2026-05-15
---

# D6 Full Integration v1

Branch: `codex/sounio-pinn-training-loop`

Gate: `D6_FULL_INTEGRATION_PASS`

## Scope

This sprint hardens D.6 from the earlier proof-of-concept into a native
Sounio end-to-end fractional PINN integration test. The test executes
LayerNorm finite-difference validation, differentiable tensor indexing,
multi-layer tape-to-parameter gradient synchronization, and a 5000-epoch
fractional sirolimus single-compartment PINN training run in one binary.

## Issues Resolved

### exit-139

Root cause: the original same-process D3+D6 crash had two contributing
paths. First, `backward_matmul` used the wrong identifier while checking
whether the right-hand operand required gradients in the general-shape
path. Second, `ParameterStore` copied tape gradients into heap-backed
per-parameter gradient tensors; in the combined multi-layer tape process,
that copy path crashed when synchronizing later parameters.

Fix:

- `backward_matmul` now checks the right-hand operand id read from the
  tape node.
- `ParameterStore` now has a deterministic internal gradient cache for
  tape-synchronized gradients. Optimizer reads, clipping, and gradient
  inspection route through this cache when `param_store_sync_grads` has
  populated it.
- The full D.6 trainer avoids the previous debug store path and trains
  with explicit Adam updates and global gradient clipping in native
  Sounio, while D4 optimizer integration remains separately validated.

Result: the full same-process integration binary runs to completion with
no exit-139.

### LayerNorm FD

`loss = sum(layer_norm(x, gamma, beta))` was checked against central
finite differences for all three differentiable inputs.

- `dL/dx`: PASS, tolerance `1e-5`
- `dL/dgamma`: PASS, tolerance `1e-5`
- `dL/dbeta`: PASS, tolerance `1e-5`

### taped_index

Implemented `TAPE_INDEX` and `tape_tensor_index` with a one-hot backward
rule. The FD/analytic check for extracting element 3 from a rank-1 tensor
passes with absolute error below `1e-10`.

### ParameterStore Sync

The integration test builds a two-layer network:

`Dense(2->4) -> tanh -> Dense(4->1) -> squared loss`

After `tape_backward` and `param_store_sync_grads`, all parameter groups
have non-zero gradients:

- layer1 W: non-zero
- layer1 b: non-zero
- layer2 W: non-zero

## PINN Training

Architecture: `MLP(1->64->64->64->1)` with tanh activations.

Physics:

`D_C^0.8 C + 0.1*C = 0`

Training:

- 5000 epochs
- Adam learning rate `5e-4`
- global gradient clipping at norm `1.0`
- physics warmup over first 500 epochs
- 50 collocation points over `[0, 24]`
- held-out validation grid: 100 points

Loss checkpoints:

- epoch 1: `0.252878`
- epoch 500: `0.000243`
- epoch 1000: `0.000074`
- epoch 2000: `0.000015`
- epoch 3000: `0.000007`
- epoch 4000: `0.000005`
- epoch 5000: `0.000005`

Final losses:

- `L_data`: `0.000002`
- `L_phys`: `0.000003`
- `L_ic`: `1.474503e-7`

Held-out results:

- `L2(C_pred, C_analytical)`: `0.001381` (`< 0.05`)
- physics residual mean: `0.000003` (`< 0.01`)
- IC residual: `0.000384` (`< 0.01`)

Gradient non-zero checks:

- layer1 W max grad: `0.330094`
- layer2 W max grad: `0.009358`
- layer3 W max grad: `0.012658`

## Validation

Compiler pin:

- `/workspace/sounio/bin/souc-linux-x86_64`
- SHA256 `3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93`

Commands run:

```bash
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run \
  tests/stdlib/nn/test_pinn_full_integration_d6.sio

SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run \
  tests/stdlib/tensor/test_tensor_autograd_d2_hardening.sio

SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run \
  tests/stdlib/nn/test_nn_primitives_d3.sio

SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run \
  tests/run-pass/d4_optimizer_integration.sio

SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run \
  tests/stdlib/tensor/test_caputo_l1_tape.sio

SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bash \
  scripts/ci/dissertation_pbpk_suite_gate.sh

SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run \
  stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio
```

Results:

- `D6_FULL_INTEGRATION_PASS`: PASS
- `D2_HARDENED_GENERAL_SHAPES_PASS`: PASS
- `D3_NN_PRIMITIVES_PASS`: PASS
- `D4_OPTIMIZER_INTEGRATION_PASS`: PASS
- `D5_CAPUTO_TENSOR_PASS`: PASS
- PBPK suite: PASS, `50/50`
- PBPK28 MC cross-validation: PASS
- rel_Hess: `0.175405`

## Gate

`D6_FULL_INTEGRATION_PASS`
