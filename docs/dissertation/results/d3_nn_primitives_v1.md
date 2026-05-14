<!-- docs:meta
topic_id: repo.docs.dissertation.results.d3-nn-primitives-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d3-nn-primitives-v1
-->

# D3 NN Primitives v1

## Scope

This lane implements the D.3 tensorized neural-network primitive surface on
branch `codex/sounio-nn-tensorized`.

Important base choice: `origin/main` contained C/M6/D.1/D.5, but did not contain
the D.2 autograd bootstrap. This lane first merged local branch
`codex/sounio-autograd-tape` at commit `8d1ccd7f` via merge commit `148ddc63`
so D.3 could compile against `stdlib/tensor/tape.sio`. The D.2 worktree itself
was not edited.

## Implemented Surface

- `stdlib/nn/layers.sio`
  - `Dense` with tensor weights, tensor bias, and D.2 tape IDs.
  - `dense_new`, deterministic Xavier-scale initialization, and
    `dense_forward`.
  - taped activation wrappers for tanh, sigmoid, relu, and gelu.
  - `layer_norm_forward` as tensor composition with arithmetic/reduction tape
    nodes and Newton sqrt approximation.
  - inverted `dropout_forward` with deterministic local D.3 RNG state.
  - `mse_loss` and `mae_loss`; MAE uses `relu(x) + relu(-x)` for the absolute
    value subgradient path.

## Validation Result

Focused test:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/stdlib/nn/test_nn_primitives_d3.sio

D3_NN_PRIMITIVES_BLOCKED
```

The bootstrap test proves a smaller D.2-compatible slice:

- Dense FD check passes for a 2x2 dense layer using
  `loss = sum(dense_forward(x))`; maximum absolute FD error observed during
  debugging was `7.076451e-11`.
- taped tanh records a D.2 activation node, but backward is not invoked in the
  final bootstrap test because `TAPE_TANH` backward exits during
  `tape_backward`.
- dropout builds a direct tensor output and tape ID on a 2x2 witness; MSE and
  MAE build tape IDs on the same witness. Direct scalar inspection of the
  returned MSE/MAE tensors was avoided in the bootstrap test because the current
  runtime showed ownership-sensitive exits after returning nested tensor tuples.
- XOR converges with the canonical 2-hidden-unit topology in a scalar SGD loop;
  this is retained as a topology/schedule sanity check, not as proof that D.2
  tensor activation backward is ready.

## Gate Status

`D3_NN_PRIMITIVES_PASS` is **not emitted**.

Blockers found in this lane:

1. `D3-DENSE-001`: requested Dense FD shape `in=4, out=3` is blocked by D.2
   taped matmul runtime behavior. `tensor_matmul` direct forward succeeds, but
   `tape_tensor_matmul` exits at runtime before `dense_forward` completes for
   the requested 3x4 x 4x1 witness. The D.2 bootstrap had only FD-proven the
   smaller 2x2 matmul path.
2. `D3-ACT-001`: `taped_tanh` forward recording succeeds, but
   `tape_backward` through `TAPE_TANH` exits at runtime. This matches the D.2
   self-audit caveat that activation backward functions exist but were not
   FD-proven.
3. `D3-LAYERNORM-001`: full LayerNorm FD validation is blocked by the same D.2
   reduction/activation/shape-depth fragility. The primitive implementation is
   present, but invoking the full composed path in the bootstrap test exits
   before the witness can be checked, so this lane does not claim the FD gate.

## Conclusion

D.3 is implemented as a coherent bootstrap layer over the actual D.2 API, but
the requested D.3 gate depends on a stronger D.2 tape than is currently present
on the imported bootstrap commit. The next action is to harden D.2 matmul shapes
and activation backward, then rerun D.3 FD gates without changing the public
marker semantics.
