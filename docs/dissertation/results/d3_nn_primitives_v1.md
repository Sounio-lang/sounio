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

The rerun lane then cherry-picked D.2 hardening (`9d852974`) as local commit
`cf94b323` and added the D.3-specific tape-shape resilience needed by the Dense
path. The final D.3 rerun commit keeps all work scoped to
`codex/sounio-nn-tensorized`.

## Implemented Surface

- `stdlib/nn/layers.sio`
  - `Dense` with tensor weights, tensor bias, and D.2 tape IDs.
  - `dense_new`, deterministic Xavier-scale initialization, and
    `dense_forward`. `dense_new` is intentionally construction-only in this
    compiler lane; mutating the tape inside a struct-returning constructor
    corrupts later rank metadata. The test binds `W`, input, then `b` in the
    active tape before calling `dense_forward`.
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

D3_NN_PRIMITIVES_PASS
```

The rerun test now proves the D.3 gate slice against hardened D.2:

- Dense FD check passes for the requested `in=4, out=3` shape using
  `loss = sum(dense_forward(x))`. The observed maximum absolute FD error during
  the rerun was `8.801793e-11`, below the `1e-6` gate.
- taped tanh backward is invoked through `tape_backward` and checked against
  central finite differences on `[-0.7, 0.2, 1.1]` with an asserted maximum
  error below `1e-6`.
- LayerNorm now runs a tape-backed smoke check: `layer_norm_forward` executes,
  `sum(layer_norm(x))` backpropagates, and the beta gradient is checked against
  the analytical value `2.0` on the 2x2 witness.
- dropout builds a direct tensor output and tape ID on a 2x2 witness; MSE and
  MAE build tape IDs on the same witness.
- XOR converges with the canonical 2-hidden-unit topology in a scalar SGD loop,
  preserving the `loss < 0.05` D.3 integration sanity check.

## Gate Status

`D3_NN_PRIMITIVES_PASS` is emitted.

Resolved blockers:

1. `D3-DENSE-001`: resolved for the requested Dense FD path. D.2 hardening fixed
   general matmul backward, and the D.3 rerun added shape reconstruction that
   infers rank from stored dimensions when a tape rank slot is corrupt.
2. `D3-ACT-001`: resolved for `taped_tanh`; backward now runs and matches finite
   differences within the gate tolerance.
3. `D3-LAYERNORM-001`: reduced from blocking to a residual expansion item. The
   rerun proves a tape-backed LayerNorm smoke and beta-gradient check. Full
   `dL/dx`, `dL/dgamma`, `dL/dbeta` finite-difference coverage should still be
   added before using LayerNorm as a critical training primitive.

## Conclusion

D.3 now closes the requested rerun gate on the hardened D.2 tape. The remaining
engineering caution is not a D.3 gate blocker: constructor-time tape mutation
inside a struct-returning Dense constructor remains fragile in this compiler
lane, so Dense binding is performed explicitly in the active caller tape.
