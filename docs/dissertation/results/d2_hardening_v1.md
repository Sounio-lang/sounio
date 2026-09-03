<!-- docs:meta
topic_id: repo.docs.dissertation.results.d2-hardening-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d2-hardening-v1
-->

# D2 Autograd Hardening v1

## Scope

This lane is a surgical hardening pass on the D.2 closure-free tensor tape. It
targets the two blockers surfaced by D.3:

- `D3-DENSE-001`: `tape_tensor_matmul` backward failed for the requested dense
  shapes beyond the original 2x2 bootstrap witness.
- `D3-ACT-001`: `TAPE_TANH` backward exited during `tape_backward`.

No new tape op tags were added.

## Root Causes

### D3-DENSE-001

The matmul backward path reconstructed input `TensorShape` values from tape
metadata during reverse traversal. On the larger dense witness, direct debug
inspection showed that rank metadata reads could return pointer-like values
even while the flat dimensions and lengths remained usable. This matched the
original D.2 self-audit warning about brittle stored shape metadata.

The fix keeps matmul backward in the flat-buffer style used by the D.2
bootstrap. It derives `m` and `n` from the saved output dimensions on the tape
node and derives `k` from the flat input length, then runs deterministic
row-major loops for both input gradients.

### D3-ACT-001

The activation backward functions allocated temporary tensors and reconstructed
tensor views from tape shapes. That path hit the same metadata brittleness.

The fix computes tanh, sigmoid, and ReLU gradients directly over the saved
forward buffers and gradient buffers:

- tanh: `grad_x = grad_out * (1 - y^2)`
- sigmoid: `grad_x = grad_out * y * (1 - y)`
- ReLU: `grad_x = grad_out` when saved input is positive, otherwise `0`

The ReLU subgradient at zero is documented as the tape convention `0`.

## Validation

Pinned compiler:

```text
/workspace/sounio/bin/souc-linux-x86_64
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

Focused hardening test:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/stdlib/tensor/test_tensor_autograd_d2_hardening.sio

D2_HARDENED_GENERAL_SHAPES_PASS
```

The hardening test checks finite-difference gradients within `1e-6` absolute
tolerance for both matmul operands on:

- `(3x4) @ (4x1)`
- `(8x6) @ (6x1)`
- `(4x4) @ (4x4)`
- `(1x5) @ (5x3)`

It also checks finite-difference gradients within `1e-6` absolute tolerance for:

- `sum(tanh(x))`
- `sum(sigmoid(x))`
- `sum(relu(x))`, with the zero subgradient convention documented as `0`

Original D.2 bootstrap regression:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/stdlib/tensor/test_tensor_autograd_d2.sio

D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS
```

## D3 Boundary

This branch intentionally does not merge or edit the D.3 worktree. The D.3 XOR
integration rerun is therefore left to the D.3 branch after it incorporates this
D.2 hardening commit. This preserves the user constraint that this lane only
touch D.2-owned files.
