<!-- docs:meta
topic_id: repo.docs.dissertation.results.d2-autograd-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d2-autograd-v1
-->

# D2 Autograd v1

## Scope

This lane adds a closure-free reverse-mode tape for the D.1 dynamic tensor stack.
The branch is based on the local `codex/sounio-tensor-stack` commit because D.1
was not present on `origin/main` when this lane was launched.

The tape uses explicit op tags and a reverse-order `if` dispatcher. It does not
store closures, callbacks, or function pointers in tape nodes.

## Implementation Notes

- `stdlib/tensor/tape.sio` defines tag constants for the D.1 arithmetic,
  matmul, reduction, shape, and activation operations.
- `TensorTape` uses fixed-capacity primitive arrays rather than `[TapeNode; N]`
  or `[Tensor; N]` fields. The array-of-struct form compiled but was unstable
  in this compiler/runtime lane.
- Gradients are allocated per tape value and accumulated in flat storage. The
  worked matmul chain uses flat backward loops to avoid relying on brittle
  intermediate shape reconstruction.
- Shape and activation backward functions are implemented, but the focused gate
  only proves forward recording for the deep shape/activation chain. The
  finite-difference proof covers the critical `matmul -> add -> sub -> mul ->
  sum` training loss.

## Validation

Pinned compiler:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

Focused run:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run tests/stdlib/tensor/test_tensor_autograd_d2.sio
D2_AUTOGRAD_TAPE_CLOSURE_FREE_PASS
```

Worked example:

```text
loss = sum((W @ x + b - y)^2)
```

The tape gradients for `W` and `b` match central finite differences within
`1e-6` absolute tolerance on the focused 2x2/2x1 witness. The dispatch also
checks repeated-parent accumulation with `x2 = (x + x)^2` and scalar division.

## Static Audit

The no-closure audit over `stdlib/tensor/tape.sio` found no closure syntax,
callback fields, or stored function pointers. The only grep matches were
`Option` match arms (`None =>`, `Some(...) =>`) and the test marker print.

## Caveat

This should be treated as a D.2 bootstrap proof, not as a fully mature tensor
autograd library. The op tags and backward functions are present, but the
compiler/runtime still shows fragile behavior around stored shape metadata for
deep intermediate tensors. D.3/D.5 tensor work should keep the same flat,
explicit-gradient style until a broader shape-metadata regression suite lands.
