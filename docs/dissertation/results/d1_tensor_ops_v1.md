<!-- docs:meta
topic_id: repo.docs.dissertation.results.d1-tensor-ops-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d1-tensor-ops-v1
-->

# D1 Tensor Ops v1

## Scope

This sprint adds a first dynamic tensor stack for the Phase D ML path. The
implementation is f64-only, rank <= 8, row-major, contiguous, heap-backed, and
copy-based for reshape/transpose/permute. No tensor operation uses closures,
callbacks, BLAS, FFI, or external dependencies.

## Files

- `stdlib/tensor/types.sio`: tensor metadata, shape helpers, row-major strides,
  flat indexing, and heap-backed load/store helpers.
- `stdlib/tensor/ops.sio`: constructors, broadcasting elementwise operations,
  batched matrix multiplication, reductions, shape operations, and activations.
- `stdlib/tensor/lib.sio`: module entry point.
- `tests/stdlib/tensor/test_tensor_ops_d1.sio`: run-pass property and regression
  checks, including a finite-difference matmul Jacobian check.

## Operations

- constructors: `tensor_zeros`, `tensor_ones`, `tensor_full`,
  `tensor_from_slice`, `tensor_scalar`, `tensor_linspace`, `tensor_free`,
  `tensor_clone`
- elementwise: `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, with
  NumPy-style trailing-dimension broadcasting
- matmul: deterministic loop-order `(..., m, k) x (..., k, n) -> (..., m, n)`
- reductions: `tensor_sum`, `tensor_mean`, with `axis: Option<usize>` and
  `keepdims`
- shape: `tensor_reshape`, `tensor_transpose`, `tensor_permute`
- activations: `tensor_tanh`, `tensor_sigmoid`, `tensor_relu`, `tensor_gelu`

## Validation

Pinned compiler:

```text
SOUC_NATIVE=/workspace/sounio/bin/souc-linux-x86_64
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

Commands:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run tests/stdlib/tensor/test_tensor_ops_d1.sio
rg -n "\bfn\s*\([^)]*\)|=>|closure|extern \"C\"|BLAS|ffi" stdlib/tensor tests/stdlib/tensor
```

Results:

- Tensor run-pass test: `D1_TENSOR_OPS_PASS`
- Numerical Jacobian for `matmul` output `[0,0]` with respect to `a[0,0]`:
  PASS within `1e-6`
- Closure grep: clean for closures/callbacks; matches only `Option` arms and
  the comment documenting no BLAS/FFI acceleration

## Caveats

The dispatch sketch used `Box<[f64]>`. The current repository's compiler-proven
dynamic heap pattern is typed pointers backed by `heap_alloc`/`heap_free`, so
`Tensor.data` is `*mut f64`. This keeps the v1 tensor storage heap-backed and
contiguous without introducing FFI.

The property file is a focused run-pass suite rather than a randomized S2
property harness because `stdlib/testing/properties.sio` is not present on the
clean `origin/main` checkout.
