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

> **Reproduction command corrected (2026-09-13).** The command first recorded here was
> `SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run <test>`. `bin/souc`
> execs that override with its arguments unchanged (it already did at the merge that added
> this file, `bebd78d74c`). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc`
> began refusing the form: lean_single stopped at
> `error: no main` (a current-source lean_single shows why: it opens `run`, which does not exist,
> as a 0-byte source). How the recorded values were originally produced cannot be established
> from this repository's history. `bin/souc` now refuses the form (exit 64). The command below uses the
> ELF's raw `<source.sio> <output>` interface. Run it from the repository root: lean_single
> resolves stdlib imports relative to the working directory. The pinned binary (sha256
> `3cbea2b4…`) is no longer in the repository; the re-run used `bin/souc-linux-x86_64`, sha256
> `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`.
> Re-run 2026-09-13: lean_single printed `D1_TENSOR_OPS_PASS`. Only that marker was re-checked;
> every other value in this file is the original record. The default Madaros engine
> (`bin/souc run <test>` with no `SOUNIO_SOUC_BIN` set; `bin/madaros-linux-x86_64` sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) rejects this test today
> with `error[E037]` in `stdlib/tensor/ops.sio`.

Commands (from the repository root):

```text
cd "$(git rev-parse --show-toplevel)" && bin/souc-linux-x86_64 tests/stdlib/tensor/test_tensor_ops_d1.sio /tmp/d1_tensor_ops.elf && /tmp/d1_tensor_ops.elf
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
