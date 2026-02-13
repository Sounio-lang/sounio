# Decision Record: Self-Hosted VM Hypercomplex Tower (Quat/Oct/Sed + ML Kernels)

**Date**: 2026-02-13
**Status**: Implemented (tests green in self-hosted bundle harness)
**Scope**: `self-hosted/ir/`, `self-hosted/vm/`, `self-hosted/test_vm.sio`

## Context

We need a self-hosted runtime (`self-hosted/vm/`) that can execute:

- multi-argument calls (>= 8 and 16 args for octonions/sedenions)
- floats (for ML)
- real heap objects (arrays/tuples/structs)

On top of that foundation, we want first-class hypercomplex values:

- quaternions (4D), octonions (8D), sedenions (16D)
- operations + component-wise activations
- forward-only dense/linear kernels for ML experiments
- explicit handling of sedenion zero divisors (exact + fast heuristics)

## Decisions

### 1) IR call arguments are preserved as a full list

**Decision**: `IrInstr.call_args` stores the full `IrRegList` for calls.

**Rationale**: `src1/src2` alone cannot represent `oct(a..h)` or dense kernels with many parameters.

**Compatibility**: `src1/src2` remain as cache of first two args; VM falls back when `call_args=None`.

### 2) VM registers use a typed value (`Val`) rather than raw `i64`

**Decision**: VM registers store a tagged value:

- `ValTag { Unit, Bool, Int, Float, Ptr }`
- `Val { tag, i, f }` (payload in `i`/`f`)

**Rationale**: ML requires correct float semantics; this also avoids ad-hoc "float_regs" drift.

**Operational semantics**:

- arithmetic promotes Int -> Float when mixed
- logical ops are Bool-only (reduce ambiguity)

### 3) Heap is ID-based with explicit `Ptr` encoding

**Decision**: heap objects are stored in `VmState.heap` and addressed by `Ptr` encoded as:

- `PTR_BASE + obj_id`

Objects support:

- `Struct` with Name-keyed fields
- `Array` / `Tuple` as a ValList (grow-on-index-set)
- `QuatObj` / `OctObj` / `SedObj` as fixed `[f64; N]` payloads

**Rationale**: avoids raw pointers; stable across execution backends and easy to serialize later.

**Field semantics**:

- `IrFieldGet/Set` ignore `field_idx` and use `Name` as the key (robust against lowering hash changes).

### 4) Hypercomplex operations are VM builtins

**Decision**: Hypercomplex values are constructed/operated via `vm_try_builtin_call(...)`.

Builtins (v1):

- Quat: `quat`, `quat_add/sub/scale/mul/conj/norm_sq/normalize`
- Oct: `oct`, `oct_add/sub/scale/mul/conj/norm_sq/normalize`, `oct_relu/sigmoid/tanh`
- Sed: `sed`, `sed_add/sub/scale/mul/conj/norm_sq/normalize`, `sed_relu/sigmoid/tanh`,
  `sed_is_zero_divisor_exact`, `sed_is_zero_divisor_fast`

**Rationale**: keeps the first demos executable without needing a full tensor stdlib.

### 5) Sedenion zero divisors: exact + fast

**Decision**:

- `sed_is_zero_divisor_exact`: build the 16x16 left-multiplication matrix and rank-test via elimination.
- `sed_is_zero_divisor_fast`: deterministic low-sparsity annihilator probe (cheap heuristic).

**Guardrail**: do not use `s * conj(s)` as a detector; it equals `norm_sq(s) * 1` under Cayley-Dickson and is not informative.

### 6) ML-first: forward-only dense/linear kernels as builtins

**Decision**:

- `oct_linear_fwd(weights,biases,x,in_size,out_size,activation) -> Array`
- `sed_linear_fwd(weights,biases,x,in_size,out_size,activation,zd_epsilon) -> Array`

Layout:

- `weights`: row-major `out_size * in_size`
- `biases`: `out_size`
- `x`: `in_size`

Activation enum:

- 0 None
- 1 ReLU
- 2 Sigmoid
- 3 Tanh
- 4 Normalize

**Sedenion regularization**:

- if `sed_is_zero_divisor_fast(weighted)` then `weighted.e0 += zd_epsilon` (identity bump)

### 7) Test strategy: run internal self-hosted tests via bundle harness

**Decision**: internal `self-hosted/test_vm.sio` runs in a concatenated bundle program and is executed via:

- `SOUNIO_SELFHOST_PIPELINE=rust souc run target/selfhost/bundle.sio`

**Rationale**: `souc run` is wired to the self-hosted compiler by default; bundling ensures all required self-hosted defs exist in one compilation unit.

## Notes / Gotchas

- `quat` is a reserved lexer token in the Rust compiler front-end; avoid using `quat` as an identifier in source where it matters.
- Avoid relying on `&&` short-circuit for bounds safety in the self-hosted VM; write explicit loops/ifs.

## Acceptance

The following are validated by `self-hosted/test_vm.sio`:

- multi-arg calls (>= 3 args tested; IR supports more)
- float arithmetic + comparisons
- heap array indexing and struct field access
- quat/oct/sed builtins end-to-end
- sedenion zero divisor exact test on a known pair
- oct/sed linear forward kernels on simple verifiable cases

