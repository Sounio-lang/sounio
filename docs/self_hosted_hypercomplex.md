# Self-Hosted Hypercomplex Runtime (Quat/Oct/Sed) + ML Kernels

This document specifies the **self-hosted** (Sounio-side) runtime behavior for:

- typed runtime values (`Val`)
- heap objects (arrays/tuples/structs + hypercomplex payloads)
- hypercomplex builtins (quaternions/octonions/sedenions)
- forward-only dense/linear kernels (`oct_linear_fwd`, `sed_linear_fwd`)
- sedenion zero-divisor checks (exact vs fast)

Implementation lives in:

- `self-hosted/vm/vm.sio`
- `self-hosted/ir/ir.sio`
- tests in `self-hosted/test_vm.sio`

## 1. Value Model

Registers store:

- `ValTag`: `Unit | Bool | Int | Float | Ptr`
- `Val`: `{ tag, i: i64, f: f64 }`

Semantics:

- arithmetic promotes Int -> Float if either operand is Float
- comparisons on Float are exact `==`/`!=` (sufficient for current tests; ML code should prefer tolerances at the language level later)
- logical ops use Bool (`&&`, `||`, `!`) via `val_truthy`

## 2. Heap Model

Pointers are encoded:

- `Ptr = PTR_BASE + obj_id` (both i64)

Heap objects:

- `Struct`: field map keyed by `Name` (field_idx ignored)
- `Array` and `Tuple`: element list (grow-on-set at `idx == len`)
- `QuatObj`: payload `[f64; 4]`
- `OctObj`: payload `[f64; 8]`
- `SedObj`: payload `[f64; 16]`

### 2.1 Field Access

- `IrFieldSet(base, field_idx, src, name)`: update by `name`
- `IrFieldGet(dst, base, field_idx, name)`: lookup by `name`, returns Unit if missing

This avoids coupling to lowering-time hash schemes.

### 2.2 Index Access

- `IrIndexSet(base, idx, src)`:
  - if `idx == len`: append
  - if `idx < len`: overwrite
  - if `idx > len`: pads Units up to idx and sets
- `IrIndexGet(dst, base, idx)`: returns element or Unit if out of bounds

## 3. Calls With Many Arguments

IR calls preserve argument lists:

- `IrInstr.call_args` stores the full `IrRegList`
- `src1/src2` cache first two arguments for compatibility

VM call setup copies N arguments into callee param registers from `call_args` when present.

## 4. Hypercomplex Builtins

All hypercomplex values are heap-allocated objects and passed around as `Ptr`.

Component ordering:

- Quaternion: `quat = (w, x, y, z)`
- Octonion: `oct = (a, b, c, d, e, f, g, h)` with basis `{1, i, j, k, l, il, jl, kl}`
- Sedenion: `sed = (e0, e1, ..., e15)` (Cayley-Dickson from octonion pairs)

### 4.1 Quaternion API

- `quat(w,x,y,z) -> Ptr(QuatObj)`
- `quat_add(q1,q2) -> Ptr(QuatObj)`
- `quat_sub(q1,q2) -> Ptr(QuatObj)`
- `quat_scale(q,k) -> Ptr(QuatObj)`
- `quat_mul(q1,q2) -> Ptr(QuatObj)` (Hamilton product)
- `quat_conj(q) -> Ptr(QuatObj)`
- `quat_norm_sq(q) -> f64`
- `quat_normalize(q) -> Ptr(QuatObj)`

### 4.2 Octonion API

- `oct(a,b,c,d,e,f,g,h) -> Ptr(OctObj)`
- `oct_add/sub/scale/mul/conj/norm_sq/normalize`
- activations (component-wise):
  - `oct_relu`
  - `oct_sigmoid`
  - `oct_tanh`

Notes:

- octonion multiplication is **non-associative** (but alternative).

### 4.3 Sedenion API

- `sed(e0..e15) -> Ptr(SedObj)`
- `sed_add/sub/scale/mul/conj/norm_sq/normalize`
- activations (component-wise): `sed_relu/sigmoid/tanh`
- zero divisor:
  - `sed_is_zero_divisor_exact(s) -> bool`
  - `sed_is_zero_divisor_fast(s) -> bool` (heuristic)

Notes:

- sedenions have **zero divisors**; there is no general multiplicative inverse.

## 5. ML Kernels (Forward Only)

### 5.1 Octonion Dense/Linear Forward

`oct_linear_fwd(weights, biases, x, in_size, out_size, activation) -> Ptr(Array)`

- `weights`: Array length `out_size * in_size`, row-major: `w[out*in_size + j]`
- `biases`: Array length `out_size`
- `x`: Array length `in_size`
- `activation`: `0 None | 1 ReLU | 2 Sigmoid | 3 Tanh | 4 Normalize`

Accumulation:

- left-to-right (deterministic)

### 5.2 Sedenion Dense/Linear Forward

`sed_linear_fwd(weights, biases, x, in_size, out_size, activation, zd_epsilon) -> Ptr(Array)`

Same layout as octonions, but with regularization:

- if `sed_is_zero_divisor_fast(weighted)` then do `weighted.e0 += zd_epsilon` before accumulation

## 6. Exact vs Fast Zero Divisor

- `exact`: rank test of left multiplication map `L_s` (matrix 16x16, elimination)
- `fast`: annihilator probe on low-sparsity candidates (deterministic, not complete)

If you need correctness: use `exact`. If you need speed/regularization: use `fast`.

## References (Background)

- John C. Baez, *The Octonions* (2002)
- Deep Quaternion Networks (arXiv:1712.04604)
- Deep Octonion Networks (arXiv:1903.08478)

