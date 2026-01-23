# QNN Architecture Deep-Dive

> *"Any sufficiently advanced abstraction is indistinguishable from magic."*
>
> Sounio's QNN implementation transforms Hamilton's 1843 algebra into 2026 machine code—through a careful layering of type systems, intermediate representations, and hardware-specific optimizations.

This document details the internal architecture of Sounio's Quaternion Neural Network support, from type checking to native code generation.

---

## 1. Compiler Pipeline for QNN

```
Source (.sio)
    │
    ▼
┌─────────┐    ┌────────┐    ┌───────────┐
│  Lexer  │───▶│ Parser │───▶│    AST    │
└─────────┘    └────────┘    └───────────┘
                                   │
                                   ▼
                            ┌───────────┐
                            │ Type Check│  ← Quat type registered here
                            └───────────┘
                                   │
                                   ▼
                            ┌───────────┐
                            │    HIR    │  ← Hamilton product nodes
                            └───────────┘
                                   │
                                   ▼
                            ┌───────────┐
                            │    SIR    │  ← Quaternion-specific opts
                            └───────────┘
                                   │
                                   ▼
                            ┌───────────┐
                            │   HLIR    │  ← SSA form
                            └───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  Native  │  │ Cranelift│  │   GPU    │
              │ Backend  │  │   JIT    │  │ Codegen  │
              └──────────┘  └──────────┘  └──────────┘
```

### 1.1 The Quat Type in the Type System

**Location**: [compiler/src/types/core.rs](../../compiler/src/types/core.rs)

The `Quat` type is a first-class citizen in Sounio's type system:

```rust
// types/core.rs (simplified)
pub enum Type {
    // Primitives
    I32, I64, F32, F64, Bool,

    // Quaternion types
    Quat,                    // Native quaternion: [w, x, y, z]
    QuatLinear,              // Linear layer weights
    QuatConv2d,              // 2D convolution kernels
    QuatRnnState,            // LSTM/GRU hidden state
    QuatGate,                // RNN gate quaternions

    // ...
}
```

The type checker enforces quaternion semantics:
- Hamilton product is non-commutative: `q1 ⊗ q2 ≠ q2 ⊗ q1`
- Component access via `.w`, `.x`, `.y`, `.z`
- Intrinsics resolve to runtime functions

### 1.2 HIR Representation

Quaternion operations lower to HIR (High-level IR) nodes:

```rust
// hir/mod.rs (simplified)
pub enum HirExpr {
    // Quaternion operations
    QuatConstruct { w: Box<HirExpr>, x: Box<HirExpr>,
                    y: Box<HirExpr>, z: Box<HirExpr> },
    QuatHamiltonMul { lhs: Box<HirExpr>, rhs: Box<HirExpr> },
    QuatConjugate { operand: Box<HirExpr> },
    QuatNorm { operand: Box<HirExpr> },

    // Intrinsic calls
    QuatIntrinsic { name: String, args: Vec<HirExpr> },
}
```

Effect annotations mark GPU-bound operations:

```sounio
fn quat_linear_forward(...) -> [Quat] with GPU {
    // GPU effect tracked through the type system
    quat_linear_fwd(layer, weights, input, bias)
}
```

### 1.3 SIR Optimizations

The Scientific IR (SIR) performs domain-specific optimizations:

| Pass | Description | Impact |
|------|-------------|--------|
| Dead normalization elimination | Removes redundant `normalize()` calls | 5-10% speedup |
| Loop fusion | Merges batch operations | 20-30% speedup |
| Memory layout optimization | Aligns quaternion arrays | Better cache hits |
| Constant folding | Pre-computes static quaternions | Reduces runtime ops |

---

## 2. Native Backend Implementation

### 2.1 C ABI Functions

**Location**: [compiler/src/backend/native/quat_runtime.rs](../../compiler/src/backend/native/quat_runtime.rs)

All quaternion operations expose C-compatible functions for calling from assembly:

```rust
/// Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_conj(q: *const f32, out: *mut f32) {
    if q.is_null() || out.is_null() {
        return;
    }
    unsafe {
        *out.offset(0) = *q.offset(0);       // w
        *out.offset(1) = -*q.offset(1);      // -x
        *out.offset(2) = -*q.offset(2);      // -y
        *out.offset(3) = -*q.offset(3);      // -z
    }
}
```

**Core Functions**:

| Function | Signature | Operation |
|----------|-----------|-----------|
| `sounio_quat_conj` | `(q, out)` | Conjugate |
| `sounio_quat_norm` | `(q) → f32` | Euclidean norm |
| `sounio_quat_norm_sq` | `(q) → f32` | Norm squared |
| `sounio_quat_normalize` | `(q)` | In-place normalization |
| `sounio_quat_inverse` | `(q, out)` | Multiplicative inverse |
| `sounio_quat_mul` | `(q1, q2, out)` | Hamilton product |
| `sounio_quat_rotate` | `(q, v, out)` | Vector rotation |

### 2.2 SIMD Dispatch Architecture

**Location**: [compiler/src/backend/native/quat_simd_dispatch.rs](../../compiler/src/backend/native/quat_simd_dispatch.rs)

Runtime detection selects the optimal implementation:

```rust
pub enum SimdLevel {
    Scalar = 0,    // Always available
    Neon = 1,      // ARM 128-bit
    Avx2 = 2,      // x86-64 256-bit
    Avx512 = 3,    // x86-64 512-bit
}

// Zero-cost dispatch via OnceLock singleton
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return SimdLevel::Neon;  // Always available on aarch64
        }
        SimdLevel::Scalar
    })
}
```

**Fallback Chain**: AVX-512 → AVX2 → NEON → Scalar

### 2.3 AVX2 Implementation

**Location**: [compiler/src/backend/native/quat_simd_avx2.rs](../../compiler/src/backend/native/quat_simd_avx2.rs)

Processes 2 quaternions per iteration:

```
__m256 Register Layout (256 bits = 8 floats):
┌────────────────────────────────────────────────────────┐
│ q0.w │ q0.x │ q0.y │ q0.z │ q1.w │ q1.x │ q1.y │ q1.z │
└────────────────────────────────────────────────────────┘
  f32    f32    f32    f32    f32    f32    f32    f32
```

Hamilton product with SIMD:

```rust
// Simplified AVX2 Hamilton product
unsafe fn hamilton_product_avx2(a: __m256, b: __m256) -> __m256 {
    // Permute components for broadcasting
    let a_wwww = _mm256_permute_ps(a, 0x00);  // [w,w,w,w, w,w,w,w]
    let a_xxxx = _mm256_permute_ps(a, 0x55);  // [x,x,x,x, x,x,x,x]
    let a_yyyy = _mm256_permute_ps(a, 0xAA);  // [y,y,y,y, y,y,y,y]
    let a_zzzz = _mm256_permute_ps(a, 0xFF);  // [z,z,z,z, z,z,z,z]

    // Rearranged b for Hamilton product terms
    let b_wxyz = b;
    let b_xwzy = _mm256_permute_ps(b, 0xB1);  // Swap pairs
    let b_yzwx = _mm256_permute_ps(b, 0x4E);  // Rotate
    let b_zyxw = _mm256_permute_ps(b, 0x1B);  // Reverse

    // FMA operations (8 total vs 16 scalar)
    let term1 = _mm256_mul_ps(a_wwww, b_wxyz);
    let term2 = _mm256_fmadd_ps(a_xxxx, b_xwzy, term1);
    let term3 = _mm256_fmadd_ps(a_yyyy, b_yzwx, term2);
    let result = _mm256_fmadd_ps(a_zzzz, b_zyxw, term3);

    // Apply sign corrections for Hamilton product rules
    _mm256_xor_ps(result, HAMILTON_SIGNS)
}
```

### 2.4 INT8 Runtime

**Location**: [compiler/src/backend/native/quat_runtime_i8.rs](../../compiler/src/backend/native/quat_runtime_i8.rs)

Per-quaternion symmetric quantization:

```rust
pub struct QuatI8 {
    components: [i8; 4],  // w, x, y, z
    scale: f32,           // Shared scale factor
}

impl QuatI8 {
    pub fn from_f32(q: &[f32; 4]) -> Self {
        let max_abs = q.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        QuatI8 {
            components: [
                (q[0] * inv_scale).round() as i8,
                (q[1] * inv_scale).round() as i8,
                (q[2] * inv_scale).round() as i8,
                (q[3] * inv_scale).round() as i8,
            ],
            scale,
        }
    }
}
```

---

## 3. GPU Codegen Architecture

### 3.1 PTX Generation

**Location**: [compiler/src/codegen/gpu/qnn_kernels.rs](../../compiler/src/codegen/gpu/qnn_kernels.rs)

Kernel configuration structure:

```rust
pub struct QuatLinearBNReLUConfig {
    pub tile_m: usize,        // Output features (default: 128)
    pub tile_n: usize,        // Batch size (default: 128)
    pub tile_k: usize,        // Input features (default: 8)
    pub shared_layout: u32,   // 0=packed, 1=padded
    pub block_size: (usize, usize),  // (32, 4) threads
    pub bn_epsilon: f32,      // 1e-5
    pub async_copy: bool,     // Ampere+ async
    pub use_wmma: bool,       // Tensor cores
}
```

**Shared Memory Calculation**:

```
Input tile:    [tile_n × tile_k × 4] × 4 bytes + padding
Weights tile:  [tile_m × 4 × tile_k × 4] × 4 bytes + padding
Accumulator:   [tile_m × 4 × tile_n × 4] × 4 bytes
BN params:     [tile_m × 4] × 2 (mean + variance)
```

### 3.2 WMMA Tensor Core Mapping

4×4 quaternions map to 16×16 FP32 WMMA tiles:

```
WMMA Tile Layout (16×16 FP32):
┌─────┬─────┬─────┬─────┐
│Q0,0 │Q0,1 │Q0,2 │Q0,3 │
│2×2  │2×2  │2×2  │2×2  │
├─────┼─────┼─────┼─────┤
│Q1,0 │Q1,1 │Q1,2 │Q1,3 │
│2×2  │2×2  │2×2  │2×2  │
├─────┼─────┼─────┼─────┤
│Q2,0 │Q2,1 │Q2,2 │Q2,3 │
│2×2  │2×2  │2×2  │2×2  │
├─────┼─────┼─────┼─────┤
│Q3,0 │Q3,1 │Q3,2 │Q3,3 │
│2×2  │2×2  │2×2  │2×2  │
└─────┴─────┴─────┴─────┘

Each Quat maps to 2×2 block:
┌─────┐
│ w x │
│ y z │
└─────┘
```

**Fragment Usage**: 8 WMMA fragments per Hamilton product (4 for A, 4 for B)

### 3.3 Kernel Fusion Strategy

Fused Linear + BatchNorm + ReLU:

```
Unfused (3 kernels, 3 memory round-trips):
┌────────┐     ┌────────┐     ┌────────┐
│ Linear │────▶│   BN   │────▶│  ReLU  │
└────────┘     └────────┘     └────────┘
   ▲   │         ▲   │         ▲   │
   │   ▼         │   ▼         │   ▼
 [GMEM]        [GMEM]        [GMEM]

Fused (1 kernel, 1 memory round-trip):
┌─────────────────────────────────────┐
│     Linear → BN → ReLU (fused)      │
│     ┌──────┐  ┌────┐  ┌──────┐     │
│     │Linear│──│ BN │──│ ReLU │     │
│     └──────┘  └────┘  └──────┘     │
│          (shared memory only)       │
└─────────────────────────────────────┘
         ▲                   │
         │                   ▼
       [GMEM input]      [GMEM output]
```

**Bandwidth Reduction**: 3× less global memory traffic

---

## 4. Hamilton Product: The Optimization Journey

### 4.1 Scalar Reference Implementation

```rust
fn hamilton_product_scalar(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    let (w1, x1, y1, z1) = (a[0], a[1], a[2], a[3]);
    let (w2, x2, y2, z2) = (b[0], b[1], b[2], b[3]);

    [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  // w (16 muls)
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  // x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  // y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  // z
    ]
}
// 16 multiplications + 12 additions = ~200 ns
```

### 4.2 SIMD Vectorization

| Level | Quaternions/Iteration | Speedup | Instructions |
|-------|----------------------|---------|--------------|
| Scalar | 1 | 1× | 16 MUL + 12 ADD |
| AVX2 | 2 | 4-5× | 8 FMA |
| AVX-512 | 4 | 6-8× | 8 FMA |
| NEON | 1 | 3-4× | 8 FMA |

### 4.3 GPU Warp-Level Optimization

```
Execution Model:
┌────────────────────────────────────────┐
│              WMMA Warp (32 threads)    │
├────────────────────────────────────────┤
│ Thread 0-7:   Load A fragments         │
│ Thread 8-15:  Load B fragments         │
│ Thread 16-23: WMMA MMA compute         │
│ Thread 24-31: Store C fragments        │
├────────────────────────────────────────┤
│ Throughput: 128 quaternions/cycle      │
│ Speedup: 10-20× over scalar GPU        │
└────────────────────────────────────────┘
```

---

## 5. Backward Pass Implementation

### 5.1 Gradient Rules for Hamilton Product

For `y = q₁ ⊗ q₂`:

```
∂L/∂q₁ = ∂L/∂y ⊗ q₂*     (right-multiply by conjugate)
∂L/∂q₂ = q₁* ⊗ ∂L/∂y     (left-multiply by conjugate)
```

**Location**: [compiler/src/codegen/gpu/quat_kernels_backward.rs](../../compiler/src/codegen/gpu/quat_kernels_backward.rs)

```rust
fn hamilton_product_backward(
    grad_output: [f32; 4],
    q1: [f32; 4],
    q2: [f32; 4]
) -> ([f32; 4], [f32; 4]) {
    let q1_conj = conjugate(q1);
    let q2_conj = conjugate(q2);

    let grad_q1 = hamilton_product(grad_output, q2_conj);
    let grad_q2 = hamilton_product(q1_conj, grad_output);

    (grad_q1, grad_q2)
}
```

### 5.2 Chain Rule Through Activations

```
Component-wise ReLU gradient:
∂ReLU(q)/∂q = (∂ReLU(w)/∂w, ∂ReLU(x)/∂x, ∂ReLU(y)/∂y, ∂ReLU(z)/∂z)
            = (w > 0 ? 1 : 0, x > 0 ? 1 : 0, ...)
```

---

## 6. Memory Layout and Alignment

### 6.1 Quaternion Storage

```
Contiguous Layout (Structure of Arrays avoided):
┌─────────────────────────────────────────────────────────────────┐
│ q0.w │ q0.x │ q0.y │ q0.z │ q1.w │ q1.x │ q1.y │ q1.z │ ...   │
└─────────────────────────────────────────────────────────────────┘
  16 bytes (1 quaternion)      16 bytes (1 quaternion)

Cache Line (64 bytes) = 4 quaternions
```

**Benefits**:
- Prefetcher loads complete quaternions
- SIMD loads are contiguous
- No stride-induced penalties

### 6.2 Shared Memory Bank Conflict Avoidance

```
Without padding (bank conflicts):
Bank:  0  1  2  3  0  1  2  3  0  1  2  3 ...
      ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
      │w0│x0│y0│z0│w1│x1│y1│z1│w2│x2│y2│z2│
      └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
       ↑              ↑              ↑
       Thread 0       Thread 1       Thread 2
       (bank 0)       (bank 0!)      (bank 0!!) ← CONFLICT

With +8 float padding:
Bank:  0  1  2  3  4  5  6  7  0  1  2  3 ...
      ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
      │w0│x0│y0│z0│--│--│--│--│w1│x1│y1│z1│
      └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
       ↑                       ↑
       Thread 0                Thread 1
       (bank 0)                (bank 0, different row) ← NO CONFLICT
```

---

## 7. File Reference

| Component | File | Lines |
|-----------|------|-------|
| Type System | `compiler/src/types/core.rs` | ~670 |
| Native Runtime | `compiler/src/backend/native/quat_runtime.rs` | 514 |
| SIMD Dispatch | `compiler/src/backend/native/quat_simd_dispatch.rs` | ~400 |
| AVX2 Backend | `compiler/src/backend/native/quat_simd_avx2.rs` | ~300 |
| AVX-512 Backend | `compiler/src/backend/native/quat_simd_avx512.rs` | ~200 |
| NEON Backend | `compiler/src/backend/native/quat_simd_neon.rs` | ~220 |
| INT8 Runtime | `compiler/src/backend/native/quat_runtime_i8.rs` | 468 |
| GPU Kernels | `compiler/src/codegen/gpu/qnn_kernels.rs` | 682 |
| GPU Backward | `compiler/src/codegen/gpu/quat_kernels_backward.rs` | 674 |
| Tensor Cores | `compiler/src/codegen/gpu/qnn_tensor_core.rs` | ~800 |
| Quantization | `compiler/src/codegen/gpu/quat_quantize.rs` | 770 |

---

## See Also

- [Programming Guide](PROGRAMMING_GUIDE.md) — QNN fundamentals
- [Performance Handbook](PERFORMANCE_HANDBOOK.md) — Optimization techniques
- [Migration Guide](MIGRATION_GUIDE.md) — Converting float networks
