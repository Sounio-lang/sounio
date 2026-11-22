<!-- docs:meta
topic_id: repo.docs.qnn.architecture-deep-dive
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.qnn.architecture-deep-dive
-->

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

**Location**: `compiler/src/types/core.rs` (historical Rust source tree, removed on `main`)

The type checker enforces quaternion semantics:
- Hamilton product is non-commutative: `q1 ⊗ q2 ≠ q2 ⊗ q1`
- Component access via `.w`, `.x`, `.y`, `.z`
- Intrinsics resolve to runtime functions

### 1.2 HIR Representation

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

**Location**: `compiler/src/backend/native/quat_runtime.rs` (historical Rust source tree, removed on `main`)

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

**Location**: `compiler/src/backend/native/quat_simd_dispatch.rs` (historical Rust source tree, removed on `main`)

**Fallback Chain**: AVX-512 → AVX2 → NEON → Scalar

### 2.3 AVX2 Implementation

**Location**: `compiler/src/backend/native/quat_simd_avx2.rs` (historical Rust source tree, removed on `main`)

Processes 2 quaternions per iteration:

```
__m256 Register Layout (256 bits = 8 floats):
┌────────────────────────────────────────────────────────┐
│ q0.w │ q0.x │ q0.y │ q0.z │ q1.w │ q1.x │ q1.y │ q1.z │
└────────────────────────────────────────────────────────┘
  f32    f32    f32    f32    f32    f32    f32    f32
```

### 2.4 INT8 Runtime

**Location**: `compiler/src/backend/native/quat_runtime_i8.rs` (historical Rust source tree, removed on `main`)

---

## 3. GPU Codegen Architecture

### 3.1 PTX Generation

**Location**: `compiler/src/codegen/gpu/qnn_kernels.rs` (historical Rust source tree, removed on `main`)

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

### 4.1 SIMD Vectorization

| Level | Quaternions/Iteration | Speedup | Instructions |
|-------|----------------------|---------|--------------|
| Scalar | 1 | 1× | 16 MUL + 12 ADD |
| AVX2 | 2 | 4-5× | 8 FMA |
| AVX-512 | 4 | 6-8× | 8 FMA |
| NEON | 1 | 3-4× | 8 FMA |

### 4.2 GPU Warp-Level Optimization

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

## See Also

- [Programming Guide](PROGRAMMING_GUIDE.md) — QNN fundamentals
- [Performance Handbook](PERFORMANCE_HANDBOOK.md) — Optimization techniques
- [Migration Guide](MIGRATION_GUIDE.md) — Converting float networks
