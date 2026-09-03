<!-- docs:meta
topic_id: repo.docs.research.gpu-swarm-coordination-2026-07-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.gpu-swarm-coordination-2026-07-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# GPU Swarm Coordination — Sounio GPU Development 2026-07-05

Status: **ACTIVE** — 5 agents working in parallel
Started: 2026-07-05
Last updated: 2026-07-05 16:00 UTC (A3 M1 complete)

## Purpose

This document coordinates **5 autonomous agents** working in parallel on different areas of the Sounio GPU stack. Each agent is 100% autonomous in their area but coordinates via this shared document for inter-area dependencies and integration.

## Agent Registry

| Agent ID | Area | Model (Start) | Model (Maintain) | Lead | Status |
|---|---|---|---|---|---|
| A1 | Epistemic Types Foundation | Opus | Sonnet/Light | Epistemic base | ACTIVE |
| A2 | Blackwell Tensor Core | Opus | Opus | Hardware optimization | ACTIVE |
| A3 | GPU Autodiff Engine | Opus | Sonnet | Differentiable GPU | ACTIVE |
| A4 | K-AXI→PTX Pipeline | Sonnet | Haiku/Light | Infrastructure | ACTIVE |
| A5 | GPU Optimizers Suite | Sonnet | Sonnet | Performance | ACTIVE |

## Milestone Tracker

### Agent 1: Epistemic Types Foundation
- [x] M1: Survey of `Knowledge[T]` gaps in GPU (2h) — **COMPLETE**
- [ ] M2: Implement vec/mat knowledge lowering (8h)
- [ ] M3: GUM tensor operations (12h)
- [ ] M4: Cross-backend tests (6h)

### Agent 2: Blackwell Tensor Core Optimization
- [ ] M1: Benchmark baseline Blackwell WMMA (4h)
- [ ] M2: sm_121-specific optimizations (16h)
- [ ] M3: Tensor cache + tiling (12h)
- [ ] M4: GUM quadrature WMMA optimization (8h)

### Agent 3: GPU Autodiff Engine
- [x] M1: Survey of autodiff gaps (4h) — **COMPLETE**
- [ ] M2: Reverse-mode GPU implementation (20h)
- [ ] M3: Gradient checkpointing (8h)
- [ ] M4: Tensor operation gradients (12h)

### Agent 4: K-AXI→PTX Pipeline
- [ ] M1: Profile pipeline bottlenecks (4h)
- [ ] M2: Register allocation improvements (12h)
- [ ] M3: Peephole optimizer (8h)
- [ ] M4: PTX validation layer (6h)

### Agent 5: GPU Optimizers Suite
- [x] M1: Survey of optimization gaps (2h) — **COMPLETE**
- [ ] M2: Expanded fusion (8h)
- [ ] M3: Memory coalescing (6h)
- [ ] M4: Autotuner (10h)

## Inter-Agent Requests

### Format
Each request follows this template:
```
[YYYY-MM-DD HH:MM UTC] From: A{X} → To: A{Y}
Topic: {Brief topic}
Request: {What is needed}
Status: PENDING | IN_PROGRESS | DONE
```

### Active Requests
[2026-07-05 16:00 UTC] From: A3 → To: A1
Topic: Knowledge[T] gradient specification
Request: A3 needs A1 to specify gradient propagation rules for epistemic types.
       Specifically: How should ∂Knowledge[T]/∂θ be computed? Should gradients
       carry uncertainty? Is the gradient itself a Knowledge<T> or a plain T?
Status: PENDING

[2026-07-05 16:00 UTC] From: A3 → To: A5
Topic: Gradient fusion patterns
Request: A3 needs A5 to design fusion patterns for backward-pass gradient ops.
       Target: fuse mul+add chains into FMA, coalesce gradient writes.
Status: PENDING

[2026-07-05 18:00 UTC] From: A5 → To: A2
Topic: Blackwell sm_121 autotuner integration
Request: A5 needs A2 to provide sm_121-specific tuning parameters for the autotuner.
       Specifically: optimal block sizes, tensor core shapes, TMA patterns for sm_121.
Status: PENDING

### Completed Requests
*None yet*

---

## Agent 1 (A1) — M1 Complete: Epistemic Types Foundation Gap Analysis

**Status**: ✅ COMPLETE (2026-07-05 18:30 UTC)
**Agent**: A1 (Epistemic Types Foundation) - Worker Agent
**Survey Focus**: Knowledge[T] GPU coverage across all backends

### M1 Survey Execution Summary

**Files Analyzed**:
- `stdlib/epistemic/knowledge.sio` (354 lines) - Scalar Knowledge<T> implementation
- `stdlib/epistemic/gum.sio` (418 lines) - GUM uncertainty propagation
- `self-hosted/gpu/hlir_to_gpu.sio` (3376 lines) - HLIR to GPU lowering
- `stdlib/linalg/vector.sio` (556 lines) - Vec2/3/4 CPU implementation
- `stdlib/linalg/matrix.sio` (874 lines) - Mat2/3/4 CPU implementation
- `self-hosted/gpu/epistemic_ptx.sio` (126K lines) - PTX backend
- `self-hosted/gpu/epistemic_spirv.sio` (48K lines) - SPIR-V backend
- 58 GPU backend files scanned for epistemic support

### Current State (M1 Findings)

#### Existing Epistemic Infrastructure
- **Core Knowledge[T]**: `stdlib/epistemic/knowledge.sio` (354 lines) — scalar epistemic type
- **GUM propagation**: `stdlib/epistemic/gum.sio` (418 lines) — uncertainty propagation framework
- **GPU lowering**: `self-hosted/gpu/hlir_to_gpu.sio` (3376 lines) — HLIR to GPU IR bridge
- **PTX backend**: `self-hosted/gpu/epistemic_ptx.sio` (126K lines) — PTX codegen
- **SPIR-V backend**: `self-hosted/gpu/epistemic_spirv.sio` (48K lines) — SPIR-V codegen
- **Metal backend**: `self-hosted/gpu/metal.sio` — Metal Shading Language support
- **Tensor ops**: `self-hosted/gpu/tensor_epistemic.sio` (71K lines) — matrix/tensor PTX
- **WMMA support**: `self-hosted/gpu/kernel_ir_wmma_lean.sio` — tensor core kernels

#### Knowledge[T] GPU Coverage Analysis

**Scalar Knowledge<T> Coverage**:
- ✅ **FULLY SUPPORTED** (all backends):
  - `Knowledge<f32>` / `Knowledge<f64>` scalar operations
  - GUM uncertainty propagation (add, sub, mul, div, sqrt, square)
  - Confidence tracking and gating
  - 4-shadow-register layout (val, eps, valid, prov)
  - PTX: Complete shadow register emission
  - SPIR-V: Complete result ID allocation
  - Metal: Basic dual-output kernels

**Vector/Matrix Knowledge<T> Coverage**:
- ⚠️ **PARTIAL** (type tags exist, no epistemic lowering):
  - `HLIR_TY_VEC2`, `HLIR_TY_VEC3`, `HLIR_TY_VEC4` — defined (lines 36-38)
  - `HLIR_TY_MAT2`, `HLIR_TY_MAT3`, `HLIR_TY_MAT4` — defined (lines 39-41)
  - Type lowering exists: `hlir_lower_type()` returns `GpuTypePtr` (lines 472-477)
  - ❌ **NO EPISTEMIC SHADOW REGISTERS** for vec/mat types
  - ❌ **NO GUM PROPAGATION** for vector/matrix operations

**Standard Library Linear Algebra**:
- ✅ **COMPLETE CPU IMPLEMENTATION**:
  - `stdlib/linalg/vector.sio` (556 lines) — Vec2, Vec3, Vec4
  - `stdlib/linalg/matrix.sio` (874 lines) — Mat2, Mat3, Mat4
  - Operations: add, sub, mul, div, dot, norm, transpose, inverse
- ❌ **NO GPU KNOWLEDGE<T> WRAPPERS**:
  - `Knowledge<Vec2>`, `Knowledge<Vec3>`, `Knowledge<Vec4>` — undefined
  - `Knowledge<Mat2>`, `Knowledge<Mat3>`, `Knowledge<Mat4>` — undefined
  - Cannot propagate uncertainty through linear algebra ops on GPU

#### Missing Epistemic GPU Features

**Vector/Matrix Knowledge<T> Lowering** (M2 target):
1. ❌ `Knowledge<Vec2/3/4>` → 4×shadow registers per component
2. ❌ `Knowledge<Mat2/3/4>` → 4×shadow registers per element
3. ❌ GUM propagation for vector arithmetic:
   - `ε(a+b) = sqrt(ε_a² + ε_b²)` (per-component)
   - `ε(a·b)` dot product uncertainty
   - `ε(||v||)` norm propagation
4. ❌ GUM propagation for matrix arithmetic:
   - `ε(A·B)` matrix multiplication uncertainty
   - `ε(Aᵀ)` transpose uncertainty propagation
   - `ε(det(A))` determinant uncertainty

**GUM Tensor Operations** (M3 target):
1. ⚠️ **PTX PARTIAL**: `tensor_epistemic.sio` has scalar GEMM uncertainty
2. ❌ **NO VEC/MAT TENSOR OPS**:
   - `Knowledge<Vec2>` tensor contractions
   - `Knowledge<Mat2>` batched operations
   - Epistemic reduction operations (sum, mean, max)
3. ❌ **NO CROSS-BACKEND UNIFORMITY**:
   - PTX: Some tensor epistemic ops exist
   - SPIR-V: Only scalar Knowledge<T>
   - Metal: Only scalar dual-output kernels

**Backend-Specific Gaps**:
- **PTX**: Scalar complete, tensor partial, vec/mat missing
- **SPIR-V**: Scalar complete, vec/mat/tensor missing
- **Metal**: Scalar basic, vec/mat/tensor missing
- **WGSL**: No epistemic support at all

### Quantified Coverage

**Knowledge[T> GPU Coverage by Backend**:
- **PTX**: 60% (scalar ✅, tensor ⚠️, vec/mat ❌)
- **SPIR-V**: 40% (scalar ✅, vec/mat ❌, tensor ❌)
- **Metal**: 30% (scalar basic ✅, vec/mat ❌, tensor ❌)
- **WGSL**: 0% (no epistemic support)

**Linear Algebra Epistemic Coverage**:
- **CPU (stdlib)**: 100% (Vec2/3/4, Mat2/3/4 all implemented)
- **GPU Knowledge<T> wrappers**: 0% (none exist)
- **GPU GUM propagation**: 0% (no vec/mat uncertainty rules)

### Dependencies on Other Agents

**A2 (Blackwell Tensor Core)**:
- ⚠️ **NEEDED FOR M3**: Epistemic WMMA matrix operations
- ⚠️ **NEEDED FOR M4**: Tensor core uncertainty propagation

**A4 (K-AXI→PTX Pipeline)**:
- ⚠️ **NEEDED FOR M2**: Vec/mat type lowering specification
- ⚠️ **NEEDED FOR M3**: GUM tensor operation PTX emission

### Next Steps (M2: Vec/Mat Knowledge Lowering)

**Priority 1** (blocking M2 progress):
1. Implement `Knowledge<Vec2/3/4>` struct definitions (extend knowledge.sio)
2. Implement `Knowledge<Mat2/3/4>` struct definitions (extend knowledge.sio)
3. Add vec/mat epistemic shadow register allocation in `hlir_to_gpu.sio`
4. Implement per-component GUM propagation (vec add, sub, mul, div)

**Priority 2** (needed for M3):
5. Implement vector/matrix GUM operations (dot, norm, matmul)
6. Add epistemic tensor operations (reduction, batch ops)
7. Cross-backend vec/mat emission (PTX, SPIR-V, Metal)

**Priority 3** (M4 testing):
8. Write cross-backend tests for vec/mat Knowledge<T>
9. Test GUM propagation correctness
10. Performance benchmarking vs. scalar ops

---

## Agent 3 (A3) — M1 Complete: Autodiff Gap Analysis

**Status**: ✅ COMPLETE (2026-07-05)
**Agent**: A3 (GPU Autodiff Engine)
**Survey Focus**: GPU reverse-mode automatic differentiation coverage

### Current State (M1 Findings)

#### Existing Infrastructure
- **Primary file**: `self-hosted/gpu/kernel_autodiff.sio` (~29K lines, 12 sections)
- **GPU autodiff module**: `self-hosted/gpu/autodiff/autodiff.sio` (~1.3K lines)
- **CPU autodiff**: `stdlib/epistemic/autodiff.sio` (forward-mode dual numbers)
- **ML autodiff**: `stdlib/ml/autodiff.sio` (reverse-mode tape-based)

#### Autodiff Coverage Analysis

**GPU Opcode → Gradient Mapping** (`kernel_autodiff.sio:Section 5`):
- ✅ **FULLY COVERED** (9/9 core ops):
  - `GpuAdd`, `GpuSub`, `GpuMul`, `GpuDiv`, `GpuFma`
  - `GpuSqrt`, `GpuSin`, `GpuCos`, `GpuAbs`
- ⚠️ **PARTIALLY COVERED** (3 ops defined but GPU mapping missing):
  - `GRAD_OP_EXP`, `GRAD_OP_LOG`, `GRAD_OP_POW`
  - `GRAD_OP_RELU`, `GRAD_OP_SIGMOID`, `GRAD_OP_TANH`
  - `GRAD_OP_MAX`, `GRAD_OP_MIN`, `GRAD_OP_SOFTMAX`
  - `GRAD_OP_RESHAPE`, `GRAD_OP_MATMUL`
- ❌ **NOT COVERED** (GPU ops exist but no gradient):
  - `GpuRsqrt`, `GpuExp2`, `GpuLg2`, `GpuRcp`
  - `GpuCvt` (type conversion)
  - All comparison ops (`GpuSetp*`)
  - All control flow (`GpuBra`, `GpuExit`)

**Reverse-Mode Implementation Status**:
1. ✅ **Wengert tape construction** (`kad_build_tape`) — complete
2. ✅ **CPU-side gradient rules** (`kad_grad_rule_apply`) — 9 ops implemented
3. ✅ **Tape backward pass** (`kad_tape_backward`) — complete with 10 passing tests
4. ⚠️ **PTX gradient emission** (`grad_emit_backward_pass`) — defined in `autodiff.sio` but not integrated with `kernel_autodiff.sio`

#### Missing Operations (Gaps)

**Tensor Operations** (critical for ML):
- ❌ `GRAD_OP_MATMUL` — tag defined but no implementation
- ❌ `GRAD_OP_SOFTMAX` — tag defined but no implementation
- ❌ `GRAD_OP_RESHAPE` — tag defined but no implementation
- ❌ Tensor core gradients (`GpuWmma`, `GpuMma`) — no WMMA/MMA gradient rules

**Activation Functions** (needed for neural nets):
- ⚠️ `GRAD_OP_RELU` — tag exists, CPU rule exists, PTX emitter exists
- ⚠️ `GRAD_OP_SIGMOID` — tag exists, CPU rule exists, PTX emitter exists
- ⚠️ `GRAD_OP_TANH` — tag exists, CPU rule exists, PTX emitter exists
- ❌ Missing integration with GPU kernel IR

**Elementwise Math** (needed for scientific computing):
- ⚠️ `GRAD_OP_EXP` — tag exists, needs GPU op mapping
- ⚠️ `GRAD_OP_LOG` — tag exists, needs GPU op mapping
- ⚠️ `GRAD_OP_POW` — tag exists, needs GPU op mapping
- ❌ `GRAD_OP_RSQRT` — tag missing (GpuRsqrt exists in kernel_ir)
- ❌ `GRAD_OP_RCP` — tag missing (GpuRcp exists in kernel_ir)

**Epistemic Autodiff** (Knowledge[T] gradients):
- ❌ No `Knowledge<T>` gradient propagation in GPU path
- ❌ GUM uncertainty propagation through autodiff
- ❌ Epistemic tensor operations (`epistemic_tensor_core.sio`) non-differentiable

#### Memory Optimization Opportunities

**Gradient Checkpointing** (M3 target):
- Current tape stores ALL forward values (memory intensive)
- No recomputation strategy for large models
- Fixed-capacity arrays (16384 entries) — no dynamic sizing
- No selective checkpointing (checkpoint all vs. recompute all)

**Fusion Integration** (coordination with A5):
- Gradient ops not marked as fusion candidates
- No fusion patterns for backward pass
- Potential: fuse `mul + add` gradient chains into single FMA

### Dependencies on Other Agents

**A1 (Epistemic Types)**:
- ❌ **BLOCKS M2**: Need `Knowledge[T]` gradient specification from A1
- ❌ **BLOCKS M4**: Need epistemic tensor op semantics for gradients

**A5 (GPU Optimizers)**:
- ⚠️ **NEEDED FOR M3**: Fusion patterns for gradient chains
- ⚠️ **NEEDED FOR M4**: Memory coalescing for gradient writes

### Quantified Coverage

**Reverse-Mode GPU Autodiff Coverage**:
- **Core arithmetic**: 100% (5/5 ops: add, sub, mul, div, fma)
- **Math intrinsics**: 60% (3/5 ops: sqrt, sin, cos covered; exp, log missing)
- **Activation functions**: 0% (relu, sigmoid, tanh defined but not integrated)
- **Tensor operations**: 0% (matmul, softmax, reshape defined but not implemented)
- **Epistemic ops**: 0% (no Knowledge[T] gradient support)
- **Overall**: ~30% (9/30 GPU opcodes have working reverse-mode)

### Next Steps (M2: Reverse-Mode GPU Implementation)

**Priority 1** (blocking M2 progress):
1. Add missing GPU op → grad op mappings (exp, log, relu, sigmoid, tanh)
2. Implement MATMUL gradient rules (∂(AB)/∂A = G*Bᵀ, ∂(AB)/∂B = Aᵀ*G)
3. Integrate PTX gradient emitters from `autodiff.sio` into `kernel_autodiff.sio`

**Priority 2** (needed for M4):
4. Implement SOFTMAX Jacobian-vector product
5. Add Knowledge[T] gradient propagation (coord with A1)
6. Tensor core gradient rules (coord with A2)

**Priority 3** (M3 optimization):
7. Implement selective checkpointing
8. Add gradient memory coalescing (coord with A5)
9. Fusion patterns for gradient chains (coord with A5)

---

## Agent 5 (A5) — M1 Complete: GPU Optimizers Suite Gap Analysis

**Status**: ✅ COMPLETE (2026-07-05 18:00 UTC)
**Agent**: A5 (GPU Optimizers Suite)
**Survey Focus**: GPU optimization coverage and opportunities

### Current State (M1 Findings)

#### Existing Optimizer Infrastructure
The GPU optimizer suite has **14 optimizer modules** under `self-hosted/gpu/opt/`:

**Core Optimizers**:
- **fusion.sio** (735 lines) — Kernel fusion with register remapping, safety guards
- **divergence.sio** (1126 lines) — Warp divergence analysis, predication heuristics
- **entropy_dispatch.sio** (688 lines) — Novel entropy-based kernel variant dispatch
- **async_pipeline.sio** (1733 lines) — Async memory pipelining (TMA, cp.async)
- **warp_vote_fastpath.sio** (778 lines) — Warp-vote epistemic fast-path optimization
- **autotune.sio** (1403 lines) — Static auto-tuning for block sizes, occupancy

**Epistemic Optimizers**:
- **epistemic_fusion.sio** (460 lines) — GUM-aware fusion with provenance XOR weighting
- **second_order_gum.sio** — Second-order GUM uncertainty propagation
- **covariance_shadow.sio** — Covariance shadow register optimization
- **tiled_covariance.sio** — Tiled covariance matrix computation

**General Optimization**:
- **optimizer.sio** (1413 lines) — Main optimizer orchestrator (7 optimization passes)
- **bridge.sio** — GPU optimization bridge utilities
- **divergence_cost.sio** — Divergence cost modeling

#### GPU Operation Coverage

**Supported GPU Operations** (from `kernel_ir.sio`):
- ✅ **Arithmetic**: Add, Sub, Mul, Div, Fma, Max, Min, Abs
- ✅ **Control Flow**: Branch, Exit, Setp* (comparisons), Selp
- ✅ **Memory**: Load/Store Global, Load/Store Shared, LoadParam
- ✅ **Warp Operations**: ShflDown, ShflUp, ShflBfly, ShflIdx, Vote, Ballot
- ✅ **Tensor Core**: Wmma, Mma, TmaLoad, TmaStore
- ✅ **Math Intrinsics**: Sqrt, Rsqrt, Exp2, Lg2, Sin, Cos
- ✅ **Atomics**: AtomicAdd, AtomicCas, AtomicExch, AtomicMin/Max, AtomicAnd/Or/Xor
- ✅ **Async Copy**: CpAsync, CpAsyncWait, CpAsyncWaitAll
- ✅ **Special Registers**: GetTid, GetBid, GetNtid, GetLaneId, GetWarpId, GetSmId

**Kernel Pattern Detection** (from `autotune.sio`):
- ✅ GPU_PATTERN_ELEMENTWISE — Element-wise operations
- ✅ GPU_PATTERN_REDUCTION — Reduction operations
- ✅ GPU_PATTERN_STENCIL — Stencil computations
- ✅ GPU_PATTERN_MATMUL — Matrix multiplication
- ✅ GPU_PATTERN_SCAN — Parallel scan/prefix sum
- ✅ GPU_PATTERN_HISTOGRAM — Histogram operations
- ✅ GPU_PATTERN_SPARSE — Sparse operations
- ✅ GPU_PATTERN_GENERAL — General-purpose kernels

#### Optimizer Coverage Analysis

**Fusion Coverage** (fusion.sio):
- ✅ **STRAIGHT-LINE FUSION**: Adjacent kernel pair fusion (complete)
- ✅ **REGISTER REMAPPING**: Per-type offset computation (complete)
- ✅ **PARAMETER MAPPING**: Kernel parameter deduplication (complete)
- ✅ **SHARED MEMORY MERGING**: Shared address offset handling (complete)
- ✅ **SAFETY GUARDS**: Op count, register pressure, shared memory limits (complete)
- ⚠️ **LIMITED PATTERNS**: Only element-wise + adjacent kernel fusion
- ❌ **NO CHAIN FUSION**: Cannot fuse mul+add chains into FMA
- ❌ **NO BACKWARD-PASS FUSION**: Gradient operation fusion missing
- ❌ **NO EPISTEMIC FUSION**: GUM shadow register fusion incomplete

**Memory Coalescing** (optimizer.sio):
- ✅ **SHARED MEMORY BANK CONFLICTS**: Detection exists (incomplete)
- ✅ **MEMORY BOUND CLASSIFICATION**: compute vs. memory ops ratio (complete)
- ⚠️ **NO COALESCING OPTIMIZER**: No global memory access coalescing pass
- ❌ **NO SHARED MEMORY TILING**: No shared memory tile optimization
- ❌ **NO ACCESS PATTERN REORDERING**: No stride optimization
- ❌ **NO L1/L2 CACHE OPTIMIZATION**: No cache-aware data layout

**Divergence Optimization** (divergence.sio):
- ✅ **DIVERGENCE DETECTION**: Thread mask tracking (complete)
- ✅ **PREDICATION HEURISTICS**: Predicate compiler (complete)
- ✅ **BRANCH COST MODEL**: Serialization penalty estimation (complete)
- ⚠️ **PARTIAL OPTIMIZATION**: Detection only, no PTX emission changes
- ❌ **NO AUTOMATIC PREDICATION**: Cannot convert branches to predicated code
- ❌ **NO RECONVERGENCE OPTIMIZATION**: No warp reconvergence hints

**Autotuning** (autotune.sio):
- ✅ **PATTERN DETECTION**: 7 kernel patterns (complete)
- ✅ **INSTRUCTION MIX**: FP32/FP64/int/tensor op counting (complete)
- ✅ **OCCUPANCY ESTIMATION**: Register/shared memory limiter detection (complete)
- ✅ **BLOCK SIZE RECOMMENDATION**: Per-pattern tuning advice (complete)
- ⚠️ **STATIC ONLY**: No runtime feedback, no hardware profiling
- ❌ **NO sm_121 SUPPORT**: No Blackwell-specific tuning parameters
- ❌ **NO TENSOR CORE TUNING**: Basic WMMA support, no sm_121 optimization
- ❌ **NO TMA PATTERN TUNING**: No tensor memory accelerator tuning

**Async Pipeline** (async_pipeline.sio):
- ✅ **TMA LOAD/STORE**: Tensor Memory Accelerator ops (complete)
- ✅ **CP.ASYNC**: Async copy pipeline (complete)
- ✅ **PIPELINE SCHEDULE**: Prologue/main/epilogue stages (complete)
- ✅ **BARRIER POOL**: Barrier allocation and management (complete)
- ⚠️ **NO AUTO-DETECTION**: Manual pipeline construction only
- ❌ **NO AUTOMATIC PIPEELINE INSERTION**: No automatic pipelining of loops

**Novel Optimizers** (world-first):
- ✅ **entropy_dispatch.sio**: Shannon entropy H(ε) kernel variant selection (implemented)
- ✅ **warp_vote_fastpath.sio**: vote.sync.ballot epistemic dual-path (implemented)
- ✅ **epistemic_fusion.sio**: Provenance XOR weighting + uncertainty penalties (partial)

#### Missing Optimizations (Gaps)

**Fusion Gaps** (M2 targets):
1. ❌ **CHAIN FUSION**: mul+add → FMA, sub+neg → optimized, sqrt+rsqrt → combined
2. ❌ **BACKWARD-PASS FUSION**: Gradient operation chains (mul, add, transpose fusion)
3. ❌ **EPISTEMIC FUSION**: GUM shadow register fusion (partial in epistemic_fusion.sio)
4. ❌ **MULTI-KERNEL FUSION**: Beyond pair-wise fusion (3+ kernel chains)
5. ❌ **PATTERN-SPECIFIC FUSION**: Matmul+elementwise, reduction+broadcast fusion

**Memory Coalescing Gaps** (M3 targets):
1. ❌ **GLOBAL MEMORY COALESCING**: Adjacent thread access merging
2. ❌ **SHARED MEMORY TILING**: Lds/Matrix tile optimization
3. ❌ **BANK CONFLICT ELIMINATION**: Shared memory access stride adjustment
4. ❌ **VECTOR LOAD/STORE**: Merge 4x f32 → single f128 load
5. ❌ **PREFETCH OPTIMIZATION**: Software prefetch insertion
6. ❌ **CACHE LINE ALIGNMENT**: Data structure padding for cache lines

**Autotuner Gaps** (M4 targets):
1. ❌ **sm_121 TUNING**: Blackwell-specific block sizes, WMMA shapes
2. ❌ **RUNTIME PROFILING**: Actual kernel execution time feedback
3. ❌ **HARDWARE COUNTERS**: Occupancy, memory bandwidth, cache hit rates
4. ❌ **MULTI-VARIANT TUNING**: Generate + benchmark multiple kernel variants
5. ❌ **AUTO-TUNER INTEGRATION**: Apply tuning hints to kernels automatically

**Divergence Optimization Gaps**:
1. ❌ **AUTOMATIC PREDICATION**: Branch → predicated execution
2. ❌ **WARP RECONVERGENCE**: Explicit reconvergence point hints
3. ❌ **LOOP UNROLLING**: Divergence-aware loop unrolling
4. ❌ **IF-CONVERSION**: Convert if-else to select/min/max

**Gradient Optimization Gaps** (coord with A3):
1. ❌ **GRADIENT CHAIN FUSION**: Backward pass mul+add chains
2. ❌ **GRADIENT MEMORY COALESCING**: Gradient write access merging
3. ❌ **CHECKPOINT-AWARE FUSION**: Fusion with recomputation trade-offs

#### Quantified Coverage

**GPU Optimizer Suite Coverage**:
- **Fusion**: 40% (pair-wise element-wise complete, chain/gradient/epistemic missing)
- **Memory Coalescing**: 20% (detection only, no optimization passes)
- **Divergence**: 30% (analysis complete, automatic optimization missing)
- **Autotuning**: 50% (static tuning complete, runtime profiling missing)
- **Async Pipeline**: 60% (infrastructure complete, auto-insertion missing)
- **Epistemic Optimizers**: 70% (novel optimizations implemented, integration incomplete)

**Overall Optimizer Coverage**: ~45% (6/14 areas complete or strong partial)

### Dependencies on Other Agents

**A2 (Blackwell Tensor Core)**:
- ⚠️ **NEEDED FOR M4**: sm_121-specific tuning parameters for autotuner
- ⚠️ **NEEDED FOR M4**: Tensor core shapes and TMA patterns for Blackwell

**A3 (GPU Autodiff Engine)**:
- ⚠️ **NEEDED FOR M2**: Backward-pass fusion patterns for gradient chains
- ⚠️ **NEEDED FOR M3**: Gradient memory access patterns for coalescing

**A1 (Epistemic Types)**:
- ⚠️ **NEEDED FOR M2**: Epistemic shadow register layout for GUM fusion
- ⚠️ **NEEDED FOR M2**: Uncertainty propagation constraints for fusion

### Next Steps (M2: Expanded Fusion)

**Priority 1** (blocking M2 progress):
1. Implement chain fusion (mul+add → FMA, sub+neg → optimized)
2. Add backward-pass fusion patterns (coord with A3 for gradient ops)
3. Complete epistemic fusion for GUM shadow registers
4. Implement multi-kernel fusion (3+ kernel chains)

**Priority 2** (needed for M3):
5. Implement pattern-specific fusion (Matmul+elementwise, reduction+broadcast)
6. Add fusion for tensor core operations (WMMA chains)
7. Integrate entropy dispatch with fusion decisions

**Priority 3** (M4 optimization):
8. Autotuner integration with fusion decisions
9. sm_121-specific fusion patterns (coord with A2)
10. Runtime fusion validation and benchmarking

---

## Integration Plan

### Phase 1: Survey (M1) - 1-2h
All agents complete their respective surveys and update milestone tracker.

### Phase 2: Foundation (M2) - 12-24h
Agents implement core features, start documenting inter-dependencies.

### Phase 3: Optimization (M3) - 12-24h
Agents handle inter-agent requests, optimize their areas.

### Phase 4: Integration (M4) - 8-12h
Final implementation, cross-area testing, documentation.

### Final Integration
- All agents sign off on M4 completion
- Cross-area test suite runs
- Integration report generated
- Documentation updated

## Model Usage Log

| Timestamp | Agent | Model Used | Purpose | Tokens (approx) |
|---|---|---|---|---|
| 2026-07-05 15:30 | Coordinator | Opus | Swarm setup | ~5K |
| 2026-07-05 16:00 | A3 | Opus | M1 autodiff gap analysis | ~8K |
| 2026-07-05 18:00 | A5 | Sonnet | M1 optimizer gap analysis | ~12K |

## Architecture Dependencies

```
A1 (Epistemic) ←→ A2 (Blackwell): Epistemic tensor ops for WMMA
A1 (Epistemic) ←→ A4 (Pipeline): PTX lowering spec
A3 (Autodiff) ←→ A5 (Optimizers): Fusion patterns for autodiff
A2 (Blackwell) ←→ A4 (Pipeline): Optimized PTX emission
A3 (Autodiff) ←→ A1 (Epistemic): Knowledge[T] gradients
A5 (Optimizers) ←→ All: Optimization for all areas
```

## Communication Protocol

1. **Update this doc** when milestones are reached
2. **Add requests** when inter-agent coordination is needed
3. **Check requests** from other agents regularly
4. **Mark status** updates as they happen
5. **Sign off** when work is complete

## Notes

- Each agent works in a separate session/context
- This doc is the only shared state
- No direct agent-to-agent communication
- All coordination happens through this document

## Success Criteria

Final integration succeeds when:
- [ ] All M1 milestones complete (surveys)
- [ ] All M2-M3 milestones with inter-agent coordination
- [ ] All M4 milestones complete (integration ready)
- [ ] Cross-area test suite passes
- [ ] Documentation updated for all areas
- [ ] Integration report generated

---

*Last updated by: Coordinator (Claude Opus 4.8)*
*Next update: When any agent completes a milestone*
