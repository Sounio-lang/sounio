# Changelog

All notable changes to the Sounio compiler will be documented in this file.

## [1.0.0] - 2026-01-29

### Highlights

**Sounio v1.0** is the first production-ready release. This release completes
all core language features and provides comprehensive documentation.

### Added

#### Async Runtime Infrastructure

Complete async/await runtime with task scheduling, combinators, and channels.

- **TaskScheduler**: Priority-based task execution with dependency tracking
  - Ready queue with FIFO scheduling
  - Suspension tracking (Await, JoinAll, SelectAny, Channel waits)
  - O(1) wake operations via waiter maps
  - Statistics tracking (polls, wakes, task states)

- **Join Combinator**: Wait for ALL tasks to complete
  - `JoinFuture::new(tasks)` - Create from TaskId vector
  - `join2(a, b)`, `join3(a, b, c)` - Convenience functions
  - Results returned as array in order

- **Select Combinator**: Wait for ANY task to complete (race)
  - `SelectFuture::new(tasks)` - Create from TaskId vector
  - `select2(a, b)`, `select3(a, b, c)` - Convenience functions
  - Returns `SelectResult { index, task_id, value }`

- **Channels**: Bounded and unbounded communication
  - `channel<T>()` - Unbounded channel
  - `bounded_channel<T>(cap)` - Bounded with capacity
  - `Sender<T>` / `Receiver<T>` - Send/receive handles
  - Clone for multi-producer/multi-consumer patterns
  - Error types: `Closed`, `Full`, `Empty`

- **37 new tests** for async runtime coverage

#### LLVM Codegen Enhancements

- **Double-precision SIMD vectors**: `Vec2d`, `Vec3d`, `Vec4d`
  - `Vec2d`: `<2 x double>` (128 bits, 16-byte aligned)
  - `Vec3d`: `<4 x double>` padded (256 bits, 32-byte aligned)
  - `Vec4d`: `<4 x double>` (256 bits, 32-byte aligned)

- **Octonion type**: 8-dimensional hypercomplex numbers
  - `<8 x float>` (256 bits, 32-byte aligned)

- **QNN types**: Quaternionic Neural Network layer types
  - `QuatLinear` - Quaternion linear layer
  - `QuatConv2d` - Quaternion 2D convolution
  - `QuatRnnState` - Quaternion RNN hidden state
  - `QuatGate` - Quaternion gate (LSTM/GRU)
  - All represented as opaque pointers

- **CString constant handling**: Global string pointers for C interop

- **Epistemic type support in DWARF**: Debug info for `Knowledge<T>` with mode-aware sizing

#### GPU Runtime Bridge

- **GpuRuntimeBridge**: Singleton for kernel launch dispatch
  - Thread-safe with `OnceLock<Mutex<...>>`
  - Kernel and buffer registries
  - Integration with `handler_stack.rs` dispatch functions

#### SMT Refinement Infrastructure

- **refine_assert.rs pass**: SIR pass for Z3 integration
- **Z3 solver integration**: Predicate verification at compile-time
- **Runtime fallback**: Assertions when Z3 unavailable

#### Package Manager HTTP Backend

- **HttpRegistry**: REST API for package registry
  - `get_versions`, `get_manifest`, `download`, `publish`, `yank`
  - Git dependency support with branch/tag/rev specifiers

### Documentation

- **INSTALLATION.md**: Comprehensive setup guide with all dependencies
- **ASYNC_RUNTIME.md**: Task scheduler, combinators, channels documentation
- **LLVM_CODEGEN.md**: Type mapping and backend documentation
- **FEATURE_FLAGS.md**: Complete build configuration reference
- **GPU_RUNTIME.md**: GPU kernel launch documentation

### Changed

- LLVM 17 is now the recommended version (15 and 16 still supported)
- `libzstd-dev` is required for LLVM builds

### Fixed

- Borrow conflicts in LLVM codegen value conversion
- Missing HLIR type handlers for SIMD and QNN types
- Scope resolution for ontology domain types
- Send trait bounds for LSP handlers

### Performance

- 3851 tests passing
- Zero compiler warnings

## [0.96.0] - 2025-12-28

### Added

#### A/B Register Allocation Policy Comparison (1,333 lines)

Infrastructure for comparing Classic vs Attention-based register allocation
strategies with epistemic awareness.

- **AllocPolicy enum**: Classic, Attention, AttentionCalibrated variants
- **AttentionConfig**: 6 tunable weights for epistemic-aware scoring
  - `w_use_density`: 1.7088 (traditional liveness)
  - `w_crosses_call`: 0.4527 (call boundary penalty)
  - `w_next_use_distance`: 0.8802 (near-future use priority)
  - `w_confidence`: 0.9374 (prioritize high-confidence values)
  - `w_uncertainty`: 0.2411 (de-prioritize uncertain values)
  - `w_provenance`: 0.2013 (preserve data lineage)
- **EpistemicMetadata**: Confidence, uncertainty, provenance tracking
- **AllocationMetrics**: Quality measurement (epistemic quality, score separation)
- **MetricsCollector**: Records spill events with reasons
- **ABComparisonResult**: Side-by-side policy evaluation

#### CLI Options for Allocation Policy

```
--alloc-policy <classic|attention|attention-calibrated>
--attention-config <path>       # Custom weight JSON file
--emit-alloc-metrics <path>     # Output metrics JSON
```

#### Bayesian Optimization Tooling

- **tune_attention_weights.py**: BO script using botorch/gpytorch
- **Random search fallback**: Works without torch dependencies
- **Output formats**: JSON config + Rust code snippet

### Performance

Benchmark improvements with calibrated weights:
- Propagation (1K): **38% faster**
- Epistemic reduction: **9% faster**
- Particle init (1K): **13% faster**
- Attention scoring throughput: **11% higher**

## [0.95.0] - 2025-12-28

### Added

#### SPIR-V Binary Emission (634 lines)

Complete SPIR-V binary encoding for GPU shader generation.

- **Opcodes Module**: 50+ SPIR-V opcode constants for instruction encoding
- **Instruction Encoding**: `SpirVInst::encode()` for all instruction variants
- **Type Conversion**: `to_spirv()` implementations for:
  - `ExecutionModeKind` - LocalSize, OriginUpperLeft, etc.
  - `Decoration` - BuiltIn, Location, Binding, Block, etc.
  - `FunctionControl` - inline, pure, const flags
  - `MemoryOperands` - volatile, aligned, nontemporal
- **Module Assembly**: `SpirVModule::assemble()` generates complete binary:
  - Header: magic 0x07230203, version 1.5, generator SOUN
  - All 12 SPIR-V sections in proper order
- **File Output**: `SpirVModule::write_to_file()` for .spv files

#### GPU Test Harness

- **Dependencies**: wgpu 23, pollster 0.4, bytemuck 1.14
- **test_spirv_assembly**: Validates SPIR-V generation
  - Builds module with SpirVModuleBuilder
  - Verifies magic number, version, generator
  - Writes .spv file for external validation
- **test_gpu_execution**: Compute shader test
  - GPU detection (tested: NVIDIA RTX 4000 Ada)
  - Full compute pipeline with storage buffers
  - Graceful fallback for WSL2 environments

### Test Results

- SPIR-V binary: 28 words (112 bytes) generated correctly
- GPU detected via Vulkan/D3D12 backend
- All tests pass

## [0.94.0] - 2025-12-27

### Added

#### SIR (Sounio Intermediate Representation) - GPU Infrastructure (20,661 lines)

This release introduces the complete GPU backend infrastructure for Sounio, enabling
epistemic-aware high-performance computing on GPUs.

##### SKIR Cost Annotation System (`cost.rs` - 1,708 lines)
- **9 Universal Performance Primitives**: Declarative annotations that separate semantics from performance
  - `Tile` - Spatial decomposition (16x16x16 for Tensor Cores)
  - `Layout` - Memory organization (RowMajor, ColMajor, Blocked, Tiled, Swizzled)
  - `VectorWidth` - SIMD lane count with hardware presets (SSE, AVX, NEON)
  - `SharedCache` - Scratchpad memory allocation with alignment
  - `Unroll` - Loop transformation factors
  - `Fuse` - Operation fusion types (Pointwise, Reduction, Tiled)
  - `ReductionMode` - Determinism semantics (Deterministic, Fast, Kahan)
  - `Precision` - Numerical precision (FP64, FP32, FP16, BF16, Mixed, TensorCore)
  - `RngMode` - Randomness control for epistemic operations
- `CostEnvelope` - Aggregates all primitives for a kernel
- `KernelVariantSet` - Collection of variants for autotuning

##### GPU Capability System (`capabilities.rs` - 1,306 lines)
- `Capability` enum with 30+ GPU features
- `TargetCapabilities` struct for device-specific limits
- Target presets:
  - `vulkan_portable()` - Baseline for portability
  - `nvidia_ampere()`, `nvidia_hopper()`, `nvidia_blackwell()`
  - `amd_rdna3()`, `amd_cdna3()`
  - `intel_arc()`, `intel_pvc()`
  - `apple_m_series()`
- `CapabilityQuery` trait for backend capability checking

##### Variant Generation for Auto-Tuning (`variants.rs` - 1,809 lines)
- `VariantSpace` - Search space definition with preset configurations
- `VariantGenerator` - Three generation strategies:
  - Latin Hypercube Sampling (LHS)
  - Sobol quasi-random sequences
  - Random sampling
- `VariantRegistry` - Caches measured variants
- `PerformanceMeasurement` - Time, memory, occupancy metrics
- `VariantSelector` - Pareto-optimal selection with constraints

##### GPU Multi-Resource Attention Policy (`gpu_attention.rs` - 1,176 lines)
- `GpuResourceBudget` - Hardware constraints (registers, shared memory, warps)
- `GpuLiveInterval` - Extends CPU intervals with GPU-specific fields:
  - Divergence risk from confidence variance
  - Bank conflict risk
  - Memory coalescing score
- `GpuAttentionScore` - Multi-objective scoring:
  - Register pressure
  - Occupancy impact
  - Shared memory cost
  - Divergence cost
  - Epistemic weight
- Occupancy cost model: `occupancy_for_registers()`, `optimal_register_target()`
- Divergence cost model: `divergence_probability()`, `divergence_cost()`
- Pareto-based spill selection: `select_spill_victim()`, `select_spill_batch()`

##### SPIR-V Backend Foundation (`spirv/mod.rs` - 2,827 lines)
- Complete SPIR-V type system: `SpirVType` enum
- Instruction representation: `SpirVInst` with 50+ opcodes
- `TargetCapabilities` for Vulkan 1.0, 1.1, 1.2, 1.3
- NVIDIA Ampere and AMD RDNA2 presets
- Execution models: Vertex, Fragment, GLCompute, Kernel
- Storage classes: Uniform, Workgroup, Private, StorageBuffer, etc.
- Reduction strategies: SubgroupShuffle, SharedMemory, Atomic

##### Monte Carlo Epistemic Kernel (`kernels/monte_carlo.rs` - 1,328 lines)
- `ParticleState` - Position, velocity, weight, confidence, provenance
- `MonteCarloConfig` - Simulation parameters
- `PriorDistribution` - Initialization distribution
- Confidence partitioning: `partition_by_confidence()`, `partition_by_attention()`
- `EpistemicReduction` - Aggregates particles preserving uncertainty
- `MonteCarloKernel` - The "soul" of epistemic-aware parallel execution:
  - Confidence flows through computation
  - High-confidence particles get RK4, low-confidence get Euler
  - Divergence-avoiding patterns for GPU efficiency

##### Core SIR Infrastructure
- `emit.rs` (2,839 lines) - Attention-based x86-64 register allocation
- `passes.rs` (2,884 lines) - Optimization passes including GUM uncertainty propagation
- `lower.rs` (1,412 lines) - HLIR to SIR lowering
- `builder.rs` (647 lines) - Fluent API for SIR construction
- `blocks.rs` (386 lines) - Basic blocks and control flow
- `module.rs` (436 lines) - Module structure
- `ops.rs` (674 lines) - SIR operations
- `types.rs` (449 lines) - Type system
- `values.rs` (380 lines) - Values and constants
- `metadata.rs` (282 lines) - Epistemic metadata

### Tests

- 129 SIR module tests passing
- Comprehensive coverage for all new modules

### Design Philosophy

This release implements the "pure core + performance dialect" approach:
- **Semantics pure at the top**: High-level code remains clean and domain-focused
- **Performance at the bottom**: Cost annotations enable optimization without polluting semantics
- **IR in command**: The compiler selects strategies based on declared capabilities
- **Backends declare, don't decide**: GPU backends expose capabilities; compiler makes decisions

## [0.93.0] - Previous Release

See git history for previous changes.
