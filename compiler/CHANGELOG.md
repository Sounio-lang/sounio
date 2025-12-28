# Changelog

All notable changes to the Sounio compiler will be documented in this file.

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
