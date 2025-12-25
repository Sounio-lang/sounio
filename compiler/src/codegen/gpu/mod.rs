//! GPU Code Generation for Sounio
//!
//! Supports:
//! - PTX (NVIDIA CUDA)
//! - SPIR-V (Vulkan, OpenCL)
//! - MSL (Apple Metal)
//!
//! Architecture:
//! ```text
//! HLIR -> GpuIR -> PTX/SPIR-V -> Driver -> GPU Execution
//! ```
//!
//! # Epistemic GPU Computing
//!
//! Sounio is the first language to track epistemic state through GPU computation:
//! - Shadow registers for uncertainty (ε)
//! - Validity predicates
//! - Provenance tracking
//! - Tensor Core operations with uncertainty propagation
//!
//! # Usage
//!
//! ```ignore
//! use sounio::codegen::gpu::{hlir_to_gpu, PtxCodegen, GpuTarget};
//!
//! let gpu_module = hlir_to_gpu::lower(&hlir, GpuTarget::Cuda { compute_capability: (8, 0) });
//! let ptx = PtxCodegen::new((8, 0)).generate(&gpu_module);
//! ```

pub mod async_pipeline;
pub mod autotune;
pub mod bio;
pub mod calibration;
pub mod collective_ops;
pub mod cooperative;
pub mod costs;
pub mod counterfactual;
pub mod counterfactual_metal;
pub mod diagnostics;
pub mod divergence;
pub mod epistemic_ptx;
pub mod fusion;
pub mod graph;
pub mod hlir_to_gpu;
pub mod intrinsics;
pub mod ir;
pub mod metal;
pub mod metal_runtime;
pub mod multi_gpu;
pub mod numerical;
pub mod optimizer;
pub mod p2p_transfer;
pub mod portable;
pub mod profiler;
pub mod ptq;
pub mod ptx;
pub mod quantize;
pub mod roofline;
pub mod runtime;
pub mod sourcemap;
pub mod sparse;
#[cfg(feature = "gpu")]
pub mod spirv;
pub mod tensor_epistemic;
pub mod tile;
pub mod uplift_trees;
pub mod validation;

pub use intrinsics::{GpuIntrinsic, all_intrinsics, get_intrinsic, is_gpu_intrinsic};
pub use ir::{
    BlockId, CoopReduceOp, CooperativeGroupId, CooperativeScope, CudaArch, CudaFeatures, Fp8Format,
    GpuBlock, GpuConstValue, GpuConstant, GpuFunction, GpuKernel, GpuModule, GpuOp, GpuParam,
    GpuTarget, GpuTerminator, GpuType, MemorySpace, MetalGpuFamily, PartitionType, QuantizeMode,
    SharedMemDecl, TileLayout, TmaReduceOp, ValueId, WarpReduceOp, WarpVoteOp,
};
pub use ptx::PtxCodegen;
pub use runtime::{
    DeviceBuffer, GpuBackend, GpuError, GpuRuntime, Kernel, KernelArg, LaunchConfig,
};
#[cfg(feature = "gpu")]
pub use spirv::SpirvCodegen;
pub use tensor_epistemic::{
    EpistemicTensorCategory, EpistemicTensorIntrinsic, EpsilonPropagationRule, TensorCoreOp,
    all_epistemic_tensor_intrinsics, get_epistemic_intrinsic, is_epistemic_tensor_intrinsic,
};

// HLIR to GPU lowering - the critical bridge
pub use hlir_to_gpu::{
    LoweringConfig, OptimizedGpuModule, compile_to_ptx, compile_to_ptx_epistemic, lower,
    lower_and_optimize, lower_with_config,
};

// Epistemic PTX emission - shadow registers for uncertainty tracking
pub use epistemic_ptx::{
    EpistemicPtxConfig, EpistemicPtxEmitter, EpistemicShadowRegs, WarpEpsilonOp,
};

// Counterfactual GPU execution - Pearl's do-calculus as GPU primitives
pub use counterfactual::{
    CounterfactualContext, CounterfactualPtxConfig, CounterfactualPtxEmitter, CounterfactualValue,
    Intervention, StructuralEqType, WorldDivergence, WorldId, WorldSnapshot,
};

// Metal Shading Language (MSL) codegen - native Apple GPU support
pub use metal::{MetalCodegen, MetalCodegenConfig, compile_to_msl, compile_to_msl_epistemic};

// Counterfactual Metal execution - Pearl's do-calculus for Apple Silicon
pub use counterfactual_metal::{
    CounterfactualMetalConfig, CounterfactualMetalEmitter, compile_counterfactual_metal,
    generate_counterfactual_metal_library,
};

// Metal runtime - native Apple GPU execution
pub use metal_runtime::{
    EpistemicMetalRunner, MetalBuffer, MetalCommandBuffer, MetalDeviceInfo, MetalDispatchSize,
    MetalError, MetalKernel, MetalLibrary, MetalResourceOptions, MetalRuntime, MetalStorageMode,
};

// Bio/Quaternion GPU kernels - from "The Quaternionic Syntax of Existence"
pub use bio::{
    add_bio_kernels, gen_dna_complement_kernel, gen_gf4_add_kernel, gen_quaternion_mul_kernel,
    gen_quaternion_normalize_kernel, gen_quaternion_slerp_kernel, gen_transmission_compose_kernel,
};

// Cooperative Groups kernel generators (CUDA 9.0+ / PTX 6.0+)
pub use cooperative::{
    add_cooperative_kernels, gen_ballot_count_kernel, gen_block_reduce_kernel,
    gen_warp_broadcast_kernel, gen_warp_inclusive_scan_kernel, gen_warp_reduce_sum_kernel,
};

// CUDA Graphs with dynamic control flow
pub use graph::{
    BufferId, BufferInfo, BufferLocation, ConditionType, ConditionalNode, GpuGraph,
    GraphExecConfig, GraphKernelArg, GraphNode, GraphNodeId, GraphNodeType, KernelNode, LoopNode,
    MemcpyNode, MemsetNode, StreamId, build_graph_from_module,
};

// Cross-platform portable GPU IR (write-once, compile-anywhere)
pub use portable::{
    AvailableBackends, BackendCapabilities, Capability, CompileError, CompileResult,
    CompiledKernel, Dimension, PortableGpuOp, PortableMemorySpace, PortableType, UnifiedCompiler,
    UnifiedKernel, UnifiedParam, UnifiedSharedMem, compile_kernel, compile_to_all,
};

// Tile programming utilities (CUDA 13+)
pub use tile::{
    align_to, compute_swizzle_pattern, element_size_bytes, recommended_tile_size,
    shared_memory_bytes, supports_tiles, supports_tma, supports_wgmma, threads_per_tile,
    validate_tile_dims, validate_tile_dims_2d,
};

// Kernel fusion optimization (Phase 3)
pub use fusion::{
    ArchConstraints, CostWeights, DependencyType, FusionAnalysis, FusionCandidate,
    FusionCandidateId, FusionConfig, FusionCostModel, FusionError, FusionGroup, FusionGroupId,
    FusionPlan, FusionStats, FusionTransformer, FusionType, KernelDependencyGraph, KernelId,
    LaunchConfig as FusionLaunchConfig, ResourceEstimate, SharedMemLayout,
    analyze_and_fuse_kernels,
};

// Auto-tuning for kernel launch configuration (Phase 4)
pub use autotune::{
    ArchConstants, AutoTuneConfig, AutoTuner, BlockShape, InstructionMix, KernelAnalyzer,
    KernelPattern, KernelProfile, MemoryPattern, OccupancyCalculator, OccupancyInfo,
    OccupancyLimiter, TileConfig, TunedConfig, TuningStrategy, tune_module,
};

// Async memory pipeline (Phase 5)
pub use async_pipeline::{
    AsyncOp, AsyncOpGraph, AsyncOpId, AsyncOpKind, AsyncPipeline, Barrier, BarrierId, BarrierKind,
    BarrierPool, BarrierScheduler, BufferConfig, DependencyAnalyzer, PipelineBuilder,
    PipelineCodegen, PipelineSchedule, PipelineScheduler, PipelineStage, ScheduledOp, StageBuffer,
    StageId, apply_pipeline, create_pipeline_from_tile, schedule_pipeline,
};

// GPU Optimization Pipeline (Phase 8)
pub use optimizer::{
    GpuOptimizer, OptimizationReport, OptimizerConfig, OptimizerError, PassStats, optimize_module,
    optimize_module_aggressive,
};

// GPU Diagnostics & Validation (Phase 9)
pub use diagnostics::{
    DiagnosticConfig, DiagnosticContext, DiagnosticReport, DiagnosticSeverity, DiagnosticSummary,
    GpuDiagnostic, GpuDiagnosticKind, GpuIrLocation, HintConfidence, RecoveryGenerator,
    RecoveryHint,
};
pub use sourcemap::{GpuSourceMapper, LocationTrace, PtxDebugEmitter, PtxLocation, SpanTracker};
pub use validation::{
    BufferComparison, CorrectnessValidator, PrecisionStats, ToleranceConfig, ValidationConfig,
    ValidationError, ValidationIssue, ValidationResult,
};

// Warp Divergence Analysis & Control Flow Optimization
pub use divergence::{
    AdaptiveDispatcher, BranchCost, BranchCostEstimator, ControlFlowOpt, ControlFlowOptReport,
    ControlFlowOptimizer, DivergenceInfo, DivergenceKind, KernelDivergenceAnalysis,
    PredicateCompiler, PredicateMask, PredicateReg, ThreadMask, WarpDivergenceAnalyzer,
};

// Multi-GPU / Distributed Computing (Phase 10)
pub use collective_ops::{
    AlgorithmSelector, CollectiveAlgorithm, CollectiveManager, CollectiveOp, CollectiveStats,
    SimulatedBuffer,
};
pub use multi_gpu::{
    DeviceGroup, DeviceId, DeviceInfo, GpuTopology, InterconnectType, MultiGpuBarrier,
    MultiGpuConfig, MultiGpuError, MultiGpuEvent, MultiGpuRuntime, P2PCapability,
};
pub use p2p_transfer::{
    AsyncTransfer, P2PManager, P2PStats, ReduceOp, RingTransferStep, TransferChunk,
    TransferDescriptor, TransferDirection, generate_allgather_steps, generate_reduce_scatter_steps,
    split_into_chunks,
};

// Quantization Pipeline (Phase 11)
pub use calibration::{
    CalibrationCollector, CalibrationMethod, CalibrationStats, PerChannelCalibrator,
};
pub use ptq::{
    ActivationQuantConfig, LayerInfo, LayerQuantStatus, PtqConfig, PtqEngine, PtqErrorSummary,
    QuantizedModule, WeightQuantConfig,
};
pub use quantize::{
    PerChannelQuantParams, QuantDtype, QuantError, QuantErrorAnalyzer, QuantParams, QuantScheme,
    QuantizedTensor, pack_int4, quantize_tensor_int4, quantize_tensor_int8, unpack_int4,
};

// Performance Profiling & Roofline Analysis (Phase 12)
pub use costs::{
    ArchPeakPerf, CostDatabase, FlopsCount, InstructionClass, InstructionCost, KernelCostEstimate,
    LimitingResource, MemoryTraffic,
};
pub use profiler::{
    Bottleneck, BottleneckKind, BottleneckSeverity, KernelPerfProfile, KernelProfiler,
    ModulePerfProfile, PerfComparison, PerfCounters, PerfScore,
};
pub use roofline::{
    Boundedness, OptimizationHint, RooflineAnalysis, RooflineModel, RooflinePlot, RooflinePoint,
};

// Numerical Stability & Error Propagation (Phase 13)
pub use numerical::{
    AppliedMitigation, ErrorBound, ErrorEvent, ErrorPropagator, MitigationStrategy,
    MixedPrecisionStrategy, Precision, PrecisionAdvisor, PropagationMode, StabilityAnalyzer,
    StabilityIssue, StabilityMitigator, StabilityRisk, StabilitySummary, UlpError,
    error_to_epistemic_epsilon, risk_to_validity_confidence, synthesize_provenance,
};

// Sparse Tensor Compiler (Phase 14)
pub use sparse::{
    AnalyzerConfig, BlockMap, SparseConvKernel, SparseFormat, SparseFusionAnalyzer,
    SparseGemmKernel, SparseMVKernel, SparseOpInfo, SparseOpType, SparsePattern, SparseTensor,
    SparsityAnalyzer, SparsityCost, StructureInfo, add_sparse_kernels,
};

// GPU-Accelerated Uplift Trees with Epistemic Heads (CausalML Phase 15)
pub use uplift_trees::{
    CustomerSegment, NodeType, SplitCriterion, UpliftHistBin, UpliftTreeGpu, UpliftTreeGpuConfig,
    UpliftTreeGpuRuntime, UpliftTreeNode, UpliftTreePtxEmitter, compile_uplift_tree_ptx,
};
