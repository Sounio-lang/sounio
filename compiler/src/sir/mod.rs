//! Sounio Intermediate Representation (SIR)
//!
//! A domain-aware low-level IR designed specifically for Sounio's unique features.
//! SIR sits between HLIR (high-level SSA) and machine code emission.
//!
//! # Design Philosophy
//!
//! Unlike LLVM IR which is designed for C/C++ patterns, SIR preserves
//! domain-specific information that enables optimizations impossible
//! in general-purpose compilers:
//!
//! - **Epistemic awareness**: Confidence and provenance flow through the IR
//! - **Numerical semantics**: IEEE 754 guarantees, precision requirements
//! - **Probability primitives**: First-class distribution sampling
//! - **Scientific patterns**: ODE solver steps, compartment models
//!
//! # Architecture
//!
//! ```text
//! HLIR (Sounio types, high-level ops)
//!   ↓ lowering
//! SIR (machine types, domain metadata)
//!   ↓ optimization passes
//! SIR (optimized)
//!   ↓ code emission
//! Machine Code (x86-64, ARM64)
//! ```
//!
//! # Example
//!
//! ```text
//! ; HLIR input:
//! ;   let x: Epistemic<f64> = 0.5 ~ confidence(0.9)
//! ;   let y = x * 2.0
//!
//! ; SIR output:
//! define @main() -> !sir.void {
//! entry:
//!   %x.val = sir.const.f64 0.5
//!   %x.conf = sir.const.f64 0.9
//!   %two = sir.const.f64 2.0
//!   %y.val = sir.mul.f64 %x.val, %two
//!   %y.conf = sir.epistemic.propagate.mul %x.conf, 1.0  ; 2.0 has confidence 1.0
//!   sir.return.void
//! }
//! ```

pub mod alloc_policy;
pub mod blocks;
pub mod builder;
pub mod capabilities;
pub mod cost;
pub mod emit;
pub mod gpu_attention;
pub mod kernels;
pub mod lower;
pub mod metadata;
pub mod module;
pub mod ops;
pub mod passes;
pub mod spirv;
pub mod types;
pub mod values;
pub mod variants;

// Re-exports
pub use blocks::*;
pub use builder::SirBuilder;
pub use capabilities::{
    Capability, CapabilityDiff, CapabilityQuery, EffectiveLimits, GpuVendor, TargetCapabilities,
};
pub use metadata::*;
pub use module::*;
pub use ops::*;
pub use types::*;
pub use values::*;
pub use variants::{
    KernelVariant, Layout, PerformanceMeasurement, Precision, SelectionConstraints, VariantGenerator,
    VariantId, VariantRegistry, VariantSelector, VariantSpace,
};

// GPU attention-based register allocation
pub use gpu_attention::{
    GpuResourceBudget, GpuLiveInterval, GpuMemorySpace, GpuAttentionScore,
    GpuAttentionConfig, GpuPressureAnalysis, GpuAllocationMetrics,
    occupancy_for_registers, optimal_register_target, combined_occupancy,
    divergence_probability, divergence_cost, active_threads_ratio,
    select_spill_victim, select_spill_batch,
};

// A/B allocation policy comparison
pub use alloc_policy::{
    AllocPolicy, AttentionConfig, ConfidenceClass, EpistemicMetadata,
    AllocationMetrics, SpillEvent, SpillReason, MetricsCollector,
    ABComparisonResult, AllocatorOptions, compute_attention_score,
    SpillCandidate, select_spill_victim as cpu_select_spill_victim,
    select_spill_victims, classify_confidence, attention_score,
};

// SKIR cost annotation system (declarative performance primitives)
// Note: Some types have similar names to variants module; use cost:: prefix if needed
pub use cost::{
    CostEnvelope, CostValidationError,
    Fuse, FusionType, OpId,
    KernelVariant as CostKernelVariant, KernelVariantSet,
    LaneGroup, Layout as CostLayout,
    Precision as CostPrecision, ScalarPrecision,
    ReductionMode, RngMode,
    SharedCache, Tile, TileError, Unroll, VectorWidth,
};
