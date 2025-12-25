//! Cross-Platform Portable GPU IR
//!
//! Defines a portable subset of GPU operations that map cleanly to all backends:
//! - CUDA (PTX)
//! - Vulkan (SPIR-V)
//! - Metal (MSL)
//! - OpenCL
//!
//! This enables writing kernels once and compiling to any supported backend,
//! similar to Rust-GPU's approach.
//!
//! # Example
//!
//! ```ignore
//! let kernel = UnifiedKernel::new("saxpy")
//!     .add_param("n", PortableType::I32)
//!     .add_param("a", PortableType::F32)
//!     .add_param("x", PortableType::Ptr(Box::new(PortableType::F32)))
//!     .add_param("y", PortableType::Ptr(Box::new(PortableType::F32)))
//!     .build();
//!
//! // Compile to any backend
//! let ptx = kernel.compile(GpuTarget::Cuda { compute_capability: (8, 0) });
//! let msl = kernel.compile(GpuTarget::Metal { gpu_family: MetalGpuFamily::Apple9 });
//! let spirv = kernel.compile(GpuTarget::Vulkan { version: (1, 3) });
//! ```

use super::ir::{
    GpuBlock, GpuKernel, GpuModule, GpuOp, GpuParam, GpuTarget, GpuTerminator, GpuType,
    MemorySpace, MetalGpuFamily, SharedMemDecl, ValueId,
};
use super::metal::MetalCodegen;
use super::ptx::PtxCodegen;
use rustc_hash::FxHashSet;

// ============================================================================
// Portable Types
// ============================================================================

/// Portable GPU types that exist on all backends
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PortableType {
    // Scalar types (universal)
    Void,
    Bool,
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,

    // Vector types (universal)
    Vec2F32,
    Vec3F32,
    Vec4F32,
    Vec2I32,
    Vec3I32,
    Vec4I32,

    // Pointer (with portable memory space)
    Ptr(Box<PortableType>, PortableMemorySpace),

    // Array
    Array(Box<PortableType>, usize),
}

/// Portable memory spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortableMemorySpace {
    /// Global device memory (all backends)
    Global,
    /// Shared/threadgroup memory (all backends)
    Shared,
    /// Private/thread-local memory (all backends)
    Private,
    /// Constant memory (all backends)
    Constant,
}

impl PortableType {
    /// Convert to backend-specific GpuType
    pub fn to_gpu_type(&self) -> GpuType {
        match self {
            PortableType::Void => GpuType::Void,
            PortableType::Bool => GpuType::Bool,
            PortableType::I32 => GpuType::I32,
            PortableType::I64 => GpuType::I64,
            PortableType::U32 => GpuType::U32,
            PortableType::U64 => GpuType::U64,
            PortableType::F32 => GpuType::F32,
            PortableType::F64 => GpuType::F64,
            PortableType::Vec2F32 => GpuType::Vec2(Box::new(GpuType::F32)),
            PortableType::Vec3F32 => GpuType::Vec3(Box::new(GpuType::F32)),
            PortableType::Vec4F32 => GpuType::Vec4(Box::new(GpuType::F32)),
            PortableType::Vec2I32 => GpuType::Vec2(Box::new(GpuType::I32)),
            PortableType::Vec3I32 => GpuType::Vec3(Box::new(GpuType::I32)),
            PortableType::Vec4I32 => GpuType::Vec4(Box::new(GpuType::I32)),
            PortableType::Ptr(inner, space) => {
                GpuType::Ptr(Box::new(inner.to_gpu_type()), space.to_memory_space())
            }
            PortableType::Array(inner, size) => {
                GpuType::Array(Box::new(inner.to_gpu_type()), *size as u32)
            }
        }
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            PortableType::Void => 0,
            PortableType::Bool => 1,
            PortableType::I32 | PortableType::U32 | PortableType::F32 => 4,
            PortableType::I64 | PortableType::U64 | PortableType::F64 => 8,
            PortableType::Vec2F32 | PortableType::Vec2I32 => 8,
            PortableType::Vec3F32 | PortableType::Vec3I32 => 12,
            PortableType::Vec4F32 | PortableType::Vec4I32 => 16,
            PortableType::Ptr(_, _) => 8,
            PortableType::Array(inner, size) => inner.size_bytes() * size,
        }
    }
}

impl PortableMemorySpace {
    /// Convert to backend-specific MemorySpace
    pub fn to_memory_space(&self) -> MemorySpace {
        match self {
            PortableMemorySpace::Global => MemorySpace::Global,
            PortableMemorySpace::Shared => MemorySpace::Shared,
            PortableMemorySpace::Private => MemorySpace::Local,
            PortableMemorySpace::Constant => MemorySpace::Constant,
        }
    }
}

// ============================================================================
// Portable Operations
// ============================================================================

/// Dimension for thread/block IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    X,
    Y,
    Z,
}

/// Portable GPU operations that work on all backends
#[derive(Debug, Clone, PartialEq)]
pub enum PortableGpuOp {
    // === Constants ===
    ConstI32(i32),
    ConstI64(i64),
    ConstU32(u32),
    ConstU64(u64),
    ConstF32(f32),
    ConstF64(f64),
    ConstBool(bool),

    // === Thread Indexing (Universal) ===
    ThreadId(Dimension),
    BlockId(Dimension),
    BlockDim(Dimension),
    GridDim(Dimension),

    // === Integer Arithmetic ===
    AddI32(ValueId, ValueId),
    SubI32(ValueId, ValueId),
    MulI32(ValueId, ValueId),
    DivI32(ValueId, ValueId),
    RemI32(ValueId, ValueId),
    NegI32(ValueId),

    AddI64(ValueId, ValueId),
    SubI64(ValueId, ValueId),
    MulI64(ValueId, ValueId),
    DivI64(ValueId, ValueId),

    // === Floating Point Arithmetic ===
    AddF32(ValueId, ValueId),
    SubF32(ValueId, ValueId),
    MulF32(ValueId, ValueId),
    DivF32(ValueId, ValueId),
    NegF32(ValueId),

    AddF64(ValueId, ValueId),
    SubF64(ValueId, ValueId),
    MulF64(ValueId, ValueId),
    DivF64(ValueId, ValueId),
    NegF64(ValueId),

    // === Fused Multiply-Add ===
    FmaF32(ValueId, ValueId, ValueId), // a * b + c
    FmaF64(ValueId, ValueId, ValueId),

    // === Math Functions (Portable subset) ===
    SqrtF32(ValueId),
    SqrtF64(ValueId),
    AbsF32(ValueId),
    AbsF64(ValueId),
    AbsI32(ValueId),
    MinF32(ValueId, ValueId),
    MaxF32(ValueId, ValueId),
    MinI32(ValueId, ValueId),
    MaxI32(ValueId, ValueId),
    MinU32(ValueId, ValueId),
    MaxU32(ValueId, ValueId),

    // === Comparisons ===
    EqI32(ValueId, ValueId),
    NeI32(ValueId, ValueId),
    LtI32(ValueId, ValueId),
    LeI32(ValueId, ValueId),
    GtI32(ValueId, ValueId),
    GeI32(ValueId, ValueId),

    LtU32(ValueId, ValueId),
    LeU32(ValueId, ValueId),
    GtU32(ValueId, ValueId),
    GeU32(ValueId, ValueId),

    EqF32(ValueId, ValueId),
    NeF32(ValueId, ValueId),
    LtF32(ValueId, ValueId),
    LeF32(ValueId, ValueId),
    GtF32(ValueId, ValueId),
    GeF32(ValueId, ValueId),

    // === Logical Operations ===
    And(ValueId, ValueId),
    Or(ValueId, ValueId),
    Not(ValueId),

    // === Bitwise Operations ===
    BitAnd(ValueId, ValueId),
    BitOr(ValueId, ValueId),
    BitXor(ValueId, ValueId),
    BitNot(ValueId),
    Shl(ValueId, ValueId),
    Shr(ValueId, ValueId),  // Arithmetic shift
    LShr(ValueId, ValueId), // Logical shift

    // === Type Conversions ===
    I32ToF32(ValueId),
    F32ToI32(ValueId),
    I32ToI64(ValueId),
    I64ToI32(ValueId),
    U32ToF32(ValueId),
    F32ToU32(ValueId),
    F32ToF64(ValueId),
    F64ToF32(ValueId),

    // === Memory Operations ===
    Load(ValueId, PortableMemorySpace),
    Store(ValueId, ValueId, PortableMemorySpace), // ptr, value, space

    // === Atomic Operations (Universal subset) ===
    AtomicAddI32(ValueId, ValueId),
    AtomicAddF32(ValueId, ValueId),
    AtomicMinI32(ValueId, ValueId),
    AtomicMaxI32(ValueId, ValueId),
    AtomicExch(ValueId, ValueId),
    AtomicCas(ValueId, ValueId, ValueId), // ptr, compare, value

    // === Synchronization ===
    Barrier,           // Full thread block barrier
    MemoryFenceBlock,  // Memory fence within block
    MemoryFenceDevice, // Memory fence device-wide

    // === Control Flow ===
    Select(ValueId, ValueId, ValueId), // condition, true_val, false_val

    // === Pointer Arithmetic ===
    PtrOffset(ValueId, ValueId), // base_ptr, offset

    // === Parameters ===
    Param(usize),
}

impl PortableGpuOp {
    /// Convert to backend-specific GpuOp
    pub fn to_gpu_op(&self) -> GpuOp {
        match self {
            // Constants
            PortableGpuOp::ConstI32(v) => GpuOp::ConstInt(*v as i64, GpuType::I32),
            PortableGpuOp::ConstI64(v) => GpuOp::ConstInt(*v, GpuType::I64),
            PortableGpuOp::ConstU32(v) => GpuOp::ConstInt(*v as i64, GpuType::U32),
            PortableGpuOp::ConstU64(v) => GpuOp::ConstInt(*v as i64, GpuType::U64),
            PortableGpuOp::ConstF32(v) => GpuOp::ConstFloat(*v as f64, GpuType::F32),
            PortableGpuOp::ConstF64(v) => GpuOp::ConstFloat(*v, GpuType::F64),
            PortableGpuOp::ConstBool(v) => GpuOp::ConstBool(*v),

            // Thread indexing
            PortableGpuOp::ThreadId(Dimension::X) => GpuOp::ThreadIdX,
            PortableGpuOp::ThreadId(Dimension::Y) => GpuOp::ThreadIdY,
            PortableGpuOp::ThreadId(Dimension::Z) => GpuOp::ThreadIdZ,
            PortableGpuOp::BlockId(Dimension::X) => GpuOp::BlockIdX,
            PortableGpuOp::BlockId(Dimension::Y) => GpuOp::BlockIdY,
            PortableGpuOp::BlockId(Dimension::Z) => GpuOp::BlockIdZ,
            PortableGpuOp::BlockDim(Dimension::X) => GpuOp::BlockDimX,
            PortableGpuOp::BlockDim(Dimension::Y) => GpuOp::BlockDimY,
            PortableGpuOp::BlockDim(Dimension::Z) => GpuOp::BlockDimZ,
            PortableGpuOp::GridDim(Dimension::X) => GpuOp::GridDimX,
            PortableGpuOp::GridDim(Dimension::Y) => GpuOp::GridDimY,
            PortableGpuOp::GridDim(Dimension::Z) => GpuOp::GridDimZ,

            // Integer arithmetic
            PortableGpuOp::AddI32(a, b) | PortableGpuOp::AddI64(a, b) => GpuOp::Add(*a, *b),
            PortableGpuOp::SubI32(a, b) | PortableGpuOp::SubI64(a, b) => GpuOp::Sub(*a, *b),
            PortableGpuOp::MulI32(a, b) | PortableGpuOp::MulI64(a, b) => GpuOp::Mul(*a, *b),
            PortableGpuOp::DivI32(a, b) | PortableGpuOp::DivI64(a, b) => GpuOp::Div(*a, *b),
            PortableGpuOp::RemI32(a, b) => GpuOp::Rem(*a, *b),
            PortableGpuOp::NegI32(a) => GpuOp::Neg(*a),

            // Float arithmetic
            PortableGpuOp::AddF32(a, b) | PortableGpuOp::AddF64(a, b) => GpuOp::FAdd(*a, *b),
            PortableGpuOp::SubF32(a, b) | PortableGpuOp::SubF64(a, b) => GpuOp::FSub(*a, *b),
            PortableGpuOp::MulF32(a, b) | PortableGpuOp::MulF64(a, b) => GpuOp::FMul(*a, *b),
            PortableGpuOp::DivF32(a, b) | PortableGpuOp::DivF64(a, b) => GpuOp::FDiv(*a, *b),
            PortableGpuOp::NegF32(a) | PortableGpuOp::NegF64(a) => GpuOp::FNeg(*a),

            // FMA
            PortableGpuOp::FmaF32(a, b, c) | PortableGpuOp::FmaF64(a, b, c) => {
                GpuOp::FMulAdd(*a, *b, *c)
            }

            // Math functions
            PortableGpuOp::SqrtF32(a) | PortableGpuOp::SqrtF64(a) => GpuOp::FastSqrt(*a),
            PortableGpuOp::AbsF32(_) | PortableGpuOp::AbsF64(_) | PortableGpuOp::AbsI32(_) => {
                // Abs needs to be implemented via select
                todo!("Abs requires lowering")
            }
            PortableGpuOp::MinF32(a, b)
            | PortableGpuOp::MinI32(a, b)
            | PortableGpuOp::MinU32(a, b) => {
                // Min needs select-based lowering for portability
                todo!("Min requires lowering")
            }
            PortableGpuOp::MaxF32(a, b)
            | PortableGpuOp::MaxI32(a, b)
            | PortableGpuOp::MaxU32(a, b) => {
                todo!("Max requires lowering")
            }

            // Comparisons
            PortableGpuOp::EqI32(a, b) => GpuOp::Eq(*a, *b),
            PortableGpuOp::NeI32(a, b) => GpuOp::Ne(*a, *b),
            PortableGpuOp::LtI32(a, b) => GpuOp::Lt(*a, *b),
            PortableGpuOp::LeI32(a, b) => GpuOp::Le(*a, *b),
            PortableGpuOp::GtI32(a, b) => GpuOp::Gt(*a, *b),
            PortableGpuOp::GeI32(a, b) => GpuOp::Ge(*a, *b),
            PortableGpuOp::LtU32(a, b) => GpuOp::Lt(*a, *b), // TODO: unsigned comparison
            PortableGpuOp::LeU32(a, b) => GpuOp::Le(*a, *b),
            PortableGpuOp::GtU32(a, b) => GpuOp::Gt(*a, *b),
            PortableGpuOp::GeU32(a, b) => GpuOp::Ge(*a, *b),
            PortableGpuOp::EqF32(a, b) => GpuOp::FEq(*a, *b),
            PortableGpuOp::NeF32(a, b) => GpuOp::FNe(*a, *b),
            PortableGpuOp::LtF32(a, b) => GpuOp::FLt(*a, *b),
            PortableGpuOp::LeF32(a, b) => GpuOp::FLe(*a, *b),
            PortableGpuOp::GtF32(a, b) => GpuOp::FGt(*a, *b),
            PortableGpuOp::GeF32(a, b) => GpuOp::FGe(*a, *b),

            // Logical
            PortableGpuOp::And(a, b) => GpuOp::And(*a, *b),
            PortableGpuOp::Or(a, b) => GpuOp::Or(*a, *b),
            PortableGpuOp::Not(a) => GpuOp::Not(*a),

            // Bitwise
            PortableGpuOp::BitAnd(a, b) => GpuOp::BitAnd(*a, *b),
            PortableGpuOp::BitOr(a, b) => GpuOp::BitOr(*a, *b),
            PortableGpuOp::BitXor(a, b) => GpuOp::BitXor(*a, *b),
            PortableGpuOp::BitNot(a) => GpuOp::BitNot(*a),
            PortableGpuOp::Shl(a, b) => GpuOp::Shl(*a, *b),
            PortableGpuOp::Shr(a, b) => GpuOp::Shr(*a, *b),
            PortableGpuOp::LShr(a, b) => GpuOp::LShr(*a, *b),

            // Conversions
            PortableGpuOp::I32ToF32(a) => GpuOp::SiToFp(*a, GpuType::F32),
            PortableGpuOp::F32ToI32(a) => GpuOp::FpToSi(*a, GpuType::I32),
            PortableGpuOp::I32ToI64(a) => GpuOp::SExt(*a, GpuType::I64),
            PortableGpuOp::I64ToI32(a) => GpuOp::Trunc(*a, GpuType::I32),
            PortableGpuOp::U32ToF32(a) => GpuOp::UiToFp(*a, GpuType::F32),
            PortableGpuOp::F32ToU32(a) => GpuOp::FpToUi(*a, GpuType::U32),
            PortableGpuOp::F32ToF64(a) => GpuOp::FpExt(*a, GpuType::F64),
            PortableGpuOp::F64ToF32(a) => GpuOp::FpTrunc(*a, GpuType::F32),

            // Memory
            PortableGpuOp::Load(ptr, space) => GpuOp::Load(*ptr, space.to_memory_space()),
            PortableGpuOp::Store(ptr, val, space) => {
                GpuOp::Store(*ptr, *val, space.to_memory_space())
            }

            // Atomics
            PortableGpuOp::AtomicAddI32(ptr, val) | PortableGpuOp::AtomicAddF32(ptr, val) => {
                GpuOp::AtomicAdd(*ptr, *val)
            }
            PortableGpuOp::AtomicMinI32(ptr, val) => GpuOp::AtomicMin(*ptr, *val),
            PortableGpuOp::AtomicMaxI32(ptr, val) => GpuOp::AtomicMax(*ptr, *val),
            PortableGpuOp::AtomicExch(ptr, val) => GpuOp::AtomicExch(*ptr, *val),
            PortableGpuOp::AtomicCas(ptr, cmp, val) => GpuOp::AtomicCas(*ptr, *cmp, *val),

            // Synchronization
            PortableGpuOp::Barrier => GpuOp::SyncThreads,
            PortableGpuOp::MemoryFenceBlock => GpuOp::MemoryFence(MemorySpace::Shared),
            PortableGpuOp::MemoryFenceDevice => GpuOp::MemoryFence(MemorySpace::Global),

            // Control flow
            PortableGpuOp::Select(cond, t, f) => GpuOp::Select(*cond, *t, *f),

            // Pointer arithmetic
            PortableGpuOp::PtrOffset(base, offset) => GpuOp::GetElementPtr(*base, vec![*offset]),

            // Parameters
            PortableGpuOp::Param(idx) => GpuOp::Param(*idx as u32),
        }
    }

    /// Check if this operation is supported on a given target
    pub fn is_supported_on(&self, target: &GpuTarget) -> bool {
        match self {
            // Universal operations supported everywhere
            PortableGpuOp::ConstI32(_)
            | PortableGpuOp::ConstU32(_)
            | PortableGpuOp::ConstF32(_)
            | PortableGpuOp::ConstBool(_)
            | PortableGpuOp::ThreadId(_)
            | PortableGpuOp::BlockId(_)
            | PortableGpuOp::BlockDim(_)
            | PortableGpuOp::GridDim(_)
            | PortableGpuOp::AddI32(_, _)
            | PortableGpuOp::SubI32(_, _)
            | PortableGpuOp::MulI32(_, _)
            | PortableGpuOp::AddF32(_, _)
            | PortableGpuOp::SubF32(_, _)
            | PortableGpuOp::MulF32(_, _)
            | PortableGpuOp::DivF32(_, _)
            | PortableGpuOp::Barrier
            | PortableGpuOp::Select(_, _, _)
            | PortableGpuOp::Load(_, _)
            | PortableGpuOp::Store(_, _, _)
            | PortableGpuOp::Param(_) => true,

            // F64 operations - check target support
            PortableGpuOp::ConstF64(_)
            | PortableGpuOp::AddF64(_, _)
            | PortableGpuOp::SubF64(_, _)
            | PortableGpuOp::MulF64(_, _)
            | PortableGpuOp::DivF64(_, _)
            | PortableGpuOp::NegF64(_)
            | PortableGpuOp::FmaF64(_, _, _)
            | PortableGpuOp::SqrtF64(_) => {
                match target {
                    GpuTarget::Cuda { .. } => true,
                    GpuTarget::Metal { gpu_family } => {
                        // Only Mac GPUs have full F64 support
                        matches!(gpu_family, MetalGpuFamily::Mac2)
                    }
                    GpuTarget::Vulkan { .. } | GpuTarget::OpenCL { .. } => true,
                    _ => false,
                }
            }

            // I64 operations
            PortableGpuOp::ConstI64(_)
            | PortableGpuOp::ConstU64(_)
            | PortableGpuOp::AddI64(_, _)
            | PortableGpuOp::SubI64(_, _)
            | PortableGpuOp::MulI64(_, _)
            | PortableGpuOp::DivI64(_, _) => true,

            // Atomic operations - most are universal
            PortableGpuOp::AtomicAddI32(_, _)
            | PortableGpuOp::AtomicMinI32(_, _)
            | PortableGpuOp::AtomicMaxI32(_, _)
            | PortableGpuOp::AtomicExch(_, _)
            | PortableGpuOp::AtomicCas(_, _, _) => true,

            // Atomic float add - needs checking
            PortableGpuOp::AtomicAddF32(_, _) => {
                match target {
                    GpuTarget::Cuda { compute_capability } => compute_capability.0 >= 2,
                    GpuTarget::Metal { .. } => true,
                    GpuTarget::Vulkan { .. } => true, // With extension
                    _ => false,
                }
            }

            // Everything else is universal
            _ => true,
        }
    }
}

// ============================================================================
// Backend Capabilities
// ============================================================================

/// Capabilities of a GPU backend
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend target
    pub target: GpuTarget,

    /// Supports F64 (double precision)
    pub supports_f64: bool,

    /// Supports atomic float add
    pub supports_atomic_float: bool,

    /// Supports warp/simd shuffle
    pub supports_shuffle: bool,

    /// Supports cooperative groups
    pub supports_cooperative_groups: bool,

    /// Supports tensor cores / matrix operations
    pub supports_tensor_ops: bool,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Maximum shared memory per block (bytes)
    pub max_shared_memory: u32,

    /// Warp/SIMD width
    pub warp_size: u32,

    /// Supported memory spaces
    pub memory_spaces: FxHashSet<PortableMemorySpace>,
}

impl BackendCapabilities {
    /// Create capabilities for a CUDA target
    pub fn cuda(compute_capability: (u32, u32)) -> Self {
        let (major, minor) = compute_capability;
        Self {
            target: GpuTarget::Cuda { compute_capability },
            supports_f64: true,
            supports_atomic_float: major >= 2,
            supports_shuffle: major >= 3,
            supports_cooperative_groups: major >= 6,
            supports_tensor_ops: major >= 7,
            max_threads_per_block: 1024,
            max_shared_memory: if major >= 8 { 164 * 1024 } else { 48 * 1024 },
            warp_size: 32,
            memory_spaces: [
                PortableMemorySpace::Global,
                PortableMemorySpace::Shared,
                PortableMemorySpace::Private,
                PortableMemorySpace::Constant,
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Create capabilities for a Metal target
    pub fn metal(gpu_family: MetalGpuFamily) -> Self {
        Self {
            target: GpuTarget::Metal { gpu_family },
            supports_f64: matches!(gpu_family, MetalGpuFamily::Mac2),
            supports_atomic_float: true,
            supports_shuffle: true,
            supports_cooperative_groups: true, // via simdgroup
            supports_tensor_ops: matches!(
                gpu_family,
                MetalGpuFamily::Apple7
                    | MetalGpuFamily::Apple8
                    | MetalGpuFamily::Apple9
                    | MetalGpuFamily::Apple10
            ),
            max_threads_per_block: gpu_family.max_threads_per_threadgroup(),
            max_shared_memory: gpu_family.max_threadgroup_memory(),
            warp_size: gpu_family.simd_width(),
            memory_spaces: [
                PortableMemorySpace::Global,
                PortableMemorySpace::Shared,
                PortableMemorySpace::Private,
                PortableMemorySpace::Constant,
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Create capabilities for a Vulkan target
    pub fn vulkan(version: (u32, u32)) -> Self {
        Self {
            target: GpuTarget::Vulkan { version },
            supports_f64: true, // With extension
            supports_atomic_float: true,
            supports_shuffle: version.0 >= 1 && version.1 >= 1,
            supports_cooperative_groups: version.0 >= 1 && version.1 >= 1,
            supports_tensor_ops: false, // Vulkan doesn't have native tensor ops
            max_threads_per_block: 1024,
            max_shared_memory: 32 * 1024,
            warp_size: 32, // Varies by vendor
            memory_spaces: [
                PortableMemorySpace::Global,
                PortableMemorySpace::Shared,
                PortableMemorySpace::Private,
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Check if an operation is supported
    pub fn supports_op(&self, op: &PortableGpuOp) -> bool {
        op.is_supported_on(&self.target)
    }
}

// ============================================================================
// Unified Kernel
// ============================================================================

/// A portable kernel that can be compiled to any backend
#[derive(Debug, Clone)]
pub struct UnifiedKernel {
    /// Kernel name
    pub name: String,

    /// Parameters
    pub params: Vec<UnifiedParam>,

    /// Instructions (portable operations)
    pub instructions: Vec<(ValueId, PortableGpuOp)>,

    /// Shared memory declarations
    pub shared_memory: Vec<UnifiedSharedMem>,

    /// Required capabilities
    pub required_capabilities: FxHashSet<Capability>,
}

/// A kernel parameter
#[derive(Debug, Clone)]
pub struct UnifiedParam {
    pub name: String,
    pub ty: PortableType,
}

/// Shared memory declaration
#[derive(Debug, Clone)]
pub struct UnifiedSharedMem {
    pub name: String,
    pub elem_type: PortableType,
    pub size: usize,
}

/// Required capability for a kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    Float64,
    AtomicFloat,
    Shuffle,
    CooperativeGroups,
    TensorOps,
}

impl UnifiedKernel {
    /// Create a new unified kernel
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            instructions: Vec::new(),
            shared_memory: Vec::new(),
            required_capabilities: FxHashSet::default(),
        }
    }

    /// Add a parameter
    pub fn add_param(&mut self, name: &str, ty: PortableType) -> &mut Self {
        self.params.push(UnifiedParam {
            name: name.into(),
            ty,
        });
        self
    }

    /// Add an instruction
    pub fn add_instruction(&mut self, id: ValueId, op: PortableGpuOp) -> &mut Self {
        // Track required capabilities
        match &op {
            PortableGpuOp::ConstF64(_)
            | PortableGpuOp::AddF64(_, _)
            | PortableGpuOp::SubF64(_, _)
            | PortableGpuOp::MulF64(_, _)
            | PortableGpuOp::DivF64(_, _) => {
                self.required_capabilities.insert(Capability::Float64);
            }
            PortableGpuOp::AtomicAddF32(_, _) => {
                self.required_capabilities.insert(Capability::AtomicFloat);
            }
            _ => {}
        }
        self.instructions.push((id, op));
        self
    }

    /// Add shared memory
    pub fn add_shared_mem(
        &mut self,
        name: &str,
        elem_type: PortableType,
        size: usize,
    ) -> &mut Self {
        self.shared_memory.push(UnifiedSharedMem {
            name: name.into(),
            elem_type,
            size,
        });
        self
    }

    /// Check if this kernel can run on a backend
    pub fn is_compatible_with(&self, capabilities: &BackendCapabilities) -> bool {
        for cap in &self.required_capabilities {
            let supported = match cap {
                Capability::Float64 => capabilities.supports_f64,
                Capability::AtomicFloat => capabilities.supports_atomic_float,
                Capability::Shuffle => capabilities.supports_shuffle,
                Capability::CooperativeGroups => capabilities.supports_cooperative_groups,
                Capability::TensorOps => capabilities.supports_tensor_ops,
            };
            if !supported {
                return false;
            }
        }
        true
    }

    /// Get unsupported capabilities for a backend
    pub fn unsupported_capabilities(&self, capabilities: &BackendCapabilities) -> Vec<Capability> {
        self.required_capabilities
            .iter()
            .filter(|cap| {
                let supported = match cap {
                    Capability::Float64 => capabilities.supports_f64,
                    Capability::AtomicFloat => capabilities.supports_atomic_float,
                    Capability::Shuffle => capabilities.supports_shuffle,
                    Capability::CooperativeGroups => capabilities.supports_cooperative_groups,
                    Capability::TensorOps => capabilities.supports_tensor_ops,
                };
                !supported
            })
            .copied()
            .collect()
    }
}

// ============================================================================
// Backend Selection
// ============================================================================

/// Available GPU backends on the current system
#[derive(Debug, Clone)]
pub struct AvailableBackends {
    pub cuda: Option<BackendCapabilities>,
    pub metal: Option<BackendCapabilities>,
    pub vulkan: Option<BackendCapabilities>,
    pub opencl: Option<BackendCapabilities>,
}

impl AvailableBackends {
    /// Create with no backends (for testing)
    pub fn none() -> Self {
        Self {
            cuda: None,
            metal: None,
            vulkan: None,
            opencl: None,
        }
    }

    /// Get the best available backend for a kernel
    pub fn best_for(&self, kernel: &UnifiedKernel) -> Option<&BackendCapabilities> {
        // Priority: CUDA > Metal > Vulkan > OpenCL
        if let Some(ref cuda) = self.cuda
            && kernel.is_compatible_with(cuda)
        {
            return Some(cuda);
        }
        if let Some(ref metal) = self.metal
            && kernel.is_compatible_with(metal)
        {
            return Some(metal);
        }
        if let Some(ref vulkan) = self.vulkan
            && kernel.is_compatible_with(vulkan)
        {
            return Some(vulkan);
        }
        if let Some(ref opencl) = self.opencl
            && kernel.is_compatible_with(opencl)
        {
            return Some(opencl);
        }
        None
    }

    /// Get all compatible backends for a kernel
    pub fn compatible_with(&self, kernel: &UnifiedKernel) -> Vec<&BackendCapabilities> {
        let mut result = Vec::new();
        if let Some(ref cuda) = self.cuda
            && kernel.is_compatible_with(cuda)
        {
            result.push(cuda);
        }
        if let Some(ref metal) = self.metal
            && kernel.is_compatible_with(metal)
        {
            result.push(metal);
        }
        if let Some(ref vulkan) = self.vulkan
            && kernel.is_compatible_with(vulkan)
        {
            result.push(vulkan);
        }
        if let Some(ref opencl) = self.opencl
            && kernel.is_compatible_with(opencl)
        {
            result.push(opencl);
        }
        result
    }
}

// ============================================================================
// Unified Kernel Compilation
// ============================================================================

/// Error during kernel compilation
#[derive(Debug, Clone)]
pub enum CompileError {
    /// Backend doesn't support required capability
    UnsupportedCapability(Capability),
    /// Multiple unsupported capabilities
    UnsupportedCapabilities(Vec<Capability>),
    /// Invalid instruction
    InvalidInstruction(String),
    /// Backend-specific error
    BackendError(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnsupportedCapability(cap) => {
                write!(f, "Backend doesn't support capability: {:?}", cap)
            }
            CompileError::UnsupportedCapabilities(caps) => {
                write!(f, "Backend doesn't support capabilities: {:?}", caps)
            }
            CompileError::InvalidInstruction(msg) => {
                write!(f, "Invalid instruction: {}", msg)
            }
            CompileError::BackendError(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for CompileError {}

/// Result type for compilation
pub type CompileResult<T> = Result<T, CompileError>;

/// Compiled kernel output
#[derive(Debug)]
pub enum CompiledKernel {
    /// PTX assembly for CUDA
    Ptx(String),
    /// Metal Shading Language
    Metal(String),
    /// SPIR-V binary
    SpirV(Vec<u32>),
    /// OpenCL C source
    OpenCL(String),
}

impl CompiledKernel {
    /// Get as PTX string
    pub fn as_ptx(&self) -> Option<&str> {
        match self {
            CompiledKernel::Ptx(s) => Some(s),
            _ => None,
        }
    }

    /// Get as Metal string
    pub fn as_metal(&self) -> Option<&str> {
        match self {
            CompiledKernel::Metal(s) => Some(s),
            _ => None,
        }
    }

    /// Get backend name
    pub fn backend_name(&self) -> &'static str {
        match self {
            CompiledKernel::Ptx(_) => "CUDA (PTX)",
            CompiledKernel::Metal(_) => "Metal (MSL)",
            CompiledKernel::SpirV(_) => "Vulkan (SPIR-V)",
            CompiledKernel::OpenCL(_) => "OpenCL",
        }
    }
}

/// Unified kernel compiler
pub struct UnifiedCompiler {
    /// Optimization level (0-3)
    pub opt_level: u32,
    /// Generate debug info
    pub debug_info: bool,
    /// Fast math approximations
    pub fast_math: bool,
}

impl Default for UnifiedCompiler {
    fn default() -> Self {
        Self {
            opt_level: 2,
            debug_info: false,
            fast_math: false,
        }
    }
}

impl UnifiedCompiler {
    /// Create a new compiler with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set optimization level
    pub fn with_opt_level(mut self, level: u32) -> Self {
        self.opt_level = level.min(3);
        self
    }

    /// Enable debug info
    pub fn with_debug_info(mut self, enabled: bool) -> Self {
        self.debug_info = enabled;
        self
    }

    /// Enable fast math
    pub fn with_fast_math(mut self, enabled: bool) -> Self {
        self.fast_math = enabled;
        self
    }

    /// Compile a unified kernel to a specific backend
    pub fn compile(
        &self,
        kernel: &UnifiedKernel,
        capabilities: &BackendCapabilities,
    ) -> CompileResult<CompiledKernel> {
        // Check compatibility
        let unsupported = kernel.unsupported_capabilities(capabilities);
        if !unsupported.is_empty() {
            if unsupported.len() == 1 {
                return Err(CompileError::UnsupportedCapability(unsupported[0]));
            }
            return Err(CompileError::UnsupportedCapabilities(unsupported));
        }

        // Lower to GpuKernel
        let gpu_kernel = self.lower_to_gpu_kernel(kernel)?;

        // Compile based on target
        match &capabilities.target {
            GpuTarget::Cuda { compute_capability } => {
                self.compile_to_ptx(&gpu_kernel, *compute_capability)
            }
            GpuTarget::Metal { gpu_family } => self.compile_to_metal(&gpu_kernel, *gpu_family),
            GpuTarget::Vulkan { .. } => self.compile_to_spirv(&gpu_kernel),
            GpuTarget::OpenCL { .. } => self.compile_to_opencl(&gpu_kernel),
            _ => Err(CompileError::BackendError("Unsupported backend".into())),
        }
    }

    /// Lower UnifiedKernel to GpuKernel
    fn lower_to_gpu_kernel(&self, kernel: &UnifiedKernel) -> CompileResult<GpuKernel> {
        // Convert parameters
        let params: Vec<GpuParam> = kernel
            .params
            .iter()
            .map(|p| {
                // Determine memory space from type
                let space = match &p.ty {
                    PortableType::Ptr(_, space) => space.to_memory_space(),
                    _ => MemorySpace::Global,
                };
                GpuParam {
                    name: p.name.clone(),
                    ty: p.ty.to_gpu_type(),
                    space,
                    restrict: false,
                }
            })
            .collect();

        // Create entry block
        let mut entry_block = GpuBlock {
            id: super::ir::BlockId(0),
            label: "entry".to_string(),
            instructions: Vec::new(),
            terminator: GpuTerminator::ReturnVoid,
        };

        // Convert instructions
        for (value_id, portable_op) in &kernel.instructions {
            let gpu_op = portable_op.to_gpu_op();
            entry_block.instructions.push((*value_id, gpu_op));
        }

        // Convert shared memory declarations
        let shared_memory: Vec<SharedMemDecl> = kernel
            .shared_memory
            .iter()
            .map(|s| SharedMemDecl {
                name: s.name.clone(),
                elem_type: s.elem_type.to_gpu_type(),
                size: s.size as u32,
                align: 16, // Default alignment
            })
            .collect();

        // Calculate shared memory size
        let shared_mem_size: u32 = shared_memory
            .iter()
            .map(|s| s.size * s.elem_type.size_bytes())
            .sum();

        Ok(GpuKernel {
            name: kernel.name.clone(),
            params,
            blocks: vec![entry_block],
            shared_memory,
            entry: super::ir::BlockId(0),
            max_threads: None,
            shared_mem_size,
        })
    }

    /// Compile to PTX
    fn compile_to_ptx(
        &self,
        kernel: &GpuKernel,
        compute_capability: (u32, u32),
    ) -> CompileResult<CompiledKernel> {
        let target = GpuTarget::Cuda { compute_capability };
        let mut module = GpuModule::new("portable_module", target);
        module.kernels.insert(kernel.name.clone(), kernel.clone());

        let mut codegen = PtxCodegen::new(compute_capability);
        let ptx = codegen.generate(&module);
        Ok(CompiledKernel::Ptx(ptx))
    }

    /// Compile to Metal
    fn compile_to_metal(
        &self,
        kernel: &GpuKernel,
        gpu_family: MetalGpuFamily,
    ) -> CompileResult<CompiledKernel> {
        let target = GpuTarget::Metal { gpu_family };
        let mut module = GpuModule::new("portable_module", target);
        module.kernels.insert(kernel.name.clone(), kernel.clone());

        let config = super::metal::MetalCodegenConfig {
            gpu_family,
            fast_math: self.fast_math,
            debug_info: self.debug_info,
            epistemic_enabled: false,
            max_threads_per_threadgroup: gpu_family.max_threads_per_threadgroup(),
        };
        let mut codegen = MetalCodegen::new(config);
        let msl = codegen.generate(&module);
        Ok(CompiledKernel::Metal(msl))
    }

    /// Compile to SPIR-V (placeholder)
    fn compile_to_spirv(&self, _kernel: &GpuKernel) -> CompileResult<CompiledKernel> {
        // SPIR-V codegen requires the "gpu" feature
        #[cfg(feature = "gpu")]
        {
            // Would use SpirvCodegen here
            Err(CompileError::BackendError(
                "SPIR-V compilation not yet implemented in portable API".into(),
            ))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(CompileError::BackendError(
                "SPIR-V requires 'gpu' feature".into(),
            ))
        }
    }

    /// Compile to OpenCL C (placeholder)
    fn compile_to_opencl(&self, _kernel: &GpuKernel) -> CompileResult<CompiledKernel> {
        Err(CompileError::BackendError(
            "OpenCL compilation not yet implemented".into(),
        ))
    }
}

/// Convenience function to compile a kernel
pub fn compile_kernel(kernel: &UnifiedKernel, target: &GpuTarget) -> CompileResult<CompiledKernel> {
    let capabilities = match target {
        GpuTarget::Cuda { compute_capability } => BackendCapabilities::cuda(*compute_capability),
        GpuTarget::Metal { gpu_family } => BackendCapabilities::metal(*gpu_family),
        GpuTarget::Vulkan { version } => BackendCapabilities::vulkan(*version),
        _ => return Err(CompileError::BackendError("Unsupported target".into())),
    };

    UnifiedCompiler::new().compile(kernel, &capabilities)
}

/// Compile a kernel to all compatible backends
pub fn compile_to_all(
    kernel: &UnifiedKernel,
    backends: &AvailableBackends,
) -> Vec<(GpuTarget, CompileResult<CompiledKernel>)> {
    let compiler = UnifiedCompiler::new();
    let mut results = Vec::new();

    for caps in backends.compatible_with(kernel) {
        let result = compiler.compile(kernel, caps);
        results.push((caps.target, result));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portable_type_conversion() {
        assert_eq!(PortableType::I32.to_gpu_type(), GpuType::I32);
        assert_eq!(PortableType::F32.to_gpu_type(), GpuType::F32);
        assert_eq!(PortableType::Bool.to_gpu_type(), GpuType::Bool);
    }

    #[test]
    fn test_portable_type_size() {
        assert_eq!(PortableType::I32.size_bytes(), 4);
        assert_eq!(PortableType::F64.size_bytes(), 8);
        assert_eq!(PortableType::Vec4F32.size_bytes(), 16);
    }

    #[test]
    fn test_portable_op_conversion() {
        let op = PortableGpuOp::AddI32(ValueId(0), ValueId(1));
        assert!(matches!(op.to_gpu_op(), GpuOp::Add(_, _)));

        let op = PortableGpuOp::ThreadId(Dimension::X);
        assert!(matches!(op.to_gpu_op(), GpuOp::ThreadIdX));
    }

    #[test]
    fn test_cuda_capabilities() {
        let caps = BackendCapabilities::cuda((8, 0));
        assert!(caps.supports_f64);
        assert!(caps.supports_atomic_float);
        assert!(caps.supports_shuffle);
        assert!(caps.supports_cooperative_groups);
        assert!(caps.supports_tensor_ops);
        assert_eq!(caps.warp_size, 32);
    }

    #[test]
    fn test_metal_capabilities() {
        let caps = BackendCapabilities::metal(MetalGpuFamily::Apple9);
        assert!(!caps.supports_f64); // Apple Silicon doesn't have F64
        assert!(caps.supports_atomic_float);
        assert!(caps.supports_shuffle);
        assert!(caps.supports_tensor_ops);
        assert_eq!(caps.warp_size, 32);
    }

    #[test]
    fn test_unified_kernel_creation() {
        let mut kernel = UnifiedKernel::new("saxpy");
        kernel
            .add_param("n", PortableType::I32)
            .add_param("a", PortableType::F32)
            .add_param(
                "x",
                PortableType::Ptr(Box::new(PortableType::F32), PortableMemorySpace::Global),
            )
            .add_param(
                "y",
                PortableType::Ptr(Box::new(PortableType::F32), PortableMemorySpace::Global),
            );

        assert_eq!(kernel.name, "saxpy");
        assert_eq!(kernel.params.len(), 4);
    }

    #[test]
    fn test_kernel_compatibility() {
        let mut kernel = UnifiedKernel::new("test");
        kernel.add_instruction(ValueId(0), PortableGpuOp::ConstF64(3.14));

        let cuda_caps = BackendCapabilities::cuda((8, 0));
        let metal_caps = BackendCapabilities::metal(MetalGpuFamily::Apple9);

        assert!(kernel.is_compatible_with(&cuda_caps));
        assert!(!kernel.is_compatible_with(&metal_caps)); // No F64 on Apple Silicon
    }

    #[test]
    fn test_backend_selection() {
        let mut backends = AvailableBackends::none();
        backends.cuda = Some(BackendCapabilities::cuda((8, 0)));
        backends.metal = Some(BackendCapabilities::metal(MetalGpuFamily::Apple9));

        // Kernel without F64 should work on both
        let kernel = UnifiedKernel::new("simple");
        let best = backends.best_for(&kernel);
        assert!(best.is_some());
        assert!(matches!(best.unwrap().target, GpuTarget::Cuda { .. }));

        // Kernel with F64 should only work on CUDA
        let mut f64_kernel = UnifiedKernel::new("f64_test");
        f64_kernel.add_instruction(ValueId(0), PortableGpuOp::ConstF64(1.0));
        let compatible = backends.compatible_with(&f64_kernel);
        assert_eq!(compatible.len(), 1);
    }

    #[test]
    fn test_operation_support() {
        let cuda = GpuTarget::Cuda {
            compute_capability: (8, 0),
        };
        let metal = GpuTarget::Metal {
            gpu_family: MetalGpuFamily::Apple9,
        };

        // F32 add is universal
        let f32_add = PortableGpuOp::AddF32(ValueId(0), ValueId(1));
        assert!(f32_add.is_supported_on(&cuda));
        assert!(f32_add.is_supported_on(&metal));

        // F64 add is not on Apple Silicon
        let f64_add = PortableGpuOp::AddF64(ValueId(0), ValueId(1));
        assert!(f64_add.is_supported_on(&cuda));
        assert!(!f64_add.is_supported_on(&metal));
    }

    #[test]
    fn test_portable_saxpy_kernel() {
        let mut kernel = UnifiedKernel::new("saxpy");

        // Parameters
        kernel
            .add_param("n", PortableType::I32)
            .add_param("a", PortableType::F32)
            .add_param(
                "x",
                PortableType::Ptr(Box::new(PortableType::F32), PortableMemorySpace::Global),
            )
            .add_param(
                "y",
                PortableType::Ptr(Box::new(PortableType::F32), PortableMemorySpace::Global),
            );

        // Compute global index
        kernel.add_instruction(ValueId(0), PortableGpuOp::ThreadId(Dimension::X));
        kernel.add_instruction(ValueId(1), PortableGpuOp::BlockId(Dimension::X));
        kernel.add_instruction(ValueId(2), PortableGpuOp::BlockDim(Dimension::X));
        kernel.add_instruction(ValueId(3), PortableGpuOp::MulI32(ValueId(1), ValueId(2)));
        kernel.add_instruction(ValueId(4), PortableGpuOp::AddI32(ValueId(0), ValueId(3)));

        // Load parameters
        kernel.add_instruction(ValueId(5), PortableGpuOp::Param(0)); // n
        kernel.add_instruction(ValueId(6), PortableGpuOp::Param(1)); // a
        kernel.add_instruction(ValueId(7), PortableGpuOp::Param(2)); // x
        kernel.add_instruction(ValueId(8), PortableGpuOp::Param(3)); // y

        // Should be compatible with both CUDA and Metal
        let cuda_caps = BackendCapabilities::cuda((8, 0));
        let metal_caps = BackendCapabilities::metal(MetalGpuFamily::Apple9);

        assert!(kernel.is_compatible_with(&cuda_caps));
        assert!(kernel.is_compatible_with(&metal_caps));
        assert!(kernel.required_capabilities.is_empty());
    }

    // =========================================================================
    // Compilation Tests
    // =========================================================================

    #[test]
    fn test_compile_simple_kernel_to_ptx() {
        let mut kernel = UnifiedKernel::new("simple_add");
        kernel.add_param("a", PortableType::I32);
        kernel.add_param("b", PortableType::I32);

        // a + b
        kernel.add_instruction(ValueId(0), PortableGpuOp::Param(0));
        kernel.add_instruction(ValueId(1), PortableGpuOp::Param(1));
        kernel.add_instruction(ValueId(2), PortableGpuOp::AddI32(ValueId(0), ValueId(1)));

        let target = GpuTarget::Cuda {
            compute_capability: (8, 0),
        };
        let result = compile_kernel(&kernel, &target);

        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(matches!(compiled, CompiledKernel::Ptx(_)));

        let ptx = compiled.as_ptx().unwrap();
        assert!(ptx.contains(".entry simple_add"));
        assert!(ptx.contains(".param"));
    }

    #[test]
    fn test_compile_simple_kernel_to_metal() {
        let mut kernel = UnifiedKernel::new("simple_mul");
        kernel.add_param("x", PortableType::F32);
        kernel.add_param("y", PortableType::F32);

        // x * y
        kernel.add_instruction(ValueId(0), PortableGpuOp::Param(0));
        kernel.add_instruction(ValueId(1), PortableGpuOp::Param(1));
        kernel.add_instruction(ValueId(2), PortableGpuOp::MulF32(ValueId(0), ValueId(1)));

        let target = GpuTarget::Metal {
            gpu_family: MetalGpuFamily::Apple9,
        };
        let result = compile_kernel(&kernel, &target);

        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(matches!(compiled, CompiledKernel::Metal(_)));

        let msl = compiled.as_metal().unwrap();
        assert!(msl.contains("kernel void simple_mul"));
    }

    #[test]
    fn test_compile_with_thread_id() {
        let mut kernel = UnifiedKernel::new("vector_scale");
        kernel.add_param(
            "data",
            PortableType::Ptr(Box::new(PortableType::F32), PortableMemorySpace::Global),
        );
        kernel.add_param("scale", PortableType::F32);

        // Get thread index
        kernel.add_instruction(ValueId(0), PortableGpuOp::ThreadId(Dimension::X));
        kernel.add_instruction(ValueId(1), PortableGpuOp::BlockId(Dimension::X));
        kernel.add_instruction(ValueId(2), PortableGpuOp::BlockDim(Dimension::X));
        kernel.add_instruction(ValueId(3), PortableGpuOp::MulI32(ValueId(1), ValueId(2)));
        kernel.add_instruction(ValueId(4), PortableGpuOp::AddI32(ValueId(0), ValueId(3)));

        // Compile to CUDA
        let cuda_result = compile_kernel(
            &kernel,
            &GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );
        assert!(cuda_result.is_ok());
        let ptx = cuda_result.unwrap().as_ptx().unwrap().to_string();
        assert!(ptx.contains("tid.x"));
        assert!(ptx.contains("ctaid.x"));
        assert!(ptx.contains("ntid.x"));

        // Compile to Metal
        let metal_result = compile_kernel(
            &kernel,
            &GpuTarget::Metal {
                gpu_family: MetalGpuFamily::Apple8,
            },
        );
        assert!(metal_result.is_ok());
        let msl = metal_result.unwrap().as_metal().unwrap().to_string();
        assert!(
            msl.contains("thread_position_in_grid")
                || msl.contains("thread_position_in_threadgroup")
        );
    }

    #[test]
    fn test_compile_f64_kernel_fails_on_apple_silicon() {
        let mut kernel = UnifiedKernel::new("f64_test");
        kernel.add_param("x", PortableType::F64);
        kernel.add_instruction(ValueId(0), PortableGpuOp::ConstF64(3.14159));

        // Should fail on Apple Silicon (no F64)
        let metal_result = compile_kernel(
            &kernel,
            &GpuTarget::Metal {
                gpu_family: MetalGpuFamily::Apple9,
            },
        );
        assert!(metal_result.is_err());
        assert!(matches!(
            metal_result.unwrap_err(),
            CompileError::UnsupportedCapability(Capability::Float64)
        ));

        // Should succeed on CUDA
        let cuda_result = compile_kernel(
            &kernel,
            &GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );
        assert!(cuda_result.is_ok());
    }

    #[test]
    fn test_unified_compiler_builder() {
        let compiler = UnifiedCompiler::new()
            .with_opt_level(3)
            .with_fast_math(true)
            .with_debug_info(true);

        assert_eq!(compiler.opt_level, 3);
        assert!(compiler.fast_math);
        assert!(compiler.debug_info);
    }

    #[test]
    fn test_compile_to_all_backends() {
        let kernel = UnifiedKernel::new("universal");

        let mut backends = AvailableBackends::none();
        backends.cuda = Some(BackendCapabilities::cuda((8, 0)));
        backends.metal = Some(BackendCapabilities::metal(MetalGpuFamily::Apple9));

        let results = compile_to_all(&kernel, &backends);

        // Should have 2 results (CUDA and Metal)
        assert_eq!(results.len(), 2);

        // Both should succeed for a simple kernel
        for (target, result) in &results {
            assert!(
                result.is_ok(),
                "Compilation failed for {:?}: {:?}",
                target,
                result
            );
        }
    }

    #[test]
    fn test_compiled_kernel_backend_name() {
        assert_eq!(CompiledKernel::Ptx("".into()).backend_name(), "CUDA (PTX)");
        assert_eq!(
            CompiledKernel::Metal("".into()).backend_name(),
            "Metal (MSL)"
        );
        assert_eq!(
            CompiledKernel::SpirV(vec![]).backend_name(),
            "Vulkan (SPIR-V)"
        );
        assert_eq!(CompiledKernel::OpenCL("".into()).backend_name(), "OpenCL");
    }
}
