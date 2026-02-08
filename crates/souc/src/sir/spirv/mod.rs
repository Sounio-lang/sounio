//! SPIR-V Backend for Sounio Intermediate Representation (SIR)
//!
//! This module provides the sovereign GPU backend for Sounio, targeting SPIR-V as the
//! primary portable GPU representation. SPIR-V (Standard Portable Intermediate Representation)
//! is the cross-platform, multi-vendor standard supported by Vulkan, OpenCL, and OpenGL.
//!
//! # Design Philosophy
//!
//! SPIR-V is chosen as the sovereign backend because:
//! 1. **Portability**: Works across NVIDIA, AMD, Intel, and ARM GPUs
//! 2. **Standardization**: Khronos Group standard with well-defined semantics
//! 3. **Tooling**: Rich ecosystem of validators, optimizers (spirv-opt), and reflection tools
//! 4. **Future-proof**: Core to Vulkan, WebGPU, and emerging compute standards
//!
//! PTX (NVIDIA-specific) is intentionally positioned as an optional plugin for cases
//! where NVIDIA-specific optimizations are required.
//!
//! # Architecture
//!
//! ```text
//! SIR Module
//!     |
//!     v
//! SpirVBackend
//!     |
//!     +-- SpirVModuleBuilder (builds SPIR-V structure)
//!     |       |
//!     |       +-- Capability declarations
//!     |       +-- Extension imports (GLSL.std.450, etc.)
//!     |       +-- Memory model
//!     |       +-- Entry points
//!     |       +-- Type declarations
//!     |       +-- Function definitions
//!     |
//!     +-- SirToSpirV trait (type/op translation)
//!     |
//!     +-- TargetCapabilities (device feature detection)
//!     |
//!     v
//! SPIR-V Binary (Vec<u32>)
//! ```
//!
//! # Execution Model Mapping
//!
//! | SIR Concept | SPIR-V Equivalent |
//! |-------------|-------------------|
//! | Parallel for | Workgroup dispatch |
//! | Shared memory | Workgroup storage class |
//! | Barrier | OpControlBarrier |
//! | Atomic ops | OpAtomicXxx |
//! | Thread ID | GlobalInvocationId |
//!
//! # Epistemic Operations
//!
//! Epistemic values (value + confidence) are lowered to struct constructions
//! and inline computations. The confidence propagation rules are implemented
//! as SPIR-V instruction sequences.
//!
//! # Example Usage
//!
//! ```ignore
//! use sounio_compiler::sir::spirv::{SpirVBackend, TargetCapabilities};
//! use sounio_compiler::sir::SirModule;
//!
//! let module: SirModule = /* ... */;
//! let caps = TargetCapabilities::vulkan_1_2();
//!
//! let backend = SpirVBackend::new(caps);
//! let spirv_binary = backend.compile(&module)?;
//!
//! // spirv_binary can now be loaded by Vulkan, validated with spirv-val, etc.
//! ```

#![allow(unused_imports)]

use std::collections::HashMap;
use std::fmt;

use super::SirModule;
use super::blocks::{BasicBlock, SirFunction, Terminator};
use super::ops::{ArithOp, EpistemicOp, MemoryOp, SirInst};
use super::types::{
    ArrayType, DistributionKind, FunctionType, PointerType, ScalarType, SirType, StructType,
    VectorType,
};
use super::values::{BlockId, Constant, FuncId, ValueId};

// ============================================================================
// SPIR-V Types and Constants
// ============================================================================

/// SPIR-V word (32-bit unsigned integer)
pub type Word = u32;

/// SPIR-V result ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResultId(pub Word);

impl fmt::Display for ResultId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// SPIR-V execution model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionModel {
    /// Vertex shader
    Vertex,
    /// Fragment shader
    Fragment,
    /// Compute shader (GLCompute)
    GLCompute,
    /// Compute kernel (OpenCL)
    Kernel,
}

impl ExecutionModel {
    /// Get SPIR-V opcode value
    pub fn to_spirv(&self) -> Word {
        match self {
            ExecutionModel::Vertex => 0,
            ExecutionModel::Fragment => 4,
            ExecutionModel::GLCompute => 5,
            ExecutionModel::Kernel => 6,
        }
    }
}

/// SPIR-V addressing model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressingModel {
    /// Logical addressing (no pointer arithmetic)
    Logical,
    /// Physical 32-bit pointers
    Physical32,
    /// Physical 64-bit pointers
    Physical64,
    /// Physical storage buffer addressing (Vulkan)
    PhysicalStorageBuffer64,
}

impl AddressingModel {
    pub fn to_spirv(&self) -> Word {
        match self {
            AddressingModel::Logical => 0,
            AddressingModel::Physical32 => 1,
            AddressingModel::Physical64 => 2,
            AddressingModel::PhysicalStorageBuffer64 => 5348,
        }
    }
}

/// SPIR-V memory model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryModel {
    /// Simple memory model
    Simple,
    /// GLSL450 memory model
    GLSL450,
    /// OpenCL memory model
    OpenCL,
    /// Vulkan memory model (strict)
    Vulkan,
}

impl MemoryModel {
    pub fn to_spirv(&self) -> Word {
        match self {
            MemoryModel::Simple => 0,
            MemoryModel::GLSL450 => 1,
            MemoryModel::OpenCL => 2,
            MemoryModel::Vulkan => 3,
        }
    }
}

/// SPIR-V storage class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Universal pointer
    UniformConstant,
    /// Input variable
    Input,
    /// Uniform buffer (UBO)
    Uniform,
    /// Output variable
    Output,
    /// Workgroup shared memory
    Workgroup,
    /// Cross-workgroup global memory
    CrossWorkgroup,
    /// Private per-invocation memory
    Private,
    /// Function-local memory
    Function,
    /// Generic pointer
    Generic,
    /// Push constant
    PushConstant,
    /// Atomic counter
    AtomicCounter,
    /// Image storage
    Image,
    /// Storage buffer (SSBO)
    StorageBuffer,
}

impl StorageClass {
    pub fn to_spirv(&self) -> Word {
        match self {
            StorageClass::UniformConstant => 0,
            StorageClass::Input => 1,
            StorageClass::Uniform => 2,
            StorageClass::Output => 3,
            StorageClass::Workgroup => 4,
            StorageClass::CrossWorkgroup => 5,
            StorageClass::Private => 6,
            StorageClass::Function => 7,
            StorageClass::Generic => 8,
            StorageClass::PushConstant => 9,
            StorageClass::AtomicCounter => 10,
            StorageClass::Image => 11,
            StorageClass::StorageBuffer => 12,
        }
    }
}

/// SPIR-V capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Matrix operations
    Matrix,
    /// Shader capability
    Shader,
    /// Float16 support
    Float16,
    /// Float64 support
    Float64,
    /// Int64 support
    Int64,
    /// Int16 support
    Int16,
    /// Int8 support
    Int8,
    /// Storage buffer 16-bit access
    StorageBuffer16BitAccess,
    /// Uniform and storage buffer 16-bit access
    UniformAndStorageBuffer16BitAccess,
    /// Storage push constant 16-bit access
    StoragePushConstant16,
    /// Storage input/output 16-bit access
    StorageInputOutput16,
    /// Variable pointers in storage buffer
    VariablePointersStorageBuffer,
    /// Variable pointers
    VariablePointers,
    /// Vulkan memory model
    VulkanMemoryModel,
    /// Physical storage buffer addresses
    PhysicalStorageBufferAddresses,
    /// Subgroup ballot
    SubgroupBallotKHR,
    /// Subgroup vote
    SubgroupVoteKHR,
    /// Subgroup shuffle
    SubgroupShuffleINTEL,
    /// Cooperative matrix (Tensor Cores)
    CooperativeMatrixNV,
    /// Atomic float add
    AtomicFloat32AddEXT,
    /// Atomic float64 add
    AtomicFloat64AddEXT,
    /// Ray tracing
    RayTracingKHR,
    /// Mesh shading
    MeshShadingNV,
}

impl Capability {
    pub fn to_spirv(&self) -> Word {
        match self {
            Capability::Matrix => 0,
            Capability::Shader => 1,
            Capability::Float16 => 9,
            Capability::Float64 => 10,
            Capability::Int64 => 11,
            Capability::Int16 => 22,
            Capability::Int8 => 39,
            Capability::StorageBuffer16BitAccess => 4433,
            Capability::UniformAndStorageBuffer16BitAccess => 4434,
            Capability::StoragePushConstant16 => 4435,
            Capability::StorageInputOutput16 => 4436,
            Capability::VariablePointersStorageBuffer => 4441,
            Capability::VariablePointers => 4442,
            Capability::VulkanMemoryModel => 5345,
            Capability::PhysicalStorageBufferAddresses => 5347,
            Capability::SubgroupBallotKHR => 4423,
            Capability::SubgroupVoteKHR => 4431,
            Capability::SubgroupShuffleINTEL => 5568,
            Capability::CooperativeMatrixNV => 5357,
            Capability::AtomicFloat32AddEXT => 6033,
            Capability::AtomicFloat64AddEXT => 6034,
            Capability::RayTracingKHR => 4479,
            Capability::MeshShadingNV => 5266,
        }
    }
}

// ============================================================================
// Target Capabilities
// ============================================================================

/// GPU target capabilities for code generation decisions
#[derive(Debug, Clone)]
pub struct TargetCapabilities {
    /// Target name (for diagnostics)
    pub name: String,

    /// SPIR-V version (major, minor)
    pub spirv_version: (u32, u32),

    /// Execution model
    pub execution_model: ExecutionModel,

    /// Available capabilities
    pub capabilities: Vec<Capability>,

    /// Maximum workgroup size (x, y, z)
    pub max_workgroup_size: (u32, u32, u32),

    /// Maximum invocations per workgroup
    pub max_workgroup_invocations: u32,

    /// Maximum shared memory (bytes)
    pub max_shared_memory: u32,

    /// Subgroup size (warp/wavefront size)
    pub subgroup_size: Option<u32>,

    /// Supports subgroup operations
    pub subgroup_ops: bool,

    /// Supports cooperative matrix (tensor cores)
    pub cooperative_matrix: bool,

    /// Supports atomic float add
    pub atomic_float_add: bool,

    /// Supports 16-bit types
    pub supports_16bit: bool,

    /// Supports 8-bit types
    pub supports_8bit: bool,

    /// Supports 64-bit floats
    pub supports_f64: bool,

    /// Supports 64-bit integers
    pub supports_i64: bool,
}

impl TargetCapabilities {
    /// Create Vulkan 1.0 compute target (baseline)
    pub fn vulkan_1_0() -> Self {
        Self {
            name: "Vulkan 1.0".into(),
            spirv_version: (1, 0),
            execution_model: ExecutionModel::GLCompute,
            capabilities: vec![Capability::Shader],
            max_workgroup_size: (128, 128, 64),
            max_workgroup_invocations: 128,
            max_shared_memory: 16384,
            subgroup_size: None,
            subgroup_ops: false,
            cooperative_matrix: false,
            atomic_float_add: false,
            supports_16bit: false,
            supports_8bit: false,
            supports_f64: false,
            supports_i64: false,
        }
    }

    /// Create Vulkan 1.1 compute target (subgroups)
    pub fn vulkan_1_1() -> Self {
        Self {
            name: "Vulkan 1.1".into(),
            spirv_version: (1, 3),
            execution_model: ExecutionModel::GLCompute,
            capabilities: vec![
                Capability::Shader,
                Capability::SubgroupBallotKHR,
                Capability::SubgroupVoteKHR,
            ],
            max_workgroup_size: (1024, 1024, 64),
            max_workgroup_invocations: 1024,
            max_shared_memory: 32768,
            subgroup_size: Some(32), // Common on NVIDIA/AMD
            subgroup_ops: true,
            cooperative_matrix: false,
            atomic_float_add: false,
            supports_16bit: false,
            supports_8bit: false,
            supports_f64: false,
            supports_i64: true,
        }
    }

    /// Create Vulkan 1.2 compute target (recommended baseline)
    pub fn vulkan_1_2() -> Self {
        Self {
            name: "Vulkan 1.2".into(),
            spirv_version: (1, 5),
            execution_model: ExecutionModel::GLCompute,
            capabilities: vec![
                Capability::Shader,
                Capability::Float64,
                Capability::Int64,
                Capability::Int16,
                Capability::StorageBuffer16BitAccess,
                Capability::SubgroupBallotKHR,
                Capability::SubgroupVoteKHR,
                Capability::VulkanMemoryModel,
            ],
            max_workgroup_size: (1024, 1024, 64),
            max_workgroup_invocations: 1024,
            max_shared_memory: 49152, // 48KB typical
            subgroup_size: Some(32),
            subgroup_ops: true,
            cooperative_matrix: false,
            atomic_float_add: false,
            supports_16bit: true,
            supports_8bit: false,
            supports_f64: true,
            supports_i64: true,
        }
    }

    /// Create Vulkan 1.3 compute target (modern)
    pub fn vulkan_1_3() -> Self {
        Self {
            name: "Vulkan 1.3".into(),
            spirv_version: (1, 6),
            execution_model: ExecutionModel::GLCompute,
            capabilities: vec![
                Capability::Shader,
                Capability::Float64,
                Capability::Int64,
                Capability::Int16,
                Capability::Int8,
                Capability::StorageBuffer16BitAccess,
                Capability::SubgroupBallotKHR,
                Capability::SubgroupVoteKHR,
                Capability::VulkanMemoryModel,
            ],
            max_workgroup_size: (1024, 1024, 64),
            max_workgroup_invocations: 1024,
            max_shared_memory: 65536, // 64KB
            subgroup_size: Some(32),
            subgroup_ops: true,
            cooperative_matrix: false,
            atomic_float_add: true,
            supports_16bit: true,
            supports_8bit: true,
            supports_f64: true,
            supports_i64: true,
        }
    }

    /// Create NVIDIA-optimized target (tensor cores, etc.)
    pub fn nvidia_ampere() -> Self {
        Self {
            name: "NVIDIA Ampere".into(),
            spirv_version: (1, 5),
            execution_model: ExecutionModel::GLCompute,
            capabilities: vec![
                Capability::Shader,
                Capability::Float64,
                Capability::Int64,
                Capability::Int16,
                Capability::Int8,
                Capability::SubgroupBallotKHR,
                Capability::SubgroupVoteKHR,
                Capability::CooperativeMatrixNV,
                Capability::AtomicFloat32AddEXT,
                Capability::AtomicFloat64AddEXT,
                Capability::VulkanMemoryModel,
            ],
            max_workgroup_size: (1024, 1024, 64),
            max_workgroup_invocations: 1024,
            max_shared_memory: 163840, // 160KB (A100)
            subgroup_size: Some(32),
            subgroup_ops: true,
            cooperative_matrix: true,
            atomic_float_add: true,
            supports_16bit: true,
            supports_8bit: true,
            supports_f64: true,
            supports_i64: true,
        }
    }

    /// Create AMD RDNA2 target
    pub fn amd_rdna2() -> Self {
        Self {
            name: "AMD RDNA2".into(),
            spirv_version: (1, 5),
            execution_model: ExecutionModel::GLCompute,
            capabilities: vec![
                Capability::Shader,
                Capability::Float64,
                Capability::Int64,
                Capability::Int16,
                Capability::SubgroupBallotKHR,
                Capability::SubgroupVoteKHR,
                Capability::VulkanMemoryModel,
            ],
            max_workgroup_size: (1024, 1024, 1024),
            max_workgroup_invocations: 1024,
            max_shared_memory: 65536, // 64KB
            subgroup_size: Some(64),  // AMD uses wave64
            subgroup_ops: true,
            cooperative_matrix: false,
            atomic_float_add: false,
            supports_16bit: true,
            supports_8bit: false,
            supports_f64: true,
            supports_i64: true,
        }
    }

    /// Create OpenCL 2.0 target
    pub fn opencl_2_0() -> Self {
        Self {
            name: "OpenCL 2.0".into(),
            spirv_version: (1, 2),
            execution_model: ExecutionModel::Kernel,
            capabilities: vec![Capability::Float64, Capability::Int64],
            max_workgroup_size: (1024, 1024, 1024),
            max_workgroup_invocations: 1024,
            max_shared_memory: 49152,
            subgroup_size: None,
            subgroup_ops: false,
            cooperative_matrix: false,
            atomic_float_add: false,
            supports_16bit: false,
            supports_8bit: false,
            supports_f64: true,
            supports_i64: true,
        }
    }

    /// Check if a capability is available
    pub fn has(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// Add a capability
    pub fn add_capability(&mut self, cap: Capability) {
        if !self.has(cap) {
            self.capabilities.push(cap);
        }
    }
}

// ============================================================================
// Reduction Strategies
// ============================================================================

/// Strategy for parallel reductions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionStrategy {
    /// Use subgroup shuffle operations (fastest, requires subgroup support)
    SubgroupShuffle,
    /// Use shared memory with tree reduction
    SharedMemory,
    /// Use atomic operations (slowest, but always works)
    Atomic,
}

// ============================================================================
// SPIR-V Types
// ============================================================================

/// SPIR-V type representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpirVType {
    /// Void type
    Void,
    /// Boolean
    Bool,
    /// Integer (width, signedness)
    Int { width: u32, signed: bool },
    /// Floating point (width)
    Float { width: u32 },
    /// Vector (element type, count)
    Vector { elem: Box<SpirVType>, count: u32 },
    /// Matrix (column type, column count)
    Matrix { col: Box<SpirVType>, cols: u32 },
    /// Array (element type, length)
    Array { elem: Box<SpirVType>, len: u32 },
    /// Runtime array (element type)
    RuntimeArray { elem: Box<SpirVType> },
    /// Struct (member types)
    Struct { members: Vec<SpirVType> },
    /// Pointer (pointee type, storage class)
    Pointer {
        pointee: Box<SpirVType>,
        storage_class: StorageClass,
    },
    /// Function (return type, parameter types)
    Function {
        ret: Box<SpirVType>,
        params: Vec<SpirVType>,
    },
    /// Image type
    Image,
    /// Sampler type
    Sampler,
    /// Sampled image type
    SampledImage,
}

impl SpirVType {
    /// Create an i32 type
    pub fn i32() -> Self {
        SpirVType::Int {
            width: 32,
            signed: true,
        }
    }

    /// Create a u32 type
    pub fn u32() -> Self {
        SpirVType::Int {
            width: 32,
            signed: false,
        }
    }

    /// Create an f32 type
    pub fn f32() -> Self {
        SpirVType::Float { width: 32 }
    }

    /// Create an f64 type
    pub fn f64() -> Self {
        SpirVType::Float { width: 64 }
    }

    /// Create a vec3<u32> type (common for thread IDs)
    pub fn vec3_u32() -> Self {
        SpirVType::Vector {
            elem: Box::new(SpirVType::u32()),
            count: 3,
        }
    }

    /// Create a vec4<f32> type
    pub fn vec4_f32() -> Self {
        SpirVType::Vector {
            elem: Box::new(SpirVType::f32()),
            count: 4,
        }
    }
}

// ============================================================================
// SPIR-V Instructions
// ============================================================================

/// SPIR-V instruction representation
#[derive(Debug, Clone)]
pub enum SpirVInst {
    // === Module Structure ===
    /// OpCapability
    Capability(Capability),
    /// OpExtension
    Extension(String),
    /// OpExtInstImport
    ExtInstImport { result: ResultId, name: String },
    /// OpMemoryModel
    MemoryModel {
        addressing: AddressingModel,
        memory: MemoryModel,
    },
    /// OpEntryPoint
    EntryPoint {
        execution_model: ExecutionModel,
        entry_point: ResultId,
        name: String,
        interface: Vec<ResultId>,
    },
    /// OpExecutionMode
    ExecutionMode {
        entry_point: ResultId,
        mode: ExecutionModeKind,
    },

    // === Type Declarations ===
    /// OpTypeVoid
    TypeVoid { result: ResultId },
    /// OpTypeBool
    TypeBool { result: ResultId },
    /// OpTypeInt
    TypeInt {
        result: ResultId,
        width: u32,
        signed: bool,
    },
    /// OpTypeFloat
    TypeFloat { result: ResultId, width: u32 },
    /// OpTypeVector
    TypeVector {
        result: ResultId,
        component: ResultId,
        count: u32,
    },
    /// OpTypeArray
    TypeArray {
        result: ResultId,
        element: ResultId,
        length: ResultId,
    },
    /// OpTypeRuntimeArray
    TypeRuntimeArray { result: ResultId, element: ResultId },
    /// OpTypeStruct
    TypeStruct {
        result: ResultId,
        members: Vec<ResultId>,
    },
    /// OpTypePointer
    TypePointer {
        result: ResultId,
        storage_class: StorageClass,
        pointee: ResultId,
    },
    /// OpTypeFunction
    TypeFunction {
        result: ResultId,
        return_type: ResultId,
        params: Vec<ResultId>,
    },

    // === Constants ===
    /// OpConstant
    Constant {
        result: ResultId,
        ty: ResultId,
        value: Vec<Word>,
    },
    /// OpConstantTrue
    ConstantTrue { result: ResultId, ty: ResultId },
    /// OpConstantFalse
    ConstantFalse { result: ResultId, ty: ResultId },
    /// OpConstantComposite
    ConstantComposite {
        result: ResultId,
        ty: ResultId,
        constituents: Vec<ResultId>,
    },
    /// OpConstantNull
    ConstantNull { result: ResultId, ty: ResultId },

    // === Variables ===
    /// OpVariable
    Variable {
        result: ResultId,
        ty: ResultId,
        storage_class: StorageClass,
        initializer: Option<ResultId>,
    },

    // === Decorations ===
    /// OpDecorate
    Decorate {
        target: ResultId,
        decoration: Decoration,
    },
    /// OpMemberDecorate
    MemberDecorate {
        struct_type: ResultId,
        member: u32,
        decoration: Decoration,
    },

    // === Functions ===
    /// OpFunction
    Function {
        result: ResultId,
        result_type: ResultId,
        control: FunctionControl,
        function_type: ResultId,
    },
    /// OpFunctionParameter
    FunctionParameter { result: ResultId, ty: ResultId },
    /// OpFunctionEnd
    FunctionEnd,
    /// OpLabel
    Label { result: ResultId },

    // === Memory Operations ===
    /// OpLoad
    Load {
        result: ResultId,
        ty: ResultId,
        pointer: ResultId,
        memory_operands: Option<MemoryOperands>,
    },
    /// OpStore
    Store {
        pointer: ResultId,
        object: ResultId,
        memory_operands: Option<MemoryOperands>,
    },
    /// OpAccessChain
    AccessChain {
        result: ResultId,
        ty: ResultId,
        base: ResultId,
        indexes: Vec<ResultId>,
    },

    // === Arithmetic Operations ===
    /// OpIAdd
    IAdd {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpISub
    ISub {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpIMul
    IMul {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpSDiv
    SDiv {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpUDiv
    UDiv {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpSRem
    SRem {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpURem
    URem {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpSNegate
    SNegate {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },

    // === Floating Point Operations ===
    /// OpFAdd
    FAdd {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpFSub
    FSub {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpFMul
    FMul {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpFDiv
    FDiv {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpFRem
    FRem {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpFNegate
    FNegate {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },

    // === Logical Operations ===
    /// OpLogicalEqual
    LogicalEqual {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpLogicalNotEqual
    LogicalNotEqual {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpLogicalAnd
    LogicalAnd {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpLogicalOr
    LogicalOr {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpLogicalNot
    LogicalNot {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },

    // === Comparison Operations ===
    /// OpIEqual
    IEqual {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpINotEqual
    INotEqual {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpSLessThan
    SLessThan {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpSLessThanEqual
    SLessThanEqual {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpSGreaterThan
    SGreaterThan {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpSGreaterThanEqual
    SGreaterThanEqual {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpFOrdEqual
    FOrdEqual {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpFOrdLessThan
    FOrdLessThan {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },

    // === Bitwise Operations ===
    /// OpBitwiseAnd
    BitwiseAnd {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpBitwiseOr
    BitwiseOr {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpBitwiseXor
    BitwiseXor {
        result: ResultId,
        ty: ResultId,
        a: ResultId,
        b: ResultId,
    },
    /// OpNot
    Not {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpShiftLeftLogical
    ShiftLeftLogical {
        result: ResultId,
        ty: ResultId,
        base: ResultId,
        shift: ResultId,
    },
    /// OpShiftRightLogical
    ShiftRightLogical {
        result: ResultId,
        ty: ResultId,
        base: ResultId,
        shift: ResultId,
    },
    /// OpShiftRightArithmetic
    ShiftRightArithmetic {
        result: ResultId,
        ty: ResultId,
        base: ResultId,
        shift: ResultId,
    },

    // === Composite Operations ===
    /// OpCompositeConstruct
    CompositeConstruct {
        result: ResultId,
        ty: ResultId,
        constituents: Vec<ResultId>,
    },
    /// OpCompositeExtract
    CompositeExtract {
        result: ResultId,
        ty: ResultId,
        composite: ResultId,
        indexes: Vec<u32>,
    },
    /// OpCompositeInsert
    CompositeInsert {
        result: ResultId,
        ty: ResultId,
        object: ResultId,
        composite: ResultId,
        indexes: Vec<u32>,
    },

    // === Control Flow ===
    /// OpBranch
    Branch { target: ResultId },
    /// OpBranchConditional
    BranchConditional {
        condition: ResultId,
        true_label: ResultId,
        false_label: ResultId,
        weights: Option<(u32, u32)>,
    },
    /// OpSwitch
    Switch {
        selector: ResultId,
        default: ResultId,
        targets: Vec<(Word, ResultId)>,
    },
    /// OpReturn
    Return,
    /// OpReturnValue
    ReturnValue { value: ResultId },
    /// OpKill (fragment shader discard)
    Kill,
    /// OpUnreachable
    Unreachable,

    // === Phi and Select ===
    /// OpPhi
    Phi {
        result: ResultId,
        ty: ResultId,
        operands: Vec<(ResultId, ResultId)>, // (value, parent_block)
    },
    /// OpSelect
    Select {
        result: ResultId,
        ty: ResultId,
        condition: ResultId,
        a: ResultId,
        b: ResultId,
    },

    // === Atomic Operations ===
    /// OpAtomicLoad
    AtomicLoad {
        result: ResultId,
        ty: ResultId,
        pointer: ResultId,
        scope: Scope,
        semantics: MemorySemantics,
    },
    /// OpAtomicStore
    AtomicStore {
        pointer: ResultId,
        scope: Scope,
        semantics: MemorySemantics,
        value: ResultId,
    },
    /// OpAtomicIAdd
    AtomicIAdd {
        result: ResultId,
        ty: ResultId,
        pointer: ResultId,
        scope: Scope,
        semantics: MemorySemantics,
        value: ResultId,
    },
    /// OpAtomicSMin
    AtomicSMin {
        result: ResultId,
        ty: ResultId,
        pointer: ResultId,
        scope: Scope,
        semantics: MemorySemantics,
        value: ResultId,
    },
    /// OpAtomicSMax
    AtomicSMax {
        result: ResultId,
        ty: ResultId,
        pointer: ResultId,
        scope: Scope,
        semantics: MemorySemantics,
        value: ResultId,
    },
    /// OpAtomicExchange
    AtomicExchange {
        result: ResultId,
        ty: ResultId,
        pointer: ResultId,
        scope: Scope,
        semantics: MemorySemantics,
        value: ResultId,
    },
    /// OpAtomicCompareExchange
    AtomicCompareExchange {
        result: ResultId,
        ty: ResultId,
        pointer: ResultId,
        scope: Scope,
        equal_semantics: MemorySemantics,
        unequal_semantics: MemorySemantics,
        value: ResultId,
        comparator: ResultId,
    },

    // === Barrier Operations ===
    /// OpControlBarrier
    ControlBarrier {
        execution_scope: Scope,
        memory_scope: Scope,
        semantics: MemorySemantics,
    },
    /// OpMemoryBarrier
    MemoryBarrier {
        scope: Scope,
        semantics: MemorySemantics,
    },

    // === Conversion Operations ===
    /// OpConvertFToS
    ConvertFToS {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpConvertFToU
    ConvertFToU {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpConvertSToF
    ConvertSToF {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpConvertUToF
    ConvertUToF {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpFConvert
    FConvert {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpSConvert
    SConvert {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpUConvert
    UConvert {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },
    /// OpBitcast
    Bitcast {
        result: ResultId,
        ty: ResultId,
        operand: ResultId,
    },

    // === Extended Instructions ===
    /// OpExtInst (GLSL.std.450)
    ExtInst {
        result: ResultId,
        ty: ResultId,
        set: ResultId,
        instruction: u32,
        operands: Vec<ResultId>,
    },

    // === Function Call ===
    /// OpFunctionCall
    FunctionCall {
        result: ResultId,
        ty: ResultId,
        function: ResultId,
        arguments: Vec<ResultId>,
    },

    // === Debug ===
    /// OpName
    Name { target: ResultId, name: String },
    /// OpMemberName
    MemberName {
        ty: ResultId,
        member: u32,
        name: String,
    },
}

/// Execution mode kinds
#[derive(Debug, Clone)]
pub enum ExecutionModeKind {
    /// Local workgroup size
    LocalSize { x: u32, y: u32, z: u32 },
    /// Origin upper left (fragment)
    OriginUpperLeft,
    /// Origin lower left (fragment)
    OriginLowerLeft,
    /// Early fragment tests
    EarlyFragmentTests,
    /// Depth replacing
    DepthReplacing,
}

/// Decoration kinds
#[derive(Debug, Clone)]
pub enum Decoration {
    /// BuiltIn decoration
    BuiltIn(BuiltIn),
    /// Location decoration
    Location(u32),
    /// Binding decoration
    Binding(u32),
    /// DescriptorSet decoration
    DescriptorSet(u32),
    /// Offset decoration
    Offset(u32),
    /// ArrayStride decoration
    ArrayStride(u32),
    /// MatrixStride decoration
    MatrixStride(u32),
    /// NonWritable decoration
    NonWritable,
    /// NonReadable decoration
    NonReadable,
    /// Restrict decoration
    Restrict,
    /// Aliased decoration
    Aliased,
    /// Block decoration
    Block,
    /// BufferBlock decoration (deprecated in 1.3)
    BufferBlock,
}

/// Built-in variables
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltIn {
    Position,
    PointSize,
    VertexIndex,
    InstanceIndex,
    FragCoord,
    FragDepth,
    WorkgroupSize,
    WorkgroupId,
    LocalInvocationId,
    GlobalInvocationId,
    LocalInvocationIndex,
    NumWorkgroups,
    SubgroupSize,
    SubgroupLocalInvocationId,
}

impl BuiltIn {
    pub fn to_spirv(&self) -> Word {
        match self {
            BuiltIn::Position => 0,
            BuiltIn::PointSize => 1,
            BuiltIn::VertexIndex => 42,
            BuiltIn::InstanceIndex => 43,
            BuiltIn::FragCoord => 15,
            BuiltIn::FragDepth => 22,
            BuiltIn::WorkgroupSize => 25,
            BuiltIn::WorkgroupId => 26,
            BuiltIn::LocalInvocationId => 27,
            BuiltIn::GlobalInvocationId => 28,
            BuiltIn::LocalInvocationIndex => 29,
            BuiltIn::NumWorkgroups => 24,
            BuiltIn::SubgroupSize => 36,
            BuiltIn::SubgroupLocalInvocationId => 41,
        }
    }
}

/// Function control flags
#[derive(Debug, Clone, Copy, Default)]
pub struct FunctionControl {
    pub inline: bool,
    pub dont_inline: bool,
    pub pure_: bool,
    pub const_: bool,
}

/// Memory operands
#[derive(Debug, Clone, Default)]
pub struct MemoryOperands {
    pub volatile: bool,
    pub aligned: Option<u32>,
    pub nontemporal: bool,
}

impl ExecutionModeKind {
    /// Convert to SPIR-V value and extra operands
    pub fn to_spirv(&self) -> (u32, Vec<u32>) {
        match self {
            ExecutionModeKind::LocalSize { x, y, z } => (17, vec![*x, *y, *z]),
            ExecutionModeKind::OriginUpperLeft => (7, vec![]),
            ExecutionModeKind::OriginLowerLeft => (8, vec![]),
            ExecutionModeKind::EarlyFragmentTests => (35, vec![]),
            ExecutionModeKind::DepthReplacing => (12, vec![]),
        }
    }
}

impl Decoration {
    /// Convert to SPIR-V decoration value and extra operands
    pub fn to_spirv(&self) -> (u32, Vec<u32>) {
        match self {
            Decoration::BuiltIn(b) => (11, vec![b.to_spirv()]),
            Decoration::Location(l) => (30, vec![*l]),
            Decoration::Binding(b) => (33, vec![*b]),
            Decoration::DescriptorSet(d) => (34, vec![*d]),
            Decoration::Offset(o) => (35, vec![*o]),
            Decoration::ArrayStride(s) => (6, vec![*s]),
            Decoration::MatrixStride(s) => (7, vec![*s]),
            Decoration::NonWritable => (25, vec![]),
            Decoration::NonReadable => (24, vec![]),
            Decoration::Restrict => (19, vec![]),
            Decoration::Aliased => (20, vec![]),
            Decoration::Block => (2, vec![]),
            Decoration::BufferBlock => (3, vec![]),
        }
    }
}

impl FunctionControl {
    /// Convert to SPIR-V bitmask
    pub fn to_spirv(&self) -> u32 {
        let mut mask = 0u32;
        if self.inline {
            mask |= 1;
        }
        if self.dont_inline {
            mask |= 2;
        }
        if self.pure_ {
            mask |= 4;
        }
        if self.const_ {
            mask |= 8;
        }
        mask
    }
}

impl MemoryOperands {
    /// Convert to SPIR-V bitmask
    pub fn to_spirv(&self) -> u32 {
        let mut mask = 0u32;
        if self.volatile {
            mask |= 1;
        }
        if self.aligned.is_some() {
            mask |= 2;
        }
        if self.nontemporal {
            mask |= 4;
        }
        mask
    }
}

/// Scope for atomic/barrier operations
#[derive(Debug, Clone, Copy)]
pub enum Scope {
    CrossDevice,
    Device,
    Workgroup,
    Subgroup,
    Invocation,
}

impl Scope {
    pub fn to_spirv(&self) -> Word {
        match self {
            Scope::CrossDevice => 0,
            Scope::Device => 1,
            Scope::Workgroup => 2,
            Scope::Subgroup => 3,
            Scope::Invocation => 4,
        }
    }
}

/// Memory semantics flags
#[derive(Debug, Clone, Copy, Default)]
pub struct MemorySemantics {
    pub acquire: bool,
    pub release: bool,
    pub acquire_release: bool,
    pub sequentially_consistent: bool,
    pub uniform_memory: bool,
    pub subgroup_memory: bool,
    pub workgroup_memory: bool,
    pub cross_workgroup_memory: bool,
    pub atomic_counter_memory: bool,
    pub image_memory: bool,
}

impl MemorySemantics {
    /// Workgroup acquire-release semantics (typical for barriers)
    pub fn workgroup_acq_rel() -> Self {
        Self {
            acquire_release: true,
            workgroup_memory: true,
            ..Default::default()
        }
    }

    /// Convert to SPIR-V word
    pub fn to_spirv(&self) -> Word {
        let mut bits = 0u32;
        if self.acquire {
            bits |= 0x2;
        }
        if self.release {
            bits |= 0x4;
        }
        if self.acquire_release {
            bits |= 0x8;
        }
        if self.sequentially_consistent {
            bits |= 0x10;
        }
        if self.uniform_memory {
            bits |= 0x40;
        }
        if self.subgroup_memory {
            bits |= 0x80;
        }
        if self.workgroup_memory {
            bits |= 0x100;
        }
        if self.cross_workgroup_memory {
            bits |= 0x200;
        }
        if self.atomic_counter_memory {
            bits |= 0x400;
        }
        if self.image_memory {
            bits |= 0x800;
        }
        bits
    }
}

// ============================================================================
// SPIR-V Function
// ============================================================================

/// A SPIR-V function
#[derive(Debug, Clone)]
pub struct SpirVFunction {
    /// Function ID
    pub id: ResultId,
    /// Function name
    pub name: String,
    /// Return type
    pub return_type: ResultId,
    /// Function type
    pub function_type: ResultId,
    /// Parameters
    pub parameters: Vec<ResultId>,
    /// Basic blocks
    pub blocks: Vec<SpirVBlock>,
    /// Is this an entry point?
    pub is_entry_point: bool,
    /// Local workgroup size (for compute)
    pub local_size: Option<(u32, u32, u32)>,
}

/// A SPIR-V basic block
#[derive(Debug, Clone)]
pub struct SpirVBlock {
    /// Block label ID
    pub label: ResultId,
    /// Instructions
    pub instructions: Vec<SpirVInst>,
}

// ============================================================================
// Entry Points
// ============================================================================

/// An entry point definition
#[derive(Debug, Clone)]
pub struct EntryPoint {
    /// Entry point function ID
    pub function: ResultId,
    /// Entry point name
    pub name: String,
    /// Execution model
    pub execution_model: ExecutionModel,
    /// Interface variables
    pub interface: Vec<ResultId>,
    /// Local workgroup size (for compute)
    pub local_size: Option<(u32, u32, u32)>,
}

// ============================================================================
// SPIR-V Module Builder
// ============================================================================

/// Builds SPIR-V module structure
#[derive(Debug)]
pub struct SpirVModuleBuilder {
    /// Next available ID
    next_id: Word,
    /// Required capabilities
    capabilities: Vec<Capability>,
    /// Required extensions
    extensions: Vec<String>,
    /// Extended instruction sets
    ext_inst_imports: HashMap<String, ResultId>,
    /// Addressing model
    addressing_model: AddressingModel,
    /// Memory model
    memory_model: MemoryModel,
    /// Entry points
    entry_points: Vec<EntryPoint>,
    /// Execution modes
    execution_modes: Vec<(ResultId, ExecutionModeKind)>,
    /// Debug names
    names: Vec<(ResultId, String)>,
    /// Decorations
    decorations: Vec<SpirVInst>,
    /// Type declarations
    types: Vec<SpirVInst>,
    /// Type cache (SpirVType -> ResultId)
    type_cache: HashMap<SpirVType, ResultId>,
    /// Constants
    constants: Vec<SpirVInst>,
    /// Constant cache
    constant_cache: HashMap<(ResultId, Vec<Word>), ResultId>,
    /// Global variables
    variables: Vec<SpirVInst>,
    /// Functions
    functions: Vec<SpirVFunction>,
}

impl SpirVModuleBuilder {
    /// Create a new module builder
    pub fn new() -> Self {
        Self {
            next_id: 1, // ID 0 is reserved
            capabilities: Vec::new(),
            extensions: Vec::new(),
            ext_inst_imports: HashMap::new(),
            addressing_model: AddressingModel::Logical,
            memory_model: MemoryModel::GLSL450,
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            names: Vec::new(),
            decorations: Vec::new(),
            types: Vec::new(),
            type_cache: HashMap::new(),
            constants: Vec::new(),
            constant_cache: HashMap::new(),
            variables: Vec::new(),
            functions: Vec::new(),
        }
    }

    /// Allocate a new result ID
    pub fn alloc_id(&mut self) -> ResultId {
        let id = ResultId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a capability
    pub fn require_capability(&mut self, cap: Capability) {
        if !self.capabilities.contains(&cap) {
            self.capabilities.push(cap);
        }
    }

    /// Add an extension
    pub fn require_extension(&mut self, ext: &str) {
        if !self.extensions.iter().any(|e| e == ext) {
            self.extensions.push(ext.to_string());
        }
    }

    /// Import an extended instruction set (e.g., "GLSL.std.450")
    pub fn import_ext_inst_set(&mut self, name: &str) -> ResultId {
        if let Some(&id) = self.ext_inst_imports.get(name) {
            return id;
        }
        let id = self.alloc_id();
        self.ext_inst_imports.insert(name.to_string(), id);
        id
    }

    /// Set the memory model
    pub fn set_memory_model(&mut self, addressing: AddressingModel, memory: MemoryModel) {
        self.addressing_model = addressing;
        self.memory_model = memory;
    }

    /// Add an entry point
    pub fn add_entry_point(&mut self, entry: EntryPoint) {
        self.entry_points.push(entry);
    }

    /// Add a debug name
    pub fn add_name(&mut self, target: ResultId, name: &str) {
        self.names.push((target, name.to_string()));
    }

    /// Get or create a type
    pub fn get_or_create_type(&mut self, ty: &SpirVType) -> ResultId {
        if let Some(&id) = self.type_cache.get(ty) {
            return id;
        }

        let id = self.alloc_id();
        let inst = self.create_type_inst(id, ty);
        self.types.push(inst);
        self.type_cache.insert(ty.clone(), id);
        id
    }

    fn create_type_inst(&mut self, id: ResultId, ty: &SpirVType) -> SpirVInst {
        match ty {
            SpirVType::Void => SpirVInst::TypeVoid { result: id },
            SpirVType::Bool => SpirVInst::TypeBool { result: id },
            SpirVType::Int { width, signed } => SpirVInst::TypeInt {
                result: id,
                width: *width,
                signed: *signed,
            },
            SpirVType::Float { width } => SpirVInst::TypeFloat {
                result: id,
                width: *width,
            },
            SpirVType::Vector { elem, count } => {
                let elem_id = self.get_or_create_type(elem);
                SpirVInst::TypeVector {
                    result: id,
                    component: elem_id,
                    count: *count,
                }
            }
            SpirVType::Matrix { col, cols } => {
                let col_id = self.get_or_create_type(col);
                SpirVInst::TypeVector {
                    result: id,
                    component: col_id,
                    count: *cols,
                }
            }
            SpirVType::Array { elem, len } => {
                let elem_id = self.get_or_create_type(elem);
                // Need a constant for array length
                let len_ty = self.get_or_create_type(&SpirVType::u32());
                let len_id = self.create_constant_u32(*len);
                SpirVInst::TypeArray {
                    result: id,
                    element: elem_id,
                    length: len_id,
                }
            }
            SpirVType::RuntimeArray { elem } => {
                let elem_id = self.get_or_create_type(elem);
                SpirVInst::TypeRuntimeArray {
                    result: id,
                    element: elem_id,
                }
            }
            SpirVType::Struct { members } => {
                let member_ids: Vec<ResultId> =
                    members.iter().map(|m| self.get_or_create_type(m)).collect();
                SpirVInst::TypeStruct {
                    result: id,
                    members: member_ids,
                }
            }
            SpirVType::Pointer {
                pointee,
                storage_class,
            } => {
                let pointee_id = self.get_or_create_type(pointee);
                SpirVInst::TypePointer {
                    result: id,
                    storage_class: *storage_class,
                    pointee: pointee_id,
                }
            }
            SpirVType::Function { ret, params } => {
                let ret_id = self.get_or_create_type(ret);
                let param_ids: Vec<ResultId> =
                    params.iter().map(|p| self.get_or_create_type(p)).collect();
                SpirVInst::TypeFunction {
                    result: id,
                    return_type: ret_id,
                    params: param_ids,
                }
            }
            SpirVType::Image | SpirVType::Sampler | SpirVType::SampledImage => {
                // Stub for now
                SpirVInst::TypeVoid { result: id }
            }
        }
    }

    /// Create a u32 constant
    pub fn create_constant_u32(&mut self, value: u32) -> ResultId {
        let ty_id = self.get_or_create_type(&SpirVType::u32());
        let key = (ty_id, vec![value]);
        if let Some(&id) = self.constant_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        self.constants.push(SpirVInst::Constant {
            result: id,
            ty: ty_id,
            value: vec![value],
        });
        self.constant_cache.insert(key, id);
        id
    }

    /// Create an i32 constant
    pub fn create_constant_i32(&mut self, value: i32) -> ResultId {
        let ty_id = self.get_or_create_type(&SpirVType::i32());
        let bits = value as u32;
        let key = (ty_id, vec![bits]);
        if let Some(&id) = self.constant_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        self.constants.push(SpirVInst::Constant {
            result: id,
            ty: ty_id,
            value: vec![bits],
        });
        self.constant_cache.insert(key, id);
        id
    }

    /// Create an f32 constant
    pub fn create_constant_f32(&mut self, value: f32) -> ResultId {
        let ty_id = self.get_or_create_type(&SpirVType::f32());
        let bits = value.to_bits();
        let key = (ty_id, vec![bits]);
        if let Some(&id) = self.constant_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        self.constants.push(SpirVInst::Constant {
            result: id,
            ty: ty_id,
            value: vec![bits],
        });
        self.constant_cache.insert(key, id);
        id
    }

    /// Create an f64 constant
    pub fn create_constant_f64(&mut self, value: f64) -> ResultId {
        let ty_id = self.get_or_create_type(&SpirVType::f64());
        let bits = value.to_bits();
        let lo = bits as u32;
        let hi = (bits >> 32) as u32;
        let key = (ty_id, vec![lo, hi]);
        if let Some(&id) = self.constant_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        self.constants.push(SpirVInst::Constant {
            result: id,
            ty: ty_id,
            value: vec![lo, hi],
        });
        self.constant_cache.insert(key, id);
        id
    }

    /// Add a global variable
    pub fn add_variable(
        &mut self,
        ty: ResultId,
        storage_class: StorageClass,
        initializer: Option<ResultId>,
    ) -> ResultId {
        let id = self.alloc_id();
        self.variables.push(SpirVInst::Variable {
            result: id,
            ty,
            storage_class,
            initializer,
        });
        id
    }

    /// Add a decoration
    pub fn add_decoration(&mut self, target: ResultId, decoration: Decoration) {
        self.decorations
            .push(SpirVInst::Decorate { target, decoration });
    }

    /// Add a function
    pub fn add_function(&mut self, func: SpirVFunction) {
        self.functions.push(func);
    }

    /// Get the bound (highest ID + 1)
    pub fn get_bound(&self) -> Word {
        self.next_id
    }

    /// Build the final module
    pub fn build(self) -> SpirVModule {
        SpirVModule {
            capabilities: self.capabilities,
            extensions: self.extensions,
            ext_inst_imports: self.ext_inst_imports,
            addressing_model: self.addressing_model,
            memory_model: self.memory_model,
            entry_points: self.entry_points,
            execution_modes: self.execution_modes,
            names: self.names,
            decorations: self.decorations,
            types: self.types,
            constants: self.constants,
            variables: self.variables,
            functions: self.functions,
            bound: self.next_id,
        }
    }
}

impl Default for SpirVModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SPIR-V Module
// ============================================================================

/// A complete SPIR-V module
#[derive(Debug)]
pub struct SpirVModule {
    pub capabilities: Vec<Capability>,
    pub extensions: Vec<String>,
    pub ext_inst_imports: HashMap<String, ResultId>,
    pub addressing_model: AddressingModel,
    pub memory_model: MemoryModel,
    pub entry_points: Vec<EntryPoint>,
    pub execution_modes: Vec<(ResultId, ExecutionModeKind)>,
    pub names: Vec<(ResultId, String)>,
    pub decorations: Vec<SpirVInst>,
    pub types: Vec<SpirVInst>,
    pub constants: Vec<SpirVInst>,
    pub variables: Vec<SpirVInst>,
    pub functions: Vec<SpirVFunction>,
    pub bound: Word,
}

// SPIR-V Opcodes
mod opcodes {
    pub const OP_NOP: u32 = 0;
    pub const OP_NAME: u32 = 5;
    pub const OP_MEMBER_NAME: u32 = 6;
    pub const OP_EXTENSION: u32 = 10;
    pub const OP_EXT_INST_IMPORT: u32 = 11;
    pub const OP_MEMORY_MODEL: u32 = 14;
    pub const OP_ENTRY_POINT: u32 = 15;
    pub const OP_EXECUTION_MODE: u32 = 16;
    pub const OP_CAPABILITY: u32 = 17;
    pub const OP_TYPE_VOID: u32 = 19;
    pub const OP_TYPE_BOOL: u32 = 20;
    pub const OP_TYPE_INT: u32 = 21;
    pub const OP_TYPE_FLOAT: u32 = 22;
    pub const OP_TYPE_VECTOR: u32 = 23;
    pub const OP_TYPE_ARRAY: u32 = 28;
    pub const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
    pub const OP_TYPE_STRUCT: u32 = 30;
    pub const OP_TYPE_POINTER: u32 = 32;
    pub const OP_TYPE_FUNCTION: u32 = 33;
    pub const OP_CONSTANT: u32 = 43;
    pub const OP_CONSTANT_TRUE: u32 = 41;
    pub const OP_CONSTANT_FALSE: u32 = 42;
    pub const OP_CONSTANT_COMPOSITE: u32 = 44;
    pub const OP_CONSTANT_NULL: u32 = 46;
    pub const OP_VARIABLE: u32 = 59;
    pub const OP_DECORATE: u32 = 71;
    pub const OP_MEMBER_DECORATE: u32 = 72;
    pub const OP_FUNCTION: u32 = 54;
    pub const OP_FUNCTION_PARAMETER: u32 = 55;
    pub const OP_FUNCTION_END: u32 = 56;
    pub const OP_LABEL: u32 = 248;
    pub const OP_LOAD: u32 = 61;
    pub const OP_STORE: u32 = 62;
    pub const OP_ACCESS_CHAIN: u32 = 65;
    pub const OP_IADD: u32 = 128;
    pub const OP_ISUB: u32 = 130;
    pub const OP_IMUL: u32 = 132;
    pub const OP_SDIV: u32 = 135;
    pub const OP_UDIV: u32 = 134;
    pub const OP_SREM: u32 = 139;
    pub const OP_UREM: u32 = 138;
    pub const OP_SNEGATE: u32 = 126;
    pub const OP_FADD: u32 = 129;
    pub const OP_FSUB: u32 = 131;
    pub const OP_FMUL: u32 = 133;
    pub const OP_FDIV: u32 = 136;
    pub const OP_FREM: u32 = 140;
    pub const OP_FNEGATE: u32 = 127;
    pub const OP_RETURN: u32 = 253;
    pub const OP_RETURN_VALUE: u32 = 254;
    pub const OP_BRANCH: u32 = 249;
    pub const OP_BRANCH_CONDITIONAL: u32 = 250;
    pub const OP_COMPOSITE_EXTRACT: u32 = 81;
    pub const OP_COMPOSITE_CONSTRUCT: u32 = 80;
}

impl SpirVInst {
    /// Encode instruction to SPIR-V words
    pub fn encode(&self) -> Vec<u32> {
        use opcodes::*;
        let mut words = Vec::new();

        match self {
            SpirVInst::Capability(cap) => {
                words.push((2 << 16) | OP_CAPABILITY);
                words.push(cap.to_spirv());
            }
            SpirVInst::Extension(name) => {
                let name_words = encode_string(name);
                words.push(((1 + name_words.len() as u32) << 16) | OP_EXTENSION);
                words.extend(name_words);
            }
            SpirVInst::ExtInstImport { result, name } => {
                let name_words = encode_string(name);
                words.push(((2 + name_words.len() as u32) << 16) | OP_EXT_INST_IMPORT);
                words.push(result.0);
                words.extend(name_words);
            }
            SpirVInst::MemoryModel { addressing, memory } => {
                words.push((3 << 16) | OP_MEMORY_MODEL);
                words.push(addressing.to_spirv());
                words.push(memory.to_spirv());
            }
            SpirVInst::EntryPoint {
                execution_model,
                entry_point,
                name,
                interface,
            } => {
                let name_words = encode_string(name);
                let len = 3 + name_words.len() as u32 + interface.len() as u32;
                words.push((len << 16) | OP_ENTRY_POINT);
                words.push(execution_model.to_spirv());
                words.push(entry_point.0);
                words.extend(name_words);
                for id in interface {
                    words.push(id.0);
                }
            }
            SpirVInst::ExecutionMode { entry_point, mode } => {
                let (mode_val, extra) = mode.to_spirv();
                let len = 3 + extra.len() as u32;
                words.push((len << 16) | OP_EXECUTION_MODE);
                words.push(entry_point.0);
                words.push(mode_val);
                words.extend(extra);
            }
            SpirVInst::TypeVoid { result } => {
                words.push((2 << 16) | OP_TYPE_VOID);
                words.push(result.0);
            }
            SpirVInst::TypeBool { result } => {
                words.push((2 << 16) | OP_TYPE_BOOL);
                words.push(result.0);
            }
            SpirVInst::TypeInt {
                result,
                width,
                signed,
            } => {
                words.push((4 << 16) | OP_TYPE_INT);
                words.push(result.0);
                words.push(*width);
                words.push(if *signed { 1 } else { 0 });
            }
            SpirVInst::TypeFloat { result, width } => {
                words.push((3 << 16) | OP_TYPE_FLOAT);
                words.push(result.0);
                words.push(*width);
            }
            SpirVInst::TypeVector {
                result,
                component,
                count,
            } => {
                words.push((4 << 16) | OP_TYPE_VECTOR);
                words.push(result.0);
                words.push(component.0);
                words.push(*count);
            }
            SpirVInst::TypeArray {
                result,
                element,
                length,
            } => {
                words.push((4 << 16) | OP_TYPE_ARRAY);
                words.push(result.0);
                words.push(element.0);
                words.push(length.0);
            }
            SpirVInst::TypeRuntimeArray { result, element } => {
                words.push((3 << 16) | OP_TYPE_RUNTIME_ARRAY);
                words.push(result.0);
                words.push(element.0);
            }
            SpirVInst::TypeStruct { result, members } => {
                let len = 2 + members.len() as u32;
                words.push((len << 16) | OP_TYPE_STRUCT);
                words.push(result.0);
                for m in members {
                    words.push(m.0);
                }
            }
            SpirVInst::TypePointer {
                result,
                storage_class,
                pointee,
            } => {
                words.push((4 << 16) | OP_TYPE_POINTER);
                words.push(result.0);
                words.push(storage_class.to_spirv());
                words.push(pointee.0);
            }
            SpirVInst::TypeFunction {
                result,
                return_type,
                params,
            } => {
                let len = 3 + params.len() as u32;
                words.push((len << 16) | OP_TYPE_FUNCTION);
                words.push(result.0);
                words.push(return_type.0);
                for p in params {
                    words.push(p.0);
                }
            }
            SpirVInst::Constant { result, ty, value } => {
                let len = 3 + value.len() as u32;
                words.push((len << 16) | OP_CONSTANT);
                words.push(ty.0);
                words.push(result.0);
                words.extend(value);
            }
            SpirVInst::ConstantTrue { result, ty } => {
                words.push((3 << 16) | OP_CONSTANT_TRUE);
                words.push(ty.0);
                words.push(result.0);
            }
            SpirVInst::ConstantFalse { result, ty } => {
                words.push((3 << 16) | OP_CONSTANT_FALSE);
                words.push(ty.0);
                words.push(result.0);
            }
            SpirVInst::ConstantComposite {
                result,
                ty,
                constituents,
            } => {
                let len = 3 + constituents.len() as u32;
                words.push((len << 16) | OP_CONSTANT_COMPOSITE);
                words.push(ty.0);
                words.push(result.0);
                for c in constituents {
                    words.push(c.0);
                }
            }
            SpirVInst::ConstantNull { result, ty } => {
                words.push((3 << 16) | OP_CONSTANT_NULL);
                words.push(ty.0);
                words.push(result.0);
            }
            SpirVInst::Variable {
                result,
                ty,
                storage_class,
                initializer,
            } => {
                let len = if initializer.is_some() { 5 } else { 4 };
                words.push((len << 16) | OP_VARIABLE);
                words.push(ty.0);
                words.push(result.0);
                words.push(storage_class.to_spirv());
                if let Some(init) = initializer {
                    words.push(init.0);
                }
            }
            SpirVInst::Decorate { target, decoration } => {
                let (dec_val, extra) = decoration.to_spirv();
                let len = 3 + extra.len() as u32;
                words.push((len << 16) | OP_DECORATE);
                words.push(target.0);
                words.push(dec_val);
                words.extend(extra);
            }
            SpirVInst::MemberDecorate {
                struct_type,
                member,
                decoration,
            } => {
                let (dec_val, extra) = decoration.to_spirv();
                let len = 4 + extra.len() as u32;
                words.push((len << 16) | OP_MEMBER_DECORATE);
                words.push(struct_type.0);
                words.push(*member);
                words.push(dec_val);
                words.extend(extra);
            }
            SpirVInst::Function {
                result,
                result_type,
                control,
                function_type,
            } => {
                words.push((5 << 16) | OP_FUNCTION);
                words.push(result_type.0);
                words.push(result.0);
                words.push(control.to_spirv());
                words.push(function_type.0);
            }
            SpirVInst::FunctionParameter { result, ty } => {
                words.push((3 << 16) | OP_FUNCTION_PARAMETER);
                words.push(ty.0);
                words.push(result.0);
            }
            SpirVInst::FunctionEnd => {
                words.push((1 << 16) | OP_FUNCTION_END);
            }
            SpirVInst::Label { result } => {
                words.push((2 << 16) | OP_LABEL);
                words.push(result.0);
            }
            SpirVInst::Load {
                result,
                ty,
                pointer,
                memory_operands,
            } => {
                let len = if memory_operands.is_some() { 5 } else { 4 };
                words.push((len << 16) | OP_LOAD);
                words.push(ty.0);
                words.push(result.0);
                words.push(pointer.0);
                if let Some(mo) = memory_operands {
                    words.push(mo.to_spirv());
                }
            }
            SpirVInst::Store {
                pointer,
                object,
                memory_operands,
            } => {
                let len = if memory_operands.is_some() { 4 } else { 3 };
                words.push((len << 16) | OP_STORE);
                words.push(pointer.0);
                words.push(object.0);
                if let Some(mo) = memory_operands {
                    words.push(mo.to_spirv());
                }
            }
            SpirVInst::AccessChain {
                result,
                ty,
                base,
                indexes,
            } => {
                let len = 4 + indexes.len() as u32;
                words.push((len << 16) | OP_ACCESS_CHAIN);
                words.push(ty.0);
                words.push(result.0);
                words.push(base.0);
                for idx in indexes {
                    words.push(idx.0);
                }
            }
            SpirVInst::IAdd { result, ty, a, b } => {
                words.push((5 << 16) | OP_IADD);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::ISub { result, ty, a, b } => {
                words.push((5 << 16) | OP_ISUB);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::IMul { result, ty, a, b } => {
                words.push((5 << 16) | OP_IMUL);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::SDiv { result, ty, a, b } => {
                words.push((5 << 16) | OP_SDIV);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::UDiv { result, ty, a, b } => {
                words.push((5 << 16) | OP_UDIV);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::SRem { result, ty, a, b } => {
                words.push((5 << 16) | OP_SREM);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::URem { result, ty, a, b } => {
                words.push((5 << 16) | OP_UREM);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::SNegate {
                result,
                ty,
                operand,
            } => {
                words.push((4 << 16) | OP_SNEGATE);
                words.push(ty.0);
                words.push(result.0);
                words.push(operand.0);
            }
            SpirVInst::FAdd { result, ty, a, b } => {
                words.push((5 << 16) | OP_FADD);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::FSub { result, ty, a, b } => {
                words.push((5 << 16) | OP_FSUB);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::FMul { result, ty, a, b } => {
                words.push((5 << 16) | OP_FMUL);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::FDiv { result, ty, a, b } => {
                words.push((5 << 16) | OP_FDIV);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::FRem { result, ty, a, b } => {
                words.push((5 << 16) | OP_FREM);
                words.push(ty.0);
                words.push(result.0);
                words.push(a.0);
                words.push(b.0);
            }
            SpirVInst::FNegate {
                result,
                ty,
                operand,
            } => {
                words.push((4 << 16) | OP_FNEGATE);
                words.push(ty.0);
                words.push(result.0);
                words.push(operand.0);
            }
            SpirVInst::Return => {
                words.push((1 << 16) | OP_RETURN);
            }
            SpirVInst::ReturnValue { value } => {
                words.push((2 << 16) | OP_RETURN_VALUE);
                words.push(value.0);
            }
            SpirVInst::Branch { target } => {
                words.push((2 << 16) | OP_BRANCH);
                words.push(target.0);
            }
            SpirVInst::BranchConditional {
                condition,
                true_label,
                false_label,
                weights,
            } => {
                let len = if weights.is_some() { 6 } else { 4 };
                words.push((len << 16) | OP_BRANCH_CONDITIONAL);
                words.push(condition.0);
                words.push(true_label.0);
                words.push(false_label.0);
                if let Some((w1, w2)) = weights {
                    words.push(*w1);
                    words.push(*w2);
                }
            }
            SpirVInst::CompositeExtract {
                result,
                ty,
                composite,
                indexes,
            } => {
                let len = 4 + indexes.len() as u32;
                words.push((len << 16) | OP_COMPOSITE_EXTRACT);
                words.push(ty.0);
                words.push(result.0);
                words.push(composite.0);
                for idx in indexes {
                    words.push(*idx);
                }
            }
            SpirVInst::CompositeConstruct {
                result,
                ty,
                constituents,
            } => {
                let len = 3 + constituents.len() as u32;
                words.push((len << 16) | OP_COMPOSITE_CONSTRUCT);
                words.push(ty.0);
                words.push(result.0);
                for c in constituents {
                    words.push(c.0);
                }
            }
            // Handle remaining instructions with defaults
            _ => {
                // NOP for unimplemented instructions
                words.push((1 << 16) | OP_NOP);
            }
        }

        words
    }
}

/// Encode a string as SPIR-V words (null-terminated, padded to word boundary)
fn encode_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let mut words = Vec::new();
    let mut word = 0u32;
    let mut byte_idx = 0;

    for (i, &b) in bytes.iter().enumerate() {
        word |= (b as u32) << ((i % 4) * 8);
        byte_idx = i % 4;
        if byte_idx == 3 {
            words.push(word);
            word = 0;
        }
    }

    // Add null terminator and final word
    if byte_idx < 3 || words.is_empty() {
        words.push(word); // Contains null terminator in remaining bytes
    } else {
        words.push(0); // Separate null terminator word
    }

    words
}

impl SpirVModule {
    /// Assemble into SPIR-V binary
    pub fn assemble(&self) -> Vec<u32> {
        use opcodes::*;
        let mut words = Vec::new();

        // Header
        words.push(0x07230203); // Magic number
        words.push(0x00010500); // Version 1.5
        words.push(0x534F554E); // Generator: "SOUN"
        words.push(self.bound); // Bound
        words.push(0); // Reserved (schema)

        // 1. Capabilities
        for cap in &self.capabilities {
            let inst = SpirVInst::Capability(cap.clone());
            words.extend(inst.encode());
        }

        // 2. Extensions
        for ext in &self.extensions {
            let inst = SpirVInst::Extension(ext.clone());
            words.extend(inst.encode());
        }

        // 3. Ext inst imports
        for (name, result) in &self.ext_inst_imports {
            let inst = SpirVInst::ExtInstImport {
                result: *result,
                name: name.clone(),
            };
            words.extend(inst.encode());
        }

        // 4. Memory model
        let mm_inst = SpirVInst::MemoryModel {
            addressing: self.addressing_model.clone(),
            memory: self.memory_model.clone(),
        };
        words.extend(mm_inst.encode());

        // 5. Entry points
        for ep in &self.entry_points {
            let inst = SpirVInst::EntryPoint {
                execution_model: ep.execution_model.clone(),
                entry_point: ep.function,
                name: ep.name.clone(),
                interface: ep.interface.clone(),
            };
            words.extend(inst.encode());
        }

        // 6. Execution modes
        for (entry_point, mode) in &self.execution_modes {
            let inst = SpirVInst::ExecutionMode {
                entry_point: *entry_point,
                mode: mode.clone(),
            };
            words.extend(inst.encode());
        }

        // 7. Debug info (names)
        for (id, name) in &self.names {
            let name_words = encode_string(name);
            let len = 2 + name_words.len() as u32;
            words.push((len << 16) | OP_NAME);
            words.push(id.0);
            words.extend(name_words);
        }

        // 8. Decorations (already SpirVInst)
        for inst in &self.decorations {
            words.extend(inst.encode());
        }

        // 9. Types (already SpirVInst)
        for inst in &self.types {
            words.extend(inst.encode());
        }

        // 10. Constants (already SpirVInst)
        for inst in &self.constants {
            words.extend(inst.encode());
        }

        // 11. Global variables (already SpirVInst)
        for inst in &self.variables {
            words.extend(inst.encode());
        }

        // 12. Functions
        for func in &self.functions {
            // Function header
            let func_inst = SpirVInst::Function {
                result: func.id,
                result_type: func.return_type,
                control: FunctionControl::default(),
                function_type: func.function_type,
            };
            words.extend(func_inst.encode());

            // Parameters
            // Note: We need type info for each parameter. If not available,
            // we'd need a separate parameter_types field. For now, skip param encoding
            // if the function has complex parameters.

            // Blocks
            for block in &func.blocks {
                // Label instruction
                let label_inst = SpirVInst::Label {
                    result: block.label,
                };
                words.extend(label_inst.encode());

                // Block instructions
                for inst in &block.instructions {
                    words.extend(inst.encode());
                }
            }

            // Function end
            words.extend(SpirVInst::FunctionEnd.encode());
        }

        words
    }

    /// Write to a .spv file
    pub fn write_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let words = self.assemble();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)
    }
}

// ============================================================================
// SIR to SPIR-V Translation Trait
// ============================================================================

/// Trait for translating SIR constructs to SPIR-V
pub trait SirToSpirV {
    /// Lower a SIR type to SPIR-V type
    fn lower_type(&self, ty: &SirType) -> SpirVType;

    /// Lower a SIR instruction to SPIR-V instructions
    fn lower_op(&self, op: &SirInst, result: ResultId) -> Vec<SpirVInst>;

    /// Lower a SIR function to SPIR-V function
    fn lower_function(&self, func: &SirFunction) -> SpirVFunction;
}

// ============================================================================
// SPIR-V Backend
// ============================================================================

/// The main SPIR-V backend for SIR compilation
pub struct SpirVBackend {
    /// Target capabilities
    pub capabilities: TargetCapabilities,
    /// Module builder
    module_builder: SpirVModuleBuilder,
    /// Entry points collected during compilation
    entry_points: Vec<EntryPoint>,
    /// Value ID to SPIR-V ID mapping
    value_map: HashMap<ValueId, ResultId>,
    /// Block ID to SPIR-V label mapping
    block_map: HashMap<BlockId, ResultId>,
    /// Function ID to SPIR-V ID mapping
    func_map: HashMap<FuncId, ResultId>,
    /// GLSL.std.450 extended instruction set ID
    glsl_ext: Option<ResultId>,
}

impl SpirVBackend {
    /// Create a new SPIR-V backend with given capabilities
    pub fn new(capabilities: TargetCapabilities) -> Self {
        let mut module_builder = SpirVModuleBuilder::new();

        // Add required capabilities
        for cap in &capabilities.capabilities {
            module_builder.require_capability(*cap);
        }

        // Set memory model based on target
        let (addressing, memory) = match capabilities.execution_model {
            ExecutionModel::Kernel => (AddressingModel::Physical64, MemoryModel::OpenCL),
            _ => {
                if capabilities.has(Capability::VulkanMemoryModel) {
                    (AddressingModel::Logical, MemoryModel::Vulkan)
                } else {
                    (AddressingModel::Logical, MemoryModel::GLSL450)
                }
            }
        };
        module_builder.set_memory_model(addressing, memory);

        Self {
            capabilities,
            module_builder,
            entry_points: Vec::new(),
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            func_map: HashMap::new(),
            glsl_ext: None,
        }
    }

    /// Get the GLSL.std.450 extended instruction set ID
    fn glsl_ext(&mut self) -> ResultId {
        if let Some(id) = self.glsl_ext {
            return id;
        }
        let id = self.module_builder.import_ext_inst_set("GLSL.std.450");
        self.glsl_ext = Some(id);
        id
    }

    /// Select reduction strategy based on capabilities
    pub fn select_reduction_strategy(&self) -> ReductionStrategy {
        if self.capabilities.subgroup_ops && self.capabilities.has(Capability::SubgroupBallotKHR) {
            ReductionStrategy::SubgroupShuffle
        } else if self.capabilities.max_shared_memory > 0 {
            ReductionStrategy::SharedMemory
        } else {
            ReductionStrategy::Atomic
        }
    }

    /// Compile a SIR module to SPIR-V
    pub fn compile(mut self, module: &SirModule) -> Result<SpirVModule, SpirVError> {
        // Lower all functions
        for func in &module.functions {
            self.lower_sir_function(func)?;
        }

        // Add entry points
        for entry in std::mem::take(&mut self.entry_points) {
            self.module_builder.add_entry_point(entry);
        }

        Ok(self.module_builder.build())
    }

    /// Lower a SIR function (internal implementation)
    fn lower_sir_function(&mut self, func: &SirFunction) -> Result<(), SpirVError> {
        // Create function type
        let ret_ty = self.lower_type(&func.ret_ty);
        let ret_ty_id = self.module_builder.get_or_create_type(&ret_ty);

        let param_types: Vec<SpirVType> = func
            .params
            .iter()
            .map(|(_, ty)| self.lower_type(ty))
            .collect();
        let param_type_ids: Vec<ResultId> = param_types
            .iter()
            .map(|ty| self.module_builder.get_or_create_type(ty))
            .collect();

        let func_ty = SpirVType::Function {
            ret: Box::new(ret_ty.clone()),
            params: param_types,
        };
        let func_ty_id = self.module_builder.get_or_create_type(&func_ty);

        // Allocate function ID
        let func_id = self.module_builder.alloc_id();
        self.func_map.insert(func.id, func_id);
        self.module_builder.add_name(func_id, &func.name);

        // Create parameter IDs
        let param_ids: Vec<ResultId> = func
            .params
            .iter()
            .map(|_| self.module_builder.alloc_id())
            .collect();

        // Allocate block labels
        for block in &func.blocks {
            let label = self.module_builder.alloc_id();
            self.block_map.insert(block.id, label);
        }

        // Lower blocks
        let mut spirv_blocks = Vec::new();
        for block in &func.blocks {
            let spirv_block = self.lower_block(block)?;
            spirv_blocks.push(spirv_block);
        }

        // Create SPIR-V function
        let spirv_func = SpirVFunction {
            id: func_id,
            name: func.name.clone(),
            return_type: ret_ty_id,
            function_type: func_ty_id,
            parameters: param_ids,
            blocks: spirv_blocks,
            is_entry_point: func.name == "main" || func.linkage == super::blocks::Linkage::External,
            local_size: None, // Set by caller if needed
        };

        // If this is an entry point (compute shader), register it
        if spirv_func.is_entry_point
            && self.capabilities.execution_model == ExecutionModel::GLCompute
        {
            // Collect interface variables (built-ins)
            let interface = self.collect_interface_variables();

            self.entry_points.push(EntryPoint {
                function: func_id,
                name: func.name.clone(),
                execution_model: self.capabilities.execution_model,
                interface,
                local_size: Some((256, 1, 1)), // Default workgroup size
            });
        }

        self.module_builder.add_function(spirv_func);
        Ok(())
    }

    /// Lower a basic block
    fn lower_block(&mut self, block: &BasicBlock) -> Result<SpirVBlock, SpirVError> {
        let label = *self
            .block_map
            .get(&block.id)
            .ok_or_else(|| SpirVError::MissingBlock(block.id))?;

        let mut instructions = Vec::new();

        // Lower each instruction
        for inst in &block.instructions {
            let result_id = if let Some(val_id) = inst.result {
                let id = self.module_builder.alloc_id();
                self.value_map.insert(val_id, id);
                Some(id)
            } else {
                None
            };

            let spirv_insts = self.lower_instruction(&inst.inst, result_id)?;
            instructions.extend(spirv_insts);
        }

        // Lower terminator
        if let Some(term) = &block.terminator {
            let term_insts = self.lower_terminator(term)?;
            instructions.extend(term_insts);
        }

        Ok(SpirVBlock {
            label,
            instructions,
        })
    }

    /// Lower a SIR instruction to SPIR-V
    fn lower_instruction(
        &mut self,
        inst: &SirInst,
        result: Option<ResultId>,
    ) -> Result<Vec<SpirVInst>, SpirVError> {
        let mut insts = Vec::new();

        match inst {
            SirInst::BinOp { op, lhs, rhs } => {
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let ty_id = self.module_builder.get_or_create_type(&SpirVType::i32());
                let lhs_id = self.get_value(*lhs)?;
                let rhs_id = self.get_value(*rhs)?;

                let spirv_inst = match op {
                    ArithOp::Add => SpirVInst::IAdd {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::Sub => SpirVInst::ISub {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::Mul => SpirVInst::IMul {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::SDiv => SpirVInst::SDiv {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::UDiv => SpirVInst::UDiv {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::FAdd => SpirVInst::FAdd {
                        result: result_id,
                        ty: self.module_builder.get_or_create_type(&SpirVType::f32()),
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::FSub => SpirVInst::FSub {
                        result: result_id,
                        ty: self.module_builder.get_or_create_type(&SpirVType::f32()),
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::FMul => SpirVInst::FMul {
                        result: result_id,
                        ty: self.module_builder.get_or_create_type(&SpirVType::f32()),
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::FDiv => SpirVInst::FDiv {
                        result: result_id,
                        ty: self.module_builder.get_or_create_type(&SpirVType::f32()),
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::And => SpirVInst::BitwiseAnd {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::Or => SpirVInst::BitwiseOr {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::Xor => SpirVInst::BitwiseXor {
                        result: result_id,
                        ty: ty_id,
                        a: lhs_id,
                        b: rhs_id,
                    },
                    ArithOp::Shl => SpirVInst::ShiftLeftLogical {
                        result: result_id,
                        ty: ty_id,
                        base: lhs_id,
                        shift: rhs_id,
                    },
                    ArithOp::LShr => SpirVInst::ShiftRightLogical {
                        result: result_id,
                        ty: ty_id,
                        base: lhs_id,
                        shift: rhs_id,
                    },
                    ArithOp::AShr => SpirVInst::ShiftRightArithmetic {
                        result: result_id,
                        ty: ty_id,
                        base: lhs_id,
                        shift: rhs_id,
                    },
                    _ => return Err(SpirVError::UnsupportedOp(format!("{:?}", op))),
                };
                insts.push(spirv_inst);
            }

            SirInst::Const(constant) => {
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let const_id = self.lower_constant(constant)?;
                // Copy the constant to the result (would be optimized away)
                // In practice, we'd map directly to the constant
                self.value_map.insert(ValueId(result_id.0), const_id);
            }

            SirInst::Epistemic(ep_op) => {
                insts.extend(self.lower_epistemic_op(ep_op, result)?);
            }

            SirInst::Memory(mem_op) => {
                insts.extend(self.lower_memory_op(mem_op, result)?);
            }

            // Stub implementations for other instruction types
            _ => {
                // Return empty for unimplemented ops
            }
        }

        Ok(insts)
    }

    /// Lower a terminator
    fn lower_terminator(&mut self, term: &Terminator) -> Result<Vec<SpirVInst>, SpirVError> {
        let mut insts = Vec::new();

        match term {
            Terminator::Return(None) => {
                insts.push(SpirVInst::Return);
            }
            Terminator::Return(Some(val)) => {
                let val_id = self.get_value(*val)?;
                insts.push(SpirVInst::ReturnValue { value: val_id });
            }
            Terminator::Br(target) => {
                let target_label = *self
                    .block_map
                    .get(target)
                    .ok_or_else(|| SpirVError::MissingBlock(*target))?;
                insts.push(SpirVInst::Branch {
                    target: target_label,
                });
            }
            Terminator::CondBr {
                cond,
                then_block,
                else_block,
            } => {
                let cond_id = self.get_value(*cond)?;
                let then_label = *self
                    .block_map
                    .get(then_block)
                    .ok_or_else(|| SpirVError::MissingBlock(*then_block))?;
                let else_label = *self
                    .block_map
                    .get(else_block)
                    .ok_or_else(|| SpirVError::MissingBlock(*else_block))?;
                insts.push(SpirVInst::BranchConditional {
                    condition: cond_id,
                    true_label: then_label,
                    false_label: else_label,
                    weights: None,
                });
            }
            Terminator::Unreachable => {
                insts.push(SpirVInst::Unreachable);
            }
            _ => {}
        }

        Ok(insts)
    }

    /// Lower an epistemic operation
    fn lower_epistemic_op(
        &mut self,
        op: &EpistemicOp,
        result: Option<ResultId>,
    ) -> Result<Vec<SpirVInst>, SpirVError> {
        let mut insts = Vec::new();

        match op {
            EpistemicOp::Create { value, confidence } => {
                // Create a struct with (value, confidence)
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let val_id = self.get_value(*value)?;
                let conf_id = self.get_value(*confidence)?;

                // Epistemic type is struct { f64, f64 }
                let epistemic_ty = SpirVType::Struct {
                    members: vec![SpirVType::f64(), SpirVType::f64()],
                };
                let ty_id = self.module_builder.get_or_create_type(&epistemic_ty);

                insts.push(SpirVInst::CompositeConstruct {
                    result: result_id,
                    ty: ty_id,
                    constituents: vec![val_id, conf_id],
                });
            }

            EpistemicOp::ExtractValue(epistemic) => {
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let ep_id = self.get_value(*epistemic)?;
                let f64_ty = self.module_builder.get_or_create_type(&SpirVType::f64());

                insts.push(SpirVInst::CompositeExtract {
                    result: result_id,
                    ty: f64_ty,
                    composite: ep_id,
                    indexes: vec![0], // First element is value
                });
            }

            EpistemicOp::ExtractConfidence(epistemic) => {
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let ep_id = self.get_value(*epistemic)?;
                let f64_ty = self.module_builder.get_or_create_type(&SpirVType::f64());

                insts.push(SpirVInst::CompositeExtract {
                    result: result_id,
                    ty: f64_ty,
                    composite: ep_id,
                    indexes: vec![1], // Second element is confidence
                });
            }

            EpistemicOp::PropagateAdd { conf_a, conf_b } => {
                // For addition: min(conf_a, conf_b)
                // Implemented via select: conf_a < conf_b ? conf_a : conf_b
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let a_id = self.get_value(*conf_a)?;
                let b_id = self.get_value(*conf_b)?;
                let f64_ty = self.module_builder.get_or_create_type(&SpirVType::f64());
                let bool_ty = self.module_builder.get_or_create_type(&SpirVType::Bool);

                let cmp_id = self.module_builder.alloc_id();
                insts.push(SpirVInst::FOrdLessThan {
                    result: cmp_id,
                    ty: bool_ty,
                    a: a_id,
                    b: b_id,
                });

                insts.push(SpirVInst::Select {
                    result: result_id,
                    ty: f64_ty,
                    condition: cmp_id,
                    a: a_id,
                    b: b_id,
                });
            }

            EpistemicOp::PropagateMul {
                val_a,
                conf_a,
                val_b,
                conf_b,
            } => {
                // For multiplication: confidence propagation is more complex
                // Simplified: min(conf_a, conf_b)
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let ca_id = self.get_value(*conf_a)?;
                let cb_id = self.get_value(*conf_b)?;
                let f64_ty = self.module_builder.get_or_create_type(&SpirVType::f64());
                let bool_ty = self.module_builder.get_or_create_type(&SpirVType::Bool);

                let cmp_id = self.module_builder.alloc_id();
                insts.push(SpirVInst::FOrdLessThan {
                    result: cmp_id,
                    ty: bool_ty,
                    a: ca_id,
                    b: cb_id,
                });

                insts.push(SpirVInst::Select {
                    result: result_id,
                    ty: f64_ty,
                    condition: cmp_id,
                    a: ca_id,
                    b: cb_id,
                });
            }

            EpistemicOp::Meet { conf_a, conf_b } => {
                // Meet: min(conf_a, conf_b)
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let a_id = self.get_value(*conf_a)?;
                let b_id = self.get_value(*conf_b)?;
                let f64_ty = self.module_builder.get_or_create_type(&SpirVType::f64());
                let bool_ty = self.module_builder.get_or_create_type(&SpirVType::Bool);

                let cmp_id = self.module_builder.alloc_id();
                insts.push(SpirVInst::FOrdLessThan {
                    result: cmp_id,
                    ty: bool_ty,
                    a: a_id,
                    b: b_id,
                });

                insts.push(SpirVInst::Select {
                    result: result_id,
                    ty: f64_ty,
                    condition: cmp_id,
                    a: a_id,
                    b: b_id,
                });
            }

            EpistemicOp::Join { conf_a, conf_b } => {
                // Join: max(conf_a, conf_b)
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let a_id = self.get_value(*conf_a)?;
                let b_id = self.get_value(*conf_b)?;
                let f64_ty = self.module_builder.get_or_create_type(&SpirVType::f64());
                let bool_ty = self.module_builder.get_or_create_type(&SpirVType::Bool);

                let cmp_id = self.module_builder.alloc_id();
                insts.push(SpirVInst::FOrdLessThan {
                    result: cmp_id,
                    ty: bool_ty,
                    a: a_id,
                    b: b_id,
                });

                // max: a < b ? b : a
                insts.push(SpirVInst::Select {
                    result: result_id,
                    ty: f64_ty,
                    condition: cmp_id,
                    a: b_id, // If a < b, return b
                    b: a_id, // Otherwise return a
                });
            }

            _ => {
                // Other epistemic ops are stubs for now
            }
        }

        Ok(insts)
    }

    /// Lower a memory operation
    fn lower_memory_op(
        &mut self,
        op: &MemoryOp,
        result: Option<ResultId>,
    ) -> Result<Vec<SpirVInst>, SpirVError> {
        let mut insts = Vec::new();

        match op {
            MemoryOp::Load { ptr, ty, .. } => {
                let result_id = result.ok_or(SpirVError::MissingResult)?;
                let ptr_id = self.get_value(*ptr)?;
                let ty_spirv = self.lower_type(ty);
                let ty_id = self.module_builder.get_or_create_type(&ty_spirv);

                insts.push(SpirVInst::Load {
                    result: result_id,
                    ty: ty_id,
                    pointer: ptr_id,
                    memory_operands: None,
                });
            }

            MemoryOp::Store { ptr, val, .. } => {
                let ptr_id = self.get_value(*ptr)?;
                let val_id = self.get_value(*val)?;

                insts.push(SpirVInst::Store {
                    pointer: ptr_id,
                    object: val_id,
                    memory_operands: None,
                });
            }

            _ => {
                // Other memory ops are stubs
            }
        }

        Ok(insts)
    }

    /// Lower a constant
    fn lower_constant(&mut self, constant: &Constant) -> Result<ResultId, SpirVError> {
        match constant {
            Constant::Bool(true) => {
                let ty_id = self.module_builder.get_or_create_type(&SpirVType::Bool);
                let id = self.module_builder.alloc_id();
                // Would add ConstantTrue instruction
                Ok(id)
            }
            Constant::Bool(false) => {
                let ty_id = self.module_builder.get_or_create_type(&SpirVType::Bool);
                let id = self.module_builder.alloc_id();
                Ok(id)
            }
            Constant::I32(v) => Ok(self.module_builder.create_constant_i32(*v)),
            Constant::I64(v) => {
                // Would need 64-bit constant support
                Ok(self.module_builder.create_constant_i32(*v as i32))
            }
            Constant::F32(v) => Ok(self.module_builder.create_constant_f32(*v)),
            Constant::F64(v) => Ok(self.module_builder.create_constant_f64(*v)),
            _ => Err(SpirVError::UnsupportedConstant(format!("{:?}", constant))),
        }
    }

    /// Get a SPIR-V ID for a SIR value
    fn get_value(&self, val: ValueId) -> Result<ResultId, SpirVError> {
        self.value_map
            .get(&val)
            .copied()
            .ok_or_else(|| SpirVError::MissingValue(val))
    }

    /// Collect interface variables for entry point
    fn collect_interface_variables(&mut self) -> Vec<ResultId> {
        // Stub: would collect built-in variables like GlobalInvocationId
        Vec::new()
    }
}

impl SirToSpirV for SpirVBackend {
    fn lower_type(&self, ty: &SirType) -> SpirVType {
        match ty {
            SirType::Void => SpirVType::Void,
            SirType::Scalar(scalar) => match scalar {
                ScalarType::Bool => SpirVType::Bool,
                ScalarType::I8 => SpirVType::Int {
                    width: 8,
                    signed: true,
                },
                ScalarType::I16 => SpirVType::Int {
                    width: 16,
                    signed: true,
                },
                ScalarType::I32 => SpirVType::i32(),
                ScalarType::I64 => SpirVType::Int {
                    width: 64,
                    signed: true,
                },
                ScalarType::F32 => SpirVType::f32(),
                ScalarType::F64 => SpirVType::f64(),
            },
            SirType::Vector(vec) => SpirVType::Vector {
                elem: Box::new(self.lower_type(&SirType::Scalar(vec.elem))),
                count: vec.lanes as u32,
            },
            SirType::Pointer(ptr) => {
                let storage_class = match ptr.address_space {
                    0 => StorageClass::Generic,
                    1 => StorageClass::Function,
                    2 => StorageClass::StorageBuffer,
                    3 => StorageClass::Workgroup,
                    _ => StorageClass::Generic,
                };
                SpirVType::Pointer {
                    pointee: Box::new(self.lower_type(&ptr.pointee)),
                    storage_class,
                }
            }
            SirType::Struct(s) => SpirVType::Struct {
                members: s.fields.iter().map(|f| self.lower_type(&f.ty)).collect(),
            },
            SirType::Array(arr) => SpirVType::Array {
                elem: Box::new(self.lower_type(&arr.elem)),
                len: arr.len as u32,
            },
            SirType::Function(f) => SpirVType::Function {
                ret: Box::new(self.lower_type(&f.ret)),
                params: f.params.iter().map(|p| self.lower_type(p)).collect(),
            },
            SirType::Epistemic(inner) => {
                // Epistemic is lowered to struct { value, confidence }
                SpirVType::Struct {
                    members: vec![self.lower_type(inner), SpirVType::f64()],
                }
            }
            SirType::Distribution(_) => {
                // Distribution handle as opaque struct
                SpirVType::Struct {
                    members: vec![SpirVType::i32(), SpirVType::f64(), SpirVType::f64()],
                }
            }
        }
    }

    fn lower_op(&self, op: &SirInst, result: ResultId) -> Vec<SpirVInst> {
        // Stub implementation
        Vec::new()
    }

    fn lower_function(&self, func: &SirFunction) -> SpirVFunction {
        // Stub implementation
        SpirVFunction {
            id: ResultId(0),
            name: func.name.clone(),
            return_type: ResultId(0),
            function_type: ResultId(0),
            parameters: Vec::new(),
            blocks: Vec::new(),
            is_entry_point: false,
            local_size: None,
        }
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during SPIR-V compilation
#[derive(Debug, Clone)]
pub enum SpirVError {
    /// Missing value in value map
    MissingValue(ValueId),
    /// Missing block in block map
    MissingBlock(BlockId),
    /// Missing result ID for instruction that produces a value
    MissingResult,
    /// Unsupported operation
    UnsupportedOp(String),
    /// Unsupported constant type
    UnsupportedConstant(String),
    /// Unsupported type
    UnsupportedType(String),
    /// Capability not available
    MissingCapability(Capability),
    /// General compilation error
    CompileError(String),
}

impl fmt::Display for SpirVError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpirVError::MissingValue(v) => write!(f, "Missing value: {:?}", v),
            SpirVError::MissingBlock(b) => write!(f, "Missing block: {:?}", b),
            SpirVError::MissingResult => write!(f, "Instruction requires result ID"),
            SpirVError::UnsupportedOp(op) => write!(f, "Unsupported operation: {}", op),
            SpirVError::UnsupportedConstant(c) => write!(f, "Unsupported constant: {}", c),
            SpirVError::UnsupportedType(t) => write!(f, "Unsupported type: {}", t),
            SpirVError::MissingCapability(c) => write!(f, "Missing capability: {:?}", c),
            SpirVError::CompileError(msg) => write!(f, "Compilation error: {}", msg),
        }
    }
}

impl std::error::Error for SpirVError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_capabilities_vulkan_1_2() {
        let caps = TargetCapabilities::vulkan_1_2();
        assert!(caps.has(Capability::Shader));
        assert!(caps.has(Capability::Float64));
        assert!(caps.has(Capability::Int64));
        assert!(caps.subgroup_ops);
        assert!(caps.supports_f64);
    }

    #[test]
    fn test_target_capabilities_nvidia_ampere() {
        let caps = TargetCapabilities::nvidia_ampere();
        assert!(caps.cooperative_matrix);
        assert!(caps.atomic_float_add);
        assert_eq!(caps.subgroup_size, Some(32));
    }

    #[test]
    fn test_reduction_strategy_selection() {
        let caps = TargetCapabilities::vulkan_1_2();
        let backend = SpirVBackend::new(caps);
        assert_eq!(
            backend.select_reduction_strategy(),
            ReductionStrategy::SubgroupShuffle
        );

        let basic_caps = TargetCapabilities::vulkan_1_0();
        let basic_backend = SpirVBackend::new(basic_caps);
        assert_eq!(
            basic_backend.select_reduction_strategy(),
            ReductionStrategy::SharedMemory
        );
    }

    #[test]
    fn test_spirv_type_creation() {
        assert_eq!(
            SpirVType::i32(),
            SpirVType::Int {
                width: 32,
                signed: true
            }
        );
        assert_eq!(SpirVType::f32(), SpirVType::Float { width: 32 });
    }

    #[test]
    fn test_module_builder_basics() {
        let mut builder = SpirVModuleBuilder::new();

        builder.require_capability(Capability::Shader);
        builder.require_capability(Capability::Float64);

        let i32_ty = builder.get_or_create_type(&SpirVType::i32());
        let f32_ty = builder.get_or_create_type(&SpirVType::f32());

        assert_ne!(i32_ty, f32_ty);

        // Same type should return same ID
        let i32_ty_2 = builder.get_or_create_type(&SpirVType::i32());
        assert_eq!(i32_ty, i32_ty_2);
    }

    #[test]
    fn test_constant_creation() {
        let mut builder = SpirVModuleBuilder::new();

        let c1 = builder.create_constant_i32(42);
        let c2 = builder.create_constant_i32(42);
        let c3 = builder.create_constant_i32(100);

        // Same constant should return same ID
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_spirv_module_assemble() {
        let builder = SpirVModuleBuilder::new();
        let module = builder.build();
        let words = module.assemble();

        // Check magic number
        assert_eq!(words[0], 0x07230203);
    }

    #[test]
    fn test_memory_semantics() {
        let sem = MemorySemantics::workgroup_acq_rel();
        assert!(sem.acquire_release);
        assert!(sem.workgroup_memory);
        assert!(!sem.acquire);

        let bits = sem.to_spirv();
        assert_ne!(bits, 0);
    }

    #[test]
    fn test_type_lowering() {
        let caps = TargetCapabilities::vulkan_1_2();
        let backend = SpirVBackend::new(caps);

        let sir_i32 = SirType::Scalar(ScalarType::I32);
        let spirv_i32 = backend.lower_type(&sir_i32);
        assert_eq!(spirv_i32, SpirVType::i32());

        let sir_f64 = SirType::Scalar(ScalarType::F64);
        let spirv_f64 = backend.lower_type(&sir_f64);
        assert_eq!(spirv_f64, SpirVType::f64());

        let sir_epistemic = SirType::Epistemic(Box::new(SirType::f64()));
        let spirv_epistemic = backend.lower_type(&sir_epistemic);
        assert!(matches!(spirv_epistemic, SpirVType::Struct { .. }));
    }
}
