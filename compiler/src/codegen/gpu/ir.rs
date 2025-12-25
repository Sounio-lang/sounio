//! GPU Intermediate Representation
//!
//! A specialized IR for GPU code that captures:
//! - Thread hierarchy (grid, block, thread)
//! - Memory spaces (global, shared, local, constant)
//! - Synchronization primitives
//! - GPU-specific operations

use rustc_hash::FxHashMap;
use std::fmt;

/// GPU module containing kernels
#[derive(Debug, Clone)]
pub struct GpuModule {
    /// Module name
    pub name: String,

    /// Kernel functions
    pub kernels: FxHashMap<String, GpuKernel>,

    /// Device functions (callable from kernels)
    pub device_functions: FxHashMap<String, GpuFunction>,

    /// Global constants
    pub constants: Vec<GpuConstant>,

    /// Target architecture
    pub target: GpuTarget,
}

/// GPU target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTarget {
    /// NVIDIA CUDA (PTX)
    Cuda { compute_capability: (u32, u32) },

    /// Vulkan SPIR-V
    Vulkan { version: (u32, u32) },

    /// OpenCL SPIR-V
    OpenCL { version: (u32, u32) },

    /// Apple Metal (MSL)
    Metal { gpu_family: MetalGpuFamily },

    /// AMD ROCm (future)
    Rocm,

    /// Intel oneAPI (future)
    OneApi,
}

/// NVIDIA GPU architecture names (for convenience)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaArch {
    /// Turing (RTX 20xx, GTX 16xx) - sm_75
    Turing,
    /// Ampere (RTX 30xx, A100) - sm_80, sm_86, sm_87
    Ampere,
    /// Ada Lovelace (RTX 40xx) - sm_89
    Ada,
    /// Hopper (H100, H200) - sm_90, sm_90a
    Hopper,
    /// Blackwell (B100, B200, GB200) - sm_100, sm_100a
    Blackwell,
    /// Blackwell Ultra (B200 Ultra) - sm_120
    BlackwellUltra,
}

impl CudaArch {
    /// Get the base compute capability for this architecture
    pub fn compute_capability(&self) -> (u32, u32) {
        match self {
            CudaArch::Turing => (7, 5),
            CudaArch::Ampere => (8, 0),
            CudaArch::Ada => (8, 9),
            CudaArch::Hopper => (9, 0),
            CudaArch::Blackwell => (10, 0),
            CudaArch::BlackwellUltra => (12, 0),
        }
    }

    /// Get the recommended PTX version for this architecture
    pub fn ptx_version(&self) -> (u32, u32) {
        match self {
            CudaArch::Turing => (6, 4),
            CudaArch::Ampere => (7, 1),
            CudaArch::Ada => (8, 1),
            CudaArch::Hopper => (8, 3),
            CudaArch::Blackwell => (8, 5),
            CudaArch::BlackwellUltra => (8, 6),
        }
    }

    /// Get architecture from compute capability
    pub fn from_compute_capability(cc: (u32, u32)) -> Option<Self> {
        match cc {
            (7, 5) => Some(CudaArch::Turing),
            (8, 0) | (8, 6) | (8, 7) => Some(CudaArch::Ampere),
            (8, 9) => Some(CudaArch::Ada),
            (9, 0) => Some(CudaArch::Hopper),
            (10, 0) => Some(CudaArch::Blackwell),
            (12, 0) => Some(CudaArch::BlackwellUltra),
            _ => None,
        }
    }

    /// Architecture name as string
    pub fn name(&self) -> &'static str {
        match self {
            CudaArch::Turing => "Turing",
            CudaArch::Ampere => "Ampere",
            CudaArch::Ada => "Ada Lovelace",
            CudaArch::Hopper => "Hopper",
            CudaArch::Blackwell => "Blackwell",
            CudaArch::BlackwellUltra => "Blackwell Ultra",
        }
    }

    /// Tensor Core generation
    pub fn tensor_core_gen(&self) -> u32 {
        match self {
            CudaArch::Turing => 2,
            CudaArch::Ampere => 3,
            CudaArch::Ada | CudaArch::Hopper => 4,
            CudaArch::Blackwell | CudaArch::BlackwellUltra => 5,
        }
    }

    /// Maximum shared memory per block (bytes)
    pub fn max_shared_memory(&self) -> u32 {
        match self {
            CudaArch::Turing => 64 * 1024,          // 64 KB
            CudaArch::Ampere => 164 * 1024,         // 164 KB (A100)
            CudaArch::Ada => 100 * 1024,            // 100 KB
            CudaArch::Hopper => 228 * 1024,         // 228 KB (H100)
            CudaArch::Blackwell => 256 * 1024,      // 256 KB (estimated)
            CudaArch::BlackwellUltra => 256 * 1024, // 256 KB (estimated)
        }
    }

    /// Maximum threads per block
    pub fn max_threads_per_block(&self) -> u32 {
        1024 // Same across all architectures
    }

    /// Maximum registers per thread
    pub fn max_registers_per_thread(&self) -> u32 {
        255 // Same across all architectures
    }

    /// L2 cache size (bytes)
    pub fn l2_cache_size(&self) -> u32 {
        match self {
            CudaArch::Turing => 6 * 1024 * 1024,           // 6 MB
            CudaArch::Ampere => 40 * 1024 * 1024,          // 40 MB (A100)
            CudaArch::Ada => 72 * 1024 * 1024,             // 72 MB (RTX 4090)
            CudaArch::Hopper => 50 * 1024 * 1024,          // 50 MB (H100)
            CudaArch::Blackwell => 96 * 1024 * 1024,       // 96 MB (estimated B100)
            CudaArch::BlackwellUltra => 128 * 1024 * 1024, // 128 MB (estimated)
        }
    }
}

/// Feature flags for CUDA compute capabilities
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaFeatures {
    /// Supports BFloat16 (sm_80+)
    pub bf16: bool,
    /// Supports FP8 (sm_89+)
    pub fp8: bool,
    /// Supports TMA (Tensor Memory Accelerator, sm_90+)
    pub tma: bool,
    /// Supports Thread Block Clusters (sm_90+)
    pub clusters: bool,
    /// Supports Distributed Shared Memory (sm_90+)
    pub distributed_shared_memory: bool,
    /// Supports FP8 in Tensor Cores (sm_89+)
    pub tensor_fp8: bool,
    /// Supports FP4/INT4 Tensor Cores (sm_100+, Blackwell)
    pub tensor_fp4: bool,
    /// Supports 2nd-gen Transformer Engine (sm_100+)
    pub transformer_engine_v2: bool,
    /// Supports dynamic shared memory carveout
    pub dynamic_shared_memory: bool,
    /// Supports asynchronous copy (sm_80+)
    pub async_copy: bool,
    /// Supports redux warp reduction (sm_80+)
    pub redux: bool,
    /// Supports mbarrier (sm_80+)
    pub mbarrier: bool,
    /// Supports 5th-gen Tensor Cores (sm_100+)
    pub tensor_core_gen5: bool,
    /// Supports NVLink 5.0 (sm_100+)
    pub nvlink5: bool,
    /// Supports Confidential Computing (sm_100+)
    pub confidential_computing: bool,
    /// Supports Decompression Engine (sm_100+)
    pub decompression_engine: bool,
}

impl CudaFeatures {
    /// Get features for a compute capability
    pub fn from_compute_capability(cc: (u32, u32)) -> Self {
        let (major, minor) = cc;
        let sm = major * 10 + minor;

        CudaFeatures {
            bf16: sm >= 80,
            fp8: sm >= 89,
            tma: sm >= 90,
            clusters: sm >= 90,
            distributed_shared_memory: sm >= 90,
            tensor_fp8: sm >= 89,
            tensor_fp4: sm >= 100,
            transformer_engine_v2: sm >= 100,
            dynamic_shared_memory: sm >= 70,
            async_copy: sm >= 80,
            redux: sm >= 80,
            mbarrier: sm >= 80,
            tensor_core_gen5: sm >= 100,
            nvlink5: sm >= 100,
            confidential_computing: sm >= 100,
            decompression_engine: sm >= 100,
        }
    }

    /// Check if all Blackwell features are available
    pub fn is_blackwell(&self) -> bool {
        self.tensor_core_gen5 && self.tensor_fp4 && self.transformer_engine_v2
    }

    /// Check if all Hopper features are available
    pub fn is_hopper(&self) -> bool {
        self.tma && self.clusters && self.distributed_shared_memory && !self.tensor_core_gen5
    }
}

/// Apple GPU family for Metal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MetalGpuFamily {
    /// Apple 7 (M1, A14) - First Apple Silicon
    Apple7,
    /// Apple 8 (M2, A15, A16) - Enhanced ML
    Apple8,
    /// Apple 9 (M3, A17) - Dynamic caching, ray tracing
    Apple9,
    /// Apple 10 (M4, A18) - Enhanced ray tracing, mesh shaders (futureproof)
    Apple10,
    /// Mac 2 (Intel discrete GPU)
    Mac2,
    /// Common subset (portable)
    Common,
}

impl MetalGpuFamily {
    /// Maximum threads per threadgroup
    pub fn max_threads_per_threadgroup(&self) -> u32 {
        match self {
            MetalGpuFamily::Apple7 => 1024,
            MetalGpuFamily::Apple8 => 1024,
            MetalGpuFamily::Apple9 => 1024,
            MetalGpuFamily::Apple10 => 1024,
            MetalGpuFamily::Mac2 => 1024,
            MetalGpuFamily::Common => 512,
        }
    }

    /// Maximum threadgroup memory (bytes)
    pub fn max_threadgroup_memory(&self) -> u32 {
        match self {
            MetalGpuFamily::Apple7 => 32768,
            MetalGpuFamily::Apple8 => 32768,
            MetalGpuFamily::Apple9 => 32768,
            MetalGpuFamily::Apple10 => 65536, // M4 has more threadgroup memory
            MetalGpuFamily::Mac2 => 65536,
            MetalGpuFamily::Common => 16384,
        }
    }

    /// Supports simdgroup operations
    pub fn supports_simdgroup(&self) -> bool {
        true // All modern Metal GPUs support simdgroup
    }

    /// Supports simdgroup matrix (for ML)
    pub fn supports_simdgroup_matrix(&self) -> bool {
        matches!(
            self,
            MetalGpuFamily::Apple7
                | MetalGpuFamily::Apple8
                | MetalGpuFamily::Apple9
                | MetalGpuFamily::Apple10
        )
    }

    /// Supports mesh shaders (Apple10+)
    pub fn supports_mesh_shaders(&self) -> bool {
        matches!(self, MetalGpuFamily::Apple10)
    }

    /// Supports hardware ray tracing
    pub fn supports_ray_tracing(&self) -> bool {
        matches!(self, MetalGpuFamily::Apple9 | MetalGpuFamily::Apple10)
    }

    /// SIMD width (threads per simdgroup)
    pub fn simd_width(&self) -> u32 {
        32 // Apple GPUs use 32-wide SIMD
    }

    /// MSL version string
    pub fn msl_version(&self) -> &'static str {
        match self {
            MetalGpuFamily::Apple7 => "2.4",
            MetalGpuFamily::Apple8 => "3.0",
            MetalGpuFamily::Apple9 => "3.1",
            MetalGpuFamily::Apple10 => "3.2", // Future MSL version
            MetalGpuFamily::Mac2 => "2.4",
            MetalGpuFamily::Common => "2.3",
        }
    }
}

impl Default for GpuTarget {
    fn default() -> Self {
        GpuTarget::Cuda {
            compute_capability: (7, 5),
        }
    }
}

impl fmt::Display for GpuTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuTarget::Cuda { compute_capability } => {
                write!(
                    f,
                    "CUDA sm_{}{}",
                    compute_capability.0, compute_capability.1
                )
            }
            GpuTarget::Vulkan { version } => {
                write!(f, "Vulkan {}.{}", version.0, version.1)
            }
            GpuTarget::OpenCL { version } => {
                write!(f, "OpenCL {}.{}", version.0, version.1)
            }
            GpuTarget::Metal { gpu_family } => {
                write!(
                    f,
                    "Metal {:?} (MSL {})",
                    gpu_family,
                    gpu_family.msl_version()
                )
            }
            GpuTarget::Rocm => write!(f, "ROCm"),
            GpuTarget::OneApi => write!(f, "oneAPI"),
        }
    }
}

/// GPU kernel function
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// Kernel name
    pub name: String,

    /// Parameters
    pub params: Vec<GpuParam>,

    /// Shared memory declarations
    pub shared_memory: Vec<SharedMemDecl>,

    /// Basic blocks
    pub blocks: Vec<GpuBlock>,

    /// Entry block
    pub entry: BlockId,

    /// Maximum threads per block (optional hint)
    pub max_threads: Option<u32>,

    /// Required shared memory (bytes)
    pub shared_mem_size: u32,
}

/// GPU device function (non-kernel, callable from GPU)
#[derive(Debug, Clone)]
pub struct GpuFunction {
    /// Function name
    pub name: String,

    /// Parameters
    pub params: Vec<GpuParam>,

    /// Return type
    pub return_type: GpuType,

    /// Basic blocks
    pub blocks: Vec<GpuBlock>,

    /// Entry block
    pub entry: BlockId,

    /// Is inline hint
    pub inline: bool,
}

/// GPU parameter
#[derive(Debug, Clone)]
pub struct GpuParam {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub ty: GpuType,

    /// Memory space
    pub space: MemorySpace,

    /// Is restrict (no aliasing)
    pub restrict: bool,
}

/// GPU type
#[derive(Debug, Clone, PartialEq)]
pub enum GpuType {
    // Scalar types
    Void,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,

    // Modern ML/scientific types (CUDA 13.x / Blackwell)
    /// BFloat16 - 1 sign, 8 exponent, 7 mantissa (same exponent range as f32)
    BF16,
    /// FP8 E4M3 - 1 sign, 4 exponent, 3 mantissa (IEEE 754 style, higher precision)
    F8E4M3,
    /// FP8 E5M2 - 1 sign, 5 exponent, 2 mantissa (IEEE 754 style, larger range)
    F8E5M2,
    /// 4-bit float for extreme quantization (2-bit exponent, 1-bit mantissa)
    F4,

    // Vector types
    Vec2(Box<GpuType>),
    Vec3(Box<GpuType>),
    Vec4(Box<GpuType>),

    // Pointer types
    Ptr(Box<GpuType>, MemorySpace),

    // Array types
    Array(Box<GpuType>, u32),

    // Struct types
    Struct(String, Vec<(String, GpuType)>),
}

impl GpuType {
    pub fn size_bytes(&self) -> u32 {
        match self {
            GpuType::Void => 0,
            GpuType::Bool | GpuType::I8 | GpuType::U8 | GpuType::F8E4M3 | GpuType::F8E5M2 => 1,
            GpuType::I16 | GpuType::U16 | GpuType::F16 | GpuType::BF16 => 2,
            GpuType::I32 | GpuType::U32 | GpuType::F32 => 4,
            GpuType::I64 | GpuType::U64 | GpuType::F64 => 8,
            GpuType::F4 => 1, // Packed: 2 values per byte, but allocate 1 byte for single value
            GpuType::Vec2(t) => t.size_bytes() * 2,
            GpuType::Vec3(t) => t.size_bytes() * 3,
            GpuType::Vec4(t) => t.size_bytes() * 4,
            GpuType::Ptr(_, _) => 8,
            GpuType::Array(t, n) => t.size_bytes() * n,
            GpuType::Struct(_, fields) => fields.iter().map(|(_, t)| t.size_bytes()).sum(),
        }
    }

    pub fn alignment(&self) -> u32 {
        match self {
            GpuType::Vec2(t) | GpuType::Vec3(t) | GpuType::Vec4(t) => t.alignment() * 2,
            GpuType::Array(t, _) => t.alignment(),
            GpuType::Struct(_, fields) => {
                fields.iter().map(|(_, t)| t.alignment()).max().unwrap_or(1)
            }
            _ => self.size_bytes().max(1),
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            GpuType::F16
                | GpuType::F32
                | GpuType::F64
                | GpuType::BF16
                | GpuType::F8E4M3
                | GpuType::F8E5M2
                | GpuType::F4
        )
    }

    /// Check if this is a signed integer type
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            GpuType::I8 | GpuType::I16 | GpuType::I32 | GpuType::I64
        )
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            GpuType::U8 | GpuType::U16 | GpuType::U32 | GpuType::U64
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        self.is_signed() || self.is_unsigned()
    }
}

impl fmt::Display for GpuType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuType::Void => write!(f, "void"),
            GpuType::Bool => write!(f, "bool"),
            GpuType::I8 => write!(f, "i8"),
            GpuType::I16 => write!(f, "i16"),
            GpuType::I32 => write!(f, "i32"),
            GpuType::I64 => write!(f, "i64"),
            GpuType::U8 => write!(f, "u8"),
            GpuType::U16 => write!(f, "u16"),
            GpuType::U32 => write!(f, "u32"),
            GpuType::U64 => write!(f, "u64"),
            GpuType::F16 => write!(f, "f16"),
            GpuType::F32 => write!(f, "f32"),
            GpuType::F64 => write!(f, "f64"),
            GpuType::BF16 => write!(f, "bf16"),
            GpuType::F8E4M3 => write!(f, "f8e4m3"),
            GpuType::F8E5M2 => write!(f, "f8e5m2"),
            GpuType::F4 => write!(f, "f4"),
            GpuType::Vec2(t) => write!(f, "vec2<{}>", t),
            GpuType::Vec3(t) => write!(f, "vec3<{}>", t),
            GpuType::Vec4(t) => write!(f, "vec4<{}>", t),
            GpuType::Ptr(t, space) => write!(f, "*{:?} {}", space, t),
            GpuType::Array(t, n) => write!(f, "[{}; {}]", t, n),
            GpuType::Struct(name, _) => write!(f, "struct {}", name),
        }
    }
}

/// Memory space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    /// Global device memory (DRAM)
    Global,

    /// Shared memory (on-chip, per block)
    Shared,

    /// Local memory (per thread, register spill)
    Local,

    /// Constant memory (cached, read-only)
    Constant,

    /// Texture memory (cached, 2D locality)
    Texture,

    /// Generic (resolved at runtime)
    Generic,
}

impl fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemorySpace::Global => write!(f, "global"),
            MemorySpace::Shared => write!(f, "shared"),
            MemorySpace::Local => write!(f, "local"),
            MemorySpace::Constant => write!(f, "constant"),
            MemorySpace::Texture => write!(f, "texture"),
            MemorySpace::Generic => write!(f, "generic"),
        }
    }
}

/// Quantization mode for FP8/F4 conversions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizeMode {
    /// Round to nearest even (default, best accuracy)
    RoundNearestEven,
    /// Round toward zero (truncate)
    RoundTowardZero,
    /// Round toward positive infinity
    RoundTowardPosInf,
    /// Round toward negative infinity
    RoundTowardNegInf,
    /// Stochastic rounding (uses random bits for tie-breaking)
    Stochastic,
}

impl fmt::Display for QuantizeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantizeMode::RoundNearestEven => write!(f, "rne"),
            QuantizeMode::RoundTowardZero => write!(f, "rtz"),
            QuantizeMode::RoundTowardPosInf => write!(f, "rtp"),
            QuantizeMode::RoundTowardNegInf => write!(f, "rtn"),
            QuantizeMode::Stochastic => write!(f, "stochastic"),
        }
    }
}

/// FP8 format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Format {
    /// E4M3: 4-bit exponent, 3-bit mantissa (higher precision, range ±448)
    E4M3,
    /// E5M2: 5-bit exponent, 2-bit mantissa (larger range, ±57344)
    E5M2,
}

impl fmt::Display for Fp8Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fp8Format::E4M3 => write!(f, "e4m3"),
            Fp8Format::E5M2 => write!(f, "e5m2"),
        }
    }
}

/// TMA reduction operation type (sm_100+)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TmaReduceOp {
    /// Atomic add
    Add,
    /// Atomic min
    Min,
    /// Atomic max
    Max,
    /// Atomic AND
    And,
    /// Atomic OR
    Or,
    /// Atomic XOR
    Xor,
}

impl fmt::Display for TmaReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TmaReduceOp::Add => write!(f, "add"),
            TmaReduceOp::Min => write!(f, "min"),
            TmaReduceOp::Max => write!(f, "max"),
            TmaReduceOp::And => write!(f, "and"),
            TmaReduceOp::Or => write!(f, "or"),
            TmaReduceOp::Xor => write!(f, "xor"),
        }
    }
}

/// Memory layout for tile data in shared memory (CUDA 13+)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileLayout {
    /// Row-major layout (C convention)
    RowMajor,
    /// Column-major layout (Fortran convention)
    ColMajor,
    /// Swizzled layout for bank conflict avoidance
    /// block_m and block_n define swizzle granularity
    Swizzled { block_m: u32, block_n: u32 },
}

impl fmt::Display for TileLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileLayout::RowMajor => write!(f, "row_major"),
            TileLayout::ColMajor => write!(f, "col_major"),
            TileLayout::Swizzled { block_m, block_n } => {
                write!(f, "swizzled({},{})", block_m, block_n)
            }
        }
    }
}

impl TileLayout {
    /// Convert layout string to TileLayout enum
    pub fn from_string(s: &str) -> Self {
        match s {
            "col_major" => TileLayout::ColMajor,
            "swizzled" => TileLayout::Swizzled {
                block_m: 8,
                block_n: 64,
            },
            _ => TileLayout::RowMajor, // Default
        }
    }
}

/// Shared memory declaration
#[derive(Debug, Clone)]
pub struct SharedMemDecl {
    /// Variable name
    pub name: String,

    /// Element type
    pub elem_type: GpuType,

    /// Number of elements
    pub size: u32,

    /// Alignment
    pub align: u32,
}

/// Global constant
#[derive(Debug, Clone)]
pub struct GpuConstant {
    /// Constant name
    pub name: String,

    /// Type
    pub ty: GpuType,

    /// Value
    pub value: GpuConstValue,
}

/// Constant value
#[derive(Debug, Clone)]
pub enum GpuConstValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<GpuConstValue>),
    Struct(Vec<GpuConstValue>),
}

/// Block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BB{}", self.0)
    }
}

/// Value identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// GPU basic block
#[derive(Debug, Clone)]
pub struct GpuBlock {
    /// Block ID
    pub id: BlockId,

    /// Block label
    pub label: String,

    /// Instructions
    pub instructions: Vec<(ValueId, GpuOp)>,

    /// Terminator
    pub terminator: GpuTerminator,
}

/// GPU operations
#[derive(Debug, Clone)]
pub enum GpuOp {
    // === Constants ===
    ConstInt(i64, GpuType),
    ConstFloat(f64, GpuType),
    ConstBool(bool),

    // === Arithmetic ===
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    Rem(ValueId, ValueId),
    Neg(ValueId),

    // === Floating-point ===
    FAdd(ValueId, ValueId),
    FSub(ValueId, ValueId),
    FMul(ValueId, ValueId),
    FDiv(ValueId, ValueId),
    FNeg(ValueId),

    // === Fast math (relaxed precision) ===
    FMulAdd(ValueId, ValueId, ValueId), // a * b + c
    FastSin(ValueId),
    FastCos(ValueId),
    FastExp(ValueId),
    FastLog(ValueId),
    FastSqrt(ValueId),
    FastRsqrt(ValueId), // 1/sqrt(x)

    // === Comparisons ===
    Eq(ValueId, ValueId),
    Ne(ValueId, ValueId),
    Lt(ValueId, ValueId),
    Le(ValueId, ValueId),
    Gt(ValueId, ValueId),
    Ge(ValueId, ValueId),

    // Float comparisons
    FEq(ValueId, ValueId),
    FNe(ValueId, ValueId),
    FLt(ValueId, ValueId),
    FLe(ValueId, ValueId),
    FGt(ValueId, ValueId),
    FGe(ValueId, ValueId),

    // === Logical ===
    And(ValueId, ValueId),
    Or(ValueId, ValueId),
    Xor(ValueId, ValueId),
    Not(ValueId),

    // === Bit operations ===
    Shl(ValueId, ValueId),
    Shr(ValueId, ValueId),  // Arithmetic
    LShr(ValueId, ValueId), // Logical
    BitAnd(ValueId, ValueId),
    BitOr(ValueId, ValueId),
    BitXor(ValueId, ValueId),
    BitNot(ValueId),
    PopCount(ValueId),
    Clz(ValueId), // Count leading zeros
    Ctz(ValueId), // Count trailing zeros

    // === Conversions ===
    Trunc(ValueId, GpuType),
    ZExt(ValueId, GpuType),
    SExt(ValueId, GpuType),
    FpTrunc(ValueId, GpuType),
    FpExt(ValueId, GpuType),
    FpToSi(ValueId, GpuType),
    FpToUi(ValueId, GpuType),
    SiToFp(ValueId, GpuType),
    UiToFp(ValueId, GpuType),
    Bitcast(ValueId, GpuType),

    // === Modern ML Type Conversions (BF16/FP8/F4) ===
    /// Convert F32 to BF16 (truncate mantissa from 23 to 7 bits)
    F32ToBF16(ValueId),
    /// Convert BF16 to F32 (extend mantissa)
    BF16ToF32(ValueId),
    /// Convert F32 to FP8 E4M3 (higher precision FP8, range ±448)
    F32ToF8E4M3(ValueId),
    /// Convert FP8 E4M3 to F32
    F8E4M3ToF32(ValueId),
    /// Convert F32 to FP8 E5M2 (larger range FP8, range ±57344)
    F32ToF8E5M2(ValueId),
    /// Convert FP8 E5M2 to F32
    F8E5M2ToF32(ValueId),
    /// Convert F32 to F4 (extreme 4-bit quantization)
    F32ToF4(ValueId),
    /// Convert F4 to F32 (dequantization)
    F4ToF32(ValueId),

    // === Packed ML Type Operations ===
    /// Pack two FP8 values into a u16
    PackF8x2(ValueId, ValueId),
    /// Unpack u16 into two FP8 values (returns low byte as index 0)
    UnpackF8x2Low(ValueId),
    /// Unpack u16 into two FP8 values (returns high byte as index 1)
    UnpackF8x2High(ValueId),
    /// Pack two F4 values into a single byte
    PackF4x2(ValueId, ValueId),
    /// Unpack byte into low F4 value (bits 0-3)
    UnpackF4x2Low(ValueId),
    /// Unpack byte into high F4 value (bits 4-7)
    UnpackF4x2High(ValueId),

    // === Quantization Utilities ===
    /// Quantize F32 to F8 with saturation and rounding
    QuantizeF32ToF8(ValueId, QuantizeMode),
    /// Dequantize F8 to F32 with optional scale
    DequantizeF8ToF32(ValueId, Option<ValueId>), // value, optional scale factor

    // === INT8/INT4 Quantization (Phase 11) ===
    /// Quantize F32 to INT8: q = clamp(round(x / scale) + zero_point, -128, 127)
    /// Used for symmetric quantization (zero_point = 0) and asymmetric
    QuantizeF32ToInt8 {
        value: ValueId,
        scale: ValueId,
        zero_point: ValueId,
        symmetric: bool,
    },

    /// Dequantize INT8 to F32: x = (q - zero_point) * scale
    DequantizeInt8ToF32 {
        value: ValueId,
        scale: ValueId,
        zero_point: ValueId,
    },

    /// Quantize F32 to UINT8: q = clamp(round(x / scale) + zero_point, 0, 255)
    /// Used for activation quantization where values are non-negative
    QuantizeF32ToUint8 {
        value: ValueId,
        scale: ValueId,
        zero_point: ValueId,
    },

    /// Dequantize UINT8 to F32: x = (q - zero_point) * scale
    DequantizeUint8ToF32 {
        value: ValueId,
        scale: ValueId,
        zero_point: ValueId,
    },

    /// Quantize F32 to INT4 (packed): two values packed into one byte
    /// q_lo = clamp(round(x_lo / scale) + zp, -8, 7)
    /// q_hi = clamp(round(x_hi / scale) + zp, -8, 7)
    /// result = (q_hi << 4) | (q_lo & 0x0F)
    QuantizeF32ToInt4 {
        value_lo: ValueId,
        value_hi: ValueId,
        scale: ValueId,
        zero_point: ValueId,
    },

    /// Dequantize INT4 (packed) to two F32 values
    /// Returns low nibble as f32
    DequantizeInt4ToF32Lo {
        packed: ValueId,
        scale: ValueId,
        zero_point: ValueId,
    },

    /// Dequantize INT4 (packed) to two F32 values
    /// Returns high nibble as f32
    DequantizeInt4ToF32Hi {
        packed: ValueId,
        scale: ValueId,
        zero_point: ValueId,
    },

    /// INT8 dot product using dp4a instruction (sm_61+)
    /// Computes: c + dot(a[0:3], b[0:3]) where a, b are packed 4x INT8
    /// a and b are interpreted as 4 signed 8-bit integers packed in 32 bits
    Dp4a {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    /// Unsigned INT8 dot product using dp4a.u32 (sm_61+)
    Dp4aUnsigned {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    /// Mixed signed/unsigned dp4a (sm_61+)
    /// a is signed, b is unsigned
    Dp4aSU {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    /// INT8 matrix multiply with tensor cores (sm_75+)
    /// Performs C = A * B + C where A and B are INT8, C is INT32
    Int8MatMul {
        a: ValueId,       // INT8 matrix A
        b: ValueId,       // INT8 matrix B
        c: ValueId,       // INT32 accumulator
        m: u32,           // M dimension
        n: u32,           // N dimension
        k: u32,           // K dimension
        a_scale: ValueId, // Scale for dequantizing A
        b_scale: ValueId, // Scale for dequantizing B
    },

    /// Per-channel quantization (for weights)
    /// Each channel has its own scale and zero_point
    QuantizePerChannel {
        values: ValueId,      // Input values
        scales: ValueId,      // Per-channel scales array
        zero_points: ValueId, // Per-channel zero points array
        axis: u32,            // Channel axis (typically 0 for output channels)
        num_channels: u32,    // Number of channels
        signed: bool,         // INT8 (true) or UINT8 (false)
    },

    /// Dequantize per-channel quantized values
    DequantizePerChannel {
        values: ValueId,
        scales: ValueId,
        zero_points: ValueId,
        axis: u32,
        num_channels: u32,
    },

    /// Compute quantization scale from min/max values
    /// scale = (max - min) / (qmax - qmin)
    ComputeQuantScale {
        min_val: ValueId,
        max_val: ValueId,
        num_bits: u32, // 8 for INT8, 4 for INT4
        symmetric: bool,
    },

    /// Compute zero point for asymmetric quantization
    /// zero_point = round(-min / scale)
    ComputeZeroPoint {
        min_val: ValueId,
        scale: ValueId,
        num_bits: u32,
    },

    /// Find min/max in a tensor (reduction)
    /// Used for calibration
    FindMinMax {
        values: ValueId,
        count: ValueId,
    },

    /// Requantize from one scale to another
    /// Useful for fusing quantized layers
    Requantize {
        value: ValueId,
        in_scale: ValueId,
        in_zero_point: ValueId,
        out_scale: ValueId,
        out_zero_point: ValueId,
    },

    // === Blackwell Features (sm_100+) ===
    // Tensor Memory Accelerator (TMA) - bulk async memory operations
    /// TMA async copy from global to shared (sm_90+, enhanced in sm_100)
    TmaLoadAsync {
        dst_shared: ValueId,
        src_global: ValueId,
        size: u32,
        barrier: ValueId,
    },
    /// TMA async copy from shared to global
    TmaStoreAsync {
        dst_global: ValueId,
        src_shared: ValueId,
        size: u32,
    },
    /// TMA multicast load to multiple CTAs in cluster
    TmaMulticastLoad {
        dst_shared: ValueId,
        src_global: ValueId,
        size: u32,
        cluster_mask: u32,
        barrier: ValueId,
    },
    /// TMA reduction operation (scatter-add)
    TmaReduceAsync {
        dst_global: ValueId,
        src_shared: ValueId,
        size: u32,
        reduce_op: TmaReduceOp,
    },

    // 5th-gen Tensor Core Operations (sm_100+)
    /// WGMMA (Warpgroup Matrix Multiply-Accumulate) with FP4
    WgmmaFp4 {
        a: ValueId,       // FP4 matrix A
        b: ValueId,       // FP4 matrix B
        c: ValueId,       // Accumulator (FP32)
        m: u32,           // M dimension
        n: u32,           // N dimension
        k: u32,           // K dimension
        scale_a: ValueId, // Scale for A
        scale_b: ValueId, // Scale for B
    },
    /// WGMMA with FP8 (E4M3 or E5M2)
    WgmmaFp8 {
        a: ValueId,
        b: ValueId,
        c: ValueId,
        m: u32,
        n: u32,
        k: u32,
        format: Fp8Format,
    },
    /// WGMMA with BF16
    WgmmaBf16 {
        a: ValueId,
        b: ValueId,
        c: ValueId,
        m: u32,
        n: u32,
        k: u32,
    },

    // Transformer Engine v2 (sm_100+)
    /// Fused attention with FP8 quantization
    TransformerEngineFusedAttention {
        q: ValueId,      // Query tensor
        k: ValueId,      // Key tensor
        v: ValueId,      // Value tensor
        scale: ValueId,  // Softmax scale
        output: ValueId, // Output tensor
        format: Fp8Format,
    },
    /// FP8 GEMM with amax tracking for dynamic scaling
    TransformerEngineFp8Gemm {
        a: ValueId,
        b: ValueId,
        c: ValueId,
        amax_out: ValueId, // Track max for scaling
        format: Fp8Format,
    },

    // Decompression Engine (sm_100+)
    /// Hardware decompression from LZ4
    DecompressLz4 {
        dst: ValueId,
        src: ValueId,
        compressed_size: u32,
        uncompressed_size: u32,
    },
    /// Hardware decompression from Snappy
    DecompressSnappy {
        dst: ValueId,
        src: ValueId,
        compressed_size: u32,
        uncompressed_size: u32,
    },
    /// Hardware decompression from Deflate
    DecompressDeflate {
        dst: ValueId,
        src: ValueId,
        compressed_size: u32,
        uncompressed_size: u32,
    },

    // Thread Block Cluster Extensions (sm_100+ enhanced)
    /// Get cluster ID within grid
    ClusterId,
    /// Get cluster dimension
    ClusterDim,
    /// Get block ID within cluster
    BlockIdInCluster,
    /// Cluster-wide barrier
    ClusterBarrier,
    /// Cluster-wide arrive (non-blocking)
    ClusterArrive(ValueId), // barrier
    /// Cluster-wide wait
    ClusterWait(ValueId), // barrier

    // NVLink 5.0 Operations (sm_100+)
    /// Remote direct memory access read
    NvlinkRead {
        dst: ValueId,
        src_gpu: u32,
        src_addr: ValueId,
        size: u32,
    },
    /// Remote direct memory access write
    NvlinkWrite {
        dst_gpu: u32,
        dst_addr: ValueId,
        src: ValueId,
        size: u32,
    },
    /// NVLink atomic operation
    NvlinkAtomicAdd {
        dst_gpu: u32,
        dst_addr: ValueId,
        value: ValueId,
    },

    // ===== TILE PROGRAMMING (CUDA 13+) =====
    /// Create a tile from a cooperative group
    /// Maps to: CoopPartitionTiled + shared memory allocation
    TileCreate {
        group: ValueId,
        tile_m: u32,
        tile_n: u32,
        element_type: GpuType,
        layout: TileLayout,
    },
    /// Load tile from global memory to shared memory (collective operation)
    /// Maps to: TmaLoadAsync (sm_90+) or manual coalesced loads
    TileLoad {
        tile: ValueId,
        src_ptr: ValueId,
        stride: ValueId,
        barrier: Option<ValueId>,
    },
    /// Store tile from shared memory to global memory (collective operation)
    /// Maps to: TmaStoreAsync (sm_90+) or manual coalesced stores
    TileStore {
        tile: ValueId,
        dst_ptr: ValueId,
        stride: ValueId,
        barrier: Option<ValueId>,
    },
    /// Matrix multiply-accumulate on tiles (Tensor Core operation)
    /// C = A @ B + C
    /// Maps to: WgmmaFp4/WgmmaFp8/WgmmaBf16 based on element type
    TileMma {
        c: ValueId,
        a: ValueId,
        b: ValueId,
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
    },
    /// Synchronize all threads in tile (collective barrier)
    /// Maps to: CoopSync(tile_group)
    TileSync(ValueId),
    /// Element-wise access to tile (single thread)
    /// Returns reference to element at (row, col)
    TileGetElement {
        tile: ValueId,
        row: ValueId,
        col: ValueId,
    },
    /// Set element in tile (single thread)
    TileSetElement {
        tile: ValueId,
        row: ValueId,
        col: ValueId,
        value: ValueId,
    },
    /// Fill entire tile with scalar value (collective broadcast)
    TileFill {
        tile: ValueId,
        value: ValueId,
    },
    /// Reduce tile to scalar (collective operation)
    /// Maps to: CoopReduce across all tile threads
    TileReduce {
        tile: ValueId,
        reduce_op: CoopReduceOp,
    },
    /// Transpose tile in-place (collective operation)
    /// Useful for memory layout conversions
    TileTranspose(ValueId),
    /// Get tile dimensions (constant folding opportunity)
    TileM(ValueId),
    TileN(ValueId),

    // === Memory ===
    Load(ValueId, MemorySpace),
    Store(ValueId, ValueId, MemorySpace), // ptr, value

    // Atomic operations
    AtomicAdd(ValueId, ValueId),
    AtomicSub(ValueId, ValueId),
    AtomicMin(ValueId, ValueId),
    AtomicMax(ValueId, ValueId),
    AtomicAnd(ValueId, ValueId),
    AtomicOr(ValueId, ValueId),
    AtomicXor(ValueId, ValueId),
    AtomicExch(ValueId, ValueId),
    AtomicCas(ValueId, ValueId, ValueId), // ptr, compare, value

    // === Address computation ===
    GetElementPtr(ValueId, Vec<ValueId>),
    PtrToInt(ValueId),
    IntToPtr(ValueId, GpuType),

    // === GPU Intrinsics ===
    ThreadIdX,
    ThreadIdY,
    ThreadIdZ,
    BlockIdX,
    BlockIdY,
    BlockIdZ,
    BlockDimX,
    BlockDimY,
    BlockDimZ,
    GridDimX,
    GridDimY,
    GridDimZ,

    WarpId,
    LaneId,
    WarpSize,

    // === Synchronization ===
    SyncThreads,   // Block-level barrier
    SyncWarp(u32), // Warp-level sync (mask)
    MemoryFence(MemorySpace),

    // === Warp operations ===
    WarpShuffle(ValueId, ValueId),     // value, lane
    WarpShuffleUp(ValueId, ValueId),   // value, delta
    WarpShuffleDown(ValueId, ValueId), // value, delta
    WarpShuffleXor(ValueId, ValueId),  // value, mask
    WarpVote(WarpVoteOp, ValueId),     // all, any, ballot
    WarpReduce(WarpReduceOp, ValueId), // sum, min, max
    WarpMatch(ValueId),                // Find matching lanes

    // === Texture/Surface ===
    TexFetch(ValueId, ValueId),            // texture, coord
    TexFetch2D(ValueId, ValueId, ValueId), // texture, x, y
    SurfRead(ValueId, ValueId),            // surface, coord
    SurfWrite(ValueId, ValueId, ValueId),  // surface, coord, value

    // === Control flow ===
    Phi(Vec<(BlockId, ValueId)>),
    Select(ValueId, ValueId, ValueId), // cond, true, false

    // === Function call ===
    Call(String, Vec<ValueId>),

    // === Parameter ===
    Param(u32),

    // === Shared memory ===
    SharedAddr(String), // Get address of shared memory variable

    // === Bio/Quaternion Operations (from Quaternionic Syntax preprint) ===
    /// Quaternion multiplication (Hamilton product)
    /// q1 * q2 is noncommutative
    QuatMul(ValueId, ValueId),

    /// Quaternion conjugate: q* = w - xi - yj - zk
    QuatConj(ValueId),

    /// Quaternion norm squared: |q|² = w² + x² + y² + z²
    QuatNormSq(ValueId),

    /// Quaternion normalize to unit: q / |q|
    QuatNormalize(ValueId),

    /// Quaternion SLERP (spherical linear interpolation)
    /// slerp(q1, q2, t) for t ∈ [0, 1]
    QuatSlerp(ValueId, ValueId, ValueId),

    /// DNA base complement (A↔T, C↔G)
    DnaComplement(ValueId),

    /// GF(4) addition (characteristic 2 field)
    Gf4Add(ValueId, ValueId),

    /// GF(4) multiplication (using α² + α + 1 = 0)
    Gf4Mul(ValueId, ValueId),

    /// Transmission channel composition (quaternion product for info channels)
    TransmissionCompose(ValueId, ValueId),

    /// Transmission distortion with renormalization
    TransmissionDistort(ValueId, ValueId, ValueId, ValueId, ValueId), // trans, dg, dt, dp, de

    // === Cooperative Groups (CUDA 9.0+ / PTX 6.0+) ===
    /// Get this thread's cooperative group at specified scope
    /// Returns a group handle for subsequent operations
    CoopThisGroup(CooperativeScope),

    /// Get group size (number of threads in the group)
    CoopGroupSize(ValueId), // group handle

    /// Get thread rank within group (0 to size-1)
    CoopThreadRank(ValueId), // group handle

    /// Check if this thread is the group leader (rank 0)
    CoopIsLeader(ValueId), // group handle

    /// Synchronize all threads in the group
    CoopSync(ValueId), // group handle

    /// Collective shuffle within group
    /// Broadcasts value from src_rank to all threads
    CoopShfl(ValueId, ValueId, ValueId), // group, value, src_rank

    /// Shuffle with index (each thread gets value from thread at idx)
    CoopShflIdx(ValueId, ValueId, ValueId), // group, value, idx

    /// Shuffle up (get value from thread rank - delta)
    CoopShflUp(ValueId, ValueId, ValueId), // group, value, delta

    /// Shuffle down (get value from thread rank + delta)
    CoopShflDown(ValueId, ValueId, ValueId), // group, value, delta

    /// Shuffle XOR (get value from thread rank ^ mask)
    CoopShflXor(ValueId, ValueId, ValueId), // group, value, mask

    /// Collective reduce within group
    CoopReduce(ValueId, ValueId, CoopReduceOp), // group, value, op

    /// Inclusive scan within group
    CoopInclusiveScan(ValueId, ValueId, CoopReduceOp), // group, value, op

    /// Exclusive scan within group
    CoopExclusiveScan(ValueId, ValueId, CoopReduceOp), // group, value, op

    /// Collective ballot (bitmask of predicate across group)
    CoopBallot(ValueId, ValueId), // group, predicate

    /// Check if all threads in group have predicate true
    CoopAll(ValueId, ValueId), // group, predicate

    /// Check if any thread in group has predicate true
    CoopAny(ValueId, ValueId), // group, predicate

    /// Partition group into tiles of specified size
    CoopPartitionTiled(ValueId, u32), // group, tile_size -> new group

    /// Partition group by predicate (binary partition)
    CoopPartitionBinary(ValueId, ValueId), // group, predicate -> new group

    /// Partition group by label (threads with same label together)
    CoopPartitionLabeled(ValueId, ValueId), // group, label -> new group

    /// Get coalesced threads (active threads in current warp)
    CoopCoalescedThreads,

    /// Elect a single leader thread from the group
    CoopElect(ValueId), // group -> bool (true for elected thread)

    /// Memory fence at group scope
    CoopMemoryFence(ValueId), // group

    // === Debug/Profiling Operations ===
    /// Printf from GPU (CUDA vprintf)
    /// Format string is stored in constant memory, args are passed via buffer
    /// Printf(format_string_id, args_buffer_ptr)
    Printf(u32, Vec<ValueId>), // format string constant index, argument values

    /// Assert condition (trap if false)
    Assert(ValueId, Option<u32>), // condition, optional message string index

    /// Trigger GPU trap (unconditional)
    Trap,

    /// Software breakpoint (for debugger)
    Brkpt,

    /// Read per-SM clock counter (low 32 bits or full 64 bits)
    Clock,

    /// Read global timer (nanoseconds since GPU reset)
    GlobalTimer,

    /// Record performance monitoring event
    PmEvent(u32), // event id
}

/// Warp vote operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpVoteOp {
    All,    // All lanes true?
    Any,    // Any lane true?
    Ballot, // Bitmask of true lanes
    Eq,     // All lanes same value?
}

/// Warp reduce operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpReduceOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
}

// ============================================================================
// Cooperative Groups (CUDA 9.0+ / PTX 6.0+)
// ============================================================================

/// Cooperative group scope - defines the set of threads that participate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CooperativeScope {
    /// Single thread (trivial group)
    Thread,
    /// Warp-level (32 threads on NVIDIA, 64 on AMD)
    Warp,
    /// Thread block level
    Block,
    /// Multiple blocks (cluster, requires sm_90+)
    Cluster,
    /// Entire grid (requires cooperative launch)
    Grid,
    /// Coalesced threads (dynamically formed from active threads)
    Coalesced,
    /// Tiled partition (static subdivision of parent group)
    TiledPartition(u32), // tile size: 1, 2, 4, 8, 16, 32
}

impl CooperativeScope {
    /// Get the maximum size for this scope
    pub fn max_size(&self) -> Option<u32> {
        match self {
            CooperativeScope::Thread => Some(1),
            CooperativeScope::Warp => Some(32),
            CooperativeScope::TiledPartition(n) => Some(*n),
            _ => None, // Dynamic or hardware-dependent
        }
    }

    /// Check if this scope requires cooperative launch
    pub fn requires_cooperative_launch(&self) -> bool {
        matches!(self, CooperativeScope::Grid)
    }

    /// Check if this scope requires sm_90+ (Hopper)
    pub fn requires_cluster_support(&self) -> bool {
        matches!(self, CooperativeScope::Cluster)
    }
}

/// Cooperative group handle (runtime representation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CooperativeGroupId(pub u32);

/// Cooperative group partition type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionType {
    /// Partition by tile size
    Tiled(u32),
    /// Partition by label (threads with same label)
    Labeled,
    /// Binary partition (predicate-based)
    Binary,
}

/// Cooperative group reduce operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopReduceOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
    Mul,
}

/// GPU terminator
#[derive(Debug, Clone)]
pub enum GpuTerminator {
    /// Unconditional branch
    Br(BlockId),

    /// Conditional branch
    CondBr(ValueId, BlockId, BlockId),

    /// Return from kernel (void)
    ReturnVoid,

    /// Return value (device function only)
    Return(ValueId),

    /// Unreachable (after divergent exit)
    Unreachable,
}

impl GpuModule {
    pub fn new(name: impl Into<String>, target: GpuTarget) -> Self {
        Self {
            name: name.into(),
            kernels: FxHashMap::default(),
            device_functions: FxHashMap::default(),
            constants: Vec::new(),
            target,
        }
    }

    pub fn add_kernel(&mut self, kernel: GpuKernel) {
        self.kernels.insert(kernel.name.clone(), kernel);
    }

    pub fn add_device_function(&mut self, func: GpuFunction) {
        self.device_functions.insert(func.name.clone(), func);
    }

    pub fn add_constant(&mut self, constant: GpuConstant) {
        self.constants.push(constant);
    }

    /// Get the total number of functions (kernels + device functions)
    pub fn function_count(&self) -> usize {
        self.kernels.len() + self.device_functions.len()
    }
}

impl GpuKernel {
    /// Create a new empty kernel
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            shared_memory: Vec::new(),
            blocks: Vec::new(),
            entry: BlockId(0),
            max_threads: None,
            shared_mem_size: 0,
        }
    }

    /// Add a parameter to the kernel
    pub fn add_param(&mut self, param: GpuParam) {
        self.params.push(param);
    }

    /// Add a shared memory declaration
    pub fn add_shared_memory(&mut self, decl: SharedMemDecl) {
        self.shared_mem_size += decl.elem_type.size_bytes() * decl.size;
        self.shared_memory.push(decl);
    }

    /// Add a basic block
    pub fn add_block(&mut self, block: GpuBlock) {
        self.blocks.push(block);
    }

    /// Get the number of parameters
    pub fn param_count(&self) -> usize {
        self.params.len()
    }
}

impl GpuFunction {
    /// Create a new empty device function
    pub fn new(name: impl Into<String>, return_type: GpuType) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            return_type,
            blocks: Vec::new(),
            entry: BlockId(0),
            inline: false,
        }
    }

    /// Add a parameter to the function
    pub fn add_param(&mut self, param: GpuParam) {
        self.params.push(param);
    }

    /// Add a basic block
    pub fn add_block(&mut self, block: GpuBlock) {
        self.blocks.push(block);
    }
}

impl GpuBlock {
    /// Create a new empty block
    pub fn new(id: BlockId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            instructions: Vec::new(),
            terminator: GpuTerminator::Unreachable,
        }
    }

    /// Add an instruction to the block
    pub fn add_instruction(&mut self, value_id: ValueId, op: GpuOp) {
        self.instructions.push((value_id, op));
    }

    /// Set the terminator for the block
    pub fn set_terminator(&mut self, terminator: GpuTerminator) {
        self.terminator = terminator;
    }
}

/// Builder for GPU modules
pub struct GpuModuleBuilder {
    module: GpuModule,
    next_value_id: u32,
    next_block_id: u32,
}

impl GpuModuleBuilder {
    pub fn new(name: impl Into<String>, target: GpuTarget) -> Self {
        Self {
            module: GpuModule::new(name, target),
            next_value_id: 0,
            next_block_id: 0,
        }
    }

    /// Get the next value ID
    pub fn next_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Get the next block ID
    pub fn next_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Add a kernel to the module
    pub fn add_kernel(&mut self, kernel: GpuKernel) {
        self.module.add_kernel(kernel);
    }

    /// Add a device function to the module
    pub fn add_device_function(&mut self, func: GpuFunction) {
        self.module.add_device_function(func);
    }

    /// Build the module
    pub fn build(self) -> GpuModule {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_type_size() {
        assert_eq!(GpuType::Void.size_bytes(), 0);
        assert_eq!(GpuType::Bool.size_bytes(), 1);
        assert_eq!(GpuType::I32.size_bytes(), 4);
        assert_eq!(GpuType::I64.size_bytes(), 8);
        assert_eq!(GpuType::F32.size_bytes(), 4);
        assert_eq!(GpuType::F64.size_bytes(), 8);
        assert_eq!(GpuType::Vec2(Box::new(GpuType::F32)).size_bytes(), 8);
        assert_eq!(GpuType::Vec3(Box::new(GpuType::F32)).size_bytes(), 12);
        assert_eq!(GpuType::Vec4(Box::new(GpuType::F32)).size_bytes(), 16);
        assert_eq!(GpuType::Array(Box::new(GpuType::F32), 10).size_bytes(), 40);
    }

    #[test]
    fn test_gpu_type_properties() {
        assert!(GpuType::F32.is_float());
        assert!(GpuType::F64.is_float());
        assert!(!GpuType::I32.is_float());

        assert!(GpuType::I32.is_signed());
        assert!(!GpuType::U32.is_signed());

        assert!(GpuType::U32.is_unsigned());
        assert!(!GpuType::I32.is_unsigned());

        assert!(GpuType::I32.is_integer());
        assert!(GpuType::U64.is_integer());
        assert!(!GpuType::F32.is_integer());
    }

    #[test]
    fn test_gpu_module_creation() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let kernel = GpuKernel::new("my_kernel");
        module.add_kernel(kernel);

        assert_eq!(module.kernels.len(), 1);
        assert!(module.kernels.contains_key("my_kernel"));
    }

    #[test]
    fn test_gpu_kernel_building() {
        let mut kernel = GpuKernel::new("add_one");

        kernel.add_param(GpuParam {
            name: "data".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        kernel.add_shared_memory(SharedMemDecl {
            name: "cache".to_string(),
            elem_type: GpuType::F32,
            size: 256,
            align: 4,
        });

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        assert_eq!(kernel.param_count(), 1);
        assert_eq!(kernel.shared_mem_size, 256 * 4);
        assert_eq!(kernel.blocks.len(), 1);
    }

    #[test]
    fn test_gpu_target_display() {
        let cuda = GpuTarget::Cuda {
            compute_capability: (8, 6),
        };
        assert_eq!(format!("{}", cuda), "CUDA sm_86");

        let vulkan = GpuTarget::Vulkan { version: (1, 2) };
        assert_eq!(format!("{}", vulkan), "Vulkan 1.2");
    }
}
