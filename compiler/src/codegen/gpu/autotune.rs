//! GPU Kernel Auto-Tuning
//!
//! Static analysis-based auto-tuning for GPU kernels to select optimal launch
//! configurations based on kernel characteristics and NVIDIA's occupancy model.
//!
//! # Architecture
//!
//! ```text
//! GpuKernel → KernelAnalyzer → KernelProfile → AutoTuner → TunedConfig
//!                  │                               │
//!           InstructionMix              OccupancyCalculator
//! ```
//!
//! # Features
//!
//! - Pattern detection (elementwise, reduction, matmul, stencil, etc.)
//! - Memory access pattern analysis (coalesced, strided, random)
//! - Architecture-aware occupancy calculation (Turing through Blackwell)
//! - Block shape optimization (1D, 2D, 3D)
//! - Tile configuration for tensor core operations
//!
//! # Usage
//!
//! ```ignore
//! use sounio::codegen::gpu::autotune::{AutoTuner, AutoTuneConfig};
//!
//! let tuner = AutoTuner::new(AutoTuneConfig::default());
//! let config = tuner.tune_kernel(&kernel);
//! println!("Recommended block size: {}x{}x{}",
//!     config.block_shape.x, config.block_shape.y, config.block_shape.z);
//! ```

use std::fmt;

use super::ir::{CudaArch, GpuKernel, GpuModule, GpuOp, GpuTarget, GpuType};

// ============================================================================
// Core Types
// ============================================================================

/// Computational pattern of a kernel
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelPattern {
    /// Element-wise operations (map pattern)
    ElementWise,
    /// Reduction pattern (sum, min, max, etc.)
    Reduction,
    /// Stencil computation (convolution, PDE solvers)
    Stencil { radius: u32 },
    /// Matrix multiplication
    MatMul { m: u32, n: u32, k: u32 },
    /// Prefix sum / scan operations
    Scan,
    /// Histogram computation
    Histogram,
    /// Sparse matrix operations
    Sparse,
    /// General / unknown pattern
    General,
}

impl fmt::Display for KernelPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelPattern::ElementWise => write!(f, "elementwise"),
            KernelPattern::Reduction => write!(f, "reduction"),
            KernelPattern::Stencil { radius } => write!(f, "stencil(r={})", radius),
            KernelPattern::MatMul { m, n, k } => write!(f, "matmul({}x{}x{})", m, n, k),
            KernelPattern::Scan => write!(f, "scan"),
            KernelPattern::Histogram => write!(f, "histogram"),
            KernelPattern::Sparse => write!(f, "sparse"),
            KernelPattern::General => write!(f, "general"),
        }
    }
}

/// Memory access pattern
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPattern {
    /// All threads access consecutive addresses (optimal)
    Coalesced,
    /// Threads access with fixed stride
    Strided { stride: u32 },
    /// Unpredictable access pattern
    Random,
    /// 2D tiled access pattern
    Tiled2D { tile_x: u32, tile_y: u32 },
    /// Mixed or unknown pattern
    Mixed,
}

impl fmt::Display for MemoryPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryPattern::Coalesced => write!(f, "coalesced"),
            MemoryPattern::Strided { stride } => write!(f, "strided(s={})", stride),
            MemoryPattern::Random => write!(f, "random"),
            MemoryPattern::Tiled2D { tile_x, tile_y } => {
                write!(f, "tiled2d({}x{})", tile_x, tile_y)
            }
            MemoryPattern::Mixed => write!(f, "mixed"),
        }
    }
}

/// Instruction mix breakdown for a kernel
#[derive(Debug, Clone, Default)]
pub struct InstructionMix {
    pub int_ops: u32,
    pub fp32_ops: u32,
    pub fp64_ops: u32,
    pub fp16_ops: u32,
    pub tensor_ops: u32,
    pub memory_ops: u32,
    pub sync_ops: u32,
    pub sfu_ops: u32,
    pub control_ops: u32,
    pub atomic_ops: u32,
    pub warp_ops: u32,
    pub coop_ops: u32,
    pub total: u32,
}

impl InstructionMix {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compute_intensity(&self) -> f64 {
        let flops =
            (self.fp32_ops + self.fp64_ops * 2 + self.fp16_ops + self.tensor_ops * 512) as f64;
        let mem_ops = self.memory_ops.max(1) as f64;
        flops / mem_ops
    }

    pub fn is_memory_bound(&self) -> bool {
        self.compute_intensity() < 10.0
    }
    pub fn is_compute_bound(&self) -> bool {
        self.compute_intensity() >= 10.0
    }
    pub fn uses_tensor_cores(&self) -> bool {
        self.tensor_ops > 0
    }
}

/// Profile of a kernel for auto-tuning decisions
#[derive(Debug, Clone)]
pub struct KernelProfile {
    pub name: String,
    pub pattern: KernelPattern,
    pub memory_pattern: MemoryPattern,
    pub instruction_mix: InstructionMix,
    pub estimated_registers: u32,
    pub static_shared_mem: u32,
    pub uses_tensor_cores: bool,
    pub primary_dtype: GpuType,
    pub block_count: u32,
    pub loop_depth: u32,
}

impl KernelProfile {
    pub fn is_simple(&self) -> bool {
        self.estimated_registers <= 32 && self.static_shared_mem <= 4096 && self.block_count <= 4
    }

    pub fn is_complex(&self) -> bool {
        self.estimated_registers > 64 || self.loop_depth > 2 || self.instruction_mix.total > 500
    }
}

/// Block shape for kernel launch configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockShape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl BlockShape {
    pub fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }
    pub fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
    pub fn total_threads(&self) -> u32 {
        self.x * self.y * self.z
    }
    pub fn warps(&self) -> u32 {
        self.total_threads().div_ceil(32)
    }
    pub fn is_warp_aligned(&self) -> bool {
        self.total_threads().is_multiple_of(32)
    }
}

impl Default for BlockShape {
    fn default() -> Self {
        Self::new_1d(256)
    }
}

impl fmt::Display for BlockShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.z == 1 && self.y == 1 {
            write!(f, "{}", self.x)
        } else if self.z == 1 {
            write!(f, "{}x{}", self.x, self.y)
        } else {
            write!(f, "{}x{}x{}", self.x, self.y, self.z)
        }
    }
}

/// Tile configuration for tensor core operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub pipeline_stages: u32,
    pub swizzled: bool,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            pipeline_stages: 2,
            swizzled: false,
        }
    }
}

/// Occupancy information
#[derive(Debug, Clone)]
pub struct OccupancyInfo {
    pub occupancy: f64,
    pub active_warps: u32,
    pub max_warps: u32,
    pub active_blocks: u32,
    pub limiting_factor: OccupancyLimiter,
}

/// What factor limits occupancy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLimiter {
    Blocks,
    Warps,
    Registers,
    SharedMemory,
    ThreadsPerBlock,
}

impl fmt::Display for OccupancyLimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OccupancyLimiter::Blocks => write!(f, "max blocks/SM"),
            OccupancyLimiter::Warps => write!(f, "max warps/SM"),
            OccupancyLimiter::Registers => write!(f, "register file"),
            OccupancyLimiter::SharedMemory => write!(f, "shared memory"),
            OccupancyLimiter::ThreadsPerBlock => write!(f, "threads/block"),
        }
    }
}

/// Tuning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuningStrategy {
    MaxOccupancy,
    BalancedOccupancyIlp,
    MemoryThroughput,
    TensorCoreOptimal,
    MinSharedMem,
    SafeDefaults,
}

impl fmt::Display for TuningStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TuningStrategy::MaxOccupancy => write!(f, "max-occupancy"),
            TuningStrategy::BalancedOccupancyIlp => write!(f, "balanced-ilp"),
            TuningStrategy::MemoryThroughput => write!(f, "memory-throughput"),
            TuningStrategy::TensorCoreOptimal => write!(f, "tensor-core"),
            TuningStrategy::MinSharedMem => write!(f, "min-shared-mem"),
            TuningStrategy::SafeDefaults => write!(f, "safe-defaults"),
        }
    }
}

/// Complete tuned configuration for a kernel
#[derive(Debug, Clone)]
pub struct TunedConfig {
    pub block_shape: BlockShape,
    pub shared_mem_bytes: u32,
    pub tile_config: Option<TileConfig>,
    pub occupancy: OccupancyInfo,
    pub confidence: f64,
    pub strategy: TuningStrategy,
    pub rationale: String,
}

// ============================================================================
// Kernel Analyzer
// ============================================================================

/// Static analyzer for GPU kernels
pub struct KernelAnalyzer;

impl KernelAnalyzer {
    pub fn analyze(kernel: &GpuKernel) -> KernelProfile {
        let instruction_mix = Self::analyze_instruction_mix(kernel);
        let pattern = Self::detect_pattern(kernel, &instruction_mix);
        let memory_pattern = Self::detect_memory_pattern(kernel);
        let estimated_registers = Self::estimate_registers(kernel, &instruction_mix);
        let primary_dtype = Self::detect_primary_dtype(kernel);

        KernelProfile {
            name: kernel.name.clone(),
            pattern,
            memory_pattern,
            instruction_mix,
            estimated_registers,
            static_shared_mem: kernel.shared_mem_size,
            uses_tensor_cores: Self::uses_tensor_cores(kernel),
            primary_dtype,
            block_count: kernel.blocks.len() as u32,
            loop_depth: Self::estimate_loop_depth(kernel),
        }
    }

    pub fn analyze_instruction_mix(kernel: &GpuKernel) -> InstructionMix {
        let mut mix = InstructionMix::new();
        for block in &kernel.blocks {
            for (_id, op) in &block.instructions {
                Self::categorize_op(op, &mut mix);
            }
        }
        mix.total = mix.int_ops
            + mix.fp32_ops
            + mix.fp64_ops
            + mix.fp16_ops
            + mix.tensor_ops
            + mix.memory_ops
            + mix.sync_ops
            + mix.sfu_ops
            + mix.control_ops
            + mix.atomic_ops
            + mix.warp_ops
            + mix.coop_ops;
        mix
    }

    fn categorize_op(op: &GpuOp, mix: &mut InstructionMix) {
        match op {
            GpuOp::Add(..)
            | GpuOp::Sub(..)
            | GpuOp::Mul(..)
            | GpuOp::Div(..)
            | GpuOp::Rem(..)
            | GpuOp::Neg(..) => mix.int_ops += 1,
            GpuOp::Shl(..)
            | GpuOp::Shr(..)
            | GpuOp::LShr(..)
            | GpuOp::BitAnd(..)
            | GpuOp::BitOr(..)
            | GpuOp::BitXor(..)
            | GpuOp::BitNot(..)
            | GpuOp::PopCount(..)
            | GpuOp::Clz(..)
            | GpuOp::Ctz(..) => mix.int_ops += 1,
            GpuOp::FAdd(..)
            | GpuOp::FSub(..)
            | GpuOp::FMul(..)
            | GpuOp::FDiv(..)
            | GpuOp::FNeg(..)
            | GpuOp::FMulAdd(..) => mix.fp32_ops += 1,
            GpuOp::Eq(..)
            | GpuOp::Ne(..)
            | GpuOp::Lt(..)
            | GpuOp::Le(..)
            | GpuOp::Gt(..)
            | GpuOp::Ge(..) => mix.int_ops += 1,
            GpuOp::FEq(..)
            | GpuOp::FNe(..)
            | GpuOp::FLt(..)
            | GpuOp::FLe(..)
            | GpuOp::FGt(..)
            | GpuOp::FGe(..) => mix.fp32_ops += 1,
            GpuOp::FastSin(..)
            | GpuOp::FastCos(..)
            | GpuOp::FastExp(..)
            | GpuOp::FastLog(..)
            | GpuOp::FastSqrt(..)
            | GpuOp::FastRsqrt(..) => mix.sfu_ops += 1,
            GpuOp::Load(..) | GpuOp::Store(..) => mix.memory_ops += 1,
            GpuOp::AtomicAdd(..)
            | GpuOp::AtomicSub(..)
            | GpuOp::AtomicMin(..)
            | GpuOp::AtomicMax(..)
            | GpuOp::AtomicAnd(..)
            | GpuOp::AtomicOr(..)
            | GpuOp::AtomicXor(..)
            | GpuOp::AtomicExch(..)
            | GpuOp::AtomicCas(..) => mix.atomic_ops += 1,
            GpuOp::SyncThreads | GpuOp::SyncWarp(..) | GpuOp::MemoryFence(..) => mix.sync_ops += 1,
            GpuOp::WarpShuffle(..)
            | GpuOp::WarpShuffleUp(..)
            | GpuOp::WarpShuffleDown(..)
            | GpuOp::WarpShuffleXor(..)
            | GpuOp::WarpVote(..)
            | GpuOp::WarpReduce(..)
            | GpuOp::WarpMatch(..) => mix.warp_ops += 1,
            GpuOp::CoopThisGroup(..)
            | GpuOp::CoopGroupSize(..)
            | GpuOp::CoopThreadRank(..)
            | GpuOp::CoopIsLeader(..)
            | GpuOp::CoopSync(..)
            | GpuOp::CoopShfl(..)
            | GpuOp::CoopShflIdx(..)
            | GpuOp::CoopShflUp(..)
            | GpuOp::CoopShflDown(..)
            | GpuOp::CoopShflXor(..)
            | GpuOp::CoopReduce(..)
            | GpuOp::CoopInclusiveScan(..)
            | GpuOp::CoopExclusiveScan(..)
            | GpuOp::CoopBallot(..)
            | GpuOp::CoopAll(..)
            | GpuOp::CoopAny(..)
            | GpuOp::CoopPartitionTiled(..)
            | GpuOp::CoopPartitionBinary(..)
            | GpuOp::CoopPartitionLabeled(..)
            | GpuOp::CoopCoalescedThreads
            | GpuOp::CoopElect(..)
            | GpuOp::CoopMemoryFence(..) => mix.coop_ops += 1,
            GpuOp::WgmmaFp4 { .. }
            | GpuOp::WgmmaFp8 { .. }
            | GpuOp::WgmmaBf16 { .. }
            | GpuOp::TransformerEngineFusedAttention { .. }
            | GpuOp::TransformerEngineFp8Gemm { .. }
            | GpuOp::TileMma { .. } => mix.tensor_ops += 1,
            GpuOp::TileLoad { .. } | GpuOp::TileStore { .. } => mix.memory_ops += 1,
            GpuOp::TileSync(..) => mix.sync_ops += 1,
            GpuOp::TileCreate { .. }
            | GpuOp::TileGetElement { .. }
            | GpuOp::TileSetElement { .. }
            | GpuOp::TileFill { .. }
            | GpuOp::TileReduce { .. }
            | GpuOp::TileTranspose(..)
            | GpuOp::TileM(..)
            | GpuOp::TileN(..) => mix.memory_ops += 1,
            GpuOp::F32ToBF16(..)
            | GpuOp::BF16ToF32(..)
            | GpuOp::F32ToF8E4M3(..)
            | GpuOp::F8E4M3ToF32(..)
            | GpuOp::F32ToF8E5M2(..)
            | GpuOp::F8E5M2ToF32(..)
            | GpuOp::F32ToF4(..)
            | GpuOp::F4ToF32(..)
            | GpuOp::PackF8x2(..)
            | GpuOp::UnpackF8x2Low(..)
            | GpuOp::UnpackF8x2High(..)
            | GpuOp::PackF4x2(..)
            | GpuOp::UnpackF4x2Low(..)
            | GpuOp::UnpackF4x2High(..)
            | GpuOp::QuantizeF32ToF8(..)
            | GpuOp::DequantizeF8ToF32(..) => mix.fp16_ops += 1,
            GpuOp::TmaLoadAsync { .. }
            | GpuOp::TmaStoreAsync { .. }
            | GpuOp::TmaMulticastLoad { .. }
            | GpuOp::TmaReduceAsync { .. } => mix.memory_ops += 1,
            GpuOp::Phi(..) | GpuOp::Select(..) => mix.control_ops += 1,
            GpuOp::GetElementPtr(..) | GpuOp::PtrToInt(..) | GpuOp::IntToPtr(..) => {
                mix.int_ops += 1
            }
            GpuOp::And(..) | GpuOp::Or(..) | GpuOp::Xor(..) | GpuOp::Not(..) => mix.int_ops += 1,
            GpuOp::TexFetch(..)
            | GpuOp::TexFetch2D(..)
            | GpuOp::SurfRead(..)
            | GpuOp::SurfWrite(..) => mix.memory_ops += 1,
            GpuOp::Call(..) | GpuOp::Param(..) => mix.control_ops += 1,
            GpuOp::QuatMul(..)
            | GpuOp::QuatConj(..)
            | GpuOp::QuatNormSq(..)
            | GpuOp::QuatNormalize(..)
            | GpuOp::QuatSlerp(..) => mix.sfu_ops += 1,
            GpuOp::DnaComplement(..) | GpuOp::Gf4Add(..) | GpuOp::Gf4Mul(..) => mix.int_ops += 1,
            GpuOp::TransmissionCompose(..) | GpuOp::TransmissionDistort(..) => mix.sfu_ops += 1,
            GpuOp::ClusterBarrier | GpuOp::ClusterArrive(..) | GpuOp::ClusterWait(..) => {
                mix.sync_ops += 1
            }
            GpuOp::NvlinkRead { .. }
            | GpuOp::NvlinkWrite { .. }
            | GpuOp::NvlinkAtomicAdd { .. } => mix.memory_ops += 1,
            GpuOp::DecompressLz4 { .. }
            | GpuOp::DecompressSnappy { .. }
            | GpuOp::DecompressDeflate { .. } => mix.memory_ops += 1,
            _ => {} // Constants, thread IDs, debug ops - don't count
        }
    }

    pub fn detect_pattern(kernel: &GpuKernel, mix: &InstructionMix) -> KernelPattern {
        if mix.tensor_ops > 0 {
            for block in &kernel.blocks {
                for (_id, op) in &block.instructions {
                    if let GpuOp::TileMma {
                        tile_m,
                        tile_n,
                        tile_k,
                        ..
                    } = op
                    {
                        return KernelPattern::MatMul {
                            m: *tile_m,
                            n: *tile_n,
                            k: *tile_k,
                        };
                    }
                    if let GpuOp::WgmmaFp4 { m, n, k, .. }
                    | GpuOp::WgmmaFp8 { m, n, k, .. }
                    | GpuOp::WgmmaBf16 { m, n, k, .. } = op
                    {
                        return KernelPattern::MatMul {
                            m: *m,
                            n: *n,
                            k: *k,
                        };
                    }
                }
            }
            return KernelPattern::MatMul {
                m: 16,
                n: 16,
                k: 16,
            };
        }

        if mix.warp_ops > 0 || mix.coop_ops > 0 {
            for block in &kernel.blocks {
                for (_id, op) in &block.instructions {
                    if matches!(op, GpuOp::WarpReduce(..) | GpuOp::CoopReduce(..)) {
                        return KernelPattern::Reduction;
                    }
                    if matches!(
                        op,
                        GpuOp::CoopInclusiveScan(..) | GpuOp::CoopExclusiveScan(..)
                    ) {
                        return KernelPattern::Scan;
                    }
                }
            }
        }

        if mix.atomic_ops > 5 && mix.atomic_ops as f64 / mix.total.max(1) as f64 > 0.1 {
            return KernelPattern::Histogram;
        }

        if mix.memory_ops > mix.fp32_ops + mix.int_ops
            && mix.sync_ops > 0
            && kernel.shared_mem_size > 0
        {
            return KernelPattern::Stencil { radius: 1 };
        }

        if mix.sync_ops == 0
            && mix.atomic_ops == 0
            && mix.warp_ops == 0
            && mix.coop_ops == 0
            && kernel.blocks.len() <= 2
        {
            return KernelPattern::ElementWise;
        }

        KernelPattern::General
    }

    pub fn detect_memory_pattern(kernel: &GpuKernel) -> MemoryPattern {
        let mut has_tile_ops = false;
        let mut tile_dims = (0u32, 0u32);

        for block in &kernel.blocks {
            for (_id, op) in &block.instructions {
                if let GpuOp::TileCreate { tile_m, tile_n, .. } = op {
                    has_tile_ops = true;
                    tile_dims = (*tile_m, *tile_n);
                }
                if matches!(op, GpuOp::TileLoad { .. } | GpuOp::TileStore { .. }) {
                    has_tile_ops = true;
                }
            }
        }

        if has_tile_ops && tile_dims.0 > 0 {
            return MemoryPattern::Tiled2D {
                tile_x: tile_dims.0,
                tile_y: tile_dims.1,
            };
        }

        if kernel.shared_mem_size == 0 && kernel.blocks.len() <= 2 {
            return MemoryPattern::Coalesced;
        }

        MemoryPattern::Mixed
    }

    pub fn estimate_registers(kernel: &GpuKernel, mix: &InstructionMix) -> u32 {
        let param_regs = kernel.params.len() as u32 * 2;
        let mut value_count = 0u32;
        for block in &kernel.blocks {
            value_count += block.instructions.len() as u32;
        }
        let value_regs = (value_count as f64 * 1.5) as u32;
        let tensor_regs = if mix.tensor_ops > 0 { 32 } else { 0 };
        (param_regs + value_regs + tensor_regs).clamp(16, 255)
    }

    pub fn detect_primary_dtype(kernel: &GpuKernel) -> GpuType {
        for param in &kernel.params {
            match &param.ty {
                GpuType::F32 | GpuType::F16 | GpuType::BF16 | GpuType::F64 => {
                    return param.ty.clone();
                }
                GpuType::Ptr(inner, _) => {
                    if matches!(
                        inner.as_ref(),
                        GpuType::F32 | GpuType::F16 | GpuType::BF16 | GpuType::F64
                    ) {
                        return *inner.clone();
                    }
                }
                _ => {}
            }
        }
        GpuType::F32
    }

    pub fn uses_tensor_cores(kernel: &GpuKernel) -> bool {
        for block in &kernel.blocks {
            for (_id, op) in &block.instructions {
                if matches!(
                    op,
                    GpuOp::WgmmaFp4 { .. }
                        | GpuOp::WgmmaFp8 { .. }
                        | GpuOp::WgmmaBf16 { .. }
                        | GpuOp::TileMma { .. }
                        | GpuOp::TransformerEngineFusedAttention { .. }
                        | GpuOp::TransformerEngineFp8Gemm { .. }
                ) {
                    return true;
                }
            }
        }
        false
    }

    fn estimate_loop_depth(kernel: &GpuKernel) -> u32 {
        if kernel.blocks.len() <= 2 {
            0
        } else if kernel.blocks.len() <= 5 {
            1
        } else {
            2
        }
    }
}

// ============================================================================
// Occupancy Calculator
// ============================================================================

/// Architecture-specific constants for occupancy calculation
#[derive(Debug, Clone)]
pub struct ArchConstants {
    pub sm_version: u32,
    pub max_warps_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub registers_per_sm: u32,
    pub register_alloc_granularity: u32,
    pub shared_memory_per_sm: u32,
    pub shared_alloc_granularity: u32,
    pub max_threads_per_block: u32,
    pub warp_size: u32,
}

impl ArchConstants {
    pub fn from_cuda_arch(arch: CudaArch) -> Self {
        match arch {
            CudaArch::Turing => Self {
                sm_version: 75,
                max_warps_per_sm: 32,
                max_blocks_per_sm: 16,
                registers_per_sm: 65536,
                register_alloc_granularity: 256,
                shared_memory_per_sm: 64 * 1024,
                shared_alloc_granularity: 256,
                max_threads_per_block: 1024,
                warp_size: 32,
            },
            CudaArch::Ampere => Self {
                sm_version: 80,
                max_warps_per_sm: 64,
                max_blocks_per_sm: 32,
                registers_per_sm: 65536,
                register_alloc_granularity: 256,
                shared_memory_per_sm: 164 * 1024,
                shared_alloc_granularity: 128,
                max_threads_per_block: 1024,
                warp_size: 32,
            },
            CudaArch::Ada => Self {
                sm_version: 89,
                max_warps_per_sm: 48,
                max_blocks_per_sm: 24,
                registers_per_sm: 65536,
                register_alloc_granularity: 256,
                shared_memory_per_sm: 100 * 1024,
                shared_alloc_granularity: 128,
                max_threads_per_block: 1024,
                warp_size: 32,
            },
            CudaArch::Hopper => Self {
                sm_version: 90,
                max_warps_per_sm: 64,
                max_blocks_per_sm: 32,
                registers_per_sm: 65536,
                register_alloc_granularity: 256,
                shared_memory_per_sm: 228 * 1024,
                shared_alloc_granularity: 128,
                max_threads_per_block: 1024,
                warp_size: 32,
            },
            CudaArch::Blackwell | CudaArch::BlackwellUltra => Self {
                sm_version: 100,
                max_warps_per_sm: 64,
                max_blocks_per_sm: 32,
                registers_per_sm: 65536,
                register_alloc_granularity: 256,
                shared_memory_per_sm: 256 * 1024,
                shared_alloc_granularity: 128,
                max_threads_per_block: 1024,
                warp_size: 32,
            },
        }
    }

    pub fn from_compute_capability(cc: (u32, u32)) -> Self {
        CudaArch::from_compute_capability(cc)
            .map(Self::from_cuda_arch)
            .unwrap_or_else(|| Self::from_cuda_arch(CudaArch::Turing))
    }
}

/// Calculator for GPU occupancy
pub struct OccupancyCalculator {
    arch: ArchConstants,
}

impl OccupancyCalculator {
    pub fn new(arch: ArchConstants) -> Self {
        Self { arch }
    }
    pub fn from_cuda_arch(arch: CudaArch) -> Self {
        Self::new(ArchConstants::from_cuda_arch(arch))
    }
    pub fn from_compute_capability(cc: (u32, u32)) -> Self {
        Self::new(ArchConstants::from_compute_capability(cc))
    }

    pub fn calculate_occupancy(
        &self,
        block_size: u32,
        registers_per_thread: u32,
        shared_mem_per_block: u32,
    ) -> OccupancyInfo {
        let warps_per_block = block_size.div_ceil(self.arch.warp_size);
        let blocks_by_warps = self.arch.max_warps_per_sm / warps_per_block;

        let regs_per_block = registers_per_thread * block_size;
        let aligned_regs = regs_per_block.div_ceil(self.arch.register_alloc_granularity)
            * self.arch.register_alloc_granularity;
        let blocks_by_regs = if aligned_regs > 0 {
            self.arch.registers_per_sm / aligned_regs
        } else {
            self.arch.max_blocks_per_sm
        };

        let aligned_shared = shared_mem_per_block.div_ceil(self.arch.shared_alloc_granularity)
            * self.arch.shared_alloc_granularity;
        let blocks_by_shared = if aligned_shared > 0 {
            self.arch.shared_memory_per_sm / aligned_shared
        } else {
            self.arch.max_blocks_per_sm
        };

        let active_blocks = blocks_by_warps
            .min(blocks_by_regs)
            .min(blocks_by_shared)
            .min(self.arch.max_blocks_per_sm);
        let active_warps = active_blocks * warps_per_block;
        let occupancy = active_warps as f64 / self.arch.max_warps_per_sm as f64;

        let limiting_factor = if active_blocks == blocks_by_regs {
            OccupancyLimiter::Registers
        } else if active_blocks == blocks_by_shared {
            OccupancyLimiter::SharedMemory
        } else if active_blocks == blocks_by_warps {
            OccupancyLimiter::Warps
        } else {
            OccupancyLimiter::Blocks
        };

        OccupancyInfo {
            occupancy,
            active_warps,
            max_warps: self.arch.max_warps_per_sm,
            active_blocks,
            limiting_factor,
        }
    }

    pub fn find_optimal_block_size(
        &self,
        registers_per_thread: u32,
        shared_mem_per_block: u32,
        strategy: TuningStrategy,
    ) -> BlockShape {
        let candidates: &[u32] = match strategy {
            TuningStrategy::MaxOccupancy => &[64, 128, 256, 512, 1024],
            TuningStrategy::BalancedOccupancyIlp => &[128, 256, 384, 512],
            TuningStrategy::TensorCoreOptimal => &[128, 256],
            TuningStrategy::MinSharedMem => &[128, 256],
            TuningStrategy::MemoryThroughput => &[256, 512],
            TuningStrategy::SafeDefaults => &[256],
        };

        let mut best_size = 256u32;
        let mut best_score = 0.0f64;

        for &size in candidates {
            if size > self.arch.max_threads_per_block {
                continue;
            }
            let occ = self.calculate_occupancy(size, registers_per_thread, shared_mem_per_block);
            let score = match strategy {
                TuningStrategy::BalancedOccupancyIlp => {
                    if occ.occupancy >= 0.5 && occ.occupancy <= 0.75 {
                        occ.occupancy + (size as f64 / 1024.0) * 0.1
                    } else {
                        occ.occupancy
                    }
                }
                _ => occ.occupancy,
            };
            if score > best_score {
                best_score = score;
                best_size = size;
            }
        }
        BlockShape::new_1d(best_size)
    }
}

// ============================================================================
// Auto-Tuner
// ============================================================================

/// Configuration for auto-tuning
#[derive(Debug, Clone)]
pub struct AutoTuneConfig {
    pub target: GpuTarget,
    pub default_strategy: TuningStrategy,
    pub prefer_tensor_cores: bool,
    pub min_occupancy: f64,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            target: GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
            default_strategy: TuningStrategy::BalancedOccupancyIlp,
            prefer_tensor_cores: true,
            min_occupancy: 0.25,
        }
    }
}

/// GPU kernel auto-tuner
pub struct AutoTuner {
    config: AutoTuneConfig,
    occupancy_calc: OccupancyCalculator,
}

impl AutoTuner {
    pub fn new(config: AutoTuneConfig) -> Self {
        let occupancy_calc = match config.target {
            GpuTarget::Cuda {
                compute_capability: cc,
            } => OccupancyCalculator::from_compute_capability(cc),
            _ => OccupancyCalculator::from_cuda_arch(CudaArch::Ampere),
        };
        Self {
            config,
            occupancy_calc,
        }
    }

    pub fn tune_kernel(&self, kernel: &GpuKernel) -> TunedConfig {
        let profile = KernelAnalyzer::analyze(kernel);
        let strategy = self.select_strategy(&profile);
        let block_shape = self.recommend_block_shape(&profile, strategy);
        let shared_mem_bytes = self.recommend_shared_memory(&profile, &block_shape);
        let tile_config = self.recommend_tile_config(&profile);
        let occupancy = self.occupancy_calc.calculate_occupancy(
            block_shape.total_threads(),
            profile.estimated_registers,
            shared_mem_bytes,
        );
        let confidence = self.calculate_confidence(&profile, &occupancy);
        let rationale = self.generate_rationale(&profile, &block_shape, &occupancy, strategy);

        TunedConfig {
            block_shape,
            shared_mem_bytes,
            tile_config,
            occupancy,
            confidence,
            strategy,
            rationale,
        }
    }

    fn select_strategy(&self, profile: &KernelProfile) -> TuningStrategy {
        match &profile.pattern {
            KernelPattern::ElementWise => TuningStrategy::BalancedOccupancyIlp,
            KernelPattern::Reduction => TuningStrategy::MaxOccupancy,
            KernelPattern::MatMul { .. } => {
                if profile.uses_tensor_cores && self.config.prefer_tensor_cores {
                    TuningStrategy::TensorCoreOptimal
                } else {
                    TuningStrategy::MemoryThroughput
                }
            }
            KernelPattern::Scan => TuningStrategy::MaxOccupancy,
            KernelPattern::Stencil { .. } => TuningStrategy::MemoryThroughput,
            KernelPattern::Histogram => TuningStrategy::MinSharedMem,
            KernelPattern::Sparse => TuningStrategy::BalancedOccupancyIlp,
            KernelPattern::General => {
                if profile.is_simple() {
                    TuningStrategy::MaxOccupancy
                } else if profile.is_complex() {
                    TuningStrategy::BalancedOccupancyIlp
                } else {
                    TuningStrategy::SafeDefaults
                }
            }
        }
    }

    fn recommend_block_shape(
        &self,
        profile: &KernelProfile,
        strategy: TuningStrategy,
    ) -> BlockShape {
        match &profile.pattern {
            KernelPattern::MatMul { m, n, .. } => {
                let (bx, by) = match strategy {
                    TuningStrategy::TensorCoreOptimal => {
                        if *m >= 16 && *n >= 16 {
                            (16, 16)
                        } else {
                            (16, 8)
                        }
                    }
                    _ => (16, 16),
                };
                BlockShape::new_2d(bx, by)
            }
            KernelPattern::Stencil { .. } => BlockShape::new_2d(16, 16),
            _ => self.occupancy_calc.find_optimal_block_size(
                profile.estimated_registers,
                profile.static_shared_mem,
                strategy,
            ),
        }
    }

    fn recommend_shared_memory(&self, profile: &KernelProfile, block_shape: &BlockShape) -> u32 {
        let mut shared_mem = profile.static_shared_mem;
        match &profile.pattern {
            KernelPattern::Reduction => {
                shared_mem = shared_mem.max(block_shape.warps() * 4 * 32);
            }
            KernelPattern::Stencil { radius } => {
                let halo_x = block_shape.x + 2 * radius;
                let halo_y = block_shape.y + 2 * radius;
                shared_mem = shared_mem.max(halo_x * halo_y * 4);
            }
            KernelPattern::MatMul { .. } => {
                if profile.uses_tensor_cores {
                    shared_mem = shared_mem.max(16 * 16 * 2 * 2);
                }
            }
            _ => {}
        }
        shared_mem
    }

    fn recommend_tile_config(&self, profile: &KernelProfile) -> Option<TileConfig> {
        if !profile.uses_tensor_cores {
            return None;
        }
        let (tile_m, tile_n, tile_k) = match &profile.pattern {
            KernelPattern::MatMul { m, n, k } => (*m, *n, *k),
            _ => (16, 16, 16),
        };
        let pipeline_stages = match self.config.target {
            GpuTarget::Cuda {
                compute_capability: (major, _),
            } => {
                if major >= 9 {
                    4
                } else {
                    2
                }
            }
            _ => 2,
        };
        Some(TileConfig {
            tile_m,
            tile_n,
            tile_k,
            pipeline_stages,
            swizzled: true,
        })
    }

    fn calculate_confidence(&self, profile: &KernelProfile, occupancy: &OccupancyInfo) -> f64 {
        let mut confidence: f64 = 0.5;
        match &profile.pattern {
            KernelPattern::ElementWise => confidence += 0.3,
            KernelPattern::Reduction => confidence += 0.25,
            KernelPattern::MatMul { .. } => confidence += 0.2,
            KernelPattern::Scan => confidence += 0.2,
            KernelPattern::General => confidence -= 0.1,
            _ => {}
        }
        if occupancy.occupancy >= 0.5 {
            confidence += 0.1;
        } else if occupancy.occupancy < 0.25 {
            confidence -= 0.2;
        }
        if profile.is_simple() {
            confidence += 0.1;
        }
        confidence.clamp(0.0, 1.0)
    }

    fn generate_rationale(
        &self,
        profile: &KernelProfile,
        block_shape: &BlockShape,
        occupancy: &OccupancyInfo,
        strategy: TuningStrategy,
    ) -> String {
        let mut r = format!("Pattern: {}. Strategy: {}. ", profile.pattern, strategy);
        r.push_str(&format!(
            "Block size {} ({} threads, {} warps). ",
            block_shape,
            block_shape.total_threads(),
            block_shape.warps()
        ));
        r.push_str(&format!(
            "Occupancy: {:.0}% ({} warps, limited by {}). ",
            occupancy.occupancy * 100.0,
            occupancy.active_warps,
            occupancy.limiting_factor
        ));
        if profile.uses_tensor_cores {
            r.push_str("Uses tensor cores. ");
        }
        if profile.instruction_mix.is_memory_bound() {
            r.push_str("Memory-bound. ");
        } else if profile.instruction_mix.is_compute_bound() {
            r.push_str("Compute-bound. ");
        }
        r
    }
}

/// Tune all kernels in a module
pub fn tune_module(module: &GpuModule, config: AutoTuneConfig) -> Vec<(String, TunedConfig)> {
    let tuner = AutoTuner::new(config);
    module
        .kernels
        .iter()
        .map(|(name, kernel)| (name.clone(), tuner.tune_kernel(kernel)))
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::gpu::ir::{
        BlockId, CoopReduceOp, CooperativeScope, GpuBlock, GpuParam, GpuTerminator, MemorySpace,
        ValueId,
    };

    fn make_kernel(name: &str, instructions: Vec<(ValueId, GpuOp)>) -> GpuKernel {
        GpuKernel {
            name: name.to_string(),
            params: vec![
                GpuParam {
                    name: "input".to_string(),
                    ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                    space: MemorySpace::Global,
                    restrict: true,
                },
                GpuParam {
                    name: "output".to_string(),
                    ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                    space: MemorySpace::Global,
                    restrict: true,
                },
                GpuParam {
                    name: "n".to_string(),
                    ty: GpuType::I32,
                    space: MemorySpace::Local,
                    restrict: false,
                },
            ],
            shared_memory: vec![],
            blocks: vec![GpuBlock {
                id: BlockId(0),
                label: "entry".to_string(),
                instructions,
                terminator: GpuTerminator::ReturnVoid,
            }],
            entry: BlockId(0),
            max_threads: None,
            shared_mem_size: 0,
        }
    }

    #[test]
    fn test_elementwise_detection() {
        let ops = vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::Param(0)),
            (
                ValueId(2),
                GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]),
            ),
            (ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global)),
            (ValueId(4), GpuOp::FMul(ValueId(3), ValueId(3))),
            (ValueId(5), GpuOp::Param(1)),
            (
                ValueId(6),
                GpuOp::GetElementPtr(ValueId(5), vec![ValueId(0)]),
            ),
            (
                ValueId(7),
                GpuOp::Store(ValueId(6), ValueId(4), MemorySpace::Global),
            ),
        ];
        let kernel = make_kernel("square", ops);
        let profile = KernelAnalyzer::analyze(&kernel);
        assert_eq!(profile.pattern, KernelPattern::ElementWise);
        assert_eq!(profile.memory_pattern, MemoryPattern::Coalesced);
    }

    #[test]
    fn test_reduction_detection() {
        let ops = vec![
            (ValueId(0), GpuOp::CoopThisGroup(CooperativeScope::Block)),
            (ValueId(1), GpuOp::Param(0)),
            (ValueId(2), GpuOp::Load(ValueId(1), MemorySpace::Global)),
            (
                ValueId(3),
                GpuOp::CoopReduce(ValueId(0), ValueId(2), CoopReduceOp::Add),
            ),
        ];
        let kernel = make_kernel("reduce_sum", ops);
        let profile = KernelAnalyzer::analyze(&kernel);
        assert_eq!(profile.pattern, KernelPattern::Reduction);
    }

    #[test]
    fn test_matmul_detection() {
        let ops = vec![(
            ValueId(0),
            GpuOp::TileMma {
                c: ValueId(1),
                a: ValueId(2),
                b: ValueId(3),
                tile_m: 16,
                tile_n: 16,
                tile_k: 16,
            },
        )];
        let kernel = make_kernel("matmul", ops);
        let profile = KernelAnalyzer::analyze(&kernel);
        assert!(matches!(
            profile.pattern,
            KernelPattern::MatMul {
                m: 16,
                n: 16,
                k: 16
            }
        ));
        assert!(profile.uses_tensor_cores);
    }

    #[test]
    fn test_occupancy_calculator() {
        let calc = OccupancyCalculator::from_cuda_arch(CudaArch::Ampere);
        let occ = calc.calculate_occupancy(256, 32, 0);
        assert!(occ.occupancy > 0.5);
        let occ_reg = calc.calculate_occupancy(256, 128, 0);
        assert!(occ_reg.occupancy < occ.occupancy);
        assert_eq!(occ_reg.limiting_factor, OccupancyLimiter::Registers);
    }

    #[test]
    fn test_architecture_differences() {
        let turing = ArchConstants::from_cuda_arch(CudaArch::Turing);
        let hopper = ArchConstants::from_cuda_arch(CudaArch::Hopper);
        assert!(hopper.max_warps_per_sm >= turing.max_warps_per_sm);
        assert!(hopper.shared_memory_per_sm > turing.shared_memory_per_sm);
    }

    #[test]
    fn test_block_shape_2d() {
        let config = AutoTuneConfig {
            target: GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
            ..Default::default()
        };
        let tuner = AutoTuner::new(config);
        let ops = vec![(
            ValueId(0),
            GpuOp::TileMma {
                c: ValueId(1),
                a: ValueId(2),
                b: ValueId(3),
                tile_m: 16,
                tile_n: 16,
                tile_k: 16,
            },
        )];
        let kernel = make_kernel("gemm", ops);
        let tuned = tuner.tune_kernel(&kernel);
        assert!(tuned.block_shape.y > 1 || tuned.block_shape.x == 16);
        assert!(tuned.tile_config.is_some());
    }

    #[test]
    fn test_safe_defaults() {
        let config = AutoTuneConfig {
            target: GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
            default_strategy: TuningStrategy::SafeDefaults,
            ..Default::default()
        };
        let tuner = AutoTuner::new(config);
        let kernel = make_kernel("empty", vec![]);
        let tuned = tuner.tune_kernel(&kernel);
        // Empty kernel is detected as ElementWise pattern, gets BalancedOccupancyIlp strategy
        // Block size should be warp-aligned and reasonable (128-512)
        assert!(tuned.block_shape.is_warp_aligned());
        assert!(
            tuned.block_shape.total_threads() >= 128 && tuned.block_shape.total_threads() <= 512
        );
        assert!(tuned.confidence > 0.0);
    }

    #[test]
    fn test_instruction_mix() {
        let mut mix = InstructionMix::new();
        mix.fp32_ops = 100;
        mix.memory_ops = 10;
        mix.total = 110;
        assert!(mix.is_compute_bound());
        assert!(!mix.is_memory_bound());
    }

    #[test]
    fn test_tune_module() {
        use rustc_hash::FxHashMap;
        let mut kernels = FxHashMap::default();
        kernels.insert(
            "k1".to_string(),
            make_kernel("k1", vec![(ValueId(0), GpuOp::ThreadIdX)]),
        );
        kernels.insert(
            "k2".to_string(),
            make_kernel("k2", vec![(ValueId(0), GpuOp::ThreadIdX)]),
        );
        let module = GpuModule {
            name: "test".to_string(),
            kernels,
            device_functions: FxHashMap::default(),
            constants: vec![],
            target: GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        };
        let results = tune_module(&module, AutoTuneConfig::default());
        assert_eq!(results.len(), 2);
    }
}
