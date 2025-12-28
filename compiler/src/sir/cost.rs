//! SKIR (Sounio Kernel IR) Cost Annotation System
//!
//! This module provides the "language of cost" - declarative performance primitives
//! for GPU and parallel execution. Unlike imperative scheduling directives, these
//! annotations express *intent* and *constraints*, allowing the compiler to generate
//! optimal implementations for different targets.
//!
//! # Design Philosophy
//!
//! Cost annotations follow a layered abstraction:
//!
//! 1. **Declarative Intent**: Express what you want (determinism, precision, locality)
//! 2. **Target-Agnostic**: Same annotations work across CUDA, Metal, Vulkan, CPU
//! 3. **Composable**: Annotations combine naturally without conflicts
//! 4. **Verifiable**: Compiler validates annotations against hardware constraints
//!
//! # The Nine Universal Performance Primitives
//!
//! These primitives capture the essential degrees of freedom in parallel computation:
//!
//! | Primitive | Purpose | Example |
//! |-----------|---------|---------|
//! | `Tile` | Spatial decomposition | 16x16x16 for WMMA |
//! | `Layout` | Memory organization | Row-major, blocked |
//! | `VectorWidth` | SIMD lane count | 8 for AVX-256 |
//! | `SharedCache` | Scratchpad memory | 48KB L1 data |
//! | `Unroll` | Loop transformation | 4x unroll |
//! | `Fuse` | Operation merging | Fuse add+mul |
//! | `ReductionMode` | Associativity semantics | Deterministic |
//! | `Precision` | Numerical accuracy | Mixed FP16/FP32 |
//! | `RngMode` | Randomness control | Counter-based |
//!
//! # Usage
//!
//! ```ignore
//! use sounio::sir::cost::*;
//!
//! // Create a cost envelope for a matrix multiplication kernel
//! let cost = CostEnvelope::new()
//!     .with_tile(Tile::new(16, 16, 16))
//!     .with_layout(Layout::RowMajor)
//!     .with_precision(Precision::Mixed {
//!         compute: ScalarPrecision::Fp16,
//!         accumulate: ScalarPrecision::Fp32
//!     })
//!     .with_shared_cache(SharedCache::bytes(48 * 1024));
//!
//! // Create multiple variants for autotuning
//! let variants = KernelVariantSet::new("matmul")
//!     .add_variant(cost.clone().with_tile(Tile::new(16, 16, 16)))
//!     .add_variant(cost.clone().with_tile(Tile::new(32, 8, 16)))
//!     .add_variant(cost.clone().with_tile(Tile::new(8, 32, 16)));
//! ```
//!
//! # Semantic Guarantees
//!
//! Cost annotations provide semantic guarantees that the compiler must preserve:
//!
//! - `ReductionMode::Deterministic` guarantees bit-exact results across runs
//! - `Precision::Fp64` guarantees IEEE 754 double precision
//! - `RngMode::CounterBased` guarantees reproducible random sequences
//!
//! These guarantees are essential for scientific computing reproducibility.

use std::collections::HashMap;
use std::fmt;

use super::values::FuncId;

// ============================================================================
// Primitive 1: Tiling
// ============================================================================

/// Tile dimensions for spatial decomposition of computation.
///
/// Tiling is the fundamental transformation for GPU programming, enabling:
/// - Shared memory reuse within thread blocks
/// - Tensor Core utilization (requires specific tile shapes)
/// - Cache-efficient memory access patterns
///
/// # Tensor Core Constraints
///
/// For Tensor Core operations, tiles must match hardware requirements:
///
/// | Data Type | Supported Shapes (M, N, K) |
/// |-----------|---------------------------|
/// | FP16/BF16 | 16x16x16, 32x8x16, 8x32x16 |
/// | FP8 | 16x16x32, 32x8x32, 8x32x32 |
/// | FP4 | 16x16x64, 32x8x64, 8x32x64 |
/// | TF32 | 16x16x8, 32x8x8, 8x32x8 |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile {
    /// M dimension (rows of output)
    pub m: u32,
    /// N dimension (columns of output)
    pub n: u32,
    /// K dimension (reduction dimension)
    pub k: u32,
}

impl Tile {
    /// Create a new tile with the given dimensions.
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    /// Create a square tile (m = n = k).
    pub fn square(size: u32) -> Self {
        Self::new(size, size, size)
    }

    /// Create a 2D tile (k = 1, for elementwise operations).
    pub fn d2(m: u32, n: u32) -> Self {
        Self::new(m, n, 1)
    }

    /// Standard 16x16x16 tile for FP16/BF16 Tensor Cores.
    pub fn tensor_core_fp16() -> Self {
        Self::new(16, 16, 16)
    }

    /// Standard 16x16x32 tile for FP8 Tensor Cores.
    pub fn tensor_core_fp8() -> Self {
        Self::new(16, 16, 32)
    }

    /// Standard 16x16x64 tile for FP4 Tensor Cores (Blackwell).
    pub fn tensor_core_fp4() -> Self {
        Self::new(16, 16, 64)
    }

    /// Total number of elements in the tile.
    pub fn elements(&self) -> u32 {
        self.m * self.n * self.k
    }

    /// Check if all dimensions are powers of two.
    pub fn is_power_of_two(&self) -> bool {
        self.m.is_power_of_two() && self.n.is_power_of_two() && self.k.is_power_of_two()
    }

    /// Validate tile dimensions against hardware constraints.
    pub fn validate(&self) -> Result<(), TileError> {
        if !self.is_power_of_two() {
            return Err(TileError::NotPowerOfTwo {
                m: self.m,
                n: self.n,
                k: self.k,
            });
        }
        if self.m > 128 || self.n > 128 || self.k > 128 {
            return Err(TileError::TooLarge {
                m: self.m,
                n: self.n,
                k: self.k,
                max: 128,
            });
        }
        Ok(())
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tile({}x{}x{})", self.m, self.n, self.k)
    }
}

/// Errors during tile validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileError {
    /// Dimensions are not powers of two.
    NotPowerOfTwo { m: u32, n: u32, k: u32 },
    /// Dimensions exceed hardware limits.
    TooLarge { m: u32, n: u32, k: u32, max: u32 },
    /// Shape not supported for the given data type.
    UnsupportedShape { m: u32, n: u32, k: u32, dtype: String },
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileError::NotPowerOfTwo { m, n, k } => {
                write!(
                    f,
                    "tile dimensions must be powers of 2, got {}x{}x{}",
                    m, n, k
                )
            }
            TileError::TooLarge { m, n, k, max } => {
                write!(
                    f,
                    "tile dimensions exceed maximum {}, got {}x{}x{}",
                    max, m, n, k
                )
            }
            TileError::UnsupportedShape { m, n, k, dtype } => {
                write!(
                    f,
                    "tile shape {}x{}x{} not supported for type {}",
                    m, n, k, dtype
                )
            }
        }
    }
}

impl std::error::Error for TileError {}

// ============================================================================
// Primitive 2: Layout
// ============================================================================

/// Memory layout for multi-dimensional data.
///
/// Layout determines how logical indices map to physical memory addresses,
/// affecting cache utilization and vectorization opportunities.
///
/// # Choosing a Layout
///
/// - **RowMajor**: Default for C/C++, good for row-wise iteration
/// - **ColMajor**: Default for Fortran, good for column-wise iteration
/// - **Blocked**: Cache-oblivious, good for matrix operations
/// - **Tiled**: Explicit tile structure, required for Tensor Cores
/// - **Swizzled**: Bank conflict avoidance in shared memory
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C-style): A[i][j] -> A[i * cols + j]
    RowMajor,

    /// Column-major (Fortran-style): A[i][j] -> A[j * rows + i]
    ColMajor,

    /// Blocked layout for cache efficiency.
    /// Data is organized into blocks of size (block_m, block_n).
    Blocked {
        /// Block size in the M dimension.
        block_m: u32,
        /// Block size in the N dimension.
        block_n: u32,
    },

    /// Tiled layout for Tensor Core operations.
    /// Each tile is stored contiguously in memory.
    Tiled {
        /// Tile dimensions.
        tile: Tile,
        /// Layout of elements within a tile.
        inner: Box<Layout>,
    },

    /// Swizzled layout for shared memory bank conflict avoidance.
    /// XOR-based swizzling pattern.
    Swizzled {
        /// Swizzle block width.
        block_m: u32,
        /// Swizzle block height.
        block_n: u32,
        /// XOR mask bits for address transformation.
        xor_mask: u32,
    },

    /// Custom layout with explicit stride.
    Custom {
        /// Stride in the innermost dimension.
        stride_inner: u32,
        /// Stride in the outer dimension.
        stride_outer: u32,
    },
}

impl Layout {
    /// Create a blocked layout with the given block size.
    pub fn blocked(block_m: u32, block_n: u32) -> Self {
        Layout::Blocked { block_m, block_n }
    }

    /// Create a tiled layout with row-major inner layout.
    pub fn tiled(tile: Tile) -> Self {
        Layout::Tiled {
            tile,
            inner: Box::new(Layout::RowMajor),
        }
    }

    /// Create a swizzled layout for shared memory.
    ///
    /// The XOR mask is computed automatically based on bank structure.
    pub fn swizzled(block_m: u32, block_n: u32) -> Self {
        // Compute XOR mask for 32-bank shared memory
        // This creates a pattern that distributes consecutive addresses across banks
        let xor_mask = 0b11111; // 5 bits for 32 banks
        Layout::Swizzled {
            block_m,
            block_n,
            xor_mask,
        }
    }

    /// Check if this layout is compatible with Tensor Core operations.
    pub fn is_tensor_core_compatible(&self) -> bool {
        matches!(
            self,
            Layout::Tiled { .. } | Layout::Swizzled { .. } | Layout::RowMajor | Layout::ColMajor
        )
    }
}

impl Default for Layout {
    fn default() -> Self {
        Layout::RowMajor
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Layout::RowMajor => write!(f, "RowMajor"),
            Layout::ColMajor => write!(f, "ColMajor"),
            Layout::Blocked { block_m, block_n } => write!(f, "Blocked({}x{})", block_m, block_n),
            Layout::Tiled { tile, inner } => write!(f, "Tiled({}, inner={})", tile, inner),
            Layout::Swizzled {
                block_m,
                block_n,
                xor_mask,
            } => {
                write!(
                    f,
                    "Swizzled({}x{}, xor={:#x})",
                    block_m, block_n, xor_mask
                )
            }
            Layout::Custom {
                stride_inner,
                stride_outer,
            } => {
                write!(f, "Custom(inner={}, outer={})", stride_inner, stride_outer)
            }
        }
    }
}

// ============================================================================
// Primitive 3: Vectorization
// ============================================================================

/// SIMD vector width for data-parallel operations.
///
/// Specifies the number of elements processed in parallel by vector instructions.
///
/// # Target-Specific Widths
///
/// | Target | Vector Width | Notes |
/// |--------|--------------|-------|
/// | SSE | 4 (F32) | 128-bit registers |
/// | AVX | 8 (F32) | 256-bit registers |
/// | AVX-512 | 16 (F32) | 512-bit registers |
/// | CUDA | 1-4 | Depends on memory coalescing |
/// | NEON | 4 (F32) | 128-bit ARM SIMD |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorWidth(pub u32);

impl VectorWidth {
    /// Scalar (no vectorization).
    pub const SCALAR: VectorWidth = VectorWidth(1);

    /// SSE-compatible width (128-bit).
    pub const SSE: VectorWidth = VectorWidth(4);

    /// AVX-compatible width (256-bit).
    pub const AVX: VectorWidth = VectorWidth(8);

    /// AVX-512 compatible width (512-bit).
    pub const AVX512: VectorWidth = VectorWidth(16);

    /// Create a vector width.
    pub fn new(width: u32) -> Self {
        Self(width)
    }

    /// Get the width value.
    pub fn width(&self) -> u32 {
        self.0
    }

    /// Check if the width is a power of two.
    pub fn is_valid(&self) -> bool {
        self.0.is_power_of_two() && self.0 <= 64
    }

    /// Get the byte width for a given scalar type size.
    pub fn bytes_for_scalar(&self, scalar_bytes: u32) -> u32 {
        self.0 * scalar_bytes
    }
}

impl Default for VectorWidth {
    fn default() -> Self {
        VectorWidth::SCALAR
    }
}

impl fmt::Display for VectorWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            1 => write!(f, "Scalar"),
            4 => write!(f, "Vec4"),
            8 => write!(f, "Vec8"),
            16 => write!(f, "Vec16"),
            n => write!(f, "Vec{}", n),
        }
    }
}

/// Lane group for cooperative SIMD operations.
///
/// Used for warp-level primitives on GPUs where lanes cooperate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneGroup(pub u32);

impl LaneGroup {
    /// Single lane (no cooperation).
    pub const SINGLE: LaneGroup = LaneGroup(1);

    /// Quarter warp (8 lanes on CUDA).
    pub const QUARTER_WARP: LaneGroup = LaneGroup(8);

    /// Half warp (16 lanes on CUDA).
    pub const HALF_WARP: LaneGroup = LaneGroup(16);

    /// Full warp (32 lanes on CUDA).
    pub const WARP: LaneGroup = LaneGroup(32);

    /// Warp group (128 lanes on Hopper+).
    pub const WARP_GROUP: LaneGroup = LaneGroup(128);

    /// Create a lane group with the given size.
    pub fn new(lanes: u32) -> Self {
        Self(lanes)
    }

    /// Get the number of lanes.
    pub fn lanes(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for LaneGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            1 => write!(f, "SingleLane"),
            8 => write!(f, "QuarterWarp"),
            16 => write!(f, "HalfWarp"),
            32 => write!(f, "Warp"),
            128 => write!(f, "WarpGroup"),
            n => write!(f, "LaneGroup({})", n),
        }
    }
}

// ============================================================================
// Primitive 4: Shared Memory / Cache
// ============================================================================

/// Shared memory or cache allocation for a kernel.
///
/// Shared memory (GPU) or L1 cache (CPU) is a fast scratchpad memory
/// shared among threads in a block or core.
///
/// # Size Constraints
///
/// | Architecture | Max Shared Memory per SM |
/// |--------------|-------------------------|
/// | Volta | 96 KB |
/// | Turing | 64 KB |
/// | Ampere | 164 KB |
/// | Hopper | 228 KB |
/// | Blackwell | 228 KB |
///
/// Actual usable size depends on occupancy requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedCache {
    /// Total bytes allocated.
    pub bytes: usize,
    /// Alignment requirement in bytes.
    pub align: usize,
    /// Whether the allocation is dynamic (sized at launch).
    pub dynamic: bool,
}

impl SharedCache {
    /// Create a static shared memory allocation.
    pub fn bytes(bytes: usize) -> Self {
        Self {
            bytes,
            align: 128, // Default to 128-byte alignment for coalescing
            dynamic: false,
        }
    }

    /// Create a dynamic shared memory allocation.
    pub fn dynamic(max_bytes: usize) -> Self {
        Self {
            bytes: max_bytes,
            align: 128,
            dynamic: true,
        }
    }

    /// Set alignment requirement.
    pub fn with_align(mut self, align: usize) -> Self {
        self.align = align;
        self
    }

    /// Calculate aligned size.
    pub fn aligned_size(&self) -> usize {
        (self.bytes + self.align - 1) & !(self.align - 1)
    }

    /// Check if the allocation fits within the given limit.
    pub fn fits_within(&self, limit: usize) -> bool {
        self.aligned_size() <= limit
    }

    /// Common presets.
    pub fn small() -> Self {
        Self::bytes(16 * 1024) // 16 KB
    }
    pub fn medium() -> Self {
        Self::bytes(48 * 1024) // 48 KB
    }
    pub fn large() -> Self {
        Self::bytes(96 * 1024) // 96 KB
    }
}

impl Default for SharedCache {
    fn default() -> Self {
        Self::bytes(0)
    }
}

impl fmt::Display for SharedCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kb = self.bytes as f64 / 1024.0;
        if self.dynamic {
            write!(f, "SharedCache(dynamic, max={:.1}KB)", kb)
        } else {
            write!(f, "SharedCache({:.1}KB)", kb)
        }
    }
}

// ============================================================================
// Primitive 5: Unrolling
// ============================================================================

/// Loop unrolling factor.
///
/// Unrolling replicates loop body to reduce branch overhead and enable
/// instruction-level parallelism.
///
/// # Trade-offs
///
/// - Higher factors: More ILP, better latency hiding, larger code size
/// - Lower factors: Smaller code, better instruction cache utilization
/// - Full unroll: Eliminates loop overhead, but may cause code bloat
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unroll {
    /// Unroll factor (1 = no unroll, 0 = full unroll).
    pub factor: u32,
    /// Emit cleanup code for non-divisible trip counts.
    pub with_remainder: bool,
}

impl Unroll {
    /// No unrolling (factor = 1).
    pub const NONE: Unroll = Unroll {
        factor: 1,
        with_remainder: false,
    };

    /// Full unrolling (factor = 0 means unroll completely).
    pub const FULL: Unroll = Unroll {
        factor: 0,
        with_remainder: false,
    };

    /// Create an unroll directive with the given factor.
    pub fn new(factor: u32) -> Self {
        Self {
            factor,
            with_remainder: factor > 1,
        }
    }

    /// Create an unroll directive that handles remainder iterations.
    pub fn with_remainder(factor: u32) -> Self {
        Self {
            factor,
            with_remainder: true,
        }
    }

    /// Check if this requests full unrolling.
    pub fn is_full(&self) -> bool {
        self.factor == 0
    }

    /// Check if any unrolling is requested.
    pub fn is_unrolled(&self) -> bool {
        self.factor != 1
    }
}

impl Default for Unroll {
    fn default() -> Self {
        Unroll::NONE
    }
}

impl fmt::Display for Unroll {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.factor {
            0 => write!(f, "Unroll(full)"),
            1 => write!(f, "Unroll(none)"),
            n if self.with_remainder => write!(f, "Unroll({}x, remainder)", n),
            n => write!(f, "Unroll({}x)", n),
        }
    }
}

// ============================================================================
// Primitive 6: Fusion
// ============================================================================

/// Operation fusion directive.
///
/// Fusion combines multiple operations into a single kernel, eliminating
/// intermediate memory traffic and enabling cross-operation optimization.
///
/// # Fusion Types
///
/// - **Horizontal**: Fuse operations on different data (batch processing)
/// - **Vertical**: Fuse producer-consumer chains (elementwise fusion)
/// - **Tile**: Fuse operations within shared memory tiles
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fuse {
    /// Operations to fuse (by their IR identifiers).
    pub ops: Vec<OpId>,
    /// Type of fusion to apply.
    pub fusion_type: FusionType,
    /// Maximum register pressure allowed.
    pub max_registers: Option<u32>,
}

/// Unique identifier for an operation in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub u32);

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "op{}", self.0)
    }
}

/// Type of fusion transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionType {
    /// Fuse operations on different data streams.
    Horizontal,
    /// Fuse producer-consumer operation chains.
    Vertical,
    /// Fuse within tile boundaries.
    Tile,
    /// Automatic selection by the compiler.
    Auto,
}

impl Fuse {
    /// Create a fusion directive for the given operations.
    pub fn new(ops: Vec<OpId>) -> Self {
        Self {
            ops,
            fusion_type: FusionType::Auto,
            max_registers: None,
        }
    }

    /// Specify the fusion type.
    pub fn with_type(mut self, fusion_type: FusionType) -> Self {
        self.fusion_type = fusion_type;
        self
    }

    /// Limit register pressure for the fused kernel.
    pub fn with_max_registers(mut self, max: u32) -> Self {
        self.max_registers = Some(max);
        self
    }

    /// Check if fusion is valid (at least 2 operations).
    pub fn is_valid(&self) -> bool {
        self.ops.len() >= 2
    }
}

impl fmt::Display for Fuse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fuse({:?}, {:?})", self.fusion_type, self.ops)
    }
}

// ============================================================================
// Primitive 7: Reduction Mode
// ============================================================================

/// Semantics for reduction operations.
///
/// Reductions (sum, min, max, etc.) can be computed in different ways with
/// different performance/determinism trade-offs.
///
/// # Scientific Computing Implications
///
/// **Deterministic mode** is essential for:
/// - Reproducible research
/// - Debugging numerical issues
/// - Validation against reference implementations
/// - Regulatory compliance (FDA, etc.)
///
/// **Fast mode** is acceptable for:
/// - Training (where stochasticity is expected)
/// - Interactive exploration
/// - Performance-critical production inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionMode {
    /// Deterministic: Same inputs always produce bit-identical outputs.
    /// Uses sequential or tree reduction with fixed order.
    Deterministic,

    /// Fast: May reorder operations for performance.
    /// Results may vary between runs due to floating-point non-associativity.
    Fast,

    /// Pairwise: Tree reduction with pairwise summation.
    /// Better numerical accuracy than naive sequential sum.
    Pairwise,

    /// Kahan: Compensated summation for high accuracy.
    /// Significantly reduces round-off error accumulation.
    Kahan,

    /// Block-deterministic: Deterministic within blocks, non-deterministic across.
    /// Good compromise for multi-GPU training.
    BlockDeterministic,
}

impl ReductionMode {
    /// Check if this mode guarantees reproducibility.
    pub fn is_deterministic(&self) -> bool {
        matches!(self, ReductionMode::Deterministic | ReductionMode::Kahan)
    }

    /// Check if this mode prioritizes accuracy.
    pub fn is_accurate(&self) -> bool {
        matches!(
            self,
            ReductionMode::Kahan | ReductionMode::Pairwise | ReductionMode::Deterministic
        )
    }
}

impl Default for ReductionMode {
    fn default() -> Self {
        ReductionMode::Fast
    }
}

impl fmt::Display for ReductionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReductionMode::Deterministic => write!(f, "Reduction(deterministic)"),
            ReductionMode::Fast => write!(f, "Reduction(fast)"),
            ReductionMode::Pairwise => write!(f, "Reduction(pairwise)"),
            ReductionMode::Kahan => write!(f, "Reduction(kahan)"),
            ReductionMode::BlockDeterministic => write!(f, "Reduction(block-deterministic)"),
        }
    }
}

// ============================================================================
// Primitive 8: Precision
// ============================================================================

/// Numerical precision requirements.
///
/// Controls the floating-point precision used for computation, affecting
/// both accuracy and performance.
///
/// # Precision Hierarchy
///
/// | Precision | Bits | Use Case |
/// |-----------|------|----------|
/// | FP64 | 64 | Scientific computing, finance |
/// | FP32 | 32 | General ML training |
/// | TF32 | 19 | NVIDIA Tensor Core training |
/// | BF16 | 16 | Training (better range than FP16) |
/// | FP16 | 16 | Inference, some training |
/// | FP8 | 8 | Inference, quantization |
/// | FP4 | 4 | Extreme quantization |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// Full 64-bit double precision.
    Fp64,

    /// Standard 32-bit single precision.
    Fp32,

    /// TensorFloat-32 (19 bits effective precision).
    Tf32,

    /// Brain floating point (16 bits, 8-bit exponent).
    Bf16,

    /// IEEE half precision (16 bits, 5-bit exponent).
    Fp16,

    /// 8-bit floating point (E4M3 format).
    Fp8E4M3,

    /// 8-bit floating point (E5M2 format).
    Fp8E5M2,

    /// 4-bit floating point (Blackwell).
    Fp4,

    /// Mixed precision with separate compute and accumulate types.
    Mixed {
        /// Precision for compute operations.
        compute: ScalarPrecision,
        /// Precision for accumulation.
        accumulate: ScalarPrecision,
    },

    /// Tensor Core optimized precision (automatically selects best format).
    TensorCore,
}

/// Scalar precision for mixed-precision configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarPrecision {
    Fp64,
    Fp32,
    Tf32,
    Bf16,
    Fp16,
    Fp8E4M3,
    Fp8E5M2,
    Fp4,
}

impl Precision {
    /// Create a mixed precision configuration.
    pub fn mixed(compute: ScalarPrecision, accumulate: ScalarPrecision) -> Self {
        Precision::Mixed { compute, accumulate }
    }

    /// Standard FP16 compute with FP32 accumulate.
    pub fn amp_fp16() -> Self {
        Precision::Mixed {
            compute: ScalarPrecision::Fp16,
            accumulate: ScalarPrecision::Fp32,
        }
    }

    /// Standard BF16 compute with FP32 accumulate.
    pub fn amp_bf16() -> Self {
        Precision::Mixed {
            compute: ScalarPrecision::Bf16,
            accumulate: ScalarPrecision::Fp32,
        }
    }

    /// Get the effective bits of precision.
    pub fn effective_bits(&self) -> u32 {
        match self {
            Precision::Fp64 => 53,
            Precision::Fp32 => 24,
            Precision::Tf32 => 11,
            Precision::Bf16 => 8,
            Precision::Fp16 => 11,
            Precision::Fp8E4M3 => 4,
            Precision::Fp8E5M2 => 3,
            Precision::Fp4 => 2,
            Precision::Mixed { accumulate, .. } => match accumulate {
                ScalarPrecision::Fp64 => 53,
                ScalarPrecision::Fp32 => 24,
                ScalarPrecision::Tf32 => 11,
                ScalarPrecision::Bf16 => 8,
                ScalarPrecision::Fp16 => 11,
                ScalarPrecision::Fp8E4M3 => 4,
                ScalarPrecision::Fp8E5M2 => 3,
                ScalarPrecision::Fp4 => 2,
            },
            Precision::TensorCore => 11, // Typically TF32
        }
    }

    /// Check if this precision is suitable for scientific computing.
    pub fn is_scientific_grade(&self) -> bool {
        matches!(self, Precision::Fp64 | Precision::Fp32)
    }
}

impl Default for Precision {
    fn default() -> Self {
        Precision::Fp32
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::Fp64 => write!(f, "FP64"),
            Precision::Fp32 => write!(f, "FP32"),
            Precision::Tf32 => write!(f, "TF32"),
            Precision::Bf16 => write!(f, "BF16"),
            Precision::Fp16 => write!(f, "FP16"),
            Precision::Fp8E4M3 => write!(f, "FP8(E4M3)"),
            Precision::Fp8E5M2 => write!(f, "FP8(E5M2)"),
            Precision::Fp4 => write!(f, "FP4"),
            Precision::Mixed { compute, accumulate } => {
                write!(f, "Mixed({:?}->{:?})", compute, accumulate)
            }
            Precision::TensorCore => write!(f, "TensorCore"),
        }
    }
}

// ============================================================================
// Primitive 9: RNG Mode
// ============================================================================

/// Random number generation mode.
///
/// Controls how randomness is generated, with implications for reproducibility
/// and parallelism.
///
/// # Mode Comparison
///
/// | Mode | Reproducible | Parallel | Quality |
/// |------|--------------|----------|---------|
/// | CounterBased | Yes | Excellent | High |
/// | Deterministic | Yes | Poor | Varies |
/// | Fast | No | Excellent | Varies |
/// | Philox | Yes | Excellent | High |
/// | Threefry | Yes | Excellent | High |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RngMode {
    /// Counter-based RNG (e.g., Philox, Threefry).
    /// Deterministic given (key, counter) pair.
    /// Excellent parallelism - each thread has independent stream.
    CounterBased,

    /// Strictly deterministic sequential RNG.
    /// Same seed always produces same sequence.
    /// Poor parallelism - requires sequential state updates.
    Deterministic,

    /// Fast non-reproducible RNG.
    /// May use hardware RNG or non-deterministic algorithms.
    Fast,

    /// Philox 4x32-10 counter-based RNG.
    /// Used by TensorFlow and JAX.
    Philox,

    /// Threefry counter-based RNG.
    /// Used by JAX.
    Threefry,

    /// cuRAND device API.
    /// NVIDIA's optimized GPU RNG.
    CuRand,
}

impl RngMode {
    /// Check if this mode guarantees reproducibility.
    pub fn is_deterministic(&self) -> bool {
        matches!(
            self,
            RngMode::CounterBased
                | RngMode::Deterministic
                | RngMode::Philox
                | RngMode::Threefry
        )
    }

    /// Check if this mode supports efficient parallel generation.
    pub fn is_parallel_friendly(&self) -> bool {
        !matches!(self, RngMode::Deterministic)
    }
}

impl Default for RngMode {
    fn default() -> Self {
        RngMode::CounterBased
    }
}

impl fmt::Display for RngMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RngMode::CounterBased => write!(f, "RNG(counter-based)"),
            RngMode::Deterministic => write!(f, "RNG(deterministic)"),
            RngMode::Fast => write!(f, "RNG(fast)"),
            RngMode::Philox => write!(f, "RNG(philox)"),
            RngMode::Threefry => write!(f, "RNG(threefry)"),
            RngMode::CuRand => write!(f, "RNG(curand)"),
        }
    }
}

// ============================================================================
// Cost Envelope: Combining Primitives
// ============================================================================

/// A complete cost specification wrapping a kernel.
///
/// The `CostEnvelope` aggregates all performance primitives into a single
/// coherent specification that can be validated and used for code generation.
///
/// # Builder Pattern
///
/// ```ignore
/// let cost = CostEnvelope::new()
///     .with_tile(Tile::tensor_core_fp16())
///     .with_layout(Layout::tiled(Tile::tensor_core_fp16()))
///     .with_precision(Precision::amp_bf16())
///     .with_reduction_mode(ReductionMode::Deterministic)
///     .with_shared_cache(SharedCache::medium());
/// ```
#[derive(Debug, Clone, Default)]
pub struct CostEnvelope {
    /// Tiling specification.
    pub tile: Option<Tile>,

    /// Memory layout.
    pub layout: Option<Layout>,

    /// Vector width for SIMD operations.
    pub vector_width: Option<VectorWidth>,

    /// Lane group for cooperative operations.
    pub lane_group: Option<LaneGroup>,

    /// Shared memory allocation.
    pub shared_cache: Option<SharedCache>,

    /// Loop unrolling factor.
    pub unroll: Option<Unroll>,

    /// Operation fusion directive.
    pub fuse: Option<Fuse>,

    /// Reduction semantics.
    pub reduction_mode: Option<ReductionMode>,

    /// Numerical precision.
    pub precision: Option<Precision>,

    /// RNG mode.
    pub rng_mode: Option<RngMode>,

    /// Custom attributes for extensibility.
    pub custom_attrs: HashMap<String, String>,
}

impl CostEnvelope {
    /// Create a new empty cost envelope.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the tile dimensions.
    pub fn with_tile(mut self, tile: Tile) -> Self {
        self.tile = Some(tile);
        self
    }

    /// Set the memory layout.
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Set the vector width.
    pub fn with_vector_width(mut self, width: VectorWidth) -> Self {
        self.vector_width = Some(width);
        self
    }

    /// Set the lane group.
    pub fn with_lane_group(mut self, group: LaneGroup) -> Self {
        self.lane_group = Some(group);
        self
    }

    /// Set shared memory allocation.
    pub fn with_shared_cache(mut self, cache: SharedCache) -> Self {
        self.shared_cache = Some(cache);
        self
    }

    /// Set unroll factor.
    pub fn with_unroll(mut self, unroll: Unroll) -> Self {
        self.unroll = Some(unroll);
        self
    }

    /// Set fusion directive.
    pub fn with_fuse(mut self, fuse: Fuse) -> Self {
        self.fuse = Some(fuse);
        self
    }

    /// Set reduction mode.
    pub fn with_reduction_mode(mut self, mode: ReductionMode) -> Self {
        self.reduction_mode = Some(mode);
        self
    }

    /// Set precision requirements.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = Some(precision);
        self
    }

    /// Set RNG mode.
    pub fn with_rng_mode(mut self, mode: RngMode) -> Self {
        self.rng_mode = Some(mode);
        self
    }

    /// Add a custom attribute.
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_attrs.insert(key.into(), value.into());
        self
    }

    /// Check if this envelope requires deterministic execution.
    pub fn requires_determinism(&self) -> bool {
        let reduction_deterministic = self
            .reduction_mode
            .map(|m| m.is_deterministic())
            .unwrap_or(false);
        let rng_deterministic = self
            .rng_mode
            .map(|m| m.is_deterministic())
            .unwrap_or(false);
        reduction_deterministic || rng_deterministic
    }

    /// Check if Tensor Cores can be used with this envelope.
    pub fn supports_tensor_cores(&self) -> bool {
        // Check if tile is Tensor Core compatible
        let tile_ok = self
            .tile
            .map(|t| {
                matches!(
                    (t.m, t.n, t.k),
                    (16, 16, 16)
                        | (32, 8, 16)
                        | (8, 32, 16)
                        | (16, 16, 32)
                        | (32, 8, 32)
                        | (8, 32, 32)
                        | (16, 16, 64)
                        | (32, 8, 64)
                        | (8, 32, 64)
                        | (16, 16, 8)
                        | (32, 8, 8)
                        | (8, 32, 8)
                )
            })
            .unwrap_or(true);

        // Check if layout is compatible
        let layout_ok = self
            .layout
            .as_ref()
            .map(|l| l.is_tensor_core_compatible())
            .unwrap_or(true);

        // Check if precision is compatible
        let precision_ok = self
            .precision
            .map(|p| {
                matches!(
                    p,
                    Precision::Fp16
                        | Precision::Bf16
                        | Precision::Tf32
                        | Precision::Fp8E4M3
                        | Precision::Fp8E5M2
                        | Precision::Fp4
                        | Precision::TensorCore
                        | Precision::Mixed { .. }
                )
            })
            .unwrap_or(true);

        tile_ok && layout_ok && precision_ok
    }

    /// Validate the envelope for internal consistency.
    pub fn validate(&self) -> Result<(), CostValidationError> {
        // Validate tile if present
        if let Some(tile) = &self.tile {
            tile.validate()
                .map_err(|e| CostValidationError::InvalidTile(e))?;
        }

        // Validate shared cache fits in reasonable bounds
        if let Some(cache) = &self.shared_cache {
            if !cache.fits_within(256 * 1024) {
                return Err(CostValidationError::SharedCacheTooLarge {
                    requested: cache.aligned_size(),
                    max: 256 * 1024,
                });
            }
        }

        // Validate vector width
        if let Some(vw) = &self.vector_width {
            if !vw.is_valid() {
                return Err(CostValidationError::InvalidVectorWidth(vw.0));
            }
        }

        // Validate fusion has at least 2 ops
        if let Some(fuse) = &self.fuse {
            if !fuse.is_valid() {
                return Err(CostValidationError::InvalidFusion(
                    "fusion requires at least 2 operations".into(),
                ));
            }
        }

        Ok(())
    }

    /// Merge with another envelope, preferring self's values.
    pub fn merge(&self, other: &CostEnvelope) -> CostEnvelope {
        CostEnvelope {
            tile: self.tile.or(other.tile),
            layout: self.layout.clone().or(other.layout.clone()),
            vector_width: self.vector_width.or(other.vector_width),
            lane_group: self.lane_group.or(other.lane_group),
            shared_cache: self.shared_cache.or(other.shared_cache),
            unroll: self.unroll.or(other.unroll),
            fuse: self.fuse.clone().or(other.fuse.clone()),
            reduction_mode: self.reduction_mode.or(other.reduction_mode),
            precision: self.precision.or(other.precision),
            rng_mode: self.rng_mode.or(other.rng_mode),
            custom_attrs: {
                let mut attrs = other.custom_attrs.clone();
                attrs.extend(self.custom_attrs.clone());
                attrs
            },
        }
    }
}

impl fmt::Display for CostEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CostEnvelope {{")?;
        let mut first = true;

        if let Some(tile) = &self.tile {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, " {}", tile)?;
            first = false;
        }
        if let Some(layout) = &self.layout {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, " {}", layout)?;
            first = false;
        }
        if let Some(precision) = &self.precision {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, " {}", precision)?;
            first = false;
        }
        if let Some(reduction) = &self.reduction_mode {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, " {}", reduction)?;
            first = false;
        }
        if let Some(cache) = &self.shared_cache {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, " {}", cache)?;
            first = false;
        }

        if first {
            write!(f, " (empty)")?;
        }

        write!(f, " }}")
    }
}

/// Errors during cost envelope validation.
#[derive(Debug, Clone)]
pub enum CostValidationError {
    /// Invalid tile configuration.
    InvalidTile(TileError),
    /// Shared cache exceeds limits.
    SharedCacheTooLarge { requested: usize, max: usize },
    /// Invalid vector width.
    InvalidVectorWidth(u32),
    /// Invalid fusion specification.
    InvalidFusion(String),
    /// Conflicting annotations.
    Conflict(String),
}

impl fmt::Display for CostValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CostValidationError::InvalidTile(e) => write!(f, "invalid tile: {}", e),
            CostValidationError::SharedCacheTooLarge { requested, max } => {
                write!(
                    f,
                    "shared cache too large: {} bytes requested, max {} bytes",
                    requested, max
                )
            }
            CostValidationError::InvalidVectorWidth(w) => {
                write!(f, "invalid vector width: {}", w)
            }
            CostValidationError::InvalidFusion(msg) => write!(f, "invalid fusion: {}", msg),
            CostValidationError::Conflict(msg) => write!(f, "conflicting annotations: {}", msg),
        }
    }
}

impl std::error::Error for CostValidationError {}

// ============================================================================
// Kernel Variant: Multiple Specializations
// ============================================================================

/// A kernel variant with specific cost annotations.
///
/// Represents one possible specialization of a kernel for a particular
/// hardware configuration or use case.
#[derive(Debug, Clone)]
pub struct KernelVariant {
    /// Unique identifier for this variant.
    pub id: u32,

    /// Human-readable name for the variant.
    pub name: String,

    /// Cost envelope for this variant.
    pub cost: CostEnvelope,

    /// Target architecture hint (e.g., "sm_80", "avx512").
    pub target_hint: Option<String>,

    /// Priority for autotuning (higher = try first).
    pub priority: u32,
}

impl KernelVariant {
    /// Create a new kernel variant.
    pub fn new(id: u32, name: impl Into<String>, cost: CostEnvelope) -> Self {
        Self {
            id,
            name: name.into(),
            cost,
            target_hint: None,
            priority: 0,
        }
    }

    /// Set the target architecture hint.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target_hint = Some(target.into());
        self
    }

    /// Set the autotuning priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

impl fmt::Display for KernelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Variant[{}] '{}': {}", self.id, self.name, self.cost)?;
        if let Some(target) = &self.target_hint {
            write!(f, " (target: {})", target)?;
        }
        Ok(())
    }
}

/// A set of kernel variants for autotuning.
///
/// Contains multiple specializations of the same semantic kernel,
/// allowing the runtime to select the best variant for the current hardware.
#[derive(Debug, Clone)]
pub struct KernelVariantSet {
    /// Name of the base kernel.
    pub kernel_name: String,

    /// Function ID of the semantic kernel.
    pub func_id: Option<FuncId>,

    /// All available variants.
    pub variants: Vec<KernelVariant>,

    /// Default variant index (used when autotuning is disabled).
    pub default_variant: usize,
}

impl KernelVariantSet {
    /// Create a new variant set for the given kernel.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            kernel_name: name.into(),
            func_id: None,
            variants: Vec::new(),
            default_variant: 0,
        }
    }

    /// Set the function ID.
    pub fn with_func_id(mut self, func_id: FuncId) -> Self {
        self.func_id = Some(func_id);
        self
    }

    /// Add a variant with automatic ID assignment.
    pub fn add_variant(mut self, cost: CostEnvelope) -> Self {
        let id = self.variants.len() as u32;
        let name = format!("variant_{}", id);
        self.variants.push(KernelVariant::new(id, name, cost));
        self
    }

    /// Add a named variant.
    pub fn add_named_variant(mut self, name: impl Into<String>, cost: CostEnvelope) -> Self {
        let id = self.variants.len() as u32;
        self.variants.push(KernelVariant::new(id, name, cost));
        self
    }

    /// Set the default variant by index.
    pub fn with_default(mut self, index: usize) -> Self {
        self.default_variant = index.min(self.variants.len().saturating_sub(1));
        self
    }

    /// Get variants sorted by priority (highest first).
    pub fn sorted_by_priority(&self) -> Vec<&KernelVariant> {
        let mut variants: Vec<_> = self.variants.iter().collect();
        variants.sort_by(|a, b| b.priority.cmp(&a.priority));
        variants
    }

    /// Filter variants compatible with the given target.
    pub fn for_target(&self, target: &str) -> Vec<&KernelVariant> {
        self.variants
            .iter()
            .filter(|v| {
                v.target_hint
                    .as_ref()
                    .map(|t| t == target || t == "any")
                    .unwrap_or(true)
            })
            .collect()
    }
}

impl fmt::Display for KernelVariantSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "KernelVariantSet '{}':", self.kernel_name)?;
        for variant in &self.variants {
            writeln!(f, "  {}", variant)?;
        }
        writeln!(f, "  default: variant_{}", self.default_variant)?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_creation() {
        let tile = Tile::new(16, 16, 16);
        assert_eq!(tile.m, 16);
        assert_eq!(tile.n, 16);
        assert_eq!(tile.k, 16);
        assert_eq!(tile.elements(), 16 * 16 * 16);
        assert!(tile.is_power_of_two());
    }

    #[test]
    fn test_tile_validation() {
        // Valid tile
        assert!(Tile::tensor_core_fp16().validate().is_ok());

        // Invalid: not power of two
        assert!(Tile::new(15, 16, 16).validate().is_err());

        // Invalid: too large
        assert!(Tile::new(256, 16, 16).validate().is_err());
    }

    #[test]
    fn test_layout_display() {
        assert_eq!(format!("{}", Layout::RowMajor), "RowMajor");
        assert_eq!(format!("{}", Layout::ColMajor), "ColMajor");
        assert_eq!(format!("{}", Layout::blocked(8, 8)), "Blocked(8x8)");
    }

    #[test]
    fn test_vector_width_presets() {
        assert_eq!(VectorWidth::SCALAR.width(), 1);
        assert_eq!(VectorWidth::SSE.width(), 4);
        assert_eq!(VectorWidth::AVX.width(), 8);
        assert_eq!(VectorWidth::AVX512.width(), 16);
    }

    #[test]
    fn test_shared_cache_alignment() {
        let cache = SharedCache::bytes(100);
        assert_eq!(cache.aligned_size(), 128); // Aligned to 128

        let cache = SharedCache::bytes(256);
        assert_eq!(cache.aligned_size(), 256); // Already aligned
    }

    #[test]
    fn test_unroll_factor() {
        assert!(!Unroll::NONE.is_unrolled());
        assert!(Unroll::FULL.is_full());
        assert!(Unroll::new(4).is_unrolled());
    }

    #[test]
    fn test_reduction_mode_determinism() {
        assert!(ReductionMode::Deterministic.is_deterministic());
        assert!(ReductionMode::Kahan.is_deterministic());
        assert!(!ReductionMode::Fast.is_deterministic());
    }

    #[test]
    fn test_precision_effective_bits() {
        assert_eq!(Precision::Fp64.effective_bits(), 53);
        assert_eq!(Precision::Fp32.effective_bits(), 24);
        assert_eq!(Precision::Fp16.effective_bits(), 11);
        assert_eq!(Precision::Bf16.effective_bits(), 8);
    }

    #[test]
    fn test_rng_mode_determinism() {
        assert!(RngMode::CounterBased.is_deterministic());
        assert!(RngMode::Philox.is_deterministic());
        assert!(!RngMode::Fast.is_deterministic());
    }

    #[test]
    fn test_cost_envelope_builder() {
        let cost = CostEnvelope::new()
            .with_tile(Tile::tensor_core_fp16())
            .with_precision(Precision::amp_bf16())
            .with_reduction_mode(ReductionMode::Deterministic)
            .with_shared_cache(SharedCache::medium());

        assert!(cost.tile.is_some());
        assert!(cost.precision.is_some());
        assert!(cost.requires_determinism());
        assert!(cost.supports_tensor_cores());
    }

    #[test]
    fn test_cost_envelope_validation() {
        // Valid envelope
        let valid = CostEnvelope::new()
            .with_tile(Tile::tensor_core_fp16())
            .with_vector_width(VectorWidth::AVX);
        assert!(valid.validate().is_ok());

        // Invalid: tile not power of 2
        let invalid_tile = CostEnvelope::new().with_tile(Tile::new(15, 16, 16));
        assert!(invalid_tile.validate().is_err());

        // Invalid: shared cache too large
        let invalid_cache = CostEnvelope::new().with_shared_cache(SharedCache::bytes(512 * 1024));
        assert!(invalid_cache.validate().is_err());
    }

    #[test]
    fn test_cost_envelope_merge() {
        let base = CostEnvelope::new()
            .with_tile(Tile::tensor_core_fp16())
            .with_precision(Precision::Fp32);

        let override_envelope = CostEnvelope::new()
            .with_precision(Precision::Fp16)
            .with_reduction_mode(ReductionMode::Deterministic);

        let merged = override_envelope.merge(&base);

        // Override takes precedence
        assert_eq!(merged.precision, Some(Precision::Fp16));
        // Base fills in missing
        assert_eq!(merged.tile, Some(Tile::tensor_core_fp16()));
        // Override's new field
        assert_eq!(merged.reduction_mode, Some(ReductionMode::Deterministic));
    }

    #[test]
    fn test_kernel_variant_set() {
        let set = KernelVariantSet::new("matmul")
            .add_named_variant(
                "tensor_core",
                CostEnvelope::new()
                    .with_tile(Tile::tensor_core_fp16())
                    .with_precision(Precision::TensorCore),
            )
            .add_named_variant(
                "fp32",
                CostEnvelope::new().with_precision(Precision::Fp32),
            );

        assert_eq!(set.variants.len(), 2);
        assert_eq!(set.variants[0].name, "tensor_core");
        assert_eq!(set.variants[1].name, "fp32");
    }

    #[test]
    fn test_cost_envelope_display() {
        let cost = CostEnvelope::new()
            .with_tile(Tile::new(16, 16, 16))
            .with_precision(Precision::Fp16);

        let display = format!("{}", cost);
        assert!(display.contains("Tile"));
        assert!(display.contains("FP16"));
    }

    #[test]
    fn test_fusion_validation() {
        // Valid fusion with 2+ ops
        let valid_fuse = Fuse::new(vec![OpId(0), OpId(1)]);
        assert!(valid_fuse.is_valid());

        // Invalid fusion with < 2 ops
        let invalid_fuse = Fuse::new(vec![OpId(0)]);
        assert!(!invalid_fuse.is_valid());

        // Empty fusion
        let empty_fuse = Fuse::new(vec![]);
        assert!(!empty_fuse.is_valid());
    }

    #[test]
    fn test_lane_group_presets() {
        assert_eq!(LaneGroup::SINGLE.lanes(), 1);
        assert_eq!(LaneGroup::QUARTER_WARP.lanes(), 8);
        assert_eq!(LaneGroup::HALF_WARP.lanes(), 16);
        assert_eq!(LaneGroup::WARP.lanes(), 32);
        assert_eq!(LaneGroup::WARP_GROUP.lanes(), 128);
    }

    #[test]
    fn test_precision_scientific_grade() {
        assert!(Precision::Fp64.is_scientific_grade());
        assert!(Precision::Fp32.is_scientific_grade());
        assert!(!Precision::Fp16.is_scientific_grade());
        assert!(!Precision::Bf16.is_scientific_grade());
        assert!(!Precision::TensorCore.is_scientific_grade());
    }
}
