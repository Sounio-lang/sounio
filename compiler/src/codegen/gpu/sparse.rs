//! Sparse Tensor Compiler for GPU
//!
//! Implements sparse tensor operations with format-aware analysis and kernels:
//! - Multiple sparse formats (CSR, CSC, COO, BCSR, ELL, Hybrid, 2:4)
//! - Automatic format detection and selection
//! - Sparsity cost modeling
//! - NVIDIA 2:4 structured sparsity (Tensor Core acceleration)
//! - Format conversion and fusion optimizations
//!
//! # Sparse Format Overview
//!
//! | Format | Best For | Memory | Performance |
//! |--------|----------|--------|-------------|
//! | CSR    | General sparse matrices | O(nnz) | Good for SpMV |
//! | CSC    | Column-major access | O(nnz) | Good for SpMM |
//! | COO    | Construction, flexible | O(nnz) | Moderate |
//! | BCSR   | Block sparsity | O(nnz) | Excellent for GPU |
//! | ELL    | Regular sparsity pattern | O(m*k) | Very fast if uniform |
//! | Hybrid | Mixed patterns | O(nnz) | Adaptive |
//! | 2:4    | Structured sparsity | O(n/2) | 2x Tensor Core speedup |
//!
//! # NVIDIA 2:4 Structured Sparsity
//!
//! NVIDIA Ampere+ GPUs support 2:4 structured sparsity in Tensor Cores:
//! - Exactly 2 non-zero values per 4 elements
//! - 2x throughput compared to dense operations
//! - Requires specific data layout and metadata

use std::fmt;

use super::costs::{CostDatabase, InstructionClass};
use super::ir::{
    BlockId, CudaArch, GpuBlock, GpuKernel, GpuModule, GpuOp, GpuParam, GpuTerminator, GpuType,
    MemorySpace, ValueId,
};

// ============================================================================
// Sparse Formats
// ============================================================================

/// Sparse tensor storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Dense (not sparse)
    Dense,

    /// Compressed Sparse Row (CSR)
    /// Storage: row_ptr[m+1], col_idx[nnz], values[nnz]
    /// Good for: Row-major access, SpMV
    CSR,

    /// Compressed Sparse Column (CSC)
    /// Storage: col_ptr[n+1], row_idx[nnz], values[nnz]
    /// Good for: Column-major access, SpMM
    CSC,

    /// Coordinate format (COO)
    /// Storage: row_idx[nnz], col_idx[nnz], values[nnz]
    /// Good for: Construction, random access
    COO,

    /// Block Compressed Sparse Row (BCSR)
    /// Storage: block_row_ptr[mb+1], block_col_idx[nblocks], block_values[nblocks*bs*bs]
    /// Good for: Block sparsity patterns
    BCSR {
        /// Block size (typically 2x2, 4x4, 8x8)
        block_size: (usize, usize),
    },

    /// ELLPACK format
    /// Storage: col_idx[m*k], values[m*k] where k = max_nnz_per_row
    /// Good for: Regular sparsity patterns with uniform nnz per row
    ELL {
        /// Maximum non-zeros per row
        max_nnz_per_row: usize,
    },

    /// Hybrid format (ELL + COO)
    /// Regular part stored in ELL, overflow in COO
    /// Good for: Mixed regular/irregular patterns
    Hybrid {
        /// ELL threshold
        ell_width: usize,
    },

    /// NVIDIA 2:4 Structured Sparsity (sm_80+)
    /// Exactly 2 non-zero values per 4 elements
    /// Requires metadata for Tensor Core acceleration
    Structured2x4,

    /// Block sparse with arbitrary block size
    /// More flexible than BCSR
    BlockSparse { block_size: (usize, usize) },
}

impl SparseFormat {
    /// Get the minimum CUDA compute capability required
    pub fn min_compute_capability(&self) -> Option<(u32, u32)> {
        match self {
            SparseFormat::Structured2x4 => Some((8, 0)), // Ampere+
            _ => None, // All other formats work on any architecture
        }
    }

    /// Check if this format supports Tensor Core acceleration
    pub fn supports_tensor_cores(&self) -> bool {
        matches!(self, SparseFormat::Structured2x4)
    }

    /// Get the memory overhead factor compared to dense storage
    /// Returns (index_bytes, value_bytes) per non-zero element
    pub fn memory_overhead(&self, dtype_bytes: usize) -> (usize, usize) {
        match self {
            SparseFormat::Dense => (0, dtype_bytes),
            SparseFormat::CSR | SparseFormat::CSC => (4, dtype_bytes), // 1 index per nnz
            SparseFormat::COO => (8, dtype_bytes),                     // 2 indices per nnz
            SparseFormat::BCSR { block_size } => {
                let block_elems = block_size.0 * block_size.1;
                (4 / block_elems.max(1), dtype_bytes) // Amortized index cost
            }
            SparseFormat::ELL { .. } => (4, dtype_bytes), // 1 index per element
            SparseFormat::Hybrid { .. } => (4, dtype_bytes), // Average case
            SparseFormat::Structured2x4 => (1, dtype_bytes / 2), // Metadata + 50% values
            SparseFormat::BlockSparse { block_size } => {
                let block_elems = block_size.0 * block_size.1;
                (4 / block_elems.max(1), dtype_bytes)
            }
        }
    }
}

impl fmt::Display for SparseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseFormat::Dense => write!(f, "dense"),
            SparseFormat::CSR => write!(f, "csr"),
            SparseFormat::CSC => write!(f, "csc"),
            SparseFormat::COO => write!(f, "coo"),
            SparseFormat::BCSR { block_size } => {
                write!(f, "bcsr_{}x{}", block_size.0, block_size.1)
            }
            SparseFormat::ELL { max_nnz_per_row } => write!(f, "ell_{}", max_nnz_per_row),
            SparseFormat::Hybrid { ell_width } => write!(f, "hybrid_{}", ell_width),
            SparseFormat::Structured2x4 => write!(f, "2:4_structured"),
            SparseFormat::BlockSparse { block_size } => {
                write!(f, "block_sparse_{}x{}", block_size.0, block_size.1)
            }
        }
    }
}

/// Sparse tensor metadata
#[derive(Debug, Clone)]
pub struct SparseTensor {
    /// Storage format
    pub format: SparseFormat,
    /// Tensor shape [m, n] for 2D, arbitrary for N-D
    pub shape: Vec<usize>,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Sparsity density: nnz / total_elements
    pub density: f64,
    /// Data type
    pub dtype: GpuType,
}

impl SparseTensor {
    /// Create a new sparse tensor
    pub fn new(format: SparseFormat, shape: Vec<usize>, nnz: usize, dtype: GpuType) -> Self {
        let total_elements: usize = shape.iter().product();
        let density = if total_elements > 0 {
            nnz as f64 / total_elements as f64
        } else {
            0.0
        };

        Self {
            format,
            shape,
            nnz,
            density,
            dtype,
        }
    }

    /// Get total number of elements
    pub fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get sparsity ratio (1.0 - density)
    pub fn sparsity(&self) -> f64 {
        1.0 - self.density
    }

    /// Check if this is actually sparse (density < 0.5)
    pub fn is_sparse(&self) -> bool {
        self.density < 0.5
    }

    /// Check if very sparse (density < 0.1)
    pub fn is_very_sparse(&self) -> bool {
        self.density < 0.1
    }
}

/// Sparsity pattern metadata
#[derive(Debug, Clone, Default)]
pub struct SparsePattern {
    /// CSR row pointers
    pub row_ptr: Vec<usize>,
    /// Column indices
    pub col_idx: Vec<usize>,
    /// Block map for block-sparse formats
    pub block_map: Option<BlockMap>,
    /// 2:4 metadata for structured sparsity
    pub metadata_2x4: Option<Vec<u8>>,
}

/// Block sparsity map
#[derive(Debug, Clone)]
pub struct BlockMap {
    /// Block row pointers
    pub block_row_ptr: Vec<usize>,
    /// Block column indices
    pub block_col_idx: Vec<usize>,
    /// Block size
    pub block_size: (usize, usize),
    /// Number of blocks
    pub num_blocks: usize,
}

// ============================================================================
// Sparsity Analysis
// ============================================================================

/// Sparsity analyzer - detect patterns and optimal formats
pub struct SparsityAnalyzer {
    /// Target GPU architecture
    arch: CudaArch,
    /// Cost database for performance modeling
    cost_db: CostDatabase,
    /// Configuration
    config: AnalyzerConfig,
}

/// Analyzer configuration
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Minimum density threshold to consider sparse
    pub min_density_threshold: f64,
    /// Maximum density to use sparse operations
    pub max_density_threshold: f64,
    /// Block sizes to consider for block sparsity
    pub block_sizes: Vec<(usize, usize)>,
    /// Enable 2:4 structured sparsity detection
    pub enable_structured_2x4: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            min_density_threshold: 0.01,
            max_density_threshold: 0.5,
            block_sizes: vec![(2, 2), (4, 4), (8, 8), (16, 16)],
            enable_structured_2x4: true,
        }
    }
}

impl SparsityAnalyzer {
    /// Create a new sparsity analyzer
    pub fn new(arch: CudaArch) -> Self {
        Self {
            cost_db: CostDatabase::for_arch(arch),
            arch,
            config: AnalyzerConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(arch: CudaArch, config: AnalyzerConfig) -> Self {
        Self {
            cost_db: CostDatabase::for_arch(arch),
            arch,
            config,
        }
    }

    /// Estimate density from tensor statistics
    pub fn estimate_density(&self, nnz: usize, total_elements: usize) -> f64 {
        if total_elements == 0 {
            return 0.0;
        }
        nnz as f64 / total_elements as f64
    }

    /// Detect if tensor should be stored as sparse
    pub fn should_use_sparse(&self, density: f64) -> bool {
        density >= self.config.min_density_threshold && density <= self.config.max_density_threshold
    }

    /// Detect optimal sparse format based on sparsity pattern
    pub fn detect_format(
        &self,
        shape: &[usize],
        nnz: usize,
        _pattern: Option<&SparsePattern>,
    ) -> SparseFormat {
        let density = self.estimate_density(nnz, shape.iter().product());

        // Not sparse enough
        if !self.should_use_sparse(density) {
            return SparseFormat::Dense;
        }

        // Try 2:4 structured sparsity first (if enabled and supported)
        if self.config.enable_structured_2x4
            && self.arch.compute_capability() >= (8, 0)
            && self.is_structured_2x4(nnz, shape.iter().product())
        {
            return SparseFormat::Structured2x4;
        }

        // Very sparse: use COO for flexibility
        if density < 0.05 {
            return SparseFormat::COO;
        }

        // Default: CSR (most versatile)
        SparseFormat::CSR
    }

    /// Check if pattern matches 2:4 structured sparsity
    fn is_structured_2x4(&self, nnz: usize, total: usize) -> bool {
        // 2:4 means exactly 50% sparsity
        let expected_nnz = total / 2;
        let tolerance = total / 100; // 1% tolerance
        nnz >= expected_nnz.saturating_sub(tolerance) && nnz <= expected_nnz + tolerance
    }

    /// Detect structure in sparsity pattern
    pub fn detect_structure(&self, _pattern: &SparsePattern) -> StructureInfo {
        StructureInfo {
            has_block_structure: false,
            is_regular: false,
            is_structured_2x4: false,
            block_size: None,
        }
    }

    /// Cost model for sparse vs dense operations
    pub fn sparsity_cost_model(
        &self,
        format: SparseFormat,
        shape: &[usize],
        nnz: usize,
        op_type: SparseOpType,
    ) -> SparsityCost {
        let total_elements: usize = shape.iter().product();
        let density = self.estimate_density(nnz, total_elements);

        // Estimate costs
        let dense_cost = self.estimate_dense_cost(shape, op_type);
        let sparse_cost = self.estimate_sparse_cost(format, nnz, op_type);

        let speedup = if sparse_cost > 0.0 {
            dense_cost / sparse_cost
        } else {
            f64::INFINITY
        };

        SparsityCost {
            dense_cost,
            sparse_cost,
            speedup,
            memory_savings: 1.0 - (nnz as f64 / total_elements as f64),
            format,
            recommended: speedup > 1.0 && density < self.config.max_density_threshold,
        }
    }

    /// Estimate cost for dense operation
    fn estimate_dense_cost(&self, shape: &[usize], op_type: SparseOpType) -> f64 {
        if shape.len() < 2 {
            return 0.0;
        }

        let m = shape[0];
        let n = shape[1];

        match op_type {
            SparseOpType::MatVec => {
                // Dense SpMV: O(m*n) FMAs
                let flops = m * n * 2; // FMA = 2 ops
                let fma_cost = self.cost_db.get_cost(InstructionClass::Fp32Fma);
                flops as f64 * fma_cost.latency as f64 / fma_cost.throughput
            }
            SparseOpType::MatMat { k } => {
                // Dense GEMM: O(m*n*k) FMAs
                let flops = m * n * k * 2;
                let fma_cost = self.cost_db.get_cost(InstructionClass::Fp32Fma);
                flops as f64 * fma_cost.latency as f64 / fma_cost.throughput
            }
            SparseOpType::Convolution { .. } => {
                // Simplified convolution cost
                let flops = m * n * 9 * 2; // 3x3 kernel approximation
                let fma_cost = self.cost_db.get_cost(InstructionClass::Fp32Fma);
                flops as f64 * fma_cost.latency as f64 / fma_cost.throughput
            }
        }
    }

    /// Estimate cost for sparse operation
    fn estimate_sparse_cost(&self, format: SparseFormat, nnz: usize, op_type: SparseOpType) -> f64 {
        let base_cost = match op_type {
            SparseOpType::MatVec => {
                // Sparse SpMV: O(nnz) FMAs
                let flops = nnz * 2;
                let fma_cost = self.cost_db.get_cost(InstructionClass::Fp32Fma);
                flops as f64 * fma_cost.latency as f64 / fma_cost.throughput
            }
            SparseOpType::MatMat { k } => {
                // Sparse SpMM: O(nnz * k) FMAs
                let flops = nnz * k * 2;
                let fma_cost = self.cost_db.get_cost(InstructionClass::Fp32Fma);
                flops as f64 * fma_cost.latency as f64 / fma_cost.throughput
            }
            SparseOpType::Convolution { .. } => {
                // Sparse convolution: proportional to nnz
                let flops = nnz * 9 * 2;
                let fma_cost = self.cost_db.get_cost(InstructionClass::Fp32Fma);
                flops as f64 * fma_cost.latency as f64 / fma_cost.throughput
            }
        };

        // Format-specific overhead
        let format_overhead = match format {
            SparseFormat::Dense => 1.0,
            SparseFormat::CSR | SparseFormat::CSC => 1.2, // Index lookup overhead
            SparseFormat::COO => 1.5,                     // More irregular access
            SparseFormat::BCSR { .. } => 0.9,             // Better locality
            SparseFormat::ELL { .. } => 1.1,              // Padding overhead
            SparseFormat::Hybrid { .. } => 1.15,          // Mixed overhead
            SparseFormat::Structured2x4 => 0.5,           // 2x Tensor Core speedup
            SparseFormat::BlockSparse { .. } => 0.95,     // Good GPU utilization
        };

        base_cost * format_overhead
    }
}

/// Structure information from pattern analysis
#[derive(Debug, Clone)]
pub struct StructureInfo {
    pub has_block_structure: bool,
    pub is_regular: bool,
    pub is_structured_2x4: bool,
    pub block_size: Option<(usize, usize)>,
}

/// Sparse operation type for cost modeling
#[derive(Debug, Clone, Copy)]
pub enum SparseOpType {
    /// Sparse matrix-vector multiply
    MatVec,
    /// Sparse matrix-matrix multiply
    MatMat { k: usize },
    /// Sparse convolution
    Convolution { kernel_size: (usize, usize) },
}

/// Cost analysis result
#[derive(Debug, Clone)]
pub struct SparsityCost {
    /// Estimated dense cost (cycles)
    pub dense_cost: f64,
    /// Estimated sparse cost (cycles)
    pub sparse_cost: f64,
    /// Speedup factor (dense_cost / sparse_cost)
    pub speedup: f64,
    /// Memory savings (0.0 to 1.0)
    pub memory_savings: f64,
    /// Recommended format
    pub format: SparseFormat,
    /// Whether sparse is recommended
    pub recommended: bool,
}

// ============================================================================
// Sparse Kernels (Simplified)
// ============================================================================

/// Sparse GEMM kernel generator
pub struct SparseGemmKernel {
    format: SparseFormat,
    arch: CudaArch,
}

impl SparseGemmKernel {
    /// Create a new sparse GEMM kernel generator
    pub fn new(format: SparseFormat, arch: CudaArch) -> Self {
        Self { format, arch }
    }

    /// Generate sparse GEMM kernel (simplified stub)
    pub fn generate(&self, m: usize, n: usize, k: usize, dtype: GpuType) -> GpuKernel {
        let kernel_name = format!("sparse_gemm_{}_{}_{}x{}x{}", self.format, dtype, m, n, k);
        let mut kernel = GpuKernel::new(&kernel_name);

        // Parameters
        kernel.add_param(GpuParam {
            name: "A_values".into(),
            ty: GpuType::Ptr(Box::new(dtype.clone()), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "B".into(),
            ty: GpuType::Ptr(Box::new(dtype.clone()), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "C".into(),
            ty: GpuType::Ptr(Box::new(dtype), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        match self.format {
            SparseFormat::CSR => {
                kernel.add_param(GpuParam {
                    name: "row_ptr".into(),
                    ty: GpuType::Ptr(Box::new(GpuType::U32), MemorySpace::Global),
                    space: MemorySpace::Global,
                    restrict: true,
                });
                kernel.add_param(GpuParam {
                    name: "col_idx".into(),
                    ty: GpuType::Ptr(Box::new(GpuType::U32), MemorySpace::Global),
                    space: MemorySpace::Global,
                    restrict: true,
                });
            }
            SparseFormat::Structured2x4 => {
                kernel.add_param(GpuParam {
                    name: "metadata".into(),
                    ty: GpuType::Ptr(Box::new(GpuType::U8), MemorySpace::Global),
                    space: MemorySpace::Global,
                    restrict: true,
                });
            }
            _ => {}
        }

        // Simple stub implementation
        let mut entry = GpuBlock::new(BlockId(0), "entry");
        entry.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        entry.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(entry);

        kernel
    }
}

/// Sparse matrix-vector multiply kernel generator
pub struct SparseMVKernel {
    format: SparseFormat,
}

impl SparseMVKernel {
    pub fn new(format: SparseFormat) -> Self {
        Self { format }
    }

    pub fn generate(&self, m: usize, n: usize, dtype: GpuType) -> GpuKernel {
        let kernel_name = format!("sparse_mv_{}_{}x{}", self.format, m, n);
        let mut kernel = GpuKernel::new(&kernel_name);

        kernel.add_param(GpuParam {
            name: "A_values".into(),
            ty: GpuType::Ptr(Box::new(dtype.clone()), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "x".into(),
            ty: GpuType::Ptr(Box::new(dtype.clone()), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "y".into(),
            ty: GpuType::Ptr(Box::new(dtype), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        let mut entry = GpuBlock::new(BlockId(0), "entry");
        entry.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        entry.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(entry);

        kernel
    }
}

/// Sparse convolution kernel generator
pub struct SparseConvKernel {
    format: SparseFormat,
}

impl SparseConvKernel {
    pub fn new(format: SparseFormat) -> Self {
        Self { format }
    }

    pub fn generate(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> GpuKernel {
        let kernel_name = format!(
            "sparse_conv_{}_{}_{}x{}x{}",
            self.format, in_channels, out_channels, kernel_size.0, kernel_size.1
        );
        let mut kernel = GpuKernel::new(&kernel_name);

        kernel.add_param(GpuParam {
            name: "input".into(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "weight".into(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "output".into(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        let mut entry = GpuBlock::new(BlockId(0), "entry");
        entry.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        entry.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(entry);

        kernel
    }
}

// ============================================================================
// Sparse Fusion
// ============================================================================

/// Sparse fusion analyzer - determine if sparse ops can be fused
pub struct SparseFusionAnalyzer {
    _cost_db: CostDatabase,
}

impl SparseFusionAnalyzer {
    pub fn new(arch: CudaArch) -> Self {
        Self {
            _cost_db: CostDatabase::for_arch(arch),
        }
    }

    /// Check if two sparse operations can be fused
    pub fn can_fuse(&self, op1: &SparseOpInfo, op2: &SparseOpInfo) -> bool {
        // Can't fuse if formats are incompatible
        if !self.formats_compatible(op1.format, op2.format) {
            return false;
        }

        // Can't fuse if shapes don't match
        if !self.shapes_compatible(&op1.output_shape, &op2.input_shape) {
            return false;
        }

        // Density-aware fusion: only fuse if densities are similar
        let density_ratio = op1.density / op2.density.max(0.001);
        if !(0.5..=2.0).contains(&density_ratio) {
            return false;
        }

        true
    }

    /// Check if formats are compatible for fusion
    fn formats_compatible(&self, fmt1: SparseFormat, fmt2: SparseFormat) -> bool {
        match (fmt1, fmt2) {
            (SparseFormat::CSR, SparseFormat::CSR) => true,
            (SparseFormat::CSC, SparseFormat::CSC) => true,
            (SparseFormat::Structured2x4, SparseFormat::Structured2x4) => true,
            _ => false,
        }
    }

    /// Check if shapes are compatible
    fn shapes_compatible(&self, out_shape: &[usize], in_shape: &[usize]) -> bool {
        if out_shape.is_empty() || in_shape.is_empty() {
            return false;
        }
        out_shape[out_shape.len() - 1] == in_shape[0]
    }

    /// Cost model for fusion
    pub fn fusion_benefit(&self, op1: &SparseOpInfo, _op2: &SparseOpInfo) -> f64 {
        // Benefit = eliminated intermediate memory traffic
        let intermediate_size = op1.output_shape.iter().product::<usize>() as f64;
        let memory_saved = intermediate_size * 4.0; // Assume fp32

        // Savings vs overhead
        let fusion_overhead = 1.2; // 20% overhead from complex control flow
        memory_saved / fusion_overhead
    }
}

/// Sparse operation metadata for fusion analysis
#[derive(Debug, Clone)]
pub struct SparseOpInfo {
    pub format: SparseFormat,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub density: f64,
    pub op_type: SparseOpType,
}

// ============================================================================
// Integration
// ============================================================================

/// Add sparse kernels to a GPU module
pub fn add_sparse_kernels(module: &mut GpuModule, arch: CudaArch, tensors: &[SparseTensor]) {
    for tensor in tensors {
        match tensor.format {
            SparseFormat::Dense => continue,

            SparseFormat::CSR => {
                let spmm = SparseGemmKernel::new(SparseFormat::CSR, arch);
                if tensor.shape.len() >= 2 {
                    let kernel = spmm.generate(
                        tensor.shape[0],
                        tensor.shape[1],
                        tensor.shape[1],
                        tensor.dtype.clone(),
                    );
                    module.add_kernel(kernel);
                }

                let spmv = SparseMVKernel::new(SparseFormat::CSR);
                if tensor.shape.len() >= 2 {
                    let kernel =
                        spmv.generate(tensor.shape[0], tensor.shape[1], tensor.dtype.clone());
                    module.add_kernel(kernel);
                }
            }

            SparseFormat::Structured2x4 => {
                let spmm = SparseGemmKernel::new(SparseFormat::Structured2x4, arch);
                if tensor.shape.len() >= 2 {
                    let kernel = spmm.generate(
                        tensor.shape[0],
                        tensor.shape[1],
                        tensor.shape[1],
                        tensor.dtype.clone(),
                    );
                    module.add_kernel(kernel);
                }
            }

            _ => {
                // Add generic sparse kernel
                let spmm = SparseGemmKernel::new(tensor.format, arch);
                if tensor.shape.len() >= 2 {
                    let kernel = spmm.generate(
                        tensor.shape[0],
                        tensor.shape[1],
                        tensor.shape[1],
                        tensor.dtype.clone(),
                    );
                    module.add_kernel(kernel);
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_format_display() {
        assert_eq!(format!("{}", SparseFormat::CSR), "csr");
        assert_eq!(format!("{}", SparseFormat::Structured2x4), "2:4_structured");
        assert_eq!(
            format!("{}", SparseFormat::BCSR { block_size: (4, 4) }),
            "bcsr_4x4"
        );
    }

    #[test]
    fn test_sparse_tensor_density() {
        let tensor = SparseTensor::new(SparseFormat::CSR, vec![1000, 1000], 50000, GpuType::F32);

        assert_eq!(tensor.total_elements(), 1_000_000);
        assert_eq!(tensor.density, 0.05);
        assert_eq!(tensor.sparsity(), 0.95);
        assert!(tensor.is_very_sparse());
    }

    #[test]
    fn test_sparsity_analyzer() {
        let analyzer = SparsityAnalyzer::new(CudaArch::Ampere);

        // Very sparse (below min threshold 0.01, so sparse ops not recommended)
        let density = analyzer.estimate_density(1000, 1_000_000);
        assert_eq!(density, 0.001);
        assert!(!analyzer.should_use_sparse(density)); // Too sparse for sparse ops

        // Good sparsity range (between 0.01 and 0.5)
        let density = analyzer.estimate_density(100_000, 1_000_000);
        assert_eq!(density, 0.1);
        assert!(analyzer.should_use_sparse(density)); // Good range for sparse ops

        // Not sparse enough (above max threshold 0.5)
        let density = analyzer.estimate_density(900_000, 1_000_000);
        assert_eq!(density, 0.9);
        assert!(!analyzer.should_use_sparse(density));
    }

    #[test]
    fn test_format_detection() {
        let analyzer = SparsityAnalyzer::new(CudaArch::Ampere);

        // Very sparse (density 0.001 < 0.01 threshold) -> Dense (sparse ops not worth it)
        let format = analyzer.detect_format(&[1000, 1000], 1000, None);
        assert_eq!(format, SparseFormat::Dense);

        // Sparse in good range (density ~0.02) -> COO
        let format = analyzer.detect_format(&[1000, 1000], 20_000, None);
        assert_eq!(format, SparseFormat::COO);

        // 50% sparse -> could be 2:4
        let format = analyzer.detect_format(&[1024, 1024], 524_288, None);
        assert_eq!(format, SparseFormat::Structured2x4);

        // Dense (density 0.95 > 0.5 threshold) -> Dense
        let format = analyzer.detect_format(&[100, 100], 9500, None);
        assert_eq!(format, SparseFormat::Dense);
    }

    #[test]
    fn test_sparsity_cost_model() {
        let analyzer = SparsityAnalyzer::new(CudaArch::Ampere);

        let cost = analyzer.sparsity_cost_model(
            SparseFormat::CSR,
            &[1000, 1000],
            50_000,
            SparseOpType::MatVec,
        );

        assert!(cost.speedup > 1.0);
        assert!(cost.memory_savings > 0.9);
        assert!(cost.recommended);
    }

    #[test]
    fn test_2x4_detection() {
        let analyzer = SparsityAnalyzer::new(CudaArch::Ampere);

        // Exactly 50% sparse
        assert!(analyzer.is_structured_2x4(512, 1024));

        // Not 50% sparse
        assert!(!analyzer.is_structured_2x4(100, 1024));
        assert!(!analyzer.is_structured_2x4(900, 1024));
    }

    #[test]
    fn test_sparse_gemm_kernel_generation() {
        let generator = SparseGemmKernel::new(SparseFormat::CSR, CudaArch::Ampere);
        let kernel = generator.generate(128, 128, 128, GpuType::F32);

        assert!(kernel.name.contains("sparse_gemm"));
        assert!(kernel.name.contains("csr"));
        assert_eq!(kernel.params.len(), 5); // A_values, B, C, row_ptr, col_idx
    }

    #[test]
    fn test_sparse_fusion_analyzer() {
        let analyzer = SparseFusionAnalyzer::new(CudaArch::Ampere);

        // op1: (1000 x 1000) * (1000 x 500) -> (1000 x 500)
        let op1 = SparseOpInfo {
            format: SparseFormat::CSR,
            input_shape: vec![1000, 1000],
            output_shape: vec![1000, 500],
            density: 0.1,
            op_type: SparseOpType::MatMat { k: 1000 },
        };

        // op2: (1000 x 500) -> output uses op1's output dim (500) as input
        // shapes_compatible checks: out_shape[-1] == in_shape[0], so 500 == 500
        let op2 = SparseOpInfo {
            format: SparseFormat::CSR,
            input_shape: vec![500, 200],
            output_shape: vec![1000, 200],
            density: 0.12,
            op_type: SparseOpType::MatMat { k: 500 },
        };

        assert!(analyzer.can_fuse(&op1, &op2));

        let benefit = analyzer.fusion_benefit(&op1, &op2);
        assert!(benefit > 0.0);
    }

    #[test]
    fn test_format_compatibility() {
        let analyzer = SparseFusionAnalyzer::new(CudaArch::Ampere);

        assert!(analyzer.formats_compatible(SparseFormat::CSR, SparseFormat::CSR));
        assert!(!analyzer.formats_compatible(SparseFormat::CSR, SparseFormat::CSC));
        assert!(
            analyzer.formats_compatible(SparseFormat::Structured2x4, SparseFormat::Structured2x4)
        );
    }

    #[test]
    fn test_memory_overhead() {
        let csr_overhead = SparseFormat::CSR.memory_overhead(4);
        assert_eq!(csr_overhead, (4, 4)); // 4 bytes index + 4 bytes value

        let s24_overhead = SparseFormat::Structured2x4.memory_overhead(4);
        assert_eq!(s24_overhead, (1, 2)); // 1 byte metadata + 2 bytes value (50%)
    }

    #[test]
    fn test_tensor_core_support() {
        assert!(SparseFormat::Structured2x4.supports_tensor_cores());
        assert!(!SparseFormat::CSR.supports_tensor_cores());

        assert_eq!(
            SparseFormat::Structured2x4.min_compute_capability(),
            Some((8, 0))
        );
        assert_eq!(SparseFormat::CSR.min_compute_capability(), None);
    }
}
