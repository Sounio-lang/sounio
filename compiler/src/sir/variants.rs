//! Kernel Variant Generation System for Auto-Tuning
//!
//! This module implements the infrastructure for generating 20-200 kernel variants,
//! measuring their performance, and selecting the optimal configuration (theta*).
//!
//! # Architecture
//!
//! ```text
//! VariantSpace (defines the search space)
//!       │
//!       ▼
//! VariantGenerator (samples the space)
//!       │
//!       ▼
//! KernelVariant[] (concrete configurations)
//!       │
//!       ▼
//! VariantRegistry (stores variants + measurements)
//!       │
//!       ▼
//! VariantSelector (selects optimal θ*)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use sounio::sir::variants::*;
//!
//! // Define the search space
//! let space = VariantSpace::default()
//!     .with_tile_options(vec![(16, 16, 16), (32, 32, 16), (64, 64, 16)])
//!     .with_unroll_options(vec![1, 2, 4, 8]);
//!
//! // Generate variants using Latin Hypercube Sampling
//! let mut generator = VariantGenerator::new(space);
//! let variants = generator.generate_latin_hypercube(50);
//!
//! // Measure and store results
//! let mut registry = VariantRegistry::new();
//! for variant in variants {
//!     registry.register(variant);
//!     // ... measure on hardware ...
//!     registry.record_measurement(variant.id, measurement);
//! }
//!
//! // Select the best variant
//! let selector = VariantSelector::new(&registry);
//! let best = selector.select_best_time();
//! ```

use rustc_hash::FxHashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for a kernel variant configuration.
///
/// The ID encodes the specific combination of tuning parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariantId(pub u64);

impl VariantId {
    /// Create a new variant ID from configuration parameters.
    pub fn from_config(
        tile_idx: usize,
        unroll_idx: usize,
        layout_idx: usize,
        vector_width_idx: usize,
        shared_mem_idx: usize,
        precision_idx: usize,
    ) -> Self {
        // Pack indices into a single 64-bit ID
        let id = (tile_idx as u64)
            | ((unroll_idx as u64) << 8)
            | ((layout_idx as u64) << 16)
            | ((vector_width_idx as u64) << 24)
            | ((shared_mem_idx as u64) << 32)
            | ((precision_idx as u64) << 40);
        Self(id)
    }

    /// Generate a new unique ID using a counter.
    pub fn new_sequential(counter: u64) -> Self {
        Self(counter)
    }
}

impl fmt::Display for VariantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "var_{:016x}", self.0)
    }
}

/// Memory layout options for kernel data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C-style) layout.
    RowMajor,
    /// Column-major (Fortran-style) layout.
    ColMajor,
    /// Swizzled layout for bank conflict avoidance.
    Swizzled { block_m: u32, block_n: u32 },
    /// Blocked layout for cache efficiency.
    Blocked { block_size: u32 },
    /// Z-order (Morton) layout for 2D locality.
    ZOrder,
}

impl Default for Layout {
    fn default() -> Self {
        Layout::RowMajor
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Layout::RowMajor => write!(f, "row_major"),
            Layout::ColMajor => write!(f, "col_major"),
            Layout::Swizzled { block_m, block_n } => write!(f, "swizzled({}x{})", block_m, block_n),
            Layout::Blocked { block_size } => write!(f, "blocked({})", block_size),
            Layout::ZOrder => write!(f, "z_order"),
        }
    }
}

/// Precision options for kernel computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// Full 64-bit floating point.
    F64,
    /// Standard 32-bit floating point.
    F32,
    /// Half precision 16-bit float.
    F16,
    /// BFloat16 (same exponent range as F32).
    BF16,
    /// FP8 E4M3 (higher precision FP8).
    F8E4M3,
    /// FP8 E5M2 (larger range FP8).
    F8E5M2,
    /// 4-bit floating point (extreme quantization).
    F4,
    /// Mixed precision (F16/BF16 compute, F32 accumulate).
    Mixed,
    /// TensorFloat-32 (TF32).
    TF32,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::F32
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::F64 => write!(f, "f64"),
            Precision::F32 => write!(f, "f32"),
            Precision::F16 => write!(f, "f16"),
            Precision::BF16 => write!(f, "bf16"),
            Precision::F8E4M3 => write!(f, "f8e4m3"),
            Precision::F8E5M2 => write!(f, "f8e5m2"),
            Precision::F4 => write!(f, "f4"),
            Precision::Mixed => write!(f, "mixed"),
            Precision::TF32 => write!(f, "tf32"),
        }
    }
}

// ============================================================================
// Variant Space
// ============================================================================

/// Defines the space of possible kernel variants for auto-tuning.
///
/// This represents the search space that the auto-tuner will explore.
/// Each dimension has a set of valid options, and the full space is
/// the Cartesian product of all dimensions.
#[derive(Debug, Clone)]
pub struct VariantSpace {
    /// Tile size options as (M, N, K) tuples for matrix operations.
    pub tile_options: Vec<(u32, u32, u32)>,

    /// Loop unroll factor options.
    pub unroll_options: Vec<u32>,

    /// Memory layout options.
    pub layout_options: Vec<Layout>,

    /// SIMD vector width options (1 = scalar, 2, 4, 8, etc.).
    pub vector_widths: Vec<u32>,

    /// Shared memory allocation options (bytes).
    pub shared_mem_options: Vec<usize>,

    /// Precision/data type options.
    pub precision_options: Vec<Precision>,

    /// Block size options (threads per block).
    pub block_sizes: Vec<u32>,

    /// Number of pipeline stages for software pipelining.
    pub pipeline_stages: Vec<u32>,

    /// Thread coarsening factor options.
    pub coarsening_factors: Vec<u32>,

    /// Prefetch distance options (number of iterations ahead).
    pub prefetch_distances: Vec<u32>,
}

impl Default for VariantSpace {
    fn default() -> Self {
        Self {
            tile_options: vec![
                (16, 16, 16),
                (32, 32, 16),
                (64, 64, 16),
                (128, 64, 16),
                (64, 128, 16),
                (128, 128, 16),
            ],
            unroll_options: vec![1, 2, 4, 8],
            layout_options: vec![Layout::RowMajor, Layout::ColMajor, Layout::Swizzled { block_m: 8, block_n: 64 }],
            vector_widths: vec![1, 2, 4, 8],
            shared_mem_options: vec![16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024],
            precision_options: vec![Precision::F32, Precision::F16, Precision::BF16],
            block_sizes: vec![64, 128, 256, 512],
            pipeline_stages: vec![1, 2, 3, 4],
            coarsening_factors: vec![1, 2, 4],
            prefetch_distances: vec![0, 1, 2, 4],
        }
    }
}

impl VariantSpace {
    /// Create a new empty variant space.
    pub fn new() -> Self {
        Self {
            tile_options: Vec::new(),
            unroll_options: Vec::new(),
            layout_options: Vec::new(),
            vector_widths: Vec::new(),
            shared_mem_options: Vec::new(),
            precision_options: Vec::new(),
            block_sizes: Vec::new(),
            pipeline_stages: Vec::new(),
            coarsening_factors: Vec::new(),
            prefetch_distances: Vec::new(),
        }
    }

    /// Set tile options (builder pattern).
    pub fn with_tile_options(mut self, options: Vec<(u32, u32, u32)>) -> Self {
        self.tile_options = options;
        self
    }

    /// Set unroll options (builder pattern).
    pub fn with_unroll_options(mut self, options: Vec<u32>) -> Self {
        self.unroll_options = options;
        self
    }

    /// Set layout options (builder pattern).
    pub fn with_layout_options(mut self, options: Vec<Layout>) -> Self {
        self.layout_options = options;
        self
    }

    /// Set vector width options (builder pattern).
    pub fn with_vector_widths(mut self, options: Vec<u32>) -> Self {
        self.vector_widths = options;
        self
    }

    /// Set shared memory options (builder pattern).
    pub fn with_shared_mem_options(mut self, options: Vec<usize>) -> Self {
        self.shared_mem_options = options;
        self
    }

    /// Set precision options (builder pattern).
    pub fn with_precision_options(mut self, options: Vec<Precision>) -> Self {
        self.precision_options = options;
        self
    }

    /// Set block size options (builder pattern).
    pub fn with_block_sizes(mut self, options: Vec<u32>) -> Self {
        self.block_sizes = options;
        self
    }

    /// Set pipeline stage options (builder pattern).
    pub fn with_pipeline_stages(mut self, options: Vec<u32>) -> Self {
        self.pipeline_stages = options;
        self
    }

    /// Set coarsening factor options (builder pattern).
    pub fn with_coarsening_factors(mut self, options: Vec<u32>) -> Self {
        self.coarsening_factors = options;
        self
    }

    /// Set prefetch distance options (builder pattern).
    pub fn with_prefetch_distances(mut self, options: Vec<u32>) -> Self {
        self.prefetch_distances = options;
        self
    }

    /// Calculate the total number of variants in the full Cartesian product.
    pub fn total_variants(&self) -> usize {
        let tile = self.tile_options.len().max(1);
        let unroll = self.unroll_options.len().max(1);
        let layout = self.layout_options.len().max(1);
        let vector = self.vector_widths.len().max(1);
        let shared = self.shared_mem_options.len().max(1);
        let precision = self.precision_options.len().max(1);
        let block = self.block_sizes.len().max(1);
        let pipeline = self.pipeline_stages.len().max(1);
        let coarsen = self.coarsening_factors.len().max(1);
        let prefetch = self.prefetch_distances.len().max(1);

        tile * unroll * layout * vector * shared * precision * block * pipeline * coarsen * prefetch
    }

    /// Get the number of dimensions with multiple options.
    pub fn active_dimensions(&self) -> usize {
        let mut count = 0;
        if self.tile_options.len() > 1 { count += 1; }
        if self.unroll_options.len() > 1 { count += 1; }
        if self.layout_options.len() > 1 { count += 1; }
        if self.vector_widths.len() > 1 { count += 1; }
        if self.shared_mem_options.len() > 1 { count += 1; }
        if self.precision_options.len() > 1 { count += 1; }
        if self.block_sizes.len() > 1 { count += 1; }
        if self.pipeline_stages.len() > 1 { count += 1; }
        if self.coarsening_factors.len() > 1 { count += 1; }
        if self.prefetch_distances.len() > 1 { count += 1; }
        count
    }

    /// Create a space optimized for GEMM (matrix multiplication) kernels.
    pub fn for_gemm() -> Self {
        Self {
            tile_options: vec![
                (64, 64, 16),
                (128, 64, 16),
                (64, 128, 16),
                (128, 128, 16),
                (128, 128, 32),
                (256, 64, 16),
                (64, 256, 16),
            ],
            unroll_options: vec![1, 2, 4],
            layout_options: vec![
                Layout::RowMajor,
                Layout::ColMajor,
                Layout::Swizzled { block_m: 8, block_n: 64 },
            ],
            vector_widths: vec![4, 8],
            shared_mem_options: vec![32 * 1024, 48 * 1024, 64 * 1024, 96 * 1024],
            precision_options: vec![Precision::F32, Precision::F16, Precision::BF16, Precision::TF32],
            block_sizes: vec![128, 256],
            pipeline_stages: vec![2, 3, 4],
            coarsening_factors: vec![1, 2],
            prefetch_distances: vec![1, 2],
        }
    }

    /// Create a space optimized for reduction kernels.
    pub fn for_reduction() -> Self {
        Self {
            tile_options: vec![(256, 1, 1), (512, 1, 1), (1024, 1, 1)],
            unroll_options: vec![2, 4, 8, 16],
            layout_options: vec![Layout::RowMajor],
            vector_widths: vec![1, 4, 8],
            shared_mem_options: vec![4 * 1024, 8 * 1024, 16 * 1024],
            precision_options: vec![Precision::F32, Precision::F64],
            block_sizes: vec![128, 256, 512, 1024],
            pipeline_stages: vec![1],
            coarsening_factors: vec![4, 8, 16, 32],
            prefetch_distances: vec![0, 2],
        }
    }

    /// Create a space optimized for stencil kernels.
    pub fn for_stencil() -> Self {
        Self {
            tile_options: vec![
                (16, 16, 1),
                (32, 32, 1),
                (32, 16, 1),
                (16, 32, 1),
            ],
            unroll_options: vec![1, 2, 4],
            layout_options: vec![
                Layout::RowMajor,
                Layout::Blocked { block_size: 16 },
                Layout::ZOrder,
            ],
            vector_widths: vec![1, 4],
            shared_mem_options: vec![16 * 1024, 32 * 1024, 48 * 1024],
            precision_options: vec![Precision::F32, Precision::F64],
            block_sizes: vec![128, 256],
            pipeline_stages: vec![1, 2],
            coarsening_factors: vec![1, 2],
            prefetch_distances: vec![0, 1],
        }
    }

    /// Create a space optimized for elementwise kernels.
    pub fn for_elementwise() -> Self {
        Self {
            tile_options: vec![(128, 1, 1), (256, 1, 1), (512, 1, 1)],
            unroll_options: vec![1, 4, 8],
            layout_options: vec![Layout::RowMajor],
            vector_widths: vec![1, 4, 8],
            shared_mem_options: vec![0], // No shared memory needed
            precision_options: vec![Precision::F32, Precision::F16],
            block_sizes: vec![128, 256, 512],
            pipeline_stages: vec![1],
            coarsening_factors: vec![1, 4, 8],
            prefetch_distances: vec![0],
        }
    }
}

// ============================================================================
// Kernel Variant
// ============================================================================

/// A concrete kernel variant with specific tuning parameters.
#[derive(Debug, Clone)]
pub struct KernelVariant {
    /// Unique identifier for this variant.
    pub id: VariantId,

    /// Name of the kernel this is a variant of.
    pub kernel_name: String,

    /// Tile dimensions (M, N, K).
    pub tile_size: (u32, u32, u32),

    /// Loop unroll factor.
    pub unroll_factor: u32,

    /// Memory layout.
    pub layout: Layout,

    /// SIMD vector width.
    pub vector_width: u32,

    /// Shared memory size in bytes.
    pub shared_mem_bytes: usize,

    /// Computational precision.
    pub precision: Precision,

    /// Block size (threads per block).
    pub block_size: u32,

    /// Number of software pipeline stages.
    pub pipeline_stages: u32,

    /// Thread coarsening factor.
    pub coarsening_factor: u32,

    /// Prefetch distance (iterations ahead).
    pub prefetch_distance: u32,

    /// Whether this variant uses tensor cores.
    pub uses_tensor_cores: bool,

    /// Estimated register usage per thread.
    pub estimated_registers: Option<u32>,

    /// Estimated occupancy (0.0-1.0).
    pub estimated_occupancy: Option<f64>,
}

impl KernelVariant {
    /// Create a new kernel variant with default values.
    pub fn new(id: VariantId, kernel_name: impl Into<String>) -> Self {
        Self {
            id,
            kernel_name: kernel_name.into(),
            tile_size: (16, 16, 16),
            unroll_factor: 1,
            layout: Layout::RowMajor,
            vector_width: 1,
            shared_mem_bytes: 0,
            precision: Precision::F32,
            block_size: 256,
            pipeline_stages: 1,
            coarsening_factor: 1,
            prefetch_distance: 0,
            uses_tensor_cores: false,
            estimated_registers: None,
            estimated_occupancy: None,
        }
    }

    /// Create a unique hash for this variant's configuration.
    pub fn config_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        self.tile_size.hash(&mut hasher);
        self.unroll_factor.hash(&mut hasher);
        // Layout hash
        match &self.layout {
            Layout::RowMajor => 0u32.hash(&mut hasher),
            Layout::ColMajor => 1u32.hash(&mut hasher),
            Layout::Swizzled { block_m, block_n } => {
                2u32.hash(&mut hasher);
                block_m.hash(&mut hasher);
                block_n.hash(&mut hasher);
            }
            Layout::Blocked { block_size } => {
                3u32.hash(&mut hasher);
                block_size.hash(&mut hasher);
            }
            Layout::ZOrder => 4u32.hash(&mut hasher),
        }
        self.vector_width.hash(&mut hasher);
        self.shared_mem_bytes.hash(&mut hasher);
        // Precision hash
        match self.precision {
            Precision::F64 => 0u32.hash(&mut hasher),
            Precision::F32 => 1u32.hash(&mut hasher),
            Precision::F16 => 2u32.hash(&mut hasher),
            Precision::BF16 => 3u32.hash(&mut hasher),
            Precision::F8E4M3 => 4u32.hash(&mut hasher),
            Precision::F8E5M2 => 5u32.hash(&mut hasher),
            Precision::F4 => 6u32.hash(&mut hasher),
            Precision::Mixed => 7u32.hash(&mut hasher),
            Precision::TF32 => 8u32.hash(&mut hasher),
        }
        self.block_size.hash(&mut hasher);
        self.pipeline_stages.hash(&mut hasher);
        self.coarsening_factor.hash(&mut hasher);
        self.prefetch_distance.hash(&mut hasher);

        hasher.finish()
    }

    /// Calculate effective threads per element.
    pub fn threads_per_element(&self) -> f64 {
        self.block_size as f64 / (self.tile_size.0 * self.tile_size.1) as f64
    }

    /// Check if this variant is likely valid for the given constraints.
    pub fn is_valid_for(&self, max_registers: u32, max_shared_mem: usize, max_block_size: u32) -> bool {
        // Check block size
        if self.block_size > max_block_size {
            return false;
        }

        // Check shared memory
        if self.shared_mem_bytes > max_shared_mem {
            return false;
        }

        // Check estimated registers if available
        if let Some(regs) = self.estimated_registers {
            if regs > max_registers {
                return false;
            }
        }

        true
    }

    /// Generate a human-readable description.
    pub fn description(&self) -> String {
        format!(
            "tile={}x{}x{}, unroll={}, layout={}, vec={}, smem={}KB, prec={}, block={}, pipe={}",
            self.tile_size.0, self.tile_size.1, self.tile_size.2,
            self.unroll_factor,
            self.layout,
            self.vector_width,
            self.shared_mem_bytes / 1024,
            self.precision,
            self.block_size,
            self.pipeline_stages,
        )
    }
}

impl fmt::Display for KernelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]: {}", self.kernel_name, self.id, self.description())
    }
}

// ============================================================================
// Performance Measurement
// ============================================================================

/// Performance measurement for a kernel variant.
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Execution time in nanoseconds.
    pub execution_time_ns: u64,

    /// Achieved memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,

    /// Achieved occupancy (0.0-1.0).
    pub occupancy: f64,

    /// Actual register usage per thread.
    pub register_usage: u32,

    /// Actual shared memory usage in bytes.
    pub shared_mem_usage: u32,

    /// Epistemic quality score from attention metrics (0.0-1.0).
    ///
    /// This captures the "confidence" in numerical results:
    /// - Higher values indicate more reliable/stable computations
    /// - Lower values indicate potential numerical issues
    pub epistemic_quality: f64,

    /// Number of samples this measurement is averaged over.
    pub sample_count: u32,

    /// Standard deviation of execution time (if multiple samples).
    pub time_stddev_ns: Option<u64>,

    /// Power consumption in Watts (if available).
    pub power_watts: Option<f64>,

    /// Achieved FLOPS (floating point operations per second).
    pub achieved_flops: Option<f64>,

    /// L2 cache hit rate (0.0-1.0).
    pub l2_hit_rate: Option<f64>,

    /// DRAM utilization (0.0-1.0).
    pub dram_utilization: Option<f64>,

    /// Compute utilization (0.0-1.0).
    pub compute_utilization: Option<f64>,

    /// Warp execution efficiency (0.0-1.0).
    pub warp_efficiency: Option<f64>,
}

impl PerformanceMeasurement {
    /// Create a new measurement with minimal required fields.
    pub fn new(execution_time_ns: u64) -> Self {
        Self {
            execution_time_ns,
            memory_bandwidth_gbps: 0.0,
            occupancy: 0.0,
            register_usage: 0,
            shared_mem_usage: 0,
            epistemic_quality: 1.0,
            sample_count: 1,
            time_stddev_ns: None,
            power_watts: None,
            achieved_flops: None,
            l2_hit_rate: None,
            dram_utilization: None,
            compute_utilization: None,
            warp_efficiency: None,
        }
    }

    /// Create a measurement from multiple samples.
    pub fn from_samples(times_ns: &[u64]) -> Self {
        if times_ns.is_empty() {
            return Self::new(0);
        }

        let sum: u64 = times_ns.iter().sum();
        let mean = sum / times_ns.len() as u64;

        let variance: f64 = times_ns.iter()
            .map(|&t| {
                let diff = t as f64 - mean as f64;
                diff * diff
            })
            .sum::<f64>() / times_ns.len() as f64;

        let stddev = variance.sqrt() as u64;

        Self {
            execution_time_ns: mean,
            memory_bandwidth_gbps: 0.0,
            occupancy: 0.0,
            register_usage: 0,
            shared_mem_usage: 0,
            epistemic_quality: 1.0,
            sample_count: times_ns.len() as u32,
            time_stddev_ns: Some(stddev),
            power_watts: None,
            achieved_flops: None,
            l2_hit_rate: None,
            dram_utilization: None,
            compute_utilization: None,
            warp_efficiency: None,
        }
    }

    /// Calculate execution time in milliseconds.
    pub fn execution_time_ms(&self) -> f64 {
        self.execution_time_ns as f64 / 1_000_000.0
    }

    /// Calculate energy efficiency (FLOPS per Watt) if data is available.
    pub fn energy_efficiency(&self) -> Option<f64> {
        match (self.achieved_flops, self.power_watts) {
            (Some(flops), Some(watts)) if watts > 0.0 => Some(flops / watts),
            _ => None,
        }
    }

    /// Calculate coefficient of variation (stddev/mean) for timing stability.
    pub fn timing_stability(&self) -> Option<f64> {
        self.time_stddev_ns.map(|stddev| {
            if self.execution_time_ns > 0 {
                1.0 - (stddev as f64 / self.execution_time_ns as f64).min(1.0)
            } else {
                0.0
            }
        })
    }

    /// Calculate a composite performance score (higher is better).
    ///
    /// This combines multiple metrics into a single score for ranking.
    pub fn composite_score(&self) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Inverse of execution time (faster is better)
        if self.execution_time_ns > 0 {
            score += (1_000_000_000.0 / self.execution_time_ns as f64) * 10.0;
            weight_sum += 10.0;
        }

        // Memory bandwidth utilization
        if self.memory_bandwidth_gbps > 0.0 {
            score += self.memory_bandwidth_gbps * 0.1;
            weight_sum += 0.1 * 3000.0; // Normalize assuming max ~3000 GB/s
        }

        // Occupancy
        if self.occupancy > 0.0 {
            score += self.occupancy * 5.0;
            weight_sum += 5.0;
        }

        // Epistemic quality
        score += self.epistemic_quality * 2.0;
        weight_sum += 2.0;

        // Compute utilization
        if let Some(util) = self.compute_utilization {
            score += util * 3.0;
            weight_sum += 3.0;
        }

        // Warp efficiency
        if let Some(eff) = self.warp_efficiency {
            score += eff * 2.0;
            weight_sum += 2.0;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.0
        }
    }

    /// Check if this measurement dominates another (Pareto dominance).
    ///
    /// Returns true if this measurement is better or equal in all metrics
    /// and strictly better in at least one.
    pub fn dominates(&self, other: &Self) -> bool {
        let mut dominated_in_at_least_one = false;

        // Time: lower is better
        if self.execution_time_ns > other.execution_time_ns {
            return false;
        }
        if self.execution_time_ns < other.execution_time_ns {
            dominated_in_at_least_one = true;
        }

        // Bandwidth: higher is better
        if self.memory_bandwidth_gbps < other.memory_bandwidth_gbps {
            return false;
        }
        if self.memory_bandwidth_gbps > other.memory_bandwidth_gbps {
            dominated_in_at_least_one = true;
        }

        // Epistemic quality: higher is better
        if self.epistemic_quality < other.epistemic_quality {
            return false;
        }
        if self.epistemic_quality > other.epistemic_quality {
            dominated_in_at_least_one = true;
        }

        dominated_in_at_least_one
    }
}

// ============================================================================
// Variant Generator
// ============================================================================

/// Generator for kernel variants from a variant space.
pub struct VariantGenerator {
    space: VariantSpace,
    next_id: u64,
    rng_state: u64,
}

impl VariantGenerator {
    /// Create a new generator for the given variant space.
    pub fn new(space: VariantSpace) -> Self {
        Self {
            space,
            next_id: 0,
            rng_state: 0x853c49e6748fea9b, // Arbitrary seed
        }
    }

    /// Create a generator with a specific random seed.
    pub fn with_seed(space: VariantSpace, seed: u64) -> Self {
        Self {
            space,
            next_id: 0,
            rng_state: seed,
        }
    }

    /// Simple xorshift64 PRNG for deterministic sampling.
    fn rand_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generate a random index in the range [0, max).
    fn rand_index(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.rand_u64() as usize) % max
    }

    /// Generate the next sequential variant ID.
    fn next_variant_id(&mut self) -> VariantId {
        let id = VariantId::new_sequential(self.next_id);
        self.next_id += 1;
        id
    }

    /// Generate all variants (full Cartesian product).
    ///
    /// Warning: This can generate a very large number of variants.
    pub fn generate_all(&mut self, kernel_name: &str) -> Vec<KernelVariant> {
        let mut variants = Vec::with_capacity(self.space.total_variants());

        let tiles = if self.space.tile_options.is_empty() { vec![(16, 16, 16)] } else { self.space.tile_options.clone() };
        let unrolls = if self.space.unroll_options.is_empty() { vec![1] } else { self.space.unroll_options.clone() };
        let layouts = if self.space.layout_options.is_empty() { vec![Layout::RowMajor] } else { self.space.layout_options.clone() };
        let vectors = if self.space.vector_widths.is_empty() { vec![1] } else { self.space.vector_widths.clone() };
        let shareds = if self.space.shared_mem_options.is_empty() { vec![0] } else { self.space.shared_mem_options.clone() };
        let precisions = if self.space.precision_options.is_empty() { vec![Precision::F32] } else { self.space.precision_options.clone() };
        let blocks = if self.space.block_sizes.is_empty() { vec![256] } else { self.space.block_sizes.clone() };
        let pipelines = if self.space.pipeline_stages.is_empty() { vec![1] } else { self.space.pipeline_stages.clone() };
        let coarsens = if self.space.coarsening_factors.is_empty() { vec![1] } else { self.space.coarsening_factors.clone() };
        let prefetches = if self.space.prefetch_distances.is_empty() { vec![0] } else { self.space.prefetch_distances.clone() };

        for &tile in &tiles {
            for &unroll in &unrolls {
                for layout in &layouts {
                    for &vector in &vectors {
                        for &shared in &shareds {
                            for &precision in &precisions {
                                for &block in &blocks {
                                    for &pipeline in &pipelines {
                                        for &coarsen in &coarsens {
                                            for &prefetch in &prefetches {
                                                let id = self.next_variant_id();
                                                let mut variant = KernelVariant::new(id, kernel_name);
                                                variant.tile_size = tile;
                                                variant.unroll_factor = unroll;
                                                variant.layout = layout.clone();
                                                variant.vector_width = vector;
                                                variant.shared_mem_bytes = shared;
                                                variant.precision = precision;
                                                variant.block_size = block;
                                                variant.pipeline_stages = pipeline;
                                                variant.coarsening_factor = coarsen;
                                                variant.prefetch_distance = prefetch;
                                                variants.push(variant);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        variants
    }

    /// Generate a random sample of n variants.
    pub fn generate_sampled(&mut self, kernel_name: &str, n: usize) -> Vec<KernelVariant> {
        let mut variants = Vec::with_capacity(n);

        for _ in 0..n {
            let id = self.next_variant_id();
            let variant = self.random_variant(id, kernel_name);
            variants.push(variant);
        }

        variants
    }

    /// Generate a single random variant.
    fn random_variant(&mut self, id: VariantId, kernel_name: &str) -> KernelVariant {
        let mut variant = KernelVariant::new(id, kernel_name);

        if !self.space.tile_options.is_empty() {
            let idx = self.rand_index(self.space.tile_options.len());
            variant.tile_size = self.space.tile_options[idx];
        }

        if !self.space.unroll_options.is_empty() {
            let idx = self.rand_index(self.space.unroll_options.len());
            variant.unroll_factor = self.space.unroll_options[idx];
        }

        if !self.space.layout_options.is_empty() {
            let idx = self.rand_index(self.space.layout_options.len());
            variant.layout = self.space.layout_options[idx].clone();
        }

        if !self.space.vector_widths.is_empty() {
            let idx = self.rand_index(self.space.vector_widths.len());
            variant.vector_width = self.space.vector_widths[idx];
        }

        if !self.space.shared_mem_options.is_empty() {
            let idx = self.rand_index(self.space.shared_mem_options.len());
            variant.shared_mem_bytes = self.space.shared_mem_options[idx];
        }

        if !self.space.precision_options.is_empty() {
            let idx = self.rand_index(self.space.precision_options.len());
            variant.precision = self.space.precision_options[idx];
        }

        if !self.space.block_sizes.is_empty() {
            let idx = self.rand_index(self.space.block_sizes.len());
            variant.block_size = self.space.block_sizes[idx];
        }

        if !self.space.pipeline_stages.is_empty() {
            let idx = self.rand_index(self.space.pipeline_stages.len());
            variant.pipeline_stages = self.space.pipeline_stages[idx];
        }

        if !self.space.coarsening_factors.is_empty() {
            let idx = self.rand_index(self.space.coarsening_factors.len());
            variant.coarsening_factor = self.space.coarsening_factors[idx];
        }

        if !self.space.prefetch_distances.is_empty() {
            let idx = self.rand_index(self.space.prefetch_distances.len());
            variant.prefetch_distance = self.space.prefetch_distances[idx];
        }

        variant
    }

    /// Generate n variants using Latin Hypercube Sampling (LHS).
    ///
    /// LHS provides better coverage of the search space than random sampling,
    /// which is particularly useful for Bayesian Optimization warm-starting.
    pub fn generate_latin_hypercube(&mut self, kernel_name: &str, n: usize) -> Vec<KernelVariant> {
        if n == 0 {
            return Vec::new();
        }

        let mut variants = Vec::with_capacity(n);

        // Generate LHS permutations for each dimension
        let tile_perm = self.lhs_permutation(n, self.space.tile_options.len());
        let unroll_perm = self.lhs_permutation(n, self.space.unroll_options.len());
        let layout_perm = self.lhs_permutation(n, self.space.layout_options.len());
        let vector_perm = self.lhs_permutation(n, self.space.vector_widths.len());
        let shared_perm = self.lhs_permutation(n, self.space.shared_mem_options.len());
        let precision_perm = self.lhs_permutation(n, self.space.precision_options.len());
        let block_perm = self.lhs_permutation(n, self.space.block_sizes.len());
        let pipeline_perm = self.lhs_permutation(n, self.space.pipeline_stages.len());
        let coarsen_perm = self.lhs_permutation(n, self.space.coarsening_factors.len());
        let prefetch_perm = self.lhs_permutation(n, self.space.prefetch_distances.len());

        for i in 0..n {
            let id = self.next_variant_id();
            let mut variant = KernelVariant::new(id, kernel_name);

            if !self.space.tile_options.is_empty() {
                variant.tile_size = self.space.tile_options[tile_perm[i]];
            }
            if !self.space.unroll_options.is_empty() {
                variant.unroll_factor = self.space.unroll_options[unroll_perm[i]];
            }
            if !self.space.layout_options.is_empty() {
                variant.layout = self.space.layout_options[layout_perm[i]].clone();
            }
            if !self.space.vector_widths.is_empty() {
                variant.vector_width = self.space.vector_widths[vector_perm[i]];
            }
            if !self.space.shared_mem_options.is_empty() {
                variant.shared_mem_bytes = self.space.shared_mem_options[shared_perm[i]];
            }
            if !self.space.precision_options.is_empty() {
                variant.precision = self.space.precision_options[precision_perm[i]];
            }
            if !self.space.block_sizes.is_empty() {
                variant.block_size = self.space.block_sizes[block_perm[i]];
            }
            if !self.space.pipeline_stages.is_empty() {
                variant.pipeline_stages = self.space.pipeline_stages[pipeline_perm[i]];
            }
            if !self.space.coarsening_factors.is_empty() {
                variant.coarsening_factor = self.space.coarsening_factors[coarsen_perm[i]];
            }
            if !self.space.prefetch_distances.is_empty() {
                variant.prefetch_distance = self.space.prefetch_distances[prefetch_perm[i]];
            }

            variants.push(variant);
        }

        variants
    }

    /// Generate LHS permutation for a dimension.
    ///
    /// Returns a vector of length n where each element is an index into
    /// an options array of the given size. The permutation ensures that
    /// each stratum of the dimension is sampled exactly once (when possible).
    fn lhs_permutation(&mut self, n: usize, options_count: usize) -> Vec<usize> {
        if options_count == 0 {
            return vec![0; n];
        }

        let mut indices: Vec<usize> = (0..n).map(|i| {
            // Map sample index to option index, ensuring coverage
            (i * options_count) / n
        }).collect();

        // Shuffle the indices using Fisher-Yates
        for i in (1..n).rev() {
            let j = self.rand_index(i + 1);
            indices.swap(i, j);
        }

        indices
    }

    /// Generate variants using Sobol sequence (quasi-random, low discrepancy).
    ///
    /// Sobol sequences provide even better coverage than LHS for higher dimensions.
    pub fn generate_sobol(&mut self, kernel_name: &str, n: usize) -> Vec<KernelVariant> {
        // Simplified Sobol-like sequence using gray code
        let mut variants = Vec::with_capacity(n);
        let dims = self.space.active_dimensions();

        for i in 0..n {
            let id = self.next_variant_id();
            let mut variant = KernelVariant::new(id, kernel_name);

            // Use gray code to generate quasi-random indices
            let gray = i ^ (i >> 1);
            let mut dim_offset = 0;

            if !self.space.tile_options.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.tile_options.len());
                variant.tile_size = self.space.tile_options[idx];
                dim_offset += 1;
            }

            if !self.space.unroll_options.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.unroll_options.len());
                variant.unroll_factor = self.space.unroll_options[idx];
                dim_offset += 1;
            }

            if !self.space.layout_options.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.layout_options.len());
                variant.layout = self.space.layout_options[idx].clone();
                dim_offset += 1;
            }

            if !self.space.vector_widths.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.vector_widths.len());
                variant.vector_width = self.space.vector_widths[idx];
                dim_offset += 1;
            }

            if !self.space.shared_mem_options.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.shared_mem_options.len());
                variant.shared_mem_bytes = self.space.shared_mem_options[idx];
                dim_offset += 1;
            }

            if !self.space.precision_options.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.precision_options.len());
                variant.precision = self.space.precision_options[idx];
                dim_offset += 1;
            }

            if !self.space.block_sizes.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.block_sizes.len());
                variant.block_size = self.space.block_sizes[idx];
                dim_offset += 1;
            }

            if !self.space.pipeline_stages.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.pipeline_stages.len());
                variant.pipeline_stages = self.space.pipeline_stages[idx];
                dim_offset += 1;
            }

            if !self.space.coarsening_factors.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.coarsening_factors.len());
                variant.coarsening_factor = self.space.coarsening_factors[idx];
                dim_offset += 1;
            }

            if !self.space.prefetch_distances.is_empty() {
                let idx = self.sobol_index(gray, dim_offset, dims, self.space.prefetch_distances.len());
                variant.prefetch_distance = self.space.prefetch_distances[idx];
            }

            variants.push(variant);
        }

        variants
    }

    /// Helper to convert a gray code value to an option index.
    fn sobol_index(&self, gray: usize, dim: usize, total_dims: usize, options: usize) -> usize {
        if options == 0 {
            return 0;
        }
        // Use bit manipulation to extract dimension-specific index
        let bits_per_dim = (64 / total_dims.max(1)).max(1);
        let shift = dim * bits_per_dim;
        let mask = (1usize << bits_per_dim) - 1;
        let val = (gray >> shift) & mask;
        val % options
    }
}

// ============================================================================
// Variant Registry
// ============================================================================

/// Registry for storing generated variants and their measured performance.
pub struct VariantRegistry {
    /// All registered variants.
    variants: FxHashMap<VariantId, KernelVariant>,

    /// Performance measurements indexed by variant ID.
    measurements: FxHashMap<VariantId, PerformanceMeasurement>,

    /// Index from kernel name to variant IDs.
    kernel_variants: FxHashMap<String, Vec<VariantId>>,

    /// Variants marked as "best" for each kernel.
    best_variants: FxHashMap<String, VariantId>,
}

impl VariantRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            variants: FxHashMap::default(),
            measurements: FxHashMap::default(),
            kernel_variants: FxHashMap::default(),
            best_variants: FxHashMap::default(),
        }
    }

    /// Register a new variant.
    pub fn register(&mut self, variant: KernelVariant) {
        let id = variant.id;
        let kernel_name = variant.kernel_name.clone();

        self.kernel_variants
            .entry(kernel_name)
            .or_insert_with(Vec::new)
            .push(id);

        self.variants.insert(id, variant);
    }

    /// Register multiple variants.
    pub fn register_all(&mut self, variants: impl IntoIterator<Item = KernelVariant>) {
        for variant in variants {
            self.register(variant);
        }
    }

    /// Record a performance measurement for a variant.
    pub fn record_measurement(&mut self, id: VariantId, measurement: PerformanceMeasurement) {
        self.measurements.insert(id, measurement);
    }

    /// Get a variant by ID.
    pub fn get_variant(&self, id: VariantId) -> Option<&KernelVariant> {
        self.variants.get(&id)
    }

    /// Get a measurement by variant ID.
    pub fn get_measurement(&self, id: VariantId) -> Option<&PerformanceMeasurement> {
        self.measurements.get(&id)
    }

    /// Get all variants for a kernel.
    pub fn get_kernel_variants(&self, kernel_name: &str) -> Vec<&KernelVariant> {
        self.kernel_variants
            .get(kernel_name)
            .map(|ids| ids.iter().filter_map(|id| self.variants.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all measured variants for a kernel.
    pub fn get_measured_variants(&self, kernel_name: &str) -> Vec<(&KernelVariant, &PerformanceMeasurement)> {
        self.kernel_variants
            .get(kernel_name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| {
                        let variant = self.variants.get(id)?;
                        let measurement = self.measurements.get(id)?;
                        Some((variant, measurement))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Set the best variant for a kernel.
    pub fn set_best_variant(&mut self, kernel_name: &str, id: VariantId) {
        self.best_variants.insert(kernel_name.to_string(), id);
    }

    /// Get the best variant for a kernel.
    pub fn get_best_variant(&self, kernel_name: &str) -> Option<&KernelVariant> {
        self.best_variants
            .get(kernel_name)
            .and_then(|id| self.variants.get(id))
    }

    /// Get the number of registered variants.
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Get the number of measured variants.
    pub fn measured_count(&self) -> usize {
        self.measurements.len()
    }

    /// Get measurement coverage ratio (measured / total).
    pub fn coverage(&self) -> f64 {
        if self.variants.is_empty() {
            0.0
        } else {
            self.measurements.len() as f64 / self.variants.len() as f64
        }
    }

    /// Get all kernel names with registered variants.
    pub fn kernel_names(&self) -> impl Iterator<Item = &String> {
        self.kernel_variants.keys()
    }

    /// Remove variants without measurements.
    pub fn prune_unmeasured(&mut self) {
        let measured_ids: Vec<VariantId> = self.measurements.keys().copied().collect();
        let measured_set: std::collections::HashSet<VariantId> = measured_ids.iter().copied().collect();

        // Remove unmeasured variants
        self.variants.retain(|id, _| measured_set.contains(id));

        // Update kernel index
        for ids in self.kernel_variants.values_mut() {
            ids.retain(|id| measured_set.contains(id));
        }
    }
}

impl Default for VariantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Variant Selector
// ============================================================================

/// Selection constraints for variant filtering.
#[derive(Debug, Clone)]
pub struct SelectionConstraints {
    /// Maximum register usage per thread.
    pub max_registers: Option<u32>,
    /// Minimum occupancy (0.0-1.0).
    pub min_occupancy: Option<f64>,
    /// Maximum shared memory in bytes.
    pub max_shared_mem: Option<usize>,
    /// Minimum epistemic quality (0.0-1.0).
    pub min_epistemic_quality: Option<f64>,
    /// Maximum execution time in nanoseconds.
    pub max_time_ns: Option<u64>,
    /// Required precision.
    pub required_precision: Option<Precision>,
}

impl Default for SelectionConstraints {
    fn default() -> Self {
        Self {
            max_registers: None,
            min_occupancy: None,
            max_shared_mem: None,
            min_epistemic_quality: None,
            max_time_ns: None,
            required_precision: None,
        }
    }
}

impl SelectionConstraints {
    /// Create new empty constraints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum register constraint.
    pub fn with_max_registers(mut self, max: u32) -> Self {
        self.max_registers = Some(max);
        self
    }

    /// Set minimum occupancy constraint.
    pub fn with_min_occupancy(mut self, min: f64) -> Self {
        self.min_occupancy = Some(min);
        self
    }

    /// Set maximum shared memory constraint.
    pub fn with_max_shared_mem(mut self, max: usize) -> Self {
        self.max_shared_mem = Some(max);
        self
    }

    /// Set minimum epistemic quality constraint.
    pub fn with_min_epistemic_quality(mut self, min: f64) -> Self {
        self.min_epistemic_quality = Some(min);
        self
    }

    /// Set maximum time constraint.
    pub fn with_max_time_ns(mut self, max: u64) -> Self {
        self.max_time_ns = Some(max);
        self
    }

    /// Set required precision constraint.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.required_precision = Some(precision);
        self
    }

    /// Check if a variant and measurement satisfy these constraints.
    pub fn is_satisfied(&self, variant: &KernelVariant, measurement: &PerformanceMeasurement) -> bool {
        if let Some(max_regs) = self.max_registers {
            if measurement.register_usage > max_regs {
                return false;
            }
        }

        if let Some(min_occ) = self.min_occupancy {
            if measurement.occupancy < min_occ {
                return false;
            }
        }

        if let Some(max_smem) = self.max_shared_mem {
            if measurement.shared_mem_usage as usize > max_smem {
                return false;
            }
        }

        if let Some(min_eq) = self.min_epistemic_quality {
            if measurement.epistemic_quality < min_eq {
                return false;
            }
        }

        if let Some(max_time) = self.max_time_ns {
            if measurement.execution_time_ns > max_time {
                return false;
            }
        }

        if let Some(req_prec) = self.required_precision {
            if variant.precision != req_prec {
                return false;
            }
        }

        true
    }
}

/// Selector for choosing the best variant from measured candidates.
pub struct VariantSelector<'a> {
    registry: &'a VariantRegistry,
}

impl<'a> VariantSelector<'a> {
    /// Create a new selector for the given registry.
    pub fn new(registry: &'a VariantRegistry) -> Self {
        Self { registry }
    }

    /// Select the variant with the best (lowest) execution time.
    pub fn select_best_time(&self, kernel_name: &str) -> Option<VariantId> {
        self.registry
            .get_measured_variants(kernel_name)
            .into_iter()
            .min_by_key(|(_, m)| m.execution_time_ns)
            .map(|(v, _)| v.id)
    }

    /// Select the variant with the highest composite performance score.
    pub fn select_best_score(&self, kernel_name: &str) -> Option<VariantId> {
        self.registry
            .get_measured_variants(kernel_name)
            .into_iter()
            .max_by(|(_, m1), (_, m2)| {
                m1.composite_score().partial_cmp(&m2.composite_score()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(v, _)| v.id)
    }

    /// Select the Pareto-optimal variants (multi-objective: time vs. epistemic quality).
    ///
    /// Returns all variants on the Pareto frontier.
    pub fn select_pareto_optimal(&self, kernel_name: &str) -> Vec<VariantId> {
        let measured = self.registry.get_measured_variants(kernel_name);
        let mut pareto_front = Vec::new();

        for (v1, m1) in measured.iter() {
            let is_dominated = measured.iter().any(|(_, m2)| {
                // m2 dominates m1 if m2 is better in all objectives and strictly better in at least one
                m2.dominates(m1)
            });

            if !is_dominated {
                pareto_front.push(v1.id);
            }
        }

        pareto_front
    }

    /// Select the best variant that satisfies the given constraints.
    pub fn select_with_constraints(
        &self,
        kernel_name: &str,
        constraints: &SelectionConstraints,
    ) -> Option<VariantId> {
        self.registry
            .get_measured_variants(kernel_name)
            .into_iter()
            .filter(|(v, m)| constraints.is_satisfied(v, m))
            .min_by_key(|(_, m)| m.execution_time_ns)
            .map(|(v, _)| v.id)
    }

    /// Select the best variant considering energy efficiency.
    pub fn select_best_efficiency(&self, kernel_name: &str) -> Option<VariantId> {
        self.registry
            .get_measured_variants(kernel_name)
            .into_iter()
            .filter_map(|(v, m)| m.energy_efficiency().map(|e| (v, e)))
            .max_by(|(_, e1), (_, e2)| e1.partial_cmp(e2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(v, _)| v.id)
    }

    /// Select the best variant for numerical stability (highest epistemic quality).
    pub fn select_most_stable(&self, kernel_name: &str) -> Option<VariantId> {
        self.registry
            .get_measured_variants(kernel_name)
            .into_iter()
            .max_by(|(_, m1), (_, m2)| {
                m1.epistemic_quality.partial_cmp(&m2.epistemic_quality).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(v, _)| v.id)
    }

    /// Get the top N variants by execution time.
    pub fn select_top_n(&self, kernel_name: &str, n: usize) -> Vec<VariantId> {
        let mut measured: Vec<_> = self.registry.get_measured_variants(kernel_name);
        measured.sort_by_key(|(_, m)| m.execution_time_ns);
        measured.into_iter().take(n).map(|(v, _)| v.id).collect()
    }

    /// Get variants within a percentage threshold of the best.
    pub fn select_within_threshold(&self, kernel_name: &str, threshold_percent: f64) -> Vec<VariantId> {
        let measured = self.registry.get_measured_variants(kernel_name);

        let best_time = measured.iter()
            .map(|(_, m)| m.execution_time_ns)
            .min()
            .unwrap_or(0);

        if best_time == 0 {
            return Vec::new();
        }

        let max_time = (best_time as f64 * (1.0 + threshold_percent / 100.0)) as u64;

        measured
            .into_iter()
            .filter(|(_, m)| m.execution_time_ns <= max_time)
            .map(|(v, _)| v.id)
            .collect()
    }

    /// Generate a summary report for all measured variants of a kernel.
    pub fn generate_report(&self, kernel_name: &str) -> String {
        let measured = self.registry.get_measured_variants(kernel_name);

        if measured.is_empty() {
            return format!("No measured variants for kernel '{}'", kernel_name);
        }

        let mut report = format!("=== Variant Report for '{}' ===\n", kernel_name);
        report.push_str(&format!("Total variants measured: {}\n\n", measured.len()));

        // Sort by execution time
        let mut sorted: Vec<_> = measured;
        sorted.sort_by_key(|(_, m)| m.execution_time_ns);

        // Best variant
        if let Some((best_v, best_m)) = sorted.first() {
            report.push_str(&format!("Best variant: {}\n", best_v.id));
            report.push_str(&format!("  Config: {}\n", best_v.description()));
            report.push_str(&format!("  Time: {:.3} ms\n", best_m.execution_time_ms()));
            report.push_str(&format!("  Occupancy: {:.1}%\n", best_m.occupancy * 100.0));
            report.push_str(&format!("  Epistemic quality: {:.3}\n\n", best_m.epistemic_quality));
        }

        // Statistics
        let times: Vec<f64> = sorted.iter().map(|(_, m)| m.execution_time_ms()).collect();
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let min_time = times.first().copied().unwrap_or(0.0);
        let max_time = times.last().copied().unwrap_or(0.0);

        report.push_str(&format!("Time statistics (ms):\n"));
        report.push_str(&format!("  Min: {:.3}, Max: {:.3}, Mean: {:.3}\n", min_time, max_time, mean_time));
        report.push_str(&format!("  Speedup range: {:.2}x\n\n", max_time / min_time.max(0.001)));

        // Pareto front
        let pareto = self.select_pareto_optimal(kernel_name);
        report.push_str(&format!("Pareto-optimal variants: {}\n", pareto.len()));

        report
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_id_from_config() {
        let id1 = VariantId::from_config(0, 1, 2, 3, 4, 5);
        let id2 = VariantId::from_config(0, 1, 2, 3, 4, 5);
        let id3 = VariantId::from_config(1, 1, 2, 3, 4, 5);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_variant_space_total() {
        let space = VariantSpace::new()
            .with_tile_options(vec![(16, 16, 16), (32, 32, 16)])
            .with_unroll_options(vec![1, 2, 4])
            .with_layout_options(vec![Layout::RowMajor, Layout::ColMajor]);

        // 2 * 3 * 2 = 12
        assert_eq!(space.total_variants(), 12);
    }

    #[test]
    fn test_variant_generator_sampled() {
        let space = VariantSpace::default();
        let mut generator = VariantGenerator::with_seed(space, 12345);
        let variants = generator.generate_sampled("test_kernel", 10);

        assert_eq!(variants.len(), 10);
        for (i, v) in variants.iter().enumerate() {
            assert_eq!(v.kernel_name, "test_kernel");
            // IDs should be sequential
            assert_eq!(v.id.0, i as u64);
        }
    }

    #[test]
    fn test_variant_generator_lhs() {
        let space = VariantSpace::new()
            .with_tile_options(vec![(16, 16, 16), (32, 32, 16), (64, 64, 16)]);
        let mut generator = VariantGenerator::with_seed(space, 42);
        let variants = generator.generate_latin_hypercube("lhs_kernel", 6);

        assert_eq!(variants.len(), 6);
        // LHS should cover all tile options at least once in 6 samples with 3 options
    }

    #[test]
    fn test_variant_registry() {
        let mut registry = VariantRegistry::new();

        let v1 = KernelVariant::new(VariantId(1), "kernel_a");
        let v2 = KernelVariant::new(VariantId(2), "kernel_a");
        let v3 = KernelVariant::new(VariantId(3), "kernel_b");

        registry.register(v1);
        registry.register(v2);
        registry.register(v3);

        assert_eq!(registry.variant_count(), 3);
        assert_eq!(registry.get_kernel_variants("kernel_a").len(), 2);
        assert_eq!(registry.get_kernel_variants("kernel_b").len(), 1);
    }

    #[test]
    fn test_performance_measurement() {
        let samples = vec![100u64, 110, 105, 95, 100];
        let measurement = PerformanceMeasurement::from_samples(&samples);

        assert_eq!(measurement.sample_count, 5);
        assert_eq!(measurement.execution_time_ns, 102); // Mean
        assert!(measurement.time_stddev_ns.is_some());
    }

    #[test]
    fn test_variant_selector_best_time() {
        let mut registry = VariantRegistry::new();

        let v1 = KernelVariant::new(VariantId(1), "test");
        let v2 = KernelVariant::new(VariantId(2), "test");

        registry.register(v1);
        registry.register(v2);

        let mut m1 = PerformanceMeasurement::new(1000);
        let mut m2 = PerformanceMeasurement::new(500);
        m1.occupancy = 0.5;
        m2.occupancy = 0.8;

        registry.record_measurement(VariantId(1), m1);
        registry.record_measurement(VariantId(2), m2);

        let selector = VariantSelector::new(&registry);
        let best = selector.select_best_time("test");

        assert_eq!(best, Some(VariantId(2))); // v2 is faster
    }

    #[test]
    fn test_pareto_dominance() {
        let m1 = PerformanceMeasurement {
            execution_time_ns: 100,
            memory_bandwidth_gbps: 100.0,
            epistemic_quality: 0.9,
            ..PerformanceMeasurement::new(100)
        };

        let m2 = PerformanceMeasurement {
            execution_time_ns: 150,
            memory_bandwidth_gbps: 80.0,
            epistemic_quality: 0.8,
            ..PerformanceMeasurement::new(150)
        };

        let m3 = PerformanceMeasurement {
            execution_time_ns: 100,
            memory_bandwidth_gbps: 100.0,
            epistemic_quality: 0.85,
            ..PerformanceMeasurement::new(100)
        };

        assert!(m1.dominates(&m2)); // m1 is better in all
        assert!(!m2.dominates(&m1));
        assert!(m1.dominates(&m3)); // m1 is better in epistemic_quality
        assert!(!m3.dominates(&m1));
    }

    #[test]
    fn test_selection_constraints() {
        let constraints = SelectionConstraints::new()
            .with_max_registers(64)
            .with_min_occupancy(0.5);

        let variant = KernelVariant::new(VariantId(1), "test");

        let good_measurement = PerformanceMeasurement {
            register_usage: 32,
            occupancy: 0.75,
            ..PerformanceMeasurement::new(100)
        };

        let bad_measurement = PerformanceMeasurement {
            register_usage: 128,
            occupancy: 0.3,
            ..PerformanceMeasurement::new(100)
        };

        assert!(constraints.is_satisfied(&variant, &good_measurement));
        assert!(!constraints.is_satisfied(&variant, &bad_measurement));
    }

    #[test]
    fn test_variant_space_presets() {
        let gemm = VariantSpace::for_gemm();
        assert!(!gemm.tile_options.is_empty());
        assert!(gemm.tile_options.iter().any(|t| t.0 >= 64 && t.1 >= 64));

        let reduction = VariantSpace::for_reduction();
        assert!(reduction.coarsening_factors.iter().any(|&c| c >= 8));

        let stencil = VariantSpace::for_stencil();
        assert!(stencil.layout_options.iter().any(|l| matches!(l, Layout::Blocked { .. })));

        let elementwise = VariantSpace::for_elementwise();
        assert!(elementwise.shared_mem_options.contains(&0));
    }

    #[test]
    fn test_kernel_variant_config_hash() {
        let mut v1 = KernelVariant::new(VariantId(1), "test");
        let mut v2 = KernelVariant::new(VariantId(2), "test");

        v1.tile_size = (32, 32, 16);
        v2.tile_size = (32, 32, 16);
        v1.unroll_factor = 4;
        v2.unroll_factor = 4;

        // Same config should produce same hash
        assert_eq!(v1.config_hash(), v2.config_hash());

        // Different config should produce different hash
        v2.unroll_factor = 8;
        assert_ne!(v1.config_hash(), v2.config_hash());
    }

    #[test]
    fn test_generate_sobol() {
        let space = VariantSpace::default();
        let mut generator = VariantGenerator::with_seed(space, 123);
        let variants = generator.generate_sobol("sobol_test", 20);

        assert_eq!(variants.len(), 20);
        // Sobol should produce diverse configurations (at least 5 unique from 20)
        let unique_configs: std::collections::HashSet<_> = variants.iter().map(|v| v.config_hash()).collect();
        assert!(unique_configs.len() >= 5, "Sobol should produce at least 5 unique configurations from 20 variants, got {}", unique_configs.len());
    }
}
