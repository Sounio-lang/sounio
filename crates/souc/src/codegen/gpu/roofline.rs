//! Roofline Model (Phase 12)
//!
//! Implements the roofline performance model for analyzing GPU kernel
//! performance characteristics and identifying optimization opportunities.

use super::costs::{ArchPeakPerf, CostDatabase, FlopsCount, MemoryTraffic};
use super::ir::{CudaArch, GpuKernel};

// ============================================================================
// Roofline Model
// ============================================================================

/// Roofline model for a specific GPU architecture
#[derive(Debug, Clone)]
pub struct RooflineModel {
    arch: CudaArch,
    /// Peak compute throughput (GFLOPS)
    peak_compute_gflops: f64,
    /// Peak FP16 compute (GFLOPS)
    peak_fp16_gflops: f64,
    /// Peak Tensor Core compute (GFLOPS)
    peak_tensor_gflops: f64,
    /// Peak memory bandwidth (GB/s)
    peak_bandwidth_gbs: f64,
    /// L2 cache bandwidth (GB/s)
    l2_bandwidth_gbs: f64,
    /// Shared memory bandwidth (GB/s)
    shared_bandwidth_gbs: f64,
    /// Ridge point for FP32 (FLOPS/byte)
    ridge_point_fp32: f64,
    /// Ridge point for Tensor Core (FLOPS/byte)
    ridge_point_tensor: f64,
}

impl RooflineModel {
    /// Create a roofline model for the given architecture
    pub fn for_arch(arch: CudaArch) -> Self {
        let peak = ArchPeakPerf::for_arch(arch);

        // Convert TFLOPS to GFLOPS
        let peak_compute_gflops = peak.fp32_tflops * 1000.0;
        let peak_fp16_gflops = peak.fp16_tflops * 1000.0;
        let peak_tensor_gflops = peak.tensor_fp16_tflops * 1000.0;

        // Ridge point = Peak GFLOPS / Peak GB/s
        let ridge_point_fp32 = peak_compute_gflops / peak.memory_bandwidth_gbs;
        let ridge_point_tensor = peak_tensor_gflops / peak.memory_bandwidth_gbs;

        Self {
            arch,
            peak_compute_gflops,
            peak_fp16_gflops,
            peak_tensor_gflops,
            peak_bandwidth_gbs: peak.memory_bandwidth_gbs,
            l2_bandwidth_gbs: peak.l2_bandwidth_gbs,
            shared_bandwidth_gbs: peak.shared_bandwidth_gbs,
            ridge_point_fp32,
            ridge_point_tensor,
        }
    }

    /// Get the architecture this model was built for
    pub fn arch(&self) -> CudaArch {
        self.arch
    }

    /// Get the ridge point for FP32 operations (FLOPS/byte)
    ///
    /// The ridge point is where the kernel transitions from memory-bound
    /// to compute-bound.
    pub fn ridge_point(&self) -> f64 {
        self.ridge_point_fp32
    }

    /// Get the ridge point for Tensor Core operations
    pub fn ridge_point_tensor(&self) -> f64 {
        self.ridge_point_tensor
    }

    /// Get peak compute performance (GFLOPS)
    pub fn peak_compute(&self) -> f64 {
        self.peak_compute_gflops
    }

    /// Get peak memory bandwidth (GB/s)
    pub fn peak_bandwidth(&self) -> f64 {
        self.peak_bandwidth_gbs
    }

    /// Calculate the peak achievable performance at a given arithmetic intensity
    ///
    /// Returns the roofline ceiling in GFLOPS for the given intensity.
    pub fn peak_at_intensity(&self, intensity: f64) -> f64 {
        // Memory-bound: performance = bandwidth * intensity
        let memory_bound_perf = self.peak_bandwidth_gbs * intensity;

        // Compute-bound: performance = peak compute
        let compute_bound_perf = self.peak_compute_gflops;

        // Roofline is the minimum of the two
        memory_bound_perf.min(compute_bound_perf)
    }

    /// Calculate the peak achievable performance with Tensor Cores
    pub fn peak_at_intensity_tensor(&self, intensity: f64) -> f64 {
        let memory_bound_perf = self.peak_bandwidth_gbs * intensity;
        let compute_bound_perf = self.peak_tensor_gflops;
        memory_bound_perf.min(compute_bound_perf)
    }

    /// Calculate the peak achievable performance with L2 cache
    pub fn peak_at_intensity_l2(&self, intensity: f64) -> f64 {
        let memory_bound_perf = self.l2_bandwidth_gbs * intensity;
        let compute_bound_perf = self.peak_compute_gflops;
        memory_bound_perf.min(compute_bound_perf)
    }

    /// Determine if a kernel is compute or memory bound
    pub fn classify_boundedness(&self, intensity: f64) -> Boundedness {
        let ratio = intensity / self.ridge_point_fp32;

        if ratio < 0.9 {
            // Well below ridge point - memory bound
            let headroom = self.ridge_point_fp32 / intensity;
            Boundedness::MemoryBound { headroom }
        } else if ratio > 1.1 {
            // Well above ridge point - compute bound
            let headroom = intensity / self.ridge_point_fp32;
            Boundedness::ComputeBound { headroom }
        } else {
            // Near the ridge point - balanced
            Boundedness::Balanced
        }
    }

    /// Analyze a kernel against the roofline model
    pub fn analyze_kernel(&self, kernel: &GpuKernel, costs: &CostDatabase) -> RooflineAnalysis {
        let flops = costs.count_flops(kernel);
        let memory = costs.count_memory_bytes(kernel);

        let arithmetic_intensity = if memory.global_traffic() > 0 {
            flops.total() as f64 / memory.global_traffic() as f64
        } else {
            f64::INFINITY
        };

        // Estimate achieved performance based on operation mix
        let uses_tensor_cores = flops.tensor_flops > flops.fp32_flops;
        let peak_gflops = if uses_tensor_cores {
            self.peak_at_intensity_tensor(arithmetic_intensity)
        } else {
            self.peak_at_intensity(arithmetic_intensity)
        };

        // Assume 70% efficiency as baseline (typical for well-optimized kernels)
        let efficiency = 0.7;
        let achieved_gflops = peak_gflops * efficiency;

        let boundedness = self.classify_boundedness(arithmetic_intensity);
        let recommendations =
            self.generate_recommendations(&flops, &memory, &boundedness, uses_tensor_cores);

        RooflineAnalysis {
            kernel_name: kernel.name.clone(),
            flops,
            memory,
            arithmetic_intensity,
            achieved_gflops,
            peak_gflops,
            efficiency,
            boundedness,
            recommendations,
            uses_tensor_cores,
        }
    }

    /// Generate optimization recommendations based on roofline position
    fn generate_recommendations(
        &self,
        flops: &FlopsCount,
        memory: &MemoryTraffic,
        boundedness: &Boundedness,
        uses_tensor_cores: bool,
    ) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        match boundedness {
            Boundedness::MemoryBound { headroom } => {
                // Memory-bound kernel - focus on reducing memory traffic
                if *headroom > 2.0 {
                    let current_intensity =
                        flops.total() as f64 / memory.global_traffic().max(1) as f64;
                    hints.push(OptimizationHint::IncreaseArithmeticIntensity {
                        current: current_intensity,
                        target: self.ridge_point_fp32,
                    });
                }

                // Suggest improving memory access patterns
                hints.push(OptimizationHint::ImproveMemoryCoalescing { efficiency: 0.8 });

                // If not using tensor cores and kernel is matrix-heavy
                if !uses_tensor_cores && flops.fp32_flops > 1000 {
                    hints.push(OptimizationHint::UseTensorCores {
                        speedup_estimate: 4.0,
                    });
                }

                // Suggest async pipeline for memory-bound kernels
                hints.push(OptimizationHint::EnableAsyncPipeline);
            }

            Boundedness::ComputeBound { headroom } => {
                // Compute-bound kernel - focus on compute efficiency
                if *headroom > 1.5 {
                    hints.push(OptimizationHint::IncreaseOccupancy {
                        current: 0.5,
                        target: 0.75,
                    });
                }

                // Suggest tensor cores if not already using them
                if !uses_tensor_cores && flops.fp32_flops > flops.int_ops {
                    hints.push(OptimizationHint::UseTensorCores {
                        speedup_estimate: 8.0,
                    });
                }

                // Suggest quantization for compute-bound kernels
                if flops.fp32_flops > flops.fp16_flops {
                    hints.push(OptimizationHint::UseQuantization { precision: "INT8" });
                }
            }

            Boundedness::Balanced => {
                // Balanced kernel - both optimizations can help
                if !uses_tensor_cores {
                    hints.push(OptimizationHint::UseTensorCores {
                        speedup_estimate: 2.0,
                    });
                }
            }
        }

        // Always suggest fusion if there are multiple memory accesses
        if memory.global_loads > 100 || memory.global_stores > 100 {
            hints.push(OptimizationHint::FuseKernels {
                candidates: vec!["adjacent_kernels".to_string()],
            });
        }

        hints
    }

    /// Generate roofline plot data for visualization
    pub fn generate_roofline_data(&self, analyses: &[RooflineAnalysis]) -> RooflinePlot {
        // Generate ceiling points (log scale from 0.1 to 1000 FLOPS/byte)
        let mut ceiling = Vec::new();
        let mut intensity = 0.1;
        while intensity <= 1000.0 {
            let perf = self.peak_at_intensity(intensity);
            ceiling.push((intensity, perf));
            intensity *= 1.2;
        }

        // Generate L2 ceiling
        let mut l2_ceiling = Vec::new();
        intensity = 0.1;
        while intensity <= 1000.0 {
            let perf = self.peak_at_intensity_l2(intensity);
            l2_ceiling.push((intensity, perf));
            intensity *= 1.2;
        }

        // Generate tensor core ceiling
        let mut tensor_ceiling = Vec::new();
        intensity = 0.1;
        while intensity <= 1000.0 {
            let perf = self.peak_at_intensity_tensor(intensity);
            tensor_ceiling.push((intensity, perf));
            intensity *= 1.2;
        }

        // Extract kernel points
        let kernels: Vec<_> = analyses
            .iter()
            .map(|a| {
                (
                    a.kernel_name.clone(),
                    a.arithmetic_intensity,
                    a.achieved_gflops,
                )
            })
            .collect();

        RooflinePlot {
            ceiling,
            l2_ceiling,
            tensor_ceiling,
            kernels,
            ridge_x: self.ridge_point_fp32,
            peak_gflops: self.peak_compute_gflops,
            peak_tensor_gflops: self.peak_tensor_gflops,
        }
    }
}

// ============================================================================
// Roofline Analysis Types
// ============================================================================

/// Whether a kernel is compute or memory bound
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boundedness {
    /// Kernel is limited by compute throughput
    ComputeBound {
        /// How far above the ridge point (>1.0)
        headroom: f64,
    },
    /// Kernel is limited by memory bandwidth
    MemoryBound {
        /// How far below the ridge point (>1.0)
        headroom: f64,
    },
    /// Kernel is at the ridge point (balanced)
    Balanced,
}

impl Boundedness {
    /// Check if the kernel is memory bound
    pub fn is_memory_bound(&self) -> bool {
        matches!(self, Boundedness::MemoryBound { .. })
    }

    /// Check if the kernel is compute bound
    pub fn is_compute_bound(&self) -> bool {
        matches!(self, Boundedness::ComputeBound { .. })
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Boundedness::ComputeBound { .. } => "compute-bound",
            Boundedness::MemoryBound { .. } => "memory-bound",
            Boundedness::Balanced => "balanced",
        }
    }
}

/// Full roofline analysis for a kernel
#[derive(Debug, Clone)]
pub struct RooflineAnalysis {
    /// Kernel name
    pub kernel_name: String,
    /// FLOPS breakdown
    pub flops: FlopsCount,
    /// Memory traffic breakdown
    pub memory: MemoryTraffic,
    /// Arithmetic intensity (FLOPS/byte)
    pub arithmetic_intensity: f64,
    /// Estimated achieved performance (GFLOPS)
    pub achieved_gflops: f64,
    /// Peak achievable at this intensity (GFLOPS)
    pub peak_gflops: f64,
    /// Efficiency (achieved / peak)
    pub efficiency: f64,
    /// Whether compute or memory bound
    pub boundedness: Boundedness,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationHint>,
    /// Whether the kernel uses tensor cores
    pub uses_tensor_cores: bool,
}

impl RooflineAnalysis {
    /// Get the roofline point for this kernel
    pub fn roofline_point(&self) -> RooflinePoint {
        RooflinePoint {
            arithmetic_intensity: self.arithmetic_intensity,
            achieved_gflops: self.achieved_gflops,
            peak_gflops: self.peak_gflops,
            efficiency: self.efficiency,
        }
    }

    /// Get the percentage of peak performance achieved
    pub fn percent_of_peak(&self) -> f64 {
        self.efficiency * 100.0
    }

    /// Check if the kernel has optimization potential
    pub fn has_optimization_potential(&self) -> bool {
        self.efficiency < 0.8 || !self.recommendations.is_empty()
    }
}

/// A point on the roofline diagram
#[derive(Debug, Clone, Copy)]
pub struct RooflinePoint {
    /// Arithmetic intensity (FLOPS/byte)
    pub arithmetic_intensity: f64,
    /// Achieved performance (GFLOPS)
    pub achieved_gflops: f64,
    /// Peak achievable at this intensity (GFLOPS)
    pub peak_gflops: f64,
    /// Performance efficiency (achieved/peak)
    pub efficiency: f64,
}

/// Data for plotting a roofline diagram
#[derive(Debug, Clone)]
pub struct RooflinePlot {
    /// FP32 roofline ceiling points (intensity, GFLOPS)
    pub ceiling: Vec<(f64, f64)>,
    /// L2 cache roofline ceiling
    pub l2_ceiling: Vec<(f64, f64)>,
    /// Tensor core roofline ceiling
    pub tensor_ceiling: Vec<(f64, f64)>,
    /// Kernel data points (name, intensity, GFLOPS)
    pub kernels: Vec<(String, f64, f64)>,
    /// Ridge point x-coordinate (FP32)
    pub ridge_x: f64,
    /// Peak FP32 GFLOPS
    pub peak_gflops: f64,
    /// Peak Tensor Core GFLOPS
    pub peak_tensor_gflops: f64,
}

impl RooflinePlot {
    /// Get the number of kernels in the plot
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }

    /// Find kernels below a certain efficiency threshold
    pub fn low_efficiency_kernels(&self, threshold: f64) -> Vec<&str> {
        // This would need the efficiency data, simplified for now
        self.kernels
            .iter()
            .map(|(name, _, _)| name.as_str())
            .collect()
    }
}

// ============================================================================
// Optimization Hints
// ============================================================================

/// Optimization recommendations based on roofline analysis
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationHint {
    /// Increase arithmetic intensity to move toward compute-bound
    IncreaseArithmeticIntensity { current: f64, target: f64 },
    /// Improve memory coalescing for better bandwidth utilization
    ImproveMemoryCoalescing { efficiency: f64 },
    /// Use Tensor Cores for matrix operations
    UseTensorCores { speedup_estimate: f64 },
    /// Increase occupancy for better latency hiding
    IncreaseOccupancy { current: f64, target: f64 },
    /// Reduce shared memory usage to increase occupancy
    ReduceSharedMemory { current: u32, limit: u32 },
    /// Use quantization for reduced precision
    UseQuantization { precision: &'static str },
    /// Enable async memory pipeline
    EnableAsyncPipeline,
    /// Fuse kernels to reduce memory traffic
    FuseKernels { candidates: Vec<String> },
    /// Use vectorized loads/stores
    UseVectorizedAccess {
        current_width: u32,
        target_width: u32,
    },
    /// Reduce register pressure
    ReduceRegisterPressure { current: u32, target: u32 },
}

impl OptimizationHint {
    /// Get a human-readable description of the hint
    pub fn description(&self) -> String {
        match self {
            OptimizationHint::IncreaseArithmeticIntensity { current, target } => {
                format!(
                    "Increase arithmetic intensity from {:.1} to {:.1} FLOPS/byte",
                    current, target
                )
            }
            OptimizationHint::ImproveMemoryCoalescing { efficiency } => {
                format!(
                    "Improve memory coalescing (target {:.0}% efficiency)",
                    efficiency * 100.0
                )
            }
            OptimizationHint::UseTensorCores { speedup_estimate } => {
                format!(
                    "Use Tensor Cores for potential {:.1}x speedup",
                    speedup_estimate
                )
            }
            OptimizationHint::IncreaseOccupancy { current, target } => {
                format!(
                    "Increase occupancy from {:.0}% to {:.0}%",
                    current * 100.0,
                    target * 100.0
                )
            }
            OptimizationHint::ReduceSharedMemory { current, limit } => {
                format!("Reduce shared memory from {} to {} bytes", current, limit)
            }
            OptimizationHint::UseQuantization { precision } => {
                format!(
                    "Consider {} quantization for compute-bound sections",
                    precision
                )
            }
            OptimizationHint::EnableAsyncPipeline => {
                "Enable async memory pipeline for better latency hiding".to_string()
            }
            OptimizationHint::FuseKernels { candidates } => {
                format!("Consider fusing with: {}", candidates.join(", "))
            }
            OptimizationHint::UseVectorizedAccess {
                current_width,
                target_width,
            } => {
                format!(
                    "Use vectorized access ({}→{} bytes per thread)",
                    current_width, target_width
                )
            }
            OptimizationHint::ReduceRegisterPressure { current, target } => {
                format!(
                    "Reduce register usage from {} to {} per thread",
                    current, target
                )
            }
        }
    }

    /// Get the estimated impact of applying this optimization
    pub fn estimated_impact(&self) -> f64 {
        match self {
            OptimizationHint::UseTensorCores { speedup_estimate } => *speedup_estimate,
            OptimizationHint::IncreaseOccupancy { current, target } => target / current,
            OptimizationHint::EnableAsyncPipeline => 1.3, // ~30% improvement typical
            OptimizationHint::FuseKernels { .. } => 1.5,  // ~50% improvement typical
            OptimizationHint::UseQuantization { .. } => 2.0, // 2x for INT8
            _ => 1.2,                                     // Default ~20% improvement
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
    fn test_roofline_model_creation() {
        let model = RooflineModel::for_arch(CudaArch::Ampere);

        assert!(model.peak_compute() > 10000.0); // > 10 TFLOPS
        assert!(model.peak_bandwidth() > 1000.0); // > 1 TB/s
        assert!(model.ridge_point() > 0.0);
    }

    #[test]
    fn test_ridge_point_calculation() {
        let model = RooflineModel::for_arch(CudaArch::Hopper);

        // Ridge point should be reasonable (typically 10-100 FLOPS/byte)
        let ridge = model.ridge_point();
        assert!(ridge > 5.0);
        assert!(ridge < 200.0);
    }

    #[test]
    fn test_peak_at_intensity() {
        let model = RooflineModel::for_arch(CudaArch::Ada);

        // Low intensity should be memory-bound
        let low_intensity_perf = model.peak_at_intensity(0.1);
        assert!(low_intensity_perf < model.peak_compute());

        // High intensity should be compute-bound (at peak)
        let high_intensity_perf = model.peak_at_intensity(1000.0);
        assert!((high_intensity_perf - model.peak_compute()).abs() < 0.01);
    }

    #[test]
    fn test_boundedness_classification() {
        let model = RooflineModel::for_arch(CudaArch::Ampere);
        let ridge = model.ridge_point();

        // Well below ridge point
        let low = model.classify_boundedness(ridge * 0.1);
        assert!(low.is_memory_bound());

        // Well above ridge point
        let high = model.classify_boundedness(ridge * 10.0);
        assert!(high.is_compute_bound());

        // Near ridge point
        let balanced = model.classify_boundedness(ridge);
        assert!(matches!(balanced, Boundedness::Balanced));
    }

    #[test]
    fn test_optimization_hint_descriptions() {
        let hint = OptimizationHint::UseTensorCores {
            speedup_estimate: 4.0,
        };
        let desc = hint.description();
        assert!(desc.contains("Tensor Cores"));
        assert!(desc.contains("4.0x"));
    }

    #[test]
    fn test_roofline_plot_generation() {
        let model = RooflineModel::for_arch(CudaArch::Turing);
        let analyses = vec![];
        let plot = model.generate_roofline_data(&analyses);

        // Ceiling should have multiple points
        assert!(plot.ceiling.len() > 10);

        // Ridge point should be positive
        assert!(plot.ridge_x > 0.0);
    }

    #[test]
    fn test_different_architectures() {
        let turing = RooflineModel::for_arch(CudaArch::Turing);
        let blackwell = RooflineModel::for_arch(CudaArch::Blackwell);

        // Blackwell should have higher peak performance
        assert!(blackwell.peak_compute() > turing.peak_compute());
        assert!(blackwell.peak_bandwidth() > turing.peak_bandwidth());
    }
}
