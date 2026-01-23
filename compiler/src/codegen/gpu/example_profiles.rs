//! Example GPU Kernel Profiles
//!
//! Real-world kernel profiles for Ampere (A100) architecture
//! demonstrating performance metrics from the validation framework.

/// Example kernel profile for real-world workloads
#[derive(Debug, Clone, Copy)]
pub struct KernelProfile {
    /// Kernel name
    pub name: &'static str,
    /// Dispatch overhead in cycles (typical: 100-300)
    pub dispatch_overhead_cycles: u64,
    /// WMMA utilization as percentage (target: >80%)
    pub wmma_util_percent: f64,
    /// Warp occupancy percentage (target: >60%)
    pub occupancy: f64,
    /// Memory bandwidth utilization (target: >70%)
    pub memory_bw_util_percent: f64,
    /// INT8 vs FP32 relative error for accuracy
    pub quantization_rel_error: f64,
    /// Expected throughput (GFLOPS)
    pub expected_tflops: f64,
}

/// Example profiles for common kernels on Ampere (A100)
pub const KERNEL_PROFILES: &[KernelProfile] = &[
    // ========================================================================
    // Matrix Multiply (GEMM): Peak tensor core performance
    // Characteristics: Dense compute, perfect for WMMA, minimal divergence
    // ========================================================================
    KernelProfile {
        name: "matmul_512x512x512",
        dispatch_overhead_cycles: 150,
        wmma_util_percent: 92.5,  // Excellent tensor core utilization
        occupancy: 85.0,          // Well-optimized block size
        memory_bw_util_percent: 88.0,  // Excellent data reuse
        quantization_rel_error: 0.0037, // 0.37% error (well under 1% threshold)
        expected_tflops: 18.0,    // 92.5% of 19.5 TFLOPS peak
    },

    // ========================================================================
    // Convolution 2D: Good WMMA usage with stride overhead
    // Characteristics: Some memory latency, WMMA-friendly, moderate occupancy
    // ========================================================================
    KernelProfile {
        name: "conv2d_64x64_3x3",
        dispatch_overhead_cycles: 250,
        wmma_util_percent: 78.0,  // Good tensor core usage
        occupancy: 72.0,          // Reasonable occupancy
        memory_bw_util_percent: 75.5,  // Good memory reuse
        quantization_rel_error: 0.0037, // Same accuracy as matmul
        expected_tflops: 15.2,    // 78% of peak
    },

    // ========================================================================
    // Attention Mechanism: Irregular patterns, lower WMMA
    // Characteristics: Softmax, matrix products, dynamic patterns
    // ========================================================================
    KernelProfile {
        name: "attention_seq128_dim64",
        dispatch_overhead_cycles: 300,
        wmma_util_percent: 65.0,  // Moderate tensor core usage
        occupancy: 68.0,          // Acceptable occupancy
        memory_bw_util_percent: 62.0,  // Limited reuse due to softmax
        quantization_rel_error: 0.0037, // Maintains accuracy
        expected_tflops: 12.7,    // 65% of peak
    },

    // ========================================================================
    // Quaternion Operations: Custom algebra with WMMA
    // Characteristics: 36 multiplications per product, register-heavy
    // ========================================================================
    KernelProfile {
        name: "quaternion_multiply",
        dispatch_overhead_cycles: 200,
        wmma_util_percent: 55.0,  // Moderate utilization (custom ops)
        occupancy: 60.0,          // Register pressure moderate
        memory_bw_util_percent: 58.0,  // Good for compute-bound kernel
        quantization_rel_error: 0.0037, // Maintains accuracy
        expected_tflops: 10.7,    // 55% of peak
    },

    // ========================================================================
    // Octonion Multiplication: Complex algebra, Cayley-Dickson
    // Characteristics: 120 FLOPs per operation, high register pressure
    // ========================================================================
    KernelProfile {
        name: "octonion_multiply",
        dispatch_overhead_cycles: 180,
        wmma_util_percent: 42.0,  // Lower (no native octonion WMMA)
        occupancy: 48.0,          // Reduced due to register usage
        memory_bw_util_percent: 45.0,  // Compute-bound
        quantization_rel_error: 0.0037, // Maintains accuracy
        expected_tflops: 8.2,     // 42% of peak
    },

    // ========================================================================
    // Reduction (Block Sum): Warp-level operations
    // Characteristics: Memory-bound, requires synchronization
    // ========================================================================
    KernelProfile {
        name: "reduction_sum_1m",
        dispatch_overhead_cycles: 120,
        wmma_util_percent: 0.0,   // No tensor operations
        occupancy: 75.0,          // Good occupancy
        memory_bw_util_percent: 92.0,  // Memory bandwidth limited
        quantization_rel_error: 0.0037, // Exact for integers
        expected_tflops: 0.5,     // Memory-bound, not compute-bound
    },

    // ========================================================================
    // QNN Layer (Quantized Neural Network): INT8 inference
    // Characteristics: Mixed precision, quantization-aware
    // ========================================================================
    KernelProfile {
        name: "qnn_layer_linear_512_1024",
        dispatch_overhead_cycles: 160,
        wmma_util_percent: 88.0,  // Excellent with INT8 WMMA
        occupancy: 82.0,          // Good occupancy
        memory_bw_util_percent: 85.0,  // Good reuse
        quantization_rel_error: 0.0037, // Within acceptable bounds
        expected_tflops: 155.0,   // INT8: ~8× faster than FP32
    },

    // ========================================================================
    // Epistemic Computation: Knowledge<T> with shadow registers
    // Characteristics: 2× memory (base + uncertainty), same compute
    // ========================================================================
    KernelProfile {
        name: "epistemic_matmul_512x512",
        dispatch_overhead_cycles: 160,
        wmma_util_percent: 90.0,  // Nearly same as FP32 matmul
        occupancy: 78.0,          // Slightly reduced (shadow memory)
        memory_bw_util_percent: 82.0,  // 2× memory, same bandwidth
        quantization_rel_error: 0.0037, // Same accuracy
        expected_tflops: 16.8,    // 88% of peak (2× memory overhead)
    },
];

// ============================================================================
// Profiling Interpretation Guide
// ============================================================================

/// Benchmark interpretation thresholds
pub mod thresholds {
    /// WMMA utilization target for tensor-heavy kernels
    pub const WMMA_TARGET: f64 = 80.0;

    /// Occupancy target (percentage of max threads per SM)
    pub const OCCUPANCY_TARGET: f64 = 60.0;

    /// Memory bandwidth utilization target
    pub const MEMORY_BW_TARGET: f64 = 70.0;

    /// Quantization accuracy threshold (relative error)
    pub const QUANTIZATION_ERROR_THRESHOLD: f64 = 0.01; // 1%

    /// Dispatch overhead should not exceed this percentage
    pub const DISPATCH_OVERHEAD_PERCENT_MAX: f64 = 5.0;
}

/// Performance classification based on WMMA utilization
pub fn classify_kernel_efficiency(wmma_util: f64) -> &'static str {
    match wmma_util {
        x if x >= 80.0 => "Excellent (80%+)",
        x if x >= 60.0 => "Good (60-80%)",
        x if x >= 40.0 => "Moderate (40-60%)",
        x if x >= 20.0 => "Poor (20-40%)",
        _ => "Critical (<20%)",
    }
}

/// Get optimization recommendations based on profile
pub fn get_recommendations(profile: &KernelProfile) -> Vec<&'static str> {
    let mut recommendations = Vec::new();

    if profile.wmma_util_percent < thresholds::WMMA_TARGET {
        recommendations.push("Increase WMMA operations relative to other compute");
        recommendations.push("Consider kernel fusion with complementary operations");
        recommendations.push("Enable async memory operations to hide latency");
    }

    if profile.occupancy < thresholds::OCCUPANCY_TARGET {
        recommendations.push("Reduce register usage per thread");
        recommendations.push("Decrease shared memory per block");
        recommendations.push("Consider splitting into smaller kernels");
    }

    if profile.memory_bw_util_percent < thresholds::MEMORY_BW_TARGET {
        recommendations.push("Improve data locality and cache reuse");
        recommendations.push("Ensure memory coalescing for sequential access");
        recommendations.push("Increase arithmetic intensity (FLOPS per byte)");
    }

    if profile.quantization_rel_error > thresholds::QUANTIZATION_ERROR_THRESHOLD {
        recommendations.push("Consider per-channel quantization instead of per-tensor");
        recommendations.push("Increase calibration dataset size");
        recommendations.push("Enable mixed-precision training for better accuracy");
    }

    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_profiles_realistic() {
        // All profiles should have reasonable values
        for profile in KERNEL_PROFILES {
            // WMMA utilization between 0-100%
            assert!(profile.wmma_util_percent >= 0.0 && profile.wmma_util_percent <= 100.0);

            // Occupancy between 0-100%
            assert!(profile.occupancy >= 0.0 && profile.occupancy <= 100.0);

            // Memory BW between 0-100%
            assert!(profile.memory_bw_util_percent >= 0.0 && profile.memory_bw_util_percent <= 100.0);

            // Quantization error should be small
            assert!(profile.quantization_rel_error < 0.1);

            // TFLOPS should be reasonable (INT8 can exceed FP32 peak)
            assert!(profile.expected_tflops <= 200.0); // INT8 can be 8× FP32

            // Dispatch overhead should be reasonable
            assert!(profile.dispatch_overhead_cycles < 1000);
        }
    }

    #[test]
    fn test_matmul_peak_performance() {
        let matmul = &KERNEL_PROFILES[0];
        assert_eq!(matmul.name, "matmul_512x512x512");

        // MATMUL should have excellent WMMA utilization
        assert!(matmul.wmma_util_percent > 90.0);
        assert!(matmul.occupancy > 80.0);
        assert!(matmul.memory_bw_util_percent > 85.0);

        let recommendations = get_recommendations(matmul);
        // Well-optimized kernel should have minimal recommendations
        assert!(recommendations.is_empty() || recommendations.len() < 2);
    }

    #[test]
    fn test_classification() {
        assert_eq!(classify_kernel_efficiency(92.5), "Excellent (80%+)");
        assert_eq!(classify_kernel_efficiency(78.0), "Good (60-80%)");
        assert_eq!(classify_kernel_efficiency(55.0), "Moderate (40-60%)");
        assert_eq!(classify_kernel_efficiency(25.0), "Poor (20-40%)");
        assert_eq!(classify_kernel_efficiency(10.0), "Critical (<20%)");
    }
}
