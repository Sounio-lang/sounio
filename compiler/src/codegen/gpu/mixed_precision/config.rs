//! Mixed-precision configuration types

use std::collections::HashSet;

use super::super::ir::GpuType;

/// Low-precision float type selection for mixed-precision training
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LowPrecision {
    /// IEEE FP16 (half precision)
    FP16,
    /// Brain Float 16 (requires sm_80+)
    BF16,
    /// FP8 E4M3 format (requires sm_89+)
    FP8E4M3,
}

impl LowPrecision {
    /// Convert to GPU type
    pub fn to_gpu_type(&self) -> GpuType {
        match self {
            LowPrecision::FP16 => GpuType::F16,
            LowPrecision::BF16 => GpuType::BF16,
            LowPrecision::FP8E4M3 => GpuType::F8E4M3,
        }
    }
}

/// Configuration for mixed-precision training
#[derive(Clone, Debug)]
pub struct MixedPrecisionConfig {
    /// Forward pass precision (FP16 or BF16)
    pub forward_precision: LowPrecision,
    /// Backward pass precision (always FP32 for gradients)
    pub backward_precision: GpuType,
    /// Master weight precision (FP32 for optimizer updates)
    pub master_weight_precision: GpuType,
    /// Initial loss scale (typically 2^15 = 32768)
    pub initial_loss_scale: f32,
    /// Loss scale growth factor when no overflow
    pub scale_growth_factor: f32,
    /// Number of steps without overflow before scaling up
    pub scale_growth_interval: u32,
    /// Enable dynamic loss scaling vs static
    pub dynamic_loss_scaling: bool,
    /// Operations that must stay in FP32 (e.g., softmax, layer_norm, exp, log)
    pub fp32_operations: HashSet<String>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            forward_precision: LowPrecision::FP16,
            backward_precision: GpuType::F32,
            master_weight_precision: GpuType::F32,
            initial_loss_scale: 32768.0,
            scale_growth_factor: 2.0,
            scale_growth_interval: 2000,
            dynamic_loss_scaling: true,
            fp32_operations: Self::default_fp32_ops(),
        }
    }
}

impl MixedPrecisionConfig {
    /// Create default FP16 configuration
    pub fn default_fp16() -> Self {
        Self {
            forward_precision: LowPrecision::FP16,
            ..Self::default()
        }
    }

    /// Create BF16 configuration (for Ampere+ GPUs)
    pub fn default_bf16() -> Self {
        Self {
            forward_precision: LowPrecision::BF16,
            ..Self::default()
        }
    }

    /// Operations that are numerically sensitive and need FP32
    fn default_fp32_ops() -> HashSet<String> {
        let ops = [
            "exp",        // exp() overflows easily in FP16
            "log",        // log() near zero needs precision
            "sqrt",       // sqrt() backward needs precision
            "sum",        // Reductions accumulate error
            "mean",       // Reductions accumulate error
            "softmax",    // exp + sum + div all sensitive
            "layer_norm", // variance computation sensitive
            "batch_norm", // variance computation sensitive
        ];
        ops.iter().map(|s| s.to_string()).collect()
    }

    /// Check if an operation should use FP32
    pub fn requires_fp32(&self, op_name: &str) -> bool {
        self.fp32_operations.contains(op_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.forward_precision, LowPrecision::FP16);
        assert_eq!(config.backward_precision, GpuType::F32);
        assert_eq!(config.initial_loss_scale, 32768.0);
        assert!(config.dynamic_loss_scaling);
    }

    #[test]
    fn test_fp32_ops() {
        let config = MixedPrecisionConfig::default();
        assert!(config.requires_fp32("exp"));
        assert!(config.requires_fp32("softmax"));
        assert!(!config.requires_fp32("relu"));
        assert!(!config.requires_fp32("matmul"));
    }
}
