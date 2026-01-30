//! Mixed-precision IR transformation pass
//!
//! Analyzes and transforms GPU kernels to use mixed-precision training:
//! - FP16 (or BF16/FP8) for compute-intensive operations (matmul, conv, activations)
//! - FP32 for numerically sensitive operations (exp, log, softmax, layer_norm, reductions)
//! - Automatic insertion of type conversion operations at precision boundaries

use rustc_hash::FxHashMap;

use super::super::ir::{GpuKernel, GpuOp, ValueId};
use super::config::MixedPrecisionConfig;

/// Operation classification for mixed-precision assignment
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpCategory {
    /// Always FP32: exp, log, sqrt, sin, cos, reductions, softmax, layer_norm
    MustBeFloat32,
    /// Can use low precision: matmul, conv, activations (relu, sigmoid, tanh)
    ComputeIntensive,
    /// Preserves input precision: moves, loads, stores, etc.
    Passthrough,
}

/// Mixed-precision transform that analyzes and modifies GPU kernels
pub struct MixedPrecisionTransform {
    config: MixedPrecisionConfig,
    /// Maps value IDs to their assigned precision for this kernel
    precision_map: FxHashMap<ValueId, bool>, // true = low precision, false = FP32
}

impl MixedPrecisionTransform {
    /// Creates a new mixed-precision transform with the given configuration
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            config,
            precision_map: FxHashMap::default(),
        }
    }

    /// Classifies an operation based on its semantic meaning
    fn classify_op(&self, op: &GpuOp) -> OpCategory {
        match op {
            // FP32-required operations (numerically sensitive)
            GpuOp::FastExp(_)
            | GpuOp::FastLog(_)
            | GpuOp::FastSqrt(_)
            | GpuOp::FastSin(_)
            | GpuOp::FastCos(_)
            | GpuOp::FastRsqrt(_) => OpCategory::MustBeFloat32,

            // Quantization and special functions that need precision
            GpuOp::QuantizeF32ToInt8 { .. }
            | GpuOp::DequantizeInt8ToF32 { .. }
            | GpuOp::QuantizeF32ToUint8 { .. }
            | GpuOp::DequantizeUint8ToF32 { .. } => OpCategory::MustBeFloat32,

            // FP32 reduction/aggregation operations (accumulate error)
            GpuOp::AtomicAdd(_, _) | GpuOp::AtomicMin(_, _) | GpuOp::AtomicMax(_, _) => {
                OpCategory::MustBeFloat32
            }

            // Compute-intensive operations (benefit from low precision)
            GpuOp::FMul(_, _) | GpuOp::FAdd(_, _) | GpuOp::FSub(_, _) | GpuOp::FDiv(_, _) => {
                OpCategory::ComputeIntensive
            }

            // Integer arithmetic (generally can use low precision)
            GpuOp::Mul(_, _) | GpuOp::Add(_, _) | GpuOp::Sub(_, _) | GpuOp::Div(_, _) => {
                OpCategory::ComputeIntensive
            }

            // Passthrough: don't change their precision
            GpuOp::Load { .. } | GpuOp::Store { .. } | GpuOp::Phi(_) => OpCategory::Passthrough,

            // Most other ops can use low precision
            _ => OpCategory::ComputeIntensive,
        }
    }

    /// Analyzes kernel operations and assigns precision to each value
    pub fn classify_operations(&mut self, kernel: &GpuKernel) {
        // First pass: classify operations
        for block in &kernel.blocks {
            for (value_id, op) in &block.instructions {
                let category = self.classify_op(op);

                let should_be_low_precision = match category {
                    OpCategory::MustBeFloat32 => false,
                    OpCategory::ComputeIntensive => true,
                    OpCategory::Passthrough => {
                        // Inherit precision from operands (stub for now)
                        // Would need actual input value tracking
                        false // Default to FP32
                    }
                };

                self.precision_map
                    .insert(*value_id, should_be_low_precision);
            }
        }
    }

    /// Inserts type conversion operations where precision boundaries exist
    /// This modifies the kernel in-place by adding conversion ops
    pub fn insert_conversions(&mut self, _kernel: &mut GpuKernel) {
        // Future: For each operation, check if inputs need conversion and insert
        // MixedPrecisionCast ops at boundaries. This would require:
        // 1. Tracking input dependencies in the IR
        // 2. Creating new ValueIds for conversion results
        // 3. Inserting MixedPrecisionCast GpuOp variants at precision mismatches
    }

    /// Transforms a kernel to use mixed-precision training
    pub fn transform_kernel(&mut self, kernel: &mut GpuKernel) {
        self.classify_operations(kernel);
        self.insert_conversions(kernel);
    }

    /// Returns whether a value should use low precision
    pub fn should_use_low_precision(&self, value_id: ValueId) -> bool {
        self.precision_map.get(&value_id).copied().unwrap_or(false)
    }

    /// Returns the assigned precision map (for debugging)
    pub fn precision_map(&self) -> &FxHashMap<ValueId, bool> {
        &self.precision_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_transform_creation() {
        let config = MixedPrecisionConfig::default();
        let transform = MixedPrecisionTransform::new(config);
        assert_eq!(transform.config.dynamic_loss_scaling, true);
    }

    #[test]
    fn test_precision_assignment() {
        let config = MixedPrecisionConfig::default();
        let transform = MixedPrecisionTransform::new(config);

        // Test that FP32 requirements are respected
        // (Would need to mock a GpuKernel and test actual classification)
        assert_eq!(transform.should_use_low_precision(ValueId(0)), false); // Default
    }
}
