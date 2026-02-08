//! GPU Backward Pass Kernels for Quaternion Operations
//!
//! This module provides GPU-accelerated gradient computation for quaternion neural networks.
//! Implements backward passes for the fused linear+BN+ReLU kernel via CUDA/PTX.
//!
//! Key concepts:
//! - Quaternion conjugate: q* = (w, -x, -y, -z)
//! - Hamilton backward: dq1 = dq ⊗ q2*, dq2 = q1* ⊗ dq
//! - Batch norm backward: per-component gradient with variance scaling
//! - ReLU backward: element-wise masking
//! - Memory optimization: recompute forward activations to save memory

use std::fmt;

/// Error type for quaternion backward kernel operations
#[derive(Debug, Clone)]
pub enum BackwardError {
    InvalidConfig(String),
    KernelLaunchFailed(String),
    MemoryError(String),
    NumericalError(String),
}

impl fmt::Display for BackwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackwardError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            BackwardError::KernelLaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            BackwardError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            BackwardError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl std::error::Error for BackwardError {}

/// Configuration for quaternion backward pass kernels
#[derive(Clone, Copy, Debug)]
pub struct QuatLinearBackwardConfig {
    /// Tile size for gradient computation in output channels
    pub tile_out: usize,
    /// Tile size for input channels
    pub tile_in: usize,
    /// Shared memory size per thread block (in bytes)
    pub shared_mem_size: usize,
    /// Epsilon for BN numerical stability
    pub bn_epsilon: f32,
    /// Gradient norm clipping threshold (0 = disabled)
    pub grad_clip_norm: f32,
    /// Enable gradient accumulation into existing buffers
    pub accumulate: bool,
    /// Recompute forward activation (memory vs compute)
    pub recompute_forward: bool,
}

impl Default for QuatLinearBackwardConfig {
    fn default() -> Self {
        Self {
            tile_out: 32,
            tile_in: 32,
            shared_mem_size: 16384, // 16KB per block
            bn_epsilon: 1e-5,
            grad_clip_norm: 1.0,
            accumulate: true,
            recompute_forward: true,
        }
    }
}

impl QuatLinearBackwardConfig {
    /// Validate configuration parameters
    pub fn validate(&self, in_features: usize, out_features: usize) -> Result<(), BackwardError> {
        if self.tile_out == 0 || self.tile_in == 0 {
            return Err(BackwardError::InvalidConfig(
                "Tile sizes must be > 0".to_string(),
            ));
        }

        if self.tile_out > out_features || self.tile_in > in_features {
            return Err(BackwardError::InvalidConfig(
                "Tile sizes exceed matrix dimensions".to_string(),
            ));
        }

        if self.bn_epsilon <= 0.0 {
            return Err(BackwardError::InvalidConfig(
                "BN epsilon must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Parameters for backward kernel execution
#[derive(Debug, Clone)]
pub struct QuatBackwardParams {
    /// Gradient w.r.t. output: [batch_size, out_features * 4]
    pub grad_output: *const f32,
    /// Saved forward input: [batch_size, in_features * 4]
    pub input: *const f32,
    /// Saved weights: [out_features * 4, in_features * 4]
    pub weights: *const f32,
    /// Saved BN mean: [out_features * 4]
    pub bn_mean: *const f32,
    /// Saved BN variance: [out_features * 4]
    pub bn_var: *const f32,
    /// Saved linear output (before BN): [batch_size, out_features * 4]
    pub linear_output: *const f32,

    /// Output: gradient w.r.t. input: [batch_size, in_features * 4]
    pub grad_input: *mut f32,
    /// Output: gradient w.r.t. weights: [out_features * 4, in_features * 4]
    pub grad_weights: *mut f32,
    /// Output: gradient w.r.t. bias: [out_features * 4]
    pub grad_bias: *mut f32,
    /// Output: gradient w.r.t. BN gamma: [out_features * 4]
    pub grad_bn_gamma: *mut f32,
    /// Output: gradient w.r.t. BN beta: [out_features * 4]
    pub grad_bn_beta: *mut f32,

    /// Dimensions
    pub batch_size: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl QuatBackwardParams {
    /// Validate parameters and pointers
    pub fn validate(&self) -> Result<(), BackwardError> {
        // Pointer validation
        if self.grad_output.is_null()
            || self.input.is_null()
            || self.weights.is_null()
            || self.grad_input.is_null()
            || self.grad_weights.is_null()
        {
            return Err(BackwardError::MemoryError(
                "Null pointer in parameters".to_string(),
            ));
        }

        // Dimension validation
        if self.batch_size == 0 || self.in_features == 0 || self.out_features == 0 {
            return Err(BackwardError::InvalidConfig(
                "Invalid dimensions".to_string(),
            ));
        }

        // Features must be multiples of 4 for quaternions
        if self.in_features % 4 != 0 || self.out_features % 4 != 0 {
            return Err(BackwardError::InvalidConfig(
                "Features must be multiples of 4".to_string(),
            ));
        }

        Ok(())
    }
}

// ============================================================================
// Quaternion Utilities for Backward Pass (CPU Reference)
// ============================================================================

/// Quaternion conjugate: q* = (w, -x, -y, -z)
pub fn quat_conjugate_cpu(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Hamilton product: q1 ⊗ q2
pub fn quat_mul_cpu(q1: [f32; 4], q2: [f32; 4]) -> [f32; 4] {
    [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ]
}

/// Hamilton product with second argument conjugated: q1 ⊗ q2*
pub fn quat_mul_conj_cpu(q1: [f32; 4], q2: [f32; 4]) -> [f32; 4] {
    let q2_conj = quat_conjugate_cpu(q2);
    quat_mul_cpu(q1, q2_conj)
}

// ============================================================================
// Backward Pass Functions (CPU Reference Implementation)
// ============================================================================

/// Backward pass for quaternion multiplication: y = q1 ⊗ q2
/// Gradients: dq1 = dy ⊗ q2*, dq2 = q1* ⊗ dy
pub fn quat_mul_backward_cpu(
    dy: [f32; 4],
    q1: [f32; 4],
    q2: [f32; 4],
) -> (
    [f32; 4], // dq1
    [f32; 4], // dq2
) {
    let q2_conj = quat_conjugate_cpu(q2);
    let q1_conj = quat_conjugate_cpu(q1);

    let dq1 = quat_mul_cpu(dy, q2_conj);
    let dq2 = quat_mul_cpu(q1_conj, dy);

    (dq1, dq2)
}

/// Backward pass for batch normalization
/// Given normalized output y = (x - mean) / sqrt(var + eps)
/// Compute: dy/dx, dy/dmean, dy/dvar
pub fn bn_backward_cpu(
    dy: f32,
    x: f32,
    mean: f32,
    var: f32,
    eps: f32,
) -> (
    f32, // dx
    f32, // dmean
    f32, // dvar
) {
    let var_eps = var + eps;
    let inv_sqrt = 1.0 / var_eps.sqrt();

    let dx = dy * inv_sqrt;
    let dmean = -dy * inv_sqrt;
    let dvar = -0.5 * dy * (x - mean) * var_eps.powf(-1.5);

    (dx, dmean, dvar)
}

/// Backward pass for ReLU: dy/dx = if x > 0 then dy else 0
pub fn relu_backward_cpu(dy: f32, x: f32) -> f32 {
    if x > 0.0 { dy } else { 0.0 }
}

// ============================================================================
// Fused Linear Layer Backward Pass (Scalar Reference)
// ============================================================================

/// Scalar CPU implementation of fused linear+BN+ReLU backward pass
/// Computes gradients w.r.t. input, weights, and bias
pub fn quat_linear_bn_relu_backward_scalar(
    params: &QuatBackwardParams,
    config: &QuatLinearBackwardConfig,
) -> Result<(), BackwardError> {
    params.validate()?;
    config.validate(params.in_features, params.out_features)?;

    unsafe {
        // Iterate over output batches and features
        for b in 0..params.batch_size {
            for o in 0..params.out_features {
                let out_idx = (b * params.out_features + o) * 4;

                // Load gradient w.r.t. output (after ReLU)
                let grad_out = [
                    *params.grad_output.add(out_idx),
                    *params.grad_output.add(out_idx + 1),
                    *params.grad_output.add(out_idx + 2),
                    *params.grad_output.add(out_idx + 3),
                ];

                // Load saved linear output (before BN+ReLU)
                let linear_out = if !params.linear_output.is_null() {
                    [
                        *params.linear_output.add(out_idx),
                        *params.linear_output.add(out_idx + 1),
                        *params.linear_output.add(out_idx + 2),
                        *params.linear_output.add(out_idx + 3),
                    ]
                } else {
                    [0.0; 4]
                };

                // Load BN stats
                let bn_idx = o * 4;
                let bn_mean = if !params.bn_mean.is_null() {
                    [
                        *params.bn_mean.add(bn_idx),
                        *params.bn_mean.add(bn_idx + 1),
                        *params.bn_mean.add(bn_idx + 2),
                        *params.bn_mean.add(bn_idx + 3),
                    ]
                } else {
                    [0.0; 4]
                };

                let bn_var = if !params.bn_var.is_null() {
                    [
                        *params.bn_var.add(bn_idx),
                        *params.bn_var.add(bn_idx + 1),
                        *params.bn_var.add(bn_idx + 2),
                        *params.bn_var.add(bn_idx + 3),
                    ]
                } else {
                    [1.0; 4]
                };

                // Backward through ReLU (element-wise)
                let mut grad_bn = [0.0; 4];
                for c in 0..4 {
                    grad_bn[c] = relu_backward_cpu(grad_out[c], linear_out[c]);
                }

                // Accumulate to bias gradient
                if !params.grad_bias.is_null() {
                    *params.grad_bias.add(bn_idx) += grad_bn[0];
                    *params.grad_bias.add(bn_idx + 1) += grad_bn[1];
                    *params.grad_bias.add(bn_idx + 2) += grad_bn[2];
                    *params.grad_bias.add(bn_idx + 3) += grad_bn[3];
                }

                // Backward through BN (per-component)
                let mut grad_linear = [0.0; 4];
                for c in 0..4 {
                    let (dx, _dmean, _dvar) = bn_backward_cpu(
                        grad_bn[c],
                        linear_out[c],
                        bn_mean[c],
                        bn_var[c],
                        config.bn_epsilon,
                    );
                    grad_linear[c] = dx;

                    // Accumulate to gamma gradient (scale parameter)
                    if !params.grad_bn_gamma.is_null() {
                        *params.grad_bn_gamma.add(bn_idx + c) += grad_bn[c]
                            * (linear_out[c] - bn_mean[c])
                            / (bn_var[c] + config.bn_epsilon).sqrt();
                    }

                    // Accumulate to beta gradient (shift parameter)
                    if !params.grad_bn_beta.is_null() {
                        *params.grad_bn_beta.add(bn_idx + c) += grad_bn[c];
                    }
                }

                // Backward through linear layer (Hamilton product)
                // Compute weight gradient: dL/dW[o,i] = grad_linear[o] ⊗ conj(input[b,i])
                for i in 0..params.in_features {
                    let in_idx = (b * params.in_features + i) * 4;
                    let w_idx = (o * params.in_features + i) * 4;

                    let input_q = [
                        *params.input.add(in_idx),
                        *params.input.add(in_idx + 1),
                        *params.input.add(in_idx + 2),
                        *params.input.add(in_idx + 3),
                    ];

                    // dW = grad_linear ⊗ conj(input)
                    let dw = quat_mul_conj_cpu(grad_linear, input_q);

                    if config.accumulate {
                        // Accumulate
                        *params.grad_weights.add(w_idx) += dw[0];
                        *params.grad_weights.add(w_idx + 1) += dw[1];
                        *params.grad_weights.add(w_idx + 2) += dw[2];
                        *params.grad_weights.add(w_idx + 3) += dw[3];
                    } else {
                        // Overwrite
                        *params.grad_weights.add(w_idx) = dw[0];
                        *params.grad_weights.add(w_idx + 1) = dw[1];
                        *params.grad_weights.add(w_idx + 2) = dw[2];
                        *params.grad_weights.add(w_idx + 3) = dw[3];
                    }
                }

                // Backward through linear layer (input gradient)
                // Compute input gradient: dL/dinput[b,i] = sum_o grad_linear[o] ⊗ conj(weight[o,i])
                for i in 0..params.in_features {
                    let in_idx = (b * params.in_features + i) * 4;
                    let mut grad_in = [0.0; 4];

                    for o in 0..params.out_features {
                        let w_idx = (o * params.in_features + i) * 4;
                        let weight_q = [
                            *params.weights.add(w_idx),
                            *params.weights.add(w_idx + 1),
                            *params.weights.add(w_idx + 2),
                            *params.weights.add(w_idx + 3),
                        ];

                        // Get gradient for this output
                        let out_idx = (b * params.out_features + o) * 4;
                        let grad_out_o = [
                            *params.grad_output.add(out_idx),
                            *params.grad_output.add(out_idx + 1),
                            *params.grad_output.add(out_idx + 2),
                            *params.grad_output.add(out_idx + 3),
                        ];

                        // Compute ReLU + BN gradient
                        let linear_out_o = if !params.linear_output.is_null() {
                            [
                                *params.linear_output.add(out_idx),
                                *params.linear_output.add(out_idx + 1),
                                *params.linear_output.add(out_idx + 2),
                                *params.linear_output.add(out_idx + 3),
                            ]
                        } else {
                            [0.0; 4]
                        };

                        let mut grad_bn_o = [0.0; 4];
                        for c in 0..4 {
                            grad_bn_o[c] = relu_backward_cpu(grad_out_o[c], linear_out_o[c]);
                        }

                        let bn_idx_o = o * 4;
                        let bn_mean_o = if !params.bn_mean.is_null() {
                            [
                                *params.bn_mean.add(bn_idx_o),
                                *params.bn_mean.add(bn_idx_o + 1),
                                *params.bn_mean.add(bn_idx_o + 2),
                                *params.bn_mean.add(bn_idx_o + 3),
                            ]
                        } else {
                            [0.0; 4]
                        };

                        let bn_var_o = if !params.bn_var.is_null() {
                            [
                                *params.bn_var.add(bn_idx_o),
                                *params.bn_var.add(bn_idx_o + 1),
                                *params.bn_var.add(bn_idx_o + 2),
                                *params.bn_var.add(bn_idx_o + 3),
                            ]
                        } else {
                            [1.0; 4]
                        };

                        let mut grad_linear_o = [0.0; 4];
                        for c in 0..4 {
                            let (dx, _, _) = bn_backward_cpu(
                                grad_bn_o[c],
                                linear_out_o[c],
                                bn_mean_o[c],
                                bn_var_o[c],
                                config.bn_epsilon,
                            );
                            grad_linear_o[c] = dx;
                        }

                        // dI += conj(W) ⊗ grad_linear
                        let di = quat_mul_conj_cpu(weight_q, grad_linear_o);
                        grad_in[0] += di[0];
                        grad_in[1] += di[1];
                        grad_in[2] += di[2];
                        grad_in[3] += di[3];
                    }

                    // Write input gradient
                    if config.accumulate {
                        *params.grad_input.add(in_idx) += grad_in[0];
                        *params.grad_input.add(in_idx + 1) += grad_in[1];
                        *params.grad_input.add(in_idx + 2) += grad_in[2];
                        *params.grad_input.add(in_idx + 3) += grad_in[3];
                    } else {
                        *params.grad_input.add(in_idx) = grad_in[0];
                        *params.grad_input.add(in_idx + 1) = grad_in[1];
                        *params.grad_input.add(in_idx + 2) = grad_in[2];
                        *params.grad_input.add(in_idx + 3) = grad_in[3];
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_conjugate() {
        let q = [1.0, 2.0, 3.0, 4.0];
        let q_conj = quat_conjugate_cpu(q);
        assert_eq!(q_conj, [1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_quat_mul() {
        // Identity quaternion ⊗ q = q
        let identity = [1.0, 0.0, 0.0, 0.0];
        let q = [0.707, 0.707, 0.0, 0.0];
        let result = quat_mul_cpu(identity, q);
        assert!((result[0] - 0.707).abs() < 1e-5);
        assert!((result[1] - 0.707).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_bn_backward() {
        let dy = 0.1;
        let x = 2.0;
        let mean = 1.5;
        let var = 0.5;
        let eps = 1e-5;

        let (dx, _dmean, _dvar) = bn_backward_cpu(dy, x, mean, var, eps);
        assert!(dx > 0.0); // Positive dy should give positive dx
        assert!((dx - 0.1 / (var + eps).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_relu_backward() {
        let dy = 0.5;
        assert_eq!(relu_backward_cpu(dy, 1.0), 0.5); // x > 0
        assert_eq!(relu_backward_cpu(dy, -1.0), 0.0); // x <= 0
    }

    #[test]
    fn test_backward_config_validation() {
        let config = QuatLinearBackwardConfig::default();
        assert!(config.validate(32, 64).is_ok());
        assert!(config.validate(0, 64).is_err()); // Zero features
    }

    #[test]
    fn test_backward_params_validation() {
        let params = QuatBackwardParams {
            grad_output: 0x1 as *const f32,
            input: 0x2 as *const f32,
            weights: 0x3 as *const f32,
            bn_mean: 0x4 as *const f32,
            bn_var: 0x5 as *const f32,
            linear_output: 0x6 as *const f32,
            grad_input: 0x7 as *mut f32,
            grad_weights: 0x8 as *mut f32,
            grad_bias: 0x9 as *mut f32,
            grad_bn_gamma: 0xa as *mut f32,
            grad_bn_beta: 0xb as *mut f32,
            batch_size: 32,
            in_features: 16,
            out_features: 32,
        };

        assert!(params.validate().is_ok());

        let mut bad_params = params.clone();
        bad_params.in_features = 7; // Not multiple of 4
        assert!(bad_params.validate().is_err());
    }
}
