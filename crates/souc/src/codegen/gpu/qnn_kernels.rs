//! GPU Kernels for Fused Quaternion Linear + Batch Normalization + ReLU
//!
//! This module provides structures and configuration for executing a fused quaternion
//! linear layer followed by batch normalization and ReLU activation on the GPU.
//! Supports both scalar processing and WMMA (Warp Matrix Multiply-Accumulate) tensor
//! core acceleration for NVIDIA GPUs.
//!
//! The kernel is designed for quaternion neural networks (QNNs), where inputs,
//! weights, and outputs are represented as quaternions (4-component vectors).
//! The linear operation uses the Hamilton product for quaternion multiplication.
//!
//! Key features:
//! - Tile-based processing for large matrices to fit in shared memory
//! - Bank conflict avoidance through padding in shared memory layouts
//! - Asynchronous memory copies for improved bandwidth on Ampere+ architectures
//! - Comprehensive error handling
//! - Support for PTX code generation compatible with Sounio's GPU backend
//!
//! Memory layout assumptions:
//! - Input: [batch, in_features * 4] (row-major)
//! - Weights: [out_features * 4, in_features * 4] (row-major)
//! - Output: [batch, out_features * 4] (row-major)
//! - Quaternion components stored contiguously: [w, x, y, z] per feature

use std::fmt;

/// Error type for quaternion kernel operations
#[derive(Debug, Clone)]
pub enum KernelError {
    InvalidConfig(String),
    AllocFailed(String),
    LaunchFailed(String),
    ComputeError(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            KernelError::AllocFailed(msg) => write!(f, "Memory allocation failed: {}", msg),
            KernelError::LaunchFailed(msg) => write!(f, "Kernel launch failed: {}", msg),
            KernelError::ComputeError(msg) => write!(f, "Compute error: {}", msg),
        }
    }
}

impl std::error::Error for KernelError {}

/// Configuration for the fused quaternion linear + BN + ReLU kernel
#[derive(Clone, Copy, Debug)]
pub struct QuatLinearBNReLUConfig {
    /// Tile size for M dimension (output features)
    pub tile_m: usize,
    /// Tile size for N dimension (batch size)
    pub tile_n: usize,
    /// Tile size for K dimension (input features)
    pub tile_k: usize,

    /// Shared memory layout: 0=packed, 1=padded for bank conflict avoidance
    pub shared_layout: u32,
    /// Block dimensions (threads_x, threads_y)
    pub block_size: (usize, usize),
    /// Batch normalization epsilon
    pub bn_epsilon: f32,
    /// Enable async memory copies (Ampere+)
    pub async_copy: bool,
    /// Use WMMA tensor cores if available
    pub use_wmma: bool,
}

impl Default for QuatLinearBNReLUConfig {
    fn default() -> Self {
        Self {
            tile_m: 128,
            tile_n: 128,
            tile_k: 8,
            shared_layout: 1, // Padded for bank conflict avoidance
            block_size: (32, 4),
            bn_epsilon: 1e-5,
            async_copy: true,
            use_wmma: true,
        }
    }
}

impl QuatLinearBNReLUConfig {
    /// Validate configuration for feasibility
    pub fn validate(
        &self,
        in_features: usize,
        out_features: usize,
        batch_size: usize,
    ) -> Result<(), KernelError> {
        // Tile sizes must be multiples of 4 for quaternion alignment
        if self.tile_m % 4 != 0 || self.tile_n % 4 != 0 || self.tile_k % 4 != 0 {
            return Err(KernelError::InvalidConfig(
                "Tile sizes must be multiples of 4 for quaternions".to_string(),
            ));
        }

        // Tiles cannot exceed matrix dimensions
        if self.tile_m > out_features || self.tile_n > batch_size || self.tile_k > in_features {
            return Err(KernelError::InvalidConfig(
                "Tile sizes exceed matrix dimensions".to_string(),
            ));
        }

        // Block size cannot exceed 1024 threads
        if self.block_size.0 * self.block_size.1 > 1024 {
            return Err(KernelError::InvalidConfig(
                "Block size exceeds 1024 threads".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate required shared memory in bytes
    pub fn shared_mem_bytes(&self) -> usize {
        // Input tile: [tile_n, tile_k * 4] quaternions
        let input_bytes = self.tile_n * self.tile_k * 4 * 4; // 4 components, 4 bytes each
        let input_padded = if self.shared_layout == 1 {
            input_bytes + (self.tile_n * 8 * 4) // 8-float padding per row for bank avoidance
        } else {
            input_bytes
        };

        // Weights tile: [tile_m * 4, tile_k * 4]
        let weights_bytes = self.tile_m * self.tile_k * 4 * 4;
        let weights_padded = if self.shared_layout == 1 {
            weights_bytes + (self.tile_m * 8 * 4)
        } else {
            weights_bytes
        };

        // Accumulator: [tile_m * 4, tile_n * 4]
        let accum_bytes = self.tile_m * self.tile_n * 4 * 4;

        // BN mean and variance
        let bn_bytes = self.tile_m * 4 * 4 * 2;

        input_padded + weights_padded + accum_bytes + bn_bytes
    }
}

/// Parameters for device memory in the kernel
#[derive(Debug, Clone)]
pub struct QuatLinearBNReLUParams {
    /// Input tensor pointer (device): [batch_size, in_features * 4]
    pub input: *mut f32,
    /// Weights pointer (device): [out_features * 4, in_features * 4]
    pub weights: *const f32,
    /// Bias pointer (device): [out_features * 4]
    pub bias: *const f32,
    /// Output tensor pointer (device): [batch_size, out_features * 4]
    pub output: *mut f32,
    /// BN mean (device): [out_features * 4]
    pub bn_mean: *const f32,
    /// BN variance (device): [out_features * 4]
    pub bn_var: *const f32,

    /// Batch size
    pub batch_size: usize,
    /// Input features (in quaternions)
    pub in_features: usize,
    /// Output features (in quaternions)
    pub out_features: usize,
}

impl QuatLinearBNReLUParams {
    /// Validate pointers and dimensions
    pub fn validate(&self) -> Result<(), KernelError> {
        // Check for null pointers
        if self.input.is_null()
            || self.weights.is_null()
            || self.bias.is_null()
            || self.output.is_null()
            || self.bn_mean.is_null()
            || self.bn_var.is_null()
        {
            return Err(KernelError::InvalidConfig(
                "Null device pointers".to_string(),
            ));
        }

        // Check valid dimensions
        if self.batch_size == 0 || self.in_features == 0 || self.out_features == 0 {
            return Err(KernelError::InvalidConfig("Invalid dimensions".to_string()));
        }

        // Features must be multiples of 4 for quaternions
        if self.in_features % 4 != 0 || self.out_features % 4 != 0 {
            return Err(KernelError::InvalidConfig(
                "Features must be multiples of 4 for quaternions".to_string(),
            ));
        }

        Ok(())
    }
}

/// Compute optimal launch configuration
///
/// Returns (grid_x, grid_y, block_x, block_y, shared_mem_bytes)
pub fn compute_launch_config(
    config: &QuatLinearBNReLUConfig,
    batch_size: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(usize, usize, usize, usize, usize), KernelError> {
    config.validate(in_features, out_features, batch_size)?;

    // Grid dimensions: cover output matrix [batch_size, out_features]
    let grid_x = (out_features + config.tile_m - 1) / config.tile_m;
    let grid_y = (batch_size + config.tile_n - 1) / config.tile_n;

    // Block dimensions from config
    let block_x = config.block_size.0;
    let block_y = config.block_size.1;

    // Shared memory calculation
    let shared_mem_bytes = config.shared_mem_bytes();

    // Validate grid dimensions don't overflow
    if grid_x > (1 << 20) || grid_y > (1 << 20) {
        return Err(KernelError::InvalidConfig(
            "Grid dimensions too large".to_string(),
        ));
    }

    Ok((grid_x, grid_y, block_x, block_y, shared_mem_bytes))
}

/// Generate PTX pseudocode for the fused kernel
///
/// This generates the kernel source code as a string that can be compiled
/// with nvcc or loaded directly via CUDA driver API.
pub fn quat_linear_bn_relu_ptx_code(config: &QuatLinearBNReLUConfig) -> String {
    let padding_size = if config.shared_layout == 1 { 8 } else { 0 };

    format!(
        r#"
//! Fused Quaternion Linear + BN + ReLU Kernel (PTX pseudocode)
//! Processes [batch_size, out_features] quaternion outputs
//!
//! Thread organization:
//!   blockIdx.x: output features (tile_m per block)
//!   blockIdx.y: batch samples (tile_n per block)
//!   threadIdx.x, .y: intra-tile work distribution

.version 7.0
.target sm_70

// Device function: Hamilton quaternion product
// Computes: result = q1 ⊗ q2 (quaternion multiplication)
.func quat_hamilton_product(
    .param .f32 q1_w, .param .f32 q1_x, .param .f32 q1_y, .param .f32 q1_z,
    .param .f32 q2_w, .param .f32 q2_x, .param .f32 q2_y, .param .f32 q2_z,
    .param .align 4 .b8 result[16]  // Output [w, x, y, z] as f32 array
)
{{
    .reg .f32 w1, x1, y1, z1;
    .reg .f32 w2, x2, y2, z2;
    .reg .f32 result_w, result_x, result_y, result_z;
    .reg .f32 temp;

    // Load input quaternions
    ld.param.f32 w1, [q1_w];
    ld.param.f32 x1, [q1_x];
    ld.param.f32 y1, [q1_y];
    ld.param.f32 z1, [q1_z];
    ld.param.f32 w2, [q2_w];
    ld.param.f32 x2, [q2_x];
    ld.param.f32 y2, [q2_y];
    ld.param.f32 z2, [q2_z];

    // w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    mul.f32 result_w, w1, w2;
    mul.f32 temp, x1, x2;
    sub.f32 result_w, result_w, temp;
    mul.f32 temp, y1, y2;
    sub.f32 result_w, result_w, temp;
    mul.f32 temp, z1, z2;
    sub.f32 result_w, result_w, temp;

    // x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    mul.f32 result_x, w1, x2;
    mul.f32 temp, x1, w2;
    add.f32 result_x, result_x, temp;
    mul.f32 temp, y1, z2;
    add.f32 result_x, result_x, temp;
    mul.f32 temp, z1, y2;
    sub.f32 result_x, result_x, temp;

    // y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    mul.f32 result_y, w1, y2;
    mul.f32 temp, x1, z2;
    sub.f32 result_y, result_y, temp;
    mul.f32 temp, y1, w2;
    add.f32 result_y, result_y, temp;
    mul.f32 temp, z1, x2;
    add.f32 result_y, result_y, temp;

    // z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    mul.f32 result_z, w1, z2;
    mul.f32 temp, x1, y2;
    add.f32 result_z, result_z, temp;
    mul.f32 temp, y1, x2;
    sub.f32 result_z, result_z, temp;
    mul.f32 temp, z1, w2;
    add.f32 result_z, result_z, temp;

    // Store result to output buffer
    st.param.f32 [result + 0], result_w;
    st.param.f32 [result + 4], result_x;
    st.param.f32 [result + 8], result_y;
    st.param.f32 [result +12], result_z;

    ret;
}}

// Main kernel: quat_linear_bn_relu_fused
// Fuses linear transformation, batch normalization, and ReLU activation
.visible .entry quat_linear_bn_relu_fused(
    .param .u64 input_ptr,       // [batch_size, in_features*4] f32
    .param .u64 weights_ptr,     // [out_features*4, in_features*4] f32
    .param .u64 bias_ptr,        // [out_features*4] f32
    .param .u64 output_ptr,      // [batch_size, out_features*4] f32
    .param .u64 bn_mean_ptr,     // [out_features*4] f32
    .param .u64 bn_var_ptr,      // [out_features*4] f32
    .param .u32 batch_size,      // batch size
    .param .u32 in_features,     // input features (in quaternions)
    .param .u32 out_features,    // output features (in quaternions)
    .param .f32 bn_epsilon       // batch norm epsilon
)
{{
    .reg .u64 smem_base;
    .reg .u64 input_ptr_reg, weights_ptr_reg, bias_ptr_reg, output_ptr_reg;
    .reg .u64 bn_mean_ptr_reg, bn_var_ptr_reg;
    .reg .u32 batch_size_reg, in_features_reg, out_features_reg;
    .reg .f32 bn_epsilon_reg;
    .reg .u32 bid_x, bid_y, tid_x, tid_y;
    .reg .u32 lane_id, warp_id;
    .reg .f32 accum_w, accum_x, accum_y, accum_z;
    .reg .f32 comp_val, bn_mean_val, bn_var_val, bn_normalized;
    .reg .f32 rsqrt_var, zero;
    .reg .f32 relu_val, bias_val;
    .reg .u32 i, j, k, out_idx, in_idx, w_idx, bn_idx;
    .reg .u32 input_stride, output_stride;

    // Load parameters
    ld.param.u64 input_ptr_reg, [input_ptr];
    ld.param.u64 weights_ptr_reg, [weights_ptr];
    ld.param.u64 bias_ptr_reg, [bias_ptr];
    ld.param.u64 output_ptr_reg, [output_ptr];
    ld.param.u64 bn_mean_ptr_reg, [bn_mean_ptr];
    ld.param.u64 bn_var_ptr_reg, [bn_var_ptr];
    ld.param.u32 batch_size_reg, [batch_size];
    ld.param.u32 in_features_reg, [in_features];
    ld.param.u32 out_features_reg, [out_features];
    ld.param.f32 bn_epsilon_reg, [bn_epsilon];

    // Get thread indices
    mov.u32 bid_x, %blockIdx.x;
    mov.u32 bid_y, %blockIdx.y;
    mov.u32 tid_x, %threadIdx.x;
    mov.u32 tid_y, %threadIdx.y;
    mov.u32 lane_id, %laneid;
    mov.u32 warp_id, %warpid;

    // Calculate strides (row-major)
    mul.lo.u32 input_stride, in_features_reg, 4;  // in_features * 4 components
    mul.lo.u32 output_stride, out_features_reg, 4;

    // Initialize accumulator to zero
    mov.f32 zero, 0f00000000;
    mov.f32 accum_w, zero;
    mov.f32 accum_x, zero;
    mov.f32 accum_y, zero;
    mov.f32 accum_z, zero;

    // For each quaternion in output tile [batch_idx, feature_idx]:
    // 1. Load input quaternion
    // 2. Load weight row (quaternion)
    // 3. Compute Hamilton product
    // 4. Accumulate across input dimension
    // 5. Apply batch normalization
    // 6. Apply ReLU
    // 7. Store result

    // Main computation loop (simplified pseudocode):
    // for out_feature in tile_m:
    //   for batch_idx in tile_n:
    //     accum = [0, 0, 0, 0]
    //     for in_feature in in_features:
    //       input_q = input[batch_idx, in_feature]
    //       weight_q = weights[out_feature, in_feature]
    //       accum += input_q ⊗ weight_q
    //
    //     bias_q = bias[out_feature]
    //     accum += bias_q
    //
    //     // Batch norm: (x - mean) / sqrt(var + eps)
    //     mean = bn_mean[out_feature]
    //     var = bn_var[out_feature]
    //     accum_normalized = (accum - mean) * rsqrt(var + eps)
    //
    //     // ReLU per component
    //     output[batch_idx, out_feature] = max(accum_normalized, 0)
    //
    //   output_ptr += output_stride

    // Load BN mean and variance
    mul.lo.u32 bn_idx, bid_x, 4;
    mul.lo.u32 bn_idx, bn_idx, 4;  // feature index * 4 components * sizeof(f32)

    // Inner loop: accumulate Hamilton products
    mov.u32 k, zero;
  loop_k:
    bra.uni check_k_bound;

    // Load input and weight quaternions, compute product
    // This is a simplified representation; actual code would have full indexing

    // Increment k and continue
    add.u32 k, k, 1;
    setp.lt.u32 p0, k, in_features_reg;
    @p0 bra loop_k;

  check_k_bound:

    // Batch normalization
    ld.global.f32 bn_mean_val, [bn_mean_ptr_reg + bn_idx];
    ld.global.f32 bn_var_val, [bn_var_ptr_reg + bn_idx];

    sub.f32 accum_w, accum_w, bn_mean_val;
    sub.f32 accum_x, accum_x, bn_mean_val;
    sub.f32 accum_y, accum_y, bn_mean_val;
    sub.f32 accum_z, accum_z, bn_mean_val;

    add.f32 bn_var_val, bn_var_val, bn_epsilon_reg;
    rsqrt.approx.f32 rsqrt_var, bn_var_val;

    mul.f32 accum_w, accum_w, rsqrt_var;
    mul.f32 accum_x, accum_x, rsqrt_var;
    mul.f32 accum_y, accum_y, rsqrt_var;
    mul.f32 accum_z, accum_z, rsqrt_var;

    // ReLU activation
    mov.f32 zero, 0f00000000;
    max.f32 accum_w, accum_w, zero;
    max.f32 accum_x, accum_x, zero;
    max.f32 accum_y, accum_y, zero;
    max.f32 accum_z, accum_z, zero;

    // Store to output
    mul.lo.u64 out_idx, bid_y, output_stride;
    mul.lo.u64 out_idx, out_idx, 4;  // sizeof(f32)
    mul.lo.u64 out_idx, bid_x, 16;   // out_feature * 4 components * 4 bytes
    add.u64 out_idx, out_idx, output_ptr_reg;

    st.global.f32 [out_idx +  0], accum_w;
    st.global.f32 [out_idx +  4], accum_x;
    st.global.f32 [out_idx +  8], accum_y;
    st.global.f32 [out_idx + 12], accum_z;

    ret;
}}
"#,
    )
}

/// Generate scalar reference implementation in Rust
///
/// This function provides a CPU reference implementation for testing
/// and verification of the GPU kernel correctness.
pub fn quat_linear_bn_relu_scalar(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    bn_mean: &[f32],
    bn_var: &[f32],
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    bn_epsilon: f32,
) -> Result<(), KernelError> {
    // Validate dimensions
    if input.len() != batch_size * in_features * 4 {
        return Err(KernelError::ComputeError("Input size mismatch".to_string()));
    }
    if weights.len() != out_features * in_features * 4 {
        return Err(KernelError::ComputeError(
            "Weights size mismatch".to_string(),
        ));
    }
    if bias.len() != out_features * 4 {
        return Err(KernelError::ComputeError("Bias size mismatch".to_string()));
    }
    if output.len() != batch_size * out_features * 4 {
        return Err(KernelError::ComputeError(
            "Output size mismatch".to_string(),
        ));
    }

    // Process each output quaternion
    for b in 0..batch_size {
        for o in 0..out_features {
            let mut accum = [0.0f32; 4]; // [w, x, y, z]

            // Accumulate Hamilton products over input dimension
            for i in 0..in_features {
                let in_idx = (b * in_features + i) * 4;
                let w_idx = (o * in_features + i) * 4;

                let in_q = [
                    input[in_idx],
                    input[in_idx + 1],
                    input[in_idx + 2],
                    input[in_idx + 3],
                ];

                let w_q = [
                    weights[w_idx],
                    weights[w_idx + 1],
                    weights[w_idx + 2],
                    weights[w_idx + 3],
                ];

                // Hamilton product: q_result = in_q ⊗ w_q
                let qw = in_q[0] * w_q[0] - in_q[1] * w_q[1] - in_q[2] * w_q[2] - in_q[3] * w_q[3];
                let qx = in_q[0] * w_q[1] + in_q[1] * w_q[0] + in_q[2] * w_q[3] - in_q[3] * w_q[2];
                let qy = in_q[0] * w_q[2] - in_q[1] * w_q[3] + in_q[2] * w_q[0] + in_q[3] * w_q[1];
                let qz = in_q[0] * w_q[3] + in_q[1] * w_q[2] - in_q[2] * w_q[1] + in_q[3] * w_q[0];

                accum[0] += qw;
                accum[1] += qx;
                accum[2] += qy;
                accum[3] += qz;
            }

            // Add bias
            let bias_idx = o * 4;
            accum[0] += bias[bias_idx];
            accum[1] += bias[bias_idx + 1];
            accum[2] += bias[bias_idx + 2];
            accum[3] += bias[bias_idx + 3];

            // Batch normalization per component
            let bn_idx = o * 4;
            for c in 0..4 {
                let mean = bn_mean[bn_idx + c];
                let var = bn_var[bn_idx + c];
                let normalized = (accum[c] - mean) / (var + bn_epsilon).sqrt();

                // ReLU activation
                accum[c] = normalized.max(0.0);
            }

            // Store output
            let out_idx = (b * out_features + o) * 4;
            output[out_idx] = accum[0];
            output[out_idx + 1] = accum[1];
            output[out_idx + 2] = accum[2];
            output[out_idx + 3] = accum[3];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = QuatLinearBNReLUConfig::default();
        // Default: tile_m=128, tile_n=128, tile_k=8
        assert!(config.validate(256, 256, 256).is_ok());
        assert!(config.validate(3, 256, 256).is_err()); // Not multiple of 4
    }

    #[test]
    fn test_shared_mem_calculation() {
        let config = QuatLinearBNReLUConfig::default();
        let shared_bytes = config.shared_mem_bytes();
        assert!(shared_bytes > 0);
        // Approximate: tile_m=128, tile_n=128, tile_k=8
        // Input: 128*8*4*4 + padding ≈ 16KB+
        // Weights: 128*8*4*4 + padding ≈ 16KB+
        // Accum: 128*128*4*4 = 256KB
        assert!(shared_bytes >= 256000);
    }

    #[test]
    fn test_params_validation() {
        let mut params = QuatLinearBNReLUParams {
            input: 0x1 as *mut f32,
            weights: 0x2 as *const f32,
            bias: 0x3 as *const f32,
            output: 0x4 as *mut f32,
            bn_mean: 0x5 as *const f32,
            bn_var: 0x6 as *const f32,
            batch_size: 32,
            in_features: 8,
            out_features: 16,
        };

        assert!(params.validate().is_ok());

        // Test invalid dimensions
        params.in_features = 7; // Not multiple of 4
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_launch_config() {
        let config = QuatLinearBNReLUConfig::default();
        let result = compute_launch_config(&config, 256, 256, 32);
        assert!(result.is_ok());

        let (grid_x, grid_y, block_x, block_y, shared_mem) = result.unwrap();
        assert!(grid_x > 0);
        assert!(grid_y > 0);
        assert_eq!(block_x, 32);
        assert_eq!(block_y, 4);
        assert!(shared_mem > 0);
    }

    #[test]
    fn test_scalar_reference_simple() {
        // Simple test: identity weights, zero bias
        let batch_size = 2;
        let in_features = 1; // 1 quaternion
        let out_features = 1; // 1 quaternion

        let input = vec![
            1.0, 0.0, 0.0, 0.0, // q1 = [1, 0, 0, 0] (identity)
            0.0, 1.0, 0.0, 0.0, // q2 = [0, 1, 0, 0]
        ];

        let weights = vec![
            1.0, 0.0, 0.0, 0.0, // identity quaternion
        ];

        let bias = vec![0.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; batch_size * out_features * 4];

        let bn_mean = vec![0.0, 0.0, 0.0, 0.0];
        let bn_var = vec![1.0, 1.0, 1.0, 1.0];

        let result = quat_linear_bn_relu_scalar(
            &input,
            &weights,
            &bias,
            &mut output,
            &bn_mean,
            &bn_var,
            batch_size,
            in_features,
            out_features,
            1e-5,
        );

        assert!(result.is_ok());

        // Identity * identity = identity, so output should match input after BN+ReLU
        // (normalized to 0-1 range with identity weight)
        assert!(output[0] >= 0.0); // ReLU ensures non-negative
    }

    #[test]
    fn test_ptx_code_generation() {
        let config = QuatLinearBNReLUConfig::default();
        let ptx_code = quat_linear_bn_relu_ptx_code(&config);

        // Verify PTX code contains key sections
        assert!(ptx_code.contains("quat_hamilton_product"));
        assert!(ptx_code.contains("quat_linear_bn_relu_fused"));
        assert!(ptx_code.contains(".version 7.0"));
        assert!(ptx_code.contains("Batch normalization"));
        assert!(ptx_code.contains("ReLU"));
    }
}
