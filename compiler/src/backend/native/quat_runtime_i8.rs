//! INT8 Quaternion Runtime Functions
//!
//! Provides efficient INT8 quaternion operations for CPU execution:
//! - INT8 Hamilton product with INT32 accumulation
//! - Batch operations with dynamic scale handling
//! - Accurate requantization with minimal error
//! - Integration with dispatch system for future SIMD optimization
//!
//! Performance Target:
//! - 1.6-2x faster than FP32 due to reduced memory bandwidth
//! - 4x memory reduction (16 bytes → 4 bytes per quaternion)
//! - Compatible with dp4a (Maxwell+) and VNNI (Skylake+) instructions
//!
//! All functions follow C ABI conventions for integration with Sounio
//! type system. Scales are passed separately for flexibility.

use std::f32;

/// Represents an INT8 quaternion with individual scale
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QuatI8 {
    pub w: i8,
    pub x: i8,
    pub y: i8,
    pub z: i8,
}

impl QuatI8 {
    pub fn new(w: i8, x: i8, y: i8, z: i8) -> Self {
        Self { w, x, y, z }
    }

    /// Dequantizes to f32 quaternion
    pub fn dequantize(&self, scale: f32) -> [f32; 4] {
        [
            (self.w as f32) * scale,
            (self.x as f32) * scale,
            (self.y as f32) * scale,
            (self.z as f32) * scale,
        ]
    }
}

/// ============================================================================
/// Single Quaternion Operations
/// ============================================================================

/// Multiplies two INT8 quaternions using Hamilton product
/// Computation in INT32 to prevent overflow, result requantized
///
/// # Arguments
/// * `q1` - First quaternion (i8 components)
/// * `scale1` - Scale factor for q1
/// * `q2` - Second quaternion (i8 components)
/// * `scale2` - Scale factor for q2
///
/// # Returns
/// (output_quat, output_scale)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_mul_i8(
    q1_w: i8,
    q1_x: i8,
    q1_y: i8,
    q1_z: i8,
    scale1: f32,
    q2_w: i8,
    q2_x: i8,
    q2_y: i8,
    q2_z: i8,
    scale2: f32,
    out_w: *mut i8,
    out_x: *mut i8,
    out_y: *mut i8,
    out_z: *mut i8,
    out_scale: *mut f32,
) {
    // Dequantize both inputs temporarily
    let q1 = [
        (q1_w as f32) * scale1,
        (q1_x as f32) * scale1,
        (q1_y as f32) * scale1,
        (q1_z as f32) * scale1,
    ];

    let q2 = [
        (q2_w as f32) * scale2,
        (q2_x as f32) * scale2,
        (q2_y as f32) * scale2,
        (q2_z as f32) * scale2,
    ];

    // Hamilton product
    let w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    let x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    let y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    let z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];

    // Requantize output
    let max_abs = w.abs().max(x.abs()).max(y.abs()).max(z.abs()).max(1e-7);
    let new_scale = max_abs / 127.0;

    unsafe {
        *out_w = (w / new_scale).clamp(-128.0, 127.0) as i8;
        *out_x = (x / new_scale).clamp(-128.0, 127.0) as i8;
        *out_y = (y / new_scale).clamp(-128.0, 127.0) as i8;
        *out_z = (z / new_scale).clamp(-128.0, 127.0) as i8;
        *out_scale = new_scale;
    }
}

/// INT32-based Hamilton product (avoids intermediate f32 conversion)
/// More efficient for GPU, shown here for reference
#[inline(never)]
pub fn quat_mul_i8_int32(
    q1: QuatI8,
    scale1: f32,
    q2: QuatI8,
    scale2: f32,
) -> (QuatI8, f32) {
    // Convert to INT32 for accumulation (prevents i8 overflow)
    let q1w = q1.w as i32;
    let q1x = q1.x as i32;
    let q1y = q1.y as i32;
    let q1z = q1.z as i32;

    let q2w = q2.w as i32;
    let q2x = q2.x as i32;
    let q2y = q2.y as i32;
    let q2z = q2.z as i32;

    // Hamilton product in INT32
    let w_i32 = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z;
    let x_i32 = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y;
    let y_i32 = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x;
    let z_i32 = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w;

    // Convert back to f32 with combined scale for requantization
    let combined_scale = scale1 * scale2;
    let w_f = (w_i32 as f32) * combined_scale;
    let x_f = (x_i32 as f32) * combined_scale;
    let y_f = (y_i32 as f32) * combined_scale;
    let z_f = (z_i32 as f32) * combined_scale;

    // Compute new scale and requantize
    let max_abs = w_f
        .abs()
        .max(x_f.abs())
        .max(y_f.abs())
        .max(z_f.abs())
        .max(1e-7);
    let new_scale = max_abs / 127.0;

    let result = QuatI8::new(
        (w_f / new_scale).clamp(-128.0, 127.0) as i8,
        (x_f / new_scale).clamp(-128.0, 127.0) as i8,
        (y_f / new_scale).clamp(-128.0, 127.0) as i8,
        (z_f / new_scale).clamp(-128.0, 127.0) as i8,
    );

    (result, new_scale)
}

/// ============================================================================
/// Linear Layer: Forward Pass (INT8 activation, FP32 weights)
/// ============================================================================

/// Computes quaternion linear layer: y = W @ x + b
///
/// # Arguments
/// * `x` - Input quaternions (i8)
/// * `x_scale` - Input scale
/// * `w` - Weight quaternions (f32)
/// * `b` - Bias quaternions (f32)
/// * `out` - Output quaternions (i8)
/// * `out_scale_ptr` - Pointer to output scale
/// * `num_out` - Number of output quaternions
/// * `num_in` - Number of input quaternions
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_linear_i8_fwd(
    x: *const i8,
    x_scale: f32,
    w: *const f32,
    b: *const f32,
    out: *mut i8,
    out_scale: *mut f32,
    num_out: i32,
    num_in: i32,
) {
    let x_ptr = unsafe { std::slice::from_raw_parts(x, (num_out * num_in * 4) as usize) };
    let w_ptr = unsafe { std::slice::from_raw_parts(w, (num_out * num_in * 4) as usize) };
    let b_ptr = unsafe { std::slice::from_raw_parts(b, (num_out * 4) as usize) };
    let out_ptr =
        unsafe { std::slice::from_raw_parts_mut(out, (num_out * 4) as usize as usize) };

    // For each output quaternion
    for o in 0..num_out as usize {
        let mut result = [0.0f32; 4];

        // Add bias
        result[0] = b_ptr[o * 4];
        result[1] = b_ptr[o * 4 + 1];
        result[2] = b_ptr[o * 4 + 2];
        result[3] = b_ptr[o * 4 + 3];

        // Accumulate weighted sum
        for i in 0..num_in as usize {
            let w_idx = (o * num_in as usize + i) * 4;
            let x_idx = i * 4;

            // Dequantize input
            let x_q = [
                (x_ptr[x_idx] as f32) * x_scale,
                (x_ptr[x_idx + 1] as f32) * x_scale,
                (x_ptr[x_idx + 2] as f32) * x_scale,
                (x_ptr[x_idx + 3] as f32) * x_scale,
            ];

            // Load weight
            let w_q = [
                w_ptr[w_idx],
                w_ptr[w_idx + 1],
                w_ptr[w_idx + 2],
                w_ptr[w_idx + 3],
            ];

            // Quaternion multiply and accumulate
            let prod_w = w_q[0] * x_q[0] - w_q[1] * x_q[1] - w_q[2] * x_q[2] - w_q[3] * x_q[3];
            let prod_x = w_q[0] * x_q[1] + w_q[1] * x_q[0] + w_q[2] * x_q[3] - w_q[3] * x_q[2];
            let prod_y = w_q[0] * x_q[2] - w_q[1] * x_q[3] + w_q[2] * x_q[0] + w_q[3] * x_q[1];
            let prod_z = w_q[0] * x_q[3] + w_q[1] * x_q[2] - w_q[2] * x_q[1] + w_q[3] * x_q[0];

            result[0] += prod_w;
            result[1] += prod_x;
            result[2] += prod_y;
            result[3] += prod_z;
        }

        // Requantize output
        let max_abs = result[0]
            .abs()
            .max(result[1].abs())
            .max(result[2].abs())
            .max(result[3].abs())
            .max(1e-7);
        let scale = max_abs / 127.0;

        out_ptr[o * 4] = (result[0] / scale).clamp(-128.0, 127.0) as i8;
        out_ptr[o * 4 + 1] = (result[1] / scale).clamp(-128.0, 127.0) as i8;
        out_ptr[o * 4 + 2] = (result[2] / scale).clamp(-128.0, 127.0) as i8;
        out_ptr[o * 4 + 3] = (result[3] / scale).clamp(-128.0, 127.0) as i8;

        unsafe {
            *out_scale.offset(o as isize) = scale;
        }
    }
}

/// ============================================================================
/// Batch Operations
/// ============================================================================

/// Batch multiply: out[i] = q1[i] * q2[i]
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_batch_mul_i8(
    q1: *const i8,
    q1_scales: *const f32,
    q2: *const i8,
    q2_scales: *const f32,
    out: *mut i8,
    out_scales: *mut f32,
    n: i32,
) {
    let q1_ptr = unsafe { std::slice::from_raw_parts(q1, (n as usize) * 4) };
    let q1_scale_ptr = unsafe { std::slice::from_raw_parts(q1_scales, n as usize) };
    let q2_ptr = unsafe { std::slice::from_raw_parts(q2, (n as usize) * 4) };
    let q2_scale_ptr = unsafe { std::slice::from_raw_parts(q2_scales, n as usize) };
    let out_ptr = unsafe { std::slice::from_raw_parts_mut(out, (n as usize) * 4) };
    let out_scale_ptr = unsafe { std::slice::from_raw_parts_mut(out_scales, n as usize) };

    for i in 0..n as usize {
        let q1_i8 = QuatI8::new(
            q1_ptr[i * 4],
            q1_ptr[i * 4 + 1],
            q1_ptr[i * 4 + 2],
            q1_ptr[i * 4 + 3],
        );
        let q2_i8 = QuatI8::new(
            q2_ptr[i * 4],
            q2_ptr[i * 4 + 1],
            q2_ptr[i * 4 + 2],
            q2_ptr[i * 4 + 3],
        );

        let (result, out_scale) = quat_mul_i8_int32(q1_i8, q1_scale_ptr[i], q2_i8, q2_scale_ptr[i]);

        out_ptr[i * 4] = result.w;
        out_ptr[i * 4 + 1] = result.x;
        out_ptr[i * 4 + 2] = result.y;
        out_ptr[i * 4 + 3] = result.z;
        out_scale_ptr[i] = out_scale;
    }
}

/// ============================================================================
/// Quantization/Dequantization
/// ============================================================================

/// Quantizes f32 quaternion to INT8 with automatic scale computation
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_quantize_f32_to_i8(
    quat: *const f32,
    out: *mut i8,
    out_scale: *mut f32,
) {
    let quat_slice = unsafe { std::slice::from_raw_parts(quat, 4) };
    let w = quat_slice[0];
    let x = quat_slice[1];
    let y = quat_slice[2];
    let z = quat_slice[3];

    let max_abs = w.abs().max(x.abs()).max(y.abs()).max(z.abs()).max(1e-7);
    let scale = max_abs / 127.0;

    unsafe {
        *out = (w / scale).clamp(-128.0, 127.0) as i8;
        *out.offset(1) = (x / scale).clamp(-128.0, 127.0) as i8;
        *out.offset(2) = (y / scale).clamp(-128.0, 127.0) as i8;
        *out.offset(3) = (z / scale).clamp(-128.0, 127.0) as i8;
        *out_scale = scale;
    }
}

/// Dequantizes INT8 quaternion to f32
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_dequantize_i8_to_f32(
    quat: *const i8,
    scale: f32,
    out: *mut f32,
) {
    let quat_slice = unsafe { std::slice::from_raw_parts(quat, 4) };

    unsafe {
        *out = (quat_slice[0] as f32) * scale;
        *out.offset(1) = (quat_slice[1] as f32) * scale;
        *out.offset(2) = (quat_slice[2] as f32) * scale;
        *out.offset(3) = (quat_slice[3] as f32) * scale;
    }
}

/// ============================================================================
/// Utility Functions
/// ============================================================================

/// Computes Frobenius norm of INT8 quaternion batch (for monitoring)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_batch_norm_i8(
    quats: *const i8,
    scales: *const f32,
    n: i32,
) -> f32 {
    let quat_ptr = unsafe { std::slice::from_raw_parts(quats, (n as usize) * 4) };
    let scale_ptr = unsafe { std::slice::from_raw_parts(scales, n as usize) };

    let mut sum_sq = 0.0f32;
    for i in 0..n as usize {
        for j in 0..4 {
            let val = (quat_ptr[i * 4 + j] as f32) * scale_ptr[i];
            sum_sq += val * val;
        }
    }

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_i8_new() {
        let q = QuatI8::new(1, 2, 3, 4);
        assert_eq!(q.w, 1);
        assert_eq!(q.x, 2);
        assert_eq!(q.y, 3);
        assert_eq!(q.z, 4);
    }

    #[test]
    fn test_dequantize() {
        let q = QuatI8::new(127, 0, 0, 0);
        let scale = 1.0 / 127.0;
        let deq = q.dequantize(scale);

        assert!((deq[0] - 1.0).abs() < 0.01);
        assert!((deq[1] - 0.0).abs() < 0.01);
        assert!((deq[2] - 0.0).abs() < 0.01);
        assert!((deq[3] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_quat_mul_i8_int32() {
        let q1 = QuatI8::new(127, 0, 0, 0); // [1.0, 0, 0, 0] * scale
        let q2 = QuatI8::new(127, 0, 0, 0);
        let scale = 1.0 / 127.0;

        let (result, out_scale) = quat_mul_i8_int32(q1, scale, q2, scale);

        // Identity * Identity = Identity
        assert!(result.w > 100); // Should be close to 127
        assert_eq!(result.x, 0);
        assert_eq!(result.y, 0);
        assert_eq!(result.z, 0);
        assert!(out_scale > 0.0);
    }

    #[test]
    fn test_batch_norm() {
        let quats = [127i8, 0, 0, 0, 63, 0, 0, 0];
        let scales = [1.0 / 127.0, 1.0 / 127.0];

        let norm = unsafe { sounio_quat_batch_norm_i8(quats.as_ptr(), scales.as_ptr(), 2) };

        assert!(norm > 1.0);
        assert!(norm < 2.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = [1.5, 0.5, 0.25, 0.125];
        let mut q_i8 = [0i8; 4];
        let mut scale = 0.0f32;

        unsafe {
            sounio_quat_quantize_f32_to_i8(original.as_ptr(), q_i8.as_mut_ptr(), &mut scale);
        }

        let mut deq = [0.0f32; 4];
        unsafe {
            sounio_quat_dequantize_i8_to_f32(q_i8.as_ptr(), scale, deq.as_mut_ptr());
        }

        // Check roundtrip error
        for i in 0..4 {
            assert!((original[i] - deq[i]).abs() < 0.05);
        }
    }

    #[test]
    fn test_identity_multiply() {
        let identity = QuatI8::new(127, 0, 0, 0);
        let other = QuatI8::new(64, 32, 16, 8);

        let scale_id = 1.0 / 127.0;
        let scale_other = 0.5 / 127.0;

        let (result, _out_scale) = quat_mul_i8_int32(identity, scale_id, other, scale_other);

        // Identity * x = x (approximately, given quantization)
        let result_deq = result.dequantize(_out_scale);
        let other_deq = other.dequantize(scale_other);

        // Results should be similar
        for i in 0..4 {
            assert!((result_deq[i] - other_deq[i]).abs() < 0.2);
        }
    }
}
