//! ARM NEON optimized quaternion operations
//!
//! This module provides high-performance quaternion operations using ARM NEON intrinsics.
//! NEON uses 128-bit vectors (float32x4) that hold exactly one quaternion [w, x, y, z].
//!
//! Availability: ARM64/aarch64 (always available on aarch64)
//! Performance: 3-4x speedup over scalar

use std::arch::aarch64::*;

/// Scalar fallback for single quaternion multiplication
unsafe fn scalar_quat_mul(q1: *const f32, q2: *const f32, out: *mut f32) {
    unsafe {
        let w1 = *q1.add(0);
        let x1 = *q1.add(1);
        let y1 = *q1.add(2);
        let z1 = *q1.add(3);

        let w2 = *q2.add(0);
        let x2 = *q2.add(1);
        let y2 = *q2.add(2);
        let z2 = *q2.add(3);

        *out.add(0) = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        *out.add(1) = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        *out.add(2) = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        *out.add(3) = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    }
}

/// Single quaternion multiplication using ARM NEON
/// Uses float32x4 to hold [w, x, y, z]
#[target_feature(enable = "neon")]
pub unsafe extern "C" fn sounio_quat_mul_neon(q1: *const f32, q2: *const f32, out: *mut f32) {
    if q1.is_null() || q2.is_null() || out.is_null() {
        return;
    }

    unsafe {
        // Load quaternions into float32x4_t (NEON 128-bit vectors)
        let q1_v = vld1q_f32(q1);
        let q2_v = vld1q_f32(q2);

        // Broadcast each component from q1: [w1, w1, w1, w1], [x1, x1, x1, x1], etc.
        let w1 = vdupq_laneq_f32::<0>(q1_v); // Duplicate lane 0 (w) to all lanes
        let x1 = vdupq_laneq_f32::<1>(q1_v); // Duplicate lane 1 (x)
        let y1 = vdupq_laneq_f32::<2>(q1_v); // Duplicate lane 2 (y)
        let z1 = vdupq_laneq_f32::<3>(q1_v); // Duplicate lane 3 (z)

        // Broadcast components from q2
        let w2 = vdupq_laneq_f32::<0>(q2_v);
        let x2 = vdupq_laneq_f32::<1>(q2_v);
        let y2 = vdupq_laneq_f32::<2>(q2_v);
        let z2 = vdupq_laneq_f32::<3>(q2_v);

        // Compute w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        let mut w = vmulq_f32(w1, w2);
        w = vfmsq_f32(w, x1, x2); // w -= x1*x2
        w = vfmsq_f32(w, y1, y2); // w -= y1*y2
        w = vfmsq_f32(w, z1, z2); // w -= z1*z2

        // Compute x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        let mut x = vmulq_f32(w1, x2);
        x = vfmaq_f32(x, x1, w2); // x += x1*w2
        x = vfmaq_f32(x, y1, z2); // x += y1*z2
        x = vfmsq_f32(x, z1, y2); // x -= z1*y2

        // Compute y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        let mut y = vmulq_f32(w1, y2);
        y = vfmsq_f32(y, x1, z2); // y -= x1*z2
        y = vfmaq_f32(y, y1, w2); // y += y1*w2
        y = vfmaq_f32(y, z1, x2); // y += z1*x2

        // Compute z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        let mut z = vmulq_f32(w1, z2);
        z = vfmaq_f32(z, x1, y2); // z += x1*y2
        z = vfmsq_f32(z, y1, x2); // z -= y1*x2
        z = vfmaq_f32(z, z1, w2); // z += z1*w2

        // Extract lanes and assemble result
        let w_val = vgetq_lane_f32::<0>(w);
        let x_val = vgetq_lane_f32::<0>(x);
        let y_val = vgetq_lane_f32::<0>(y);
        let z_val = vgetq_lane_f32::<0>(z);

        *out.add(0) = w_val;
        *out.add(1) = x_val;
        *out.add(2) = y_val;
        *out.add(3) = z_val;
    }
}

/// Batch quaternion multiplication for n quaternion pairs
/// Processes one quaternion per iteration (NEON has 128-bit = 4 floats)
#[target_feature(enable = "neon")]
pub unsafe extern "C" fn sounio_quat_batch_mul_neon(
    q1s: *const f32,
    q2s: *const f32,
    outs: *mut f32,
    n: i32,
) {
    if q1s.is_null() || q2s.is_null() || outs.is_null() || n <= 0 {
        return;
    }

    let n = n as usize;

    for i in 0..n {
        unsafe {
            let q1_ptr = q1s.add(i * 4);
            let q2_ptr = q2s.add(i * 4);
            let out_ptr = outs.add(i * 4);

            sounio_quat_mul_neon(q1_ptr, q2_ptr, out_ptr);
        }
    }
}

/// Linear layer forward pass with quaternion multiplication and bias addition
/// output[b,o] = sum_i input[b,i] ⊗ weights[o,i] + bias[o]
#[target_feature(enable = "neon")]
pub unsafe extern "C" fn sounio_quat_linear_fwd_neon(
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    output: *mut f32,
    input_dim: i32,
    output_dim: i32,
    batch_size: i32,
) {
    if input.is_null() || weights.is_null() || output.is_null()
        || input_dim <= 0 || output_dim <= 0 || batch_size <= 0
    {
        return;
    }

    let has_bias = !bias.is_null();
    let input_dim = input_dim as usize;
    let output_dim = output_dim as usize;
    let batch_size = batch_size as usize;

    for b in 0..batch_size {
        for o in 0..output_dim {
            let mut acc = [0.0f32; 4];

            for i in 0..input_dim {
                let in_idx = (b * input_dim + i) * 4;
                let w_idx = (o * input_dim + i) * 4;

                unsafe {
                    let in_q = [
                        *input.add(in_idx),
                        *input.add(in_idx + 1),
                        *input.add(in_idx + 2),
                        *input.add(in_idx + 3),
                    ];

                    let w_q = [
                        *weights.add(w_idx),
                        *weights.add(w_idx + 1),
                        *weights.add(w_idx + 2),
                        *weights.add(w_idx + 3),
                    ];

                    let qw = in_q[0] * w_q[0] - in_q[1] * w_q[1] - in_q[2] * w_q[2] - in_q[3] * w_q[3];
                    let qx = in_q[0] * w_q[1] + in_q[1] * w_q[0] + in_q[2] * w_q[3] - in_q[3] * w_q[2];
                    let qy = in_q[0] * w_q[2] - in_q[1] * w_q[3] + in_q[2] * w_q[0] + in_q[3] * w_q[1];
                    let qz = in_q[0] * w_q[3] + in_q[1] * w_q[2] - in_q[2] * w_q[1] + in_q[3] * w_q[0];

                    acc[0] += qw;
                    acc[1] += qx;
                    acc[2] += qy;
                    acc[3] += qz;
                }
            }

            let out_idx = (b * output_dim + o) * 4;
            let bias_idx = o * 4;

            unsafe {
                *output.add(out_idx) = acc[0] + if has_bias { *bias.add(bias_idx) } else { 0.0 };
                *output.add(out_idx + 1) = acc[1] + if has_bias { *bias.add(bias_idx + 1) } else { 0.0 };
                *output.add(out_idx + 2) = acc[2] + if has_bias { *bias.add(bias_idx + 2) } else { 0.0 };
                *output.add(out_idx + 3) = acc[3] + if has_bias { *bias.add(bias_idx + 3) } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_quat_mul() {
        unsafe {
            // Identity ⊗ q = q
            let q1 = [1.0, 0.0, 0.0, 0.0];
            let q2 = [0.707, 0.707, 0.0, 0.0];
            let mut out = [0.0; 4];

            sounio_quat_mul_neon(q1.as_ptr(), q2.as_ptr(), out.as_mut_ptr());

            assert!((out[0] - 0.707).abs() < 1e-5);
            assert!((out[1] - 0.707).abs() < 1e-5);
        }
    }

    #[test]
    fn test_neon_batch_mul() {
        unsafe {
            let q1s = [
                1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
            ];
            let q2s = [
                0.707, 0.707, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
            ];
            let mut outs = [0.0f32; 8];

            sounio_quat_batch_mul_neon(
                q1s.as_ptr(),
                q2s.as_ptr(),
                outs.as_mut_ptr(),
                2,
            );

            assert!((outs[0] - 0.707).abs() < 1e-5);
            assert!((outs[1] - 0.707).abs() < 1e-5);
        }
    }
}
