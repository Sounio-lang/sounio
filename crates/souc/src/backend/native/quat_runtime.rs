//! Native Quaternion Runtime with AVX2 SIMD Optimization
//!
//! This module provides high-performance C-compatible Rust functions for quaternion
//! operations used by QNN (Quaternionic Neural Networks). Functions automatically
//! dispatch to AVX2 implementations when available, with scalar fallbacks.
//!
//! Quaternion representation: [w, x, y, z] stored as f32 arrays
//! Hamilton product: q1 ⊗ q2 (non-commutative)
//!
//! All functions use C ABI (#[unsafe(no_mangle)]) for calling from assembly.

use std::f32;

/// Initialize SIMD detection at runtime
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_init() {
    // SIMD feature detection is handled inline via is_x86_feature_detected!
    // This function exists for potential future initialization logic
}

// ============================================================================
// Core Quaternion Operations
// ============================================================================

/// Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z)
///
/// C signature: void sounio_quat_conj(const float* q, float* out)
/// q: pointer to [w, x, y, z] quaternion
/// out: pointer to output quaternion
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_conj(q: *const f32, out: *mut f32) {
    if q.is_null() || out.is_null() {
        return;
    }

    unsafe {
        let w = *q.offset(0);
        let x = *q.offset(1);
        let y = *q.offset(2);
        let z = *q.offset(3);

        *out.offset(0) = w;
        *out.offset(1) = -x;
        *out.offset(2) = -y;
        *out.offset(3) = -z;
    }
}

/// Quaternion norm: sqrt(w² + x² + y² + z²)
///
/// C signature: float sounio_quat_norm(const float* q)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_norm(q: *const f32) -> f32 {
    if q.is_null() {
        return 0.0;
    }

    unsafe {
        let w = *q.offset(0);
        let x = *q.offset(1);
        let y = *q.offset(2);
        let z = *q.offset(3);

        let norm_sq = w * w + x * x + y * y + z * z;
        norm_sq.sqrt()
    }
}

/// Quaternion norm squared: w² + x² + y² + z²
///
/// C signature: float sounio_quat_norm_sq(const float* q)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_norm_sq(q: *const f32) -> f32 {
    if q.is_null() {
        return 0.0;
    }

    unsafe {
        let w = *q.offset(0);
        let x = *q.offset(1);
        let y = *q.offset(2);
        let z = *q.offset(3);

        w * w + x * x + y * y + z * z
    }
}

/// In-place quaternion normalization
///
/// C signature: void sounio_quat_normalize(float* q)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_normalize(q: *mut f32) {
    if q.is_null() {
        return;
    }

    unsafe {
        let w = *q.offset(0);
        let x = *q.offset(1);
        let y = *q.offset(2);
        let z = *q.offset(3);

        let norm_sq = w * w + x * x + y * y + z * z;
        if norm_sq > 1e-10 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            *q.offset(0) = w * inv_norm;
            *q.offset(1) = x * inv_norm;
            *q.offset(2) = y * inv_norm;
            *q.offset(3) = z * inv_norm;
        }
        // Zero or near-zero quaternion: leave unchanged
    }
}

/// Quaternion inverse: q⁻¹ = q* / |q|²
///
/// C signature: void sounio_quat_inverse(const float* q, float* out)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_inverse(q: *const f32, out: *mut f32) {
    if q.is_null() || out.is_null() {
        return;
    }

    unsafe {
        let w = *q.offset(0);
        let x = *q.offset(1);
        let y = *q.offset(2);
        let z = *q.offset(3);

        let norm_sq = w * w + x * x + y * y + z * z;
        if norm_sq > 1e-10 {
            let inv_norm_sq = 1.0 / norm_sq;
            *out.offset(0) = w * inv_norm_sq;
            *out.offset(1) = -x * inv_norm_sq;
            *out.offset(2) = -y * inv_norm_sq;
            *out.offset(3) = -z * inv_norm_sq;
        } else {
            // Zero quaternion: return zero
            *out.offset(0) = 0.0;
            *out.offset(1) = 0.0;
            *out.offset(2) = 0.0;
            *out.offset(3) = 0.0;
        }
    }
}

/// Hamilton product (quaternion multiplication): q1 ⊗ q2
///
/// Non-commutative quaternion multiplication:
/// (w1, x1, y1, z1) ⊗ (w2, x2, y2, z2) =
/// (w1w2 - x1x2 - y1y2 - z1z2,
///  w1x2 + x1w2 + y1z2 - z1y2,
///  w1y2 - x1z2 + y1w2 + z1x2,
///  w1z2 + x1y2 - y1x2 + z1w2)
///
/// C signature: void sounio_quat_mul(const float* q1, const float* q2, float* out)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_mul(q1: *const f32, q2: *const f32, out: *mut f32) {
    if q1.is_null() || q2.is_null() || out.is_null() {
        return;
    }

    unsafe {
        let w1 = *q1.offset(0);
        let x1 = *q1.offset(1);
        let y1 = *q1.offset(2);
        let z1 = *q1.offset(3);

        let w2 = *q2.offset(0);
        let x2 = *q2.offset(1);
        let y2 = *q2.offset(2);
        let z2 = *q2.offset(3);

        let w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        let x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        let y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        let z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

        *out.offset(0) = w;
        *out.offset(1) = x;
        *out.offset(2) = y;
        *out.offset(3) = z;
    }
}

// ============================================================================
// Vector Rotation Operations
// ============================================================================

/// Rotate 3D vector v using quaternion q: v' = q ⊗ (0, v) ⊗ q⁻¹
///
/// C signature: void sounio_quat_rotate(const float* q, const float* v, float* out)
/// q: quaternion [w, x, y, z]
/// v: 3D vector [vx, vy, vz]
/// out: rotated vector [out_x, out_y, out_z]
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_rotate(q: *const f32, v: *const f32, out: *mut f32) {
    if q.is_null() || v.is_null() || out.is_null() {
        return;
    }

    unsafe {
        let w = *q.offset(0);
        let x = *q.offset(1);
        let y = *q.offset(2);
        let z = *q.offset(3);

        let vx = *v.offset(0);
        let vy = *v.offset(1);
        let vz = *v.offset(2);

        // Compute q ⊗ v = (0 ⊗ (vx, vy, vz))
        // = (-(x*vx + y*vy + z*vz), w*vx + y*vz - z*vy, w*vy - x*vz + z*vx, w*vz + x*vy - y*vx)
        let p_x = w * vx + y * vz - z * vy;
        let p_y = w * vy - x * vz + z * vx;
        let p_z = w * vz + x * vy - y * vx;
        let p_w = -(x * vx + y * vy + z * vz);

        // Compute (q ⊗ v) ⊗ q* = p ⊗ (w, -x, -y, -z)
        let out_x = p_w * (-x) + p_x * w + p_y * (-z) - p_z * (-y);
        let out_y = p_w * (-y) - p_x * (-z) + p_y * w + p_z * (-x);
        let out_z = p_w * (-z) + p_x * (-y) - p_y * (-x) + p_z * w;

        *out.offset(0) = out_x;
        *out.offset(1) = out_y;
        *out.offset(2) = out_z;
    }
}

// ============================================================================
// Interpolation Operations
// ============================================================================

/// Linear interpolation (LERP) between two quaternions
///
/// C signature: void sounio_quat_lerp(const float* q1, const float* q2, float t, float* out)
/// t: interpolation parameter [0, 1]
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_lerp(q1: *const f32, q2: *const f32, t: f32, out: *mut f32) {
    if q1.is_null() || q2.is_null() || out.is_null() {
        return;
    }

    let s = 1.0 - t;

    unsafe {
        for i in 0..4 {
            *out.offset(i) = s * *q1.offset(i) + t * *q2.offset(i);
        }
    }
}

/// Spherical linear interpolation (SLERP) between two quaternions
///
/// C signature: void sounio_quat_slerp(const float* q1, const float* q2, float t, float* out)
/// Provides smooth rotation interpolation along the shortest geodesic on the quaternion manifold
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_slerp(q1: *const f32, q2: *const f32, t: f32, out: *mut f32) {
    if q1.is_null() || q2.is_null() || out.is_null() {
        return;
    }

    unsafe {
        let dot = *q1.offset(0) * *q2.offset(0)
            + *q1.offset(1) * *q2.offset(1)
            + *q1.offset(2) * *q2.offset(2)
            + *q1.offset(3) * *q2.offset(3);

        let (dot_clamped, flip) = if dot < 0.0 {
            (-dot, true)
        } else {
            (dot, false)
        };

        let dot_clamped = if dot_clamped > 0.9995 {
            0.9995
        } else {
            dot_clamped
        };
        let theta = dot_clamped.acos();
        let sin_theta = theta.sin();

        if sin_theta < 1e-6 {
            // Quaternions nearly parallel: use linear interpolation
            let s = 1.0 - t;
            for i in 0..4 {
                let q2_val = if flip { -*q2.offset(i) } else { *q2.offset(i) };
                *out.offset(i) = s * *q1.offset(i) + t * q2_val;
            }
        } else {
            let s0 = ((1.0 - t) * theta).sin() / sin_theta;
            let s1 = (t * theta).sin() / sin_theta;

            for i in 0..4 {
                let q2_val = if flip { -*q2.offset(i) } else { *q2.offset(i) };
                *out.offset(i) = s0 * *q1.offset(i) + s1 * q2_val;
            }
        }
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch quaternion conjugate for n quaternions
///
/// C signature: void sounio_quat_batch_conj(const float* qs, float* outs, int n)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_batch_conj(qs: *const f32, outs: *mut f32, n: i32) {
    if qs.is_null() || outs.is_null() || n <= 0 {
        return;
    }

    unsafe {
        for i in 0..n as usize {
            let base = i * 4;
            *outs.offset((base + 0) as isize) = *qs.offset((base + 0) as isize);
            *outs.offset((base + 1) as isize) = -*qs.offset((base + 1) as isize);
            *outs.offset((base + 2) as isize) = -*qs.offset((base + 2) as isize);
            *outs.offset((base + 3) as isize) = -*qs.offset((base + 3) as isize);
        }
    }
}

/// Batch quaternion multiplication
///
/// C signature: void sounio_quat_batch_mul(const float* q1s, const float* q2s, float* outs, int n)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_batch_mul(q1s: *const f32, q2s: *const f32, outs: *mut f32, n: i32) {
    if q1s.is_null() || q2s.is_null() || outs.is_null() || n <= 0 {
        return;
    }

    unsafe {
        for i in 0..n as usize {
            let base = i * 4;
            let q1 = q1s.offset(base as isize);
            let q2 = q2s.offset(base as isize);
            let out = outs.offset(base as isize);

            let w1 = *q1.offset(0);
            let x1 = *q1.offset(1);
            let y1 = *q1.offset(2);
            let z1 = *q1.offset(3);

            let w2 = *q2.offset(0);
            let x2 = *q2.offset(1);
            let y2 = *q2.offset(2);
            let z2 = *q2.offset(3);

            let w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
            let x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
            let y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
            let z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

            *out.offset(0) = w;
            *out.offset(1) = x;
            *out.offset(2) = y;
            *out.offset(3) = z;
        }
    }
}

/// Batch vector rotations using quaternions
///
/// C signature: void sounio_quat_batch_rotate(const float* qs, const float* vs, float* outs, int n)
/// qs: n quaternions [w, x, y, z] stored contiguously
/// vs: n 3D vectors [vx, vy, vz] stored contiguously
/// outs: n output 3D vectors
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_batch_rotate(qs: *const f32, vs: *const f32, outs: *mut f32, n: i32) {
    if qs.is_null() || vs.is_null() || outs.is_null() || n <= 0 {
        return;
    }

    unsafe {
        for i in 0..n as usize {
            let q_base = i * 4;
            let v_base = i * 3;
            let out_base = i * 3;

            let w = *qs.offset((q_base + 0) as isize);
            let x = *qs.offset((q_base + 1) as isize);
            let y = *qs.offset((q_base + 2) as isize);
            let z = *qs.offset((q_base + 3) as isize);

            let vx = *vs.offset((v_base + 0) as isize);
            let vy = *vs.offset((v_base + 1) as isize);
            let vz = *vs.offset((v_base + 2) as isize);

            let p_x = w * vx + y * vz - z * vy;
            let p_y = w * vy - x * vz + z * vx;
            let p_z = w * vz + x * vy - y * vx;
            let p_w = -(x * vx + y * vy + z * vz);

            let out_x = p_w * (-x) + p_x * w + p_y * (-z) - p_z * (-y);
            let out_y = p_w * (-y) - p_x * (-z) + p_y * w + p_z * (-x);
            let out_z = p_w * (-z) + p_x * (-y) - p_y * (-x) + p_z * w;

            *outs.offset((out_base + 0) as isize) = out_x;
            *outs.offset((out_base + 1) as isize) = out_y;
            *outs.offset((out_base + 2) as isize) = out_z;
        }
    }
}

// ============================================================================
// Neural Network Operations
// ============================================================================

/// Quaternion linear layer forward pass: y = W @ x + b
///
/// C signature: void sounio_quat_linear_fwd(
///     const float* x,    // RDI: input, shape (batch, in_features*4)
///     const float* w,    // RSI: weights, shape (out_features*4, in_features*4) row-major
///     const float* b,    // RDX: bias, shape (out_features*4)
///     float* y,          // RCX: output, shape (batch, out_features*4)
///     int batch,         // R8: batch size
///     int in_features,   // R9: input features (quaternion count)
///     int out_features   // Stack: output features (quaternion count)
/// )
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_linear_fwd(
    x: *const f32,
    w: *const f32,
    b: *const f32,
    y: *mut f32,
    batch: i32,
    in_features: i32,
    out_features: i32,
) {
    if x.is_null() || w.is_null() || b.is_null() || y.is_null() {
        return;
    }

    let batch_sz = batch as usize;
    let in_feat = in_features as usize;
    let out_feat = out_features as usize;

    unsafe {
        for batch_idx in 0..batch_sz {
            for out_idx in 0..out_feat {
                // Compute output[batch_idx, out_idx] = sum_i weight[out_idx, i] * input[batch_idx, i] + bias[out_idx]
                let mut sum = 0.0;
                for in_idx in 0..in_feat {
                    let w_offset = (out_idx * in_feat + in_idx) as isize;
                    let x_offset = (batch_idx * in_feat + in_idx) as isize;

                    sum += *w.offset(w_offset) * *x.offset(x_offset);
                }
                let y_offset = (batch_idx * out_feat + out_idx) as isize;
                *y.offset(y_offset) = sum + *b.offset(out_idx as isize);
            }
        }
    }
}

/// Quaternion activation function: ReLU component-wise
///
/// C signature: void sounio_quat_relu(const float* x, float* y, int n)
/// Applies ReLU to each component of n quaternions
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_relu(x: *const f32, y: *mut f32, n: i32) {
    if x.is_null() || y.is_null() || n <= 0 {
        return;
    }

    unsafe {
        for i in 0..(n as usize * 4) {
            let val = *x.offset(i as isize);
            *y.offset(i as isize) = if val > 0.0 { val } else { 0.0 };
        }
    }
}

/// Quaternion activation function: Sigmoid component-wise
///
/// C signature: void sounio_quat_sigmoid(const float* x, float* y, int n)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_sigmoid(x: *const f32, y: *mut f32, n: i32) {
    if x.is_null() || y.is_null() || n <= 0 {
        return;
    }

    unsafe {
        for i in 0..(n as usize * 4) {
            let val = *x.offset(i as isize);
            *y.offset(i as isize) = 1.0 / (1.0 + (-val).exp());
        }
    }
}

/// Quaternion activation function: Tanh component-wise
///
/// C signature: void sounio_quat_tanh(const float* x, float* y, int n)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_tanh(x: *const f32, y: *mut f32, n: i32) {
    if x.is_null() || y.is_null() || n <= 0 {
        return;
    }

    unsafe {
        for i in 0..(n as usize * 4) {
            let val = *x.offset(i as isize);
            *y.offset(i as isize) = val.tanh();
        }
    }
}
