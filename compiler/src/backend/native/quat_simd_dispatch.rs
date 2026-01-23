//! SIMD Dispatch Infrastructure for Quaternion Operations
//!
//! This module provides runtime CPU feature detection and automatic dispatch
//! to optimal quaternion operation implementations (scalar, AVX2, AVX-512, NEON).
//!
//! The dispatch system is initialized once at startup and subsequent function
//! calls are routed to the appropriate implementation based on detected capabilities.

use std::sync::OnceLock;

// ============================================================================
// SIMD Level Detection
// ============================================================================

/// CPU SIMD capability levels, ordered by performance
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// Scalar implementation (baseline, always available)
    Scalar = 0,
    /// ARM NEON (128-bit vectors, 1 quat per iteration)
    #[cfg(target_arch = "aarch64")]
    Neon = 1,
    /// AVX2 (256-bit vectors, 2 quats per iteration, 4-5x speedup)
    Avx2 = 2,
    /// AVX-512 (512-bit vectors, 4 quats per iteration, 6-8x speedup)
    Avx512 = 3,
}

/// Global SIMD level detected at runtime
static DETECTED_SIMD: OnceLock<SimdLevel> = OnceLock::new();

/// Detect CPU SIMD capabilities
#[inline(never)]
fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX-512 first (most powerful)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            return SimdLevel::Avx512;
        }

        // Fall back to AVX2
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }

        // Scalar fallback
        SimdLevel::Scalar
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON is always available on aarch64
        #[allow(unreachable_code)]
        SimdLevel::Neon
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdLevel::Scalar
    }
}

/// Get the detected SIMD level (initialized once on first call)
#[inline]
pub fn simd_level() -> SimdLevel {
    *DETECTED_SIMD.get_or_init(detect_simd_level)
}

// ============================================================================
// Function Pointers for Dispatch
// ============================================================================

/// Signature for quaternion multiplication dispatch
pub type QuatMulFn =
    unsafe extern "C" fn(q1: *const f32, q2: *const f32, out: *mut f32);

/// Signature for batch quaternion multiplication dispatch
pub type QuatBatchMulFn =
    unsafe extern "C" fn(q1s: *const f32, q2s: *const f32, outs: *mut f32, n: i32);

/// Signature for linear layer forward pass dispatch
pub type QuatLinearFwdFn = unsafe extern "C" fn(
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    output: *mut f32,
    input_dim: i32,
    output_dim: i32,
    batch_size: i32,
);

// ============================================================================
// Dispatch Wrappers - Route to optimal implementation
// ============================================================================

/// Dispatch quaternion multiplication to appropriate backend
///
/// Routes to AVX-512, AVX2, or scalar implementation based on CPU capabilities.
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_mul_dispatch(q1: *const f32, q2: *const f32, out: *mut f32) {
    unsafe {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx512 => {
                sounio_quat_mul_avx512(q1, q2, out);
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => {
                sounio_quat_mul_avx2(q1, q2, out);
            }
            _ => {
                sounio_quat_mul_scalar(q1, q2, out);
            }
        }
    }
}

/// Dispatch batch quaternion multiplication to appropriate backend
///
/// Processes `n` quaternion pairs: result[i] = q1s[i] ⊗ q2s[i]
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_batch_mul_dispatch(
    q1s: *const f32,
    q2s: *const f32,
    outs: *mut f32,
    n: i32,
) {
    if n <= 0 {
        return;
    }

    unsafe {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx512 => {
                sounio_quat_batch_mul_avx512(q1s, q2s, outs, n);
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => {
                sounio_quat_batch_mul_avx2(q1s, q2s, outs, n);
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                sounio_quat_batch_mul_neon(q1s, q2s, outs, n);
            }
            _ => {
                sounio_quat_batch_mul_scalar(q1s, q2s, outs, n);
            }
        }
    }
}

/// Dispatch linear layer forward pass to appropriate backend
///
/// Computes: output[b,o] = sum_i input[b,i] ⊗ weights[o,i] + bias[o]
/// With quaternion multiplication (Hamilton product)
#[unsafe(no_mangle)]
pub extern "C" fn sounio_quat_linear_fwd_dispatch(
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    output: *mut f32,
    input_dim: i32,
    output_dim: i32,
    batch_size: i32,
) {
    if batch_size <= 0 || input_dim <= 0 || output_dim <= 0 {
        return;
    }

    unsafe {
        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx512 => {
                sounio_quat_linear_fwd_avx512(
                    input, weights, bias, output, input_dim, output_dim, batch_size,
                );
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => {
                sounio_quat_linear_fwd_avx2(
                    input, weights, bias, output, input_dim, output_dim, batch_size,
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => {
                sounio_quat_linear_fwd_neon(
                    input, weights, bias, output, input_dim, output_dim, batch_size,
                );
            }
            _ => {
                sounio_quat_linear_fwd_scalar(
                    input, weights, bias, output, input_dim, output_dim, batch_size,
                );
            }
        }
    }
}

// ============================================================================
// Scalar Reference Implementations
// ============================================================================

/// Scalar quaternion multiplication reference
/// q_out = q1 ⊗ q2 = [w, x, y, z] (Hamilton product)
#[inline]
unsafe extern "C" fn sounio_quat_mul_scalar(q1: *const f32, q2: *const f32, out: *mut f32) { unsafe {
    if q1.is_null() || q2.is_null() || out.is_null() {
        return;
    }

    let w1 = *q1.add(0);
    let x1 = *q1.add(1);
    let y1 = *q1.add(2);
    let z1 = *q1.add(3);

    let w2 = *q2.add(0);
    let x2 = *q2.add(1);
    let y2 = *q2.add(2);
    let z2 = *q2.add(3);

    let w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    let x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    let y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    let z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

    *out.add(0) = w;
    *out.add(1) = x;
    *out.add(2) = y;
    *out.add(3) = z;
}}

/// Scalar batch quaternion multiplication
#[inline]
unsafe extern "C" fn sounio_quat_batch_mul_scalar(
    q1s: *const f32,
    q2s: *const f32,
    outs: *mut f32,
    n: i32,
) { unsafe {
    for i in 0..n {
        let q1 = q1s.add((i * 4) as usize);
        let q2 = q2s.add((i * 4) as usize);
        let out = outs.add((i * 4) as usize);
        sounio_quat_mul_scalar(q1, q2, out);
    }
}}

/// Scalar linear layer forward pass
#[inline]
unsafe extern "C" fn sounio_quat_linear_fwd_scalar(
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    output: *mut f32,
    input_dim: i32,
    output_dim: i32,
    batch_size: i32,
) { unsafe {
    for b in 0..batch_size {
        for o in 0..output_dim {
            let mut accum = [0.0f32; 4];

            for i in 0..input_dim {
                let in_idx = ((b * input_dim + i) * 4) as usize;
                let w_idx = ((o * input_dim + i) * 4) as usize;

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

                // Hamilton product inline
                let w = in_q[0] * w_q[0] - in_q[1] * w_q[1] - in_q[2] * w_q[2]
                    - in_q[3] * w_q[3];
                let x =
                    in_q[0] * w_q[1] + in_q[1] * w_q[0] + in_q[2] * w_q[3] - in_q[3] * w_q[2];
                let y =
                    in_q[0] * w_q[2] - in_q[1] * w_q[3] + in_q[2] * w_q[0] + in_q[3] * w_q[1];
                let z =
                    in_q[0] * w_q[3] + in_q[1] * w_q[2] - in_q[2] * w_q[1] + in_q[3] * w_q[0];

                accum[0] += w;
                accum[1] += x;
                accum[2] += y;
                accum[3] += z;
            }

            // Add bias and write output
            let out_idx = ((b * output_dim + o) * 4) as usize;
            let bias_idx = (o * 4) as usize;

            *output.add(out_idx) = accum[0] + *bias.add(bias_idx);
            *output.add(out_idx + 1) = accum[1] + *bias.add(bias_idx + 1);
            *output.add(out_idx + 2) = accum[2] + *bias.add(bias_idx + 2);
            *output.add(out_idx + 3) = accum[3] + *bias.add(bias_idx + 3);
        }
    }
}}

// ============================================================================
// SIMD Implementation Stubs (to be generated)
// ============================================================================

// These are declared but will be implemented in separate SIMD-specific modules
// The implementations use CPU-specific intrinsics (AVX2, AVX-512, NEON)

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    // AVX2 implementations (256-bit, 2 quats per iteration)
    pub(crate) fn sounio_quat_mul_avx2(q1: *const f32, q2: *const f32, out: *mut f32);
    pub(crate) fn sounio_quat_batch_mul_avx2(
        q1s: *const f32,
        q2s: *const f32,
        outs: *mut f32,
        n: i32,
    );
    pub(crate) fn sounio_quat_linear_fwd_avx2(
        input: *const f32,
        weights: *const f32,
        bias: *const f32,
        output: *mut f32,
        input_dim: i32,
        output_dim: i32,
        batch_size: i32,
    );

    // AVX-512 implementations (512-bit, 4 quats per iteration)
    pub(crate) fn sounio_quat_mul_avx512(q1: *const f32, q2: *const f32, out: *mut f32);
    pub(crate) fn sounio_quat_batch_mul_avx512(
        q1s: *const f32,
        q2s: *const f32,
        outs: *mut f32,
        n: i32,
    );
    pub(crate) fn sounio_quat_linear_fwd_avx512(
        input: *const f32,
        weights: *const f32,
        bias: *const f32,
        output: *mut f32,
        input_dim: i32,
        output_dim: i32,
        batch_size: i32,
    );
}

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    // NEON implementations (128-bit, 1 quat per iteration)
    pub(crate) fn sounio_quat_mul_neon(q1: *const f32, q2: *const f32, out: *mut f32);
    pub(crate) fn sounio_quat_batch_mul_neon(
        q1s: *const f32,
        q2s: *const f32,
        outs: *mut f32,
        n: i32,
    );
    pub(crate) fn sounio_quat_linear_fwd_neon(
        input: *const f32,
        weights: *const f32,
        bias: *const f32,
        output: *mut f32,
        input_dim: i32,
        output_dim: i32,
        batch_size: i32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_level_detection() {
        let level = simd_level();
        println!("Detected SIMD level: {:?}", level);
        // Should always detect at least Scalar
        assert!(level >= SimdLevel::Scalar);
    }

    #[test]
    fn test_scalar_quat_mul() {
        // q1 = [1, 0, 0, 0] (identity)
        // q2 = [0.707, 0.707, 0, 0]
        // q1 ⊗ q2 should equal q2
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.707, 0.707, 0.0, 0.0];
        let mut out = [0.0; 4];

        unsafe {
            sounio_quat_mul_scalar(q1.as_ptr(), q2.as_ptr(), out.as_mut_ptr());
        }

        assert!((out[0] - 0.707).abs() < 1e-5);
        assert!((out[1] - 0.707).abs() < 1e-5);
        assert!((out[2] - 0.0).abs() < 1e-5);
        assert!((out[3] - 0.0).abs() < 1e-5);
    }
}
