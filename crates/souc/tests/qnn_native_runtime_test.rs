//! Integration tests for quaternion native runtime
//!
//! Tests the performance and correctness of quaternion operations in the native backend,
//! including verification of the Hamilton product, vector rotations, and interpolations.

#[cfg(test)]
mod quat_runtime_tests {
    use std::f32::consts::PI;

    // Import the runtime functions directly from the library
    use sounio::backend::native::quat_runtime::{
        sounio_quat_batch_conj, sounio_quat_batch_mul, sounio_quat_batch_rotate, sounio_quat_conj,
        sounio_quat_inverse, sounio_quat_lerp, sounio_quat_linear_fwd, sounio_quat_mul,
        sounio_quat_norm, sounio_quat_norm_sq, sounio_quat_normalize, sounio_quat_relu,
        sounio_quat_rotate, sounio_quat_sigmoid, sounio_quat_slerp, sounio_quat_tanh,
    };

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_slice(a: &[f32], b: &[f32]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < EPSILON)
    }

    #[test]
    fn test_quat_identity() {
        // Identity quaternion: (1, 0, 0, 0)
        let q = [1.0, 0.0, 0.0, 0.0];
        let mut result = [0.0; 4];

        unsafe {
            sounio_quat_norm(q.as_ptr());
            sounio_quat_conj(q.as_ptr(), result.as_mut_ptr());
        }

        assert!(approx_eq_slice(&result, &[1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_quat_conjugate() {
        // Test conjugate: (w, x, y, z) → (w, -x, -y, -z)
        let q = [1.0, 2.0, 3.0, 4.0];
        let mut result = [0.0; 4];

        unsafe {
            sounio_quat_conj(q.as_ptr(), result.as_mut_ptr());
        }

        let expected = [1.0, -2.0, -3.0, -4.0];
        assert!(approx_eq_slice(&result, &expected));
    }

    #[test]
    fn test_quat_norm() {
        // Test norm: sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let q = [1.0, 2.0, 3.0, 4.0];
        let norm = unsafe { sounio_quat_norm(q.as_ptr()) };

        let expected = 30.0_f32.sqrt();
        assert!(approx_eq(norm, expected));
    }

    #[test]
    fn test_quat_norm_sq() {
        // Test norm squared: 1 + 4 + 9 + 16 = 30
        let q = [1.0, 2.0, 3.0, 4.0];
        let norm_sq = unsafe { sounio_quat_norm_sq(q.as_ptr()) };

        let expected = 30.0;
        assert!(approx_eq(norm_sq, expected));
    }

    #[test]
    fn test_quat_normalize() {
        // Test normalization
        let mut q = [2.0, 0.0, 0.0, 0.0];
        unsafe {
            sounio_quat_normalize(q.as_mut_ptr());
        }

        let expected = [1.0, 0.0, 0.0, 0.0];
        assert!(approx_eq_slice(&q, &expected));
    }

    #[test]
    fn test_quat_inverse() {
        // Test inverse: for unit quaternion, inverse = conjugate
        let q = [1.0, 0.0, 0.0, 0.0]; // Unit quaternion (identity)
        let mut result = [0.0; 4];

        unsafe {
            sounio_quat_inverse(q.as_ptr(), result.as_mut_ptr());
        }

        assert!(approx_eq_slice(&result, &[1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_quat_hamilton_product() {
        // Test Hamilton product: identity * q = q
        let identity = [1.0, 0.0, 0.0, 0.0];
        let q = [0.7071, 0.7071, 0.0, 0.0]; // 90° rotation about x-axis
        let mut result = [0.0; 4];

        unsafe {
            sounio_quat_mul(identity.as_ptr(), q.as_ptr(), result.as_mut_ptr());
        }

        assert!(approx_eq_slice(&result, &q));
    }

    #[test]
    fn test_quat_hamilton_noncommutative() {
        // Test non-commutativity: q1 ⊗ q2 ≠ q2 ⊗ q1
        let q1 = [1.0, 1.0, 0.0, 0.0];
        let q2 = [0.0, 0.0, 1.0, 1.0];
        let mut result1 = [0.0; 4];
        let mut result2 = [0.0; 4];

        unsafe {
            sounio_quat_mul(q1.as_ptr(), q2.as_ptr(), result1.as_mut_ptr());
            sounio_quat_mul(q2.as_ptr(), q1.as_ptr(), result2.as_mut_ptr());
        }

        // They should NOT be equal
        assert!(!approx_eq_slice(&result1, &result2));
    }

    #[test]
    fn test_quat_rotate_identity() {
        // Rotating a vector with identity quaternion should return the same vector
        let q_identity = [1.0, 0.0, 0.0, 0.0];
        let v = [1.0, 0.0, 0.0];
        let mut result = [0.0; 3];

        unsafe {
            sounio_quat_rotate(q_identity.as_ptr(), v.as_ptr(), result.as_mut_ptr());
        }

        assert!(approx_eq_slice(&result, &v));
    }

    #[test]
    fn test_quat_rotate_90deg_x() {
        // Rotate vector (0, 1, 0) by 90° about x-axis
        // Result should be approximately (0, 0, 1)
        let cos_45 = (PI / 4.0).cos();
        let sin_45 = (PI / 4.0).sin();
        let q = [cos_45, sin_45, 0.0, 0.0]; // 90° rotation about x
        let v = [0.0, 1.0, 0.0];
        let mut result = [0.0; 3];

        unsafe {
            sounio_quat_rotate(q.as_ptr(), v.as_ptr(), result.as_mut_ptr());
        }

        // Expected: roughly (0, 0, 1) after 90° rotation
        assert!(result[0].abs() < EPSILON);
        assert!(result[2].abs() - 1.0 < EPSILON);
    }

    #[test]
    fn test_quat_lerp() {
        // Linear interpolation at t=0 should give q1
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.0, 1.0, 0.0, 0.0];
        let mut result_t0 = [0.0; 4];
        let mut result_t1 = [0.0; 4];

        unsafe {
            sounio_quat_lerp(q1.as_ptr(), q2.as_ptr(), 0.0, result_t0.as_mut_ptr());
            sounio_quat_lerp(q1.as_ptr(), q2.as_ptr(), 1.0, result_t1.as_mut_ptr());
        }

        assert!(approx_eq_slice(&result_t0, &q1));
        assert!(approx_eq_slice(&result_t1, &q2));
    }

    #[test]
    fn test_quat_slerp() {
        // SLERP at t=0 should give q1
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.0, 1.0, 0.0, 0.0];
        let mut result_t0 = [0.0; 4];
        let mut result_t1 = [0.0; 4];

        unsafe {
            sounio_quat_slerp(q1.as_ptr(), q2.as_ptr(), 0.0, result_t0.as_mut_ptr());
            sounio_quat_slerp(q1.as_ptr(), q2.as_ptr(), 1.0, result_t1.as_mut_ptr());
        }

        assert!(approx_eq_slice(&result_t0, &q1));
        assert!(approx_eq_slice(&result_t1, &q2));
    }

    #[test]
    fn test_quat_batch_conj() {
        // Test batch conjugate operation
        let qs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 quaternions
        let mut results = [0.0; 8];

        unsafe {
            sounio_quat_batch_conj(qs.as_ptr(), results.as_mut_ptr(), 2);
        }

        let expected = [1.0, -2.0, -3.0, -4.0, 5.0, -6.0, -7.0, -8.0];
        assert!(approx_eq_slice(&results, &expected));
    }

    #[test]
    fn test_quat_batch_mul() {
        // Test batch multiplication
        let identity = [1.0, 0.0, 0.0, 0.0];
        let q1 = [0.7071, 0.7071, 0.0, 0.0];
        let q1s = [
            identity[0],
            identity[1],
            identity[2],
            identity[3],
            q1[0],
            q1[1],
            q1[2],
            q1[3],
        ];
        let q2s = [
            q1[0],
            q1[1],
            q1[2],
            q1[3],
            identity[0],
            identity[1],
            identity[2],
            identity[3],
        ];
        let mut results = [0.0; 8];

        unsafe {
            sounio_quat_batch_mul(q1s.as_ptr(), q2s.as_ptr(), results.as_mut_ptr(), 2);
        }

        // identity * q1 = q1, and q1 * identity = q1
        assert!(approx_eq_slice(&results[0..4], &q1));
        assert!(approx_eq_slice(&results[4..8], &q1));
    }

    #[test]
    fn test_quat_batch_rotate() {
        // Test batch rotation
        let q_identity = [1.0, 0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let qs = [
            q_identity[0],
            q_identity[1],
            q_identity[2],
            q_identity[3],
            q_identity[0],
            q_identity[1],
            q_identity[2],
            q_identity[3],
        ];
        let vs = [v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]];
        let mut results = [0.0; 6];

        unsafe {
            sounio_quat_batch_rotate(qs.as_ptr(), vs.as_ptr(), results.as_mut_ptr(), 2);
        }

        // Identity rotation should preserve vectors
        assert!(approx_eq_slice(&results[0..3], &v1));
        assert!(approx_eq_slice(&results[3..6], &v2));
    }

    #[test]
    fn test_quat_linear_fwd() {
        // Simple linear layer test: 4 input features → 4 output features
        // (this is a scalar linear layer, not quaternion-specific multiplication)
        let x = [1.0, 0.0, 0.0, 0.0]; // Input vector
        let w = [
            1.0, 0.0, 0.0, 0.0, // Weight matrix (4x4 identity)
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = [0.1, 0.2, 0.3, 0.4]; // Bias
        let mut y = [0.0; 4];

        sounio_quat_linear_fwd(
            x.as_ptr(),
            w.as_ptr(),
            b.as_ptr(),
            y.as_mut_ptr(),
            1, // batch
            4, // in_features (4 scalar elements)
            4, // out_features (4 scalar elements)
        );

        // y = W @ x + b = identity @ [1,0,0,0] + [0.1,0.2,0.3,0.4]
        //               = [1,0,0,0] + [0.1,0.2,0.3,0.4] = [1.1, 0.2, 0.3, 0.4]
        let expected = [1.1, 0.2, 0.3, 0.4];
        assert!(approx_eq_slice(&y, &expected));
    }

    #[test]
    fn test_quat_relu() {
        // ReLU: max(0, x) applied component-wise
        let x = [1.0, -0.5, 0.0, -2.0];
        let mut y = [0.0; 4];

        unsafe {
            sounio_quat_relu(x.as_ptr(), y.as_mut_ptr(), 1);
        }

        let expected = [1.0, 0.0, 0.0, 0.0];
        assert!(approx_eq_slice(&y, &expected));
    }

    #[test]
    fn test_quat_sigmoid() {
        // Sigmoid: 1 / (1 + exp(-x))
        // At x=0, sigmoid(0) ≈ 0.5
        let x = [0.0, 0.0, 0.0, 0.0];
        let mut y = [0.0; 4];

        unsafe {
            sounio_quat_sigmoid(x.as_ptr(), y.as_mut_ptr(), 1);
        }

        // Each component should be approximately 0.5
        for val in &y {
            assert!(approx_eq(*val, 0.5));
        }
    }

    #[test]
    fn test_quat_tanh() {
        // Tanh: at x=0, tanh(0) = 0
        let x = [0.0, 0.0, 0.0, 0.0];
        let mut y = [0.0; 4];

        unsafe {
            sounio_quat_tanh(x.as_ptr(), y.as_mut_ptr(), 1);
        }

        for val in &y {
            assert!(approx_eq(*val, 0.0));
        }
    }
}
