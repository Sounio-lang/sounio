//! Integration tests for native tensor runtime functions
//!
//! Tests matrix multiplication, matrix-vector multiplication, transpose,
//! and other tensor operations using the C-compatible runtime functions.

use sounio::backend::native::tensor_runtime::*;

/// Test matrix multiplication: C = A @ B
#[test]
fn test_tensor_matmul() {
    // A: 2×3 matrix
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // B: 3×2 matrix
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    // C: 2×2 matrix (result)
    let mut c = vec![0.0; 4];

    unsafe {
        sounio_tensor_matmul(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            2, // M
            3, // K
            2, // N
        );
    }

    // Expected: C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    //           C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    //           C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    //           C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    assert!((c[0] - 58.0).abs() < 1e-10);
    assert!((c[1] - 64.0).abs() < 1e-10);
    assert!((c[2] - 139.0).abs() < 1e-10);
    assert!((c[3] - 154.0).abs() < 1e-10);
}

/// Test matrix-vector multiplication: y = A @ x
#[test]
fn test_tensor_matvec() {
    // A: 2×3 matrix
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // x: 3-element vector
    let x = vec![2.0, 3.0, 4.0];
    // y: 2-element vector (result)
    let mut y = vec![0.0; 2];

    unsafe {
        sounio_tensor_matvec(
            a.as_ptr(),
            x.as_ptr(),
            y.as_mut_ptr(),
            2, // M
            3, // N
        );
    }

    // Expected: y[0] = 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    //           y[1] = 4*2 + 5*3 + 6*4 = 8 + 15 + 24 = 47
    assert!((y[0] - 20.0).abs() < 1e-10);
    assert!((y[1] - 47.0).abs() < 1e-10);
}

/// Test matrix transpose: B = A^T
#[test]
fn test_tensor_transpose() {
    // A: 2×3 matrix
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // B: 3×2 matrix (result)
    let mut b = vec![0.0; 6];

    unsafe {
        sounio_tensor_transpose(a.as_ptr(), b.as_mut_ptr(), 2, 3);
    }

    // Expected: B = A^T
    // B[0,0] = A[0,0] = 1
    // B[0,1] = A[1,0] = 4
    // B[1,0] = A[0,1] = 2
    // B[1,1] = A[1,1] = 5
    // B[2,0] = A[0,2] = 3
    // B[2,1] = A[1,2] = 6
    assert!((b[0] - 1.0).abs() < 1e-10); // B[0,0]
    assert!((b[1] - 4.0).abs() < 1e-10); // B[0,1]
    assert!((b[2] - 2.0).abs() < 1e-10); // B[1,0]
    assert!((b[3] - 5.0).abs() < 1e-10); // B[1,1]
    assert!((b[4] - 3.0).abs() < 1e-10); // B[2,0]
    assert!((b[5] - 6.0).abs() < 1e-10); // B[2,1]
}

/// Test element-wise addition: C = A + B
#[test]
fn test_tensor_add() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];

    unsafe {
        sounio_tensor_add(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 4);
    }

    assert!((c[0] - 6.0).abs() < 1e-10);
    assert!((c[1] - 8.0).abs() < 1e-10);
    assert!((c[2] - 10.0).abs() < 1e-10);
    assert!((c[3] - 12.0).abs() < 1e-10);
}

/// Test element-wise multiplication: C = A * B
#[test]
fn test_tensor_mul() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let mut c = vec![0.0; 4];

    unsafe {
        sounio_tensor_mul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 4);
    }

    assert!((c[0] - 2.0).abs() < 1e-10);
    assert!((c[1] - 6.0).abs() < 1e-10);
    assert!((c[2] - 12.0).abs() < 1e-10);
    assert!((c[3] - 20.0).abs() < 1e-10);
}

/// Test tensor scaling: B = alpha * A
#[test]
fn test_tensor_scale() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let alpha = 2.5;
    let mut b = vec![0.0; 4];

    unsafe {
        sounio_tensor_scale(a.as_ptr(), alpha, b.as_mut_ptr(), 4);
    }

    assert!((b[0] - 2.5).abs() < 1e-10);
    assert!((b[1] - 5.0).abs() < 1e-10);
    assert!((b[2] - 7.5).abs() < 1e-10);
    assert!((b[3] - 10.0).abs() < 1e-10);
}

/// Test tensor reshape (just copies data)
#[test]
fn test_tensor_reshape() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![0.0; 6];
    let input_shape = [2, 3];
    let output_shape = [3, 2];

    unsafe {
        sounio_tensor_reshape(
            input.as_ptr(),
            output.as_mut_ptr(),
            input_shape.as_ptr(),
            2, // input_rank
            output_shape.as_ptr(),
            2, // output_rank
            6, // total_elements
        );
    }

    // Reshape just copies data (row-major)
    for i in 0..6 {
        assert!((output[i] - input[i]).abs() < 1e-10);
    }
}

/// Test identity matrix multiplication
#[test]
fn test_matmul_identity() {
    // Identity matrix 3×3
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut result = vec![0.0; 9];

    unsafe {
        sounio_tensor_matmul(
            identity.as_ptr(),
            a.as_ptr(),
            result.as_mut_ptr(),
            3, // M
            3, // K
            3, // N
        );
    }

    // I @ A = A
    for i in 0..9 {
        assert!((result[i] - a[i]).abs() < 1e-10);
    }
}

/// Test zero matrix multiplication
#[test]
fn test_matmul_zero() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let zero = vec![0.0, 0.0, 0.0, 0.0];
    let mut result = vec![1.0; 4]; // Initialize to non-zero

    unsafe {
        sounio_tensor_matmul(
            a.as_ptr(),
            zero.as_ptr(),
            result.as_mut_ptr(),
            2, // M
            2, // K
            2, // N
        );
    }

    // A @ 0 = 0
    for i in 0..4 {
        assert!((result[i] - 0.0).abs() < 1e-10);
    }
}

/// Test SIMD-optimized element-wise addition
#[test]
fn test_tensor_add_simd() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let mut c = vec![0.0; 5];

    unsafe {
        sounio_tensor_add(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 5);
    }

    assert_eq!(c, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
}

/// Test SIMD-optimized element-wise multiplication
#[test]
fn test_tensor_mul_simd() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let mut c = vec![0.0; 4];

    unsafe {
        sounio_tensor_mul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 4);
    }

    assert_eq!(c, vec![2.0, 6.0, 12.0, 20.0]);
}

/// Test SIMD-optimized tensor scaling
#[test]
fn test_tensor_scale_simd() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let alpha = 2.5;
    let mut b = vec![0.0; 5];

    unsafe {
        sounio_tensor_scale(a.as_ptr(), alpha, b.as_mut_ptr(), 5);
    }

    assert_eq!(b, vec![2.5, 5.0, 7.5, 10.0, 12.5]);
}

/// Test SIMD operations with odd-sized arrays (remainder handling)
#[test]
fn test_tensor_simd_odd_size() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![10.0, 20.0, 30.0];
    let mut c = vec![0.0; 3];

    unsafe {
        sounio_tensor_add(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3);
    }

    assert_eq!(c, vec![11.0, 22.0, 33.0]);
}

/// Test SIMD operations with single-element arrays (scalar fallback)
#[test]
fn test_tensor_simd_single_element() {
    let a = vec![5.0];
    let b = vec![10.0];
    let mut c = vec![0.0];

    unsafe {
        sounio_tensor_add(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1);
    }

    assert_eq!(c, vec![15.0]);
}
