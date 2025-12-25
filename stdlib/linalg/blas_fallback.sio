// blas_fallback.d - Pure D Fallback Implementations for BLAS
//
// These implementations are used when native BLAS libraries are not available.
// They provide correct (but potentially slower) alternatives for all BLAS operations.
//
// Performance notes:
// - Native BLAS (OpenBLAS, MKL, ATLAS) is 5-50x faster for large matrices
// - These fallbacks are suitable for:
//   - Small matrices (n < 100)
//   - Development/testing without BLAS
//   - Embedded systems without BLAS
//
// The module system will automatically use these when BLAS is not linked.

module linalg::blas_fallback

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = x
    for _ in 0..20 {
        y = 0.5 * (y + x / y)
    }
    return y
}

// ============================================================================
// LEVEL 1 BLAS: VECTOR-VECTOR OPERATIONS
// ============================================================================

/// DAXPY: y = alpha * x + y
/// Computes y := alpha * x + y where x and y are vectors
pub fn daxpy_fallback(n: i32, alpha: f64, x: &[f64], incx: i32, y: &![f64], incy: i32) {
    if alpha == 0.0 { return }

    let mut ix = 0
    let mut iy = 0

    for _ in 0..n {
        y[iy as usize] = y[iy as usize] + alpha * x[ix as usize]
        ix = ix + incx
        iy = iy + incy
    }
}

/// DDOT: dot product of two vectors
/// Returns x . y = sum(x_i * y_i)
pub fn ddot_fallback(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    let mut result = 0.0
    let mut ix = 0
    let mut iy = 0

    for _ in 0..n {
        result = result + x[ix as usize] * y[iy as usize]
        ix = ix + incx
        iy = iy + incy
    }

    return result
}

/// DNRM2: Euclidean norm of a vector
/// Returns ||x||_2 = sqrt(sum(x_i^2))
pub fn dnrm2_fallback(n: i32, x: &[f64], incx: i32) -> f64 {
    // Use Kahan summation for better accuracy
    let mut sum = 0.0
    let mut scale = 0.0
    let mut ix = 0

    for _ in 0..n {
        let val = abs_f64(x[ix as usize])
        if val > 0.0 {
            if scale < val {
                let ratio = scale / val
                sum = 1.0 + sum * ratio * ratio
                scale = val
            } else {
                let ratio = val / scale
                sum = sum + ratio * ratio
            }
        }
        ix = ix + incx
    }

    return scale * sqrt_f64(sum)
}

/// DSCAL: scale a vector
/// Computes x := alpha * x
pub fn dscal_fallback(n: i32, alpha: f64, x: &![f64], incx: i32) {
    let mut ix = 0

    for _ in 0..n {
        x[ix as usize] = alpha * x[ix as usize]
        ix = ix + incx
    }
}

/// DASUM: sum of absolute values
/// Returns ||x||_1 = sum(|x_i|)
pub fn dasum_fallback(n: i32, x: &[f64], incx: i32) -> f64 {
    let mut sum = 0.0
    let mut ix = 0

    for _ in 0..n {
        sum = sum + abs_f64(x[ix as usize])
        ix = ix + incx
    }

    return sum
}

/// IDAMAX: index of maximum absolute value
/// Returns argmax_i |x_i|
pub fn idamax_fallback(n: i32, x: &[f64], incx: i32) -> i32 {
    if n <= 0 { return 0 }

    let mut max_idx = 0
    let mut max_val = abs_f64(x[0])
    let mut ix = incx

    for i in 1..n {
        let val = abs_f64(x[ix as usize])
        if val > max_val {
            max_val = val
            max_idx = i
        }
        ix = ix + incx
    }

    return max_idx
}

/// DCOPY: copy a vector
/// Computes y := x
pub fn dcopy_fallback(n: i32, x: &[f64], incx: i32, y: &![f64], incy: i32) {
    let mut ix = 0
    let mut iy = 0

    for _ in 0..n {
        y[iy as usize] = x[ix as usize]
        ix = ix + incx
        iy = iy + incy
    }
}

/// DSWAP: swap two vectors
/// Computes x <-> y
pub fn dswap_fallback(n: i32, x: &![f64], incx: i32, y: &![f64], incy: i32) {
    let mut ix = 0
    let mut iy = 0

    for _ in 0..n {
        let temp = x[ix as usize]
        x[ix as usize] = y[iy as usize]
        y[iy as usize] = temp
        ix = ix + incx
        iy = iy + incy
    }
}

/// DROT: apply Givens rotation
pub fn drot_fallback(n: i32, x: &![f64], incx: i32, y: &![f64], incy: i32, c: f64, s: f64) {
    let mut ix = 0
    let mut iy = 0

    for _ in 0..n {
        let temp = c * x[ix as usize] + s * y[iy as usize]
        y[iy as usize] = c * y[iy as usize] - s * x[ix as usize]
        x[ix as usize] = temp
        ix = ix + incx
        iy = iy + incy
    }
}

// ============================================================================
// LEVEL 2 BLAS: MATRIX-VECTOR OPERATIONS
// ============================================================================

/// DGEMV: matrix-vector multiply
/// y := alpha * op(A) * x + beta * y
/// where op(A) = A or A^T depending on trans
pub fn dgemv_fallback(
    order: i32,       // 0 = row-major, 1 = col-major
    trans: i32,       // 0 = no trans, 1 = trans
    m: i32,           // rows of A
    n: i32,           // cols of A
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &![f64],
    incy: i32
) {
    // Determine effective dimensions based on transpose
    let nrow = if trans == 0 { m } else { n }
    let ncol = if trans == 0 { n } else { m }

    // Scale y by beta
    if beta != 1.0 {
        let mut iy = 0
        for _ in 0..nrow {
            y[iy as usize] = beta * y[iy as usize]
            iy = iy + incy
        }
    }

    if alpha == 0.0 { return }

    // Perform matrix-vector multiply
    let mut iy = 0
    for i in 0..nrow {
        let mut sum = 0.0
        let mut ix = 0

        for j in 0..ncol {
            // Get A[i,j] based on storage order and transpose
            let a_ij = if order == 1 {  // col-major
                if trans == 0 {
                    a[(j * lda + i) as usize]  // A[i,j]
                } else {
                    a[(i * lda + j) as usize]  // A[j,i]
                }
            } else {  // row-major
                if trans == 0 {
                    a[(i * lda + j) as usize]  // A[i,j]
                } else {
                    a[(j * lda + i) as usize]  // A[j,i]
                }
            }

            sum = sum + a_ij * x[ix as usize]
            ix = ix + incx
        }

        y[iy as usize] = y[iy as usize] + alpha * sum
        iy = iy + incy
    }
}

/// DSYMV: symmetric matrix-vector multiply
/// y := alpha * A * x + beta * y
/// where A is symmetric (only upper or lower triangle stored)
pub fn dsymv_fallback(
    order: i32,       // 0 = row-major, 1 = col-major
    uplo: i32,        // 0 = upper, 1 = lower
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &![f64],
    incy: i32
) {
    // Scale y by beta
    if beta != 1.0 {
        let mut iy = 0
        for _ in 0..n {
            y[iy as usize] = beta * y[iy as usize]
            iy = iy + incy
        }
    }

    if alpha == 0.0 { return }

    // Perform symmetric matrix-vector multiply
    let mut iy = 0
    for i in 0..n {
        let mut sum = 0.0
        let mut ix = 0

        for j in 0..n {
            // Get A[i,j] from symmetric matrix
            let row = if uplo == 0 { if i <= j { i } else { j } } else { if i >= j { i } else { j } }
            let col = if uplo == 0 { if i <= j { j } else { i } } else { if i >= j { j } else { i } }

            let a_ij = if order == 1 {  // col-major
                a[(col * lda + row) as usize]
            } else {  // row-major
                a[(row * lda + col) as usize]
            }

            sum = sum + a_ij * x[ix as usize]
            ix = ix + incx
        }

        y[iy as usize] = y[iy as usize] + alpha * sum
        iy = iy + incy
    }
}

/// DTRSV: triangular solve
/// Solve op(A) * x = b where A is triangular
pub fn dtrsv_fallback(
    order: i32,       // 0 = row-major, 1 = col-major
    uplo: i32,        // 0 = upper, 1 = lower
    trans: i32,       // 0 = no trans, 1 = trans
    diag: i32,        // 0 = non-unit, 1 = unit diagonal
    n: i32,
    a: &[f64],
    lda: i32,
    x: &![f64],       // on entry: b, on exit: x
    incx: i32
) {
    // Determine if we solve forward or backward
    let forward = (uplo == 1) != (trans != 0)  // lower and no-trans, or upper and trans

    if forward {
        // Forward substitution
        let mut ix = 0
        for i in 0..n {
            let mut sum = x[ix as usize]
            let mut jx = 0

            for j in 0..i {
                let a_ij = if order == 1 {  // col-major
                    if trans == 0 {
                        a[(j * lda + i) as usize]  // A[i,j]
                    } else {
                        a[(i * lda + j) as usize]  // A[j,i]
                    }
                } else {  // row-major
                    if trans == 0 {
                        a[(i * lda + j) as usize]
                    } else {
                        a[(j * lda + i) as usize]
                    }
                }

                sum = sum - a_ij * x[jx as usize]
                jx = jx + incx
            }

            // Divide by diagonal element
            if diag == 0 {
                let a_ii = if order == 1 {
                    a[(i * lda + i) as usize]
                } else {
                    a[(i * lda + i) as usize]
                }
                sum = sum / a_ii
            }

            x[ix as usize] = sum
            ix = ix + incx
        }
    } else {
        // Backward substitution
        let mut ix = ((n - 1) * incx) as usize
        for ii in 0..n {
            let i = n - 1 - ii
            let mut sum = x[ix]
            let mut jx = ((n - 1) * incx) as usize

            for jj in 0..(n - 1 - i) {
                let j = n - 1 - jj
                let a_ij = if order == 1 {  // col-major
                    if trans == 0 {
                        a[(j * lda + i) as usize]
                    } else {
                        a[(i * lda + j) as usize]
                    }
                } else {
                    if trans == 0 {
                        a[(i * lda + j) as usize]
                    } else {
                        a[(j * lda + i) as usize]
                    }
                }

                sum = sum - a_ij * x[jx]
                jx = jx - incx as usize
            }

            // Divide by diagonal element
            if diag == 0 {
                let a_ii = a[(i * lda + i) as usize]
                sum = sum / a_ii
            }

            x[ix] = sum
            if i > 0 {
                ix = ix - incx as usize
            }
        }
    }
}

// ============================================================================
// LEVEL 3 BLAS: MATRIX-MATRIX OPERATIONS
// ============================================================================

/// DGEMM: general matrix multiply
/// C := alpha * op(A) * op(B) + beta * C
pub fn dgemm_fallback(
    order: i32,       // 0 = row-major, 1 = col-major
    transa: i32,      // 0 = no trans, 1 = trans
    transb: i32,      // 0 = no trans, 1 = trans
    m: i32,           // rows of C
    n: i32,           // cols of C
    k: i32,           // inner dimension
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &![f64],
    ldc: i32
) {
    // Scale C by beta
    for i in 0..m {
        for j in 0..n {
            let idx = if order == 1 {
                (j * ldc + i) as usize
            } else {
                (i * ldc + j) as usize
            }
            c[idx] = beta * c[idx]
        }
    }

    if alpha == 0.0 { return }

    // Perform C += alpha * op(A) * op(B)
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0

            for l in 0..k {
                // Get A[i,l] based on transpose
                let a_il = if order == 1 {  // col-major
                    if transa == 0 {
                        a[(l * lda + i) as usize]  // A[i,l]
                    } else {
                        a[(i * lda + l) as usize]  // A[l,i]
                    }
                } else {  // row-major
                    if transa == 0 {
                        a[(i * lda + l) as usize]
                    } else {
                        a[(l * lda + i) as usize]
                    }
                }

                // Get B[l,j] based on transpose
                let b_lj = if order == 1 {  // col-major
                    if transb == 0 {
                        b[(j * ldb + l) as usize]  // B[l,j]
                    } else {
                        b[(l * ldb + j) as usize]  // B[j,l]
                    }
                } else {  // row-major
                    if transb == 0 {
                        b[(l * ldb + j) as usize]
                    } else {
                        b[(j * ldb + l) as usize]
                    }
                }

                sum = sum + a_il * b_lj
            }

            let c_idx = if order == 1 {
                (j * ldc + i) as usize
            } else {
                (i * ldc + j) as usize
            }
            c[c_idx] = c[c_idx] + alpha * sum
        }
    }
}

/// DSYRK: symmetric rank-k update
/// C := alpha * A * A^T + beta * C  (or alpha * A^T * A + beta * C)
pub fn dsyrk_fallback(
    order: i32,       // 0 = row-major, 1 = col-major
    uplo: i32,        // 0 = upper, 1 = lower
    trans: i32,       // 0 = no trans (C = A*A^T), 1 = trans (C = A^T*A)
    n: i32,           // order of C
    k: i32,           // cols of A (if no-trans) or rows of A (if trans)
    alpha: f64,
    a: &[f64],
    lda: i32,
    beta: f64,
    c: &![f64],
    ldc: i32
) {
    // Scale C by beta (only the stored triangle)
    for i in 0..n {
        let j_start = if uplo == 0 { i } else { 0 }
        let j_end = if uplo == 0 { n } else { i + 1 }

        for j in j_start..j_end {
            let idx = if order == 1 {
                (j * ldc + i) as usize
            } else {
                (i * ldc + j) as usize
            }
            c[idx] = beta * c[idx]
        }
    }

    if alpha == 0.0 { return }

    // Perform C += alpha * A * A^T or alpha * A^T * A
    for i in 0..n {
        let j_start = if uplo == 0 { i } else { 0 }
        let j_end = if uplo == 0 { n } else { i + 1 }

        for j in j_start..j_end {
            let mut sum = 0.0

            for l in 0..k {
                // Get A[i,l] and A[j,l] based on transpose
                let a_il: f64
                let a_jl: f64

                if trans == 0 {
                    // C = A * A^T: A is n x k
                    a_il = if order == 1 {
                        a[(l * lda + i) as usize]
                    } else {
                        a[(i * lda + l) as usize]
                    }
                    a_jl = if order == 1 {
                        a[(l * lda + j) as usize]
                    } else {
                        a[(j * lda + l) as usize]
                    }
                } else {
                    // C = A^T * A: A is k x n
                    a_il = if order == 1 {
                        a[(i * lda + l) as usize]
                    } else {
                        a[(l * lda + i) as usize]
                    }
                    a_jl = if order == 1 {
                        a[(j * lda + l) as usize]
                    } else {
                        a[(l * lda + j) as usize]
                    }
                }

                sum = sum + a_il * a_jl
            }

            let c_idx = if order == 1 {
                (j * ldc + i) as usize
            } else {
                (i * ldc + j) as usize
            }
            c[c_idx] = c[c_idx] + alpha * sum
        }
    }
}

/// DTRSM: triangular solve with multiple right-hand sides
/// Solve op(A) * X = alpha * B  or  X * op(A) = alpha * B
pub fn dtrsm_fallback(
    order: i32,       // 0 = row-major, 1 = col-major
    side: i32,        // 0 = left (A*X=B), 1 = right (X*A=B)
    uplo: i32,        // 0 = upper, 1 = lower
    trans: i32,       // 0 = no trans, 1 = trans
    diag: i32,        // 0 = non-unit, 1 = unit diagonal
    m: i32,           // rows of B
    n: i32,           // cols of B
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &![f64],       // on entry: B, on exit: X
    ldb: i32
) {
    // Scale B by alpha
    for i in 0..m {
        for j in 0..n {
            let idx = if order == 1 {
                (j * ldb + i) as usize
            } else {
                (i * ldb + j) as usize
            }
            b[idx] = alpha * b[idx]
        }
    }

    if side == 0 {
        // Solve A*X = B (or A^T*X = B): solve for each column of B
        for j in 0..n {
            // Extract column j of B
            let forward = (uplo == 1) != (trans != 0)

            if forward {
                // Forward substitution
                for i in 0..m {
                    let mut sum = if order == 1 {
                        b[(j * ldb + i) as usize]
                    } else {
                        b[(i * ldb + j) as usize]
                    }

                    for k in 0..i {
                        let a_ik = if order == 1 {
                            if trans == 0 {
                                a[(k * lda + i) as usize]
                            } else {
                                a[(i * lda + k) as usize]
                            }
                        } else {
                            if trans == 0 {
                                a[(i * lda + k) as usize]
                            } else {
                                a[(k * lda + i) as usize]
                            }
                        }

                        let b_kj = if order == 1 {
                            b[(j * ldb + k) as usize]
                        } else {
                            b[(k * ldb + j) as usize]
                        }

                        sum = sum - a_ik * b_kj
                    }

                    if diag == 0 {
                        let a_ii = a[(i * lda + i) as usize]
                        sum = sum / a_ii
                    }

                    if order == 1 {
                        b[(j * ldb + i) as usize] = sum
                    } else {
                        b[(i * ldb + j) as usize] = sum
                    }
                }
            } else {
                // Backward substitution
                for ii in 0..m {
                    let i = m - 1 - ii
                    let mut sum = if order == 1 {
                        b[(j * ldb + i) as usize]
                    } else {
                        b[(i * ldb + j) as usize]
                    }

                    for k in (i + 1)..m {
                        let a_ik = if order == 1 {
                            if trans == 0 {
                                a[(k * lda + i) as usize]
                            } else {
                                a[(i * lda + k) as usize]
                            }
                        } else {
                            if trans == 0 {
                                a[(i * lda + k) as usize]
                            } else {
                                a[(k * lda + i) as usize]
                            }
                        }

                        let b_kj = if order == 1 {
                            b[(j * ldb + k) as usize]
                        } else {
                            b[(k * ldb + j) as usize]
                        }

                        sum = sum - a_ik * b_kj
                    }

                    if diag == 0 {
                        let a_ii = a[(i * lda + i) as usize]
                        sum = sum / a_ii
                    }

                    if order == 1 {
                        b[(j * ldb + i) as usize] = sum
                    } else {
                        b[(i * ldb + j) as usize] = sum
                    }
                }
            }
        }
    } else {
        // Solve X*A = B (or X*A^T = B): solve for each row of B
        // This is handled by solving the transposed system A^T * X^T = B^T
        // For simplicity, we iterate over rows instead of columns

        let forward = (uplo == 0) != (trans != 0)

        if forward {
            for j in 0..n {
                for i in 0..m {
                    let mut sum = if order == 1 {
                        b[(j * ldb + i) as usize]
                    } else {
                        b[(i * ldb + j) as usize]
                    }

                    for k in 0..j {
                        let a_kj = if order == 1 {
                            if trans == 0 {
                                a[(j * lda + k) as usize]
                            } else {
                                a[(k * lda + j) as usize]
                            }
                        } else {
                            if trans == 0 {
                                a[(k * lda + j) as usize]
                            } else {
                                a[(j * lda + k) as usize]
                            }
                        }

                        let b_ik = if order == 1 {
                            b[(k * ldb + i) as usize]
                        } else {
                            b[(i * ldb + k) as usize]
                        }

                        sum = sum - b_ik * a_kj
                    }

                    if diag == 0 {
                        let a_jj = a[(j * lda + j) as usize]
                        sum = sum / a_jj
                    }

                    if order == 1 {
                        b[(j * ldb + i) as usize] = sum
                    } else {
                        b[(i * ldb + j) as usize] = sum
                    }
                }
            }
        } else {
            for jj in 0..n {
                let j = n - 1 - jj
                for i in 0..m {
                    let mut sum = if order == 1 {
                        b[(j * ldb + i) as usize]
                    } else {
                        b[(i * ldb + j) as usize]
                    }

                    for k in (j + 1)..n {
                        let a_kj = if order == 1 {
                            if trans == 0 {
                                a[(j * lda + k) as usize]
                            } else {
                                a[(k * lda + j) as usize]
                            }
                        } else {
                            if trans == 0 {
                                a[(k * lda + j) as usize]
                            } else {
                                a[(j * lda + k) as usize]
                            }
                        }

                        let b_ik = if order == 1 {
                            b[(k * ldb + i) as usize]
                        } else {
                            b[(i * ldb + k) as usize]
                        }

                        sum = sum - b_ik * a_kj
                    }

                    if diag == 0 {
                        let a_jj = a[(j * lda + j) as usize]
                        sum = sum / a_jj
                    }

                    if order == 1 {
                        b[(j * ldb + i) as usize] = sum
                    } else {
                        b[(i * ldb + j) as usize] = sum
                    }
                }
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[test]
fn test_daxpy_fallback() {
    let x = &[1.0, 2.0, 3.0]
    let mut y = [4.0, 5.0, 6.0]

    daxpy_fallback(3, 2.0, x, 1, &!y, 1)

    // y = 2*x + y = [2+4, 4+5, 6+6] = [6, 9, 12]
    assert_approx(y[0], 6.0)
    assert_approx(y[1], 9.0)
    assert_approx(y[2], 12.0)
}

#[test]
fn test_ddot_fallback() {
    let x = &[1.0, 2.0, 3.0]
    let y = &[4.0, 5.0, 6.0]

    let result = ddot_fallback(3, x, 1, y, 1)

    // 1*4 + 2*5 + 3*6 = 32
    assert_approx(result, 32.0)
}

#[test]
fn test_dnrm2_fallback() {
    let x = &[3.0, 4.0]

    let result = dnrm2_fallback(2, x, 1)

    // sqrt(9 + 16) = 5
    assert_approx(result, 5.0)
}

#[test]
fn test_dgemm_fallback() {
    // A = [[1, 2], [3, 4]] (col-major: [1, 3, 2, 4])
    // B = I
    // C = A * B = A
    let a = &[1.0, 3.0, 2.0, 4.0]  // col-major
    let b = &[1.0, 0.0, 0.0, 1.0]  // I col-major
    let mut c = [0.0, 0.0, 0.0, 0.0]

    dgemm_fallback(1, 0, 0, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, &!c, 2)

    // C should equal A
    assert_approx(c[0], 1.0)  // C[0,0]
    assert_approx(c[1], 3.0)  // C[1,0]
    assert_approx(c[2], 2.0)  // C[0,1]
    assert_approx(c[3], 4.0)  // C[1,1]
}
