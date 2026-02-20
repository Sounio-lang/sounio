//! CBLAS FFI bindings for Sounio
//!
//! Provides C-compatible bindings to the standard CBLAS (C interface to BLAS)
//! library for linear algebra operations. This is Sounio's bridge to hardware-
//! optimized matrix operations (OpenBLAS, MKL, Accelerate).
//!
//! # Linking
//!
//! Requires a CBLAS implementation at link time:
//! - Linux: `libopenblas` or `libblas`
//! - macOS: Accelerate framework (system default)
//! - Windows: Intel MKL or OpenBLAS

use std::os::raw::{c_double, c_float, c_int};

/// CBLAS matrix storage order
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CblasOrder {
    RowMajor = 101,
    ColMajor = 102,
}

/// CBLAS transpose operation
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CblasTranspose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
}

#[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
#[cfg_attr(
    not(target_os = "macos"),
    link(name = "openblas", kind = "dylib")
)]
extern "C" {
    // Level 1: vector operations
    pub fn cblas_sdot(
        n: c_int,
        x: *const c_float,
        incx: c_int,
        y: *const c_float,
        incy: c_int,
    ) -> c_float;
    pub fn cblas_ddot(
        n: c_int,
        x: *const c_double,
        incx: c_int,
        y: *const c_double,
        incy: c_int,
    ) -> c_double;
    pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;
    pub fn cblas_saxpy(
        n: c_int,
        alpha: c_float,
        x: *const c_float,
        incx: c_int,
        y: *mut c_float,
        incy: c_int,
    );
    pub fn cblas_daxpy(
        n: c_int,
        alpha: c_double,
        x: *const c_double,
        incx: c_int,
        y: *mut c_double,
        incy: c_int,
    );
    pub fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
    pub fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);

    // Level 2: matrix-vector operations
    pub fn cblas_sgemv(
        order: CblasOrder,
        transa: CblasTranspose,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: c_float,
        y: *mut c_float,
        incy: c_int,
    );
    pub fn cblas_dgemv(
        order: CblasOrder,
        transa: CblasTranspose,
        m: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        x: *const c_double,
        incx: c_int,
        beta: c_double,
        y: *mut c_double,
        incy: c_int,
    );

    // Level 3: matrix-matrix operations
    pub fn cblas_sgemm(
        order: CblasOrder,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
    pub fn cblas_dgemm(
        order: CblasOrder,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );
}

// =============================================================================
// Safe wrappers
// =============================================================================

/// Single-precision dot product: x . y
pub fn sdot(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "dot product: vectors must have same length");
    unsafe { cblas_sdot(x.len() as c_int, x.as_ptr(), 1, y.as_ptr(), 1) }
}

/// Double-precision dot product: x . y
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "dot product: vectors must have same length");
    unsafe { cblas_ddot(x.len() as c_int, x.as_ptr(), 1, y.as_ptr(), 1) }
}

/// Single-precision Euclidean norm: ||x||_2
pub fn snrm2(x: &[f32]) -> f32 {
    unsafe { cblas_snrm2(x.len() as c_int, x.as_ptr(), 1) }
}

/// Double-precision Euclidean norm: ||x||_2
pub fn dnrm2(x: &[f64]) -> f64 {
    unsafe { cblas_dnrm2(x.len() as c_int, x.as_ptr(), 1) }
}

/// Single-precision GEMM: C = alpha * A * B + beta * C (row-major, no transpose)
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");
    assert_eq!(c.len(), m * n, "C dimensions mismatch");
    unsafe {
        cblas_sgemm(
            CblasOrder::RowMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            n as c_int,
            beta,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

/// Double-precision GEMM: C = alpha * A * B + beta * C (row-major, no transpose)
pub fn dgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) {
    assert_eq!(a.len(), m * k, "A dimensions mismatch");
    assert_eq!(b.len(), k * n, "B dimensions mismatch");
    assert_eq!(c.len(), m * n, "C dimensions mismatch");
    unsafe {
        cblas_dgemm(
            CblasOrder::RowMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            n as c_int,
            beta,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: these tests require a BLAS library at link time.
    // They will fail to link if no BLAS is installed.
    // Run with: cargo test -p souc --features blas

    #[test]
    #[cfg(feature = "blas")]
    fn test_sdot() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        let result = sdot(&x, &y);
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "blas")]
    fn test_sgemm_2x2() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        sgemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }
}
