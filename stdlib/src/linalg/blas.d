//! BLAS (Basic Linear Algebra Subprograms) bindings

use super::matrix::{Matrix, Vector, Layout}

/// BLAS Level 1: Vector-vector operations

/// DAXPY: y = alpha * x + y
pub fn daxpy(alpha: f64, x: &Vector<f64>, y: &!Vector<f64>) with IO {
    assert!(x.len() == y.len());

    extern "C" {
        fn cblas_daxpy(
            n: i32,
            alpha: f64,
            x: *const f64,
            incx: i32,
            y: *mut f64,
            incy: i32
        );
    }

    unsafe {
        cblas_daxpy(
            x.len() as i32,
            alpha,
            x.as_ptr(),
            1,
            y.as_mut_ptr(),
            1
        );
    }
}

/// DDOT: dot product
pub fn ddot(x: &Vector<f64>, y: &Vector<f64>) -> f64 with IO {
    assert!(x.len() == y.len());

    extern "C" {
        fn cblas_ddot(
            n: i32,
            x: *const f64,
            incx: i32,
            y: *const f64,
            incy: i32
        ) -> f64;
    }

    unsafe {
        cblas_ddot(x.len() as i32, x.as_ptr(), 1, y.as_ptr(), 1)
    }
}

/// DNRM2: Euclidean norm
pub fn dnrm2(x: &Vector<f64>) -> f64 with IO {
    extern "C" {
        fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
    }

    unsafe {
        cblas_dnrm2(x.len() as i32, x.as_ptr(), 1)
    }
}

/// DSCAL: x = alpha * x
pub fn dscal(alpha: f64, x: &!Vector<f64>) with IO {
    extern "C" {
        fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
    }

    unsafe {
        cblas_dscal(x.len() as i32, alpha, x.as_mut_ptr(), 1);
    }
}

/// DASUM: sum of absolute values
pub fn dasum(x: &Vector<f64>) -> f64 with IO {
    extern "C" {
        fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64;
    }

    unsafe {
        cblas_dasum(x.len() as i32, x.as_ptr(), 1)
    }
}

/// IDAMAX: index of max absolute value
pub fn idamax(x: &Vector<f64>) -> usize with IO {
    extern "C" {
        fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32;
    }

    unsafe {
        cblas_idamax(x.len() as i32, x.as_ptr(), 1) as usize
    }
}

/// BLAS Level 2: Matrix-vector operations

/// Transpose flag for BLAS
pub enum Transpose {
    NoTrans = 111,  // CblasNoTrans
    Trans = 112,    // CblasTrans
    ConjTrans = 113 // CblasConjTrans
}

/// DGEMV: y = alpha * A * x + beta * y (or A^T)
pub fn dgemv(
    trans: Transpose,
    alpha: f64,
    a: &Matrix<f64>,
    x: &Vector<f64>,
    beta: f64,
    y: &!Vector<f64>
) with IO {
    let (m, n) = a.shape();
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        _ => (m, n),
    };

    assert!(x.len() == x_len && y.len() == y_len);

    extern "C" {
        fn cblas_dgemv(
            order: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            x: *const f64,
            incx: i32,
            beta: f64,
            y: *mut f64,
            incy: i32
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;

    unsafe {
        cblas_dgemv(
            CBLAS_ROW_MAJOR,
            trans as i32,
            m as i32,
            n as i32,
            alpha,
            a.as_ptr(),
            a.ld() as i32,
            x.as_ptr(),
            1,
            beta,
            y.as_mut_ptr(),
            1
        );
    }
}

/// DSYMV: y = alpha * A * x + beta * y (symmetric A)
pub fn dsymv(
    uplo: UpLo,
    alpha: f64,
    a: &Matrix<f64>,
    x: &Vector<f64>,
    beta: f64,
    y: &!Vector<f64>
) with IO {
    assert!(a.is_square() && a.nrows() == x.len() && x.len() == y.len());

    extern "C" {
        fn cblas_dsymv(
            order: i32,
            uplo: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            x: *const f64,
            incx: i32,
            beta: f64,
            y: *mut f64,
            incy: i32
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;

    unsafe {
        cblas_dsymv(
            CBLAS_ROW_MAJOR,
            uplo as i32,
            a.nrows() as i32,
            alpha,
            a.as_ptr(),
            a.ld() as i32,
            x.as_ptr(),
            1,
            beta,
            y.as_mut_ptr(),
            1
        );
    }
}

/// Upper or lower triangular
pub enum UpLo {
    Upper = 121, // CblasUpper
    Lower = 122, // CblasLower
}

/// DTRSV: solve triangular system A*x = b
pub fn dtrsv(
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    a: &Matrix<f64>,
    x: &!Vector<f64>
) with IO {
    assert!(a.is_square() && a.nrows() == x.len());

    extern "C" {
        fn cblas_dtrsv(
            order: i32,
            uplo: i32,
            trans: i32,
            diag: i32,
            n: i32,
            a: *const f64,
            lda: i32,
            x: *mut f64,
            incx: i32
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;

    unsafe {
        cblas_dtrsv(
            CBLAS_ROW_MAJOR,
            uplo as i32,
            trans as i32,
            diag as i32,
            a.nrows() as i32,
            a.as_ptr(),
            a.ld() as i32,
            x.as_mut_ptr(),
            1
        );
    }
}

/// Unit or non-unit diagonal
pub enum Diag {
    NonUnit = 131, // CblasNonUnit
    Unit = 132,    // CblasUnit
}

/// BLAS Level 3: Matrix-matrix operations

/// DGEMM: C = alpha * A * B + beta * C
pub fn dgemm(
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: f64,
    a: &Matrix<f64>,
    b: &Matrix<f64>,
    beta: f64,
    c: &!Matrix<f64>
) with IO {
    // Compute dimensions
    let m = match trans_a {
        Transpose::NoTrans => a.nrows(),
        _ => a.ncols(),
    };
    let k = match trans_a {
        Transpose::NoTrans => a.ncols(),
        _ => a.nrows(),
    };
    let n = match trans_b {
        Transpose::NoTrans => b.ncols(),
        _ => b.nrows(),
    };

    // Verify dimensions
    let k_b = match trans_b {
        Transpose::NoTrans => b.nrows(),
        _ => b.ncols(),
    };
    assert!(k == k_b, "inner dimensions must match");
    assert!(c.nrows() == m && c.ncols() == n, "output dimensions mismatch");

    extern "C" {
        fn cblas_dgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;

    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            trans_a as i32,
            trans_b as i32,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            a.ld() as i32,
            b.as_ptr(),
            b.ld() as i32,
            beta,
            c.as_mut_ptr(),
            c.ld() as i32
        );
    }
}

/// DSYRK: C = alpha * A * A^T + beta * C (symmetric rank-k update)
pub fn dsyrk(
    uplo: UpLo,
    trans: Transpose,
    alpha: f64,
    a: &Matrix<f64>,
    beta: f64,
    c: &!Matrix<f64>
) with IO {
    let n = match trans {
        Transpose::NoTrans => a.nrows(),
        _ => a.ncols(),
    };
    let k = match trans {
        Transpose::NoTrans => a.ncols(),
        _ => a.nrows(),
    };

    assert!(c.nrows() == n && c.ncols() == n);

    extern "C" {
        fn cblas_dsyrk(
            order: i32,
            uplo: i32,
            trans: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;

    unsafe {
        cblas_dsyrk(
            CBLAS_ROW_MAJOR,
            uplo as i32,
            trans as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            a.ld() as i32,
            beta,
            c.as_mut_ptr(),
            c.ld() as i32
        );
    }
}

/// DTRSM: solve triangular matrix equation A*X = B or X*A = B
pub fn dtrsm(
    side: Side,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    alpha: f64,
    a: &Matrix<f64>,
    b: &!Matrix<f64>
) with IO {
    let m = b.nrows();
    let n = b.ncols();

    extern "C" {
        fn cblas_dtrsm(
            order: i32,
            side: i32,
            uplo: i32,
            trans: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *mut f64,
            ldb: i32
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;

    unsafe {
        cblas_dtrsm(
            CBLAS_ROW_MAJOR,
            side as i32,
            uplo as i32,
            trans as i32,
            diag as i32,
            m as i32,
            n as i32,
            alpha,
            a.as_ptr(),
            a.ld() as i32,
            b.as_mut_ptr(),
            b.ld() as i32
        );
    }
}

/// Left or right side
pub enum Side {
    Left = 141,  // CblasLeft
    Right = 142, // CblasRight
}

/// High-level matrix multiplication operator
impl Mul<&Matrix<f64>> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, other: &Matrix<f64>) -> Matrix<f64> with IO {
        let mut c = Matrix::zeros(self.nrows(), other.ncols());
        dgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            self,
            other,
            0.0,
            &!c
        );
        c
    }
}

/// Matrix-vector multiplication
impl Mul<&Vector<f64>> for &Matrix<f64> {
    type Output = Vector<f64>;

    fn mul(self, v: &Vector<f64>) -> Vector<f64> with IO {
        let mut y = Vector::zeros(self.nrows());
        dgemv(Transpose::NoTrans, 1.0, self, v, 0.0, &!y);
        y
    }
}
