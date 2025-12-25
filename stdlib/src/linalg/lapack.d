//! LAPACK (Linear Algebra PACKage) bindings

use super::matrix::{Matrix, Vector}
use super::blas::{UpLo, Transpose}

/// Result of matrix decomposition
pub struct DecompResult<T> {
    /// Whether the operation succeeded
    pub success: bool,

    /// LAPACK info code (0 = success, <0 = illegal arg, >0 = failure)
    pub info: i32,

    /// Optional error message
    pub message: Option<string>,
}

/// LU decomposition result
pub struct LU {
    /// LU factors (L is unit lower triangular, U is upper triangular)
    pub factors: Matrix<f64>,

    /// Pivot indices
    pub pivots: Vector<i32>,
}

/// Compute LU decomposition with partial pivoting: P * A = L * U
pub fn lu(a: &Matrix<f64>) -> Result<LU, string> with IO {
    assert!(a.is_square(), "LU requires square matrix");

    let n = a.nrows();
    let mut factors = a.clone();
    let mut pivots = Vector::<i32>::new(n);

    extern "C" {
        fn LAPACK_dgetrf(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            info: *mut i32
        );
    }

    let mut info: i32 = 0;
    let n_i32 = n as i32;
    let lda = factors.ld() as i32;

    unsafe {
        LAPACK_dgetrf(
            &n_i32,
            &n_i32,
            factors.as_mut_ptr(),
            &lda,
            pivots.as_mut_ptr() as *mut i32,
            &mut info
        );
    }

    if info == 0 {
        Ok(LU { factors, pivots })
    } else if info < 0 {
        Err(format!("illegal argument at position {}", -info))
    } else {
        Err(format!("singular matrix: U({},{}) is zero", info, info))
    }
}

/// Solve linear system A*x = b using LU decomposition
pub fn solve(a: &Matrix<f64>, b: &Vector<f64>) -> Result<Vector<f64>, string> with IO {
    let lu_result = lu(a)?;
    solve_lu(&lu_result, b)
}

/// Solve using pre-computed LU decomposition
pub fn solve_lu(lu: &LU, b: &Vector<f64>) -> Result<Vector<f64>, string> with IO {
    let n = lu.factors.nrows();
    assert!(b.len() == n, "dimension mismatch");

    let mut x = b.clone();

    extern "C" {
        fn LAPACK_dgetrs(
            trans: *const i8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            ipiv: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32
        );
    }

    let mut info: i32 = 0;
    let n_i32 = n as i32;
    let nrhs: i32 = 1;
    let lda = lu.factors.ld() as i32;
    let trans: i8 = b'N' as i8;

    unsafe {
        LAPACK_dgetrs(
            &trans,
            &n_i32,
            &nrhs,
            lu.factors.as_ptr(),
            &lda,
            lu.pivots.as_ptr() as *const i32,
            x.as_mut_ptr(),
            &n_i32,
            &mut info
        );
    }

    if info == 0 {
        Ok(x)
    } else {
        Err(format!("dgetrs failed with info = {}", info))
    }
}

/// Compute matrix inverse
pub fn inv(a: &Matrix<f64>) -> Result<Matrix<f64>, string> with IO {
    let lu_result = lu(a)?;

    let n = lu_result.factors.nrows();
    let mut work = Vector::<f64>::new(n * 64); // workspace
    let mut inv = lu_result.factors.clone();

    extern "C" {
        fn LAPACK_dgetri(
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *const i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32
        );
    }

    let mut info: i32 = 0;
    let n_i32 = n as i32;
    let lda = inv.ld() as i32;
    let lwork = work.len() as i32;

    unsafe {
        LAPACK_dgetri(
            &n_i32,
            inv.as_mut_ptr(),
            &lda,
            lu_result.pivots.as_ptr() as *const i32,
            work.as_mut_ptr(),
            &lwork,
            &mut info
        );
    }

    if info == 0 {
        Ok(inv)
    } else {
        Err(format!("matrix inversion failed with info = {}", info))
    }
}

/// Compute matrix determinant
pub fn det(a: &Matrix<f64>) -> Result<f64, string> with IO {
    let lu_result = lu(a)?;

    let n = lu_result.factors.nrows();
    let mut det = 1.0;
    let mut sign = 1;

    for i in 0..n {
        det *= lu_result.factors[(i, i)];
        // Check if pivot was swapped
        if lu_result.pivots[i] as usize != i + 1 {
            sign = -sign;
        }
    }

    Ok(det * sign as f64)
}

/// Cholesky decomposition result
pub struct Cholesky {
    /// Lower triangular factor L such that A = L * L^T
    pub factor: Matrix<f64>,
}

/// Compute Cholesky decomposition for symmetric positive definite matrix
pub fn cholesky(a: &Matrix<f64>) -> Result<Cholesky, string> with IO {
    assert!(a.is_square(), "Cholesky requires square matrix");

    let n = a.nrows();
    let mut factor = a.clone();

    extern "C" {
        fn LAPACK_dpotrf(
            uplo: *const i8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            info: *mut i32
        );
    }

    let mut info: i32 = 0;
    let n_i32 = n as i32;
    let lda = factor.ld() as i32;
    let uplo: i8 = b'L' as i8;

    unsafe {
        LAPACK_dpotrf(
            &uplo,
            &n_i32,
            factor.as_mut_ptr(),
            &lda,
            &mut info
        );
    }

    if info == 0 {
        // Zero out upper triangle
        for i in 0..n {
            for j in (i+1)..n {
                factor[(i, j)] = 0.0;
            }
        }
        Ok(Cholesky { factor })
    } else if info > 0 {
        Err(format!("matrix is not positive definite (leading minor {} is not positive)", info))
    } else {
        Err(format!("illegal argument at position {}", -info))
    }
}

/// QR decomposition result
pub struct QR {
    /// Matrix containing R in upper triangle and Householder vectors below
    pub factors: Matrix<f64>,

    /// Scalar factors of Householder reflectors
    pub tau: Vector<f64>,
}

/// Compute QR decomposition: A = Q * R
pub fn qr(a: &Matrix<f64>) -> Result<QR, string> with IO {
    let (m, n) = a.shape();
    let k = m.min(n);

    let mut factors = a.clone();
    let mut tau = Vector::<f64>::new(k);

    extern "C" {
        fn LAPACK_dgeqrf(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            tau: *mut f64,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32
        );
    }

    // Query optimal workspace size
    let mut info: i32 = 0;
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let lda = factors.ld() as i32;
    let mut work_size: f64 = 0.0;
    let lwork_query: i32 = -1;

    unsafe {
        LAPACK_dgeqrf(
            &m_i32, &n_i32,
            factors.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            &mut work_size,
            &lwork_query,
            &mut info
        );
    }

    let lwork = work_size as i32;
    let mut work = Vector::<f64>::new(lwork as usize);

    unsafe {
        LAPACK_dgeqrf(
            &m_i32, &n_i32,
            factors.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info
        );
    }

    if info == 0 {
        Ok(QR { factors, tau })
    } else {
        Err(format!("QR decomposition failed with info = {}", info))
    }
}

/// Extract Q matrix from QR decomposition
pub fn qr_q(qr: &QR) -> Matrix<f64> with IO {
    let (m, n) = qr.factors.shape();
    let k = m.min(n);

    let mut q = qr.factors.clone();

    extern "C" {
        fn LAPACK_dorgqr(
            m: *const i32,
            n: *const i32,
            k: *const i32,
            a: *mut f64,
            lda: *const i32,
            tau: *const f64,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32
        );
    }

    let mut info: i32 = 0;
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;
    let lda = q.ld() as i32;

    // Query workspace
    let mut work_size: f64 = 0.0;
    let lwork_query: i32 = -1;

    unsafe {
        LAPACK_dorgqr(
            &m_i32, &m_i32, &k_i32,
            q.as_mut_ptr(),
            &lda,
            qr.tau.as_ptr(),
            &mut work_size,
            &lwork_query,
            &mut info
        );
    }

    let lwork = work_size as i32;
    let mut work = Vector::<f64>::new(lwork as usize);

    unsafe {
        LAPACK_dorgqr(
            &m_i32, &m_i32, &k_i32,
            q.as_mut_ptr(),
            &lda,
            qr.tau.as_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info
        );
    }

    q
}

/// SVD decomposition result
pub struct SVD {
    /// Left singular vectors (m x m)
    pub u: Matrix<f64>,

    /// Singular values (min(m,n))
    pub s: Vector<f64>,

    /// Right singular vectors transposed (n x n)
    pub vt: Matrix<f64>,
}

/// Compute Singular Value Decomposition: A = U * S * V^T
pub fn svd(a: &Matrix<f64>) -> Result<SVD, string> with IO {
    let (m, n) = a.shape();
    let k = m.min(n);

    let mut work_a = a.clone();
    let mut u = Matrix::<f64>::zeros(m, m);
    let mut s = Vector::<f64>::new(k);
    let mut vt = Matrix::<f64>::zeros(n, n);

    extern "C" {
        fn LAPACK_dgesvd(
            jobu: *const i8,
            jobvt: *const i8,
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            s: *mut f64,
            u: *mut f64,
            ldu: *const i32,
            vt: *mut f64,
            ldvt: *const i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32
        );
    }

    let jobu: i8 = b'A' as i8;   // All of U
    let jobvt: i8 = b'A' as i8;  // All of V^T
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let lda = work_a.ld() as i32;
    let ldu = u.ld() as i32;
    let ldvt = vt.ld() as i32;
    let mut info: i32 = 0;

    // Query workspace
    let mut work_size: f64 = 0.0;
    let lwork_query: i32 = -1;

    unsafe {
        LAPACK_dgesvd(
            &jobu, &jobvt,
            &m_i32, &n_i32,
            work_a.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            &mut work_size,
            &lwork_query,
            &mut info
        );
    }

    let lwork = work_size as i32;
    let mut work = Vector::<f64>::new(lwork as usize);

    unsafe {
        LAPACK_dgesvd(
            &jobu, &jobvt,
            &m_i32, &n_i32,
            work_a.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            &mut info
        );
    }

    if info == 0 {
        Ok(SVD { u, s, vt })
    } else {
        Err(format!("SVD failed with info = {}", info))
    }
}

/// Eigenvalue decomposition result
pub struct Eigen {
    /// Eigenvalues (real parts)
    pub values_real: Vector<f64>,

    /// Eigenvalues (imaginary parts)
    pub values_imag: Vector<f64>,

    /// Right eigenvectors (columns)
    pub vectors: Matrix<f64>,
}

/// Compute eigenvalues and eigenvectors of a general matrix
pub fn eig(a: &Matrix<f64>) -> Result<Eigen, string> with IO {
    assert!(a.is_square(), "eigenvalue decomposition requires square matrix");

    let n = a.nrows();
    let mut work_a = a.clone();
    let mut wr = Vector::<f64>::new(n); // Real parts
    let mut wi = Vector::<f64>::new(n); // Imaginary parts
    let mut vl = Matrix::<f64>::zeros(n, n); // Left eigenvectors (not computed)
    let mut vr = Matrix::<f64>::zeros(n, n); // Right eigenvectors

    extern "C" {
        fn LAPACK_dgeev(
            jobvl: *const i8,
            jobvr: *const i8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            wr: *mut f64,
            wi: *mut f64,
            vl: *mut f64,
            ldvl: *const i32,
            vr: *mut f64,
            ldvr: *const i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32
        );
    }

    let jobvl: i8 = b'N' as i8;  // Don't compute left eigenvectors
    let jobvr: i8 = b'V' as i8;  // Compute right eigenvectors
    let n_i32 = n as i32;
    let lda = work_a.ld() as i32;
    let ldvl = vl.ld() as i32;
    let ldvr = vr.ld() as i32;
    let mut info: i32 = 0;

    // Query workspace
    let mut work_size: f64 = 0.0;
    let lwork_query: i32 = -1;

    unsafe {
        LAPACK_dgeev(
            &jobvl, &jobvr,
            &n_i32,
            work_a.as_mut_ptr(),
            &lda,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            &ldvl,
            vr.as_mut_ptr(),
            &ldvr,
            &mut work_size,
            &lwork_query,
            &mut info
        );
    }

    let lwork = work_size as i32;
    let mut work = Vector::<f64>::new(lwork as usize);

    unsafe {
        LAPACK_dgeev(
            &jobvl, &jobvr,
            &n_i32,
            work_a.as_mut_ptr(),
            &lda,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            &ldvl,
            vr.as_mut_ptr(),
            &ldvr,
            work.as_mut_ptr(),
            &lwork,
            &mut info
        );
    }

    if info == 0 {
        Ok(Eigen {
            values_real: wr,
            values_imag: wi,
            vectors: vr,
        })
    } else {
        Err(format!("eigenvalue computation failed with info = {}", info))
    }
}

/// Symmetric eigenvalue decomposition
pub struct SymEigen {
    /// Eigenvalues (ascending order)
    pub values: Vector<f64>,

    /// Eigenvectors (columns)
    pub vectors: Matrix<f64>,
}

/// Compute eigenvalues and eigenvectors of a symmetric matrix
pub fn eigh(a: &Matrix<f64>) -> Result<SymEigen, string> with IO {
    assert!(a.is_square(), "symmetric eigenvalue decomposition requires square matrix");

    let n = a.nrows();
    let mut work_a = a.clone();
    let mut w = Vector::<f64>::new(n);

    extern "C" {
        fn LAPACK_dsyev(
            jobz: *const i8,
            uplo: *const i8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            w: *mut f64,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32
        );
    }

    let jobz: i8 = b'V' as i8;  // Compute eigenvectors
    let uplo: i8 = b'U' as i8;  // Upper triangle
    let n_i32 = n as i32;
    let lda = work_a.ld() as i32;
    let mut info: i32 = 0;

    // Query workspace
    let mut work_size: f64 = 0.0;
    let lwork_query: i32 = -1;

    unsafe {
        LAPACK_dsyev(
            &jobz, &uplo,
            &n_i32,
            work_a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            &mut work_size,
            &lwork_query,
            &mut info
        );
    }

    let lwork = work_size as i32;
    let mut work = Vector::<f64>::new(lwork as usize);

    unsafe {
        LAPACK_dsyev(
            &jobz, &uplo,
            &n_i32,
            work_a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info
        );
    }

    if info == 0 {
        Ok(SymEigen {
            values: w,
            vectors: work_a,
        })
    } else {
        Err(format!("symmetric eigenvalue computation failed with info = {}", info))
    }
}

/// Compute condition number of a matrix
pub fn cond(a: &Matrix<f64>) -> Result<f64, string> with IO {
    let svd_result = svd(a)?;

    let s_max = svd_result.s[0];
    let s_min = svd_result.s[svd_result.s.len() - 1];

    if s_min == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(s_max / s_min)
    }
}

/// Compute matrix rank (numerical)
pub fn rank(a: &Matrix<f64>, tol: f64) -> Result<usize, string> with IO {
    let svd_result = svd(a)?;

    let rank = svd_result.s.iter()
        .filter(|&s| *s > tol)
        .count();

    Ok(rank)
}

/// Compute pseudoinverse using SVD
pub fn pinv(a: &Matrix<f64>, tol: f64) -> Result<Matrix<f64>, string> with IO {
    let svd_result = svd(a)?;

    let (m, n) = a.shape();

    // Compute S^+ (pseudoinverse of singular values)
    let mut s_inv = Vector::<f64>::zeros(svd_result.s.len());
    for i in 0..svd_result.s.len() {
        if svd_result.s[i] > tol {
            s_inv[i] = 1.0 / svd_result.s[i];
        }
    }

    // A^+ = V * S^+ * U^T
    // First compute V * S^+
    let mut vs = Matrix::<f64>::zeros(n, svd_result.s.len());
    for i in 0..n {
        for j in 0..svd_result.s.len() {
            vs[(i, j)] = svd_result.vt[(j, i)] * s_inv[j];
        }
    }

    // Then multiply by U^T
    &vs * &svd_result.u.t()
}
