//! Dynamic-sized linear algebra primitives for Sounio
//!
//! Provides runtime-sized matrices that complement the fixed-size `Matrix8`
//! in the stdlib. Backed by flat `Vec<T>` with row-major storage.
//!
//! For hardware-accelerated operations, see `runtime::ffi::ffi_blas`.

use std::fmt;
use std::ops::{Add, Mul};

/// A dynamically sized matrix with row-major storage.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T> DynMatrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    /// Creates a new zero-filled matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Creates an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self
    where
        T: From<i32>,
    {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat.data[i * n + i] = T::from(1);
        }
        mat
    }

    /// Creates a matrix from a flat vector of data.
    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Data length {} doesn't match dimensions {}x{}",
                data.len(),
                rows,
                cols
            ));
        }
        Ok(Self { data, rows, cols })
    }

    /// Returns the element at position (i, j).
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.rows && j < self.cols, "index out of bounds");
        self.data[i * self.cols + j]
    }

    /// Sets the element at position (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        assert!(i < self.rows && j < self.cols, "index out of bounds");
        self.data[i * self.cols + j] = val;
    }

    /// Returns the matrix dimensions as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Returns a reference to the underlying data slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a vector representing the i-th row.
    pub fn row(&self, i: usize) -> Vec<T> {
        assert!(i < self.rows, "row index out of bounds");
        let start = i * self.cols;
        self.data[start..start + self.cols].to_vec()
    }

    /// Returns a vector representing the j-th column.
    pub fn col(&self, j: usize) -> Vec<T> {
        assert!(j < self.cols, "column index out of bounds");
        self.data.iter().skip(j).step_by(self.cols).copied().collect()
    }

    /// Transposes the matrix, returning a new matrix.
    pub fn transposed(&self) -> Self {
        let mut new_data = vec![T::default(); self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Self {
            data: new_data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Multiplies every element by a scalar.
    pub fn scale(&mut self, scalar: T) {
        for val in &mut self.data {
            *val = *val * scalar;
        }
    }

    /// Element-wise addition. Panics on dimension mismatch.
    pub fn add_assign(&mut self, other: &DynMatrix<T>) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "dimension mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a + *b;
        }
    }

    /// Matrix multiplication: self (m x k) * other (k x n) -> result (m x n).
    pub fn matmul(&self, other: &DynMatrix<T>) -> DynMatrix<T> {
        assert_eq!(
            self.cols, other.rows,
            "matmul dimension mismatch: {}x{} * {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }
        result
    }
}

impl<T: fmt::Display + Copy + Default + Add<Output = T> + Mul<Output = T>> fmt::Display
    for DynMatrix<T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{}", self.data[i * self.cols + j])?;
                if j < self.cols - 1 {
                    write!(f, " ")?;
                }
            }
            if i < self.rows - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// A dynamic matrix with epistemic uncertainty tracking.
pub struct EpistemicDynMatrix {
    /// Matrix values
    pub values: DynMatrix<f32>,
    /// Per-element uncertainty bounds
    pub epsilon: DynMatrix<f32>,
    /// Optional per-element confidence
    pub confidence: Option<DynMatrix<f32>>,
}

impl EpistemicDynMatrix {
    /// Creates a new epistemic matrix with given values and epsilon.
    pub fn new(
        values: DynMatrix<f32>,
        epsilon: DynMatrix<f32>,
        confidence: Option<DynMatrix<f32>>,
    ) -> Result<Self, String> {
        if values.shape() != epsilon.shape() {
            return Err(format!(
                "values/epsilon shape mismatch: {:?} vs {:?}",
                values.shape(),
                epsilon.shape()
            ));
        }
        if let Some(ref conf) = confidence {
            if values.shape() != conf.shape() {
                return Err(format!(
                    "values/confidence shape mismatch: {:?} vs {:?}",
                    values.shape(),
                    conf.shape()
                ));
            }
        }
        Ok(Self {
            values,
            epsilon,
            confidence,
        })
    }

    /// Creates an epistemic matrix from values only, with zero epsilon.
    pub fn from_values(values: DynMatrix<f32>) -> Self {
        let eps = DynMatrix::zeros(values.nrows(), values.ncols());
        Self {
            values,
            epsilon: eps,
            confidence: None,
        }
    }

    /// Matrix multiply with uncertainty propagation.
    ///
    /// ε_C[i,j] = sqrt(Σ_k (ε_A[i,k]*B[k,j])² + (A[i,k]*ε_B[k,j])²)
    pub fn epistemic_matmul(&self, other: &EpistemicDynMatrix) -> Result<EpistemicDynMatrix, String> {
        let result_values = self.values.matmul(&other.values);
        let result_epsilon = self.propagate_uncertainty(other)?;
        Ok(EpistemicDynMatrix {
            values: result_values,
            epsilon: result_epsilon,
            confidence: None,
        })
    }

    fn propagate_uncertainty(&self, other: &EpistemicDynMatrix) -> Result<DynMatrix<f32>, String> {
        let (rows_a, cols_a) = self.values.shape();
        let (rows_b, cols_b) = other.values.shape();

        if cols_a != rows_b {
            return Err("dimension mismatch for uncertainty propagation".to_string());
        }

        let mut result = DynMatrix::zeros(rows_a, cols_b);
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0f32;
                for k in 0..cols_a {
                    let a_val = self.values.get(i, k);
                    let a_eps = self.epsilon.get(i, k);
                    let b_val = other.values.get(k, j);
                    let b_eps = other.epsilon.get(k, j);
                    sum += (a_eps * b_val).powi(2) + (a_val * b_eps).powi(2);
                }
                result.set(i, j, sum.sqrt());
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let mat = DynMatrix::<i32>::zeros(2, 3);
        assert_eq!(mat.shape(), (2, 3));
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(mat.get(i, j), 0);
            }
        }
    }

    #[test]
    fn test_identity() {
        let mat = DynMatrix::<i32>::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(mat.get(i, j), if i == j { 1 } else { 0 });
            }
        }
    }

    #[test]
    fn test_from_vec() {
        let mat = DynMatrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3).unwrap();
        assert_eq!(mat.shape(), (2, 3));
        assert_eq!(mat.get(1, 1), 5);
    }

    #[test]
    fn test_transpose() {
        let mat = DynMatrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3).unwrap();
        let t = mat.transposed();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.as_slice(), &[1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_add_assign() {
        let mut a = DynMatrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        let b = DynMatrix::from_vec(vec![5, 6, 7, 8], 2, 2).unwrap();
        a.add_assign(&b);
        assert_eq!(a.as_slice(), &[6, 8, 10, 12]);
    }

    #[test]
    fn test_matmul() {
        let a = DynMatrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        let b = DynMatrix::from_vec(vec![2, 0, 1, 2], 2, 2).unwrap();
        let c = a.matmul(&b);
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.as_slice(), &[4, 4, 10, 8]);
    }

    #[test]
    fn test_epistemic_matmul() {
        let values_a = DynMatrix::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let epsilon_a = DynMatrix::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], 2, 2).unwrap();
        let ea = EpistemicDynMatrix::new(values_a, epsilon_a, None).unwrap();

        let values_b = DynMatrix::from_vec(vec![2.0f32, 0.0, 1.0, 2.0], 2, 2).unwrap();
        let epsilon_b = DynMatrix::from_vec(vec![0.2f32, 0.1, 0.3, 0.2], 2, 2).unwrap();
        let eb = EpistemicDynMatrix::new(values_b, epsilon_b, None).unwrap();

        let result = ea.epistemic_matmul(&eb).unwrap();
        assert_eq!(result.values.shape(), (2, 2));
        assert_eq!(result.epsilon.shape(), (2, 2));

        // Uncertainty should be positive for all elements
        for i in 0..2 {
            for j in 0..2 {
                assert!(result.epsilon.get(i, j) >= 0.0);
            }
        }
    }

    #[test]
    fn test_display() {
        let mat = DynMatrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        assert_eq!(format!("{mat}"), "1 2\n3 4");
    }
}
