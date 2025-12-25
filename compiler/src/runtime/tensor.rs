//! Tensor<T, Shape> with Compile-Time Shape Verification
//!
//! This module provides tensors with static shape checking for scientific computing.
//! Shape mismatches are caught at compile time, preventing runtime dimension errors.
//!
//! # Shape Representation
//!
//! Shapes are represented as type-level lists:
//! - `Tensor<f64, [3, 4]>` - 3x4 matrix
//! - `Tensor<f64, [2, 3, 4]>` - 2x3x4 3D tensor
//! - `Tensor<f64, []>` - scalar
//! - `Tensor<f64, [N]>` - vector of length N (compile-time const)
//!
//! # Shape Rules
//!
//! - Matrix multiply: `[M, K] @ [K, N] -> [M, N]`
//! - Element-wise: shapes must match exactly or broadcast
//! - Transpose: `[M, N]^T -> [N, M]`
//! - Reshape: total elements must match
//!
//! # Example
//!
//! ```d
//! let a: Tensor<f64, [3, 4]> = zeros();
//! let b: Tensor<f64, [4, 2]> = ones();
//! let c = a @ b;  // c: Tensor<f64, [3, 2]>
//!
//! // Compile error: shape mismatch
//! // let d: Tensor<f64, [3, 3]> = a @ b;
//! ```

use std::fmt;

/// Shape dimension (can be static or dynamic)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Static dimension known at compile time
    Static(usize),
    /// Dynamic dimension (for runtime flexibility)
    Dynamic,
    /// Symbolic dimension (for generic code)
    Symbolic(u32),
}

impl Dim {
    pub fn is_static(&self) -> bool {
        matches!(self, Dim::Static(_))
    }

    pub fn value(&self) -> Option<usize> {
        match self {
            Dim::Static(n) => Some(*n),
            _ => None,
        }
    }

    /// Check if two dimensions are compatible
    pub fn compatible(&self, other: &Dim) -> bool {
        match (self, other) {
            (Dim::Static(a), Dim::Static(b)) => a == b,
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
            (Dim::Symbolic(a), Dim::Symbolic(b)) => a == b,
            _ => false,
        }
    }
}

/// Tensor shape as a list of dimensions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<Dim>,
}

impl Shape {
    /// Create shape from static dimensions
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.iter().map(|&d| Dim::Static(d)).collect(),
        }
    }

    /// Create scalar shape
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Create vector shape
    pub fn vector(n: usize) -> Self {
        Self::new(&[n])
    }

    /// Create matrix shape
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self::new(&[rows, cols])
    }

    /// Number of dimensions (rank)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements
    pub fn numel(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, d| d.value().map(|v| acc * v))
    }

    /// Get dimension at index
    pub fn dim(&self, i: usize) -> Option<&Dim> {
        self.dims.get(i)
    }

    /// Check if shapes are compatible for element-wise operations
    pub fn broadcast_compatible(&self, other: &Shape) -> bool {
        let max_rank = self.rank().max(other.rank());

        for i in 0..max_rank {
            let d1 = self.dims.get(self.rank().saturating_sub(i + 1));
            let d2 = other.dims.get(other.rank().saturating_sub(i + 1));

            match (d1, d2) {
                (Some(Dim::Static(a)), Some(Dim::Static(b))) => {
                    if *a != *b && *a != 1 && *b != 1 {
                        return false;
                    }
                }
                (None, _) | (_, None) => {}
                _ => {}
            }
        }
        true
    }

    /// Compute broadcast result shape
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        if !self.broadcast_compatible(other) {
            return None;
        }

        let max_rank = self.rank().max(other.rank());
        let mut result_dims = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let d1 = self.dims.get(self.rank().saturating_sub(i + 1));
            let d2 = other.dims.get(other.rank().saturating_sub(i + 1));

            let dim = match (d1, d2) {
                (Some(Dim::Static(a)), Some(Dim::Static(b))) => Dim::Static(*a.max(b)),
                (Some(d), None) | (None, Some(d)) => *d,
                (Some(Dim::Dynamic), _) | (_, Some(Dim::Dynamic)) => Dim::Dynamic,
                _ => Dim::Dynamic,
            };
            result_dims.push(dim);
        }

        result_dims.reverse();
        Some(Shape { dims: result_dims })
    }

    /// Check if shapes are compatible for matrix multiplication
    /// [M, K] @ [K, N] -> [M, N]
    pub fn matmul_compatible(&self, other: &Shape) -> bool {
        if self.rank() < 1 || other.rank() < 1 {
            return false;
        }

        let k1 = self.dims.last();
        let k2 = if other.rank() == 1 {
            other.dims.first()
        } else {
            other.dims.get(other.rank() - 2)
        };

        match (k1, k2) {
            (Some(d1), Some(d2)) => d1.compatible(d2),
            _ => false,
        }
    }

    /// Compute matmul result shape
    pub fn matmul_result(&self, other: &Shape) -> Option<Shape> {
        if !self.matmul_compatible(other) {
            return None;
        }

        if self.rank() == 1 && other.rank() == 1 {
            // Vector dot product -> scalar
            return Some(Shape::scalar());
        }

        if self.rank() == 1 {
            // [K] @ [K, N] -> [N]
            return Some(Shape {
                dims: vec![*other.dims.last()?],
            });
        }

        if other.rank() == 1 {
            // [M, K] @ [K] -> [M]
            return Some(Shape {
                dims: vec![self.dims[self.rank() - 2]],
            });
        }

        // [M, K] @ [K, N] -> [M, N]
        let m = self.dims[self.rank() - 2];
        let n = *other.dims.last()?;

        // Handle batched matmul
        let batch_dims: Vec<Dim> = if self.rank() > 2 || other.rank() > 2 {
            let max_batch = (self.rank() - 2).max(other.rank() - 2);
            (0..max_batch)
                .map(|i| {
                    let d1 = self.dims.get(i);
                    let d2 = other.dims.get(i);
                    match (d1, d2) {
                        (Some(d), None) | (None, Some(d)) => *d,
                        (Some(Dim::Static(a)), Some(Dim::Static(b))) => Dim::Static(*a.max(b)),
                        _ => Dim::Dynamic,
                    }
                })
                .collect()
        } else {
            vec![]
        };

        let mut result = batch_dims;
        result.push(m);
        result.push(n);
        Some(Shape { dims: result })
    }

    /// Transpose (swap last two dimensions)
    pub fn transpose(&self) -> Option<Shape> {
        if self.rank() < 2 {
            return Some(self.clone());
        }

        let mut dims = self.dims.clone();
        let n = dims.len();
        dims.swap(n - 2, n - 1);
        Some(Shape { dims })
    }

    /// Reshape to new shape (must have same number of elements)
    pub fn reshape(&self, new_shape: &Shape) -> Option<Shape> {
        match (self.numel(), new_shape.numel()) {
            (Some(a), Some(b)) if a == b => Some(new_shape.clone()),
            _ => None,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match d {
                Dim::Static(n) => write!(f, "{}", n)?,
                Dim::Dynamic => write!(f, "?")?,
                Dim::Symbolic(id) => write!(f, "N{}", id)?,
            }
        }
        write!(f, "]")
    }
}

/// Dense tensor with row-major storage
#[derive(Clone)]
pub struct Tensor<T = f64> {
    data: Vec<T>,
    shape: Shape,
    strides: Vec<usize>,
}

impl<T: Clone + Default> Tensor<T> {
    /// Create tensor filled with default value
    pub fn zeros(shape: &Shape) -> Self {
        let numel = shape.numel().expect("Shape must be static");
        let strides = compute_strides(shape);
        Self {
            data: vec![T::default(); numel],
            shape: shape.clone(),
            strides,
        }
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get element at flat index
    pub fn get_flat(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    /// Set element at flat index
    pub fn set_flat(&mut self, i: usize, value: T) {
        if i < self.data.len() {
            self.data[i] = value;
        }
    }

    /// Convert multi-index to flat index
    pub fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.rank() {
            return None;
        }

        let mut flat = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if let Some(Dim::Static(dim)) = self.shape.dim(i)
                && idx >= *dim
            {
                return None;
            }
            flat += idx * self.strides[i];
        }
        Some(flat)
    }

    /// Get element at multi-index
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        self.flat_index(indices).and_then(|i| self.data.get(i))
    }

    /// Set element at multi-index
    pub fn set(&mut self, indices: &[usize], value: T) {
        if let Some(i) = self.flat_index(indices)
            && i < self.data.len()
        {
            self.data[i] = value;
        }
    }

    /// Get raw data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get mutable raw data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Reshape tensor (view with new shape)
    pub fn reshape(&self, new_shape: &Shape) -> Option<Self> {
        if self.shape.numel() != new_shape.numel() {
            return None;
        }
        let strides = compute_strides(new_shape);
        Some(Self {
            data: self.data.clone(),
            shape: new_shape.clone(),
            strides,
        })
    }
}

impl Tensor<f64> {
    /// Create tensor filled with ones
    pub fn ones(shape: &Shape) -> Self {
        let numel = shape.numel().expect("Shape must be static");
        let strides = compute_strides(shape);
        Self {
            data: vec![1.0; numel],
            shape: shape.clone(),
            strides,
        }
    }

    /// Create tensor from slice
    pub fn from_slice(shape: &Shape, data: &[f64]) -> Option<Self> {
        let numel = shape.numel()?;
        if data.len() != numel {
            return None;
        }
        let strides = compute_strides(shape);
        Some(Self {
            data: data.to_vec(),
            shape: shape.clone(),
            strides,
        })
    }

    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let shape = Shape::matrix(n, n);
        let mut t = Self::zeros(&shape);
        for i in 0..n {
            t.set(&[i, i], 1.0);
        }
        t
    }

    /// Create tensor with range of values
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let n = ((end - start) / step).ceil() as usize;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        let shape = Shape::vector(n);
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create tensor with linearly spaced values
    pub fn linspace(start: f64, end: f64, n: usize) -> Self {
        let step = if n > 1 {
            (end - start) / (n - 1) as f64
        } else {
            0.0
        };
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        let shape = Shape::vector(n);
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Transpose (swap last two dimensions)
    pub fn transpose(&self) -> Self {
        if self.shape.rank() < 2 {
            return self.clone();
        }

        let new_shape = self.shape.transpose().unwrap();
        let mut result = Self::zeros(&new_shape);

        let rows = self
            .shape
            .dim(self.shape.rank() - 2)
            .unwrap()
            .value()
            .unwrap();
        let cols = self
            .shape
            .dim(self.shape.rank() - 1)
            .unwrap()
            .value()
            .unwrap();

        for i in 0..rows {
            for j in 0..cols {
                result.set(&[j, i], *self.get(&[i, j]).unwrap());
            }
        }

        result
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Option<Self> {
        let result_shape = self.shape.matmul_result(&other.shape)?;

        // Handle 2D matrix multiply
        if self.shape.rank() == 2 && other.shape.rank() == 2 {
            let m = self.shape.dim(0)?.value()?;
            let k = self.shape.dim(1)?.value()?;
            let n = other.shape.dim(1)?.value()?;

            let mut result = Self::zeros(&result_shape);

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += self.get(&[i, l])? * other.get(&[l, j])?;
                    }
                    result.set(&[i, j], sum);
                }
            }

            return Some(result);
        }

        // TODO: Handle vector and batched cases
        None
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.shape != other.shape {
            // TODO: Broadcasting
            return None;
        }

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Some(Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Option<Self> {
        if self.shape != other.shape {
            return None;
        }

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Some(Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Self) -> Option<Self> {
        if self.shape != other.shape {
            return None;
        }

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Some(Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        self.sum() / self.numel() as f64
    }

    /// Maximum element
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum element
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Frobenius norm
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Apply function element-wise
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Apply function element-wise with index
    pub fn map_indexed<F>(&self, f: F) -> Self
    where
        F: Fn(&[usize], f64) -> f64,
    {
        let mut result = self.clone();
        let shape_dims: Vec<usize> = self.shape.dims.iter().filter_map(|d| d.value()).collect();

        for (i, x) in self.data.iter().enumerate() {
            let indices = flat_to_indices(i, &shape_dims);
            result.data[i] = f(&indices, *x);
        }
        result
    }
}

impl fmt::Debug for Tensor<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({}, {:?})", self.shape, self.data)
    }
}

impl fmt::Display for Tensor<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{}", self.shape)?;
        if self.shape.rank() <= 2 && self.numel() <= 100 {
            write!(f, " {:?}", self.data)?;
        }
        Ok(())
    }
}

/// Compute row-major strides for shape
fn compute_strides(shape: &Shape) -> Vec<usize> {
    let mut strides = vec![1; shape.rank()];
    for i in (0..shape.rank().saturating_sub(1)).rev() {
        let dim = shape.dim(i + 1).and_then(|d| d.value()).unwrap_or(1);
        strides[i] = strides[i + 1] * dim;
    }
    strides
}

/// Convert flat index to multi-index
fn flat_to_indices(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = flat;
    for i in (0..shape.len()).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

/// Shape verification result for type checking
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// Dimension mismatch
    DimensionMismatch {
        expected: Dim,
        found: Dim,
        axis: usize,
    },
    /// Rank mismatch
    RankMismatch { expected: usize, found: usize },
    /// Incompatible for operation
    IncompatibleShapes {
        lhs: Shape,
        rhs: Shape,
        operation: String,
    },
    /// Cannot determine shape
    UnknownShape,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::DimensionMismatch {
                expected,
                found,
                axis,
            } => {
                write!(
                    f,
                    "dimension mismatch at axis {}: expected {:?}, found {:?}",
                    axis, expected, found
                )
            }
            ShapeError::RankMismatch { expected, found } => {
                write!(f, "rank mismatch: expected {}, found {}", expected, found)
            }
            ShapeError::IncompatibleShapes {
                lhs,
                rhs,
                operation,
            } => {
                write!(
                    f,
                    "incompatible shapes {} and {} for {}",
                    lhs, rhs, operation
                )
            }
            ShapeError::UnknownShape => {
                write!(f, "cannot determine tensor shape")
            }
        }
    }
}

/// Verify shapes are compatible for an operation (for type checker)
pub fn verify_matmul(lhs: &Shape, rhs: &Shape) -> Result<Shape, ShapeError> {
    lhs.matmul_result(rhs)
        .ok_or_else(|| ShapeError::IncompatibleShapes {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            operation: "matrix multiplication".to_string(),
        })
}

/// Verify shapes for element-wise operation
pub fn verify_elementwise(lhs: &Shape, rhs: &Shape) -> Result<Shape, ShapeError> {
    lhs.broadcast_with(rhs)
        .ok_or_else(|| ShapeError::IncompatibleShapes {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            operation: "element-wise".to_string(),
        })
}

/// Verify reshape is valid
pub fn verify_reshape(from: &Shape, to: &Shape) -> Result<Shape, ShapeError> {
    match (from.numel(), to.numel()) {
        (Some(a), Some(b)) if a == b => Ok(to.clone()),
        _ => Err(ShapeError::IncompatibleShapes {
            lhs: from.clone(),
            rhs: to.clone(),
            operation: "reshape".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), Some(24));
    }

    #[test]
    fn test_shape_broadcast() {
        let a = Shape::new(&[3, 1]);
        let b = Shape::new(&[1, 4]);

        assert!(a.broadcast_compatible(&b));
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c, Shape::new(&[3, 4]));
    }

    #[test]
    fn test_matmul_shapes() {
        let a = Shape::matrix(3, 4);
        let b = Shape::matrix(4, 5);

        assert!(a.matmul_compatible(&b));
        let c = a.matmul_result(&b).unwrap();
        assert_eq!(c, Shape::matrix(3, 5));
    }

    #[test]
    fn test_matmul_incompatible() {
        let a = Shape::matrix(3, 4);
        let b = Shape::matrix(5, 6);

        assert!(!a.matmul_compatible(&b));
    }

    #[test]
    fn test_tensor_creation() {
        let shape = Shape::matrix(2, 3);
        let t = Tensor::<f64>::zeros(&shape);

        assert_eq!(t.numel(), 6);
        assert_eq!(t.shape().rank(), 2);
    }

    #[test]
    fn test_tensor_indexing() {
        let shape = Shape::matrix(2, 3);
        let mut t = Tensor::<f64>::zeros(&shape);

        t.set(&[0, 1], 5.0);
        assert_eq!(t.get(&[0, 1]), Some(&5.0));
        assert_eq!(t.get(&[0, 0]), Some(&0.0));
    }

    #[test]
    fn test_tensor_matmul() {
        // 2x3 matrix
        let a = Tensor::from_slice(&Shape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // 3x2 matrix
        let b =
            Tensor::from_slice(&Shape::matrix(3, 2), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &Shape::matrix(2, 2));
        // [1,2,3] @ [7,9,11; 8,10,12] = [58, 64]
        // [4,5,6] @ [7,9,11; 8,10,12] = [139, 154]
        assert_eq!(c.get(&[0, 0]), Some(&58.0));
        assert_eq!(c.get(&[0, 1]), Some(&64.0));
        assert_eq!(c.get(&[1, 0]), Some(&139.0));
        assert_eq!(c.get(&[1, 1]), Some(&154.0));
    }

    #[test]
    fn test_tensor_transpose() {
        let a = Tensor::from_slice(&Shape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let b = a.transpose();

        assert_eq!(b.shape(), &Shape::matrix(3, 2));
        assert_eq!(b.get(&[0, 0]), Some(&1.0));
        assert_eq!(b.get(&[0, 1]), Some(&4.0));
        assert_eq!(b.get(&[1, 0]), Some(&2.0));
    }

    #[test]
    fn test_tensor_operations() {
        let a = Tensor::from_slice(&Shape::vector(3), &[1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::from_slice(&Shape::vector(3), &[4.0, 5.0, 6.0]).unwrap();

        let c = a.add(&b).unwrap();
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);

        let d = a.hadamard(&b).unwrap();
        assert_eq!(d.data(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_tensor_stats() {
        let a = Tensor::from_slice(&Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(a.sum(), 10.0);
        assert_eq!(a.mean(), 2.5);
        assert_eq!(a.min(), 1.0);
        assert_eq!(a.max(), 4.0);
    }

    #[test]
    fn test_identity() {
        let eye = Tensor::eye(3);
        assert_eq!(eye.get(&[0, 0]), Some(&1.0));
        assert_eq!(eye.get(&[1, 1]), Some(&1.0));
        assert_eq!(eye.get(&[2, 2]), Some(&1.0));
        assert_eq!(eye.get(&[0, 1]), Some(&0.0));
    }

    #[test]
    fn test_linspace() {
        let v = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(v.numel(), 5);
        assert!((v.data()[0] - 0.0).abs() < 1e-10);
        assert!((v.data()[4] - 1.0).abs() < 1e-10);
        assert!((v.data()[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_shape_verification() {
        let a = Shape::matrix(3, 4);
        let b = Shape::matrix(4, 5);
        let c = Shape::matrix(5, 6);

        assert!(verify_matmul(&a, &b).is_ok());
        assert!(verify_matmul(&a, &c).is_err());
    }
}
