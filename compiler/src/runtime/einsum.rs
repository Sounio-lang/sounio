//! Einstein Notation (einsum) Runtime
//!
//! This module provides Einstein summation convention for tensor operations,
//! enabling concise expression of complex tensor contractions.
//!
//! # Notation
//!
//! - `"ij,jk->ik"` - Matrix multiplication
//! - `"ii->"` - Trace
//! - `"ij->ji"` - Transpose
//! - `"ijk,ikl->ijl"` - Batched matrix multiplication
//! - `"i,j->ij"` - Outer product
//! - `"ij,ij->"` - Frobenius inner product
//!
//! # Example
//!
//! ```d
//! let A: Tensor<f64, [3, 4]> = ...;
//! let B: Tensor<f64, [4, 5]> = ...;
//! let C = einsum("ij,jk->ik", A, B);  // C: [3, 5]
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Parsed einsum expression
#[derive(Debug, Clone)]
pub struct EinsumExpr {
    /// Input subscripts for each operand
    pub inputs: Vec<Vec<char>>,
    /// Output subscripts
    pub output: Vec<char>,
    /// All indices that appear
    pub all_indices: HashSet<char>,
    /// Contracted (summed) indices
    pub contracted: HashSet<char>,
    /// Batch indices (appear in all inputs and output)
    pub batch: HashSet<char>,
}

impl EinsumExpr {
    /// Parse einsum notation string
    /// Format: "inputs->output" where inputs are comma-separated
    pub fn parse(notation: &str) -> Result<Self, EinsumError> {
        let parts: Vec<&str> = notation.split("->").collect();
        if parts.len() != 2 {
            return Err(EinsumError::InvalidNotation(
                "Expected format 'inputs->output'".to_string(),
            ));
        }

        let input_str = parts[0];
        let output_str = parts[1];

        let inputs: Vec<Vec<char>> = input_str
            .split(',')
            .map(|s| s.chars().filter(|c| c.is_alphabetic()).collect())
            .collect();

        let output: Vec<char> = output_str.chars().filter(|c| c.is_alphabetic()).collect();

        // Collect all indices
        let mut all_indices = HashSet::new();
        for input in &inputs {
            for &c in input {
                all_indices.insert(c);
            }
        }

        // Contracted indices: appear in inputs but not in output
        let output_set: HashSet<char> = output.iter().copied().collect();
        let contracted: HashSet<char> = all_indices
            .iter()
            .filter(|c| !output_set.contains(c))
            .copied()
            .collect();

        // Batch indices: appear in all inputs and in output
        let mut batch = output_set.clone();
        for input in &inputs {
            let input_set: HashSet<char> = input.iter().copied().collect();
            batch = batch.intersection(&input_set).copied().collect();
        }

        Ok(Self {
            inputs,
            output,
            all_indices,
            contracted,
            batch,
        })
    }

    /// Check if this is a valid expression for given input shapes
    pub fn validate(&self, shapes: &[&[usize]]) -> Result<Vec<usize>, EinsumError> {
        if shapes.len() != self.inputs.len() {
            return Err(EinsumError::OperandCountMismatch {
                expected: self.inputs.len(),
                got: shapes.len(),
            });
        }

        // Build index -> size mapping
        let mut index_sizes: HashMap<char, usize> = HashMap::new();

        for (input, shape) in self.inputs.iter().zip(shapes.iter()) {
            if input.len() != shape.len() {
                return Err(EinsumError::RankMismatch {
                    subscripts: input.clone(),
                    shape: shape.to_vec(),
                });
            }

            for (&idx, &size) in input.iter().zip(shape.iter()) {
                if let Some(&existing) = index_sizes.get(&idx) {
                    if existing != size {
                        return Err(EinsumError::DimensionMismatch {
                            index: idx,
                            sizes: vec![existing, size],
                        });
                    }
                } else {
                    index_sizes.insert(idx, size);
                }
            }
        }

        // Compute output shape
        let output_shape: Vec<usize> = self
            .output
            .iter()
            .map(|&idx| {
                index_sizes
                    .get(&idx)
                    .copied()
                    .ok_or(EinsumError::InvalidNotation(format!(
                        "Output index '{}' not found in inputs",
                        idx
                    )))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(output_shape)
    }
}

/// Einsum error types
#[derive(Debug, Clone)]
pub enum EinsumError {
    InvalidNotation(String),
    OperandCountMismatch {
        expected: usize,
        got: usize,
    },
    RankMismatch {
        subscripts: Vec<char>,
        shape: Vec<usize>,
    },
    DimensionMismatch {
        index: char,
        sizes: Vec<usize>,
    },
}

impl fmt::Display for EinsumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EinsumError::InvalidNotation(msg) => write!(f, "Invalid einsum notation: {}", msg),
            EinsumError::OperandCountMismatch { expected, got } => {
                write!(f, "Expected {} operands, got {}", expected, got)
            }
            EinsumError::RankMismatch { subscripts, shape } => {
                write!(
                    f,
                    "Subscripts {:?} don't match shape {:?}",
                    subscripts, shape
                )
            }
            EinsumError::DimensionMismatch { index, sizes } => {
                write!(f, "Index '{}' has inconsistent sizes: {:?}", index, sizes)
            }
        }
    }
}

impl std::error::Error for EinsumError {}

/// Execute einsum operation
pub fn einsum(
    notation: &str,
    operands: &[&[f64]],
    shapes: &[&[usize]],
) -> Result<(Vec<f64>, Vec<usize>), EinsumError> {
    let expr = EinsumExpr::parse(notation)?;
    let output_shape = expr.validate(shapes)?;

    // Special case optimizations
    if expr.inputs.len() == 2 {
        // Check for common patterns
        if let Some(result) = try_optimized_binary(&expr, operands, shapes) {
            return Ok((result, output_shape));
        }
    }

    // General implementation using nested loops
    let result = einsum_general(&expr, operands, shapes, &output_shape);
    Ok((result, output_shape))
}

/// Try optimized implementations for common binary operations
fn try_optimized_binary(
    expr: &EinsumExpr,
    operands: &[&[f64]],
    shapes: &[&[usize]],
) -> Option<Vec<f64>> {
    let a = operands[0];
    let b = operands[1];

    // Matrix multiplication: ij,jk->ik
    if expr.inputs[0] == ['i', 'j'] && expr.inputs[1] == ['j', 'k'] && expr.output == ['i', 'k'] {
        let m = shapes[0][0];
        let k = shapes[0][1];
        let n = shapes[1][1];
        return Some(matmul(a, b, m, k, n));
    }

    // Batched matmul: bij,bjk->bik
    if expr.inputs[0] == ['b', 'i', 'j']
        && expr.inputs[1] == ['b', 'j', 'k']
        && expr.output == ['b', 'i', 'k']
    {
        let batch = shapes[0][0];
        let m = shapes[0][1];
        let k = shapes[0][2];
        let n = shapes[1][2];
        return Some(batched_matmul(a, b, batch, m, k, n));
    }

    // Outer product: i,j->ij
    if expr.inputs[0] == ['i'] && expr.inputs[1] == ['j'] && expr.output == ['i', 'j'] {
        let m = shapes[0][0];
        let n = shapes[1][0];
        return Some(outer_product(a, b, m, n));
    }

    // Dot product: i,i->
    if expr.inputs[0] == ['i'] && expr.inputs[1] == ['i'] && expr.output.is_empty() {
        return Some(vec![dot_product(a, b)]);
    }

    // Element-wise multiply with sum: ij,ij->
    if expr.inputs[0] == ['i', 'j'] && expr.inputs[1] == ['i', 'j'] && expr.output.is_empty() {
        return Some(vec![a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()]);
    }

    None
}

/// General einsum implementation
fn einsum_general(
    expr: &EinsumExpr,
    operands: &[&[f64]],
    shapes: &[&[usize]],
    output_shape: &[usize],
) -> Vec<f64> {
    // Build index -> size mapping
    let mut index_sizes: HashMap<char, usize> = HashMap::new();
    for (input, shape) in expr.inputs.iter().zip(shapes.iter()) {
        for (&idx, &size) in input.iter().zip(shape.iter()) {
            index_sizes.insert(idx, size);
        }
    }

    // Compute output strides
    let output_size: usize = output_shape.iter().product();
    let mut result = vec![0.0; output_size.max(1)];

    let output_strides = compute_strides(output_shape);

    // Input strides
    let input_strides: Vec<Vec<usize>> = shapes.iter().map(|s| compute_strides(s)).collect();

    // All indices to iterate over
    let all_indices: Vec<char> = expr.all_indices.iter().copied().collect();
    let index_ranges: Vec<usize> = all_indices
        .iter()
        .map(|&c| *index_sizes.get(&c).unwrap_or(&1))
        .collect();

    // Iterate over all index combinations
    let mut indices = vec![0usize; all_indices.len()];
    let total_iters: usize = index_ranges.iter().product();

    for _ in 0..total_iters {
        // Compute product of operand elements
        let mut product = 1.0;
        for (op_idx, (operand, input_subscripts)) in
            operands.iter().zip(expr.inputs.iter()).enumerate()
        {
            let mut flat_idx = 0;
            for (sub_idx, &subscript) in input_subscripts.iter().enumerate() {
                let idx_pos = all_indices.iter().position(|&c| c == subscript).unwrap();
                flat_idx += indices[idx_pos] * input_strides[op_idx][sub_idx];
            }
            product *= operand[flat_idx];
        }

        // Compute output index
        let mut out_flat_idx = 0;
        for (sub_idx, &subscript) in expr.output.iter().enumerate() {
            let idx_pos = all_indices.iter().position(|&c| c == subscript).unwrap();
            out_flat_idx += indices[idx_pos] * output_strides[sub_idx];
        }

        if output_size > 0 {
            result[out_flat_idx] += product;
        } else {
            result[0] += product;
        }

        // Increment indices
        for i in (0..indices.len()).rev() {
            indices[i] += 1;
            if indices[i] < index_ranges[i] {
                break;
            }
            indices[i] = 0;
        }
    }

    result
}

/// Compute row-major strides
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Optimized implementations

fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn batched_matmul(a: &[f64], b: &[f64], batch: usize, m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; batch * m * n];
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let c_batch_stride = m * n;

    for bat in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum +=
                        a[bat * a_batch_stride + i * k + l] * b[bat * b_batch_stride + l * n + j];
                }
                c[bat * c_batch_stride + i * n + j] = sum;
            }
        }
    }
    c
}

fn outer_product(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = a[i] * b[j];
        }
    }
    c
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Trace of a matrix: ii->
pub fn trace(a: &[f64], n: usize) -> f64 {
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Transpose: ij->ji
pub fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut b = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            b[j * m + i] = a[i * n + j];
        }
    }
    b
}

/// Diagonal extraction: ii->i
pub fn diagonal(a: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Sum over axis: ij->i (sum over j)
pub fn sum_axis(a: &[f64], m: usize, n: usize, axis: usize) -> Vec<f64> {
    if axis == 1 {
        // Sum over columns (j)
        (0..m).map(|i| (0..n).map(|j| a[i * n + j]).sum()).collect()
    } else {
        // Sum over rows (i)
        (0..n).map(|j| (0..m).map(|i| a[i * n + j]).sum()).collect()
    }
}

/// Contract tensors along specified axes
pub fn tensordot(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    axes_a: &[usize],
    axes_b: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    // Build einsum notation from axes
    let mut notation = String::new();
    let mut next_idx = 'a';

    // First operand subscripts
    let mut a_subscripts = Vec::new();
    for i in 0..a_shape.len() {
        a_subscripts.push(next_idx);
        next_idx = (next_idx as u8 + 1) as char;
    }
    notation.push_str(&a_subscripts.iter().collect::<String>());
    notation.push(',');

    // Second operand subscripts
    let mut b_subscripts = Vec::new();
    for i in 0..b_shape.len() {
        if let Some(pos) = axes_b.iter().position(|&x| x == i) {
            // Use the same index as the corresponding axis in a
            b_subscripts.push(a_subscripts[axes_a[pos]]);
        } else {
            b_subscripts.push(next_idx);
            next_idx = (next_idx as u8 + 1) as char;
        }
    }
    notation.push_str(&b_subscripts.iter().collect::<String>());
    notation.push_str("->");

    // Output subscripts (non-contracted from both)
    for (i, &c) in a_subscripts.iter().enumerate() {
        if !axes_a.contains(&i) {
            notation.push(c);
        }
    }
    for (i, &c) in b_subscripts.iter().enumerate() {
        if !axes_b.contains(&i) {
            notation.push(c);
        }
    }

    einsum(&notation, &[a, b], &[a_shape, b_shape]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_matmul() {
        let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
        assert_eq!(expr.inputs.len(), 2);
        assert_eq!(expr.inputs[0], vec!['i', 'j']);
        assert_eq!(expr.inputs[1], vec!['j', 'k']);
        assert_eq!(expr.output, vec!['i', 'k']);
        assert!(expr.contracted.contains(&'j'));
    }

    #[test]
    fn test_parse_trace() {
        let expr = EinsumExpr::parse("ii->").unwrap();
        assert_eq!(expr.inputs[0], vec!['i', 'i']);
        assert!(expr.output.is_empty());
    }

    #[test]
    fn test_validate() {
        let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
        let shapes = vec![&[3usize, 4][..], &[4usize, 5][..]];
        let output_shape = expr.validate(&shapes).unwrap();
        assert_eq!(output_shape, vec![3, 5]);
    }

    #[test]
    fn test_validate_mismatch() {
        let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
        let shapes = vec![&[3usize, 4][..], &[5usize, 6][..]]; // j sizes don't match
        assert!(expr.validate(&shapes).is_err());
    }

    #[test]
    fn test_einsum_matmul() {
        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let (result, shape) = einsum("ij,jk->ik", &[&a, &b], &[&[2, 3], &[3, 2]]).unwrap();

        assert_eq!(shape, vec![2, 2]);
        // [1,2,3] @ [7,9,11; 8,10,12]^T = [58, 64; 139, 154]
        assert_eq!(result[0], 58.0);
        assert_eq!(result[1], 64.0);
        assert_eq!(result[2], 139.0);
        assert_eq!(result[3], 154.0);
    }

    #[test]
    fn test_einsum_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let (result, shape) = einsum("i,i->", &[&a, &b], &[&[3], &[3]]).unwrap();

        assert!(shape.is_empty());
        assert_eq!(result[0], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_einsum_outer() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];

        let (result, shape) = einsum("i,j->ij", &[&a, &b], &[&[2], &[3]]).unwrap();

        assert_eq!(shape, vec![2, 3]);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_einsum_trace() {
        let a = vec![1.0, 2.0, 3.0, 4.0];

        let (result, shape) = einsum("ii->", &[&a], &[&[2, 2]]).unwrap();

        assert!(shape.is_empty());
        assert_eq!(result[0], 5.0); // 1 + 4
    }

    #[test]
    fn test_einsum_transpose() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (result, shape) = einsum("ij->ji", &[&a], &[&[2, 3]]).unwrap();

        assert_eq!(shape, vec![3, 2]);
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_trace_fn() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(trace(&a, 3), 15.0); // 1 + 5 + 9
    }

    #[test]
    fn test_transpose_fn() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = transpose(&a, 2, 3);
        assert_eq!(b, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_diagonal() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let d = diagonal(&a, 3);
        assert_eq!(d, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_sum_axis() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Sum over columns (axis 1): [6, 15]
        let sum_cols = sum_axis(&a, 2, 3, 1);
        assert_eq!(sum_cols, vec![6.0, 15.0]);

        // Sum over rows (axis 0): [5, 7, 9]
        let sum_rows = sum_axis(&a, 2, 3, 0);
        assert_eq!(sum_rows, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_batched_matmul() {
        // 2 batches of 2x3 @ 3x2
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
        ];

        let (result, shape) = einsum("bij,bjk->bik", &[&a, &b], &[&[2, 2, 3], &[2, 3, 2]]).unwrap();

        assert_eq!(shape, vec![2, 2, 2]);
    }

    #[test]
    fn test_tensordot() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        // Contract last axis of a with first axis of b
        let (result, shape) = tensordot(&a, &[2, 3], &b, &[3, 2], &[1], &[0]);

        assert_eq!(shape, vec![2, 2]);
    }
}
