//! Dimension Solver for Scientific Types
//!
//! Provides symbolic dimension unification and shape inference for:
//! - Array<T, N> with const generic dimensions
//! - Matrix<T, M, N> with parametric row/column counts
//! - Tensor shapes in operations like matmul: [M, K] @ [K, N] -> [M, N]
//!
//! The solver supports:
//! - Unification of concrete, symbolic, and inferred dimensions
//! - Shape inference for binary operations (matmul, broadcast, concat)
//! - Constraint collection and solving for type inference
//!
//! # Example
//!
//! ```ignore
//! fn matmul<M, K, N>(a: Matrix<f64, M, K>, b: Matrix<f64, K, N>) -> Matrix<f64, M, N>
//! ```
//!
//! The dimension solver unifies the inner dimension K and produces the result shape [M, N].

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::collections::HashMap;

use super::core::{DimOp, DimSize, DimVar, TensorShape};

/// Result type for dimension solving operations
pub type DimResult<T> = Result<T, DimensionError>;

/// Errors that can occur during dimension solving
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionError {
    /// Dimension mismatch: expected N, found M
    Mismatch { expected: DimSize, found: DimSize },
    /// Shape rank mismatch: expected N dimensions, found M
    RankMismatch { expected: usize, found: usize },
    /// Inner dimension mismatch for matrix multiplication
    MatmulInnerMismatch {
        left_cols: DimSize,
        right_rows: DimSize,
    },
    /// Cannot broadcast shapes
    BroadcastIncompatible {
        left: Vec<DimSize>,
        right: Vec<DimSize>,
    },
    /// Infinite dimension (occurs check failed)
    InfiniteDimension(DimVar),
    /// Cannot infer dimension
    CannotInfer { context: String },
    /// Division by zero in dimension computation
    DivisionByZero,
    /// Negative dimension after computation
    NegativeDimension { value: i64 },
    /// Symbolic dimension constraint violation
    SymbolicConstraint { name: String, message: String },
}

impl std::fmt::Display for DimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mismatch { expected, found } => {
                write!(
                    f,
                    "dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            Self::RankMismatch { expected, found } => {
                write!(
                    f,
                    "rank mismatch: expected {} dimensions, found {}",
                    expected, found
                )
            }
            Self::MatmulInnerMismatch {
                left_cols,
                right_rows,
            } => {
                write!(
                    f,
                    "matrix multiplication inner dimension mismatch: {} vs {}",
                    left_cols, right_rows
                )
            }
            Self::BroadcastIncompatible { left, right } => {
                let left_str: Vec<_> = left.iter().map(|d| d.to_string()).collect();
                let right_str: Vec<_> = right.iter().map(|d| d.to_string()).collect();
                write!(
                    f,
                    "cannot broadcast shapes [{}] and [{}]",
                    left_str.join(", "),
                    right_str.join(", ")
                )
            }
            Self::InfiniteDimension(var) => {
                write!(
                    f,
                    "infinite dimension: ?D{} occurs in its own definition",
                    var.0
                )
            }
            Self::CannotInfer { context } => {
                write!(f, "cannot infer dimension: {}", context)
            }
            Self::DivisionByZero => write!(f, "division by zero in dimension computation"),
            Self::NegativeDimension { value } => {
                write!(f, "negative dimension: {}", value)
            }
            Self::SymbolicConstraint { name, message } => {
                write!(f, "constraint on '{}': {}", name, message)
            }
        }
    }
}

impl std::error::Error for DimensionError {}

/// Dimension solver for unifying and inferring array/matrix dimensions
pub struct DimensionSolver {
    /// Counter for generating fresh dimension variables
    next_var: u32,
    /// Substitution map for dimension variables
    substitutions: HashMap<DimVar, DimSize>,
    /// Symbolic dimension bindings (name -> DimSize)
    symbolic_bindings: HashMap<String, DimSize>,
    /// Constraints collected during type checking
    constraints: Vec<DimConstraint>,
}

/// A constraint between two dimensions
#[derive(Debug, Clone)]
pub struct DimConstraint {
    pub left: DimSize,
    pub right: DimSize,
    pub context: String,
}

impl DimensionSolver {
    /// Create a new dimension solver
    pub fn new() -> Self {
        Self {
            next_var: 0,
            substitutions: HashMap::new(),
            symbolic_bindings: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Generate a fresh dimension variable
    pub fn fresh_var(&mut self) -> DimVar {
        let var = DimVar(self.next_var);
        self.next_var += 1;
        var
    }

    /// Generate a fresh symbolic dimension with a given prefix
    pub fn fresh_symbolic(&mut self, prefix: &str) -> DimSize {
        let name = format!("{}_{}", prefix, self.next_var);
        self.next_var += 1;
        DimSize::Symbolic(name)
    }

    /// Bind a symbolic dimension name to a concrete or inferred dimension
    pub fn bind_symbolic(&mut self, name: &str, dim: DimSize) -> DimResult<()> {
        if let Some(existing) = self.symbolic_bindings.get(name) {
            // Already bound - must unify
            self.unify(existing.clone(), dim)?;
        } else {
            self.symbolic_bindings.insert(name.to_string(), dim);
        }
        Ok(())
    }

    /// Look up a symbolic dimension binding
    pub fn lookup_symbolic(&self, name: &str) -> Option<&DimSize> {
        self.symbolic_bindings.get(name)
    }

    /// Apply substitutions to a dimension
    pub fn apply(&self, dim: &DimSize) -> DimSize {
        match dim {
            DimSize::Var(var) => {
                if let Some(subst) = self.substitutions.get(var) {
                    self.apply(subst)
                } else {
                    dim.clone()
                }
            }
            DimSize::Symbolic(name) => {
                if let Some(bound) = self.symbolic_bindings.get(name) {
                    self.apply(bound)
                } else {
                    dim.clone()
                }
            }
            DimSize::BinOp { op, left, right } => {
                let l = self.apply(left);
                let r = self.apply(right);
                // Try to simplify if both are now constants
                if let (Some(lv), Some(rv)) = (l.as_const(), r.as_const()) {
                    let result = match op {
                        DimOp::Add => lv.checked_add(rv),
                        DimOp::Sub => lv.checked_sub(rv),
                        DimOp::Mul => lv.checked_mul(rv),
                        DimOp::Div => {
                            if rv == 0 {
                                None
                            } else {
                                Some(lv / rv)
                            }
                        }
                        DimOp::Max => Some(lv.max(rv)),
                        DimOp::Min => Some(lv.min(rv)),
                    };
                    if let Some(v) = result {
                        return DimSize::Const(v);
                    }
                }
                DimSize::BinOp {
                    op: *op,
                    left: Box::new(l),
                    right: Box::new(r),
                }
            }
            _ => dim.clone(),
        }
    }

    /// Unify two dimensions, updating substitutions
    pub fn unify(&mut self, a: DimSize, b: DimSize) -> DimResult<()> {
        let a = self.apply(&a);
        let b = self.apply(&b);

        match (&a, &b) {
            // Same dimension
            _ if a == b => Ok(()),

            // Const dimensions must match exactly
            (DimSize::Const(n1), DimSize::Const(n2)) => {
                if n1 == n2 {
                    Ok(())
                } else {
                    Err(DimensionError::Mismatch {
                        expected: a.clone(),
                        found: b.clone(),
                    })
                }
            }

            // Variable unification
            (DimSize::Var(var), other) | (other, DimSize::Var(var)) => {
                // Occurs check
                if self.occurs_in(*var, other) {
                    return Err(DimensionError::InfiniteDimension(*var));
                }
                self.substitutions.insert(*var, other.clone());
                Ok(())
            }

            // Symbolic with same name
            (DimSize::Symbolic(s1), DimSize::Symbolic(s2)) if s1 == s2 => Ok(()),

            // Symbolic with const - bind the symbolic
            (DimSize::Symbolic(name), DimSize::Const(n))
            | (DimSize::Const(n), DimSize::Symbolic(name)) => {
                self.bind_symbolic(name, DimSize::Const(*n))
            }

            // NOTE: Symbolic with var is already handled by the DimSize::Var arm above
            // since it comes first and matches (Var, anything) or (anything, Var)

            // Different symbolics - create constraint
            (DimSize::Symbolic(s1), DimSize::Symbolic(s2)) => {
                // Bind s1 to s2's value (or create a shared var)
                let var = self.fresh_var();
                self.bind_symbolic(s1, DimSize::Var(var))?;
                self.bind_symbolic(s2, DimSize::Var(var))
            }

            // Dynamic is compatible with anything (runtime check)
            (DimSize::Dynamic, _) | (_, DimSize::Dynamic) => Ok(()),

            // BinOp - try to unify structurally or evaluate
            (DimSize::BinOp { .. }, _) | (_, DimSize::BinOp { .. }) => {
                // Try to evaluate both sides
                if let (Some(n1), Some(n2)) = (a.try_eval(), b.try_eval()) {
                    if n1 == n2 {
                        Ok(())
                    } else {
                        Err(DimensionError::Mismatch {
                            expected: a.clone(),
                            found: b.clone(),
                        })
                    }
                } else {
                    // Cannot unify complex expressions statically
                    // Add as constraint for later solving
                    self.constraints.push(DimConstraint {
                        left: a.clone(),
                        right: b.clone(),
                        context: "binary operation unification".to_string(),
                    });
                    Ok(())
                }
            }
        }
    }

    /// Check if a variable occurs in a dimension (for occurs check)
    fn occurs_in(&self, var: DimVar, dim: &DimSize) -> bool {
        match dim {
            DimSize::Var(v) => {
                if *v == var {
                    true
                } else if let Some(subst) = self.substitutions.get(v) {
                    self.occurs_in(var, subst)
                } else {
                    false
                }
            }
            DimSize::BinOp { left, right, .. } => {
                self.occurs_in(var, left) || self.occurs_in(var, right)
            }
            _ => false,
        }
    }

    /// Add a constraint to be solved later
    pub fn add_constraint(&mut self, left: DimSize, right: DimSize, context: &str) {
        self.constraints.push(DimConstraint {
            left,
            right,
            context: context.to_string(),
        });
    }

    /// Solve all collected constraints
    pub fn solve_constraints(&mut self) -> DimResult<()> {
        // Iteratively solve constraints until fixed point
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            let constraints = std::mem::take(&mut self.constraints);
            for constraint in constraints {
                let left = self.apply(&constraint.left);
                let right = self.apply(&constraint.right);

                if left != right {
                    // Try to unify
                    match self.unify(left.clone(), right.clone()) {
                        Ok(()) => changed = true,
                        Err(e) => {
                            // Re-add constraint if not solvable yet
                            if !matches!(e, DimensionError::Mismatch { .. }) {
                                self.constraints.push(DimConstraint {
                                    left,
                                    right,
                                    context: constraint.context,
                                });
                            } else {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // Shape Operations
    // ========================================================================

    /// Unify two tensor shapes
    pub fn unify_shapes(&mut self, a: &TensorShape, b: &TensorShape) -> DimResult<TensorShape> {
        let dims_a = a.to_parametric();
        let dims_b = b.to_parametric();

        if dims_a.len() != dims_b.len() {
            return Err(DimensionError::RankMismatch {
                expected: dims_a.len(),
                found: dims_b.len(),
            });
        }

        let mut result = Vec::with_capacity(dims_a.len());
        for (da, db) in dims_a.into_iter().zip(dims_b.into_iter()) {
            self.unify(da.clone(), db.clone())?;
            result.push(self.apply(&da));
        }

        Ok(TensorShape::from_dims(result))
    }

    /// Infer the result shape for matrix multiplication
    ///
    /// For [M, K] @ [K, N] -> [M, N]
    pub fn matmul_shape(
        &mut self,
        left: &TensorShape,
        right: &TensorShape,
    ) -> DimResult<TensorShape> {
        let left_dims = left.to_parametric();
        let right_dims = right.to_parametric();

        // Need at least 2D for matmul
        if left_dims.len() < 2 || right_dims.len() < 2 {
            return Err(DimensionError::RankMismatch {
                expected: 2,
                found: left_dims.len().min(right_dims.len()),
            });
        }

        // Get the relevant dimensions
        let m = &left_dims[left_dims.len() - 2];
        let k1 = &left_dims[left_dims.len() - 1];
        let k2 = &right_dims[right_dims.len() - 2];
        let n = &right_dims[right_dims.len() - 1];

        // Unify inner dimension K
        self.unify(k1.clone(), k2.clone())
            .map_err(|_| DimensionError::MatmulInnerMismatch {
                left_cols: k1.clone(),
                right_rows: k2.clone(),
            })?;

        // Build result shape: batch dims + [M, N]
        let mut result = Vec::new();

        // Handle batch dimensions (broadcasting)
        let left_batch = &left_dims[..left_dims.len() - 2];
        let right_batch = &right_dims[..right_dims.len() - 2];
        let batch = self.broadcast_dims(left_batch, right_batch)?;
        result.extend(batch);

        // Add M and N
        result.push(self.apply(m));
        result.push(self.apply(n));

        Ok(TensorShape::from_dims(result))
    }

    /// Broadcast two dimension lists according to NumPy-style broadcasting rules
    pub fn broadcast_dims(
        &mut self,
        left: &[DimSize],
        right: &[DimSize],
    ) -> DimResult<Vec<DimSize>> {
        let max_len = left.len().max(right.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            // Use checked subtraction to properly handle rank extension
            // When i+1 > len, the dimension is missing (implicitly 1)
            let ld = if i + 1 <= left.len() {
                Some(&left[left.len() - 1 - i])
            } else {
                None
            };
            let rd = if i + 1 <= right.len() {
                Some(&right[right.len() - 1 - i])
            } else {
                None
            };

            let dim = match (ld, rd) {
                // Missing dimension is implicitly 1, so use the other
                (None, Some(d)) | (Some(d), None) => d.clone(),
                // If either is 1, use the other (broadcasting rule)
                (Some(DimSize::Const(1)), Some(d)) | (Some(d), Some(DimSize::Const(1))) => {
                    d.clone()
                }
                (Some(a), Some(b)) => {
                    // Try to unify (same dimensions)
                    match self.unify(a.clone(), b.clone()) {
                        Ok(()) => self.apply(a),
                        Err(_) => {
                            // Check if either is 1 (after resolution)
                            let a_resolved = self.apply(a);
                            let b_resolved = self.apply(b);
                            if a_resolved.as_const() == Some(1) {
                                b_resolved
                            } else if b_resolved.as_const() == Some(1) {
                                a_resolved
                            } else {
                                return Err(DimensionError::BroadcastIncompatible {
                                    left: left.to_vec(),
                                    right: right.to_vec(),
                                });
                            }
                        }
                    }
                }
                (None, None) => unreachable!(),
            };
            result.push(dim);
        }

        result.reverse();
        Ok(result)
    }

    /// Infer the result shape for element-wise operations with broadcasting
    pub fn broadcast_shape(
        &mut self,
        left: &TensorShape,
        right: &TensorShape,
    ) -> DimResult<TensorShape> {
        let left_dims = left.to_parametric();
        let right_dims = right.to_parametric();
        let result = self.broadcast_dims(&left_dims, &right_dims)?;
        Ok(TensorShape::from_dims(result))
    }

    /// Infer the result shape for concatenation along an axis
    pub fn concat_shape(&mut self, shapes: &[TensorShape], axis: usize) -> DimResult<TensorShape> {
        if shapes.is_empty() {
            return Err(DimensionError::CannotInfer {
                context: "concatenation requires at least one input".to_string(),
            });
        }

        let first = shapes[0].to_parametric();
        if axis >= first.len() {
            return Err(DimensionError::RankMismatch {
                expected: axis + 1,
                found: first.len(),
            });
        }

        let mut result = first.clone();
        let mut concat_dim = result[axis].clone();

        for shape in shapes.iter().skip(1) {
            let dims = shape.to_parametric();
            if dims.len() != first.len() {
                return Err(DimensionError::RankMismatch {
                    expected: first.len(),
                    found: dims.len(),
                });
            }

            // Check all non-axis dimensions match
            for (i, (d1, d2)) in result.iter().zip(dims.iter()).enumerate() {
                if i != axis {
                    self.unify(d1.clone(), d2.clone())?;
                }
            }

            // Add to concat dimension
            concat_dim = DimSize::BinOp {
                op: DimOp::Add,
                left: Box::new(concat_dim),
                right: Box::new(dims[axis].clone()),
            };
        }

        result[axis] = self.apply(&concat_dim);
        Ok(TensorShape::from_dims(result))
    }

    /// Infer the shape after slicing
    pub fn slice_shape(
        &mut self,
        shape: &TensorShape,
        axis: usize,
        start: Option<DimSize>,
        end: Option<DimSize>,
    ) -> DimResult<TensorShape> {
        let mut dims = shape.to_parametric();

        if axis >= dims.len() {
            return Err(DimensionError::RankMismatch {
                expected: axis + 1,
                found: dims.len(),
            });
        }

        let original = &dims[axis];
        let start = start.unwrap_or(DimSize::Const(0));
        let end = end.unwrap_or_else(|| original.clone());

        // New size = end - start
        dims[axis] = DimSize::BinOp {
            op: DimOp::Sub,
            left: Box::new(end),
            right: Box::new(start),
        };

        // Apply and simplify
        dims[axis] = self.apply(&dims[axis]);

        Ok(TensorShape::from_dims(dims))
    }

    /// Infer the shape after reshaping
    ///
    /// One dimension can be -1 (inferred from total size)
    pub fn reshape_shape(
        &mut self,
        original: &TensorShape,
        new_shape: &[DimSize],
    ) -> DimResult<TensorShape> {
        // Check if we can compute the original total size
        let orig_dims = original.to_parametric();
        let mut orig_total = DimSize::Const(1);
        for d in &orig_dims {
            orig_total = DimSize::BinOp {
                op: DimOp::Mul,
                left: Box::new(orig_total),
                right: Box::new(d.clone()),
            };
        }
        let orig_total = self.apply(&orig_total);

        // Find -1 dimension (if any) and compute known product
        let mut infer_idx = None;
        let mut known_product = DimSize::Const(1);

        for (i, d) in new_shape.iter().enumerate() {
            if let DimSize::Const(n) = d {
                if *n == usize::MAX {
                    // Convention: MAX represents -1 (infer)
                    if infer_idx.is_some() {
                        return Err(DimensionError::CannotInfer {
                            context: "only one dimension can be inferred in reshape".to_string(),
                        });
                    }
                    infer_idx = Some(i);
                    continue;
                }
            }
            known_product = DimSize::BinOp {
                op: DimOp::Mul,
                left: Box::new(known_product),
                right: Box::new(d.clone()),
            };
        }

        let mut result = new_shape.to_vec();
        if let Some(idx) = infer_idx {
            // Inferred dim = total / known_product
            result[idx] = DimSize::BinOp {
                op: DimOp::Div,
                left: Box::new(orig_total),
                right: Box::new(self.apply(&known_product)),
            };
            result[idx] = self.apply(&result[idx]);
        }

        Ok(TensorShape::from_dims(result))
    }

    /// Get all unsolved dimension variables
    pub fn unsolved_vars(&self) -> Vec<DimVar> {
        let mut vars = Vec::new();
        for (var, _) in &self.substitutions {
            // Check if var is fully resolved
            let resolved = self.apply(&DimSize::Var(*var));
            if matches!(resolved, DimSize::Var(_)) {
                vars.push(*var);
            }
        }
        vars
    }

    /// Get all unsolved symbolic dimensions
    pub fn unsolved_symbolics(&self) -> Vec<String> {
        let mut symbolics = Vec::new();
        for (name, _) in &self.symbolic_bindings {
            let resolved = self
                .symbolic_bindings
                .get(name)
                .map(|d| self.apply(d))
                .unwrap_or(DimSize::Symbolic(name.clone()));
            if matches!(resolved, DimSize::Symbolic(_) | DimSize::Var(_)) {
                symbolics.push(name.clone());
            }
        }
        symbolics
    }
}

impl Default for DimensionSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper functions for common operations
// ============================================================================

/// Create a matrix type shape [M, N]
pub fn matrix_shape(rows: DimSize, cols: DimSize) -> TensorShape {
    TensorShape::Parametric(vec![rows, cols])
}

/// Create a vector type shape [N]
pub fn vector_shape(size: DimSize) -> TensorShape {
    TensorShape::Parametric(vec![size])
}

/// Check if a dimension is statically known to be positive
pub fn is_positive(dim: &DimSize) -> bool {
    match dim {
        DimSize::Const(n) => *n > 0,
        DimSize::Dynamic => true, // Assume dynamic is valid
        _ => true,                // Conservative: assume valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_const() {
        let mut solver = DimensionSolver::new();
        assert!(solver.unify(DimSize::Const(3), DimSize::Const(3)).is_ok());
        assert!(solver.unify(DimSize::Const(3), DimSize::Const(4)).is_err());
    }

    #[test]
    fn test_unify_symbolic() {
        let mut solver = DimensionSolver::new();
        let n = DimSize::Symbolic("N".to_string());
        let m = DimSize::Symbolic("M".to_string());

        // Unify N with a constant
        assert!(solver.unify(n.clone(), DimSize::Const(5)).is_ok());
        assert_eq!(solver.apply(&n), DimSize::Const(5));

        // Unify M with N (should get same value)
        assert!(solver.unify(m.clone(), n.clone()).is_ok());
        assert_eq!(solver.apply(&m), DimSize::Const(5));
    }

    #[test]
    fn test_unify_var() {
        let mut solver = DimensionSolver::new();
        let v1 = solver.fresh_var();
        let v2 = solver.fresh_var();

        assert!(solver.unify(DimSize::Var(v1), DimSize::Const(10)).is_ok());
        assert_eq!(solver.apply(&DimSize::Var(v1)), DimSize::Const(10));

        assert!(solver.unify(DimSize::Var(v2), DimSize::Var(v1)).is_ok());
        assert_eq!(solver.apply(&DimSize::Var(v2)), DimSize::Const(10));
    }

    #[test]
    fn test_matmul_shape() {
        let mut solver = DimensionSolver::new();

        // [3, 4] @ [4, 5] -> [3, 5]
        let left = TensorShape::Static(vec![3, 4]);
        let right = TensorShape::Static(vec![4, 5]);
        let result = solver.matmul_shape(&left, &right).unwrap();
        assert_eq!(result.try_static_dims(), Some(vec![3, 5]));
    }

    #[test]
    fn test_matmul_shape_symbolic() {
        let mut solver = DimensionSolver::new();

        // [M, K] @ [K, N] -> [M, N]
        let m = DimSize::Symbolic("M".to_string());
        let k = DimSize::Symbolic("K".to_string());
        let n = DimSize::Symbolic("N".to_string());

        let left = TensorShape::Parametric(vec![m.clone(), k.clone()]);
        let right = TensorShape::Parametric(vec![k.clone(), n.clone()]);

        let result = solver.matmul_shape(&left, &right).unwrap();
        let dims = result.parametric_dims().unwrap();
        assert_eq!(dims.len(), 2);

        // The result should be [M, N] - both still symbolic since no concrete values
        match (&dims[0], &dims[1]) {
            (DimSize::Symbolic(s1), DimSize::Symbolic(s2)) => {
                assert_eq!(s1, "M");
                assert_eq!(s2, "N");
            }
            _ => panic!("Expected symbolic dimensions"),
        }
    }

    #[test]
    fn test_matmul_mismatch() {
        let mut solver = DimensionSolver::new();

        // [3, 4] @ [5, 6] -> error (4 != 5)
        let left = TensorShape::Static(vec![3, 4]);
        let right = TensorShape::Static(vec![5, 6]);
        assert!(solver.matmul_shape(&left, &right).is_err());
    }

    #[test]
    fn test_broadcast() {
        let mut solver = DimensionSolver::new();

        // [3, 1] broadcast [1, 4] -> [3, 4]
        let left = TensorShape::Static(vec![3, 1]);
        let right = TensorShape::Static(vec![1, 4]);
        let result = solver.broadcast_shape(&left, &right).unwrap();
        assert_eq!(result.try_static_dims(), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_rank_extension() {
        let mut solver = DimensionSolver::new();

        // [3] broadcast [2, 3] -> [2, 3]
        let left = TensorShape::Static(vec![3]);
        let right = TensorShape::Static(vec![2, 3]);
        let result = solver.broadcast_shape(&left, &right).unwrap();
        assert_eq!(result.try_static_dims(), Some(vec![2, 3]));
    }

    #[test]
    fn test_concat_shape() {
        let mut solver = DimensionSolver::new();

        // Concat [3, 4] and [5, 4] along axis 0 -> [8, 4]
        let shapes = vec![
            TensorShape::Static(vec![3, 4]),
            TensorShape::Static(vec![5, 4]),
        ];
        let result = solver.concat_shape(&shapes, 0).unwrap();
        assert_eq!(result.try_static_dims(), Some(vec![8, 4]));
    }

    #[test]
    fn test_slice_shape() {
        let mut solver = DimensionSolver::new();

        // [10, 5] slice axis 0, start=2, end=7 -> [5, 5]
        let shape = TensorShape::Static(vec![10, 5]);
        let result = solver
            .slice_shape(&shape, 0, Some(DimSize::Const(2)), Some(DimSize::Const(7)))
            .unwrap();
        assert_eq!(result.try_static_dims(), Some(vec![5, 5]));
    }

    #[test]
    fn test_binop_simplification() {
        let solver = DimensionSolver::new();

        let dim = DimSize::BinOp {
            op: DimOp::Add,
            left: Box::new(DimSize::Const(3)),
            right: Box::new(DimSize::Const(4)),
        };
        assert_eq!(solver.apply(&dim), DimSize::Const(7));
    }
}
