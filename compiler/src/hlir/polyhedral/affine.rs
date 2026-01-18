//! Affine Expressions and Maps
//!
//! This module provides data structures for representing affine expressions
//! and maps, which are fundamental to the polyhedral model.
//!
//! # Affine Expressions
//!
//! An affine expression is a linear combination of variables plus a constant:
//!   e = a₁x₁ + a₂x₂ + ... + aₙxₙ + c
//!
//! # Affine Maps
//!
//! An affine map transforms a point in one integer space to a point in another:
//!   f: Z^n → Z^m
//!   f(x) = Ax + b
//!
//! # Access Relations
//!
//! Access relations describe how array elements are accessed as a function
//! of loop iterators.


/// An affine expression: linear combination of variables plus a constant
///
/// Represents: a₁x₁ + a₂x₂ + ... + aₙxₙ + b₁p₁ + ... + bₘpₘ + c
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineExpr {
    /// Coefficients for iteration variables
    pub iter_coeffs: Vec<i64>,
    /// Coefficients for parameters
    pub param_coeffs: Vec<i64>,
    /// Constant term
    pub constant: i64,
}

impl AffineExpr {
    /// Create a constant affine expression
    pub fn constant(c: i64) -> Self {
        Self {
            iter_coeffs: vec![],
            param_coeffs: vec![],
            constant: c,
        }
    }

    /// Create an expression for a single iteration variable
    pub fn iter_var(num_iters: usize, var_idx: usize) -> Self {
        let mut coeffs = vec![0; num_iters];
        coeffs[var_idx] = 1;
        Self {
            iter_coeffs: coeffs,
            param_coeffs: vec![],
            constant: 0,
        }
    }

    /// Create an expression for a single parameter
    pub fn param(num_iters: usize, num_params: usize, param_idx: usize) -> Self {
        let mut param_coeffs = vec![0; num_params];
        param_coeffs[param_idx] = 1;
        Self {
            iter_coeffs: vec![0; num_iters],
            param_coeffs,
            constant: 0,
        }
    }

    /// Create an expression: coeff * iter_var
    pub fn scaled_iter(num_iters: usize, var_idx: usize, coeff: i64) -> Self {
        let mut coeffs = vec![0; num_iters];
        coeffs[var_idx] = coeff;
        Self {
            iter_coeffs: coeffs,
            param_coeffs: vec![],
            constant: 0,
        }
    }

    /// Evaluate the expression at a given point
    pub fn evaluate(&self, iter_values: &[i64], param_values: &[i64]) -> i64 {
        let mut result = self.constant;

        for (coeff, value) in self.iter_coeffs.iter().zip(iter_values.iter()) {
            result += coeff * value;
        }

        for (coeff, value) in self.param_coeffs.iter().zip(param_values.iter()) {
            result += coeff * value;
        }

        result
    }

    /// Add two affine expressions
    pub fn add(&self, other: &AffineExpr) -> AffineExpr {
        let num_iters = self.iter_coeffs.len().max(other.iter_coeffs.len());
        let num_params = self.param_coeffs.len().max(other.param_coeffs.len());

        let mut iter_coeffs = vec![0; num_iters];
        let mut param_coeffs = vec![0; num_params];

        for (i, c) in self.iter_coeffs.iter().enumerate() {
            iter_coeffs[i] += c;
        }
        for (i, c) in other.iter_coeffs.iter().enumerate() {
            iter_coeffs[i] += c;
        }

        for (i, c) in self.param_coeffs.iter().enumerate() {
            param_coeffs[i] += c;
        }
        for (i, c) in other.param_coeffs.iter().enumerate() {
            param_coeffs[i] += c;
        }

        AffineExpr {
            iter_coeffs,
            param_coeffs,
            constant: self.constant + other.constant,
        }
    }

    /// Subtract another affine expression
    pub fn sub(&self, other: &AffineExpr) -> AffineExpr {
        self.add(&other.negate())
    }

    /// Negate the expression
    pub fn negate(&self) -> AffineExpr {
        AffineExpr {
            iter_coeffs: self.iter_coeffs.iter().map(|c| -c).collect(),
            param_coeffs: self.param_coeffs.iter().map(|c| -c).collect(),
            constant: -self.constant,
        }
    }

    /// Scale by a constant
    pub fn scale(&self, factor: i64) -> AffineExpr {
        AffineExpr {
            iter_coeffs: self.iter_coeffs.iter().map(|c| c * factor).collect(),
            param_coeffs: self.param_coeffs.iter().map(|c| c * factor).collect(),
            constant: self.constant * factor,
        }
    }

    /// Check if expression depends only on outer loops (indices < depth)
    pub fn is_loop_invariant_at(&self, depth: usize) -> bool {
        self.iter_coeffs.iter().skip(depth).all(|c| *c == 0)
    }

    /// Check if expression is constant (no variable dependencies)
    pub fn is_constant(&self) -> bool {
        self.iter_coeffs.iter().all(|c| *c == 0)
            && self.param_coeffs.iter().all(|c| *c == 0)
    }

    /// Get the coefficient for a specific iteration variable
    pub fn get_iter_coeff(&self, idx: usize) -> i64 {
        self.iter_coeffs.get(idx).copied().unwrap_or(0)
    }
}

impl std::fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut terms = Vec::new();

        for (i, coeff) in self.iter_coeffs.iter().enumerate() {
            if *coeff != 0 {
                let term = if *coeff == 1 {
                    format!("i{}", i)
                } else if *coeff == -1 {
                    format!("-i{}", i)
                } else {
                    format!("{}*i{}", coeff, i)
                };
                terms.push(term);
            }
        }

        for (i, coeff) in self.param_coeffs.iter().enumerate() {
            if *coeff != 0 {
                let term = if *coeff == 1 {
                    format!("p{}", i)
                } else if *coeff == -1 {
                    format!("-p{}", i)
                } else {
                    format!("{}*p{}", coeff, i)
                };
                terms.push(term);
            }
        }

        if self.constant != 0 || terms.is_empty() {
            terms.push(format!("{}", self.constant));
        }

        write!(f, "{}", terms.join(" + ").replace("+ -", "- "))
    }
}

impl std::ops::Add for AffineExpr {
    type Output = AffineExpr;

    fn add(self, rhs: AffineExpr) -> AffineExpr {
        AffineExpr::add(&self, &rhs)
    }
}

impl std::ops::Sub for AffineExpr {
    type Output = AffineExpr;

    fn sub(self, rhs: AffineExpr) -> AffineExpr {
        AffineExpr::sub(&self, &rhs)
    }
}

/// An affine map from one integer space to another
///
/// Represents: f: Z^n → Z^m where f(x) = [e₁(x), e₂(x), ..., eₘ(x)]
/// Each eᵢ is an affine expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineMap {
    /// Number of input dimensions
    pub num_inputs: usize,
    /// Number of output dimensions
    pub num_outputs: usize,
    /// Number of parameters
    pub num_params: usize,
    /// Output expressions (one per output dimension)
    pub outputs: Vec<AffineExpr>,
}

impl AffineMap {
    /// Create a new affine map
    pub fn new(num_inputs: usize, num_params: usize, outputs: Vec<AffineExpr>) -> Self {
        Self {
            num_inputs,
            num_outputs: outputs.len(),
            num_params,
            outputs,
        }
    }

    /// Create the identity map for n dimensions
    pub fn identity(n: usize) -> Self {
        let outputs = (0..n)
            .map(|i| AffineExpr::iter_var(n, i))
            .collect();
        Self {
            num_inputs: n,
            num_outputs: n,
            num_params: 0,
            outputs,
        }
    }

    /// Create a projection map that drops the last k dimensions
    pub fn project(n: usize, k: usize) -> Self {
        assert!(k <= n);
        let outputs = (0..(n - k))
            .map(|i| AffineExpr::iter_var(n, i))
            .collect();
        Self {
            num_inputs: n,
            num_outputs: n - k,
            num_params: 0,
            outputs,
        }
    }

    /// Create a loop interchange map
    ///
    /// Swaps dimensions i and j in an n-dimensional space
    pub fn interchange(n: usize, i: usize, j: usize) -> Self {
        assert!(i < n && j < n);
        let outputs = (0..n)
            .map(|dim| {
                if dim == i {
                    AffineExpr::iter_var(n, j)
                } else if dim == j {
                    AffineExpr::iter_var(n, i)
                } else {
                    AffineExpr::iter_var(n, dim)
                }
            })
            .collect();
        Self {
            num_inputs: n,
            num_outputs: n,
            num_params: 0,
            outputs,
        }
    }

    /// Apply the map to a point
    pub fn apply(&self, input: &[i64], params: &[i64]) -> Vec<i64> {
        self.outputs
            .iter()
            .map(|expr| expr.evaluate(input, params))
            .collect()
    }

    /// Compose two affine maps: (self . other)(x) = self(other(x))
    pub fn compose(&self, other: &AffineMap) -> AffineMap {
        assert_eq!(self.num_inputs, other.num_outputs);

        let outputs = self
            .outputs
            .iter()
            .map(|expr| {
                // Substitute other's outputs into expr
                let mut result = AffineExpr::constant(expr.constant);

                // Add param contributions (unchanged)
                result.param_coeffs = expr.param_coeffs.clone();

                // For each input coefficient in expr, substitute the corresponding
                // output from other
                for (i, coeff) in expr.iter_coeffs.iter().enumerate() {
                    if *coeff != 0 {
                        result = result.add(&other.outputs[i].scale(*coeff));
                    }
                }

                result
            })
            .collect();

        AffineMap {
            num_inputs: other.num_inputs,
            num_outputs: self.num_outputs,
            num_params: self.num_params.max(other.num_params),
            outputs,
        }
    }

    /// Check if this is the identity map
    pub fn is_identity(&self) -> bool {
        if self.num_inputs != self.num_outputs {
            return false;
        }
        for (i, expr) in self.outputs.iter().enumerate() {
            if expr.constant != 0 {
                return false;
            }
            for (j, coeff) in expr.iter_coeffs.iter().enumerate() {
                let expected = if i == j { 1 } else { 0 };
                if *coeff != expected {
                    return false;
                }
            }
        }
        true
    }
}

impl std::fmt::Display for AffineMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, expr) in self.outputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", expr)?;
        }
        write!(f, "]")
    }
}

/// An access relation describes how an array is accessed
///
/// Maps from iteration space to array indices
#[derive(Debug, Clone)]
pub struct AccessRelation {
    /// Name of the array being accessed
    pub array_name: String,
    /// The affine map from iteration space to array indices
    pub map: AffineMap,
    /// Whether this is a read or write access
    pub is_write: bool,
    /// Statement ID that performs this access
    pub statement_id: usize,
}

impl AccessRelation {
    /// Create a new access relation
    pub fn new(
        array_name: String,
        map: AffineMap,
        is_write: bool,
        statement_id: usize,
    ) -> Self {
        Self {
            array_name,
            map,
            is_write,
            statement_id,
        }
    }

    /// Create a simple 1D access: A[i]
    pub fn simple_1d(array_name: &str, num_loops: usize, loop_idx: usize, is_write: bool) -> Self {
        let map = AffineMap::new(
            num_loops,
            0,
            vec![AffineExpr::iter_var(num_loops, loop_idx)],
        );
        Self {
            array_name: array_name.to_string(),
            map,
            is_write,
            statement_id: 0,
        }
    }

    /// Create a 2D access: A[i][j]
    pub fn simple_2d(array_name: &str, num_loops: usize, i_idx: usize, j_idx: usize, is_write: bool) -> Self {
        let map = AffineMap::new(
            num_loops,
            0,
            vec![
                AffineExpr::iter_var(num_loops, i_idx),
                AffineExpr::iter_var(num_loops, j_idx),
            ],
        );
        Self {
            array_name: array_name.to_string(),
            map,
            is_write,
            statement_id: 0,
        }
    }

    /// Create access with stride: A[stride * i + offset]
    pub fn strided(
        array_name: &str,
        num_loops: usize,
        loop_idx: usize,
        stride: i64,
        offset: i64,
        is_write: bool,
    ) -> Self {
        let mut expr = AffineExpr::scaled_iter(num_loops, loop_idx, stride);
        expr.constant = offset;
        let map = AffineMap::new(num_loops, 0, vec![expr]);
        Self {
            array_name: array_name.to_string(),
            map,
            is_write,
            statement_id: 0,
        }
    }

    /// Get the array indices for a given iteration point
    pub fn get_indices(&self, iter_point: &[i64], params: &[i64]) -> Vec<i64> {
        self.map.apply(iter_point, params)
    }

    /// Check if two accesses may refer to the same array element
    /// (May-alias check)
    pub fn may_conflict(&self, other: &AccessRelation) -> bool {
        // Different arrays never conflict
        if self.array_name != other.array_name {
            return false;
        }

        // At least one must be a write for a true conflict
        if !self.is_write && !other.is_write {
            return false;
        }

        // Simplified check: assume they may conflict if same array and one is write
        // A proper implementation would check if the access polyhedra intersect
        true
    }
}

impl std::fmt::Display for AccessRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let access_type = if self.is_write { "write" } else { "read" };
        write!(f, "{}[S{}] -> {}[{}]", access_type, self.statement_id, self.array_name, self.map)
    }
}

/// Subscript pair for dependence analysis
///
/// Represents a pair of subscript expressions from source and sink accesses
#[derive(Debug, Clone)]
pub struct SubscriptPair {
    /// Source subscript expression
    pub source: AffineExpr,
    /// Sink subscript expression
    pub sink: AffineExpr,
    /// Dimension index in the array
    pub dimension: usize,
}

impl SubscriptPair {
    /// Create a new subscript pair
    pub fn new(source: AffineExpr, sink: AffineExpr, dimension: usize) -> Self {
        Self {
            source,
            sink,
            dimension,
        }
    }

    /// Check if this pair can be solved independently (separable)
    pub fn is_separable(&self) -> bool {
        // A pair is separable if source and sink involve disjoint sets of loop indices
        // Simplified: check if they use the same loop index patterns
        let source_vars: Vec<usize> = self
            .source
            .iter_coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, _)| i)
            .collect();

        let sink_vars: Vec<usize> = self
            .sink
            .iter_coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, _)| i)
            .collect();

        // If they use different loop variables, they're separable
        source_vars.iter().all(|v| !sink_vars.contains(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_expr_constant() {
        let expr = AffineExpr::constant(42);
        assert_eq!(expr.evaluate(&[], &[]), 42);
        assert!(expr.is_constant());
    }

    #[test]
    fn test_affine_expr_iter_var() {
        let expr = AffineExpr::iter_var(2, 0); // i0
        assert_eq!(expr.evaluate(&[5, 10], &[]), 5);

        let expr2 = AffineExpr::iter_var(2, 1); // i1
        assert_eq!(expr2.evaluate(&[5, 10], &[]), 10);
    }

    #[test]
    fn test_affine_expr_add() {
        let e1 = AffineExpr::iter_var(2, 0);
        let e2 = AffineExpr::iter_var(2, 1);
        let sum = e1.add(&e2); // i0 + i1

        assert_eq!(sum.evaluate(&[3, 4], &[]), 7);
    }

    #[test]
    fn test_affine_expr_scale() {
        let expr = AffineExpr::iter_var(1, 0).scale(3); // 3*i0
        assert_eq!(expr.evaluate(&[5], &[]), 15);
    }

    #[test]
    fn test_affine_map_identity() {
        let id = AffineMap::identity(3);
        assert!(id.is_identity());
        assert_eq!(id.apply(&[1, 2, 3], &[]), vec![1, 2, 3]);
    }

    #[test]
    fn test_affine_map_interchange() {
        let swap = AffineMap::interchange(3, 0, 2);
        assert_eq!(swap.apply(&[1, 2, 3], &[]), vec![3, 2, 1]);
    }

    #[test]
    fn test_affine_map_compose() {
        let swap01 = AffineMap::interchange(3, 0, 1);
        let swap12 = AffineMap::interchange(3, 1, 2);
        let composed = swap12.compose(&swap01);

        // swap01: (i, j, k) -> (j, i, k)
        // swap12: (i, j, k) -> (i, k, j)
        // composed: (i, j, k) -> (j, k, i)
        assert_eq!(composed.apply(&[1, 2, 3], &[]), vec![2, 3, 1]);
    }

    #[test]
    fn test_access_relation_simple() {
        let access = AccessRelation::simple_1d("A", 2, 0, true);
        assert_eq!(access.get_indices(&[5, 10], &[]), vec![5]);
    }

    #[test]
    fn test_access_relation_strided() {
        // A[2*i + 1]
        let access = AccessRelation::strided("A", 1, 0, 2, 1, false);
        assert_eq!(access.get_indices(&[5], &[]), vec![11]); // 2*5 + 1 = 11
    }

    #[test]
    fn test_access_relation_2d() {
        let access = AccessRelation::simple_2d("A", 2, 0, 1, false);
        assert_eq!(access.get_indices(&[3, 4], &[]), vec![3, 4]);
    }

    #[test]
    fn test_may_conflict() {
        let read = AccessRelation::simple_1d("A", 1, 0, false);
        let write = AccessRelation::simple_1d("A", 1, 0, true);
        let other_read = AccessRelation::simple_1d("B", 1, 0, false);

        assert!(read.may_conflict(&write)); // Read after write
        assert!(!read.may_conflict(&other_read)); // Different arrays
    }

    #[test]
    fn test_loop_invariant() {
        // Expression: i0 + 5
        let expr = AffineExpr {
            iter_coeffs: vec![1, 0],
            param_coeffs: vec![],
            constant: 5,
        };

        // At depth 0, nothing is invariant (i0 is used)
        assert!(!expr.is_loop_invariant_at(0));
        // At depth 1, it's invariant (only uses i0, which is outer)
        assert!(expr.is_loop_invariant_at(1));
        // At depth 2, still invariant
        assert!(expr.is_loop_invariant_at(2));
    }
}
