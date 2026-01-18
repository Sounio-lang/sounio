//! Integer Sets and Iteration Domains
//!
//! This module provides the core data structures for representing iteration
//! spaces as integer polyhedra. An iteration domain is the set of all valid
//! iteration points for a loop nest.
//!
//! # Mathematical Foundation
//!
//! An integer set S is defined by a set of affine constraints:
//!   S = { x ∈ Z^n | Ax ≤ b }
//!
//! where:
//! - x is a vector of iteration variables
//! - A is the constraint matrix
//! - b is the constant vector
//!
//! # Example
//!
//! For a loop nest:
//! ```text
//! for i = 0 to N-1:
//!     for j = 0 to M-1:
//!         S(i, j)
//! ```
//!
//! The iteration domain is:
//!   { (i, j) | 0 ≤ i < N ∧ 0 ≤ j < M }

use std::collections::HashMap;

/// A constraint in the polyhedral model
///
/// Represents an affine inequality of the form:
///   a₁x₁ + a₂x₂ + ... + aₙxₙ + c ≥ 0
///
/// or equality:
///   a₁x₁ + a₂x₂ + ... + aₙxₙ + c = 0
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    /// Coefficients for iteration variables (in order)
    pub iter_coeffs: Vec<i64>,
    /// Coefficients for parameters (symbolic constants like N, M)
    pub param_coeffs: Vec<i64>,
    /// Constant term
    pub constant: i64,
    /// Whether this is an equality (true) or inequality >= 0 (false)
    pub is_equality: bool,
}

impl Constraint {
    /// Create a new inequality constraint: expr >= 0
    pub fn inequality(iter_coeffs: Vec<i64>, param_coeffs: Vec<i64>, constant: i64) -> Self {
        Self {
            iter_coeffs,
            param_coeffs,
            constant,
            is_equality: false,
        }
    }

    /// Create a new equality constraint: expr = 0
    pub fn equality(iter_coeffs: Vec<i64>, param_coeffs: Vec<i64>, constant: i64) -> Self {
        Self {
            iter_coeffs,
            param_coeffs,
            constant,
            is_equality: true,
        }
    }

    /// Create constraint: var_idx >= lower_bound
    /// Represents: x_i - lb >= 0
    pub fn lower_bound(num_iters: usize, var_idx: usize, lower_bound: i64) -> Self {
        let mut iter_coeffs = vec![0; num_iters];
        iter_coeffs[var_idx] = 1;
        Self::inequality(iter_coeffs, vec![], -lower_bound)
    }

    /// Create constraint: var_idx < upper_bound (as upper_bound - var_idx - 1 >= 0)
    /// Represents: ub - x_i - 1 >= 0  =>  x_i <= ub - 1  =>  x_i < ub
    pub fn upper_bound_strict(num_iters: usize, var_idx: usize, upper_bound: i64) -> Self {
        let mut iter_coeffs = vec![0; num_iters];
        iter_coeffs[var_idx] = -1;
        Self::inequality(iter_coeffs, vec![], upper_bound - 1)
    }

    /// Create constraint: var_idx <= upper_bound
    /// Represents: ub - x_i >= 0
    pub fn upper_bound(num_iters: usize, var_idx: usize, upper_bound: i64) -> Self {
        let mut iter_coeffs = vec![0; num_iters];
        iter_coeffs[var_idx] = -1;
        Self::inequality(iter_coeffs, vec![], upper_bound)
    }

    /// Create constraint with parametric upper bound: var_idx < param
    /// Represents: param - x_i - 1 >= 0
    pub fn param_upper_bound(
        num_iters: usize,
        num_params: usize,
        var_idx: usize,
        param_idx: usize,
    ) -> Self {
        let mut iter_coeffs = vec![0; num_iters];
        iter_coeffs[var_idx] = -1;
        let mut param_coeffs = vec![0; num_params];
        param_coeffs[param_idx] = 1;
        Self::inequality(iter_coeffs, param_coeffs, -1)
    }

    /// Evaluate the constraint at a given point
    ///
    /// Returns the value of the left-hand side of the constraint.
    /// For inequalities, a non-negative value means the constraint is satisfied.
    /// For equalities, zero means the constraint is satisfied.
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

    /// Check if the constraint is satisfied at a given point
    pub fn is_satisfied(&self, iter_values: &[i64], param_values: &[i64]) -> bool {
        let value = self.evaluate(iter_values, param_values);
        if self.is_equality {
            value == 0
        } else {
            value >= 0
        }
    }

    /// Negate the constraint (flip the inequality direction)
    pub fn negate(&self) -> Self {
        if self.is_equality {
            // Negating equality doesn't make sense in the usual way
            // Return the same constraint
            self.clone()
        } else {
            // Negate all coefficients and subtract 1 to flip >= to <
            // x >= 0 becomes -x - 1 >= 0 (i.e., x < 0, or x <= -1)
            Self {
                iter_coeffs: self.iter_coeffs.iter().map(|c| -c).collect(),
                param_coeffs: self.param_coeffs.iter().map(|c| -c).collect(),
                constant: -self.constant - 1,
                is_equality: false,
            }
        }
    }
}

impl std::fmt::Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut terms = Vec::new();

        // Iteration variable terms
        for (i, coeff) in self.iter_coeffs.iter().enumerate() {
            if *coeff != 0 {
                if *coeff == 1 {
                    terms.push(format!("i{}", i));
                } else if *coeff == -1 {
                    terms.push(format!("-i{}", i));
                } else {
                    terms.push(format!("{}*i{}", coeff, i));
                }
            }
        }

        // Parameter terms
        for (i, coeff) in self.param_coeffs.iter().enumerate() {
            if *coeff != 0 {
                if *coeff == 1 {
                    terms.push(format!("p{}", i));
                } else if *coeff == -1 {
                    terms.push(format!("-p{}", i));
                } else {
                    terms.push(format!("{}*p{}", coeff, i));
                }
            }
        }

        // Constant term
        if self.constant != 0 || terms.is_empty() {
            terms.push(format!("{}", self.constant));
        }

        let expr = if terms.is_empty() {
            "0".to_string()
        } else {
            terms.join(" + ").replace("+ -", "- ")
        };

        if self.is_equality {
            write!(f, "{} = 0", expr)
        } else {
            write!(f, "{} >= 0", expr)
        }
    }
}

/// An integer set represented as a conjunction of affine constraints
///
/// Represents: { x ∈ Z^n | ∀c ∈ constraints: c(x) }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerSet {
    /// Number of iteration variables (dimensions)
    pub num_dims: usize,
    /// Number of symbolic parameters
    pub num_params: usize,
    /// Names of iteration variables (for debugging)
    pub dim_names: Vec<String>,
    /// Names of parameters (for debugging)
    pub param_names: Vec<String>,
    /// The constraints defining the set
    pub constraints: Vec<Constraint>,
}

impl IntegerSet {
    /// Create a new empty integer set with given dimensions
    pub fn new(num_dims: usize, num_params: usize) -> Self {
        Self {
            num_dims,
            num_params,
            dim_names: (0..num_dims).map(|i| format!("i{}", i)).collect(),
            param_names: (0..num_params).map(|i| format!("p{}", i)).collect(),
            constraints: Vec::new(),
        }
    }

    /// Create an integer set with named dimensions
    pub fn with_names(dim_names: Vec<String>, param_names: Vec<String>) -> Self {
        Self {
            num_dims: dim_names.len(),
            num_params: param_names.len(),
            dim_names,
            param_names,
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the set
    pub fn add_constraint(&mut self, constraint: Constraint) {
        debug_assert_eq!(constraint.iter_coeffs.len(), self.num_dims);
        debug_assert!(
            constraint.param_coeffs.is_empty()
                || constraint.param_coeffs.len() == self.num_params
        );
        self.constraints.push(constraint);
    }

    /// Add lower bound: dim >= value
    pub fn add_lower_bound(&mut self, dim: usize, value: i64) {
        self.add_constraint(Constraint::lower_bound(self.num_dims, dim, value));
    }

    /// Add upper bound: dim < value (strict)
    pub fn add_upper_bound_strict(&mut self, dim: usize, value: i64) {
        self.add_constraint(Constraint::upper_bound_strict(self.num_dims, dim, value));
    }

    /// Add upper bound: dim <= value
    pub fn add_upper_bound(&mut self, dim: usize, value: i64) {
        self.add_constraint(Constraint::upper_bound(self.num_dims, dim, value));
    }

    /// Add parametric upper bound: dim < param
    pub fn add_param_upper_bound(&mut self, dim: usize, param: usize) {
        self.add_constraint(Constraint::param_upper_bound(
            self.num_dims,
            self.num_params,
            dim,
            param,
        ));
    }

    /// Check if a point is in the set
    pub fn contains(&self, iter_values: &[i64], param_values: &[i64]) -> bool {
        self.constraints
            .iter()
            .all(|c| c.is_satisfied(iter_values, param_values))
    }

    /// Check if the set is empty (no integer points)
    ///
    /// This uses a simple heuristic - for a proper implementation,
    /// we would use the Omega test or ISL.
    pub fn is_empty(&self) -> bool {
        // Simple check: if there are contradictory bounds on any dimension
        // This is a conservative approximation
        for dim in 0..self.num_dims {
            if let Some((lb, ub)) = self.get_constant_bounds(dim) {
                if lb > ub {
                    return true;
                }
            }
        }
        false
    }

    /// Get constant lower and upper bounds for a dimension (if they exist)
    pub fn get_constant_bounds(&self, dim: usize) -> Option<(i64, i64)> {
        let mut lower = i64::MIN;
        let mut upper = i64::MAX;

        for constraint in &self.constraints {
            // Check if this constraint involves only this dimension
            let is_simple = constraint
                .iter_coeffs
                .iter()
                .enumerate()
                .all(|(i, c)| i == dim || *c == 0)
                && constraint.param_coeffs.iter().all(|c| *c == 0);

            if !is_simple {
                continue;
            }

            let coeff = constraint.iter_coeffs[dim];
            if coeff == 0 {
                continue;
            }

            // Constraint: coeff * dim + constant >= 0
            // If coeff > 0: dim >= -constant/coeff
            // If coeff < 0: dim <= -constant/coeff
            if constraint.is_equality {
                if coeff != 0 && constraint.constant % coeff == 0 {
                    let bound = -constraint.constant / coeff;
                    lower = lower.max(bound);
                    upper = upper.min(bound);
                }
            } else if coeff > 0 {
                // Lower bound
                let lb = (-constraint.constant + coeff - 1) / coeff; // Ceiling division
                lower = lower.max(lb);
            } else {
                // Upper bound (coeff < 0)
                let ub = constraint.constant / (-coeff);
                upper = upper.min(ub);
            }
        }

        if lower != i64::MIN && upper != i64::MAX {
            Some((lower, upper))
        } else {
            None
        }
    }

    /// Compute the intersection with another integer set
    pub fn intersect(&self, other: &IntegerSet) -> Self {
        assert_eq!(self.num_dims, other.num_dims);
        assert_eq!(self.num_params, other.num_params);

        let mut result = self.clone();
        for constraint in &other.constraints {
            result.add_constraint(constraint.clone());
        }
        result
    }

    /// Project out a dimension (existentially quantify)
    ///
    /// Returns a new set with dimension `dim` projected out.
    /// This is a simplified version that only works for simple cases.
    pub fn project_out(&self, dim: usize) -> Self {
        assert!(dim < self.num_dims);

        let mut new_dim_names = self.dim_names.clone();
        new_dim_names.remove(dim);

        let mut result = IntegerSet::with_names(new_dim_names, self.param_names.clone());

        // Copy constraints that don't involve the projected dimension
        for constraint in &self.constraints {
            if constraint.iter_coeffs[dim] == 0 {
                let mut new_coeffs = constraint.iter_coeffs.clone();
                new_coeffs.remove(dim);
                result.add_constraint(Constraint {
                    iter_coeffs: new_coeffs,
                    param_coeffs: constraint.param_coeffs.clone(),
                    constant: constraint.constant,
                    is_equality: constraint.is_equality,
                });
            }
        }

        // Note: A proper implementation would use Fourier-Motzkin elimination
        // to derive new constraints from the projected ones

        result
    }
}

impl std::fmt::Display for IntegerSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;

        // Print dimensions
        write!(f, "[")?;
        for (i, name) in self.dim_names.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", name)?;
        }
        write!(f, "]")?;

        // Print parameters
        if !self.param_names.is_empty() {
            write!(f, " -> [")?;
            for (i, name) in self.param_names.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", name)?;
            }
            write!(f, "]")?;
        }

        write!(f, " : ")?;

        // Print constraints
        if self.constraints.is_empty() {
            write!(f, "true")?;
        } else {
            for (i, constraint) in self.constraints.iter().enumerate() {
                if i > 0 {
                    write!(f, " and ")?;
                }
                write!(f, "{}", constraint)?;
            }
        }

        write!(f, " }}")
    }
}

/// Iteration domain for a statement in a loop nest
///
/// The iteration domain captures all valid iteration points where
/// a statement can execute.
#[derive(Debug, Clone)]
pub struct IterationDomain {
    /// The underlying integer set
    pub set: IntegerSet,
    /// Mapping from loop indices to dimension indices
    pub loop_indices: Vec<String>,
    /// Mapping from parameter names to their HLIR values (if known)
    pub param_values: HashMap<String, i64>,
}

impl IterationDomain {
    /// Create a new iteration domain
    pub fn new(loop_indices: Vec<String>, params: Vec<String>) -> Self {
        Self {
            set: IntegerSet::with_names(loop_indices.clone(), params),
            loop_indices,
            param_values: HashMap::new(),
        }
    }

    /// Create a simple rectangular domain for a loop nest
    ///
    /// Each loop has bounds: lower[i] <= loop_var[i] < upper[i]
    pub fn rectangular(
        loop_indices: Vec<String>,
        lowers: Vec<i64>,
        uppers: Vec<i64>,
    ) -> Self {
        let num_loops = loop_indices.len();
        let mut domain = Self::new(loop_indices, vec![]);

        for i in 0..num_loops {
            domain.set.add_lower_bound(i, lowers[i]);
            domain.set.add_upper_bound_strict(i, uppers[i]);
        }

        domain
    }

    /// Create a triangular domain (common in scientific computing)
    ///
    /// Example: for i = 0 to N-1; for j = i to N-1
    pub fn triangular_upper(n: usize, param_name: &str) -> Self {
        let loop_indices = vec!["i".to_string(), "j".to_string()];
        let params = vec![param_name.to_string()];
        let mut domain = Self::new(loop_indices, params);

        // i >= 0
        domain.set.add_lower_bound(0, 0);
        // i < N (parametric)
        domain.set.add_param_upper_bound(0, 0);
        // j >= i (j - i >= 0)
        domain.set.add_constraint(Constraint::inequality(
            vec![-1, 1], // -i + j
            vec![],
            0, // -i + j >= 0
        ));
        // j < N (parametric)
        domain.set.add_param_upper_bound(1, 0);

        domain
    }

    /// Get the number of loops (dimensions)
    pub fn depth(&self) -> usize {
        self.set.num_dims
    }

    /// Check if a point is in the domain
    pub fn contains(&self, point: &[i64]) -> bool {
        let param_values: Vec<i64> = self
            .set
            .param_names
            .iter()
            .map(|name| self.param_values.get(name).copied().unwrap_or(0))
            .collect();
        self.set.contains(point, &param_values)
    }

    /// Set a known parameter value
    pub fn set_param(&mut self, name: &str, value: i64) {
        self.param_values.insert(name.to_string(), value);
    }
}

impl std::fmt::Display for IterationDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_constraint() {
        // Constraint: i >= 0
        let c = Constraint::lower_bound(1, 0, 0);
        assert!(c.is_satisfied(&[0], &[]));
        assert!(c.is_satisfied(&[5], &[]));
        assert!(!c.is_satisfied(&[-1], &[]));
    }

    #[test]
    fn test_upper_bound_constraint() {
        // Constraint: i < 10
        let c = Constraint::upper_bound_strict(1, 0, 10);
        assert!(c.is_satisfied(&[0], &[]));
        assert!(c.is_satisfied(&[9], &[]));
        assert!(!c.is_satisfied(&[10], &[]));
    }

    #[test]
    fn test_rectangular_domain() {
        // for i = 0 to 9; for j = 0 to 9
        let domain = IterationDomain::rectangular(
            vec!["i".to_string(), "j".to_string()],
            vec![0, 0],
            vec![10, 10],
        );

        assert!(domain.contains(&[0, 0]));
        assert!(domain.contains(&[5, 5]));
        assert!(domain.contains(&[9, 9]));
        assert!(!domain.contains(&[10, 0]));
        assert!(!domain.contains(&[0, 10]));
        assert!(!domain.contains(&[-1, 0]));
    }

    #[test]
    fn test_integer_set_display() {
        let mut set = IntegerSet::with_names(
            vec!["i".to_string(), "j".to_string()],
            vec!["N".to_string()],
        );
        set.add_lower_bound(0, 0);
        set.add_param_upper_bound(0, 0);

        let display = format!("{}", set);
        assert!(display.contains("i"));
        assert!(display.contains("N"));
    }

    #[test]
    fn test_constraint_negate() {
        // i >= 0 negated becomes i < 0 (or -i - 1 >= 0)
        let c = Constraint::lower_bound(1, 0, 0);
        let neg = c.negate();

        assert!(c.is_satisfied(&[0], &[]));
        assert!(!neg.is_satisfied(&[0], &[]));
        assert!(neg.is_satisfied(&[-1], &[]));
    }

    #[test]
    fn test_constant_bounds() {
        let mut set = IntegerSet::new(2, 0);
        // 0 <= i < 10
        set.add_lower_bound(0, 0);
        set.add_upper_bound_strict(0, 10);
        // 5 <= j <= 15
        set.add_lower_bound(1, 5);
        set.add_upper_bound(1, 15);

        assert_eq!(set.get_constant_bounds(0), Some((0, 9)));
        assert_eq!(set.get_constant_bounds(1), Some((5, 15)));
    }

    #[test]
    fn test_empty_set_detection() {
        let mut set = IntegerSet::new(1, 0);
        // i >= 10 and i < 5 (empty)
        set.add_lower_bound(0, 10);
        set.add_upper_bound_strict(0, 5);

        assert!(set.is_empty());
    }

    #[test]
    fn test_set_intersection() {
        let mut set1 = IntegerSet::new(1, 0);
        set1.add_lower_bound(0, 0);

        let mut set2 = IntegerSet::new(1, 0);
        set2.add_upper_bound_strict(0, 10);

        let intersection = set1.intersect(&set2);
        assert_eq!(intersection.constraints.len(), 2);
        assert!(intersection.contains(&[5], &[]));
        assert!(!intersection.contains(&[-1], &[]));
        assert!(!intersection.contains(&[10], &[]));
    }
}
