// Sounio Compiler - Tropical Geometry for Resource Analysis
// Min-plus algebra for compile-time resource bounds and causal path optimization

use std::ops::{Add, Mul};

/// Tropical number: element of ℝ ∪ {∞} with min-plus arithmetic
///
/// Tropical Addition: a ⊕ b = min(a, b)
/// Tropical Multiplication: a ⊗ b = a + b
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct TropicalNumber(pub f64);

impl Add for TropicalNumber {
    type Output = Self;

    /// Tropical addition: a ⊕ b = min(a, b)
    fn add(self, other: Self) -> Self::Output {
        TropicalNumber(self.0.min(other.0))
    }
}

impl Mul for TropicalNumber {
    type Output = Self;

    /// Tropical multiplication: a ⊗ b = a + b
    fn mul(self, other: Self) -> Self::Output {
        TropicalNumber(self.0 + other.0)
    }
}

impl TropicalNumber {
    /// Tropical identity for addition (neutral element for ⊕): ∞
    pub fn additive_identity() -> Self {
        TropicalNumber(f64::INFINITY)
    }

    /// Tropical identity for multiplication (neutral element for ⊗): 0
    pub fn multiplicative_identity() -> Self {
        TropicalNumber(0.0)
    }

    /// Check if this is the additive identity (∞)
    pub fn is_additive_identity(&self) -> bool {
        self.0.is_infinite() && self.0 > 0.0
    }

    /// Check if this is the multiplicative identity (0)
    pub fn is_multiplicative_identity(&self) -> bool {
        (self.0 - 0.0).abs() < 1e-10
    }
}

/// Matrix of tropical numbers
#[derive(Clone, Debug, PartialEq)]
pub struct TropicalMatrix(pub Vec<Vec<TropicalNumber>>);

impl TropicalMatrix {
    /// Create a new tropical matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        TropicalMatrix(vec![vec![TropicalNumber::additive_identity(); cols]; rows])
    }

    /// Get dimensions (rows, cols)
    pub fn dimensions(&self) -> (usize, usize) {
        if self.0.is_empty() {
            (0, 0)
        } else {
            (self.0.len(), self.0[0].len())
        }
    }

    /// Get element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<TropicalNumber> {
        self.0.get(i)?.get(j).copied()
    }

    /// Set element at (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: TropicalNumber) {
        if i < self.0.len() && j < self.0[i].len() {
            self.0[i][j] = value;
        }
    }
}

/// Tropical matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])
///
/// In tropical algebra: min_k(a + b) replaces sum_k(ab) in linear algebra
pub fn tropical_matmul(a: &TropicalMatrix, b: &TropicalMatrix) -> Result<TropicalMatrix, String> {
    let (a_rows, a_cols) = a.dimensions();
    let (b_rows, b_cols) = b.dimensions();

    if a_cols != b_rows {
        return Err(format!(
            "Incompatible dimensions: {}x{} * {}x{}",
            a_rows, a_cols, b_rows, b_cols
        ));
    }

    let mut result = TropicalMatrix::new(a_rows, b_cols);

    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut min_cost = f64::INFINITY;
            for k in 0..a_cols {
                if let (Some(a_ik), Some(b_kj)) = (a.get(i, k), b.get(k, j)) {
                    let cost = a_ik.mul(b_kj).0;
                    min_cost = min_cost.min(cost);
                }
            }
            result.set(i, j, TropicalNumber(min_cost));
        }
    }

    Ok(result)
}

/// Create tropical identity matrix (multiplicative neutral element)
///
/// Main diagonal: 0 (multiplicative identity)
/// Off-diagonal: ∞ (additive identity)
fn tropical_identity_matrix(size: usize) -> TropicalMatrix {
    let mut mat = TropicalMatrix::new(size, size);
    for i in 0..size {
        mat.set(i, i, TropicalNumber::multiplicative_identity());
    }
    mat
}

/// Tropical matrix power: A^(⊗n) = A ⊗ A ⊗ ... ⊗ A (n times)
///
/// Uses binary exponentiation for efficiency O(log n)
/// Represents paths of length at most n in a weighted directed graph
pub fn tropical_power(m: &TropicalMatrix, n: usize) -> Result<TropicalMatrix, String> {
    let (rows, cols) = m.dimensions();
    if rows != cols {
        return Err("Matrix must be square for power computation".to_string());
    }

    if rows == 0 {
        return Ok(TropicalMatrix(vec![]));
    }

    let mut result = tropical_identity_matrix(rows);
    let mut base = m.clone();
    let mut exp = n;

    while exp > 0 {
        if exp % 2 == 1 {
            result = tropical_matmul(&result, &base)?;
        }
        base = tropical_matmul(&base, &base)?;
        exp /= 2;
    }

    Ok(result)
}

/// Resource types for compile-time analysis
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResourceType {
    /// Time resource (in abstract units)
    Time(TropicalNumber),
    /// Space resource (in abstract units)
    Space(TropicalNumber),
    /// Energy resource (in abstract units)
    Energy(TropicalNumber),
}

/// Sequential composition of resources: r1 ⊗ r2
///
/// When computing sequentially, costs add: time_total = time1 + time2
pub fn sequential_compose(r1: &ResourceType, r2: &ResourceType) -> Result<ResourceType, String> {
    match (r1, r2) {
        (ResourceType::Time(t1), ResourceType::Time(t2)) => Ok(ResourceType::Time(*t1 * *t2)),
        (ResourceType::Space(s1), ResourceType::Space(s2)) => Ok(ResourceType::Space(*s1 * *s2)),
        (ResourceType::Energy(e1), ResourceType::Energy(e2)) => Ok(ResourceType::Energy(*e1 * *e2)),
        _ => Err("Cannot sequentially compose different resource types".to_string()),
    }
}

/// Parallel composition of resources: r1 ⊕ r2
///
/// When computing in parallel, take minimum cost: time_total = min(time1, time2)
pub fn parallel_compose(r1: &ResourceType, r2: &ResourceType) -> Result<ResourceType, String> {
    match (r1, r2) {
        (ResourceType::Time(t1), ResourceType::Time(t2)) => Ok(ResourceType::Time(*t1 + *t2)),
        (ResourceType::Space(s1), ResourceType::Space(s2)) => Ok(ResourceType::Space(*s1 + *s2)),
        (ResourceType::Energy(e1), ResourceType::Energy(e2)) => Ok(ResourceType::Energy(*e1 + *e2)),
        _ => Err("Cannot parallel compose different resource types".to_string()),
    }
}

/// Plain tropical operations on f64
pub fn tropical_add(a: f64, b: f64) -> f64 {
    a.min(b)
}

pub fn tropical_mul(a: f64, b: f64) -> f64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_number_add() {
        let a = TropicalNumber(3.0);
        let b = TropicalNumber(5.0);
        let result = a + b;
        assert_eq!(result, TropicalNumber(3.0)); // min(3, 5) = 3
    }

    #[test]
    fn test_tropical_number_mul() {
        let a = TropicalNumber(3.0);
        let b = TropicalNumber(5.0);
        let result = a * b;
        assert_eq!(result, TropicalNumber(8.0)); // 3 + 5 = 8
    }

    #[test]
    fn test_tropical_identities() {
        let zero = TropicalNumber::multiplicative_identity();
        let inf = TropicalNumber::additive_identity();

        assert!(zero.is_multiplicative_identity());
        assert!(inf.is_additive_identity());
    }

    #[test]
    fn test_tropical_matrix_mul() {
        let mut a = TropicalMatrix::new(2, 2);
        a.set(0, 0, TropicalNumber(1.0));
        a.set(0, 1, TropicalNumber(2.0));
        a.set(1, 0, TropicalNumber(3.0));
        a.set(1, 1, TropicalNumber(4.0));

        let result = tropical_matmul(&a, &a);
        assert!(result.is_ok());

        // result[0][0] = min(1+1, 2+3) = min(2, 5) = 2
        let res = result.unwrap();
        assert_eq!(res.get(0, 0), Some(TropicalNumber(2.0)));
    }

    #[test]
    fn test_tropical_identity_matrix() {
        let id = tropical_identity_matrix(3);
        assert_eq!(id.get(0, 0), Some(TropicalNumber(0.0)));
        assert_eq!(id.get(0, 1), Some(TropicalNumber(f64::INFINITY)));
        assert_eq!(id.get(1, 1), Some(TropicalNumber(0.0)));
    }

    #[test]
    fn test_tropical_power_identity() {
        let id = tropical_identity_matrix(2);
        let result = tropical_power(&id, 5);
        assert!(result.is_ok());
        // Identity^n = Identity
        let powered = result.unwrap();
        assert_eq!(powered.get(0, 0), Some(TropicalNumber(0.0)));
    }

    #[test]
    fn test_sequential_compose_time() {
        let t1 = ResourceType::Time(TropicalNumber(3.0));
        let t2 = ResourceType::Time(TropicalNumber(5.0));
        let result = sequential_compose(&t1, &t2);
        assert!(result.is_ok());
        match result.unwrap() {
            ResourceType::Time(t) => assert_eq!(t, TropicalNumber(8.0)), // 3 + 5
            _ => panic!("Wrong resource type"),
        }
    }

    #[test]
    fn test_parallel_compose_time() {
        let t1 = ResourceType::Time(TropicalNumber(3.0));
        let t2 = ResourceType::Time(TropicalNumber(5.0));
        let result = parallel_compose(&t1, &t2);
        assert!(result.is_ok());
        match result.unwrap() {
            ResourceType::Time(t) => assert_eq!(t, TropicalNumber(3.0)), // min(3, 5)
            _ => panic!("Wrong resource type"),
        }
    }

    #[test]
    fn test_type_mismatch_sequential() {
        let t = ResourceType::Time(TropicalNumber(3.0));
        let s = ResourceType::Space(TropicalNumber(5.0));
        let result = sequential_compose(&t, &s);
        assert!(result.is_err());
    }

    #[test]
    fn test_tropical_add_basic() {
        assert_eq!(tropical_add(3.0, 5.0), 3.0);
        assert_eq!(tropical_add(10.0, 2.0), 2.0);
    }

    #[test]
    fn test_tropical_mul_basic() {
        assert_eq!(tropical_mul(3.0, 5.0), 8.0);
        assert_eq!(tropical_mul(2.0, -1.0), 1.0);
    }
}
