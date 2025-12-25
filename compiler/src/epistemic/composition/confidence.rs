//! Hybrid Confidence System for Epistemic Composition
//!
//! Provides multiple strategies for combining confidence values based on
//! the semantic context of the operation.
//!
//! # Combination Strategies
//!
//! | Strategy | Formula | Use Case |
//! |----------|---------|----------|
//! | Multiplicative | ε₁ × ε₂ | Independent knowledge (tensor) |
//! | Dempster-Shafer | 1 - (1-ε₁)(1-ε₂) | Concordant sources (join) |
//! | Penalized Average | avg(ε₁,ε₂) × (1-κ) | Conflicting sources |
//! | Bayesian | P(H\|E) = L×P(H)/P(E) | Evidence conditioning |
//! | Jeffrey | Σ P(H\|Eᵢ)×P'(Eᵢ) | Uncertain evidence |

use std::fmt;

/// A confidence value in the range [0, 1]
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct ConfidenceValue(f64);

impl ConfidenceValue {
    /// Create a new confidence value, clamping to [0, 1]
    pub fn new(value: f64) -> Result<Self, ConfidenceError> {
        if value.is_nan() {
            Err(ConfidenceError::NotANumber)
        } else {
            Ok(ConfidenceValue(value.clamp(0.0, 1.0)))
        }
    }

    /// Create with unchecked value (for internal use)
    pub(crate) fn new_unchecked(value: f64) -> Self {
        ConfidenceValue(value.clamp(0.0, 1.0))
    }

    /// Absolute certainty (ε = 1.0)
    pub fn certain() -> Self {
        ConfidenceValue(1.0)
    }

    /// Complete uncertainty (ε = 0.0)
    pub fn uncertain() -> Self {
        ConfidenceValue(0.0)
    }

    /// Zero confidence (alias for uncertain)
    pub fn zero() -> Self {
        ConfidenceValue(0.0)
    }

    /// Default epistemic threshold (ε = 0.5)
    pub fn threshold() -> Self {
        ConfidenceValue(0.5)
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if confidence meets a threshold
    pub fn meets(&self, threshold: ConfidenceValue) -> bool {
        self.0 >= threshold.0
    }

    /// Complement: 1 - ε
    pub fn complement(&self) -> Self {
        ConfidenceValue(1.0 - self.0)
    }

    /// Product of two confidences
    pub fn product(&self, other: ConfidenceValue) -> Self {
        ConfidenceValue::new_unchecked(self.0 * other.0)
    }

    /// Scale by a factor
    pub fn scale(&self, factor: f64) -> Self {
        ConfidenceValue::new_unchecked(self.0 * factor)
    }
}

impl Default for ConfidenceValue {
    fn default() -> Self {
        ConfidenceValue(0.5)
    }
}

impl fmt::Debug for ConfidenceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ε={:.3}", self.0)
    }
}

impl fmt::Display for ConfidenceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}%", self.0 * 100.0)
    }
}

/// Errors when creating confidence values
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceError {
    /// Value was NaN
    NotANumber,
    /// Value was infinite
    Infinite,
}

impl fmt::Display for ConfidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceError::NotANumber => write!(f, "confidence value cannot be NaN"),
            ConfidenceError::Infinite => write!(f, "confidence value cannot be infinite"),
        }
    }
}

impl std::error::Error for ConfidenceError {}

/// Strategy for combining confidence values
#[derive(Clone, Debug, PartialEq)]
pub enum CombinationStrategy {
    /// Simple product: ε₁ × ε₂
    /// Used for independent knowledge (tensor)
    Multiplicative,

    /// Dempster-Shafer combination: 1 - (1-ε₁)(1-ε₂)
    /// Used when sources agree (concordant join)
    DempsterShafer,

    /// Penalized average: avg(ε₁,ε₂) × (1-κ)
    /// Used when sources conflict
    PenalizedAverage {
        /// Conflict level κ ∈ [0, 1]
        conflict: f64,
    },

    /// Bayesian update: P(H|E) = L×P(H)/P(E)
    /// Used for evidence conditioning with certain evidence
    Bayesian {
        /// Likelihood P(E|H)
        likelihood: f64,
        /// Complement likelihood P(E|¬H)
        complement: f64,
    },

    /// Jeffrey conditioning: Σ P(H|Eᵢ)×P'(Eᵢ)
    /// Used for uncertain evidence
    Jeffrey {
        /// Partition probabilities: (P'(Eᵢ), P(H|Eᵢ))
        partition_probs: Vec<(f64, f64)>,
    },

    /// Weighted combination with correlation adjustment
    /// Used for tensor with ontology overlap
    WeightedProduct {
        /// Correlation factor γ ∈ [0.5, 1.0]
        correlation: f64,
    },

    /// Minimum (conservative)
    /// Used when we want the most pessimistic estimate
    Minimum,

    /// Maximum (optimistic)
    /// Used for "at least one is right" scenarios
    Maximum,
}

/// Compute combined confidence using the specified strategy
pub fn combine_confidence(
    epsilon1: ConfidenceValue,
    epsilon2: ConfidenceValue,
    strategy: &CombinationStrategy,
) -> ConfidenceValue {
    let e1 = epsilon1.value();
    let e2 = epsilon2.value();

    let result = match strategy {
        CombinationStrategy::Multiplicative => e1 * e2,

        CombinationStrategy::DempsterShafer => {
            // ε* = 1 - (1-ε₁)(1-ε₂)
            let p_both_wrong = (1.0 - e1) * (1.0 - e2);
            1.0 - p_both_wrong
        }

        CombinationStrategy::PenalizedAverage { conflict } => {
            // ε* = avg(ε₁,ε₂) × (1-κ)
            let avg = (e1 + e2) / 2.0;
            avg * (1.0 - conflict)
        }

        CombinationStrategy::Bayesian {
            likelihood,
            complement,
        } => {
            // P(H|E) = P(E|H)×P(H) / P(E)
            // where P(E) = P(E|H)×P(H) + P(E|¬H)×P(¬H)
            let prior = e1;
            let p_e = likelihood * prior + complement * (1.0 - prior);
            if p_e > 1e-10 {
                (likelihood * prior) / p_e
            } else {
                prior // No update if evidence impossible
            }
        }

        CombinationStrategy::Jeffrey { partition_probs } => {
            // P'(H) = Σᵢ P(H|Eᵢ) × P'(Eᵢ)
            partition_probs
                .iter()
                .map(|(p_prime_ei, p_h_given_ei)| p_prime_ei * p_h_given_ei)
                .sum()
        }

        CombinationStrategy::WeightedProduct { correlation } => {
            // ε* = ε₁ × ε₂ × γ
            e1 * e2 * correlation
        }

        CombinationStrategy::Minimum => e1.min(e2),

        CombinationStrategy::Maximum => e1.max(e2),
    };

    ConfidenceValue::new_unchecked(result.clamp(0.0, 1.0))
}

/// Select appropriate combination strategy based on operation context
pub fn select_strategy(
    operation: &str,
    evidence_confidence: Option<f64>,
    conflict_level: Option<f64>,
    correlation: Option<f64>,
) -> CombinationStrategy {
    match operation {
        "tensor" => {
            if let Some(corr) = correlation {
                CombinationStrategy::WeightedProduct { correlation: corr }
            } else {
                CombinationStrategy::Multiplicative
            }
        }

        "join" => match conflict_level {
            Some(k) if k < 0.05 => CombinationStrategy::DempsterShafer,
            Some(k) => CombinationStrategy::PenalizedAverage { conflict: k },
            None => CombinationStrategy::DempsterShafer,
        },

        "condition" => match evidence_confidence {
            Some(e) if e >= 0.95 => CombinationStrategy::Bayesian {
                likelihood: 0.9,
                complement: 0.1,
            },
            _ => CombinationStrategy::Jeffrey {
                partition_probs: vec![],
            },
        },

        _ => CombinationStrategy::Multiplicative,
    }
}

/// Confidence level categories for display and thresholds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfidenceLevel {
    /// Very low confidence (ε < 0.3)
    VeryLow,
    /// Low confidence (0.3 ≤ ε < 0.5)
    Low,
    /// Moderate confidence (0.5 ≤ ε < 0.7)
    Moderate,
    /// High confidence (0.7 ≤ ε < 0.85)
    High,
    /// Very high confidence (ε ≥ 0.85)
    VeryHigh,
}

impl ConfidenceLevel {
    /// Get confidence level from a numeric value
    pub fn from_value(confidence: f64) -> Self {
        if confidence < 0.3 {
            ConfidenceLevel::VeryLow
        } else if confidence < 0.5 {
            ConfidenceLevel::Low
        } else if confidence < 0.7 {
            ConfidenceLevel::Moderate
        } else if confidence < 0.85 {
            ConfidenceLevel::High
        } else {
            ConfidenceLevel::VeryHigh
        }
    }

    /// Get the minimum threshold for this level
    pub fn min_threshold(&self) -> f64 {
        match self {
            ConfidenceLevel::VeryLow => 0.0,
            ConfidenceLevel::Low => 0.3,
            ConfidenceLevel::Moderate => 0.5,
            ConfidenceLevel::High => 0.7,
            ConfidenceLevel::VeryHigh => 0.85,
        }
    }
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceLevel::VeryLow => write!(f, "very low"),
            ConfidenceLevel::Low => write!(f, "low"),
            ConfidenceLevel::Moderate => write!(f, "moderate"),
            ConfidenceLevel::High => write!(f, "high"),
            ConfidenceLevel::VeryHigh => write!(f, "very high"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_value_creation() {
        let c = ConfidenceValue::new(0.75).unwrap();
        assert!((c.value() - 0.75).abs() < f64::EPSILON);

        // Clamping
        let high = ConfidenceValue::new(1.5).unwrap();
        assert!((high.value() - 1.0).abs() < f64::EPSILON);

        let low = ConfidenceValue::new(-0.5).unwrap();
        assert!((low.value() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_nan() {
        let result = ConfidenceValue::new(f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiplicative_combination() {
        let c1 = ConfidenceValue::new(0.8).unwrap();
        let c2 = ConfidenceValue::new(0.9).unwrap();

        let result = combine_confidence(c1, c2, &CombinationStrategy::Multiplicative);
        assert!((result.value() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_dempster_shafer_combination() {
        let c1 = ConfidenceValue::new(0.8).unwrap();
        let c2 = ConfidenceValue::new(0.9).unwrap();

        let result = combine_confidence(c1, c2, &CombinationStrategy::DempsterShafer);
        // 1 - (0.2)(0.1) = 0.98
        assert!((result.value() - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_penalized_average() {
        let c1 = ConfidenceValue::new(0.8).unwrap();
        let c2 = ConfidenceValue::new(0.6).unwrap();

        let result = combine_confidence(
            c1,
            c2,
            &CombinationStrategy::PenalizedAverage { conflict: 0.2 },
        );
        // avg = 0.7, penalty = 0.8 → 0.56
        assert!((result.value() - 0.56).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_update() {
        let prior = ConfidenceValue::new(0.3).unwrap();
        let dummy = ConfidenceValue::certain();

        let result = combine_confidence(
            prior,
            dummy,
            &CombinationStrategy::Bayesian {
                likelihood: 0.95,
                complement: 0.10,
            },
        );

        // P(E) = 0.95×0.3 + 0.10×0.7 = 0.355
        // P(H|E) = (0.95 × 0.3) / 0.355 ≈ 0.803
        assert!((result.value() - 0.803).abs() < 0.01);
    }

    #[test]
    fn test_confidence_level() {
        assert_eq!(ConfidenceLevel::from_value(0.1), ConfidenceLevel::VeryLow);
        assert_eq!(ConfidenceLevel::from_value(0.4), ConfidenceLevel::Low);
        assert_eq!(ConfidenceLevel::from_value(0.6), ConfidenceLevel::Moderate);
        assert_eq!(ConfidenceLevel::from_value(0.75), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_value(0.9), ConfidenceLevel::VeryHigh);
    }
}
