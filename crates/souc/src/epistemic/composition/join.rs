//! JOIN (⊔) Operator - Fusing Knowledge with Conflict Resolution
//!
//! The join operator combines knowledge about the SAME phenomenon,
//! with explicit handling of agreement and conflict.
//!
//! # Typing Rule
//!
//! ```text
//! Γ ⊢ e₁ : Knowledge[τ, ε₁, δ₁, φ₁]
//! Γ ⊢ e₂ : Knowledge[τ, ε₂, δ₂, φ₂]      // SAME τ!
//! τ : Fusible                             // τ implements Fusible
//! ─────────────────────────────────────────────────────────
//! Γ ⊢ e₁ ⊔ e₂ : JoinResult[τ, ε*, δ₁∪δ₂, φ₁⊕φ₂]
//! ```
//!
//! # Fusion Cases
//!
//! | Case | Condition | Confidence | Result |
//! |------|-----------|------------|--------|
//! | Concordant | κ < 0.05 | 1-(1-ε₁)(1-ε₂) (boost) | Merged value |
//! | Resolved | 0.05 ≤ κ < θ | avg × (1-κ) (penalty) | Merged value |
//! | Irreconcilable | κ ≥ θ | - | Both preserved |

use super::confidence::{CombinationStrategy, combine_confidence};
use super::knowledge::EpistemicValue;
use super::provenance::ProvenanceNode;
use std::fmt;

/// Conflict level between two values
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ConflictLevel(f64);

impl ConflictLevel {
    /// Create a new conflict level, clamping to [0, 1]
    pub fn new(value: f64) -> Self {
        ConflictLevel(value.clamp(0.0, 1.0))
    }

    /// No conflict (identical values)
    pub fn none() -> Self {
        ConflictLevel(0.0)
    }

    /// Maximum conflict
    pub fn total() -> Self {
        ConflictLevel(1.0)
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if concordant (κ < 0.05)
    pub fn is_concordant(&self) -> bool {
        self.0 < 0.05
    }

    /// Check if resolvable given threshold
    pub fn is_resolvable(&self, threshold: f64) -> bool {
        self.0 < threshold
    }
}

impl fmt::Display for ConflictLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 < 0.05 {
            write!(f, "concordant")
        } else if self.0 < 0.3 {
            write!(f, "minor conflict ({:.1}%)", self.0 * 100.0)
        } else if self.0 < 0.7 {
            write!(f, "moderate conflict ({:.1}%)", self.0 * 100.0)
        } else {
            write!(f, "severe conflict ({:.1}%)", self.0 * 100.0)
        }
    }
}

/// Trait for types that can be fused in a join operation
///
/// Implementors must define how to compute conflict between values
/// and how to merge them with weighted averaging.
pub trait Fusible: Sized {
    /// Compute conflict between two values
    ///
    /// Returns κ ∈ [0, 1]:
    /// - 0 = identical values
    /// - 1 = completely different (maximum conflict)
    fn conflict(&self, other: &Self) -> ConflictLevel;

    /// Merge two values with weights
    ///
    /// The weights are typically the confidence values.
    fn weighted_merge(&self, other: &Self, w1: f64, w2: f64) -> Self;
}

// Implement Fusible for common numeric types
impl Fusible for f64 {
    fn conflict(&self, other: &Self) -> ConflictLevel {
        let max_val = self.abs().max(other.abs());
        if max_val < 1e-10 {
            ConflictLevel::none()
        } else {
            ConflictLevel::new((self - other).abs() / max_val)
        }
    }

    fn weighted_merge(&self, other: &Self, w1: f64, w2: f64) -> Self {
        let total = w1 + w2;
        if total < 1e-10 {
            (*self + *other) / 2.0
        } else {
            (self * w1 + other * w2) / total
        }
    }
}

impl Fusible for f32 {
    fn conflict(&self, other: &Self) -> ConflictLevel {
        (*self as f64).conflict(&(*other as f64))
    }

    fn weighted_merge(&self, other: &Self, w1: f64, w2: f64) -> Self {
        (*self as f64).weighted_merge(&(*other as f64), w1, w2) as f32
    }
}

impl Fusible for i32 {
    fn conflict(&self, other: &Self) -> ConflictLevel {
        (*self as f64).conflict(&(*other as f64))
    }

    fn weighted_merge(&self, other: &Self, w1: f64, w2: f64) -> Self {
        (*self as f64)
            .weighted_merge(&(*other as f64), w1, w2)
            .round() as i32
    }
}

impl Fusible for i64 {
    fn conflict(&self, other: &Self) -> ConflictLevel {
        (*self as f64).conflict(&(*other as f64))
    }

    fn weighted_merge(&self, other: &Self, w1: f64, w2: f64) -> Self {
        (*self as f64)
            .weighted_merge(&(*other as f64), w1, w2)
            .round() as i64
    }
}

impl Fusible for bool {
    fn conflict(&self, other: &Self) -> ConflictLevel {
        if self == other {
            ConflictLevel::none()
        } else {
            ConflictLevel::total()
        }
    }

    fn weighted_merge(&self, other: &Self, w1: f64, w2: f64) -> Self {
        // Majority vote weighted by confidence
        if self == other {
            *self
        } else if w1 > w2 {
            *self
        } else {
            *other
        }
    }
}

/// Result of a join operation
#[derive(Debug, Clone)]
pub enum JoinResult<T> {
    /// Values agreed (κ < 0.05), confidence boosted
    Concordant(EpistemicValue<T>),

    /// Minor conflict resolved with penalty
    Resolved {
        /// The merged result
        result: EpistemicValue<T>,
        /// The conflict level that was resolved
        conflict_level: ConflictLevel,
    },

    /// Conflict too large to resolve
    Irreconcilable {
        /// First value
        k1: EpistemicValue<T>,
        /// Second value
        k2: EpistemicValue<T>,
        /// The conflict level
        conflict_level: ConflictLevel,
    },
}

impl<T> JoinResult<T> {
    /// Check if the join was successful (concordant or resolved)
    pub fn is_success(&self) -> bool {
        !matches!(self, JoinResult::Irreconcilable { .. })
    }

    /// Get the result if successful
    pub fn result(&self) -> Option<&EpistemicValue<T>> {
        match self {
            JoinResult::Concordant(r) => Some(r),
            JoinResult::Resolved { result, .. } => Some(result),
            JoinResult::Irreconcilable { .. } => None,
        }
    }

    /// Get the conflict level
    pub fn conflict_level(&self) -> Option<ConflictLevel> {
        match self {
            JoinResult::Concordant(_) => Some(ConflictLevel::none()),
            JoinResult::Resolved { conflict_level, .. } => Some(*conflict_level),
            JoinResult::Irreconcilable { conflict_level, .. } => Some(*conflict_level),
        }
    }

    /// Unwrap the result, panicking if irreconcilable
    pub fn unwrap(self) -> EpistemicValue<T> {
        match self {
            JoinResult::Concordant(r) => r,
            JoinResult::Resolved { result, .. } => result,
            JoinResult::Irreconcilable { conflict_level, .. } => {
                panic!("Called unwrap on irreconcilable join: {}", conflict_level)
            }
        }
    }

    /// Unwrap or provide a default
    pub fn unwrap_or(self, default: EpistemicValue<T>) -> EpistemicValue<T> {
        match self {
            JoinResult::Concordant(r) => r,
            JoinResult::Resolved { result, .. } => result,
            JoinResult::Irreconcilable { .. } => default,
        }
    }

    /// Prefer higher confidence when irreconcilable
    pub fn prefer_higher_confidence(self) -> EpistemicValue<T> {
        match self {
            JoinResult::Concordant(r) => r,
            JoinResult::Resolved { result, .. } => result,
            JoinResult::Irreconcilable { k1, k2, .. } => {
                if k1.confidence().value() >= k2.confidence().value() {
                    k1
                } else {
                    k2
                }
            }
        }
    }
}

impl<T: Clone> EpistemicValue<T>
where
    T: Fusible,
{
    /// Join operator: K₁ ⊔ K₂
    ///
    /// Fuses knowledge about the SAME phenomenon with conflict resolution.
    ///
    /// # Arguments
    /// * `other` - The other knowledge to join
    /// * `conflict_threshold` - Maximum conflict level to attempt resolution
    ///
    /// # Algebraic Properties
    ///
    /// - Commutative: K₁ ⊔ K₂ = K₂ ⊔ K₁
    /// - Idempotent: K ⊔ K = K' where ε(K') ≥ ε(K)
    /// - Concordance boosts: conflict = 0 ⟹ ε(K₁ ⊔ K₂) > max(ε(K₁), ε(K₂))
    pub fn join(self, other: EpistemicValue<T>, conflict_threshold: f64) -> JoinResult<T> {
        let conflict = self.value.conflict(other.value());
        let e1 = self.confidence.value();
        let e2 = other.confidence.value();

        // Union of ontologies
        let mut combined_ontology = self.ontology.clone();
        combined_ontology.extend(other.ontology.iter().cloned());

        if conflict.is_concordant() {
            // CONCORDANT: Values agree, boost confidence via Dempster-Shafer
            let merged_value = self.value.weighted_merge(other.value(), e1, e2);
            let boosted_confidence = combine_confidence(
                self.confidence,
                other.confidence,
                &CombinationStrategy::DempsterShafer,
            );

            let result = EpistemicValue::new(
                merged_value,
                boosted_confidence,
                combined_ontology,
                ProvenanceNode::derived(
                    "join_concordant",
                    vec![self.provenance.clone(), other.provenance.clone()],
                ),
            );

            JoinResult::Concordant(result)
        } else if conflict.is_resolvable(conflict_threshold) {
            // RESOLVABLE: Merge with confidence penalty
            let merged_value = self.value.weighted_merge(other.value(), e1, e2);
            let penalized_confidence = combine_confidence(
                self.confidence,
                other.confidence,
                &CombinationStrategy::PenalizedAverage {
                    conflict: conflict.value(),
                },
            );

            let result = EpistemicValue::new(
                merged_value,
                penalized_confidence,
                combined_ontology,
                ProvenanceNode::derived_with_metadata(
                    "join_resolved",
                    vec![self.provenance.clone(), other.provenance.clone()],
                    format!("conflict={:.3}", conflict.value()),
                ),
            );

            JoinResult::Resolved {
                result,
                conflict_level: conflict,
            }
        } else {
            // IRRECONCILABLE: Preserve both
            JoinResult::Irreconcilable {
                k1: self,
                k2: other,
                conflict_level: conflict,
            }
        }
    }

    /// Join with default threshold (0.3)
    pub fn join_default(self, other: EpistemicValue<T>) -> JoinResult<T> {
        self.join(other, 0.3)
    }

    /// Force merge even if conflicting (uses weighted average)
    pub fn force_merge(self, other: EpistemicValue<T>) -> EpistemicValue<T> {
        let conflict = self.value.conflict(other.value());
        let e1 = self.confidence.value();
        let e2 = other.confidence.value();

        let merged_value = self.value.weighted_merge(other.value(), e1, e2);

        let mut combined_ontology = self.ontology.clone();
        combined_ontology.extend(other.ontology.iter().cloned());

        // Use penalized confidence even for high conflict
        let penalized_confidence = combine_confidence(
            self.confidence,
            other.confidence,
            &CombinationStrategy::PenalizedAverage {
                conflict: conflict.value(),
            },
        );

        EpistemicValue::new(
            merged_value,
            penalized_confidence,
            combined_ontology,
            ProvenanceNode::derived_with_metadata(
                "force_merge",
                vec![self.provenance.clone(), other.provenance.clone()],
                format!("conflict={:.3}", conflict.value()),
            ),
        )
    }
}

// Syntactic sugar: K₁ | K₂ (with default threshold)
impl<T: Clone + Fusible> std::ops::BitOr<EpistemicValue<T>> for EpistemicValue<T> {
    type Output = JoinResult<T>;

    fn bitor(self, rhs: EpistemicValue<T>) -> Self::Output {
        self.join(rhs, 0.3)
    }
}

/// Join multiple values with pairwise resolution
pub fn join_all<T: Clone + Fusible>(
    values: Vec<EpistemicValue<T>>,
    conflict_threshold: f64,
) -> Result<EpistemicValue<T>, Vec<EpistemicValue<T>>> {
    if values.is_empty() {
        panic!("Cannot join empty list");
    }
    if values.len() == 1 {
        return Ok(values.into_iter().next().unwrap());
    }

    let mut iter = values.into_iter();
    let mut acc = iter.next().unwrap();

    for v in iter {
        match acc.join(v, conflict_threshold) {
            JoinResult::Concordant(r) | JoinResult::Resolved { result: r, .. } => {
                acc = r;
            }
            JoinResult::Irreconcilable { k1, k2, .. } => {
                return Err(vec![k1, k2]);
            }
        }
    }

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_level_f64() {
        let a = 5.0_f64;
        let b = 5.1_f64;
        let conflict = a.conflict(&b);
        // |5.0 - 5.1| / 5.1 ≈ 0.0196
        assert!(conflict.is_concordant());
    }

    #[test]
    fn test_conflict_level_high() {
        let a = 5.0_f64;
        let b = 10.0_f64;
        let conflict = a.conflict(&b);
        // |5.0 - 10.0| / 10.0 = 0.5
        assert!(!conflict.is_concordant());
        assert!((conflict.value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_merge() {
        let a = 5.0_f64;
        let b = 10.0_f64;
        let merged = a.weighted_merge(&b, 0.8, 0.2);
        // (5.0 * 0.8 + 10.0 * 0.2) / 1.0 = 6.0
        assert!((merged - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_join_concordant() {
        let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);
        let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(5.05, 0.75);

        let result = k1.join(k2, 0.3);

        match result {
            JoinResult::Concordant(r) => {
                // Values should be merged
                assert!((*r.value() - 5.0).abs() < 0.1);
                // Confidence should be boosted: 1 - (1-0.8)(1-0.75) = 0.95
                assert!(r.confidence().value() > 0.8);
            }
            _ => panic!("Expected concordant join"),
        }
    }

    #[test]
    fn test_join_resolved() {
        let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);
        let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(6.0, 0.75);

        let result = k1.join(k2, 0.3);

        match result {
            JoinResult::Resolved {
                result: r,
                conflict_level,
            } => {
                // Conflict = |5-6|/6 ≈ 0.167
                assert!(conflict_level.value() > 0.05);
                assert!(conflict_level.value() < 0.3);
                // Confidence should be penalized
                assert!(r.confidence().value() < 0.8);
            }
            _ => panic!("Expected resolved join"),
        }
    }

    #[test]
    fn test_join_irreconcilable() {
        let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);
        let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(15.0, 0.75);

        let result = k1.join(k2, 0.3);

        match result {
            JoinResult::Irreconcilable { conflict_level, .. } => {
                // Conflict = |5-15|/15 ≈ 0.667
                assert!(conflict_level.value() > 0.3);
            }
            _ => panic!("Expected irreconcilable join"),
        }
    }

    #[test]
    fn test_join_commutativity() {
        let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);
        let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(5.5, 0.75);

        let result1 = k1.clone().join(k2.clone(), 0.3);
        let result2 = k2.join(k1, 0.3);

        // Both should be resolved
        let r1 = result1.result().unwrap();
        let r2 = result2.result().unwrap();

        assert!((*r1.value() - *r2.value()).abs() < 1e-10);
        assert!((r1.confidence().value() - r2.confidence().value()).abs() < 1e-10);
    }

    #[test]
    fn test_join_idempotence() {
        let k: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);

        let result = k.clone().join(k.clone(), 0.3);

        match result {
            JoinResult::Concordant(r) => {
                assert!((*r.value() - 5.0).abs() < 1e-10);
                // Confidence should be boosted (Dempster-Shafer)
                assert!(r.confidence().value() >= 0.8);
            }
            _ => panic!("Self-join should be concordant"),
        }
    }

    #[test]
    fn test_join_bitor_syntax() {
        let k1 = EpistemicValue::with_confidence(5.0, 0.8);
        let k2 = EpistemicValue::with_confidence(5.0, 0.75);

        let result = k1 | k2;
        assert!(result.is_success());
    }

    #[test]
    fn test_prefer_higher_confidence() {
        let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.9);
        let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(50.0, 0.7);

        let result = k1.join(k2, 0.1); // Low threshold forces irreconcilable
        let chosen = result.prefer_higher_confidence();

        // Should choose k1 (higher confidence)
        assert!((*chosen.value() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_force_merge() {
        let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(5.0, 0.8);
        let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(15.0, 0.6);

        let result = k1.force_merge(k2);

        // Should merge even with high conflict
        // Weighted: (5*0.8 + 15*0.6) / 1.4 ≈ 9.29
        assert!((*result.value() - 9.286).abs() < 0.01);
        // Confidence heavily penalized
        assert!(result.confidence().value() < 0.5);
    }

    #[test]
    fn test_bool_fusion() {
        let k1 = EpistemicValue::with_confidence(true, 0.8);
        let k2 = EpistemicValue::with_confidence(true, 0.7);

        let result = k1.join(k2, 0.3);

        match result {
            JoinResult::Concordant(r) => {
                assert!(*r.value());
            }
            _ => panic!("Same booleans should be concordant"),
        }
    }

    #[test]
    fn test_bool_conflict() {
        let k1 = EpistemicValue::with_confidence(true, 0.8);
        let k2 = EpistemicValue::with_confidence(false, 0.7);

        let result = k1.join(k2, 0.3);

        // Different booleans = total conflict
        match result {
            JoinResult::Irreconcilable { .. } => {}
            _ => panic!("Different booleans should be irreconcilable"),
        }
    }
}
