//! TENSOR (⊗) Operator - Composing Independent Knowledge
//!
//! The tensor product combines knowledge about INDEPENDENT aspects.
//! Confidence is multiplicative, adjusted by ontology correlation.
//!
//! # Typing Rule
//!
//! ```text
//! Γ ⊢ e₁ : Knowledge[τ₁, ε₁, δ₁, φ₁]
//! Γ ⊢ e₂ : Knowledge[τ₂, ε₂, δ₂, φ₂]
//! γ = correlation(δ₁, δ₂)
//! ─────────────────────────────────────────────────────────
//! Γ ⊢ e₁ ⊗ e₂ : Knowledge[(τ₁, τ₂), ε₁·ε₂·γ, δ₁∪δ₂, φ₁⊕φ₂]
//! ```
//!
//! # Confidence Formula
//!
//! ```text
//! ε(K₁ ⊗ K₂) = ε₁ × ε₂ × correlation(δ₁, δ₂)
//!
//! where correlation(δ₁, δ₂) = 1 - jaccard(δ₁, δ₂) / 2
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! let absorption: EpistemicValue<Rate> = /* ka = 1.2/h, ε = 0.90 */;
//! let elimination: EpistemicValue<Rate> = /* ke = 0.3/h, ε = 0.85 */;
//! let pk_model = absorption.tensor(elimination);
//! // ε = 0.90 × 0.85 × 1.0 = 0.765 (ontologies disjoint)
//! ```

use super::confidence::{CombinationStrategy, ConfidenceValue, combine_confidence};
use super::knowledge::{EpistemicValue, OntologyRef};
use super::provenance::ProvenanceNode;
use std::collections::HashSet;

/// Compute correlation factor based on ontology overlap
///
/// - No overlap (δ₁ ∩ δ₂ = ∅): γ = 1.0 (fully independent)
/// - Full overlap (δ₁ = δ₂): γ = 0.5 (not independent)
///
/// Formula: γ = 1 - jaccard(δ₁, δ₂) / 2
pub fn ontology_correlation(d1: &HashSet<OntologyRef>, d2: &HashSet<OntologyRef>) -> f64 {
    if d1.is_empty() && d2.is_empty() {
        return 1.0; // No ontology = assume fully independent
    }

    let intersection = d1.intersection(d2).count() as f64;
    let union = d1.union(d2).count() as f64;

    if union == 0.0 {
        1.0
    } else {
        let jaccard = intersection / union;
        1.0 - 0.5 * jaccard // Full overlap → 0.5, no overlap → 1.0
    }
}

impl<A: Clone> EpistemicValue<A> {
    /// Tensor product: K₁ ⊗ K₂
    ///
    /// Combines independent knowledge about different aspects.
    /// The result is a tuple of both values with combined metadata.
    ///
    /// # Algebraic Properties
    ///
    /// - Associative: (K₁ ⊗ K₂) ⊗ K₃ ≅ K₁ ⊗ (K₂ ⊗ K₃)
    /// - Commutative: K₁ ⊗ K₂ ≅ K₂ ⊗ K₁
    /// - Identity: K ⊗ I = K where I = certain(())
    /// - Monotonic: ε(K₁) ≤ ε(K₂) ⟹ ε(K₁⊗K) ≤ ε(K₂⊗K)
    pub fn tensor<B: Clone>(self, other: EpistemicValue<B>) -> EpistemicValue<(A, B)> {
        // Compute correlation factor based on ontology overlap
        let gamma = ontology_correlation(&self.ontology, &other.ontology);

        // Combine confidences: ε₁ × ε₂ × γ
        let combined_confidence = combine_confidence(
            self.confidence,
            other.confidence,
            &CombinationStrategy::WeightedProduct { correlation: gamma },
        );

        // Union of ontologies
        let mut combined_ontology = self.ontology.clone();
        combined_ontology.extend(other.ontology.iter().cloned());

        // Merge provenance
        let provenance = ProvenanceNode::derived(
            "tensor",
            vec![self.provenance.clone(), other.provenance.clone()],
        );

        EpistemicValue::new(
            (self.value().clone(), other.value().clone()),
            combined_confidence,
            combined_ontology,
            provenance,
        )
    }

    /// Tensor with explicit correlation override
    ///
    /// Use when you know the independence structure better than
    /// ontology overlap would indicate.
    pub fn tensor_with_correlation<B: Clone>(
        self,
        other: EpistemicValue<B>,
        correlation: f64,
    ) -> EpistemicValue<(A, B)> {
        let combined_confidence = combine_confidence(
            self.confidence,
            other.confidence,
            &CombinationStrategy::WeightedProduct {
                correlation: correlation.clamp(0.0, 1.0),
            },
        );

        let mut combined_ontology = self.ontology.clone();
        combined_ontology.extend(other.ontology.iter().cloned());

        let provenance = ProvenanceNode::derived_with_metadata(
            "tensor",
            vec![self.provenance.clone(), other.provenance.clone()],
            format!("correlation={:.3}", correlation),
        );

        EpistemicValue::new(
            (self.value.clone(), other.value.clone()),
            combined_confidence,
            combined_ontology,
            provenance,
        )
    }
}

/// Tensor identity element: () with certainty
pub fn tensor_identity() -> EpistemicValue<()> {
    EpistemicValue::certain(())
}

// Syntactic sugar: K₁ & K₂
impl<A: Clone, B: Clone> std::ops::BitAnd<EpistemicValue<B>> for EpistemicValue<A> {
    type Output = EpistemicValue<(A, B)>;

    fn bitand(self, rhs: EpistemicValue<B>) -> Self::Output {
        self.tensor(rhs)
    }
}

/// Helper to tensor multiple values
pub fn tensor_all<T: Clone>(values: Vec<EpistemicValue<T>>) -> EpistemicValue<Vec<T>> {
    if values.is_empty() {
        return EpistemicValue::certain(Vec::new());
    }

    let mut result_values = Vec::with_capacity(values.len());
    let mut combined_confidence = ConfidenceValue::certain();
    let mut combined_ontology = HashSet::new();
    let mut provenances = Vec::with_capacity(values.len());

    for v in values {
        // Compute rolling correlation with accumulated ontologies
        let gamma = ontology_correlation(&combined_ontology, v.ontology());

        // Update confidence: ε_acc × ε_new × γ
        combined_confidence = combine_confidence(
            combined_confidence,
            v.confidence(),
            &CombinationStrategy::WeightedProduct { correlation: gamma },
        );

        // Accumulate ontologies
        combined_ontology.extend(v.ontology().iter().cloned());

        // Collect values and provenances
        result_values.push(v.value().clone());
        provenances.push(v.provenance().clone());
    }

    let provenance = ProvenanceNode::derived("tensor_all", provenances);

    EpistemicValue::new(
        result_values,
        combined_confidence,
        combined_ontology,
        provenance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ontology_correlation_disjoint() {
        let d1: HashSet<OntologyRef> = [OntologyRef::new("A", "1")].into_iter().collect();
        let d2: HashSet<OntologyRef> = [OntologyRef::new("B", "2")].into_iter().collect();

        let gamma = ontology_correlation(&d1, &d2);
        assert!((gamma - 1.0).abs() < 1e-10); // No overlap = fully independent
    }

    #[test]
    fn test_ontology_correlation_identical() {
        let d1: HashSet<OntologyRef> = [OntologyRef::new("A", "1")].into_iter().collect();
        let d2: HashSet<OntologyRef> = [OntologyRef::new("A", "1")].into_iter().collect();

        let gamma = ontology_correlation(&d1, &d2);
        assert!((gamma - 0.5).abs() < 1e-10); // Full overlap = γ = 0.5
    }

    #[test]
    fn test_ontology_correlation_partial() {
        let d1: HashSet<OntologyRef> = [OntologyRef::new("A", "1"), OntologyRef::new("B", "2")]
            .into_iter()
            .collect();
        let d2: HashSet<OntologyRef> = [OntologyRef::new("A", "1"), OntologyRef::new("C", "3")]
            .into_iter()
            .collect();

        let gamma = ontology_correlation(&d1, &d2);
        // Jaccard = 1/3, so γ = 1 - 0.5 * (1/3) = 1 - 0.167 ≈ 0.833
        assert!((gamma - (1.0 - 1.0 / 6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_product() {
        let k1 = EpistemicValue::with_confidence(10.0, 0.9);
        let k2 = EpistemicValue::with_confidence(20.0, 0.8);

        let result = k1.tensor(k2);

        assert_eq!(result.value(), &(10.0, 20.0));
        // Empty ontologies → γ = 1.0, so ε = 0.9 × 0.8 × 1.0 = 0.72
        assert!((result.confidence().value() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_with_ontology_overlap() {
        let k1 = EpistemicValue::with_confidence(10.0, 0.9)
            .with_ontology(OntologyRef::new("PKPD", "clearance"));
        let k2 = EpistemicValue::with_confidence(20.0, 0.8)
            .with_ontology(OntologyRef::new("PKPD", "clearance"));

        let result = k1.tensor(k2);

        // Same ontology → γ = 0.5, so ε = 0.9 × 0.8 × 0.5 = 0.36
        assert!((result.confidence().value() - 0.36).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_bitand_syntax() {
        let k1 = EpistemicValue::with_confidence(1, 0.9);
        let k2 = EpistemicValue::with_confidence(2, 0.8);

        let result = k1 & k2;
        assert_eq!(result.value(), &(1, 2));
    }

    #[test]
    fn test_tensor_associativity() {
        let k1: EpistemicValue<f64> = EpistemicValue::with_confidence(1.0, 0.9);
        let k2: EpistemicValue<f64> = EpistemicValue::with_confidence(2.0, 0.8);
        let k3: EpistemicValue<f64> = EpistemicValue::with_confidence(3.0, 0.7);

        // (K₁ ⊗ K₂) ⊗ K₃
        let left = k1.clone().tensor(k2.clone()).tensor(k3.clone());

        // K₁ ⊗ (K₂ ⊗ K₃)
        let right = k1.tensor(k2.tensor(k3));

        // Values are isomorphic: ((a,b),c) vs (a,(b,c))
        let ((a, b), c) = left.value();
        let (a2, (b2, c2)) = right.value();

        assert!((*a - *a2).abs() < 1e-10);
        assert!((*b - *b2).abs() < 1e-10);
        assert!((*c - *c2).abs() < 1e-10);

        // Confidences should be equal
        assert!((left.confidence().value() - right.confidence().value()).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_monotonicity() {
        let k_low = EpistemicValue::with_confidence(1.0, 0.5);
        let k_high = EpistemicValue::with_confidence(1.0, 0.9);
        let k_other = EpistemicValue::with_confidence(2.0, 0.8);

        let result_low = k_low.tensor(k_other.clone());
        let result_high = k_high.tensor(k_other);

        assert!(result_low.confidence().value() <= result_high.confidence().value());
    }

    #[test]
    fn test_tensor_identity() {
        let k: EpistemicValue<f64> = EpistemicValue::with_confidence(42.0, 0.8);
        let id = tensor_identity();

        let result = k.clone().tensor(id);

        // Value should be (42.0, ())
        assert!((result.value().0 - 42.0).abs() < 1e-10);
        // Confidence should be preserved (times 1.0)
        assert!((result.confidence().value() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_all() {
        let values = vec![
            EpistemicValue::with_confidence(1.0, 0.9),
            EpistemicValue::with_confidence(2.0, 0.8),
            EpistemicValue::with_confidence(3.0, 0.7),
        ];

        let result = tensor_all(values);

        assert_eq!(result.value().len(), 3);
        // All empty ontologies → γ = 1.0 each step
        // 0.9 × 0.8 × 0.7 = 0.504
        assert!((result.confidence().value() - 0.504).abs() < 1e-10);
    }
}
