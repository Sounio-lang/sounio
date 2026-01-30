//! CONDITION (|) Operator - Bayesian and Jeffrey Conditioning
//!
//! The condition operator updates knowledge given new evidence,
//! using either standard Bayesian or Jeffrey conditioning.
//!
//! # Typing Rule
//!
//! ```text
//! Γ ⊢ e : Knowledge[τ, ε, δ, φ]
//! Γ ⊢ E : Evidence
//! Γ ⊢ L : Fn(&τ, &Evidence) → f64  // Likelihood function
//! ─────────────────────────────────────────────────────────
//! Γ ⊢ e | E WITH L : Knowledge[τ, ε', δ, φ']
//! ```
//!
//! # Conditioning Methods
//!
//! | Method | Formula | When to Use |
//! |--------|---------|-------------|
//! | Bayesian | P(H\|E) = P(E\|H)P(H)/P(E) | Evidence is certain |
//! | Jeffrey | P'(H) = Σ P(H\|Eᵢ)P'(Eᵢ) | Evidence is uncertain |
//!
//! # Example
//!
//! ```rust,ignore
//! let prior = EpistemicValue::with_confidence(Disease::TypeA, 0.30);
//! let posterior = prior.condition(
//!     &test_result,
//!     |disease, test| 0.95,  // Sensitivity
//!     0.10,                   // False positive rate
//! );
//! // P(H|E) = (0.95 × 0.30) / (0.95×0.30 + 0.10×0.70) ≈ 0.803
//! ```

use super::confidence::ConfidenceValue;
use super::knowledge::EpistemicValue;
use super::provenance::{ProvenanceNode, SourceInfo};
use std::fmt;

/// Evidence for conditioning
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Description of the evidence
    pub description: String,
    /// How reliable is this evidence?
    pub reliability: ConfidenceValue,
    /// Source of the evidence
    pub source: SourceInfo,
}

impl Evidence {
    /// Create new evidence
    pub fn new(description: impl Into<String>, reliability: f64, source: SourceInfo) -> Self {
        Evidence {
            description: description.into(),
            reliability: ConfidenceValue::new(reliability).unwrap_or_default(),
            source,
        }
    }

    /// Create certain evidence (reliability = 1.0)
    pub fn certain(description: impl Into<String>, source: SourceInfo) -> Self {
        Evidence {
            description: description.into(),
            reliability: ConfidenceValue::certain(),
            source,
        }
    }

    /// Create uncertain evidence
    pub fn uncertain(description: impl Into<String>, reliability: f64) -> Self {
        Evidence {
            description: description.into(),
            reliability: ConfidenceValue::new(reliability).unwrap_or_default(),
            source: SourceInfo::Primitive,
        }
    }

    /// Check if evidence is considered certain (≥ 0.95)
    pub fn is_certain(&self) -> bool {
        self.reliability.value() >= 0.95
    }
}

impl fmt::Display for Evidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.description, self.reliability)
    }
}

/// Strength of evidence effect
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvidenceStrength {
    /// Strong evidence (likelihood ratio > 10)
    Strong,
    /// Moderate evidence (2 < LR < 10)
    Moderate,
    /// Weak evidence (1 < LR < 2)
    Weak,
    /// Neutral evidence (LR ≈ 1)
    Neutral,
    /// Counter-evidence (LR < 1)
    Counter,
}

impl EvidenceStrength {
    /// Compute strength from likelihood ratio
    pub fn from_likelihood_ratio(lr: f64) -> Self {
        if lr > 10.0 {
            EvidenceStrength::Strong
        } else if lr > 2.0 {
            EvidenceStrength::Moderate
        } else if lr > 1.1 {
            EvidenceStrength::Weak
        } else if lr > 0.9 {
            EvidenceStrength::Neutral
        } else {
            EvidenceStrength::Counter
        }
    }
}

/// Strategy for conditioning
#[derive(Clone, Debug)]
pub enum ConditioningStrategy {
    /// Standard Bayesian: P(H|E) = P(E|H)P(H)/P(E)
    Bayesian {
        /// Likelihood P(E|H)
        likelihood: f64,
        /// Complement likelihood P(E|¬H)
        complement_likelihood: f64,
    },

    /// Jeffrey conditioning for uncertain evidence
    Jeffrey {
        /// Partition probabilities: (P'(Eᵢ), P(H|Eᵢ))
        partition: Vec<(f64, f64)>,
    },

    /// Simple observation update (soft evidence)
    Observation {
        /// Positive or negative evidence
        positive: bool,
        /// Strength of update
        strength: f64,
    },

    /// Likelihood ratio update
    LikelihoodRatio {
        /// The ratio P(E|H) / P(E|¬H)
        ratio: f64,
    },
}

impl<T: Clone> EpistemicValue<T> {
    /// Bayesian conditioning: K | E WITH likelihood
    ///
    /// Updates confidence using Bayes' theorem:
    /// P(H|E) = P(E|H) × P(H) / P(E)
    /// where P(E) = P(E|H)×P(H) + P(E|¬H)×P(¬H)
    ///
    /// # Arguments
    /// * `evidence` - The evidence (used for provenance)
    /// * `likelihood` - Function computing P(E|H) given the value
    /// * `complement_likelihood` - P(E|¬H), the false positive rate
    pub fn condition<E>(
        self,
        _evidence: &E,
        likelihood: impl Fn(&T) -> f64,
        complement_likelihood: f64,
    ) -> Self {
        let prior = self.confidence.value();
        let p_e_given_h = likelihood(&self.value);

        // Bayes' theorem
        // P(H|E) = P(E|H) × P(H) / P(E)
        // P(E) = P(E|H)×P(H) + P(E|¬H)×P(¬H)
        let p_e = p_e_given_h * prior + complement_likelihood * (1.0 - prior);

        let posterior = if p_e > 1e-10 {
            (p_e_given_h * prior) / p_e
        } else {
            prior // No update if evidence impossible
        };

        EpistemicValue::new(
            self.value,
            ConfidenceValue::new(posterior.clamp(0.0, 1.0)).unwrap(),
            self.ontology,
            ProvenanceNode::updated(self.provenance, "bayesian_condition"),
        )
    }

    /// Jeffrey conditioning for uncertain evidence
    ///
    /// P'(H) = Σᵢ P(H|Eᵢ) × P'(Eᵢ)
    ///
    /// # Arguments
    /// * `partition_probs` - List of (P'(Eᵢ), P(H|Eᵢ)) pairs
    pub fn condition_jeffrey(self, partition_probs: Vec<(f64, f64)>) -> Self {
        // P'(H) = Σᵢ P(H|Eᵢ) × P'(Eᵢ)
        let posterior: f64 = partition_probs
            .iter()
            .map(|(p_prime_ei, p_h_given_ei)| p_prime_ei * p_h_given_ei)
            .sum();

        EpistemicValue::new(
            self.value,
            ConfidenceValue::new(posterior.clamp(0.0, 1.0)).unwrap(),
            self.ontology,
            ProvenanceNode::updated(self.provenance, "jeffrey_condition"),
        )
    }

    /// Update with a likelihood ratio
    ///
    /// LR = P(E|H) / P(E|¬H)
    ///
    /// Posterior odds = Prior odds × LR
    pub fn condition_lr(self, likelihood_ratio: f64) -> Self {
        let prior = self.confidence.value();

        // Convert to odds, multiply by LR, convert back
        let prior_odds = prior / (1.0 - prior + 1e-10);
        let posterior_odds = prior_odds * likelihood_ratio;
        let posterior = posterior_odds / (1.0 + posterior_odds);

        EpistemicValue::new(
            self.value,
            ConfidenceValue::new(posterior.clamp(0.0, 1.0)).unwrap(),
            self.ontology,
            ProvenanceNode::updated(self.provenance, "lr_condition"),
        )
    }

    /// Simple observation update (soft evidence)
    ///
    /// - Positive: ε' = ε + strength × (1 - ε)
    /// - Negative: ε' = ε × (1 - strength)
    pub fn observe_evidence(self, positive: bool, strength: f64) -> Self {
        let prior = self.confidence.value();

        let posterior = if positive {
            // Positive evidence increases confidence
            prior + strength.clamp(0.0, 1.0) * (1.0 - prior)
        } else {
            // Negative evidence decreases confidence
            prior * (1.0 - strength.clamp(0.0, 1.0))
        };

        EpistemicValue::new(
            self.value,
            ConfidenceValue::new(posterior).unwrap(),
            self.ontology,
            ProvenanceNode::updated(self.provenance, "observe"),
        )
    }

    /// Apply conditioning using a strategy
    pub fn apply_conditioning(self, strategy: ConditioningStrategy) -> Self {
        match strategy {
            ConditioningStrategy::Bayesian {
                likelihood,
                complement_likelihood,
            } => self.condition(&(), |_| likelihood, complement_likelihood),

            ConditioningStrategy::Jeffrey { partition } => self.condition_jeffrey(partition),

            ConditioningStrategy::Observation { positive, strength } => {
                self.observe_evidence(positive, strength)
            }

            ConditioningStrategy::LikelihoodRatio { ratio } => self.condition_lr(ratio),
        }
    }

    /// Combine multiple pieces of evidence
    ///
    /// Applies conditioning sequentially (assumes independence).
    pub fn condition_multiple<E>(self, evidence_list: &[(E, impl Fn(&T) -> f64, f64)]) -> Self {
        let mut result = self;
        for (_evidence, likelihood, complement) in evidence_list {
            result = result.condition(&(), |v| likelihood(v), *complement);
        }
        result
    }
}

/// Compute likelihood ratio from sensitivity and specificity
pub fn likelihood_ratio(sensitivity: f64, specificity: f64) -> f64 {
    // LR+ = sensitivity / (1 - specificity)
    sensitivity / (1.0 - specificity + 1e-10)
}

/// Compute negative likelihood ratio
pub fn negative_likelihood_ratio(sensitivity: f64, specificity: f64) -> f64 {
    // LR- = (1 - sensitivity) / specificity
    (1.0 - sensitivity) / (specificity + 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_condition_positive() {
        // Prior: 30% probability of disease
        let prior = EpistemicValue::with_confidence(true, 0.30);

        // Test with sensitivity = 0.95, specificity = 0.90
        // So P(E|H) = 0.95, P(E|¬H) = 0.10
        let posterior = prior.condition(&"test", |_| 0.95, 0.10);

        // P(E) = 0.95×0.30 + 0.10×0.70 = 0.285 + 0.070 = 0.355
        // P(H|E) = (0.95 × 0.30) / 0.355 ≈ 0.803
        assert!((posterior.confidence().value() - 0.803).abs() < 0.01);
    }

    #[test]
    fn test_bayesian_condition_negative() {
        // Prior: 30% probability of disease
        let prior = EpistemicValue::with_confidence(true, 0.30);

        // Low likelihood for hypothesis
        let posterior = prior.condition(&"test", |_| 0.10, 0.90);

        // P(E) = 0.10×0.30 + 0.90×0.70 = 0.03 + 0.63 = 0.66
        // P(H|E) = (0.10 × 0.30) / 0.66 ≈ 0.045
        assert!((posterior.confidence().value() - 0.045).abs() < 0.01);
    }

    #[test]
    fn test_neutral_evidence() {
        let prior = EpistemicValue::with_confidence(true, 0.50);

        // Neutral: P(E|H) = P(E|¬H) = 0.5
        let posterior = prior.condition(&"neutral", |_| 0.5, 0.5);

        // Should be unchanged
        assert!((posterior.confidence().value() - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_jeffrey_conditioning() {
        let prior = EpistemicValue::with_confidence(true, 0.50);

        // Uncertain evidence: 60% likely E₁ (supports H), 40% likely E₂ (opposes H)
        let posterior = prior.condition_jeffrey(vec![
            (0.6, 0.8), // If E₁, P(H|E₁) = 0.8
            (0.4, 0.2), // If E₂, P(H|E₂) = 0.2
        ]);

        // P'(H) = 0.6×0.8 + 0.4×0.2 = 0.48 + 0.08 = 0.56
        assert!((posterior.confidence().value() - 0.56).abs() < 0.01);
    }

    #[test]
    fn test_likelihood_ratio_update() {
        let prior = EpistemicValue::with_confidence(true, 0.30);

        // LR = 10 (strong evidence)
        let posterior = prior.condition_lr(10.0);

        // Prior odds = 0.30/0.70 ≈ 0.43
        // Posterior odds = 0.43 × 10 = 4.3
        // Posterior = 4.3/5.3 ≈ 0.81
        assert!((posterior.confidence().value() - 0.81).abs() < 0.02);
    }

    #[test]
    fn test_observe_positive() {
        let prior = EpistemicValue::with_confidence(true, 0.50);

        let posterior = prior.observe_evidence(true, 0.3);

        // ε' = 0.50 + 0.3 × 0.50 = 0.65
        assert!((posterior.confidence().value() - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_observe_negative() {
        let prior = EpistemicValue::with_confidence(true, 0.80);

        let posterior = prior.observe_evidence(false, 0.25);

        // ε' = 0.80 × 0.75 = 0.60
        assert!((posterior.confidence().value() - 0.60).abs() < 0.01);
    }

    #[test]
    fn test_apply_conditioning_strategy() {
        let prior = EpistemicValue::with_confidence(true, 0.30);

        let strategy = ConditioningStrategy::Bayesian {
            likelihood: 0.95,
            complement_likelihood: 0.10,
        };

        let posterior = prior.apply_conditioning(strategy);
        assert!((posterior.confidence().value() - 0.803).abs() < 0.01);
    }

    #[test]
    fn test_likelihood_ratio_computation() {
        // Sensitivity = 0.95, Specificity = 0.90
        let lr = likelihood_ratio(0.95, 0.90);
        // LR+ = 0.95 / 0.10 = 9.5
        assert!((lr - 9.5).abs() < 0.1);

        let nlr = negative_likelihood_ratio(0.95, 0.90);
        // LR- = 0.05 / 0.90 ≈ 0.056
        assert!((nlr - 0.056).abs() < 0.01);
    }

    #[test]
    fn test_evidence_strength() {
        assert_eq!(
            EvidenceStrength::from_likelihood_ratio(15.0),
            EvidenceStrength::Strong
        );
        assert_eq!(
            EvidenceStrength::from_likelihood_ratio(5.0),
            EvidenceStrength::Moderate
        );
        assert_eq!(
            EvidenceStrength::from_likelihood_ratio(1.5),
            EvidenceStrength::Weak
        );
        assert_eq!(
            EvidenceStrength::from_likelihood_ratio(1.0),
            EvidenceStrength::Neutral
        );
        assert_eq!(
            EvidenceStrength::from_likelihood_ratio(0.5),
            EvidenceStrength::Counter
        );
    }
}
