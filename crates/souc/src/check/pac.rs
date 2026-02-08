//! PAC Learning Type Checker
//!
//! This module implements compile-time verification of PAC (Probably Approximately Correct)
//! learning theory constraints. It ensures that machine learning code has sufficient samples
//! for the required accuracy and confidence guarantees.
//!
//! # Theory
//!
//! PAC learning (Valiant 1984) provides mathematical guarantees for machine learning:
//! - **VC Dimension**: Measures hypothesis class complexity (shattering coefficient)
//! - **Sample Complexity**: Minimum samples needed for (ε, δ)-learning
//! - **Generalization Bounds**: With probability ≥ 1-δ, test_error ≤ train_error + ε
//!
//! # Compile-Time Verification
//!
//! The checker verifies:
//! 1. Sample size meets the complexity bound for the hypothesis class
//! 2. Epsilon (accuracy) and delta (confidence) parameters are valid (0 < x < 1)
//! 3. VC dimension is positive for the hypothesis class
//! 4. Generalization guarantees are properly propagated to output types
//!
//! # References
//!
//! - Valiant, L. (1984). A theory of the learnable. CACM 27(11).
//! - Blumer et al. (1989). Learnability and VC dimension. JACM 36(4).
//! - Hanneke, S. (2016). The optimal sample complexity of PAC learning. JMLR 17(1).
//! - McAllester, D. (1999). PAC-Bayesian stochastic model selection. Machine Learning.

use std::collections::HashMap;

use crate::common::Span;
use crate::types::pac::{
    compute_sample_complexity, compute_sample_complexity_tight, DeltaBound, EpsilonBound,
    GeneralizationBound, HypothesisClass, SampleBound, VCDimension,
};

/// Error from PAC type checking
#[derive(Debug, Clone)]
pub struct PACError {
    /// Error message
    pub message: String,
    /// Source location
    pub span: Option<Span>,
    /// Error kind
    pub kind: PACErrorKind,
    /// Suggested fix
    pub suggestion: Option<String>,
}

impl PACError {
    /// Create a new PAC error
    pub fn new(message: impl Into<String>, kind: PACErrorKind) -> Self {
        Self {
            message: message.into(),
            span: None,
            kind,
            suggestion: None,
        }
    }

    /// Add a source span
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

impl std::fmt::Display for PACError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PAC learning error: {}", self.message)?;
        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  suggestion: {}", suggestion)?;
        }
        Ok(())
    }
}

impl std::error::Error for PACError {}

/// Kind of PAC error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PACErrorKind {
    /// Insufficient samples for the required guarantee
    InsufficientSamples,
    /// Invalid epsilon (accuracy) parameter
    InvalidEpsilon,
    /// Invalid delta (confidence) parameter
    InvalidDelta,
    /// Invalid or unknown VC dimension
    InvalidVCDim,
    /// Unknown hypothesis class
    UnknownHypothesisClass,
    /// Conflicting PAC constraints
    ConflictingConstraints,
    /// Missing PAC annotation
    MissingAnnotation,
}

/// Result of PAC verification
#[derive(Debug, Clone)]
pub enum PACVerifyResult {
    /// Constraint is satisfied
    Satisfied {
        /// The generalization bound that holds
        bound: GeneralizationBound,
        /// Actual samples used
        samples: u64,
        /// Minimum samples required
        min_samples: u64,
    },
    /// Constraint might be satisfied (symbolic)
    PossiblySatisfied {
        /// The symbolic constraint
        constraint: String,
    },
    /// Constraint is violated
    Violated {
        /// Error details
        error: PACError,
    },
}

impl PACVerifyResult {
    /// Check if the result is satisfied
    pub fn is_satisfied(&self) -> bool {
        matches!(self, PACVerifyResult::Satisfied { .. })
    }

    /// Check if the result is violated
    pub fn is_violated(&self) -> bool {
        matches!(self, PACVerifyResult::Violated { .. })
    }
}

/// PAC type constraint
#[derive(Debug, Clone)]
pub struct PACConstraint {
    /// Hypothesis class for the learning task
    pub hypothesis: HypothesisClass,
    /// Required accuracy (epsilon)
    pub epsilon: EpsilonBound,
    /// Required confidence (1 - delta)
    pub delta: DeltaBound,
    /// Actual sample size (if known)
    pub sample_size: Option<SampleBound>,
    /// Source span for error reporting
    pub span: Option<Span>,
}

impl PACConstraint {
    /// Create a new PAC constraint
    pub fn new(hypothesis: HypothesisClass, epsilon: f64, delta: f64) -> Self {
        Self {
            hypothesis,
            epsilon: EpsilonBound::Const(epsilon),
            delta: DeltaBound::Const(delta),
            sample_size: None,
            span: None,
        }
    }

    /// Set the sample size
    pub fn with_samples(mut self, samples: u64) -> Self {
        self.sample_size = Some(SampleBound::Const(samples));
        self
    }

    /// Set the source span
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

/// PAC type checker
///
/// Verifies PAC learning constraints during type checking. Maintains a registry
/// of known hypothesis classes and validates sample complexity requirements.
#[derive(Debug)]
pub struct PACChecker {
    /// Registry of known hypothesis classes by name
    hypothesis_registry: HashMap<String, HypothesisClass>,
    /// Collected constraints to verify
    constraints: Vec<PACConstraint>,
    /// Whether to use tight bounds (Hanneke 2016)
    use_tight_bounds: bool,
}

impl Default for PACChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl PACChecker {
    /// Create a new PAC checker with standard hypothesis classes
    pub fn new() -> Self {
        let mut checker = Self {
            hypothesis_registry: HashMap::new(),
            constraints: Vec::new(),
            use_tight_bounds: false,
        };

        // Register standard hypothesis classes
        checker.register_standard_classes();

        checker
    }

    /// Enable tight sample complexity bounds (Hanneke 2016)
    pub fn with_tight_bounds(mut self) -> Self {
        self.use_tight_bounds = true;
        self
    }

    /// Register standard hypothesis classes
    fn register_standard_classes(&mut self) {
        // Linear classifiers in d dimensions: VC dim = d + 1
        self.hypothesis_registry.insert(
            "LinearClassifier".to_string(),
            HypothesisClass::linear_classifier(10), // Default 10 dimensions
        );

        // Decision stump (1D threshold): VC dim = 1
        self.hypothesis_registry.insert(
            "DecisionStump".to_string(),
            HypothesisClass::decision_stump(),
        );

        // Axis-aligned rectangles in d dimensions: VC dim = 2d
        self.hypothesis_registry.insert(
            "AxisAlignedRectangle".to_string(),
            HypothesisClass::axis_aligned_rectangle(2), // Default 2D
        );

        // Neural network (parameterized by layers and width)
        self.hypothesis_registry.insert(
            "NeuralNetwork".to_string(),
            HypothesisClass::neural_network(3, 100), // Default 3 layers, 100 width
        );

        // Decision tree
        self.hypothesis_registry.insert(
            "DecisionTree".to_string(),
            HypothesisClass::decision_tree(5, 10), // Default depth 5, 10 features
        );
    }

    /// Register a custom hypothesis class
    pub fn register_hypothesis(&mut self, name: impl Into<String>, class: HypothesisClass) {
        self.hypothesis_registry.insert(name.into(), class);
    }

    /// Look up a hypothesis class by name
    pub fn lookup_hypothesis(&self, name: &str) -> Option<&HypothesisClass> {
        self.hypothesis_registry.get(name)
    }

    /// Add a constraint to verify
    pub fn add_constraint(&mut self, constraint: PACConstraint) {
        self.constraints.push(constraint);
    }

    /// Check a training call for sample complexity
    ///
    /// Verifies that the provided sample size is sufficient for the required
    /// (epsilon, delta)-learning guarantee with the given hypothesis class.
    pub fn check_train_call(
        &mut self,
        hypothesis_name: &str,
        sample_size: u64,
        epsilon: f64,
        delta: f64,
        span: Option<Span>,
    ) -> Result<GeneralizationBound, PACError> {
        // Validate epsilon
        if epsilon <= 0.0 || epsilon >= 1.0 {
            return Err(PACError::new(
                format!("Invalid epsilon: {} (must be in (0, 1))", epsilon),
                PACErrorKind::InvalidEpsilon,
            )
            .with_suggestion("Use epsilon values like 0.01 (99% accuracy) or 0.1 (90% accuracy)"));
        }

        // Validate delta
        if delta <= 0.0 || delta >= 1.0 {
            return Err(PACError::new(
                format!("Invalid delta: {} (must be in (0, 1))", delta),
                PACErrorKind::InvalidDelta,
            )
            .with_suggestion(
                "Use delta values like 0.05 (95% confidence) or 0.01 (99% confidence)",
            ));
        }

        // Look up hypothesis class
        let hypothesis = self
            .hypothesis_registry
            .get(hypothesis_name)
            .ok_or_else(|| {
                PACError::new(
                    format!("Unknown hypothesis class: {}", hypothesis_name),
                    PACErrorKind::UnknownHypothesisClass,
                )
                .with_suggestion(format!(
                    "Known classes: {}",
                    self.hypothesis_registry
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;

        // Get VC dimension
        let vc_dim = match &hypothesis.vc_dim {
            VCDimension::Const(d) => *d,
            VCDimension::Infinite => {
                return Err(PACError::new(
                    format!(
                        "Hypothesis class {} has infinite VC dimension",
                        hypothesis_name
                    ),
                    PACErrorKind::InvalidVCDim,
                )
                .with_suggestion("Use a hypothesis class with finite VC dimension"));
            }
            _ => {
                // For symbolic VC dimensions, we can't compute concrete bounds
                // Return a symbolic result
                return Ok(GeneralizationBound {
                    train_error: crate::types::pac::ErrorBound::Symbolic("train_err".to_string()),
                    gap_bound: crate::types::pac::ErrorBound::Symbolic(format!(
                        "gap({}, {}, {})",
                        hypothesis_name, epsilon, delta
                    )),
                    confidence: 1.0 - delta,
                });
            }
        };

        // Compute required sample complexity
        let required_samples = if self.use_tight_bounds {
            compute_sample_complexity_tight(vc_dim, epsilon, delta)
        } else {
            compute_sample_complexity(vc_dim, epsilon, delta)
        };

        // Check if we have enough samples
        if sample_size < required_samples {
            let mut err = PACError::new(
                format!(
                    "Insufficient samples for PAC learning: have {}, need {} \
                     (VC dim = {}, ε = {}, δ = {})",
                    sample_size, required_samples, vc_dim, epsilon, delta
                ),
                PACErrorKind::InsufficientSamples,
            )
            .with_suggestion(format!(
                "Either: (1) use {} more samples, (2) increase ε to accept less accuracy, \
                 or (3) increase δ to accept less confidence",
                required_samples - sample_size
            ));

            if let Some(s) = span {
                err = err.with_span(s);
            }

            return Err(err);
        }

        // Compute the actual generalization gap for the samples we have
        let gap = crate::types::pac::compute_generalization_gap(vc_dim, sample_size, delta);

        // Add constraint for tracking
        let constraint = PACConstraint {
            hypothesis: hypothesis.clone(),
            epsilon: EpsilonBound::Const(epsilon),
            delta: DeltaBound::Const(delta),
            sample_size: Some(SampleBound::Const(sample_size)),
            span,
        };
        self.constraints.push(constraint);

        Ok(GeneralizationBound {
            train_error: crate::types::pac::ErrorBound::Unknown,
            gap_bound: crate::types::pac::ErrorBound::Const(gap),
            confidence: 1.0 - delta,
        })
    }

    /// Verify all collected constraints
    pub fn verify_all(&self) -> Vec<PACVerifyResult> {
        self.constraints
            .iter()
            .map(|c| self.verify_constraint(c))
            .collect()
    }

    /// Verify a single constraint
    fn verify_constraint(&self, constraint: &PACConstraint) -> PACVerifyResult {
        // Get concrete values if available
        let epsilon = match &constraint.epsilon {
            EpsilonBound::Const(e) => *e,
            EpsilonBound::Symbolic(s) => {
                return PACVerifyResult::PossiblySatisfied {
                    constraint: format!("epsilon = {}", s),
                };
            }
            EpsilonBound::Unknown => {
                return PACVerifyResult::PossiblySatisfied {
                    constraint: "epsilon unknown".to_string(),
                };
            }
        };

        let delta = match &constraint.delta {
            DeltaBound::Const(d) => *d,
            DeltaBound::Symbolic(s) => {
                return PACVerifyResult::PossiblySatisfied {
                    constraint: format!("delta = {}", s),
                };
            }
            DeltaBound::Unknown => {
                return PACVerifyResult::PossiblySatisfied {
                    constraint: "delta unknown".to_string(),
                };
            }
        };

        let samples = match &constraint.sample_size {
            Some(SampleBound::Const(s)) => *s,
            Some(SampleBound::Symbolic(s)) => {
                return PACVerifyResult::PossiblySatisfied {
                    constraint: format!("samples = {}", s),
                };
            }
            Some(SampleBound::Expr(_)) | Some(SampleBound::Unknown) => {
                return PACVerifyResult::PossiblySatisfied {
                    constraint: "samples symbolic".to_string(),
                };
            }
            None => {
                return PACVerifyResult::Violated {
                    error: PACError::new(
                        "No sample size specified",
                        PACErrorKind::MissingAnnotation,
                    ),
                };
            }
        };

        // Get VC dimension
        let vc_dim = match &constraint.hypothesis.vc_dim {
            VCDimension::Const(d) => *d,
            VCDimension::Infinite => {
                return PACVerifyResult::Violated {
                    error: PACError::new("Infinite VC dimension", PACErrorKind::InvalidVCDim),
                };
            }
            _ => {
                return PACVerifyResult::PossiblySatisfied {
                    constraint: "symbolic VC dimension".to_string(),
                };
            }
        };

        // Compute required samples
        let min_samples = if self.use_tight_bounds {
            compute_sample_complexity_tight(vc_dim, epsilon, delta)
        } else {
            compute_sample_complexity(vc_dim, epsilon, delta)
        };

        if samples >= min_samples {
            let gap = crate::types::pac::compute_generalization_gap(vc_dim, samples, delta);
            PACVerifyResult::Satisfied {
                bound: GeneralizationBound {
                    train_error: crate::types::pac::ErrorBound::Unknown,
                    gap_bound: crate::types::pac::ErrorBound::Const(gap),
                    confidence: 1.0 - delta,
                },
                samples,
                min_samples,
            }
        } else {
            PACVerifyResult::Violated {
                error: PACError::new(
                    format!(
                        "Insufficient samples: have {}, need {}",
                        samples, min_samples
                    ),
                    PACErrorKind::InsufficientSamples,
                ),
            }
        }
    }

    /// Get the number of registered hypothesis classes
    pub fn hypothesis_count(&self) -> usize {
        self.hypothesis_registry.len()
    }

    /// Get the number of pending constraints
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Clear all constraints (for reuse)
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }
}

/// Extension trait for integrating PAC checking with Knowledge types
pub trait PACKnowledgeExt {
    /// Get the PAC learning confidence (1 - delta)
    fn pac_confidence(&self) -> Option<f64>;

    /// Get the generalization gap bound
    fn generalization_gap(&self) -> Option<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pac_checker_creation() {
        let checker = PACChecker::new();
        assert!(checker.hypothesis_count() >= 5);
        assert_eq!(checker.constraint_count(), 0);
    }

    #[test]
    fn test_lookup_standard_classes() {
        let checker = PACChecker::new();

        assert!(checker.lookup_hypothesis("LinearClassifier").is_some());
        assert!(checker.lookup_hypothesis("DecisionStump").is_some());
        assert!(checker.lookup_hypothesis("NeuralNetwork").is_some());
        assert!(checker.lookup_hypothesis("Unknown").is_none());
    }

    #[test]
    fn test_sufficient_samples() {
        let mut checker = PACChecker::new();

        // Decision stump has VC dim = 1
        // For epsilon = 0.1, delta = 0.1, need roughly 100 samples
        let result = checker.check_train_call("DecisionStump", 1000, 0.1, 0.1, None);
        assert!(result.is_ok());

        let bound = result.unwrap();
        assert!(bound.confidence >= 0.9);
    }

    #[test]
    fn test_insufficient_samples() {
        let mut checker = PACChecker::new();

        // Decision stump with very few samples
        let result = checker.check_train_call("DecisionStump", 5, 0.01, 0.01, None);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.kind, PACErrorKind::InsufficientSamples);
    }

    #[test]
    fn test_invalid_epsilon() {
        let mut checker = PACChecker::new();

        let result = checker.check_train_call("DecisionStump", 1000, 1.5, 0.1, None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PACErrorKind::InvalidEpsilon);

        let result = checker.check_train_call("DecisionStump", 1000, -0.1, 0.1, None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PACErrorKind::InvalidEpsilon);
    }

    #[test]
    fn test_invalid_delta() {
        let mut checker = PACChecker::new();

        let result = checker.check_train_call("DecisionStump", 1000, 0.1, 1.5, None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PACErrorKind::InvalidDelta);
    }

    #[test]
    fn test_unknown_hypothesis() {
        let mut checker = PACChecker::new();

        let result = checker.check_train_call("UnknownHypothesis", 1000, 0.1, 0.1, None);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            PACErrorKind::UnknownHypothesisClass
        );
    }

    #[test]
    fn test_constraint_collection() {
        let mut checker = PACChecker::new();

        // Add some successful calls with sufficient samples
        // DecisionStump: VC dim = 1, needs ~400 samples for eps=0.1, delta=0.1
        checker
            .check_train_call("DecisionStump", 1000, 0.1, 0.1, None)
            .ok();
        // LinearClassifier (10D): VC dim = 11, needs ~5600 samples for eps=0.1, delta=0.1
        checker
            .check_train_call("LinearClassifier", 10000, 0.1, 0.1, None)
            .ok();

        assert_eq!(checker.constraint_count(), 2);

        let results = checker.verify_all();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_satisfied()));
    }

    #[test]
    fn test_tight_bounds() {
        // Tight bounds should generally require fewer samples
        // (though the difference may be small for some parameters)
        let normal_samples = compute_sample_complexity(10, 0.1, 0.1);
        let tight_samples = compute_sample_complexity_tight(10, 0.1, 0.1);

        // Tight bounds should be less or equal
        assert!(tight_samples <= normal_samples);
    }

    #[test]
    fn test_custom_hypothesis() {
        let mut checker = PACChecker::new();

        // Register a custom hypothesis class
        let custom = HypothesisClass {
            name: "CustomClass".to_string(),
            vc_dim: VCDimension::Const(5),
            model_type: "Custom".to_string(),
            is_finite: true,
            rademacher: None,
        };
        checker.register_hypothesis("CustomClass", custom);

        let result = checker.check_train_call("CustomClass", 10000, 0.1, 0.1, None);
        assert!(result.is_ok());
    }
}
