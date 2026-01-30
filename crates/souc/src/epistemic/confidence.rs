//! Epistemic status tracking: confidence, revisability, source
//!
//! Every piece of knowledge in Sounio carries information about
//! how certain we are about it and where it came from.
//!
//! # Design Principles
//!
//! 1. **Confidence propagates**: When knowledge passes through transformations,
//!    confidence is updated based on the transformation's reliability.
//!
//! 2. **Sources are tracked**: Every piece of knowledge knows its origin,
//!    whether from measurement, computation, assertion, or external source.
//!
//! 3. **Revisability is explicit**: Some knowledge is axiomatic (definitions),
//!    while other knowledge can be revised with new evidence.

use super::Transformation;
use std::fmt;

/// Complete epistemic status of a knowledge value
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicStatus {
    /// How confident are we? (0.0 - 1.0)
    pub confidence: Confidence,

    /// Can this be revised with new evidence?
    pub revisability: Revisability,

    /// Where did this knowledge originate?
    pub source: Source,

    /// Evidence chain supporting this knowledge
    pub evidence: Vec<Evidence>,
}

impl EpistemicStatus {
    /// Axiomatic knowledge - certain, non-revisable, from definition
    ///
    /// Used for literals, constants, and definitions.
    pub fn axiomatic() -> Self {
        Self {
            confidence: Confidence::certain(),
            revisability: Revisability::NonRevisable,
            source: Source::Axiom,
            evidence: vec![],
        }
    }

    /// Empirical knowledge - uncertain, revisable, from measurement
    pub fn empirical(confidence: f64, source: Source) -> Self {
        Self {
            confidence: Confidence::new(confidence),
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".into()],
            },
            source,
            evidence: vec![],
        }
    }

    /// Derived knowledge - inherits from dependencies
    ///
    /// Confidence is the product of dependencies (conservative).
    /// Revisable if any dependency is revisable.
    pub fn derived(dependencies: &[&EpistemicStatus], derivation: &str) -> Self {
        // Confidence is product of dependencies (conservative)
        let confidence = dependencies
            .iter()
            .map(|e| e.confidence.value())
            .product::<f64>();

        // Revisable if any dependency is revisable
        let revisability = if dependencies.iter().any(|e| e.revisability.is_revisable()) {
            Revisability::Revisable {
                conditions: vec![format!("revision of {}", derivation)],
            }
        } else {
            Revisability::NonRevisable
        };

        Self {
            confidence: Confidence::new(confidence),
            revisability,
            source: Source::Derivation(derivation.to_string()),
            evidence: vec![],
        }
    }

    /// Propagate epistemic status through a transformation
    ///
    /// The confidence is reduced by the transformation's confidence factor.
    pub fn propagate(&self, transformation: &Transformation) -> Self {
        Self {
            confidence: self
                .confidence
                .propagate(transformation.confidence_factor()),
            revisability: self.revisability.clone(),
            source: Source::Transformation {
                original: Box::new(self.source.clone()),
                via: transformation.name().to_string(),
            },
            evidence: self.evidence.clone(),
        }
    }

    /// Add evidence to this epistemic status
    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence.push(evidence);
        self
    }

    /// Create from measurement with instrument
    pub fn from_measurement(confidence: f64, instrument: &str) -> Self {
        Self::empirical(
            confidence,
            Source::Measurement {
                instrument: Some(instrument.to_string()),
                protocol: None,
                timestamp: None,
            },
        )
    }
}

impl Default for EpistemicStatus {
    fn default() -> Self {
        Self::axiomatic()
    }
}

/// Confidence level with optional bounds
///
/// Represents epistemic confidence as a value between 0.0 and 1.0,
/// optionally with lower and upper bounds for interval estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence {
    /// Point estimate (0.0 - 1.0)
    value: f64,
    /// Lower bound (optional)
    lower: Option<f64>,
    /// Upper bound (optional)
    upper: Option<f64>,
}

impl Confidence {
    /// Create a new confidence with point estimate
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            lower: None,
            upper: None,
        }
    }

    /// Create confidence with interval bounds
    pub fn with_bounds(value: f64, lower: f64, upper: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            lower: Some(lower.clamp(0.0, value)),
            upper: Some(upper.clamp(value, 1.0)),
        }
    }

    /// Certain knowledge (confidence = 1.0)
    pub fn certain() -> Self {
        Self {
            value: 1.0,
            lower: Some(1.0),
            upper: Some(1.0),
        }
    }

    /// Completely uncertain (confidence = 0.5, full range)
    pub fn uncertain() -> Self {
        Self {
            value: 0.5,
            lower: Some(0.0),
            upper: Some(1.0),
        }
    }

    /// Get the point estimate
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the lower bound if available
    pub fn lower_bound(&self) -> Option<f64> {
        self.lower
    }

    /// Get the upper bound if available
    pub fn upper_bound(&self) -> Option<f64> {
        self.upper
    }

    /// Propagate confidence through transformation
    pub fn propagate(&self, factor: f64) -> Self {
        Self {
            value: (self.value * factor).clamp(0.0, 1.0),
            lower: self.lower.map(|l| (l * factor).clamp(0.0, 1.0)),
            upper: self.upper.map(|u| (u * factor).clamp(0.0, 1.0)),
        }
    }

    /// Combine two confidence values (for conjunction)
    pub fn combine(&self, other: &Confidence) -> Self {
        Self {
            value: self.value * other.value,
            lower: match (self.lower, other.lower) {
                (Some(a), Some(b)) => Some(a * b),
                _ => None,
            },
            upper: match (self.upper, other.upper) {
                (Some(a), Some(b)) => Some(a * b),
                _ => None,
            },
        }
    }

    /// Disjunctive combination (for alternatives)
    pub fn disjunction(&self, other: &Confidence) -> Self {
        // P(A or B) = P(A) + P(B) - P(A)*P(B) assuming independence
        let combined = self.value + other.value - (self.value * other.value);
        Self::new(combined)
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self::certain()
    }
}

/// Whether knowledge can be revised
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Revisability {
    /// Cannot be revised (axioms, definitions)
    #[default]
    NonRevisable,
    /// Can be revised under conditions
    Revisable { conditions: Vec<String> },
    /// Must be revised (known to be provisional)
    MustRevise { reason: String },
}

impl Revisability {
    /// Check if this knowledge can be revised
    pub fn is_revisable(&self) -> bool {
        !matches!(self, Revisability::NonRevisable)
    }

    /// Check if this knowledge must be revised
    pub fn must_revise(&self) -> bool {
        matches!(self, Revisability::MustRevise { .. })
    }
}

/// Source of knowledge
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Source {
    /// Axiomatic (by definition)
    Axiom,

    /// From measurement/observation
    Measurement {
        instrument: Option<String>,
        protocol: Option<String>,
        timestamp: Option<String>,
    },

    /// From computation/derivation
    Derivation(String),

    /// From external source
    External {
        uri: String,
        accessed: Option<String>,
    },

    /// From ontology assertion
    OntologyAssertion { ontology: String, term: String },

    /// From model prediction
    ModelPrediction {
        model: String,
        version: Option<String>,
    },

    /// Transformed from another source
    Transformation { original: Box<Source>, via: String },

    /// Human assertion
    HumanAssertion { asserter: Option<String> },

    /// Unknown source (should be rare)
    #[default]
    Unknown,
}

/// Evidence supporting knowledge
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    /// What kind of evidence
    pub kind: EvidenceKind,
    /// Reference to evidence
    pub reference: String,
    /// Strength of evidence
    pub strength: Confidence,
}

impl Evidence {
    /// Create new evidence
    pub fn new(kind: EvidenceKind, reference: impl Into<String>, strength: f64) -> Self {
        Self {
            kind,
            reference: reference.into(),
            strength: Confidence::new(strength),
        }
    }

    /// Create evidence from publication
    pub fn publication(doi: &str, strength: f64) -> Self {
        Self::new(
            EvidenceKind::Publication {
                doi: Some(doi.to_string()),
            },
            doi,
            strength,
        )
    }

    /// Create evidence from experiment
    pub fn experiment(protocol: &str, strength: f64) -> Self {
        Self::new(
            EvidenceKind::Experiment {
                protocol: protocol.to_string(),
            },
            protocol,
            strength,
        )
    }
}

/// Kind of evidence
#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceKind {
    /// Published research
    Publication { doi: Option<String> },
    /// Dataset
    Dataset { uri: String },
    /// Experimental result
    Experiment { protocol: String },
    /// Computational result
    Computation { code_ref: String },
    /// Expert opinion
    ExpertOpinion { source: String },
    /// Verified by external process
    Verified { verifier: String },
    /// Human assertion or review
    HumanAssertion { reviewer: String },
}

impl fmt::Display for EpistemicStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ε(c={:.2}, {:?}, {:?})",
            self.confidence.value(),
            self.revisability,
            self.source
        )
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let (Some(l), Some(u)) = (self.lower, self.upper) {
            write!(f, "{:.2} [{:.2}, {:.2}]", self.value, l, u)
        } else {
            write!(f, "{:.2}", self.value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_propagation() {
        let conf = Confidence::new(0.9);
        let propagated = conf.propagate(0.8);
        assert!((propagated.value() - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_confidence_combine() {
        let a = Confidence::new(0.8);
        let b = Confidence::new(0.9);
        let combined = a.combine(&b);
        assert!((combined.value() - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_axiomatic_status() {
        let status = EpistemicStatus::axiomatic();
        assert!((status.confidence.value() - 1.0).abs() < 0.001);
        assert!(!status.revisability.is_revisable());
    }

    #[test]
    fn test_derived_status() {
        let a = EpistemicStatus::empirical(0.9, Source::Axiom);
        let b = EpistemicStatus::empirical(0.8, Source::Axiom);
        let derived = EpistemicStatus::derived(&[&a, &b], "computation");
        assert!((derived.confidence.value() - 0.72).abs() < 0.001);
        assert!(derived.revisability.is_revisable());
    }
}
