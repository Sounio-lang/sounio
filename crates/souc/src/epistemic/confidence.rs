//! Epistemic status tracking: confidence, revisability, source
//!
//! Every piece of knowledge in Sounio carries information about
//! how certain we are about it and where it came from.
//!
//! # Canonical Definition of ε
//!
//! **ε = posterior mean of Beta(α, β) = α / (α + β)**
//!
//! This is the ONE canonical interpretation. Other uses are views or approximations:
//! - Born rule: `ε = |ψ|²` — valid for pure quantum states (Born is Bayesian under Cox's theorem)
//! - Variational free energy: `-log(ε)` — monotone proxy (Friston active inference)
//! - Fuzzy membership: `ε` as degree — acknowledged approximation, not dissolution
//!
//! # Knightian ε=⊥ is NOT a Beta distribution
//!
//! `KnightianMode::Undefined` is a **type-level flag** indicating the Beta model does not apply.
//! It lives orthogonally to the `confidence` field — `ε=⊥` means "no probability model
//! is defined for this quantity", which is categorically different from `ε=0.0` ("known false").
//! These must never be conflated.
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
//!
//! 4. **Knightian propagates**: Any computation involving `ε=⊥` must return `ε=⊥`.
//!    You cannot compute your way out of Knightian uncertainty without new information.

use super::Transformation;
use std::fmt;

/// Knightian epistemic mode — distinguishes risk from radical uncertainty
///
/// Frank Knight (1921) distinguished:
/// - **Risk**: unknown outcome, but known probability distribution → `Defined`
/// - **Uncertainty**: unknown outcome, unknown distribution → `Undefined`
///
/// This is not a stylistic distinction. They require different decision theories:
/// - Defined confidence admits Bayesian updating, expected utility maximization
/// - Undefined confidence admits only maximin / robust decision rules
///
/// # Type Rules
/// - `Undefined` propagates: any function of `Undefined` returns `Undefined`
/// - `Undefined` + `Defined` via Bayesian fusion is ill-typed (mixing frameworks)
/// - Transition from `Undefined` to `Defined` requires explicit `model_discovery`
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum KnightianMode {
    /// Risk: confidence is a known probability distribution
    /// This is the standard Bayesian case — ε ∈ [0, 1] is meaningful
    #[default]
    Defined,

    /// Knightian uncertainty: the model space itself is unknown
    /// ε=⊥ — not zero confidence, genuinely undefined
    /// Example: dark matter candidates, pandemic novel pathogens
    Undefined,

    /// Dempster-Shafer conflict: multiple irreconcilable evidence bodies
    /// Cannot be resolved by Bayesian fusion without a prior over the conflict
    Conflicted,
}

impl KnightianMode {
    /// Knightian uncertainty propagates through composition
    ///
    /// If any input is `Undefined`, the output must be `Undefined`.
    /// This is the key propagation rule: you cannot compute your way
    /// out of Knightian uncertainty without new information.
    pub fn compose(&self, other: &Self) -> Self {
        match (self, other) {
            (KnightianMode::Undefined, _) | (_, KnightianMode::Undefined) => {
                KnightianMode::Undefined
            }
            (KnightianMode::Conflicted, _) | (_, KnightianMode::Conflicted) => {
                KnightianMode::Conflicted
            }
            _ => KnightianMode::Defined,
        }
    }

    /// Whether confidence has a meaningful value (can be used in Bayesian updates)
    pub fn is_defined(&self) -> bool {
        matches!(self, KnightianMode::Defined)
    }

    /// Whether this is genuine Knightian uncertainty (model space unknown)
    pub fn is_undefined(&self) -> bool {
        matches!(self, KnightianMode::Undefined)
    }
}

/// Complete epistemic status of a knowledge value
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicStatus {
    /// How confident are we? (0.0 - 1.0)
    pub confidence: Confidence,

    /// Knightian mode: is confidence defined at all?
    /// `Undefined` (ε=⊥) is formally distinct from `Defined` with value 0.0
    pub knightian: KnightianMode,

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
            knightian: KnightianMode::Defined,
            revisability: Revisability::NonRevisable,
            source: Source::Axiom,
            evidence: vec![],
        }
    }

    /// Empirical knowledge - uncertain, revisable, from measurement
    pub fn empirical(confidence: f64, source: Source) -> Self {
        Self {
            confidence: Confidence::new(confidence),
            knightian: KnightianMode::Defined,
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".into()],
            },
            source,
            evidence: vec![],
        }
    }

    /// Knightian uncertainty — the model space itself is unknown
    ///
    /// Use when we cannot assign a probability distribution because we don't
    /// even know what the candidate models are. This is formally distinct from
    /// `empirical(0.0, ...)` which means "known to be unlikely".
    ///
    /// # Examples
    /// - Dark matter candidates (model class unknown, not just parameters)
    /// - Novel pandemic pathogens (no base rate exists)
    /// - Black swan financial events (tail distribution not knowable)
    pub fn knightian_undefined(source: Source) -> Self {
        Self {
            // Confidence value is meaningless in Knightian mode, set to 0.5
            // as a placeholder — but KnightianMode::Undefined is the signal.
            confidence: Confidence::new(0.5),
            knightian: KnightianMode::Undefined,
            revisability: Revisability::Revisable {
                conditions: vec!["model_discovery".into()],
            },
            source,
            evidence: vec![],
        }
    }

    /// Dempster-Shafer conflict — irreconcilable evidence bodies
    pub fn conflicted(sources: Vec<Source>) -> Self {
        let evidence = sources
            .into_iter()
            .map(|s| Evidence {
                kind: EvidenceKind::ExpertOpinion {
                    source: format!("{:?}", s),
                },
                reference: "conflict".to_string(),
                strength: Confidence::new(0.5),
            })
            .collect();
        Self {
            confidence: Confidence::new(0.5),
            knightian: KnightianMode::Conflicted,
            revisability: Revisability::Revisable {
                conditions: vec!["conflict_resolution".into()],
            },
            source: Source::Unknown,
            evidence,
        }
    }

    /// Whether this is Knightian uncertainty (ε=⊥)
    pub fn is_knightian_undefined(&self) -> bool {
        self.knightian.is_undefined()
    }

    /// Whether confidence can be meaningfully compared or updated via Bayes
    pub fn is_defined(&self) -> bool {
        self.knightian.is_defined()
    }

    /// Compose two epistemic statuses — Knightian propagates
    ///
    /// If either input is `KnightianMode::Undefined`, the result must be too.
    /// This enforces the key propagation rule: you cannot compute your way
    /// out of Knightian uncertainty.
    pub fn compose(&self, other: &EpistemicStatus, derivation: &str) -> Self {
        let knightian = self.knightian.compose(&other.knightian);
        if knightian.is_undefined() {
            return Self {
                confidence: Confidence::new(0.5),
                knightian: KnightianMode::Undefined,
                revisability: Revisability::Revisable {
                    conditions: vec!["model_discovery".into()],
                },
                source: Source::Derivation(derivation.to_string()),
                evidence: vec![],
            };
        }
        Self::derived(&[self, other], derivation)
    }

    /// Derived knowledge - inherits from dependencies
    ///
    /// Confidence is the product of dependencies (conservative).
    /// Revisable if any dependency is revisable.
    pub fn derived(dependencies: &[&EpistemicStatus], derivation: &str) -> Self {
        // If any dependency is Knightian, the result is Knightian
        let knightian = dependencies
            .iter()
            .fold(KnightianMode::Defined, |acc, e| acc.compose(&e.knightian));

        if knightian.is_undefined() {
            return Self {
                confidence: Confidence::new(0.5),
                knightian: KnightianMode::Undefined,
                revisability: Revisability::Revisable {
                    conditions: vec!["model_discovery".into()],
                },
                source: Source::Derivation(derivation.to_string()),
                evidence: vec![],
            };
        }

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
            knightian: KnightianMode::Defined,
            revisability,
            source: Source::Derivation(derivation.to_string()),
            evidence: vec![],
        }
    }

    /// Propagate epistemic status through a transformation
    ///
    /// The confidence is reduced by the transformation's confidence factor.
    /// Knightian status propagates unchanged.
    pub fn propagate(&self, transformation: &Transformation) -> Self {
        Self {
            confidence: self
                .confidence
                .propagate(transformation.confidence_factor()),
            knightian: self.knightian.clone(),
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
        match self.knightian {
            KnightianMode::Undefined => write!(
                f,
                "ε=⊥ [Knightian: model space unknown, {:?}]",
                self.source
            ),
            KnightianMode::Conflicted => write!(
                f,
                "ε=conflict [Dempster-Shafer: {} conflicting bodies]",
                self.evidence.len()
            ),
            KnightianMode::Defined => write!(
                f,
                "ε(c={:.2}, {:?}, {:?})",
                self.confidence.value(),
                self.revisability,
                self.source
            ),
        }
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

    // ── Knightian uncertainty tests ─────────────────────────────────────────

    #[test]
    fn test_knightian_undefined_is_distinct_from_zero_confidence() {
        // ε=⊥ (Knightian) is formally distinct from ε=0.0 (known unlikely)
        let knightian = EpistemicStatus::knightian_undefined(Source::Unknown);
        let zero_conf = EpistemicStatus::empirical(0.0, Source::Unknown);

        assert!(knightian.is_knightian_undefined());
        assert!(!zero_conf.is_knightian_undefined());

        // Zero confidence is still Bayesian (defined)
        assert!(!knightian.is_defined());
        assert!(zero_conf.is_defined());
    }

    #[test]
    fn test_knightian_display_shows_bottom() {
        let k = EpistemicStatus::knightian_undefined(Source::Unknown);
        let display = format!("{}", k);
        // Must show ε=⊥, not a numeric confidence value
        assert!(display.contains("ε=⊥"), "Display was: {}", display);
        assert!(display.contains("Knightian"), "Display was: {}", display);
    }

    #[test]
    fn test_knightian_propagates_through_derivation() {
        // Any derivation involving a Knightian input must produce Knightian output
        let dark_matter = EpistemicStatus::knightian_undefined(Source::Unknown);
        let detector_efficiency = EpistemicStatus::empirical(0.95, Source::Axiom);

        // Composing Knightian with Bayesian → Knightian (the model space stays unknown)
        let composed = dark_matter.compose(&detector_efficiency, "detection_rate");
        assert!(
            composed.is_knightian_undefined(),
            "Knightian must propagate through composition"
        );
    }

    #[test]
    fn test_knightian_propagates_in_derived() {
        // EpistemicStatus::derived must propagate Knightian from any dependency
        let k = EpistemicStatus::knightian_undefined(Source::Unknown);
        let e = EpistemicStatus::empirical(0.8, Source::Axiom);

        let result = EpistemicStatus::derived(&[&k, &e], "multi_input");
        assert!(
            result.is_knightian_undefined(),
            "derived() must propagate Knightian uncertainty from any dependency"
        );
    }

    #[test]
    fn test_knightian_propagates_through_transform() {
        // propagate() through a transformation must preserve Knightian mode
        use super::super::Transformation;
        let k = EpistemicStatus::knightian_undefined(Source::Unknown);
        let result = k.propagate(&Transformation::function("identity"));
        assert!(
            result.is_knightian_undefined(),
            "Knightian mode must survive transformation"
        );
    }

    #[test]
    fn test_knightian_mode_compose_rules() {
        // Composition rules for KnightianMode
        assert_eq!(
            KnightianMode::Undefined.compose(&KnightianMode::Defined),
            KnightianMode::Undefined
        );
        assert_eq!(
            KnightianMode::Defined.compose(&KnightianMode::Undefined),
            KnightianMode::Undefined
        );
        assert_eq!(
            KnightianMode::Defined.compose(&KnightianMode::Defined),
            KnightianMode::Defined
        );
        assert_eq!(
            KnightianMode::Conflicted.compose(&KnightianMode::Defined),
            KnightianMode::Conflicted
        );
    }
}
