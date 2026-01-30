//! Core EpistemicValue Type for Composition Algebra
//!
//! This module provides the runtime representation of epistemic knowledge
//! with confidence, ontology binding, and provenance tracking.
//!
//! # Type Structure
//!
//! ```text
//! EpistemicValue<T> {
//!     value: T,                    // The wrapped value
//!     confidence: ConfidenceValue, // ε ∈ [0, 1]
//!     ontology: HashSet<OntologyRef>,  // δ - domain bindings
//!     provenance: ProvenanceNode,  // Φ - derivation history
//! }
//! ```

use super::confidence::ConfidenceValue;
use super::provenance::{ProvenanceNode, SourceInfo};
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

/// Reference to an ontology term for domain binding
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OntologyRef {
    /// Ontology prefix (e.g., "BFO", "PATO", "ChEBI")
    pub prefix: String,
    /// Term identifier
    pub term: String,
}

impl OntologyRef {
    /// Create a new ontology reference
    pub fn new(prefix: impl Into<String>, term: impl Into<String>) -> Self {
        OntologyRef {
            prefix: prefix.into(),
            term: term.into(),
        }
    }

    /// Parse from string like "BFO:0000001"
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() == 2 {
            Some(OntologyRef::new(parts[0], parts[1]))
        } else {
            None
        }
    }

    /// Get the full URI
    pub fn to_uri(&self) -> String {
        format!("{}:{}", self.prefix, self.term)
    }
}

impl fmt::Display for OntologyRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.prefix, self.term)
    }
}

/// Error when extracting a value below confidence threshold
#[derive(Debug, Clone)]
pub struct ExtractError {
    /// The confidence of the value
    pub actual_confidence: ConfidenceValue,
    /// The required threshold
    pub required_threshold: ConfidenceValue,
    /// Description for debugging
    pub message: String,
}

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cannot extract: confidence {} < threshold {} ({})",
            self.actual_confidence, self.required_threshold, self.message
        )
    }
}

impl std::error::Error for ExtractError {}

/// An epistemic value with confidence, ontology binding, and provenance
///
/// This is the core type for the epistemic composition algebra.
/// It wraps any value with epistemic metadata.
#[derive(Debug, Clone)]
pub struct EpistemicValue<T> {
    /// The wrapped value
    pub(crate) value: T,
    /// Confidence level ε ∈ [0, 1]
    pub(crate) confidence: ConfidenceValue,
    /// Domain ontology bindings
    pub(crate) ontology: HashSet<OntologyRef>,
    /// Provenance tracking
    pub(crate) provenance: ProvenanceNode,
}

impl<T> EpistemicValue<T> {
    /// Create a new epistemic value with full metadata
    pub fn new(
        value: T,
        confidence: ConfidenceValue,
        ontology: HashSet<OntologyRef>,
        provenance: ProvenanceNode,
    ) -> Self {
        EpistemicValue {
            value,
            confidence,
            ontology,
            provenance,
        }
    }

    /// Create a certain (ε = 1.0) epistemic value with no ontology
    ///
    /// This is the LIFT functor: τ → Knowledge[τ, 1.0, ∅, Primitive]
    pub fn certain(value: T) -> Self {
        EpistemicValue {
            value,
            confidence: ConfidenceValue::certain(),
            ontology: HashSet::new(),
            provenance: ProvenanceNode::primitive(),
        }
    }

    /// Create an epistemic value with given confidence
    pub fn with_confidence(value: T, confidence: f64) -> Self {
        EpistemicValue {
            value,
            confidence: ConfidenceValue::new(confidence).unwrap_or(ConfidenceValue::uncertain()),
            ontology: HashSet::new(),
            provenance: ProvenanceNode::primitive(),
        }
    }

    /// Create an epistemic value from a source
    pub fn from_source(value: T, confidence: f64, source: SourceInfo) -> Self {
        EpistemicValue {
            value,
            confidence: ConfidenceValue::new(confidence).unwrap_or(ConfidenceValue::uncertain()),
            ontology: HashSet::new(),
            provenance: ProvenanceNode::source(source),
        }
    }

    /// Get a reference to the wrapped value
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get the confidence level
    pub fn confidence(&self) -> ConfidenceValue {
        self.confidence
    }

    /// Get the ontology bindings
    pub fn ontology(&self) -> &HashSet<OntologyRef> {
        &self.ontology
    }

    /// Get the provenance
    pub fn provenance(&self) -> &ProvenanceNode {
        &self.provenance
    }

    /// Add an ontology binding
    pub fn with_ontology(mut self, ont_ref: OntologyRef) -> Self {
        self.ontology.insert(ont_ref);
        self
    }

    /// Add multiple ontology bindings
    pub fn with_ontologies(mut self, refs: impl IntoIterator<Item = OntologyRef>) -> Self {
        self.ontology.extend(refs);
        self
    }

    /// Set the provenance
    pub fn with_provenance(mut self, provenance: ProvenanceNode) -> Self {
        self.provenance = provenance;
        self
    }

    /// Safe extraction with threshold check
    ///
    /// Returns Some(&T) if confidence >= threshold, None otherwise.
    /// This is the EXTRACT operator: Knowledge[τ] → Option<τ>
    pub fn extract(&self, min_confidence: ConfidenceValue) -> Option<&T> {
        if self.confidence.meets(min_confidence) {
            Some(&self.value)
        } else {
            None
        }
    }

    /// Extract with explicit error
    pub fn try_extract(&self, min_confidence: ConfidenceValue) -> Result<&T, ExtractError> {
        if self.confidence.meets(min_confidence) {
            Ok(&self.value)
        } else {
            Err(ExtractError {
                actual_confidence: self.confidence,
                required_threshold: min_confidence,
                message: format!(
                    "Value has {} confidence but {} required",
                    self.confidence, min_confidence
                ),
            })
        }
    }

    /// Unsafe extraction - bypasses confidence check
    ///
    /// # Safety
    /// Caller must handle uncertainty externally. Use with caution.
    pub fn force_extract(&self) -> &T {
        &self.value
    }

    /// Extract and consume the value
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Check if this value meets a confidence threshold
    pub fn is_confident(&self, threshold: ConfidenceValue) -> bool {
        self.confidence.meets(threshold)
    }

    /// Check if this is certain (ε = 1.0)
    pub fn is_certain(&self) -> bool {
        self.confidence.value() >= 1.0 - f64::EPSILON
    }

    /// Check if this is uncertain (ε < 0.5)
    pub fn is_uncertain(&self) -> bool {
        self.confidence.value() < 0.5
    }

    /// Map the inner value while preserving epistemic metadata
    pub fn map<U, F>(self, f: F) -> EpistemicValue<U>
    where
        F: FnOnce(T) -> U,
    {
        EpistemicValue {
            value: f(self.value),
            confidence: self.confidence,
            ontology: self.ontology,
            provenance: ProvenanceNode::derived("map", vec![self.provenance]),
        }
    }

    /// Map with a confidence factor for the transformation
    pub fn map_with_confidence<U, F>(self, f: F, confidence_factor: f64) -> EpistemicValue<U>
    where
        F: FnOnce(T) -> U,
    {
        EpistemicValue {
            value: f(self.value),
            confidence: self.confidence.scale(confidence_factor),
            ontology: self.ontology,
            provenance: ProvenanceNode::derived("map", vec![self.provenance]),
        }
    }

    /// Observe evidence that affects confidence
    pub fn observe(self, positive: bool, strength: f64) -> Self {
        let new_confidence = if positive {
            // Positive evidence increases confidence
            self.confidence.value() + strength * (1.0 - self.confidence.value())
        } else {
            // Negative evidence decreases confidence
            self.confidence.value() * (1.0 - strength)
        };

        EpistemicValue {
            value: self.value,
            confidence: ConfidenceValue::new(new_confidence)
                .unwrap_or(ConfidenceValue::uncertain()),
            ontology: self.ontology,
            provenance: ProvenanceNode::updated(self.provenance, "observe"),
        }
    }
}

impl<T: Clone> EpistemicValue<T> {
    /// Clone the inner value
    pub fn clone_value(&self) -> T {
        self.value.clone()
    }
}

impl<T: Default> Default for EpistemicValue<T> {
    fn default() -> Self {
        EpistemicValue::certain(T::default())
    }
}

impl<T: fmt::Display> fmt::Display for EpistemicValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.value, self.confidence)
    }
}

/// Macro for creating ontology sets
#[macro_export]
macro_rules! ontology {
    ($($prefix:ident : $term:expr),* $(,)?) => {{
        let mut set = std::collections::HashSet::new();
        $(
            set.insert($crate::epistemic::composition::OntologyRef::new(
                stringify!($prefix),
                $term
            ));
        )*
        set
    }};
    ($($s:expr),* $(,)?) => {{
        let mut set = std::collections::HashSet::new();
        $(
            if let Some(r) = $crate::epistemic::composition::OntologyRef::parse($s) {
                set.insert(r);
            }
        )*
        set
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certain_value() {
        let v: EpistemicValue<f64> = EpistemicValue::certain(42.0);
        assert!(v.is_certain());
        assert!((v.value() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_confidence() {
        let v = EpistemicValue::with_confidence(100, 0.75);
        assert!((v.confidence().value() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_success() {
        let v = EpistemicValue::with_confidence(42, 0.8);
        let threshold = ConfidenceValue::new(0.7).unwrap();
        assert!(v.extract(threshold).is_some());
        assert_eq!(*v.extract(threshold).unwrap(), 42);
    }

    #[test]
    fn test_extract_failure() {
        let v = EpistemicValue::with_confidence(42, 0.5);
        let threshold = ConfidenceValue::new(0.7).unwrap();
        assert!(v.extract(threshold).is_none());
    }

    #[test]
    fn test_map() {
        let v = EpistemicValue::with_confidence(10, 0.8);
        let doubled = v.map(|x| x * 2);
        assert_eq!(*doubled.value(), 20);
        assert!((doubled.confidence().value() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_map_with_confidence() {
        let v = EpistemicValue::with_confidence(10, 0.8);
        let transformed = v.map_with_confidence(|x| x * 2, 0.9);
        assert_eq!(*transformed.value(), 20);
        // 0.8 * 0.9 = 0.72
        assert!((transformed.confidence().value() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_observe_positive() {
        let v = EpistemicValue::with_confidence(true, 0.5);
        let updated = v.observe(true, 0.3);
        // 0.5 + 0.3 * 0.5 = 0.65
        assert!((updated.confidence().value() - 0.65).abs() < 1e-10);
    }

    #[test]
    fn test_observe_negative() {
        let v = EpistemicValue::with_confidence(true, 0.8);
        let updated = v.observe(false, 0.25);
        // 0.8 * 0.75 = 0.6
        assert!((updated.confidence().value() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_ontology_ref() {
        let r = OntologyRef::parse("BFO:0000001").unwrap();
        assert_eq!(r.prefix, "BFO");
        assert_eq!(r.term, "0000001");
        assert_eq!(r.to_uri(), "BFO:0000001");
    }

    #[test]
    fn test_with_ontology() {
        let v = EpistemicValue::certain(42.0)
            .with_ontology(OntologyRef::new("PATO", "mass"))
            .with_ontology(OntologyRef::new("UO", "kilogram"));

        assert_eq!(v.ontology().len(), 2);
    }
}
