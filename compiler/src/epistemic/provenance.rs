//! Provenance tracking: Φ functor trace
//!
//! Every transformation a value undergoes is recorded in its type.
//! This enables complete audit trails and reproducibility.
//!
//! # Design
//!
//! Provenance is tracked as a Directed Acyclic Graph (DAG), inspired by
//! categorical semantics (Ologs). This allows representing:
//! - Multiple sources contributing to a single value
//! - Branching and merging of data flows
//! - Complete transformation history
//!
//! # Example
//!
//! ```text
//! Source(sensor_A) ──┐
//!                    ├─→ Transform(average) ──→ Transform(scale) ──→ Value
//! Source(sensor_B) ──┘
//! ```

use std::collections::VecDeque;

/// Complete provenance of a knowledge value
#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    /// Ordered sequence of transformations
    pub trace: FunctorTrace,

    /// Original creation point
    pub origin: Origin,

    /// Hash for integrity verification
    pub integrity_hash: Option<String>,
}

impl Provenance {
    /// Provenance for a literal value
    pub fn literal() -> Self {
        Self {
            trace: FunctorTrace::empty(),
            origin: Origin::Literal,
            integrity_hash: None,
        }
    }

    /// Provenance from external source
    pub fn external(uri: &str) -> Self {
        Self {
            trace: FunctorTrace::empty(),
            origin: Origin::External {
                uri: uri.to_string(),
            },
            integrity_hash: None,
        }
    }

    /// Provenance from computation
    pub fn computed(function: &str) -> Self {
        Self {
            trace: FunctorTrace::empty(),
            origin: Origin::Computed {
                function: function.to_string(),
            },
            integrity_hash: None,
        }
    }

    /// Provenance from ontology assertion
    pub fn ontology_assertion(ontology: &str, term: &str) -> Self {
        Self {
            trace: FunctorTrace::empty(),
            origin: Origin::OntologyAssertion {
                ontology: ontology.to_string(),
                term: term.to_string(),
            },
            integrity_hash: None,
        }
    }

    /// Extend provenance with new transformation
    pub fn extend(&self, transformation: Transformation) -> Self {
        Self {
            trace: self.trace.append(transformation),
            origin: self.origin.clone(),
            integrity_hash: None, // Invalidated by transformation
        }
    }

    /// Check if provenance includes specific transformation type
    pub fn includes(&self, kind: TransformationKind) -> bool {
        self.trace.steps.iter().any(|t| t.kind == kind)
    }

    /// Get the full transformation path as string
    pub fn path_string(&self) -> String {
        if self.trace.steps.is_empty() {
            return format!("{:?}", self.origin);
        }

        let steps: Vec<_> = self.trace.steps.iter().map(|t| t.name.clone()).collect();

        format!("{:?} → {}", self.origin, steps.join(" → "))
    }

    /// Compute total confidence factor through the transformation chain
    pub fn total_confidence_factor(&self) -> f64 {
        self.trace
            .steps
            .iter()
            .map(|t| t.confidence_factor)
            .product()
    }

    /// Get the number of transformations
    pub fn depth(&self) -> usize {
        self.trace.steps.len()
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Self::literal()
    }
}

/// Sequence of transformations (the Φ functor trace)
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FunctorTrace {
    /// Ordered transformation steps
    pub steps: VecDeque<Transformation>,
    /// Maximum trace length before compression
    pub max_length: usize,
}

impl FunctorTrace {
    /// Create an empty trace
    pub fn empty() -> Self {
        Self {
            steps: VecDeque::new(),
            max_length: 100,
        }
    }

    /// Create a trace with custom max length
    pub fn with_max_length(max_length: usize) -> Self {
        Self {
            steps: VecDeque::new(),
            max_length,
        }
    }

    /// Append a transformation to the trace
    pub fn append(&self, transformation: Transformation) -> Self {
        let mut new_steps = self.steps.clone();
        new_steps.push_back(transformation);

        // Compress if too long (keep first and last N)
        if new_steps.len() > self.max_length {
            let compressed = Self::compress(&new_steps);
            Self {
                steps: compressed,
                max_length: self.max_length,
            }
        } else {
            Self {
                steps: new_steps,
                max_length: self.max_length,
            }
        }
    }

    /// Compress a trace that's too long
    fn compress(steps: &VecDeque<Transformation>) -> VecDeque<Transformation> {
        // Keep first 10 and last 10, mark middle as compressed
        let mut result = VecDeque::new();
        let len = steps.len();
        let keep = 10;

        for (i, step) in steps.iter().enumerate() {
            if i < keep || i >= len - keep {
                result.push_back(step.clone());
            } else if i == keep {
                result.push_back(Transformation::compressed(len - 2 * keep));
            }
        }

        result
    }

    /// Compose this trace with another
    pub fn compose(&self, other: &FunctorTrace) -> FunctorTrace {
        let mut combined = self.steps.clone();
        combined.extend(other.steps.iter().cloned());

        FunctorTrace {
            steps: combined,
            max_length: self.max_length,
        }
    }

    /// Check if the trace is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the length of the trace
    pub fn len(&self) -> usize {
        self.steps.len()
    }
}

/// A single transformation in the functor trace
#[derive(Debug, Clone, PartialEq)]
pub struct Transformation {
    /// Name of the transformation
    pub name: String,

    /// Kind of transformation
    pub kind: TransformationKind,

    /// Input types (for type-level tracking)
    pub inputs: Vec<String>,

    /// Output type
    pub output: String,

    /// Confidence factor (how much does this affect confidence?)
    pub confidence_factor: f64,

    /// Source location where transformation occurred
    pub location: Option<crate::common::Span>,

    /// Additional metadata
    pub metadata: TransformationMetadata,
}

impl Transformation {
    /// Create a new transformation
    pub fn new(name: &str, kind: TransformationKind) -> Self {
        Self {
            name: name.to_string(),
            kind,
            inputs: vec![],
            output: String::new(),
            confidence_factor: 1.0,
            location: None,
            metadata: TransformationMetadata::default(),
        }
    }

    /// Create a function transformation
    pub fn function(name: &str) -> Self {
        Self::new(name, TransformationKind::Function)
    }

    /// Create a conversion transformation
    pub fn conversion(name: &str) -> Self {
        Self::new(name, TransformationKind::Conversion)
    }

    /// Create a translation transformation
    pub fn translation(name: &str) -> Self {
        Self::new(name, TransformationKind::Translation).with_confidence(0.95)
    }

    /// Create a compressed placeholder
    pub fn compressed(count: usize) -> Self {
        Self {
            name: format!("[{} steps compressed]", count),
            kind: TransformationKind::Compressed,
            inputs: vec![],
            output: String::new(),
            confidence_factor: 1.0,
            location: None,
            metadata: TransformationMetadata::default(),
        }
    }

    /// Get the confidence factor
    pub fn confidence_factor(&self) -> f64 {
        self.confidence_factor
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set confidence factor
    pub fn with_confidence(mut self, factor: f64) -> Self {
        self.confidence_factor = factor;
        self
    }

    /// Set input types
    pub fn with_inputs(mut self, inputs: Vec<String>) -> Self {
        self.inputs = inputs;
        self
    }

    /// Set output type
    pub fn with_output(mut self, output: String) -> Self {
        self.output = output;
        self
    }

    /// Set location
    pub fn with_location(mut self, location: crate::common::Span) -> Self {
        self.location = Some(location);
        self
    }
}

/// Kind of transformation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransformationKind {
    /// Pure function application
    #[default]
    Function,
    /// Type conversion/coercion
    Conversion,
    /// Ontology translation
    Translation,
    /// Aggregation/reduction
    Aggregation,
    /// Filtering/selection
    Filter,
    /// Statistical transformation
    Statistical,
    /// Machine learning inference
    MLInference,
    /// External API call
    ExternalCall,
    /// Compressed (placeholder for many steps)
    Compressed,
}

impl TransformationKind {
    /// Get default confidence factor for this kind
    pub fn default_confidence_factor(&self) -> f64 {
        match self {
            TransformationKind::Function => 1.0,
            TransformationKind::Conversion => 0.99,
            TransformationKind::Translation => 0.95,
            TransformationKind::Aggregation => 0.98,
            TransformationKind::Filter => 1.0,
            TransformationKind::Statistical => 0.95,
            TransformationKind::MLInference => 0.85,
            TransformationKind::ExternalCall => 0.90,
            TransformationKind::Compressed => 1.0,
        }
    }
}

/// Metadata for a transformation
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TransformationMetadata {
    /// Git commit hash if available
    pub commit: Option<String>,
    /// Timestamp
    pub timestamp: Option<String>,
    /// Additional key-value pairs
    pub extra: Vec<(String, String)>,
}

impl TransformationMetadata {
    /// Create empty metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Set commit hash
    pub fn with_commit(mut self, commit: &str) -> Self {
        self.commit = Some(commit.to_string());
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: &str) -> Self {
        self.timestamp = Some(timestamp.to_string());
        self
    }

    /// Add extra metadata
    pub fn with_extra(mut self, key: &str, value: &str) -> Self {
        self.extra.push((key.to_string(), value.to_string()));
        self
    }
}

/// Origin of a knowledge value
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Origin {
    /// Created as a literal in source code
    #[default]
    Literal,
    /// Loaded from external source
    External { uri: String },
    /// Result of computation
    Computed { function: String },
    /// User input
    UserInput { context: String },
    /// From database query
    Database { query: String, connection: String },
    /// From ontology assertion
    OntologyAssertion { ontology: String, term: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_literal() {
        let prov = Provenance::literal();
        assert_eq!(prov.origin, Origin::Literal);
        assert!(prov.trace.is_empty());
    }

    #[test]
    fn test_provenance_extend() {
        let prov = Provenance::literal();
        let extended = prov.extend(Transformation::function("add"));

        assert_eq!(extended.trace.len(), 1);
        assert_eq!(extended.trace.steps[0].name, "add");
    }

    #[test]
    fn test_functor_trace_compose() {
        let t1 = FunctorTrace::empty().append(Transformation::function("f"));
        let t2 = FunctorTrace::empty().append(Transformation::function("g"));

        let composed = t1.compose(&t2);
        assert_eq!(composed.len(), 2);
    }

    #[test]
    fn test_transformation_confidence() {
        let t = Transformation::translation("ChEBI_to_FHIR");
        assert!((t.confidence_factor - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_provenance_path_string() {
        let prov = Provenance::literal()
            .extend(Transformation::function("measure"))
            .extend(Transformation::function("calibrate"))
            .extend(Transformation::function("convert"));

        let path = prov.path_string();
        assert!(path.contains("measure"));
        assert!(path.contains("calibrate"));
        assert!(path.contains("convert"));
    }
}
