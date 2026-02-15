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

use std::collections::{HashMap, VecDeque};

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

// ============================================================
// Provenance Query API
// ============================================================

/// Predicate for filtering transformations
#[derive(Debug, Clone)]
pub enum TransformationFilter {
    /// Match specific transformation kind
    ByKind(TransformationKind),
    /// Match confidence above threshold
    MinConfidence(f64),
    /// Match confidence below threshold
    MaxConfidence(f64),
    /// Match transformations with specific metadata key (in extra pairs)
    HasMetadata(String),
    /// Match transformations with specific metadata key-value (in extra pairs)
    MetadataEquals(String, String),
    /// Match any of the given filters (OR)
    Any(Vec<TransformationFilter>),
    /// Match all of the given filters (AND)
    All(Vec<TransformationFilter>),
}

/// Result of a provenance query
#[derive(Debug, Clone)]
pub struct ProvenanceQueryResult {
    /// Matched transformations with their indices in the original chain
    pub matched_transformations: Vec<(usize, Transformation)>,
    /// Total number of transformations scanned
    pub total_scanned: usize,
    /// Number of matches
    pub match_count: usize,
    /// Product of matched transformations' confidence factors
    pub confidence_product: f64,
}

/// Provenance chain summary
#[derive(Debug, Clone)]
pub struct ProvenanceSummary {
    /// The origin of the provenance chain
    pub origin: Origin,
    /// Total number of transformation steps
    pub total_steps: usize,
    /// Total confidence (product of all steps)
    pub total_confidence: f64,
    /// Count of each transformation kind
    pub kind_counts: HashMap<TransformationKind, usize>,
    /// Weakest link: (index, confidence_factor) of lowest-confidence step
    pub min_confidence_step: Option<(usize, f64)>,
    /// Whether any step is a user override (ExternalCall kind used as proxy)
    pub has_user_override: bool,
    /// Whether any step is ML inference
    pub has_ml_inference: bool,
}

/// Check if a transformation matches a filter predicate
fn matches_filter(t: &Transformation, filter: &TransformationFilter) -> bool {
    match filter {
        TransformationFilter::ByKind(kind) => t.kind == *kind,
        TransformationFilter::MinConfidence(threshold) => t.confidence_factor >= *threshold,
        TransformationFilter::MaxConfidence(threshold) => t.confidence_factor <= *threshold,
        TransformationFilter::HasMetadata(key) => {
            t.metadata.extra.iter().any(|(k, _)| k == key)
                || (key == "commit" && t.metadata.commit.is_some())
                || (key == "timestamp" && t.metadata.timestamp.is_some())
        }
        TransformationFilter::MetadataEquals(key, value) => {
            t.metadata.extra.iter().any(|(k, v)| k == key && v == value)
                || (key == "commit"
                    && t.metadata.commit.as_deref() == Some(value.as_str()))
                || (key == "timestamp"
                    && t.metadata.timestamp.as_deref() == Some(value.as_str()))
        }
        TransformationFilter::Any(filters) => filters.iter().any(|f| matches_filter(t, f)),
        TransformationFilter::All(filters) => filters.iter().all(|f| matches_filter(t, f)),
    }
}

impl Provenance {
    /// Filter transformations matching a predicate
    pub fn query(&self, filter: &TransformationFilter) -> ProvenanceQueryResult {
        let total_scanned = self.trace.steps.len();
        let mut matched = Vec::new();
        let mut confidence_product = 1.0;

        for (i, t) in self.trace.steps.iter().enumerate() {
            if matches_filter(t, filter) {
                confidence_product *= t.confidence_factor;
                matched.push((i, t.clone()));
            }
        }

        let match_count = matched.len();
        ProvenanceQueryResult {
            matched_transformations: matched,
            total_scanned,
            match_count,
            confidence_product,
        }
    }

    /// Get a summary of the provenance chain
    pub fn summarize(&self) -> ProvenanceSummary {
        let mut kind_counts: HashMap<TransformationKind, usize> = HashMap::new();
        let mut min_step: Option<(usize, f64)> = None;
        let mut has_user_override = false;
        let mut has_ml_inference = false;

        for (i, t) in self.trace.steps.iter().enumerate() {
            *kind_counts.entry(t.kind).or_insert(0) += 1;

            match min_step {
                None => min_step = Some((i, t.confidence_factor)),
                Some((_, prev_min)) if t.confidence_factor < prev_min => {
                    min_step = Some((i, t.confidence_factor));
                }
                _ => {}
            }

            if t.kind == TransformationKind::ExternalCall {
                has_user_override = true;
            }
            if t.kind == TransformationKind::MLInference {
                has_ml_inference = true;
            }
        }

        ProvenanceSummary {
            origin: self.origin.clone(),
            total_steps: self.trace.steps.len(),
            total_confidence: self.total_confidence_factor(),
            kind_counts,
            min_confidence_step: min_step,
            has_user_override,
            has_ml_inference,
        }
    }

    /// Extract a sub-chain between two indices (inclusive)
    pub fn subchain(&self, start: usize, end: usize) -> Option<Vec<Transformation>> {
        if start > end || end >= self.trace.steps.len() {
            return None;
        }
        Some(
            self.trace
                .steps
                .iter()
                .skip(start)
                .take(end - start + 1)
                .cloned()
                .collect(),
        )
    }

    /// Find the weakest link (lowest confidence step)
    pub fn weakest_link(&self) -> Option<(usize, &Transformation)> {
        self.trace
            .steps
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.confidence_factor
                    .partial_cmp(&b.confidence_factor)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Check if provenance contains any user overrides (ExternalCall kind)
    pub fn has_user_override(&self) -> bool {
        self.trace
            .steps
            .iter()
            .any(|t| t.kind == TransformationKind::ExternalCall)
    }

    /// Check if provenance involves ML inference
    pub fn has_ml_inference(&self) -> bool {
        self.trace
            .steps
            .iter()
            .any(|t| t.kind == TransformationKind::MLInference)
    }

    /// Get all transformation kinds present in the chain
    pub fn transformation_kinds(&self) -> Vec<TransformationKind> {
        let mut kinds: Vec<TransformationKind> = self
            .trace
            .steps
            .iter()
            .map(|t| t.kind)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        kinds.sort_by_key(|k| format!("{:?}", k));
        kinds
    }

    /// Count transformations by kind
    pub fn count_by_kind(&self, kind: &TransformationKind) -> usize {
        self.trace.steps.iter().filter(|t| t.kind == *kind).count()
    }
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

    // ================================================================
    // Provenance Query API tests
    // ================================================================

    /// Helper: build a multi-step provenance chain for testing
    fn sample_provenance() -> Provenance {
        Provenance::external("https://sensor.example.com")
            .extend(Transformation::function("read_sensor").with_confidence(1.0))
            .extend(Transformation::new("calibrate", TransformationKind::Conversion).with_confidence(0.98))
            .extend(Transformation::new("scale", TransformationKind::Statistical).with_confidence(0.92))
            .extend(Transformation::new("predict", TransformationKind::MLInference).with_confidence(0.85))
            .extend(Transformation::new("override_val", TransformationKind::ExternalCall).with_confidence(0.90))
    }

    #[test]
    fn test_query_by_kind() {
        let prov = sample_provenance();
        let result = prov.query(&TransformationFilter::ByKind(TransformationKind::MLInference));
        assert_eq!(result.match_count, 1);
        assert_eq!(result.matched_transformations[0].0, 3); // index 3
        assert_eq!(result.matched_transformations[0].1.name, "predict");
    }

    #[test]
    fn test_query_min_confidence() {
        let prov = sample_provenance();
        let result = prov.query(&TransformationFilter::MinConfidence(0.95));
        // read_sensor(1.0) and calibrate(0.98)
        assert_eq!(result.match_count, 2);
    }

    #[test]
    fn test_query_max_confidence() {
        let prov = sample_provenance();
        let result = prov.query(&TransformationFilter::MaxConfidence(0.90));
        // predict(0.85) and override_val(0.90)
        assert_eq!(result.match_count, 2);
    }

    #[test]
    fn test_query_any_filter() {
        let prov = sample_provenance();
        let filter = TransformationFilter::Any(vec![
            TransformationFilter::ByKind(TransformationKind::MLInference),
            TransformationFilter::ByKind(TransformationKind::ExternalCall),
        ]);
        let result = prov.query(&filter);
        assert_eq!(result.match_count, 2);
    }

    #[test]
    fn test_query_all_filter() {
        let prov = sample_provenance();
        let filter = TransformationFilter::All(vec![
            TransformationFilter::ByKind(TransformationKind::Statistical),
            TransformationFilter::MinConfidence(0.90),
        ]);
        let result = prov.query(&filter);
        assert_eq!(result.match_count, 1);
        assert_eq!(result.matched_transformations[0].1.name, "scale");
    }

    #[test]
    fn test_query_has_metadata() {
        let prov = Provenance::literal()
            .extend({
                let mut t = Transformation::function("step1");
                t.metadata = TransformationMetadata::new().with_extra("author", "alice");
                t
            })
            .extend(Transformation::function("step2"));

        let result = prov.query(&TransformationFilter::HasMetadata("author".to_string()));
        assert_eq!(result.match_count, 1);
        assert_eq!(result.matched_transformations[0].1.name, "step1");
    }

    #[test]
    fn test_query_metadata_equals() {
        let prov = Provenance::literal()
            .extend({
                let mut t = Transformation::function("step1");
                t.metadata = TransformationMetadata::new().with_extra("lab", "boston");
                t
            })
            .extend({
                let mut t = Transformation::function("step2");
                t.metadata = TransformationMetadata::new().with_extra("lab", "zurich");
                t
            });

        let result = prov.query(&TransformationFilter::MetadataEquals(
            "lab".to_string(),
            "zurich".to_string(),
        ));
        assert_eq!(result.match_count, 1);
        assert_eq!(result.matched_transformations[0].1.name, "step2");
    }

    #[test]
    fn test_query_has_metadata_commit() {
        let prov = Provenance::literal().extend({
            let mut t = Transformation::function("deploy");
            t.metadata = TransformationMetadata::new().with_commit("abc123");
            t
        });

        let result = prov.query(&TransformationFilter::HasMetadata("commit".to_string()));
        assert_eq!(result.match_count, 1);
    }

    #[test]
    fn test_query_empty_chain() {
        let prov = Provenance::literal();
        let result = prov.query(&TransformationFilter::ByKind(TransformationKind::Function));
        assert_eq!(result.match_count, 0);
        assert_eq!(result.total_scanned, 0);
        assert!((result.confidence_product - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_query_confidence_product() {
        let prov = sample_provenance();
        let result = prov.query(&TransformationFilter::MaxConfidence(0.95));
        // scale(0.92), predict(0.85), override_val(0.90)
        let expected = 0.92 * 0.85 * 0.90;
        assert!((result.confidence_product - expected).abs() < 0.001);
    }

    #[test]
    fn test_summarize() {
        let prov = sample_provenance();
        let summary = prov.summarize();

        assert_eq!(summary.total_steps, 5);
        assert_eq!(summary.origin, Origin::External { uri: "https://sensor.example.com".to_string() });
        assert!(summary.has_user_override);
        assert!(summary.has_ml_inference);
        assert_eq!(*summary.kind_counts.get(&TransformationKind::Function).unwrap_or(&0), 1);
        assert_eq!(*summary.kind_counts.get(&TransformationKind::MLInference).unwrap_or(&0), 1);

        // Weakest link should be predict at 0.85
        let (idx, conf) = summary.min_confidence_step.unwrap();
        assert_eq!(idx, 3);
        assert!((conf - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_subchain() {
        let prov = sample_provenance();
        let sub = prov.subchain(1, 3).unwrap();
        assert_eq!(sub.len(), 3);
        assert_eq!(sub[0].name, "calibrate");
        assert_eq!(sub[2].name, "predict");
    }

    #[test]
    fn test_subchain_out_of_bounds() {
        let prov = sample_provenance();
        assert!(prov.subchain(3, 1).is_none()); // start > end
        assert!(prov.subchain(0, 100).is_none()); // end out of bounds
    }

    #[test]
    fn test_weakest_link() {
        let prov = sample_provenance();
        let (idx, t) = prov.weakest_link().unwrap();
        assert_eq!(idx, 3);
        assert_eq!(t.name, "predict");
        assert!((t.confidence_factor - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_weakest_link_empty() {
        let prov = Provenance::literal();
        assert!(prov.weakest_link().is_none());
    }

    #[test]
    fn test_has_user_override() {
        let prov = sample_provenance();
        assert!(prov.has_user_override());

        let prov2 = Provenance::literal().extend(Transformation::function("f"));
        assert!(!prov2.has_user_override());
    }

    #[test]
    fn test_has_ml_inference() {
        let prov = sample_provenance();
        assert!(prov.has_ml_inference());

        let prov2 = Provenance::literal().extend(Transformation::function("f"));
        assert!(!prov2.has_ml_inference());
    }

    #[test]
    fn test_transformation_kinds() {
        let prov = sample_provenance();
        let kinds = prov.transformation_kinds();
        assert!(kinds.contains(&TransformationKind::Function));
        assert!(kinds.contains(&TransformationKind::MLInference));
        assert!(kinds.contains(&TransformationKind::Statistical));
        assert!(!kinds.contains(&TransformationKind::Compressed));
    }

    #[test]
    fn test_count_by_kind() {
        let prov = sample_provenance();
        assert_eq!(prov.count_by_kind(&TransformationKind::Function), 1);
        assert_eq!(prov.count_by_kind(&TransformationKind::MLInference), 1);
        assert_eq!(prov.count_by_kind(&TransformationKind::Compressed), 0);
    }

    #[test]
    fn test_query_nested_all_in_any() {
        let prov = sample_provenance();
        // (Statistical AND MinConfidence(0.90)) OR MLInference
        let filter = TransformationFilter::Any(vec![
            TransformationFilter::All(vec![
                TransformationFilter::ByKind(TransformationKind::Statistical),
                TransformationFilter::MinConfidence(0.90),
            ]),
            TransformationFilter::ByKind(TransformationKind::MLInference),
        ]);
        let result = prov.query(&filter);
        // scale(Statistical, 0.92) matches first branch, predict(ML) matches second
        assert_eq!(result.match_count, 2);
    }

    #[test]
    fn test_summarize_empty_chain() {
        let prov = Provenance::literal();
        let summary = prov.summarize();
        assert_eq!(summary.total_steps, 0);
        assert!((summary.total_confidence - 1.0).abs() < f64::EPSILON);
        assert!(summary.min_confidence_step.is_none());
        assert!(!summary.has_user_override);
        assert!(!summary.has_ml_inference);
    }
}
