//! Type Compatibility Checking using Semantic Distance
//!
//! This is where Sounio differs fundamentally from other languages:
//! types are compatible if they're "close enough" in semantic space.
//!
//! # The Core Innovation
//!
//! Traditional type systems ask: "Is A equal to B?"
//! Sounio asks: "How far is A from B in semantic space?"
//!
//! With 15M ontological types, rigid equality is useless — `ChEBI.Aspirin`
//! would never equal `DRUGBANK.DB00945` even though they denote the same
//! molecule. But if we compute their semantic distance as ~0.02, we can
//! treat them as compatible.

use std::sync::Arc;

use std::hash::{Hash, Hasher};

use crate::ontology::distance::cache::TieredDistanceCache;
use crate::ontology::distance::{SemanticDistance, SemanticDistanceIndex};
use crate::ontology::loader::IRI;

/// Type compatibility result
#[derive(Debug, Clone)]
pub enum Compatibility {
    /// Types are identical
    Identical,
    /// Types are compatible (within threshold)
    Compatible {
        /// The computed semantic distance
        distance: SemanticDistance,
        /// How compatibility was determined
        method: CompatibilityMethod,
    },
    /// Types are incompatible (beyond threshold)
    Incompatible {
        /// The computed semantic distance
        distance: SemanticDistance,
        /// Why incompatible
        reason: IncompatibilityReason,
    },
    /// Compatibility cannot be determined
    Unknown {
        /// Why unknown
        reason: String,
    },
}

impl Compatibility {
    /// Check if types are compatible
    pub fn is_compatible(&self) -> bool {
        matches!(
            self,
            Compatibility::Identical | Compatibility::Compatible { .. }
        )
    }

    /// Get distance if available
    pub fn distance(&self) -> Option<&SemanticDistance> {
        match self {
            Compatibility::Compatible { distance, .. } => Some(distance),
            Compatibility::Incompatible { distance, .. } => Some(distance),
            _ => None,
        }
    }

    /// Get conceptual distance value
    pub fn conceptual_distance(&self) -> Option<f64> {
        self.distance().map(|d| d.conceptual)
    }
}

/// How compatibility was determined
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityMethod {
    /// Exact type match
    Exact,
    /// Subsumption (is-a relationship)
    Subsumption,
    /// SSSOM mapping
    Mapping,
    /// Path-based distance
    PathDistance,
    /// Embedding similarity
    EmbeddingSimilarity,
    /// Hybrid (combination of methods)
    Hybrid,
}

/// Reason for type incompatibility
#[derive(Debug, Clone)]
pub enum IncompatibilityReason {
    /// Distance exceeds threshold
    DistanceExceedsThreshold { distance: f64, threshold: f64 },
    /// Cross-ontology without mapping
    CrossOntologyNoMapping {
        source_ontology: String,
        target_ontology: String,
    },
    /// Types are unrelated
    Unrelated,
    /// Structural mismatch
    StructuralMismatch(String),
}

impl std::fmt::Display for IncompatibilityReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncompatibilityReason::DistanceExceedsThreshold {
                distance,
                threshold,
            } => {
                write!(
                    f,
                    "semantic distance {:.3} exceeds threshold {:.3}",
                    distance, threshold
                )
            }
            IncompatibilityReason::CrossOntologyNoMapping {
                source_ontology,
                target_ontology,
            } => {
                write!(
                    f,
                    "no mapping between {} and {}",
                    source_ontology, target_ontology
                )
            }
            IncompatibilityReason::Unrelated => {
                write!(f, "types are unrelated")
            }
            IncompatibilityReason::StructuralMismatch(msg) => {
                write!(f, "structural mismatch: {}", msg)
            }
        }
    }
}

/// Compatibility context with configurable thresholds
#[derive(Debug, Clone)]
pub struct CompatibilityContext {
    /// Default threshold for general compatibility
    pub default_threshold: f64,
    /// Strict threshold for safety-critical contexts
    pub strict_threshold: f64,
    /// Loose threshold for coercion contexts
    pub loose_threshold: f64,
    /// Whether to allow cross-ontology compatibility
    pub allow_cross_ontology: bool,
    /// Whether to use embeddings for similarity
    pub use_embeddings: bool,
}

impl Default for CompatibilityContext {
    fn default() -> Self {
        Self {
            default_threshold: 0.15,
            strict_threshold: 0.05,
            loose_threshold: 0.25,
            allow_cross_ontology: true,
            use_embeddings: true,
        }
    }
}

impl CompatibilityContext {
    /// Strict context for medical/safety-critical code
    pub fn strict() -> Self {
        Self {
            default_threshold: 0.05,
            strict_threshold: 0.02,
            loose_threshold: 0.10,
            allow_cross_ontology: false,
            use_embeddings: true,
        }
    }

    /// Permissive context for exploratory code
    pub fn permissive() -> Self {
        Self {
            default_threshold: 0.25,
            strict_threshold: 0.10,
            loose_threshold: 0.40,
            allow_cross_ontology: true,
            use_embeddings: true,
        }
    }

    /// Get threshold for annotation
    pub fn threshold_for(&self, annotation: &CompatibilityAnnotation) -> f64 {
        match annotation {
            CompatibilityAnnotation::Exact => 0.0,
            CompatibilityAnnotation::Strict => self.strict_threshold,
            CompatibilityAnnotation::Default => self.default_threshold,
            CompatibilityAnnotation::Loose => self.loose_threshold,
            CompatibilityAnnotation::Custom(t) => *t,
        }
    }
}

/// Compatibility annotations for fine-grained control
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompatibilityAnnotation {
    /// Exact match required (distance = 0)
    Exact,
    /// Strict threshold
    Strict,
    /// Default threshold
    Default,
    /// Loose threshold
    Loose,
    /// Custom threshold
    Custom(f64),
}

impl CompatibilityAnnotation {
    /// Parse from string (for attribute parsing)
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "exact" => Some(Self::Exact),
            "strict" => Some(Self::Strict),
            "default" => Some(Self::Default),
            "loose" => Some(Self::Loose),
            _ => s.parse::<f64>().ok().map(Self::Custom),
        }
    }
}

impl std::fmt::Display for CompatibilityAnnotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompatibilityAnnotation::Exact => write!(f, "exact"),
            CompatibilityAnnotation::Strict => write!(f, "strict"),
            CompatibilityAnnotation::Default => write!(f, "default"),
            CompatibilityAnnotation::Loose => write!(f, "loose"),
            CompatibilityAnnotation::Custom(t) => write!(f, "{:.3}", t),
        }
    }
}

/// Type compatibility checker
pub struct CompatibilityChecker {
    /// Semantic distance index
    distance_index: Arc<SemanticDistanceIndex>,
    /// Distance cache
    cache: TieredDistanceCache,
    /// Context
    context: CompatibilityContext,
}

impl CompatibilityChecker {
    /// Create a new compatibility checker
    pub fn new(distance_index: Arc<SemanticDistanceIndex>, context: CompatibilityContext) -> Self {
        Self {
            distance_index,
            cache: TieredDistanceCache::new(10_000, 100_000),
            context,
        }
    }

    /// Create with default context
    pub fn with_defaults(distance_index: Arc<SemanticDistanceIndex>) -> Self {
        Self::new(distance_index, CompatibilityContext::default())
    }

    /// Check if source type is compatible with target type
    pub fn check(&self, source: &IRI, target: &IRI) -> Compatibility {
        self.check_with_threshold(source, target, self.context.default_threshold)
    }

    /// Check with explicit threshold
    pub fn check_with_threshold(
        &self,
        source: &IRI,
        target: &IRI,
        threshold: f64,
    ) -> Compatibility {
        // Identical types
        if source == target {
            return Compatibility::Identical;
        }

        // Check cross-ontology restriction
        let source_ont = source.ontology();
        let target_ont = target.ontology();
        let is_cross_ontology = source_ont != target_ont;

        if is_cross_ontology && !self.context.allow_cross_ontology {
            return Compatibility::Incompatible {
                distance: SemanticDistance::MAX,
                reason: IncompatibilityReason::CrossOntologyNoMapping {
                    source_ontology: source_ont.to_string(),
                    target_ontology: target_ont.to_string(),
                },
            };
        }

        // Compute distance (with caching)
        let distance = self.compute_distance_cached(source, target);

        // Determine compatibility method
        let method = self.determine_method(source, target, &distance);

        // Check against threshold
        if distance.conceptual <= threshold {
            Compatibility::Compatible { distance, method }
        } else {
            Compatibility::Incompatible {
                distance,
                reason: IncompatibilityReason::DistanceExceedsThreshold {
                    distance: distance.conceptual,
                    threshold,
                },
            }
        }
    }

    /// Check with annotation
    pub fn check_with_annotation(
        &self,
        source: &IRI,
        target: &IRI,
        annotation: CompatibilityAnnotation,
    ) -> Compatibility {
        let threshold = self.context.threshold_for(&annotation);
        self.check_with_threshold(source, target, threshold)
    }

    /// Compute distance with caching
    fn compute_distance_cached(&self, source: &IRI, target: &IRI) -> SemanticDistance {
        // Use hash of IRIs for cache key
        let source_hash = Self::hash_iri(source);
        let target_hash = Self::hash_iri(target);

        let cached = self.cache.get(source_hash, target_hash);
        if let Some(dist) = cached {
            return SemanticDistance::new(dist as f64);
        }

        let distance = self.distance_index.distance(source, target);
        self.cache
            .insert(source_hash, target_hash, distance.conceptual as f32);
        distance
    }

    /// Hash an IRI to a u32 for cache key
    fn hash_iri(iri: &IRI) -> u32 {
        let mut hasher = rustc_hash::FxHasher::default();
        iri.as_str().hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Determine how compatibility was established
    fn determine_method(
        &self,
        source: &IRI,
        target: &IRI,
        distance: &SemanticDistance,
    ) -> CompatibilityMethod {
        if source == target {
            return CompatibilityMethod::Exact;
        }

        // Check subsumption
        if self.distance_index.is_subtype(source, target) {
            return CompatibilityMethod::Subsumption;
        }

        // Check SSSOM mapping
        if self
            .distance_index
            .find_sssom_mapping(source, target)
            .is_some()
        {
            return CompatibilityMethod::Mapping;
        }

        // Check if embeddings were used
        if self.distance_index.has_embeddings() && source.ontology() != target.ontology() {
            return CompatibilityMethod::EmbeddingSimilarity;
        }

        if distance.conceptual < 0.5 {
            CompatibilityMethod::PathDistance
        } else {
            CompatibilityMethod::Hybrid
        }
    }

    /// Find compatible types from a set
    pub fn find_compatible<'a>(
        &self,
        source: &IRI,
        candidates: impl Iterator<Item = &'a IRI>,
        threshold: f64,
    ) -> Vec<(&'a IRI, SemanticDistance)> {
        candidates
            .filter_map(
                |candidate| match self.check_with_threshold(source, candidate, threshold) {
                    Compatibility::Identical => Some((candidate, SemanticDistance::ZERO)),
                    Compatibility::Compatible { distance, .. } => Some((candidate, distance)),
                    _ => None,
                },
            )
            .collect()
    }

    /// Find the closest compatible type
    pub fn find_closest<'a>(
        &self,
        source: &IRI,
        candidates: impl Iterator<Item = &'a IRI>,
    ) -> Option<(&'a IRI, SemanticDistance)> {
        candidates
            .map(|candidate| {
                let distance = self.compute_distance_cached(source, candidate);
                (candidate, distance)
            })
            .min_by(|a, b| {
                a.1.conceptual
                    .partial_cmp(&b.1.conceptual)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> crate::ontology::distance::cache::TieredCacheStats {
        self.cache.tiered_stats()
    }

    /// Clear the distance cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get the context
    pub fn context(&self) -> &CompatibilityContext {
        &self.context
    }
}

/// Error for incompatible types
#[derive(Debug, Clone)]
pub struct IncompatibleTypesError {
    /// Source type IRI
    pub source: IRI,
    /// Target type IRI
    pub target: IRI,
    /// Computed distance
    pub distance: SemanticDistance,
    /// Threshold that was exceeded
    pub threshold: f64,
    /// Suggested alternative (if any)
    pub suggestion: Option<IRI>,
    /// Detailed reason
    pub reason: IncompatibilityReason,
}

impl std::fmt::Display for IncompatibleTypesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "type `{}` is not compatible with `{}` (distance {:.3} > threshold {:.3})",
            self.source, self.target, self.distance.conceptual, self.threshold
        )?;

        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  suggestion: use `{}` instead", suggestion)?;
        }

        Ok(())
    }
}

impl std::error::Error for IncompatibleTypesError {}

/// Builder for compatibility checking with fluent API
pub struct CompatibilityCheck<'a> {
    checker: &'a CompatibilityChecker,
    source: &'a IRI,
    target: &'a IRI,
    annotation: CompatibilityAnnotation,
}

impl<'a> CompatibilityCheck<'a> {
    /// Create a new compatibility check
    pub fn new(checker: &'a CompatibilityChecker, source: &'a IRI, target: &'a IRI) -> Self {
        Self {
            checker,
            source,
            target,
            annotation: CompatibilityAnnotation::Default,
        }
    }

    /// Set exact matching
    pub fn exact(mut self) -> Self {
        self.annotation = CompatibilityAnnotation::Exact;
        self
    }

    /// Set strict threshold
    pub fn strict(mut self) -> Self {
        self.annotation = CompatibilityAnnotation::Strict;
        self
    }

    /// Set loose threshold
    pub fn loose(mut self) -> Self {
        self.annotation = CompatibilityAnnotation::Loose;
        self
    }

    /// Set custom threshold
    pub fn threshold(mut self, t: f64) -> Self {
        self.annotation = CompatibilityAnnotation::Custom(t);
        self
    }

    /// Execute the check
    pub fn check(self) -> Compatibility {
        self.checker
            .check_with_annotation(self.source, self.target, self.annotation)
    }

    /// Check and return Result
    pub fn require(self) -> Result<SemanticDistance, IncompatibleTypesError> {
        // Capture fields before consuming self
        let source = self.source.clone();
        let target = self.target.clone();
        let annotation = self.annotation;
        let threshold = self.checker.context.threshold_for(&annotation);

        match self.check() {
            Compatibility::Identical => Ok(SemanticDistance::ZERO),
            Compatibility::Compatible { distance, .. } => Ok(distance),
            Compatibility::Incompatible { distance, reason } => Err(IncompatibleTypesError {
                source,
                target,
                distance,
                threshold,
                suggestion: None,
                reason,
            }),
            Compatibility::Unknown { reason } => Err(IncompatibleTypesError {
                source,
                target,
                distance: SemanticDistance::MAX,
                threshold,
                suggestion: None,
                reason: IncompatibilityReason::StructuralMismatch(reason),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compatibility_annotation_parse() {
        assert_eq!(
            CompatibilityAnnotation::parse("exact"),
            Some(CompatibilityAnnotation::Exact)
        );
        assert_eq!(
            CompatibilityAnnotation::parse("strict"),
            Some(CompatibilityAnnotation::Strict)
        );
        assert_eq!(
            CompatibilityAnnotation::parse("LOOSE"),
            Some(CompatibilityAnnotation::Loose)
        );
        assert_eq!(
            CompatibilityAnnotation::parse("0.15"),
            Some(CompatibilityAnnotation::Custom(0.15))
        );
        assert_eq!(CompatibilityAnnotation::parse("invalid"), None);
    }

    #[test]
    fn test_context_thresholds() {
        let ctx = CompatibilityContext::default();

        assert_eq!(ctx.threshold_for(&CompatibilityAnnotation::Exact), 0.0);
        assert_eq!(
            ctx.threshold_for(&CompatibilityAnnotation::Strict),
            ctx.strict_threshold
        );
        assert_eq!(
            ctx.threshold_for(&CompatibilityAnnotation::Default),
            ctx.default_threshold
        );
        assert_eq!(
            ctx.threshold_for(&CompatibilityAnnotation::Custom(0.42)),
            0.42
        );
    }

    #[test]
    fn test_strict_context() {
        let ctx = CompatibilityContext::strict();

        assert!(ctx.default_threshold < 0.10);
        assert!(!ctx.allow_cross_ontology);
    }

    #[test]
    fn test_permissive_context() {
        let ctx = CompatibilityContext::permissive();

        assert!(ctx.default_threshold > 0.20);
        assert!(ctx.allow_cross_ontology);
    }

    #[test]
    fn test_compatibility_is_compatible() {
        let identical = Compatibility::Identical;
        assert!(identical.is_compatible());

        let compatible = Compatibility::Compatible {
            distance: SemanticDistance::new(0.1),
            method: CompatibilityMethod::PathDistance,
        };
        assert!(compatible.is_compatible());

        let incompatible = Compatibility::Incompatible {
            distance: SemanticDistance::new(0.5),
            reason: IncompatibilityReason::Unrelated,
        };
        assert!(!incompatible.is_compatible());
    }

    #[test]
    fn test_annotation_display() {
        assert_eq!(format!("{}", CompatibilityAnnotation::Exact), "exact");
        assert_eq!(format!("{}", CompatibilityAnnotation::Strict), "strict");
        assert_eq!(
            format!("{}", CompatibilityAnnotation::Custom(0.123)),
            "0.123"
        );
    }
}
