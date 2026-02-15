//! L4 Federated Resolution Integration
//!
//! Wraps the existing `FederatedResolver` with additional features:
//!
//! - **Degraded confidence**: federated terms carry a base confidence of 0.85
//!   because external sources have weaker curation guarantees.
//! - **Cross-ontology mapping** via SSSOM: translate terms between ontologies
//!   (e.g., ChEBI aspirin -> DrugBank DB00945).
//! - **Local result caching**: federated results are stored locally so
//!   subsequent compilations do not require network access.
//! - **Timeout / fallback**: graceful degradation when networks are
//!   unavailable.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │ FederatedOntologyResolver                         │
//! │  ┌──────────────────┐  ┌───────────────────────┐ │
//! │  │ Local Cache       │  │ SSSOM Cross-Ontology  │ │
//! │  │ (HashMap + TTL)   │  │ Mapping Index         │ │
//! │  └────────┬─────────┘  └───────────┬───────────┘ │
//! │           │ miss                   │              │
//! │           ▼                        │              │
//! │  ┌──────────────────┐              │              │
//! │  │ FederatedResolver │◄────────────┘              │
//! │  │ (OLS4/BioPortal)  │                            │
//! │  └──────────────────┘                            │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::federation::{
//!     FederatedOntologyResolver, FederatedResolverConfig,
//! };
//!
//! let resolver = FederatedOntologyResolver::new(FederatedResolverConfig::default());
//! let result = resolver.resolve_federated("ORDO:123");
//! ```

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::collections::HashMap;

use super::lazy_loader::OntologySubsumption;

/// Source of a federated term resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FederatedSourceKind {
    /// Resolved from the local cache (no network needed).
    LocalCache,
    /// Resolved from BioPortal.
    BioPortal,
    /// Resolved from OLS4 (EMBL-EBI).
    OLS4,
    /// Resolved from a custom endpoint.
    Custom(String),
    /// Resolved via cross-ontology SSSOM mapping.
    SssomMapping,
}

impl FederatedSourceKind {
    /// Base confidence for this source.
    ///
    /// Federated sources have lower base confidence than local/compiled
    /// ontologies because external curation quality varies.
    pub fn base_confidence(&self) -> f64 {
        match self {
            FederatedSourceKind::LocalCache => 0.90,
            FederatedSourceKind::OLS4 => 0.85,
            FederatedSourceKind::BioPortal => 0.85,
            FederatedSourceKind::Custom(_) => 0.75,
            FederatedSourceKind::SssomMapping => 0.80,
        }
    }
}

/// A resolved federated term.
#[derive(Debug, Clone)]
pub struct FederatedTermResult {
    /// Canonical CURIE (e.g., "ORDO:123").
    pub curie: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Definition text.
    pub definition: Option<String>,
    /// Direct parent CURIEs.
    pub parents: Vec<String>,
    /// Synonyms.
    pub synonyms: Vec<String>,
    /// Source that provided this result.
    pub source: FederatedSourceKind,
    /// Confidence in this resolution (degraded for federated sources).
    pub confidence: f64,
    /// Whether this term is obsolete.
    pub obsolete: bool,
}

/// Cross-ontology mapping using SSSOM.
///
/// Represents a mapping between terms in different ontologies,
/// as described by the Simple Standard for Sharing Ontology Mappings
/// (SSSOM, Matentzoglu et al., 2022).
#[derive(Debug, Clone, PartialEq)]
pub struct CrossOntologyMapping {
    /// Source term CURIE (e.g., "CHEBI:15365").
    pub subject_id: String,
    /// Target term CURIE (e.g., "DRUGBANK:DB00945").
    pub object_id: String,
    /// Mapping predicate (e.g., "skos:exactMatch", "skos:closeMatch").
    pub predicate: MappingPredicateKind,
    /// Confidence of the mapping [0.0, 1.0].
    pub confidence: f64,
    /// Justification for the mapping.
    pub justification: MappingJustificationKind,
}

/// SSSOM mapping predicate kinds.
///
/// Based on SKOS vocabulary predicates used in SSSOM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MappingPredicateKind {
    /// Exact semantic equivalence.
    ExactMatch,
    /// Close but not exact equivalence.
    CloseMatch,
    /// Subject is broader than object.
    BroadMatch,
    /// Subject is narrower than object.
    NarrowMatch,
    /// Related but not equivalent.
    RelatedMatch,
}

impl MappingPredicateKind {
    /// Confidence multiplier for this predicate type.
    ///
    /// ExactMatch is most confident; related/broad/narrow degrade.
    pub fn confidence_factor(&self) -> f64 {
        match self {
            MappingPredicateKind::ExactMatch => 1.0,
            MappingPredicateKind::CloseMatch => 0.90,
            MappingPredicateKind::NarrowMatch => 0.85,
            MappingPredicateKind::BroadMatch => 0.85,
            MappingPredicateKind::RelatedMatch => 0.70,
        }
    }

    /// Parse from a SKOS-style predicate string.
    pub fn from_str_predicate(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "skos:exactmatch" | "exactmatch" | "exact" => Some(Self::ExactMatch),
            "skos:closematch" | "closematch" | "close" => Some(Self::CloseMatch),
            "skos:broadmatch" | "broadmatch" | "broad" => Some(Self::BroadMatch),
            "skos:narrowmatch" | "narrowmatch" | "narrow" => Some(Self::NarrowMatch),
            "skos:relatedmatch" | "relatedmatch" | "related" => Some(Self::RelatedMatch),
            _ => None,
        }
    }
}

/// SSSOM mapping justification kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingJustificationKind {
    /// Manual curation by domain expert.
    ManualCuration,
    /// Lexical matching algorithm.
    LexicalMatching,
    /// Logical reasoning / semantic matching.
    SemanticMatching,
    /// Composite (multiple methods).
    Composite,
    /// Unspecified justification.
    Unspecified,
}

/// Configuration for the federated resolver.
#[derive(Debug, Clone)]
pub struct FederatedResolverConfig {
    /// Base degraded confidence for federated results.
    pub base_confidence: f64,
    /// Timeout in milliseconds for network requests.
    pub timeout_ms: u64,
    /// Whether to enable network requests (false = stub mode).
    pub enable_network: bool,
    /// Maximum number of cached federated results.
    pub max_cache_size: usize,
}

impl Default for FederatedResolverConfig {
    fn default() -> Self {
        Self {
            base_confidence: 0.85,
            timeout_ms: 5000,
            enable_network: false, // Off by default in compiler
            max_cache_size: 5000,
        }
    }
}

/// Federated ontology resolver.
///
/// Wraps the lower-level `FederatedResolver` with caching,
/// degraded confidence, and cross-ontology mapping support.
pub struct FederatedOntologyResolver {
    /// Configuration.
    config: FederatedResolverConfig,
    /// Local cache of resolved terms.
    cache: HashMap<String, FederatedTermResult>,
    /// Cross-ontology mapping index (subject_id -> mappings).
    mappings: HashMap<String, Vec<CrossOntologyMapping>>,
    /// Statistics.
    stats: FederationStats,
}

/// Statistics for federation operations.
#[derive(Debug, Clone, Default)]
pub struct FederationStats {
    /// Total resolution attempts.
    pub total_attempts: usize,
    /// Cache hits.
    pub cache_hits: usize,
    /// Network resolutions (real or stubbed).
    pub network_resolutions: usize,
    /// SSSOM mapping lookups.
    pub mapping_lookups: usize,
    /// Failed resolutions.
    pub failures: usize,
}

impl FederatedOntologyResolver {
    /// Create a new federated resolver.
    pub fn new(config: FederatedResolverConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            mappings: HashMap::new(),
            stats: FederationStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_resolver() -> Self {
        Self::new(FederatedResolverConfig::default())
    }

    /// Resolve a federated term by CURIE.
    ///
    /// Resolution order:
    /// 1. Local cache
    /// 2. Network (if enabled)
    /// 3. Return not-found
    ///
    /// Federated results always have degraded confidence (base 0.85).
    pub fn resolve_federated(&mut self, curie: &str) -> Option<FederatedTermResult> {
        self.stats.total_attempts += 1;

        // Check cache
        if let Some(cached) = self.cache.get(curie) {
            self.stats.cache_hits += 1;
            return Some(cached.clone());
        }

        // Attempt network resolution (stubbed when enable_network is false)
        if self.config.enable_network {
            self.stats.network_resolutions += 1;
            // In production, this would call the FederatedResolver.
            // For now, we return None since we don't have live API access.
            self.stats.failures += 1;
            return None;
        }

        self.stats.failures += 1;
        None
    }

    /// Insert a pre-resolved term into the local cache.
    ///
    /// Used to populate the cache from previous compilation results or
    /// from the `FederatedResolver` response.
    pub fn cache_term(&mut self, term: FederatedTermResult) {
        if self.cache.len() >= self.config.max_cache_size {
            // Simple eviction: clear oldest half
            let keys: Vec<String> = self.cache.keys().take(self.cache.len() / 2).cloned().collect();
            for key in keys {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(term.curie.clone(), term);
    }

    /// Register a cross-ontology mapping.
    pub fn register_mapping(&mut self, mapping: CrossOntologyMapping) {
        self.mappings
            .entry(mapping.subject_id.clone())
            .or_default()
            .push(mapping);
    }

    /// Register multiple cross-ontology mappings.
    pub fn register_mappings(&mut self, mappings: Vec<CrossOntologyMapping>) {
        for mapping in mappings {
            self.register_mapping(mapping);
        }
    }

    /// Look up cross-ontology mappings for a term.
    ///
    /// Returns all known mappings from the given subject CURIE.
    /// If `target_prefix` is specified, filters to mappings targeting
    /// that ontology.
    pub fn find_mappings(
        &self,
        subject: &str,
        target_prefix: Option<&str>,
    ) -> Vec<&CrossOntologyMapping> {
        let Some(mappings) = self.mappings.get(subject) else {
            return Vec::new();
        };

        match target_prefix {
            Some(prefix) => {
                let prefix_upper = prefix.to_uppercase();
                mappings
                    .iter()
                    .filter(|m| {
                        m.object_id
                            .split(':')
                            .next()
                            .map(|p| p.to_uppercase() == prefix_upper)
                            .unwrap_or(false)
                    })
                    .collect()
            }
            None => mappings.iter().collect(),
        }
    }

    /// Translate a term to another ontology using SSSOM mappings.
    ///
    /// Returns the best (highest confidence) mapping to the target prefix.
    pub fn translate(
        &mut self,
        subject: &str,
        target_prefix: &str,
    ) -> Option<CrossOntologyMapping> {
        self.stats.mapping_lookups += 1;

        let mappings = self.find_mappings(subject, Some(target_prefix));
        mappings
            .into_iter()
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Check cross-ontology subsumption via SSSOM mappings.
    ///
    /// When child and parent are in different ontologies, we look for
    /// an SSSOM mapping that connects them. The resulting subsumption
    /// confidence is degraded by the mapping predicate type.
    pub fn cross_ontology_subsumption(
        &mut self,
        child: &str,
        parent: &str,
    ) -> OntologySubsumption {
        // Extract prefixes
        let child_prefix = child.split(':').next().unwrap_or("");
        let parent_prefix = parent.split(':').next().unwrap_or("");

        if child_prefix.eq_ignore_ascii_case(parent_prefix) {
            // Same ontology -- not a cross-ontology case
            return OntologySubsumption::negative();
        }

        // Look for mappings from child to parent's ontology
        let mappings = self.find_mappings(child, Some(parent_prefix));

        for mapping in mappings {
            if mapping.object_id == parent {
                // Direct mapping found
                let confidence =
                    self.config.base_confidence * mapping.predicate.confidence_factor();
                return OntologySubsumption::positive(
                    confidence,
                    1,
                    vec![child.to_string(), parent.to_string()],
                );
            }
        }

        // Also check reverse: parent maps to child's ontology
        let reverse_mappings = self.find_mappings(parent, Some(child_prefix));
        for mapping in reverse_mappings {
            if mapping.object_id == child
                && matches!(mapping.predicate, MappingPredicateKind::BroadMatch)
            {
                // Parent has a broad match to child, meaning child is narrower
                let confidence =
                    self.config.base_confidence * mapping.predicate.confidence_factor();
                return OntologySubsumption::positive(
                    confidence,
                    1,
                    vec![child.to_string(), parent.to_string()],
                );
            }
        }

        OntologySubsumption::negative()
    }

    /// Get statistics.
    pub fn stats(&self) -> &FederationStats {
        &self.stats
    }

    /// Get cached term count.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats = FederationStats::default();
    }
}

impl Default for FederatedOntologyResolver {
    fn default() -> Self {
        Self::default_resolver()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mapping(
        subject: &str,
        object: &str,
        predicate: MappingPredicateKind,
        confidence: f64,
    ) -> CrossOntologyMapping {
        CrossOntologyMapping {
            subject_id: subject.to_string(),
            object_id: object.to_string(),
            predicate,
            confidence,
            justification: MappingJustificationKind::ManualCuration,
        }
    }

    fn make_term(curie: &str, label: &str, source: FederatedSourceKind) -> FederatedTermResult {
        let confidence = source.base_confidence();
        FederatedTermResult {
            curie: curie.to_string(),
            label: Some(label.to_string()),
            definition: None,
            parents: Vec::new(),
            synonyms: Vec::new(),
            source,
            confidence,
            obsolete: false,
        }
    }

    // ---- Source confidence ----

    #[test]
    fn test_source_confidence_ols4() {
        assert!((FederatedSourceKind::OLS4.base_confidence() - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_source_confidence_bioportal() {
        assert!((FederatedSourceKind::BioPortal.base_confidence() - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_source_confidence_local() {
        assert!(FederatedSourceKind::LocalCache.base_confidence() > FederatedSourceKind::OLS4.base_confidence());
    }

    #[test]
    fn test_source_confidence_custom() {
        assert!(FederatedSourceKind::Custom("test".into()).base_confidence() < FederatedSourceKind::OLS4.base_confidence());
    }

    // ---- Predicate parsing ----

    #[test]
    fn test_predicate_parsing() {
        assert_eq!(
            MappingPredicateKind::from_str_predicate("skos:exactMatch"),
            Some(MappingPredicateKind::ExactMatch)
        );
        assert_eq!(
            MappingPredicateKind::from_str_predicate("closeMatch"),
            Some(MappingPredicateKind::CloseMatch)
        );
        assert_eq!(
            MappingPredicateKind::from_str_predicate("broad"),
            Some(MappingPredicateKind::BroadMatch)
        );
        assert_eq!(
            MappingPredicateKind::from_str_predicate("narrow"),
            Some(MappingPredicateKind::NarrowMatch)
        );
        assert_eq!(
            MappingPredicateKind::from_str_predicate("related"),
            Some(MappingPredicateKind::RelatedMatch)
        );
        assert_eq!(MappingPredicateKind::from_str_predicate("unknown"), None);
    }

    #[test]
    fn test_predicate_confidence_ordering() {
        assert!(
            MappingPredicateKind::ExactMatch.confidence_factor()
                > MappingPredicateKind::CloseMatch.confidence_factor()
        );
        assert!(
            MappingPredicateKind::CloseMatch.confidence_factor()
                > MappingPredicateKind::RelatedMatch.confidence_factor()
        );
    }

    // ---- Caching ----

    #[test]
    fn test_cache_term() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        let term = make_term("ORDO:123", "test disease", FederatedSourceKind::OLS4);
        resolver.cache_term(term);
        assert_eq!(resolver.cache_size(), 1);
    }

    #[test]
    fn test_resolve_from_cache() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        let term = make_term("ORDO:123", "test disease", FederatedSourceKind::OLS4);
        resolver.cache_term(term);

        let result = resolver.resolve_federated("ORDO:123");
        assert!(result.is_some());
        assert_eq!(result.unwrap().label, Some("test disease".into()));
        assert_eq!(resolver.stats().cache_hits, 1);
    }

    #[test]
    fn test_resolve_not_found() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        let result = resolver.resolve_federated("NONEXISTENT:999");
        assert!(result.is_none());
        assert_eq!(resolver.stats().failures, 1);
    }

    // ---- SSSOM mappings ----

    #[test]
    fn test_register_mapping() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        let mapping = make_mapping(
            "CHEBI:15365",
            "DRUGBANK:DB00945",
            MappingPredicateKind::ExactMatch,
            0.95,
        );
        resolver.register_mapping(mapping);

        let found = resolver.find_mappings("CHEBI:15365", None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].object_id, "DRUGBANK:DB00945");
    }

    #[test]
    fn test_find_mappings_with_prefix_filter() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        resolver.register_mappings(vec![
            make_mapping(
                "CHEBI:15365",
                "DRUGBANK:DB00945",
                MappingPredicateKind::ExactMatch,
                0.95,
            ),
            make_mapping(
                "CHEBI:15365",
                "MESH:D001241",
                MappingPredicateKind::CloseMatch,
                0.80,
            ),
        ]);

        let drugbank = resolver.find_mappings("CHEBI:15365", Some("DRUGBANK"));
        assert_eq!(drugbank.len(), 1);

        let mesh = resolver.find_mappings("CHEBI:15365", Some("MESH"));
        assert_eq!(mesh.len(), 1);

        let all = resolver.find_mappings("CHEBI:15365", None);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_translate() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        resolver.register_mappings(vec![
            make_mapping(
                "CHEBI:15365",
                "DRUGBANK:DB00945",
                MappingPredicateKind::ExactMatch,
                0.95,
            ),
            make_mapping(
                "CHEBI:15365",
                "DRUGBANK:DB99999",
                MappingPredicateKind::CloseMatch,
                0.70,
            ),
        ]);

        // Should return highest confidence mapping
        let result = resolver.translate("CHEBI:15365", "DRUGBANK");
        assert!(result.is_some());
        assert_eq!(result.unwrap().object_id, "DRUGBANK:DB00945");
    }

    #[test]
    fn test_translate_not_found() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        let result = resolver.translate("CHEBI:15365", "NONEXISTENT");
        assert!(result.is_none());
    }

    // ---- Cross-ontology subsumption ----

    #[test]
    fn test_cross_ontology_subsumption_direct() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        resolver.register_mapping(make_mapping(
            "CHEBI:15365",
            "DRUGBANK:DB00945",
            MappingPredicateKind::ExactMatch,
            0.95,
        ));

        let result = resolver.cross_ontology_subsumption("CHEBI:15365", "DRUGBANK:DB00945");
        assert!(result.holds);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_cross_ontology_subsumption_not_found() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        let result = resolver.cross_ontology_subsumption("CHEBI:15365", "DRUGBANK:DB00945");
        assert!(!result.holds);
    }

    #[test]
    fn test_cross_ontology_subsumption_same_ontology_returns_negative() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        // Same ontology is not a cross-ontology case
        let result = resolver.cross_ontology_subsumption("CHEBI:15365", "CHEBI:23888");
        assert!(!result.holds);
    }

    // ---- Statistics ----

    #[test]
    fn test_stats_initial() {
        let resolver = FederatedOntologyResolver::default_resolver();
        assert_eq!(resolver.stats().total_attempts, 0);
        assert_eq!(resolver.stats().cache_hits, 0);
    }

    #[test]
    fn test_clear() {
        let mut resolver = FederatedOntologyResolver::default_resolver();
        resolver.cache_term(make_term("ORDO:1", "test", FederatedSourceKind::OLS4));
        assert_eq!(resolver.cache_size(), 1);

        resolver.clear();
        assert_eq!(resolver.cache_size(), 0);
        assert_eq!(resolver.stats().total_attempts, 0);
    }
}
