//! Ontology Resolver - Unified Interface for All Layers
//!
//! This module provides the main interface for resolving ontology terms
//! across all four layers of the Sounio ontology architecture.
//!
//! # Resolution Strategy
//!
//! Terms are resolved in priority order:
//! 1. L1 Primitive (BFO, RO, COB) - instant, compiled-in
//! 2. L2 Foundation (PATO, UO, IAO, Schema.org, FHIR) - fast, file-based
//! 3. L3 Domain (ChEBI, GO, DOID, etc.) - medium, SQLite
//! 4. L4 Federated (BioPortal, OLS4) - slow, network
//!
//! # Caching
//!
//! All resolved terms are cached using a tiered LRU cache:
//! - Hot: Most recently used terms
//! - Warm: Frequently used terms
//! - Cold: Less frequently used terms
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::{OntologyResolver, ResolverConfig};
//!
//! let config = ResolverConfig::default();
//! let resolver = OntologyResolver::new(config)?;
//!
//! // Resolve a term
//! let term = resolver.resolve("CHEBI:15365")?;
//!
//! // Check subsumption
//! let is_drug = resolver.is_subclass_of("CHEBI:15365", "CHEBI:23888")?;
//!
//! // Translate between ontologies
//! let fhir = resolver.translate("CHEBI:15365", "fhir")?;
//! ```

use std::path::PathBuf;

use super::cache::{CacheConfig, CachedTermData, OntologyCache, SubsumptionCache};
use super::primitive::{PRIMITIVE_BFO, PRIMITIVE_COB, PRIMITIVE_RO};
use super::sssom::SssomMappingSet;
use super::{OntologyError, OntologyLayer, OntologyResult, OntologyStats, ParsedTermRef};

/// Configuration for the ontology resolver
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// Cache configuration
    pub cache: CacheConfig,
    /// Path to data directory (for SQLite databases)
    pub data_dir: Option<PathBuf>,
    /// Whether to enable federated queries
    pub enable_federated: bool,
    /// Timeout for network requests (ms)
    pub network_timeout_ms: u64,
    /// Maximum number of federated retries
    pub max_retries: u32,
    /// Whether to use offline mode (no network)
    pub offline_mode: bool,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            cache: CacheConfig::default(),
            data_dir: None,
            enable_federated: true,
            network_timeout_ms: 5000,
            max_retries: 3,
            offline_mode: false,
        }
    }
}

impl ResolverConfig {
    /// Create config with data directory
    pub fn with_data_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.data_dir = Some(dir.into());
        self
    }

    /// Enable offline mode
    pub fn offline(mut self) -> Self {
        self.offline_mode = true;
        self.enable_federated = false;
        self
    }

    /// Disable federated queries
    pub fn no_federated(mut self) -> Self {
        self.enable_federated = false;
        self
    }
}

/// Resolved term information
#[derive(Debug, Clone)]
pub struct ResolvedTerm {
    /// Canonical CURIE (e.g., "CHEBI:15365")
    pub curie: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Definition
    pub definition: Option<String>,
    /// Direct superclass CURIEs
    pub superclasses: Vec<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Which layer this term came from
    pub layer: OntologyLayer,
    /// Full IRI
    pub iri: Option<String>,
}

impl ResolvedTerm {
    /// Get the display label (label or CURIE if no label)
    pub fn display(&self) -> &str {
        self.label.as_deref().unwrap_or(&self.curie)
    }

    /// Convert to cached term data
    pub fn to_cached_data(&self) -> CachedTermData {
        CachedTermData {
            curie: self.curie.clone(),
            label: self.label.clone(),
            definition: self.definition.clone(),
            superclasses: self.superclasses.clone(),
            subclasses: vec![],
            synonyms: self.synonyms.clone(),
            xrefs: vec![],
        }
    }
}

/// Information about a relation/property
#[derive(Debug, Clone)]
pub struct RelationInfo {
    /// Canonical CURIE
    pub curie: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Inverse relation if exists
    pub inverse: Option<String>,
    /// Domain (subject type)
    pub domain: Option<String>,
    /// Range (object type)
    pub range: Option<String>,
}

/// Metadata about a term
#[derive(Debug, Clone)]
pub struct TermInfo {
    /// Canonical CURIE
    pub curie: String,
    /// Label
    pub label: Option<String>,
    /// Is this term obsolete?
    pub obsolete: bool,
    /// Replaced by (if obsolete)
    pub replaced_by: Option<String>,
    /// Consider using (if obsolete)
    pub consider: Vec<String>,
    /// Creation date
    pub created: Option<String>,
    /// Creator
    pub creator: Option<String>,
}

/// Metadata about an ontology
#[derive(Debug, Clone)]
pub struct OntologyMetadata {
    /// Ontology ID/prefix
    pub id: String,
    /// Full name
    pub title: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Version
    pub version: Option<String>,
    /// License
    pub license: Option<String>,
    /// Number of classes
    pub class_count: Option<usize>,
    /// Number of properties
    pub property_count: Option<usize>,
    /// Home page URL
    pub homepage: Option<String>,
}

/// Result of subsumption check
#[derive(Debug, Clone)]
pub enum SubsumptionResult {
    /// Child is a subclass of parent
    IsSubclass,
    /// Child is not a subclass of parent
    NotSubclass,
    /// Cannot determine (e.g., different ontologies without mapping)
    Unknown,
    /// Terms are equivalent
    Equivalent,
}

/// Error during resolution
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResolutionError {
    #[error("Term not found: {0}")]
    NotFound(String),
    #[error("Ontology not available: {0}")]
    OntologyUnavailable(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Invalid term format: {0}")]
    InvalidFormat(String),
}

/// The main ontology resolver
pub struct OntologyResolver {
    /// Configuration
    config: ResolverConfig,
    /// Term cache
    cache: OntologyCache,
    /// Subsumption cache
    subsumption_cache: SubsumptionCache,
    /// SSSOM mappings (loaded lazily)
    mappings: Option<SssomMappingSet>,
    /// Statistics
    stats: OntologyStats,
}

impl OntologyResolver {
    /// Create a new resolver with the given configuration
    pub fn new(config: ResolverConfig) -> OntologyResult<Self> {
        let cache = OntologyCache::new(config.cache.clone());
        let subsumption_cache = SubsumptionCache::new(10000);

        Ok(Self {
            config,
            cache,
            subsumption_cache,
            mappings: None,
            stats: OntologyStats::default(),
        })
    }

    /// Create a resolver with default configuration
    pub fn default_resolver() -> OntologyResult<Self> {
        Self::new(ResolverConfig::default())
    }

    /// Resolve a term by its CURIE or IRI
    pub fn resolve(&mut self, id: &str) -> OntologyResult<ResolvedTerm> {
        // Parse the term reference
        let parsed = ParsedTermRef::parse(id)?;

        // Check cache first
        if let Some(cached) = self.cache.get(&parsed.curie) {
            return Ok(ResolvedTerm {
                curie: cached.data.curie.clone(),
                label: cached.data.label.clone(),
                definition: cached.data.definition.clone(),
                superclasses: cached.data.superclasses.clone(),
                synonyms: cached.data.synonyms.clone(),
                layer: cached.source_layer,
                iri: None,
            });
        }

        // Check if known missing
        if self.cache.is_known_missing(&parsed.curie) {
            return Err(OntologyError::TermNotFound {
                ontology: parsed.prefix.clone(),
                term: parsed.local_id.clone(),
            });
        }

        // Try each layer in order
        let result = self.resolve_from_layers(&parsed);

        match &result {
            Ok(term) => {
                // Cache the result
                self.cache
                    .insert(term.curie.clone(), term.to_cached_data(), term.layer);
            }
            Err(_) => {
                // Cache negative result
                self.cache.insert_negative(parsed.curie.clone());
            }
        }

        result
    }

    /// Try to resolve from each layer
    fn resolve_from_layers(&mut self, parsed: &ParsedTermRef) -> OntologyResult<ResolvedTerm> {
        // L1: Primitive
        if let Some(term) = self.resolve_primitive(parsed) {
            self.stats.primitive_hits += 1;
            return Ok(term);
        }

        // L2: Foundation
        if let Some(term) = self.resolve_foundation(parsed) {
            self.stats.foundation_hits += 1;
            return Ok(term);
        }

        // L3: Domain (SQLite)
        #[cfg(feature = "ontology")]
        if let Some(term) = self.resolve_domain(parsed)? {
            self.stats.domain_hits += 1;
            return Ok(term);
        }

        // L4: Federated (network)
        if self.config.enable_federated
            && !self.config.offline_mode
            && let Some(term) = self.resolve_federated(parsed)?
        {
            self.stats.federated_hits += 1;
            return Ok(term);
        }

        Err(OntologyError::TermNotFound {
            ontology: parsed.prefix.clone(),
            term: parsed.local_id.clone(),
        })
    }

    /// Resolve from L1 primitive ontologies
    fn resolve_primitive(&self, parsed: &ParsedTermRef) -> Option<ResolvedTerm> {
        match parsed.prefix.as_str() {
            "BFO" => {
                let term = PRIMITIVE_BFO.get_by_id(&parsed.curie)?;
                Some(ResolvedTerm {
                    curie: parsed.curie.clone(),
                    label: Some(term.label.to_string()),
                    definition: Some(term.definition.to_string()),
                    superclasses: PRIMITIVE_BFO
                        .direct_superclasses(term.variant)
                        .iter()
                        .map(|c| c.id().to_string())
                        .collect(),
                    synonyms: vec![],
                    layer: OntologyLayer::Primitive,
                    iri: Some(term.iri.to_string()),
                })
            }
            "RO" => {
                let term = PRIMITIVE_RO.get_by_id(&parsed.curie)?;
                Some(ResolvedTerm {
                    curie: parsed.curie.clone(),
                    label: Some(term.label.to_string()),
                    definition: Some(term.definition.to_string()),
                    superclasses: vec![],
                    synonyms: vec![],
                    layer: OntologyLayer::Primitive,
                    iri: Some(term.iri.to_string()),
                })
            }
            "COB" => {
                let term = PRIMITIVE_COB.get_by_id(&parsed.curie)?;
                Some(ResolvedTerm {
                    curie: parsed.curie.clone(),
                    label: Some(term.label.to_string()),
                    definition: Some(term.definition.to_string()),
                    superclasses: vec![],
                    synonyms: vec![],
                    layer: OntologyLayer::Primitive,
                    iri: Some(term.iri.to_string()),
                })
            }
            _ => None,
        }
    }

    /// Resolve from L2 foundation ontologies
    fn resolve_foundation(&self, parsed: &ParsedTermRef) -> Option<ResolvedTerm> {
        // Foundation ontologies would be loaded from embedded data or files
        // For now, return None as placeholder
        match parsed.prefix.as_str() {
            "PATO" | "UO" | "IAO" | "SCHEMA" | "FHIR" => {
                // TODO: Implement foundation resolution
                None
            }
            _ => None,
        }
    }

    /// Resolve from L3 domain ontologies (SQLite)
    #[cfg(feature = "ontology")]
    fn resolve_domain(&self, parsed: &ParsedTermRef) -> OntologyResult<Option<ResolvedTerm>> {
        // Would use SemanticSqlManager to look up from database
        // For now, return None as placeholder
        Ok(None)
    }

    /// Resolve from L4 federated sources (network)
    fn resolve_federated(&self, parsed: &ParsedTermRef) -> OntologyResult<Option<ResolvedTerm>> {
        // Would use FederatedResolver to query BioPortal/OLS4
        // For now, return None as placeholder
        Ok(None)
    }

    /// Check if child term is a subclass of parent term
    pub fn is_subclass_of(
        &mut self,
        child: &str,
        parent: &str,
    ) -> OntologyResult<SubsumptionResult> {
        self.stats.subsumption_checks += 1;

        // Check cache first
        if let Some(result) = self.subsumption_cache.get(child, parent) {
            return Ok(if result {
                SubsumptionResult::IsSubclass
            } else {
                SubsumptionResult::NotSubclass
            });
        }

        // Parse terms
        let child_parsed = ParsedTermRef::parse(child)?;
        let parent_parsed = ParsedTermRef::parse(parent)?;

        // Same term
        if child_parsed.curie == parent_parsed.curie {
            return Ok(SubsumptionResult::Equivalent);
        }

        // Same ontology - use native subsumption
        if child_parsed.prefix == parent_parsed.prefix {
            let result = self.check_native_subsumption(&child_parsed, &parent_parsed)?;
            self.subsumption_cache.insert(
                child,
                parent,
                matches!(result, SubsumptionResult::IsSubclass),
            );
            return Ok(result);
        }

        // Different ontologies - would need mapping
        Ok(SubsumptionResult::Unknown)
    }

    /// Check subsumption within the same ontology
    fn check_native_subsumption(
        &self,
        child: &ParsedTermRef,
        parent: &ParsedTermRef,
    ) -> OntologyResult<SubsumptionResult> {
        // L1: Primitive
        match child.prefix.as_str() {
            "BFO" => {
                // Parse local IDs to BfoClass
                let child_term = PRIMITIVE_BFO.get_by_id(&child.curie);
                let parent_term = PRIMITIVE_BFO.get_by_id(&parent.curie);

                if let (Some(c), Some(p)) = (child_term, parent_term)
                    && PRIMITIVE_BFO.is_subclass(c.variant, p.variant)
                {
                    return Ok(SubsumptionResult::IsSubclass);
                }
                return Ok(SubsumptionResult::NotSubclass);
            }
            "COB" => {
                let child_term = PRIMITIVE_COB.get_by_id(&child.curie);
                let parent_term = PRIMITIVE_COB.get_by_id(&parent.curie);

                if let (Some(c), Some(p)) = (child_term, parent_term)
                    && PRIMITIVE_COB.is_subclass(c.variant, p.variant)
                {
                    return Ok(SubsumptionResult::IsSubclass);
                }
                return Ok(SubsumptionResult::NotSubclass);
            }
            _ => {}
        }

        // L3: Domain (would use SQLite)
        #[cfg(feature = "ontology")]
        {
            // Would query semantic-sql database
        }

        Ok(SubsumptionResult::Unknown)
    }

    /// Translate a term to another ontology using SSSOM mappings
    pub fn translate(&mut self, from: &str, to_prefix: &str) -> OntologyResult<Option<String>> {
        self.stats.mapping_lookups += 1;

        let parsed = ParsedTermRef::parse(from)?;

        // Check mappings
        if let Some(mappings) = &self.mappings
            && let Some(best) = mappings.find_best_mapping(&parsed.curie, Some(to_prefix))
        {
            return Ok(Some(best.object_id.clone()));
        }

        Ok(None)
    }

    /// Load SSSOM mappings from a file
    pub fn load_mappings(&mut self, path: impl AsRef<std::path::Path>) -> OntologyResult<()> {
        let new_mappings = super::sssom::load_sssom_mappings(path)?;

        if let Some(existing) = &mut self.mappings {
            existing.merge(new_mappings);
        } else {
            self.mappings = Some(new_mappings);
        }

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> &OntologyStats {
        &self.stats
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &super::cache::CacheStats {
        self.cache.stats()
    }

    /// Clear all caches
    pub fn clear_caches(&mut self) {
        self.cache.clear();
        self.subsumption_cache.clear();
    }

    /// Check if a term exists
    pub fn exists(&mut self, id: &str) -> bool {
        self.resolve(id).is_ok()
    }

    /// Get all ancestors (transitive superclasses) of a term
    pub fn get_ancestors(&mut self, id: &str) -> OntologyResult<Vec<String>> {
        let term = self.resolve(id)?;

        // For primitives, compute transitive closure
        let parsed = ParsedTermRef::parse(id)?;
        if parsed.prefix.as_str() == "BFO"
            && let Some(t) = PRIMITIVE_BFO.get_by_id(&parsed.curie)
        {
            let mut ancestors = vec![];
            let mut stack = vec![t.variant];
            let mut visited = std::collections::HashSet::new();

            while let Some(current) = stack.pop() {
                if visited.insert(current) {
                    for parent in PRIMITIVE_BFO.direct_superclasses(current) {
                        ancestors.push(parent.id().to_string());
                        stack.push(parent);
                    }
                }
            }

            return Ok(ancestors);
        }

        // For domain ontologies, return direct superclasses
        Ok(term.superclasses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_bfo() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        let term = resolver.resolve("BFO:0000001");
        assert!(term.is_ok());
        let term = term.unwrap();
        assert_eq!(term.label, Some("entity".to_string()));
        assert_eq!(term.layer, OntologyLayer::Primitive);
    }

    #[test]
    fn test_subsumption_bfo() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        // Object is a subclass of Entity
        let result = resolver.is_subclass_of("BFO:0000027", "BFO:0000001");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), SubsumptionResult::IsSubclass));

        // Entity is not a subclass of Object
        let result = resolver.is_subclass_of("BFO:0000001", "BFO:0000027");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), SubsumptionResult::NotSubclass));
    }

    #[test]
    fn test_same_term_equivalent() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        let result = resolver.is_subclass_of("BFO:0000001", "BFO:0000001");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), SubsumptionResult::Equivalent));
    }

    #[test]
    fn test_term_not_found() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        let result = resolver.resolve("NONEXISTENT:12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_caching() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        // First resolve (cache miss)
        let _ = resolver.resolve("BFO:0000001");

        // Second resolve (cache hit)
        let _ = resolver.resolve("BFO:0000001");

        assert!(resolver.cache_stats().total_hits() > 0);
    }

    #[test]
    fn test_exists() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        assert!(resolver.exists("BFO:0000001"));
        assert!(!resolver.exists("NONEXISTENT:12345"));
    }
}
