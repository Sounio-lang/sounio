//! Ontology Integration for Sounio
//!
//! This module implements the 4-layer ontology architecture that enables
//! 15 million ontology terms to serve as native types in Sounio.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │ L4: Federated (~15M terms)                                     │
//! │     BioPortal, OLS4 - Runtime resolution via HTTP              │
//! ├────────────────────────────────────────────────────────────────┤
//! │ L3: Domain (~500K terms)                                       │
//! │     ChEBI, GO, DOID via Semantic-SQL (lazy-loaded SQLite)      │
//! ├────────────────────────────────────────────────────────────────┤
//! │ L2: Foundation (~8K terms)                                     │
//! │     PATO, UO, IAO, Schema.org, FHIR - shipped with stdlib      │
//! ├────────────────────────────────────────────────────────────────┤
//! │ L1: Primitive (~850 terms)                                     │
//! │     BFO, RO, COB - compiled into the compiler                  │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Design Principles
//!
//! 1. **Performance**: Primitive ontologies are compiled in; domain ontologies
//!    use pre-built SQLite databases; federated queries are cached aggressively.
//!
//! 2. **Correctness**: Every ontology term has a canonical IRI. Subsumption
//!    reasoning respects the ontology hierarchy.
//!
//! 3. **Interoperability**: SSSOM mappings enable translation between ontologies.
//!    CWA (Closed World) for type checking, with graceful OWA (Open World) fallback.
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::{OntologyResolver, ResolvedTerm};
//!
//! let resolver = OntologyResolver::new()?;
//!
//! // Resolve a term
//! let aspirin = resolver.resolve("ChEBI:15365")?;
//!
//! // Check subsumption
//! let is_drug = resolver.is_subclass_of(&aspirin, "ChEBI:23888")?;
//!
//! // Find mapping to FHIR
//! let fhir_code = resolver.translate(&aspirin, "fhir")?;
//! ```

pub mod alignment;
pub mod build;
pub mod cache;
pub mod distance;
pub mod domain;
pub mod embedding;
pub mod fidelity;
pub mod foundation;
pub mod llm_gen;
pub mod loader;
pub mod memory;
pub mod native;
#[cfg(feature = "ontology")]
pub mod semantic_sql;
pub mod sssom;
pub mod version;

mod federated;
mod primitive;
mod resolver;

pub use cache::{CacheConfig, CacheStats, OntologyCache};
pub use distance::{
    DistanceConfig, PhysicalCost, SemanticDistance, SemanticDistanceIndex,
    information_content::{ICConfig, ICIndex, ICSimilarity},
    path::{HierarchyGraph, HierarchyStats, LCAResult, PathResult},
    sssom::{MappingPredicate, SSSOMIndex, SSSOMMapping, SSSOMParser, SSSOMSet},
};
pub use domain::{
    DomainOntologies, DomainStats, DomainTerm, OntologyMetadata as DomainOntologyMetadata,
};
pub use federated::{FederatedQuery, FederatedResolver, FederatedSource};
pub use fidelity::{
    AggregateFidelity, FidelityCaveat, FidelityConfig, FidelityIssue, FidelityResult,
    FidelityStats, ProvenanceAudit, ProvenanceIssue, SubsumptionFidelity, ViolationDetails,
    ViolationType, WorldFidelityChecker,
};
pub use foundation::{
    CurationStatus, FoundationOntologies, FoundationTerm, OntologySource, TermEntry, TermMapping,
    augmentation::EpistemicAugmenter, fhir::FHIROntology, iao::IAOOntology, pato::PATOOntology,
    schema_org::SchemaOrgOntology, uo::UOOntology,
};
pub use loader::{
    IRI, LoadedTerm, OntologyId, OntologyLoader, OntologyLoaderConfig,
    bioportal::BioPortalClient,
    obo_parser::{OboParseError, parse_obo_file},
};
pub use primitive::{
    BfoClass, CobClass, PRIMITIVE_BFO, PRIMITIVE_COB, PRIMITIVE_RO, PrimitiveStore, RoRelation,
};
pub use resolver::{
    OntologyMetadata, OntologyResolver, RelationInfo, ResolutionError, ResolvedTerm,
    SubsumptionResult, TermInfo,
};
#[cfg(feature = "ontology")]
pub use semantic_sql::{SemanticSqlStore, SqlOntology, SqlTerm};
pub use sssom::{
    MappingDirection, MappingJustification, SssomMapping, SssomMappingSet, load_sssom_mappings,
};

use crate::epistemic::{OntologyBinding, OntologyRef, TermId};

/// Result type for ontology operations
pub type OntologyResult<T> = Result<T, OntologyError>;

/// Errors that can occur during ontology operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum OntologyError {
    #[error("Term not found: {ontology}:{term}")]
    TermNotFound { ontology: String, term: String },

    #[error("Ontology not available: {0}")]
    OntologyNotAvailable(String),

    #[error("Invalid term format: {0}")]
    InvalidTermFormat(String),

    #[error("Resolution failed: {0}")]
    ResolutionFailed(String),

    #[error("Subsumption check failed: {0}")]
    SubsumptionFailed(String),

    #[error("Mapping not found from {from} to {to}")]
    MappingNotFound { from: String, to: String },

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("SSSOM parse error: {0}")]
    SssomParseError(String),
}

/// Layer in the ontology hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OntologyLayer {
    /// L1: Primitive (BFO, RO, COB) - compiled into compiler
    Primitive,
    /// L2: Foundation (PATO, UO, IAO, Schema.org, FHIR) - shipped with stdlib
    Foundation,
    /// L3: Domain (ChEBI, GO, DOID, etc.) - lazy loaded SQLite
    Domain,
    /// L4: Federated (BioPortal, OLS4) - runtime HTTP resolution
    Federated,
}

impl OntologyLayer {
    /// Get the priority (lower = higher priority)
    pub fn priority(&self) -> u8 {
        match self {
            OntologyLayer::Primitive => 0,
            OntologyLayer::Foundation => 1,
            OntologyLayer::Domain => 2,
            OntologyLayer::Federated => 3,
        }
    }

    /// Check if this layer requires network access
    pub fn requires_network(&self) -> bool {
        matches!(self, OntologyLayer::Federated)
    }

    /// Check if this layer supports offline mode
    pub fn supports_offline(&self) -> bool {
        !self.requires_network()
    }
}

impl From<&OntologyRef> for OntologyLayer {
    fn from(ontology_ref: &OntologyRef) -> Self {
        match ontology_ref {
            OntologyRef::Primitive(_) => OntologyLayer::Primitive,
            OntologyRef::Foundation(_) => OntologyLayer::Foundation,
            OntologyRef::Domain(_) => OntologyLayer::Domain,
            OntologyRef::Federated(_) => OntologyLayer::Federated,
        }
    }
}

/// A parsed ontology term reference
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsedTermRef {
    /// Ontology prefix (e.g., "ChEBI", "GO", "BFO")
    pub prefix: String,
    /// Local identifier (e.g., "15365", "0000001")
    pub local_id: String,
    /// Full CURIE (e.g., "ChEBI:15365")
    pub curie: String,
}

impl ParsedTermRef {
    /// Parse a CURIE-style term reference
    ///
    /// Accepts formats:
    /// - `PREFIX:ID` (e.g., "ChEBI:15365")
    /// - `PREFIX_ID` (e.g., "CHEBI_15365")
    /// - Full IRI (e.g., "http://purl.obolibrary.org/obo/CHEBI_15365")
    pub fn parse(input: &str) -> OntologyResult<Self> {
        // Try to extract from full IRI first (must come before CURIE check)
        if input.starts_with("http://") || input.starts_with("https://") {
            // OBO format: http://purl.obolibrary.org/obo/CHEBI_15365
            if input.contains("obolibrary.org/obo/")
                && let Some(term) = input.rsplit('/').next()
                && let Some((prefix, local_id)) = term.split_once('_')
            {
                return Ok(Self {
                    prefix: prefix.to_uppercase(),
                    local_id: local_id.to_string(),
                    curie: format!("{}:{}", prefix.to_uppercase(), local_id),
                });
            }
            return Err(OntologyError::InvalidTermFormat(format!(
                "Cannot parse IRI: {}",
                input
            )));
        }

        // Try CURIE format (PREFIX:ID)
        if let Some((prefix, local_id)) = input.split_once(':') {
            return Ok(Self {
                prefix: prefix.to_uppercase(),
                local_id: local_id.to_string(),
                curie: format!("{}:{}", prefix.to_uppercase(), local_id),
            });
        }

        // Try OBO-style format (PREFIX_ID)
        if let Some((prefix, local_id)) = input.split_once('_')
            && prefix.chars().all(|c| c.is_ascii_alphabetic())
        {
            return Ok(Self {
                prefix: prefix.to_uppercase(),
                local_id: local_id.to_string(),
                curie: format!("{}:{}", prefix.to_uppercase(), local_id),
            });
        }

        Err(OntologyError::InvalidTermFormat(format!(
            "Cannot parse term reference: {}",
            input
        )))
    }

    /// Convert to an OntologyBinding
    pub fn to_binding(&self) -> OntologyBinding {
        OntologyBinding {
            ontology: self.infer_ontology_ref(),
            term: TermId::new(&self.local_id),
            constraint: None,
        }
    }

    /// Infer the OntologyRef from the prefix
    fn infer_ontology_ref(&self) -> OntologyRef {
        use crate::epistemic::{
            DomainOntology, FederatedRef, FoundationOntology, PrimitiveOntology,
        };

        match self.prefix.as_str() {
            // L1 Primitive
            "BFO" => OntologyRef::Primitive(PrimitiveOntology::BFO),
            "RO" => OntologyRef::Primitive(PrimitiveOntology::RO),
            "COB" => OntologyRef::Primitive(PrimitiveOntology::COB),

            // L2 Foundation
            "PATO" => OntologyRef::Foundation(FoundationOntology::PATO),
            "UO" => OntologyRef::Foundation(FoundationOntology::UO),
            "IAO" => OntologyRef::Foundation(FoundationOntology::IAO),
            "SCHEMA" => OntologyRef::Foundation(FoundationOntology::SchemaOrg),
            "FHIR" => OntologyRef::Foundation(FoundationOntology::FHIR),

            // L3 Domain (common OBO ontologies)
            prefix @ ("CHEBI" | "GO" | "DOID" | "HP" | "MONDO" | "UBERON" | "CL" | "NCBITaxon"
            | "PR" | "SO" | "ENVO" | "OBI" | "OMIM" | "ORDO" | "NCIT") => {
                OntologyRef::Domain(DomainOntology {
                    id: prefix.to_lowercase(),
                    version: None,
                })
            }

            // L4 Federated (unknown prefix)
            other => OntologyRef::Federated(FederatedRef {
                acronym: other.to_string(),
                version: "latest".to_string(),
            }),
        }
    }
}

/// Statistics about ontology usage
#[derive(Debug, Clone, Default)]
pub struct OntologyStats {
    /// Number of L1 primitive terms resolved
    pub primitive_hits: usize,
    /// Number of L2 foundation terms resolved
    pub foundation_hits: usize,
    /// Number of L3 domain terms resolved (from SQLite)
    pub domain_hits: usize,
    /// Number of L4 federated terms resolved (from network)
    pub federated_hits: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of subsumption checks
    pub subsumption_checks: usize,
    /// Number of mapping lookups
    pub mapping_lookups: usize,
}

impl OntologyStats {
    /// Get total term resolutions
    pub fn total_resolutions(&self) -> usize {
        self.primitive_hits + self.foundation_hits + self.domain_hits + self.federated_hits
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

// ============================================================================
// General Ontology Trait
// ============================================================================

/// A concept from an ontology.
///
/// This is a simplified representation used for querying and analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OntologyConcept {
    /// The CURIE identifier (e.g., "CHEBI:15365")
    pub curie: String,
    /// Human-readable label
    pub label: String,
}

impl OntologyConcept {
    /// Create a new concept.
    pub fn new(curie: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            curie: curie.into(),
            label: label.into(),
        }
    }

    /// Get the CURIE.
    pub fn curie(&self) -> &str {
        &self.curie
    }

    /// Get the label.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Trait for ontology access.
///
/// This provides a unified interface for querying ontology data across
/// all layers (primitive, foundation, domain, federated). Implementations
/// can be backed by in-memory stores, SQLite databases, or network services.
///
/// # Example
///
/// ```rust,ignore
/// use sounio::ontology::{OntologyAccess, OntologyConcept};
///
/// fn find_related_concepts<O: OntologyAccess>(ont: &O, term: &str) {
///     let ancestors = ont.ancestors(term);
///     let descendants = ont.descendants(term);
///
///     for anc in &ancestors {
///         if ont.is_subclass(term, anc) {
///             println!("{} is a subclass of {}", term, anc);
///         }
///     }
/// }
/// ```
pub trait OntologyAccess {
    /// Search for concepts matching a query string.
    ///
    /// Returns up to `limit` concepts whose labels or CURIEs match the query.
    fn search(&self, query: &str, limit: usize) -> Vec<OntologyConcept>;

    /// Get all ancestors of a concept.
    ///
    /// Returns the CURIEs of all superclasses in the ontology hierarchy.
    fn ancestors(&self, curie: &str) -> Vec<String>;

    /// Get all descendants of a concept.
    ///
    /// Returns the CURIEs of all subclasses in the ontology hierarchy.
    fn descendants(&self, curie: &str) -> Vec<String>;

    /// Check if `child` is a subclass of `parent`.
    ///
    /// This performs a transitive subclass check through the ontology hierarchy.
    fn is_subclass(&self, child: &str, parent: &str) -> bool;

    /// Get a concept by its CURIE.
    ///
    /// Returns `None` if the concept is not found.
    fn get(&self, curie: &str) -> Option<OntologyConcept> {
        self.search(curie, 1).into_iter().find(|c| c.curie == curie)
    }

    /// Get the distance between two concepts in the hierarchy.
    ///
    /// Returns `None` if no path exists between them.
    fn distance(&self, from: &str, to: &str) -> Option<usize> {
        // Check direct relationship
        if from == to {
            return Some(0);
        }

        // Check if `to` is an ancestor of `from`
        let ancestors = self.ancestors(from);
        for (i, anc) in ancestors.iter().enumerate() {
            if anc == to {
                return Some(i + 1);
            }
        }

        // Check if `to` is a descendant of `from`
        let descendants = self.descendants(from);
        for (i, desc) in descendants.iter().enumerate() {
            if desc == to {
                return Some(i + 1);
            }
        }

        None
    }
}

/// Adapter to convert NativeOntology to OntologyAccess trait.
pub struct NativeOntologyAdapter {
    ontology: native::NativeOntology,
}

impl NativeOntologyAdapter {
    /// Create a new adapter wrapping a NativeOntology.
    pub fn new(ontology: native::NativeOntology) -> Self {
        Self { ontology }
    }

    /// Get a reference to the underlying NativeOntology.
    pub fn inner(&self) -> &native::NativeOntology {
        &self.ontology
    }
}

impl OntologyAccess for NativeOntologyAdapter {
    fn search(&self, query: &str, limit: usize) -> Vec<OntologyConcept> {
        self.ontology
            .search(query, limit)
            .into_iter()
            .map(|(curie, label)| OntologyConcept::new(curie, label))
            .collect()
    }

    fn ancestors(&self, curie: &str) -> Vec<String> {
        self.ontology
            .get_ancestors(curie)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn descendants(&self, _curie: &str) -> Vec<String> {
        // NativeOntology doesn't have descendants method - return empty
        Vec::new()
    }

    fn is_subclass(&self, child: &str, parent: &str) -> bool {
        self.ontology.is_subclass(child, parent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_curie() {
        let term = ParsedTermRef::parse("ChEBI:15365").unwrap();
        assert_eq!(term.prefix, "CHEBI");
        assert_eq!(term.local_id, "15365");
        assert_eq!(term.curie, "CHEBI:15365");
    }

    #[test]
    fn test_parse_obo_style() {
        let term = ParsedTermRef::parse("CHEBI_15365").unwrap();
        assert_eq!(term.prefix, "CHEBI");
        assert_eq!(term.local_id, "15365");
    }

    #[test]
    fn test_parse_iri() {
        let term = ParsedTermRef::parse("http://purl.obolibrary.org/obo/CHEBI_15365").unwrap();
        assert_eq!(term.prefix, "CHEBI");
        assert_eq!(term.local_id, "15365");
    }

    #[test]
    fn test_ontology_layer_priority() {
        assert!(OntologyLayer::Primitive.priority() < OntologyLayer::Foundation.priority());
        assert!(OntologyLayer::Foundation.priority() < OntologyLayer::Domain.priority());
        assert!(OntologyLayer::Domain.priority() < OntologyLayer::Federated.priority());
    }

    #[test]
    fn test_infer_ontology_ref() {
        let bfo = ParsedTermRef::parse("BFO:0000001").unwrap();
        assert!(matches!(
            bfo.infer_ontology_ref(),
            OntologyRef::Primitive(_)
        ));

        let chebi = ParsedTermRef::parse("ChEBI:15365").unwrap();
        assert!(matches!(chebi.infer_ontology_ref(), OntologyRef::Domain(_)));

        let unknown = ParsedTermRef::parse("UNKNOWN:123").unwrap();
        assert!(matches!(
            unknown.infer_ontology_ref(),
            OntologyRef::Federated(_)
        ));
    }
}
