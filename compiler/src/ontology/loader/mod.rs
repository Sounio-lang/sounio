//! Native Ontology Loading Infrastructure
//!
//! This module implements the first programming language where 15+ million
//! ontological terms are compiled into first-class types.
//!
//! # Architecture
//!
//! ```text
//! OWL/OBO Files → Parser → Type Graph → Type Cache → Compiler Integration
//!                              ↓
//!                    Semantic Distance Index
//! ```
//!
//! # The Paradigm
//!
//! Traditional: Ontology is DATA you query at runtime
//! Sounio: Ontology is the TYPE SYSTEM the compiler uses
//!
//! When you write `let x: ChEBI.Aspirin`, the compiler:
//! 1. Resolves ChEBI.Aspirin to IRI http://purl.obolibrary.org/obo/CHEBI_15365
//! 2. Loads the class definition from the ontology
//! 3. Extracts: superclasses, properties, restrictions
//! 4. Constructs a Sounio type with full semantic information
//! 5. Uses this for type checking, not just documentation

pub mod bioportal;
pub mod loader_cache;
pub mod obo_parser;
pub mod optimized;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Internationalized Resource Identifier for ontology terms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IRI(pub String);

impl IRI {
    pub fn new(s: &str) -> Self {
        Self(s.to_string())
    }

    pub fn from_string(s: &str) -> Self {
        Self(s.to_string())
    }

    pub fn from_curie(prefix: &str, local: &str) -> Self {
        let base = match prefix.to_uppercase().as_str() {
            "CHEBI" => "http://purl.obolibrary.org/obo/CHEBI_",
            "GO" => "http://purl.obolibrary.org/obo/GO_",
            "DOID" => "http://purl.obolibrary.org/obo/DOID_",
            "HP" => "http://purl.obolibrary.org/obo/HP_",
            "PATO" => "http://purl.obolibrary.org/obo/PATO_",
            "UO" => "http://purl.obolibrary.org/obo/UO_",
            "BFO" => "http://purl.obolibrary.org/obo/BFO_",
            "RO" => "http://purl.obolibrary.org/obo/RO_",
            "IAO" => "http://purl.obolibrary.org/obo/IAO_",
            "OBI" => "http://purl.obolibrary.org/obo/OBI_",
            "UBERON" => "http://purl.obolibrary.org/obo/UBERON_",
            "CL" => "http://purl.obolibrary.org/obo/CL_",
            "NCBITaxon" | "NCBITAXON" => "http://purl.obolibrary.org/obo/NCBITaxon_",
            "PR" => "http://purl.obolibrary.org/obo/PR_",
            "SO" => "http://purl.obolibrary.org/obo/SO_",
            "MONDO" => "http://purl.obolibrary.org/obo/MONDO_",
            "MAXO" => "http://purl.obolibrary.org/obo/MAXO_",
            "DRUGBANK" | "DB" => "http://identifiers.org/drugbank/",
            "UNIPROT" => "http://identifiers.org/uniprot/",
            "MESH" => "http://id.nlm.nih.gov/mesh/",
            "SNOMED" | "SNOMEDCT" => "http://snomed.info/id/",
            "ICD10" => "http://hl7.org/fhir/sid/icd-10/",
            "LOINC" => "http://loinc.org/",
            "RXNORM" => "http://www.nlm.nih.gov/research/umls/rxnorm/",
            "FHIR" => "http://hl7.org/fhir/",
            "SCHEMA" => "https://schema.org/",
            _ => {
                // Unknown prefix, construct a generic OBO IRI
                return Self(format!(
                    "http://purl.obolibrary.org/obo/{}_{}",
                    prefix, local
                ));
            }
        };
        Self(format!("{}{}", base, local))
    }

    pub fn to_curie(&self) -> Option<(String, String)> {
        let s = &self.0;

        // Try to extract CURIE from common patterns
        if let Some(rest) = s.strip_prefix("http://purl.obolibrary.org/obo/")
            && let Some(idx) = rest.find('_')
        {
            let prefix = &rest[..idx];
            let local = &rest[idx + 1..];
            return Some((prefix.to_string(), local.to_string()));
        }

        if s.starts_with("https://schema.org/") {
            return Some(("SCHEMA".to_string(), s[19..].to_string()));
        }

        if s.starts_with("http://hl7.org/fhir/") {
            return Some(("FHIR".to_string(), s[20..].to_string()));
        }

        None
    }

    pub fn ontology(&self) -> OntologyId {
        if let Some((prefix, _)) = self.to_curie() {
            OntologyId::from_prefix(&prefix)
        } else {
            OntologyId::Unknown
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for IRI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Ontology identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OntologyId {
    // Foundation (BFO-based)
    BFO, // Basic Formal Ontology
    RO,  // Relation Ontology
    COB, // Core Ontology for Biology
    IAO, // Information Artifact Ontology

    // Measurement & Quality
    PATO, // Phenotype and Trait Ontology
    UO,   // Units of Measurement Ontology

    // Biology
    GO,        // Gene Ontology
    ChEBI,     // Chemical Entities of Biological Interest
    CL,        // Cell Ontology
    UBERON,    // Uber-anatomy Ontology
    PR,        // Protein Ontology
    SO,        // Sequence Ontology
    NCBITaxon, // NCBI Taxonomy

    // Disease & Phenotype
    DOID,  // Disease Ontology
    HP,    // Human Phenotype Ontology
    MONDO, // Mondo Disease Ontology
    MAXO,  // Medical Action Ontology

    // Clinical
    SNOMED, // SNOMED CT
    ICD10,  // ICD-10
    LOINC,  // Logical Observation Identifiers
    RxNorm, // RxNorm (drugs)
    FHIR,   // HL7 FHIR

    // External
    DrugBank,
    UniProt,
    MeSH,

    // Web
    SchemaOrg,

    // Unknown
    Unknown,
}

impl OntologyId {
    pub fn from_prefix(prefix: &str) -> Self {
        match prefix.to_uppercase().as_str() {
            "BFO" => Self::BFO,
            "RO" => Self::RO,
            "COB" => Self::COB,
            "IAO" => Self::IAO,
            "PATO" => Self::PATO,
            "UO" => Self::UO,
            "GO" => Self::GO,
            "CHEBI" => Self::ChEBI,
            "CL" => Self::CL,
            "UBERON" => Self::UBERON,
            "PR" => Self::PR,
            "SO" => Self::SO,
            "NCBITAXON" => Self::NCBITaxon,
            "DOID" => Self::DOID,
            "HP" => Self::HP,
            "MONDO" => Self::MONDO,
            "MAXO" => Self::MAXO,
            "SNOMED" | "SNOMEDCT" => Self::SNOMED,
            "ICD10" => Self::ICD10,
            "LOINC" => Self::LOINC,
            "RXNORM" => Self::RxNorm,
            "FHIR" => Self::FHIR,
            "DRUGBANK" | "DB" => Self::DrugBank,
            "UNIPROT" => Self::UniProt,
            "MESH" => Self::MeSH,
            "SCHEMA" => Self::SchemaOrg,
            _ => Self::Unknown,
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self {
            Self::BFO => "BFO",
            Self::RO => "RO",
            Self::COB => "COB",
            Self::IAO => "IAO",
            Self::PATO => "PATO",
            Self::UO => "UO",
            Self::GO => "GO",
            Self::ChEBI => "CHEBI",
            Self::CL => "CL",
            Self::UBERON => "UBERON",
            Self::PR => "PR",
            Self::SO => "SO",
            Self::NCBITaxon => "NCBITaxon",
            Self::DOID => "DOID",
            Self::HP => "HP",
            Self::MONDO => "MONDO",
            Self::MAXO => "MAXO",
            Self::SNOMED => "SNOMED",
            Self::ICD10 => "ICD10",
            Self::LOINC => "LOINC",
            Self::RxNorm => "RXNORM",
            Self::FHIR => "FHIR",
            Self::DrugBank => "DRUGBANK",
            Self::UniProt => "UNIPROT",
            Self::MeSH => "MESH",
            Self::SchemaOrg => "SCHEMA",
            Self::Unknown => "UNKNOWN",
        }
    }

    pub fn bioportal_acronym(&self) -> &'static str {
        match self {
            Self::BFO => "BFO",
            Self::RO => "RO",
            Self::GO => "GO",
            Self::ChEBI => "CHEBI",
            Self::DOID => "DOID",
            Self::HP => "HP",
            Self::MONDO => "MONDO",
            Self::UBERON => "UBERON",
            Self::CL => "CL",
            Self::PATO => "PATO",
            Self::UO => "UO",
            Self::NCBITaxon => "NCBITAXON",
            Self::SNOMED => "SNOMEDCT",
            Self::MeSH => "MESH",
            Self::LOINC => "LOINC",
            Self::RxNorm => "RXNORM",
            _ => "UNKNOWN",
        }
    }

    pub fn current_version(&self) -> &'static str {
        match self {
            Self::GO => "2024-01-01",
            Self::ChEBI => "231",
            Self::DOID => "2024-01-31",
            Self::HP => "2024-01-16",
            Self::MONDO => "2024-01-03",
            Self::UBERON => "2024-01-18",
            Self::PATO => "2024-01-16",
            _ => "unknown",
        }
    }
}

impl std::fmt::Display for OntologyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.prefix())
    }
}

/// Configuration for ontology loading
#[derive(Debug, Clone)]
pub struct OntologyLoaderConfig {
    /// Local cache directory for downloaded ontologies
    pub cache_dir: PathBuf,

    /// Maximum terms to keep in L1 (hot) cache
    pub l1_cache_size: usize,

    /// Maximum terms to keep in L2 (warm) cache
    pub l2_cache_size: usize,

    /// Enable federated resolution via BioPortal API
    pub enable_federated: bool,

    /// BioPortal API key (required for federated)
    pub bioportal_api_key: Option<String>,

    /// Preload these ontologies at startup
    pub preload_ontologies: Vec<OntologyId>,

    /// Build semantic distance index for these ontologies
    pub index_ontologies: Vec<OntologyId>,

    /// Network timeout in seconds
    pub network_timeout_secs: u64,
}

impl Default for OntologyLoaderConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".sounio/ontology_cache"),
            l1_cache_size: 10_000,
            l2_cache_size: 100_000,
            enable_federated: true,
            bioportal_api_key: std::env::var("BIOPORTAL_API_KEY").ok(),
            preload_ontologies: vec![
                OntologyId::BFO,
                OntologyId::RO,
                OntologyId::COB,
                OntologyId::SchemaOrg,
                OntologyId::FHIR,
            ],
            index_ontologies: vec![
                OntologyId::ChEBI,
                OntologyId::GO,
                OntologyId::DOID,
                OntologyId::PATO,
                OntologyId::UO,
            ],
            network_timeout_secs: 30,
        }
    }
}

/// A fully loaded ontological term with all semantic information
#[derive(Debug, Clone)]
pub struct LoadedTerm {
    /// The canonical IRI
    pub iri: IRI,

    /// Human-readable label
    pub label: String,

    /// Which ontology this term belongs to
    pub ontology: OntologyId,

    /// Direct superclasses (is-a relations)
    pub superclasses: Vec<IRI>,

    /// Direct subclasses
    pub subclasses: Vec<IRI>,

    /// Properties/relations this term can have
    pub properties: Vec<PropertyDefinition>,

    /// OWL restrictions (e.g., "has_part some Cell")
    pub restrictions: Vec<Restriction>,

    /// Cross-references to other ontologies
    pub xrefs: Vec<CrossReference>,

    /// Definition text
    pub definition: Option<String>,

    /// Synonyms
    pub synonyms: Vec<Synonym>,

    /// Precomputed: depth in hierarchy (for distance calculation)
    pub hierarchy_depth: u32,

    /// Precomputed: information content (for similarity)
    pub information_content: f64,

    /// Is this term obsolete?
    pub is_obsolete: bool,

    /// Replacement term if obsolete
    pub replaced_by: Option<IRI>,
}

impl LoadedTerm {
    pub fn new(iri: IRI, label: String, ontology: OntologyId) -> Self {
        Self {
            iri,
            label,
            ontology,
            superclasses: Vec::new(),
            subclasses: Vec::new(),
            properties: Vec::new(),
            restrictions: Vec::new(),
            xrefs: Vec::new(),
            definition: None,
            synonyms: Vec::new(),
            hierarchy_depth: 0,
            information_content: 0.0,
            is_obsolete: false,
            replaced_by: None,
        }
    }
}

/// Property definition from ontology
#[derive(Debug, Clone)]
pub struct PropertyDefinition {
    /// Property IRI
    pub iri: IRI,

    /// Human-readable label
    pub label: String,

    /// Domain (what types can have this property)
    pub domain: Option<IRI>,

    /// Range (what values this property can have)
    pub range: Option<IRI>,

    /// Is this property optional?
    pub optional: bool,

    /// Cardinality (min, max)
    pub cardinality: Option<(u32, Option<u32>)>,
}

/// OWL restriction
#[derive(Debug, Clone)]
pub struct Restriction {
    /// The property being restricted
    pub property: IRI,

    /// Type of restriction
    pub restriction_type: RestrictionType,

    /// The class or value being restricted to
    pub filler: IRI,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestrictionType {
    /// someValuesFrom (existential)
    Some,
    /// allValuesFrom (universal)
    All,
    /// hasValue
    Value,
    /// exactCardinality
    Exact(u32),
    /// minCardinality
    Min(u32),
    /// maxCardinality
    Max(u32),
}

/// Cross-reference to another ontology
#[derive(Debug, Clone)]
pub struct CrossReference {
    /// Target IRI
    pub target: IRI,

    /// Mapping confidence (0.0 - 1.0)
    pub confidence: f64,

    /// Source of the mapping
    pub source: String,
}

/// Synonym with scope
#[derive(Debug, Clone)]
pub struct Synonym {
    /// Synonym text
    pub text: String,

    /// Scope of the synonym
    pub scope: SynonymScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynonymScope {
    Exact,
    Narrow,
    Broad,
    Related,
}

/// Ontology metadata
#[derive(Debug, Clone)]
pub struct OntologyMetadata {
    /// Ontology identifier
    pub id: OntologyId,

    /// Full name
    pub name: String,

    /// Version string
    pub version: String,

    /// Number of terms
    pub term_count: usize,

    /// Last updated
    pub last_updated: Option<String>,

    /// License
    pub license: Option<String>,

    /// Description
    pub description: Option<String>,
}

/// Resolution error
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResolutionError {
    #[error("Term not found: {0}")]
    NotFound(IRI),

    #[error("Ontology not available: {0}")]
    OntologyNotAvailable(OntologyId),

    #[error("API error: status {status}, message: {message}")]
    ApiError { status: u16, message: String },

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Rate limited, retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },
}

/// The main ontology loader
pub struct OntologyLoader {
    config: OntologyLoaderConfig,

    /// L1 cache: Hot terms, fully in memory
    l1_cache: RwLock<lru::LruCache<IRI, Arc<LoadedTerm>>>,

    /// L2 cache: Warm terms, backed by on-disk storage
    l2_cache: loader_cache::L2Cache,

    /// Federated resolver for BioPortal/OBO
    federated: Option<bioportal::BioPortalClient>,

    /// Ontology metadata
    metadata: RwLock<HashMap<OntologyId, OntologyMetadata>>,

    /// Local OBO files
    local_ontologies: RwLock<HashMap<OntologyId, Vec<LoadedTerm>>>,
}

impl OntologyLoader {
    /// Create a new ontology loader with the given configuration
    pub fn new(config: OntologyLoaderConfig) -> Result<Self, OntologyLoadError> {
        // Create cache directory
        std::fs::create_dir_all(&config.cache_dir)
            .map_err(|e| OntologyLoadError::IoError(e.to_string()))?;

        // Initialize L1 cache
        let l1_cache = RwLock::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(config.l1_cache_size).unwrap(),
        ));

        // Initialize L2 cache
        let l2_cache = loader_cache::L2Cache::open(&config.cache_dir.join("l2_cache.db"))?;

        // Initialize federated resolver if enabled
        let federated = if config.enable_federated {
            Some(bioportal::BioPortalClient::new(
                config.bioportal_api_key.clone(),
                config.network_timeout_secs,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            l1_cache,
            l2_cache,
            federated,
            metadata: RwLock::new(HashMap::new()),
            local_ontologies: RwLock::new(HashMap::new()),
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, OntologyLoadError> {
        Self::new(OntologyLoaderConfig::default())
    }

    /// Get the configuration
    pub fn config(&self) -> &OntologyLoaderConfig {
        &self.config
    }

    /// Load an OBO file into the local cache
    pub fn load_obo_file(
        &self,
        path: &std::path::Path,
        ontology_id: OntologyId,
    ) -> Result<usize, OntologyLoadError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| OntologyLoadError::IoError(e.to_string()))?;

        let terms = obo_parser::parse_obo_file(&content)?;
        let count = terms.len();

        // Store in local cache
        if let Ok(mut local) = self.local_ontologies.write() {
            local.insert(ontology_id, terms);
        }

        // Update metadata
        if let Ok(mut meta) = self.metadata.write() {
            meta.insert(
                ontology_id,
                OntologyMetadata {
                    id: ontology_id,
                    name: format!("{} (local)", ontology_id.prefix()),
                    version: "local".to_string(),
                    term_count: count,
                    last_updated: None,
                    license: None,
                    description: None,
                },
            );
        }

        Ok(count)
    }

    /// Resolve an IRI to a LoadedTerm
    pub fn resolve(&self, iri: &IRI) -> Result<Arc<LoadedTerm>, ResolutionError> {
        // L1 cache hit?
        if let Ok(cache) = self.l1_cache.read()
            && let Some(term) = cache.peek(iri)
        {
            return Ok(term.clone());
        }

        // L2 cache hit?
        if let Some(term) = self.l2_cache.get(iri)? {
            // Promote to L1
            if let Ok(mut cache) = self.l1_cache.write() {
                cache.put(iri.clone(), term.clone());
            }
            return Ok(term);
        }

        // Check local ontologies
        let ontology_id = iri.ontology();
        if let Ok(local) = self.local_ontologies.read()
            && let Some(terms) = local.get(&ontology_id)
            && let Some(term) = terms.iter().find(|t| &t.iri == iri)
        {
            let term = Arc::new(term.clone());
            if let Ok(mut cache) = self.l1_cache.write() {
                cache.put(iri.clone(), term.clone());
            }
            return Ok(term);
        }

        // Federated resolution
        if let Some(ref federated) = self.federated {
            let term = federated.resolve_term(iri)?;
            let term = Arc::new(term);

            // Cache in both levels
            self.l2_cache.put(iri, &term)?;
            if let Ok(mut cache) = self.l1_cache.write() {
                cache.put(iri.clone(), term.clone());
            }

            return Ok(term);
        }

        Err(ResolutionError::NotFound(iri.clone()))
    }

    /// Resolve a CURIE (e.g., "CHEBI:15365")
    pub fn resolve_curie(&self, curie: &str) -> Result<Arc<LoadedTerm>, ResolutionError> {
        let iri = if let Some((prefix, local)) = curie.split_once(':') {
            IRI::from_curie(prefix, local)
        } else {
            return Err(ResolutionError::ParseError(format!(
                "Invalid CURIE: {}",
                curie
            )));
        };

        self.resolve(&iri)
    }

    /// Check if a term is a subclass of another
    pub fn is_subclass_of(&self, sub: &IRI, super_: &IRI) -> Result<bool, ResolutionError> {
        if sub == super_ {
            return Ok(true);
        }

        let term = self.resolve(sub)?;

        // Direct parent?
        if term.superclasses.contains(super_) {
            return Ok(true);
        }

        // Check ancestors recursively (with cycle detection)
        let mut visited = std::collections::HashSet::new();
        let mut queue = term.superclasses.clone();

        while let Some(parent) = queue.pop() {
            if &parent == super_ {
                return Ok(true);
            }

            if visited.insert(parent.clone())
                && let Ok(parent_term) = self.resolve(&parent)
            {
                queue.extend(parent_term.superclasses.clone());
            }
        }

        Ok(false)
    }

    /// Get all superclasses (transitive closure)
    pub fn get_ancestors(&self, iri: &IRI) -> Result<Vec<IRI>, ResolutionError> {
        let mut ancestors = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = vec![iri.clone()];

        while let Some(current) = queue.pop() {
            if visited.insert(current.clone())
                && let Ok(term) = self.resolve(&current)
            {
                for parent in &term.superclasses {
                    if !visited.contains(parent) {
                        ancestors.push(parent.clone());
                        queue.push(parent.clone());
                    }
                }
            }
        }

        Ok(ancestors)
    }

    /// Get lowest common ancestor of two terms
    pub fn lowest_common_ancestor(&self, a: &IRI, b: &IRI) -> Result<Option<IRI>, ResolutionError> {
        let ancestors_a = self.get_ancestors(a)?;
        let ancestors_b: std::collections::HashSet<_> =
            self.get_ancestors(b)?.into_iter().collect();

        // Find first common ancestor (closest to both)
        for ancestor in ancestors_a {
            if ancestors_b.contains(&ancestor) {
                return Ok(Some(ancestor));
            }
        }

        Ok(None)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let l1_size = self.l1_cache.read().map(|c| c.len()).unwrap_or(0);
        CacheStats {
            l1_size,
            l1_capacity: self.config.l1_cache_size,
            l2_size: self.l2_cache.len(),
            l2_capacity: self.config.l2_cache_size,
        }
    }

    /// Clear all caches
    pub fn clear_caches(&self) {
        if let Ok(mut cache) = self.l1_cache.write() {
            cache.clear();
        }
        self.l2_cache.clear();
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub l1_size: usize,
    pub l1_capacity: usize,
    pub l2_size: usize,
    pub l2_capacity: usize,
}

/// Ontology loading error
#[derive(Debug, Clone, thiserror::Error)]
pub enum OntologyLoadError {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Federated resolution error: {0}")]
    FederatedError(String),
}

impl From<obo_parser::OboParseError> for OntologyLoadError {
    fn from(e: obo_parser::OboParseError) -> Self {
        OntologyLoadError::ParseError(e.to_string())
    }
}

impl From<loader_cache::CacheError> for OntologyLoadError {
    fn from(e: loader_cache::CacheError) -> Self {
        OntologyLoadError::CacheError(e.to_string())
    }
}

impl From<bioportal::BioportalError> for OntologyLoadError {
    fn from(e: bioportal::BioportalError) -> Self {
        OntologyLoadError::FederatedError(e.to_string())
    }
}

impl From<loader_cache::CacheError> for ResolutionError {
    fn from(e: loader_cache::CacheError) -> Self {
        ResolutionError::CacheError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iri_from_curie() {
        let iri = IRI::from_curie("CHEBI", "15365");
        assert_eq!(iri.0, "http://purl.obolibrary.org/obo/CHEBI_15365");

        let iri = IRI::from_curie("GO", "0008150");
        assert_eq!(iri.0, "http://purl.obolibrary.org/obo/GO_0008150");

        let iri = IRI::from_curie("SCHEMA", "Drug");
        assert_eq!(iri.0, "https://schema.org/Drug");
    }

    #[test]
    fn test_iri_to_curie() {
        let iri = IRI::new("http://purl.obolibrary.org/obo/CHEBI_15365");
        let curie = iri.to_curie();
        assert_eq!(curie, Some(("CHEBI".to_string(), "15365".to_string())));
    }

    #[test]
    fn test_ontology_id_from_prefix() {
        assert_eq!(OntologyId::from_prefix("CHEBI"), OntologyId::ChEBI);
        assert_eq!(OntologyId::from_prefix("GO"), OntologyId::GO);
        assert_eq!(OntologyId::from_prefix("DOID"), OntologyId::DOID);
        assert_eq!(OntologyId::from_prefix("unknown"), OntologyId::Unknown);
    }

    #[test]
    fn test_loaded_term_creation() {
        let term = LoadedTerm::new(
            IRI::from_curie("CHEBI", "15365"),
            "aspirin".to_string(),
            OntologyId::ChEBI,
        );

        assert_eq!(term.label, "aspirin");
        assert_eq!(term.ontology, OntologyId::ChEBI);
        assert!(term.superclasses.is_empty());
    }
}
