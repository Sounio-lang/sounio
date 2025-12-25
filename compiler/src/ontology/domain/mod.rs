//! L3: Domain Ontologies
//!
//! ~500,000 terms for specialized domains like chemistry, biology, and medicine.
//! Loaded lazily from pre-built Semantic-SQL (SQLite) databases.
//!
//! Supported ontologies:
//! - ChEBI: Chemical Entities of Biological Interest (~180,000 terms)
//! - GO: Gene Ontology (~45,000 terms)
//! - DOID: Disease Ontology (~20,000 terms)
//! - HP: Human Phenotype Ontology (~16,000 terms)
//! - MONDO: Mondo Disease Ontology (~30,000 terms)
//! - UBERON: Anatomy Ontology (~15,000 terms)
//! - CL: Cell Ontology (~2,500 terms)
//! - NCBITaxon: NCBI Taxonomy (~2M terms, subset loaded)
//! - PR: Protein Ontology (~400,000 terms)
//! - SO: Sequence Ontology (~2,500 terms)
//! - ENVO: Environment Ontology (~7,000 terms)
//! - OBI: Ontology for Biomedical Investigations (~4,500 terms)
//!
//! # Performance
//!
//! - SQLite databases are memory-mapped for fast queries
//! - Index on term IDs for O(1) lookups
//! - Transitive closure tables for efficient subsumption
//! - LRU cache for hot terms

pub mod loader;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::epistemic::{EpistemicStatus, TermId};
use crate::ontology::OntologyError;

/// L3 Domain Ontologies collection
pub struct DomainOntologies {
    /// Loaded ontologies by prefix
    ontologies: HashMap<String, LazyOntology>,
    /// Path to ontology database directory
    db_path: PathBuf,
    /// Cache of recently accessed terms
    cache: lru::LruCache<String, DomainTerm>,
}

/// A lazily-loaded domain ontology
pub struct LazyOntology {
    /// Ontology metadata
    pub metadata: OntologyMetadata,
    /// Whether the ontology has been fully loaded
    loaded: bool,
    /// In-memory index (populated on first access)
    index: Option<DomainIndex>,
}

/// Metadata about a domain ontology
#[derive(Debug, Clone)]
pub struct OntologyMetadata {
    /// Ontology prefix (e.g., "ChEBI", "GO")
    pub prefix: String,
    /// Full name
    pub name: String,
    /// Description
    pub description: String,
    /// Version
    pub version: String,
    /// Approximate term count
    pub term_count: usize,
    /// Database file path
    pub db_file: PathBuf,
    /// Whether ontology is available
    pub available: bool,
}

/// In-memory index for a loaded ontology
pub(crate) struct DomainIndex {
    /// Term lookup by ID
    pub(crate) terms: HashMap<String, DomainTerm>,
    /// Subsumption cache (child -> ancestors)
    pub(crate) ancestors: HashMap<String, Vec<String>>,
}

/// A term from a domain ontology
#[derive(Debug, Clone)]
pub struct DomainTerm {
    /// Term identifier
    pub id: TermId,
    /// Source ontology prefix
    pub ontology: String,
    /// Definition text
    pub definition: Option<String>,
    /// Direct parent terms (is_a relations)
    pub parents: Vec<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Cross-references to other ontologies
    pub xrefs: Vec<String>,
    /// Computed epistemic status
    pub epistemic: EpistemicStatus,
}

impl DomainOntologies {
    /// Create a new domain ontologies manager
    pub fn new() -> Self {
        Self::with_path(Self::default_db_path())
    }

    /// Create with a specific database path
    pub fn with_path(db_path: PathBuf) -> Self {
        let mut ontologies = HashMap::new();

        // Register known domain ontologies
        for meta in Self::known_ontologies(&db_path) {
            ontologies.insert(
                meta.prefix.clone(),
                LazyOntology {
                    metadata: meta,
                    loaded: false,
                    index: None,
                },
            );
        }

        Self {
            ontologies,
            db_path,
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(10000).unwrap()),
        }
    }

    /// Get default database path
    fn default_db_path() -> PathBuf {
        // Check environment variable
        if let Ok(path) = std::env::var("DEMETRIOS_ONTOLOGY_DB") {
            return PathBuf::from(path);
        }

        // Check relative to executable
        if let Ok(exe) = std::env::current_exe()
            && let Some(parent) = exe.parent()
        {
            let db_path = parent.join("ontology_db");
            if db_path.exists() {
                return db_path;
            }
        }

        // Default location
        PathBuf::from("/usr/share/sounio/ontology_db")
    }

    /// List of known domain ontologies
    fn known_ontologies(db_path: &Path) -> Vec<OntologyMetadata> {
        vec![
            OntologyMetadata {
                prefix: "CHEBI".into(),
                name: "Chemical Entities of Biological Interest".into(),
                description: "Ontology of molecular entities focused on small chemical compounds"
                    .into(),
                version: "2024-01".into(),
                term_count: 180000,
                db_file: db_path.join("chebi.db"),
                available: db_path.join("chebi.db").exists(),
            },
            OntologyMetadata {
                prefix: "GO".into(),
                name: "Gene Ontology".into(),
                description: "Ontology of genes, gene products, and their biological attributes"
                    .into(),
                version: "2024-01".into(),
                term_count: 45000,
                db_file: db_path.join("go.db"),
                available: db_path.join("go.db").exists(),
            },
            OntologyMetadata {
                prefix: "DOID".into(),
                name: "Disease Ontology".into(),
                description: "Ontology of human disease".into(),
                version: "2024-01".into(),
                term_count: 20000,
                db_file: db_path.join("doid.db"),
                available: db_path.join("doid.db").exists(),
            },
            OntologyMetadata {
                prefix: "HP".into(),
                name: "Human Phenotype Ontology".into(),
                description: "Ontology of phenotypic abnormalities in human disease".into(),
                version: "2024-01".into(),
                term_count: 16000,
                db_file: db_path.join("hp.db"),
                available: db_path.join("hp.db").exists(),
            },
            OntologyMetadata {
                prefix: "MONDO".into(),
                name: "Mondo Disease Ontology".into(),
                description: "Integrated disease ontology".into(),
                version: "2024-01".into(),
                term_count: 30000,
                db_file: db_path.join("mondo.db"),
                available: db_path.join("mondo.db").exists(),
            },
            OntologyMetadata {
                prefix: "UBERON".into(),
                name: "Uberon Multi-Species Anatomy Ontology".into(),
                description: "Cross-species anatomy ontology".into(),
                version: "2024-01".into(),
                term_count: 15000,
                db_file: db_path.join("uberon.db"),
                available: db_path.join("uberon.db").exists(),
            },
            OntologyMetadata {
                prefix: "CL".into(),
                name: "Cell Ontology".into(),
                description: "Ontology of cell types".into(),
                version: "2024-01".into(),
                term_count: 2500,
                db_file: db_path.join("cl.db"),
                available: db_path.join("cl.db").exists(),
            },
            OntologyMetadata {
                prefix: "NCBITAXON".into(),
                name: "NCBI Taxonomy".into(),
                description: "NCBI organismal classification".into(),
                version: "2024-01".into(),
                term_count: 2000000,
                db_file: db_path.join("ncbitaxon.db"),
                available: db_path.join("ncbitaxon.db").exists(),
            },
            OntologyMetadata {
                prefix: "PR".into(),
                name: "Protein Ontology".into(),
                description: "Ontology of protein-related entities".into(),
                version: "2024-01".into(),
                term_count: 400000,
                db_file: db_path.join("pr.db"),
                available: db_path.join("pr.db").exists(),
            },
            OntologyMetadata {
                prefix: "SO".into(),
                name: "Sequence Ontology".into(),
                description: "Ontology of biological sequence features".into(),
                version: "2024-01".into(),
                term_count: 2500,
                db_file: db_path.join("so.db"),
                available: db_path.join("so.db").exists(),
            },
            OntologyMetadata {
                prefix: "ENVO".into(),
                name: "Environment Ontology".into(),
                description: "Ontology of environmental features and habitats".into(),
                version: "2024-01".into(),
                term_count: 7000,
                db_file: db_path.join("envo.db"),
                available: db_path.join("envo.db").exists(),
            },
            OntologyMetadata {
                prefix: "OBI".into(),
                name: "Ontology for Biomedical Investigations".into(),
                description: "Ontology of investigations, protocols, and instruments".into(),
                version: "2024-01".into(),
                term_count: 4500,
                db_file: db_path.join("obi.db"),
                available: db_path.join("obi.db").exists(),
            },
        ]
    }

    /// Get metadata for an ontology
    pub fn get_metadata(&self, prefix: &str) -> Option<&OntologyMetadata> {
        self.ontologies.get(prefix).map(|o| &o.metadata)
    }

    /// Check if an ontology is available
    pub fn is_available(&self, prefix: &str) -> bool {
        self.ontologies
            .get(prefix)
            .map(|o| o.metadata.available)
            .unwrap_or(false)
    }

    /// List available ontologies
    pub fn available_ontologies(&self) -> Vec<&OntologyMetadata> {
        self.ontologies
            .values()
            .filter(|o| o.metadata.available)
            .map(|o| &o.metadata)
            .collect()
    }

    /// Resolve a term from domain ontologies
    pub fn resolve(&mut self, curie: &str) -> Result<Option<DomainTerm>, OntologyError> {
        // Check cache first
        if let Some(term) = self.cache.get(curie) {
            return Ok(Some(term.clone()));
        }

        // Parse CURIE to get prefix
        let (prefix, local_id) = curie.split_once(':').ok_or_else(|| {
            OntologyError::InvalidTermFormat(format!("Invalid CURIE format: {}", curie))
        })?;

        let prefix = prefix.to_uppercase();

        // Check if ontology exists
        if !self.ontologies.contains_key(&prefix) {
            return Ok(None);
        }

        // Load ontology if needed
        self.ensure_loaded(&prefix)?;

        // Look up term
        if let Some(ontology) = self.ontologies.get(&prefix)
            && let Some(index) = &ontology.index
            && let Some(term) = index.terms.get(curie)
        {
            // Cache the result
            self.cache.put(curie.to_string(), term.clone());
            return Ok(Some(term.clone()));
        }

        Ok(None)
    }

    /// Check subsumption between two terms
    pub fn is_subclass_of(&mut self, child: &str, parent: &str) -> Result<bool, OntologyError> {
        // Same term is always subclass of itself
        if child == parent {
            return Ok(true);
        }

        // Parse CURIEs
        let (child_prefix, _) = child
            .split_once(':')
            .ok_or_else(|| OntologyError::InvalidTermFormat(format!("Invalid CURIE: {}", child)))?;

        let (parent_prefix, _) = parent.split_once(':').ok_or_else(|| {
            OntologyError::InvalidTermFormat(format!("Invalid CURIE: {}", parent))
        })?;

        // Cross-ontology subsumption not supported at this layer
        if child_prefix.to_uppercase() != parent_prefix.to_uppercase() {
            return Ok(false);
        }

        let prefix = child_prefix.to_uppercase();
        self.ensure_loaded(&prefix)?;

        // Check transitive closure
        if let Some(ontology) = self.ontologies.get(&prefix)
            && let Some(index) = &ontology.index
            && let Some(ancestors) = index.ancestors.get(child)
        {
            return Ok(ancestors.contains(&parent.to_string()));
        }

        Ok(false)
    }

    /// Ensure an ontology is loaded into memory
    fn ensure_loaded(&mut self, prefix: &str) -> Result<(), OntologyError> {
        let ontology = self.ontologies.get_mut(prefix).ok_or_else(|| {
            OntologyError::OntologyNotAvailable(format!("Unknown ontology: {}", prefix))
        })?;

        if ontology.loaded {
            return Ok(());
        }

        if !ontology.metadata.available {
            return Err(OntologyError::OntologyNotAvailable(format!(
                "Ontology {} database not found at {:?}",
                prefix, ontology.metadata.db_file
            )));
        }

        // Load from SQLite database
        #[cfg(feature = "ontology")]
        {
            let index = loader::load_ontology_from_sqlite(&ontology.metadata)?;
            ontology.index = Some(index);
            ontology.loaded = true;
        }

        #[cfg(not(feature = "ontology"))]
        {
            // Bootstrap mode - create empty index
            ontology.index = Some(DomainIndex {
                terms: HashMap::new(),
                ancestors: HashMap::new(),
            });
            ontology.loaded = true;
        }

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> DomainStats {
        let mut available = 0;
        let mut loaded = 0;
        let mut total_terms = 0;

        for ontology in self.ontologies.values() {
            if ontology.metadata.available {
                available += 1;
                if ontology.loaded {
                    loaded += 1;
                    if let Some(index) = &ontology.index {
                        total_terms += index.terms.len();
                    }
                }
            }
        }

        DomainStats {
            registered: self.ontologies.len(),
            available,
            loaded,
            cached_terms: self.cache.len(),
            total_loaded_terms: total_terms,
        }
    }
}

impl Default for DomainOntologies {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about domain ontology usage
#[derive(Debug, Clone)]
pub struct DomainStats {
    /// Number of registered ontologies
    pub registered: usize,
    /// Number of ontologies with available databases
    pub available: usize,
    /// Number of currently loaded ontologies
    pub loaded: usize,
    /// Number of cached terms
    pub cached_terms: usize,
    /// Total terms in loaded ontologies
    pub total_loaded_terms: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_ontologies_new() {
        let domains = DomainOntologies::new();
        assert!(domains.ontologies.len() > 0);
    }

    #[test]
    fn test_known_ontologies() {
        let db_path = PathBuf::from("/tmp");
        let known = DomainOntologies::known_ontologies(&db_path);

        // Should have ChEBI, GO, DOID, HP, etc.
        let prefixes: Vec<_> = known.iter().map(|m| m.prefix.as_str()).collect();
        assert!(prefixes.contains(&"CHEBI"));
        assert!(prefixes.contains(&"GO"));
        assert!(prefixes.contains(&"DOID"));
    }

    #[test]
    fn test_get_metadata() {
        let domains = DomainOntologies::new();

        let chebi = domains.get_metadata("CHEBI");
        assert!(chebi.is_some());
        assert_eq!(
            chebi.unwrap().name,
            "Chemical Entities of Biological Interest"
        );

        let unknown = domains.get_metadata("UNKNOWN");
        assert!(unknown.is_none());
    }

    #[test]
    fn test_stats() {
        let domains = DomainOntologies::new();
        let stats = domains.stats();

        assert!(stats.registered > 0);
        // No databases available in test environment
        assert_eq!(stats.loaded, 0);
    }
}
