//! Semantic-SQL Store for L3 Domain Ontologies
//!
//! This module provides access to OBO Foundry ontologies via pre-built
//! SQLite databases from the semantic-sql project.
//!
//! # Data Source
//!
//! Pre-built SQLite databases are available at:
//! <https://s3.amazonaws.com/bbop-sqlite/>
//!
//! Example files:
//! - chebi.db (~2GB for full ChEBI)
//! - go.db (~500MB for Gene Ontology)
//! - doid.db (~50MB for Disease Ontology)
//!
//! # Schema
//!
//! The semantic-sql schema includes:
//! - `statements`: RDF-like triples (subject, predicate, object)
//! - `rdfs_subclass_of_statement`: Materialized subclass relationships
//! - `entailed_edge`: Transitive closure of relationships
//! - `class_node`: Class definitions
//! - `object_property_node`: Property definitions
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::semantic_sql::SemanticSqlStore;
//!
//! let store = SemanticSqlStore::open("path/to/chebi.db")?;
//!
//! // Get term info
//! let aspirin = store.get_term("CHEBI:15365")?;
//!
//! // Check subsumption
//! let is_drug = store.is_subclass_of("CHEBI:15365", "CHEBI:23888")?;
//!
//! // Get superclasses
//! let parents = store.get_superclasses("CHEBI:15365")?;
//! ```

use std::collections::HashSet;
use std::path::Path;

use rusqlite::{Connection, OpenFlags, params};

use super::{OntologyError, OntologyResult};

/// A SQLite-backed ontology store using the semantic-sql schema
pub struct SemanticSqlStore {
    /// Database connection
    conn: Connection,
    /// Ontology identifier (e.g., "chebi", "go")
    ontology_id: String,
    /// Whether this database has entailed edges (transitive closure)
    has_entailed_edges: bool,
}

impl SemanticSqlStore {
    /// Open an existing semantic-sql database
    pub fn open(path: impl AsRef<Path>) -> OntologyResult<Self> {
        let path = path.as_ref();

        // Extract ontology ID from filename
        let ontology_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_lowercase();

        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|e| OntologyError::DatabaseError(format!("Cannot open database: {}", e)))?;

        // Check if entailed_edge table exists
        let has_entailed_edges = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='entailed_edge'")
            .and_then(|mut stmt| stmt.exists([]))
            .unwrap_or(false);

        Ok(Self {
            conn,
            ontology_id,
            has_entailed_edges,
        })
    }

    /// Get the ontology identifier
    pub fn ontology_id(&self) -> &str {
        &self.ontology_id
    }

    /// Check if this database has precomputed entailed edges
    pub fn has_entailed_edges(&self) -> bool {
        self.has_entailed_edges
    }

    /// Get a term by its CURIE or IRI
    pub fn get_term(&self, id: &str) -> OntologyResult<SqlTerm> {
        // Normalize to IRI
        let iri = self.curie_to_iri(id);

        // Query class_node or rdfs_label
        let mut stmt = self
            .conn
            .prepare(
                r#"
            SELECT
                cn.id,
                COALESCE(l.value, cn.id) as label,
                def.value as definition
            FROM class_node cn
            LEFT JOIN statements l ON cn.id = l.subject
                AND l.predicate = 'http://www.w3.org/2000/01/rdf-schema#label'
            LEFT JOIN statements def ON cn.id = def.subject
                AND def.predicate = 'http://purl.obolibrary.org/obo/IAO_0000115'
            WHERE cn.id = ?1
            LIMIT 1
            "#,
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let result = stmt
            .query_row([&iri], |row| {
                Ok(SqlTerm {
                    id: row.get(0)?,
                    label: row.get(1)?,
                    definition: row.get(2)?,
                    synonyms: vec![], // Loaded separately
                    xrefs: vec![],    // Loaded separately
                    obsolete: false,  // Loaded separately
                })
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => OntologyError::TermNotFound {
                    ontology: self.ontology_id.clone(),
                    term: id.to_string(),
                },
                _ => OntologyError::DatabaseError(e.to_string()),
            })?;

        // Load synonyms
        let synonyms = self.get_synonyms(&iri)?;

        Ok(SqlTerm { synonyms, ..result })
    }

    /// Get synonyms for a term
    fn get_synonyms(&self, iri: &str) -> OntologyResult<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare(
                r#"
            SELECT value FROM statements
            WHERE subject = ?1
            AND predicate IN (
                'http://www.geneontology.org/formats/oboInOwl#hasExactSynonym',
                'http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym',
                'http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym',
                'http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym'
            )
            "#,
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let synonyms = stmt
            .query_map([iri], |row| row.get(0))
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(synonyms)
    }

    /// Check if child is a subclass of parent (direct or transitive)
    pub fn is_subclass_of(&self, child: &str, parent: &str) -> OntologyResult<bool> {
        let child_iri = self.curie_to_iri(child);
        let parent_iri = self.curie_to_iri(parent);

        if child_iri == parent_iri {
            return Ok(true);
        }

        // Use entailed_edge if available (faster)
        if self.has_entailed_edges {
            let mut stmt = self
                .conn
                .prepare(
                    r#"
                SELECT 1 FROM entailed_edge
                WHERE subject = ?1 AND object = ?2
                AND predicate = 'http://www.w3.org/2000/01/rdf-schema#subClassOf'
                LIMIT 1
                "#,
                )
                .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

            return stmt
                .exists([&child_iri, &parent_iri])
                .map_err(|e| OntologyError::DatabaseError(e.to_string()));
        }

        // Fall back to recursive query
        let mut stmt = self
            .conn
            .prepare(
                r#"
            WITH RECURSIVE ancestors(id) AS (
                SELECT ?1
                UNION
                SELECT object FROM rdfs_subclass_of_statement
                WHERE subject IN (SELECT id FROM ancestors)
            )
            SELECT 1 FROM ancestors WHERE id = ?2 LIMIT 1
            "#,
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        stmt.exists([&child_iri, &parent_iri])
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))
    }

    /// Get direct superclasses
    pub fn get_superclasses(&self, id: &str) -> OntologyResult<Vec<SqlTerm>> {
        let iri = self.curie_to_iri(id);

        let mut stmt = self
            .conn
            .prepare(
                r#"
            SELECT
                s.object as id,
                COALESCE(l.value, s.object) as label
            FROM rdfs_subclass_of_statement s
            LEFT JOIN statements l ON s.object = l.subject
                AND l.predicate = 'http://www.w3.org/2000/01/rdf-schema#label'
            WHERE s.subject = ?1
            "#,
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let terms = stmt
            .query_map([&iri], |row| {
                Ok(SqlTerm {
                    id: row.get(0)?,
                    label: row.get(1)?,
                    definition: None,
                    synonyms: vec![],
                    xrefs: vec![],
                    obsolete: false,
                })
            })
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(terms)
    }

    /// Get direct subclasses
    pub fn get_subclasses(&self, id: &str) -> OntologyResult<Vec<SqlTerm>> {
        let iri = self.curie_to_iri(id);

        let mut stmt = self
            .conn
            .prepare(
                r#"
            SELECT
                s.subject as id,
                COALESCE(l.value, s.subject) as label
            FROM rdfs_subclass_of_statement s
            LEFT JOIN statements l ON s.subject = l.subject
                AND l.predicate = 'http://www.w3.org/2000/01/rdf-schema#label'
            WHERE s.object = ?1
            "#,
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let terms = stmt
            .query_map([&iri], |row| {
                Ok(SqlTerm {
                    id: row.get(0)?,
                    label: row.get(1)?,
                    definition: None,
                    synonyms: vec![],
                    xrefs: vec![],
                    obsolete: false,
                })
            })
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(terms)
    }

    /// Get all ancestors (transitive superclasses)
    pub fn get_ancestors(&self, id: &str) -> OntologyResult<Vec<String>> {
        let iri = self.curie_to_iri(id);

        if self.has_entailed_edges {
            let mut stmt = self
                .conn
                .prepare(
                    r#"
                SELECT DISTINCT object FROM entailed_edge
                WHERE subject = ?1
                AND predicate = 'http://www.w3.org/2000/01/rdf-schema#subClassOf'
                "#,
                )
                .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

            let ancestors = stmt
                .query_map([&iri], |row| row.get(0))
                .map_err(|e| OntologyError::DatabaseError(e.to_string()))?
                .filter_map(|r| r.ok())
                .map(|iri: String| self.iri_to_curie(&iri))
                .collect();

            Ok(ancestors)
        } else {
            // Recursive query
            let mut stmt = self
                .conn
                .prepare(
                    r#"
                WITH RECURSIVE ancestors(id) AS (
                    SELECT object FROM rdfs_subclass_of_statement WHERE subject = ?1
                    UNION
                    SELECT object FROM rdfs_subclass_of_statement
                    WHERE subject IN (SELECT id FROM ancestors)
                )
                SELECT DISTINCT id FROM ancestors
                "#,
                )
                .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

            let ancestors = stmt
                .query_map([&iri], |row| row.get(0))
                .map_err(|e| OntologyError::DatabaseError(e.to_string()))?
                .filter_map(|r| r.ok())
                .map(|iri: String| self.iri_to_curie(&iri))
                .collect();

            Ok(ancestors)
        }
    }

    /// Search for terms by label (case-insensitive)
    pub fn search(&self, query: &str, limit: usize) -> OntologyResult<Vec<SqlTerm>> {
        let pattern = format!("%{}%", query.to_lowercase());

        let mut stmt = self
            .conn
            .prepare(
                r#"
            SELECT DISTINCT
                s.subject as id,
                s.value as label
            FROM statements s
            WHERE s.predicate = 'http://www.w3.org/2000/01/rdf-schema#label'
            AND LOWER(s.value) LIKE ?1
            LIMIT ?2
            "#,
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let terms = stmt
            .query_map(params![pattern, limit as i64], |row| {
                Ok(SqlTerm {
                    id: row.get(0)?,
                    label: row.get(1)?,
                    definition: None,
                    synonyms: vec![],
                    xrefs: vec![],
                    obsolete: false,
                })
            })
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(terms)
    }

    /// Get relationship (edge) between two terms
    pub fn get_relationship(&self, subject: &str, object: &str) -> OntologyResult<Vec<String>> {
        let subject_iri = self.curie_to_iri(subject);
        let object_iri = self.curie_to_iri(object);

        let mut stmt = self
            .conn
            .prepare(
                r#"
            SELECT DISTINCT predicate FROM statements
            WHERE subject = ?1 AND object = ?2
            "#,
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let predicates = stmt
            .query_map([&subject_iri, &object_iri], |row| row.get(0))
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(predicates)
    }

    /// Get total term count
    pub fn term_count(&self) -> OntologyResult<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM class_node", [], |row| row.get(0))
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        Ok(count as usize)
    }

    /// Convert CURIE to full IRI
    fn curie_to_iri(&self, curie: &str) -> String {
        // Already an IRI
        if curie.starts_with("http://") || curie.starts_with("https://") {
            return curie.to_string();
        }

        // Parse CURIE
        if let Some((prefix, local_id)) = curie.split_once(':') {
            let prefix_upper = prefix.to_uppercase();

            // Common OBO prefixes
            let namespace = match prefix_upper.as_str() {
                "CHEBI" => "http://purl.obolibrary.org/obo/CHEBI_",
                "GO" => "http://purl.obolibrary.org/obo/GO_",
                "DOID" => "http://purl.obolibrary.org/obo/DOID_",
                "HP" => "http://purl.obolibrary.org/obo/HP_",
                "MONDO" => "http://purl.obolibrary.org/obo/MONDO_",
                "UBERON" => "http://purl.obolibrary.org/obo/UBERON_",
                "CL" => "http://purl.obolibrary.org/obo/CL_",
                "PR" => "http://purl.obolibrary.org/obo/PR_",
                "SO" => "http://purl.obolibrary.org/obo/SO_",
                "ENVO" => "http://purl.obolibrary.org/obo/ENVO_",
                "OBI" => "http://purl.obolibrary.org/obo/OBI_",
                "PATO" => "http://purl.obolibrary.org/obo/PATO_",
                "BFO" => "http://purl.obolibrary.org/obo/BFO_",
                "RO" => "http://purl.obolibrary.org/obo/RO_",
                "IAO" => "http://purl.obolibrary.org/obo/IAO_",
                "NCBITaxon" | "NCBITAXON" => "http://purl.obolibrary.org/obo/NCBITaxon_",
                _ => {
                    // Generic OBO pattern
                    return format!("http://purl.obolibrary.org/obo/{}", curie.replace(':', "_"));
                }
            };

            format!("{}{}", namespace, local_id)
        } else {
            curie.to_string()
        }
    }

    /// Convert IRI to CURIE
    fn iri_to_curie(&self, iri: &str) -> String {
        // OBO pattern: http://purl.obolibrary.org/obo/PREFIX_ID
        if iri.starts_with("http://purl.obolibrary.org/obo/") {
            let local = &iri[31..]; // Skip base URL
            if let Some(idx) = local.find('_') {
                let prefix = &local[..idx];
                let id = &local[idx + 1..];
                return format!("{}:{}", prefix, id);
            }
        }

        // Return as-is if can't convert
        iri.to_string()
    }
}

/// A term from the semantic-sql database
#[derive(Debug, Clone)]
pub struct SqlTerm {
    /// Full IRI or CURIE
    pub id: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Definition text
    pub definition: Option<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Cross-references
    pub xrefs: Vec<String>,
    /// Whether this term is obsolete
    pub obsolete: bool,
}

impl SqlTerm {
    /// Get the CURIE form of the ID
    pub fn curie(&self) -> String {
        if self.id.starts_with("http://purl.obolibrary.org/obo/") {
            let local = &self.id[31..];
            if let Some(idx) = local.find('_') {
                let prefix = &local[..idx];
                let id = &local[idx + 1..];
                return format!("{}:{}", prefix, id);
            }
        }
        self.id.clone()
    }

    /// Get the display label (label or ID if no label)
    pub fn display_label(&self) -> &str {
        self.label.as_deref().unwrap_or(&self.id)
    }
}

/// Information about an available ontology database
#[derive(Debug, Clone)]
pub struct SqlOntology {
    /// Ontology identifier (e.g., "chebi")
    pub id: String,
    /// Full name
    pub name: String,
    /// Path to database file
    pub path: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Number of terms
    pub term_count: Option<usize>,
    /// Version
    pub version: Option<String>,
}

/// Manager for multiple semantic-sql databases
pub struct SemanticSqlManager {
    /// Available ontologies
    ontologies: Vec<SqlOntology>,
    /// Open stores (lazy loaded)
    stores: std::collections::HashMap<String, SemanticSqlStore>,
    /// Base directory for database files
    base_dir: Option<String>,
}

impl SemanticSqlManager {
    /// Create a new manager
    pub fn new() -> Self {
        Self {
            ontologies: vec![],
            stores: std::collections::HashMap::new(),
            base_dir: None,
        }
    }

    /// Set the base directory for database files
    pub fn with_base_dir(mut self, dir: impl Into<String>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    /// Register an available ontology
    pub fn register(&mut self, ontology: SqlOntology) {
        self.ontologies.push(ontology);
    }

    /// Get a store by ontology ID (lazy loaded)
    pub fn get_store(&mut self, ontology_id: &str) -> OntologyResult<&SemanticSqlStore> {
        if !self.stores.contains_key(ontology_id) {
            let ontology = self
                .ontologies
                .iter()
                .find(|o| o.id == ontology_id)
                .ok_or_else(|| OntologyError::OntologyNotAvailable(ontology_id.to_string()))?;

            let store = SemanticSqlStore::open(&ontology.path)?;
            self.stores.insert(ontology_id.to_string(), store);
        }

        Ok(self.stores.get(ontology_id).unwrap())
    }

    /// List available ontologies
    pub fn available_ontologies(&self) -> &[SqlOntology] {
        &self.ontologies
    }

    /// Check if an ontology is available
    pub fn is_available(&self, ontology_id: &str) -> bool {
        self.ontologies.iter().any(|o| o.id == ontology_id)
    }
}

impl Default for SemanticSqlManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curie_to_iri() {
        // Create a dummy store for testing
        // In real tests, we'd use a test database
        let store = SemanticSqlStore {
            conn: Connection::open_in_memory().unwrap(),
            ontology_id: "test".to_string(),
            has_entailed_edges: false,
        };

        assert_eq!(
            store.curie_to_iri("CHEBI:15365"),
            "http://purl.obolibrary.org/obo/CHEBI_15365"
        );
        assert_eq!(
            store.curie_to_iri("GO:0008150"),
            "http://purl.obolibrary.org/obo/GO_0008150"
        );
    }

    #[test]
    fn test_iri_to_curie() {
        let store = SemanticSqlStore {
            conn: Connection::open_in_memory().unwrap(),
            ontology_id: "test".to_string(),
            has_entailed_edges: false,
        };

        assert_eq!(
            store.iri_to_curie("http://purl.obolibrary.org/obo/CHEBI_15365"),
            "CHEBI:15365"
        );
    }

    #[test]
    fn test_sql_term_curie() {
        let term = SqlTerm {
            id: "http://purl.obolibrary.org/obo/CHEBI_15365".to_string(),
            label: Some("aspirin".to_string()),
            definition: None,
            synonyms: vec![],
            xrefs: vec![],
            obsolete: false,
        };

        assert_eq!(term.curie(), "CHEBI:15365");
    }
}
