//! SQLite Loader for Domain Ontologies
//!
//! Loads ontology data from pre-built Semantic-SQL databases.
//! These databases follow the OBO Foundry Semantic-SQL schema.

use std::collections::HashMap;

use crate::epistemic::{Confidence, EpistemicStatus, Evidence, EvidenceKind, Revisability, Source};
use crate::ontology::OntologyError;

use super::{DomainIndex, OntologyMetadata};

/// Load an ontology from its SQLite database
#[cfg(feature = "ontology")]
pub(crate) fn load_ontology_from_sqlite(
    metadata: &OntologyMetadata,
) -> Result<DomainIndex, OntologyError> {
    use rusqlite::Connection;

    let conn = Connection::open(&metadata.db_file).map_err(|e| {
        OntologyError::DatabaseError(format!(
            "Failed to open database {:?}: {}",
            metadata.db_file, e
        ))
    })?;

    let mut terms = HashMap::new();
    let mut ancestors = HashMap::new();

    // Load terms from statements table (Semantic-SQL schema)
    // The schema typically has:
    // - statements(stanza, subject, predicate, object, value, datatype, language)
    // - edges(subject, predicate, object)
    // - entailed_edge (for transitive closure)

    // Check which tables exist
    let has_statements: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='statements'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    let has_edges: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='edge'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if has_statements {
        load_from_statements(&conn, &metadata.prefix, &mut terms)?;
    }

    if has_edges {
        load_parents_from_edges(&conn, &metadata.prefix, &mut terms)?;
        load_ancestors(&conn, &metadata.prefix, &mut ancestors)?;
    }

    Ok(DomainIndex { terms, ancestors })
}

#[cfg(feature = "ontology")]
fn load_from_statements(
    conn: &rusqlite::Connection,
    prefix: &str,
    terms: &mut HashMap<String, DomainTerm>,
) -> Result<(), OntologyError> {
    // Query for term labels (rdfs:label)
    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT subject, value FROM statements
             WHERE predicate = 'rdfs:label'
             AND subject LIKE ? || ':%'",
        )
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    let rows = stmt
        .query_map([prefix], |row| {
            let subject: String = row.get(0)?;
            let label: String = row.get(1)?;
            Ok((subject, label))
        })
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    for row in rows {
        let (subject, label) = row.map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let term = terms.entry(subject.clone()).or_insert_with(|| DomainTerm {
            id: TermId::with_label(&subject, &label),
            ontology: prefix.to_string(),
            definition: None,
            parents: vec![],
            synonyms: vec![],
            xrefs: vec![],
            epistemic: default_epistemic(&subject, prefix),
        });
        term.id = TermId::with_label(&subject, &label);
    }

    // Query for definitions (IAO:0000115 or obo:IAO_0000115)
    let mut def_stmt = conn
        .prepare(
            "SELECT subject, value FROM statements
             WHERE (predicate = 'IAO:0000115' OR predicate = 'obo:IAO_0000115' OR predicate LIKE '%definition%')
             AND subject LIKE ? || ':%'",
        )
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    let def_rows = def_stmt
        .query_map([prefix], |row| {
            let subject: String = row.get(0)?;
            let definition: String = row.get(1)?;
            Ok((subject, definition))
        })
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    for row in def_rows {
        let (subject, definition) = row.map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        if let Some(term) = terms.get_mut(&subject) {
            term.definition = Some(definition);
        }
    }

    // Query for synonyms
    let mut syn_stmt = conn
        .prepare(
            "SELECT subject, value FROM statements
             WHERE predicate LIKE '%synonym%'
             AND subject LIKE ? || ':%'",
        )
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    let syn_rows = syn_stmt
        .query_map([prefix], |row| {
            let subject: String = row.get(0)?;
            let synonym: String = row.get(1)?;
            Ok((subject, synonym))
        })
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    for row in syn_rows {
        let (subject, synonym) = row.map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        if let Some(term) = terms.get_mut(&subject) {
            term.synonyms.push(synonym);
        }
    }

    Ok(())
}

#[cfg(feature = "ontology")]
fn load_parents_from_edges(
    conn: &rusqlite::Connection,
    prefix: &str,
    terms: &mut HashMap<String, DomainTerm>,
) -> Result<(), OntologyError> {
    // Query for is_a (subClassOf) relations
    let mut stmt = conn
        .prepare(
            "SELECT subject, object FROM edge
             WHERE predicate = 'rdfs:subClassOf'
             AND subject LIKE ? || ':%'",
        )
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    let rows = stmt
        .query_map([prefix], |row| {
            let subject: String = row.get(0)?;
            let object: String = row.get(1)?;
            Ok((subject, object))
        })
        .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

    for row in rows {
        let (subject, object) = row.map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        if let Some(term) = terms.get_mut(&subject) {
            term.parents.push(object);
        }
    }

    Ok(())
}

#[cfg(feature = "ontology")]
fn load_ancestors(
    conn: &rusqlite::Connection,
    prefix: &str,
    ancestors: &mut HashMap<String, Vec<String>>,
) -> Result<(), OntologyError> {
    // Check if entailed_edge table exists (transitive closure)
    let has_entailed: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='entailed_edge'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if has_entailed {
        let mut stmt = conn
            .prepare(
                "SELECT subject, object FROM entailed_edge
                 WHERE predicate = 'rdfs:subClassOf'
                 AND subject LIKE ? || ':%'",
            )
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        let rows = stmt
            .query_map([prefix], |row| {
                let subject: String = row.get(0)?;
                let object: String = row.get(1)?;
                Ok((subject, object))
            })
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        for row in rows {
            let (subject, object) = row.map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
            ancestors
                .entry(subject)
                .or_insert_with(Vec::new)
                .push(object);
        }
    }

    Ok(())
}

/// Create default epistemic status for a domain term
fn default_epistemic(term_id: &str, ontology: &str) -> EpistemicStatus {
    EpistemicStatus {
        confidence: Confidence::new(0.90), // Domain ontologies are expert-curated
        revisability: Revisability::Revisable {
            conditions: vec!["ontology_update".into(), "domain_evidence".into()],
        },
        source: Source::OntologyAssertion {
            ontology: format!(
                "http://purl.obolibrary.org/obo/{}.owl",
                ontology.to_lowercase()
            ),
            term: term_id.to_string(),
        },
        evidence: vec![Evidence {
            kind: EvidenceKind::Publication { doi: None },
            reference: format!("OBO Foundry: {}", ontology),
            strength: Confidence::new(0.90),
        }],
    }
}

/// Bootstrap loader for when SQLite is not available
#[cfg(not(feature = "ontology"))]
pub(crate) fn load_ontology_from_sqlite(
    _metadata: &OntologyMetadata,
) -> Result<DomainIndex, OntologyError> {
    // Return empty index in bootstrap mode
    Ok(DomainIndex {
        terms: HashMap::new(),
        ancestors: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_epistemic() {
        let status = default_epistemic("CHEBI:15365", "CHEBI");
        assert!(status.confidence.value() >= 0.9);
    }

    #[test]
    #[cfg(not(feature = "ontology"))]
    fn test_bootstrap_loader() {
        let metadata = OntologyMetadata {
            prefix: "TEST".into(),
            name: "Test".into(),
            description: "Test".into(),
            version: "1.0".into(),
            term_count: 0,
            db_file: std::path::PathBuf::from("/nonexistent"),
            available: false,
        };

        let index = load_ontology_from_sqlite(&metadata).unwrap();
        assert!(index.terms.is_empty());
    }
}
