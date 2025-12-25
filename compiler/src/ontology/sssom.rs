//! SSSOM - Simple Standard for Sharing Ontology Mappings
//!
//! This module implements support for SSSOM (Simple Standard for Sharing
//! Ontology Mappings), a TSV-based format for ontology alignment files.
//!
//! SSSOM enables translation between ontology terms from different sources.
//!
//! # Format
//!
//! SSSOM files are TSV with a YAML metadata header:
//!
//! ```text
//! #curie_map:
//! #  CHEBI: http://purl.obolibrary.org/obo/CHEBI_
//! #  DRUGBANK: http://identifiers.org/drugbank/
//! subject_id	predicate_id	object_id	mapping_justification
//! CHEBI:15365	skos:exactMatch	DRUGBANK:DB00945	semapv:ManualMappingCuration
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::sssom::{load_sssom_mappings, SssomMappingSet};
//!
//! let mappings = load_sssom_mappings("path/to/mappings.sssom.tsv")?;
//!
//! // Find mappings from ChEBI to DrugBank
//! let translations = mappings.find_mappings("CHEBI:15365", Some("DRUGBANK"));
//! ```
//!
//! # References
//!
//! - SSSOM specification: https://mapping-commons.github.io/sssom/
//! - OAK SSSOM support: https://incatools.github.io/ontology-access-kit/

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{OntologyError, OntologyResult};

/// A set of SSSOM mappings with metadata
#[derive(Debug, Clone)]
pub struct SssomMappingSet {
    /// CURIE prefix map (prefix -> IRI namespace)
    pub curie_map: HashMap<String, String>,
    /// Mapping set ID
    pub mapping_set_id: Option<String>,
    /// Mapping set version
    pub mapping_set_version: Option<String>,
    /// Creator(s) of this mapping set
    pub creator_id: Vec<String>,
    /// License for the mappings
    pub license: Option<String>,
    /// All mappings in this set
    pub mappings: Vec<SssomMapping>,
    /// Index: subject_id -> mapping indices
    subject_index: HashMap<String, Vec<usize>>,
    /// Index: object_id -> mapping indices
    object_index: HashMap<String, Vec<usize>>,
}

impl SssomMappingSet {
    /// Create an empty mapping set
    pub fn new() -> Self {
        Self {
            curie_map: HashMap::new(),
            mapping_set_id: None,
            mapping_set_version: None,
            creator_id: vec![],
            license: None,
            mappings: vec![],
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
        }
    }

    /// Add a mapping to the set
    pub fn add_mapping(&mut self, mapping: SssomMapping) {
        let index = self.mappings.len();

        // Update indices
        self.subject_index
            .entry(mapping.subject_id.clone())
            .or_default()
            .push(index);
        self.object_index
            .entry(mapping.object_id.clone())
            .or_default()
            .push(index);

        self.mappings.push(mapping);
    }

    /// Rebuild indices (call after bulk loading)
    pub fn rebuild_indices(&mut self) {
        self.subject_index.clear();
        self.object_index.clear();

        for (index, mapping) in self.mappings.iter().enumerate() {
            self.subject_index
                .entry(mapping.subject_id.clone())
                .or_default()
                .push(index);
            self.object_index
                .entry(mapping.object_id.clone())
                .or_default()
                .push(index);
        }
    }

    /// Find mappings for a subject term
    pub fn find_mappings_from(&self, subject_id: &str) -> Vec<&SssomMapping> {
        self.subject_index
            .get(subject_id)
            .map(|indices| indices.iter().map(|&i| &self.mappings[i]).collect())
            .unwrap_or_default()
    }

    /// Find mappings to an object term
    pub fn find_mappings_to(&self, object_id: &str) -> Vec<&SssomMapping> {
        self.object_index
            .get(object_id)
            .map(|indices| indices.iter().map(|&i| &self.mappings[i]).collect())
            .unwrap_or_default()
    }

    /// Find mappings from a subject to a specific target ontology
    pub fn find_mappings(
        &self,
        subject_id: &str,
        target_prefix: Option<&str>,
    ) -> Vec<&SssomMapping> {
        let mappings = self.find_mappings_from(subject_id);

        if let Some(prefix) = target_prefix {
            mappings
                .into_iter()
                .filter(|m| m.object_id.starts_with(prefix))
                .collect()
        } else {
            mappings
        }
    }

    /// Find best mapping (highest confidence exact match)
    pub fn find_best_mapping(
        &self,
        subject_id: &str,
        target_prefix: Option<&str>,
    ) -> Option<&SssomMapping> {
        let mappings = self.find_mappings(subject_id, target_prefix);

        // Prefer exact matches, then close matches
        mappings
            .iter()
            .filter(|m| m.predicate == MappingPredicate::ExactMatch)
            .max_by(|a, b| {
                a.confidence
                    .unwrap_or(0.0)
                    .partial_cmp(&b.confidence.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| {
                mappings
                    .iter()
                    .filter(|m| m.predicate == MappingPredicate::CloseMatch)
                    .max_by(|a, b| {
                        a.confidence
                            .unwrap_or(0.0)
                            .partial_cmp(&b.confidence.unwrap_or(0.0))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .copied()
    }

    /// Get total number of mappings
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Check if mapping set is empty
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Merge another mapping set into this one
    pub fn merge(&mut self, other: SssomMappingSet) {
        // Merge CURIE maps
        self.curie_map.extend(other.curie_map);

        // Add all mappings
        for mapping in other.mappings {
            self.add_mapping(mapping);
        }
    }

    /// Filter mappings by predicate type
    pub fn filter_by_predicate(&self, predicate: MappingPredicate) -> Vec<&SssomMapping> {
        self.mappings
            .iter()
            .filter(|m| m.predicate == predicate)
            .collect()
    }

    /// Get all unique source prefixes
    pub fn source_prefixes(&self) -> Vec<String> {
        let mut prefixes: Vec<String> = self
            .mappings
            .iter()
            .filter_map(|m| m.subject_id.split(':').next())
            .map(String::from)
            .collect();
        prefixes.sort();
        prefixes.dedup();
        prefixes
    }

    /// Get all unique target prefixes
    pub fn target_prefixes(&self) -> Vec<String> {
        let mut prefixes: Vec<String> = self
            .mappings
            .iter()
            .filter_map(|m| m.object_id.split(':').next())
            .map(String::from)
            .collect();
        prefixes.sort();
        prefixes.dedup();
        prefixes
    }
}

impl Default for SssomMappingSet {
    fn default() -> Self {
        Self::new()
    }
}

/// A single SSSOM mapping
#[derive(Debug, Clone)]
pub struct SssomMapping {
    /// Subject term CURIE (e.g., "CHEBI:15365")
    pub subject_id: String,
    /// Subject label (optional)
    pub subject_label: Option<String>,
    /// Subject source ontology
    pub subject_source: Option<String>,

    /// Mapping predicate (exact match, close match, etc.)
    pub predicate: MappingPredicate,

    /// Object term CURIE (e.g., "DRUGBANK:DB00945")
    pub object_id: String,
    /// Object label (optional)
    pub object_label: Option<String>,
    /// Object source ontology
    pub object_source: Option<String>,

    /// Mapping justification
    pub mapping_justification: MappingJustification,

    /// Confidence score (0.0 - 1.0)
    pub confidence: Option<f64>,

    /// Semantic similarity score
    pub semantic_similarity_score: Option<f64>,

    /// Comment or notes
    pub comment: Option<String>,

    /// Mapping provider
    pub mapping_provider: Option<String>,

    /// Author of this specific mapping
    pub author_id: Vec<String>,
}

impl SssomMapping {
    /// Create a new mapping
    pub fn new(
        subject_id: impl Into<String>,
        predicate: MappingPredicate,
        object_id: impl Into<String>,
        justification: MappingJustification,
    ) -> Self {
        Self {
            subject_id: subject_id.into(),
            subject_label: None,
            subject_source: None,
            predicate,
            object_id: object_id.into(),
            object_label: None,
            object_source: None,
            mapping_justification: justification,
            confidence: None,
            semantic_similarity_score: None,
            comment: None,
            mapping_provider: None,
            author_id: vec![],
        }
    }

    /// Create an exact match mapping
    pub fn exact_match(
        subject: impl Into<String>,
        object: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            subject_id: subject.into(),
            subject_label: None,
            subject_source: None,
            predicate: MappingPredicate::ExactMatch,
            object_id: object.into(),
            object_label: None,
            object_source: None,
            mapping_justification: MappingJustification::ManualMappingCuration,
            confidence: Some(confidence),
            semantic_similarity_score: None,
            comment: None,
            mapping_provider: None,
            author_id: vec![],
        }
    }

    /// Create a close match mapping
    pub fn close_match(
        subject: impl Into<String>,
        object: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            subject_id: subject.into(),
            subject_label: None,
            subject_source: None,
            predicate: MappingPredicate::CloseMatch,
            object_id: object.into(),
            object_label: None,
            object_source: None,
            mapping_justification: MappingJustification::ManualMappingCuration,
            confidence: Some(confidence),
            semantic_similarity_score: None,
            comment: None,
            mapping_provider: None,
            author_id: vec![],
        }
    }

    /// Get the confidence, defaulting to predicate-based estimate
    pub fn effective_confidence(&self) -> f64 {
        self.confidence
            .unwrap_or_else(|| self.predicate.default_confidence())
    }

    /// Check if this is a bidirectional mapping
    pub fn is_bidirectional(&self) -> bool {
        matches!(
            self.predicate,
            MappingPredicate::ExactMatch | MappingPredicate::CloseMatch
        )
    }

    /// Get the reverse mapping (swap subject and object)
    pub fn reverse(&self) -> Self {
        Self {
            subject_id: self.object_id.clone(),
            subject_label: self.object_label.clone(),
            subject_source: self.object_source.clone(),
            predicate: self.predicate.reverse(),
            object_id: self.subject_id.clone(),
            object_label: self.subject_label.clone(),
            object_source: self.subject_source.clone(),
            mapping_justification: self.mapping_justification.clone(),
            confidence: self.confidence,
            semantic_similarity_score: self.semantic_similarity_score,
            comment: self.comment.clone(),
            mapping_provider: self.mapping_provider.clone(),
            author_id: self.author_id.clone(),
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set labels
    pub fn with_labels(
        mut self,
        subject_label: impl Into<String>,
        object_label: impl Into<String>,
    ) -> Self {
        self.subject_label = Some(subject_label.into());
        self.object_label = Some(object_label.into());
        self
    }
}

/// SKOS-based mapping predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MappingPredicate {
    /// skos:exactMatch - identical meaning
    ExactMatch,
    /// skos:closeMatch - similar meaning
    CloseMatch,
    /// skos:broadMatch - subject is narrower than object
    BroadMatch,
    /// skos:narrowMatch - subject is broader than object
    NarrowMatch,
    /// skos:relatedMatch - some relation exists
    RelatedMatch,
    /// owl:sameAs - identity in OWL sense
    SameAs,
    /// owl:equivalentClass - equivalent OWL classes
    EquivalentClass,
    /// rdfs:subClassOf - subject is subclass of object
    SubClassOf,
    /// rdfs:superClassOf - subject is superclass of object
    SuperClassOf,
    /// Custom/unknown predicate
    Other(u32),
}

impl MappingPredicate {
    /// Parse from CURIE or IRI
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "skos:exactmatch" | "exactmatch" => MappingPredicate::ExactMatch,
            "skos:closematch" | "closematch" => MappingPredicate::CloseMatch,
            "skos:broadmatch" | "broadmatch" => MappingPredicate::BroadMatch,
            "skos:narrowmatch" | "narrowmatch" => MappingPredicate::NarrowMatch,
            "skos:relatedmatch" | "relatedmatch" => MappingPredicate::RelatedMatch,
            "owl:sameas" | "sameas" => MappingPredicate::SameAs,
            "owl:equivalentclass" | "equivalentclass" => MappingPredicate::EquivalentClass,
            "rdfs:subclassof" | "subclassof" => MappingPredicate::SubClassOf,
            _ => MappingPredicate::Other(0),
        }
    }

    /// Get the default confidence for this predicate type
    pub fn default_confidence(&self) -> f64 {
        match self {
            MappingPredicate::ExactMatch | MappingPredicate::SameAs => 1.0,
            MappingPredicate::EquivalentClass => 0.99,
            MappingPredicate::CloseMatch => 0.9,
            MappingPredicate::BroadMatch | MappingPredicate::NarrowMatch => 0.8,
            MappingPredicate::SubClassOf | MappingPredicate::SuperClassOf => 0.85,
            MappingPredicate::RelatedMatch => 0.7,
            MappingPredicate::Other(_) => 0.5,
        }
    }

    /// Get the CURIE representation
    pub fn curie(&self) -> &'static str {
        match self {
            MappingPredicate::ExactMatch => "skos:exactMatch",
            MappingPredicate::CloseMatch => "skos:closeMatch",
            MappingPredicate::BroadMatch => "skos:broadMatch",
            MappingPredicate::NarrowMatch => "skos:narrowMatch",
            MappingPredicate::RelatedMatch => "skos:relatedMatch",
            MappingPredicate::SameAs => "owl:sameAs",
            MappingPredicate::EquivalentClass => "owl:equivalentClass",
            MappingPredicate::SubClassOf => "rdfs:subClassOf",
            MappingPredicate::SuperClassOf => "rdfs:superClassOf",
            MappingPredicate::Other(_) => "sssom:unknown",
        }
    }

    /// Get the reverse predicate
    pub fn reverse(&self) -> Self {
        match self {
            MappingPredicate::BroadMatch => MappingPredicate::NarrowMatch,
            MappingPredicate::NarrowMatch => MappingPredicate::BroadMatch,
            MappingPredicate::SubClassOf => MappingPredicate::SuperClassOf,
            MappingPredicate::SuperClassOf => MappingPredicate::SubClassOf,
            // Symmetric predicates
            other => *other,
        }
    }
}

/// Justification for why a mapping was created
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MappingJustification {
    /// Manual curation by human expert
    ManualMappingCuration,
    /// Lexical matching (label similarity)
    LexicalMatching,
    /// Logical reasoning/inference
    LogicalReasoning,
    /// Semantic similarity matching
    SemanticSimilarityMatching,
    /// Cross-reference extraction
    MappingFromXref,
    /// Composite/combined methods
    CompositeMatching,
    /// Unknown or unspecified
    Unknown(String),
}

impl MappingJustification {
    /// Parse from CURIE or IRI
    pub fn parse(s: &str) -> Self {
        let s_lower = s.to_lowercase();
        if s_lower.contains("manualmappingcuration") || s_lower.contains("manual") {
            MappingJustification::ManualMappingCuration
        } else if s_lower.contains("lexical") {
            MappingJustification::LexicalMatching
        } else if s_lower.contains("logical") {
            MappingJustification::LogicalReasoning
        } else if s_lower.contains("semantic") {
            MappingJustification::SemanticSimilarityMatching
        } else if s_lower.contains("xref") {
            MappingJustification::MappingFromXref
        } else if s_lower.contains("composite") {
            MappingJustification::CompositeMatching
        } else {
            MappingJustification::Unknown(s.to_string())
        }
    }

    /// Get the CURIE representation
    pub fn curie(&self) -> String {
        match self {
            MappingJustification::ManualMappingCuration => {
                "semapv:ManualMappingCuration".to_string()
            }
            MappingJustification::LexicalMatching => "semapv:LexicalMatching".to_string(),
            MappingJustification::LogicalReasoning => "semapv:LogicalReasoning".to_string(),
            MappingJustification::SemanticSimilarityMatching => {
                "semapv:SemanticSimilarityMatching".to_string()
            }
            MappingJustification::MappingFromXref => "semapv:MappingFromXref".to_string(),
            MappingJustification::CompositeMatching => "semapv:CompositeMatching".to_string(),
            MappingJustification::Unknown(s) => s.clone(),
        }
    }
}

/// Direction of mapping lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingDirection {
    /// Forward: subject -> object
    Forward,
    /// Reverse: object -> subject
    Reverse,
    /// Both directions
    Bidirectional,
}

/// Load SSSOM mappings from a file
pub fn load_sssom_mappings(path: impl AsRef<Path>) -> OntologyResult<SssomMappingSet> {
    let file = File::open(path.as_ref())
        .map_err(|e| OntologyError::SssomParseError(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);

    let mut mapping_set = SssomMappingSet::new();
    let mut header_columns: Vec<String> = vec![];

    for line in reader.lines() {
        let line =
            line.map_err(|e| OntologyError::SssomParseError(format!("Read error: {}", e)))?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Parse metadata comments
        if line.starts_with('#') {
            let content = line.trim_start_matches('#').trim();

            // Parse YAML-like metadata
            if content.starts_with("curie_map:") {
                // Multi-line CURIE map handled below
            } else if let Some((key, value)) = content.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                match key {
                    "mapping_set_id" => mapping_set.mapping_set_id = Some(value.to_string()),
                    "mapping_set_version" => {
                        mapping_set.mapping_set_version = Some(value.to_string())
                    }
                    "license" => mapping_set.license = Some(value.to_string()),
                    _ => {
                        // Could be CURIE map entry
                        if !value.is_empty() && value.starts_with("http") {
                            mapping_set
                                .curie_map
                                .insert(key.to_uppercase(), value.to_string());
                        }
                    }
                }
            }
            continue;
        }

        // First non-comment line is the header
        if header_columns.is_empty() {
            header_columns = line.split('\t').map(|s| s.to_string()).collect();
            continue;
        }

        // Parse data row
        let values: Vec<&str> = line.split('\t').collect();
        if values.len() < 3 {
            continue; // Skip malformed lines
        }

        // Map column names to values
        let row: HashMap<&str, &str> = header_columns
            .iter()
            .zip(values.iter())
            .map(|(k, v)| (k.as_str(), *v))
            .collect();

        // Extract required fields
        let subject_id = row.get("subject_id").copied().unwrap_or("");
        let predicate_id = row
            .get("predicate_id")
            .copied()
            .unwrap_or("skos:exactMatch");
        let object_id = row.get("object_id").copied().unwrap_or("");

        if subject_id.is_empty() || object_id.is_empty() {
            continue;
        }

        let mut mapping = SssomMapping::new(
            subject_id,
            MappingPredicate::parse(predicate_id),
            object_id,
            MappingJustification::parse(row.get("mapping_justification").copied().unwrap_or("")),
        );

        // Optional fields
        mapping.subject_label = row.get("subject_label").map(|s| s.to_string());
        mapping.object_label = row.get("object_label").map(|s| s.to_string());
        mapping.subject_source = row.get("subject_source").map(|s| s.to_string());
        mapping.object_source = row.get("object_source").map(|s| s.to_string());
        mapping.comment = row.get("comment").map(|s| s.to_string());
        mapping.mapping_provider = row.get("mapping_provider").map(|s| s.to_string());

        if let Some(conf) = row.get("confidence") {
            mapping.confidence = conf.parse().ok();
        }
        if let Some(sim) = row.get("semantic_similarity_score") {
            mapping.semantic_similarity_score = sim.parse().ok();
        }

        mapping_set.add_mapping(mapping);
    }

    Ok(mapping_set)
}

/// Load SSSOM mappings from a string (for testing)
pub fn parse_sssom_string(content: &str) -> OntologyResult<SssomMappingSet> {
    let mut mapping_set = SssomMappingSet::new();
    let mut header_columns: Vec<String> = vec![];

    for line in content.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        if header_columns.is_empty() {
            header_columns = line.split('\t').map(|s| s.to_string()).collect();
            continue;
        }

        let values: Vec<&str> = line.split('\t').collect();
        if values.len() < 3 {
            continue;
        }

        let row: HashMap<&str, &str> = header_columns
            .iter()
            .zip(values.iter())
            .map(|(k, v)| (k.as_str(), *v))
            .collect();

        let subject_id = row.get("subject_id").copied().unwrap_or("");
        let predicate_id = row
            .get("predicate_id")
            .copied()
            .unwrap_or("skos:exactMatch");
        let object_id = row.get("object_id").copied().unwrap_or("");

        if subject_id.is_empty() || object_id.is_empty() {
            continue;
        }

        let mut mapping = SssomMapping::new(
            subject_id,
            MappingPredicate::parse(predicate_id),
            object_id,
            MappingJustification::ManualMappingCuration,
        );

        if let Some(conf) = row.get("confidence") {
            mapping.confidence = conf.parse().ok();
        }

        mapping_set.add_mapping(mapping);
    }

    Ok(mapping_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_predicate_parse() {
        assert_eq!(
            MappingPredicate::parse("skos:exactMatch"),
            MappingPredicate::ExactMatch
        );
        assert_eq!(
            MappingPredicate::parse("SKOS:CLOSEMATCH"),
            MappingPredicate::CloseMatch
        );
        assert_eq!(
            MappingPredicate::parse("owl:sameAs"),
            MappingPredicate::SameAs
        );
    }

    #[test]
    fn test_mapping_predicate_reverse() {
        assert_eq!(
            MappingPredicate::BroadMatch.reverse(),
            MappingPredicate::NarrowMatch
        );
        assert_eq!(
            MappingPredicate::ExactMatch.reverse(),
            MappingPredicate::ExactMatch
        );
    }

    #[test]
    fn test_sssom_mapping_creation() {
        let mapping = SssomMapping::exact_match("CHEBI:15365", "DRUGBANK:DB00945", 0.99);
        assert_eq!(mapping.subject_id, "CHEBI:15365");
        assert_eq!(mapping.object_id, "DRUGBANK:DB00945");
        assert_eq!(mapping.predicate, MappingPredicate::ExactMatch);
        assert_eq!(mapping.confidence, Some(0.99));
    }

    #[test]
    fn test_sssom_mapping_reverse() {
        let mapping = SssomMapping::exact_match("CHEBI:15365", "DRUGBANK:DB00945", 0.99);
        let reversed = mapping.reverse();

        assert_eq!(reversed.subject_id, "DRUGBANK:DB00945");
        assert_eq!(reversed.object_id, "CHEBI:15365");
    }

    #[test]
    fn test_parse_sssom_string() {
        let content = r#"subject_id	predicate_id	object_id	confidence
CHEBI:15365	skos:exactMatch	DRUGBANK:DB00945	0.99
CHEBI:30762	skos:closeMatch	DRUGBANK:DB00537	0.85"#;

        let mapping_set = parse_sssom_string(content).unwrap();
        assert_eq!(mapping_set.len(), 2);

        let mappings = mapping_set.find_mappings_from("CHEBI:15365");
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].object_id, "DRUGBANK:DB00945");
    }

    #[test]
    fn test_find_best_mapping() {
        let content = r#"subject_id	predicate_id	object_id	confidence
CHEBI:15365	skos:exactMatch	DRUGBANK:DB00945	0.99
CHEBI:15365	skos:closeMatch	MESH:D001241	0.80"#;

        let mapping_set = parse_sssom_string(content).unwrap();

        // Should prefer exact match
        let best = mapping_set.find_best_mapping("CHEBI:15365", None);
        assert!(best.is_some());
        assert_eq!(best.unwrap().object_id, "DRUGBANK:DB00945");

        // Should find MESH mapping when filtered
        let mesh = mapping_set.find_best_mapping("CHEBI:15365", Some("MESH"));
        assert!(mesh.is_some());
        assert_eq!(mesh.unwrap().object_id, "MESH:D001241");
    }

    #[test]
    fn test_mapping_set_prefixes() {
        let content = r#"subject_id	predicate_id	object_id	confidence
CHEBI:15365	skos:exactMatch	DRUGBANK:DB00945	0.99
GO:0008150	skos:closeMatch	MESH:D001687	0.85"#;

        let mapping_set = parse_sssom_string(content).unwrap();

        let sources = mapping_set.source_prefixes();
        assert!(sources.contains(&"CHEBI".to_string()));
        assert!(sources.contains(&"GO".to_string()));

        let targets = mapping_set.target_prefixes();
        assert!(targets.contains(&"DRUGBANK".to_string()));
        assert!(targets.contains(&"MESH".to_string()));
    }
}
