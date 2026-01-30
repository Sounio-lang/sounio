//! BioPortal LOOM (Lexical OWL Ontology Matcher) Integration
//!
//! LOOM is BioPortal's automated ontology alignment service that generates
//! mappings between ontologies based on lexical similarity and structural
//! analysis.
//!
//! # Features
//!
//! - Lexical matching (label similarity)
//! - Structural matching (hierarchy alignment)
//! - CUI-based matching (UMLS integration)
//! - LOOM mappings available via BioPortal API
//!
//! # API Endpoints
//!
//! - GET /ontologies/{ont}/mappings - All mappings for an ontology
//! - GET /ontologies/{ont}/classes/{cls}/mappings - Mappings for a class
//!
//! # Example Response
//!
//! ```json
//! {
//!   "classes": [
//!     {
//!       "@id": "http://purl.obolibrary.org/obo/MONDO_0005148",
//!       "prefLabel": "type 2 diabetes mellitus",
//!       "links": { "mappings": "..." }
//!     }
//!   ],
//!   "mappings": [
//!     {
//!       "source": "MONDO",
//!       "sourceClassId": "http://purl.obolibrary.org/obo/MONDO_0005148",
//!       "targetOntology": "DOID",
//!       "targetClassId": "http://purl.obolibrary.org/obo/DOID_9352",
//!       "relation": "http://www.w3.org/2004/02/skos/core#exactMatch",
//!       "confidence": 0.95
//!     }
//!   ]
//! }
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::ontology::distance::sssom::MappingPredicate;
use crate::ontology::loader::IRI;

/// BioPortal API base URL
pub const BIOPORTAL_API_BASE: &str = "https://data.bioontology.org";

/// A LOOM mapping from BioPortal
#[derive(Debug, Clone)]
pub struct LOOMMapping {
    /// Source ontology acronym
    pub source_ontology: String,
    /// Source class IRI
    pub source_class: IRI,
    /// Source class label
    pub source_label: Option<String>,
    /// Target ontology acronym
    pub target_ontology: String,
    /// Target class IRI
    pub target_class: IRI,
    /// Target class label
    pub target_label: Option<String>,
    /// Mapping relation (predicate)
    pub relation: MappingPredicate,
    /// Confidence score
    pub confidence: f64,
    /// Mapping source (LOOM, CUI, SAME_URI, etc.)
    pub mapping_source: LOOMMappingSource,
}

/// Source of a LOOM mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LOOMMappingSource {
    /// Lexical matching (label similarity)
    LOOM,
    /// UMLS CUI matching
    CUI,
    /// Same URI across ontologies
    SameURI,
    /// Manual curation
    Manual,
    /// Unknown source
    Unknown,
}

impl LOOMMappingSource {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "LOOM" => Self::LOOM,
            "CUI" => Self::CUI,
            "SAME_URI" | "SAMEURI" => Self::SameURI,
            "MANUAL" | "CURATED" => Self::Manual,
            _ => Self::Unknown,
        }
    }

    pub fn reliability(&self) -> f64 {
        match self {
            Self::Manual => 1.0,
            Self::CUI => 0.90,
            Self::SameURI => 0.95,
            Self::LOOM => 0.75,
            Self::Unknown => 0.50,
        }
    }
}

/// LOOM client for fetching and caching BioPortal mappings
pub struct LOOMClient {
    /// API key for BioPortal
    api_key: Option<String>,
    /// Cache directory
    cache_dir: Option<std::path::PathBuf>,
    /// In-memory cache of mappings by ontology
    cached_mappings: HashMap<String, Vec<LOOMMapping>>,
    /// Mappings indexed by source IRI
    by_source: HashMap<IRI, Vec<LOOMMapping>>,
    /// Mappings indexed by target IRI
    by_target: HashMap<IRI, Vec<LOOMMapping>>,
}

impl LOOMClient {
    /// Create a new LOOM client
    pub fn new() -> Self {
        Self {
            api_key: None,
            cache_dir: None,
            cached_mappings: HashMap::new(),
            by_source: HashMap::new(),
            by_target: HashMap::new(),
        }
    }

    /// Set API key for BioPortal
    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = Some(key);
        self
    }

    /// Set cache directory
    pub fn with_cache_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Load mappings from a cached JSON file
    pub fn load_cache(&mut self, path: &Path) -> Result<usize, LOOMError> {
        let file = File::open(path).map_err(LOOMError::IoError)?;
        let reader = BufReader::new(file);
        let mut loaded = 0;

        // Parse JSON (simplified - in production use serde_json)
        for line in reader.lines() {
            let line = line.map_err(LOOMError::IoError)?;
            if let Some(mapping) = self.parse_json_mapping(&line) {
                self.add_mapping(mapping);
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Load mappings from a TSV file
    /// Format: source_ont\tsource_iri\ttarget_ont\ttarget_iri\trelation\tconfidence\tsource
    pub fn load_tsv(&mut self, path: &Path) -> Result<usize, LOOMError> {
        let file = File::open(path).map_err(LOOMError::IoError)?;
        let reader = BufReader::new(file);
        let mut loaded = 0;
        let mut is_header = true;

        for line in reader.lines() {
            let line = line.map_err(LOOMError::IoError)?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if is_header {
                is_header = false;
                continue;
            }

            if let Some(mapping) = self.parse_tsv_line(line) {
                self.add_mapping(mapping);
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Parse a TSV line into a mapping
    fn parse_tsv_line(&self, line: &str) -> Option<LOOMMapping> {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 6 {
            return None;
        }

        Some(LOOMMapping {
            source_ontology: parts[0].to_string(),
            source_class: IRI::new(parts[1]),
            source_label: None,
            target_ontology: parts[2].to_string(),
            target_class: IRI::new(parts[3]),
            target_label: None,
            relation: MappingPredicate::from_str(parts[4]),
            confidence: parts[5].parse().unwrap_or(0.5),
            mapping_source: parts
                .get(6)
                .map(|s| LOOMMappingSource::from_str(s))
                .unwrap_or(LOOMMappingSource::LOOM),
        })
    }

    /// Parse a JSON mapping (simplified)
    fn parse_json_mapping(&self, _json: &str) -> Option<LOOMMapping> {
        // In production, use serde_json
        // This is a placeholder
        None
    }

    /// Add a mapping to the index
    pub fn add_mapping(&mut self, mapping: LOOMMapping) {
        self.by_source
            .entry(mapping.source_class.clone())
            .or_default()
            .push(mapping.clone());

        self.by_target
            .entry(mapping.target_class.clone())
            .or_default()
            .push(mapping.clone());

        self.cached_mappings
            .entry(mapping.source_ontology.clone())
            .or_default()
            .push(mapping);
    }

    /// Get mappings from a source IRI
    pub fn get_mappings_from(&self, source: &IRI) -> &[LOOMMapping] {
        self.by_source
            .get(source)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get mappings to a target IRI
    pub fn get_mappings_to(&self, target: &IRI) -> &[LOOMMapping] {
        self.by_target
            .get(target)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all mappings for an IRI (as source or target)
    pub fn get_all_mappings(&self, iri: &IRI) -> Vec<&LOOMMapping> {
        let mut result = Vec::new();
        result.extend(self.get_mappings_from(iri));
        result.extend(self.get_mappings_to(iri));
        result
    }

    /// Find best mapping between two IRIs
    pub fn find_mapping(&self, source: &IRI, target: &IRI) -> Option<&LOOMMapping> {
        // Check direct mapping
        for mapping in self.get_mappings_from(source) {
            if &mapping.target_class == target {
                return Some(mapping);
            }
        }

        // Check reverse mapping
        self.get_mappings_to(source)
            .iter()
            .find(|&mapping| &mapping.source_class == target)
            .map(|v| v as _)
    }

    /// Compute distance between two IRIs via LOOM mappings
    pub fn loom_distance(&self, a: &IRI, b: &IRI) -> Option<(f64, f64)> {
        if let Some(mapping) = self.find_mapping(a, b) {
            let predicate_weight = mapping.relation.semantic_weight();
            let source_reliability = mapping.mapping_source.reliability();
            let effective_confidence = mapping.confidence * source_reliability;
            let distance = 1.0 - (predicate_weight * effective_confidence);
            return Some((distance, effective_confidence));
        }
        None
    }

    /// Get mappings for an ontology
    pub fn get_ontology_mappings(&self, ontology: &str) -> &[LOOMMapping] {
        self.cached_mappings
            .get(ontology)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get statistics
    pub fn stats(&self) -> LOOMStats {
        let mut ontology_pairs: HashMap<(String, String), usize> = HashMap::new();
        let mut source_counts: HashMap<LOOMMappingSource, usize> = HashMap::new();
        let mut confidence_sum = 0.0;
        let mut total = 0;

        for mappings in self.by_source.values() {
            for m in mappings {
                let pair = if m.source_ontology <= m.target_ontology {
                    (m.source_ontology.clone(), m.target_ontology.clone())
                } else {
                    (m.target_ontology.clone(), m.source_ontology.clone())
                };
                *ontology_pairs.entry(pair).or_insert(0) += 1;
                *source_counts.entry(m.mapping_source).or_insert(0) += 1;
                confidence_sum += m.confidence;
                total += 1;
            }
        }

        LOOMStats {
            total_mappings: total,
            unique_sources: self.by_source.len(),
            unique_targets: self.by_target.len(),
            ontology_pairs: ontology_pairs.len(),
            source_counts,
            avg_confidence: if total > 0 {
                confidence_sum / total as f64
            } else {
                0.0
            },
        }
    }

    /// Total number of mappings
    pub fn len(&self) -> usize {
        self.by_source.values().map(|v| v.len()).sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.by_source.is_empty()
    }

    /// Export mappings to TSV
    pub fn export_tsv(&self, path: &Path) -> Result<(), LOOMError> {
        let mut file = File::create(path).map_err(LOOMError::IoError)?;

        // Header
        writeln!(
            file,
            "source_ontology\tsource_class\ttarget_ontology\ttarget_class\trelation\tconfidence\tmapping_source"
        )
        .map_err(LOOMError::IoError)?;

        // Data rows
        for mappings in self.by_source.values() {
            for m in mappings {
                writeln!(
                    file,
                    "{}\t{}\t{}\t{}\t{:?}\t{}\t{:?}",
                    m.source_ontology,
                    m.source_class,
                    m.target_ontology,
                    m.target_class,
                    m.relation,
                    m.confidence,
                    m.mapping_source,
                )
                .map_err(LOOMError::IoError)?;
            }
        }

        Ok(())
    }
}

impl Default for LOOMClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for LOOM mappings
#[derive(Debug, Clone)]
pub struct LOOMStats {
    pub total_mappings: usize,
    pub unique_sources: usize,
    pub unique_targets: usize,
    pub ontology_pairs: usize,
    pub source_counts: HashMap<LOOMMappingSource, usize>,
    pub avg_confidence: f64,
}

/// LOOM error types
#[derive(Debug)]
pub enum LOOMError {
    IoError(std::io::Error),
    ApiError(String),
    ParseError(String),
    RateLimited,
    Unauthorized,
}

impl std::fmt::Display for LOOMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ApiError(msg) => write!(f, "API error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::RateLimited => write!(f, "Rate limited by BioPortal API"),
            Self::Unauthorized => write!(f, "Unauthorized - check API key"),
        }
    }
}

impl std::error::Error for LOOMError {}

/// Builder for fetching LOOM mappings
pub struct LOOMFetcher {
    api_key: String,
    ontologies: Vec<String>,
    cache_dir: std::path::PathBuf,
}

impl LOOMFetcher {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            ontologies: Vec::new(),
            cache_dir: std::path::PathBuf::from(".loom_cache"),
        }
    }

    /// Add ontology to fetch
    pub fn ontology(mut self, ont: &str) -> Self {
        self.ontologies.push(ont.to_string());
        self
    }

    /// Add multiple ontologies
    pub fn ontologies(mut self, onts: &[&str]) -> Self {
        self.ontologies.extend(onts.iter().map(|s| s.to_string()));
        self
    }

    /// Set cache directory
    pub fn cache_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    /// Build URL for ontology mappings
    pub fn build_url(&self, ontology: &str) -> String {
        format!(
            "{}/ontologies/{}/mappings?apikey={}",
            BIOPORTAL_API_BASE, ontology, self.api_key
        )
    }

    /// Get cache file path for an ontology
    pub fn cache_path(&self, ontology: &str) -> std::path::PathBuf {
        self.cache_dir.join(format!("{}_mappings.tsv", ontology))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loom_mapping_source() {
        assert_eq!(LOOMMappingSource::from_str("LOOM"), LOOMMappingSource::LOOM);
        assert_eq!(LOOMMappingSource::from_str("CUI"), LOOMMappingSource::CUI);
        assert!(LOOMMappingSource::CUI.reliability() > LOOMMappingSource::LOOM.reliability());
    }

    #[test]
    fn test_loom_client_basic() {
        let mut client = LOOMClient::new();

        let mapping = LOOMMapping {
            source_ontology: "MONDO".to_string(),
            source_class: IRI::from_curie("MONDO", "0005148"),
            source_label: Some("type 2 diabetes mellitus".to_string()),
            target_ontology: "DOID".to_string(),
            target_class: IRI::from_curie("DOID", "9352"),
            target_label: Some("type 2 diabetes mellitus".to_string()),
            relation: MappingPredicate::ExactMatch,
            confidence: 0.95,
            mapping_source: LOOMMappingSource::LOOM,
        };

        client.add_mapping(mapping);

        let mondo = IRI::from_curie("MONDO", "0005148");
        let doid = IRI::from_curie("DOID", "9352");

        assert!(client.find_mapping(&mondo, &doid).is_some());

        let result = client.loom_distance(&mondo, &doid);
        assert!(result.is_some());
    }

    #[test]
    fn test_loom_stats() {
        let mut client = LOOMClient::new();

        client.add_mapping(LOOMMapping {
            source_ontology: "CHEBI".to_string(),
            source_class: IRI::from_curie("CHEBI", "15365"),
            source_label: None,
            target_ontology: "DrugBank".to_string(),
            target_class: IRI::from_curie("DrugBank", "DB00945"),
            target_label: None,
            relation: MappingPredicate::ExactMatch,
            confidence: 0.90,
            mapping_source: LOOMMappingSource::CUI,
        });

        let stats = client.stats();
        assert_eq!(stats.total_mappings, 1);
        assert_eq!(stats.ontology_pairs, 1);
    }
}
