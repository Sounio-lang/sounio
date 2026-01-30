// Sounio Compiler - SSSOM Cross-Ontology Mapping Integration
// Simple Standard for Sharing Ontology Mappings (SSSOM)
// https://mapping-commons.github.io/sssom/

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::ontology::loader::IRI;

/// Predicate types for ontology mappings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MappingPredicate {
    /// Exact match (owl:equivalentClass, skos:exactMatch)
    ExactMatch,

    /// Close match (skos:closeMatch)
    CloseMatch,

    /// Narrow match (skos:narrowMatch) - subject is narrower than object
    NarrowMatch,

    /// Broad match (skos:broadMatch) - subject is broader than object
    BroadMatch,

    /// Related match (skos:relatedMatch)
    RelatedMatch,

    /// Database cross-reference (oboInOwl:hasDbXref)
    DbXref,

    /// Custom predicate
    Custom,
}

impl MappingPredicate {
    /// Parse predicate from SSSOM string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "skos:exactmatch" | "owl:equivalentclass" | "owl:sameas" => Self::ExactMatch,
            "skos:closematch" => Self::CloseMatch,
            "skos:narrowmatch" => Self::NarrowMatch,
            "skos:broadmatch" => Self::BroadMatch,
            "skos:relatedmatch" => Self::RelatedMatch,
            "oboinowl:hasdbxref" | "obo:hasdatabasecrossreference" => Self::DbXref,
            _ => Self::Custom,
        }
    }

    /// Get semantic weight for mapping type
    /// Higher weight = more semantically equivalent
    pub fn semantic_weight(&self) -> f64 {
        match self {
            Self::ExactMatch => 1.0,
            Self::CloseMatch => 0.9,
            Self::NarrowMatch => 0.7,
            Self::BroadMatch => 0.7,
            Self::RelatedMatch => 0.5,
            Self::DbXref => 0.8,
            Self::Custom => 0.5,
        }
    }
}

/// A single SSSOM mapping entry
#[derive(Debug, Clone)]
pub struct SSSOMMapping {
    /// Subject entity (source)
    pub subject_id: IRI,

    /// Subject label (optional)
    pub subject_label: Option<String>,

    /// Object entity (target)
    pub object_id: IRI,

    /// Object label (optional)
    pub object_label: Option<String>,

    /// Mapping predicate (relationship type)
    pub predicate: MappingPredicate,

    /// Mapping justification (how mapping was derived)
    pub mapping_justification: Option<String>,

    /// Confidence score [0, 1]
    pub confidence: f64,

    /// Source of the mapping
    pub mapping_source: Option<String>,

    /// Creator of the mapping
    pub creator: Option<String>,

    /// Comment or notes
    pub comment: Option<String>,
}

impl SSSOMMapping {
    /// Create a new exact match mapping
    pub fn exact(subject: IRI, object: IRI, confidence: f64) -> Self {
        Self {
            subject_id: subject,
            subject_label: None,
            object_id: object,
            object_label: None,
            predicate: MappingPredicate::ExactMatch,
            mapping_justification: None,
            confidence,
            mapping_source: None,
            creator: None,
            comment: None,
        }
    }

    /// Create a new mapping with predicate
    pub fn with_predicate(
        subject: IRI,
        object: IRI,
        predicate: MappingPredicate,
        confidence: f64,
    ) -> Self {
        Self {
            subject_id: subject,
            subject_label: None,
            object_id: object,
            object_label: None,
            predicate,
            mapping_justification: None,
            confidence,
            mapping_source: None,
            creator: None,
            comment: None,
        }
    }

    /// Compute effective confidence (confidence * predicate weight)
    pub fn effective_confidence(&self) -> f64 {
        self.confidence * self.predicate.semantic_weight()
    }
}

/// SSSOM mapping set (collection of mappings with metadata)
#[derive(Debug, Clone)]
pub struct SSSOMSet {
    /// All mappings in the set
    pub mappings: Vec<SSSOMMapping>,

    /// Mapping set ID
    pub mapping_set_id: Option<String>,

    /// License for the mapping set
    pub license: Option<String>,

    /// Subject source ontology
    pub subject_source: Option<String>,

    /// Object source ontology
    pub object_source: Option<String>,

    /// Creation date
    pub created: Option<String>,
}

impl SSSOMSet {
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
            mapping_set_id: None,
            license: None,
            subject_source: None,
            object_source: None,
            created: None,
        }
    }

    /// Add a mapping to the set
    pub fn add(&mut self, mapping: SSSOMMapping) {
        self.mappings.push(mapping);
    }
}

impl Default for SSSOMSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Index for fast cross-ontology mapping lookup
pub struct SSSOMIndex {
    /// Mappings indexed by subject
    by_subject: HashMap<IRI, Vec<SSSOMMapping>>,

    /// Mappings indexed by object (for reverse lookup)
    by_object: HashMap<IRI, Vec<SSSOMMapping>>,

    /// Total number of mappings
    count: usize,

    /// Prefix mappings for CURIE expansion
    prefixes: HashMap<String, String>,
}

impl SSSOMIndex {
    /// Create empty index
    pub fn new() -> Self {
        Self {
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
            count: 0,
            prefixes: Self::default_prefixes(),
        }
    }

    /// Default prefix mappings
    fn default_prefixes() -> HashMap<String, String> {
        let mut prefixes = HashMap::new();
        prefixes.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "obo".to_string(),
            "http://purl.obolibrary.org/obo/".to_string(),
        );
        prefixes.insert(
            "CHEBI".to_string(),
            "http://purl.obolibrary.org/obo/CHEBI_".to_string(),
        );
        prefixes.insert(
            "GO".to_string(),
            "http://purl.obolibrary.org/obo/GO_".to_string(),
        );
        prefixes.insert(
            "HP".to_string(),
            "http://purl.obolibrary.org/obo/HP_".to_string(),
        );
        prefixes.insert(
            "MONDO".to_string(),
            "http://purl.obolibrary.org/obo/MONDO_".to_string(),
        );
        prefixes.insert(
            "DOID".to_string(),
            "http://purl.obolibrary.org/obo/DOID_".to_string(),
        );
        prefixes.insert(
            "NCBITaxon".to_string(),
            "http://purl.obolibrary.org/obo/NCBITaxon_".to_string(),
        );
        prefixes.insert(
            "UBERON".to_string(),
            "http://purl.obolibrary.org/obo/UBERON_".to_string(),
        );
        prefixes.insert(
            "CL".to_string(),
            "http://purl.obolibrary.org/obo/CL_".to_string(),
        );
        prefixes.insert("SNOMED".to_string(), "http://snomed.info/id/".to_string());
        prefixes.insert(
            "ICD10".to_string(),
            "http://hl7.org/fhir/sid/icd-10/".to_string(),
        );
        prefixes
    }

    /// Add a prefix mapping
    pub fn add_prefix(&mut self, prefix: String, expansion: String) {
        self.prefixes.insert(prefix, expansion);
    }

    /// Expand CURIE to IRI
    pub fn expand_curie(&self, curie: &str) -> Option<IRI> {
        if curie.starts_with("http://") || curie.starts_with("https://") {
            return Some(IRI::new(curie));
        }

        let parts: Vec<&str> = curie.splitn(2, ':').collect();
        if parts.len() != 2 {
            return None;
        }

        let prefix = parts[0];
        let local = parts[1];

        self.prefixes
            .get(prefix)
            .map(|expansion| IRI::new(&format!("{}{}", expansion, local)))
    }

    /// Index a mapping
    pub fn add(&mut self, mapping: SSSOMMapping) {
        self.by_subject
            .entry(mapping.subject_id.clone())
            .or_default()
            .push(mapping.clone());

        self.by_object
            .entry(mapping.object_id.clone())
            .or_default()
            .push(mapping);

        self.count += 1;
    }

    /// Index an entire mapping set
    pub fn add_set(&mut self, set: SSSOMSet) {
        for mapping in set.mappings {
            self.add(mapping);
        }
    }

    /// Get mappings where IRI is the subject
    pub fn get_mappings_from(&self, iri: &IRI) -> &[SSSOMMapping] {
        self.by_subject
            .get(iri)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get mappings where IRI is the object
    pub fn get_mappings_to(&self, iri: &IRI) -> &[SSSOMMapping] {
        self.by_object.get(iri).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all mappings for an IRI (as subject or object)
    pub fn get_all_mappings(&self, iri: &IRI) -> Vec<&SSSOMMapping> {
        let mut mappings = Vec::new();

        if let Some(from_mappings) = self.by_subject.get(iri) {
            mappings.extend(from_mappings.iter());
        }

        if let Some(to_mappings) = self.by_object.get(iri) {
            mappings.extend(to_mappings.iter());
        }

        mappings
    }

    /// Find equivalent terms across ontologies
    pub fn find_equivalents(&self, iri: &IRI) -> Vec<(IRI, f64)> {
        let mut equivalents = Vec::new();

        for mapping in self.get_mappings_from(iri) {
            if mapping.predicate == MappingPredicate::ExactMatch {
                equivalents.push((mapping.object_id.clone(), mapping.confidence));
            }
        }

        for mapping in self.get_mappings_to(iri) {
            if mapping.predicate == MappingPredicate::ExactMatch {
                equivalents.push((mapping.subject_id.clone(), mapping.confidence));
            }
        }

        // Deduplicate and take highest confidence
        let mut deduped: HashMap<IRI, f64> = HashMap::new();
        for (eq_iri, conf) in equivalents {
            let entry = deduped.entry(eq_iri).or_insert(0.0);
            *entry = entry.max(conf);
        }

        deduped.into_iter().collect()
    }

    /// Compute cross-ontology distance via mappings
    /// Returns (distance, confidence) where distance is 1.0 - predicate_weight
    pub fn cross_ontology_distance(&self, a: &IRI, b: &IRI) -> Option<(f64, f64)> {
        // Direct mapping?
        for mapping in self.get_mappings_from(a) {
            if &mapping.object_id == b {
                let distance = 1.0 - mapping.predicate.semantic_weight();
                return Some((distance, mapping.confidence));
            }
        }

        // Reverse mapping?
        for mapping in self.get_mappings_to(a) {
            if &mapping.subject_id == b {
                let distance = 1.0 - mapping.predicate.semantic_weight();
                return Some((distance, mapping.confidence));
            }
        }

        // Transitive via common equivalent?
        let a_equiv = self.find_equivalents(a);
        let b_equiv = self.find_equivalents(b);

        for (a_eq, a_conf) in &a_equiv {
            for (b_eq, b_conf) in &b_equiv {
                if a_eq == b_eq {
                    // Both map to same term - distance is sum of hop distances
                    let distance = 0.1; // Small penalty for indirection
                    let confidence = a_conf.min(*b_conf);
                    return Some((distance, confidence));
                }
            }
        }

        None
    }

    /// Total number of mappings
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get statistics
    pub fn stats(&self) -> SSSOMStats {
        let mut predicate_counts: HashMap<MappingPredicate, usize> = HashMap::new();
        let mut confidence_sum = 0.0;

        for mappings in self.by_subject.values() {
            for m in mappings {
                *predicate_counts.entry(m.predicate).or_insert(0) += 1;
                confidence_sum += m.confidence;
            }
        }

        SSSOMStats {
            total_mappings: self.count,
            unique_subjects: self.by_subject.len(),
            unique_objects: self.by_object.len(),
            predicate_counts,
            avg_confidence: if self.count > 0 {
                confidence_sum / self.count as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for SSSOMIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about SSSOM index
#[derive(Debug, Clone)]
pub struct SSSOMStats {
    pub total_mappings: usize,
    pub unique_subjects: usize,
    pub unique_objects: usize,
    pub predicate_counts: HashMap<MappingPredicate, usize>,
    pub avg_confidence: f64,
}

/// SSSOM TSV parser
pub struct SSSOMParser {
    /// Column indices
    col_subject_id: Option<usize>,
    col_object_id: Option<usize>,
    col_predicate: Option<usize>,
    col_confidence: Option<usize>,
    col_subject_label: Option<usize>,
    col_object_label: Option<usize>,
    col_mapping_justification: Option<usize>,
    col_mapping_source: Option<usize>,
    col_creator: Option<usize>,
    col_comment: Option<usize>,

    /// Prefix map from header
    prefixes: HashMap<String, String>,
}

impl SSSOMParser {
    pub fn new() -> Self {
        Self {
            col_subject_id: None,
            col_object_id: None,
            col_predicate: None,
            col_confidence: None,
            col_subject_label: None,
            col_object_label: None,
            col_mapping_justification: None,
            col_mapping_source: None,
            col_creator: None,
            col_comment: None,
            prefixes: HashMap::new(),
        }
    }

    /// Parse SSSOM TSV file
    pub fn parse_file(&mut self, path: &Path) -> Result<SSSOMSet, SSSOMParseError> {
        let file = File::open(path).map_err(SSSOMParseError::IoError)?;
        let reader = BufReader::new(file);

        let mut set = SSSOMSet::new();
        let mut in_header = true;
        let mut header_parsed = false;

        for line in reader.lines() {
            let line = line.map_err(SSSOMParseError::IoError)?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // YAML metadata header
            if line.starts_with('#') {
                if line.starts_with("#curie_map:") || line.starts_with("# curie_map:") {
                    // Parse prefix map from YAML
                    // Simplified handling
                }
                if line.starts_with("#mapping_set_id:") {
                    set.mapping_set_id = Some(
                        line.split(':')
                            .skip(1)
                            .collect::<Vec<_>>()
                            .join(":")
                            .trim()
                            .to_string(),
                    );
                }
                if line.starts_with("#license:") {
                    set.license = Some(
                        line.split(':')
                            .skip(1)
                            .collect::<Vec<_>>()
                            .join(":")
                            .trim()
                            .to_string(),
                    );
                }
                continue;
            }

            // Column headers
            if in_header && !header_parsed {
                self.parse_header(line);
                header_parsed = true;
                in_header = false;
                continue;
            }

            // Data row
            if let Some(mapping) = self.parse_row(line) {
                set.mappings.push(mapping);
            }
        }

        Ok(set)
    }

    /// Parse TSV header row
    fn parse_header(&mut self, line: &str) {
        let cols: Vec<&str> = line.split('\t').collect();

        for (i, col) in cols.iter().enumerate() {
            match col.to_lowercase().as_str() {
                "subject_id" => self.col_subject_id = Some(i),
                "object_id" => self.col_object_id = Some(i),
                "predicate_id" | "predicate" => self.col_predicate = Some(i),
                "confidence" => self.col_confidence = Some(i),
                "subject_label" => self.col_subject_label = Some(i),
                "object_label" => self.col_object_label = Some(i),
                "mapping_justification" => self.col_mapping_justification = Some(i),
                "mapping_source" => self.col_mapping_source = Some(i),
                "creator_id" | "creator" => self.col_creator = Some(i),
                "comment" => self.col_comment = Some(i),
                _ => {}
            }
        }
    }

    /// Parse a data row
    fn parse_row(&self, line: &str) -> Option<SSSOMMapping> {
        let cols: Vec<&str> = line.split('\t').collect();

        let subject_id = self
            .col_subject_id
            .and_then(|i| cols.get(i))
            .filter(|s| !s.is_empty())
            .map(|s| IRI::new(s))?;

        let object_id = self
            .col_object_id
            .and_then(|i| cols.get(i))
            .filter(|s| !s.is_empty())
            .map(|s| IRI::new(s))?;

        let predicate = self
            .col_predicate
            .and_then(|i| cols.get(i))
            .map(|s| MappingPredicate::from_str(s))
            .unwrap_or(MappingPredicate::RelatedMatch);

        let confidence = self
            .col_confidence
            .and_then(|i| cols.get(i))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        Some(SSSOMMapping {
            subject_id,
            subject_label: self
                .col_subject_label
                .and_then(|i| cols.get(i))
                .map(|s| s.to_string()),
            object_id,
            object_label: self
                .col_object_label
                .and_then(|i| cols.get(i))
                .map(|s| s.to_string()),
            predicate,
            mapping_justification: self
                .col_mapping_justification
                .and_then(|i| cols.get(i))
                .map(|s| s.to_string()),
            confidence,
            mapping_source: self
                .col_mapping_source
                .and_then(|i| cols.get(i))
                .map(|s| s.to_string()),
            creator: self
                .col_creator
                .and_then(|i| cols.get(i))
                .map(|s| s.to_string()),
            comment: self
                .col_comment
                .and_then(|i| cols.get(i))
                .map(|s| s.to_string()),
        })
    }
}

impl Default for SSSOMParser {
    fn default() -> Self {
        Self::new()
    }
}

/// SSSOM parse error
#[derive(Debug)]
pub enum SSSOMParseError {
    IoError(std::io::Error),
    MissingColumn(String),
    InvalidData(String),
}

impl std::fmt::Display for SSSOMParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::MissingColumn(col) => write!(f, "Missing required column: {}", col),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for SSSOMParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_predicate() {
        assert_eq!(
            MappingPredicate::from_str("skos:exactMatch"),
            MappingPredicate::ExactMatch
        );
        assert_eq!(
            MappingPredicate::from_str("skos:closeMatch"),
            MappingPredicate::CloseMatch
        );
        assert_eq!(
            MappingPredicate::from_str("owl:equivalentClass"),
            MappingPredicate::ExactMatch
        );
    }

    #[test]
    fn test_sssom_index() {
        let mut index = SSSOMIndex::new();

        let chebi = IRI::new("http://purl.obolibrary.org/obo/CHEBI_15365");
        let kegg = IRI::new("http://identifiers.org/kegg.compound/C00186");

        index.add(SSSOMMapping::exact(chebi.clone(), kegg.clone(), 0.95));

        let mappings = index.get_mappings_from(&chebi);
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].object_id, kegg);

        let equivalents = index.find_equivalents(&chebi);
        assert_eq!(equivalents.len(), 1);
    }

    #[test]
    fn test_cross_ontology_distance() {
        let mut index = SSSOMIndex::new();

        let mondo = IRI::new("http://purl.obolibrary.org/obo/MONDO_0005148");
        let doid = IRI::new("http://purl.obolibrary.org/obo/DOID_9352");

        // Type 2 diabetes exact match between MONDO and DOID
        index.add(SSSOMMapping::exact(mondo.clone(), doid.clone(), 1.0));

        let (distance, confidence) = index.cross_ontology_distance(&mondo, &doid).unwrap();
        assert_eq!(distance, 0.0); // Exact match = 0 distance
        assert_eq!(confidence, 1.0);
    }
}
