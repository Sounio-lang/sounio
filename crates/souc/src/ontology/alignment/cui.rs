//! UMLS CUI (Concept Unique Identifier) Bridging
//!
//! The Unified Medical Language System (UMLS) provides a common framework for
//! integrating biomedical vocabularies. Each concept in UMLS has a unique
//! identifier (CUI) that can serve as a bridge between different ontologies.
//!
//! # Example
//!
//! ```text
//! ChEBI:15365 (Aspirin)
//!     └── maps to → UMLS:C0004057 (Aspirin)
//!                        ↓
//! DrugBank:DB00945 (Acetylsalicylic acid)
//!     └── maps to → UMLS:C0004057 (Aspirin)
//!
//! Therefore: ChEBI:15365 ≈ DrugBank:DB00945 (via CUI C0004057)
//! ```
//!
//! # Data Sources
//!
//! - UMLS Metathesaurus (requires UMLS license)
//! - MRCONSO.RRF: Concept names and sources
//! - MRREL.RRF: Relationships between concepts
//! - MRSTY.RRF: Semantic types

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::ontology::loader::IRI;

/// UMLS Concept Unique Identifier
pub type CUI = String;

/// A UMLS concept with its mappings
#[derive(Debug, Clone)]
pub struct UMLSConcept {
    /// Concept Unique Identifier
    pub cui: CUI,
    /// Preferred term (English)
    pub preferred_term: String,
    /// Semantic types (e.g., "Disease or Syndrome")
    pub semantic_types: Vec<String>,
    /// Source vocabularies that include this concept
    pub sources: Vec<SourceAtom>,
}

/// An atom from a source vocabulary
#[derive(Debug, Clone)]
pub struct SourceAtom {
    /// Source vocabulary abbreviation (e.g., "SNOMEDCT_US", "ICD10CM")
    pub source: String,
    /// Atom identifier in the source
    pub code: String,
    /// Term string in the source
    pub term: String,
    /// Term type (e.g., "PT" for Preferred Term)
    pub term_type: String,
}

impl SourceAtom {
    /// Convert to IRI based on source vocabulary
    pub fn to_iri(&self) -> Option<IRI> {
        let prefix = source_to_prefix(&self.source)?;
        Some(IRI::from_curie(prefix, &self.code))
    }
}

/// Convert UMLS source abbreviation to ontology prefix
fn source_to_prefix(source: &str) -> Option<&'static str> {
    match source {
        "SNOMEDCT_US" | "SNOMEDCT" => Some("SNOMED"),
        "ICD10CM" | "ICD10" => Some("ICD10"),
        "ICD9CM" | "ICD9" => Some("ICD9"),
        "HPO" => Some("HP"),
        "GO" => Some("GO"),
        "OMIM" => Some("OMIM"),
        "NCBI" | "NCBITAXON" => Some("NCBITaxon"),
        "MESH" | "MSH" => Some("MESH"),
        "RXNORM" => Some("RXNORM"),
        "NCI" | "NCIT" => Some("NCIT"),
        "DRUGBANK" => Some("DrugBank"),
        "CHEBI" => Some("CHEBI"),
        "DOID" => Some("DOID"),
        "MONDO" => Some("MONDO"),
        "ORPHANET" | "ORDO" => Some("Orphanet"),
        "UBERON" => Some("UBERON"),
        "CL" => Some("CL"),
        "FMA" => Some("FMA"),
        _ => None,
    }
}

/// Mapping between an ontology term and a UMLS CUI
#[derive(Debug, Clone)]
pub struct CUIMapping {
    /// The ontology term IRI
    pub term: IRI,
    /// The UMLS CUI
    pub cui: CUI,
    /// Source vocabulary
    pub source: String,
    /// Confidence in the mapping
    pub confidence: f64,
}

/// CUI Bridge for cross-ontology alignment via UMLS
pub struct CUIBridge {
    /// CUI to concept information
    concepts: HashMap<CUI, UMLSConcept>,
    /// Term IRI to CUI mappings
    term_to_cui: HashMap<IRI, Vec<CUI>>,
    /// CUI to term IRI mappings
    cui_to_terms: HashMap<CUI, Vec<IRI>>,
    /// Supported source vocabularies
    supported_sources: HashSet<String>,
}

impl CUIBridge {
    /// Create empty bridge
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            term_to_cui: HashMap::new(),
            cui_to_terms: HashMap::new(),
            supported_sources: Self::default_sources(),
        }
    }

    /// Default supported source vocabularies
    fn default_sources() -> HashSet<String> {
        [
            "SNOMEDCT_US",
            "ICD10CM",
            "ICD9CM",
            "HPO",
            "GO",
            "OMIM",
            "MESH",
            "RXNORM",
            "NCI",
            "DRUGBANK",
            "CHEBI",
            "DOID",
            "MONDO",
            "ORPHANET",
            "UBERON",
            "CL",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Load from UMLS RRF files
    pub fn load_from_rrf(&mut self, mrconso_path: &Path) -> Result<usize, CUILoadError> {
        let file = File::open(mrconso_path).map_err(CUILoadError::IoError)?;
        let reader = BufReader::new(file);
        let mut loaded = 0;

        for line in reader.lines() {
            let line = line.map_err(CUILoadError::IoError)?;
            if let Some(atom) = self.parse_mrconso_line(&line) {
                self.add_atom(atom);
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Parse a line from MRCONSO.RRF
    /// Format: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
    fn parse_mrconso_line(&self, line: &str) -> Option<(CUI, SourceAtom)> {
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() < 15 {
            return None;
        }

        let cui = fields[0].to_string();
        let language = fields[1];
        let source = fields[11].to_string();
        let term_type = fields[12].to_string();
        let code = fields[13].to_string();
        let term = fields[14].to_string();

        // Only English terms from supported sources
        if language != "ENG" || !self.supported_sources.contains(&source) {
            return None;
        }

        let atom = SourceAtom {
            source,
            code,
            term,
            term_type,
        };

        Some((cui, atom))
    }

    /// Add an atom to the bridge
    fn add_atom(&mut self, (cui, atom): (CUI, SourceAtom)) {
        // Get or create concept
        let concept = self
            .concepts
            .entry(cui.clone())
            .or_insert_with(|| UMLSConcept {
                cui: cui.clone(),
                preferred_term: String::new(),
                semantic_types: Vec::new(),
                sources: Vec::new(),
            });

        // Update preferred term if this is a PT (Preferred Term)
        if atom.term_type == "PT" && concept.preferred_term.is_empty() {
            concept.preferred_term = atom.term.clone();
        }

        // Convert to IRI and index
        if let Some(iri) = atom.to_iri() {
            self.term_to_cui
                .entry(iri.clone())
                .or_default()
                .push(cui.clone());
            self.cui_to_terms.entry(cui.clone()).or_default().push(iri);
        }

        concept.sources.push(atom);
    }

    /// Add a direct mapping
    pub fn add_mapping(&mut self, term: IRI, cui: CUI) {
        self.term_to_cui
            .entry(term.clone())
            .or_default()
            .push(cui.clone());
        self.cui_to_terms.entry(cui).or_default().push(term);
    }

    /// Get CUIs for a term
    pub fn get_cuis(&self, term: &IRI) -> &[CUI] {
        self.term_to_cui
            .get(term)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get terms mapped to a CUI
    pub fn get_terms(&self, cui: &CUI) -> &[IRI] {
        self.cui_to_terms
            .get(cui)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get concept information for a CUI
    pub fn get_concept(&self, cui: &CUI) -> Option<&UMLSConcept> {
        self.concepts.get(cui)
    }

    /// Find equivalent terms across ontologies via CUI bridging
    pub fn find_equivalents(&self, term: &IRI) -> Vec<(IRI, CUI, f64)> {
        let mut equivalents = Vec::new();
        let cuis = self.get_cuis(term);

        for cui in cuis {
            let terms = self.get_terms(cui);
            for eq_term in terms {
                if eq_term != term {
                    // Confidence based on source reliability
                    let confidence = self.compute_confidence(term, eq_term, cui);
                    equivalents.push((eq_term.clone(), cui.clone(), confidence));
                }
            }
        }

        // Deduplicate by IRI, keeping highest confidence
        let mut best: HashMap<IRI, (CUI, f64)> = HashMap::new();
        for (iri, cui, conf) in equivalents {
            let entry = best.entry(iri).or_insert((cui.clone(), 0.0));
            if conf > entry.1 {
                *entry = (cui, conf);
            }
        }

        best.into_iter()
            .map(|(iri, (cui, conf))| (iri, cui, conf))
            .collect()
    }

    /// Compute confidence for a CUI-based alignment
    fn compute_confidence(&self, source: &IRI, target: &IRI, _cui: &CUI) -> f64 {
        // Base confidence for CUI-based alignment
        // Note: We don't require a concept entry - the CUI itself is evidence enough
        let mut confidence: f64 = 0.8;

        // Boost if both are from well-curated sources
        use crate::ontology::loader::OntologyId;
        let source_ont = source.ontology();
        let target_ont = target.ontology();

        let curated = [
            OntologyId::SNOMED,
            OntologyId::ChEBI,
            OntologyId::MONDO,
            OntologyId::DOID,
            OntologyId::HP,
            OntologyId::GO,
        ];
        if curated.contains(&source_ont) && curated.contains(&target_ont) {
            confidence += 0.1;
        }

        // Slight penalty for different semantic domains (would need semantic types)
        // For now, just return base confidence
        confidence.min(1.0)
    }

    /// Check if two terms share a CUI
    pub fn share_cui(&self, a: &IRI, b: &IRI) -> Option<CUI> {
        let cuis_a: HashSet<_> = self.get_cuis(a).iter().collect();
        let cuis_b: HashSet<_> = self.get_cuis(b).iter().collect();

        cuis_a.intersection(&cuis_b).next().map(|c| (*c).clone())
    }

    /// Compute distance between terms via CUI bridging
    /// Returns (distance, confidence, bridging_cui)
    pub fn cui_distance(&self, a: &IRI, b: &IRI) -> Option<(f64, f64, CUI)> {
        // Direct CUI match
        if let Some(cui) = self.share_cui(a, b) {
            let confidence = self.compute_confidence(a, b, &cui);
            return Some((0.0, confidence, cui));
        }

        // One-hop through related CUIs (would need MRREL.RRF)
        // For now, just check direct matches
        None
    }

    /// Get statistics
    pub fn stats(&self) -> CUIBridgeStats {
        let mut ontology_counts: HashMap<String, usize> = HashMap::new();

        for iri in self.term_to_cui.keys() {
            let ont = iri.ontology().to_string();
            *ontology_counts.entry(ont).or_insert(0) += 1;
        }

        CUIBridgeStats {
            total_concepts: self.concepts.len(),
            total_mappings: self.term_to_cui.values().map(|v| v.len()).sum(),
            unique_terms: self.term_to_cui.len(),
            ontology_counts,
        }
    }

    /// Number of concepts
    pub fn len(&self) -> usize {
        self.concepts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.concepts.is_empty()
    }
}

impl Default for CUIBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for CUI bridge
#[derive(Debug, Clone)]
pub struct CUIBridgeStats {
    pub total_concepts: usize,
    pub total_mappings: usize,
    pub unique_terms: usize,
    pub ontology_counts: HashMap<String, usize>,
}

/// Error loading CUI data
#[derive(Debug)]
pub enum CUILoadError {
    IoError(std::io::Error),
    ParseError(String),
}

impl std::fmt::Display for CUILoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for CUILoadError {}

/// Builder for creating CUI bridges from multiple sources
pub struct CUIBridgeBuilder {
    bridge: CUIBridge,
}

impl CUIBridgeBuilder {
    pub fn new() -> Self {
        Self {
            bridge: CUIBridge::new(),
        }
    }

    /// Load from UMLS MRCONSO.RRF
    pub fn load_mrconso(mut self, path: &Path) -> Result<Self, CUILoadError> {
        self.bridge.load_from_rrf(path)?;
        Ok(self)
    }

    /// Add manual mappings from a simple TSV file
    /// Format: term_iri<TAB>cui
    pub fn load_mappings_tsv(mut self, path: &Path) -> Result<Self, CUILoadError> {
        let file = File::open(path).map_err(CUILoadError::IoError)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.map_err(CUILoadError::IoError)?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                let term = IRI::new(parts[0]);
                let cui = parts[1].to_string();
                self.bridge.add_mapping(term, cui);
            }
        }

        Ok(self)
    }

    /// Add a single mapping
    pub fn add_mapping(mut self, term: IRI, cui: CUI) -> Self {
        self.bridge.add_mapping(term, cui);
        self
    }

    /// Build the bridge
    pub fn build(self) -> CUIBridge {
        self.bridge
    }
}

impl Default for CUIBridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_to_prefix() {
        assert_eq!(source_to_prefix("SNOMEDCT_US"), Some("SNOMED"));
        assert_eq!(source_to_prefix("ICD10CM"), Some("ICD10"));
        assert_eq!(source_to_prefix("CHEBI"), Some("CHEBI"));
        assert_eq!(source_to_prefix("UNKNOWN"), None);
    }

    #[test]
    fn test_cui_bridge_basic() {
        let mut bridge = CUIBridge::new();

        let chebi = IRI::from_curie("CHEBI", "15365");
        let drugbank = IRI::from_curie("DrugBank", "DB00945");
        let cui = "C0004057".to_string();

        bridge.add_mapping(chebi.clone(), cui.clone());
        bridge.add_mapping(drugbank.clone(), cui.clone());

        assert!(bridge.share_cui(&chebi, &drugbank).is_some());

        let equiv = bridge.find_equivalents(&chebi);
        assert_eq!(equiv.len(), 1);
        assert_eq!(equiv[0].0, drugbank);
    }

    #[test]
    fn test_cui_distance() {
        let mut bridge = CUIBridge::new();

        let mondo = IRI::from_curie("MONDO", "0005148");
        let doid = IRI::from_curie("DOID", "9352");
        let cui = "C0011860".to_string(); // Type 2 diabetes

        bridge.add_mapping(mondo.clone(), cui.clone());
        bridge.add_mapping(doid.clone(), cui.clone());

        let result = bridge.cui_distance(&mondo, &doid);
        assert!(result.is_some());

        let (distance, confidence, bridging_cui) = result.unwrap();
        assert_eq!(distance, 0.0);
        assert!(confidence > 0.5);
        assert_eq!(bridging_cui, cui);
    }

    #[test]
    fn test_cui_bridge_stats() {
        let mut bridge = CUIBridge::new();

        bridge.add_mapping(IRI::from_curie("CHEBI", "1"), "C001".to_string());
        bridge.add_mapping(IRI::from_curie("CHEBI", "2"), "C002".to_string());
        bridge.add_mapping(IRI::from_curie("MONDO", "1"), "C001".to_string());

        let stats = bridge.stats();
        assert_eq!(stats.unique_terms, 3);
        assert_eq!(stats.total_mappings, 3);
    }
}
