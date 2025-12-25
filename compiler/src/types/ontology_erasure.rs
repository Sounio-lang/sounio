//! Ontology-Specific Erasure Analysis
//!
//! This module handles erasure specifically for ontological types.
//! The key insight: 15M+ ontological types exist only at compile-time
//! to guide type checking. At runtime, they are completely erased.
//!
//! What gets erased:
//! - IRI type annotations (e.g., `snomed:73211009`)
//! - Subsumption constraints (e.g., `: Disease`)
//! - Semantic distance computations
//! - Ontology hierarchy lookups
//!
//! What remains at runtime:
//! - The actual data values
//! - Core type tags (for dynamic dispatch if needed)
//! - Confidence values (for epistemic types)

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::erasure::{ErasureCategory, ErasureInfo, ErasureSet};
use super::multiplicity::Multiplicity;

/// Statistics for ontology erasure
static ONTOLOGY_TYPES_ERASED: AtomicUsize = AtomicUsize::new(0);
static SUBSUMPTION_CHECKS_ERASED: AtomicUsize = AtomicUsize::new(0);
static SEMANTIC_COMPUTATIONS_ERASED: AtomicUsize = AtomicUsize::new(0);

/// Get current erasure statistics
pub fn erasure_stats() -> OntologyErasureStats {
    OntologyErasureStats {
        types_erased: ONTOLOGY_TYPES_ERASED.load(Ordering::Relaxed),
        subsumption_checks: SUBSUMPTION_CHECKS_ERASED.load(Ordering::Relaxed),
        semantic_computations: SEMANTIC_COMPUTATIONS_ERASED.load(Ordering::Relaxed),
    }
}

/// Reset erasure statistics (for testing)
pub fn reset_stats() {
    ONTOLOGY_TYPES_ERASED.store(0, Ordering::Relaxed);
    SUBSUMPTION_CHECKS_ERASED.store(0, Ordering::Relaxed);
    SEMANTIC_COMPUTATIONS_ERASED.store(0, Ordering::Relaxed);
}

/// Statistics about ontology erasure
#[derive(Debug, Clone, Default)]
pub struct OntologyErasureStats {
    /// Number of ontological types erased
    pub types_erased: usize,

    /// Number of subsumption checks erased
    pub subsumption_checks: usize,

    /// Number of semantic distance computations erased
    pub semantic_computations: usize,
}

impl OntologyErasureStats {
    /// Estimate memory saved by erasure (rough estimate)
    pub fn memory_saved_bytes(&self) -> usize {
        // Each ontological type annotation: ~64 bytes (IRI string + metadata)
        // Each subsumption check: ~128 bytes (two IRIs + result caching)
        // Each semantic computation: ~256 bytes (embeddings, distances)
        self.types_erased * 64 + self.subsumption_checks * 128 + self.semantic_computations * 256
    }
}

/// Represents an ontological type that will be erased
#[derive(Debug, Clone)]
pub struct OntologicalType {
    /// The IRI identifying this type
    pub iri: String,

    /// Parent types in the ontology hierarchy
    pub parents: Vec<String>,

    /// Whether this type has been verified against the ontology
    pub verified: bool,

    /// Multiplicity (always Zero for ontological types)
    pub multiplicity: Multiplicity,
}

impl OntologicalType {
    /// Create a new ontological type
    pub fn new(iri: impl Into<String>) -> Self {
        OntologicalType {
            iri: iri.into(),
            parents: Vec::new(),
            verified: false,
            multiplicity: Multiplicity::Zero,
        }
    }

    /// Add a parent type
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parents.push(parent.into());
        self
    }

    /// Mark as verified
    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Check if this type should be erased (always true for ontological types)
    pub fn should_erase(&self) -> bool {
        // Ontological types are always erased - they exist only for type checking
        true
    }
}

/// Analyzer specifically for ontological type erasure
pub struct OntologyErasureAnalyzer {
    /// Types that have been analyzed
    analyzed_types: HashMap<String, OntologicalType>,

    /// Subsumption relationships checked
    subsumption_checks: Vec<(String, String)>,

    /// Semantic computations performed
    semantic_computations: Vec<String>,

    /// Resulting erasure set
    erasures: ErasureSet,
}

impl OntologyErasureAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        OntologyErasureAnalyzer {
            analyzed_types: HashMap::new(),
            subsumption_checks: Vec::new(),
            semantic_computations: Vec::new(),
            erasures: ErasureSet::new(),
        }
    }

    /// Analyze an ontological type annotation
    pub fn analyze_type(&mut self, iri: &str) -> ErasureInfo {
        ONTOLOGY_TYPES_ERASED.fetch_add(1, Ordering::Relaxed);

        let onto_type = OntologicalType::new(iri);
        self.analyzed_types.insert(iri.to_string(), onto_type);

        self.erasures.erase_binding(
            iri.to_string(),
            ErasureCategory::Ontological,
            "ontological type annotation",
        );

        ErasureInfo::erased(
            ErasureCategory::Ontological,
            format!("type '{}' erased after verification", iri),
        )
    }

    /// Record a subsumption check (will be erased)
    pub fn record_subsumption(&mut self, subtype: &str, supertype: &str) {
        SUBSUMPTION_CHECKS_ERASED.fetch_add(1, Ordering::Relaxed);
        self.subsumption_checks
            .push((subtype.to_string(), supertype.to_string()));
    }

    /// Record a semantic computation (will be erased)
    pub fn record_semantic_computation(&mut self, description: &str) {
        SEMANTIC_COMPUTATIONS_ERASED.fetch_add(1, Ordering::Relaxed);
        self.semantic_computations.push(description.to_string());
    }

    /// Get all analyzed types
    pub fn types(&self) -> impl Iterator<Item = (&String, &OntologicalType)> {
        self.analyzed_types.iter()
    }

    /// Get erasure statistics
    pub fn stats(&self) -> OntologyErasureStats {
        OntologyErasureStats {
            types_erased: self.analyzed_types.len(),
            subsumption_checks: self.subsumption_checks.len(),
            semantic_computations: self.semantic_computations.len(),
        }
    }

    /// Consume analyzer and return erasure set
    pub fn finish(self) -> ErasureSet {
        self.erasures
    }
}

impl Default for OntologyErasureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents what remains after ontological erasure
#[derive(Debug, Clone)]
pub struct ErasedRepresentation {
    /// Core type tag (if runtime type info needed)
    pub type_tag: Option<u32>,

    /// Size of the erased type's data
    pub data_size: usize,

    /// Alignment requirement
    pub alignment: usize,
}

impl ErasedRepresentation {
    /// Create representation for a fully erased type (no runtime presence)
    pub fn fully_erased() -> Self {
        ErasedRepresentation {
            type_tag: None,
            data_size: 0,
            alignment: 1,
        }
    }

    /// Create representation with a type tag for dynamic dispatch
    pub fn with_tag(tag: u32, data_size: usize, alignment: usize) -> Self {
        ErasedRepresentation {
            type_tag: Some(tag),
            data_size,
            alignment,
        }
    }

    /// Total runtime size (0 for fully erased types)
    pub fn runtime_size(&self) -> usize {
        if self.type_tag.is_some() {
            4 + self.data_size // 4 bytes for tag + data
        } else {
            0
        }
    }
}

/// Tracks erasure decisions for a compilation unit
#[derive(Debug, Default)]
pub struct CompilationErasure {
    /// Per-function erasure info
    function_erasures: HashMap<String, FunctionErasureInfo>,

    /// Global type erasures
    global_types: ErasureSet,
}

/// Erasure information for a function
#[derive(Debug, Clone)]
pub struct FunctionErasureInfo {
    /// Function name
    pub name: String,

    /// Parameters that are erased
    pub erased_params: Vec<String>,

    /// Return type erased?
    pub return_erased: bool,

    /// Type parameters that are erased
    pub erased_type_params: Vec<String>,

    /// Ontological constraints (all erased)
    pub ontological_constraints: usize,
}

impl CompilationErasure {
    /// Create new compilation erasure tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record erasure info for a function
    pub fn record_function(&mut self, info: FunctionErasureInfo) {
        self.function_erasures.insert(info.name.clone(), info);
    }

    /// Add global type erasure
    pub fn add_global_erasure(&mut self, erasures: ErasureSet) {
        self.global_types.merge(erasures);
    }

    /// Get summary statistics
    pub fn summary(&self) -> CompilationErasureSummary {
        let total_erased_params: usize = self
            .function_erasures
            .values()
            .map(|f| f.erased_params.len())
            .sum();

        let total_erased_type_params: usize = self
            .function_erasures
            .values()
            .map(|f| f.erased_type_params.len())
            .sum();

        let total_ontological: usize = self
            .function_erasures
            .values()
            .map(|f| f.ontological_constraints)
            .sum();

        CompilationErasureSummary {
            functions_analyzed: self.function_erasures.len(),
            erased_params: total_erased_params,
            erased_type_params: total_erased_type_params,
            ontological_constraints: total_ontological,
            global_erasures: self.global_types.erased_count(),
        }
    }
}

/// Summary of erasure for a compilation
#[derive(Debug, Clone)]
pub struct CompilationErasureSummary {
    /// Functions analyzed
    pub functions_analyzed: usize,

    /// Total erased parameters
    pub erased_params: usize,

    /// Total erased type parameters
    pub erased_type_params: usize,

    /// Total ontological constraints erased
    pub ontological_constraints: usize,

    /// Global erasures
    pub global_erasures: usize,
}

impl std::fmt::Display for CompilationErasureSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Compilation Erasure Summary:")?;
        writeln!(f, "  Functions: {}", self.functions_analyzed)?;
        writeln!(f, "  Erased params: {}", self.erased_params)?;
        writeln!(f, "  Erased type params: {}", self.erased_type_params)?;
        writeln!(
            f,
            "  Ontological constraints: {}",
            self.ontological_constraints
        )?;
        writeln!(f, "  Global erasures: {}", self.global_erasures)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ontological_type() {
        let ty = OntologicalType::new("snomed:73211009")
            .with_parent("snomed:64572001")
            .verified();

        assert_eq!(ty.iri, "snomed:73211009");
        assert_eq!(ty.parents, vec!["snomed:64572001"]);
        assert!(ty.verified);
        assert!(ty.should_erase()); // Always erased
        assert_eq!(ty.multiplicity, Multiplicity::Zero);
    }

    #[test]
    fn test_ontology_erasure_analyzer() {
        reset_stats();

        let mut analyzer = OntologyErasureAnalyzer::new();

        // Analyze some types
        let info = analyzer.analyze_type("snomed:73211009");
        assert!(info.erased);

        analyzer.analyze_type("snomed:64572001");
        analyzer.record_subsumption("snomed:73211009", "snomed:64572001");
        analyzer.record_semantic_computation("distance calculation");

        let stats = analyzer.stats();
        assert_eq!(stats.types_erased, 2);
        assert_eq!(stats.subsumption_checks, 1);
        assert_eq!(stats.semantic_computations, 1);

        // Memory saved estimate
        assert!(stats.memory_saved_bytes() > 0);
    }

    #[test]
    fn test_erased_representation() {
        let fully_erased = ErasedRepresentation::fully_erased();
        assert_eq!(fully_erased.runtime_size(), 0);

        let with_tag = ErasedRepresentation::with_tag(42, 16, 8);
        assert_eq!(with_tag.runtime_size(), 20); // 4 + 16
    }

    #[test]
    fn test_compilation_erasure() {
        let mut comp = CompilationErasure::new();

        comp.record_function(FunctionErasureInfo {
            name: "process_diagnosis".to_string(),
            erased_params: vec!["T".to_string()],
            return_erased: false,
            erased_type_params: vec!["Ontology".to_string()],
            ontological_constraints: 5,
        });

        comp.record_function(FunctionErasureInfo {
            name: "validate_treatment".to_string(),
            erased_params: vec![],
            return_erased: true,
            erased_type_params: vec!["O1".to_string(), "O2".to_string()],
            ontological_constraints: 3,
        });

        let summary = comp.summary();
        assert_eq!(summary.functions_analyzed, 2);
        assert_eq!(summary.erased_params, 1);
        assert_eq!(summary.erased_type_params, 3);
        assert_eq!(summary.ontological_constraints, 8);
    }

    #[test]
    fn test_global_stats() {
        // Note: This test checks that global stats increment correctly.
        // Since tests run in parallel, we check relative increments, not absolute values.
        reset_stats();
        let baseline = erasure_stats().types_erased;

        let mut analyzer1 = OntologyErasureAnalyzer::new();
        analyzer1.analyze_type("test:1");
        analyzer1.analyze_type("test:2");

        let mut analyzer2 = OntologyErasureAnalyzer::new();
        analyzer2.analyze_type("test:3");

        let global = erasure_stats();
        // We added 3 types, so count should increase by at least 3
        assert!(
            global.types_erased >= baseline + 3,
            "Expected at least {} types erased, got {}",
            baseline + 3,
            global.types_erased
        );
    }
}
