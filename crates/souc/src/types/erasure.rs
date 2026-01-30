//! Erasure Analysis for Quantitative Type Theory
//!
//! This module implements erasure analysis, which determines what parts of
//! a program can be erased at compile-time. In Sounio, this is critical
//! for ontological types: all 15M+ types are erased after type checking,
//! producing zero runtime overhead.
//!
//! Erasure works in conjunction with multiplicities:
//! - Multiplicity 0 → Always erased
//! - Multiplicity 1 → Runtime relevant, linear
//! - Multiplicity ω → Runtime relevant, unrestricted

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::multiplicity::{Multiplicity, QType};

/// Categories of erasable content
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErasureCategory {
    /// Type parameters and type-level computation
    TypeLevel,

    /// Ontological type annotations (IRI-based types)
    Ontological,

    /// Proof terms and witnesses
    Proof,

    /// Compile-time computed values
    CompileTimeValue,

    /// Phantom type parameters
    Phantom,
}

impl fmt::Display for ErasureCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErasureCategory::TypeLevel => write!(f, "type-level"),
            ErasureCategory::Ontological => write!(f, "ontological"),
            ErasureCategory::Proof => write!(f, "proof"),
            ErasureCategory::CompileTimeValue => write!(f, "compile-time"),
            ErasureCategory::Phantom => write!(f, "phantom"),
        }
    }
}

/// Result of erasure analysis for a single item
#[derive(Debug, Clone)]
pub struct ErasureInfo {
    /// Whether this item should be erased
    pub erased: bool,

    /// The multiplicity of this item
    pub multiplicity: Multiplicity,

    /// Category of erasure (if erased)
    pub category: Option<ErasureCategory>,

    /// Reason for erasure decision
    pub reason: String,
}

impl ErasureInfo {
    /// Create info for an erased item
    pub fn erased(category: ErasureCategory, reason: impl Into<String>) -> Self {
        ErasureInfo {
            erased: true,
            multiplicity: Multiplicity::Zero,
            category: Some(category),
            reason: reason.into(),
        }
    }

    /// Create info for a runtime-relevant item
    pub fn runtime(multiplicity: Multiplicity, reason: impl Into<String>) -> Self {
        ErasureInfo {
            erased: false,
            multiplicity,
            category: None,
            reason: reason.into(),
        }
    }
}

/// Tracks what has been marked for erasure
#[derive(Debug, Clone, Default)]
pub struct ErasureSet {
    /// Set of erased type parameters
    type_params: HashSet<String>,

    /// Set of erased value bindings
    value_bindings: HashSet<String>,

    /// Detailed info for each erased item
    info: HashMap<String, ErasureInfo>,
}

impl ErasureSet {
    /// Create a new empty erasure set
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a type parameter as erased
    pub fn erase_type_param(&mut self, name: String, category: ErasureCategory) {
        self.type_params.insert(name.clone());
        self.info.insert(
            name,
            ErasureInfo::erased(category, "type parameter has multiplicity 0"),
        );
    }

    /// Mark a value binding as erased
    pub fn erase_binding(&mut self, name: String, category: ErasureCategory, reason: &str) {
        self.value_bindings.insert(name.clone());
        self.info
            .insert(name, ErasureInfo::erased(category, reason));
    }

    /// Check if a type parameter is erased
    pub fn is_type_erased(&self, name: &str) -> bool {
        self.type_params.contains(name)
    }

    /// Check if a binding is erased
    pub fn is_binding_erased(&self, name: &str) -> bool {
        self.value_bindings.contains(name)
    }

    /// Get erasure info for an item
    pub fn get_info(&self, name: &str) -> Option<&ErasureInfo> {
        self.info.get(name)
    }

    /// Get total count of erased items
    pub fn erased_count(&self) -> usize {
        self.type_params.len() + self.value_bindings.len()
    }

    /// Iterate over all erased type parameters
    pub fn erased_types(&self) -> impl Iterator<Item = &String> {
        self.type_params.iter()
    }

    /// Iterate over all erased bindings
    pub fn erased_bindings(&self) -> impl Iterator<Item = &String> {
        self.value_bindings.iter()
    }

    /// Merge another erasure set into this one
    pub fn merge(&mut self, other: ErasureSet) {
        self.type_params.extend(other.type_params);
        self.value_bindings.extend(other.value_bindings);
        self.info.extend(other.info);
    }
}

/// Erasure analyzer that determines what can be erased
pub struct ErasureAnalyzer {
    /// Current erasure set being built
    erasures: ErasureSet,

    /// Stack of scopes for nested analysis
    scope_stack: Vec<ErasureSet>,

    /// Configuration options
    config: ErasureConfig,
}

/// Configuration for erasure analysis
#[derive(Debug, Clone)]
pub struct ErasureConfig {
    /// Whether to erase all ontological types (recommended: true)
    pub erase_ontological: bool,

    /// Whether to erase proof terms
    pub erase_proofs: bool,

    /// Whether to keep phantom types for debugging
    pub keep_phantom_debug: bool,

    /// Log erasure decisions
    pub verbose: bool,
}

impl Default for ErasureConfig {
    fn default() -> Self {
        ErasureConfig {
            erase_ontological: true,
            erase_proofs: true,
            keep_phantom_debug: false,
            verbose: false,
        }
    }
}

impl ErasureAnalyzer {
    /// Create a new erasure analyzer with default config
    pub fn new() -> Self {
        Self::with_config(ErasureConfig::default())
    }

    /// Create an erasure analyzer with custom config
    pub fn with_config(config: ErasureConfig) -> Self {
        ErasureAnalyzer {
            erasures: ErasureSet::new(),
            scope_stack: Vec::new(),
            config,
        }
    }

    /// Enter a new scope for erasure analysis
    pub fn enter_scope(&mut self) {
        self.scope_stack.push(std::mem::take(&mut self.erasures));
    }

    /// Exit scope, merging erasures into parent
    pub fn exit_scope(&mut self) -> ErasureSet {
        let scope_erasures = std::mem::take(&mut self.erasures);
        if let Some(parent) = self.scope_stack.pop() {
            self.erasures = parent;
        }
        scope_erasures
    }

    /// Analyze a type parameter and determine if it should be erased
    pub fn analyze_type_param<T>(&mut self, name: &str, qtype: &QType<T>) -> ErasureInfo {
        if qtype.multiplicity == Multiplicity::Zero {
            self.erasures
                .erase_type_param(name.to_string(), ErasureCategory::TypeLevel);
            ErasureInfo::erased(
                ErasureCategory::TypeLevel,
                format!("type parameter '{}' has multiplicity 0", name),
            )
        } else {
            ErasureInfo::runtime(
                qtype.multiplicity,
                format!(
                    "type parameter '{}' has multiplicity {}",
                    name, qtype.multiplicity
                ),
            )
        }
    }

    /// Analyze an ontological type annotation
    pub fn analyze_ontological(&mut self, iri: &str) -> ErasureInfo {
        if self.config.erase_ontological {
            self.erasures.erase_binding(
                iri.to_string(),
                ErasureCategory::Ontological,
                "ontological types are erased after type checking",
            );
            ErasureInfo::erased(
                ErasureCategory::Ontological,
                format!("ontological type '{}' erased", iri),
            )
        } else {
            ErasureInfo::runtime(Multiplicity::Many, "ontological erasure disabled")
        }
    }

    /// Analyze a proof term
    pub fn analyze_proof(&mut self, name: &str) -> ErasureInfo {
        if self.config.erase_proofs {
            self.erasures.erase_binding(
                name.to_string(),
                ErasureCategory::Proof,
                "proof terms are erased after verification",
            );
            ErasureInfo::erased(ErasureCategory::Proof, format!("proof '{}' erased", name))
        } else {
            ErasureInfo::runtime(Multiplicity::Many, "proof erasure disabled")
        }
    }

    /// Analyze a compile-time value
    pub fn analyze_comptime_value(&mut self, name: &str) -> ErasureInfo {
        self.erasures.erase_binding(
            name.to_string(),
            ErasureCategory::CompileTimeValue,
            "compile-time values are evaluated during compilation",
        );
        ErasureInfo::erased(
            ErasureCategory::CompileTimeValue,
            format!("comptime value '{}' erased", name),
        )
    }

    /// Get the current erasure set
    pub fn erasures(&self) -> &ErasureSet {
        &self.erasures
    }

    /// Consume analyzer and return final erasure set
    pub fn finish(self) -> ErasureSet {
        self.erasures
    }

    /// Get statistics about erasure
    pub fn stats(&self) -> ErasureStats {
        let mut by_category: HashMap<ErasureCategory, usize> = HashMap::new();

        for info in self.erasures.info.values() {
            if let Some(cat) = info.category {
                *by_category.entry(cat).or_insert(0) += 1;
            }
        }

        ErasureStats {
            total_erased: self.erasures.erased_count(),
            type_params_erased: self.erasures.type_params.len(),
            bindings_erased: self.erasures.value_bindings.len(),
            by_category,
        }
    }
}

impl Default for ErasureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about erasure analysis
#[derive(Debug, Clone)]
pub struct ErasureStats {
    /// Total items erased
    pub total_erased: usize,

    /// Type parameters erased
    pub type_params_erased: usize,

    /// Value bindings erased
    pub bindings_erased: usize,

    /// Breakdown by category
    pub by_category: HashMap<ErasureCategory, usize>,
}

impl fmt::Display for ErasureStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Erasure Statistics:")?;
        writeln!(f, "  Total erased: {}", self.total_erased)?;
        writeln!(f, "  Type params: {}", self.type_params_erased)?;
        writeln!(f, "  Bindings: {}", self.bindings_erased)?;
        writeln!(f, "  By category:")?;
        for (cat, count) in &self.by_category {
            writeln!(f, "    {}: {}", cat, count)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erasure_info_creation() {
        let erased = ErasureInfo::erased(ErasureCategory::Ontological, "test erasure");
        assert!(erased.erased);
        assert_eq!(erased.multiplicity, Multiplicity::Zero);
        assert_eq!(erased.category, Some(ErasureCategory::Ontological));

        let runtime = ErasureInfo::runtime(Multiplicity::One, "linear value");
        assert!(!runtime.erased);
        assert_eq!(runtime.multiplicity, Multiplicity::One);
        assert_eq!(runtime.category, None);
    }

    #[test]
    fn test_erasure_set() {
        let mut set = ErasureSet::new();

        set.erase_type_param("T".to_string(), ErasureCategory::TypeLevel);
        set.erase_binding("proof".to_string(), ErasureCategory::Proof, "verified");

        assert!(set.is_type_erased("T"));
        assert!(!set.is_type_erased("U"));
        assert!(set.is_binding_erased("proof"));
        assert_eq!(set.erased_count(), 2);
    }

    #[test]
    fn test_erasure_analyzer_ontological() {
        let mut analyzer = ErasureAnalyzer::new();

        let info = analyzer.analyze_ontological("http://snomed.info/id/73211009");
        assert!(info.erased);
        assert_eq!(info.category, Some(ErasureCategory::Ontological));

        assert!(
            analyzer
                .erasures()
                .is_binding_erased("http://snomed.info/id/73211009")
        );
    }

    #[test]
    fn test_erasure_analyzer_type_params() {
        let mut analyzer = ErasureAnalyzer::new();

        // Erased type param (multiplicity 0)
        let erased_type = QType::erased("OntologyMarker");
        let info = analyzer.analyze_type_param("T", &erased_type);
        assert!(info.erased);

        // Runtime type param (multiplicity ω)
        let runtime_type = QType::unrestricted("Int");
        let info = analyzer.analyze_type_param("U", &runtime_type);
        assert!(!info.erased);

        assert!(analyzer.erasures().is_type_erased("T"));
        assert!(!analyzer.erasures().is_type_erased("U"));
    }

    #[test]
    fn test_erasure_scopes() {
        let mut analyzer = ErasureAnalyzer::new();

        analyzer.analyze_ontological("outer");

        analyzer.enter_scope();
        analyzer.analyze_ontological("inner");
        let inner_set = analyzer.exit_scope();

        assert!(inner_set.is_binding_erased("inner"));
        assert!(!inner_set.is_binding_erased("outer"));

        // Outer scope still has its erasure
        assert!(analyzer.erasures().is_binding_erased("outer"));
    }

    #[test]
    fn test_erasure_stats() {
        let mut analyzer = ErasureAnalyzer::new();

        analyzer.analyze_ontological("snomed:123");
        analyzer.analyze_ontological("snomed:456");
        analyzer.analyze_proof("witness1");

        let erased_type = QType::erased("Marker");
        analyzer.analyze_type_param("M", &erased_type);

        let stats = analyzer.stats();
        assert_eq!(stats.total_erased, 4);
        assert_eq!(stats.type_params_erased, 1);
        assert_eq!(stats.bindings_erased, 3);
        assert_eq!(
            stats.by_category.get(&ErasureCategory::Ontological),
            Some(&2)
        );
        assert_eq!(stats.by_category.get(&ErasureCategory::Proof), Some(&1));
    }
}
