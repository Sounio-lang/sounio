//! Semantic Annotations for Diagnostics
//!
//! Adds ontology-specific context to error messages.
//! Explains WHY types are incompatible in domain terms.

use crate::ontology::distance::SemanticDistance;
use crate::ontology::loader::IRI;

/// Semantic context for a type mismatch
#[derive(Debug, Clone)]
pub struct SemanticContext {
    /// Human-readable description of expected type
    pub expected_description: String,
    /// Human-readable description of found type
    pub found_description: String,
    /// Why these types are incompatible
    pub incompatibility_reason: String,
    /// Additional notes about the types
    pub type_notes: Vec<String>,
    /// Domain-specific explanation
    pub domain_explanation: Option<String>,
}

impl SemanticContext {
    /// Create a new semantic context
    pub fn new(expected: &str, found: &str) -> Self {
        Self {
            expected_description: expected.to_string(),
            found_description: found.to_string(),
            incompatibility_reason: String::new(),
            type_notes: Vec::new(),
            domain_explanation: None,
        }
    }

    /// Add incompatibility reason
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.incompatibility_reason = reason.into();
        self
    }

    /// Add a note
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.type_notes.push(note.into());
        self
    }

    /// Add domain explanation
    pub fn with_domain_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.domain_explanation = Some(explanation.into());
        self
    }
}

/// Information about an ontological term
#[derive(Debug, Clone)]
pub struct TermInfo {
    /// The term IRI
    pub iri: String,
    /// Human-readable label
    pub label: String,
    /// Short description (e.g., "a pain reliever")
    pub short_description: String,
    /// Full description/definition
    pub full_description: String,
    /// Branch path (e.g., "drugs/analgesics/NSAIDs")
    pub branch_path: String,
    /// Ontology prefix (e.g., "ChEBI")
    pub ontology_prefix: String,
}

impl TermInfo {
    /// Create minimal term info
    pub fn minimal(iri: &str, label: &str) -> Self {
        Self {
            iri: iri.to_string(),
            label: label.to_string(),
            short_description: format!("a {}", label),
            full_description: label.to_string(),
            branch_path: String::new(),
            ontology_prefix: extract_prefix(iri),
        }
    }

    /// Create from IRI with extracted info
    pub fn from_iri(iri: &IRI) -> Self {
        let (prefix, local) = iri
            .to_curie()
            .unwrap_or_else(|| ("unknown".to_string(), iri.as_str().to_string()));

        Self {
            iri: iri.as_str().to_string(),
            label: local.clone(),
            short_description: format!("a {}", local),
            full_description: local,
            branch_path: String::new(),
            ontology_prefix: prefix,
        }
    }
}

/// Distance components for explanation
#[derive(Debug, Clone, Copy)]
pub struct DistanceComponents {
    /// Path-based distance in hierarchy
    pub path: Option<f64>,
    /// Information content distance
    pub ic: Option<f64>,
    /// Embedding-based distance
    pub embedding: Option<f64>,
    /// Overall combined distance
    pub combined: f64,
}

impl DistanceComponents {
    /// Create from semantic distance
    pub fn from_semantic_distance(dist: &SemanticDistance) -> Self {
        Self {
            path: None, // Would need to track this separately
            ic: None,
            embedding: None,
            combined: dist.conceptual,
        }
    }

    /// Create with explicit components
    pub fn new(combined: f64) -> Self {
        Self {
            path: None,
            ic: None,
            embedding: None,
            combined,
        }
    }

    /// Set path distance
    pub fn with_path(mut self, path: f64) -> Self {
        self.path = Some(path);
        self
    }

    /// Set IC distance
    pub fn with_ic(mut self, ic: f64) -> Self {
        self.ic = Some(ic);
        self
    }

    /// Set embedding distance
    pub fn with_embedding(mut self, embedding: f64) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Semantic annotator for enriching diagnostics
pub struct SemanticAnnotator {
    /// Cache of term information
    term_cache: std::collections::HashMap<String, TermInfo>,
}

impl SemanticAnnotator {
    /// Create a new semantic annotator
    pub fn new() -> Self {
        Self {
            term_cache: std::collections::HashMap::new(),
        }
    }

    /// Generate semantic context for a type mismatch
    pub fn annotate_mismatch(
        &self,
        expected: &TermInfo,
        found: &TermInfo,
        distance: &DistanceComponents,
    ) -> SemanticContext {
        let mut context = SemanticContext::new(&expected.full_description, &found.full_description);

        // Build incompatibility reason from distance components
        let reason = self.explain_incompatibility(distance);
        context = context.with_reason(reason);

        // Add notes about each type
        context = context.with_note(format!(
            "`{}` is {} (branch: {})",
            expected.iri,
            expected.short_description,
            if expected.branch_path.is_empty() {
                &expected.ontology_prefix
            } else {
                &expected.branch_path
            }
        ));

        context = context.with_note(format!(
            "`{}` is {} (branch: {})",
            found.iri,
            found.short_description,
            if found.branch_path.is_empty() {
                &found.ontology_prefix
            } else {
                &found.branch_path
            }
        ));

        // Check if different ontologies
        if expected.ontology_prefix != found.ontology_prefix {
            context = context.with_note(format!(
                "types are from different ontologies ({} vs {})",
                expected.ontology_prefix, found.ontology_prefix
            ));
        }

        // Domain-specific explanation
        if let Some(explanation) = self.generate_domain_explanation(expected, found) {
            context = context.with_domain_explanation(explanation);
        }

        context
    }

    /// Explain why types are incompatible based on distance components
    fn explain_incompatibility(&self, distance: &DistanceComponents) -> String {
        let mut reasons = Vec::new();

        // Explain path distance
        if let Some(path) = distance.path {
            if path > 0.5 {
                reasons.push("different branches of the ontology".to_string());
            } else if path > 0.3 {
                reasons.push("distantly related in hierarchy".to_string());
            }
        }

        // Explain IC distance
        if let Some(ic) = distance.ic
            && ic > 0.5
        {
            reasons.push("low shared information content".to_string());
        }

        // Explain embedding distance
        if let Some(emb) = distance.embedding {
            if emb > 0.5 {
                reasons.push("semantically unrelated concepts".to_string());
            } else if emb > 0.3 {
                reasons.push("weak semantic similarity".to_string());
            }
        }

        if reasons.is_empty() {
            format!(
                "semantic distance {:.3} exceeds threshold",
                distance.combined
            )
        } else {
            reasons.join("; ")
        }
    }

    /// Generate domain-specific explanation based on ontology type
    fn generate_domain_explanation(&self, expected: &TermInfo, found: &TermInfo) -> Option<String> {
        match expected.ontology_prefix.as_str() {
            "CHEBI" => self.explain_chebi_mismatch(expected, found),
            "GO" => self.explain_go_mismatch(expected, found),
            "HP" => self.explain_hp_mismatch(expected, found),
            "MONDO" | "DOID" => self.explain_disease_mismatch(expected, found),
            "UO" => self.explain_unit_mismatch(expected, found),
            "PATO" => self.explain_quality_mismatch(expected, found),
            _ => None,
        }
    }

    fn explain_chebi_mismatch(&self, expected: &TermInfo, found: &TermInfo) -> Option<String> {
        // Check for common ChEBI category mismatches
        let expected_lower = expected.label.to_lowercase();
        let found_lower = found.label.to_lowercase();

        // Drug vs non-drug
        if (expected_lower.contains("drug") || expected_lower.contains("pharmaceutical"))
            && !found_lower.contains("drug")
        {
            return Some(format!(
                "{} is a pharmaceutical compound; {} may not be suitable for therapeutic use",
                expected.label, found.label
            ));
        }

        // Hormone vs other
        if found_lower.contains("hormone") && !expected_lower.contains("hormone") {
            return Some(format!(
                "{} is a hormone; expected {} which has a different biological role",
                found.label, expected.label
            ));
        }

        None
    }

    fn explain_go_mismatch(&self, expected: &TermInfo, found: &TermInfo) -> Option<String> {
        // Gene Ontology has three main branches
        let expected_branch = categorize_go_term(&expected.branch_path);
        let found_branch = categorize_go_term(&found.branch_path);

        if expected_branch != found_branch {
            return Some(format!(
                "GO terms are in different categories: {} vs {}",
                expected_branch, found_branch
            ));
        }

        None
    }

    fn explain_hp_mismatch(&self, _expected: &TermInfo, _found: &TermInfo) -> Option<String> {
        // Human Phenotype Ontology explanations
        None
    }

    fn explain_disease_mismatch(&self, expected: &TermInfo, found: &TermInfo) -> Option<String> {
        // Disease ontology explanations
        Some(format!(
            "{} and {} are different disease entities",
            expected.label, found.label
        ))
    }

    fn explain_unit_mismatch(&self, expected: &TermInfo, found: &TermInfo) -> Option<String> {
        // Unit ontology - incompatible units
        Some(format!(
            "incompatible units: {} cannot be converted to {}",
            found.label, expected.label
        ))
    }

    fn explain_quality_mismatch(&self, expected: &TermInfo, found: &TermInfo) -> Option<String> {
        // PATO quality mismatch
        Some(format!(
            "different quality types: {} vs {}",
            expected.label, found.label
        ))
    }

    /// Cache term info
    pub fn cache_term(&mut self, iri: &str, info: TermInfo) {
        self.term_cache.insert(iri.to_string(), info);
    }

    /// Get cached term info
    pub fn get_cached(&self, iri: &str) -> Option<&TermInfo> {
        self.term_cache.get(iri)
    }
}

impl Default for SemanticAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

/// Semantic suggestion with distance and description
#[derive(Debug, Clone)]
pub struct SemanticSuggestion {
    /// The suggested term IRI
    pub term_iri: String,
    /// Display text (e.g., "ChEBI.Aspirin")
    pub text: String,
    /// Semantic distance from original
    pub distance: f32,
    /// Brief description
    pub description: String,
    /// Full label
    pub label: String,
}

impl SemanticSuggestion {
    /// Create a new suggestion
    pub fn new(term_iri: &str, text: &str, distance: f32) -> Self {
        Self {
            term_iri: term_iri.to_string(),
            text: text.to_string(),
            distance,
            description: String::new(),
            label: text.to_string(),
        }
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

/// Enrich a list of raw suggestions with semantic information
pub fn enrich_suggestions(
    suggestions: &[(String, f32)], // (IRI, distance)
    annotator: &SemanticAnnotator,
) -> Vec<SemanticSuggestion> {
    suggestions
        .iter()
        .map(|(iri, distance)| {
            if let Some(info) = annotator.get_cached(iri) {
                SemanticSuggestion {
                    term_iri: iri.clone(),
                    text: format!("{}.{}", info.ontology_prefix, info.label),
                    distance: *distance,
                    description: truncate_description(&info.full_description, 50),
                    label: info.label.clone(),
                }
            } else {
                let (prefix, local) = iri.split_once(':').unwrap_or(("", iri));
                SemanticSuggestion::new(iri, &format!("{}.{}", prefix, local), *distance)
            }
        })
        .collect()
}

/// Extract ontology prefix from IRI
fn extract_prefix(iri: &str) -> String {
    if let Some((prefix, _)) = iri.split_once(':') {
        prefix.to_string()
    } else if iri.contains('/') {
        // Try to extract from URL-style IRI
        iri.rsplit('/')
            .next()
            .and_then(|s| s.split('_').next())
            .unwrap_or("unknown")
            .to_string()
    } else {
        "unknown".to_string()
    }
}

/// Categorize GO term by branch
fn categorize_go_term(branch_path: &str) -> &'static str {
    let lower = branch_path.to_lowercase();
    if lower.contains("biological_process") || lower.contains("bp") {
        "biological process"
    } else if lower.contains("molecular_function") || lower.contains("mf") {
        "molecular function"
    } else if lower.contains("cellular_component") || lower.contains("cc") {
        "cellular component"
    } else {
        "unknown"
    }
}

/// Truncate description to max length
fn truncate_description(desc: &str, max_len: usize) -> String {
    if desc.len() <= max_len {
        desc.to_string()
    } else {
        format!("{}...", &desc[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_info_minimal() {
        let info = TermInfo::minimal("ChEBI:15365", "aspirin");
        assert_eq!(info.label, "aspirin");
        assert_eq!(info.ontology_prefix, "ChEBI");
    }

    #[test]
    fn test_semantic_context_builder() {
        let ctx = SemanticContext::new("Analgesic", "Hormone")
            .with_reason("different therapeutic categories")
            .with_note("test note");

        assert_eq!(ctx.expected_description, "Analgesic");
        assert_eq!(ctx.found_description, "Hormone");
        assert_eq!(ctx.type_notes.len(), 1);
    }

    #[test]
    fn test_distance_components() {
        let dist = DistanceComponents::new(0.65)
            .with_path(0.72)
            .with_embedding(0.58);

        assert_eq!(dist.combined, 0.65);
        assert_eq!(dist.path, Some(0.72));
        assert_eq!(dist.embedding, Some(0.58));
    }

    #[test]
    fn test_extract_prefix() {
        assert_eq!(extract_prefix("ChEBI:15365"), "ChEBI");
        assert_eq!(extract_prefix("GO:0008150"), "GO");
    }

    #[test]
    fn test_truncate_description() {
        assert_eq!(truncate_description("short", 50), "short");
        assert_eq!(
            truncate_description("this is a very long description that needs truncation", 20),
            "this is a very lo..."
        );
    }

    #[test]
    fn test_categorize_go_term() {
        assert_eq!(
            categorize_go_term("biological_process/metabolism"),
            "biological process"
        );
        assert_eq!(
            categorize_go_term("molecular_function/binding"),
            "molecular function"
        );
        assert_eq!(
            categorize_go_term("cellular_component/membrane"),
            "cellular component"
        );
    }
}
