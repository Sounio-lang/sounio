//! Ontology Generator using LLMs
//!
//! Extracts terms, builds taxonomies, and generates ontology fragments
//! from natural language text using LLM-based analysis.

use super::{GenerationError, GenerationResult};
use crate::llm::{LLMClientRegistry, OntologyTask};
use std::collections::HashMap;

/// Configuration for ontology generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Minimum confidence threshold for including terms
    pub min_confidence: f64,
    /// Maximum number of terms to extract per text
    pub max_terms: usize,
    /// Whether to generate taxonomy relationships
    pub generate_taxonomy: bool,
    /// Whether to extract non-taxonomic relations
    pub extract_relations: bool,
    /// Whether to generate definitions
    pub generate_definitions: bool,
    /// Domain context for generation
    pub domain: String,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_terms: 50,
            generate_taxonomy: true,
            extract_relations: true,
            generate_definitions: true,
            domain: "general".to_string(),
        }
    }
}

impl GenerationConfig {
    /// Create a config for a specific domain
    pub fn for_domain(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            ..Default::default()
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Enable or disable taxonomy generation
    pub fn with_taxonomy(mut self, enabled: bool) -> Self {
        self.generate_taxonomy = enabled;
        self
    }
}

/// Main ontology generator
pub struct OntologyGenerator {
    registry: LLMClientRegistry,
    config: GenerationConfig,
}

impl OntologyGenerator {
    /// Create a new generator with an LLM client registry
    pub fn new(registry: LLMClientRegistry) -> Self {
        Self {
            registry,
            config: GenerationConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(registry: LLMClientRegistry, config: GenerationConfig) -> Self {
        Self { registry, config }
    }

    /// Check if generation is available
    pub fn is_available(&self) -> bool {
        self.registry.is_available()
    }

    /// Generate an ontology fragment from natural language text
    pub fn generate_from_text(
        &self,
        text: &str,
        domain: &str,
    ) -> GenerationResult<GeneratedOntologyFragment> {
        if !self.is_available() {
            return Err(GenerationError::NotAvailable);
        }

        let mut fragment = GeneratedOntologyFragment::new(domain);
        let mut stats = GenerationStats::default();

        // Step 1: Extract terms
        let terms = self.extract_terms(text, domain)?;
        stats.terms_extracted = terms.len();

        if terms.is_empty() {
            return Err(GenerationError::NoTermsExtracted);
        }

        // Filter by confidence
        let filtered_terms: Vec<_> = terms
            .into_iter()
            .filter(|t| t.confidence >= self.config.min_confidence)
            .take(self.config.max_terms)
            .collect();

        stats.terms_accepted = filtered_terms.len();

        // Step 2: Type each term (classify into BFO categories)
        for term in &filtered_terms {
            match self.type_term(term, domain) {
                Ok(class) => {
                    fragment.classes.push(class);
                }
                Err(e) => {
                    // Log but continue with other terms
                    stats.typing_errors += 1;
                }
            }
        }

        // Step 3: Build taxonomy (if enabled)
        if self.config.generate_taxonomy && fragment.classes.len() >= 2 {
            let taxonomy = self.build_taxonomy(&fragment.classes, domain)?;
            for rel in taxonomy {
                fragment.axioms.push(GeneratedAxiom::SubClassOf {
                    subclass: rel.subclass,
                    superclass: rel.superclass,
                    confidence: rel.confidence,
                });
            }
            stats.taxonomy_relations = fragment
                .axioms
                .iter()
                .filter(|a| matches!(a, GeneratedAxiom::SubClassOf { .. }))
                .count();
        }

        // Step 4: Extract relations (if enabled)
        if self.config.extract_relations {
            let relations = self.extract_relations(text, &fragment.classes, domain)?;
            stats.relations_extracted = relations.len();
            for rel in relations {
                fragment
                    .axioms
                    .push(GeneratedAxiom::ObjectPropertyAssertion {
                        subject: rel.subject,
                        predicate: rel.predicate,
                        object: rel.object,
                        confidence: rel.confidence,
                    });
            }
        }

        // Step 5: Generate definitions (if enabled)
        if self.config.generate_definitions {
            for class in &mut fragment.classes {
                if class.definition.is_none()
                    && let Ok(def) = self.generate_definition(class, domain)
                {
                    class.definition = Some(def);
                    stats.definitions_generated += 1;
                }
            }
        }

        fragment.stats = stats;
        Ok(fragment)
    }

    /// Extract terms from text
    fn extract_terms(&self, text: &str, domain: &str) -> GenerationResult<Vec<ExtractedTerm>> {
        let mut params = HashMap::new();
        params.insert("domain".to_string(), domain.to_string());
        params.insert("text".to_string(), text.to_string());

        let request = self
            .registry
            .prompts()
            .build_prompt(&OntologyTask::TermExtraction, &params)
            .ok_or_else(|| GenerationError::ParseError("Failed to build prompt".into()))?;

        let response = self.registry.default_client()?.query(&request)?;

        // Parse JSON response
        self.parse_term_extraction_response(&response.content, response.estimated_confidence())
    }

    /// Parse the term extraction response from LLM
    fn parse_term_extraction_response(
        &self,
        content: &str,
        base_confidence: f64,
    ) -> GenerationResult<Vec<ExtractedTerm>> {
        // Find JSON in response
        let json_str = extract_json_from_response(content)?;

        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| GenerationError::ParseError(e.to_string()))?;

        let terms_array = parsed
            .get("terms")
            .and_then(|t| t.as_array())
            .ok_or_else(|| GenerationError::ParseError("Missing 'terms' array".into()))?;

        let mut terms = Vec::new();
        for term_obj in terms_array {
            let term = term_obj
                .get("term")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();

            let type_hint = term_obj
                .get("type_hint")
                .and_then(|t| t.as_str())
                .unwrap_or("class")
                .to_string();

            let confidence = term_obj
                .get("confidence")
                .and_then(|c| c.as_f64())
                .unwrap_or(0.5)
                * base_confidence; // Combine with response confidence

            if !term.is_empty() {
                terms.push(ExtractedTerm {
                    term,
                    type_hint,
                    confidence,
                    context: None,
                });
            }
        }

        Ok(terms)
    }

    /// Type a term using BFO classification
    fn type_term(&self, term: &ExtractedTerm, domain: &str) -> GenerationResult<GeneratedClass> {
        let mut params = HashMap::new();
        params.insert("term".to_string(), term.term.clone());
        params.insert("domain".to_string(), domain.to_string());
        params.insert(
            "context".to_string(),
            term.context.clone().unwrap_or_default(),
        );

        let request = self
            .registry
            .prompts()
            .build_prompt(&OntologyTask::TermTyping, &params)
            .ok_or_else(|| GenerationError::ParseError("Failed to build prompt".into()))?;

        let response = self.registry.default_client()?.query(&request)?;

        self.parse_term_typing_response(
            &response.content,
            &term.term,
            response.estimated_confidence(),
        )
    }

    /// Parse term typing response
    fn parse_term_typing_response(
        &self,
        content: &str,
        term: &str,
        base_confidence: f64,
    ) -> GenerationResult<GeneratedClass> {
        let json_str = extract_json_from_response(content)?;

        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| GenerationError::ParseError(e.to_string()))?;

        let bfo_category = parsed
            .get("bfo_category")
            .and_then(|c| c.as_str())
            .unwrap_or("BFO:0000001")
            .to_string();

        let bfo_label = parsed
            .get("bfo_label")
            .and_then(|l| l.as_str())
            .unwrap_or("entity")
            .to_string();

        let reasoning = parsed
            .get("reasoning")
            .and_then(|r| r.as_str())
            .map(|s| s.to_string());

        let confidence = parsed
            .get("confidence")
            .and_then(|c| c.as_f64())
            .unwrap_or(0.7)
            * base_confidence;

        Ok(GeneratedClass {
            name: term.to_string(),
            label: term.to_string(),
            bfo_parent: bfo_category,
            bfo_parent_label: bfo_label,
            definition: None,
            reasoning,
            confidence,
            provenance: LLMProvenance::new("term_typing"),
        })
    }

    /// Build taxonomy relationships between classes
    fn build_taxonomy(
        &self,
        classes: &[GeneratedClass],
        domain: &str,
    ) -> GenerationResult<Vec<TaxonomicRelation>> {
        let mut relations = Vec::new();

        // Compare each pair of classes
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                if let Ok(rel) = self.check_taxonomy_relation(&classes[i], &classes[j], domain)
                    && let Some(r) = rel
                {
                    relations.push(r);
                }
            }
        }

        Ok(relations)
    }

    /// Check taxonomic relationship between two classes
    fn check_taxonomy_relation(
        &self,
        class_a: &GeneratedClass,
        class_b: &GeneratedClass,
        domain: &str,
    ) -> GenerationResult<Option<TaxonomicRelation>> {
        let mut params = HashMap::new();
        params.insert("term_a".to_string(), class_a.name.clone());
        params.insert("term_b".to_string(), class_b.name.clone());
        params.insert("domain".to_string(), domain.to_string());

        let request = self
            .registry
            .prompts()
            .build_prompt(&OntologyTask::TaxonomyDiscovery, &params)
            .ok_or_else(|| GenerationError::ParseError("Failed to build prompt".into()))?;

        let response = self.registry.default_client()?.query(&request)?;

        self.parse_taxonomy_response(
            &response.content,
            class_a,
            class_b,
            response.estimated_confidence(),
        )
    }

    /// Parse taxonomy discovery response
    fn parse_taxonomy_response(
        &self,
        content: &str,
        class_a: &GeneratedClass,
        class_b: &GeneratedClass,
        base_confidence: f64,
    ) -> GenerationResult<Option<TaxonomicRelation>> {
        let json_str = extract_json_from_response(content)?;

        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| GenerationError::ParseError(e.to_string()))?;

        let relationship = parsed
            .get("relationship")
            .and_then(|r| r.as_str())
            .unwrap_or("none");

        let confidence = parsed
            .get("confidence")
            .and_then(|c| c.as_f64())
            .unwrap_or(0.5)
            * base_confidence;

        // Only accept high-confidence taxonomy relations
        if confidence < self.config.min_confidence {
            return Ok(None);
        }

        match relationship {
            "subclass_of" => Ok(Some(TaxonomicRelation {
                subclass: class_a.name.clone(),
                superclass: class_b.name.clone(),
                confidence,
            })),
            "superclass_of" => Ok(Some(TaxonomicRelation {
                subclass: class_b.name.clone(),
                superclass: class_a.name.clone(),
                confidence,
            })),
            _ => Ok(None),
        }
    }

    /// Extract non-taxonomic relations
    fn extract_relations(
        &self,
        text: &str,
        classes: &[GeneratedClass],
        domain: &str,
    ) -> GenerationResult<Vec<ExtractedRelation>> {
        let mut params = HashMap::new();
        params.insert("domain".to_string(), domain.to_string());
        params.insert("text".to_string(), text.to_string());

        let request = self
            .registry
            .prompts()
            .build_prompt(&OntologyTask::RelationExtraction, &params)
            .ok_or_else(|| GenerationError::ParseError("Failed to build prompt".into()))?;

        let response = self.registry.default_client()?.query(&request)?;

        self.parse_relation_extraction_response(&response.content, response.estimated_confidence())
    }

    /// Parse relation extraction response
    fn parse_relation_extraction_response(
        &self,
        content: &str,
        base_confidence: f64,
    ) -> GenerationResult<Vec<ExtractedRelation>> {
        let json_str = extract_json_from_response(content)?;

        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| GenerationError::ParseError(e.to_string()))?;

        let relations_array = parsed
            .get("relations")
            .and_then(|r| r.as_array())
            .unwrap_or(&Vec::new())
            .clone();

        let mut relations = Vec::new();
        for rel_obj in &relations_array {
            let subject = rel_obj
                .get("subject")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();

            let predicate = rel_obj
                .get("predicate")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();

            let object = rel_obj
                .get("object")
                .and_then(|o| o.as_str())
                .unwrap_or("")
                .to_string();

            let evidence = rel_obj
                .get("evidence")
                .and_then(|e| e.as_str())
                .map(|s| s.to_string());

            let confidence = rel_obj
                .get("confidence")
                .and_then(|c| c.as_f64())
                .unwrap_or(0.5)
                * base_confidence;

            if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                relations.push(ExtractedRelation {
                    subject,
                    predicate,
                    object,
                    evidence,
                    confidence,
                });
            }
        }

        Ok(relations)
    }

    /// Generate a definition for a class
    fn generate_definition(
        &self,
        class: &GeneratedClass,
        domain: &str,
    ) -> GenerationResult<String> {
        let mut params = HashMap::new();
        params.insert("term".to_string(), class.name.clone());
        params.insert("parent".to_string(), class.bfo_parent_label.clone());
        params.insert("domain".to_string(), domain.to_string());
        params.insert("context".to_string(), domain.to_string());

        let request = self
            .registry
            .prompts()
            .build_prompt(&OntologyTask::DefinitionGeneration, &params)
            .ok_or_else(|| GenerationError::ParseError("Failed to build prompt".into()))?;

        let response = self.registry.default_client()?.query(&request)?;

        // Extract definition from response
        let json_str = extract_json_from_response(&response.content)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| GenerationError::ParseError(e.to_string()))?;

        parsed
            .get("definition")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| GenerationError::ParseError("No definition in response".into()))
    }
}

/// Extract JSON object from LLM response (handles markdown code blocks)
fn extract_json_from_response(content: &str) -> GenerationResult<String> {
    // Try to find JSON in markdown code block
    if let Some(start) = content.find("```json")
        && let Some(end) = content[start..]
            .find("```\n")
            .or(content[start..].rfind("```"))
    {
        let json_start = start + 7; // Skip "```json"
        if json_start < start + end {
            return Ok(content[json_start..start + end].trim().to_string());
        }
    }

    // Try to find raw JSON object
    if let Some(start) = content.find('{')
        && let Some(end) = content.rfind('}')
        && start < end
    {
        return Ok(content[start..=end].to_string());
    }

    Err(GenerationError::ParseError(
        "Could not find JSON in response".into(),
    ))
}

// ============================================================================
// Data Structures
// ============================================================================

/// A term extracted from natural language
#[derive(Debug, Clone)]
pub struct ExtractedTerm {
    /// The term string
    pub term: String,
    /// Hint about what type this is (class, property, individual)
    pub type_hint: String,
    /// Confidence in extraction
    pub confidence: f64,
    /// Optional context where term was found
    pub context: Option<String>,
}

/// A generated ontology class
#[derive(Debug, Clone)]
pub struct GeneratedClass {
    /// Class name (identifier)
    pub name: String,
    /// Human-readable label
    pub label: String,
    /// BFO parent category (e.g., "BFO:0000040")
    pub bfo_parent: String,
    /// BFO parent label (e.g., "material entity")
    pub bfo_parent_label: String,
    /// Optional formal definition
    pub definition: Option<String>,
    /// Reasoning for classification
    pub reasoning: Option<String>,
    /// Confidence in classification
    pub confidence: f64,
    /// Provenance information
    pub provenance: LLMProvenance,
}

/// A generated ontology property
#[derive(Debug, Clone)]
pub struct GeneratedProperty {
    /// Property name
    pub name: String,
    /// Human-readable label
    pub label: String,
    /// Domain class
    pub domain: Option<String>,
    /// Range class
    pub range: Option<String>,
    /// RO parent relation
    pub ro_parent: Option<String>,
    /// Confidence
    pub confidence: f64,
    /// Provenance
    pub provenance: LLMProvenance,
}

/// A generated OWL axiom
#[derive(Debug, Clone)]
pub enum GeneratedAxiom {
    /// Subclass relationship
    SubClassOf {
        subclass: String,
        superclass: String,
        confidence: f64,
    },
    /// Equivalent class
    EquivalentClass {
        class1: String,
        class2: String,
        confidence: f64,
    },
    /// Disjoint classes
    DisjointWith {
        class1: String,
        class2: String,
        confidence: f64,
    },
    /// Object property assertion
    ObjectPropertyAssertion {
        subject: String,
        predicate: String,
        object: String,
        confidence: f64,
    },
    /// Annotation assertion (definition, label, etc.)
    AnnotationAssertion {
        subject: String,
        property: String,
        value: String,
    },
}

/// Taxonomic (is-a) relationship
#[derive(Debug, Clone)]
pub struct TaxonomicRelation {
    /// Subclass term
    pub subclass: String,
    /// Superclass term
    pub superclass: String,
    /// Confidence in relationship
    pub confidence: f64,
}

/// Non-taxonomic relation extracted from text
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    /// Subject entity
    pub subject: String,
    /// Predicate/relation type
    pub predicate: String,
    /// Object entity
    pub object: String,
    /// Text evidence
    pub evidence: Option<String>,
    /// Confidence
    pub confidence: f64,
}

/// A generated ontology fragment
#[derive(Debug, Clone)]
pub struct GeneratedOntologyFragment {
    /// Domain this fragment is for
    pub domain: String,
    /// Generated classes
    pub classes: Vec<GeneratedClass>,
    /// Generated properties
    pub properties: Vec<GeneratedProperty>,
    /// Generated axioms
    pub axioms: Vec<GeneratedAxiom>,
    /// Generation statistics
    pub stats: GenerationStats,
    /// Overall provenance
    pub provenance: LLMProvenance,
}

impl GeneratedOntologyFragment {
    /// Create a new empty fragment
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            classes: Vec::new(),
            properties: Vec::new(),
            axioms: Vec::new(),
            stats: GenerationStats::default(),
            provenance: LLMProvenance::new("ontology_generation"),
        }
    }

    /// Check if fragment is empty
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.properties.is_empty() && self.axioms.is_empty()
    }

    /// Get average confidence across all elements
    pub fn average_confidence(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0;

        for class in &self.classes {
            total += class.confidence;
            count += 1;
        }

        for prop in &self.properties {
            total += prop.confidence;
            count += 1;
        }

        for axiom in &self.axioms {
            total += match axiom {
                GeneratedAxiom::SubClassOf { confidence, .. }
                | GeneratedAxiom::EquivalentClass { confidence, .. }
                | GeneratedAxiom::DisjointWith { confidence, .. }
                | GeneratedAxiom::ObjectPropertyAssertion { confidence, .. } => *confidence,
                GeneratedAxiom::AnnotationAssertion { .. } => 1.0,
            };
            count += 1;
        }

        if count > 0 { total / count as f64 } else { 0.0 }
    }
}

/// Statistics about generation process
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Terms extracted from text
    pub terms_extracted: usize,
    /// Terms accepted (above confidence threshold)
    pub terms_accepted: usize,
    /// Errors during term typing
    pub typing_errors: usize,
    /// Taxonomy relations discovered
    pub taxonomy_relations: usize,
    /// Non-taxonomic relations extracted
    pub relations_extracted: usize,
    /// Definitions generated
    pub definitions_generated: usize,
}

/// Provenance information for LLM-generated content
#[derive(Debug, Clone)]
pub struct LLMProvenance {
    /// Task that generated this
    pub task: String,
    /// Timestamp of generation
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Model used (if known)
    pub model: Option<String>,
    /// Confidence from LLM response analysis
    pub response_confidence: Option<f64>,
}

impl LLMProvenance {
    /// Create new provenance with current timestamp
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            timestamp: chrono::Utc::now(),
            model: None,
            response_confidence: None,
        }
    }

    /// Set the model used
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set response confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.response_confidence = Some(confidence);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert!(config.min_confidence > 0.0);
        assert!(config.generate_taxonomy);
        assert!(config.extract_relations);
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::for_domain("biomaterials")
            .with_min_confidence(0.8)
            .with_taxonomy(false);

        assert_eq!(config.domain, "biomaterials");
        assert!((config.min_confidence - 0.8).abs() < f64::EPSILON);
        assert!(!config.generate_taxonomy);
    }

    #[test]
    fn test_extract_json_from_response() {
        // Test markdown code block
        let content = r#"Here is the result:
```json
{"terms": [{"term": "scaffold", "confidence": 0.9}]}
```
"#;
        let json = extract_json_from_response(content).unwrap();
        assert!(json.contains("scaffold"));

        // Test raw JSON
        let content2 = r#"The output is {"terms": [{"term": "test"}]}"#;
        let json2 = extract_json_from_response(content2).unwrap();
        assert!(json2.contains("test"));
    }

    #[test]
    fn test_generated_fragment_empty() {
        let fragment = GeneratedOntologyFragment::new("test");
        assert!(fragment.is_empty());
        assert_eq!(fragment.average_confidence(), 0.0);
    }

    #[test]
    fn test_generated_fragment_with_classes() {
        let mut fragment = GeneratedOntologyFragment::new("test");
        fragment.classes.push(GeneratedClass {
            name: "scaffold".to_string(),
            label: "Scaffold".to_string(),
            bfo_parent: "BFO:0000040".to_string(),
            bfo_parent_label: "material entity".to_string(),
            definition: None,
            reasoning: None,
            confidence: 0.9,
            provenance: LLMProvenance::new("test"),
        });

        assert!(!fragment.is_empty());
        assert!((fragment.average_confidence() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_llm_provenance() {
        let prov = LLMProvenance::new("term_extraction")
            .with_model("gpt-4")
            .with_confidence(0.85);

        assert_eq!(prov.task, "term_extraction");
        assert_eq!(prov.model, Some("gpt-4".to_string()));
        assert!((prov.response_confidence.unwrap() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_extracted_term() {
        let term = ExtractedTerm {
            term: "porous scaffold".to_string(),
            type_hint: "class".to_string(),
            confidence: 0.92,
            context: Some("tissue engineering".to_string()),
        };

        assert_eq!(term.term, "porous scaffold");
        assert!(term.confidence > 0.9);
    }
}
