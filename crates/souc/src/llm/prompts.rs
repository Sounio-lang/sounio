//! Prompt Templates for Ontology Learning
//!
//! Based on LLMs4OL 2025 Challenge findings:
//! - Chain-of-Thought prompting improves term typing accuracy
//! - Few-shot examples improve taxonomy discovery
//! - RAG integration improves non-taxonomic relation extraction
//!
//! # Task Types
//!
//! | Task | Description | Typical Accuracy |
//! |------|-------------|------------------|
//! | Term Extraction | Extract domain terms from text | ~85% |
//! | Term Typing | Classify terms into BFO categories | ~78% |
//! | Taxonomy Discovery | Find is-a relationships | ~72% |
//! | Relation Extraction | Find non-taxonomic relations | ~65% |
//! | Competency to OWL | Generate OWL from questions | ~60% |
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::llm::prompts::{PromptTemplates, OntologyTask};
//! use std::collections::HashMap;
//!
//! let templates = PromptTemplates::default();
//!
//! let mut params = HashMap::new();
//! params.insert("domain".to_string(), "biomaterials".to_string());
//! params.insert("text".to_string(), "Porous scaffolds support cell migration.".to_string());
//!
//! let request = templates.build_prompt(&OntologyTask::TermExtraction, &params)?;
//! ```

use super::LLMRequest;
use std::collections::HashMap;

/// Prompt template types for ontology learning tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OntologyTask {
    /// Extract domain terms from natural language text
    TermExtraction,
    /// Classify terms into BFO ontological categories
    TermTyping,
    /// Discover taxonomic (is-a) relationships between terms
    TaxonomyDiscovery,
    /// Extract non-taxonomic relations (part-of, causes, etc.)
    RelationExtraction,
    /// Generate OWL axioms from competency questions
    CompetencyToOWL,
    /// Validate generated ontology for consistency
    OntologyValidation,
    /// Suggest missing terms or relations
    OntologyEnrichment,
    /// Generate natural language definitions for terms
    DefinitionGeneration,
}

impl std::fmt::Display for OntologyTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OntologyTask::TermExtraction => write!(f, "Term Extraction"),
            OntologyTask::TermTyping => write!(f, "Term Typing"),
            OntologyTask::TaxonomyDiscovery => write!(f, "Taxonomy Discovery"),
            OntologyTask::RelationExtraction => write!(f, "Relation Extraction"),
            OntologyTask::CompetencyToOWL => write!(f, "Competency to OWL"),
            OntologyTask::OntologyValidation => write!(f, "Ontology Validation"),
            OntologyTask::OntologyEnrichment => write!(f, "Ontology Enrichment"),
            OntologyTask::DefinitionGeneration => write!(f, "Definition Generation"),
        }
    }
}

/// Prompt template manager
#[derive(Debug, Clone)]
pub struct PromptTemplates {
    templates: HashMap<OntologyTask, PromptTemplate>,
}

/// A single prompt template with placeholders
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// System message defining the LLM's role
    pub system: String,
    /// User message template with {placeholders}
    pub user_template: String,
    /// Few-shot examples for in-context learning
    pub examples: Vec<PromptExample>,
    /// Chain-of-thought instruction (prepended to user message)
    pub cot_instruction: Option<String>,
    /// Expected output format description
    pub output_format: String,
    /// Recommended temperature for this task
    pub recommended_temperature: f32,
}

/// A few-shot example for in-context learning
#[derive(Debug, Clone)]
pub struct PromptExample {
    /// Example input
    pub input: String,
    /// Expected output
    pub output: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        let mut templates = HashMap::new();

        // ====================================================================
        // Term Extraction (Text2Onto)
        // ====================================================================
        templates.insert(
            OntologyTask::TermExtraction,
            PromptTemplate {
                system: r#"You are an expert ontology engineer specializing in biomedical and scientific domains.
Your task is to extract domain-specific terms from text that could form concepts in an ontology.

Focus on:
- Nouns and noun phrases that represent domain concepts
- Technical terms and jargon specific to the domain
- Named entities (substances, processes, structures)

Exclude:
- Common words and stopwords
- Verbs and adjectives (unless they form compound terms)
- Pronouns and determiners"#
                    .to_string(),

                user_template: r#"Extract ontology terms from the following text. Output as JSON.

Domain context: {domain}

Text:
{text}

Output format:
{{
  "terms": [
    {{"term": "...", "type_hint": "class|property|individual", "confidence": 0.0-1.0}}
  ]
}}"#
                    .to_string(),

                examples: vec![PromptExample {
                    input: r#"Domain: biomaterials
Text: Porous scaffolds with interconnected porosity support cell migration and tissue ingrowth during bone regeneration."#
                        .to_string(),
                    output: r#"{"terms": [
  {"term": "porous scaffold", "type_hint": "class", "confidence": 0.95},
  {"term": "interconnected porosity", "type_hint": "property", "confidence": 0.85},
  {"term": "cell migration", "type_hint": "class", "confidence": 0.92},
  {"term": "tissue ingrowth", "type_hint": "class", "confidence": 0.88},
  {"term": "bone regeneration", "type_hint": "class", "confidence": 0.94}
]}"#
                    .to_string(),
                }],

                cot_instruction: Some(
                    r#"Think step by step:
1. Identify all noun phrases in the text
2. Filter for domain-specific technical terms
3. Classify each term as class (concept), property (attribute), or individual (instance)
4. Assess confidence based on how clearly the term fits the domain"#
                        .to_string(),
                ),

                output_format: "JSON with terms array".to_string(),
                recommended_temperature: 0.2,
            },
        );

        // ====================================================================
        // Term Typing (BFO Classification)
        // ====================================================================
        templates.insert(
            OntologyTask::TermTyping,
            PromptTemplate {
                system: r#"You are an expert in ontology design using BFO (Basic Formal Ontology) as the upper ontology.

BFO divides entities into:
- Continuants: Entities that persist through time while maintaining identity
  - Independent continuant: Material entities (objects, organisms)
  - Dependent continuant: Qualities, roles, functions, dispositions
- Occurrents: Entities that unfold through time (processes, events)

Key BFO classes:
- BFO:0000040 material entity (physical objects)
- BFO:0000019 quality (inherent attributes)
- BFO:0000023 role (externally grounded realizable)
- BFO:0000034 function (internally grounded realizable)
- BFO:0000016 disposition (tendency to behave)
- BFO:0000015 process (temporal unfolding)"#
                    .to_string(),

                user_template: r#"Classify the following term according to BFO categories.

Term: {term}
Domain: {domain}
Context: {context}

Output format:
{{
  "term": "...",
  "bfo_category": "BFO:XXXXXXX",
  "bfo_label": "...",
  "reasoning": "...",
  "confidence": 0.0-1.0
}}"#
                    .to_string(),

                examples: vec![
                    PromptExample {
                        input: r#"Term: scaffold
Domain: biomaterials
Context: 3D printed scaffolds for bone regeneration"#
                            .to_string(),
                        output: r#"{"term": "scaffold", "bfo_category": "BFO:0000040", "bfo_label": "material entity", "reasoning": "A scaffold is a physical object with spatial extent that persists through time. It is manufactured and has measurable properties.", "confidence": 0.95}"#
                            .to_string(),
                    },
                    PromptExample {
                        input: r#"Term: porosity
Domain: biomaterials
Context: high porosity scaffold for vascularization"#
                            .to_string(),
                        output: r#"{"term": "porosity", "bfo_category": "BFO:0000019", "bfo_label": "quality", "reasoning": "Porosity is an inherent quality of a material that describes the proportion of void space. It inheres in the scaffold.", "confidence": 0.92}"#
                            .to_string(),
                    },
                ],

                cot_instruction: Some(
                    r#"Reason step by step:
1. Does this entity have spatial parts and persist through time? (continuant vs occurrent)
2. If continuant: Is it independent (can exist on its own) or dependent (needs a bearer)?
3. If dependent: Is it a quality (inherent), role (external), function (internal purpose), or disposition (tendency)?
4. If occurrent: Is it a process, temporal region, or spatiotemporal region?
5. Justify with domain-specific reasoning"#
                        .to_string(),
                ),

                output_format: "JSON with BFO classification".to_string(),
                recommended_temperature: 0.1,
            },
        );

        // ====================================================================
        // Taxonomy Discovery
        // ====================================================================
        templates.insert(
            OntologyTask::TaxonomyDiscovery,
            PromptTemplate {
                system: r#"You are an expert in building taxonomies (is-a hierarchies) for scientific ontologies.

Determine if one term is a subclass of another based on their definitions and domain knowledge.
Use strict subsumption: A is-a B means every instance of A is necessarily also an instance of B.

Key principles:
- Transitivity: if A is-a B and B is-a C, then A is-a C
- Single inheritance preferred when possible
- Avoid circular definitions
- Consider necessary vs sufficient conditions"#
                    .to_string(),

                user_template: r#"Determine the taxonomic relationship between these terms.

Term A: {term_a}
Term B: {term_b}
Domain: {domain}

Consider:
- Is every {term_a} necessarily a {term_b}? (subclass_of)
- Is every {term_b} necessarily a {term_a}? (superclass_of)
- Are they logically equivalent? (equivalent)
- Are they mutually exclusive? (disjoint)
- No clear relationship? (none)

Output format:
{{
  "term_a": "...",
  "term_b": "...",
  "relationship": "subclass_of|superclass_of|equivalent|disjoint|none",
  "reasoning": "...",
  "confidence": 0.0-1.0
}}"#
                    .to_string(),

                examples: vec![PromptExample {
                    input: r#"Term A: ceramic scaffold
Term B: scaffold
Domain: biomaterials"#
                        .to_string(),
                    output: r#"{"term_a": "ceramic scaffold", "term_b": "scaffold", "relationship": "subclass_of", "reasoning": "A ceramic scaffold is a specific type of scaffold made from ceramic materials. Every ceramic scaffold is necessarily a scaffold, but not every scaffold is ceramic (could be polymer, metal, etc.).", "confidence": 0.96}"#
                        .to_string(),
                }],

                cot_instruction: Some(
                    r#"Consider step by step:
1. Define term A precisely in this domain
2. Define term B precisely in this domain
3. Can you think of any instance of A that is NOT an instance of B? (if yes, not subclass)
4. Can you think of any instance of B that is NOT an instance of A? (if yes, not superclass)
5. What is the most specific accurate relationship?"#
                        .to_string(),
                ),

                output_format: "JSON with relationship and reasoning".to_string(),
                recommended_temperature: 0.15,
            },
        );

        // ====================================================================
        // Relation Extraction
        // ====================================================================
        templates.insert(
            OntologyTask::RelationExtraction,
            PromptTemplate {
                system: r#"You are an expert in extracting semantic relations from scientific text for ontology construction.

Common relation types from RO (Relation Ontology):
- part_of: X is a component of Y
- has_part: inverse of part_of
- participates_in: X is involved in process Y
- has_participant: inverse
- located_in: X is spatially within Y
- causes: X directly leads to Y
- regulates: X modulates Y
- derives_from: X originates from Y
- adjacent_to: X is next to Y
- capable_of: X has the ability to do Y"#
                    .to_string(),

                user_template: r#"Extract semantic relations from this text.

Domain: {domain}
Text: {text}

Output format:
{{
  "relations": [
    {{
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "evidence": "quoted text supporting this",
      "confidence": 0.0-1.0
    }}
  ]
}}"#
                    .to_string(),

                examples: vec![PromptExample {
                    input: r#"Domain: cell biology
Text: Osteoblasts are located in bone tissue and participate in bone formation by secreting collagen."#
                        .to_string(),
                    output: r#"{"relations": [
  {"subject": "osteoblast", "predicate": "located_in", "object": "bone tissue", "evidence": "Osteoblasts are located in bone tissue", "confidence": 0.95},
  {"subject": "osteoblast", "predicate": "participates_in", "object": "bone formation", "evidence": "participate in bone formation", "confidence": 0.92},
  {"subject": "osteoblast", "predicate": "capable_of", "object": "collagen secretion", "evidence": "by secreting collagen", "confidence": 0.88}
]}"#
                        .to_string(),
                }],

                cot_instruction: Some(
                    r#"For each potential relation:
1. Identify subject and object entities in the text
2. Determine the semantic relationship type from RO vocabulary
3. Quote the textual evidence
4. Assess confidence based on explicitness of the relation"#
                        .to_string(),
                ),

                output_format: "JSON with relations array".to_string(),
                recommended_temperature: 0.2,
            },
        );

        // ====================================================================
        // Competency Questions to OWL
        // ====================================================================
        templates.insert(
            OntologyTask::CompetencyToOWL,
            PromptTemplate {
                system: r#"You are an expert ontology engineer who converts natural language competency questions into OWL axioms.

Use Manchester syntax for OWL expressions:
- Class definitions: Class: ClassName SubClassOf: ParentClass
- Restrictions: SomeProperty some ValueClass
- Intersections: Class1 and Class2
- Unions: Class1 or Class2
- Disjointness: DisjointWith: Class2

Ensure logical consistency with existing ontology classes and properties."#
                    .to_string(),

                user_template: r#"Generate OWL axioms to answer this competency question.

Competency Question: {question}
Existing Classes: {existing_classes}
Existing Properties: {existing_properties}
Domain: {domain}

Output format:
{{
  "interpretation": "How this question should be answered by the ontology",
  "new_classes": ["list of new classes needed"],
  "new_properties": ["list of new properties needed"],
  "axioms": ["list of Manchester syntax axioms"],
  "reasoning": "Why these axioms answer the question",
  "confidence": 0.0-1.0
}}"#
                    .to_string(),

                examples: vec![],

                cot_instruction: Some(
                    r#"Process step by step:
1. Parse the question to identify what is being asked
2. Map question components to ontology concepts
3. Identify if new classes or properties are needed
4. Construct OWL axioms in Manchester syntax
5. Verify the axioms can answer the original question"#
                        .to_string(),
                ),

                output_format: "JSON with OWL axioms".to_string(),
                recommended_temperature: 0.15,
            },
        );

        // ====================================================================
        // Ontology Validation
        // ====================================================================
        templates.insert(
            OntologyTask::OntologyValidation,
            PromptTemplate {
                system: r#"You are an expert ontology validator. Check ontologies for:
1. Logical consistency (no contradictions)
2. Structural issues (orphan classes, missing definitions)
3. Naming conventions (CamelCase classes, snake_case properties)
4. Documentation completeness (labels, definitions, examples)
5. Best practices (single inheritance where possible, proper use of restrictions)"#
                    .to_string(),

                user_template: r#"Validate this ontology fragment for issues.

Ontology:
{ontology}

Domain: {domain}

Check for:
- Logical inconsistencies
- Structural problems
- Missing documentation
- Naming convention violations
- Best practice violations

Output format:
{{
  "issues": [
    {{
      "severity": "error|warning|suggestion",
      "type": "consistency|structure|documentation|naming|best_practice",
      "element": "affected class/property",
      "message": "description of issue",
      "suggestion": "how to fix"
    }}
  ],
  "summary": {{
    "errors": 0,
    "warnings": 0,
    "suggestions": 0,
    "overall_quality": 0.0-1.0
  }}
}}"#
                .to_string(),

                examples: vec![],
                cot_instruction: None,
                output_format: "JSON with validation results".to_string(),
                recommended_temperature: 0.1,
            },
        );

        // ====================================================================
        // Ontology Enrichment
        // ====================================================================
        templates.insert(
            OntologyTask::OntologyEnrichment,
            PromptTemplate {
                system: r#"You are an expert ontology engineer specializing in ontology enrichment and completion.
Your task is to suggest missing concepts, relationships, or axioms that would improve ontology coverage and usefulness.

Consider:
- Common domain concepts that might be missing
- Useful subclass hierarchies
- Important relationships between existing concepts
- Symmetric, transitive, or inverse property declarations"#
                    .to_string(),

                user_template: r#"Suggest enrichments for this ontology.

Current classes: {classes}
Current properties: {properties}
Domain: {domain}

Suggest:
1. Missing subclasses for existing classes
2. Missing sibling classes
3. Missing relationships
4. Missing property declarations

Output format:
{{
  "suggestions": [
    {{
      "type": "new_class|new_property|new_axiom|property_characteristic",
      "suggestion": "...",
      "rationale": "why this would be useful",
      "priority": "high|medium|low",
      "confidence": 0.0-1.0
    }}
  ]
}}"#
                    .to_string(),

                examples: vec![],
                cot_instruction: Some(
                    r#"Consider:
1. What important domain concepts are not yet represented?
2. Are there obvious subtype hierarchies missing?
3. What relationships typically exist between these concepts?
4. Are there property characteristics (symmetry, transitivity) that should be declared?"#
                        .to_string(),
                ),
                output_format: "JSON with enrichment suggestions".to_string(),
                recommended_temperature: 0.3,
            },
        );

        // ====================================================================
        // Definition Generation
        // ====================================================================
        templates.insert(
            OntologyTask::DefinitionGeneration,
            PromptTemplate {
                system: r#"You are an expert at writing precise ontology definitions following Aristotelian form:
"A [term] is a [genus] that [differentia]"

Good definitions:
- Use the immediate parent class as genus
- Include necessary and sufficient conditions
- Are unambiguous and precise
- Avoid circular references
- Use controlled vocabulary from the ontology"#
                    .to_string(),

                user_template: r#"Generate a formal definition for this term.

Term: {term}
Parent class: {parent}
Domain: {domain}
Context: {context}

Output format:
{{
  "term": "...",
  "definition": "A [term] is a [genus] that [differentia]",
  "genus": "immediate parent class used",
  "differentia": ["distinguishing characteristics"],
  "necessary_conditions": ["conditions that must hold"],
  "sufficient_conditions": ["conditions that guarantee membership"],
  "examples": ["instances of this class"],
  "counterexamples": ["things that are NOT instances"],
  "confidence": 0.0-1.0
}}"#
                    .to_string(),

                examples: vec![PromptExample {
                    input: r#"Term: ceramic scaffold
Parent class: scaffold
Domain: biomaterials
Context: Used in bone tissue engineering"#
                        .to_string(),
                    output: r#"{"term": "ceramic scaffold", "definition": "A ceramic scaffold is a scaffold that is composed primarily of ceramic materials and is designed for tissue engineering applications.", "genus": "scaffold", "differentia": ["composed primarily of ceramic materials", "designed for tissue engineering"], "necessary_conditions": ["must be a porous structure", "must be made of ceramic (e.g., hydroxyapatite, tricalcium phosphate)"], "sufficient_conditions": ["porous structure made of biocompatible ceramic for tissue engineering"], "examples": ["hydroxyapatite scaffold", "bioglass scaffold", "TCP scaffold"], "counterexamples": ["PLGA scaffold (polymer)", "titanium mesh (metal)"], "confidence": 0.91}"#
                        .to_string(),
                }],

                cot_instruction: Some(
                    r#"To create a good definition:
1. Identify the immediate parent class (genus)
2. List characteristics that distinguish this from sibling classes (differentia)
3. Determine necessary conditions (what MUST be true)
4. Determine sufficient conditions (what guarantees membership)
5. Think of clear examples and counterexamples"#
                        .to_string(),
                ),

                output_format: "JSON with Aristotelian definition".to_string(),
                recommended_temperature: 0.2,
            },
        );

        Self { templates }
    }
}

impl PromptTemplates {
    /// Get a template by task type
    pub fn get(&self, task: &OntologyTask) -> Option<&PromptTemplate> {
        self.templates.get(task)
    }

    /// Get mutable template for customization
    pub fn get_mut(&mut self, task: &OntologyTask) -> Option<&mut PromptTemplate> {
        self.templates.get_mut(task)
    }

    /// Register a custom template
    pub fn register(&mut self, task: OntologyTask, template: PromptTemplate) {
        self.templates.insert(task, template);
    }

    /// Build a complete LLM request from template and parameters
    ///
    /// Parameters are substituted for {placeholder} strings in the template.
    pub fn build_prompt(
        &self,
        task: &OntologyTask,
        params: &HashMap<String, String>,
    ) -> Option<LLMRequest> {
        let template = self.get(task)?;

        let mut user_message = template.user_template.clone();
        for (key, value) in params {
            user_message = user_message.replace(&format!("{{{}}}", key), value);
        }

        // Add chain-of-thought instruction if available
        if let Some(cot) = &template.cot_instruction {
            user_message = format!("{}\n\n{}", cot, user_message);
        }

        // Add few-shot examples if available
        if !template.examples.is_empty() {
            let examples_text: String = template
                .examples
                .iter()
                .enumerate()
                .map(|(i, ex)| {
                    format!(
                        "Example {}:\nInput:\n{}\n\nOutput:\n{}",
                        i + 1,
                        ex.input,
                        ex.output
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n---\n\n");
            user_message = format!("{}\n\n---\n\nNow process:\n{}", examples_text, user_message);
        }

        Some(LLMRequest {
            prompt: user_message,
            system: Some(template.system.clone()),
            temperature: template.recommended_temperature,
            max_tokens: 4096,
            stop_sequences: vec!["```".to_string(), "\n\n\n".to_string()],
        })
    }

    /// List all available task types
    pub fn available_tasks(&self) -> Vec<OntologyTask> {
        self.templates.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_templates() {
        let templates = PromptTemplates::default();

        // All core tasks should have templates
        assert!(templates.get(&OntologyTask::TermExtraction).is_some());
        assert!(templates.get(&OntologyTask::TermTyping).is_some());
        assert!(templates.get(&OntologyTask::TaxonomyDiscovery).is_some());
        assert!(templates.get(&OntologyTask::RelationExtraction).is_some());
        assert!(templates.get(&OntologyTask::CompetencyToOWL).is_some());
        assert!(templates.get(&OntologyTask::OntologyValidation).is_some());
    }

    #[test]
    fn test_build_prompt_term_extraction() {
        let templates = PromptTemplates::default();

        let mut params = HashMap::new();
        params.insert("domain".to_string(), "biomaterials".to_string());
        params.insert(
            "text".to_string(),
            "Scaffolds support cell growth.".to_string(),
        );

        let request = templates
            .build_prompt(&OntologyTask::TermExtraction, &params)
            .unwrap();

        assert!(request.prompt.contains("biomaterials"));
        assert!(request.prompt.contains("Scaffolds support cell growth"));
        assert!(request.system.is_some());
        assert!(request.temperature < 0.5); // Should be low for precision
    }

    #[test]
    fn test_build_prompt_with_examples() {
        let templates = PromptTemplates::default();

        let mut params = HashMap::new();
        params.insert("term".to_string(), "scaffold".to_string());
        params.insert("domain".to_string(), "biomaterials".to_string());
        params.insert("context".to_string(), "tissue engineering".to_string());

        let request = templates
            .build_prompt(&OntologyTask::TermTyping, &params)
            .unwrap();

        // Should contain few-shot examples
        assert!(request.prompt.contains("Example"));
        assert!(request.prompt.contains("BFO:0000040"));
    }

    #[test]
    fn test_build_prompt_missing_params() {
        let templates = PromptTemplates::default();

        let params = HashMap::new(); // Empty - missing required params

        let request = templates
            .build_prompt(&OntologyTask::TermExtraction, &params)
            .unwrap();

        // Placeholders should remain if not substituted
        assert!(request.prompt.contains("{domain}"));
        assert!(request.prompt.contains("{text}"));
    }

    #[test]
    fn test_ontology_task_display() {
        assert_eq!(
            format!("{}", OntologyTask::TermExtraction),
            "Term Extraction"
        );
        assert_eq!(format!("{}", OntologyTask::TermTyping), "Term Typing");
        assert_eq!(
            format!("{}", OntologyTask::TaxonomyDiscovery),
            "Taxonomy Discovery"
        );
    }

    #[test]
    fn test_custom_template() {
        let mut templates = PromptTemplates::default();

        let custom = PromptTemplate {
            system: "Custom system".to_string(),
            user_template: "Custom user {param}".to_string(),
            examples: vec![],
            cot_instruction: None,
            output_format: "JSON".to_string(),
            recommended_temperature: 0.5,
        };

        templates.register(OntologyTask::OntologyEnrichment, custom);

        let mut params = HashMap::new();
        params.insert("param".to_string(), "value".to_string());

        let request = templates
            .build_prompt(&OntologyTask::OntologyEnrichment, &params)
            .unwrap();

        assert!(request.prompt.contains("Custom user value"));
    }
}
