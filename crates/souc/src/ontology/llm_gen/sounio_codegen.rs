//! Sounio Struct Generation from Ontology
//!
//! Generates Sounio struct definitions from ontology classes.
//! Uses ontology term definitions, superclasses, and relations
//! to create type-safe Sounio code with Knowledge<T> types.
//!
//! # Example
//!
//! ```sounio
//! /// A benzoic acid derivative
//! /// @ontology CHEBI:15365
//! struct Aspirin {
//!     benzene_ring: BenzeneRing,
//!     role: Knowledge<AnalgesicRole>,
//! }
//! ```

use super::GenerationError;
use crate::ontology::{OntologyResolver, ResolverConfig};

/// Field naming style for generated structs
#[derive(Clone, Copy, Debug, Default)]
pub enum FieldStyle {
    #[default]
    SnakeCase,
    CamelCase,
}

/// Configuration for Sounio struct generation
#[derive(Clone, Debug)]
pub struct SounioCodegenConfig {
    /// Include definition as doc comment
    pub include_comments: bool,
    /// Include @ontology annotation
    pub include_ontology_annotations: bool,
    /// Field naming style
    pub field_style: FieldStyle,
    /// Generate Knowledge<T> wrappers for ontology types
    pub use_knowledge_types: bool,
}

impl Default for SounioCodegenConfig {
    fn default() -> Self {
        Self {
            include_comments: true,
            include_ontology_annotations: true,
            field_style: FieldStyle::SnakeCase,
            use_knowledge_types: true,
        }
    }
}

/// Generator for Sounio structs from ontology classes
pub struct SounioStructGenerator {
    resolver: OntologyResolver,
    config: SounioCodegenConfig,
}

impl SounioStructGenerator {
    /// Create a new generator with default resolver
    pub fn new() -> Result<Self, GenerationError> {
        let config = ResolverConfig::default().offline();
        let resolver = OntologyResolver::new(config)
            .map_err(|e| GenerationError::Custom(format!("Failed to create resolver: {}", e)))?;
        Ok(Self {
            resolver,
            config: SounioCodegenConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: SounioCodegenConfig) -> Result<Self, GenerationError> {
        let resolver_config = ResolverConfig::default().offline();
        let resolver = OntologyResolver::new(resolver_config)
            .map_err(|e| GenerationError::Custom(format!("Failed to create resolver: {}", e)))?;
        Ok(Self { resolver, config })
    }

    /// Generate a Sounio struct definition from an ontology CURIE
    pub fn generate_struct(&mut self, curie: &str) -> Result<String, GenerationError> {
        // Resolve the term
        let term = self
            .resolver
            .resolve(curie)
            .map_err(|e| GenerationError::Custom(format!("Failed to resolve {}: {}", curie, e)))?;

        let local = local_name(curie);
        let struct_name = to_pascal_case(&local);

        let mut lines = Vec::new();

        // Add definition doc comment if enabled
        if self.config.include_comments {
            if let Some(def) = &term.definition {
                for line in def.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        lines.push(format!("/// {}", trimmed));
                    }
                }
            }
        }

        // Add ontology annotation comment if enabled
        if self.config.include_ontology_annotations {
            lines.push(format!("/// @ontology {}", curie));
        }

        // Struct header
        lines.push(format!("struct {} {{", struct_name));

        // Generate fields from superclasses
        // Each direct superclass becomes a field representing the parent type
        for superclass in &term.superclasses {
            let parent_local = local_name(superclass);
            let parent_type = to_pascal_case(&parent_local);
            let field_name = to_field_name(&parent_local, self.config.field_style);

            if !field_name.is_empty() && !parent_type.is_empty() {
                if self.config.use_knowledge_types {
                    lines.push(format!("    {}: Knowledge<{}>,", field_name, parent_type));
                } else {
                    lines.push(format!("    {}: {},", field_name, parent_type));
                }
            }
        }

        // Add label as optional display name
        if let Some(label) = &term.label {
            if self.config.include_comments {
                lines.push(format!("    // label: \"{}\"", label));
            }
        }

        lines.push("}".to_string());

        Ok(lines.join("\n"))
    }

    /// Generate multiple structs for related terms
    pub fn generate_hierarchy(&mut self, root_curie: &str) -> Result<Vec<String>, GenerationError> {
        let mut results = Vec::new();

        // Generate root struct
        let root_struct = self.generate_struct(root_curie)?;
        results.push(root_struct);

        // Resolve term to get superclasses
        if let Ok(term) = self.resolver.resolve(root_curie) {
            // Generate structs for each superclass (limited depth to prevent explosion)
            for superclass in term.superclasses.iter().take(3) {
                if let Ok(parent_struct) = self.generate_struct(superclass) {
                    if !results.contains(&parent_struct) {
                        results.push(parent_struct);
                    }
                }
            }
        }

        Ok(results)
    }
}

/// Extract local name from CURIE (part after the colon)
fn local_name(curie: &str) -> String {
    curie
        .rsplit_once(':')
        .map_or_else(|| curie.to_string(), |(_, local)| local.to_string())
}

/// Convert string to PascalCase for type names
fn to_pascal_case(s: &str) -> String {
    let words: Vec<String> = s
        .split(|c: char| c == '_' || c == '-' || c.is_whitespace())
        .filter(|w| !w.trim().is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect();
    words.join("")
}

/// Convert string to field name according to style
fn to_field_name(label: &str, style: FieldStyle) -> String {
    let lowered = label.to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| c.is_whitespace() || c == '_' || c == '-')
        .filter(|w| !w.trim().is_empty())
        .collect();

    if words.is_empty() {
        return String::new();
    }

    match style {
        FieldStyle::SnakeCase => words.join("_"),
        FieldStyle::CamelCase => {
            let mut camel = words[0].to_string();
            for word in &words[1..] {
                let mut chars = word.chars();
                if let Some(first) = chars.next() {
                    camel.push_str(&first.to_uppercase().to_string());
                }
                camel.push_str(chars.as_str());
            }
            camel
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_name() {
        assert_eq!(local_name("CHEBI:15365"), "15365");
        assert_eq!(local_name("GO:0008150"), "0008150");
        assert_eq!(local_name("BFO:0000001"), "0000001");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("benzene_ring"), "BenzeneRing");
        assert_eq!(to_pascal_case("analgesic-role"), "AnalgesicRole");
        assert_eq!(to_pascal_case("cell migration"), "CellMigration");
    }

    #[test]
    fn test_to_field_name() {
        assert_eq!(
            to_field_name("Benzene Ring", FieldStyle::SnakeCase),
            "benzene_ring"
        );
        assert_eq!(
            to_field_name("Benzene Ring", FieldStyle::CamelCase),
            "benzeneRing"
        );
    }
}
