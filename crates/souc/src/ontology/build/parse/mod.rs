//! Ontology file parsers
//!
//! Supports: OBO, OWL/RDF, JSON-LD, FHIR JSON
//!
//! All parsers implement streaming iteration to handle large files
//! without loading everything into memory.

pub mod fhir;
pub mod jsonld;
pub mod obo;
pub mod owl;

use std::io::Read;
use std::path::Path;

/// A relation between terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    /// The predicate/relationship type IRI
    pub predicate: String,
    /// The target term IRI
    pub target: String,
}

/// A raw term as parsed from source files
#[derive(Debug, Clone, Default)]
pub struct RawTerm {
    /// Full IRI (e.g., "http://purl.obolibrary.org/obo/CHEBI_15365")
    pub iri: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Definition/description
    pub definition: Option<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Parent terms (is-a relations)
    pub parents: Vec<String>,
    /// Other relations (part-of, has-role, etc.)
    pub relations: Vec<Relation>,
    /// Deprecated flag
    pub deprecated: bool,
    /// Source ontology identifier
    pub source: String,
    /// Additional metadata as key-value pairs
    pub metadata: std::collections::HashMap<String, String>,
}

impl RawTerm {
    /// Create a new raw term with just an IRI
    pub fn new(iri: impl Into<String>) -> Self {
        RawTerm {
            iri: iri.into(),
            label: None,
            definition: None,
            synonyms: Vec::new(),
            parents: Vec::new(),
            relations: Vec::new(),
            deprecated: false,
            source: String::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Add a parent
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parents.push(parent.into());
        self
    }
}

/// Parser trait for ontology formats
pub trait OntologyParser: Send + Sync {
    /// Parse from a reader, yielding terms incrementally
    fn parse<'a>(
        &'a self,
        reader: Box<dyn Read + 'a>,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a>;

    /// File extensions this parser handles
    fn extensions(&self) -> &[&str];

    /// Detect if this parser can handle the given file
    fn can_parse(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                self.extensions()
                    .iter()
                    .any(|ext| ext.eq_ignore_ascii_case(e))
            })
            .unwrap_or(false)
    }

    /// Parser name for diagnostics
    fn name(&self) -> &str;
}

/// Errors that can occur during parsing
#[derive(Debug, Clone)]
pub enum ParseError {
    /// IO error
    IoError(String),
    /// Invalid format at specific location
    Format { line: usize, message: String },
    /// Invalid format (generic)
    InvalidFormat(String),
    /// Missing required field
    MissingField(String),
    /// Invalid IRI format
    InvalidIri(String),
    /// JSON parsing error
    Json(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::IoError(msg) => write!(f, "IO error: {}", msg),
            ParseError::Format { line, message } => {
                write!(f, "Format error at line {}: {}", line, message)
            }
            ParseError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ParseError::MissingField(field) => write!(f, "Missing required field: {}", field),
            ParseError::InvalidIri(iri) => write!(f, "Invalid IRI: {}", iri),
            ParseError::Json(msg) => write!(f, "JSON error: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<std::io::Error> for ParseError {
    fn from(e: std::io::Error) -> Self {
        ParseError::IoError(e.to_string())
    }
}

/// Registry of available parsers
pub struct ParserRegistry {
    parsers: Vec<Box<dyn OntologyParser>>,
}

impl ParserRegistry {
    /// Create a new registry with all available parsers
    pub fn new() -> Self {
        Self {
            parsers: vec![
                Box::new(obo::OboParser::new()),
                Box::new(owl::OwlParser::new()),
                Box::new(jsonld::JsonLdParser::new()),
                Box::new(fhir::FhirParser::new()),
            ],
        }
    }

    /// Find a parser that can handle the given path
    pub fn parser_for(&self, path: &Path) -> Option<&dyn OntologyParser> {
        self.parsers
            .iter()
            .find(|p| p.can_parse(path))
            .map(|p| p.as_ref())
    }

    /// Get all registered parsers
    pub fn parsers(&self) -> impl Iterator<Item = &dyn OntologyParser> {
        self.parsers.iter().map(|p| p.as_ref())
    }

    /// List supported extensions
    pub fn supported_extensions(&self) -> Vec<&str> {
        self.parsers
            .iter()
            .flat_map(|p| p.extensions().iter().copied())
            .collect()
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_term_builder() {
        let term = RawTerm::new("http://example.org/term1")
            .with_label("Test Term")
            .with_source("test")
            .with_parent("http://example.org/parent");

        assert_eq!(term.iri, "http://example.org/term1");
        assert_eq!(term.label, Some("Test Term".to_string()));
        assert_eq!(term.source, "test");
        assert_eq!(term.parents.len(), 1);
    }

    #[test]
    fn test_parser_registry() {
        let registry = ParserRegistry::new();

        // Check OBO parser
        let path = Path::new("test.obo");
        assert!(registry.parser_for(path).is_some());

        // Check unknown extension
        let path = Path::new("test.xyz");
        assert!(registry.parser_for(path).is_none());
    }

    #[test]
    fn test_supported_extensions() {
        let registry = ParserRegistry::new();
        let exts = registry.supported_extensions();

        assert!(exts.contains(&"obo"));
        assert!(exts.contains(&"owl"));
        assert!(exts.contains(&"json"));
    }
}
