//! Rich Semantic Error Messages
//!
//! Error messages that explain *why* types are incompatible,
//! with suggestions for fixing the issue.
//!
//! # Philosophy
//!
//! Good error messages answer:
//! 1. What went wrong?
//! 2. Where did it go wrong?
//! 3. Why is it wrong?
//! 4. How can I fix it?
//!
//! For semantic types, we add:
//! 5. What is the semantic distance?
//! 6. Is there a mapping I could use?
//! 7. Would explicit coercion work?

use crate::ontology::distance::SemanticDistance;
use crate::ontology::loader::{IRI, OntologyId};
use std::fmt;

/// Source span (byte offsets)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn empty() -> Self {
        Self { start: 0, end: 0 }
    }
}

/// A semantic type error with rich context
#[derive(Debug)]
pub struct SemanticTypeError {
    /// The error kind
    pub kind: SemanticErrorKind,

    /// Source location
    pub span: Span,

    /// Expected type
    pub expected: TypeInfo,

    /// Found type
    pub found: TypeInfo,

    /// Semantic distance between types
    pub distance: Option<SemanticDistance>,

    /// Available mappings
    pub mappings: Vec<AvailableMapping>,

    /// Suggested fixes
    pub suggestions: Vec<Suggestion>,
}

/// Kind of semantic error
#[derive(Debug, Clone)]
pub enum SemanticErrorKind {
    /// Types are too far apart for implicit coercion
    DistanceTooLarge { actual: f64, threshold: f64 },

    /// Types are from disjoint branches
    DisjointTypes,

    /// No known mapping between ontologies
    NoMapping {
        from_ontology: OntologyId,
        to_ontology: OntologyId,
    },

    /// Missing explicit cast
    RequiresExplicitCast,

    /// Type not found in ontology
    UnknownOntologyType { ontology: OntologyId, term: String },

    /// Confidence too low
    InsufficientConfidence { required: f64, actual: f64 },

    /// Cross-ontology coercion without mapping
    CrossOntologyMismatch,

    /// Semantic type used in incompatible context
    IncompatibleContext { expected_context: String },
}

/// Information about a type involved in an error
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// IRI if this is an ontology type
    pub iri: Option<IRI>,
    /// Human-readable label
    pub label: String,
    /// Ontology this type belongs to
    pub ontology: Option<OntologyId>,
    /// Definition from the ontology
    pub definition: Option<String>,
    /// Source span where type is declared
    pub declaration_span: Option<Span>,
}

impl TypeInfo {
    pub fn new(label: &str) -> Self {
        Self {
            iri: None,
            label: label.to_string(),
            ontology: None,
            definition: None,
            declaration_span: None,
        }
    }

    pub fn with_iri(mut self, iri: IRI) -> Self {
        self.iri = Some(iri);
        self
    }

    pub fn with_ontology(mut self, ontology: OntologyId) -> Self {
        self.ontology = Some(ontology);
        self
    }

    pub fn with_definition(mut self, definition: &str) -> Self {
        self.definition = Some(definition.to_string());
        self
    }
}

/// An available mapping between ontologies
#[derive(Debug, Clone)]
pub struct AvailableMapping {
    /// Source IRI
    pub from: IRI,
    /// Target IRI
    pub to: IRI,
    /// Mapping predicate (e.g., "skos:exactMatch")
    pub predicate: String,
    /// Confidence in the mapping
    pub confidence: f64,
}

/// A suggested fix for the error
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// Human-readable suggestion message
    pub message: String,
    /// Optional code replacement
    pub replacement: Option<String>,
    /// Span to replace (if applicable)
    pub span: Option<Span>,
    /// Priority (higher = more likely to be helpful)
    pub priority: u8,
}

impl Suggestion {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            replacement: None,
            span: None,
            priority: 0,
        }
    }

    pub fn with_replacement(mut self, replacement: &str) -> Self {
        self.replacement = Some(replacement.to_string());
        self
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

impl SemanticTypeError {
    /// Create a new semantic type error
    pub fn new(kind: SemanticErrorKind, span: Span, expected: TypeInfo, found: TypeInfo) -> Self {
        Self {
            kind,
            span,
            expected,
            found,
            distance: None,
            mappings: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Add semantic distance information
    pub fn with_distance(mut self, distance: SemanticDistance) -> Self {
        self.distance = Some(distance);
        self
    }

    /// Add available mappings
    pub fn with_mappings(mut self, mappings: Vec<AvailableMapping>) -> Self {
        self.mappings = mappings;
        self
    }

    /// Add fix suggestions
    pub fn with_suggestions(mut self, suggestions: Vec<Suggestion>) -> Self {
        self.suggestions = suggestions;
        self
    }

    /// Get the main error message
    pub fn main_message(&self) -> String {
        match &self.kind {
            SemanticErrorKind::DistanceTooLarge { actual, threshold } => {
                format!(
                    "semantic distance too large: {:.2} > {:.2} threshold",
                    actual, threshold
                )
            }
            SemanticErrorKind::DisjointTypes => {
                format!(
                    "`{}` and `{}` are disjoint types in the ontology",
                    self.expected.label, self.found.label
                )
            }
            SemanticErrorKind::NoMapping {
                from_ontology,
                to_ontology,
            } => {
                format!("no mapping from {} to {}", from_ontology, to_ontology)
            }
            SemanticErrorKind::RequiresExplicitCast => {
                format!(
                    "cannot implicitly convert `{}` to `{}`",
                    self.found.label, self.expected.label
                )
            }
            SemanticErrorKind::UnknownOntologyType { ontology, term } => {
                format!("`{}` is not a valid term in {}", term, ontology)
            }
            SemanticErrorKind::InsufficientConfidence { required, actual } => {
                format!(
                    "confidence {:.0}% is below required {:.0}%",
                    actual * 100.0,
                    required * 100.0
                )
            }
            SemanticErrorKind::CrossOntologyMismatch => {
                "cannot convert between different ontologies without explicit mapping".to_string()
            }
            SemanticErrorKind::IncompatibleContext { expected_context } => {
                format!(
                    "`{}` cannot be used in {} context",
                    self.found.label, expected_context
                )
            }
        }
    }

    /// Get the primary label for the error span
    pub fn primary_label(&self) -> String {
        format!("found `{}`", self.found.label)
    }

    /// Get explanation of the semantic distance
    pub fn distance_explanation(&self) -> Option<String> {
        let distance = self.distance.as_ref()?;

        let mut parts = vec![format!("semantic distance: {:.3}", distance.conceptual)];

        if distance.conceptual < 0.3 {
            parts.push("(close: implicit coercion normally allowed)".to_string());
        } else if distance.conceptual < 0.7 {
            parts.push("(moderate: explicit cast required)".to_string());
        } else {
            parts.push("(distant: types are semantically unrelated)".to_string());
        }

        // Provenance depth
        if distance.provenance_depth > 0 {
            parts.push(format!(
                "provenance depth: {} steps",
                distance.provenance_depth
            ));
        }

        // Confidence retention from the distance struct
        parts.push(format!(
            "confidence retention: {:.0}%",
            distance.confidence_retention * 100.0
        ));

        Some(parts.join("\n  "))
    }

    /// Format as a rich text report
    pub fn format(&self, source: &str, filename: &str) -> String {
        let mut output = String::new();

        // Error header
        output.push_str(&format!("error: {}\n", self.main_message()));
        output.push_str(&format!("   --> {}:{}\n", filename, self.span.start));
        output.push_str("    |\n");

        // Extract the relevant line
        let (line_num, col, line_content) = find_line_and_column(source, self.span.start);

        if line_num > 0 && line_num <= source.lines().count() {
            output.push_str(&format!("{:>4} | {}\n", line_num, line_content));

            // Underline the error span
            let underline_start = col;
            let underline_len =
                (self.span.end - self.span.start).min(line_content.len().saturating_sub(col));
            output.push_str(&format!(
                "    | {}{} {}\n",
                " ".repeat(underline_start),
                "^".repeat(underline_len.max(1)),
                self.primary_label()
            ));
        }

        output.push_str("    |\n");

        // Expected type info
        if let Some(expected_span) = self.expected.declaration_span {
            let (exp_line, _, exp_content) = find_line_and_column(source, expected_span.start);
            output.push_str(&format!(
                "    = note: expected `{}` because of:\n",
                self.expected.label
            ));
            output.push_str(&format!("{:>4} | {}\n", exp_line, exp_content));
        } else {
            output.push_str(&format!("    = note: expected `{}`\n", self.expected.label));
        }

        // Semantic distance explanation
        if let Some(explanation) = self.distance_explanation() {
            output.push_str(&format!("    = note: {}\n", explanation));
        }

        // Type definitions from ontology
        if let Some(ref def) = self.expected.definition {
            let truncated = truncate_definition(def, 80);
            output.push_str(&format!(
                "    = note: {} is: {}\n",
                self.expected.label, truncated
            ));
        }
        if let Some(ref def) = self.found.definition {
            let truncated = truncate_definition(def, 80);
            output.push_str(&format!(
                "    = note: {} is: {}\n",
                self.found.label, truncated
            ));
        }

        // Available mappings
        for mapping in &self.mappings {
            let from_str = mapping
                .from
                .to_curie()
                .map(|(p, l)| format!("{}:{}", p, l))
                .unwrap_or_else(|| mapping.from.to_string());
            let to_str = mapping
                .to
                .to_curie()
                .map(|(p, l)| format!("{}:{}", p, l))
                .unwrap_or_else(|| mapping.to.to_string());
            output.push_str(&format!(
                "    = note: mapping available: {} {} {} (confidence: {:.0}%)\n",
                from_str,
                mapping.predicate,
                to_str,
                mapping.confidence * 100.0,
            ));
        }

        // Suggestions
        let mut sorted_suggestions = self.suggestions.clone();
        sorted_suggestions.sort_by(|a, b| b.priority.cmp(&a.priority));

        for suggestion in &sorted_suggestions {
            if let Some(ref replacement) = suggestion.replacement {
                output.push_str(&format!(
                    "    = help: {}: replace with `{}`\n",
                    suggestion.message, replacement
                ));
            } else {
                output.push_str(&format!("    = help: {}\n", suggestion.message));
            }
        }

        output
    }

    /// Format as JSON for tooling integration
    pub fn format_json(&self) -> String {
        serde_json::json!({
            "type": "semantic_type_error",
            "message": self.main_message(),
            "span": {
                "start": self.span.start,
                "end": self.span.end,
            },
            "expected": {
                "label": self.expected.label,
                "iri": self.expected.iri.as_ref().map(|i| i.to_string()),
                "ontology": self.expected.ontology.as_ref().map(|o| o.to_string()),
            },
            "found": {
                "label": self.found.label,
                "iri": self.found.iri.as_ref().map(|i| i.to_string()),
                "ontology": self.found.ontology.as_ref().map(|o| o.to_string()),
            },
            "distance": self.distance.as_ref().map(|d| d.conceptual),
            "suggestions": self.suggestions.iter().map(|s| &s.message).collect::<Vec<_>>(),
        })
        .to_string()
    }
}

impl fmt::Display for SemanticTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.main_message())
    }
}

impl std::error::Error for SemanticTypeError {}

/// Generate fix suggestions based on error kind
pub fn generate_suggestions(error: &SemanticTypeError) -> Vec<Suggestion> {
    let mut suggestions = Vec::new();

    match &error.kind {
        SemanticErrorKind::RequiresExplicitCast => {
            suggestions.push(
                Suggestion::new("add explicit cast")
                    .with_replacement(&format!("<expr> as {}", error.expected.label))
                    .with_priority(10),
            );
        }

        SemanticErrorKind::DistanceTooLarge { actual, .. } => {
            if *actual < 0.5 {
                suggestions.push(
                    Suggestion::new("use explicit cast to acknowledge semantic distance")
                        .with_replacement(&format!("<expr> as {}", error.expected.label))
                        .with_priority(8),
                );
            } else {
                suggestions.push(
                    Suggestion::new("types are semantically distant; consider using a different type or intermediate conversion")
                        .with_priority(5),
                );
            }
        }

        SemanticErrorKind::NoMapping {
            from_ontology,
            to_ontology,
        } => {
            suggestions.push(
                Suggestion::new(&format!(
                    "check SSSOM mappings between {} and {}",
                    from_ontology, to_ontology
                ))
                .with_priority(7),
            );
            suggestions.push(
                Suggestion::new("consider using a common upper ontology term").with_priority(5),
            );
        }

        SemanticErrorKind::UnknownOntologyType { ontology, term } => {
            suggestions.push(
                Suggestion::new(&format!(
                    "search for similar terms in {}: dc ontology search '{}' --ontology={}",
                    ontology, term, ontology
                ))
                .with_priority(8),
            );
        }

        SemanticErrorKind::InsufficientConfidence { required, actual } => {
            let boost_needed = required - actual;
            suggestions.push(
                Suggestion::new(&format!(
                    "confidence is too low; need {:.0}% more certainty",
                    boost_needed * 100.0
                ))
                .with_priority(6),
            );
            suggestions.push(
                Suggestion::new(
                    "consider adding additional evidence or using a more specific type",
                )
                .with_priority(4),
            );
        }

        SemanticErrorKind::DisjointTypes => {
            suggestions.push(
                Suggestion::new(
                    "these types have no common supertype; this is likely a logic error",
                )
                .with_priority(9),
            );
        }

        SemanticErrorKind::CrossOntologyMismatch => {
            suggestions.push(
                Suggestion::new("import the target ontology and use a mapping").with_priority(7),
            );
        }

        SemanticErrorKind::IncompatibleContext { .. } => {
            suggestions.push(
                Suggestion::new("ensure the type is appropriate for this context").with_priority(5),
            );
        }
    }

    // Add mapping-based suggestions
    for mapping in &error.mappings {
        let from_str = mapping
            .from
            .to_curie()
            .map(|(p, l)| format!("{}:{}", p, l))
            .unwrap_or_else(|| mapping.from.to_string());
        let to_str = mapping
            .to
            .to_curie()
            .map(|(p, l)| format!("{}:{}", p, l))
            .unwrap_or_else(|| mapping.to.to_string());
        suggestions.push(
            Suggestion::new(&format!(
                "use mapping {} {} {}",
                from_str, mapping.predicate, to_str
            ))
            .with_priority(6),
        );
    }

    suggestions
}

/// Find line number, column, and line content for a byte offset
fn find_line_and_column(source: &str, offset: usize) -> (usize, usize, &str) {
    let mut line_num = 1;
    let mut line_start = 0;

    for (i, c) in source.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line_num += 1;
            line_start = i + 1;
        }
    }

    let line_end = source[line_start..]
        .find('\n')
        .map(|i| line_start + i)
        .unwrap_or(source.len());

    let column = offset.saturating_sub(line_start);
    let line_content = &source[line_start..line_end];

    (line_num, column, line_content)
}

/// Truncate a definition string for display
fn truncate_definition(def: &str, max_len: usize) -> String {
    let trimmed = def.trim();
    if trimmed.len() <= max_len {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_error_message() {
        let error = SemanticTypeError::new(
            SemanticErrorKind::DistanceTooLarge {
                actual: 0.82,
                threshold: 0.30,
            },
            Span::new(100, 110),
            TypeInfo::new("Drug"),
            TypeInfo::new("Aspirin"),
        );

        let msg = error.main_message();
        assert!(msg.contains("0.82"));
        assert!(msg.contains("0.30"));
    }

    #[test]
    fn test_suggestion_generation() {
        let error = SemanticTypeError::new(
            SemanticErrorKind::RequiresExplicitCast,
            Span::new(0, 10),
            TypeInfo::new("Drug"),
            TypeInfo::new("Aspirin"),
        );

        let suggestions = generate_suggestions(&error);
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].replacement.is_some());
    }

    #[test]
    fn test_find_line_and_column() {
        let source = "line1\nline2\nline3";

        let (line, col, content) = find_line_and_column(source, 0);
        assert_eq!(line, 1);
        assert_eq!(col, 0);
        assert_eq!(content, "line1");

        let (line, col, content) = find_line_and_column(source, 6);
        assert_eq!(line, 2);
        assert_eq!(col, 0);
        assert_eq!(content, "line2");

        let (line, col, content) = find_line_and_column(source, 8);
        assert_eq!(line, 2);
        assert_eq!(col, 2);
        assert_eq!(content, "line2");
    }
}
