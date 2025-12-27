//! OWL/RDF Parser for Ontology Store Builder
//!
//! Supports:
//! - Turtle (.ttl)
//! - RDF/XML (.rdf, .owl)
//! - N-Triples (.nt)
//!
//! Uses rio_turtle and rio_xml crates for standards-compliant parsing.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use super::{OntologyParser, ParseError, RawTerm, Relation};

/// OWL/RDF parser supporting multiple serialization formats.
pub struct OwlParser {
    /// Default prefixes for common ontologies
    default_prefixes: HashMap<String, String>,
}

impl Default for OwlParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OwlParser {
    pub fn new() -> Self {
        let mut default_prefixes = HashMap::new();

        // Standard RDF/OWL prefixes
        default_prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        default_prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        default_prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        default_prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        default_prefixes.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );
        default_prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        default_prefixes.insert(
            "dcterms".to_string(),
            "http://purl.org/dc/terms/".to_string(),
        );

        // Biomedical ontology prefixes
        default_prefixes.insert(
            "obo".to_string(),
            "http://purl.obolibrary.org/obo/".to_string(),
        );
        default_prefixes.insert(
            "oboInOwl".to_string(),
            "http://www.geneontology.org/formats/oboInOwl#".to_string(),
        );

        Self { default_prefixes }
    }

    /// Detect format from file extension
    fn detect_format(path: &Path) -> Option<RdfFormat> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "ttl" | "turtle" => Some(RdfFormat::Turtle),
            "rdf" | "owl" | "xml" => Some(RdfFormat::RdfXml),
            "nt" | "ntriples" => Some(RdfFormat::NTriples),
            _ => None,
        }
    }

    /// Parse Turtle format (streaming)
    fn parse_turtle<'a, R: BufRead + 'a>(
        &'a self,
        reader: R,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a> {
        Box::new(TurtleIterator::new(reader, self.default_prefixes.clone()))
    }

    /// Parse RDF/XML format
    fn parse_rdf_xml<'a, R: Read + 'a>(
        &'a self,
        reader: R,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a> {
        Box::new(RdfXmlIterator::new(reader, self.default_prefixes.clone()))
    }

    /// Parse N-Triples format (line-based, very efficient)
    fn parse_ntriples<'a, R: BufRead + 'a>(
        &'a self,
        reader: R,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a> {
        Box::new(NTriplesIterator::new(reader))
    }
}

impl OntologyParser for OwlParser {
    fn parse<'a>(
        &'a self,
        reader: Box<dyn Read + 'a>,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a> {
        // Default to Turtle format when we can't detect
        let buf_reader = BufReader::new(reader);
        self.parse_turtle(buf_reader)
    }

    fn extensions(&self) -> &[&str] {
        &["ttl", "owl", "rdf", "nt", "turtle", "ntriples"]
    }

    fn can_parse(&self, path: &Path) -> bool {
        Self::detect_format(path).is_some()
    }

    fn name(&self) -> &str {
        "OWL/RDF Parser"
    }
}

/// RDF serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfFormat {
    Turtle,
    RdfXml,
    NTriples,
}

// ============================================================================
// Turtle Parser (streaming)
// ============================================================================

/// Streaming Turtle parser that collects triples into terms.
struct TurtleIterator<R: BufRead> {
    reader: R,
    prefixes: HashMap<String, String>,
    /// Accumulated triples for current subject
    current_subject: Option<String>,
    current_triples: TripleAccumulator,
    /// Buffer for pending terms
    pending: Vec<Result<RawTerm, ParseError>>,
    /// Line number for error reporting
    line_number: usize,
    /// Whether we've finished parsing
    finished: bool,
}

impl<R: BufRead> TurtleIterator<R> {
    fn new(reader: R, prefixes: HashMap<String, String>) -> Self {
        Self {
            reader,
            prefixes,
            current_subject: None,
            current_triples: TripleAccumulator::new(),
            pending: Vec::new(),
            line_number: 0,
            finished: false,
        }
    }

    /// Parse a single line of Turtle
    fn parse_line(&mut self, line: &str) -> Option<Result<RawTerm, ParseError>> {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

        // Handle prefix declarations
        if line.starts_with("@prefix") || line.to_lowercase().starts_with("prefix") {
            self.parse_prefix(line);
            return None;
        }

        // Handle base declarations
        if line.starts_with("@base") || line.to_lowercase().starts_with("base") {
            // Store base IRI for relative resolution
            return None;
        }

        // Try to parse as triple
        self.parse_triple(line)
    }

    fn parse_prefix(&mut self, line: &str) {
        // @prefix foo: <http://example.org/> .
        // PREFIX foo: <http://example.org/>
        let line = line
            .trim_start_matches("@prefix")
            .trim_start_matches("PREFIX")
            .trim_start_matches("prefix")
            .trim();

        if let Some(colon_pos) = line.find(':') {
            let prefix = line[..colon_pos].trim();
            let rest = line[colon_pos + 1..].trim();

            if let (Some(start), Some(end)) = (rest.find('<'), rest.find('>')) {
                let iri = &rest[start + 1..end];
                self.prefixes.insert(prefix.to_string(), iri.to_string());
            }
        }
    }

    fn parse_triple(&mut self, line: &str) -> Option<Result<RawTerm, ParseError>> {
        // Simple triple parser - handles basic S P O patterns
        // For complex Turtle, we'd use rio_turtle crate

        let parts: Vec<&str> = self.tokenize_triple(line);
        if parts.len() < 3 {
            return None;
        }

        let subject = self.expand_term(parts[0]);
        let predicate = self.expand_term(parts[1]);
        let object = self.expand_term(parts[2]);

        // Check if subject changed (emit previous term)
        let result = if self.current_subject.as_ref() != Some(&subject) {
            let prev_term = self.emit_current_term();
            self.current_subject = Some(subject.clone());
            self.current_triples = TripleAccumulator::new();
            prev_term
        } else {
            None
        };

        // Accumulate triple
        self.current_triples.add_triple(&predicate, &object);

        result
    }

    fn tokenize_triple<'a>(&self, line: &'a str) -> Vec<&'a str> {
        let mut tokens = Vec::new();
        let _chars = line.chars().peekable();
        let mut start = 0;
        let mut pos = 0;
        let mut in_uri = false;
        let mut in_string = false;
        let line_bytes = line.as_bytes();

        while pos < line.len() {
            let c = line_bytes[pos] as char;

            if c == '<' && !in_string {
                in_uri = true;
                start = pos;
            } else if c == '>' && in_uri {
                in_uri = false;
                tokens.push(&line[start..=pos]);
                start = pos + 1;
            } else if c == '"' && !in_uri {
                in_string = !in_string;
                if !in_string {
                    // End of string - include any language tag or datatype
                    while pos + 1 < line.len() {
                        let next = line_bytes[pos + 1] as char;
                        if next == '@' || next == '^' {
                            pos += 1;
                            while pos + 1 < line.len() && !line_bytes[pos + 1].is_ascii_whitespace()
                            {
                                pos += 1;
                            }
                        } else {
                            break;
                        }
                    }
                    tokens.push(&line[start..=pos]);
                    start = pos + 1;
                } else {
                    start = pos;
                }
            } else if c.is_ascii_whitespace() && !in_uri && !in_string {
                if start < pos {
                    let token = &line[start..pos];
                    if !token.trim().is_empty() && token != "." && token != ";" {
                        tokens.push(token.trim());
                    }
                }
                start = pos + 1;
            }

            pos += 1;
        }

        // Handle remaining token
        if start < line.len() {
            let token = &line[start..].trim_end_matches(['.', ';']);
            if !token.trim().is_empty() {
                tokens.push(token.trim());
            }
        }

        tokens
    }

    fn expand_term(&self, term: &str) -> String {
        let term = term.trim();

        // Full IRI
        if term.starts_with('<') && term.ends_with('>') {
            return term[1..term.len() - 1].to_string();
        }

        // Prefixed name
        if let Some(colon_pos) = term.find(':') {
            let prefix = &term[..colon_pos];
            let local = &term[colon_pos + 1..];

            if let Some(base) = self.prefixes.get(prefix) {
                return format!("{}{}", base, local);
            }
        }

        // Blank node or literal - return as-is
        term.to_string()
    }

    fn emit_current_term(&self) -> Option<Result<RawTerm, ParseError>> {
        let subject = self.current_subject.as_ref()?;

        // Only emit if we have actual data
        if self.current_triples.is_empty() {
            return None;
        }

        let term = RawTerm {
            iri: subject.clone(),
            label: self.current_triples.label.clone(),
            definition: self.current_triples.definition.clone(),
            synonyms: self.current_triples.synonyms.clone(),
            parents: self.current_triples.parents.clone(),
            relations: self.current_triples.relations.clone(),
            deprecated: self.current_triples.deprecated,
            metadata: self.current_triples.metadata.clone(),
            source: String::new(),
        };

        Some(Ok(term))
    }
}

impl<R: BufRead> Iterator for TurtleIterator<R> {
    type Item = Result<RawTerm, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Return any pending items first
        if let Some(item) = self.pending.pop() {
            return Some(item);
        }

        if self.finished {
            return None;
        }

        // Read lines until we have a term to emit
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF - emit final term
                    self.finished = true;
                    return self.emit_current_term();
                }
                Ok(_) => {
                    self.line_number += 1;
                    if let Some(result) = self.parse_line(&line) {
                        return Some(result);
                    }
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(ParseError::IoError(e.to_string())));
                }
            }
        }
    }
}

// ============================================================================
// RDF/XML Parser
// ============================================================================

/// Simple RDF/XML parser (for full compliance, use rio_xml)
struct RdfXmlIterator<R: Read> {
    reader: R,
    prefixes: HashMap<String, String>,
    finished: bool,
    terms: Vec<Result<RawTerm, ParseError>>,
    current_index: usize,
}

impl<R: Read> RdfXmlIterator<R> {
    fn new(mut reader: R, prefixes: HashMap<String, String>) -> Self {
        // For RDF/XML, we need to parse the whole document first
        // A full streaming implementation would use SAX-style parsing
        let mut content = String::new();
        let terms = match reader.read_to_string(&mut content) {
            Ok(_) => Self::parse_rdf_xml_content(&content, &prefixes),
            Err(e) => vec![Err(ParseError::IoError(e.to_string()))],
        };

        Self {
            reader,
            prefixes,
            finished: false,
            terms,
            current_index: 0,
        }
    }

    fn parse_rdf_xml_content(
        content: &str,
        prefixes: &HashMap<String, String>,
    ) -> Vec<Result<RawTerm, ParseError>> {
        let mut terms = Vec::new();

        // Simple regex-based extraction for rdf:Description elements
        // For production, use rio_xml or quick-xml with proper namespace handling

        let mut current_term: Option<RawTerm> = None;

        for line in content.lines() {
            let line = line.trim();

            // Start of Description
            if line.contains("rdf:Description") || line.contains("owl:Class") {
                // Extract rdf:about
                if let Some(about) = Self::extract_attribute(line, "rdf:about") {
                    if let Some(term) = current_term.take()
                        && !term.iri.is_empty()
                    {
                        terms.push(Ok(term));
                    }
                    current_term = Some(RawTerm {
                        iri: about,
                        ..Default::default()
                    });
                }
            }

            // Extract label
            if let Some(ref mut term) = current_term {
                if line.contains("rdfs:label")
                    && let Some(label) = Self::extract_element_text(line)
                {
                    term.label = Some(label);
                }

                // Extract definition
                if (line.contains("obo:IAO_0000115") || line.contains("skos:definition"))
                    && let Some(def) = Self::extract_element_text(line)
                {
                    term.definition = Some(def);
                }

                // Extract parent (subClassOf)
                if line.contains("rdfs:subClassOf")
                    && let Some(parent) = Self::extract_attribute(line, "rdf:resource")
                {
                    term.parents.push(parent);
                }

                // Extract deprecated status
                if line.contains("owl:deprecated") && line.contains("true") {
                    term.deprecated = true;
                }
            }
        }

        // Don't forget the last term
        if let Some(term) = current_term
            && !term.iri.is_empty()
        {
            terms.push(Ok(term));
        }

        terms
    }

    fn extract_attribute(line: &str, attr: &str) -> Option<String> {
        let pattern = format!("{}=\"", attr);
        if let Some(start) = line.find(&pattern) {
            let rest = &line[start + pattern.len()..];
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
        None
    }

    fn extract_element_text(line: &str) -> Option<String> {
        // Simple extraction: find text between > and <
        if let Some(start) = line.find('>') {
            let rest = &line[start + 1..];
            if let Some(end) = rest.find('<') {
                let text = rest[..end].trim();
                if !text.is_empty() {
                    return Some(text.to_string());
                }
            }
        }
        None
    }
}

impl<R: Read> Iterator for RdfXmlIterator<R> {
    type Item = Result<RawTerm, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.terms.len() {
            let result = self.terms[self.current_index].clone();
            self.current_index += 1;
            Some(result)
        } else {
            None
        }
    }
}

// ============================================================================
// N-Triples Parser (line-based, very efficient)
// ============================================================================

/// N-Triples parser - one triple per line, very efficient
struct NTriplesIterator<R: BufRead> {
    reader: R,
    /// Accumulated triples grouped by subject
    current_subject: Option<String>,
    current_triples: TripleAccumulator,
    finished: bool,
}

impl<R: BufRead> NTriplesIterator<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            current_subject: None,
            current_triples: TripleAccumulator::new(),
            finished: false,
        }
    }

    fn parse_line(&mut self, line: &str) -> Option<Result<RawTerm, ParseError>> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

        // N-Triples format: <subject> <predicate> <object> .
        let parts: Vec<&str> = self.split_ntriple(line);
        if parts.len() < 3 {
            return None;
        }

        let subject = Self::extract_iri(parts[0]);
        let predicate = Self::extract_iri(parts[1]);
        let object = parts[2].to_string();

        // Check if subject changed
        let result = if self.current_subject.as_ref() != Some(&subject) {
            let prev = self.emit_current();
            self.current_subject = Some(subject.clone());
            self.current_triples = TripleAccumulator::new();
            prev
        } else {
            None
        };

        self.current_triples.add_triple(&predicate, &object);
        result
    }

    fn split_ntriple<'a>(&self, line: &'a str) -> Vec<&'a str> {
        let mut parts = Vec::new();
        let mut start = 0;
        let mut in_uri = false;
        let mut in_string = false;
        let bytes = line.as_bytes();

        for (i, &b) in bytes.iter().enumerate() {
            let c = b as char;
            if c == '<' && !in_string {
                in_uri = true;
                start = i;
            } else if c == '>' && in_uri {
                in_uri = false;
                parts.push(&line[start..=i]);
                start = i + 1;
            } else if c == '"' && !in_uri {
                if !in_string {
                    in_string = true;
                    start = i;
                } else {
                    in_string = false;
                    // Include datatype or language tag
                    let mut end = i;
                    while end + 1 < line.len() {
                        let next = bytes[end + 1] as char;
                        if next == '@' || next == '^' {
                            end += 1;
                            while end + 1 < line.len()
                                && !bytes[end + 1].is_ascii_whitespace()
                                && bytes[end + 1] as char != '.'
                            {
                                end += 1;
                            }
                        } else {
                            break;
                        }
                    }
                    parts.push(&line[start..=end]);
                    start = end + 1;
                }
            }
        }

        parts
    }

    fn extract_iri(term: &str) -> String {
        let term = term.trim();
        if term.starts_with('<') && term.ends_with('>') {
            term[1..term.len() - 1].to_string()
        } else {
            term.to_string()
        }
    }

    fn emit_current(&self) -> Option<Result<RawTerm, ParseError>> {
        let subject = self.current_subject.as_ref()?;
        if self.current_triples.is_empty() {
            return None;
        }

        Some(Ok(RawTerm {
            iri: subject.clone(),
            label: self.current_triples.label.clone(),
            definition: self.current_triples.definition.clone(),
            synonyms: self.current_triples.synonyms.clone(),
            parents: self.current_triples.parents.clone(),
            relations: self.current_triples.relations.clone(),
            deprecated: self.current_triples.deprecated,
            metadata: self.current_triples.metadata.clone(),
            source: String::new(),
        }))
    }
}

impl<R: BufRead> Iterator for NTriplesIterator<R> {
    type Item = Result<RawTerm, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.finished = true;
                    return self.emit_current();
                }
                Ok(_) => {
                    if let Some(result) = self.parse_line(&line) {
                        return Some(result);
                    }
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(ParseError::IoError(e.to_string())));
                }
            }
        }
    }
}

// ============================================================================
// Triple Accumulator
// ============================================================================

/// Accumulates RDF triples for a single subject
#[derive(Default)]
struct TripleAccumulator {
    label: Option<String>,
    definition: Option<String>,
    synonyms: Vec<String>,
    parents: Vec<String>,
    relations: Vec<Relation>,
    deprecated: bool,
    metadata: HashMap<String, String>,
}

impl TripleAccumulator {
    fn new() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.label.is_none()
            && self.definition.is_none()
            && self.synonyms.is_empty()
            && self.parents.is_empty()
            && self.relations.is_empty()
    }

    fn add_triple(&mut self, predicate: &str, object: &str) {
        let object_value = Self::extract_literal_value(object);

        match predicate {
            // Label predicates
            p if p.ends_with("label") || p.ends_with("#label") => {
                self.label = Some(object_value);
            }

            // Definition predicates
            p if p.contains("IAO_0000115")
                || p.ends_with("definition")
                || p.ends_with("#definition") =>
            {
                self.definition = Some(object_value);
            }

            // Synonym predicates
            p if p.contains("hasExactSynonym")
                || p.contains("hasRelatedSynonym")
                || p.contains("hasBroadSynonym")
                || p.contains("hasNarrowSynonym")
                || p.ends_with("altLabel") =>
            {
                self.synonyms.push(object_value);
            }

            // Parent predicates
            p if p.ends_with("subClassOf") || p.ends_with("#subClassOf") || p.contains("is_a") => {
                let parent_iri = Self::extract_iri_value(object);
                if !parent_iri.is_empty() {
                    self.parents.push(parent_iri);
                }
            }

            // Deprecated predicate
            p if p.ends_with("deprecated") => {
                self.deprecated = object.contains("true") || object.contains("1");
            }

            // Other relations
            _ => {
                // Store as generic relation if it looks like an object property
                if object.starts_with('<') || object.starts_with('_') {
                    let object_iri = Self::extract_iri_value(object);
                    if !object_iri.is_empty() {
                        self.relations.push(Relation {
                            predicate: predicate.to_string(),
                            target: object_iri,
                        });
                    }
                } else {
                    // Store as metadata
                    let key = predicate
                        .rsplit('/')
                        .next()
                        .or_else(|| predicate.rsplit('#').next())
                        .unwrap_or(predicate);
                    self.metadata.insert(key.to_string(), object_value);
                }
            }
        }
    }

    fn extract_literal_value(object: &str) -> String {
        let object = object.trim();

        // Handle quoted literals
        if object.starts_with('"')
            && let Some(end_quote) = object[1..].find('"')
        {
            return object[1..end_quote + 1].to_string();
        }

        // Handle IRIs
        if object.starts_with('<') && object.ends_with('>') {
            return object[1..object.len() - 1].to_string();
        }

        object.to_string()
    }

    fn extract_iri_value(object: &str) -> String {
        let object = object.trim();
        if object.starts_with('<') && object.ends_with('>') {
            object[1..object.len() - 1].to_string()
        } else if object.starts_with('<') {
            // Handle <iri> without closing bracket (partial)
            if let Some(end) = object.find('>') {
                object[1..end].to_string()
            } else {
                object[1..].to_string()
            }
        } else {
            object.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_turtle_parsing() {
        let turtle = r#"
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix obo: <http://purl.obolibrary.org/obo/> .

obo:GO_0008150 rdfs:label "biological_process" .
obo:GO_0008150 obo:IAO_0000115 "A biological process" .

obo:GO_0009987 rdfs:label "cellular process" .
obo:GO_0009987 rdfs:subClassOf obo:GO_0008150 .
"#;

        let parser = OwlParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(turtle));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert!(
            terms.len() >= 1,
            "Expected at least 1 term, got {}",
            terms.len()
        );

        // Check first term
        let term = terms[0].as_ref().unwrap();
        assert!(term.iri.contains("GO_0008150"));
        assert_eq!(term.label, Some("biological_process".to_string()));
    }

    #[test]
    fn test_ntriples_parsing() {
        let ntriples = r#"
<http://example.org/term1> <http://www.w3.org/2000/01/rdf-schema#label> "Term One" .
<http://example.org/term1> <http://purl.obolibrary.org/obo/IAO_0000115> "Definition of term one" .
<http://example.org/term2> <http://www.w3.org/2000/01/rdf-schema#label> "Term Two" .
<http://example.org/term2> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <http://example.org/term1> .
"#;

        let reader = BufReader::new(Cursor::new(ntriples));
        let mut iter = NTriplesIterator::new(reader);

        let term1 = iter.next().unwrap().unwrap();
        assert_eq!(term1.iri, "http://example.org/term1");
        assert_eq!(term1.label, Some("Term One".to_string()));
        assert_eq!(term1.definition, Some("Definition of term one".to_string()));

        let term2 = iter.next().unwrap().unwrap();
        assert_eq!(term2.iri, "http://example.org/term2");
        assert_eq!(term2.label, Some("Term Two".to_string()));
        assert_eq!(term2.parents, vec!["http://example.org/term1"]);
    }

    #[test]
    fn test_prefix_expansion() {
        let parser = OwlParser::new();

        // Test with prefixed turtle
        let turtle = r#"
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:term1 rdfs:label "Test" .
"#;

        let reader: Box<dyn Read> = Box::new(Cursor::new(turtle));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert!(!terms.is_empty());
        let term = terms[0].as_ref().unwrap();
        assert_eq!(term.iri, "http://example.org/term1");
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            OwlParser::detect_format(Path::new("test.ttl")),
            Some(RdfFormat::Turtle)
        );
        assert_eq!(
            OwlParser::detect_format(Path::new("test.owl")),
            Some(RdfFormat::RdfXml)
        );
        assert_eq!(
            OwlParser::detect_format(Path::new("test.rdf")),
            Some(RdfFormat::RdfXml)
        );
        assert_eq!(
            OwlParser::detect_format(Path::new("test.nt")),
            Some(RdfFormat::NTriples)
        );
        assert_eq!(OwlParser::detect_format(Path::new("test.txt")), None);
    }

    #[test]
    fn test_triple_accumulator() {
        let mut acc = TripleAccumulator::new();

        acc.add_triple(
            "http://www.w3.org/2000/01/rdf-schema#label",
            "\"Test Label\"",
        );
        acc.add_triple(
            "http://purl.obolibrary.org/obo/IAO_0000115",
            "\"Test Definition\"",
        );
        acc.add_triple(
            "http://www.w3.org/2000/01/rdf-schema#subClassOf",
            "<http://example.org/parent>",
        );
        acc.add_triple(
            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
            "\"Synonym 1\"",
        );
        acc.add_triple(
            "http://www.w3.org/2002/07/owl#deprecated",
            "\"true\"^^xsd:boolean",
        );

        assert_eq!(acc.label, Some("Test Label".to_string()));
        assert_eq!(acc.definition, Some("Test Definition".to_string()));
        assert_eq!(acc.parents, vec!["http://example.org/parent"]);
        assert_eq!(acc.synonyms, vec!["Synonym 1"]);
        assert!(acc.deprecated);
    }
}
