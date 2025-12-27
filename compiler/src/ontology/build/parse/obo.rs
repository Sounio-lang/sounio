//! OBO format parser (Open Biomedical Ontologies)
//!
//! Streaming parser that yields terms as they're parsed,
//! keeping memory usage constant regardless of file size.
//!
//! Supports OBO 1.4 format as specified at:
//! http://owlcollab.github.io/oboformat/doc/GO.format.obo-1_4.html

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

use super::{OntologyParser, ParseError, RawTerm, Relation};

/// OBO format parser
pub struct OboParser {
    /// Prefix mappings (e.g., "CHEBI" -> "http://purl.obolibrary.org/obo/CHEBI_")
    prefixes: HashMap<String, String>,
}

impl OboParser {
    /// Create a new OBO parser with default prefixes
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Standard OBO prefixes
        prefixes.insert(
            "CHEBI".into(),
            "http://purl.obolibrary.org/obo/CHEBI_".into(),
        );
        prefixes.insert("GO".into(), "http://purl.obolibrary.org/obo/GO_".into());
        prefixes.insert("DOID".into(), "http://purl.obolibrary.org/obo/DOID_".into());
        prefixes.insert("HP".into(), "http://purl.obolibrary.org/obo/HP_".into());
        prefixes.insert(
            "MONDO".into(),
            "http://purl.obolibrary.org/obo/MONDO_".into(),
        );
        prefixes.insert(
            "UBERON".into(),
            "http://purl.obolibrary.org/obo/UBERON_".into(),
        );
        prefixes.insert("CL".into(), "http://purl.obolibrary.org/obo/CL_".into());
        prefixes.insert("PATO".into(), "http://purl.obolibrary.org/obo/PATO_".into());
        prefixes.insert("BFO".into(), "http://purl.obolibrary.org/obo/BFO_".into());
        prefixes.insert("RO".into(), "http://purl.obolibrary.org/obo/RO_".into());
        prefixes.insert("IAO".into(), "http://purl.obolibrary.org/obo/IAO_".into());
        prefixes.insert("SNOMED".into(), "http://snomed.info/id/".into());
        prefixes.insert(
            "ICD10".into(),
            "http://purl.bioontology.org/ontology/ICD10/".into(),
        );

        Self { prefixes }
    }

    /// Add a custom prefix mapping
    pub fn with_prefix(mut self, prefix: impl Into<String>, expansion: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), expansion.into());
        self
    }

    /// Expand a CURIE to full IRI
    pub fn expand_curie(&self, curie: &str) -> String {
        // Already a full IRI
        if curie.starts_with("http://") || curie.starts_with("https://") {
            return curie.to_string();
        }

        // Try to expand prefix
        if let Some(colon_pos) = curie.find(':') {
            let prefix = &curie[..colon_pos];
            let local = &curie[colon_pos + 1..];

            if let Some(base) = self.prefixes.get(prefix) {
                return format!("{}{}", base, local);
            }
        }

        // Default OBO namespace
        format!("http://purl.obolibrary.org/obo/{}", curie.replace(':', "_"))
    }
}

impl Default for OboParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OntologyParser for OboParser {
    fn extensions(&self) -> &[&str] {
        &["obo"]
    }

    fn name(&self) -> &str {
        "OBO"
    }

    fn parse<'a>(
        &'a self,
        reader: Box<dyn Read + 'a>,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a> {
        Box::new(OboIterator::new(BufReader::new(reader), self))
    }
}

/// Streaming iterator over OBO terms
struct OboIterator<'a, R: BufRead> {
    reader: R,
    parser: &'a OboParser,
    line_number: usize,
    current_stanza: Option<OboStanza>,
    source: String,
    line_buffer: String,
    finished: bool,
}

/// Accumulated data for a stanza being parsed
#[derive(Default)]
struct OboStanza {
    stanza_type: String,
    id: Option<String>,
    name: Option<String>,
    def: Option<String>,
    synonyms: Vec<String>,
    is_a: Vec<String>,
    relationships: Vec<(String, String)>,
    is_obsolete: bool,
}

impl<'a, R: BufRead> OboIterator<'a, R> {
    fn new(reader: R, parser: &'a OboParser) -> Self {
        Self {
            reader,
            parser,
            line_number: 0,
            current_stanza: None,
            source: String::new(),
            line_buffer: String::new(),
            finished: false,
        }
    }

    fn parse_line(&mut self, line: &str) {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('!') {
            return;
        }

        // Stanza header
        if line.starts_with('[') && line.ends_with(']') {
            let stanza_type = &line[1..line.len() - 1];
            self.current_stanza = Some(OboStanza {
                stanza_type: stanza_type.to_string(),
                ..Default::default()
            });
            return;
        }

        // Tag-value pair
        if let Some(colon_pos) = line.find(':') {
            let tag = line[..colon_pos].trim();
            let value = line[colon_pos + 1..].trim();

            // Remove trailing comments
            let value = value.split(" !").next().unwrap_or(value).trim();

            // Handle header tags (before any stanza)
            if self.current_stanza.is_none() {
                if tag == "ontology" {
                    self.source = value.to_string();
                }
                return;
            }

            if let Some(ref mut stanza) = self.current_stanza {
                match tag {
                    "id" => stanza.id = Some(value.to_string()),
                    "name" => stanza.name = Some(value.to_string()),
                    "def" => {
                        // Extract definition from quotes: "definition text" [refs]
                        if let Some(start) = value.find('"')
                            && let Some(end) = value[start + 1..].find('"')
                        {
                            stanza.def = Some(value[start + 1..start + 1 + end].to_string());
                        }
                    }
                    "synonym" => {
                        // Extract synonym from quotes
                        if let Some(start) = value.find('"')
                            && let Some(end) = value[start + 1..].find('"')
                        {
                            stanza
                                .synonyms
                                .push(value[start + 1..start + 1 + end].to_string());
                        }
                    }
                    "is_a" => {
                        // is_a: PARENT_ID ! optional comment
                        let parent = value.split_whitespace().next().unwrap_or(value);
                        stanza.is_a.push(parent.to_string());
                    }
                    "relationship" => {
                        // relationship: relation_type target_id
                        let parts: Vec<&str> = value.split_whitespace().collect();
                        if parts.len() >= 2 {
                            stanza
                                .relationships
                                .push((parts[0].to_string(), parts[1].to_string()));
                        }
                    }
                    "is_obsolete" => {
                        stanza.is_obsolete = value == "true";
                    }
                    _ => {}
                }
            }
        }
    }

    fn finalize_stanza(&mut self) -> Option<RawTerm> {
        let stanza = self.current_stanza.take()?;

        // Only emit [Term] stanzas
        if stanza.stanza_type != "Term" {
            return None;
        }

        let id = stanza.id?;

        Some(RawTerm {
            iri: self.parser.expand_curie(&id),
            label: stanza.name,
            definition: stanza.def,
            synonyms: stanza.synonyms,
            parents: stanza
                .is_a
                .iter()
                .map(|p| self.parser.expand_curie(p))
                .collect(),
            relations: stanza
                .relationships
                .iter()
                .map(|(r, t)| Relation {
                    predicate: r.clone(),
                    target: self.parser.expand_curie(t),
                })
                .collect(),
            deprecated: stanza.is_obsolete,
            source: self.source.clone(),
            metadata: HashMap::new(),
        })
    }
}

impl<'a, R: BufRead> Iterator for OboIterator<'a, R> {
    type Item = Result<RawTerm, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            self.line_buffer.clear();
            match self.reader.read_line(&mut self.line_buffer) {
                Ok(0) => {
                    // EOF - emit final stanza if any
                    self.finished = true;
                    return self.finalize_stanza().map(Ok);
                }
                Ok(_) => {
                    self.line_number += 1;
                    // Clone the line to avoid borrowing issues
                    let line = self.line_buffer.trim().to_string();

                    // New stanza starting - emit previous if it was a Term
                    if line.starts_with('[') && self.current_stanza.is_some() {
                        let term = self.finalize_stanza();

                        // Parse the new stanza header
                        self.parse_line(&line);

                        if let Some(t) = term {
                            return Some(Ok(t));
                        }
                    } else {
                        self.parse_line(&line);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_curie() {
        let parser = OboParser::new();

        assert_eq!(
            parser.expand_curie("CHEBI:15365"),
            "http://purl.obolibrary.org/obo/CHEBI_15365"
        );

        assert_eq!(
            parser.expand_curie("http://example.org/term"),
            "http://example.org/term"
        );

        assert_eq!(
            parser.expand_curie("UNKNOWN:123"),
            "http://purl.obolibrary.org/obo/UNKNOWN_123"
        );
    }

    #[test]
    fn test_parse_simple_term() {
        let obo = r#"
format-version: 1.4
ontology: test

[Term]
id: TEST:0001
name: Test Term
def: "A test term for testing" [TEST:ref]
is_a: TEST:0000

[Term]
id: TEST:0002
name: Another Term
is_a: TEST:0001
"#;

        let parser = OboParser::new();
        let terms: Vec<_> = parser.parse(Box::new(obo.as_bytes())).collect();

        assert_eq!(terms.len(), 2);

        let term = terms[0].as_ref().unwrap();
        assert!(term.iri.contains("TEST_0001"));
        assert_eq!(term.label, Some("Test Term".to_string()));
        assert_eq!(term.definition, Some("A test term for testing".to_string()));
        assert_eq!(term.parents.len(), 1);

        let term2 = terms[1].as_ref().unwrap();
        assert!(term2.iri.contains("TEST_0002"));
    }

    #[test]
    fn test_parse_synonyms() {
        let obo = r#"
[Term]
id: TEST:0001
name: Main Name
synonym: "Alt Name 1" EXACT []
synonym: "Alt Name 2" RELATED []
"#;

        let parser = OboParser::new();
        let terms: Vec<_> = parser.parse(Box::new(obo.as_bytes())).collect();

        assert_eq!(terms.len(), 1);
        let term = terms[0].as_ref().unwrap();
        assert_eq!(term.synonyms.len(), 2);
        assert!(term.synonyms.contains(&"Alt Name 1".to_string()));
        assert!(term.synonyms.contains(&"Alt Name 2".to_string()));
    }

    #[test]
    fn test_parse_relationships() {
        let obo = r#"
[Term]
id: TEST:0001
name: Test
relationship: part_of TEST:0002
relationship: has_role TEST:0003
"#;

        let parser = OboParser::new();
        let terms: Vec<_> = parser.parse(Box::new(obo.as_bytes())).collect();

        assert_eq!(terms.len(), 1);
        let term = terms[0].as_ref().unwrap();
        assert_eq!(term.relations.len(), 2);
    }

    #[test]
    fn test_parse_obsolete() {
        let obo = r#"
[Term]
id: TEST:0001
name: Obsolete Term
is_obsolete: true
"#;

        let parser = OboParser::new();
        let terms: Vec<_> = parser.parse(Box::new(obo.as_bytes())).collect();

        assert_eq!(terms.len(), 1);
        let term = terms[0].as_ref().unwrap();
        assert!(term.deprecated);
    }

    #[test]
    fn test_skip_typedef() {
        let obo = r#"
[Term]
id: TEST:0001
name: Term

[Typedef]
id: part_of
name: part of

[Term]
id: TEST:0002
name: Term 2
"#;

        let parser = OboParser::new();
        let terms: Vec<_> = parser.parse(Box::new(obo.as_bytes())).collect();

        // Should only have Terms, not Typedefs
        assert_eq!(terms.len(), 2);
    }
}
