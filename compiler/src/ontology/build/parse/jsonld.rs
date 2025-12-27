//! JSON-LD Parser for Ontology Store Builder
//!
//! Parses JSON-LD format commonly used by Schema.org and other modern ontologies.
//! Supports both single objects and arrays of objects.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use serde_json::{Map, Value};

use super::{OntologyParser, ParseError, RawTerm, Relation};

/// JSON-LD parser for Schema.org and similar ontologies.
pub struct JsonLdParser {
    /// Context for prefix expansion
    context: HashMap<String, String>,
}

impl Default for JsonLdParser {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonLdParser {
    pub fn new() -> Self {
        let mut context = HashMap::new();

        // Schema.org context
        context.insert("schema".to_string(), "https://schema.org/".to_string());
        context.insert("@vocab".to_string(), "https://schema.org/".to_string());

        // Common JSON-LD prefixes
        context.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        context.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        context.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        context.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        context.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );

        Self { context }
    }

    /// Parse JSON-LD content into terms
    fn parse_content(&self, content: &str) -> Result<Vec<RawTerm>, ParseError> {
        let json: Value = serde_json::from_str(content)
            .map_err(|e| ParseError::InvalidFormat(format!("Invalid JSON: {}", e)))?;

        let mut terms = Vec::new();
        let mut context = self.context.clone();

        // Extract context if present
        if let Some(ctx) = json.get("@context") {
            self.merge_context(&mut context, ctx);
        }

        // Handle different JSON-LD structures
        match &json {
            Value::Object(obj) => {
                // Single object or graph container
                if let Some(graph) = obj.get("@graph") {
                    // Graph container
                    if let Value::Array(items) = graph {
                        for item in items {
                            if let Some(term) = self.parse_node(item, &context) {
                                terms.push(term);
                            }
                        }
                    }
                } else {
                    // Single object
                    if let Some(term) = self.parse_node(&json, &context) {
                        terms.push(term);
                    }
                }
            }
            Value::Array(arr) => {
                // Array of objects
                for item in arr {
                    if let Some(term) = self.parse_node(item, &context) {
                        terms.push(term);
                    }
                }
            }
            _ => {}
        }

        Ok(terms)
    }

    /// Merge context definitions
    fn merge_context(&self, context: &mut HashMap<String, String>, ctx: &Value) {
        match ctx {
            Value::String(url) => {
                // Remote context - we'd need to fetch it in production
                // For now, just store the URL
                context.insert("@import".to_string(), url.clone());
            }
            Value::Object(obj) => {
                for (key, value) in obj {
                    if let Value::String(iri) = value {
                        context.insert(key.clone(), iri.clone());
                    } else if let Value::Object(term_def) = value {
                        // Term definition object
                        if let Some(Value::String(id)) = term_def.get("@id") {
                            context.insert(key.clone(), id.clone());
                        }
                    }
                }
            }
            Value::Array(arr) => {
                // Array of contexts - merge all
                for item in arr {
                    self.merge_context(context, item);
                }
            }
            _ => {}
        }
    }

    /// Parse a single JSON-LD node into a RawTerm
    fn parse_node(&self, node: &Value, context: &HashMap<String, String>) -> Option<RawTerm> {
        let obj = node.as_object()?;

        // Get the @id (IRI)
        let iri = self.extract_id(obj, context)?;

        // Skip blank nodes for now
        if iri.starts_with("_:") {
            return None;
        }

        let mut term = RawTerm {
            iri,
            ..Default::default()
        };

        // Extract label
        term.label = self.extract_string_property(
            obj,
            &["rdfs:label", "schema:name", "name", "label"],
            context,
        );

        // Extract definition
        term.definition = self.extract_string_property(
            obj,
            &[
                "rdfs:comment",
                "schema:description",
                "description",
                "comment",
            ],
            context,
        );

        // Extract synonyms
        if let Some(synonyms) =
            self.extract_string_array(obj, &["skos:altLabel", "alternateName"], context)
        {
            term.synonyms = synonyms;
        }

        // Extract parents (subClassOf, subPropertyOf)
        term.parents = self.extract_parent_refs(obj, context);

        // Extract type-based relations
        term.relations = self.extract_relations(obj, context);

        // Extract deprecated status
        if let Some(Value::Bool(true)) = obj
            .get("schema:supersededBy")
            .or_else(|| obj.get("supersededBy"))
        {
            term.deprecated = true;
        }
        if let Some(Value::Bool(dep)) = obj.get("owl:deprecated") {
            term.deprecated = *dep;
        }

        // Store @type as metadata
        if let Some(types) = self.extract_types(obj, context) {
            term.metadata.insert("@type".to_string(), types.join(", "));
        }

        Some(term)
    }

    /// Extract @id from a node
    fn extract_id(
        &self,
        obj: &Map<String, Value>,
        context: &HashMap<String, String>,
    ) -> Option<String> {
        if let Some(Value::String(id)) = obj.get("@id") {
            return Some(self.expand_iri(id, context));
        }
        None
    }

    /// Extract @type values
    fn extract_types(
        &self,
        obj: &Map<String, Value>,
        context: &HashMap<String, String>,
    ) -> Option<Vec<String>> {
        let type_val = obj.get("@type")?;

        match type_val {
            Value::String(t) => Some(vec![self.expand_iri(t, context)]),
            Value::Array(arr) => {
                let types: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| self.expand_iri(s, context))
                    .collect();
                if types.is_empty() { None } else { Some(types) }
            }
            _ => None,
        }
    }

    /// Extract a string property trying multiple keys
    fn extract_string_property(
        &self,
        obj: &Map<String, Value>,
        keys: &[&str],
        context: &HashMap<String, String>,
    ) -> Option<String> {
        for key in keys {
            // Try direct key
            if let Some(value) = obj.get(*key)
                && let Some(s) = self.value_to_string(value)
            {
                return Some(s);
            }

            // Try expanded key
            let expanded = self.expand_iri(key, context);
            if let Some(value) = obj.get(&expanded)
                && let Some(s) = self.value_to_string(value)
            {
                return Some(s);
            }
        }
        None
    }

    /// Extract an array of strings from a property
    fn extract_string_array(
        &self,
        obj: &Map<String, Value>,
        keys: &[&str],
        context: &HashMap<String, String>,
    ) -> Option<Vec<String>> {
        for key in keys {
            if let Some(value) = obj
                .get(*key)
                .or_else(|| obj.get(&self.expand_iri(key, context)))
            {
                match value {
                    Value::String(s) => return Some(vec![s.clone()]),
                    Value::Array(arr) => {
                        let strings: Vec<String> =
                            arr.iter().filter_map(|v| self.value_to_string(v)).collect();
                        if !strings.is_empty() {
                            return Some(strings);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Extract parent references
    fn extract_parent_refs(
        &self,
        obj: &Map<String, Value>,
        context: &HashMap<String, String>,
    ) -> Vec<String> {
        let parent_keys = [
            "rdfs:subClassOf",
            "subClassOf",
            "rdfs:subPropertyOf",
            "subPropertyOf",
            "schema:isPartOf",
            "isPartOf",
        ];

        let mut parents = Vec::new();

        for key in parent_keys {
            if let Some(value) = obj
                .get(key)
                .or_else(|| obj.get(&self.expand_iri(key, context)))
            {
                match value {
                    Value::String(s) => {
                        parents.push(self.expand_iri(s, context));
                    }
                    Value::Object(ref_obj) => {
                        if let Some(Value::String(id)) = ref_obj.get("@id") {
                            parents.push(self.expand_iri(id, context));
                        }
                    }
                    Value::Array(arr) => {
                        for item in arr {
                            match item {
                                Value::String(s) => {
                                    parents.push(self.expand_iri(s, context));
                                }
                                Value::Object(ref_obj) => {
                                    if let Some(Value::String(id)) = ref_obj.get("@id") {
                                        parents.push(self.expand_iri(id, context));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        parents
    }

    /// Extract relations from object properties
    fn extract_relations(
        &self,
        obj: &Map<String, Value>,
        context: &HashMap<String, String>,
    ) -> Vec<Relation> {
        let mut relations = Vec::new();

        // Schema.org specific relations
        let relation_keys = [
            ("schema:domainIncludes", "domainIncludes"),
            ("schema:rangeIncludes", "rangeIncludes"),
            ("schema:inverseOf", "inverseOf"),
            ("rdfs:domain", "domain"),
            ("rdfs:range", "range"),
        ];

        for (full_key, short_key) in relation_keys {
            if let Some(value) = obj.get(full_key).or_else(|| obj.get(short_key)) {
                let targets = self.extract_refs(value, context);
                for target in targets {
                    relations.push(Relation {
                        predicate: short_key.to_string(),
                        target,
                    });
                }
            }
        }

        relations
    }

    /// Extract references from a value (string, object with @id, or array)
    fn extract_refs(&self, value: &Value, context: &HashMap<String, String>) -> Vec<String> {
        let mut refs = Vec::new();

        match value {
            Value::String(s) => {
                refs.push(self.expand_iri(s, context));
            }
            Value::Object(obj) => {
                if let Some(Value::String(id)) = obj.get("@id") {
                    refs.push(self.expand_iri(id, context));
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    refs.extend(self.extract_refs(item, context));
                }
            }
            _ => {}
        }

        refs
    }

    /// Convert a JSON value to a string
    fn value_to_string(&self, value: &Value) -> Option<String> {
        match value {
            Value::String(s) => Some(s.clone()),
            Value::Object(obj) => {
                // Language-tagged string or typed literal
                if let Some(Value::String(v)) = obj.get("@value") {
                    Some(v.clone())
                } else {
                    None
                }
            }
            Value::Array(arr) => {
                // Take first string value
                arr.iter().find_map(|v| self.value_to_string(v))
            }
            Value::Number(n) => Some(n.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            _ => None,
        }
    }

    /// Expand a potentially prefixed IRI
    fn expand_iri(&self, iri: &str, context: &HashMap<String, String>) -> String {
        // Already a full IRI
        if iri.starts_with("http://") || iri.starts_with("https://") {
            return iri.to_string();
        }

        // Prefixed name
        if let Some(colon_pos) = iri.find(':') {
            let prefix = &iri[..colon_pos];
            let local = &iri[colon_pos + 1..];

            if let Some(base) = context.get(prefix) {
                return format!("{}{}", base, local);
            }
        }

        // Use @vocab if available
        if let Some(vocab) = context.get("@vocab") {
            return format!("{}{}", vocab, iri);
        }

        iri.to_string()
    }
}

impl OntologyParser for JsonLdParser {
    fn parse<'a>(
        &'a self,
        reader: Box<dyn Read + 'a>,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a> {
        // Read entire content (JSON needs to be parsed as a whole)
        let mut content = String::new();
        let mut reader = reader;

        match reader.read_to_string(&mut content) {
            Ok(_) => match self.parse_content(&content) {
                Ok(terms) => Box::new(terms.into_iter().map(Ok)),
                Err(e) => Box::new(std::iter::once(Err(e))),
            },
            Err(e) => Box::new(std::iter::once(Err(ParseError::IoError(e.to_string())))),
        }
    }

    fn extensions(&self) -> &[&str] {
        &["jsonld", "json-ld"]
    }

    fn can_parse(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap_or("").to_lowercase();
            return ext == "jsonld" || ext == "json-ld";
        }

        // Check if filename contains schema.org pattern
        if let Some(name) = path.file_name() {
            let name = name.to_str().unwrap_or("");
            return name.contains("schema") && name.ends_with(".json");
        }

        false
    }

    fn name(&self) -> &str {
        "JSON-LD Parser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_simple_jsonld() {
        let jsonld = r#"
        {
            "@context": "https://schema.org/",
            "@type": "Person",
            "@id": "https://example.org/person/1",
            "name": "John Doe",
            "description": "A test person"
        }
        "#;

        let parser = JsonLdParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(jsonld));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 1);
        let term = terms[0].as_ref().unwrap();
        assert_eq!(term.iri, "https://example.org/person/1");
        assert_eq!(term.label, Some("John Doe".to_string()));
        assert_eq!(term.definition, Some("A test person".to_string()));
    }

    #[test]
    fn test_graph_container() {
        let jsonld = r#"
        {
            "@context": {
                "schema": "https://schema.org/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "@graph": [
                {
                    "@id": "schema:Thing",
                    "@type": "rdfs:Class",
                    "rdfs:label": "Thing",
                    "rdfs:comment": "The most generic type"
                },
                {
                    "@id": "schema:Person",
                    "@type": "rdfs:Class",
                    "rdfs:label": "Person",
                    "rdfs:comment": "A person",
                    "rdfs:subClassOf": {"@id": "schema:Thing"}
                }
            ]
        }
        "#;

        let parser = JsonLdParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(jsonld));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 2);

        let thing = terms[0].as_ref().unwrap();
        assert!(thing.iri.contains("Thing"));
        assert_eq!(thing.label, Some("Thing".to_string()));

        let person = terms[1].as_ref().unwrap();
        assert!(person.iri.contains("Person"));
        assert_eq!(person.label, Some("Person".to_string()));
        assert!(!person.parents.is_empty());
    }

    #[test]
    fn test_schema_org_class() {
        let jsonld = r#"
        {
            "@context": "https://schema.org/",
            "@type": "rdfs:Class",
            "@id": "schema:MedicalCondition",
            "name": "MedicalCondition",
            "description": "Any condition affecting health",
            "subClassOf": {"@id": "schema:MedicalEntity"}
        }
        "#;

        let parser = JsonLdParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(jsonld));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 1);
        let term = terms[0].as_ref().unwrap();
        assert!(term.iri.contains("MedicalCondition"));
        assert_eq!(term.label, Some("MedicalCondition".to_string()));
        assert!(!term.parents.is_empty());
    }

    #[test]
    fn test_expand_iri() {
        let parser = JsonLdParser::new();
        let mut context = parser.context.clone();
        context.insert("ex".to_string(), "http://example.org/".to_string());

        assert_eq!(
            parser.expand_iri("http://full.url/term", &context),
            "http://full.url/term"
        );

        assert_eq!(
            parser.expand_iri("ex:term1", &context),
            "http://example.org/term1"
        );

        assert_eq!(
            parser.expand_iri("schema:Person", &context),
            "https://schema.org/Person"
        );
    }

    #[test]
    fn test_language_tagged_strings() {
        let jsonld = r#"
        {
            "@id": "https://example.org/term",
            "rdfs:label": {"@value": "Label in English", "@language": "en"},
            "rdfs:comment": [
                {"@value": "English description", "@language": "en"},
                {"@value": "Descripción en español", "@language": "es"}
            ]
        }
        "#;

        let parser = JsonLdParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(jsonld));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 1);
        let term = terms[0].as_ref().unwrap();
        assert_eq!(term.label, Some("Label in English".to_string()));
        // Takes first value from array
        assert_eq!(term.definition, Some("English description".to_string()));
    }

    #[test]
    fn test_domain_range_relations() {
        let jsonld = r#"
        {
            "@context": "https://schema.org/",
            "@type": "rdf:Property",
            "@id": "schema:birthDate",
            "name": "birthDate",
            "description": "Date of birth",
            "domainIncludes": {"@id": "schema:Person"},
            "rangeIncludes": {"@id": "schema:Date"}
        }
        "#;

        let parser = JsonLdParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(jsonld));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 1);
        let term = terms[0].as_ref().unwrap();

        // Check relations were extracted
        assert!(
            term.relations
                .iter()
                .any(|r| r.predicate == "domainIncludes")
        );
        assert!(
            term.relations
                .iter()
                .any(|r| r.predicate == "rangeIncludes")
        );
    }
}
