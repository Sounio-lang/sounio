//! FHIR JSON Parser for Ontology Store Builder
//!
//! Parses FHIR (Fast Healthcare Interoperability Resources) JSON format.
//! Supports:
//! - CodeSystem resources
//! - ValueSet resources
//! - StructureDefinition resources
//! - ConceptMap resources

use std::io::Read;
use std::path::Path;

use serde_json::{Map, Value};

use super::{OntologyParser, ParseError, RawTerm, Relation};

/// FHIR JSON parser for healthcare terminology.
pub struct FhirParser {
    /// Base URL for FHIR resources
    base_url: String,
}

impl Default for FhirParser {
    fn default() -> Self {
        Self::new()
    }
}

impl FhirParser {
    pub fn new() -> Self {
        Self {
            base_url: "http://hl7.org/fhir/".to_string(),
        }
    }

    /// Create parser with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Parse FHIR JSON content
    fn parse_content(&self, content: &str) -> Result<Vec<RawTerm>, ParseError> {
        let json: Value = serde_json::from_str(content)
            .map_err(|e| ParseError::InvalidFormat(format!("Invalid JSON: {}", e)))?;

        let obj = json
            .as_object()
            .ok_or_else(|| ParseError::InvalidFormat("Expected JSON object".to_string()))?;

        // Check resourceType to determine parsing strategy
        let resource_type = obj
            .get("resourceType")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        match resource_type {
            "CodeSystem" => self.parse_code_system(obj),
            "ValueSet" => self.parse_value_set(obj),
            "StructureDefinition" => self.parse_structure_definition(obj),
            "ConceptMap" => self.parse_concept_map(obj),
            "Bundle" => self.parse_bundle(obj),
            _ => Err(ParseError::InvalidFormat(format!(
                "Unsupported FHIR resource type: {}",
                resource_type
            ))),
        }
    }

    /// Parse a FHIR CodeSystem resource
    fn parse_code_system(&self, obj: &Map<String, Value>) -> Result<Vec<RawTerm>, ParseError> {
        let mut terms = Vec::new();

        // Get the CodeSystem URL (serves as prefix for concept IRIs)
        let system_url = obj
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.base_url);

        let system_name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        let system_version = obj.get("version").and_then(|v| v.as_str());

        // Parse concepts
        if let Some(Value::Array(concepts)) = obj.get("concept") {
            self.parse_concepts(&mut terms, concepts, system_url, None)?;
        }

        // Add metadata as a root term for the CodeSystem itself
        if let Some(url) = obj.get("url").and_then(|v| v.as_str()) {
            let mut root = RawTerm {
                iri: url.to_string(),
                label: Some(system_name.to_string()),
                definition: obj
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                ..Default::default()
            };

            if let Some(v) = system_version {
                root.metadata.insert("version".to_string(), v.to_string());
            }

            root.metadata
                .insert("resourceType".to_string(), "CodeSystem".to_string());
            terms.insert(0, root);
        }

        Ok(terms)
    }

    /// Recursively parse FHIR concepts (supports hierarchy)
    fn parse_concepts(
        &self,
        terms: &mut Vec<RawTerm>,
        concepts: &[Value],
        system_url: &str,
        parent_code: Option<&str>,
    ) -> Result<(), ParseError> {
        for concept in concepts {
            if let Some(obj) = concept.as_object() {
                let code = obj
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ParseError::MissingField("code".to_string()))?;

                // Construct IRI from system URL and code
                let iri = format!("{}#{}", system_url.trim_end_matches('/'), code);

                let mut term = RawTerm {
                    iri,
                    label: obj
                        .get("display")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    definition: obj
                        .get("definition")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    ..Default::default()
                };

                // Add parent relationship
                if let Some(parent) = parent_code {
                    term.parents
                        .push(format!("{}#{}", system_url.trim_end_matches('/'), parent));
                }

                // Extract designations (synonyms)
                if let Some(Value::Array(designations)) = obj.get("designation") {
                    for des in designations {
                        if let Some(des_obj) = des.as_object()
                            && let Some(Value::String(value)) = des_obj.get("value")
                        {
                            // Don't add if it's the same as the display
                            if term.label.as_ref() != Some(value) {
                                term.synonyms.push(value.clone());
                            }
                        }
                    }
                }

                // Extract properties
                if let Some(Value::Array(properties)) = obj.get("property") {
                    for prop in properties {
                        if let Some(prop_obj) = prop.as_object() {
                            let prop_code = prop_obj.get("code").and_then(|v| v.as_str());

                            // Handle different property value types
                            let prop_value = prop_obj
                                .get("valueCode")
                                .or_else(|| prop_obj.get("valueString"))
                                .or_else(|| prop_obj.get("valueCoding"))
                                .or_else(|| prop_obj.get("valueBoolean"));

                            if let (Some(code), Some(value)) = (prop_code, prop_value) {
                                match value {
                                    Value::String(s) => {
                                        // Check for deprecation
                                        if code == "status" && s == "deprecated" {
                                            term.deprecated = true;
                                        } else if code == "parent" {
                                            // Parent reference via property
                                            term.parents.push(format!(
                                                "{}#{}",
                                                system_url.trim_end_matches('/'),
                                                s
                                            ));
                                        } else {
                                            term.metadata.insert(code.to_string(), s.clone());
                                        }
                                    }
                                    Value::Bool(b) => {
                                        if code == "deprecated" && *b {
                                            term.deprecated = true;
                                        }
                                        term.metadata.insert(code.to_string(), b.to_string());
                                    }
                                    Value::Object(coding) => {
                                        // valueCoding - extract code
                                        if let Some(Value::String(c)) = coding.get("code") {
                                            if code == "parent" {
                                                let sys = coding
                                                    .get("system")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or(system_url);
                                                term.parents.push(format!(
                                                    "{}#{}",
                                                    sys.trim_end_matches('/'),
                                                    c
                                                ));
                                            } else {
                                                term.relations.push(Relation {
                                                    predicate: code.to_string(),
                                                    target: c.clone(),
                                                });
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                terms.push(term);

                // Recursively parse child concepts
                if let Some(Value::Array(children)) = obj.get("concept") {
                    self.parse_concepts(terms, children, system_url, Some(code))?;
                }
            }
        }

        Ok(())
    }

    /// Parse a FHIR ValueSet resource
    fn parse_value_set(&self, obj: &Map<String, Value>) -> Result<Vec<RawTerm>, ParseError> {
        let mut terms = Vec::new();

        let valueset_url = obj
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.base_url);

        // Add ValueSet as a term
        let mut root = RawTerm {
            iri: valueset_url.to_string(),
            label: obj
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            definition: obj
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            ..Default::default()
        };
        root.metadata
            .insert("resourceType".to_string(), "ValueSet".to_string());
        terms.push(root);

        // Parse compose section
        if let Some(compose) = obj.get("compose").and_then(|v| v.as_object())
            && let Some(Value::Array(includes)) = compose.get("include")
        {
            for include in includes {
                if let Some(inc_obj) = include.as_object() {
                    let system = inc_obj.get("system").and_then(|v| v.as_str()).unwrap_or("");

                    // Parse concepts in the include
                    if let Some(Value::Array(concepts)) = inc_obj.get("concept") {
                        for concept in concepts {
                            if let Some(c_obj) = concept.as_object()
                                && let Some(Value::String(code)) = c_obj.get("code")
                            {
                                let term = RawTerm {
                                    iri: format!("{}#{}", system.trim_end_matches('/'), code),
                                    label: c_obj
                                        .get("display")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string()),
                                    ..Default::default()
                                };
                                terms.push(term);
                            }
                        }
                    }
                }
            }
        }

        // Parse expansion section (if present)
        if let Some(expansion) = obj.get("expansion").and_then(|v| v.as_object())
            && let Some(Value::Array(contains)) = expansion.get("contains")
        {
            self.parse_expansion_contains(&mut terms, contains)?;
        }

        Ok(terms)
    }

    /// Parse ValueSet expansion contains
    fn parse_expansion_contains(
        &self,
        terms: &mut Vec<RawTerm>,
        contains: &[Value],
    ) -> Result<(), ParseError> {
        for item in contains {
            if let Some(obj) = item.as_object() {
                let system = obj.get("system").and_then(|v| v.as_str()).unwrap_or("");

                if let Some(Value::String(code)) = obj.get("code") {
                    let term = RawTerm {
                        iri: format!("{}#{}", system.trim_end_matches('/'), code),
                        label: obj
                            .get("display")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        deprecated: obj
                            .get("inactive")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        ..Default::default()
                    };
                    terms.push(term);
                }

                // Recursively parse nested contains
                if let Some(Value::Array(nested)) = obj.get("contains") {
                    self.parse_expansion_contains(terms, nested)?;
                }
            }
        }

        Ok(())
    }

    /// Parse a FHIR StructureDefinition resource
    fn parse_structure_definition(
        &self,
        obj: &Map<String, Value>,
    ) -> Result<Vec<RawTerm>, ParseError> {
        let mut terms = Vec::new();

        let url = obj
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ParseError::MissingField("url".to_string()))?;

        let mut root = RawTerm {
            iri: url.to_string(),
            label: obj
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            definition: obj
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            ..Default::default()
        };

        // Add kind and type as metadata
        if let Some(Value::String(kind)) = obj.get("kind") {
            root.metadata.insert("kind".to_string(), kind.clone());
        }
        if let Some(Value::String(type_)) = obj.get("type") {
            root.metadata.insert("type".to_string(), type_.clone());
        }

        // Add base type as parent
        if let Some(Value::String(base)) = obj.get("baseDefinition") {
            root.parents.push(base.clone());
        }

        root.metadata.insert(
            "resourceType".to_string(),
            "StructureDefinition".to_string(),
        );
        terms.push(root);

        // Parse elements if this is a detailed structure
        if let Some(snapshot) = obj.get("snapshot").and_then(|v| v.as_object())
            && let Some(Value::Array(elements)) = snapshot.get("element")
        {
            for element in elements {
                if let Some(elem_obj) = element.as_object()
                    && let Some(term) = self.parse_element(elem_obj, url)
                {
                    terms.push(term);
                }
            }
        }

        Ok(terms)
    }

    /// Parse a StructureDefinition element
    fn parse_element(&self, obj: &Map<String, Value>, base_url: &str) -> Option<RawTerm> {
        let id = obj.get("id").and_then(|v| v.as_str())?;
        let path = obj.get("path").and_then(|v| v.as_str())?;

        // Skip the root element
        if !path.contains('.') {
            return None;
        }

        let mut term = RawTerm {
            iri: format!("{}#{}", base_url, id),
            label: obj
                .get("short")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            definition: obj
                .get("definition")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            ..Default::default()
        };

        // Add path as metadata
        term.metadata.insert("path".to_string(), path.to_string());

        // Extract type information
        if let Some(Value::Array(types)) = obj.get("type") {
            for type_val in types {
                if let Some(type_obj) = type_val.as_object()
                    && let Some(Value::String(code)) = type_obj.get("code")
                {
                    term.relations.push(Relation {
                        predicate: "type".to_string(),
                        target: code.clone(),
                    });
                }
            }
        }

        // Check cardinality for optionality
        let min = obj.get("min").and_then(|v| v.as_u64()).unwrap_or(0);
        let max = obj.get("max").and_then(|v| v.as_str()).unwrap_or("*");
        term.metadata
            .insert("cardinality".to_string(), format!("{}..{}", min, max));

        Some(term)
    }

    /// Parse a FHIR ConceptMap resource
    fn parse_concept_map(&self, obj: &Map<String, Value>) -> Result<Vec<RawTerm>, ParseError> {
        let mut terms = Vec::new();

        let url = obj
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.base_url);

        // Add ConceptMap as root term
        let mut root = RawTerm {
            iri: url.to_string(),
            label: obj
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            definition: obj
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            ..Default::default()
        };
        root.metadata
            .insert("resourceType".to_string(), "ConceptMap".to_string());

        // Add source and target as relations
        if let Some(Value::String(source)) =
            obj.get("sourceUri").or_else(|| obj.get("sourceCanonical"))
        {
            root.relations.push(Relation {
                predicate: "sourceUri".to_string(),
                target: source.clone(),
            });
        }
        if let Some(Value::String(target)) =
            obj.get("targetUri").or_else(|| obj.get("targetCanonical"))
        {
            root.relations.push(Relation {
                predicate: "targetUri".to_string(),
                target: target.clone(),
            });
        }

        terms.push(root);

        // Parse groups (mappings)
        if let Some(Value::Array(groups)) = obj.get("group") {
            for group in groups {
                if let Some(group_obj) = group.as_object() {
                    let source_sys = group_obj
                        .get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let target_sys = group_obj
                        .get("target")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if let Some(Value::Array(elements)) = group_obj.get("element") {
                        for element in elements {
                            if let Some(elem_obj) = element.as_object() {
                                self.parse_mapping_element(
                                    &mut terms, elem_obj, source_sys, target_sys,
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(terms)
    }

    /// Parse a ConceptMap element (mapping)
    fn parse_mapping_element(
        &self,
        terms: &mut Vec<RawTerm>,
        obj: &Map<String, Value>,
        source_sys: &str,
        target_sys: &str,
    ) {
        let source_code = match obj.get("code").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return,
        };

        let source_iri = format!("{}#{}", source_sys.trim_end_matches('/'), source_code);

        let mut term = RawTerm {
            iri: source_iri,
            label: obj
                .get("display")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            ..Default::default()
        };

        // Parse targets
        if let Some(Value::Array(targets)) = obj.get("target") {
            for target in targets {
                if let Some(t_obj) = target.as_object()
                    && let Some(Value::String(t_code)) = t_obj.get("code")
                {
                    let equivalence = t_obj
                        .get("equivalence")
                        .and_then(|v| v.as_str())
                        .unwrap_or("equivalent");

                    term.relations.push(Relation {
                        predicate: format!("mapsTo:{}", equivalence),
                        target: format!("{}#{}", target_sys.trim_end_matches('/'), t_code),
                    });
                }
            }
        }

        terms.push(term);
    }

    /// Parse a FHIR Bundle resource
    fn parse_bundle(&self, obj: &Map<String, Value>) -> Result<Vec<RawTerm>, ParseError> {
        let mut terms = Vec::new();

        if let Some(Value::Array(entries)) = obj.get("entry") {
            for entry in entries {
                if let Some(entry_obj) = entry.as_object()
                    && let Some(resource) = entry_obj.get("resource").and_then(|v| v.as_object())
                {
                    // Recursively parse each resource in the bundle
                    let resource_type = resource
                        .get("resourceType")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown");

                    let resource_terms = match resource_type {
                        "CodeSystem" => self.parse_code_system(resource),
                        "ValueSet" => self.parse_value_set(resource),
                        "StructureDefinition" => self.parse_structure_definition(resource),
                        "ConceptMap" => self.parse_concept_map(resource),
                        _ => continue,
                    };

                    if let Ok(t) = resource_terms {
                        terms.extend(t);
                    }
                }
            }
        }

        Ok(terms)
    }
}

impl OntologyParser for FhirParser {
    fn parse<'a>(
        &'a self,
        reader: Box<dyn Read + 'a>,
    ) -> Box<dyn Iterator<Item = Result<RawTerm, ParseError>> + 'a> {
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
        &["json"]
    }

    fn can_parse(&self, path: &Path) -> bool {
        // Check for FHIR-specific patterns
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            let name_lower = name.to_lowercase();

            // Common FHIR file naming patterns
            if name_lower.contains("codesystem")
                || name_lower.contains("valueset")
                || name_lower.contains("structuredefinition")
                || name_lower.contains("conceptmap")
                || name_lower.contains("fhir")
            {
                return name.ends_with(".json");
            }
        }

        false
    }

    fn name(&self) -> &str {
        "FHIR JSON Parser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_code_system_parsing() {
        let fhir = r#"
        {
            "resourceType": "CodeSystem",
            "url": "http://example.org/fhir/CodeSystem/test",
            "name": "TestCodeSystem",
            "description": "A test code system",
            "version": "1.0.0",
            "concept": [
                {
                    "code": "A",
                    "display": "Concept A",
                    "definition": "First concept",
                    "concept": [
                        {
                            "code": "A1",
                            "display": "Concept A1",
                            "definition": "Child of A"
                        }
                    ]
                },
                {
                    "code": "B",
                    "display": "Concept B"
                }
            ]
        }
        "#;

        let parser = FhirParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(fhir));
        let terms: Vec<_> = parser.parse(reader).collect();

        // Root term + 3 concepts (A, A1, B)
        assert_eq!(terms.len(), 4);

        // Check root term
        let root = terms[0].as_ref().unwrap();
        assert!(root.iri.contains("test"));
        assert_eq!(root.label, Some("TestCodeSystem".to_string()));

        // Check concept A
        let a = terms[1].as_ref().unwrap();
        assert!(a.iri.ends_with("#A"));
        assert_eq!(a.label, Some("Concept A".to_string()));

        // Check concept A1 has parent A
        let a1 = terms[2].as_ref().unwrap();
        assert!(a1.iri.ends_with("#A1"));
        assert!(a1.parents.iter().any(|p| p.ends_with("#A")));
    }

    #[test]
    fn test_value_set_parsing() {
        let fhir = r#"
        {
            "resourceType": "ValueSet",
            "url": "http://example.org/fhir/ValueSet/test",
            "name": "TestValueSet",
            "compose": {
                "include": [
                    {
                        "system": "http://example.org/system",
                        "concept": [
                            {"code": "X", "display": "Concept X"},
                            {"code": "Y", "display": "Concept Y"}
                        ]
                    }
                ]
            }
        }
        "#;

        let parser = FhirParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(fhir));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 3); // ValueSet + 2 concepts

        let x = terms[1].as_ref().unwrap();
        assert!(x.iri.contains("#X"));
    }

    #[test]
    fn test_structure_definition_parsing() {
        let fhir = r#"
        {
            "resourceType": "StructureDefinition",
            "url": "http://hl7.org/fhir/StructureDefinition/Patient",
            "name": "Patient",
            "description": "Demographics and administrative information about a patient",
            "kind": "resource",
            "type": "Patient",
            "baseDefinition": "http://hl7.org/fhir/StructureDefinition/DomainResource",
            "snapshot": {
                "element": [
                    {
                        "id": "Patient",
                        "path": "Patient"
                    },
                    {
                        "id": "Patient.name",
                        "path": "Patient.name",
                        "short": "A name associated with the patient",
                        "definition": "A name associated with the individual.",
                        "min": 0,
                        "max": "*",
                        "type": [{"code": "HumanName"}]
                    }
                ]
            }
        }
        "#;

        let parser = FhirParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(fhir));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 2); // Patient + name element

        let patient = terms[0].as_ref().unwrap();
        assert!(patient.iri.contains("Patient"));
        assert!(patient.parents.iter().any(|p| p.contains("DomainResource")));

        let name = terms[1].as_ref().unwrap();
        assert!(name.iri.contains("Patient.name"));
        assert!(name.relations.iter().any(|r| r.target == "HumanName"));
    }

    #[test]
    fn test_concept_map_parsing() {
        let fhir = r#"
        {
            "resourceType": "ConceptMap",
            "url": "http://example.org/fhir/ConceptMap/test",
            "name": "TestMapping",
            "sourceUri": "http://source.org/CodeSystem/A",
            "targetUri": "http://target.org/CodeSystem/B",
            "group": [
                {
                    "source": "http://source.org/CodeSystem/A",
                    "target": "http://target.org/CodeSystem/B",
                    "element": [
                        {
                            "code": "S1",
                            "display": "Source 1",
                            "target": [
                                {"code": "T1", "equivalence": "equivalent"}
                            ]
                        }
                    ]
                }
            ]
        }
        "#;

        let parser = FhirParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(fhir));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 2); // ConceptMap + 1 mapping

        let mapping = terms[1].as_ref().unwrap();
        assert!(mapping.iri.contains("#S1"));
        assert!(mapping.relations.iter().any(|r| r.target.contains("#T1")));
    }

    #[test]
    fn test_bundle_parsing() {
        let fhir = r#"
        {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "CodeSystem",
                        "url": "http://example.org/cs1",
                        "name": "CS1",
                        "concept": [{"code": "A", "display": "A"}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "ValueSet",
                        "url": "http://example.org/vs1",
                        "name": "VS1"
                    }
                }
            ]
        }
        "#;

        let parser = FhirParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(fhir));
        let terms: Vec<_> = parser.parse(reader).collect();

        // CS1 root + concept A + VS1 root
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_can_parse() {
        let parser = FhirParser::new();

        assert!(parser.can_parse(Path::new("CodeSystem-example.json")));
        assert!(parser.can_parse(Path::new("valueset-test.json")));
        assert!(parser.can_parse(Path::new("StructureDefinition-Patient.json")));
        assert!(parser.can_parse(Path::new("fhir-bundle.json")));
        assert!(!parser.can_parse(Path::new("regular-data.json")));
    }

    #[test]
    fn test_deprecated_concepts() {
        let fhir = r#"
        {
            "resourceType": "CodeSystem",
            "url": "http://example.org/cs",
            "name": "Test",
            "concept": [
                {
                    "code": "OLD",
                    "display": "Old Concept",
                    "property": [
                        {"code": "deprecated", "valueBoolean": true}
                    ]
                }
            ]
        }
        "#;

        let parser = FhirParser::new();
        let reader: Box<dyn Read> = Box::new(Cursor::new(fhir));
        let terms: Vec<_> = parser.parse(reader).collect();

        assert_eq!(terms.len(), 2);

        let old = terms[1].as_ref().unwrap();
        assert!(old.deprecated);
    }
}
