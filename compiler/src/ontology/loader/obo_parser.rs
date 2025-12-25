//! OBO Format Parser
//!
//! Parses OBO Foundry ontologies into LoadedTerms.
//! OBO is the standard format for biological ontologies.
//!
//! # Format Example
//!
//! ```obo
//! [Term]
//! id: GO:0008150
//! name: biological_process
//! namespace: biological_process
//! def: "A biological process represents a specific objective..."
//! synonym: "biological process" EXACT []
//! synonym: "physiological process" EXACT []
//! is_a: BFO:0000015 ! process
//! ```
//!
//! # Supported Tags
//!
//! - id: Term identifier (CURIE format)
//! - name: Human-readable label
//! - namespace: Ontology namespace
//! - def: Definition with dbxrefs
//! - synonym: Synonym with scope
//! - is_a: Superclass relation
//! - relationship: Other relations
//! - xref: Cross-references
//! - is_obsolete: Obsolescence flag
//! - replaced_by: Replacement term
//! - consider: Alternative terms
//! - property_value: Property annotations
//! - intersection_of: Genus-differentia definition
//! - union_of: Union definition
//! - disjoint_from: Disjointness axiom

use super::{CrossReference, IRI, LoadedTerm, Restriction, RestrictionType, Synonym, SynonymScope};

/// Parse error for OBO files
#[derive(Debug, Clone, thiserror::Error)]
pub enum OboParseError {
    #[error("Invalid line: {0}")]
    InvalidLine(String),

    #[error("Invalid ID: {0}")]
    InvalidId(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid format in tag {tag}: {message}")]
    InvalidFormat { tag: String, message: String },
}

/// Builder for constructing LoadedTerm from OBO stanza
#[derive(Debug, Default)]
struct TermBuilder {
    id: Option<IRI>,
    label: Option<String>,
    namespace: Option<String>,
    definition: Option<String>,
    synonyms: Vec<Synonym>,
    superclasses: Vec<IRI>,
    relationships: Vec<(String, IRI)>,
    xrefs: Vec<CrossReference>,
    is_obsolete: bool,
    replaced_by: Option<IRI>,
    consider: Vec<IRI>,
    property_values: Vec<PropertyValue>,
    intersection_of: Vec<IntersectionComponent>,
    union_of: Vec<IRI>,
    disjoint_from: Vec<IRI>,
    subsets: Vec<String>,
    created_by: Option<String>,
    creation_date: Option<String>,
    annotations: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct PropertyValue {
    property: String,
    value: String,
    datatype: Option<String>,
}

#[derive(Debug, Clone)]
enum IntersectionComponent {
    Genus(IRI),
    Differentia { relation: String, target: IRI },
}

impl TermBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn build(self) -> Result<LoadedTerm, OboParseError> {
        let iri = self
            .id
            .ok_or(OboParseError::MissingField("id".to_string()))?;
        let label = self.label.unwrap_or_else(|| {
            // Use local part of IRI as fallback label
            iri.to_curie()
                .map(|(_, local)| local)
                .unwrap_or_else(|| iri.0.clone())
        });

        let ontology = iri.ontology();

        let mut term = LoadedTerm::new(iri, label, ontology);
        term.definition = self.definition;
        term.synonyms = self.synonyms;
        term.superclasses = self.superclasses;
        term.is_obsolete = self.is_obsolete;
        term.replaced_by = self.replaced_by;

        // Convert relationships to restrictions
        for (rel, target) in self.relationships {
            let rel_iri = IRI::from_curie("RO", &rel);
            term.restrictions.push(Restriction {
                property: rel_iri,
                restriction_type: RestrictionType::Some,
                filler: target,
            });
        }

        // Convert xrefs
        term.xrefs = self.xrefs;

        Ok(term)
    }
}

/// Parse an entire OBO file
pub fn parse_obo_file(content: &str) -> Result<Vec<LoadedTerm>, OboParseError> {
    let mut terms = Vec::new();
    let mut current_term: Option<TermBuilder> = None;
    let mut in_term_stanza = false;

    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('!') {
            continue;
        }

        // Check for stanza headers
        if line == "[Term]" {
            // Save previous term if any
            if let Some(builder) = current_term.take()
                && (!builder.is_obsolete || builder.replaced_by.is_some())
            {
                terms.push(builder.build()?);
            }
            current_term = Some(TermBuilder::new());
            in_term_stanza = true;
            continue;
        }

        if line.starts_with('[') {
            // Other stanza type ([Typedef], [Instance], etc.)
            if let Some(builder) = current_term.take()
                && (!builder.is_obsolete || builder.replaced_by.is_some())
            {
                terms.push(builder.build()?);
            }
            current_term = None;
            in_term_stanza = false;
            continue;
        }

        // Parse tag-value pairs within Term stanza
        if in_term_stanza && let Some(ref mut builder) = current_term {
            parse_term_line(line, builder)?;
        }
    }

    // Don't forget the last term
    if let Some(builder) = current_term
        && (!builder.is_obsolete || builder.replaced_by.is_some())
    {
        terms.push(builder.build()?);
    }

    Ok(terms)
}

fn parse_term_line(line: &str, builder: &mut TermBuilder) -> Result<(), OboParseError> {
    let (tag, value) = line
        .split_once(':')
        .ok_or_else(|| OboParseError::InvalidLine(line.to_string()))?;

    let tag = tag.trim();
    let value = value.trim();

    match tag {
        "id" => builder.id = Some(parse_id(value)?),
        "name" => builder.label = Some(value.to_string()),
        "namespace" => builder.namespace = Some(value.to_string()),
        "def" => builder.definition = Some(parse_def(value)?),
        "synonym" => builder.synonyms.push(parse_synonym(value)?),
        "is_a" => builder.superclasses.push(parse_is_a(value)?),
        "relationship" => {
            let (rel, target) = parse_relationship(value)?;
            builder.relationships.push((rel, target));
        }
        "xref" => builder.xrefs.push(parse_xref(value)?),
        "is_obsolete" => {
            builder.is_obsolete = value == "true";
        }
        "replaced_by" => builder.replaced_by = Some(parse_id(value)?),
        "consider" => builder.consider.push(parse_id(value)?),
        "property_value" => {
            if let Ok(pv) = parse_property_value(value) {
                builder.property_values.push(pv);
            }
        }
        "intersection_of" => {
            if let Ok(ic) = parse_intersection(value) {
                builder.intersection_of.push(ic);
            }
        }
        "union_of" => {
            if let Ok(id) = parse_id(value) {
                builder.union_of.push(id);
            }
        }
        "disjoint_from" => {
            if let Ok(id) = parse_id(value) {
                builder.disjoint_from.push(id);
            }
        }
        "subset" => builder.subsets.push(value.to_string()),
        "created_by" => builder.created_by = Some(value.to_string()),
        "creation_date" => builder.creation_date = Some(value.to_string()),
        "alt_id" | "comment" | "equivalent_to" => {
            // Store as annotation
            builder
                .annotations
                .push((tag.to_string(), value.to_string()));
        }
        _ => {
            // Unknown tag, store as annotation
            builder
                .annotations
                .push((tag.to_string(), value.to_string()));
        }
    }

    Ok(())
}

/// Parse an OBO ID like "GO:0008150" into an IRI
fn parse_id(s: &str) -> Result<IRI, OboParseError> {
    // Remove trailing comment (! ...)
    let s = s.split('!').next().unwrap().trim();

    if let Some((prefix, local)) = s.split_once(':') {
        Ok(IRI::from_curie(prefix.trim(), local.trim()))
    } else {
        Err(OboParseError::InvalidId(s.to_string()))
    }
}

/// Parse a definition: "text" [dbxrefs]
fn parse_def(s: &str) -> Result<String, OboParseError> {
    // Definition format: "text" [dbxref, dbxref, ...]
    if s.starts_with('"') {
        // Find the closing quote
        if let Some(end) = s[1..].find('"') {
            return Ok(s[1..end + 1].to_string());
        }
    }

    // Fallback: return as-is
    Ok(s.to_string())
}

/// Parse a synonym: "text" SCOPE [dbxrefs]
fn parse_synonym(s: &str) -> Result<Synonym, OboParseError> {
    // Synonym format: "text" SCOPE [dbxref, ...] {annotations}

    // Find the quoted text
    if !s.starts_with('"') {
        return Ok(Synonym {
            text: s.to_string(),
            scope: SynonymScope::Related,
        });
    }

    // Find closing quote
    let end_quote = s[1..]
        .find('"')
        .ok_or_else(|| OboParseError::InvalidFormat {
            tag: "synonym".to_string(),
            message: "missing closing quote".to_string(),
        })?;

    let text = s[1..end_quote + 1].to_string();
    let rest = s[end_quote + 2..].trim();

    // Parse scope
    let scope = if rest.starts_with("EXACT") {
        SynonymScope::Exact
    } else if rest.starts_with("NARROW") {
        SynonymScope::Narrow
    } else if rest.starts_with("BROAD") {
        SynonymScope::Broad
    } else {
        SynonymScope::Related
    };

    Ok(Synonym { text, scope })
}

/// Parse is_a: PARENT_ID ! comment
fn parse_is_a(s: &str) -> Result<IRI, OboParseError> {
    parse_id(s)
}

/// Parse relationship: REL_TYPE TARGET_ID ! comment
fn parse_relationship(s: &str) -> Result<(String, IRI), OboParseError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(OboParseError::InvalidFormat {
            tag: "relationship".to_string(),
            message: "expected 'relation target'".to_string(),
        });
    }

    let relation = parts[0].to_string();
    let target = parse_id(parts[1])?;

    Ok((relation, target))
}

/// Parse xref: PREFIX:ID or PREFIX:ID "description"
fn parse_xref(s: &str) -> Result<CrossReference, OboParseError> {
    // Remove any trailing description in quotes
    let s = if let Some(idx) = s.find('"') {
        s[..idx].trim()
    } else {
        s.trim()
    };

    let target = parse_id(s)?;

    Ok(CrossReference {
        target,
        confidence: 0.8, // Default confidence for xrefs
        source: "obo_xref".to_string(),
    })
}

/// Parse property_value: PROPERTY VALUE DATATYPE
fn parse_property_value(s: &str) -> Result<PropertyValue, OboParseError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return Err(OboParseError::InvalidFormat {
            tag: "property_value".to_string(),
            message: "empty property_value".to_string(),
        });
    }

    let property = parts[0].to_string();
    let value = if parts.len() > 1 {
        // Handle quoted values
        if parts[1].starts_with('"') {
            let rest = parts[1..].join(" ");
            if let Some(end) = rest[1..].find('"') {
                rest[1..end + 1].to_string()
            } else {
                rest
            }
        } else {
            parts[1].to_string()
        }
    } else {
        String::new()
    };

    let datatype = parts.get(2).map(|s| s.to_string());

    Ok(PropertyValue {
        property,
        value,
        datatype,
    })
}

/// Parse intersection_of: RELATION TARGET or just TARGET (for genus)
fn parse_intersection(s: &str) -> Result<IntersectionComponent, OboParseError> {
    let parts: Vec<&str> = s.split_whitespace().collect();

    if parts.len() == 1 {
        // Just a genus (class ID)
        Ok(IntersectionComponent::Genus(parse_id(parts[0])?))
    } else if parts.len() >= 2 {
        // Differentia: relation target
        let relation = parts[0].to_string();
        let target = parse_id(parts[1])?;
        Ok(IntersectionComponent::Differentia { relation, target })
    } else {
        Err(OboParseError::InvalidFormat {
            tag: "intersection_of".to_string(),
            message: "expected 'class' or 'relation class'".to_string(),
        })
    }
}

/// Parse OBO header to extract metadata
pub fn parse_obo_header(content: &str) -> OboHeader {
    let mut header = OboHeader::default();

    for line in content.lines() {
        let line = line.trim();

        // Stop at first stanza
        if line.starts_with('[') {
            break;
        }

        if line.is_empty() || line.starts_with('!') {
            continue;
        }

        if let Some((tag, value)) = line.split_once(':') {
            let tag = tag.trim();
            let value = value.trim();

            match tag {
                "format-version" => header.format_version = Some(value.to_string()),
                "data-version" => header.data_version = Some(value.to_string()),
                "ontology" => header.ontology = Some(value.to_string()),
                "date" => header.date = Some(value.to_string()),
                "default-namespace" => header.default_namespace = Some(value.to_string()),
                "saved-by" => header.saved_by = Some(value.to_string()),
                "auto-generated-by" => header.auto_generated_by = Some(value.to_string()),
                "subsetdef" => header.subsetdefs.push(value.to_string()),
                "import" => header.imports.push(value.to_string()),
                "synonymtypedef" => header.synonym_typedefs.push(value.to_string()),
                "idspace" => header.idspaces.push(value.to_string()),
                "remark" => header.remarks.push(value.to_string()),
                _ => {}
            }
        }
    }

    header
}

/// OBO file header information
#[derive(Debug, Default, Clone)]
pub struct OboHeader {
    pub format_version: Option<String>,
    pub data_version: Option<String>,
    pub ontology: Option<String>,
    pub date: Option<String>,
    pub default_namespace: Option<String>,
    pub saved_by: Option<String>,
    pub auto_generated_by: Option<String>,
    pub subsetdefs: Vec<String>,
    pub imports: Vec<String>,
    pub synonym_typedefs: Vec<String>,
    pub idspaces: Vec<String>,
    pub remarks: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_id() {
        let iri = parse_id("GO:0008150").unwrap();
        assert!(iri.0.contains("GO_0008150"));

        let iri = parse_id("GO:0008150 ! biological_process").unwrap();
        assert!(iri.0.contains("GO_0008150"));
    }

    #[test]
    fn test_parse_def() {
        let def = parse_def(r#""A biological process" [GOC:mtg_sensu]"#).unwrap();
        assert_eq!(def, "A biological process");
    }

    #[test]
    fn test_parse_synonym() {
        let syn = parse_synonym(r#""physiological process" EXACT []"#).unwrap();
        assert_eq!(syn.text, "physiological process");
        assert_eq!(syn.scope, SynonymScope::Exact);

        let syn = parse_synonym(r#""cell growth" NARROW []"#).unwrap();
        assert_eq!(syn.scope, SynonymScope::Narrow);
    }

    #[test]
    fn test_parse_relationship() {
        let (rel, target) = parse_relationship("part_of GO:0008150").unwrap();
        assert_eq!(rel, "part_of");
        assert!(target.0.contains("GO_0008150"));
    }

    #[test]
    fn test_parse_simple_obo() {
        let obo = r#"
format-version: 1.2
ontology: go

[Term]
id: GO:0008150
name: biological_process
def: "A biological process" [GOC:go]
synonym: "biological process" EXACT []
is_a: BFO:0000015

[Term]
id: GO:0009987
name: cellular process
def: "A process that is carried out at the cellular level" [GOC:go]
is_a: GO:0008150
"#;

        let terms = parse_obo_file(obo).unwrap();
        assert_eq!(terms.len(), 2);

        let bp = &terms[0];
        assert_eq!(bp.label, "biological_process");
        assert!(!bp.superclasses.is_empty());

        let cp = &terms[1];
        assert_eq!(cp.label, "cellular process");
        assert!(cp.superclasses.iter().any(|s| s.0.contains("GO_0008150")));
    }

    #[test]
    fn test_parse_header() {
        let obo = r#"
format-version: 1.2
data-version: releases/2024-01-01
ontology: go
default-namespace: gene_ontology

[Term]
id: GO:0008150
"#;

        let header = parse_obo_header(obo);
        assert_eq!(header.format_version, Some("1.2".to_string()));
        assert_eq!(header.ontology, Some("go".to_string()));
    }

    #[test]
    fn test_obsolete_term_handling() {
        let obo = r#"
[Term]
id: GO:0000001
name: obsolete term
is_obsolete: true

[Term]
id: GO:0000002
name: active term
"#;

        let terms = parse_obo_file(obo).unwrap();
        // Obsolete term without replacement should be excluded
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].label, "active term");
    }
}
