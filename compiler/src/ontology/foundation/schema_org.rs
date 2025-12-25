//! Schema.org - Web Vocabulary
//!
//! ~2,850 types describing web content and structured data.
//! Essential for interoperability with web standards.

use std::collections::HashMap;
use std::path::Path;

use crate::epistemic::TermId;

use super::{CurationStatus, OntologySource, TermEntry, TermMapping};
use crate::ontology::OntologyError;

pub struct SchemaOrgOntology {
    types: HashMap<String, SchemaType>,
    properties: HashMap<String, SchemaProperty>,
    mappings: HashMap<String, Vec<TermMapping>>,
}

#[derive(Debug, Clone)]
pub struct SchemaType {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub parents: Vec<String>,
    pub properties: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SchemaProperty {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub domain: Vec<String>,
    pub range: Vec<String>,
}

impl SchemaOrgOntology {
    pub fn load(_path: &Path) -> Result<Self, OntologyError> {
        Ok(Self::bootstrap())
    }

    /// Create with bootstrap data (essential Schema.org types)
    pub fn bootstrap() -> Self {
        let mut types = HashMap::new();
        let mut properties = HashMap::new();

        // Root type
        types.insert(
            "schema:Thing".into(),
            SchemaType {
                id: "schema:Thing".into(),
                label: "Thing".into(),
                description: Some("The most generic type of item".into()),
                parents: vec![],
                properties: vec![
                    "schema:name".into(),
                    "schema:description".into(),
                    "schema:identifier".into(),
                ],
            },
        );

        // Creative works
        types.insert(
            "schema:CreativeWork".into(),
            SchemaType {
                id: "schema:CreativeWork".into(),
                label: "CreativeWork".into(),
                description: Some("The most generic kind of creative work".into()),
                parents: vec!["schema:Thing".into()],
                properties: vec![
                    "schema:author".into(),
                    "schema:dateCreated".into(),
                    "schema:license".into(),
                ],
            },
        );

        types.insert(
            "schema:Dataset".into(),
            SchemaType {
                id: "schema:Dataset".into(),
                label: "Dataset".into(),
                description: Some("A body of structured information".into()),
                parents: vec!["schema:CreativeWork".into()],
                properties: vec![
                    "schema:distribution".into(),
                    "schema:variableMeasured".into(),
                ],
            },
        );

        types.insert(
            "schema:SoftwareApplication".into(),
            SchemaType {
                id: "schema:SoftwareApplication".into(),
                label: "SoftwareApplication".into(),
                description: Some("A software application".into()),
                parents: vec!["schema:CreativeWork".into()],
                properties: vec![
                    "schema:softwareVersion".into(),
                    "schema:operatingSystem".into(),
                ],
            },
        );

        types.insert(
            "schema:SoftwareSourceCode".into(),
            SchemaType {
                id: "schema:SoftwareSourceCode".into(),
                label: "SoftwareSourceCode".into(),
                description: Some("Computer programming source code".into()),
                parents: vec!["schema:CreativeWork".into()],
                properties: vec![
                    "schema:programmingLanguage".into(),
                    "schema:codeRepository".into(),
                ],
            },
        );

        types.insert(
            "schema:ScholarlyArticle".into(),
            SchemaType {
                id: "schema:ScholarlyArticle".into(),
                label: "ScholarlyArticle".into(),
                description: Some("A scholarly article".into()),
                parents: vec!["schema:Article".into()],
                properties: vec![],
            },
        );

        types.insert(
            "schema:Article".into(),
            SchemaType {
                id: "schema:Article".into(),
                label: "Article".into(),
                description: Some("An article, such as a news or blog post".into()),
                parents: vec!["schema:CreativeWork".into()],
                properties: vec!["schema:articleBody".into()],
            },
        );

        // Organizations and people
        types.insert(
            "schema:Organization".into(),
            SchemaType {
                id: "schema:Organization".into(),
                label: "Organization".into(),
                description: Some("An organization such as a company or NGO".into()),
                parents: vec!["schema:Thing".into()],
                properties: vec!["schema:member".into(), "schema:department".into()],
            },
        );

        types.insert(
            "schema:Person".into(),
            SchemaType {
                id: "schema:Person".into(),
                label: "Person".into(),
                description: Some("A person".into()),
                parents: vec!["schema:Thing".into()],
                properties: vec![
                    "schema:givenName".into(),
                    "schema:familyName".into(),
                    "schema:email".into(),
                ],
            },
        );

        // Events
        types.insert(
            "schema:Event".into(),
            SchemaType {
                id: "schema:Event".into(),
                label: "Event".into(),
                description: Some("An event happening at a certain time and location".into()),
                parents: vec!["schema:Thing".into()],
                properties: vec![
                    "schema:startDate".into(),
                    "schema:endDate".into(),
                    "schema:location".into(),
                ],
            },
        );

        // Places
        types.insert(
            "schema:Place".into(),
            SchemaType {
                id: "schema:Place".into(),
                label: "Place".into(),
                description: Some("Entities with a physical location".into()),
                parents: vec!["schema:Thing".into()],
                properties: vec!["schema:geo".into(), "schema:address".into()],
            },
        );

        // Intangibles
        types.insert(
            "schema:Intangible".into(),
            SchemaType {
                id: "schema:Intangible".into(),
                label: "Intangible".into(),
                description: Some("A utility class for things that are not physical".into()),
                parents: vec!["schema:Thing".into()],
                properties: vec![],
            },
        );

        types.insert(
            "schema:DefinedTerm".into(),
            SchemaType {
                id: "schema:DefinedTerm".into(),
                label: "DefinedTerm".into(),
                description: Some("A word, name, or phrase defined in a DefinedTermSet".into()),
                parents: vec!["schema:Intangible".into()],
                properties: vec!["schema:termCode".into()],
            },
        );

        types.insert(
            "schema:PropertyValue".into(),
            SchemaType {
                id: "schema:PropertyValue".into(),
                label: "PropertyValue".into(),
                description: Some("A property-value pair".into()),
                parents: vec!["schema:Intangible".into()],
                properties: vec![
                    "schema:value".into(),
                    "schema:unitCode".into(),
                    "schema:unitText".into(),
                ],
            },
        );

        types.insert(
            "schema:QuantitativeValue".into(),
            SchemaType {
                id: "schema:QuantitativeValue".into(),
                label: "QuantitativeValue".into(),
                description: Some("A point value or interval for product characteristics".into()),
                parents: vec!["schema:Intangible".into()],
                properties: vec![
                    "schema:value".into(),
                    "schema:minValue".into(),
                    "schema:maxValue".into(),
                    "schema:unitCode".into(),
                ],
            },
        );

        // Medical/Health (relevant for FHIR interop)
        types.insert(
            "schema:MedicalEntity".into(),
            SchemaType {
                id: "schema:MedicalEntity".into(),
                label: "MedicalEntity".into(),
                description: Some("The most generic type of medical entity".into()),
                parents: vec!["schema:Thing".into()],
                properties: vec!["schema:code".into()],
            },
        );

        types.insert(
            "schema:Drug".into(),
            SchemaType {
                id: "schema:Drug".into(),
                label: "Drug".into(),
                description: Some("A chemical or biologic substance used to treat disease".into()),
                parents: vec!["schema:MedicalEntity".into()],
                properties: vec!["schema:activeIngredient".into(), "schema:dosageForm".into()],
            },
        );

        types.insert(
            "schema:MedicalCondition".into(),
            SchemaType {
                id: "schema:MedicalCondition".into(),
                label: "MedicalCondition".into(),
                description: Some("Any condition of the human body".into()),
                parents: vec!["schema:MedicalEntity".into()],
                properties: vec![],
            },
        );

        // Common properties
        properties.insert(
            "schema:name".into(),
            SchemaProperty {
                id: "schema:name".into(),
                label: "name".into(),
                description: Some("The name of the item".into()),
                domain: vec!["schema:Thing".into()],
                range: vec!["schema:Text".into()],
            },
        );

        properties.insert(
            "schema:description".into(),
            SchemaProperty {
                id: "schema:description".into(),
                label: "description".into(),
                description: Some("A description of the item".into()),
                domain: vec!["schema:Thing".into()],
                range: vec!["schema:Text".into()],
            },
        );

        properties.insert(
            "schema:identifier".into(),
            SchemaProperty {
                id: "schema:identifier".into(),
                label: "identifier".into(),
                description: Some("The identifier property".into()),
                domain: vec!["schema:Thing".into()],
                range: vec![
                    "schema:Text".into(),
                    "schema:URL".into(),
                    "schema:PropertyValue".into(),
                ],
            },
        );

        properties.insert(
            "schema:value".into(),
            SchemaProperty {
                id: "schema:value".into(),
                label: "value".into(),
                description: Some("The value of the quantitative value".into()),
                domain: vec!["schema:PropertyValue".into()],
                range: vec!["schema:Number".into(), "schema:Text".into()],
            },
        );

        properties.insert(
            "schema:unitCode".into(),
            SchemaProperty {
                id: "schema:unitCode".into(),
                label: "unitCode".into(),
                description: Some("The unit of measurement (UN/CEFACT)".into()),
                domain: vec!["schema:PropertyValue".into()],
                range: vec!["schema:Text".into(), "schema:URL".into()],
            },
        );

        Self {
            types,
            properties,
            mappings: HashMap::new(),
        }
    }

    pub fn get_type(&self, id: &str) -> Option<&SchemaType> {
        self.types.get(id)
    }

    pub fn get_property(&self, id: &str) -> Option<&SchemaProperty> {
        self.properties.get(id)
    }

    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    pub fn property_count(&self) -> usize {
        self.properties.len()
    }
}

impl OntologySource for SchemaOrgOntology {
    fn terms(&self) -> Vec<TermEntry> {
        self.types
            .values()
            .map(|t| TermEntry {
                id: TermId {
                    id: t.id.clone(),
                    label: Some(t.label.clone()),
                },
                ontology: "Schema.org".into(),
                definition: t.description.clone(),
                parents: t.parents.clone(),
            })
            .collect()
    }

    fn curation_status(&self) -> CurationStatus {
        CurationStatus::CommunityCurated
    }

    fn provenance(&self) -> &str {
        "https://schema.org/"
    }

    fn get_mappings(&self, term: &TermId) -> Option<Vec<TermMapping>> {
        self.mappings.get(&term.id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap() {
        let schema = SchemaOrgOntology::bootstrap();
        assert!(schema.type_count() > 10);
        assert!(schema.property_count() > 3);
    }

    #[test]
    fn test_thing_exists() {
        let schema = SchemaOrgOntology::bootstrap();
        let thing = schema.get_type("schema:Thing");
        assert!(thing.is_some());
        assert_eq!(thing.unwrap().label, "Thing");
    }

    #[test]
    fn test_dataset_hierarchy() {
        let schema = SchemaOrgOntology::bootstrap();
        let dataset = schema.get_type("schema:Dataset").unwrap();
        assert!(dataset.parents.contains(&"schema:CreativeWork".to_string()));

        let creative_work = schema.get_type("schema:CreativeWork").unwrap();
        assert!(creative_work.parents.contains(&"schema:Thing".to_string()));
    }

    #[test]
    fn test_terms_vec() {
        let schema = SchemaOrgOntology::bootstrap();
        let terms = schema.terms();
        assert!(!terms.is_empty());
    }
}
