//! IAO - Information Artifact Ontology
//!
//! ~300 terms describing information entities.
//! Essential for modeling data, documents, and computational artifacts.

use std::collections::HashMap;
use std::path::Path;

use crate::epistemic::TermId;

use super::{CurationStatus, OntologySource, TermEntry, TermMapping};
use crate::ontology::OntologyError;

pub struct IAOOntology {
    classes: HashMap<String, IAOClass>,
    mappings: HashMap<String, Vec<TermMapping>>,
}

#[derive(Debug, Clone)]
pub struct IAOClass {
    pub id: String,
    pub label: String,
    pub definition: Option<String>,
    pub parents: Vec<String>,
    pub synonyms: Vec<String>,
}

impl IAOOntology {
    pub fn load(_path: &Path) -> Result<Self, OntologyError> {
        Ok(Self::bootstrap())
    }

    /// Create with bootstrap data (essential IAO classes)
    pub fn bootstrap() -> Self {
        let mut classes = HashMap::new();

        // Root: Information content entity
        classes.insert(
            "IAO:0000030".into(),
            IAOClass {
                id: "IAO:0000030".into(),
                label: "information content entity".into(),
                definition: Some(
                    "A generically dependent continuant that is about some thing".into(),
                ),
                parents: vec!["BFO:0000031".into()],
                synonyms: vec!["ICE".into()],
            },
        );

        // Data items
        classes.insert(
            "IAO:0000027".into(),
            IAOClass {
                id: "IAO:0000027".into(),
                label: "data item".into(),
                definition: Some(
                    "An information content entity that is intended to be a truthful statement about something"
                        .into(),
                ),
                parents: vec!["IAO:0000030".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "IAO:0000100".into(),
            IAOClass {
                id: "IAO:0000100".into(),
                label: "data set".into(),
                definition: Some("A data item that is an aggregate of other data items".into()),
                parents: vec!["IAO:0000027".into()],
                synonyms: vec!["dataset".into()],
            },
        );

        // Measurement data
        classes.insert(
            "IAO:0000109".into(),
            IAOClass {
                id: "IAO:0000109".into(),
                label: "measurement datum".into(),
                definition: Some(
                    "A data item that is a recording of the output of a measurement".into(),
                ),
                parents: vec!["IAO:0000027".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "IAO:0000032".into(),
            IAOClass {
                id: "IAO:0000032".into(),
                label: "scalar measurement datum".into(),
                definition: Some("A measurement datum with a single numeric value".into()),
                parents: vec!["IAO:0000109".into()],
                synonyms: vec![],
            },
        );

        // Identifiers
        classes.insert(
            "IAO:0020000".into(),
            IAOClass {
                id: "IAO:0020000".into(),
                label: "identifier".into(),
                definition: Some(
                    "An information content entity used to uniquely identify something".into(),
                ),
                parents: vec!["IAO:0000030".into()],
                synonyms: vec!["ID".into()],
            },
        );

        classes.insert(
            "IAO:0000578".into(),
            IAOClass {
                id: "IAO:0000578".into(),
                label: "centrally registered identifier".into(),
                definition: Some("An identifier registered in a central authority".into()),
                parents: vec!["IAO:0020000".into()],
                synonyms: vec!["CRID".into()],
            },
        );

        // Documents
        classes.insert(
            "IAO:0000310".into(),
            IAOClass {
                id: "IAO:0000310".into(),
                label: "document".into(),
                definition: Some(
                    "A collection of information content entities intended to be understood together"
                        .into(),
                ),
                parents: vec!["IAO:0000030".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "IAO:0000013".into(),
            IAOClass {
                id: "IAO:0000013".into(),
                label: "journal article".into(),
                definition: Some("A document published in a peer-reviewed journal".into()),
                parents: vec!["IAO:0000310".into()],
                synonyms: vec!["paper".into(), "publication".into()],
            },
        );

        classes.insert(
            "IAO:0000311".into(),
            IAOClass {
                id: "IAO:0000311".into(),
                label: "publication".into(),
                definition: Some("A document that has been made available to the public".into()),
                parents: vec!["IAO:0000310".into()],
                synonyms: vec![],
            },
        );

        // Software and algorithms
        classes.insert(
            "IAO:0000064".into(),
            IAOClass {
                id: "IAO:0000064".into(),
                label: "algorithm".into(),
                definition: Some(
                    "A directive information entity specifying a procedure for computation".into(),
                ),
                parents: vec!["IAO:0000104".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "IAO:0000010".into(),
            IAOClass {
                id: "IAO:0000010".into(),
                label: "software".into(),
                definition: Some(
                    "A directive information entity that can be executed by a computer".into(),
                ),
                parents: vec!["IAO:0000104".into()],
                synonyms: vec!["program".into()],
            },
        );

        classes.insert(
            "IAO:0000104".into(),
            IAOClass {
                id: "IAO:0000104".into(),
                label: "plan specification".into(),
                definition: Some(
                    "A directive information entity with parts for an objective and actions".into(),
                ),
                parents: vec!["IAO:0000033".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "IAO:0000033".into(),
            IAOClass {
                id: "IAO:0000033".into(),
                label: "directive information entity".into(),
                definition: Some(
                    "An information content entity that describes an action to be performed".into(),
                ),
                parents: vec!["IAO:0000030".into()],
                synonyms: vec![],
            },
        );

        // Ontology terms
        classes.insert(
            "IAO:0000078".into(),
            IAOClass {
                id: "IAO:0000078".into(),
                label: "curation status specification".into(),
                definition: Some("A specification of the curation status of a term".into()),
                parents: vec!["IAO:0000102".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "IAO:0000102".into(),
            IAOClass {
                id: "IAO:0000102".into(),
                label: "data about an ontology part".into(),
                definition: Some("Data that is about a part of an ontology".into()),
                parents: vec!["IAO:0000027".into()],
                synonyms: vec![],
            },
        );

        // Definitions and descriptions
        classes.insert(
            "IAO:0000115".into(),
            IAOClass {
                id: "IAO:0000115".into(),
                label: "definition".into(),
                definition: Some(
                    "A textual definition that states necessary and sufficient conditions".into(),
                ),
                parents: vec!["IAO:0000030".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "IAO:0000600".into(),
            IAOClass {
                id: "IAO:0000600".into(),
                label: "elucidation".into(),
                definition: Some(
                    "A statement that clarifies the meaning of a primitive term".into(),
                ),
                parents: vec!["IAO:0000030".into()],
                synonyms: vec![],
            },
        );

        // Symbol and representation
        classes.insert(
            "IAO:0000028".into(),
            IAOClass {
                id: "IAO:0000028".into(),
                label: "symbol".into(),
                definition: Some(
                    "An information content entity that stands for something else".into(),
                ),
                parents: vec!["IAO:0000030".into()],
                synonyms: vec![],
            },
        );

        Self {
            classes,
            mappings: HashMap::new(),
        }
    }

    pub fn get_class(&self, id: &str) -> Option<&IAOClass> {
        self.classes.get(id)
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }
}

impl OntologySource for IAOOntology {
    fn terms(&self) -> Vec<TermEntry> {
        self.classes
            .values()
            .map(|c| TermEntry {
                id: TermId {
                    id: c.id.clone(),
                    label: Some(c.label.clone()),
                },
                ontology: "IAO".into(),
                definition: c.definition.clone(),
                parents: c.parents.clone(),
            })
            .collect()
    }

    fn curation_status(&self) -> CurationStatus {
        CurationStatus::ExpertCurated
    }

    fn provenance(&self) -> &str {
        "http://purl.obolibrary.org/obo/iao.owl"
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
        let iao = IAOOntology::bootstrap();
        assert!(iao.class_count() > 10);
    }

    #[test]
    fn test_ice_exists() {
        let iao = IAOOntology::bootstrap();
        let ice = iao.get_class("IAO:0000030");
        assert!(ice.is_some());
        assert_eq!(ice.unwrap().label, "information content entity");
    }

    #[test]
    fn test_data_item_hierarchy() {
        let iao = IAOOntology::bootstrap();
        let data_item = iao.get_class("IAO:0000027").unwrap();
        assert!(data_item.parents.contains(&"IAO:0000030".to_string()));

        let dataset = iao.get_class("IAO:0000100").unwrap();
        assert!(dataset.parents.contains(&"IAO:0000027".to_string()));
    }

    #[test]
    fn test_terms_vec() {
        let iao = IAOOntology::bootstrap();
        let terms = iao.terms();
        assert!(!terms.is_empty());
    }
}
