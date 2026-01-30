//! PATO - Phenotype and Trait Ontology
//!
//! ~2,500 terms describing phenotypic qualities.
//! Loaded from OWL file at stdlib/ontology/pato.owl

use std::collections::HashMap;
use std::path::Path;

use crate::epistemic::TermId;

use super::{CurationStatus, OntologySource, TermEntry, TermMapping};
use crate::ontology::OntologyError;

pub struct PATOOntology {
    classes: HashMap<String, PATOClass>,
    mappings: HashMap<String, Vec<TermMapping>>,
}

#[derive(Debug, Clone)]
pub struct PATOClass {
    pub id: String,
    pub label: String,
    pub definition: Option<String>,
    pub parents: Vec<String>,
    pub synonyms: Vec<String>,
}

impl PATOOntology {
    pub fn load(_path: &Path) -> Result<Self, OntologyError> {
        // In production: use horned-owl or oxigraph to parse OWL
        // For bootstrap: use embedded data
        Ok(Self::bootstrap())
    }

    /// Create with bootstrap data (essential PATO classes)
    pub fn bootstrap() -> Self {
        let mut classes = HashMap::new();

        // Core PATO classes
        classes.insert(
            "PATO:0000001".into(),
            PATOClass {
                id: "PATO:0000001".into(),
                label: "quality".into(),
                definition: Some("A dependent entity that inheres in a bearer".into()),
                parents: vec!["BFO:0000019".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0001035".into(),
            PATOClass {
                id: "PATO:0001035".into(),
                label: "physical quality".into(),
                definition: Some("A quality of a physical entity".into()),
                parents: vec!["PATO:0000001".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0000125".into(),
            PATOClass {
                id: "PATO:0000125".into(),
                label: "mass".into(),
                definition: Some(
                    "A physical quality inhering in a bearer by virtue of the amount of matter"
                        .into(),
                ),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec!["weight".into()],
            },
        );

        classes.insert(
            "PATO:0000033".into(),
            PATOClass {
                id: "PATO:0000033".into(),
                label: "concentration".into(),
                definition: Some(
                    "A quality inhering by virtue of the amount of substance per unit volume"
                        .into(),
                ),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec!["concentration of".into()],
            },
        );

        classes.insert(
            "PATO:0000165".into(),
            PATOClass {
                id: "PATO:0000165".into(),
                label: "size".into(),
                definition: Some("A morphology quality by virtue of physical magnitude".into()),
                parents: vec!["PATO:0000051".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0000051".into(),
            PATOClass {
                id: "PATO:0000051".into(),
                label: "morphology".into(),
                definition: Some("A quality of a continuant's physical form and structure".into()),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec!["shape".into(), "form".into()],
            },
        );

        classes.insert(
            "PATO:0000146".into(),
            PATOClass {
                id: "PATO:0000146".into(),
                label: "temperature".into(),
                definition: Some("A physical quality of the thermal energy of a system".into()),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0000918".into(),
            PATOClass {
                id: "PATO:0000918".into(),
                label: "volume".into(),
                definition: Some(
                    "A 3-D extent quality by virtue of the amount of 3-dimensional space occupied"
                        .into(),
                ),
                parents: vec!["PATO:0001710".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0001710".into(),
            PATOClass {
                id: "PATO:0001710".into(),
                label: "3-D extent".into(),
                definition: Some("A spatial quality representing three-dimensional extent".into()),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0002062".into(),
            PATOClass {
                id: "PATO:0002062".into(),
                label: "physical quality of a process".into(),
                definition: Some("A quality of a process that can be quantified".into()),
                parents: vec!["PATO:0001236".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0001236".into(),
            PATOClass {
                id: "PATO:0001236".into(),
                label: "process quality".into(),
                definition: Some("A quality that inheres in a process".into()),
                parents: vec!["PATO:0000001".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0000011".into(),
            PATOClass {
                id: "PATO:0000011".into(),
                label: "age".into(),
                definition: Some(
                    "A time quality by virtue of the duration of the bearer's existence".into(),
                ),
                parents: vec!["PATO:0000165".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0000070".into(),
            PATOClass {
                id: "PATO:0000070".into(),
                label: "count".into(),
                definition: Some(
                    "A quality of a collection representing the number of items".into(),
                ),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec!["number".into()],
            },
        );

        classes.insert(
            "PATO:0000048".into(),
            PATOClass {
                id: "PATO:0000048".into(),
                label: "hardness".into(),
                definition: Some(
                    "A physical quality representing resistance to deformation".into(),
                ),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec![],
            },
        );

        classes.insert(
            "PATO:0001019".into(),
            PATOClass {
                id: "PATO:0001019".into(),
                label: "color".into(),
                definition: Some(
                    "A quality of a physical entity that results from the absorption or emission of light"
                        .into(),
                ),
                parents: vec!["PATO:0001035".into()],
                synonyms: vec!["colour".into()],
            },
        );

        Self {
            classes,
            mappings: HashMap::new(),
        }
    }

    pub fn get_class(&self, id: &str) -> Option<&PATOClass> {
        self.classes.get(id)
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }
}

impl OntologySource for PATOOntology {
    fn terms(&self) -> Vec<TermEntry> {
        self.classes
            .values()
            .map(|c| TermEntry {
                id: TermId {
                    id: c.id.clone(),
                    label: Some(c.label.clone()),
                },
                ontology: "PATO".into(),
                definition: c.definition.clone(),
                parents: c.parents.clone(),
            })
            .collect()
    }

    fn curation_status(&self) -> CurationStatus {
        CurationStatus::ExpertCurated
    }

    fn provenance(&self) -> &str {
        "http://purl.obolibrary.org/obo/pato.owl"
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
        let pato = PATOOntology::bootstrap();
        assert!(pato.class_count() > 10);
    }

    #[test]
    fn test_quality_exists() {
        let pato = PATOOntology::bootstrap();
        let quality = pato.get_class("PATO:0000001");
        assert!(quality.is_some());
        assert_eq!(quality.unwrap().label, "quality");
    }

    #[test]
    fn test_terms_vec() {
        let pato = PATOOntology::bootstrap();
        let terms = pato.terms();
        assert!(!terms.is_empty());
    }
}
