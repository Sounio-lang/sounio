//! FHIR R5 - Fast Healthcare Interoperability Resources
//!
//! ~1,150 resources for healthcare data exchange.
//! Essential for biomedical computing and clinical research.

use std::collections::HashMap;
use std::path::Path;

use crate::epistemic::TermId;

use super::{CurationStatus, OntologySource, TermEntry, TermMapping};
use crate::ontology::OntologyError;

pub struct FHIROntology {
    resources: HashMap<String, FHIRResource>,
    data_types: HashMap<String, FHIRDataType>,
    mappings: HashMap<String, Vec<TermMapping>>,
}

#[derive(Debug, Clone)]
pub struct FHIRResource {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub parents: Vec<String>,
    pub elements: Vec<FHIRElement>,
    /// Maturity level (0-5)
    pub maturity: u8,
}

#[derive(Debug, Clone)]
pub struct FHIRElement {
    pub path: String,
    pub type_code: String,
    pub cardinality: (u32, Option<u32>), // (min, max) where None = *
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FHIRDataType {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub kind: DataTypeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataTypeKind {
    Primitive,
    Complex,
    Resource,
    Logical,
}

impl FHIROntology {
    pub fn load(_path: &Path) -> Result<Self, OntologyError> {
        Ok(Self::bootstrap())
    }

    /// Create with bootstrap data (essential FHIR R5 resources)
    pub fn bootstrap() -> Self {
        let mut resources = HashMap::new();
        let mut data_types = HashMap::new();

        // Primitive data types
        Self::add_primitive_types(&mut data_types);

        // Complex data types
        Self::add_complex_types(&mut data_types);

        // Base resources
        Self::add_base_resources(&mut resources);

        // Clinical resources
        Self::add_clinical_resources(&mut resources);

        // Administrative resources
        Self::add_administrative_resources(&mut resources);

        Self {
            resources,
            data_types,
            mappings: HashMap::new(),
        }
    }

    fn add_primitive_types(types: &mut HashMap<String, FHIRDataType>) {
        let primitives = [
            ("boolean", "Value of true or false"),
            ("integer", "A signed integer"),
            ("string", "A sequence of Unicode characters"),
            ("decimal", "A rational number"),
            ("uri", "A Uniform Resource Identifier"),
            ("url", "A Uniform Resource Locator"),
            (
                "canonical",
                "A URI that refers to a resource by canonical URL",
            ),
            ("base64Binary", "A stream of bytes, base64 encoded"),
            (
                "instant",
                "An instant in time (YYYY-MM-DDThh:mm:ss.sss+zz:zz)",
            ),
            ("date", "A date (YYYY, YYYY-MM, or YYYY-MM-DD)"),
            ("dateTime", "A date, date-time or partial date"),
            ("time", "A time during the day (hh:mm:ss)"),
            ("code", "A string with a restricted set of values"),
            ("oid", "An OID represented as a URI"),
            ("id", "A string constrained to [A-Za-z0-9\\-\\.]{1,64}"),
            ("markdown", "A string that may contain markdown"),
            ("unsignedInt", "An integer >= 0"),
            ("positiveInt", "An integer > 0"),
            ("uuid", "A UUID represented as a URI"),
        ];

        for (name, desc) in primitives {
            types.insert(
                format!("fhir:{}", name),
                FHIRDataType {
                    id: format!("fhir:{}", name),
                    name: name.into(),
                    description: Some(desc.into()),
                    kind: DataTypeKind::Primitive,
                },
            );
        }
    }

    fn add_complex_types(types: &mut HashMap<String, FHIRDataType>) {
        let complex = [
            ("Quantity", "A measured amount with a unit"),
            ("Money", "An amount of economic utility in some currency"),
            ("Range", "Set of values bounded by low and high"),
            ("Ratio", "A ratio of two Quantity values"),
            ("Period", "Time range defined by start and end date/time"),
            (
                "Coding",
                "A reference to a code defined by a terminology system",
            ),
            (
                "CodeableConcept",
                "Concept - reference to a terminology or text",
            ),
            ("Identifier", "An identifier for this resource"),
            ("HumanName", "Name of a human"),
            ("Address", "An address expressed using postal conventions"),
            ("ContactPoint", "Details of a contact point"),
            ("Attachment", "Content in a format defined elsewhere"),
            ("Reference", "A reference from one resource to another"),
            ("Annotation", "Text with attribution"),
            ("Signature", "A digital signature"),
            ("SampledData", "Series of measurements"),
            ("Age", "A duration expressed as a Quantity (years)"),
            ("Duration", "A length of time"),
            ("Distance", "A length - a value with a unit of distance"),
            ("Count", "A measured count (integer) quantity"),
            ("Dosage", "How the medication is/was taken"),
        ];

        for (name, desc) in complex {
            types.insert(
                format!("fhir:{}", name),
                FHIRDataType {
                    id: format!("fhir:{}", name),
                    name: name.into(),
                    description: Some(desc.into()),
                    kind: DataTypeKind::Complex,
                },
            );
        }
    }

    fn add_base_resources(resources: &mut HashMap<String, FHIRResource>) {
        // Resource - the root
        resources.insert(
            "fhir:Resource".into(),
            FHIRResource {
                id: "fhir:Resource".into(),
                name: "Resource".into(),
                description: Some("Base Resource".into()),
                parents: vec![],
                elements: vec![
                    FHIRElement {
                        path: "Resource.id".into(),
                        type_code: "id".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Logical id of this artifact".into()),
                    },
                    FHIRElement {
                        path: "Resource.meta".into(),
                        type_code: "Meta".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Metadata about the resource".into()),
                    },
                ],
                maturity: 5,
            },
        );

        // DomainResource
        resources.insert(
            "fhir:DomainResource".into(),
            FHIRResource {
                id: "fhir:DomainResource".into(),
                name: "DomainResource".into(),
                description: Some(
                    "A resource with narrative, extensions, and contained resources".into(),
                ),
                parents: vec!["fhir:Resource".into()],
                elements: vec![
                    FHIRElement {
                        path: "DomainResource.text".into(),
                        type_code: "Narrative".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Text summary of the resource".into()),
                    },
                    FHIRElement {
                        path: "DomainResource.contained".into(),
                        type_code: "Resource".into(),
                        cardinality: (0, None),
                        description: Some("Contained, inline resources".into()),
                    },
                ],
                maturity: 5,
            },
        );

        // Bundle
        resources.insert(
            "fhir:Bundle".into(),
            FHIRResource {
                id: "fhir:Bundle".into(),
                name: "Bundle".into(),
                description: Some("Contains a collection of resources".into()),
                parents: vec!["fhir:Resource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Bundle.type".into(),
                        type_code: "code".into(),
                        cardinality: (1, Some(1)),
                        description: Some("document | message | transaction | etc.".into()),
                    },
                    FHIRElement {
                        path: "Bundle.entry".into(),
                        type_code: "BackboneElement".into(),
                        cardinality: (0, None),
                        description: Some("Entry in the bundle".into()),
                    },
                ],
                maturity: 5,
            },
        );
    }

    fn add_clinical_resources(resources: &mut HashMap<String, FHIRResource>) {
        // Patient
        resources.insert(
            "fhir:Patient".into(),
            FHIRResource {
                id: "fhir:Patient".into(),
                name: "Patient".into(),
                description: Some(
                    "Demographics and administrative information about a person".into(),
                ),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Patient.identifier".into(),
                        type_code: "Identifier".into(),
                        cardinality: (0, None),
                        description: Some("An identifier for this patient".into()),
                    },
                    FHIRElement {
                        path: "Patient.name".into(),
                        type_code: "HumanName".into(),
                        cardinality: (0, None),
                        description: Some("A name associated with the patient".into()),
                    },
                    FHIRElement {
                        path: "Patient.birthDate".into(),
                        type_code: "date".into(),
                        cardinality: (0, Some(1)),
                        description: Some("The date of birth for the individual".into()),
                    },
                    FHIRElement {
                        path: "Patient.gender".into(),
                        type_code: "code".into(),
                        cardinality: (0, Some(1)),
                        description: Some("male | female | other | unknown".into()),
                    },
                ],
                maturity: 5,
            },
        );

        // Observation
        resources.insert(
            "fhir:Observation".into(),
            FHIRResource {
                id: "fhir:Observation".into(),
                name: "Observation".into(),
                description: Some("Measurements and simple assertions made about a patient".into()),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Observation.status".into(),
                        type_code: "code".into(),
                        cardinality: (1, Some(1)),
                        description: Some("registered | preliminary | final | amended +".into()),
                    },
                    FHIRElement {
                        path: "Observation.code".into(),
                        type_code: "CodeableConcept".into(),
                        cardinality: (1, Some(1)),
                        description: Some("Type of observation (code / type)".into()),
                    },
                    FHIRElement {
                        path: "Observation.value[x]".into(),
                        type_code: "Quantity|CodeableConcept|string|boolean|integer|Range|Ratio"
                            .into(),
                        cardinality: (0, Some(1)),
                        description: Some("Actual result".into()),
                    },
                    FHIRElement {
                        path: "Observation.subject".into(),
                        type_code: "Reference(Patient|Group|Device|Location)".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Who and/or what this is about".into()),
                    },
                ],
                maturity: 5,
            },
        );

        // Condition
        resources.insert(
            "fhir:Condition".into(),
            FHIRResource {
                id: "fhir:Condition".into(),
                name: "Condition".into(),
                description: Some(
                    "A clinical condition, problem, diagnosis, or other event".into(),
                ),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Condition.clinicalStatus".into(),
                        type_code: "CodeableConcept".into(),
                        cardinality: (0, Some(1)),
                        description: Some(
                            "active | recurrence | relapse | inactive | remission | resolved"
                                .into(),
                        ),
                    },
                    FHIRElement {
                        path: "Condition.code".into(),
                        type_code: "CodeableConcept".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Identification of the condition".into()),
                    },
                    FHIRElement {
                        path: "Condition.subject".into(),
                        type_code: "Reference(Patient|Group)".into(),
                        cardinality: (1, Some(1)),
                        description: Some("Who has the condition?".into()),
                    },
                ],
                maturity: 4,
            },
        );

        // MedicationRequest
        resources.insert(
            "fhir:MedicationRequest".into(),
            FHIRResource {
                id: "fhir:MedicationRequest".into(),
                name: "MedicationRequest".into(),
                description: Some("Ordering of medication for patient or group".into()),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "MedicationRequest.status".into(),
                        type_code: "code".into(),
                        cardinality: (1, Some(1)),
                        description: Some("active | on-hold | cancelled | completed | etc.".into()),
                    },
                    FHIRElement {
                        path: "MedicationRequest.medication".into(),
                        type_code: "CodeableReference(Medication)".into(),
                        cardinality: (1, Some(1)),
                        description: Some("Medication to be taken".into()),
                    },
                    FHIRElement {
                        path: "MedicationRequest.subject".into(),
                        type_code: "Reference(Patient|Group)".into(),
                        cardinality: (1, Some(1)),
                        description: Some("Who the medication request is for".into()),
                    },
                    FHIRElement {
                        path: "MedicationRequest.dosageInstruction".into(),
                        type_code: "Dosage".into(),
                        cardinality: (0, None),
                        description: Some("How the medication should be taken".into()),
                    },
                ],
                maturity: 3,
            },
        );

        // DiagnosticReport
        resources.insert(
            "fhir:DiagnosticReport".into(),
            FHIRResource {
                id: "fhir:DiagnosticReport".into(),
                name: "DiagnosticReport".into(),
                description: Some("A Diagnostic report - combination of request information, results, and interpretation".into()),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "DiagnosticReport.status".into(),
                        type_code: "code".into(),
                        cardinality: (1, Some(1)),
                        description: Some("registered | partial | preliminary | modified | final | etc.".into()),
                    },
                    FHIRElement {
                        path: "DiagnosticReport.code".into(),
                        type_code: "CodeableConcept".into(),
                        cardinality: (1, Some(1)),
                        description: Some("Name/Code for this diagnostic report".into()),
                    },
                    FHIRElement {
                        path: "DiagnosticReport.result".into(),
                        type_code: "Reference(Observation)".into(),
                        cardinality: (0, None),
                        description: Some("Observations".into()),
                    },
                ],
                maturity: 4,
            },
        );

        // Specimen
        resources.insert(
            "fhir:Specimen".into(),
            FHIRResource {
                id: "fhir:Specimen".into(),
                name: "Specimen".into(),
                description: Some("Sample for analysis".into()),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Specimen.type".into(),
                        type_code: "CodeableConcept".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Kind of material that forms the specimen".into()),
                    },
                    FHIRElement {
                        path: "Specimen.subject".into(),
                        type_code: "Reference(Patient|Group|Device|BiologicallyDerivedProduct|Substance|Location)".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Where the specimen came from".into()),
                    },
                ],
                maturity: 3,
            },
        );

        // Substance
        resources.insert(
            "fhir:Substance".into(),
            FHIRResource {
                id: "fhir:Substance".into(),
                name: "Substance".into(),
                description: Some("A homogeneous material with a definite composition".into()),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Substance.code".into(),
                        type_code: "CodeableReference(SubstanceDefinition)".into(),
                        cardinality: (1, Some(1)),
                        description: Some("What substance this is".into()),
                    },
                    FHIRElement {
                        path: "Substance.quantity".into(),
                        type_code: "Quantity".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Amount of substance in the package".into()),
                    },
                ],
                maturity: 3,
            },
        );
    }

    fn add_administrative_resources(resources: &mut HashMap<String, FHIRResource>) {
        // Organization
        resources.insert(
            "fhir:Organization".into(),
            FHIRResource {
                id: "fhir:Organization".into(),
                name: "Organization".into(),
                description: Some(
                    "A formally or informally recognized grouping of people or organizations"
                        .into(),
                ),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Organization.identifier".into(),
                        type_code: "Identifier".into(),
                        cardinality: (0, None),
                        description: Some("Identifies this organization".into()),
                    },
                    FHIRElement {
                        path: "Organization.name".into(),
                        type_code: "string".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Name used for the organization".into()),
                    },
                    FHIRElement {
                        path: "Organization.type".into(),
                        type_code: "CodeableConcept".into(),
                        cardinality: (0, None),
                        description: Some("Kind of organization".into()),
                    },
                ],
                maturity: 5,
            },
        );

        // Practitioner
        resources.insert(
            "fhir:Practitioner".into(),
            FHIRResource {
                id: "fhir:Practitioner".into(),
                name: "Practitioner".into(),
                description: Some(
                    "A person with a formal responsibility in healthcare delivery".into(),
                ),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Practitioner.identifier".into(),
                        type_code: "Identifier".into(),
                        cardinality: (0, None),
                        description: Some("An identifier for the person".into()),
                    },
                    FHIRElement {
                        path: "Practitioner.name".into(),
                        type_code: "HumanName".into(),
                        cardinality: (0, None),
                        description: Some("The name(s) associated with the practitioner".into()),
                    },
                    FHIRElement {
                        path: "Practitioner.qualification".into(),
                        type_code: "BackboneElement".into(),
                        cardinality: (0, None),
                        description: Some(
                            "Qualifications obtained by training and certification".into(),
                        ),
                    },
                ],
                maturity: 4,
            },
        );

        // Location
        resources.insert(
            "fhir:Location".into(),
            FHIRResource {
                id: "fhir:Location".into(),
                name: "Location".into(),
                description: Some("Details and position information for a physical place".into()),
                parents: vec!["fhir:DomainResource".into()],
                elements: vec![
                    FHIRElement {
                        path: "Location.name".into(),
                        type_code: "string".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Name of the location".into()),
                    },
                    FHIRElement {
                        path: "Location.address".into(),
                        type_code: "Address".into(),
                        cardinality: (0, Some(1)),
                        description: Some("Physical location".into()),
                    },
                    FHIRElement {
                        path: "Location.position".into(),
                        type_code: "BackboneElement".into(),
                        cardinality: (0, Some(1)),
                        description: Some("The absolute geographic location".into()),
                    },
                ],
                maturity: 4,
            },
        );
    }

    pub fn get_resource(&self, id: &str) -> Option<&FHIRResource> {
        self.resources.get(id)
    }

    pub fn get_data_type(&self, id: &str) -> Option<&FHIRDataType> {
        self.data_types.get(id)
    }

    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    pub fn data_type_count(&self) -> usize {
        self.data_types.len()
    }

    /// Check if a resource inherits from another
    pub fn inherits_from(&self, child: &str, parent: &str) -> bool {
        if child == parent {
            return true;
        }

        if let Some(resource) = self.get_resource(child) {
            for p in &resource.parents {
                if p == parent || self.inherits_from(p, parent) {
                    return true;
                }
            }
        }

        false
    }
}

impl OntologySource for FHIROntology {
    fn terms(&self) -> Vec<TermEntry> {
        self.resources
            .values()
            .map(|r| TermEntry {
                id: TermId {
                    id: r.id.clone(),
                    label: Some(r.name.clone()),
                },
                ontology: "FHIR".into(),
                definition: r.description.clone(),
                parents: r.parents.clone(),
            })
            .collect()
    }

    fn curation_status(&self) -> CurationStatus {
        CurationStatus::ExpertCurated
    }

    fn provenance(&self) -> &str {
        "http://hl7.org/fhir/R5"
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
        let fhir = FHIROntology::bootstrap();
        assert!(fhir.resource_count() > 10);
        assert!(fhir.data_type_count() > 30);
    }

    #[test]
    fn test_patient_exists() {
        let fhir = FHIROntology::bootstrap();
        let patient = fhir.get_resource("fhir:Patient");
        assert!(patient.is_some());
        assert_eq!(patient.unwrap().name, "Patient");
    }

    #[test]
    fn test_patient_inherits_from_domain_resource() {
        let fhir = FHIROntology::bootstrap();
        assert!(fhir.inherits_from("fhir:Patient", "fhir:DomainResource"));
        assert!(fhir.inherits_from("fhir:Patient", "fhir:Resource"));
    }

    #[test]
    fn test_primitive_types() {
        let fhir = FHIROntology::bootstrap();
        let string_type = fhir.get_data_type("fhir:string");
        assert!(string_type.is_some());
        assert_eq!(string_type.unwrap().kind, DataTypeKind::Primitive);
    }

    #[test]
    fn test_complex_types() {
        let fhir = FHIROntology::bootstrap();
        let quantity = fhir.get_data_type("fhir:Quantity");
        assert!(quantity.is_some());
        assert_eq!(quantity.unwrap().kind, DataTypeKind::Complex);
    }

    #[test]
    fn test_terms_vec() {
        let fhir = FHIROntology::bootstrap();
        let terms = fhir.terms();
        assert!(!terms.is_empty());
    }
}
