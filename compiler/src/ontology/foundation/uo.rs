//! UO - Units of Measurement Ontology
//!
//! ~1,000 terms describing units of measurement.
//! Critical for dimensional analysis and scientific computing.

use std::collections::HashMap;
use std::path::Path;

use crate::epistemic::TermId;

use super::{CurationStatus, OntologySource, TermEntry, TermMapping};
use crate::ontology::OntologyError;

pub struct UOOntology {
    units: HashMap<String, UOUnit>,
    mappings: HashMap<String, Vec<TermMapping>>,
}

#[derive(Debug, Clone)]
pub struct UOUnit {
    pub id: String,
    pub label: String,
    pub definition: Option<String>,
    pub parents: Vec<String>,
    pub synonyms: Vec<String>,
    /// SI base units this derives from
    pub base_units: Option<BaseUnits>,
}

/// SI base unit dimensions
#[derive(Debug, Clone, Default)]
pub struct BaseUnits {
    pub meter: i8,    // length
    pub kilogram: i8, // mass
    pub second: i8,   // time
    pub ampere: i8,   // electric current
    pub kelvin: i8,   // temperature
    pub mole: i8,     // amount of substance
    pub candela: i8,  // luminous intensity
}

impl UOOntology {
    pub fn load(_path: &Path) -> Result<Self, OntologyError> {
        Ok(Self::bootstrap())
    }

    /// Create with bootstrap data (essential UO units)
    pub fn bootstrap() -> Self {
        let mut units = HashMap::new();

        // Root unit
        units.insert(
            "UO:0000000".into(),
            UOUnit {
                id: "UO:0000000".into(),
                label: "unit".into(),
                definition: Some("A unit of measurement".into()),
                parents: vec![],
                synonyms: vec![],
                base_units: None,
            },
        );

        // Base SI units
        units.insert(
            "UO:0000001".into(),
            UOUnit {
                id: "UO:0000001".into(),
                label: "length unit".into(),
                definition: Some("A unit which is a standard measure of distance".into()),
                parents: vec!["UO:0000000".into()],
                synonyms: vec![],
                base_units: Some(BaseUnits {
                    meter: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000002".into(),
            UOUnit {
                id: "UO:0000002".into(),
                label: "mass unit".into(),
                definition: Some("A unit which is a standard measure of mass".into()),
                parents: vec!["UO:0000000".into()],
                synonyms: vec![],
                base_units: Some(BaseUnits {
                    kilogram: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000003".into(),
            UOUnit {
                id: "UO:0000003".into(),
                label: "time unit".into(),
                definition: Some("A unit which is a standard measure of time".into()),
                parents: vec!["UO:0000000".into()],
                synonyms: vec![],
                base_units: Some(BaseUnits {
                    second: 1,
                    ..Default::default()
                }),
            },
        );

        // Specific length units
        units.insert(
            "UO:0000008".into(),
            UOUnit {
                id: "UO:0000008".into(),
                label: "meter".into(),
                definition: Some("The SI base unit of length".into()),
                parents: vec!["UO:0000001".into()],
                synonyms: vec!["m".into(), "metre".into()],
                base_units: Some(BaseUnits {
                    meter: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000015".into(),
            UOUnit {
                id: "UO:0000015".into(),
                label: "centimeter".into(),
                definition: Some("A length unit equal to one hundredth of a meter".into()),
                parents: vec!["UO:0000001".into()],
                synonyms: vec!["cm".into()],
                base_units: Some(BaseUnits {
                    meter: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000016".into(),
            UOUnit {
                id: "UO:0000016".into(),
                label: "millimeter".into(),
                definition: Some("A length unit equal to one thousandth of a meter".into()),
                parents: vec!["UO:0000001".into()],
                synonyms: vec!["mm".into()],
                base_units: Some(BaseUnits {
                    meter: 1,
                    ..Default::default()
                }),
            },
        );

        // Mass units
        units.insert(
            "UO:0000009".into(),
            UOUnit {
                id: "UO:0000009".into(),
                label: "kilogram".into(),
                definition: Some("The SI base unit of mass".into()),
                parents: vec!["UO:0000002".into()],
                synonyms: vec!["kg".into()],
                base_units: Some(BaseUnits {
                    kilogram: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000021".into(),
            UOUnit {
                id: "UO:0000021".into(),
                label: "gram".into(),
                definition: Some("A mass unit equal to one thousandth of a kilogram".into()),
                parents: vec!["UO:0000002".into()],
                synonyms: vec!["g".into()],
                base_units: Some(BaseUnits {
                    kilogram: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000022".into(),
            UOUnit {
                id: "UO:0000022".into(),
                label: "milligram".into(),
                definition: Some("A mass unit equal to one thousandth of a gram".into()),
                parents: vec!["UO:0000002".into()],
                synonyms: vec!["mg".into()],
                base_units: Some(BaseUnits {
                    kilogram: 1,
                    ..Default::default()
                }),
            },
        );

        // Time units
        units.insert(
            "UO:0000010".into(),
            UOUnit {
                id: "UO:0000010".into(),
                label: "second".into(),
                definition: Some("The SI base unit of time".into()),
                parents: vec!["UO:0000003".into()],
                synonyms: vec!["s".into(), "sec".into()],
                base_units: Some(BaseUnits {
                    second: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000031".into(),
            UOUnit {
                id: "UO:0000031".into(),
                label: "minute".into(),
                definition: Some("A time unit equal to 60 seconds".into()),
                parents: vec!["UO:0000003".into()],
                synonyms: vec!["min".into()],
                base_units: Some(BaseUnits {
                    second: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000032".into(),
            UOUnit {
                id: "UO:0000032".into(),
                label: "hour".into(),
                definition: Some("A time unit equal to 3600 seconds".into()),
                parents: vec!["UO:0000003".into()],
                synonyms: vec!["h".into(), "hr".into()],
                base_units: Some(BaseUnits {
                    second: 1,
                    ..Default::default()
                }),
            },
        );

        // Derived units
        units.insert(
            "UO:0000095".into(),
            UOUnit {
                id: "UO:0000095".into(),
                label: "volume unit".into(),
                definition: Some("A unit of 3-dimensional extent".into()),
                parents: vec!["UO:0000000".into()],
                synonyms: vec![],
                base_units: Some(BaseUnits {
                    meter: 3,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000099".into(),
            UOUnit {
                id: "UO:0000099".into(),
                label: "liter".into(),
                definition: Some("A volume unit equal to one cubic decimeter".into()),
                parents: vec!["UO:0000095".into()],
                synonyms: vec!["L".into(), "litre".into()],
                base_units: Some(BaseUnits {
                    meter: 3,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000098".into(),
            UOUnit {
                id: "UO:0000098".into(),
                label: "milliliter".into(),
                definition: Some("A volume unit equal to one thousandth of a liter".into()),
                parents: vec!["UO:0000095".into()],
                synonyms: vec!["mL".into()],
                base_units: Some(BaseUnits {
                    meter: 3,
                    ..Default::default()
                }),
            },
        );

        // Concentration units
        units.insert(
            "UO:0000051".into(),
            UOUnit {
                id: "UO:0000051".into(),
                label: "concentration unit".into(),
                definition: Some("A unit for amount of substance per unit volume".into()),
                parents: vec!["UO:0000000".into()],
                synonyms: vec![],
                base_units: Some(BaseUnits {
                    mole: 1,
                    meter: -3,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000062".into(),
            UOUnit {
                id: "UO:0000062".into(),
                label: "molar".into(),
                definition: Some("A concentration unit equal to moles per liter".into()),
                parents: vec!["UO:0000051".into()],
                synonyms: vec!["M".into(), "mol/L".into()],
                base_units: Some(BaseUnits {
                    mole: 1,
                    meter: -3,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000063".into(),
            UOUnit {
                id: "UO:0000063".into(),
                label: "millimolar".into(),
                definition: Some("A concentration unit equal to millimoles per liter".into()),
                parents: vec!["UO:0000051".into()],
                synonyms: vec!["mM".into()],
                base_units: Some(BaseUnits {
                    mole: 1,
                    meter: -3,
                    ..Default::default()
                }),
            },
        );

        // Temperature units
        units.insert(
            "UO:0000004".into(),
            UOUnit {
                id: "UO:0000004".into(),
                label: "temperature unit".into(),
                definition: Some("A unit of temperature".into()),
                parents: vec!["UO:0000000".into()],
                synonyms: vec![],
                base_units: Some(BaseUnits {
                    kelvin: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000012".into(),
            UOUnit {
                id: "UO:0000012".into(),
                label: "kelvin".into(),
                definition: Some("The SI base unit of temperature".into()),
                parents: vec!["UO:0000004".into()],
                synonyms: vec!["K".into()],
                base_units: Some(BaseUnits {
                    kelvin: 1,
                    ..Default::default()
                }),
            },
        );

        units.insert(
            "UO:0000027".into(),
            UOUnit {
                id: "UO:0000027".into(),
                label: "degree Celsius".into(),
                definition: Some("A temperature unit equal to kelvin minus 273.15".into()),
                parents: vec!["UO:0000004".into()],
                synonyms: vec!["°C".into(), "C".into()],
                base_units: Some(BaseUnits {
                    kelvin: 1,
                    ..Default::default()
                }),
            },
        );

        Self {
            units,
            mappings: HashMap::new(),
        }
    }

    pub fn get_unit(&self, id: &str) -> Option<&UOUnit> {
        self.units.get(id)
    }

    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Check if two units are dimensionally compatible
    pub fn dimensionally_compatible(&self, a: &str, b: &str) -> bool {
        let unit_a = self.get_unit(a);
        let unit_b = self.get_unit(b);

        match (unit_a, unit_b) {
            (Some(ua), Some(ub)) => match (&ua.base_units, &ub.base_units) {
                (Some(ba), Some(bb)) => {
                    ba.meter == bb.meter
                        && ba.kilogram == bb.kilogram
                        && ba.second == bb.second
                        && ba.ampere == bb.ampere
                        && ba.kelvin == bb.kelvin
                        && ba.mole == bb.mole
                        && ba.candela == bb.candela
                }
                _ => false,
            },
            _ => false,
        }
    }
}

impl OntologySource for UOOntology {
    fn terms(&self) -> Vec<TermEntry> {
        self.units
            .values()
            .map(|u| TermEntry {
                id: TermId {
                    id: u.id.clone(),
                    label: Some(u.label.clone()),
                },
                ontology: "UO".into(),
                definition: u.definition.clone(),
                parents: u.parents.clone(),
            })
            .collect()
    }

    fn curation_status(&self) -> CurationStatus {
        CurationStatus::ExpertCurated
    }

    fn provenance(&self) -> &str {
        "http://purl.obolibrary.org/obo/uo.owl"
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
        let uo = UOOntology::bootstrap();
        assert!(uo.unit_count() > 15);
    }

    #[test]
    fn test_meter_exists() {
        let uo = UOOntology::bootstrap();
        let meter = uo.get_unit("UO:0000008");
        assert!(meter.is_some());
        assert_eq!(meter.unwrap().label, "meter");
    }

    #[test]
    fn test_dimensional_compatibility() {
        let uo = UOOntology::bootstrap();

        // Same dimension
        assert!(uo.dimensionally_compatible("UO:0000008", "UO:0000015")); // meter, centimeter
        assert!(uo.dimensionally_compatible("UO:0000021", "UO:0000022")); // gram, milligram

        // Different dimensions
        assert!(!uo.dimensionally_compatible("UO:0000008", "UO:0000021")); // meter, gram
    }

    #[test]
    fn test_terms_vec() {
        let uo = UOOntology::bootstrap();
        let terms = uo.terms();
        assert!(!terms.is_empty());
    }
}
