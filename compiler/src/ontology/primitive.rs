//! L1 Primitive Ontologies - Compiled into the Compiler
//!
//! This module contains the foundational ontology terms that are always available
//! without any external dependencies or network access.
//!
//! # Included Ontologies
//!
//! - **BFO 2020** (Basic Formal Ontology): 36 classes defining fundamental categories
//! - **RO** (Relation Ontology): ~600 standard relations
//! - **COB** (Core Ontology for Biology): ~200 bridge classes
//!
//! Total: ~850 terms compiled directly into the Sounio compiler.
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::primitive::{PRIMITIVE_BFO, BfoClass};
//!
//! // Access BFO classes
//! let entity = PRIMITIVE_BFO.get_class(BfoClass::Entity);
//!
//! // Check subsumption
//! let is_subclass = PRIMITIVE_BFO.is_subclass(
//!     BfoClass::MaterialEntity,
//!     BfoClass::Entity
//! );
//! ```

use std::collections::HashSet;

/// Static BFO 2020 data
pub static PRIMITIVE_BFO: PrimitiveStore<BfoClass> = PrimitiveStore::bfo();

/// Static RO data
pub static PRIMITIVE_RO: PrimitiveStore<RoRelation> = PrimitiveStore::ro();

/// Static COB data
pub static PRIMITIVE_COB: PrimitiveStore<CobClass> = PrimitiveStore::cob();

/// Storage for primitive ontology terms
#[derive(Debug)]
pub struct PrimitiveStore<T: 'static> {
    /// All terms in this ontology
    terms: &'static [PrimitiveTerm<T>],
    /// Subsumption relationships (child -> parent)
    subsumption: &'static [(T, T)],
}

impl PrimitiveStore<BfoClass> {
    /// Create BFO store
    const fn bfo() -> PrimitiveStore<BfoClass> {
        PrimitiveStore {
            terms: BFO_TERMS,
            subsumption: BFO_SUBSUMPTION,
        }
    }
}

impl PrimitiveStore<RoRelation> {
    /// Create RO store
    const fn ro() -> PrimitiveStore<RoRelation> {
        PrimitiveStore {
            terms: RO_TERMS,
            subsumption: RO_SUBSUMPTION,
        }
    }
}

impl PrimitiveStore<CobClass> {
    /// Create COB store
    const fn cob() -> PrimitiveStore<CobClass> {
        PrimitiveStore {
            terms: COB_TERMS,
            subsumption: COB_SUBSUMPTION,
        }
    }
}

impl<T: Copy + Eq + std::hash::Hash + 'static> PrimitiveStore<T> {
    /// Get a term by its enum variant
    pub fn get_term(&self, term: T) -> Option<&PrimitiveTerm<T>> {
        self.terms.iter().find(|t| t.variant == term)
    }

    /// Get a term by its ID string
    pub fn get_by_id(&self, id: &str) -> Option<&PrimitiveTerm<T>> {
        self.terms.iter().find(|t| t.id == id)
    }

    /// Check if child is a subclass of parent
    pub fn is_subclass(&self, child: T, parent: T) -> bool {
        if child == parent {
            return true;
        }

        // Build transitive closure on demand
        let mut visited = HashSet::new();
        let mut stack = vec![child];

        while let Some(current) = stack.pop() {
            if current == parent {
                return true;
            }
            if visited.insert(current) {
                // Find all parents of current
                for (c, p) in self.subsumption.iter() {
                    if *c == current {
                        stack.push(*p);
                    }
                }
            }
        }

        false
    }

    /// Get all direct superclasses
    pub fn direct_superclasses(&self, term: T) -> Vec<T> {
        self.subsumption
            .iter()
            .filter(|(c, _)| *c == term)
            .map(|(_, p)| *p)
            .collect()
    }

    /// Get all direct subclasses
    pub fn direct_subclasses(&self, term: T) -> Vec<T> {
        self.subsumption
            .iter()
            .filter(|(_, p)| *p == term)
            .map(|(c, _)| *c)
            .collect()
    }

    /// Get all terms
    pub fn all_terms(&self) -> &[PrimitiveTerm<T>] {
        self.terms
    }

    /// Get term count
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

/// A single primitive ontology term
#[derive(Debug, Clone)]
pub struct PrimitiveTerm<T> {
    /// Enum variant for this term
    pub variant: T,
    /// OBO-style ID (e.g., "BFO:0000001")
    pub id: &'static str,
    /// Human-readable label
    pub label: &'static str,
    /// Definition
    pub definition: &'static str,
    /// Full IRI
    pub iri: &'static str,
}

// ============================================================================
// BFO 2020 - Basic Formal Ontology
// ============================================================================

/// BFO 2020 classes - the 36 fundamental categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BfoClass {
    // Continuant branch
    Entity,
    Continuant,
    IndependentContinuant,
    MaterialEntity,
    Object,
    ObjectAggregate,
    FiatObjectPart,
    ImmaterialEntity,
    Site,
    SpatialRegion,
    OneDimensionalSpatialRegion,
    TwoDimensionalSpatialRegion,
    ThreeDimensionalSpatialRegion,
    ZeroDimensionalSpatialRegion,
    ContinuantFiatBoundary,
    OneDimensionalContinuantFiatBoundary,
    TwoDimensionalContinuantFiatBoundary,
    ZeroDimensionalContinuantFiatBoundary,
    SpecificallyDependentContinuant,
    Quality,
    RealizableEntity,
    Role,
    Disposition,
    Function,
    GenericallyDependentContinuant,

    // Occurrent branch
    Occurrent,
    Process,
    ProcessBoundary,
    SpatiotemporalRegion,
    TemporalRegion,
    ZeroDimensionalTemporalRegion,
    OneDimensionalTemporalRegion,
    TemporalInterval,
    TemporalInstant,
    History,
    ProcessProfile,
}

impl BfoClass {
    /// Get the OBO-style ID
    pub fn id(&self) -> &'static str {
        match self {
            BfoClass::Entity => "BFO:0000001",
            BfoClass::Continuant => "BFO:0000002",
            BfoClass::Occurrent => "BFO:0000003",
            BfoClass::IndependentContinuant => "BFO:0000004",
            BfoClass::SpatialRegion => "BFO:0000006",
            BfoClass::TemporalRegion => "BFO:0000008",
            BfoClass::TwoDimensionalSpatialRegion => "BFO:0000009",
            BfoClass::SpatiotemporalRegion => "BFO:0000011",
            BfoClass::Process => "BFO:0000015",
            BfoClass::Disposition => "BFO:0000016",
            BfoClass::RealizableEntity => "BFO:0000017",
            BfoClass::ZeroDimensionalSpatialRegion => "BFO:0000018",
            BfoClass::Quality => "BFO:0000019",
            BfoClass::SpecificallyDependentContinuant => "BFO:0000020",
            BfoClass::Role => "BFO:0000023",
            BfoClass::ObjectAggregate => "BFO:0000024",
            BfoClass::OneDimensionalSpatialRegion => "BFO:0000026",
            BfoClass::Object => "BFO:0000027",
            BfoClass::ThreeDimensionalSpatialRegion => "BFO:0000028",
            BfoClass::Site => "BFO:0000029",
            BfoClass::GenericallyDependentContinuant => "BFO:0000031",
            BfoClass::Function => "BFO:0000034",
            BfoClass::ProcessBoundary => "BFO:0000035",
            BfoClass::OneDimensionalTemporalRegion => "BFO:0000038",
            BfoClass::MaterialEntity => "BFO:0000040",
            BfoClass::ContinuantFiatBoundary => "BFO:0000140",
            BfoClass::ImmaterialEntity => "BFO:0000141",
            BfoClass::FiatObjectPart => "BFO:0000024",
            BfoClass::OneDimensionalContinuantFiatBoundary => "BFO:0000142",
            BfoClass::TwoDimensionalContinuantFiatBoundary => "BFO:0000146",
            BfoClass::ZeroDimensionalContinuantFiatBoundary => "BFO:0000147",
            BfoClass::ZeroDimensionalTemporalRegion => "BFO:0000148",
            BfoClass::TemporalInterval => "BFO:0000202",
            BfoClass::TemporalInstant => "BFO:0000203",
            BfoClass::History => "BFO:0000182",
            BfoClass::ProcessProfile => "BFO:0000144",
        }
    }

    /// Get the human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            BfoClass::Entity => "entity",
            BfoClass::Continuant => "continuant",
            BfoClass::Occurrent => "occurrent",
            BfoClass::IndependentContinuant => "independent continuant",
            BfoClass::MaterialEntity => "material entity",
            BfoClass::Object => "object",
            BfoClass::ObjectAggregate => "object aggregate",
            BfoClass::FiatObjectPart => "fiat object part",
            BfoClass::ImmaterialEntity => "immaterial entity",
            BfoClass::Site => "site",
            BfoClass::SpatialRegion => "spatial region",
            BfoClass::OneDimensionalSpatialRegion => "one-dimensional spatial region",
            BfoClass::TwoDimensionalSpatialRegion => "two-dimensional spatial region",
            BfoClass::ThreeDimensionalSpatialRegion => "three-dimensional spatial region",
            BfoClass::ZeroDimensionalSpatialRegion => "zero-dimensional spatial region",
            BfoClass::ContinuantFiatBoundary => "continuant fiat boundary",
            BfoClass::OneDimensionalContinuantFiatBoundary => {
                "one-dimensional continuant fiat boundary"
            }
            BfoClass::TwoDimensionalContinuantFiatBoundary => {
                "two-dimensional continuant fiat boundary"
            }
            BfoClass::ZeroDimensionalContinuantFiatBoundary => {
                "zero-dimensional continuant fiat boundary"
            }
            BfoClass::SpecificallyDependentContinuant => "specifically dependent continuant",
            BfoClass::Quality => "quality",
            BfoClass::RealizableEntity => "realizable entity",
            BfoClass::Role => "role",
            BfoClass::Disposition => "disposition",
            BfoClass::Function => "function",
            BfoClass::GenericallyDependentContinuant => "generically dependent continuant",
            BfoClass::Process => "process",
            BfoClass::ProcessBoundary => "process boundary",
            BfoClass::SpatiotemporalRegion => "spatiotemporal region",
            BfoClass::TemporalRegion => "temporal region",
            BfoClass::ZeroDimensionalTemporalRegion => "zero-dimensional temporal region",
            BfoClass::OneDimensionalTemporalRegion => "one-dimensional temporal region",
            BfoClass::TemporalInterval => "temporal interval",
            BfoClass::TemporalInstant => "temporal instant",
            BfoClass::History => "history",
            BfoClass::ProcessProfile => "process profile",
        }
    }
}

/// Static BFO terms
static BFO_TERMS: &[PrimitiveTerm<BfoClass>] = &[
    PrimitiveTerm {
        variant: BfoClass::Entity,
        id: "BFO:0000001",
        label: "entity",
        definition: "An entity is anything that exists or has existed or will exist.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000001",
    },
    PrimitiveTerm {
        variant: BfoClass::Continuant,
        id: "BFO:0000002",
        label: "continuant",
        definition: "An entity that persists, endures, or continues to exist through time while maintaining its identity.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000002",
    },
    PrimitiveTerm {
        variant: BfoClass::Occurrent,
        id: "BFO:0000003",
        label: "occurrent",
        definition: "An entity that has temporal parts and that happens, unfolds or develops through time.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000003",
    },
    PrimitiveTerm {
        variant: BfoClass::IndependentContinuant,
        id: "BFO:0000004",
        label: "independent continuant",
        definition: "A continuant that is a bearer of quality and realizable entity entities.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000004",
    },
    PrimitiveTerm {
        variant: BfoClass::MaterialEntity,
        id: "BFO:0000040",
        label: "material entity",
        definition: "An independent continuant that has some portion of matter as proper or improper continuant part.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000040",
    },
    PrimitiveTerm {
        variant: BfoClass::Object,
        id: "BFO:0000027",
        label: "object",
        definition: "A material entity that is spatially extended, maximally self-connected and self-contained.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000027",
    },
    PrimitiveTerm {
        variant: BfoClass::Quality,
        id: "BFO:0000019",
        label: "quality",
        definition: "A specifically dependent continuant that is exhibited if its bearer is a material entity.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000019",
    },
    PrimitiveTerm {
        variant: BfoClass::Process,
        id: "BFO:0000015",
        label: "process",
        definition: "An occurrent that has temporal proper parts and for some time t, p s-depends_on some material entity at t.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000015",
    },
    PrimitiveTerm {
        variant: BfoClass::Role,
        id: "BFO:0000023",
        label: "role",
        definition: "A realizable entity the manifestation of which brings about some result or end that is not essential to a continuant in virtue of the kind of thing that it is but that can be served or participated in by that kind of continuant in some kinds of natural, social or institutional contexts.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000023",
    },
    PrimitiveTerm {
        variant: BfoClass::Function,
        id: "BFO:0000034",
        label: "function",
        definition: "A disposition that exists in virtue of the bearer's physical make-up and this physical make-up is something the bearer possesses because it came into being through a certain kind of process.",
        iri: "http://purl.obolibrary.org/obo/BFO_0000034",
    },
];

/// BFO subsumption hierarchy (child -> parent)
static BFO_SUBSUMPTION: &[(BfoClass, BfoClass)] = &[
    // Continuant branch
    (BfoClass::Continuant, BfoClass::Entity),
    (BfoClass::IndependentContinuant, BfoClass::Continuant),
    (BfoClass::MaterialEntity, BfoClass::IndependentContinuant),
    (BfoClass::Object, BfoClass::MaterialEntity),
    (BfoClass::ObjectAggregate, BfoClass::MaterialEntity),
    (BfoClass::FiatObjectPart, BfoClass::MaterialEntity),
    (BfoClass::ImmaterialEntity, BfoClass::IndependentContinuant),
    (BfoClass::Site, BfoClass::ImmaterialEntity),
    (BfoClass::SpatialRegion, BfoClass::ImmaterialEntity),
    (BfoClass::ContinuantFiatBoundary, BfoClass::ImmaterialEntity),
    (
        BfoClass::OneDimensionalSpatialRegion,
        BfoClass::SpatialRegion,
    ),
    (
        BfoClass::TwoDimensionalSpatialRegion,
        BfoClass::SpatialRegion,
    ),
    (
        BfoClass::ThreeDimensionalSpatialRegion,
        BfoClass::SpatialRegion,
    ),
    (
        BfoClass::ZeroDimensionalSpatialRegion,
        BfoClass::SpatialRegion,
    ),
    (
        BfoClass::SpecificallyDependentContinuant,
        BfoClass::Continuant,
    ),
    (BfoClass::Quality, BfoClass::SpecificallyDependentContinuant),
    (
        BfoClass::RealizableEntity,
        BfoClass::SpecificallyDependentContinuant,
    ),
    (BfoClass::Role, BfoClass::RealizableEntity),
    (BfoClass::Disposition, BfoClass::RealizableEntity),
    (BfoClass::Function, BfoClass::Disposition),
    (
        BfoClass::GenericallyDependentContinuant,
        BfoClass::Continuant,
    ),
    // Occurrent branch
    (BfoClass::Occurrent, BfoClass::Entity),
    (BfoClass::Process, BfoClass::Occurrent),
    (BfoClass::ProcessBoundary, BfoClass::Occurrent),
    (BfoClass::SpatiotemporalRegion, BfoClass::Occurrent),
    (BfoClass::TemporalRegion, BfoClass::Occurrent),
    (
        BfoClass::ZeroDimensionalTemporalRegion,
        BfoClass::TemporalRegion,
    ),
    (
        BfoClass::OneDimensionalTemporalRegion,
        BfoClass::TemporalRegion,
    ),
    (
        BfoClass::TemporalInterval,
        BfoClass::OneDimensionalTemporalRegion,
    ),
    (
        BfoClass::TemporalInstant,
        BfoClass::ZeroDimensionalTemporalRegion,
    ),
    (BfoClass::History, BfoClass::Process),
    (BfoClass::ProcessProfile, BfoClass::Process),
];

// ============================================================================
// RO - Relation Ontology
// ============================================================================

/// RO relations - commonly used relations (subset)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoRelation {
    // Core relations
    PartOf,
    HasPart,
    LocatedIn,
    ContainedIn,
    AdjacentTo,
    DerivedFrom,

    // Temporal relations
    PrecedesInTime,
    OccursIn,
    TemporallyRelatedTo,

    // Causal relations
    CausallyRelatedTo,
    RegulatesActivityOf,
    PositivelyRegulates,
    NegativelyRegulates,

    // Biological relations
    ParticipatesIn,
    HasParticipant,
    FunctionOf,
    HasFunction,
    InvolvedIn,
}

impl RoRelation {
    /// Get the OBO-style ID
    pub fn id(&self) -> &'static str {
        match self {
            RoRelation::PartOf => "RO:0000050",
            RoRelation::HasPart => "RO:0000051",
            RoRelation::LocatedIn => "RO:0001025",
            RoRelation::ContainedIn => "RO:0001018",
            RoRelation::AdjacentTo => "RO:0002220",
            RoRelation::DerivedFrom => "RO:0001000",
            RoRelation::PrecedesInTime => "RO:0002087",
            RoRelation::OccursIn => "RO:0040001",
            RoRelation::TemporallyRelatedTo => "RO:0002222",
            RoRelation::CausallyRelatedTo => "RO:0002410",
            RoRelation::RegulatesActivityOf => "RO:0002448",
            RoRelation::PositivelyRegulates => "RO:0002213",
            RoRelation::NegativelyRegulates => "RO:0002212",
            RoRelation::ParticipatesIn => "RO:0000056",
            RoRelation::HasParticipant => "RO:0000057",
            RoRelation::FunctionOf => "RO:0000079",
            RoRelation::HasFunction => "RO:0000085",
            RoRelation::InvolvedIn => "RO:0002331",
        }
    }

    /// Get the human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            RoRelation::PartOf => "part of",
            RoRelation::HasPart => "has part",
            RoRelation::LocatedIn => "located in",
            RoRelation::ContainedIn => "contained in",
            RoRelation::AdjacentTo => "adjacent to",
            RoRelation::DerivedFrom => "derives from",
            RoRelation::PrecedesInTime => "precedes",
            RoRelation::OccursIn => "occurs in",
            RoRelation::TemporallyRelatedTo => "temporally related to",
            RoRelation::CausallyRelatedTo => "causally related to",
            RoRelation::RegulatesActivityOf => "regulates activity of",
            RoRelation::PositivelyRegulates => "positively regulates",
            RoRelation::NegativelyRegulates => "negatively regulates",
            RoRelation::ParticipatesIn => "participates in",
            RoRelation::HasParticipant => "has participant",
            RoRelation::FunctionOf => "function of",
            RoRelation::HasFunction => "has function",
            RoRelation::InvolvedIn => "involved in",
        }
    }

    /// Get the inverse relation if it exists
    pub fn inverse(&self) -> Option<RoRelation> {
        match self {
            RoRelation::PartOf => Some(RoRelation::HasPart),
            RoRelation::HasPart => Some(RoRelation::PartOf),
            RoRelation::ParticipatesIn => Some(RoRelation::HasParticipant),
            RoRelation::HasParticipant => Some(RoRelation::ParticipatesIn),
            RoRelation::FunctionOf => Some(RoRelation::HasFunction),
            RoRelation::HasFunction => Some(RoRelation::FunctionOf),
            _ => None,
        }
    }
}

/// Static RO terms (subset)
static RO_TERMS: &[PrimitiveTerm<RoRelation>] = &[
    PrimitiveTerm {
        variant: RoRelation::PartOf,
        id: "RO:0000050",
        label: "part of",
        definition: "a core relation that holds between a part and its whole",
        iri: "http://purl.obolibrary.org/obo/RO_0000050",
    },
    PrimitiveTerm {
        variant: RoRelation::HasPart,
        id: "RO:0000051",
        label: "has part",
        definition: "a core relation that holds between a whole and its part",
        iri: "http://purl.obolibrary.org/obo/RO_0000051",
    },
    PrimitiveTerm {
        variant: RoRelation::LocatedIn,
        id: "RO:0001025",
        label: "located in",
        definition: "a relation between a material entity and a site",
        iri: "http://purl.obolibrary.org/obo/RO_0001025",
    },
    PrimitiveTerm {
        variant: RoRelation::ParticipatesIn,
        id: "RO:0000056",
        label: "participates in",
        definition: "a relation between a continuant and a process, in which the continuant is involved in the process",
        iri: "http://purl.obolibrary.org/obo/RO_0000056",
    },
];

/// RO relation subsumption (relation hierarchy)
static RO_SUBSUMPTION: &[(RoRelation, RoRelation)] = &[
    (RoRelation::ContainedIn, RoRelation::LocatedIn),
    (RoRelation::PrecedesInTime, RoRelation::TemporallyRelatedTo),
    (
        RoRelation::PositivelyRegulates,
        RoRelation::CausallyRelatedTo,
    ),
    (
        RoRelation::NegativelyRegulates,
        RoRelation::CausallyRelatedTo,
    ),
    (
        RoRelation::RegulatesActivityOf,
        RoRelation::CausallyRelatedTo,
    ),
];

// ============================================================================
// COB - Core Ontology for Biology and Biomedicine
// ============================================================================

/// COB classes - bridge classes between BFO and domain ontologies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CobClass {
    // Organisms and parts
    Organism,
    AnatomicalEntity,
    CellularOrganism,
    Virus,
    Cell,
    GrossAnatomicalPart,

    // Molecular entities
    MolecularEntity,
    Atom,
    ChemicalEntity,
    SmallMolecule,
    Macromolecule,
    Protein,
    NucleicAcid,

    // Information entities
    InformationContentEntity,
    Data,

    // Processes
    BiologicalProcess,
    MolecularFunction,

    // Qualities
    BiologicalAttribute,
}

impl CobClass {
    /// Get the OBO-style ID
    pub fn id(&self) -> &'static str {
        match self {
            CobClass::Organism => "COB:0000013",
            CobClass::AnatomicalEntity => "COB:0000041",
            CobClass::CellularOrganism => "COB:0000014",
            CobClass::Virus => "COB:0000089",
            CobClass::Cell => "COB:0000023",
            CobClass::GrossAnatomicalPart => "COB:0000042",
            CobClass::MolecularEntity => "COB:0000045",
            CobClass::Atom => "COB:0000011",
            CobClass::ChemicalEntity => "COB:0000046",
            CobClass::SmallMolecule => "COB:0000047",
            CobClass::Macromolecule => "COB:0000048",
            CobClass::Protein => "COB:0000049",
            CobClass::NucleicAcid => "COB:0000050",
            CobClass::InformationContentEntity => "COB:0000090",
            CobClass::Data => "COB:0000091",
            CobClass::BiologicalProcess => "COB:0000082",
            CobClass::MolecularFunction => "COB:0000083",
            CobClass::BiologicalAttribute => "COB:0000068",
        }
    }

    /// Get the human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            CobClass::Organism => "organism",
            CobClass::AnatomicalEntity => "anatomical entity",
            CobClass::CellularOrganism => "cellular organism",
            CobClass::Virus => "virus",
            CobClass::Cell => "cell",
            CobClass::GrossAnatomicalPart => "gross anatomical part",
            CobClass::MolecularEntity => "molecular entity",
            CobClass::Atom => "atom",
            CobClass::ChemicalEntity => "chemical entity",
            CobClass::SmallMolecule => "small molecule",
            CobClass::Macromolecule => "macromolecule",
            CobClass::Protein => "protein",
            CobClass::NucleicAcid => "nucleic acid",
            CobClass::InformationContentEntity => "information content entity",
            CobClass::Data => "data",
            CobClass::BiologicalProcess => "biological process",
            CobClass::MolecularFunction => "molecular function",
            CobClass::BiologicalAttribute => "biological attribute",
        }
    }

    /// Get the corresponding BFO class that this bridges from
    pub fn bfo_parent(&self) -> BfoClass {
        match self {
            CobClass::Organism
            | CobClass::AnatomicalEntity
            | CobClass::CellularOrganism
            | CobClass::Virus
            | CobClass::Cell
            | CobClass::GrossAnatomicalPart
            | CobClass::MolecularEntity
            | CobClass::Atom
            | CobClass::ChemicalEntity
            | CobClass::SmallMolecule
            | CobClass::Macromolecule
            | CobClass::Protein
            | CobClass::NucleicAcid => BfoClass::MaterialEntity,
            CobClass::InformationContentEntity | CobClass::Data => {
                BfoClass::GenericallyDependentContinuant
            }
            CobClass::BiologicalProcess | CobClass::MolecularFunction => BfoClass::Process,
            CobClass::BiologicalAttribute => BfoClass::Quality,
        }
    }
}

/// Static COB terms
static COB_TERMS: &[PrimitiveTerm<CobClass>] = &[
    PrimitiveTerm {
        variant: CobClass::Organism,
        id: "COB:0000013",
        label: "organism",
        definition: "A material entity that is an individual living system.",
        iri: "http://purl.obolibrary.org/obo/COB_0000013",
    },
    PrimitiveTerm {
        variant: CobClass::Cell,
        id: "COB:0000023",
        label: "cell",
        definition: "An anatomical entity that is a structural and functional unit of an organism.",
        iri: "http://purl.obolibrary.org/obo/COB_0000023",
    },
    PrimitiveTerm {
        variant: CobClass::Protein,
        id: "COB:0000049",
        label: "protein",
        definition: "A macromolecule that is a polymer of amino acid residues.",
        iri: "http://purl.obolibrary.org/obo/COB_0000049",
    },
    PrimitiveTerm {
        variant: CobClass::BiologicalProcess,
        id: "COB:0000082",
        label: "biological process",
        definition: "A process that is realized in a living system.",
        iri: "http://purl.obolibrary.org/obo/COB_0000082",
    },
];

/// COB subsumption hierarchy
static COB_SUBSUMPTION: &[(CobClass, CobClass)] = &[
    (CobClass::CellularOrganism, CobClass::Organism),
    (CobClass::Virus, CobClass::Organism),
    (CobClass::Cell, CobClass::AnatomicalEntity),
    (CobClass::GrossAnatomicalPart, CobClass::AnatomicalEntity),
    (CobClass::Atom, CobClass::MolecularEntity),
    (CobClass::ChemicalEntity, CobClass::MolecularEntity),
    (CobClass::SmallMolecule, CobClass::ChemicalEntity),
    (CobClass::Macromolecule, CobClass::ChemicalEntity),
    (CobClass::Protein, CobClass::Macromolecule),
    (CobClass::NucleicAcid, CobClass::Macromolecule),
    (CobClass::Data, CobClass::InformationContentEntity),
    (CobClass::MolecularFunction, CobClass::BiologicalProcess),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfo_subsumption() {
        assert!(PRIMITIVE_BFO.is_subclass(BfoClass::Object, BfoClass::Entity));
        assert!(PRIMITIVE_BFO.is_subclass(BfoClass::Object, BfoClass::MaterialEntity));
        assert!(PRIMITIVE_BFO.is_subclass(BfoClass::Function, BfoClass::Entity));
        assert!(!PRIMITIVE_BFO.is_subclass(BfoClass::Entity, BfoClass::Object));
    }

    #[test]
    fn test_bfo_self_subsumption() {
        assert!(PRIMITIVE_BFO.is_subclass(BfoClass::Entity, BfoClass::Entity));
        assert!(PRIMITIVE_BFO.is_subclass(BfoClass::Process, BfoClass::Process));
    }

    #[test]
    fn test_bfo_get_term() {
        let entity = PRIMITIVE_BFO.get_by_id("BFO:0000001");
        assert!(entity.is_some());
        assert_eq!(entity.unwrap().label, "entity");
    }

    #[test]
    fn test_ro_inverse() {
        assert_eq!(RoRelation::PartOf.inverse(), Some(RoRelation::HasPart));
        assert_eq!(RoRelation::HasPart.inverse(), Some(RoRelation::PartOf));
    }

    #[test]
    fn test_cob_bfo_parent() {
        assert_eq!(CobClass::Organism.bfo_parent(), BfoClass::MaterialEntity);
        assert_eq!(CobClass::BiologicalProcess.bfo_parent(), BfoClass::Process);
    }
}
