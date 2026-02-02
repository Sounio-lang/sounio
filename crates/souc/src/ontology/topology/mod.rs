// Sounio Compiler - Topological Analysis via Persistent Homology
// Detect ontology structure using Rips complexes and persistent diagrams

pub mod persistent;

pub use persistent::{
    PersistenceDiagram, PersistentHomologyAnalyzer, RipsComplex, TopologicalFeature,
};
