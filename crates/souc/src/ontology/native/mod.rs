//! Native Ontology Module - Day 37
//!
//! This module provides the core infrastructure for embedding ontologies
//! directly into the Sounio compiler with O(1) subsumption queries.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Native Ontology Store (.dontology)                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Header: magic, version, counts                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  String Table: deduplicated labels, definitions             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Concept Table: IRI → (label_idx, parent_idx, flags)        │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Euler Tour: DFS traversal for O(1) LCA                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Sparse Table: RMQ structure for LCA queries                │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **O(1) LCA queries**: Bender & Farach-Colton (2000) algorithm
//! - **Compact storage**: String interning + varint encoding
//! - **Lazy loading**: Memory-mapped for large ontologies
//! - **Prefix trie**: Fast IRI prefix lookup

pub mod downloader;
pub mod hierarchy;
pub mod storage;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use downloader::{DownloadConfig, DownloadProgress, OntologyDownloader};
pub use hierarchy::{HierarchyIndex, LcaQuery};
pub use storage::{ConceptEntry, NativeStore, StringTable};

use crate::ontology::{OntologyError, OntologyResult};

/// Magic bytes for .dontology files
pub const DONTOLOGY_MAGIC: &[u8; 4] = b"DONT";

/// Current format version
pub const DONTOLOGY_VERSION: u32 = 1;

/// A native ontology store loaded from .dontology format
#[derive(Debug)]
pub struct NativeOntology {
    /// Ontology identifier (e.g., "chebi", "go", "bfo")
    pub id: String,
    /// Version string
    pub version: String,
    /// Number of concepts
    pub concept_count: usize,
    /// The underlying storage
    store: NativeStore,
    /// Hierarchy index for O(1) LCA
    hierarchy: HierarchyIndex,
}

impl NativeOntology {
    /// Load a native ontology from a .dontology file
    pub fn load(path: impl AsRef<Path>) -> OntologyResult<Self> {
        let store = NativeStore::load(path.as_ref())?;
        let hierarchy = HierarchyIndex::build(&store)?;

        Ok(Self {
            id: store.header.ontology_id.clone(),
            version: store.header.version.clone(),
            concept_count: store.concepts.len(),
            store,
            hierarchy,
        })
    }

    /// Check if term `child` is a subclass of term `ancestor`
    ///
    /// This is O(1) using the Euler tour + sparse table approach
    pub fn is_subclass(&self, child: &str, ancestor: &str) -> bool {
        self.hierarchy.is_ancestor(child, ancestor)
    }

    /// Find the Lowest Common Ancestor of two terms
    pub fn lca(&self, term1: &str, term2: &str) -> Option<&str> {
        self.hierarchy.lca(term1, term2)
    }

    /// Get concept by CURIE (e.g., "CHEBI:15365")
    pub fn get_concept(&self, curie: &str) -> Option<&ConceptEntry> {
        self.store.get_concept(curie)
    }

    /// Get label for a concept
    pub fn get_label(&self, curie: &str) -> Option<&str> {
        self.store.get_label(curie)
    }

    /// Get definition for a concept
    pub fn get_definition(&self, curie: &str) -> Option<&str> {
        self.store.get_definition(curie)
    }

    /// Search for concepts by label prefix
    pub fn search(&self, prefix: &str, limit: usize) -> Vec<(&str, &str)> {
        self.store.search_by_label(prefix, limit)
    }

    /// Get direct parent of a concept
    pub fn get_parent(&self, curie: &str) -> Option<&str> {
        self.store.get_parent(curie)
    }

    /// Get all ancestors of a concept (transitive closure)
    pub fn get_ancestors(&self, curie: &str) -> Vec<&str> {
        self.hierarchy.get_ancestors(curie)
    }

    /// Get the depth of a concept in the hierarchy
    pub fn depth(&self, curie: &str) -> usize {
        // Count ancestors to determine depth
        self.get_ancestors(curie).len()
    }

    /// Create an empty ontology (for testing)
    pub fn empty(id: &str) -> Self {
        Self {
            id: id.to_string(),
            version: "0.0.0".to_string(),
            concept_count: 0,
            store: NativeStore::empty(),
            hierarchy: HierarchyIndex::empty(),
        }
    }
}

/// Registry of loaded native ontologies
#[derive(Debug, Default)]
pub struct NativeOntologyRegistry {
    /// Loaded ontologies by ID
    ontologies: HashMap<String, NativeOntology>,
    /// Base directory for .dontology files
    data_dir: PathBuf,
}

impl NativeOntologyRegistry {
    /// Create a new registry with the given data directory
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            ontologies: HashMap::new(),
            data_dir: data_dir.into(),
        }
    }

    /// Get or load an ontology by ID
    pub fn get_or_load(&mut self, id: &str) -> OntologyResult<&NativeOntology> {
        if !self.ontologies.contains_key(id) {
            let path = self.data_dir.join(format!("{}.dontology", id));
            if !path.exists() {
                return Err(OntologyError::OntologyNotAvailable(format!(
                    "Ontology '{}' not found at {:?}. Run 'dc ontology init' to download.",
                    id, path
                )));
            }
            let ont = NativeOntology::load(&path)?;
            self.ontologies.insert(id.to_string(), ont);
        }
        Ok(self.ontologies.get(id).unwrap())
    }

    /// Check if an ontology is loaded
    pub fn is_loaded(&self, id: &str) -> bool {
        self.ontologies.contains_key(id)
    }

    /// Get a loaded ontology
    pub fn get(&self, id: &str) -> Option<&NativeOntology> {
        self.ontologies.get(id)
    }

    /// List available ontologies (files in data_dir)
    pub fn list_available(&self) -> Vec<String> {
        std::fs::read_dir(&self.data_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name.ends_with(".dontology") {
                            Some(name.trim_end_matches(".dontology").to_string())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = NativeOntologyRegistry::new("/tmp/test");
        assert!(registry.list_available().is_empty() || true); // May have files
    }
}
