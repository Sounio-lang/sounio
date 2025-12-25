//! In-Memory Embedding Storage
//!
//! Simple HashMap-based storage for development and small datasets.

use std::collections::HashMap;
use std::sync::RwLock;

use super::super::{Embedding, EmbeddingError, EmbeddingStore};
use crate::ontology::loader::IRI;

/// In-memory embedding store
pub struct MemoryStore {
    embeddings: RwLock<HashMap<IRI, Embedding>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            embeddings: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            embeddings: RwLock::new(HashMap::with_capacity(capacity)),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingStore for MemoryStore {
    fn get(&self, iri: &IRI) -> Result<Option<Embedding>, EmbeddingError> {
        if let Ok(store) = self.embeddings.read() {
            Ok(store.get(iri).cloned())
        } else {
            Ok(None)
        }
    }

    fn put(&mut self, iri: &IRI, embedding: &Embedding) -> Result<(), EmbeddingError> {
        if let Ok(mut store) = self.embeddings.write() {
            store.insert(iri.clone(), embedding.clone());
        }
        Ok(())
    }

    fn delete(&mut self, iri: &IRI) -> Result<(), EmbeddingError> {
        if let Ok(mut store) = self.embeddings.write() {
            store.remove(iri);
        }
        Ok(())
    }

    fn all(&self) -> Result<Vec<Embedding>, EmbeddingError> {
        if let Ok(store) = self.embeddings.read() {
            Ok(store.values().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    fn count(&self) -> Result<usize, EmbeddingError> {
        if let Ok(store) = self.embeddings.read() {
            Ok(store.len())
        } else {
            Ok(0)
        }
    }

    fn contains(&self, iri: &IRI) -> Result<bool, EmbeddingError> {
        if let Ok(store) = self.embeddings.read() {
            Ok(store.contains_key(iri))
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::embedding::EmbeddingModel;

    #[test]
    fn test_memory_store_basic() {
        let mut store = MemoryStore::new();
        let iri = IRI::new("http://example.org/term1");
        let embedding = Embedding::new(iri.clone(), vec![1.0, 0.0, 0.0], EmbeddingModel::Random);

        store.put(&iri, &embedding).unwrap();

        assert!(store.contains(&iri).unwrap());
        assert_eq!(store.count().unwrap(), 1);

        let retrieved = store.get(&iri).unwrap().unwrap();
        assert_eq!(retrieved.iri, iri);
    }

    #[test]
    fn test_memory_store_delete() {
        let mut store = MemoryStore::new();
        let iri = IRI::new("http://example.org/term1");
        let embedding = Embedding::new(iri.clone(), vec![1.0, 0.0, 0.0], EmbeddingModel::Random);

        store.put(&iri, &embedding).unwrap();
        assert!(store.contains(&iri).unwrap());

        store.delete(&iri).unwrap();
        assert!(!store.contains(&iri).unwrap());
    }
}
