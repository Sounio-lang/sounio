//! L2 Cache for Ontology Terms
//!
//! Provides persistent storage for ontology terms using an embedded database.
//! This serves as the "warm" cache tier between the in-memory L1 cache and
//! federated resolution.

use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap;

use super::{
    CrossReference, IRI, LoadedTerm, OntologyId, Restriction, RestrictionType, Synonym,
    SynonymScope,
};

/// Cache error type
#[derive(Debug, Clone, thiserror::Error)]
pub enum CacheError {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),
}

/// L2 cache backed by a simple file-based store
/// In production, this would use SQLite or DuckDB
pub struct L2Cache {
    /// In-memory backing store (simulating disk cache)
    store: RwLock<FxHashMap<String, CachedTerm>>,

    /// Cache file path
    path: std::path::PathBuf,

    /// Number of entries
    count: AtomicUsize,

    /// Maximum entries
    max_entries: usize,
}

/// Serializable cached term
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CachedTerm {
    iri: String,
    label: String,
    ontology: String,
    superclasses: Vec<String>,
    subclasses: Vec<String>,
    definition: Option<String>,
    synonyms: Vec<CachedSynonym>,
    xrefs: Vec<CachedXref>,
    restrictions: Vec<CachedRestriction>,
    hierarchy_depth: u32,
    information_content: f64,
    is_obsolete: bool,
    replaced_by: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CachedSynonym {
    text: String,
    scope: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CachedXref {
    target: String,
    confidence: f64,
    source: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CachedRestriction {
    property: String,
    restriction_type: String,
    filler: String,
}

impl L2Cache {
    /// Open or create an L2 cache at the given path
    pub fn open(path: &Path) -> Result<Self, CacheError> {
        let store = if path.exists() {
            // Try to load existing cache
            match std::fs::read(path) {
                Ok(data) => {
                    bincode::deserialize::<FxHashMap<String, CachedTerm>>(&data).unwrap_or_default()
                }
                Err(_) => FxHashMap::default(),
            }
        } else {
            FxHashMap::default()
        };

        let count = store.len();

        Ok(Self {
            store: RwLock::new(store),
            path: path.to_path_buf(),
            count: AtomicUsize::new(count),
            max_entries: 100_000,
        })
    }

    /// Get a term from the cache
    pub fn get(&self, iri: &IRI) -> Result<Option<Arc<LoadedTerm>>, CacheError> {
        let store = self
            .store
            .read()
            .map_err(|e| CacheError::DatabaseError(e.to_string()))?;

        if let Some(cached) = store.get(&iri.0) {
            Ok(Some(Arc::new(cached.to_loaded_term())))
        } else {
            Ok(None)
        }
    }

    /// Put a term into the cache
    pub fn put(&self, iri: &IRI, term: &LoadedTerm) -> Result<(), CacheError> {
        let cached = CachedTerm::from_loaded_term(term);

        {
            let mut store = self
                .store
                .write()
                .map_err(|e| CacheError::DatabaseError(e.to_string()))?;

            // Evict if at capacity
            if store.len() >= self.max_entries {
                // Simple eviction: remove ~10% of entries
                let to_remove: Vec<String> =
                    store.keys().take(self.max_entries / 10).cloned().collect();
                for key in to_remove {
                    store.remove(&key);
                }
            }

            store.insert(iri.0.clone(), cached);
            self.count.store(store.len(), Ordering::Relaxed);
        }

        Ok(())
    }

    /// Check if a term is in the cache
    pub fn contains(&self, iri: &IRI) -> bool {
        self.store
            .read()
            .map(|s| s.contains_key(&iri.0))
            .unwrap_or(false)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut store) = self.store.write() {
            store.clear();
            self.count.store(0, Ordering::Relaxed);
        }
    }

    /// Persist the cache to disk
    pub fn flush(&self) -> Result<(), CacheError> {
        let store = self
            .store
            .read()
            .map_err(|e| CacheError::DatabaseError(e.to_string()))?;
        let data = bincode::serialize(&*store)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        std::fs::write(&self.path, data).map_err(|e| CacheError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Get all IRIs in the cache for a given ontology
    pub fn get_ontology_iris(&self, ontology: OntologyId) -> Vec<IRI> {
        let prefix = ontology.prefix();
        let Ok(store) = self.store.read() else {
            return Vec::new();
        };
        store
            .keys()
            .filter(|k| {
                if let Some((p, _)) = IRI::new(k).to_curie() {
                    p.to_uppercase() == prefix
                } else {
                    false
                }
            })
            .map(|k| IRI::new(k))
            .collect()
    }

    /// Bulk insert terms
    pub fn put_batch(&self, terms: &[LoadedTerm]) -> Result<usize, CacheError> {
        let mut store = self
            .store
            .write()
            .map_err(|e| CacheError::DatabaseError(e.to_string()))?;
        let mut count = 0;

        for term in terms {
            if store.len() >= self.max_entries {
                break;
            }

            let cached = CachedTerm::from_loaded_term(term);
            store.insert(term.iri.0.clone(), cached);
            count += 1;
        }

        self.count.store(store.len(), Ordering::Relaxed);
        Ok(count)
    }
}

impl Drop for L2Cache {
    fn drop(&mut self) {
        // Try to persist on drop
        let _ = self.flush();
    }
}

impl CachedTerm {
    fn from_loaded_term(term: &LoadedTerm) -> Self {
        Self {
            iri: term.iri.0.clone(),
            label: term.label.clone(),
            ontology: term.ontology.prefix().to_string(),
            superclasses: term.superclasses.iter().map(|i| i.0.clone()).collect(),
            subclasses: term.subclasses.iter().map(|i| i.0.clone()).collect(),
            definition: term.definition.clone(),
            synonyms: term
                .synonyms
                .iter()
                .map(|s| CachedSynonym {
                    text: s.text.clone(),
                    scope: match s.scope {
                        SynonymScope::Exact => "EXACT",
                        SynonymScope::Narrow => "NARROW",
                        SynonymScope::Broad => "BROAD",
                        SynonymScope::Related => "RELATED",
                    }
                    .to_string(),
                })
                .collect(),
            xrefs: term
                .xrefs
                .iter()
                .map(|x| CachedXref {
                    target: x.target.0.clone(),
                    confidence: x.confidence,
                    source: x.source.clone(),
                })
                .collect(),
            restrictions: term
                .restrictions
                .iter()
                .map(|r| CachedRestriction {
                    property: r.property.0.clone(),
                    restriction_type: match r.restriction_type {
                        RestrictionType::Some => "SOME".to_string(),
                        RestrictionType::All => "ALL".to_string(),
                        RestrictionType::Value => "VALUE".to_string(),
                        RestrictionType::Exact(n) => format!("EXACT:{}", n),
                        RestrictionType::Min(n) => format!("MIN:{}", n),
                        RestrictionType::Max(n) => format!("MAX:{}", n),
                    },
                    filler: r.filler.0.clone(),
                })
                .collect(),
            hierarchy_depth: term.hierarchy_depth,
            information_content: term.information_content,
            is_obsolete: term.is_obsolete,
            replaced_by: term.replaced_by.as_ref().map(|i| i.0.clone()),
        }
    }

    fn to_loaded_term(&self) -> LoadedTerm {
        let iri = IRI::new(&self.iri);
        let ontology = OntologyId::from_prefix(&self.ontology);

        let mut term = LoadedTerm::new(iri, self.label.clone(), ontology);
        term.superclasses = self.superclasses.iter().map(|s| IRI::new(s)).collect();
        term.subclasses = self.subclasses.iter().map(|s| IRI::new(s)).collect();
        term.definition = self.definition.clone();
        term.synonyms = self
            .synonyms
            .iter()
            .map(|s| Synonym {
                text: s.text.clone(),
                scope: match s.scope.as_str() {
                    "EXACT" => SynonymScope::Exact,
                    "NARROW" => SynonymScope::Narrow,
                    "BROAD" => SynonymScope::Broad,
                    _ => SynonymScope::Related,
                },
            })
            .collect();
        term.xrefs = self
            .xrefs
            .iter()
            .map(|x| CrossReference {
                target: IRI::new(&x.target),
                confidence: x.confidence,
                source: x.source.clone(),
            })
            .collect();
        term.restrictions = self
            .restrictions
            .iter()
            .map(|r| {
                let restriction_type = if r.restriction_type == "SOME" {
                    RestrictionType::Some
                } else if r.restriction_type == "ALL" {
                    RestrictionType::All
                } else if r.restriction_type == "VALUE" {
                    RestrictionType::Value
                } else if r.restriction_type.starts_with("EXACT:") {
                    RestrictionType::Exact(r.restriction_type[6..].parse().unwrap_or(1))
                } else if r.restriction_type.starts_with("MIN:") {
                    RestrictionType::Min(r.restriction_type[4..].parse().unwrap_or(0))
                } else if r.restriction_type.starts_with("MAX:") {
                    RestrictionType::Max(r.restriction_type[4..].parse().unwrap_or(u32::MAX))
                } else {
                    RestrictionType::Some
                };

                Restriction {
                    property: IRI::new(&r.property),
                    restriction_type,
                    filler: IRI::new(&r.filler),
                }
            })
            .collect();
        term.hierarchy_depth = self.hierarchy_depth;
        term.information_content = self.information_content;
        term.is_obsolete = self.is_obsolete;
        term.replaced_by = self.replaced_by.as_ref().map(|s| IRI::new(s));

        term
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_operations() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("test_cache.db");

        let cache = L2Cache::open(&cache_path).unwrap();

        // Create a test term
        let iri = IRI::from_curie("CHEBI", "15365");
        let mut term = LoadedTerm::new(iri.clone(), "aspirin".to_string(), OntologyId::ChEBI);
        term.definition = Some("A drug".to_string());
        term.superclasses.push(IRI::from_curie("CHEBI", "35475"));

        // Put and get
        cache.put(&iri, &term).unwrap();
        assert!(cache.contains(&iri));

        let retrieved = cache.get(&iri).unwrap().unwrap();
        assert_eq!(retrieved.label, "aspirin");
        assert_eq!(retrieved.definition, Some("A drug".to_string()));
        assert!(!retrieved.superclasses.is_empty());
    }

    #[test]
    fn test_cache_persistence() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("persist_cache.db");

        // Write to cache
        {
            let cache = L2Cache::open(&cache_path).unwrap();
            let iri = IRI::from_curie("GO", "0008150");
            let term = LoadedTerm::new(
                iri.clone(),
                "biological_process".to_string(),
                OntologyId::GO,
            );
            cache.put(&iri, &term).unwrap();
            cache.flush().unwrap();
        }

        // Reopen and verify
        {
            let cache = L2Cache::open(&cache_path).unwrap();
            let iri = IRI::from_curie("GO", "0008150");
            let retrieved = cache.get(&iri).unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().label, "biological_process");
        }
    }

    #[test]
    fn test_batch_insert() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("batch_cache.db");

        let cache = L2Cache::open(&cache_path).unwrap();

        let terms: Vec<LoadedTerm> = (0..100)
            .map(|i| {
                LoadedTerm::new(
                    IRI::from_curie("TEST", &format!("{:04}", i)),
                    format!("term_{}", i),
                    OntologyId::Unknown,
                )
            })
            .collect();

        let count = cache.put_batch(&terms).unwrap();
        assert_eq!(count, 100);
        assert_eq!(cache.len(), 100);
    }
}
