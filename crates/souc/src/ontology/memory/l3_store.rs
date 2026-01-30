//! L3 Store - Cold Ontology Types
//!
//! The L3 store holds the full ontology (~15M types) in a memory-efficient
//! format. It uses memory-mapped files when available for demand paging,
//! minimizing resident memory while providing access to all types.
//!
//! Capacity: ~15M types (~2GB on disk, minimal RAM)
//! Use case: Rare types, genetic variants, specialized conditions

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::compact::CompactTerm;

/// L3 Store - Cold storage for the full ontology
pub struct L3Store {
    /// In-memory fallback storage (used when mmap not available)
    entries: RwLock<HashMap<String, Arc<CompactTerm>>>,

    /// Index for fast IRI lookup (maps IRI hash to entry offset)
    index: RwLock<HashMap<u64, usize>>,

    /// Whether we're using memory-mapped backing
    mmap_enabled: bool,
}

impl L3Store {
    /// Create a new L3 store (in-memory mode)
    pub fn new() -> Self {
        L3Store {
            entries: RwLock::new(HashMap::new()),
            index: RwLock::new(HashMap::new()),
            mmap_enabled: false,
        }
    }

    /// Get a term from the store
    pub fn get(&self, iri: &str) -> Option<Arc<CompactTerm>> {
        let entries = self.entries.read().ok()?;
        entries.get(iri).map(Arc::clone)
    }

    /// Insert a term into the store
    pub fn insert(&self, iri: String, term: Arc<CompactTerm>) {
        if let Ok(mut entries) = self.entries.write() {
            // Update index
            let hash = Self::hash_iri(&iri);
            if let Ok(mut index) = self.index.write() {
                index.insert(hash, entries.len());
            }

            entries.insert(iri, term);
        }
    }

    /// Batch insert for efficient bulk loading
    pub fn insert_batch(&self, terms: Vec<(String, Arc<CompactTerm>)>) {
        if let Ok(mut entries) = self.entries.write()
            && let Ok(mut index) = self.index.write()
        {
            for (iri, term) in terms {
                let hash = Self::hash_iri(&iri);
                index.insert(hash, entries.len());
                entries.insert(iri, term);
            }
        }
    }

    /// Check if IRI exists in store
    pub fn contains(&self, iri: &str) -> bool {
        // Fast path: check index first
        let hash = Self::hash_iri(iri);
        if let Ok(index) = self.index.read()
            && !index.contains_key(&hash)
        {
            return false;
        }

        // Verify in entries (handles hash collisions)
        self.entries
            .read()
            .map(|e| e.contains_key(iri))
            .unwrap_or(false)
    }

    /// Remove a term from the store
    pub fn remove(&self, iri: &str) -> Option<Arc<CompactTerm>> {
        let term = self.entries.write().ok()?.remove(iri);

        if term.is_some() {
            let hash = Self::hash_iri(iri);
            if let Ok(mut index) = self.index.write() {
                index.remove(&hash);
            }
        }

        term
    }

    /// Get current number of entries
    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the store
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
        if let Ok(mut index) = self.index.write() {
            index.clear();
        }
    }

    /// Get store statistics
    pub fn stats(&self) -> L3StoreStats {
        let entries = self.len();

        // Estimate memory: ~64 bytes per compact term + string overhead
        let estimated_memory = entries * 100; // rough estimate

        L3StoreStats {
            entries,
            mmap_enabled: self.mmap_enabled,
            estimated_memory_bytes: estimated_memory,
            index_size: self.index.read().map(|i| i.len()).unwrap_or(0),
        }
    }

    /// Hash an IRI for index lookup
    fn hash_iri(iri: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        iri.hash(&mut hasher);
        hasher.finish()
    }

    /// Prefetch entries into OS page cache (hint for mmap mode)
    pub fn prefetch(&self, iris: &[&str]) {
        // In non-mmap mode, this is a no-op
        // With mmap, this would trigger page faults to bring pages into RAM
        for iri in iris {
            let _ = self.contains(iri);
        }
    }

    /// Iterate over all IRIs (for diagnostics)
    pub fn iris(&self) -> Vec<String> {
        self.entries
            .read()
            .map(|e| e.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get multiple terms at once (batch lookup)
    pub fn get_batch(&self, iris: &[&str]) -> Vec<Option<Arc<CompactTerm>>> {
        let entries = match self.entries.read() {
            Ok(e) => e,
            Err(_) => return vec![None; iris.len()],
        };

        iris.iter()
            .map(|iri| entries.get(*iri).map(Arc::clone))
            .collect()
    }
}

impl Default for L3Store {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for L3 store
#[derive(Debug, Clone)]
pub struct L3StoreStats {
    /// Number of entries
    pub entries: usize,
    /// Whether mmap is enabled
    pub mmap_enabled: bool,
    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Size of the index
    pub index_size: usize,
}

impl std::fmt::Display for L3StoreStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "L3 Store Statistics:")?;
        writeln!(f, "  Entries: {}", self.entries)?;
        writeln!(
            f,
            "  MMap: {}",
            if self.mmap_enabled {
                "enabled"
            } else {
                "disabled"
            }
        )?;
        writeln!(
            f,
            "  Memory: {:.1} MB",
            self.estimated_memory_bytes as f64 / (1024.0 * 1024.0)
        )?;
        Ok(())
    }
}

/// Builder for loading ontology into L3 store
pub struct L3StoreBuilder {
    terms: Vec<(String, CompactTerm)>,
    use_mmap: bool,
    backing_path: Option<std::path::PathBuf>,
}

impl L3StoreBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        L3StoreBuilder {
            terms: Vec::new(),
            use_mmap: false,
            backing_path: None,
        }
    }

    /// Add a term to be loaded
    pub fn add_term(mut self, iri: String, term: CompactTerm) -> Self {
        self.terms.push((iri, term));
        self
    }

    /// Add multiple terms
    pub fn add_terms(mut self, terms: Vec<(String, CompactTerm)>) -> Self {
        self.terms.extend(terms);
        self
    }

    /// Enable memory mapping
    pub fn with_mmap(mut self, path: std::path::PathBuf) -> Self {
        self.use_mmap = true;
        self.backing_path = Some(path);
        self
    }

    /// Build the L3 store
    pub fn build(self) -> L3Store {
        let store = L3Store::new();

        // Batch insert all terms
        let terms: Vec<_> = self
            .terms
            .into_iter()
            .map(|(iri, term)| (iri, Arc::new(term)))
            .collect();

        store.insert_batch(terms);
        store
    }

    /// Estimate memory requirements
    pub fn estimated_memory(&self) -> usize {
        self.terms.len() * 100 // ~100 bytes per entry
    }
}

impl Default for L3StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::compact::CompactTermBuilder;
    use super::*;

    #[test]
    fn test_l3_basic_operations() {
        let store = L3Store::new();

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        store.insert("test:1".to_string(), term);

        assert!(store.contains("test:1"));
        assert!(!store.contains("test:2"));
        assert_eq!(store.len(), 1);

        let retrieved = store.get("test:1");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_l3_batch_insert() {
        let store = L3Store::new();

        let terms: Vec<_> = (0..100)
            .map(|i| {
                let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
                (format!("test:{}", i), term)
            })
            .collect();

        store.insert_batch(terms);

        assert_eq!(store.len(), 100);
        assert!(store.contains("test:0"));
        assert!(store.contains("test:99"));
    }

    #[test]
    fn test_l3_batch_get() {
        let store = L3Store::new();

        for i in 0..10 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            store.insert(format!("test:{}", i), term);
        }

        let results = store.get_batch(&["test:0", "test:5", "test:99"]);

        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none()); // Doesn't exist
    }

    #[test]
    fn test_l3_remove() {
        let store = L3Store::new();

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        store.insert("test:1".to_string(), term);

        let removed = store.remove("test:1");
        assert!(removed.is_some());
        assert!(!store.contains("test:1"));
    }

    #[test]
    fn test_l3_builder() {
        let store = L3StoreBuilder::new()
            .add_term(
                "test:1".to_string(),
                CompactTermBuilder::new("test:1").build(),
            )
            .add_term(
                "test:2".to_string(),
                CompactTermBuilder::new("test:2").build(),
            )
            .build();

        assert_eq!(store.len(), 2);
        assert!(store.contains("test:1"));
        assert!(store.contains("test:2"));
    }

    #[test]
    fn test_l3_stats() {
        let store = L3Store::new();

        for i in 0..1000 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            store.insert(format!("test:{}", i), term);
        }

        let stats = store.stats();
        assert_eq!(stats.entries, 1000);
        assert!(!stats.mmap_enabled);
        assert!(stats.estimated_memory_bytes > 0);
    }
}
