//! L2 Cache - Warm Ontology Types
//!
//! The L2 cache holds medium-frequency ontological types in a more
//! compact representation than L1. It acts as a buffer between the
//! hot L1 cache and the cold L3 store.
//!
//! Capacity: ~100,000 types (~20MB)
//! Use case: Moderately common types, recently promoted from L3

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::compact::CompactTerm;

/// Entry in the L2 cache
#[derive(Debug)]
struct L2Entry {
    /// The cached term (compact representation)
    term: Arc<CompactTerm>,
    /// Access count for promotion decisions
    access_count: u32,
}

/// L2 Cache - Warm storage for medium-frequency types
pub struct L2Cache {
    /// The cache storage
    entries: RwLock<HashMap<String, L2Entry>>,
    /// Maximum number of entries
    capacity: usize,
    /// Access counts for promotion tracking (separate for lock granularity)
    access_tracker: RwLock<HashMap<String, u32>>,
}

impl L2Cache {
    /// Create a new L2 cache with default capacity (100,000)
    pub fn new() -> Self {
        Self::with_capacity(100_000)
    }

    /// Create with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        L2Cache {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            access_tracker: RwLock::new(HashMap::new()),
        }
    }

    /// Get a term from the cache
    pub fn get(&self, iri: &str) -> Option<Arc<CompactTerm>> {
        // Get term
        let term = {
            let entries = self.entries.read().ok()?;
            entries.get(iri).map(|e| Arc::clone(&e.term))
        };

        // Update access count
        if term.is_some()
            && let Ok(mut tracker) = self.access_tracker.write()
        {
            *tracker.entry(iri.to_string()).or_insert(0) += 1;
        }

        term
    }

    /// Insert a term, returning evicted entry if cache is full
    pub fn insert(
        &self,
        iri: String,
        term: Arc<CompactTerm>,
    ) -> Option<(String, Arc<CompactTerm>)> {
        let mut entries = self.entries.write().ok()?;

        // Check if we need to evict
        let evicted = if entries.len() >= self.capacity && !entries.contains_key(&iri) {
            self.evict_one(&mut entries)
        } else {
            None
        };

        let entry = L2Entry {
            term,
            access_count: 0,
        };

        entries.insert(iri.clone(), entry);

        // Initialize access tracker
        if let Ok(mut tracker) = self.access_tracker.write() {
            tracker.insert(iri, 0);
        }

        evicted
    }

    /// Check if IRI exists in cache
    pub fn contains(&self, iri: &str) -> bool {
        self.entries
            .read()
            .map(|e| e.contains_key(iri))
            .unwrap_or(false)
    }

    /// Remove an entry from the cache
    pub fn remove(&self, iri: &str) -> Option<Arc<CompactTerm>> {
        let term = self.entries.write().ok()?.remove(iri).map(|e| e.term);

        if term.is_some()
            && let Ok(mut tracker) = self.access_tracker.write()
        {
            tracker.remove(iri);
        }

        term
    }

    /// Get access count for an IRI (for promotion decisions)
    pub fn access_count(&self, iri: &str) -> u32 {
        self.access_tracker
            .read()
            .ok()
            .and_then(|t| t.get(iri).copied())
            .unwrap_or(0)
    }

    /// Get current number of entries
    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
        if let Ok(mut tracker) = self.access_tracker.write() {
            tracker.clear();
        }
    }

    /// Evict one entry (FIFO-like with access count consideration)
    fn evict_one(
        &self,
        entries: &mut HashMap<String, L2Entry>,
    ) -> Option<(String, Arc<CompactTerm>)> {
        // Get access counts
        let tracker = self.access_tracker.read().ok()?;

        // Find entry with lowest access count
        let victim = entries
            .iter()
            .min_by_key(|(k, _)| tracker.get(*k).copied().unwrap_or(0))
            .map(|(k, _)| k.clone());

        drop(tracker);

        victim.and_then(|k| {
            if let Ok(mut tracker) = self.access_tracker.write() {
                tracker.remove(&k);
            }
            entries.remove(&k).map(|e| (k, e.term))
        })
    }

    /// Get cache statistics
    pub fn stats(&self) -> L2CacheStats {
        let len = self.len();

        let total_access = self
            .access_tracker
            .read()
            .map(|t| t.values().map(|&v| v as u64).sum())
            .unwrap_or(0);

        L2CacheStats {
            entries: len,
            capacity: self.capacity,
            total_accesses: total_access,
            utilization: len as f64 / self.capacity as f64,
        }
    }

    /// Iterate over all entries (for debugging/diagnostics)
    pub fn entries(&self) -> Vec<(String, Arc<CompactTerm>)> {
        self.entries
            .read()
            .map(|e| {
                e.iter()
                    .map(|(k, v)| (k.clone(), Arc::clone(&v.term)))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for L2Cache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for L2 cache
#[derive(Debug, Clone)]
pub struct L2CacheStats {
    /// Current number of entries
    pub entries: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Total accesses
    pub total_accesses: u64,
    /// Cache utilization
    pub utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::super::compact::CompactTermBuilder;
    use super::*;

    #[test]
    fn test_l2_basic_operations() {
        let cache = L2Cache::with_capacity(100);

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        cache.insert("test:1".to_string(), term);

        assert!(cache.contains("test:1"));
        assert!(!cache.contains("test:2"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_l2_access_tracking() {
        let cache = L2Cache::with_capacity(100);

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        cache.insert("test:1".to_string(), term);

        assert_eq!(cache.access_count("test:1"), 0);

        cache.get("test:1");
        cache.get("test:1");

        assert_eq!(cache.access_count("test:1"), 2);
    }

    #[test]
    fn test_l2_eviction() {
        let cache = L2Cache::with_capacity(3);

        // Fill cache
        for i in 0..3 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            cache.insert(format!("test:{}", i), term);
        }

        // Access test:1 and test:2
        cache.get("test:1");
        cache.get("test:2");

        // Insert new entry, should evict test:0
        let term = Arc::new(CompactTermBuilder::new("test:3").build());
        let evicted = cache.insert("test:3".to_string(), term);

        assert!(evicted.is_some());
        assert!(!cache.contains("test:0"));
        assert!(cache.contains("test:3"));
    }

    #[test]
    fn test_l2_remove() {
        let cache = L2Cache::with_capacity(100);

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        cache.insert("test:1".to_string(), term);

        let removed = cache.remove("test:1");
        assert!(removed.is_some());
        assert!(!cache.contains("test:1"));
        assert_eq!(cache.access_count("test:1"), 0);
    }
}
