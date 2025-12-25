//! L1 Cache - Hot Ontology Types
//!
//! The L1 cache holds the most frequently accessed ontological types
//! in a HashMap for O(1) lookup. This is the fastest tier but has
//! the highest memory overhead per entry.
//!
//! Capacity: ~10,000 types (~2MB)
//! Use case: Common medical types (Diabetes, Hypertension, etc.)

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::compact::CompactTerm;

/// Entry in the L1 cache with access metadata
#[derive(Debug)]
struct L1Entry {
    /// The cached term
    term: Arc<CompactTerm>,
    /// Access count for LRU-like eviction
    access_count: u32,
    /// Last access timestamp (for aging)
    last_access: std::time::Instant,
}

/// L1 Cache - HashMap-based hot cache for frequently accessed types
pub struct L1Cache {
    /// The cache storage
    entries: RwLock<HashMap<String, L1Entry>>,
    /// Maximum number of entries
    capacity: usize,
}

impl L1Cache {
    /// Create a new L1 cache with default capacity (10,000)
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        L1Cache {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
        }
    }

    /// Get a term from the cache, updating access metadata
    pub fn get(&self, iri: &str) -> Option<Arc<CompactTerm>> {
        // Always use write lock to update access count
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(iri)
        {
            entry.access_count = entry.access_count.saturating_add(1);
            entry.last_access = std::time::Instant::now();
            return Some(Arc::clone(&entry.term));
        }

        None
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

        let entry = L1Entry {
            term,
            access_count: 1,
            last_access: std::time::Instant::now(),
        };

        entries.insert(iri, entry);
        evicted
    }

    /// Check if IRI exists in cache (without updating access count)
    pub fn contains(&self, iri: &str) -> bool {
        self.entries
            .read()
            .map(|e| e.contains_key(iri))
            .unwrap_or(false)
    }

    /// Remove an entry from the cache
    pub fn remove(&self, iri: &str) -> Option<Arc<CompactTerm>> {
        self.entries.write().ok()?.remove(iri).map(|e| e.term)
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
    }

    /// Get access count for an IRI
    pub fn access_count(&self, iri: &str) -> u32 {
        self.entries
            .read()
            .ok()
            .and_then(|e| e.get(iri).map(|entry| entry.access_count))
            .unwrap_or(0)
    }

    /// Evict one entry (LRU-based)
    fn evict_one(
        &self,
        entries: &mut HashMap<String, L1Entry>,
    ) -> Option<(String, Arc<CompactTerm>)> {
        // Find entry with lowest access count, oldest access time
        let victim = entries
            .iter()
            .min_by(|a, b| {
                // Primary: access count
                // Secondary: last access time (older = evict first)
                a.1.access_count
                    .cmp(&b.1.access_count)
                    .then_with(|| a.1.last_access.cmp(&b.1.last_access))
            })
            .map(|(k, _)| k.clone());

        victim.and_then(|k| entries.remove(&k).map(|e| (k, e.term)))
    }

    /// Get cache statistics
    pub fn stats(&self) -> L1CacheStats {
        let entries = self.entries.read().ok();

        let (len, total_access, max_access) = entries
            .map(|e| {
                let total: u64 = e.values().map(|v| v.access_count as u64).sum();
                let max = e.values().map(|v| v.access_count).max().unwrap_or(0);
                (e.len(), total, max)
            })
            .unwrap_or((0, 0, 0));

        L1CacheStats {
            entries: len,
            capacity: self.capacity,
            total_accesses: total_access,
            max_access_count: max_access,
            utilization: len as f64 / self.capacity as f64,
        }
    }
}

impl Default for L1Cache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for L1 cache
#[derive(Debug, Clone)]
pub struct L1CacheStats {
    /// Current number of entries
    pub entries: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Total accesses across all entries
    pub total_accesses: u64,
    /// Highest access count for any entry
    pub max_access_count: u32,
    /// Cache utilization (0.0 to 1.0)
    pub utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::super::compact::CompactTermBuilder;
    use super::*;

    #[test]
    fn test_l1_basic_operations() {
        let cache = L1Cache::with_capacity(100);

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        cache.insert("test:1".to_string(), term);

        assert!(cache.contains("test:1"));
        assert!(!cache.contains("test:2"));
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get("test:1");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_l1_access_counting() {
        let cache = L1Cache::with_capacity(100);

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        cache.insert("test:1".to_string(), term);

        assert_eq!(cache.access_count("test:1"), 1);

        cache.get("test:1");
        cache.get("test:1");
        cache.get("test:1");

        assert_eq!(cache.access_count("test:1"), 4); // 1 initial + 3 gets
    }

    #[test]
    fn test_l1_eviction() {
        let cache = L1Cache::with_capacity(3);

        // Fill cache
        for i in 0..3 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            cache.insert(format!("test:{}", i), term);
        }

        // Access test:1 and test:2 to increase their counts
        cache.get("test:1");
        cache.get("test:2");

        // Insert new entry, should evict test:0 (lowest access count)
        let term = Arc::new(CompactTermBuilder::new("test:3").build());
        let evicted = cache.insert("test:3".to_string(), term);

        assert!(evicted.is_some());
        let (evicted_iri, _) = evicted.unwrap();
        assert_eq!(evicted_iri, "test:0");

        // Verify test:0 is gone
        assert!(!cache.contains("test:0"));
        assert!(cache.contains("test:3"));
    }

    #[test]
    fn test_l1_remove() {
        let cache = L1Cache::with_capacity(100);

        let term = Arc::new(CompactTermBuilder::new("test:1").build());
        cache.insert("test:1".to_string(), term);

        let removed = cache.remove("test:1");
        assert!(removed.is_some());
        assert!(!cache.contains("test:1"));
    }

    #[test]
    fn test_l1_stats() {
        let cache = L1Cache::with_capacity(100);

        for i in 0..50 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            cache.insert(format!("test:{}", i), term);
        }

        // Access some entries multiple times
        for _ in 0..10 {
            cache.get("test:0");
        }

        let stats = cache.stats();
        assert_eq!(stats.entries, 50);
        assert_eq!(stats.capacity, 100);
        assert!(stats.utilization > 0.49 && stats.utilization < 0.51);
        assert_eq!(stats.max_access_count, 11); // 1 initial + 10 gets
    }
}
