//! LRU Cache for Ontology Terms
//!
//! Provides efficient caching for ontology term lookups across all layers.
//! Uses a tiered caching strategy:
//!
//! - Hot cache: Most recently accessed terms (small, fast)
//! - Warm cache: Frequently used terms (medium)
//! - Cold cache: Less frequently used terms (large, disk-backed optional)
//!
//! # Configuration
//!
//! ```rust,ignore
//! let config = CacheConfig::default()
//!     .with_max_entries(10000)
//!     .with_ttl_seconds(3600);
//!
//! let cache = OntologyCache::new(config);
//! ```

use std::time::{Duration, Instant};

use lru::LruCache;

use super::OntologyLayer;

/// Configuration for the ontology cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the hot cache
    pub hot_cache_size: usize,
    /// Maximum number of entries in the warm cache
    pub warm_cache_size: usize,
    /// Maximum number of entries in the cold cache
    pub cold_cache_size: usize,
    /// Time-to-live for cached entries (None = no expiry)
    pub ttl: Option<Duration>,
    /// Whether to cache negative lookups (term not found)
    pub cache_negatives: bool,
    /// Maximum size for negative cache
    pub negative_cache_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            hot_cache_size: 1000,
            warm_cache_size: 10000,
            cold_cache_size: 100000,
            ttl: Some(Duration::from_secs(3600)), // 1 hour default
            cache_negatives: true,
            negative_cache_size: 1000,
        }
    }
}

impl CacheConfig {
    /// Create a new configuration with custom max entries
    pub fn with_max_entries(mut self, entries: usize) -> Self {
        // Distribute entries across tiers (10%, 30%, 60%)
        self.hot_cache_size = entries / 10;
        self.warm_cache_size = (entries * 3) / 10;
        self.cold_cache_size = (entries * 6) / 10;
        self
    }

    /// Set TTL in seconds
    pub fn with_ttl_seconds(mut self, seconds: u64) -> Self {
        self.ttl = Some(Duration::from_secs(seconds));
        self
    }

    /// Disable TTL (entries never expire)
    pub fn no_ttl(mut self) -> Self {
        self.ttl = None;
        self
    }

    /// Enable or disable negative caching
    pub fn with_negative_caching(mut self, enabled: bool) -> Self {
        self.cache_negatives = enabled;
        self
    }

    /// Create a minimal cache for testing
    pub fn minimal() -> Self {
        Self {
            hot_cache_size: 10,
            warm_cache_size: 50,
            cold_cache_size: 100,
            ttl: Some(Duration::from_secs(60)),
            cache_negatives: false,
            negative_cache_size: 10,
        }
    }

    /// Create a large cache for production
    pub fn large() -> Self {
        Self {
            hot_cache_size: 10000,
            warm_cache_size: 100000,
            cold_cache_size: 1000000,
            ttl: Some(Duration::from_secs(86400)), // 24 hours
            cache_negatives: true,
            negative_cache_size: 10000,
        }
    }
}

/// A cached term entry
#[derive(Debug, Clone)]
pub struct CachedTerm {
    /// The resolved term data
    pub data: CachedTermData,
    /// When this entry was cached
    pub cached_at: Instant,
    /// Number of times this entry has been accessed
    pub access_count: usize,
    /// Which layer this term came from
    pub source_layer: OntologyLayer,
}

/// The actual cached data for a term
#[derive(Debug, Clone)]
pub struct CachedTermData {
    /// Full CURIE (e.g., "CHEBI:15365")
    pub curie: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Definition text
    pub definition: Option<String>,
    /// Direct superclass CURIEs
    pub superclasses: Vec<String>,
    /// Direct subclass CURIEs (limited)
    pub subclasses: Vec<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Cross-references to other ontologies
    pub xrefs: Vec<String>,
}

impl CachedTerm {
    /// Check if this entry has expired
    pub fn is_expired(&self, ttl: Option<Duration>) -> bool {
        match ttl {
            Some(ttl) => self.cached_at.elapsed() > ttl,
            None => false,
        }
    }

    /// Record an access
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }
}

/// Statistics about cache usage
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of lookups
    pub lookups: usize,
    /// Number of hot cache hits
    pub hot_hits: usize,
    /// Number of warm cache hits
    pub warm_hits: usize,
    /// Number of cold cache hits
    pub cold_hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of negative cache hits
    pub negative_hits: usize,
    /// Number of evictions
    pub evictions: usize,
    /// Number of expirations
    pub expirations: usize,
}

impl CacheStats {
    /// Get the hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        let hits = self.hot_hits + self.warm_hits + self.cold_hits;
        hits as f64 / self.lookups as f64
    }

    /// Get total hits
    pub fn total_hits(&self) -> usize {
        self.hot_hits + self.warm_hits + self.cold_hits
    }

    /// Get total misses
    pub fn total_misses(&self) -> usize {
        self.misses
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.lookups = 0;
        self.hot_hits = 0;
        self.warm_hits = 0;
        self.cold_hits = 0;
        self.misses = 0;
        self.negative_hits = 0;
        self.evictions = 0;
        self.expirations = 0;
    }
}

/// Tiered LRU cache for ontology terms
pub struct OntologyCache {
    /// Hot cache - most recently accessed
    hot: LruCache<String, CachedTerm>,
    /// Warm cache - frequently used
    warm: LruCache<String, CachedTerm>,
    /// Cold cache - less frequently used
    cold: LruCache<String, CachedTerm>,
    /// Negative cache - terms we know don't exist
    negative: LruCache<String, Instant>,
    /// Configuration
    config: CacheConfig,
    /// Statistics
    stats: CacheStats,
}

impl OntologyCache {
    /// Create a new cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        Self {
            hot: LruCache::new(std::num::NonZeroUsize::new(config.hot_cache_size.max(1)).unwrap()),
            warm: LruCache::new(
                std::num::NonZeroUsize::new(config.warm_cache_size.max(1)).unwrap(),
            ),
            cold: LruCache::new(
                std::num::NonZeroUsize::new(config.cold_cache_size.max(1)).unwrap(),
            ),
            negative: LruCache::new(
                std::num::NonZeroUsize::new(config.negative_cache_size.max(1)).unwrap(),
            ),
            config,
            stats: CacheStats::default(),
        }
    }

    /// Create a cache with default configuration
    pub fn default_cache() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get a term from the cache
    pub fn get(&mut self, curie: &str) -> Option<&CachedTerm> {
        self.stats.lookups += 1;

        // Check negative cache first
        if self.config.cache_negatives
            && let Some(cached_at) = self.negative.get(curie)
            && let Some(ttl) = self.config.ttl
            && cached_at.elapsed() <= ttl
        {
            self.stats.negative_hits += 1;
            return None;
        }

        // Check hot cache
        if let Some(entry) = self.hot.get_mut(curie) {
            if !entry.is_expired(self.config.ttl) {
                entry.record_access();
                self.stats.hot_hits += 1;
                return self.hot.get(curie);
            } else {
                self.stats.expirations += 1;
                self.hot.pop(curie);
            }
        }

        // Check warm cache and promote to hot if found
        if let Some(entry) = self.warm.pop(curie) {
            if !entry.is_expired(self.config.ttl) {
                self.stats.warm_hits += 1;
                let mut promoted = entry;
                promoted.record_access();
                // Cascade eviction when promoting to hot
                if let Some((evicted_key, evicted_entry)) =
                    self.hot.push(curie.to_string(), promoted)
                    && let Some((warm_evicted_key, warm_evicted_entry)) =
                        self.warm.push(evicted_key, evicted_entry)
                    && self
                        .cold
                        .push(warm_evicted_key, warm_evicted_entry)
                        .is_some()
                {
                    self.stats.evictions += 1;
                }
                return self.hot.get(curie);
            } else {
                self.stats.expirations += 1;
            }
        }

        // Check cold cache and promote to warm if found
        if let Some(entry) = self.cold.pop(curie) {
            if !entry.is_expired(self.config.ttl) {
                self.stats.cold_hits += 1;
                let mut promoted = entry;
                promoted.record_access();
                // Cascade eviction when promoting to warm
                if let Some((evicted_key, evicted_entry)) =
                    self.warm.push(curie.to_string(), promoted)
                    && self.cold.push(evicted_key, evicted_entry).is_some()
                {
                    self.stats.evictions += 1;
                }
                return self.warm.get(curie);
            } else {
                self.stats.expirations += 1;
            }
        }

        self.stats.misses += 1;
        None
    }

    /// Insert a term into the cache
    pub fn insert(&mut self, curie: String, term: CachedTermData, layer: OntologyLayer) {
        let entry = CachedTerm {
            data: term,
            cached_at: Instant::now(),
            access_count: 1,
            source_layer: layer,
        };

        // Remove from negative cache if present
        self.negative.pop(&curie);

        // Insert into hot cache with cascading eviction to warm/cold
        if let Some((evicted_key, evicted_entry)) = self.hot.push(curie, entry) {
            // Cascade evicted item to warm cache
            if let Some((warm_evicted_key, warm_evicted_entry)) =
                self.warm.push(evicted_key, evicted_entry)
            {
                // Cascade warm eviction to cold cache
                if self
                    .cold
                    .push(warm_evicted_key, warm_evicted_entry)
                    .is_some()
                {
                    self.stats.evictions += 1;
                }
            }
        }
    }

    /// Record a negative lookup (term doesn't exist)
    pub fn insert_negative(&mut self, curie: String) {
        if self.config.cache_negatives {
            self.negative.push(curie, Instant::now());
        }
    }

    /// Check if a term is known to not exist
    pub fn is_known_missing(&mut self, curie: &str) -> bool {
        if !self.config.cache_negatives {
            return false;
        }

        if let Some(cached_at) = self.negative.get(curie) {
            if let Some(ttl) = self.config.ttl {
                if cached_at.elapsed() <= ttl {
                    return true;
                }
            } else {
                return true;
            }
        }
        false
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.hot.clear();
        self.warm.clear();
        self.cold.clear();
        self.negative.clear();
        self.stats.reset();
    }

    /// Get total number of cached entries
    pub fn len(&self) -> usize {
        self.hot.len() + self.warm.len() + self.cold.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Demote entries from hot to warm, warm to cold
    ///
    /// This is called periodically to ensure the tiered structure is maintained.
    pub fn demote(&mut self) {
        // Demote least recently used from hot to warm
        while self.hot.len() > self.config.hot_cache_size {
            if let Some((key, value)) = self.hot.pop_lru()
                && self.warm.push(key, value).is_some()
            {
                self.stats.evictions += 1;
            }
        }

        // Demote least recently used from warm to cold
        while self.warm.len() > self.config.warm_cache_size {
            if let Some((key, value)) = self.warm.pop_lru()
                && self.cold.push(key, value).is_some()
            {
                self.stats.evictions += 1;
            }
        }
    }

    /// Evict expired entries from all caches
    pub fn evict_expired(&mut self) {
        if self.config.ttl.is_none() {
            return;
        }

        let ttl = self.config.ttl.unwrap();

        // Helper to collect expired keys
        fn collect_expired(cache: &LruCache<String, CachedTerm>, ttl: Duration) -> Vec<String> {
            cache
                .iter()
                .filter(|(_, v)| v.cached_at.elapsed() > ttl)
                .map(|(k, _)| k.clone())
                .collect()
        }

        // Evict from hot
        let expired: Vec<String> = collect_expired(&self.hot, ttl);
        for key in expired {
            self.hot.pop(&key);
            self.stats.expirations += 1;
        }

        // Evict from warm
        let expired: Vec<String> = collect_expired(&self.warm, ttl);
        for key in expired {
            self.warm.pop(&key);
            self.stats.expirations += 1;
        }

        // Evict from cold
        let expired: Vec<String> = collect_expired(&self.cold, ttl);
        for key in expired {
            self.cold.pop(&key);
            self.stats.expirations += 1;
        }
    }
}

/// Cache for subsumption (is-a) relationships
pub struct SubsumptionCache {
    /// Cache of (child, parent) -> is_subclass_of result
    cache: LruCache<(String, String), bool>,
    /// Statistics
    hits: usize,
    misses: usize,
}

impl SubsumptionCache {
    /// Create a new subsumption cache
    pub fn new(size: usize) -> Self {
        Self {
            cache: LruCache::new(std::num::NonZeroUsize::new(size.max(1)).unwrap()),
            hits: 0,
            misses: 0,
        }
    }

    /// Check if a subsumption relationship is cached
    pub fn get(&mut self, child: &str, parent: &str) -> Option<bool> {
        let key = (child.to_string(), parent.to_string());
        if let Some(&result) = self.cache.get(&key) {
            self.hits += 1;
            Some(result)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Cache a subsumption result
    pub fn insert(&mut self, child: &str, parent: &str, result: bool) {
        let key = (child.to_string(), parent.to_string());
        self.cache.push(key, result);
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = OntologyCache::new(CacheConfig::minimal());

        let term_data = CachedTermData {
            curie: "CHEBI:15365".to_string(),
            label: Some("aspirin".to_string()),
            definition: Some("A benzoic acid".to_string()),
            superclasses: vec!["CHEBI:22586".to_string()],
            subclasses: vec![],
            synonyms: vec!["acetylsalicylic acid".to_string()],
            xrefs: vec![],
        };

        cache.insert("CHEBI:15365".to_string(), term_data, OntologyLayer::Domain);

        let retrieved = cache.get("CHEBI:15365");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data.label, Some("aspirin".to_string()));
    }

    #[test]
    fn test_cache_promotion() {
        let config = CacheConfig {
            hot_cache_size: 2,
            warm_cache_size: 2,
            cold_cache_size: 2,
            ttl: None,
            cache_negatives: false,
            negative_cache_size: 1,
        };

        let mut cache = OntologyCache::new(config);

        // Insert 3 items, forcing promotion
        for i in 0..3 {
            let term_data = CachedTermData {
                curie: format!("TEST:{}", i),
                label: Some(format!("test{}", i)),
                definition: None,
                superclasses: vec![],
                subclasses: vec![],
                synonyms: vec![],
                xrefs: vec![],
            };
            cache.insert(format!("TEST:{}", i), term_data, OntologyLayer::Domain);
        }

        // All items should be retrievable
        assert!(cache.get("TEST:0").is_some());
        assert!(cache.get("TEST:1").is_some());
        assert!(cache.get("TEST:2").is_some());
    }

    #[test]
    fn test_negative_cache() {
        let config = CacheConfig {
            cache_negatives: true,
            negative_cache_size: 10,
            ..CacheConfig::minimal()
        };
        let mut cache = OntologyCache::new(config);

        // Record negative lookup
        cache.insert_negative("MISSING:123".to_string());

        // Should be known missing
        assert!(cache.is_known_missing("MISSING:123"));
        assert!(!cache.is_known_missing("OTHER:456"));
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = OntologyCache::new(CacheConfig::minimal());

        // Miss
        cache.get("NONEXISTENT:1");
        assert_eq!(cache.stats().total_misses(), 1);

        // Insert and hit
        let term_data = CachedTermData {
            curie: "TEST:1".to_string(),
            label: Some("test".to_string()),
            definition: None,
            superclasses: vec![],
            subclasses: vec![],
            synonyms: vec![],
            xrefs: vec![],
        };
        cache.insert("TEST:1".to_string(), term_data, OntologyLayer::Domain);
        cache.get("TEST:1");

        assert_eq!(cache.stats().total_hits(), 1);
        assert!(cache.stats().hit_rate() > 0.0);
    }

    #[test]
    fn test_subsumption_cache() {
        let mut cache = SubsumptionCache::new(100);

        // Miss
        assert!(cache.get("CHEBI:15365", "CHEBI:23888").is_none());

        // Insert
        cache.insert("CHEBI:15365", "CHEBI:23888", true);

        // Hit
        assert_eq!(cache.get("CHEBI:15365", "CHEBI:23888"), Some(true));
    }
}
