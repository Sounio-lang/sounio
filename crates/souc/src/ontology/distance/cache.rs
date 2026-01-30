//! Distance Cache for Hot-Path Type Checking
//!
//! During compilation, the same type pairs are checked repeatedly.
//! Caching distances dramatically improves compile times.
//!
//! # Architecture
//!
//! Two-tier cache:
//! - L1: Small (10K), fast, frequently accessed pairs
//! - L2: Large (100K), warm pairs promoted to L1 after repeated access

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache key for distance lookups (normalized for symmetry)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DistanceCacheKey {
    /// Lower IRI ID (normalized for symmetry)
    lo: u32,
    /// Higher IRI ID
    hi: u32,
}

impl DistanceCacheKey {
    /// Create a new cache key, normalizing order for symmetry
    pub fn new(a: u32, b: u32) -> Self {
        if a <= b {
            Self { lo: a, hi: b }
        } else {
            Self { lo: b, hi: a }
        }
    }

    /// Get the pair as a tuple
    pub fn as_pair(&self) -> (u32, u32) {
        (self.lo, self.hi)
    }
}

/// Cached distance value
#[derive(Debug, Clone, Copy)]
pub struct CachedDistance {
    /// The computed distance value [0, 1]
    pub value: f32,
    /// Last access timestamp (clock tick)
    pub last_access: u64,
    /// Number of times accessed
    pub access_count: u32,
}

impl CachedDistance {
    pub fn new(value: f32, clock: u64) -> Self {
        Self {
            value,
            last_access: clock,
            access_count: 1,
        }
    }
}

/// Statistics for cache performance monitoring
#[derive(Debug, Default)]
pub struct CacheStats {
    /// L1 cache hits
    pub l1_hits: AtomicU64,
    /// L2 cache hits
    pub l2_hits: AtomicU64,
    /// Total misses
    pub misses: AtomicU64,
    /// Total insertions
    pub insertions: AtomicU64,
    /// L1 evictions
    pub l1_evictions: AtomicU64,
    /// L2 evictions
    pub l2_evictions: AtomicU64,
    /// Promotions from L2 to L1
    pub promotions: AtomicU64,
}

impl CacheStats {
    /// Get overall hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.l1_hits.load(Ordering::Relaxed) + self.l2_hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get L1 hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let l1_hits = self.l1_hits.load(Ordering::Relaxed);
        let total =
            l1_hits + self.l2_hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            l1_hits as f64 / total as f64
        }
    }

    /// Get summary as string
    pub fn summary(&self) -> String {
        format!(
            "L1 hits: {}, L2 hits: {}, misses: {}, hit rate: {:.2}%, promotions: {}",
            self.l1_hits.load(Ordering::Relaxed),
            self.l2_hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.hit_rate() * 100.0,
            self.promotions.load(Ordering::Relaxed),
        )
    }
}

/// Single-tier LRU-like distance cache
pub struct DistanceCache {
    /// Cache storage
    cache: RwLock<HashMap<DistanceCacheKey, CachedDistance>>,
    /// Maximum entries
    capacity: usize,
    /// Eviction clock for approximate LRU
    clock: AtomicU64,
    /// Statistics
    stats: CacheStats,
}

impl DistanceCache {
    /// Create a new cache with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            clock: AtomicU64::new(0),
            stats: CacheStats::default(),
        }
    }

    /// Get cached distance
    pub fn get(&self, a: u32, b: u32) -> Option<f32> {
        let key = DistanceCacheKey::new(a, b);

        // First try read lock
        if let Ok(cache) = self.cache.read()
            && let Some(entry) = cache.get(&key)
        {
            self.stats.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.value);
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert distance into cache
    pub fn insert(&self, a: u32, b: u32, distance: f32) {
        let key = DistanceCacheKey::new(a, b);

        if let Ok(mut cache) = self.cache.write() {
            // Check capacity and evict if needed
            if cache.len() >= self.capacity {
                self.evict_oldest_inner(&mut cache);
            }

            let clock = self.clock.fetch_add(1, Ordering::Relaxed);
            let entry = CachedDistance::new(distance, clock);

            cache.insert(key, entry);
            self.stats.insertions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get or compute distance
    pub fn get_or_compute<F>(&self, a: u32, b: u32, compute: F) -> f32
    where
        F: FnOnce() -> f32,
    {
        if let Some(cached) = self.get(a, b) {
            return cached;
        }

        let distance = compute();
        self.insert(a, b, distance);
        distance
    }

    /// Evict oldest entries (approximate LRU) - internal version
    fn evict_oldest_inner(&self, cache: &mut HashMap<DistanceCacheKey, CachedDistance>) {
        let current_clock = self.clock.load(Ordering::Relaxed);
        let threshold = current_clock.saturating_sub(self.capacity as u64 / 2);

        let mut evicted = 0u64;
        cache.retain(|_, entry| {
            if entry.last_access < threshold {
                evicted += 1;
                false
            } else {
                true
            }
        });

        self.stats
            .l1_evictions
            .fetch_add(evicted, Ordering::Relaxed);
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Current cache size
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.read().map(|c| c.is_empty()).unwrap_or(true)
    }

    /// Memory usage estimate in bytes
    pub fn memory_bytes(&self) -> usize {
        self.len()
            * (std::mem::size_of::<DistanceCacheKey>() + std::mem::size_of::<CachedDistance>() + 32) // HashMap overhead estimate
    }
}

/// Two-tier distance cache: L1 (hot) + L2 (warm)
///
/// Implements a promotion policy where frequently accessed
/// items in L2 get promoted to L1.
pub struct TieredDistanceCache {
    /// L1: Small, fast, frequently accessed pairs
    l1: RwLock<HashMap<DistanceCacheKey, CachedDistance>>,
    /// L2: Larger, less frequently accessed pairs
    l2: RwLock<HashMap<DistanceCacheKey, CachedDistance>>,
    /// L1 capacity
    l1_capacity: usize,
    /// L2 capacity
    l2_capacity: usize,
    /// Promotion threshold (promote after N accesses in L2)
    promotion_threshold: u32,
    /// Eviction clock
    clock: AtomicU64,
    /// Statistics
    stats: CacheStats,
}

impl TieredDistanceCache {
    /// Create a new tiered cache
    pub fn new(l1_capacity: usize, l2_capacity: usize) -> Self {
        Self {
            l1: RwLock::new(HashMap::with_capacity(l1_capacity)),
            l2: RwLock::new(HashMap::with_capacity(l2_capacity)),
            l1_capacity,
            l2_capacity,
            promotion_threshold: 3,
            clock: AtomicU64::new(0),
            stats: CacheStats::default(),
        }
    }

    /// Create with custom promotion threshold
    pub fn with_promotion_threshold(mut self, threshold: u32) -> Self {
        self.promotion_threshold = threshold;
        self
    }

    /// Get cached distance
    pub fn get(&self, a: u32, b: u32) -> Option<f32> {
        let key = DistanceCacheKey::new(a, b);
        let clock = self.clock.fetch_add(1, Ordering::Relaxed);

        // Check L1 first
        if let Ok(mut l1) = self.l1.write()
            && let Some(entry) = l1.get_mut(&key)
        {
            entry.last_access = clock;
            entry.access_count = entry.access_count.saturating_add(1);
            self.stats.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.value);
        }

        // Check L2
        if let Ok(mut l2) = self.l2.write()
            && let Some(entry) = l2.get_mut(&key)
        {
            entry.last_access = clock;
            entry.access_count = entry.access_count.saturating_add(1);
            let value = entry.value;
            let count = entry.access_count;

            self.stats.l2_hits.fetch_add(1, Ordering::Relaxed);

            // Promote to L1 if hot enough
            if count >= self.promotion_threshold {
                l2.remove(&key);
                drop(l2); // Release L2 lock before acquiring L1
                self.promote_to_l1(key, value, clock);
            }

            return Some(value);
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert distance into cache (goes to L2 first)
    pub fn insert(&self, a: u32, b: u32, distance: f32) {
        let key = DistanceCacheKey::new(a, b);

        if let Ok(mut l2) = self.l2.write() {
            // Check capacity and evict if needed
            if l2.len() >= self.l2_capacity {
                self.evict_l2_inner(&mut l2);
            }

            let clock = self.clock.fetch_add(1, Ordering::Relaxed);
            let entry = CachedDistance::new(distance, clock);

            l2.insert(key, entry);
            self.stats.insertions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get or compute distance
    pub fn get_or_compute<F>(&self, a: u32, b: u32, compute: F) -> f32
    where
        F: FnOnce() -> f32,
    {
        if let Some(cached) = self.get(a, b) {
            return cached;
        }

        let distance = compute();
        self.insert(a, b, distance);
        distance
    }

    /// Promote an entry to L1
    fn promote_to_l1(&self, key: DistanceCacheKey, value: f32, clock: u64) {
        if let Ok(mut l1) = self.l1.write() {
            // Evict from L1 if needed
            if l1.len() >= self.l1_capacity {
                self.evict_l1_inner(&mut l1);
            }

            let entry = CachedDistance {
                value,
                last_access: clock,
                access_count: self.promotion_threshold,
            };
            l1.insert(key, entry);

            self.stats.promotions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Evict oldest entries from L1
    fn evict_l1_inner(&self, l1: &mut HashMap<DistanceCacheKey, CachedDistance>) {
        let current_clock = self.clock.load(Ordering::Relaxed);
        let threshold = current_clock.saturating_sub(self.l1_capacity as u64);

        let mut evicted = 0u64;
        l1.retain(|_, entry| {
            if entry.last_access < threshold {
                evicted += 1;
                false
            } else {
                true
            }
        });

        self.stats
            .l1_evictions
            .fetch_add(evicted, Ordering::Relaxed);
    }

    /// Evict oldest entries from L2
    fn evict_l2_inner(&self, l2: &mut HashMap<DistanceCacheKey, CachedDistance>) {
        let current_clock = self.clock.load(Ordering::Relaxed);
        let threshold = current_clock.saturating_sub(self.l2_capacity as u64 / 2);

        let mut evicted = 0u64;
        l2.retain(|_, entry| {
            if entry.last_access < threshold {
                evicted += 1;
                false
            } else {
                true
            }
        });

        self.stats
            .l2_evictions
            .fetch_add(evicted, Ordering::Relaxed);
    }

    /// Clear all caches
    pub fn clear(&self) {
        if let Ok(mut l1) = self.l1.write() {
            l1.clear();
        }
        if let Ok(mut l2) = self.l2.write() {
            l2.clear();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get tiered statistics
    pub fn tiered_stats(&self) -> TieredCacheStats {
        let l1_size = self.l1.read().map(|l| l.len()).unwrap_or(0);
        let l2_size = self.l2.read().map(|l| l.len()).unwrap_or(0);

        TieredCacheStats {
            l1_size,
            l2_size,
            l1_capacity: self.l1_capacity,
            l2_capacity: self.l2_capacity,
            l1_hit_rate: self.stats.l1_hit_rate(),
            overall_hit_rate: self.stats.hit_rate(),
            promotions: self.stats.promotions.load(Ordering::Relaxed),
        }
    }

    /// Total entries across both tiers
    pub fn len(&self) -> usize {
        let l1_len = self.l1.read().map(|l| l.len()).unwrap_or(0);
        let l2_len = self.l2.read().map(|l| l.len()).unwrap_or(0);
        l1_len + l2_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        let l1_empty = self.l1.read().map(|l| l.is_empty()).unwrap_or(true);
        let l2_empty = self.l2.read().map(|l| l.is_empty()).unwrap_or(true);
        l1_empty && l2_empty
    }

    /// Memory usage estimate
    pub fn memory_bytes(&self) -> usize {
        let entry_size =
            std::mem::size_of::<DistanceCacheKey>() + std::mem::size_of::<CachedDistance>() + 32;
        self.len() * entry_size
    }
}

/// Statistics for tiered cache
#[derive(Debug, Clone)]
pub struct TieredCacheStats {
    pub l1_size: usize,
    pub l2_size: usize,
    pub l1_capacity: usize,
    pub l2_capacity: usize,
    pub l1_hit_rate: f64,
    pub overall_hit_rate: f64,
    pub promotions: u64,
}

impl std::fmt::Display for TieredCacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "L1: {}/{} ({:.1}% hit), L2: {}/{}, overall: {:.1}% hit, {} promotions",
            self.l1_size,
            self.l1_capacity,
            self.l1_hit_rate * 100.0,
            self.l2_size,
            self.l2_capacity,
            self.overall_hit_rate * 100.0,
            self.promotions,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_symmetry() {
        let key1 = DistanceCacheKey::new(10, 20);
        let key2 = DistanceCacheKey::new(20, 10);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_basic_cache() {
        let cache = DistanceCache::new(100);

        cache.insert(1, 2, 0.5);
        assert_eq!(cache.get(1, 2), Some(0.5));
        assert_eq!(cache.get(2, 1), Some(0.5)); // Symmetric
        assert_eq!(cache.get(1, 3), None);
    }

    #[test]
    fn test_get_or_compute() {
        let cache = DistanceCache::new(100);

        let mut computed = false;
        let value = cache.get_or_compute(1, 2, || {
            computed = true;
            0.42
        });

        assert!(computed);
        assert_eq!(value, 0.42);

        // Second call should use cache
        computed = false;
        let value = cache.get_or_compute(1, 2, || {
            computed = true;
            0.99
        });

        assert!(!computed);
        assert_eq!(value, 0.42);
    }

    #[test]
    fn test_tiered_cache() {
        let cache = TieredDistanceCache::new(10, 100);

        // Insert goes to L2
        cache.insert(1, 2, 0.5);

        // First access from L2
        assert_eq!(cache.get(1, 2), Some(0.5));
        assert_eq!(cache.stats.l2_hits.load(Ordering::Relaxed), 1);

        // Access multiple times to trigger promotion
        for _ in 0..3 {
            cache.get(1, 2);
        }

        // Should now be in L1
        cache.get(1, 2);
        assert!(cache.stats.l1_hits.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = DistanceCache::new(10);

        // Fill cache
        for i in 0..20 {
            cache.insert(i, i + 100, 0.1 * i as f32);
        }

        // Cache should have evicted some entries
        assert!(cache.len() <= 10);
    }

    #[test]
    fn test_hit_rate() {
        let cache = DistanceCache::new(100);

        cache.insert(1, 2, 0.5);

        // 5 hits
        for _ in 0..5 {
            cache.get(1, 2);
        }

        // 5 misses
        for i in 0..5 {
            cache.get(i + 10, i + 20);
        }

        let hit_rate = cache.stats.hit_rate();
        assert!((hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tiered_stats() {
        let cache = TieredDistanceCache::new(5, 50);

        for i in 0..10 {
            cache.insert(i, i + 100, 0.1);
        }

        let stats = cache.tiered_stats();
        assert_eq!(stats.l2_size, 10);
        assert_eq!(stats.l1_size, 0); // Nothing promoted yet
    }
}
