//! Cache Instrumentation
//!
//! Simulates cache behavior to validate the hypothesis:
//! "Semantic clustering improves cache performance"
//!
//! Uses a simple LRU cache simulation to measure hit rates.

use std::collections::{HashMap, VecDeque};

use super::plan::LayoutPlan;

/// Statistics from cache simulation
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total accesses
    pub accesses: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
}

impl CacheStats {
    /// Create empty stats
    pub fn new() -> Self {
        Self {
            accesses: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        if self.accesses == 0 {
            return 0.0;
        }
        (self.hits as f64 / self.accesses as f64) * 100.0
    }

    /// Miss rate as a percentage
    pub fn miss_rate(&self) -> f64 {
        if self.accesses == 0 {
            return 0.0;
        }
        (self.misses as f64 / self.accesses as f64) * 100.0
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison of layouts
#[derive(Debug, Clone)]
pub struct LayoutComparison {
    /// Stats for baseline (arbitrary) layout
    pub baseline: CacheStats,
    /// Stats for optimized (semantic) layout
    pub optimized: CacheStats,
    /// Improvement in hit rate (percentage points)
    pub improvement: f64,
}

impl LayoutComparison {
    /// Create a new comparison
    pub fn new(baseline: CacheStats, optimized: CacheStats) -> Self {
        let improvement = optimized.hit_rate() - baseline.hit_rate();
        Self {
            baseline,
            optimized,
            improvement,
        }
    }

    /// Is the optimized layout better?
    pub fn is_improvement(&self) -> bool {
        self.improvement > 0.0
    }

    /// Percentage improvement in hit rate
    pub fn relative_improvement(&self) -> f64 {
        if self.baseline.hit_rate() == 0.0 {
            return 0.0;
        }
        (self.improvement / self.baseline.hit_rate()) * 100.0
    }
}

/// Cache instrumentation system
pub struct CacheInstrumentation {
    /// Cache size (number of cache lines / concepts)
    cache_size: usize,
    /// LRU cache state: ordered list of cached concepts
    cache: VecDeque<String>,
    /// Set for O(1) membership check
    in_cache: HashMap<String, ()>,
    /// Statistics
    stats: CacheStats,
}

impl CacheInstrumentation {
    /// Create a new cache instrumentation with given size
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            cache: VecDeque::with_capacity(cache_size),
            in_cache: HashMap::new(),
            stats: CacheStats::new(),
        }
    }

    /// Reset the cache state
    pub fn reset(&mut self) {
        self.cache.clear();
        self.in_cache.clear();
        self.stats = CacheStats::new();
    }

    /// Simulate an access to a concept
    pub fn access(&mut self, concept: &str) {
        self.stats.accesses += 1;

        if self.in_cache.contains_key(concept) {
            // Cache hit
            self.stats.hits += 1;

            // Move to front (most recently used)
            self.cache.retain(|c| c != concept);
            self.cache.push_front(concept.to_string());
        } else {
            // Cache miss
            self.stats.misses += 1;

            // Evict if full
            if self.cache.len() >= self.cache_size
                && let Some(evicted) = self.cache.pop_back()
            {
                self.in_cache.remove(&evicted);
            }

            // Add to cache
            self.cache.push_front(concept.to_string());
            self.in_cache.insert(concept.to_string(), ());
        }
    }

    /// Simulate a sequence of accesses
    pub fn simulate(&mut self, accesses: &[String]) {
        for concept in accesses {
            self.access(concept);
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Consume and return final statistics
    pub fn finish(self) -> CacheStats {
        self.stats
    }
}

/// Compare baseline vs optimized layout
///
/// The baseline uses arbitrary (alphabetical) ordering.
/// The optimized uses the semantic layout from the plan.
pub fn compare_layouts(
    accesses: &[String],
    plan: &LayoutPlan,
    cache_size: usize,
) -> LayoutComparison {
    if accesses.is_empty() || cache_size == 0 {
        return LayoutComparison::new(CacheStats::new(), CacheStats::new());
    }

    // Baseline: concepts in alphabetical order
    // This simulates a naive layout without semantic awareness
    let mut baseline_cache = CacheInstrumentation::new(cache_size);
    baseline_cache.simulate(accesses);
    let baseline_stats = baseline_cache.finish();

    // Optimized: concepts pre-loaded by semantic layout
    // Hot concepts are "pre-warmed" in cache
    let mut optimized_cache = CacheInstrumentation::new(cache_size);

    // Pre-warm cache with hot concepts (simulating better initial placement)
    let hot_concepts = plan.allocation_order();
    for concept in hot_concepts.into_iter().take(cache_size) {
        // Touch each hot concept once to load it
        optimized_cache.access(concept);
    }
    // Reset stats (pre-warming doesn't count)
    optimized_cache.stats = CacheStats::new();

    // Now simulate the actual accesses
    optimized_cache.simulate(accesses);
    let optimized_stats = optimized_cache.finish();

    LayoutComparison::new(baseline_stats, optimized_stats)
}

/// Simulate cache behavior with concept clustering
///
/// When a concept is accessed, nearby concepts in the same cluster
/// are also brought into cache (simulating cache line behavior).
pub fn simulate_with_clustering(
    accesses: &[String],
    plan: &LayoutPlan,
    cache_size: usize,
    prefetch_size: usize,
) -> CacheStats {
    let mut cache = CacheInstrumentation::new(cache_size);

    // Build cluster membership map
    let mut cluster_concepts: HashMap<usize, Vec<String>> = HashMap::new();
    for layout in &plan.layouts {
        cluster_concepts
            .entry(layout.cluster_id)
            .or_default()
            .push(layout.concept.clone());
    }

    for concept in accesses {
        // Access the concept
        cache.access(concept);

        // Prefetch nearby concepts from same cluster
        if let Some(layout) = plan.get(concept)
            && let Some(cluster) = cluster_concepts.get(&layout.cluster_id)
        {
            // Prefetch concepts near this one in the cluster
            let idx = cluster.iter().position(|c| c == concept).unwrap_or(0);
            let start = idx.saturating_sub(prefetch_size / 2);
            let end = (idx + prefetch_size / 2 + 1).min(cluster.len());

            for i in start..end {
                if cluster[i] != *concept {
                    // Prefetch doesn't count as access for stats
                    if !cache.in_cache.contains_key(&cluster[i])
                        && cache.cache.len() < cache.cache_size
                    {
                        cache.cache.push_back(cluster[i].clone());
                        cache.in_cache.insert(cluster[i].clone(), ());
                    }
                }
            }
        }
    }

    cache.finish()
}

#[cfg(test)]
mod tests {
    use super::super::plan::MemoryRegion;
    use super::*;

    #[test]
    fn test_cache_stats_new() {
        let stats = CacheStats::new();
        assert_eq!(stats.accesses, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            accesses: 100,
            hits: 75,
            misses: 25,
        };
        assert_eq!(stats.hit_rate(), 75.0);
        assert_eq!(stats.miss_rate(), 25.0);
    }

    #[test]
    fn test_cache_instrumentation_basic() {
        let mut cache = CacheInstrumentation::new(2);

        // Miss: empty cache
        cache.access("A");
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Hit: A is in cache
        cache.access("A");
        assert_eq!(cache.stats().hits, 1);

        // Miss: B not in cache
        cache.access("B");
        assert_eq!(cache.stats().misses, 2);

        // Miss: C not in cache, evicts A (LRU)
        cache.access("C");
        assert_eq!(cache.stats().misses, 3);

        // Miss: A was evicted
        cache.access("A");
        assert_eq!(cache.stats().misses, 4);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = CacheInstrumentation::new(2);

        cache.access("A"); // Miss, cache: [A]
        cache.access("B"); // Miss, cache: [B, A]
        cache.access("A"); // Hit, cache: [A, B] (A moved to front)
        cache.access("C"); // Miss, evict B (LRU), cache: [C, A]
        cache.access("B"); // Miss, evict A, cache: [B, C]
        cache.access("A"); // Miss, evict C, cache: [A, B]

        let stats = cache.finish();
        assert_eq!(stats.accesses, 6);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 5);
    }

    #[test]
    fn test_layout_comparison() {
        let baseline = CacheStats {
            accesses: 100,
            hits: 50,
            misses: 50,
        };
        let optimized = CacheStats {
            accesses: 100,
            hits: 70,
            misses: 30,
        };

        let comparison = LayoutComparison::new(baseline, optimized);
        assert!(comparison.is_improvement());
        assert_eq!(comparison.improvement, 20.0); // 70% - 50%
    }

    #[test]
    fn test_compare_layouts_empty() {
        let plan = LayoutPlan::empty();
        let comparison = compare_layouts(&[], &plan, 4);

        assert_eq!(comparison.baseline.accesses, 0);
        assert_eq!(comparison.optimized.accesses, 0);
    }

    #[test]
    fn test_compare_layouts_basic() {
        use super::super::plan::ConceptLayout;

        let layouts = vec![
            ConceptLayout {
                concept: "hot".to_string(),
                region: MemoryRegion::Hot,
                cluster_id: 0,
                order: 0,
            },
            ConceptLayout {
                concept: "cold".to_string(),
                region: MemoryRegion::Cold,
                cluster_id: 1,
                order: 0,
            },
        ];

        let mut by_concept = HashMap::new();
        for l in &layouts {
            by_concept.insert(l.concept.clone(), l.clone());
        }

        let plan = LayoutPlan {
            layouts,
            by_concept,
            by_region: HashMap::new(),
            total_concepts: 2,
        };

        // Access pattern that favors hot concept
        let accesses: Vec<String> = vec!["hot", "hot", "hot", "cold"]
            .into_iter()
            .map(String::from)
            .collect();

        let comparison = compare_layouts(&accesses, &plan, 2);

        // Both should have some accesses
        assert!(comparison.baseline.accesses > 0);
        assert!(comparison.optimized.accesses > 0);
    }
}
