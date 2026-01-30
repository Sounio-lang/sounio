//! Tiered Memory Model for Ontology Storage
//!
//! This module implements a three-tier memory hierarchy optimized for
//! handling 15M+ ontological types efficiently:
//!
//! - **L1 Cache**: Hot types in HashMap (~10K types, ~2MB)
//!   - Frequently accessed types (top of hierarchy, common diseases)
//!   - O(1) lookup, highest memory overhead per type
//!
//! - **L2 Cache**: Warm types in memory-mapped storage (~100K types, ~20MB)
//!   - Medium-frequency types
//!   - Compact representation, good locality
//!
//! - **L3 Store**: Cold types in memory-mapped file (~15M types, ~2GB)
//!   - Full ontology backing store
//!   - Minimal memory footprint, demand paging
//!
//! The tiered approach ensures that common medical types like "Diabetes"
//! or "Hypertension" are always fast to access, while rare types like
//! specific genetic variants incur minimal memory overhead.

pub mod compact;
pub mod l1_cache;
pub mod l2_cache;
pub mod l3_store;

pub use compact::{CompactTerm, CompactTermBuilder, TermFlags};
pub use l1_cache::L1Cache;
pub use l2_cache::L2Cache;
pub use l3_store::L3Store;

use std::sync::Arc;

/// Statistics for the tiered memory system
#[derive(Debug, Clone, Default)]
pub struct TieredMemoryStats {
    /// L1 cache statistics
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l1_size: usize,

    /// L2 cache statistics
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l2_size: usize,

    /// L3 store statistics
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub l3_size: usize,

    /// Promotion/demotion counts
    pub promotions_l3_to_l2: u64,
    pub promotions_l2_to_l1: u64,
    pub demotions_l1_to_l2: u64,
    pub demotions_l2_to_l3: u64,
}

impl TieredMemoryStats {
    /// Calculate overall hit rate
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits + self.l3_hits;
        let total_accesses = total_hits + self.l1_misses;
        if total_accesses == 0 {
            0.0
        } else {
            total_hits as f64 / total_accesses as f64
        }
    }

    /// Calculate L1 hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total == 0 {
            0.0
        } else {
            self.l1_hits as f64 / total as f64
        }
    }

    /// Estimate total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // L1: ~200 bytes per entry (HashMap overhead + full Term)
        // L2: ~80 bytes per entry (compact representation)
        // L3: ~64 bytes per entry (minimal compact)
        self.l1_size * 200 + self.l2_size * 80 + self.l3_size * 64
    }
}

impl std::fmt::Display for TieredMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tiered Memory Statistics:")?;
        writeln!(
            f,
            "  L1 Cache: {} entries, {:.1}% hit rate",
            self.l1_size,
            self.l1_hit_rate() * 100.0
        )?;
        writeln!(f, "  L2 Cache: {} entries", self.l2_size)?;
        writeln!(f, "  L3 Store: {} entries", self.l3_size)?;
        writeln!(f, "  Overall hit rate: {:.1}%", self.hit_rate() * 100.0)?;
        writeln!(
            f,
            "  Estimated memory: {:.1} MB",
            self.memory_usage() as f64 / (1024.0 * 1024.0)
        )?;
        Ok(())
    }
}

/// Configuration for the tiered memory system
#[derive(Debug, Clone)]
pub struct TieredMemoryConfig {
    /// Maximum entries in L1 cache
    pub l1_max_entries: usize,

    /// Maximum entries in L2 cache
    pub l2_max_entries: usize,

    /// Path to L3 backing file (if using file-backed storage)
    pub l3_backing_path: Option<std::path::PathBuf>,

    /// Whether to use memory mapping for L3
    pub l3_use_mmap: bool,

    /// Promotion threshold (access count before promoting to higher tier)
    pub promotion_threshold: u32,

    /// Enable statistics collection
    pub collect_stats: bool,
}

impl Default for TieredMemoryConfig {
    fn default() -> Self {
        TieredMemoryConfig {
            l1_max_entries: 10_000,
            l2_max_entries: 100_000,
            l3_backing_path: None,
            l3_use_mmap: true,
            promotion_threshold: 3,
            collect_stats: true,
        }
    }
}

/// Unified interface for the tiered memory system
pub struct TieredOntologyMemory {
    /// L1 hot cache
    l1: L1Cache,

    /// L2 warm cache
    l2: L2Cache,

    /// L3 cold store
    l3: L3Store,

    /// Configuration
    config: TieredMemoryConfig,

    /// Statistics
    stats: std::sync::RwLock<TieredMemoryStats>,
}

impl TieredOntologyMemory {
    /// Create a new tiered memory system with default config
    pub fn new() -> Self {
        Self::with_config(TieredMemoryConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: TieredMemoryConfig) -> Self {
        TieredOntologyMemory {
            l1: L1Cache::with_capacity(config.l1_max_entries),
            l2: L2Cache::with_capacity(config.l2_max_entries),
            l3: L3Store::new(),
            config,
            stats: std::sync::RwLock::new(TieredMemoryStats::default()),
        }
    }

    /// Look up a term by IRI, checking all tiers
    pub fn get(&self, iri: &str) -> Option<Arc<CompactTerm>> {
        // Try L1 first (hot cache)
        if let Some(term) = self.l1.get(iri) {
            if self.config.collect_stats
                && let Ok(mut stats) = self.stats.write()
            {
                stats.l1_hits += 1;
            }
            return Some(term);
        }

        // L1 miss
        if self.config.collect_stats
            && let Ok(mut stats) = self.stats.write()
        {
            stats.l1_misses += 1;
        }

        // Try L2 (warm cache)
        if let Some(term) = self.l2.get(iri) {
            if self.config.collect_stats
                && let Ok(mut stats) = self.stats.write()
            {
                stats.l2_hits += 1;
            }
            // Consider promotion to L1
            self.maybe_promote_to_l1(iri, &term);
            return Some(term);
        }

        if self.config.collect_stats
            && let Ok(mut stats) = self.stats.write()
        {
            stats.l2_misses += 1;
        }

        // Try L3 (cold store)
        if let Some(term) = self.l3.get(iri) {
            if self.config.collect_stats
                && let Ok(mut stats) = self.stats.write()
            {
                stats.l3_hits += 1;
            }
            // Promote to L2
            self.promote_to_l2(iri, &term);
            return Some(term);
        }

        if self.config.collect_stats
            && let Ok(mut stats) = self.stats.write()
        {
            stats.l3_misses += 1;
        }

        None
    }

    /// Insert a term, placing it in the appropriate tier based on expected access pattern
    pub fn insert(&self, iri: String, term: CompactTerm, hot: bool) {
        let term = Arc::new(term);

        if hot {
            // Hot terms go directly to L1
            if let Some(evicted) = self.l1.insert(iri.clone(), Arc::clone(&term)) {
                // Demote evicted entry to L2
                self.l2.insert(evicted.0, evicted.1);
                if self.config.collect_stats
                    && let Ok(mut stats) = self.stats.write()
                {
                    stats.demotions_l1_to_l2 += 1;
                }
            }
        } else {
            // Cold terms go to L3
            self.l3.insert(iri, term);
        }

        self.update_size_stats();
    }

    /// Batch insert terms (optimized for bulk loading)
    pub fn insert_batch(&self, terms: Vec<(String, CompactTerm, bool)>) {
        let mut hot_terms = Vec::new();
        let mut cold_terms = Vec::new();

        for (iri, term, hot) in terms {
            if hot {
                hot_terms.push((iri, Arc::new(term)));
            } else {
                cold_terms.push((iri, Arc::new(term)));
            }
        }

        // Bulk insert to L1
        for (iri, term) in hot_terms {
            if let Some(evicted) = self.l1.insert(iri, term) {
                self.l2.insert(evicted.0, evicted.1);
            }
        }

        // Bulk insert to L3
        self.l3.insert_batch(cold_terms);

        self.update_size_stats();
    }

    /// Check if a term exists (without updating access counts)
    pub fn contains(&self, iri: &str) -> bool {
        self.l1.contains(iri) || self.l2.contains(iri) || self.l3.contains(iri)
    }

    /// Get current statistics
    pub fn stats(&self) -> TieredMemoryStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Clear all caches (keeps L3 backing store)
    pub fn clear_caches(&self) {
        self.l1.clear();
        self.l2.clear();
        self.update_size_stats();
    }

    /// Prefetch common terms into L1/L2
    pub fn prefetch(&self, iris: &[&str]) {
        for iri in iris {
            // Just access each IRI to trigger caching
            let _ = self.get(iri);
        }
    }

    // Internal: maybe promote from L2 to L1
    fn maybe_promote_to_l1(&self, iri: &str, term: &Arc<CompactTerm>) {
        if self.l2.access_count(iri) >= self.config.promotion_threshold {
            if let Some(evicted) = self.l1.insert(iri.to_string(), Arc::clone(term)) {
                self.l2.insert(evicted.0, evicted.1);
            }
            self.l2.remove(iri);

            if self.config.collect_stats
                && let Ok(mut stats) = self.stats.write()
            {
                stats.promotions_l2_to_l1 += 1;
            }
        }
    }

    // Internal: promote from L3 to L2
    fn promote_to_l2(&self, iri: &str, term: &Arc<CompactTerm>) {
        if let Some(evicted) = self.l2.insert(iri.to_string(), Arc::clone(term)) {
            // L2 is full, demote to L3
            self.l3.insert(evicted.0, evicted.1);
            if self.config.collect_stats
                && let Ok(mut stats) = self.stats.write()
            {
                stats.demotions_l2_to_l3 += 1;
            }
        }

        if self.config.collect_stats
            && let Ok(mut stats) = self.stats.write()
        {
            stats.promotions_l3_to_l2 += 1;
        }
    }

    // Internal: update size statistics
    fn update_size_stats(&self) {
        if self.config.collect_stats
            && let Ok(mut stats) = self.stats.write()
        {
            stats.l1_size = self.l1.len();
            stats.l2_size = self.l2.len();
            stats.l3_size = self.l3.len();
        }
    }
}

impl Default for TieredOntologyMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiered_memory_basic() {
        let memory = TieredOntologyMemory::new();

        // Insert hot term
        let term = CompactTermBuilder::new("snomed:73211009")
            .with_label("Diabetes mellitus")
            .build();
        memory.insert("snomed:73211009".to_string(), term, true);

        // Should be in L1
        assert!(memory.l1.contains("snomed:73211009"));

        // Lookup should work
        let result = memory.get("snomed:73211009");
        assert!(result.is_some());

        let stats = memory.stats();
        assert_eq!(stats.l1_hits, 1);
    }

    #[test]
    fn test_tiered_memory_cold() {
        let memory = TieredOntologyMemory::new();

        // Insert cold term
        let term = CompactTermBuilder::new("snomed:rare123")
            .with_label("Rare condition")
            .build();
        memory.insert("snomed:rare123".to_string(), term, false);

        // Should be in L3
        assert!(memory.l3.contains("snomed:rare123"));
        assert!(!memory.l1.contains("snomed:rare123"));

        // Lookup should promote to L2
        let result = memory.get("snomed:rare123");
        assert!(result.is_some());

        let stats = memory.stats();
        assert_eq!(stats.l3_hits, 1);
        assert_eq!(stats.promotions_l3_to_l2, 1);
    }

    #[test]
    fn test_stats_display() {
        let stats = TieredMemoryStats {
            l1_hits: 100,
            l1_misses: 10,
            l1_size: 1000,
            l2_hits: 50,
            l2_misses: 5,
            l2_size: 10000,
            l3_hits: 20,
            l3_misses: 2,
            l3_size: 100000,
            ..Default::default()
        };

        let display = format!("{}", stats);
        assert!(display.contains("L1 Cache"));
        assert!(display.contains("hit rate"));
    }
}
