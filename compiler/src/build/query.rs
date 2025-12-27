//! Incremental computation system (Salsa-style query database).
//!
//! This module provides a query system for incremental compilation, where
//! computations are memoized and automatically recomputed when dependencies change.

use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Revision number for tracking changes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Revision(u64);

impl Revision {
    /// Initial revision
    pub const ZERO: Revision = Revision(0);

    /// Create a new revision
    pub fn new(n: u64) -> Self {
        Revision(n)
    }

    /// Increment revision
    pub fn increment(&mut self) {
        self.0 += 1;
    }

    /// Get next revision
    pub fn next(self) -> Self {
        Revision(self.0 + 1)
    }
}

/// Query key (identifies a specific query invocation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryKey {
    /// Query type name
    pub query_type: &'static str,

    /// Query input (serialized)
    pub input_hash: u64,
}

impl QueryKey {
    /// Create a new query key
    pub fn new<T: Hash>(query_type: &'static str, input: &T) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);

        QueryKey {
            query_type,
            input_hash: hasher.finish(),
        }
    }
}

/// Stored query result
#[derive(Clone)]
pub struct QueryResult {
    /// The computed value (type-erased)
    pub value: Arc<dyn Any + Send + Sync>,

    /// Revision when this was computed
    pub computed_at: Revision,

    /// Revisions of dependencies when this was computed
    pub dependencies: Vec<(QueryKey, Revision)>,

    /// Duration of computation in microseconds
    pub duration_us: u64,
}

impl QueryResult {
    /// Create a new query result
    pub fn new<T: Any + Send + Sync>(
        value: T,
        computed_at: Revision,
        dependencies: Vec<(QueryKey, Revision)>,
        duration_us: u64,
    ) -> Self {
        QueryResult {
            value: Arc::new(value),
            computed_at,
            dependencies,
            duration_us,
        }
    }

    /// Try to downcast the value
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.value.downcast_ref()
    }
}

/// Query database for incremental computation
pub struct QueryDb {
    /// Current revision
    current_revision: Revision,

    /// Stored query results
    cache: HashMap<QueryKey, QueryResult>,

    /// Input revisions (when inputs last changed)
    input_revisions: HashMap<QueryKey, Revision>,

    /// Statistics
    stats: QueryStats,
}

impl QueryDb {
    /// Create a new query database
    pub fn new() -> Self {
        QueryDb {
            current_revision: Revision::ZERO,
            cache: HashMap::new(),
            input_revisions: HashMap::new(),
            stats: QueryStats::default(),
        }
    }

    /// Get current revision
    pub fn current_revision(&self) -> Revision {
        self.current_revision
    }

    /// Bump the revision (call when inputs change)
    pub fn bump_revision(&mut self) -> Revision {
        self.current_revision.increment();
        self.current_revision
    }

    /// Mark an input as changed
    pub fn set_input_changed<T: Hash>(&mut self, query_type: &'static str, input: &T) {
        let key = QueryKey::new(query_type, input);
        self.input_revisions.insert(key, self.current_revision);
    }

    /// Check if a cached result is still valid
    fn is_valid(&self, key: &QueryKey, result: &QueryResult) -> bool {
        // Check if this key's input itself has changed
        if let Some(&input_revision) = self.input_revisions.get(key)
            && input_revision > result.computed_at
        {
            return false;
        }

        // Check if any dependency has changed since this was computed
        for (dep_key, dep_revision) in &result.dependencies {
            if let Some(&current_revision) = self.input_revisions.get(dep_key)
                && current_revision > *dep_revision
            {
                return false;
            }

            // Also check cached dependency results
            if let Some(cached) = self.cache.get(dep_key)
                && cached.computed_at > *dep_revision
            {
                return false;
            }
        }

        true
    }

    /// Execute a query with dependency tracking
    pub fn query<I, O, F>(&mut self, query_type: &'static str, input: &I, compute: F) -> Arc<O>
    where
        I: Hash + Clone,
        O: Any + Send + Sync + Clone,
        F: FnOnce(&mut QueryContext) -> O,
    {
        let key = QueryKey::new(query_type, input);

        // Check cache
        if let Some(cached) = self.cache.get(&key) {
            if self.is_valid(&key, cached) {
                self.stats.cache_hits += 1;

                if let Some(value) = cached.downcast_ref::<O>() {
                    return Arc::new(value.clone());
                }
            } else {
                self.stats.invalidations += 1;
            }
        }

        self.stats.cache_misses += 1;

        // Execute query with dependency tracking
        let mut ctx = QueryContext::new(self.current_revision);
        let start = std::time::Instant::now();
        let value = compute(&mut ctx);
        let duration_us = start.elapsed().as_micros() as u64;

        // Store result
        let result = QueryResult::new(
            value.clone(),
            self.current_revision,
            ctx.dependencies,
            duration_us,
        );

        self.cache.insert(key, result);

        Arc::new(value)
    }

    /// Get statistics
    pub fn stats(&self) -> &QueryStats {
        &self.stats
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.stats.cache_hits as f64 / total as f64
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats = QueryStats::default();
    }

    /// Get number of cached queries
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Garbage collect unused entries
    pub fn gc(&mut self, keep_recent: usize) {
        if self.cache.len() <= keep_recent {
            return;
        }

        let mut entries: Vec<_> = self.cache.iter().collect();
        entries.sort_by_key(|(_, result)| std::cmp::Reverse(result.computed_at));

        let to_keep: Vec<_> = entries
            .iter()
            .take(keep_recent)
            .map(|(key, _)| (*key).clone())
            .collect();

        self.cache.retain(|key, _| to_keep.contains(key));
    }
}

impl Default for QueryDb {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for tracking query dependencies
pub struct QueryContext {
    /// Current revision
    revision: Revision,

    /// Dependencies recorded during query execution
    dependencies: Vec<(QueryKey, Revision)>,
}

impl QueryContext {
    /// Create a new query context
    fn new(revision: Revision) -> Self {
        QueryContext {
            revision,
            dependencies: Vec::new(),
        }
    }

    /// Record a dependency on another query
    pub fn depends_on<T: Hash>(&mut self, query_type: &'static str, input: &T) {
        let key = QueryKey::new(query_type, input);
        self.dependencies.push((key, self.revision));
    }

    /// Get current revision
    pub fn revision(&self) -> Revision {
        self.revision
    }
}

/// Query database statistics
#[derive(Debug, Default, Clone)]
pub struct QueryStats {
    /// Number of cache hits
    pub cache_hits: u64,

    /// Number of cache misses
    pub cache_misses: u64,

    /// Number of invalidations
    pub invalidations: u64,
}

/// Example query types for the Sounio compiler
pub mod queries {
    use std::path::PathBuf;

    /// Input: source file path
    /// Output: parsed AST
    pub const PARSE: &str = "parse";

    /// Input: module path
    /// Output: resolved symbols
    pub const RESOLVE: &str = "resolve";

    /// Input: module path
    /// Output: type-checked HIR
    pub const TYPECHECK: &str = "typecheck";

    /// Input: module path
    /// Output: lowered HLIR
    pub const LOWER: &str = "lower";

    /// Input: module path
    /// Output: optimized HLIR
    pub const OPTIMIZE: &str = "optimize";

    /// Helper to create a file input key
    pub fn file_input(path: &PathBuf) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revision() {
        let mut rev = Revision::ZERO;
        assert_eq!(rev, Revision(0));

        rev.increment();
        assert_eq!(rev, Revision(1));

        let next = rev.next();
        assert_eq!(next, Revision(2));
    }

    #[test]
    fn test_query_key() {
        let key1 = QueryKey::new("test", &42);
        let key2 = QueryKey::new("test", &42);
        let key3 = QueryKey::new("test", &43);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_query_cache() {
        let mut db = QueryDb::new();

        let mut call_count = 0;

        // First call should execute
        let result1 = db.query("test", &42, |_ctx| {
            call_count += 1;
            100
        });
        assert_eq!(*result1, 100);
        assert_eq!(call_count, 1);
        assert_eq!(db.stats().cache_hits, 0);
        assert_eq!(db.stats().cache_misses, 1);

        // Second call should use cache
        let result2 = db.query("test", &42, |_ctx| {
            call_count += 1;
            200
        });
        assert_eq!(*result2, 100); // Still returns cached value
        assert_eq!(call_count, 1); // Not executed again
        assert_eq!(db.stats().cache_hits, 1);
    }

    #[test]
    fn test_query_invalidation() {
        let mut db = QueryDb::new();

        // First computation
        let result1 = db.query("test", &42, |_ctx| 100);
        assert_eq!(*result1, 100);

        // Mark input as changed
        db.bump_revision();
        db.set_input_changed("test", &42);

        // Should recompute
        let result2 = db.query("test", &42, |_ctx| 200);
        assert_eq!(*result2, 200);
        assert_eq!(db.stats().invalidations, 1);
    }

    #[test]
    fn test_query_dependencies() {
        let mut db = QueryDb::new();

        // Query A depends on query B
        let b_result = db.query("query_b", &1, |_ctx| 10);

        let a_result = db.query("query_a", &2, |ctx| {
            ctx.depends_on("query_b", &1);
            *b_result + 5
        });

        assert_eq!(*a_result, 15);

        // Change query B's input
        db.bump_revision();
        db.set_input_changed("query_b", &1);

        // Query A should be invalidated because it depends on B
        let new_b = db.query("query_b", &1, |_ctx| 20);
        let new_a = db.query("query_a", &2, |ctx| {
            ctx.depends_on("query_b", &1);
            *new_b + 5
        });

        assert_eq!(*new_a, 25);
    }

    #[test]
    fn test_gc() {
        let mut db = QueryDb::new();

        // Add many queries
        for i in 0..100 {
            db.query("test", &i, |_ctx| i * 2);
        }

        assert_eq!(db.len(), 100);

        // Garbage collect, keeping only 10 most recent
        db.gc(10);

        assert_eq!(db.len(), 10);
    }

    #[test]
    fn test_hit_rate() {
        let mut db = QueryDb::new();

        // 5 unique queries
        for i in 0..5 {
            db.query("test", &i, |_ctx| i);
        }

        // 5 cached queries
        for i in 0..5 {
            db.query("test", &i, |_ctx| i);
        }

        // Hit rate should be 50%
        assert_eq!(db.hit_rate(), 0.5);
    }
}
