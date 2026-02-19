//! Incremental/query-based scaffold for MIR function optimization.
//!
//! This module provides a small P0 wrapper that memoizes optimization
//! results per function fingerprint + optimization level.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::build::query::{QueryDb, QueryStats};
use crate::mir::MirFunction;

use super::{OptimizationLevel, PassResult};

const QUERY_TYPE_MIR_OPTIMIZE_FUNCTION: &str = "mir_optimize_function_v1";

/// Cached value for a function optimization query.
#[derive(Debug, Clone)]
pub struct CachedOptimizationValue {
    /// Optimized function body.
    pub optimized_function: MirFunction,
    /// Pass result produced while optimizing the function.
    pub pass_result: PassResult,
}

/// Output returned for each optimization request.
#[derive(Debug, Clone)]
pub struct IncrementalOptimizationOutput {
    /// Optimized function body.
    pub optimized_function: MirFunction,
    /// Pass result produced while optimizing the function.
    pub pass_result: PassResult,
    /// Whether the result came from the query cache.
    pub cache_hit: bool,
}

/// Query key for incremental function optimization.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FunctionOptimizationKey {
    function_fingerprint: u64,
    opt_level_tag: u8,
    pipeline_epoch: u32,
}

impl FunctionOptimizationKey {
    fn new(func: &MirFunction, opt_level: OptimizationLevel, pipeline_epoch: u32) -> Self {
        Self {
            function_fingerprint: hash_function(func),
            opt_level_tag: optimization_level_tag(opt_level),
            pipeline_epoch,
        }
    }
}

/// P0 query-backed optimizer scaffold.
///
/// The cache key intentionally includes:
/// - function fingerprint (content + shape),
/// - optimization level (`O0..O3`),
/// - pipeline epoch (manual knob for pass-order/logic changes).
pub struct IncrementalFunctionOptimizer {
    query_db: QueryDb,
    pipeline_epoch: u32,
}

impl IncrementalFunctionOptimizer {
    /// Create a new optimizer with default pipeline epoch 1.
    pub fn new() -> Self {
        Self {
            query_db: QueryDb::new(),
            pipeline_epoch: 1,
        }
    }

    /// Create a new optimizer with a custom pipeline epoch.
    pub fn with_pipeline_epoch(pipeline_epoch: u32) -> Self {
        Self {
            query_db: QueryDb::new(),
            pipeline_epoch,
        }
    }

    /// Set the pipeline epoch used for future keys.
    pub fn set_pipeline_epoch(&mut self, pipeline_epoch: u32) {
        self.pipeline_epoch = pipeline_epoch;
    }

    /// Optimize a function with query-backed memoization.
    ///
    /// The closure runs only on cache miss.
    pub fn optimize_function<F>(
        &mut self,
        func: &MirFunction,
        opt_level: OptimizationLevel,
        run_passes: F,
    ) -> Result<IncrementalOptimizationOutput, String>
    where
        F: FnOnce(&mut MirFunction) -> Result<PassResult, String>,
    {
        let key = FunctionOptimizationKey::new(func, opt_level, self.pipeline_epoch);
        let hits_before = self.query_db.stats().cache_hits;
        let cached = self
            .query_db
            .query(QUERY_TYPE_MIR_OPTIMIZE_FUNCTION, &key, |_ctx| {
                let mut candidate = func.clone();
                let pass_result = run_passes(&mut candidate)?;
                Ok::<CachedOptimizationValue, String>(CachedOptimizationValue {
                    optimized_function: candidate,
                    pass_result,
                })
            });
        let cache_hit = self.query_db.stats().cache_hits > hits_before;

        let value = cached.as_ref().map_err(Clone::clone)?;

        Ok(IncrementalOptimizationOutput {
            optimized_function: value.optimized_function.clone(),
            pass_result: value.pass_result.clone(),
            cache_hit,
        })
    }

    /// Query statistics for cache-hit/miss observability.
    pub fn stats(&self) -> &QueryStats {
        self.query_db.stats()
    }

    /// Query cache hit-rate.
    pub fn hit_rate(&self) -> f64 {
        self.query_db.hit_rate()
    }

    /// Clear cached optimization results.
    pub fn clear(&mut self) {
        self.query_db.clear();
    }
}

impl Default for IncrementalFunctionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

fn hash_function(func: &MirFunction) -> u64 {
    let mut hasher = DefaultHasher::new();
    func.hash(&mut hasher);
    hasher.finish()
}

fn optimization_level_tag(level: OptimizationLevel) -> u8 {
    match level {
        OptimizationLevel::O0 => 0,
        OptimizationLevel::O1 => 1,
        OptimizationLevel::O2 => 2,
        OptimizationLevel::O3 => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{BlockId, FuncId, MirBlock, MirTerminator, MirType};

    fn pass_result(modified: bool) -> PassResult {
        PassResult {
            modified,
            passes_applied: Vec::new(),
            instructions_reduced: 0,
        }
    }

    fn sample_function(name: &str) -> MirFunction {
        let mut func = MirFunction::new(FuncId(0), name.to_string(), MirType::Unit);
        let mut entry = MirBlock::new(BlockId(0), "entry".to_string());
        entry.set_terminator(MirTerminator::Return { value: None });
        func.add_block(entry);
        func
    }

    #[test]
    fn reuses_cached_optimization_for_same_key() {
        let mut optimizer = IncrementalFunctionOptimizer::new();
        let func = sample_function("cached");
        let mut calls = 0usize;

        let first = optimizer
            .optimize_function(&func, OptimizationLevel::O2, |candidate| {
                calls += 1;
                candidate.name = format!("{}_optimized", candidate.name);
                Ok(pass_result(true))
            })
            .expect("first optimization should succeed");

        assert!(!first.cache_hit);
        assert_eq!(calls, 1);
        assert_eq!(first.optimized_function.name, "cached_optimized");

        let second = optimizer
            .optimize_function(&func, OptimizationLevel::O2, |candidate| {
                calls += 1;
                candidate.name = format!("{}_optimized_again", candidate.name);
                Ok(pass_result(true))
            })
            .expect("second optimization should succeed");

        assert!(second.cache_hit);
        assert_eq!(calls, 1);
        assert_eq!(second.optimized_function.name, "cached_optimized");
    }

    #[test]
    fn opt_level_changes_cache_key() {
        let mut optimizer = IncrementalFunctionOptimizer::new();
        let func = sample_function("level_sensitive");
        let mut calls = 0usize;

        let _ = optimizer
            .optimize_function(&func, OptimizationLevel::O1, |_candidate| {
                calls += 1;
                Ok(pass_result(false))
            })
            .expect("o1 optimization should succeed");

        let second = optimizer
            .optimize_function(&func, OptimizationLevel::O2, |_candidate| {
                calls += 1;
                Ok(pass_result(false))
            })
            .expect("o2 optimization should succeed");

        assert!(!second.cache_hit);
        assert_eq!(calls, 2);
    }

    #[test]
    fn pipeline_epoch_changes_cache_key() {
        let mut optimizer = IncrementalFunctionOptimizer::with_pipeline_epoch(1);
        let func = sample_function("epoch_sensitive");
        let mut calls = 0usize;

        let _ = optimizer
            .optimize_function(&func, OptimizationLevel::O2, |_candidate| {
                calls += 1;
                Ok(pass_result(false))
            })
            .expect("epoch-1 optimization should succeed");

        optimizer.set_pipeline_epoch(2);

        let second = optimizer
            .optimize_function(&func, OptimizationLevel::O2, |_candidate| {
                calls += 1;
                Ok(pass_result(false))
            })
            .expect("epoch-2 optimization should succeed");

        assert!(!second.cache_hit);
        assert_eq!(calls, 2);
    }
}
