//! Build system, caching, and incremental compilation.
//!
//! This module provides the core build system for the Sounio compiler,
//! including dependency tracking, incremental compilation, artifact caching,
//! and parallel execution.
//!
//! # Architecture
//!
//! The build system consists of several components:
//!
//! - **Build Graph** (`graph`): Tracks compilation units and their dependencies
//! - **Change Detection** (`change`): Monitors source files for modifications
//! - **Artifact Cache** (`cache`): Content-addressed storage for compiled artifacts
//! - **Query System** (`query`): Salsa-style incremental computation
//! - **Parallel Executor** (`parallel`): Work-stealing parallel compilation
//! - **Configuration** (`config`): Build profiles and compiler flags
//!
//! # Example
//!
//! ```no_run
//! use sounio::build::{BuildGraph, BuildConfig, ParallelExecutor};
//! use std::path::PathBuf;
//!
//! // Create build graph
//! let mut graph = BuildGraph::new();
//!
//! // Configure build
//! let config = BuildConfig::release();
//!
//! // Execute parallel build
//! let mut executor = ParallelExecutor::from_cpu_count();
//! ```

pub mod cache;
pub mod change;
pub mod config;
pub mod graph;
pub mod parallel;
pub mod query;

// Re-export commonly used types
pub use cache::{ArtifactCache, CacheConfig, CacheEntry, CacheKey, EvictionPolicy};
pub use change::{ChangeDetector, FileState};
pub use config::{BuildConfig, BuildProfile, CompilerFlags, FeatureConfig, OptLevel, TargetTriple};
pub use graph::{BuildGraph, CompilationUnit, ContentHash, UnitId};
pub use parallel::{BuildTask, JobServer, ParallelExecutor};
pub use query::{QueryDb, QueryKey, Revision};

// Re-export the build manager
// (BuildManager is defined in this module below, so no need to use path)

use std::path::Path;
use std::time::Instant;

/// High-level build manager that orchestrates the entire build process
pub struct BuildManager {
    /// Build configuration
    config: BuildConfig,

    /// Dependency graph
    graph: BuildGraph,

    /// Change detector
    change_detector: ChangeDetector,

    /// Artifact cache
    cache: ArtifactCache,

    /// Query database
    query_db: QueryDb,

    /// Parallel executor
    executor: ParallelExecutor,
}

impl BuildManager {
    /// Create a new build manager
    pub fn new(config: BuildConfig) -> Result<Self, BuildError> {
        let cache_dir = config.paths.cache_dir.clone();
        let mut cache = ArtifactCache::new(cache_dir);
        cache.init()?;

        let executor = ParallelExecutor::from_cpu_count();

        Ok(BuildManager {
            config,
            graph: BuildGraph::new(),
            change_detector: ChangeDetector::new(),
            cache,
            query_db: QueryDb::new(),
            executor,
        })
    }

    /// Initialize the build (scan sources, load cache)
    pub fn init(&mut self, source_dir: &Path) -> Result<(), BuildError> {
        // Try to load existing graph
        let graph_path = self.config.paths.build_dir.join("graph.bin");
        if graph_path.exists() {
            match BuildGraph::load(&graph_path) {
                Ok(loaded_graph) => {
                    self.graph = loaded_graph;
                    if self.config.flags.verbose {
                        println!("Loaded build graph with {} units", self.graph.len());
                    }
                }
                Err(e) => {
                    if self.config.flags.verbose {
                        println!("Could not load build graph: {}, starting fresh", e);
                    }
                }
            }
        }

        // Scan for source files
        let file_count = self.change_detector.scan_directory(source_dir)?;

        if self.config.flags.verbose {
            println!("Found {} source files", file_count);
        }

        // Create CompilationUnits for any new files not in the graph
        let tracked_files = self.change_detector.tracked_files();
        let mut new_units = 0;

        for path in tracked_files {
            if self.graph.get_unit_id(&path).is_none() {
                // New file - create a CompilationUnit
                if let Ok(content_hash) = ContentHash::from_file(&path) {
                    let unit = CompilationUnit::new(path.clone(), content_hash);
                    let unit_id = unit.id;
                    self.graph.add_unit(unit);

                    // Mark as root if it's a main module or has no dependencies yet
                    // (dependencies will be resolved during actual compilation)
                    self.graph.add_root(unit_id);
                    new_units += 1;
                }
            }
        }

        if self.config.flags.verbose && new_units > 0 {
            println!("Added {} new compilation units", new_units);
        }

        // Prune units for files that no longer exist
        self.graph.prune_missing();

        Ok(())
    }

    /// Perform an incremental build
    pub fn build(&mut self) -> Result<BuildReport, BuildError> {
        let start = Instant::now();

        // Check for changes
        let changed = self.change_detector.check_changes()?;

        if !changed.is_empty() {
            if self.config.flags.verbose {
                println!("Detected {} changed files", changed.len());
            }

            // Apply changes to graph
            self.change_detector
                .apply_changes(&mut self.graph, &changed);

            // Bump query revision
            self.query_db.bump_revision();
        }

        // Get units to compile
        let dirty_units = self.graph.dirty_units();

        if dirty_units.is_empty() {
            return Ok(BuildReport {
                duration: start.elapsed(),
                units_compiled: 0,
                cache_hits: 0,
                cache_misses: 0,
                success: true,
            });
        }

        if self.config.flags.verbose {
            println!("Compiling {} units", dirty_units.len());
        }

        // Collect unit info before compilation (to avoid borrow conflicts)
        let units_to_compile: Vec<(UnitId, std::path::PathBuf)> = dirty_units
            .iter()
            .filter_map(|&id| self.graph.get_unit(id).map(|u| (id, u.path.clone())))
            .collect();

        let verbose = self.config.flags.verbose;

        // Compile each unit using the actual Sounio compiler
        let mut results = Vec::new();
        for (unit_id, path) in &units_to_compile {
            let compile_result = Self::compile_unit_file(path, verbose);
            results.push(parallel::BuildResult {
                unit_id: *unit_id,
                success: compile_result.is_ok(),
                duration: std::time::Duration::ZERO,
                error: compile_result.err(),
            });
        }

        // Process results
        let mut compiled = 0;
        let mut failed = 0;

        for result in &results {
            if result.success {
                self.graph.mark_clean(result.unit_id);
                compiled += 1;
            } else {
                failed += 1;
                if let Some(error) = &result.error {
                    eprintln!("Error compiling {:?}: {}", result.unit_id, error);
                }
            }
        }

        // Get cache stats before save
        let cache_hits = self.cache.stats().hits;
        let cache_misses = self.cache.stats().misses;

        // Save cache and graph
        self.cache.save()?;
        self.graph
            .save(&self.config.paths.build_dir.join("graph.bin"))?;

        Ok(BuildReport {
            duration: start.elapsed(),
            units_compiled: compiled,
            cache_hits,
            cache_misses,
            success: failed == 0,
        })
    }

    /// Compile a single unit (placeholder for future implementation)
    /// Compile a single source file
    fn compile_unit_file(path: &Path, verbose: bool) -> Result<(), String> {
        // Read source file
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        // Compile using the Sounio compiler pipeline
        // For now, just do check (type-checking) - full compilation would use compile()
        let tokens = crate::lexer::lex(&source)
            .map_err(|e| format!("Lexer error in {}: {}", path.display(), e))?;
        let ast = crate::parser::parse(&tokens, &source)
            .map_err(|e| format!("Parser error in {}: {}", path.display(), e))?;
        let _hir = crate::check::check(&ast)
            .map_err(|e| format!("Type error in {}: {}", path.display(), e))?;

        if verbose {
            println!("  Compiled: {}", path.display());
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn compile_unit(&mut self, unit_id: UnitId) -> Result<(), String> {
        let unit = self
            .graph
            .get_unit(unit_id)
            .ok_or_else(|| format!("Unit {:?} not found", unit_id))?;

        // Check cache
        let cache_key = CacheKey::new(
            unit.content_hash.clone(),
            env!("CARGO_PKG_VERSION").to_string(),
            self.config.hash(),
        );

        if let Some(_cached) = self.cache.get(&cache_key) {
            // Cache hit - no compilation needed
            if self.config.flags.verbose {
                println!("  Cached: {}", unit.path.display());
            }
            return Ok(());
        }

        // Compile the unit using the actual compiler pipeline
        let path = unit.path.clone();
        let verbose = self.config.flags.verbose;
        Self::compile_unit_file(&path, verbose)?;

        // Store compilation result in cache
        // For now, we store a simple marker since we're only doing type-checking
        let metadata = cache::CacheMetadata::default();
        let _ = self
            .cache
            .put(cache_key, path.as_os_str().as_encoded_bytes(), metadata);

        // Mark unit as clean
        self.graph.mark_clean(unit_id);

        Ok(())
    }

    /// Clean build artifacts
    pub fn clean(&mut self) -> Result<(), BuildError> {
        self.cache.clear()?;
        self.graph = BuildGraph::new();
        self.query_db.clear();

        Ok(())
    }

    /// Get build statistics
    pub fn stats(&self) -> BuildStats {
        BuildStats {
            total_units: self.graph.len(),
            dirty_units: self.graph.dirty_units().len(),
            cache_size: self.cache.size(),
            cache_entries: self.cache.len(),
            cache_hit_rate: self.cache.hit_rate(),
            query_hit_rate: self.query_db.hit_rate(),
        }
    }
}

/// Build report
#[derive(Debug, Clone)]
pub struct BuildReport {
    /// Total build duration
    pub duration: std::time::Duration,

    /// Number of units compiled
    pub units_compiled: usize,

    /// Cache hits
    pub cache_hits: u64,

    /// Cache misses
    pub cache_misses: u64,

    /// Build succeeded
    pub success: bool,
}

impl BuildReport {
    /// Print a summary of the build
    pub fn print_summary(&self) {
        println!();
        println!("Build completed in {:.2}s", self.duration.as_secs_f64());
        println!("  Units compiled: {}", self.units_compiled);
        println!("  Cache hits: {}", self.cache_hits);
        println!("  Cache misses: {}", self.cache_misses);

        if self.cache_hits + self.cache_misses > 0 {
            let hit_rate = self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64;
            println!("  Cache hit rate: {:.1}%", hit_rate * 100.0);
        }

        if self.success {
            println!("  Status: SUCCESS");
        } else {
            println!("  Status: FAILED");
        }
    }
}

/// Build statistics
#[derive(Debug, Clone)]
pub struct BuildStats {
    /// Total compilation units
    pub total_units: usize,

    /// Dirty units needing recompilation
    pub dirty_units: usize,

    /// Cache size in bytes
    pub cache_size: u64,

    /// Number of cached entries
    pub cache_entries: usize,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Query database hit rate
    pub query_hit_rate: f64,
}

/// Build errors
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("Graph error: {0}")]
    GraphError(#[from] graph::BuildGraphError),

    #[error("Change detection error: {0}")]
    ChangeError(#[from] change::ChangeDetectorError),

    #[error("Cache error: {0}")]
    CacheError(#[from] cache::CacheError),

    #[error("Parallel execution error: {0}")]
    ParallelError(#[from] parallel::ParallelError),

    #[error("Configuration error: {0}")]
    ConfigError(#[from] config::ConfigError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_build_manager_creation() {
        let config = BuildConfig::dev();
        let _manager = BuildManager::new(config);
        // Just ensure it compiles and constructs
    }

    #[test]
    fn test_build_report() {
        let report = BuildReport {
            duration: std::time::Duration::from_secs(5),
            units_compiled: 10,
            cache_hits: 5,
            cache_misses: 5,
            success: true,
        };

        assert_eq!(report.units_compiled, 10);
        assert!(report.success);
    }

    #[test]
    fn test_build_stats() {
        let stats = BuildStats {
            total_units: 100,
            dirty_units: 10,
            cache_size: 1024 * 1024,
            cache_entries: 50,
            cache_hit_rate: 0.8,
            query_hit_rate: 0.9,
        };

        assert_eq!(stats.total_units, 100);
        assert_eq!(stats.dirty_units, 10);
        assert_eq!(stats.cache_hit_rate, 0.8);
    }
}
