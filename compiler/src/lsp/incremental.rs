//! Incremental compilation for LSP
//!
//! Provides dependency-aware incremental analysis to avoid re-parsing
//! and re-analyzing unchanged files. Key features:
//!
//! - Content hashing to detect changes efficiently
//! - Dependency graph integration for transitive invalidation
//! - Cached parse results with version tracking
//! - Efficient re-parsing only when content changes
//!
//! # Architecture
//!
//! The `IncrementalAnalyzer` sits between the LSP server and the analysis
//! pipeline. When a document changes:
//!
//! 1. Content hash is computed and compared to cached version
//! 2. If unchanged, cached results are returned immediately
//! 3. If changed, the file is re-analyzed and dependents are invalidated
//! 4. Transitive dependents are marked dirty for lazy re-analysis
//!
//! # Example
//!
//! ```ignore
//! let mut analyzer = IncrementalAnalyzer::new();
//!
//! // On document change
//! analyzer.invalidate(&uri);
//!
//! // Get affected files for re-analysis
//! let affected = analyzer.get_affected_files(&uri);
//!
//! // Check if cache is valid before analyzing
//! if !analyzer.is_cached_valid(&uri, content_hash) {
//!     let result = analyze_file(&source);
//!     analyzer.cache_result(&uri, content_hash, result);
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tower_lsp::lsp_types::*;

use crate::ast::Ast;
use crate::lexer;
use crate::parser;

use super::workspace::DependencyGraph;

/// Cache entry for a parsed file
///
/// Note: `SymbolTable` is not cached because it doesn't implement Clone
/// and is relatively cheap to regenerate from the AST. The main benefit
/// of caching comes from avoiding re-lexing and re-parsing unchanged files.
#[derive(Debug)]
pub struct CacheEntry {
    /// Content hash when this entry was created
    pub content_hash: u64,

    /// Parsed AST (if parsing succeeded)
    pub ast: Option<Ast>,

    /// Diagnostics from analysis
    pub diagnostics: Vec<Diagnostic>,

    /// Document version when cached
    pub version: i32,

    /// Timestamp when entry was created
    pub created_at: Instant,

    /// Number of times this entry has been accessed
    pub access_count: u64,

    /// Files this file imports (extracted from AST)
    pub imports: HashSet<PathBuf>,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(content_hash: u64, version: i32) -> Self {
        Self {
            content_hash,
            ast: None,
            diagnostics: Vec::new(),
            version,
            created_at: Instant::now(),
            access_count: 0,
            imports: HashSet::new(),
        }
    }

    /// Update entry with analysis results
    pub fn with_results(mut self, ast: Option<Ast>, diagnostics: Vec<Diagnostic>) -> Self {
        self.ast = ast;
        self.diagnostics = diagnostics;
        self
    }

    /// Check if entry has expired (for cache eviction)
    pub fn is_expired(&self, max_age: Duration) -> bool {
        self.created_at.elapsed() > max_age
    }

    /// Record an access to this entry
    pub fn touch(&mut self) {
        self.access_count += 1;
    }
}

/// Cache statistics for monitoring and debugging
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,

    /// Number of cache misses
    pub misses: u64,

    /// Number of invalidations
    pub invalidations: u64,

    /// Number of entries currently cached
    pub entries: usize,

    /// Total bytes of cached ASTs (approximate)
    pub estimated_memory: usize,
}

impl CacheStats {
    /// Calculate cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Result from incremental analysis
///
/// This is a simpler version of `AnalysisResult` that doesn't include
/// `SymbolTable` since it doesn't implement `Clone`. The symbol table
/// should be regenerated from the AST when needed.
#[derive(Debug)]
pub struct IncrementalResult {
    /// Parsed AST (if parsing succeeded)
    pub ast: Option<Ast>,

    /// Diagnostics from parsing
    pub diagnostics: Vec<Diagnostic>,

    /// Document version when analyzed
    pub version: i32,

    /// Whether this was a cache hit
    pub cache_hit: bool,
}

impl IncrementalResult {
    /// Create a new incremental result
    pub fn new(ast: Option<Ast>, diagnostics: Vec<Diagnostic>, version: i32, cache_hit: bool) -> Self {
        Self {
            ast,
            diagnostics,
            version,
            cache_hit,
        }
    }
}

/// Incremental analyzer with dependency-aware caching
///
/// This analyzer maintains a cache of parse results and tracks
/// dependencies between files to enable efficient incremental
/// re-analysis when files change.
#[derive(Debug)]
pub struct IncrementalAnalyzer {
    /// Cached parse results indexed by URI
    /// Stores (content_hash, CacheEntry)
    parse_cache: HashMap<Url, CacheEntry>,

    /// Dependency graph for tracking imports
    /// Owned copy that can be synced with workspace
    dependencies: DependencyGraph,

    /// Set of URIs that need reanalysis
    dirty: HashSet<Url>,

    /// URI to file path mapping
    uri_to_path: HashMap<Url, PathBuf>,

    /// Cache statistics
    stats: CacheStats,

    /// Maximum cache age before eviction
    max_cache_age: Duration,

    /// Maximum number of cache entries
    max_cache_entries: usize,
}

impl IncrementalAnalyzer {
    /// Create a new incremental analyzer
    pub fn new() -> Self {
        Self {
            parse_cache: HashMap::new(),
            dependencies: DependencyGraph::new(),
            dirty: HashSet::new(),
            uri_to_path: HashMap::new(),
            stats: CacheStats::default(),
            max_cache_age: Duration::from_secs(300), // 5 minutes
            max_cache_entries: 1000,
        }
    }

    /// Create analyzer with custom cache settings
    pub fn with_settings(max_cache_age: Duration, max_cache_entries: usize) -> Self {
        Self {
            parse_cache: HashMap::new(),
            dependencies: DependencyGraph::new(),
            dirty: HashSet::new(),
            uri_to_path: HashMap::new(),
            stats: CacheStats::default(),
            max_cache_age,
            max_cache_entries,
        }
    }

    /// Compute a hash of source content
    pub fn hash_content(source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }

    /// Register a URI to path mapping
    pub fn register_path(&mut self, uri: &Url, path: PathBuf) {
        self.uri_to_path.insert(uri.clone(), path);
    }

    /// Invalidate a file and mark dependents as dirty
    ///
    /// Call this when a document changes. This will:
    /// 1. Remove the file from cache
    /// 2. Mark all files that depend on it as dirty
    /// 3. Update statistics
    pub fn invalidate(&mut self, uri: &Url) {
        // Remove from cache
        self.parse_cache.remove(uri);
        self.stats.invalidations += 1;

        // Mark this file as dirty
        self.dirty.insert(uri.clone());

        // Mark dependents as dirty
        if let Some(path) = self.uri_to_path.get(uri) {
            let affected = self.dependencies.get_affected_files(path);
            for affected_path in affected {
                if let Ok(affected_uri) = Url::from_file_path(&affected_path) {
                    self.dirty.insert(affected_uri.clone());
                    // Don't remove from cache yet - lazy invalidation
                    // The entry will be revalidated on next access
                    if let Some(entry) = self.parse_cache.get_mut(&affected_uri) {
                        // Mark as stale by setting hash to 0
                        entry.content_hash = 0;
                    }
                }
            }
        }

        self.update_stats();
    }

    /// Get all files that need reanalysis after a change
    ///
    /// Returns URIs of all files affected by a change to the given file,
    /// including transitive dependents.
    pub fn get_affected_files(&self, uri: &Url) -> Vec<Url> {
        let mut affected = Vec::new();

        if let Some(path) = self.uri_to_path.get(uri) {
            let affected_paths = self.dependencies.get_affected_files(path);
            for affected_path in affected_paths {
                if let Ok(affected_uri) = Url::from_file_path(&affected_path) {
                    affected.push(affected_uri);
                }
            }
        }

        affected
    }

    /// Check if a cached result is still valid
    ///
    /// Returns true if:
    /// 1. There is a cache entry for this URI
    /// 2. The content hash matches
    /// 3. The file is not marked as dirty
    pub fn is_cached_valid(&self, uri: &Url, content_hash: u64) -> bool {
        if self.dirty.contains(uri) {
            return false;
        }

        if let Some(entry) = self.parse_cache.get(uri) {
            entry.content_hash == content_hash && content_hash != 0
        } else {
            false
        }
    }

    /// Get cached analysis result if valid
    ///
    /// Returns the cached result if the hash matches and the entry
    /// is not dirty. Updates cache statistics.
    pub fn get_cached(&mut self, uri: &Url, content_hash: u64) -> Option<&CacheEntry> {
        if !self.is_cached_valid(uri, content_hash) {
            self.stats.misses += 1;
            return None;
        }

        // Update access count
        if let Some(entry) = self.parse_cache.get_mut(uri) {
            entry.touch();
        }

        self.stats.hits += 1;
        self.parse_cache.get(uri)
    }

    /// Get or compute analysis result
    ///
    /// This is the main entry point for incremental analysis. It:
    /// 1. Checks if cached result is valid
    /// 2. If valid, returns cached result
    /// 3. If invalid, parses and analyzes the file
    /// 4. Caches the new result
    /// 5. Updates dependency graph based on imports
    ///
    /// Note: This returns `IncrementalResult` which does not include
    /// `SymbolTable` since it doesn't implement Clone. Generate a new
    /// `SymbolTable` from the AST if needed.
    pub fn get_or_analyze(
        &mut self,
        uri: &Url,
        source: &str,
        version: i32,
    ) -> IncrementalResult {
        let content_hash = Self::hash_content(source);

        // Check cache first
        if self.is_cached_valid(uri, content_hash) {
            if let Some(entry) = self.parse_cache.get_mut(uri) {
                entry.touch();
                self.stats.hits += 1;

                return IncrementalResult::new(
                    entry.ast.clone(),
                    entry.diagnostics.clone(),
                    entry.version,
                    true, // cache hit
                );
            }
        }

        self.stats.misses += 1;

        // Parse and analyze
        let (ast, diagnostics) = self.analyze_source(source);

        // Extract imports from AST for dependency tracking
        let imports = ast.as_ref().map(|a| self.extract_imports(a)).unwrap_or_default();

        // Update dependency graph
        if let Some(path) = self.uri_to_path.get(uri).cloned() {
            // Clear old dependencies for this file
            self.dependencies.clear_file(&path);

            // Add new dependencies
            for import_path in &imports {
                self.dependencies.add_dependency(&path, import_path);
            }
        }

        // Cache the result
        let mut entry = CacheEntry::new(content_hash, version);
        entry.ast = ast.clone();
        entry.diagnostics = diagnostics.clone();
        entry.imports = imports;

        // Remove from dirty set since we just analyzed it
        self.dirty.remove(uri);

        // Evict old entries if necessary
        self.maybe_evict();

        self.parse_cache.insert(uri.clone(), entry);
        self.update_stats();

        IncrementalResult::new(ast, diagnostics, version, false)
    }

    /// Cache an externally computed analysis result
    ///
    /// Use this to cache results from external analysis pipelines.
    pub fn cache_result(
        &mut self,
        uri: &Url,
        content_hash: u64,
        ast: Option<Ast>,
        diagnostics: Vec<Diagnostic>,
        version: i32,
    ) {
        let entry = CacheEntry::new(content_hash, version).with_results(ast, diagnostics);

        self.dirty.remove(uri);
        self.maybe_evict();
        self.parse_cache.insert(uri.clone(), entry);
        self.update_stats();
    }

    /// Parse and analyze source code
    ///
    /// Returns the AST and diagnostics. Symbol table generation is left
    /// to the caller since it doesn't need to be cached.
    fn analyze_source(&self, source: &str) -> (Option<Ast>, Vec<Diagnostic>) {
        let mut diagnostics = Vec::new();
        let mut ast = None;

        // Phase 1: Lexing
        match lexer::lex(source) {
            Ok(tokens) => {
                // Phase 2: Parsing
                match parser::parse(&tokens, source) {
                    Ok(parsed_ast) => {
                        ast = Some(parsed_ast);
                    }
                    Err(err) => {
                        diagnostics.push(self.miette_to_diagnostic(&err, source));
                    }
                }
            }
            Err(err) => {
                diagnostics.push(self.miette_to_diagnostic(&err, source));
            }
        }

        (ast, diagnostics)
    }

    /// Extract import paths from an AST
    ///
    /// Scans the AST for import statements and converts module paths
    /// to file paths. This enables dependency tracking between files.
    fn extract_imports(&self, ast: &Ast) -> HashSet<PathBuf> {
        let mut imports = HashSet::new();

        for item in &ast.items {
            if let crate::ast::Item::Import(import) = item {
                // Convert module path to file path
                // This is a simplified version - full implementation would
                // use workspace module resolution to handle:
                // - Multiple file extensions (.sio, .d)
                // - Module directories (mod.sio)
                // - External packages
                let path_str = import.path.segments.join("/");
                imports.insert(PathBuf::from(format!("{}.sio", path_str)));

                // Also try the legacy .d extension
                imports.insert(PathBuf::from(format!("{}.d", path_str)));
            }
        }

        imports
    }

    /// Convert miette error to LSP Diagnostic
    fn miette_to_diagnostic(&self, err: &dyn miette::Diagnostic, source: &str) -> Diagnostic {
        let message = err.to_string();

        // Try to extract span information
        let range = if let Some(labels) = err.labels() {
            let mut range = Range::default();
            for label in labels {
                let start = self.offset_to_position(source, label.offset());
                let end = self.offset_to_position(source, label.offset() + label.len());
                range = Range { start, end };
                break; // Use first label
            }
            range
        } else {
            Range::default()
        };

        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::ERROR),
            code: err.code().map(|c| NumberOrString::String(c.to_string())),
            source: Some("sounio".to_string()),
            message,
            related_information: None,
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Convert byte offset to LSP Position
    fn offset_to_position(&self, source: &str, offset: usize) -> Position {
        let offset = offset.min(source.len());
        let mut line = 0;
        let mut col = 0;

        for (i, c) in source.char_indices() {
            if i >= offset {
                break;
            }
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }

        Position {
            line,
            character: col,
        }
    }

    /// Evict old cache entries if necessary
    fn maybe_evict(&mut self) {
        // First, remove expired entries
        let max_age = self.max_cache_age;
        self.parse_cache.retain(|_, entry| !entry.is_expired(max_age));

        // If still over limit, remove least recently used
        if self.parse_cache.len() > self.max_cache_entries {
            let mut entries: Vec<_> = self.parse_cache.iter().collect();
            entries.sort_by_key(|(_, e)| std::cmp::Reverse(e.access_count));

            let to_remove: Vec<Url> = entries
                .iter()
                .skip(self.max_cache_entries)
                .map(|(uri, _)| (*uri).clone())
                .collect();

            for uri in to_remove {
                self.parse_cache.remove(&uri);
            }
        }
    }

    /// Update cache statistics
    fn update_stats(&mut self) {
        self.stats.entries = self.parse_cache.len();
        // Rough memory estimate: ~1KB per AST on average
        self.stats.estimated_memory = self.parse_cache.len() * 1024;
    }

    /// Get current cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get all dirty files
    pub fn dirty_files(&self) -> &HashSet<Url> {
        &self.dirty
    }

    /// Check if a file is dirty
    pub fn is_dirty(&self, uri: &Url) -> bool {
        self.dirty.contains(uri)
    }

    /// Clear all dirty flags
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Clear the entire cache
    pub fn clear_cache(&mut self) {
        self.parse_cache.clear();
        self.dirty.clear();
        self.stats = CacheStats::default();
    }

    /// Sync dependency graph from workspace
    ///
    /// Call this to update the internal dependency graph from
    /// the workspace's authoritative version.
    pub fn sync_dependencies(&mut self, workspace_deps: &DependencyGraph) {
        // Clone the dependency graph from workspace
        // In a more sophisticated implementation, we'd share the graph
        self.dependencies = DependencyGraph::new();

        // Note: DependencyGraph doesn't expose iteration, so we'd need
        // to add that functionality or track dependencies ourselves
    }

    /// Add a dependency between two files
    pub fn add_dependency(&mut self, from: &Url, to: &Url) {
        if let (Some(from_path), Some(to_path)) = (
            self.uri_to_path.get(from).cloned(),
            self.uri_to_path.get(to).cloned(),
        ) {
            self.dependencies.add_dependency(&from_path, &to_path);
        }
    }

    /// Get files that the given file depends on
    pub fn get_dependencies(&self, uri: &Url) -> Vec<Url> {
        let mut deps = Vec::new();

        if let Some(path) = self.uri_to_path.get(uri) {
            if let Some(imports) = self.dependencies.get_imports(path) {
                for import_path in imports {
                    if let Ok(import_uri) = Url::from_file_path(import_path) {
                        deps.push(import_uri);
                    }
                }
            }
        }

        deps
    }

    /// Get files that depend on the given file
    pub fn get_dependents(&self, uri: &Url) -> Vec<Url> {
        let mut deps = Vec::new();

        if let Some(path) = self.uri_to_path.get(uri) {
            if let Some(dependents) = self.dependencies.get_dependents(path) {
                for dep_path in dependents {
                    if let Ok(dep_uri) = Url::from_file_path(dep_path) {
                        deps.push(dep_uri);
                    }
                }
            }
        }

        deps
    }
}

impl Default for IncrementalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for IncrementalAnalyzer with configuration options
pub struct IncrementalAnalyzerBuilder {
    max_cache_age: Duration,
    max_cache_entries: usize,
}

impl IncrementalAnalyzerBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            max_cache_age: Duration::from_secs(300),
            max_cache_entries: 1000,
        }
    }

    /// Set maximum cache age
    pub fn max_cache_age(mut self, duration: Duration) -> Self {
        self.max_cache_age = duration;
        self
    }

    /// Set maximum number of cache entries
    pub fn max_cache_entries(mut self, count: usize) -> Self {
        self.max_cache_entries = count;
        self
    }

    /// Build the analyzer
    pub fn build(self) -> IncrementalAnalyzer {
        IncrementalAnalyzer::with_settings(self.max_cache_age, self.max_cache_entries)
    }
}

impl Default for IncrementalAnalyzerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hashing() {
        let hash1 = IncrementalAnalyzer::hash_content("fn main() {}");
        let hash2 = IncrementalAnalyzer::hash_content("fn main() {}");
        let hash3 = IncrementalAnalyzer::hash_content("fn main() { }");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_entry_creation() {
        let entry = CacheEntry::new(12345, 1);
        assert_eq!(entry.content_hash, 12345);
        assert_eq!(entry.version, 1);
        assert!(entry.ast.is_none());
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_cache_entry_touch() {
        let mut entry = CacheEntry::new(12345, 1);
        assert_eq!(entry.access_count, 0);

        entry.touch();
        assert_eq!(entry.access_count, 1);

        entry.touch();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_incremental_analyzer_creation() {
        let analyzer = IncrementalAnalyzer::new();
        assert_eq!(analyzer.stats().entries, 0);
        assert_eq!(analyzer.stats().hits, 0);
        assert_eq!(analyzer.stats().misses, 0);
    }

    #[test]
    fn test_cache_validity() {
        let mut analyzer = IncrementalAnalyzer::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let source = "fn main() {}";
        let hash = IncrementalAnalyzer::hash_content(source);

        // Initially not cached
        assert!(!analyzer.is_cached_valid(&uri, hash));

        // After caching
        let result = AnalysisResult::default();
        analyzer.cache_result(&uri, hash, result);

        assert!(analyzer.is_cached_valid(&uri, hash));

        // Different hash should be invalid
        let different_hash = IncrementalAnalyzer::hash_content("fn main() { }");
        assert!(!analyzer.is_cached_valid(&uri, different_hash));
    }

    #[test]
    fn test_invalidation() {
        let mut analyzer = IncrementalAnalyzer::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let hash = IncrementalAnalyzer::hash_content("fn main() {}");

        // Cache a result
        analyzer.cache_result(&uri, hash, AnalysisResult::default());
        assert!(analyzer.is_cached_valid(&uri, hash));

        // Invalidate
        analyzer.invalidate(&uri);
        assert!(!analyzer.is_cached_valid(&uri, hash));
        assert!(analyzer.is_dirty(&uri));
        assert_eq!(analyzer.stats().invalidations, 1);
    }

    #[test]
    fn test_dirty_tracking() {
        let mut analyzer = IncrementalAnalyzer::new();
        let uri = Url::parse("file:///test.sio").unwrap();

        assert!(!analyzer.is_dirty(&uri));

        analyzer.invalidate(&uri);
        assert!(analyzer.is_dirty(&uri));

        analyzer.clear_dirty();
        assert!(!analyzer.is_dirty(&uri));
    }

    #[test]
    fn test_cache_stats() {
        let mut analyzer = IncrementalAnalyzer::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let source = "fn main() {}";
        let hash = IncrementalAnalyzer::hash_content(source);

        // Miss on first access
        assert!(analyzer.get_cached(&uri, hash).is_none());
        assert_eq!(analyzer.stats().misses, 1);

        // Cache a result
        analyzer.cache_result(&uri, hash, AnalysisResult::default());

        // Hit on subsequent access
        assert!(analyzer.get_cached(&uri, hash).is_some());
        assert_eq!(analyzer.stats().hits, 1);
    }

    #[test]
    fn test_hit_ratio() {
        let stats = CacheStats {
            hits: 80,
            misses: 20,
            ..Default::default()
        };
        assert!((stats.hit_ratio() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_builder() {
        let analyzer = IncrementalAnalyzerBuilder::new()
            .max_cache_age(Duration::from_secs(60))
            .max_cache_entries(100)
            .build();

        assert_eq!(analyzer.max_cache_age, Duration::from_secs(60));
        assert_eq!(analyzer.max_cache_entries, 100);
    }

    #[test]
    fn test_path_registration() {
        let mut analyzer = IncrementalAnalyzer::new();
        let uri = Url::parse("file:///project/test.sio").unwrap();
        let path = PathBuf::from("/project/test.sio");

        analyzer.register_path(&uri, path.clone());

        // Dependencies should work after registration
        assert!(analyzer.get_dependencies(&uri).is_empty());
    }

    #[test]
    fn test_clear_cache() {
        let mut analyzer = IncrementalAnalyzer::new();
        let uri = Url::parse("file:///test.sio").unwrap();

        analyzer.cache_result(&uri, 12345, AnalysisResult::default());
        assert_eq!(analyzer.stats().entries, 1);

        analyzer.clear_cache();
        assert_eq!(analyzer.stats().entries, 0);
        assert!(!analyzer.is_cached_valid(&uri, 12345));
    }
}
