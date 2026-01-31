//! Query Bridge: Connects the incremental query system to the LSP
//!
//! This module bridges the gap between the compiler's query system and the LSP,
//! enabling demand-driven incremental analysis for fast editor responsiveness.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Item};
use crate::build::query::{QueryDb, Revision};
use crate::hir::Hir;
use crate::resolve::{ResolvedAst, SymbolTable};

use super::workspace::Workspace;

// ============================================================================
// Rich Type Information (for SOTA++ LSP features)
// ============================================================================

/// Effect information for a function or expression
#[derive(Debug, Clone, Default)]
pub struct EffectInfo {
    /// Effects this function/expression has
    pub effects: Vec<String>,

    /// Source locations where each effect originates
    pub sources: Vec<EffectSource>,
}

/// Source of an effect
#[derive(Debug, Clone)]
pub struct EffectSource {
    /// Effect name (IO, Mut, Alloc, etc.)
    pub effect: String,

    /// Line where the effect originates
    pub line: u32,

    /// Expression that causes the effect
    pub expr: String,
}

/// Unit/dimensional information for an expression
#[derive(Debug, Clone, Default)]
pub struct UnitInfo {
    /// The unit type (e.g., "kg*m/s^2")
    pub unit: String,

    /// Dimension representation (M, L, T, etc.)
    pub dimension: String,

    /// Whether this is dimensionless
    pub is_dimensionless: bool,
}

/// Epistemic information for a value
#[derive(Debug, Clone, Default)]
pub struct EpistemicInfo {
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,

    /// Lower confidence bound (if known)
    pub confidence_lower: Option<f64>,

    /// Upper confidence bound (if known)
    pub confidence_upper: Option<f64>,

    /// Source of the knowledge (axiomatic, empirical, derived, etc.)
    pub source: String,

    /// Whether this value can be revised
    pub revisable: bool,

    /// Evidence chain
    pub evidence: Vec<EvidenceItem>,

    /// Provenance information
    pub provenance: Option<ProvenanceInfo>,
}

/// A piece of evidence supporting a knowledge claim
#[derive(Debug, Clone)]
pub struct EvidenceItem {
    /// Type of evidence
    pub kind: String,

    /// Reference (citation, experiment ID, etc.)
    pub reference: String,

    /// Strength of this evidence (0.0 - 1.0)
    pub strength: f64,
}

/// Provenance information for tracing data lineage
#[derive(Debug, Clone)]
pub struct ProvenanceInfo {
    /// Original source
    pub origin: String,

    /// Transformations applied
    pub transformations: Vec<String>,

    /// Dependencies (other values this depends on)
    pub dependencies: Vec<String>,
}

/// Refinement type information
#[derive(Debug, Clone)]
pub struct RefinementInfo {
    /// Variable name
    pub variable: String,

    /// Base type
    pub base_type: String,

    /// Predicate (as string)
    pub predicate: String,

    /// SMT verification status
    pub status: SmtStatus,

    /// Counterexample (if verification failed)
    pub counterexample: Option<Counterexample>,

    /// Span in source
    pub span: (u32, u32),
}

/// SMT verification status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtStatus {
    /// Verified by SMT solver
    Verified,
    /// Failed verification
    Failed,
    /// Unknown (solver timeout or couldn't determine)
    Unknown,
    /// Not yet checked
    Pending,
}

impl Default for SmtStatus {
    fn default() -> Self {
        SmtStatus::Pending
    }
}

/// Counterexample from SMT solver
#[derive(Debug, Clone)]
pub struct Counterexample {
    /// Variable assignments that violate the constraint
    pub bindings: Vec<(String, String)>,
}

// ============================================================================
// LSP Analysis Result
// ============================================================================

/// Combined analysis result for LSP features
#[derive(Debug, Clone, Default)]
pub struct LspAnalysisResult {
    /// Parsed AST
    pub ast: Option<Arc<Ast>>,

    /// Resolved AST with symbol table
    pub resolved: Option<Arc<ResolvedAst>>,

    /// Type-checked HIR
    pub hir: Option<Arc<Hir>>,

    /// Symbol table for quick lookups
    pub symbols: Option<SymbolTable>,

    /// Diagnostics from all phases
    pub diagnostics: Vec<Diagnostic>,

    /// Effect information per function
    pub effects: HashMap<String, EffectInfo>,

    /// Unit information per expression (keyed by node id)
    pub units: HashMap<u32, UnitInfo>,

    /// Epistemic information per variable
    pub epistemic: HashMap<String, EpistemicInfo>,

    /// Refinement information
    pub refinements: Vec<RefinementInfo>,

    /// Source text (for span conversion)
    pub source: String,

    /// Revision when this was computed
    pub revision: u64,
}

// ============================================================================
// LSP Query Database
// ============================================================================

/// LSP-specific wrapper around the build system's QueryDb
pub struct LspQueryDatabase {
    /// The underlying query database
    inner: QueryDb,

    /// Workspace for cross-file features
    workspace: Option<Arc<RwLock<Workspace>>>,

    /// URI to PathBuf mapping
    uri_to_path: HashMap<Url, PathBuf>,

    /// Cached analysis results (URI → result)
    /// This is a secondary cache on top of QueryDb for LSP-specific data
    analysis_cache: HashMap<Url, LspAnalysisResult>,

    /// File contents cache (for incremental updates)
    file_contents: HashMap<Url, String>,
}

impl LspQueryDatabase {
    /// Create a new LSP query database
    pub fn new() -> Self {
        Self {
            inner: QueryDb::new(),
            workspace: None,
            uri_to_path: HashMap::new(),
            analysis_cache: HashMap::new(),
            file_contents: HashMap::new(),
        }
    }

    /// Create with workspace reference
    pub fn with_workspace(workspace: Arc<RwLock<Workspace>>) -> Self {
        Self {
            inner: QueryDb::new(),
            workspace: Some(workspace),
            uri_to_path: HashMap::new(),
            analysis_cache: HashMap::new(),
            file_contents: HashMap::new(),
        }
    }

    /// Set workspace reference
    pub fn set_workspace(&mut self, workspace: Arc<RwLock<Workspace>>) {
        self.workspace = Some(workspace);
    }

    /// Get workspace reference
    pub fn workspace(&self) -> Option<&Arc<RwLock<Workspace>>> {
        self.workspace.as_ref()
    }

    /// Notify that a file has changed
    pub fn file_changed(&mut self, uri: &Url, content: &str) {
        let path = self.uri_to_path(uri);

        // Update file contents cache
        self.file_contents.insert(uri.clone(), content.to_string());

        // Bump revision and mark input as changed
        self.inner.bump_revision();
        self.inner.set_input_changed("file_contents", &path);

        // Invalidate analysis cache for this file
        self.analysis_cache.remove(uri);
    }

    /// Notify that a file was closed
    pub fn file_closed(&mut self, uri: &Url) {
        self.uri_to_path.remove(uri);
        self.analysis_cache.remove(uri);
        self.file_contents.remove(uri);
    }

    /// Get analysis result for a file (with caching)
    pub fn analyze(&mut self, uri: &Url) -> LspAnalysisResult {
        // Check if we have a valid cached result
        if let Some(cached) = self.analysis_cache.get(uri) {
            return cached.clone();
        }

        // Perform analysis
        let result = self.do_analyze(uri);

        // Cache the result
        self.analysis_cache.insert(uri.clone(), result.clone());

        result
    }

    /// Perform the actual analysis
    fn do_analyze(&mut self, uri: &Url) -> LspAnalysisResult {
        let mut result = LspAnalysisResult::default();
        result.revision = self.inner.current_revision().get();

        // Get file contents
        let contents = match self.file_contents.get(uri) {
            Some(c) => c.clone(),
            None => return result,
        };
        result.source = contents.clone();

        // Phase 1: Lexing
        let tokens = match crate::lexer::lex(&contents) {
            Ok(tokens) => tokens,
            Err(err) => {
                result
                    .diagnostics
                    .push(Self::message_to_diagnostic(&err.to_string(), &contents));
                return result;
            }
        };

        // Phase 2: Parsing
        let ast = match crate::parser::parse(&tokens, &contents) {
            Ok(ast) => {
                result.ast = Some(Arc::new(ast.clone()));
                ast
            }
            Err(err) => {
                result
                    .diagnostics
                    .push(Self::message_to_diagnostic(&err.to_string(), &contents));
                return result;
            }
        };

        // Phase 3: Name resolution
        let resolved = match crate::resolve::resolve(ast.clone()) {
            Ok(resolved) => {
                // Extract symbols before wrapping in Arc (ResolvedAst doesn't impl Clone)
                result.symbols = Some(resolved.symbols.clone());
                let resolved_arc = Arc::new(resolved);
                result.resolved = Some(resolved_arc.clone());
                resolved_arc
            }
            Err(err) => {
                result
                    .diagnostics
                    .push(Self::message_to_diagnostic(&err.to_string(), &contents));
                return result;
            }
        };

        // Phase 4: Type checking (optional, may fail for incomplete code)
        match crate::check::check(&resolved) {
            Ok(hir) => {
                result.hir = Some(Arc::new(hir));

                // Extract rich type information from the AST
                Self::extract_effects(&result.ast.as_ref().unwrap(), &mut result.effects);
                // TODO: Extract units, epistemic, refinements from HIR
            }
            Err(_err) => {
                // Type checking failed, but we still have AST and symbols
                // Don't add error - user is likely still typing
            }
        }

        result
    }

    /// Get just the AST (faster if you don't need full analysis)
    pub fn get_ast(&mut self, uri: &Url) -> Option<Arc<Ast>> {
        self.analyze(uri).ast
    }

    /// Get the symbol table
    pub fn get_symbols(&mut self, uri: &Url) -> Option<SymbolTable> {
        self.analyze(uri).symbols
    }

    /// Get diagnostics for a file
    pub fn get_diagnostics(&mut self, uri: &Url) -> Vec<Diagnostic> {
        self.analyze(uri).diagnostics
    }

    /// Get effect information for a function
    pub fn get_effects(&mut self, uri: &Url, function_name: &str) -> Option<EffectInfo> {
        self.analyze(uri).effects.get(function_name).cloned()
    }

    /// Get unit information for an expression
    pub fn get_unit(&mut self, uri: &Url, node_id: u32) -> Option<UnitInfo> {
        self.analyze(uri).units.get(&node_id).cloned()
    }

    /// Get epistemic information for a variable
    pub fn get_epistemic(&mut self, uri: &Url, var_name: &str) -> Option<EpistemicInfo> {
        self.analyze(uri).epistemic.get(var_name).cloned()
    }

    /// Get refinement information
    pub fn get_refinements(&mut self, uri: &Url) -> Vec<RefinementInfo> {
        self.analyze(uri).refinements
    }

    /// Get current revision
    pub fn current_revision(&self) -> Revision {
        self.inner.current_revision()
    }

    /// Get cache statistics
    pub fn cache_hit_rate(&self) -> f64 {
        self.inner.hit_rate()
    }

    /// Convert URI to PathBuf, caching the result
    fn uri_to_path(&mut self, uri: &Url) -> PathBuf {
        if let Some(path) = self.uri_to_path.get(uri) {
            return path.clone();
        }

        let path = uri
            .to_file_path()
            .unwrap_or_else(|_| PathBuf::from(uri.path()));
        self.uri_to_path.insert(uri.clone(), path.clone());
        path
    }

    /// Convert an error message to LSP diagnostic
    fn message_to_diagnostic(message: &str, source: &str) -> Diagnostic {
        // Try to extract span from error message
        let range = Self::extract_range_from_message(message, source).unwrap_or(Range {
            start: Position::new(0, 0),
            end: Position::new(0, 1),
        });

        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("sounio".to_string()),
            message: message.to_string(),
            related_information: None,
            tags: None,
            data: None,
        }
    }

    /// Extract range from error message
    fn extract_range_from_message(message: &str, source: &str) -> Option<Range> {
        // Try to find line:col pattern in message (e.g., "at 5:10" or "5:10")
        for (i, c) in message.char_indices() {
            if c.is_ascii_digit() {
                let rest = &message[i..];
                if let Some(colon_pos) = rest.find(':') {
                    let before_colon = &rest[..colon_pos];

                    // Check if before_colon is all digits
                    if before_colon.chars().all(|c| c.is_ascii_digit()) {
                        let after_colon = &rest[colon_pos + 1..];
                        let col_end = after_colon
                            .find(|c: char| !c.is_ascii_digit())
                            .unwrap_or(after_colon.len());
                        let col_str = &after_colon[..col_end];

                        if !col_str.is_empty() {
                            if let (Ok(line), Ok(col)) =
                                (before_colon.parse::<u32>(), col_str.parse::<u32>())
                            {
                                return Some(Range {
                                    start: Position::new(
                                        line.saturating_sub(1),
                                        col.saturating_sub(1),
                                    ),
                                    end: Position::new(line.saturating_sub(1), col),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Default to first line
        Some(Range {
            start: Position::new(0, 0),
            end: Position::new(
                0,
                source.lines().next().map(|l| l.len() as u32).unwrap_or(1),
            ),
        })
    }

    /// Extract effect information from AST
    fn extract_effects(ast: &Ast, effects: &mut HashMap<String, EffectInfo>) {
        for item in &ast.items {
            if let Item::Function(f) = item {
                let mut info = EffectInfo::default();

                // Get declared effects from function signature
                for effect in &f.effects {
                    info.effects.push(effect.name.to_string());
                }

                // TODO: Walk function body to find effect sources
                // This would require deeper AST analysis with the effect checker

                effects.insert(f.name.clone(), info);
            }
        }
    }
}

impl Default for LspQueryDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsp_query_database_creation() {
        let db = LspQueryDatabase::new();
        assert!(db.workspace.is_none());
    }

    #[test]
    fn test_uri_to_path_conversion() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///home/test/foo.sio").unwrap();
        let path = db.uri_to_path(&uri);
        assert_eq!(path, PathBuf::from("/home/test/foo.sio"));
    }

    #[test]
    fn test_file_changed_creates_entry() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        db.file_changed(&uri, "fn main() {}");

        // File contents should be cached
        assert!(db.file_contents.contains_key(&uri));
    }

    #[test]
    fn test_analyze_empty_file() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        db.file_changed(&uri, "");

        let result = db.analyze(&uri);
        // Empty file should parse successfully
        assert!(result.ast.is_some());
    }

    #[test]
    fn test_analyze_simple_function() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        db.file_changed(&uri, "fn hello() -> i32 { 42 }");

        let result = db.analyze(&uri);
        assert!(result.ast.is_some());

        // Should have extracted effect info for the function
        assert!(result.effects.contains_key("hello"));
    }

    #[test]
    fn test_cache_invalidation() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();

        // Initial analysis
        db.file_changed(&uri, "fn foo() {}");
        let result1 = db.analyze(&uri);
        let rev1 = result1.revision;

        // Change file - should invalidate cache
        db.file_changed(&uri, "fn bar() {}");
        let result2 = db.analyze(&uri);

        // Revision should have changed
        assert!(result2.revision > rev1);
    }

    #[test]
    fn test_extract_range_from_message() {
        let source = "fn main() {\n    let x = 5;\n}";

        // Test with line:col pattern
        let range = LspQueryDatabase::extract_range_from_message("error at 2:5: undefined", source);
        assert!(range.is_some());
        let r = range.unwrap();
        assert_eq!(r.start.line, 1); // 0-indexed
        assert_eq!(r.start.character, 4); // 0-indexed
    }

    #[test]
    fn test_analyze_with_symbols() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let code = r#"
fn add(x: i32, y: i32) -> i32 {
    x + y
}

fn main() -> i32 {
    add(1, 2)
}
"#;
        db.file_changed(&uri, code);
        let result = db.analyze(&uri);

        // Should have symbols
        assert!(result.symbols.is_some(), "Should have symbol table");

        if let Some(ref symbols) = result.symbols {
            // Should find the 'add' function
            let add_def = symbols.lookup("add");
            assert!(add_def.is_some(), "Should find 'add' function");

            // Should find the 'main' function
            let main_def = symbols.lookup("main");
            assert!(main_def.is_some(), "Should find 'main' function");
        }

        // Should have effect info for functions
        assert!(
            result.effects.contains_key("add"),
            "Should have effect info for 'add'"
        );
        assert!(
            result.effects.contains_key("main"),
            "Should have effect info for 'main'"
        );
    }

    #[test]
    fn test_analyze_function_with_effects() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let code = r#"
fn print_value(x: i32) -> () with IO {
    // uses IO effect
}
"#;
        db.file_changed(&uri, code);
        let result = db.analyze(&uri);

        // Should extract effect information
        assert!(result.effects.contains_key("print_value"));
        let effect_info = result.effects.get("print_value").unwrap();
        assert!(
            effect_info.effects.contains(&"IO".to_string()),
            "Should have IO effect"
        );
    }

    #[test]
    fn test_analyze_struct_definition() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let code = r#"
struct Point {
    x: i32,
    y: i32
}
"#;
        db.file_changed(&uri, code);
        let result = db.analyze(&uri);

        assert!(result.symbols.is_some());
        if let Some(ref symbols) = result.symbols {
            let point_def = symbols.lookup_type("Point");
            assert!(point_def.is_some(), "Should find 'Point' struct");
        }
    }

    #[test]
    fn test_analyze_syntax_error_produces_diagnostics() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let code = r#"
fn broken() -> i32 {
    let x =  // missing expression
}
"#;
        db.file_changed(&uri, code);
        let result = db.analyze(&uri);

        // Should have diagnostics for the syntax error
        assert!(
            !result.diagnostics.is_empty(),
            "Should have diagnostics for syntax error"
        );
    }

    #[test]
    fn test_file_closed_removes_cache() {
        let mut db = LspQueryDatabase::new();
        let uri = Url::parse("file:///test.sio").unwrap();

        db.file_changed(&uri, "fn test() {}");
        assert!(db.file_contents.contains_key(&uri));

        db.file_closed(&uri);
        assert!(!db.file_contents.contains_key(&uri));
        assert!(!db.analysis_cache.contains_key(&uri));
    }

    #[test]
    fn test_multiple_files() {
        let mut db = LspQueryDatabase::new();
        let uri1 = Url::parse("file:///test1.sio").unwrap();
        let uri2 = Url::parse("file:///test2.sio").unwrap();

        db.file_changed(&uri1, "fn foo() -> i32 { 1 }");
        db.file_changed(&uri2, "fn bar() -> i32 { 2 }");

        let result1 = db.analyze(&uri1);
        let result2 = db.analyze(&uri2);

        // Both should have their respective symbols
        assert!(result1.effects.contains_key("foo"));
        assert!(result2.effects.contains_key("bar"));

        // Ensure no cross-contamination
        assert!(!result1.effects.contains_key("bar"));
        assert!(!result2.effects.contains_key("foo"));
    }
}
