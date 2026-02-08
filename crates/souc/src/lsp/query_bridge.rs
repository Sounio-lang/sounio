//! Query Bridge: Connects the incremental query system to the LSP
//!
//! This module bridges the gap between the compiler's query system and the LSP,
//! enabling demand-driven incremental analysis for fast editor responsiveness.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Expr, Item, Literal, Pattern, Stmt, TypeExpr, UnitExpr};
use crate::build::query::{QueryDb, Revision};
use crate::common::{NodeId, Span};
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
                let ast = result.ast.as_ref().unwrap();
                Self::extract_effects(ast, &contents, &mut result.effects);
                Self::extract_semantic_facts(
                    ast,
                    &contents,
                    &mut result.units,
                    &mut result.epistemic,
                    &mut result.refinements,
                );
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

    /// Extract effect information from AST and function bodies.
    fn extract_effects(ast: &Ast, source: &str, effects: &mut HashMap<String, EffectInfo>) {
        for item in &ast.items {
            if let Item::Function(f) = item {
                let mut info = EffectInfo::default();
                let mut seen_effects = HashSet::new();
                let mut seen_sources = HashSet::new();

                // Declared effects from function signature.
                for effect in &f.effects {
                    let name = effect.name.to_string();
                    if seen_effects.insert(name.clone()) {
                        info.effects.push(name);
                    }
                }

                // Heuristically infer effect sources from function body.
                Self::collect_effect_sources_from_block(
                    &f.body,
                    ast,
                    source,
                    &mut info.effects,
                    &mut seen_effects,
                    &mut info.sources,
                    &mut seen_sources,
                );

                effects.insert(f.name.clone(), info);
            }
        }
    }

    fn collect_effect_sources_from_block(
        block: &crate::ast::Block,
        ast: &Ast,
        source: &str,
        effect_names: &mut Vec<String>,
        seen_effects: &mut HashSet<String>,
        sources: &mut Vec<EffectSource>,
        seen_sources: &mut HashSet<(String, u32, String)>,
    ) {
        for stmt in &block.stmts {
            Self::collect_effect_sources_from_stmt(
                stmt,
                ast,
                source,
                effect_names,
                seen_effects,
                sources,
                seen_sources,
            );
        }
    }

    fn collect_effect_sources_from_stmt(
        stmt: &Stmt,
        ast: &Ast,
        source: &str,
        effect_names: &mut Vec<String>,
        seen_effects: &mut HashSet<String>,
        sources: &mut Vec<EffectSource>,
        seen_sources: &mut HashSet<(String, u32, String)>,
    ) {
        match stmt {
            Stmt::Let { value, .. } => {
                if let Some(expr) = value {
                    Self::collect_effect_sources_from_expr(
                        expr,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Stmt::Expr { expr, .. } => {
                Self::collect_effect_sources_from_expr(
                    expr,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Stmt::Assign { target, value, .. } => {
                let line = Self::expr_line(target, ast, source);
                Self::record_effect_source(
                    "Mut",
                    "assignment",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    target,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    value,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Stmt::Empty | Stmt::MacroInvocation(_) | Stmt::LocalExtern(_) => {}
        }
    }

    fn collect_effect_sources_from_expr(
        expr: &Expr,
        ast: &Ast,
        source: &str,
        effect_names: &mut Vec<String>,
        seen_effects: &mut HashSet<String>,
        sources: &mut Vec<EffectSource>,
        seen_sources: &mut HashSet<(String, u32, String)>,
    ) {
        match expr {
            Expr::Perform {
                effect, op, args, ..
            } => {
                let effect_name = effect.name().unwrap_or("Effect").to_string();
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    &effect_name,
                    &format!("perform {}::{}", effect, op),
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                for arg in args {
                    Self::collect_effect_sources_from_expr(
                        arg,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Sample { distribution, .. } => {
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    "Prob",
                    "sample",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    distribution,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Observe {
                data, distribution, ..
            } => {
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    "Prob",
                    "observe",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    data,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    distribution,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Await { expr, .. } | Expr::Spawn { expr, .. } => {
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    "Async",
                    "async operation",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    expr,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Handle { expr, .. } | Expr::Try { expr, .. } | Expr::Unary { expr, .. } => {
                Self::collect_effect_sources_from_expr(
                    expr,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::AsyncBlock { block, .. } => {
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    "Async",
                    "async block",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_block(
                    block,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::AsyncClosure { body, .. } => {
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    "Async",
                    "async closure",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    body,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Select { arms, .. } => {
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    "Async",
                    "select",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                for arm in arms {
                    Self::collect_effect_sources_from_expr(
                        &arm.future,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                    Self::collect_effect_sources_from_expr(
                        &arm.body,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Join { futures, .. } => {
                let line = Self::expr_line(expr, ast, source);
                Self::record_effect_source(
                    "Async",
                    "join",
                    line,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                for future in futures {
                    Self::collect_effect_sources_from_expr(
                        future,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Call { callee, args, .. } => {
                if let Some(name) = Self::call_name(callee) {
                    if Self::is_io_call(name) {
                        let line = Self::expr_line(expr, ast, source);
                        Self::record_effect_source(
                            "IO",
                            &format!("call {}", name),
                            line,
                            effect_names,
                            seen_effects,
                            sources,
                            seen_sources,
                        );
                    }
                }
                Self::collect_effect_sources_from_expr(
                    callee,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                for arg in args {
                    Self::collect_effect_sources_from_expr(
                        arg,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                if Self::is_mutating_method(method) {
                    let line = Self::expr_line(expr, ast, source);
                    Self::record_effect_source(
                        "Mut",
                        &format!("method .{}()", method),
                        line,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
                Self::collect_effect_sources_from_expr(
                    receiver,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                for arg in args {
                    Self::collect_effect_sources_from_expr(
                        arg,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Binary { left, right, .. } => {
                Self::collect_effect_sources_from_expr(
                    left,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    right,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                Self::collect_effect_sources_from_expr(
                    condition,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_block(
                    then_branch,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                if let Some(else_expr) = else_branch {
                    Self::collect_effect_sources_from_expr(
                        else_expr,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Block { block, .. }
            | Expr::Loop { body: block, .. }
            | Expr::UnsafeBlock { block, .. } => {
                Self::collect_effect_sources_from_block(
                    block,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::While {
                condition, body, ..
            } => {
                Self::collect_effect_sources_from_expr(
                    condition,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_block(
                    body,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::For { iter, body, .. } => {
                Self::collect_effect_sources_from_expr(
                    iter,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_block(
                    body,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
                for e in elements {
                    Self::collect_effect_sources_from_expr(
                        e,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::ArrayRepeat { value, count, .. } => {
                Self::collect_effect_sources_from_expr(
                    value,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    count,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::StructLit { fields, .. } => {
                for (_, value) in fields {
                    Self::collect_effect_sources_from_expr(
                        value,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
                Self::collect_effect_sources_from_expr(
                    base,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Index { base, index, .. } => {
                Self::collect_effect_sources_from_expr(
                    base,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    index,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Cast { expr, .. }
            | Expr::Resume { value: expr, .. }
            | Expr::GpuAnnotated { expr, .. } => {
                Self::collect_effect_sources_from_expr(
                    expr,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Return { value, .. } | Expr::Break { value, .. } => {
                if let Some(v) = value {
                    Self::collect_effect_sources_from_expr(
                        v,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => {
                Self::collect_effect_sources_from_expr(
                    factual,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    intervention,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    outcome,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::KnowledgeExpr {
                value,
                epsilon,
                validity,
                provenance,
                ..
            } => {
                Self::collect_effect_sources_from_expr(
                    value,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                if let Some(eps) = epsilon {
                    Self::collect_effect_sources_from_expr(
                        eps,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
                if let Some(v) = validity {
                    Self::collect_effect_sources_from_expr(
                        v,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
                if let Some(p) = provenance {
                    Self::collect_effect_sources_from_expr(
                        p,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Uncertain {
                value, uncertainty, ..
            } => {
                Self::collect_effect_sources_from_expr(
                    value,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                Self::collect_effect_sources_from_expr(
                    uncertainty,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
            }
            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                Self::collect_effect_sources_from_expr(
                    target,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                for g in given {
                    Self::collect_effect_sources_from_expr(
                        g,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
                for (_, v) in interventions {
                    Self::collect_effect_sources_from_expr(
                        v,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                Self::collect_effect_sources_from_expr(
                    scrutinee,
                    ast,
                    source,
                    effect_names,
                    seen_effects,
                    sources,
                    seen_sources,
                );
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        Self::collect_effect_sources_from_expr(
                            guard,
                            ast,
                            source,
                            effect_names,
                            seen_effects,
                            sources,
                            seen_sources,
                        );
                    }
                    Self::collect_effect_sources_from_expr(
                        &arm.body,
                        ast,
                        source,
                        effect_names,
                        seen_effects,
                        sources,
                        seen_sources,
                    );
                }
            }
            Expr::Literal { .. }
            | Expr::Path { .. }
            | Expr::Continue { .. }
            | Expr::MacroInvocation(_)
            | Expr::Do { .. }
            | Expr::OntologyTerm { .. }
            | Expr::Range { .. } => {}
        }
    }

    fn extract_semantic_facts(
        ast: &Ast,
        source: &str,
        units: &mut HashMap<u32, UnitInfo>,
        epistemic: &mut HashMap<String, EpistemicInfo>,
        refinements: &mut Vec<RefinementInfo>,
    ) {
        for item in &ast.items {
            match item {
                Item::Function(f) => {
                    for param in &f.params {
                        if let Some(name) = Self::binding_name(&param.pattern) {
                            Self::extract_type_facts(
                                name,
                                &param.ty,
                                ast,
                                source,
                                epistemic,
                                refinements,
                            );
                        }
                    }
                    for stmt in &f.body.stmts {
                        Self::extract_stmt_facts(stmt, ast, source, units, epistemic, refinements);
                    }
                }
                Item::Global(g) => {
                    let binding = Self::binding_name(&g.pattern).map(str::to_string);
                    if let (Some(name), Some(ty)) = (&binding, g.ty.as_ref()) {
                        Self::extract_type_facts(name, ty, ast, source, epistemic, refinements);
                    }
                    Self::extract_expr_facts(
                        &g.value,
                        binding.as_deref(),
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
                Item::Impl(impl_def) => {
                    for impl_item in &impl_def.items {
                        if let crate::ast::ImplItem::Fn(f) = impl_item {
                            for param in &f.params {
                                if let Some(name) = Self::binding_name(&param.pattern) {
                                    Self::extract_type_facts(
                                        name,
                                        &param.ty,
                                        ast,
                                        source,
                                        epistemic,
                                        refinements,
                                    );
                                }
                            }
                            for stmt in &f.body.stmts {
                                Self::extract_stmt_facts(
                                    stmt,
                                    ast,
                                    source,
                                    units,
                                    epistemic,
                                    refinements,
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn extract_stmt_facts(
        stmt: &Stmt,
        ast: &Ast,
        source: &str,
        units: &mut HashMap<u32, UnitInfo>,
        epistemic: &mut HashMap<String, EpistemicInfo>,
        refinements: &mut Vec<RefinementInfo>,
    ) {
        match stmt {
            Stmt::Let {
                pattern, ty, value, ..
            } => {
                let binding = Self::binding_name(pattern).map(str::to_string);
                if let (Some(name), Some(ty)) = (&binding, ty.as_ref()) {
                    Self::extract_type_facts(name, ty, ast, source, epistemic, refinements);
                }
                if let Some(expr) = value {
                    Self::extract_expr_facts(
                        expr,
                        binding.as_deref(),
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
            }
            Stmt::Expr { expr, .. } => {
                Self::extract_expr_facts(expr, None, ast, source, units, epistemic, refinements);
            }
            Stmt::Assign { target, value, .. } => {
                let binding = Self::path_binding(target).map(str::to_string);
                Self::extract_expr_facts(target, None, ast, source, units, epistemic, refinements);
                Self::extract_expr_facts(
                    value,
                    binding.as_deref(),
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
            }
            Stmt::Empty | Stmt::MacroInvocation(_) | Stmt::LocalExtern(_) => {}
        }
    }

    fn extract_expr_facts(
        expr: &Expr,
        binding: Option<&str>,
        ast: &Ast,
        source: &str,
        units: &mut HashMap<u32, UnitInfo>,
        epistemic: &mut HashMap<String, EpistemicInfo>,
        refinements: &mut Vec<RefinementInfo>,
    ) {
        match expr {
            Expr::Literal { id, value } => {
                if let Some(unit_name) = Self::literal_unit(value) {
                    units.insert(
                        id.0,
                        UnitInfo {
                            unit: unit_name.to_string(),
                            dimension: Self::infer_dimension(unit_name),
                            is_dimensionless: false,
                        },
                    );
                }
            }
            Expr::KnowledgeExpr {
                epsilon,
                provenance,
                ..
            } => {
                if let Some(name) = binding {
                    let mut info = EpistemicInfo {
                        confidence: 0.8,
                        confidence_lower: None,
                        confidence_upper: None,
                        source: "derived".to_string(),
                        revisable: true,
                        evidence: Vec::new(),
                        provenance: None,
                    };
                    if let Some(eps_expr) = epsilon {
                        if let Some(eps) = Self::numeric_literal(eps_expr) {
                            let confidence = (1.0 - eps.abs()).clamp(0.0, 1.0);
                            info.confidence = confidence;
                            info.confidence_lower = Some((confidence - 0.1).max(0.0));
                            info.confidence_upper = Some((confidence + 0.1).min(1.0));
                        }
                    }
                    if let Some(src_expr) = provenance {
                        let label = Self::expr_label(src_expr);
                        info.source = label.clone();
                        info.provenance = Some(ProvenanceInfo {
                            origin: label,
                            transformations: Vec::new(),
                            dependencies: Vec::new(),
                        });
                    }
                    epistemic.insert(name.to_string(), info);
                }
            }
            Expr::Uncertain { uncertainty, .. } => {
                if let Some(name) = binding {
                    let sigma = Self::numeric_literal(uncertainty).unwrap_or(0.25);
                    let confidence = (1.0 / (1.0 + sigma.abs())).clamp(0.0, 1.0);
                    epistemic.insert(
                        name.to_string(),
                        EpistemicInfo {
                            confidence,
                            confidence_lower: Some((confidence - 0.1).max(0.0)),
                            confidence_upper: Some((confidence + 0.1).min(1.0)),
                            source: "uncertainty-propagated".to_string(),
                            revisable: true,
                            evidence: Vec::new(),
                            provenance: None,
                        },
                    );
                }
            }
            Expr::Cast { expr, ty, .. } => {
                if let Some(name) = binding {
                    Self::extract_type_facts(name, ty, ast, source, epistemic, refinements);
                }
                Self::extract_expr_facts(expr, binding, ast, source, units, epistemic, refinements);
            }
            Expr::Call { callee, args, .. } => {
                Self::extract_expr_facts(callee, None, ast, source, units, epistemic, refinements);
                for arg in args {
                    Self::extract_expr_facts(arg, None, ast, source, units, epistemic, refinements);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                Self::extract_expr_facts(
                    receiver,
                    None,
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
                for arg in args {
                    Self::extract_expr_facts(arg, None, ast, source, units, epistemic, refinements);
                }
            }
            Expr::Binary { left, right, .. } => {
                Self::extract_expr_facts(left, None, ast, source, units, epistemic, refinements);
                Self::extract_expr_facts(right, None, ast, source, units, epistemic, refinements);
            }
            Expr::Unary { expr, .. }
            | Expr::Try { expr, .. }
            | Expr::Await { expr, .. }
            | Expr::Spawn { expr, .. }
            | Expr::Handle { expr, .. }
            | Expr::Resume { value: expr, .. }
            | Expr::GpuAnnotated { expr, .. } => {
                Self::extract_expr_facts(expr, None, ast, source, units, epistemic, refinements);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                Self::extract_expr_facts(
                    condition,
                    None,
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
                for stmt in &then_branch.stmts {
                    Self::extract_stmt_facts(stmt, ast, source, units, epistemic, refinements);
                }
                if let Some(else_expr) = else_branch {
                    Self::extract_expr_facts(
                        else_expr,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
            }
            Expr::Block { block, .. }
            | Expr::Loop { body: block, .. }
            | Expr::UnsafeBlock { block, .. }
            | Expr::AsyncBlock { block, .. } => {
                for stmt in &block.stmts {
                    Self::extract_stmt_facts(stmt, ast, source, units, epistemic, refinements);
                }
            }
            Expr::While {
                condition, body, ..
            } => {
                Self::extract_expr_facts(
                    condition,
                    None,
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
                for stmt in &body.stmts {
                    Self::extract_stmt_facts(stmt, ast, source, units, epistemic, refinements);
                }
            }
            Expr::For { iter, body, .. } => {
                Self::extract_expr_facts(iter, None, ast, source, units, epistemic, refinements);
                for stmt in &body.stmts {
                    Self::extract_stmt_facts(stmt, ast, source, units, epistemic, refinements);
                }
            }
            Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
                for elem in elements {
                    Self::extract_expr_facts(
                        elem,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
            }
            Expr::ArrayRepeat { value, count, .. } => {
                Self::extract_expr_facts(value, None, ast, source, units, epistemic, refinements);
                Self::extract_expr_facts(count, None, ast, source, units, epistemic, refinements);
            }
            Expr::StructLit { fields, .. } => {
                for (_, value) in fields {
                    Self::extract_expr_facts(
                        value,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
            }
            Expr::Index { base, index, .. } => {
                Self::extract_expr_facts(base, None, ast, source, units, epistemic, refinements);
                Self::extract_expr_facts(index, None, ast, source, units, epistemic, refinements);
            }
            Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
                Self::extract_expr_facts(base, None, ast, source, units, epistemic, refinements);
            }
            Expr::Return { value, .. } | Expr::Break { value, .. } => {
                if let Some(v) = value {
                    Self::extract_expr_facts(v, None, ast, source, units, epistemic, refinements);
                }
            }
            Expr::Select { arms, .. } => {
                for arm in arms {
                    Self::extract_expr_facts(
                        &arm.future,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                    Self::extract_expr_facts(
                        &arm.body,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
            }
            Expr::Join { futures, .. } => {
                for future in futures {
                    Self::extract_expr_facts(
                        future,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
            }
            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => {
                Self::extract_expr_facts(factual, None, ast, source, units, epistemic, refinements);
                Self::extract_expr_facts(
                    intervention,
                    None,
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
                Self::extract_expr_facts(outcome, None, ast, source, units, epistemic, refinements);
            }
            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                Self::extract_expr_facts(target, None, ast, source, units, epistemic, refinements);
                for g in given {
                    Self::extract_expr_facts(g, None, ast, source, units, epistemic, refinements);
                }
                for (_, v) in interventions {
                    Self::extract_expr_facts(v, None, ast, source, units, epistemic, refinements);
                }
            }
            Expr::Sample { distribution, .. } => {
                Self::extract_expr_facts(
                    distribution,
                    None,
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
            }
            Expr::Observe {
                data, distribution, ..
            } => {
                Self::extract_expr_facts(data, None, ast, source, units, epistemic, refinements);
                Self::extract_expr_facts(
                    distribution,
                    None,
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
            }
            Expr::Range { start, end, .. } => {
                if let Some(start) = start {
                    Self::extract_expr_facts(
                        start,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
                if let Some(end) = end {
                    Self::extract_expr_facts(end, None, ast, source, units, epistemic, refinements);
                }
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                Self::extract_expr_facts(
                    scrutinee,
                    None,
                    ast,
                    source,
                    units,
                    epistemic,
                    refinements,
                );
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        Self::extract_expr_facts(
                            guard,
                            None,
                            ast,
                            source,
                            units,
                            epistemic,
                            refinements,
                        );
                    }
                    Self::extract_expr_facts(
                        &arm.body,
                        None,
                        ast,
                        source,
                        units,
                        epistemic,
                        refinements,
                    );
                }
            }
            Expr::Literal { .. }
            | Expr::Path { .. }
            | Expr::Continue { .. }
            | Expr::Perform { .. }
            | Expr::MacroInvocation(_)
            | Expr::Do { .. }
            | Expr::OntologyTerm { .. } => {}
        }
    }

    fn extract_type_facts(
        binding: &str,
        ty: &TypeExpr,
        ast: &Ast,
        source: &str,
        epistemic: &mut HashMap<String, EpistemicInfo>,
        refinements: &mut Vec<RefinementInfo>,
    ) {
        match ty {
            TypeExpr::Knowledge {
                epsilon,
                validity,
                provenance,
                ..
            } => {
                let mut info = EpistemicInfo {
                    confidence: 0.85,
                    confidence_lower: None,
                    confidence_upper: None,
                    source: "knowledge-type".to_string(),
                    revisable: validity.is_some(),
                    evidence: Vec::new(),
                    provenance: None,
                };

                if let Some(eps) = epsilon {
                    if let Some(value) = Self::numeric_literal(&eps.value) {
                        let confidence = (1.0 - value.abs()).clamp(0.0, 1.0);
                        info.confidence = confidence;
                        info.confidence_lower = Some((confidence - 0.1).max(0.0));
                        info.confidence_upper = Some((confidence + 0.1).min(1.0));
                    }
                }

                if let Some(prov) = provenance {
                    let origin = if let Some(src) = &prov.source {
                        Self::expr_label(src)
                    } else {
                        format!("{:?}", prov.kind).to_lowercase()
                    };
                    info.source = origin.clone();
                    info.provenance = Some(ProvenanceInfo {
                        origin,
                        transformations: Vec::new(),
                        dependencies: Vec::new(),
                    });
                }

                epistemic.insert(binding.to_string(), info);
            }
            TypeExpr::Refinement {
                var,
                base_type,
                predicate,
            } => {
                let span = ast
                    .node_spans
                    .get(&Self::expr_id(predicate))
                    .copied()
                    .unwrap_or_else(Span::dummy);
                let line = Self::line_from_offset(source, span.start);
                refinements.push(RefinementInfo {
                    variable: binding.to_string(),
                    base_type: Self::type_label(base_type),
                    predicate: format!("{} (binder `{}`)", Self::expr_label(predicate), var),
                    status: SmtStatus::Pending,
                    counterexample: None,
                    span: (line, line),
                });
            }
            TypeExpr::Reference { inner, .. }
            | TypeExpr::RawPointer { inner, .. }
            | TypeExpr::Linear { inner, .. }
            | TypeExpr::Effected { inner, .. }
            | TypeExpr::Forall { inner, .. } => {
                Self::extract_type_facts(binding, inner, ast, source, epistemic, refinements);
            }
            _ => {}
        }
    }

    fn record_effect_source(
        effect: &str,
        expr: &str,
        line: u32,
        effect_names: &mut Vec<String>,
        seen_effects: &mut HashSet<String>,
        sources: &mut Vec<EffectSource>,
        seen_sources: &mut HashSet<(String, u32, String)>,
    ) {
        if seen_effects.insert(effect.to_string()) {
            effect_names.push(effect.to_string());
        }
        let key = (effect.to_string(), line, expr.to_string());
        if seen_sources.insert(key.clone()) {
            sources.push(EffectSource {
                effect: key.0,
                line: key.1,
                expr: key.2,
            });
        }
    }

    fn is_io_call(name: &str) -> bool {
        matches!(
            name,
            "print"
                | "println"
                | "read"
                | "read_file"
                | "write"
                | "write_file"
                | "open"
                | "close"
                | "http_get"
                | "http_post"
                | "fetch"
        )
    }

    fn is_mutating_method(method: &str) -> bool {
        matches!(
            method,
            "push"
                | "pop"
                | "insert"
                | "remove"
                | "clear"
                | "append"
                | "extend"
                | "retain"
                | "sort"
                | "reverse"
        )
    }

    fn call_name(callee: &Expr) -> Option<&str> {
        if let Expr::Path { path, .. } = callee {
            path.name()
        } else {
            None
        }
    }

    fn expr_line(expr: &Expr, ast: &Ast, source: &str) -> u32 {
        let span = ast
            .node_spans
            .get(&Self::expr_id(expr))
            .copied()
            .unwrap_or_else(Span::dummy);
        Self::line_from_offset(source, span.start)
    }

    fn line_from_offset(source: &str, offset: usize) -> u32 {
        let clamped = offset.min(source.len());
        source[..clamped].bytes().filter(|b| *b == b'\n').count() as u32 + 1
    }

    fn binding_name(pattern: &Pattern) -> Option<&str> {
        if let Pattern::Binding { name, .. } = pattern {
            Some(name)
        } else {
            None
        }
    }

    fn path_binding(expr: &Expr) -> Option<&str> {
        if let Expr::Path { path, .. } = expr {
            path.name()
        } else {
            None
        }
    }

    fn literal_unit(lit: &Literal) -> Option<&str> {
        match lit {
            Literal::IntUnit(_, unit) | Literal::FloatUnit(_, unit) => Some(unit.as_str()),
            _ => None,
        }
    }

    fn infer_dimension(unit: &str) -> String {
        if unit.is_empty() || unit == "1" {
            return "dimensionless".to_string();
        }
        let lower = unit.to_ascii_lowercase();
        if matches!(lower.as_str(), "kg" | "g" | "mg" | "ug" | "ng") {
            return "M".to_string();
        }
        if matches!(lower.as_str(), "m" | "cm" | "mm" | "um" | "nm" | "km") {
            return "L".to_string();
        }
        if matches!(lower.as_str(), "s" | "ms" | "min" | "h" | "d") {
            return "T".to_string();
        }
        if matches!(lower.as_str(), "l" | "ml" | "ul") {
            return "L^3".to_string();
        }
        if lower.contains('/') {
            return "ratio".to_string();
        }
        "composite".to_string()
    }

    fn numeric_literal(expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Literal { value, .. } => match value {
                Literal::Int(v) | Literal::TypedInt(v, _) | Literal::IntUnit(v, _) => {
                    Some(*v as f64)
                }
                Literal::Float(v) | Literal::TypedFloat(v, _) | Literal::FloatUnit(v, _) => {
                    Some(*v)
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn type_label(ty: &TypeExpr) -> String {
        match ty {
            TypeExpr::Unit => "()".to_string(),
            TypeExpr::Never => "!".to_string(),
            TypeExpr::SelfType => "Self".to_string(),
            TypeExpr::Named { path, args, unit } => {
                let mut out = path.to_string();
                if !args.is_empty() {
                    let args_str = args
                        .iter()
                        .map(Self::type_label)
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push('<');
                    out.push_str(&args_str);
                    out.push('>');
                }
                if let Some(unit) = unit {
                    out.push('@');
                    out.push_str(unit);
                }
                out
            }
            TypeExpr::Reference { mutable, inner } => {
                if *mutable {
                    format!("&!{}", Self::type_label(inner))
                } else {
                    format!("&{}", Self::type_label(inner))
                }
            }
            TypeExpr::RawPointer { mutable, inner } => {
                if *mutable {
                    format!("*mut {}", Self::type_label(inner))
                } else {
                    format!("*const {}", Self::type_label(inner))
                }
            }
            TypeExpr::Array { element, .. } => format!("[{}]", Self::type_label(element)),
            TypeExpr::Tuple(items) => {
                let parts = items
                    .iter()
                    .map(Self::type_label)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", parts)
            }
            TypeExpr::Function {
                params,
                return_type,
                ..
            } => {
                let params = params
                    .iter()
                    .map(Self::type_label)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("fn({}) -> {}", params, Self::type_label(return_type))
            }
            TypeExpr::Knowledge { value_type, .. } => {
                format!("Knowledge[{}]", Self::type_label(value_type))
            }
            TypeExpr::Quantity { numeric_type, unit } => {
                format!(
                    "Quantity[{}, {}]",
                    Self::type_label(numeric_type),
                    Self::unit_label(unit)
                )
            }
            TypeExpr::Refinement { base_type, .. } => {
                format!("{{ {} | ... }}", Self::type_label(base_type))
            }
            _ => format!("{:?}", ty),
        }
    }

    fn unit_label(unit: &UnitExpr) -> String {
        if unit.base_units.is_empty() {
            return "1".to_string();
        }
        let mut pieces = Vec::new();
        for (name, power) in &unit.base_units {
            if *power == 1 {
                pieces.push(name.clone());
            } else {
                pieces.push(format!("{}^{}", name, power));
            }
        }
        pieces.join("*")
    }

    fn expr_label(expr: &Expr) -> String {
        match expr {
            Expr::Literal { value, .. } => match value {
                Literal::Bool(v) => v.to_string(),
                Literal::Int(v) => v.to_string(),
                Literal::Float(v) => v.to_string(),
                Literal::String(v) => format!("{:?}", v),
                Literal::IntUnit(v, unit) => format!("{}_{}", v, unit),
                Literal::FloatUnit(v, unit) => format!("{}_{}", v, unit),
                _ => format!("{:?}", value),
            },
            Expr::Path { path, .. } => path.to_string(),
            Expr::Call { callee, .. } => format!("{}(...)", Self::expr_label(callee)),
            Expr::MethodCall { method, .. } => format!(".{}(...)", method),
            Expr::Binary {
                op, left, right, ..
            } => format!(
                "({} {:?} {})",
                Self::expr_label(left),
                op,
                Self::expr_label(right)
            ),
            Expr::Unary { op, expr, .. } => format!("{:?}{}", op, Self::expr_label(expr)),
            _ => format!("{:?}", expr),
        }
    }

    fn expr_id(expr: &Expr) -> NodeId {
        match expr {
            Expr::Literal { id, .. } => *id,
            Expr::Path { id, .. } => *id,
            Expr::Binary { id, .. } => *id,
            Expr::Unary { id, .. } => *id,
            Expr::Call { id, .. } => *id,
            Expr::MethodCall { id, .. } => *id,
            Expr::Field { id, .. } => *id,
            Expr::TupleField { id, .. } => *id,
            Expr::Index { id, .. } => *id,
            Expr::Cast { id, .. } => *id,
            Expr::Block { id, .. } => *id,
            Expr::If { id, .. } => *id,
            Expr::Match { id, .. } => *id,
            Expr::Loop { id, .. } => *id,
            Expr::While { id, .. } => *id,
            Expr::For { id, .. } => *id,
            Expr::Return { id, .. } => *id,
            Expr::Break { id, .. } => *id,
            Expr::Continue { id, .. } => *id,
            Expr::Closure { id, .. } => *id,
            Expr::Tuple { id, .. } => *id,
            Expr::Array { id, .. } => *id,
            Expr::ArrayRepeat { id, .. } => *id,
            Expr::Range { id, .. } => *id,
            Expr::StructLit { id, .. } => *id,
            Expr::Try { id, .. } => *id,
            Expr::Perform { id, .. } => *id,
            Expr::Handle { id, .. } => *id,
            Expr::Resume { id, .. } => *id,
            Expr::Sample { id, .. } => *id,
            Expr::Await { id, .. } => *id,
            Expr::AsyncBlock { id, .. } => *id,
            Expr::UnsafeBlock { id, .. } => *id,
            Expr::AsyncClosure { id, .. } => *id,
            Expr::Spawn { id, .. } => *id,
            Expr::Select { id, .. } => *id,
            Expr::Join { id, .. } => *id,
            Expr::MacroInvocation(m) => m.id,
            Expr::Do { id, .. } => *id,
            Expr::Counterfactual { id, .. } => *id,
            Expr::KnowledgeExpr { id, .. } => *id,
            Expr::Uncertain { id, .. } => *id,
            Expr::GpuAnnotated { id, .. } => *id,
            Expr::Observe { id, .. } => *id,
            Expr::Query { id, .. } => *id,
            Expr::OntologyTerm { id, .. } => *id,
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
