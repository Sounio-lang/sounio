//! Semantic analysis for LSP features
//!
//! Manages parsing, name resolution, and type checking with caching.
//! Uses LspQueryDatabase for demand-driven incremental analysis.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::lsp_types::*;

use super::completion::CompletionProvider;
use super::definition::DefinitionProvider;
use super::diagnostics::DiagnosticsProvider;
use super::document::Document;
use super::hover::HoverProvider;
use super::inlay_hints::InlayHintProvider;
use super::query_bridge::{LspQueryDatabase, SmtStatus};
use super::references::ReferencesProvider;
use super::rich_info::confidence_badge;
use super::semantic_tokens::SemanticTokensProvider;
use super::workspace::Workspace;

use crate::ast::Ast;
use crate::common::Span;
use crate::fmt::Formatter as SourceFormatter;
use crate::lexer;
use crate::resolve::SymbolTable;

/// Cached analysis result for a document
#[derive(Debug)]
pub struct AnalysisResult {
    /// Parsed AST
    pub ast: Option<Ast>,

    /// Symbol table from name resolution
    pub symbols: Option<SymbolTable>,

    /// Diagnostics from all phases
    pub diagnostics: Vec<Diagnostic>,

    /// Source version when analyzed
    pub version: i32,
}

impl Default for AnalysisResult {
    fn default() -> Self {
        Self {
            ast: None,
            symbols: None,
            diagnostics: Vec::new(),
            version: 0,
        }
    }
}

/// Analysis host manages incremental analysis and caching
///
/// Uses LspQueryDatabase for demand-driven incremental analysis,
/// with a legacy cache for backward compatibility with providers.
pub struct AnalysisHost {
    /// Query-based incremental database (primary)
    query_db: LspQueryDatabase,

    /// Legacy cached results per file (for backward compat with providers)
    cache: HashMap<Url, AnalysisResult>,

    /// Completion provider
    completion: CompletionProvider,

    /// Hover provider
    hover: HoverProvider,

    /// Definition provider
    definition: DefinitionProvider,

    /// References provider
    references: ReferencesProvider,

    /// Semantic tokens provider
    semantic_tokens: SemanticTokensProvider,

    /// Diagnostics provider
    diagnostics_provider: DiagnosticsProvider,

    /// Inlay hints provider
    inlay_hints: InlayHintProvider,

    /// Workspace reference for cross-file features
    workspace: Option<Arc<RwLock<Workspace>>>,
}

impl AnalysisHost {
    /// Create a new analysis host
    pub fn new() -> Self {
        Self {
            query_db: LspQueryDatabase::new(),
            cache: HashMap::new(),
            completion: CompletionProvider::new(),
            hover: HoverProvider::new(),
            definition: DefinitionProvider::new(),
            references: ReferencesProvider::new(),
            semantic_tokens: SemanticTokensProvider::new(),
            diagnostics_provider: DiagnosticsProvider::new(),
            inlay_hints: InlayHintProvider::new(),
            workspace: None,
        }
    }

    /// Create a new analysis host with workspace
    pub fn with_workspace(workspace: Arc<RwLock<Workspace>>) -> Self {
        Self {
            query_db: LspQueryDatabase::new(),
            cache: HashMap::new(),
            completion: CompletionProvider::new(),
            hover: HoverProvider::new(),
            definition: DefinitionProvider::new(),
            references: ReferencesProvider::new(),
            semantic_tokens: SemanticTokensProvider::new(),
            diagnostics_provider: DiagnosticsProvider::new(),
            inlay_hints: InlayHintProvider::new(),
            workspace: Some(workspace),
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

    /// Analyze a document and return diagnostics
    ///
    /// Uses the query database for incremental analysis, then populates
    /// the legacy cache for backward compatibility with existing providers.
    pub fn analyze(&mut self, source: &str, uri: &Url) -> Vec<Diagnostic> {
        // Update the query database with new file contents
        self.query_db.file_changed(uri, source);

        // Get analysis result from query database
        let result = self.query_db.analyze(uri);

        // Convert to legacy cache format for backward compatibility
        let ast = result.ast.as_ref().map(|a| (**a).clone());
        let symbols = result.symbols.clone();
        let diagnostics = result.diagnostics.clone();

        // Populate legacy cache
        self.cache.insert(
            uri.clone(),
            AnalysisResult {
                ast,
                symbols,
                diagnostics: diagnostics.clone(),
                version: 0,
            },
        );

        diagnostics
    }

    /// Get the query database (for advanced features)
    pub fn query_db(&self) -> &LspQueryDatabase {
        &self.query_db
    }

    /// Get mutable query database
    pub fn query_db_mut(&mut self) -> &mut LspQueryDatabase {
        &mut self.query_db
    }

    /// Notify that a file was closed
    pub fn file_closed(&mut self, uri: &Url) {
        self.query_db.file_closed(uri);
        self.cache.remove(uri);
    }

    /// Get effect information for a function (from query database)
    pub fn get_effect_info(
        &mut self,
        uri: &Url,
        function_name: &str,
    ) -> Option<super::query_bridge::EffectInfo> {
        self.query_db.get_effects(uri, function_name)
    }

    /// Get unit information for an expression (from query database)
    pub fn get_unit_info(
        &mut self,
        uri: &Url,
        node_id: u32,
    ) -> Option<super::query_bridge::UnitInfo> {
        self.query_db.get_unit(uri, node_id)
    }

    /// Get epistemic information for a variable (from query database)
    pub fn get_epistemic_info(
        &mut self,
        uri: &Url,
        var_name: &str,
    ) -> Option<super::query_bridge::EpistemicInfo> {
        self.query_db.get_epistemic(uri, var_name)
    }

    /// Get hover information at a position
    ///
    /// This method combines:
    /// - Basic symbol information (type, kind)
    /// - Effect annotations (IO, Mut, etc.)
    /// - Epistemic status (confidence, source)
    /// - Unit information (dimensional analysis)
    /// - Refinement predicates (when available)
    pub fn hover(&self, doc: &Document, pos: Position, uri: &Url) -> Option<Hover> {
        let (word, range) = doc.word_at(pos)?;

        // Check cached analysis
        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                // Get basic symbol hover
                let base_hover = self.hover.hover_for_symbol(&word, range, symbols);

                // Enhance with rich information from query_db
                return self.enhance_hover_with_rich_info(base_hover, &word, range, uri);
            }
        }

        // Fallback to keyword hover
        self.hover.hover_for_keyword(&word, range)
    }

    /// Enhance a hover with effect, epistemic, and unit information
    fn enhance_hover_with_rich_info(
        &self,
        base_hover: Option<Hover>,
        word: &str,
        range: Range,
        uri: &Url,
    ) -> Option<Hover> {
        // Get the base hover content
        let base_content = match &base_hover {
            Some(hover) => match &hover.contents {
                HoverContents::Markup(m) => m.value.clone(),
                HoverContents::Scalar(MarkedString::String(s)) => s.clone(),
                HoverContents::Scalar(MarkedString::LanguageString(ls)) => {
                    format!("```{}\n{}\n```", ls.language, ls.value)
                }
                HoverContents::Array(arr) => arr
                    .iter()
                    .map(|m| match m {
                        MarkedString::String(s) => s.clone(),
                        MarkedString::LanguageString(ls) => {
                            format!("```{}\n{}\n```", ls.language, ls.value)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            },
            None => String::new(),
        };

        // Build enhanced content
        let mut content = base_content;

        // Try to get effect information (for functions)
        if let Some(analysis_result) = self.cache.get(uri) {
            // Check if this is a function with effects
            if let Some(effect_info) = analysis_result
                .ast
                .as_ref()
                .and_then(|_ast| self.get_effect_info_for_symbol(word, uri))
            {
                if !effect_info.effects.is_empty() {
                    let effects_str = effect_info.effects.join(", ");
                    content.push_str(&format!("\n\n**Effects:** `{}`", effects_str));

                    // Add effect sources if available
                    if !effect_info.sources.is_empty() {
                        content.push_str("\n");
                        for src in &effect_info.sources {
                            content.push_str(&format!(
                                "\n- `{}` from line {} (`{}`)",
                                src.effect, src.line, src.expr
                            ));
                        }
                    }
                } else {
                    content.push_str("\n\n**Effects:** *pure* (no side effects)");
                }
            }
        }

        // Try to get epistemic information (for variables)
        if let Some(epistemic_info) = self.get_epistemic_info_for_symbol(word, uri) {
            let badge = confidence_badge(epistemic_info.confidence);
            content.push_str(&format!(
                "\n\n**Confidence:** {} {:.0}%",
                badge,
                epistemic_info.confidence * 100.0
            ));

            // Add confidence bounds if available
            if let (Some(lower), Some(upper)) = (
                epistemic_info.confidence_lower,
                epistemic_info.confidence_upper,
            ) {
                content.push_str(&format!(" [{:.0}% - {:.0}%]", lower * 100.0, upper * 100.0));
            }

            // Add source information
            if !epistemic_info.source.is_empty() {
                content.push_str(&format!("\n**Source:** {}", epistemic_info.source));
            }

            // Add revisability
            if epistemic_info.revisable {
                content.push_str("\n**Revisability:** revisable");
            } else {
                content.push_str("\n**Revisability:** non-revisable");
            }

            // Add evidence if available
            if !epistemic_info.evidence.is_empty() {
                content.push_str("\n**Evidence:**");
                for ev in &epistemic_info.evidence {
                    content.push_str(&format!(
                        "\n- {}: {} (strength: {:.0}%)",
                        ev.kind,
                        ev.reference,
                        ev.strength * 100.0
                    ));
                }
            }
        }

        // Try to get unit information
        if let Some(unit_info) = self.get_unit_info_for_symbol(word, uri) {
            content.push_str(&format!("\n\n**Unit:** `{}`", unit_info.unit));
            if !unit_info.dimension.is_empty() {
                content.push_str(&format!(" (dimension: {})", unit_info.dimension));
            }
            if unit_info.is_dimensionless {
                content.push_str(" (dimensionless)");
            }
        }

        // Try to get refinement information
        if let Some(refinement_info) = self.get_refinement_info_for_symbol(word, uri) {
            content.push_str(&format!(
                "\n\n**Refinement:** `{{ {}: {} | {} }}`",
                refinement_info.variable, refinement_info.base_type, refinement_info.predicate
            ));

            let status_str = match refinement_info.status {
                SmtStatus::Verified => "verified",
                SmtStatus::Failed => "failed",
                SmtStatus::Unknown => "unknown",
                SmtStatus::Pending => "pending",
            };
            content.push_str(&format!("\n**SMT Status:** {}", status_str));

            if let Some(ref cex) = refinement_info.counterexample {
                content.push_str("\n**Counterexample:**");
                for (var, val) in &cex.bindings {
                    content.push_str(&format!("\n- {} = {}", var, val));
                }
            }
        }

        // Return enhanced hover or base hover
        if content.is_empty() && base_hover.is_none() {
            return None;
        }

        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: Some(range),
        })
    }

    /// Get effect information for a symbol (function name)
    fn get_effect_info_for_symbol(
        &self,
        name: &str,
        uri: &Url,
    ) -> Option<super::query_bridge::EffectInfo> {
        // Query the analysis result for effect information
        let analysis_result = self.cache.get(uri)?;

        // Look up in the query database cache
        // Note: In the current implementation, we need to look at the AST
        // to find functions and their declared effects
        if let Some(ref ast) = analysis_result.ast {
            for item in &ast.items {
                if let crate::ast::Item::Function(f) = item {
                    if f.name == name {
                        return Some(super::query_bridge::EffectInfo {
                            effects: f.effects.iter().map(|e| e.name.to_string()).collect(),
                            sources: vec![],
                        });
                    }
                }
            }
        }

        None
    }

    /// Get epistemic information for a symbol (variable name)
    fn get_epistemic_info_for_symbol(
        &self,
        name: &str,
        uri: &Url,
    ) -> Option<super::query_bridge::EpistemicInfo> {
        let analysis_result = self.cache.get(uri)?;
        let ast = analysis_result.ast.as_ref()?;

        for item in &ast.items {
            match item {
                crate::ast::Item::Function(f) => {
                    for param in &f.params {
                        if Self::binding_name(&param.pattern) == Some(name) {
                            if let Some(info) = Self::epistemic_from_type(&param.ty) {
                                return Some(info);
                            }
                        }
                    }
                    if let Some(info) = Self::epistemic_from_block(&f.body, name) {
                        return Some(info);
                    }
                }
                crate::ast::Item::Global(g) => {
                    if Self::binding_name(&g.pattern) == Some(name) {
                        if let Some(ty) = &g.ty {
                            if let Some(info) = Self::epistemic_from_type(ty) {
                                return Some(info);
                            }
                        }
                        if let Some(info) = Self::epistemic_from_expr(&g.value) {
                            return Some(info);
                        }
                    }
                }
                crate::ast::Item::Impl(impl_def) => {
                    for impl_item in &impl_def.items {
                        if let crate::ast::ImplItem::Fn(f) = impl_item {
                            for param in &f.params {
                                if Self::binding_name(&param.pattern) == Some(name) {
                                    if let Some(info) = Self::epistemic_from_type(&param.ty) {
                                        return Some(info);
                                    }
                                }
                            }
                            if let Some(info) = Self::epistemic_from_block(&f.body, name) {
                                return Some(info);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Get unit information for a symbol
    fn get_unit_info_for_symbol(
        &self,
        name: &str,
        uri: &Url,
    ) -> Option<super::query_bridge::UnitInfo> {
        let analysis_result = self.cache.get(uri)?;
        let ast = analysis_result.ast.as_ref()?;

        for item in &ast.items {
            match item {
                crate::ast::Item::Function(f) => {
                    for param in &f.params {
                        if Self::binding_name(&param.pattern) == Some(name) {
                            if let Some(info) = Self::unit_from_type(&param.ty) {
                                return Some(info);
                            }
                        }
                    }
                    if let Some(info) = Self::unit_from_block(&f.body, name) {
                        return Some(info);
                    }
                }
                crate::ast::Item::Global(g) => {
                    if Self::binding_name(&g.pattern) == Some(name) {
                        if let Some(ty) = &g.ty {
                            if let Some(info) = Self::unit_from_type(ty) {
                                return Some(info);
                            }
                        }
                        if let Some(info) = Self::unit_from_expr(&g.value) {
                            return Some(info);
                        }
                    }
                }
                crate::ast::Item::Impl(impl_def) => {
                    for impl_item in &impl_def.items {
                        if let crate::ast::ImplItem::Fn(f) = impl_item {
                            for param in &f.params {
                                if Self::binding_name(&param.pattern) == Some(name) {
                                    if let Some(info) = Self::unit_from_type(&param.ty) {
                                        return Some(info);
                                    }
                                }
                            }
                            if let Some(info) = Self::unit_from_block(&f.body, name) {
                                return Some(info);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Get refinement information for a symbol
    fn get_refinement_info_for_symbol(
        &self,
        name: &str,
        uri: &Url,
    ) -> Option<super::query_bridge::RefinementInfo> {
        let analysis_result = self.cache.get(uri)?;
        let ast = analysis_result.ast.as_ref()?;

        for item in &ast.items {
            match item {
                crate::ast::Item::Function(f) => {
                    for param in &f.params {
                        if Self::binding_name(&param.pattern) == Some(name) {
                            if let Some(info) = Self::refinement_from_type(name, &param.ty) {
                                return Some(info);
                            }
                        }
                    }
                    if let Some(info) = Self::refinement_from_block(&f.body, name) {
                        return Some(info);
                    }
                }
                crate::ast::Item::Global(g) => {
                    if Self::binding_name(&g.pattern) == Some(name) {
                        if let Some(ty) = &g.ty {
                            if let Some(info) = Self::refinement_from_type(name, ty) {
                                return Some(info);
                            }
                        }
                    }
                }
                crate::ast::Item::Impl(impl_def) => {
                    for impl_item in &impl_def.items {
                        if let crate::ast::ImplItem::Fn(f) = impl_item {
                            for param in &f.params {
                                if Self::binding_name(&param.pattern) == Some(name) {
                                    if let Some(info) = Self::refinement_from_type(name, &param.ty)
                                    {
                                        return Some(info);
                                    }
                                }
                            }
                            if let Some(info) = Self::refinement_from_block(&f.body, name) {
                                return Some(info);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn binding_name(pattern: &crate::ast::Pattern) -> Option<&str> {
        if let crate::ast::Pattern::Binding { name, .. } = pattern {
            Some(name)
        } else {
            None
        }
    }

    fn numeric_literal(expr: &crate::ast::Expr) -> Option<f64> {
        match expr {
            crate::ast::Expr::Literal { value, .. } => match value {
                crate::ast::Literal::Int(v)
                | crate::ast::Literal::TypedInt(v, _)
                | crate::ast::Literal::IntUnit(v, _) => Some(*v as f64),
                crate::ast::Literal::Float(v)
                | crate::ast::Literal::TypedFloat(v, _)
                | crate::ast::Literal::FloatUnit(v, _) => Some(*v),
                _ => None,
            },
            _ => None,
        }
    }

    fn unit_expr_label(unit: &crate::ast::UnitExpr) -> String {
        if unit.base_units.is_empty() {
            return "1".to_string();
        }
        unit.base_units
            .iter()
            .map(|(name, power)| {
                if *power == 1 {
                    name.clone()
                } else {
                    format!("{}^{}", name, power)
                }
            })
            .collect::<Vec<_>>()
            .join("*")
    }

    fn infer_unit_dimension(unit: &str) -> String {
        if unit.is_empty() || unit == "1" {
            return "dimensionless".to_string();
        }
        let u = unit.to_ascii_lowercase();
        if matches!(u.as_str(), "kg" | "g" | "mg" | "ug" | "ng") {
            return "M".to_string();
        }
        if matches!(u.as_str(), "m" | "cm" | "mm" | "um" | "nm" | "km") {
            return "L".to_string();
        }
        if matches!(u.as_str(), "s" | "ms" | "min" | "h" | "d") {
            return "T".to_string();
        }
        if matches!(u.as_str(), "l" | "ml" | "ul") {
            return "L^3".to_string();
        }
        if u.contains('/') {
            return "ratio".to_string();
        }
        "composite".to_string()
    }

    fn epistemic_from_type(
        ty: &crate::ast::TypeExpr,
    ) -> Option<super::query_bridge::EpistemicInfo> {
        if let crate::ast::TypeExpr::Knowledge {
            epsilon,
            validity,
            provenance,
            ..
        } = ty
        {
            let mut info = super::query_bridge::EpistemicInfo {
                confidence: 0.85,
                confidence_lower: None,
                confidence_upper: None,
                source: "knowledge-type".to_string(),
                revisable: validity.is_some(),
                evidence: Vec::new(),
                provenance: None,
            };

            if let Some(eps) = epsilon {
                if let Some(v) = Self::numeric_literal(&eps.value) {
                    let confidence = (1.0 - v.abs()).clamp(0.0, 1.0);
                    info.confidence = confidence;
                    info.confidence_lower = Some((confidence - 0.1).max(0.0));
                    info.confidence_upper = Some((confidence + 0.1).min(1.0));
                }
            }

            if let Some(prov) = provenance {
                let source = prov
                    .source
                    .as_ref()
                    .map(|e| Self::expr_label(e))
                    .unwrap_or_else(|| format!("{:?}", prov.kind).to_lowercase());
                info.source = source.clone();
                info.provenance = Some(super::query_bridge::ProvenanceInfo {
                    origin: source,
                    transformations: Vec::new(),
                    dependencies: Vec::new(),
                });
            }

            return Some(info);
        }

        None
    }

    fn epistemic_from_expr(expr: &crate::ast::Expr) -> Option<super::query_bridge::EpistemicInfo> {
        match expr {
            crate::ast::Expr::KnowledgeExpr {
                epsilon,
                provenance,
                ..
            } => {
                let mut info = super::query_bridge::EpistemicInfo {
                    confidence: 0.8,
                    confidence_lower: None,
                    confidence_upper: None,
                    source: "derived".to_string(),
                    revisable: true,
                    evidence: Vec::new(),
                    provenance: None,
                };
                if let Some(eps) = epsilon {
                    if let Some(v) = Self::numeric_literal(eps) {
                        let confidence = (1.0 - v.abs()).clamp(0.0, 1.0);
                        info.confidence = confidence;
                        info.confidence_lower = Some((confidence - 0.1).max(0.0));
                        info.confidence_upper = Some((confidence + 0.1).min(1.0));
                    }
                }
                if let Some(src) = provenance {
                    let origin = Self::expr_label(src);
                    info.source = origin.clone();
                    info.provenance = Some(super::query_bridge::ProvenanceInfo {
                        origin,
                        transformations: Vec::new(),
                        dependencies: Vec::new(),
                    });
                }
                Some(info)
            }
            crate::ast::Expr::Uncertain { uncertainty, .. } => {
                let sigma = Self::numeric_literal(uncertainty).unwrap_or(0.25);
                let confidence = (1.0 / (1.0 + sigma.abs())).clamp(0.0, 1.0);
                Some(super::query_bridge::EpistemicInfo {
                    confidence,
                    confidence_lower: Some((confidence - 0.1).max(0.0)),
                    confidence_upper: Some((confidence + 0.1).min(1.0)),
                    source: "uncertainty-propagated".to_string(),
                    revisable: true,
                    evidence: Vec::new(),
                    provenance: None,
                })
            }
            _ => None,
        }
    }

    fn epistemic_from_block(
        block: &crate::ast::Block,
        name: &str,
    ) -> Option<super::query_bridge::EpistemicInfo> {
        for stmt in &block.stmts {
            if let Some(info) = Self::epistemic_from_stmt(stmt, name) {
                return Some(info);
            }
        }
        None
    }

    fn epistemic_from_stmt(
        stmt: &crate::ast::Stmt,
        name: &str,
    ) -> Option<super::query_bridge::EpistemicInfo> {
        match stmt {
            crate::ast::Stmt::Let {
                pattern, ty, value, ..
            } => {
                if Self::binding_name(pattern) == Some(name) {
                    if let Some(ty) = ty {
                        if let Some(info) = Self::epistemic_from_type(ty) {
                            return Some(info);
                        }
                    }
                    if let Some(expr) = value {
                        if let Some(info) = Self::epistemic_from_expr(expr) {
                            return Some(info);
                        }
                    }
                }
                if let Some(expr) = value {
                    return Self::epistemic_in_expr(expr, name);
                }
                None
            }
            crate::ast::Stmt::Expr { expr, .. } => Self::epistemic_in_expr(expr, name),
            crate::ast::Stmt::Assign { target, value, .. } => {
                if let crate::ast::Expr::Path { path, .. } = target {
                    if path.name() == Some(name) {
                        return Self::epistemic_from_expr(value);
                    }
                }
                Self::epistemic_in_expr(value, name)
            }
            crate::ast::Stmt::Empty
            | crate::ast::Stmt::MacroInvocation(_)
            | crate::ast::Stmt::LocalExtern(_) => None,
        }
    }

    fn epistemic_in_expr(
        expr: &crate::ast::Expr,
        name: &str,
    ) -> Option<super::query_bridge::EpistemicInfo> {
        match expr {
            crate::ast::Expr::Block { block, .. }
            | crate::ast::Expr::Loop { body: block, .. }
            | crate::ast::Expr::AsyncBlock { block, .. }
            | crate::ast::Expr::UnsafeBlock { block, .. } => {
                Self::epistemic_from_block(block, name)
            }
            crate::ast::Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                if let Some(info) = Self::epistemic_from_block(then_branch, name) {
                    return Some(info);
                }
                if let Some(else_expr) = else_branch {
                    return Self::epistemic_in_expr(else_expr, name);
                }
                None
            }
            crate::ast::Expr::Match { arms, .. } => {
                for arm in arms {
                    if let Some(info) = Self::epistemic_in_expr(&arm.body, name) {
                        return Some(info);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn unit_from_type(ty: &crate::ast::TypeExpr) -> Option<super::query_bridge::UnitInfo> {
        match ty {
            crate::ast::TypeExpr::Quantity { unit, .. } => {
                let unit_label = Self::unit_expr_label(unit);
                Some(super::query_bridge::UnitInfo {
                    dimension: Self::infer_unit_dimension(&unit_label),
                    is_dimensionless: unit_label == "1",
                    unit: unit_label,
                })
            }
            crate::ast::TypeExpr::Named {
                unit: Some(unit), ..
            } => Some(super::query_bridge::UnitInfo {
                unit: unit.clone(),
                dimension: Self::infer_unit_dimension(unit),
                is_dimensionless: unit == "1",
            }),
            _ => None,
        }
    }

    fn unit_from_expr(expr: &crate::ast::Expr) -> Option<super::query_bridge::UnitInfo> {
        if let crate::ast::Expr::Literal { value, .. } = expr {
            let unit = match value {
                crate::ast::Literal::IntUnit(_, unit) | crate::ast::Literal::FloatUnit(_, unit) => {
                    unit.clone()
                }
                _ => return None,
            };
            return Some(super::query_bridge::UnitInfo {
                dimension: Self::infer_unit_dimension(&unit),
                is_dimensionless: unit == "1",
                unit,
            });
        }
        None
    }

    fn unit_from_block(
        block: &crate::ast::Block,
        name: &str,
    ) -> Option<super::query_bridge::UnitInfo> {
        for stmt in &block.stmts {
            if let Some(info) = Self::unit_from_stmt(stmt, name) {
                return Some(info);
            }
        }
        None
    }

    fn unit_from_stmt(
        stmt: &crate::ast::Stmt,
        name: &str,
    ) -> Option<super::query_bridge::UnitInfo> {
        match stmt {
            crate::ast::Stmt::Let {
                pattern, ty, value, ..
            } => {
                if Self::binding_name(pattern) == Some(name) {
                    if let Some(ty) = ty {
                        if let Some(info) = Self::unit_from_type(ty) {
                            return Some(info);
                        }
                    }
                    if let Some(value) = value {
                        if let Some(info) = Self::unit_from_expr(value) {
                            return Some(info);
                        }
                    }
                }
                if let Some(value) = value {
                    return Self::unit_in_expr(value, name);
                }
                None
            }
            crate::ast::Stmt::Expr { expr, .. } => Self::unit_in_expr(expr, name),
            crate::ast::Stmt::Assign { target, value, .. } => {
                if let crate::ast::Expr::Path { path, .. } = target {
                    if path.name() == Some(name) {
                        return Self::unit_from_expr(value);
                    }
                }
                Self::unit_in_expr(value, name)
            }
            crate::ast::Stmt::Empty
            | crate::ast::Stmt::MacroInvocation(_)
            | crate::ast::Stmt::LocalExtern(_) => None,
        }
    }

    fn unit_in_expr(expr: &crate::ast::Expr, name: &str) -> Option<super::query_bridge::UnitInfo> {
        match expr {
            crate::ast::Expr::Block { block, .. }
            | crate::ast::Expr::Loop { body: block, .. }
            | crate::ast::Expr::AsyncBlock { block, .. }
            | crate::ast::Expr::UnsafeBlock { block, .. } => Self::unit_from_block(block, name),
            crate::ast::Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                if let Some(info) = Self::unit_from_block(then_branch, name) {
                    return Some(info);
                }
                if let Some(else_expr) = else_branch {
                    return Self::unit_in_expr(else_expr, name);
                }
                None
            }
            crate::ast::Expr::Match { arms, .. } => {
                for arm in arms {
                    if let Some(info) = Self::unit_in_expr(&arm.body, name) {
                        return Some(info);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn refinement_from_type(
        binding: &str,
        ty: &crate::ast::TypeExpr,
    ) -> Option<super::query_bridge::RefinementInfo> {
        if let crate::ast::TypeExpr::Refinement {
            var,
            base_type,
            predicate,
        } = ty
        {
            return Some(super::query_bridge::RefinementInfo {
                variable: binding.to_string(),
                base_type: format_type(base_type),
                predicate: format!("{} (binder `{}`)", Self::expr_label(predicate), var),
                status: SmtStatus::Pending,
                counterexample: None,
                span: (0, 0),
            });
        }
        None
    }

    fn refinement_from_block(
        block: &crate::ast::Block,
        name: &str,
    ) -> Option<super::query_bridge::RefinementInfo> {
        for stmt in &block.stmts {
            if let Some(info) = Self::refinement_from_stmt(stmt, name) {
                return Some(info);
            }
        }
        None
    }

    fn refinement_from_stmt(
        stmt: &crate::ast::Stmt,
        name: &str,
    ) -> Option<super::query_bridge::RefinementInfo> {
        match stmt {
            crate::ast::Stmt::Let { pattern, ty, .. } => {
                if Self::binding_name(pattern) == Some(name) {
                    return ty
                        .as_ref()
                        .and_then(|ty| Self::refinement_from_type(name, ty));
                }
                None
            }
            crate::ast::Stmt::Expr { expr, .. } => Self::refinement_in_expr(expr, name),
            crate::ast::Stmt::Assign { value, .. } => Self::refinement_in_expr(value, name),
            crate::ast::Stmt::Empty
            | crate::ast::Stmt::MacroInvocation(_)
            | crate::ast::Stmt::LocalExtern(_) => None,
        }
    }

    fn refinement_in_expr(
        expr: &crate::ast::Expr,
        name: &str,
    ) -> Option<super::query_bridge::RefinementInfo> {
        match expr {
            crate::ast::Expr::Block { block, .. }
            | crate::ast::Expr::Loop { body: block, .. }
            | crate::ast::Expr::AsyncBlock { block, .. }
            | crate::ast::Expr::UnsafeBlock { block, .. } => {
                Self::refinement_from_block(block, name)
            }
            crate::ast::Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                if let Some(info) = Self::refinement_from_block(then_branch, name) {
                    return Some(info);
                }
                if let Some(else_expr) = else_branch {
                    return Self::refinement_in_expr(else_expr, name);
                }
                None
            }
            crate::ast::Expr::Match { arms, .. } => {
                for arm in arms {
                    if let Some(info) = Self::refinement_in_expr(&arm.body, name) {
                        return Some(info);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn expr_label(expr: &crate::ast::Expr) -> String {
        match expr {
            crate::ast::Expr::Literal { value, .. } => match value {
                crate::ast::Literal::Bool(v) => v.to_string(),
                crate::ast::Literal::Int(v) => v.to_string(),
                crate::ast::Literal::Float(v) => v.to_string(),
                crate::ast::Literal::String(v) => format!("{:?}", v),
                crate::ast::Literal::IntUnit(v, unit) => format!("{}_{}", v, unit),
                crate::ast::Literal::FloatUnit(v, unit) => format!("{}_{}", v, unit),
                _ => format!("{:?}", value),
            },
            crate::ast::Expr::Path { path, .. } => path.to_string(),
            crate::ast::Expr::Call { callee, .. } => format!("{}(...)", Self::expr_label(callee)),
            crate::ast::Expr::MethodCall { method, .. } => format!(".{}(...)", method),
            crate::ast::Expr::Binary {
                op, left, right, ..
            } => format!(
                "({} {:?} {})",
                Self::expr_label(left),
                op,
                Self::expr_label(right)
            ),
            _ => format!("{:?}", expr),
        }
    }

    /// Go to definition
    pub fn goto_definition(
        &self,
        doc: &Document,
        pos: Position,
        uri: &Url,
    ) -> Option<GotoDefinitionResponse> {
        let (word, _) = doc.word_at(pos)?;

        // First try local file symbols
        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                if let Some(response) = self.definition.find_definition(&word, symbols, uri) {
                    return Some(response);
                }
            }
        }

        None
    }

    /// Cross-file goto definition using workspace
    pub async fn goto_definition_cross_file(
        &self,
        doc: &Document,
        pos: Position,
        uri: &Url,
    ) -> Option<GotoDefinitionResponse> {
        let (word, _) = doc.word_at(pos)?;

        // First try local file symbols
        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                if let Some(response) = self.definition.find_definition(&word, symbols, uri) {
                    return Some(response);
                }
            }
        }

        // Fall back to workspace-wide lookup
        if let Some(ref workspace) = self.workspace {
            let ws = workspace.read().await;
            return ws.goto_definition(&word, uri);
        }

        None
    }

    /// Find all references
    pub fn find_references(
        &self,
        doc: &Document,
        pos: Position,
        uri: &Url,
    ) -> Option<Vec<Location>> {
        let (word, _) = doc.word_at(pos)?;

        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                return Some(self.references.find_references(&word, symbols, uri));
            }
        }

        None
    }

    /// Cross-file find references using workspace
    pub async fn find_references_cross_file(
        &self,
        doc: &Document,
        pos: Position,
        uri: &Url,
        include_declaration: bool,
    ) -> Option<Vec<Location>> {
        let (word, _) = doc.word_at(pos)?;

        // Get local references first
        let mut locations = Vec::new();
        if let Some(result) = self.cache.get(uri) {
            if let Some(ref symbols) = result.symbols {
                locations.extend(self.references.find_references(&word, symbols, uri));
            }
        }

        // Add workspace-wide references
        if let Some(ref workspace) = self.workspace {
            let ws = workspace.read().await;
            let workspace_refs = ws.find_references(&word, include_declaration);

            // Merge and deduplicate
            for loc in workspace_refs {
                if !locations
                    .iter()
                    .any(|l| l.uri == loc.uri && l.range == loc.range)
                {
                    locations.push(loc);
                }
            }
        }

        if locations.is_empty() {
            None
        } else {
            Some(locations)
        }
    }

    /// Get workspace symbols for workspace/symbol request
    pub async fn workspace_symbols(&self, query: &str) -> Vec<SymbolInformation> {
        if let Some(ref workspace) = self.workspace {
            let ws = workspace.read().await;
            ws.workspace_symbols(query)
        } else {
            Vec::new()
        }
    }

    /// Get completions at a position
    pub fn completions(&self, doc: &Document, pos: Position) -> Vec<CompletionItem> {
        let source = doc.text();
        let offset = doc.position_to_offset(pos).unwrap_or(0);
        let context = doc.get_context(pos, 50).unwrap_or_default();

        self.completion
            .complete(&source, offset, &context, &self.cache)
    }

    /// Get semantic tokens for highlighting
    pub fn semantic_tokens(&self, doc: &Document) -> SemanticTokens {
        let source = doc.text();
        self.semantic_tokens.tokenize(&source)
    }

    /// Get document symbols (outline)
    pub fn document_symbols(&self, doc: &Document, uri: &Url) -> Vec<DocumentSymbol> {
        if let Some(result) = self.cache.get(uri) {
            if let Some(ref ast) = result.ast {
                return self.extract_symbols(ast, &doc.text());
            }
        }

        Vec::new()
    }

    /// Get signature help
    pub fn signature_help(&self, doc: &Document, pos: Position) -> Option<SignatureHelp> {
        let source = doc.text();
        let offset = doc.position_to_offset(pos)?;

        // Find the function call context
        let before = &source[..offset];
        let paren_pos = before.rfind('(')?;

        // Count commas to determine active parameter
        let in_call = &source[paren_pos..offset];
        let active_param = in_call.matches(',').count();

        // Extract function name
        let func_end = before[..paren_pos].trim_end().len();
        let mut func_start = func_end;
        let bytes = before.as_bytes();
        while func_start > 0
            && (bytes[func_start - 1].is_ascii_alphanumeric() || bytes[func_start - 1] == b'_')
        {
            func_start -= 1;
        }

        let func_name = &before[func_start..func_end];

        // Look up function signature
        self.get_function_signature(func_name, active_param as u32)
    }

    /// Rename symbol
    pub fn rename(
        &self,
        doc: &Document,
        pos: Position,
        new_name: &str,
        uri: &Url,
    ) -> Option<WorkspaceEdit> {
        let (old_name, _) = doc.word_at(pos)?;

        // Find all references and create edits
        let locations = self.find_references(doc, pos, uri)?;

        let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();

        for loc in locations {
            let edit = TextEdit {
                range: loc.range,
                new_text: new_name.to_string(),
            };

            changes.entry(loc.uri).or_default().push(edit);
        }

        Some(WorkspaceEdit {
            changes: Some(changes),
            ..Default::default()
        })
    }

    /// Get code actions
    pub fn code_actions(
        &self,
        _doc: &Document,
        _range: Range,
        diagnostics: &[Diagnostic],
        _uri: &Url,
    ) -> CodeActionResponse {
        let mut actions = Vec::new();

        for diag in diagnostics {
            // Generate quick fixes based on diagnostic code
            if let Some(NumberOrString::String(code)) = &diag.code {
                match code.as_str() {
                    "E0002" => {
                        // Undefined variable
                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: "Add variable declaration".to_string(),
                            kind: Some(CodeActionKind::QUICKFIX),
                            diagnostics: Some(vec![diag.clone()]),
                            edit: None,
                            ..Default::default()
                        }));
                    }
                    "E0006" => {
                        // Effect not declared
                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: "Add effect to function signature".to_string(),
                            kind: Some(CodeActionKind::QUICKFIX),
                            diagnostics: Some(vec![diag.clone()]),
                            edit: None,
                            ..Default::default()
                        }));
                    }
                    _ => {}
                }
            }
        }

        actions
    }

    /// Get inlay hints
    ///
    /// Returns inlay hints for the given document and range, including:
    /// - Type hints for let bindings without explicit types
    /// - Parameter hints at call sites
    /// - Tensor shape hints
    ///
    /// Note: Epistemic/unit/effect hints from query database require
    /// architectural changes (RwLock on query_db) for safe &self access.
    /// These will be enabled in a future iteration.
    pub fn inlay_hints(&self, doc: &Document, range: Range, uri: &Url) -> Vec<InlayHint> {
        let source = doc.text();

        // Get hints from the provider (type hints, parameter hints, tensor hints)
        let ast = self.cache.get(uri).and_then(|r| r.ast.as_ref());
        let symbols = self.cache.get(uri).and_then(|r| r.symbols.as_ref());
        self.inlay_hints.hints(&source, range, ast, symbols)
    }

    /// Format document
    pub fn format(&self, doc: &Document) -> Option<Vec<TextEdit>> {
        let source = doc.text();
        let mut formatter = SourceFormatter::default();
        let formatted = formatter.format(&source).ok()?;

        if formatted == source {
            return Some(Vec::new());
        }

        Some(vec![TextEdit {
            range: Range {
                start: Position::new(0, 0),
                end: doc.offset_to_position(doc.len()),
            },
            new_text: formatted,
        }])
    }

    /// Get folding ranges
    pub fn folding_ranges(&self, doc: &Document) -> Vec<FoldingRange> {
        let mut ranges = Vec::new();
        let source = doc.text();

        // Track brace depth for folding
        if let Ok(tokens) = lexer::lex(&source) {
            let mut brace_stack: Vec<(u32, FoldingRangeKind)> = Vec::new();

            for token in &tokens {
                let pos = doc.offset_to_position(token.span.start);

                match token.kind {
                    crate::lexer::TokenKind::LBrace => {
                        brace_stack.push((pos.line, FoldingRangeKind::Region));
                    }
                    crate::lexer::TokenKind::RBrace => {
                        if let Some((start_line, kind)) = brace_stack.pop() {
                            if pos.line > start_line {
                                ranges.push(FoldingRange {
                                    start_line,
                                    start_character: None,
                                    end_line: pos.line,
                                    end_character: None,
                                    kind: Some(kind),
                                    collapsed_text: None,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        ranges
    }

    /// Extract document symbols from AST
    fn extract_symbols(&self, ast: &Ast, source: &str) -> Vec<DocumentSymbol> {
        let mut symbols = Vec::new();

        for item in &ast.items {
            match item {
                crate::ast::Item::Function(f) => {
                    let range = span_to_range(&f.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: f.name.clone(),
                        detail: Some(format_fn_signature(f)),
                        kind: SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Struct(s) => {
                    let range = span_to_range(&s.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: s.name.clone(),
                        detail: Some(format!("struct {}", s.name)),
                        kind: SymbolKind::STRUCT,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Enum(e) => {
                    let range = span_to_range(&e.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: e.name.clone(),
                        detail: Some(format!("enum {}", e.name)),
                        kind: SymbolKind::ENUM,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::TypeAlias(t) => {
                    let range = span_to_range(&t.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: t.name.clone(),
                        detail: Some(format!("type {}", t.name)),
                        kind: SymbolKind::TYPE_PARAMETER,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Effect(e) => {
                    let range = span_to_range(&e.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: e.name.clone(),
                        detail: Some(format!("effect {}", e.name)),
                        kind: SymbolKind::INTERFACE,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Trait(t) => {
                    let range = span_to_range(&t.span, source);
                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: t.name.clone(),
                        detail: Some(format!("trait {}", t.name)),
                        kind: SymbolKind::INTERFACE,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                _ => {}
            }
        }

        symbols
    }

    /// Get function signature for signature help
    fn get_function_signature(&self, name: &str, active_param: u32) -> Option<SignatureHelp> {
        // Check cache for function definitions
        for result in self.cache.values() {
            if let Some(ref ast) = result.ast {
                for item in &ast.items {
                    if let crate::ast::Item::Function(f) = item {
                        if f.name == name {
                            let params: Vec<ParameterInformation> = f
                                .params
                                .iter()
                                .map(|p| {
                                    let param_name = match &p.pattern {
                                        crate::ast::Pattern::Binding { name, .. } => name.clone(),
                                        _ => "_".to_string(),
                                    };
                                    ParameterInformation {
                                        label: ParameterLabel::Simple(param_name),
                                        documentation: None,
                                    }
                                })
                                .collect();

                            return Some(SignatureHelp {
                                signatures: vec![SignatureInformation {
                                    label: format_fn_signature(f),
                                    documentation: None,
                                    parameters: Some(params),
                                    active_parameter: Some(active_param),
                                }],
                                active_signature: Some(0),
                                active_parameter: Some(active_param),
                            });
                        }
                    }
                }
            }
        }

        // Check for built-in functions
        get_builtin_signature(name, active_param)
    }
}

impl Default for AnalysisHost {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert Span to LSP Range
fn span_to_range(span: &Span, source: &str) -> Range {
    let (start_line, start_col) = offset_to_line_col(source, span.start);
    let (end_line, end_col) = offset_to_line_col(source, span.end);

    Range {
        start: Position {
            line: start_line as u32,
            character: start_col as u32,
        },
        end: Position {
            line: end_line as u32,
            character: end_col as u32,
        },
    }
}

/// Convert byte offset to line/column
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
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

    (line, col)
}

/// Format a function signature
fn format_fn_signature(f: &crate::ast::FnDef) -> String {
    let params: Vec<String> = f
        .params
        .iter()
        .map(|p| {
            let name = match &p.pattern {
                crate::ast::Pattern::Binding { name, .. } => name.clone(),
                _ => "_".to_string(),
            };
            format!("{}: {}", name, format_type(&p.ty))
        })
        .collect();

    let ret = f
        .return_type
        .as_ref()
        .map(|t| format!(" -> {}", format_type(t)))
        .unwrap_or_default();

    let effects = if f.effects.is_empty() {
        String::new()
    } else {
        let eff_names: Vec<String> = f.effects.iter().map(|e| e.name.to_string()).collect();
        format!(" with {}", eff_names.join(", "))
    };

    format!("fn {}({}){}{}", f.name, params.join(", "), ret, effects)
}

/// Format a type expression
fn format_type(ty: &crate::ast::TypeExpr) -> String {
    match ty {
        crate::ast::TypeExpr::Unit => "()".to_string(),
        crate::ast::TypeExpr::SelfType => "Self".to_string(),
        crate::ast::TypeExpr::Named { path, args, unit } => {
            let base = path.to_string();
            let type_args = if args.is_empty() {
                String::new()
            } else {
                let arg_strs: Vec<String> = args.iter().map(format_type).collect();
                format!("<{}>", arg_strs.join(", "))
            };
            let unit_suffix = unit
                .as_ref()
                .map(|u| format!("<{}>", u))
                .unwrap_or_default();
            format!("{}{}{}", base, type_args, unit_suffix)
        }
        crate::ast::TypeExpr::Reference { mutable, inner } => {
            if *mutable {
                format!("&!{}", format_type(inner)) // D uses &! for mutable refs
            } else {
                format!("&{}", format_type(inner))
            }
        }
        crate::ast::TypeExpr::Array { element, size } => {
            if let Some(_size) = size {
                format!("[{}; N]", format_type(element))
            } else {
                format!("[{}]", format_type(element))
            }
        }
        crate::ast::TypeExpr::Tuple(types) => {
            let inner: Vec<String> = types.iter().map(format_type).collect();
            format!("({})", inner.join(", "))
        }
        crate::ast::TypeExpr::Function {
            params,
            return_type,
            ..
        } => {
            let p: Vec<String> = params.iter().map(format_type).collect();
            format!("fn({}) -> {}", p.join(", "), format_type(return_type))
        }
        crate::ast::TypeExpr::Infer => "_".to_string(),
        // Tensor types with shape information
        crate::ast::TypeExpr::Tensor {
            element_type,
            shape,
        } => {
            let dims_str = shape
                .iter()
                .map(format_tensor_dim)
                .collect::<Vec<_>>()
                .join(", ");
            format!("Tensor[{}, ({})]", format_type(element_type), dims_str)
        }
        // Tile types for GPU programming
        crate::ast::TypeExpr::Tile {
            element_type,
            tile_m,
            tile_n,
            layout,
        } => {
            let layout_str = layout
                .as_ref()
                .map(|l| format!(", {}", l))
                .unwrap_or_default();
            format!(
                "tile<{}, {}, {}{}>",
                format_type(element_type),
                tile_m,
                tile_n,
                layout_str
            )
        }
        // Quantity types with units
        crate::ast::TypeExpr::Quantity { numeric_type, unit } => {
            let unit_str = unit
                .base_units
                .iter()
                .map(|(u, e)| {
                    if *e == 1 {
                        u.clone()
                    } else {
                        format!("{}^{}", u, e)
                    }
                })
                .collect::<Vec<_>>()
                .join("*");
            format!("{}@{}", format_type(numeric_type), unit_str)
        }
        // Ontology types
        crate::ast::TypeExpr::Ontology { ontology, term } => {
            if let Some(t) = term {
                format!("{}:{}", ontology, t)
            } else {
                ontology.clone()
            }
        }
        // Linear/affine type annotations
        crate::ast::TypeExpr::Linear { inner, linearity } => {
            let modifier = match linearity {
                crate::ast::LinearityKind::Linear => "linear ",
                crate::ast::LinearityKind::Affine => "affine ",
                crate::ast::LinearityKind::Relevant => "relevant ",
                crate::ast::LinearityKind::Unrestricted => "",
            };
            format!("{}{}", modifier, format_type(inner))
        }
        // Effected types
        crate::ast::TypeExpr::Effected { inner, effects } => {
            let effects_str = effects.effects.join(", ");
            if effects_str.is_empty() {
                format_type(inner)
            } else {
                format!("{} with {}", format_type(inner), effects_str)
            }
        }
        // Refinement types
        crate::ast::TypeExpr::Refinement {
            var,
            base_type,
            predicate: _,
        } => {
            format!("{{ {}: {} | ... }}", var, format_type(base_type))
        }
        // Knowledge types (epistemic)
        crate::ast::TypeExpr::Knowledge {
            value_type,
            epsilon,
            validity: _,
            provenance: _,
        } => {
            let eps_str = epsilon
                .as_ref()
                .map(|e| format!(", e {}", format_epsilon_bound(e)))
                .unwrap_or_default();
            format!("Knowledge[{}{}]", format_type(value_type), eps_str)
        }
        // Raw pointer types (for FFI)
        crate::ast::TypeExpr::RawPointer { mutable, inner } => {
            if *mutable {
                format!("*!{}", format_type(inner))
            } else {
                format!("*{}", format_type(inner))
            }
        }
        // Never type (bottom type, e.g., for diverging functions)
        crate::ast::TypeExpr::Never => "!".to_string(),
        // Scientific array with dimension
        crate::ast::TypeExpr::ScientificArray { element_type, dim } => {
            format!(
                "Array[{}, {}]",
                format_type(element_type),
                format_tensor_dim(dim)
            )
        }
        // Scientific matrix
        crate::ast::TypeExpr::ScientificMatrix {
            element_type,
            rows,
            cols,
        } => {
            format!(
                "Matrix[{}, {}, {}]",
                format_type(element_type),
                format_tensor_dim(rows),
                format_tensor_dim(cols)
            )
        }
        // Forall quantified types
        crate::ast::TypeExpr::Forall { vars, inner } => {
            let vars_str = vars
                .iter()
                .map(|v| v.name.clone())
                .collect::<Vec<_>>()
                .join(", ");
            format!("forall<{}> {}", vars_str, format_type(inner))
        }
    }
}

/// Format a tensor dimension
fn format_tensor_dim(dim: &crate::ast::TensorDim) -> String {
    match dim {
        crate::ast::TensorDim::Named(name) => name.clone(),
        crate::ast::TensorDim::Fixed(size) => size.to_string(),
        crate::ast::TensorDim::Dynamic => "?".to_string(),
        crate::ast::TensorDim::Expr(_) => "_".to_string(),
    }
}

/// Format an epsilon bound for epistemic types
fn format_epsilon_bound(bound: &crate::ast::EpsilonBound) -> String {
    let op = match bound.operator {
        crate::ast::ComparisonOp::Lt => "<",
        crate::ast::ComparisonOp::Le => "<=",
        crate::ast::ComparisonOp::Eq => "=",
        crate::ast::ComparisonOp::Ge => ">=",
        crate::ast::ComparisonOp::Gt => ">",
    };
    format!("{} ...", op)
}

/// Get built-in function signature
fn get_builtin_signature(name: &str, active_param: u32) -> Option<SignatureHelp> {
    let (label, params): (&str, Vec<&str>) = match name {
        "print" => ("fn print(value: T)", vec!["value: T"]),
        "println" => ("fn println(value: T)", vec!["value: T"]),
        "len" => ("fn len(collection: C) -> usize", vec!["collection: C"]),
        "push" => (
            "fn push(vec: &mut Vec<T>, value: T)",
            vec!["vec: &mut Vec<T>", "value: T"],
        ),
        "pop" => (
            "fn pop(vec: &mut Vec<T>) -> Option<T>",
            vec!["vec: &mut Vec<T>"],
        ),
        "sample" => (
            "fn sample(distribution: D) -> T with Prob",
            vec!["distribution: D"],
        ),
        "observe" => (
            "fn observe(distribution: D, value: T) with Prob",
            vec!["distribution: D", "value: T"],
        ),
        _ => return None,
    };

    Some(SignatureHelp {
        signatures: vec![SignatureInformation {
            label: label.to_string(),
            documentation: None,
            parameters: Some(
                params
                    .iter()
                    .map(|p| ParameterInformation {
                        label: ParameterLabel::Simple(p.to_string()),
                        documentation: None,
                    })
                    .collect(),
            ),
            active_parameter: Some(active_param),
        }],
        active_signature: Some(0),
        active_parameter: Some(active_param),
    })
}
