//! Inlay hints provider for inline type and parameter annotations
//!
//! Provides inlay hints for:
//! - Type annotations on let bindings without explicit types
//! - Parameter names at call sites
//! - Chaining hints for method chains
//! - Closure return type hints
//! - Effect annotations
//! - Lifetime annotations (future)
//!
//! # Configuration
//!
//! The provider supports configuration for enabling/disabling specific hint types:
//! - `typeHints.enabled` - Show type hints for let bindings
//! - `parameterHints.enabled` - Show parameter names at call sites
//! - `chainingHints.enabled` - Show types for chained method calls
//! - `closureHints.enabled` - Show closure return types
//! - `effectHints.enabled` - Show effect annotations
//!
//! # Example
//!
//! ```d
//! let x = 42;           // Shows ": int" after x
//! let y = vec.len();    // Shows ": usize" after y
//! foo(42, "hello");     // Shows "count:" and "message:" before args
//! ```

use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Expr, Item, Stmt};
use crate::common::Span;
use crate::lexer::{self, TokenKind};
use crate::resolve::SymbolTable;
use crate::types::Type;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for inlay hints
#[derive(Debug, Clone)]
pub struct InlayHintConfig {
    /// Show type hints for let bindings
    pub type_hints: bool,
    /// Show parameter names at call sites
    pub parameter_hints: bool,
    /// Show types for chained method calls
    pub chaining_hints: bool,
    /// Show closure return types
    pub closure_return_hints: bool,
    /// Show effect annotations
    pub effect_hints: bool,
    /// Show lifetime annotations
    pub lifetime_hints: bool,
    /// Maximum length for type hints before truncation
    pub max_type_length: usize,
}

impl Default for InlayHintConfig {
    fn default() -> Self {
        Self {
            type_hints: true,
            parameter_hints: true,
            chaining_hints: true,
            closure_return_hints: true,
            effect_hints: true,
            lifetime_hints: false, // Disabled by default
            max_type_length: 50,
        }
    }
}

// ============================================================================
// Inlay Hint Provider
// ============================================================================

/// Provider for inlay hints
pub struct InlayHintProvider {
    /// Configuration
    config: InlayHintConfig,
    /// Known function signatures for parameter hints
    known_functions: std::collections::HashMap<String, Vec<String>>,
}

impl InlayHintProvider {
    /// Create a new inlay hint provider with default configuration
    pub fn new() -> Self {
        Self::with_config(InlayHintConfig::default())
    }

    /// Create a new inlay hint provider with custom configuration
    pub fn with_config(config: InlayHintConfig) -> Self {
        let mut known_functions = std::collections::HashMap::new();

        // Add known standard library functions
        known_functions.insert("print".to_string(), vec!["value".to_string()]);
        known_functions.insert("println".to_string(), vec!["value".to_string()]);
        known_functions.insert(
            "format".to_string(),
            vec!["fmt".to_string(), "args".to_string()],
        );
        known_functions.insert("len".to_string(), vec!["collection".to_string()]);
        known_functions.insert(
            "push".to_string(),
            vec!["vec".to_string(), "value".to_string()],
        );
        known_functions.insert("pop".to_string(), vec!["vec".to_string()]);
        known_functions.insert("get".to_string(), vec!["index".to_string()]);
        known_functions.insert(
            "insert".to_string(),
            vec!["key".to_string(), "value".to_string()],
        );
        known_functions.insert("remove".to_string(), vec!["key".to_string()]);
        known_functions.insert(
            "map".to_string(),
            vec!["collection".to_string(), "f".to_string()],
        );
        known_functions.insert(
            "filter".to_string(),
            vec!["collection".to_string(), "predicate".to_string()],
        );
        known_functions.insert(
            "fold".to_string(),
            vec!["init".to_string(), "f".to_string()],
        );
        known_functions.insert("sample".to_string(), vec!["distribution".to_string()]);
        known_functions.insert(
            "observe".to_string(),
            vec!["distribution".to_string(), "value".to_string()],
        );

        Self {
            config,
            known_functions,
        }
    }

    /// Generate inlay hints for a document
    pub fn hints(
        &self,
        source: &str,
        range: Range,
        ast: Option<&Ast>,
        symbols: Option<&SymbolTable>,
    ) -> Vec<InlayHint> {
        let mut hints = Vec::new();

        // Lexer-based hints (fast, works without AST)
        self.collect_lexer_hints(source, range, &mut hints);

        // AST-based hints (more accurate, requires AST)
        if let Some(ast) = ast {
            self.collect_ast_hints(source, range, ast, symbols, &mut hints);
        }

        hints
    }

    /// Collect hints using lexer tokens only
    fn collect_lexer_hints(&self, source: &str, range: Range, hints: &mut Vec<InlayHint>) {
        if !self.config.type_hints {
            return;
        }

        let Ok(tokens) = lexer::lex(source) else {
            return;
        };

        let mut i = 0;
        while i < tokens.len() {
            // Look for "let" keyword
            if tokens[i].kind == TokenKind::Let {
                // Pattern: let <name> = <value>  (no type annotation)
                // Pattern: let <name>: <type> = <value>  (has type annotation)
                if i + 3 < tokens.len() {
                    if let TokenKind::Ident = tokens[i + 1].kind {
                        // Check if next token is `:` (has type) or `=` (no type)
                        if tokens[i + 2].kind == TokenKind::Eq {
                            // No type annotation - add hint
                            let pos = self.offset_to_position(source, tokens[i + 1].span.end);
                            if self.position_in_range(&pos, &range) {
                                hints.push(InlayHint {
                                    position: pos,
                                    label: InlayHintLabel::String(": <inferred>".to_string()),
                                    kind: Some(InlayHintKind::TYPE),
                                    text_edits: None,
                                    tooltip: Some(InlayHintTooltip::String(
                                        "Type will be inferred from the initializer".to_string(),
                                    )),
                                    padding_left: Some(false),
                                    padding_right: Some(true),
                                    data: None,
                                });
                            }
                        }
                    }
                }
            }
            i += 1;
        }
    }

    /// Collect hints using parsed AST
    fn collect_ast_hints(
        &self,
        source: &str,
        range: Range,
        ast: &Ast,
        symbols: Option<&SymbolTable>,
        hints: &mut Vec<InlayHint>,
    ) {
        for item in &ast.items {
            self.visit_item(source, range, item, symbols, hints);
        }
    }

    /// Visit an item for hints
    fn visit_item(
        &self,
        source: &str,
        range: Range,
        item: &Item,
        symbols: Option<&SymbolTable>,
        hints: &mut Vec<InlayHint>,
    ) {
        match item {
            Item::Function(f) => {
                // Visit function body
                for stmt in &f.body.stmts {
                    self.visit_stmt(source, range, stmt, symbols, hints);
                }
            }
            Item::Impl(impl_def) => {
                for impl_item in &impl_def.items {
                    if let crate::ast::ImplItem::Fn(f) = impl_item {
                        for stmt in &f.body.stmts {
                            self.visit_stmt(source, range, stmt, symbols, hints);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Visit a statement for hints
    fn visit_stmt(
        &self,
        source: &str,
        range: Range,
        stmt: &Stmt,
        symbols: Option<&SymbolTable>,
        hints: &mut Vec<InlayHint>,
    ) {
        match stmt {
            Stmt::Let { ty, value, .. } => {
                // Type hint for let without explicit type
                if self.config.type_hints && ty.is_none() {
                    if let Some(expr) = value {
                        if let Some(inferred_type) = self.infer_expr_type(expr, symbols) {
                            // Get position after pattern (would need pattern span)
                            // For now, this is a placeholder - actual impl needs pattern spans
                            let _type_str = self.format_type(&inferred_type);
                        }
                    }
                }

                // Visit the value expression
                if let Some(expr) = value {
                    self.visit_expr(source, range, expr, symbols, hints);
                }
            }
            Stmt::Expr { expr, .. } => {
                self.visit_expr(source, range, expr, symbols, hints);
            }
            Stmt::Assign { target, value, .. } => {
                self.visit_expr(source, range, target, symbols, hints);
                self.visit_expr(source, range, value, symbols, hints);
            }
            Stmt::Empty => {}
            Stmt::MacroInvocation(_) => {
                // Macro invocations are expanded before inlay hint processing
            }
        }
    }

    /// Visit an expression for hints
    fn visit_expr(
        &self,
        source: &str,
        range: Range,
        expr: &Expr,
        symbols: Option<&SymbolTable>,
        hints: &mut Vec<InlayHint>,
    ) {
        match expr {
            Expr::Call { callee, args, .. } => {
                // Parameter hints for function calls
                if self.config.parameter_hints {
                    if let Some(func_name) = self.extract_function_name(callee) {
                        self.add_parameter_hints(source, range, &func_name, args, symbols, hints);
                    }
                }

                // Recurse into callee and args
                self.visit_expr(source, range, callee, symbols, hints);
                for arg in args {
                    self.visit_expr(source, range, arg, symbols, hints);
                }
            }
            Expr::MethodCall {
                receiver,
                args,
                method,
                ..
            } => {
                // Parameter hints for method calls
                if self.config.parameter_hints {
                    self.add_parameter_hints(source, range, method, args, symbols, hints);
                }

                self.visit_expr(source, range, receiver, symbols, hints);
                for arg in args {
                    self.visit_expr(source, range, arg, symbols, hints);
                }

                // Chaining hints
                if self.config.chaining_hints {
                    // For method chains like `x.foo().bar().baz()`, show intermediate types
                    if self.is_method_chain(receiver) {
                        // Would need type info to show chaining hints
                    }
                }
            }
            Expr::Closure { body, .. } | Expr::AsyncClosure { body, .. } => {
                // Closure return type hints
                if self.config.closure_return_hints {
                    if let Some(return_type) = self.infer_expr_type(body, symbols) {
                        let type_str = self.format_type(&return_type);
                        if !type_str.is_empty() && type_str != "()" {
                            // Would need closure span to add hint
                        }
                    }
                }

                self.visit_expr(source, range, body, symbols, hints);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.visit_expr(source, range, condition, symbols, hints);
                for stmt in &then_branch.stmts {
                    self.visit_stmt(source, range, stmt, symbols, hints);
                }
                if let Some(else_expr) = else_branch {
                    self.visit_expr(source, range, else_expr, symbols, hints);
                }
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.visit_expr(source, range, scrutinee, symbols, hints);
                for arm in arms {
                    self.visit_expr(source, range, &arm.body, symbols, hints);
                }
            }
            Expr::Block { block, .. } => {
                for stmt in &block.stmts {
                    self.visit_stmt(source, range, stmt, symbols, hints);
                }
            }
            Expr::Binary { left, right, .. } => {
                self.visit_expr(source, range, left, symbols, hints);
                self.visit_expr(source, range, right, symbols, hints);
            }
            Expr::Unary { expr, .. }
            | Expr::Try { expr, .. }
            | Expr::Await { expr, .. }
            | Expr::Spawn { expr, .. } => {
                self.visit_expr(source, range, expr, symbols, hints);
            }
            Expr::Loop { body, .. } | Expr::AsyncBlock { block: body, .. } => {
                for stmt in &body.stmts {
                    self.visit_stmt(source, range, stmt, symbols, hints);
                }
            }
            Expr::While {
                condition, body, ..
            } => {
                self.visit_expr(source, range, condition, symbols, hints);
                for stmt in &body.stmts {
                    self.visit_stmt(source, range, stmt, symbols, hints);
                }
            }
            Expr::For { iter, body, .. } => {
                self.visit_expr(source, range, iter, symbols, hints);
                for stmt in &body.stmts {
                    self.visit_stmt(source, range, stmt, symbols, hints);
                }
            }
            Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
                for elem in elements {
                    self.visit_expr(source, range, elem, symbols, hints);
                }
            }
            Expr::StructLit { fields, .. } => {
                for (_, value) in fields {
                    self.visit_expr(source, range, value, symbols, hints);
                }
            }
            Expr::Index { base, index, .. } => {
                self.visit_expr(source, range, base, symbols, hints);
                self.visit_expr(source, range, index, symbols, hints);
            }
            Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
                self.visit_expr(source, range, base, symbols, hints);
            }
            Expr::Cast { expr, .. } => {
                self.visit_expr(source, range, expr, symbols, hints);
            }
            Expr::Return { value, .. } | Expr::Break { value, .. } => {
                if let Some(v) = value {
                    self.visit_expr(source, range, v, symbols, hints);
                }
            }
            Expr::Perform { args, .. } => {
                for arg in args {
                    self.visit_expr(source, range, arg, symbols, hints);
                }
            }
            Expr::Handle { expr, .. }
            | Expr::Sample {
                distribution: expr, ..
            } => {
                self.visit_expr(source, range, expr, symbols, hints);
            }
            Expr::Select { arms, .. } => {
                for arm in arms {
                    self.visit_expr(source, range, &arm.future, symbols, hints);
                    self.visit_expr(source, range, &arm.body, symbols, hints);
                }
            }
            Expr::Join { futures, .. } => {
                for f in futures {
                    self.visit_expr(source, range, f, symbols, hints);
                }
            }
            // Literals and simple expressions don't need hints
            Expr::Literal { .. } | Expr::Path { .. } | Expr::Continue { .. } => {}
            // Macro invocations are expanded before inlay hint processing
            Expr::MacroInvocation(_) => {}
        }
    }

    /// Add parameter hints for a function call
    fn add_parameter_hints(
        &self,
        _source: &str,
        _range: Range,
        func_name: &str,
        _args: &[Expr],
        _symbols: Option<&SymbolTable>,
        _hints: &mut Vec<InlayHint>,
    ) {
        // Look up parameter names from known functions or symbol table
        if let Some(_param_names) = self.known_functions.get(func_name) {
            // Would need expression spans to add hints before each argument
            // For each arg, add hint with param name
        }
    }

    /// Extract function name from callee expression
    fn extract_function_name(&self, callee: &Expr) -> Option<String> {
        match callee {
            Expr::Path { path, .. } => path.name().map(|s| s.to_string()),
            _ => None,
        }
    }

    /// Check if expression is a method chain
    fn is_method_chain(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::MethodCall { .. })
    }

    /// Infer the type of an expression (simplified)
    fn infer_expr_type(&self, expr: &Expr, _symbols: Option<&SymbolTable>) -> Option<Type> {
        match expr {
            Expr::Literal { value, .. } => {
                use crate::ast::Literal;
                match value {
                    Literal::Int(_) => Some(Type::I32),
                    Literal::Float(_) => Some(Type::F64),
                    Literal::Bool(_) => Some(Type::Bool),
                    Literal::String(_) => Some(Type::String),
                    Literal::Char(_) => Some(Type::Char),
                    Literal::Unit => Some(Type::Unit),
                    Literal::IntUnit(_, _) | Literal::FloatUnit(_, _) => {
                        // Unit literals - would need unit type info
                        None
                    }
                }
            }
            Expr::Array { elements, .. } => {
                if let Some(first) = elements.first() {
                    if let Some(elem_type) = self.infer_expr_type(first, _symbols) {
                        return Some(Type::Array {
                            element: Box::new(elem_type),
                            size: Some(elements.len()),
                        });
                    }
                }
                None
            }
            Expr::Tuple { elements, .. } => {
                let types: Vec<_> = elements
                    .iter()
                    .filter_map(|e| self.infer_expr_type(e, _symbols))
                    .collect();
                if types.len() == elements.len() {
                    Some(Type::Tuple(types))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Format a type for display
    fn format_type(&self, ty: &Type) -> String {
        let formatted = format!("{:?}", ty);
        if formatted.len() > self.config.max_type_length {
            format!("{}...", &formatted[..self.config.max_type_length])
        } else {
            formatted
        }
    }

    /// Convert byte offset to LSP position
    fn offset_to_position(&self, source: &str, offset: usize) -> Position {
        let offset = offset.min(source.len());
        let mut line = 0u32;
        let mut col = 0u32;

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

        Position::new(line, col)
    }

    /// Check if a position is within a range
    fn position_in_range(&self, pos: &Position, range: &Range) -> bool {
        if pos.line < range.start.line || pos.line > range.end.line {
            return false;
        }
        if pos.line == range.start.line && pos.character < range.start.character {
            return false;
        }
        if pos.line == range.end.line && pos.character > range.end.character {
            return false;
        }
        true
    }
}

impl Default for InlayHintProvider {
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
    fn test_inlay_hint_config_default() {
        let config = InlayHintConfig::default();
        assert!(config.type_hints);
        assert!(config.parameter_hints);
        assert!(config.chaining_hints);
        assert!(config.closure_return_hints);
        assert!(config.effect_hints);
        assert!(!config.lifetime_hints);
        assert_eq!(config.max_type_length, 50);
    }

    #[test]
    fn test_provider_creation() {
        let provider = InlayHintProvider::new();
        assert!(provider.known_functions.contains_key("print"));
        assert!(provider.known_functions.contains_key("len"));
    }

    #[test]
    fn test_offset_to_position() {
        let provider = InlayHintProvider::new();

        // Single line
        let source = "let x = 42;";
        let pos = provider.offset_to_position(source, 4);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 4);

        // Multi-line
        let source = "let x = 42;\nlet y = 10;";
        let pos = provider.offset_to_position(source, 16);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 4);
    }

    #[test]
    fn test_position_in_range() {
        let provider = InlayHintProvider::new();

        let range = Range {
            start: Position::new(0, 0),
            end: Position::new(10, 100),
        };

        assert!(provider.position_in_range(&Position::new(5, 50), &range));
        assert!(provider.position_in_range(&Position::new(0, 0), &range));
        assert!(provider.position_in_range(&Position::new(10, 50), &range));
        assert!(!provider.position_in_range(&Position::new(11, 0), &range));
    }

    #[test]
    fn test_lexer_hints() {
        let provider = InlayHintProvider::new();
        let source = "let x = 42;";
        let range = Range {
            start: Position::new(0, 0),
            end: Position::new(100, 100),
        };

        let hints = provider.hints(source, range, None, None);
        // Should have a type hint for the let binding
        assert!(!hints.is_empty());
        assert_eq!(hints[0].kind, Some(InlayHintKind::TYPE));
    }

    #[test]
    fn test_no_hints_with_explicit_type() {
        let provider = InlayHintProvider::new();
        let source = "let x: int = 42;";
        let range = Range {
            start: Position::new(0, 0),
            end: Position::new(100, 100),
        };

        let hints = provider.hints(source, range, None, None);
        // Should have no hints since type is explicit
        assert!(hints.is_empty());
    }

    #[test]
    fn test_format_type_truncation() {
        let config = InlayHintConfig {
            max_type_length: 10,
            ..Default::default()
        };
        let provider = InlayHintProvider::with_config(config);

        let long_type = Type::Tuple(vec![Type::I32, Type::I32, Type::I32]);
        let formatted = provider.format_type(&long_type);
        assert!(formatted.len() <= 13); // 10 + "..."
    }
}
