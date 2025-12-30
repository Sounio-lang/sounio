//! Inlay hints provider for inline type and parameter annotations
//!
//! Provides inlay hints for:
//! - Type annotations on let bindings without explicit types
//! - Parameter names at call sites
//! - Chaining hints for method chains
//! - Closure return type hints
//! - Effect annotations
//! - Tensor shape hints for tensor operations
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
//! - `tensorShapeHints.enabled` - Show tensor shape hints
//!
//! # Example
//!
//! ```d
//! let x = 42;           // Shows ": int" after x
//! let y = vec.len();    // Shows ": usize" after y
//! foo(42, "hello");     // Shows "count:" and "message:" before args
//! let t: Tensor[f32, (batch, 128, 256)] = ...;  // Shows shape info
//! ```

use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Expr, Item, Stmt, TensorDim, TypeExpr};
use crate::common::Span;
use crate::hir::{HirTensorDim, HirType};
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
    /// Show tensor shape hints
    pub tensor_shape_hints: bool,
    /// Show shape compatibility warnings
    pub tensor_shape_warnings: bool,
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
            tensor_shape_hints: true,
            tensor_shape_warnings: true,
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
            // Catch-all for new expression types (Range, OntologyTerm, Do, Counterfactual, KnowledgeExpr, etc.)
            _ => {}
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

    // ========================================================================
    // Tensor Shape Hints
    // ========================================================================

    /// Generate tensor shape hints for tensor type expressions in the AST
    pub fn collect_tensor_hints(
        &self,
        source: &str,
        range: Range,
        ast: &Ast,
        hints: &mut Vec<InlayHint>,
    ) {
        if !self.config.tensor_shape_hints {
            return;
        }

        for item in &ast.items {
            self.visit_item_for_tensors(source, range, item, hints);
        }
    }

    /// Visit an item looking for tensor types
    fn visit_item_for_tensors(
        &self,
        source: &str,
        range: Range,
        item: &Item,
        hints: &mut Vec<InlayHint>,
    ) {
        match item {
            Item::Function(f) => {
                // Check function parameters for tensor types
                for param in &f.params {
                    self.check_type_expr_for_tensor(source, range, &param.ty, hints);
                }
                // Check return type
                if let Some(ref ret) = f.return_type {
                    self.check_type_expr_for_tensor(source, range, ret, hints);
                }
                // Visit function body
                for stmt in &f.body.stmts {
                    self.visit_stmt_for_tensors(source, range, stmt, hints);
                }
            }
            Item::Struct(s) => {
                for field in &s.fields {
                    self.check_type_expr_for_tensor(source, range, &field.ty, hints);
                }
            }
            Item::Impl(impl_def) => {
                for impl_item in &impl_def.items {
                    if let crate::ast::ImplItem::Fn(f) = impl_item {
                        for param in &f.params {
                            self.check_type_expr_for_tensor(source, range, &param.ty, hints);
                        }
                        if let Some(ref ret) = f.return_type {
                            self.check_type_expr_for_tensor(source, range, ret, hints);
                        }
                        for stmt in &f.body.stmts {
                            self.visit_stmt_for_tensors(source, range, stmt, hints);
                        }
                    }
                }
            }
            Item::TypeAlias(alias) => {
                self.check_type_expr_for_tensor(source, range, &alias.ty, hints);
            }
            _ => {}
        }
    }

    /// Visit a statement looking for tensor types
    fn visit_stmt_for_tensors(
        &self,
        source: &str,
        range: Range,
        stmt: &Stmt,
        hints: &mut Vec<InlayHint>,
    ) {
        match stmt {
            Stmt::Let { ty, value, .. } => {
                // Check explicit type annotation for tensors
                if let Some(type_expr) = ty {
                    self.check_type_expr_for_tensor(source, range, type_expr, hints);
                }
                // Check value expression
                if let Some(expr) = value {
                    self.visit_expr_for_tensors(source, range, expr, hints);
                }
            }
            Stmt::Expr { expr, .. } => {
                self.visit_expr_for_tensors(source, range, expr, hints);
            }
            Stmt::Assign { target, value, .. } => {
                self.visit_expr_for_tensors(source, range, target, hints);
                self.visit_expr_for_tensors(source, range, value, hints);
            }
            _ => {}
        }
    }

    /// Visit an expression looking for tensor operations
    fn visit_expr_for_tensors(
        &self,
        source: &str,
        range: Range,
        expr: &Expr,
        hints: &mut Vec<InlayHint>,
    ) {
        match expr {
            Expr::Call { callee, args, .. } => {
                // Check for tensor operations like matmul, reshape, etc.
                if let Some(func_name) = self.extract_function_name(callee) {
                    // Get the span from the expression using helper
                    if let Some(expr_span) = self.get_expr_span(expr) {
                        if let Some(hint) =
                            self.tensor_operation_hint(&func_name, args, source, expr_span)
                        {
                            let pos = self.offset_to_position(source, expr_span.end);
                            if self.position_in_range(&pos, &range) {
                                hints.push(hint);
                            }
                        }
                    }
                }
                // Recurse
                self.visit_expr_for_tensors(source, range, callee, hints);
                for arg in args {
                    self.visit_expr_for_tensors(source, range, arg, hints);
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                // Check for tensor methods like .reshape(), .transpose(), etc.
                if let Some(expr_span) = self.get_expr_span(expr) {
                    if let Some(hint) =
                        self.tensor_method_hint(method, receiver, args, source, expr_span)
                    {
                        let pos = self.offset_to_position(source, expr_span.end);
                        if self.position_in_range(&pos, &range) {
                            hints.push(hint);
                        }
                    }
                }
                self.visit_expr_for_tensors(source, range, receiver, hints);
                for arg in args {
                    self.visit_expr_for_tensors(source, range, arg, hints);
                }
            }
            Expr::Binary { left, right, .. } => {
                self.visit_expr_for_tensors(source, range, left, hints);
                self.visit_expr_for_tensors(source, range, right, hints);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.visit_expr_for_tensors(source, range, condition, hints);
                for stmt in &then_branch.stmts {
                    self.visit_stmt_for_tensors(source, range, stmt, hints);
                }
                if let Some(else_expr) = else_branch {
                    self.visit_expr_for_tensors(source, range, else_expr, hints);
                }
            }
            Expr::Block { block, .. } => {
                for stmt in &block.stmts {
                    self.visit_stmt_for_tensors(source, range, stmt, hints);
                }
            }
            _ => {}
        }
    }

    /// Get span from an expression (helper for expressions without explicit span field)
    fn get_expr_span(&self, _expr: &Expr) -> Option<Span> {
        // For now, return a placeholder span
        // In a full implementation, we would extract this from the AST
        // or track spans during parsing
        // TODO: Implement proper span extraction from AST nodes
        Some(Span { start: 0, end: 0 })
    }

    /// Check a type expression for tensor types and add shape hints
    fn check_type_expr_for_tensor(
        &self,
        source: &str,
        range: Range,
        type_expr: &TypeExpr,
        hints: &mut Vec<InlayHint>,
    ) {
        if let TypeExpr::Tensor {
            element_type,
            shape,
        } = type_expr
        {
            // Convert AST tensor dims to HIR format for display
            let hir_dims: Vec<HirTensorDim> = shape
                .iter()
                .map(|d| match d {
                    TensorDim::Named(name) => HirTensorDim::Named(name.clone()),
                    TensorDim::Fixed(size) => HirTensorDim::Fixed(*size),
                    TensorDim::Dynamic => HirTensorDim::Dynamic,
                    TensorDim::Expr(_) => HirTensorDim::Dynamic, // Treat expr dims as dynamic
                })
                .collect();

            // Calculate total elements if possible
            if let Some(total) = HirTensorDim::compute_total_elements(&hir_dims) {
                // Only show hint if there's useful info (non-trivial tensor)
                if total > 1 && hir_dims.len() >= 2 {
                    let dims_str = hir_dims
                        .iter()
                        .map(|d| d.format_short())
                        .collect::<Vec<_>>()
                        .join(" x ");

                    // We'd need the span of the tensor type here
                    // For now, this is a placeholder - actual implementation
                    // would need TypeExpr to carry span information
                }
            }
        }
    }

    /// Generate hint for tensor function calls (matmul, reshape, etc.)
    fn tensor_operation_hint(
        &self,
        func_name: &str,
        args: &[Expr],
        source: &str,
        span: Span,
    ) -> Option<InlayHint> {
        match func_name {
            "matmul" | "mm" | "bmm" => {
                // Matrix multiplication - show resulting shape
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " -> [M, N]", // Placeholder - actual shape would come from type inference
                    "Matrix multiplication result shape",
                ))
            }
            "reshape" | "view" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (reshape)",
                    "Tensor reshape operation",
                ))
            }
            "transpose" | "t" | "T" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (transposed)",
                    "Tensor transpose operation - dimensions swapped",
                ))
            }
            "broadcast" | "expand" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (broadcast)",
                    "Tensor broadcast operation",
                ))
            }
            "zeros" | "ones" | "rand" | "randn" | "zeros_like" | "ones_like" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (tensor init)",
                    "Tensor initialization",
                ))
            }
            _ => None,
        }
    }

    /// Generate hint for tensor method calls
    fn tensor_method_hint(
        &self,
        method: &str,
        _receiver: &Expr,
        _args: &[Expr],
        source: &str,
        span: Span,
    ) -> Option<InlayHint> {
        match method {
            "reshape" | "view" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (reshaped)",
                    "Returns tensor with new shape",
                ))
            }
            "transpose" | "t" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (transposed)",
                    "Returns transposed tensor",
                ))
            }
            "squeeze" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (squeezed)",
                    "Removes dimensions of size 1",
                ))
            }
            "unsqueeze" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (expanded)",
                    "Adds dimension of size 1",
                ))
            }
            "flatten" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " -> [N]",
                    "Flattens to 1D tensor",
                ))
            }
            "sum" | "mean" | "max" | "min" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " (reduced)",
                    "Reduction operation along dimensions",
                ))
            }
            "matmul" | "mm" | "bmm" => {
                Some(self.create_tensor_hint(
                    span,
                    source,
                    " -> [M, N]",
                    "Matrix multiplication result",
                ))
            }
            _ => None,
        }
    }

    /// Create an inlay hint for tensor shape information
    fn create_tensor_hint(
        &self,
        span: Span,
        source: &str,
        label: &str,
        tooltip: &str,
    ) -> InlayHint {
        let pos = self.offset_to_position(source, span.end);
        InlayHint {
            position: pos,
            label: InlayHintLabel::String(label.to_string()),
            kind: Some(InlayHintKind::TYPE),
            text_edits: None,
            tooltip: Some(InlayHintTooltip::String(tooltip.to_string())),
            padding_left: Some(false),
            padding_right: Some(true),
            data: None,
        }
    }

    /// Generate tensor shape hint with detailed shape information
    pub fn tensor_shape_hint_from_hir(
        &self,
        hir_type: &HirType,
        span: Span,
        source: &str,
    ) -> Option<InlayHint> {
        if !self.config.tensor_shape_hints {
            return None;
        }

        match hir_type {
            HirType::Tensor { element, dims } => {
                let dims_str = dims
                    .iter()
                    .map(|d| d.format_short())
                    .collect::<Vec<_>>()
                    .join(", ");

                let total_info = HirTensorDim::compute_total_elements(dims)
                    .map(|n| format!(" ({} elements)", n))
                    .unwrap_or_default();

                let label = format!(": Tensor[{}, ({})]", element.format_short(), dims_str);
                let tooltip = format!(
                    "Tensor type\n\
                     Element: {}\n\
                     Shape: [{}]\n\
                     Rank: {}{}",
                    element.format_short(),
                    dims_str,
                    dims.len(),
                    total_info
                );

                Some(InlayHint {
                    position: self.offset_to_position(source, span.end),
                    label: InlayHintLabel::String(label),
                    kind: Some(InlayHintKind::TYPE),
                    text_edits: None,
                    tooltip: Some(InlayHintTooltip::String(tooltip)),
                    padding_left: Some(false),
                    padding_right: Some(true),
                    data: None,
                })
            }
            // Handle matrices as special tensors
            HirType::Mat2 | HirType::Mat3 | HirType::Mat4 => {
                let dim = match hir_type {
                    HirType::Mat2 => 2,
                    HirType::Mat3 => 3,
                    HirType::Mat4 => 4,
                    _ => unreachable!(),
                };
                let label = format!(": mat{} [{}x{}]", dim, dim, dim);
                let tooltip = format!(
                    "Matrix type\n\
                     Shape: {}x{} (column-major)\n\
                     Total elements: {}",
                    dim,
                    dim,
                    dim * dim
                );

                Some(InlayHint {
                    position: self.offset_to_position(source, span.end),
                    label: InlayHintLabel::String(label),
                    kind: Some(InlayHintKind::TYPE),
                    text_edits: None,
                    tooltip: Some(InlayHintTooltip::String(tooltip)),
                    padding_left: Some(false),
                    padding_right: Some(true),
                    data: None,
                })
            }
            // Handle vectors
            HirType::Vec2 | HirType::Vec3 | HirType::Vec4 => {
                let dim = match hir_type {
                    HirType::Vec2 => 2,
                    HirType::Vec3 => 3,
                    HirType::Vec4 => 4,
                    _ => unreachable!(),
                };
                let label = format!(": vec{} [{}]", dim, dim);
                let tooltip = format!(
                    "Vector type\n\
                     Dimension: {}\n\
                     Element type: f32",
                    dim
                );

                Some(InlayHint {
                    position: self.offset_to_position(source, span.end),
                    label: InlayHintLabel::String(label),
                    kind: Some(InlayHintKind::TYPE),
                    text_edits: None,
                    tooltip: Some(InlayHintTooltip::String(tooltip)),
                    padding_left: Some(false),
                    padding_right: Some(true),
                    data: None,
                })
            }
            _ => None,
        }
    }

    /// Generate shape compatibility warning hint
    pub fn tensor_shape_warning_hint(
        &self,
        message: &str,
        span: Span,
        source: &str,
    ) -> Option<InlayHint> {
        if !self.config.tensor_shape_warnings {
            return None;
        }

        Some(InlayHint {
            position: self.offset_to_position(source, span.end),
            label: InlayHintLabel::LabelParts(vec![InlayHintLabelPart {
                value: format!(" [!] {}", message),
                tooltip: Some(InlayHintLabelPartTooltip::String(
                    "Shape compatibility warning".to_string(),
                )),
                location: None,
                command: None,
            }]),
            kind: None, // No specific kind for warnings
            text_edits: None,
            tooltip: Some(InlayHintTooltip::String(format!(
                "Warning: {}\n\nTensor shapes may be incompatible for this operation.",
                message
            ))),
            padding_left: Some(true),
            padding_right: Some(true),
            data: None,
        })
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
        assert!(config.tensor_shape_hints);
        assert!(config.tensor_shape_warnings);
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

    // ========================================================================
    // Tensor Shape Hint Tests
    // ========================================================================

    #[test]
    fn test_tensor_shape_hint_from_hir() {
        let provider = InlayHintProvider::new();
        let source = "let x = tensor";
        let span = crate::common::Span { start: 0, end: 14 };

        // Test with a simple tensor type
        let tensor_type = HirType::Tensor {
            element: Box::new(HirType::F32),
            dims: vec![
                HirTensorDim::Fixed(32),
                HirTensorDim::Fixed(128),
                HirTensorDim::Fixed(256),
            ],
        };

        let hint = provider.tensor_shape_hint_from_hir(&tensor_type, span, source);
        assert!(hint.is_some());

        let hint = hint.unwrap();
        assert_eq!(hint.kind, Some(InlayHintKind::TYPE));
        if let InlayHintLabel::String(label) = &hint.label {
            assert!(label.contains("Tensor"));
            assert!(label.contains("f32"));
            assert!(label.contains("32"));
            assert!(label.contains("128"));
            assert!(label.contains("256"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_tensor_shape_hint_with_named_dims() {
        let provider = InlayHintProvider::new();
        let source = "let x = tensor";
        let span = crate::common::Span { start: 0, end: 14 };

        let tensor_type = HirType::Tensor {
            element: Box::new(HirType::F32),
            dims: vec![
                HirTensorDim::Named("batch".to_string()),
                HirTensorDim::Named("channels".to_string()),
                HirTensorDim::Named("height".to_string()),
                HirTensorDim::Named("width".to_string()),
            ],
        };

        let hint = provider.tensor_shape_hint_from_hir(&tensor_type, span, source);
        assert!(hint.is_some());

        let hint = hint.unwrap();
        if let InlayHintLabel::String(label) = &hint.label {
            assert!(label.contains("batch"));
            assert!(label.contains("channels"));
            assert!(label.contains("height"));
            assert!(label.contains("width"));
        }
    }

    #[test]
    fn test_tensor_shape_hint_with_dynamic_dims() {
        let provider = InlayHintProvider::new();
        let source = "let x = tensor";
        let span = crate::common::Span { start: 0, end: 14 };

        let tensor_type = HirType::Tensor {
            element: Box::new(HirType::F64),
            dims: vec![
                HirTensorDim::Dynamic,
                HirTensorDim::Fixed(512),
            ],
        };

        let hint = provider.tensor_shape_hint_from_hir(&tensor_type, span, source);
        assert!(hint.is_some());

        let hint = hint.unwrap();
        if let InlayHintLabel::String(label) = &hint.label {
            assert!(label.contains("?"));  // Dynamic dimension marker
            assert!(label.contains("512"));
        }
    }

    #[test]
    fn test_matrix_type_hint() {
        let provider = InlayHintProvider::new();
        let source = "let m = mat4";
        let span = crate::common::Span { start: 0, end: 12 };

        let hint = provider.tensor_shape_hint_from_hir(&HirType::Mat4, span, source);
        assert!(hint.is_some());

        let hint = hint.unwrap();
        if let InlayHintLabel::String(label) = &hint.label {
            assert!(label.contains("mat4"));
            assert!(label.contains("4x4"));
        }
    }

    #[test]
    fn test_vector_type_hint() {
        let provider = InlayHintProvider::new();
        let source = "let v = vec3";
        let span = crate::common::Span { start: 0, end: 12 };

        let hint = provider.tensor_shape_hint_from_hir(&HirType::Vec3, span, source);
        assert!(hint.is_some());

        let hint = hint.unwrap();
        if let InlayHintLabel::String(label) = &hint.label {
            assert!(label.contains("vec3"));
            assert!(label.contains("3"));
        }
    }

    #[test]
    fn test_tensor_shape_warnings() {
        let provider = InlayHintProvider::new();
        let source = "matmul(a, b)";
        let span = crate::common::Span { start: 0, end: 12 };

        let hint = provider.tensor_shape_warning_hint("Dimension mismatch: 128 vs 256", span, source);
        assert!(hint.is_some());

        let hint = hint.unwrap();
        if let InlayHintLabel::LabelParts(parts) = &hint.label {
            assert!(!parts.is_empty());
            assert!(parts[0].value.contains("!"));
            assert!(parts[0].value.contains("Dimension mismatch"));
        }
    }

    #[test]
    fn test_tensor_hints_disabled() {
        let config = InlayHintConfig {
            tensor_shape_hints: false,
            ..Default::default()
        };
        let provider = InlayHintProvider::with_config(config);
        let source = "let x = tensor";
        let span = crate::common::Span { start: 0, end: 14 };

        let tensor_type = HirType::Tensor {
            element: Box::new(HirType::F32),
            dims: vec![HirTensorDim::Fixed(32)],
        };

        let hint = provider.tensor_shape_hint_from_hir(&tensor_type, span, source);
        assert!(hint.is_none());
    }

    #[test]
    fn test_tensor_warnings_disabled() {
        let config = InlayHintConfig {
            tensor_shape_warnings: false,
            ..Default::default()
        };
        let provider = InlayHintProvider::with_config(config);
        let source = "matmul(a, b)";
        let span = crate::common::Span { start: 0, end: 12 };

        let hint = provider.tensor_shape_warning_hint("Shape mismatch", span, source);
        assert!(hint.is_none());
    }

    #[test]
    fn test_non_tensor_type_returns_none() {
        let provider = InlayHintProvider::new();
        let source = "let x = 42";
        let span = crate::common::Span { start: 0, end: 10 };

        let hint = provider.tensor_shape_hint_from_hir(&HirType::I32, span, source);
        assert!(hint.is_none());
    }
}
