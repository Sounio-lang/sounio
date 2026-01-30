//! Rename Refactoring
//!
//! Provides symbol renaming with the following capabilities:
//! - Rename local variables, parameters, and functions
//! - Rename types (structs, enums, traits)
//! - Rename fields and methods
//! - Update all references across files
//! - Conflict detection

use super::{
    RefactoringContext, ReferenceKind, SymbolLocation, TextEdit, WorkspaceEdit, validate_new_name,
};
use crate::ast::{Ast, Expr, Item, Pattern, Stmt, TypeExpr};
use crate::common::Span;
use std::collections::HashSet;

/// Rename refactoring operation
pub struct RenameRefactoring {
    /// The symbol to rename
    pub old_name: String,
    /// The new name for the symbol
    pub new_name: String,
    /// Position where rename was triggered (for finding symbol)
    pub position: usize,
    /// Whether to preview changes without applying
    pub preview_only: bool,
}

impl RenameRefactoring {
    /// Create a new rename operation
    pub fn new(old_name: impl Into<String>, new_name: impl Into<String>) -> Self {
        Self {
            old_name: old_name.into(),
            new_name: new_name.into(),
            position: 0,
            preview_only: false,
        }
    }

    /// Create rename at a specific position
    pub fn at_position(position: usize, new_name: impl Into<String>) -> Self {
        Self {
            old_name: String::new(), // Will be determined from position
            new_name: new_name.into(),
            position,
            preview_only: false,
        }
    }

    /// Set preview mode
    pub fn preview(mut self) -> Self {
        self.preview_only = true;
        self
    }

    /// Execute the rename operation
    pub fn execute(&self, ctx: &RefactoringContext) -> Result<RenameResult, RenameError> {
        // Validate new name
        validate_new_name(&self.new_name).map_err(|e| RenameError::InvalidName(e.to_string()))?;

        let ast = ctx.current_ast().ok_or_else(|| RenameError::NoModule)?;
        let source = ctx.current_source().ok_or_else(|| RenameError::NoSource)?;

        // Find the symbol at position or by name
        let symbol_name = if self.old_name.is_empty() {
            self.find_symbol_at_position(source, self.position)?
        } else {
            self.old_name.clone()
        };

        // Check for conflicts
        if self.check_name_conflict(ast, &self.new_name) {
            return Err(RenameError::Conflict(self.new_name.clone()));
        }

        // Find all references
        let references = self.find_all_references(ast, &symbol_name, &ctx.current_file);

        if references.is_empty() {
            return Err(RenameError::SymbolNotFound(symbol_name));
        }

        // Generate edits
        let mut workspace_edit = WorkspaceEdit::new();
        for loc in &references {
            let edit = TextEdit::replace(loc.span.start..loc.span.end, self.new_name.clone());
            workspace_edit.add_edit(&loc.file, edit);
        }

        Ok(RenameResult {
            old_name: symbol_name,
            new_name: self.new_name.clone(),
            references,
            edits: workspace_edit,
        })
    }

    /// Find the symbol at the given position
    fn find_symbol_at_position(
        &self,
        source: &str,
        position: usize,
    ) -> Result<String, RenameError> {
        // Simple approach: find identifier boundaries around position
        let chars: Vec<char> = source.chars().collect();

        if position >= chars.len() {
            return Err(RenameError::InvalidPosition(position));
        }

        // Find start of identifier
        let mut start = position;
        while start > 0 {
            let c = chars[start - 1];
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            start -= 1;
        }

        // Find end of identifier
        let mut end = position;
        while end < chars.len() {
            let c = chars[end];
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            end += 1;
        }

        if start == end {
            return Err(RenameError::InvalidPosition(position));
        }

        let name: String = chars[start..end].iter().collect();
        Ok(name)
    }

    /// Check if new name conflicts with existing symbols
    fn check_name_conflict(&self, ast: &Ast, new_name: &str) -> bool {
        for item in &ast.items {
            match item {
                Item::Function(f) if f.name == new_name => return true,
                Item::Struct(s) if s.name == new_name => return true,
                Item::Enum(e) if e.name == new_name => return true,
                Item::Const(c) if c.name == new_name => return true,
                Item::TypeAlias(t) if t.name == new_name => return true,
                Item::Trait(t) if t.name == new_name => return true,
                _ => {}
            }
        }
        false
    }

    /// Find all references to a symbol
    fn find_all_references(&self, ast: &Ast, name: &str, file: &str) -> Vec<SymbolLocation> {
        let mut finder = ReferenceFinder::new(name, file);
        finder.visit_ast(ast);
        finder.locations
    }
}

/// Result of a rename operation
#[derive(Debug, Clone)]
pub struct RenameResult {
    /// Original name
    pub old_name: String,
    /// New name
    pub new_name: String,
    /// All locations that were renamed
    pub references: Vec<SymbolLocation>,
    /// Edits to apply
    pub edits: WorkspaceEdit,
}

impl RenameResult {
    /// Number of references renamed
    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    /// Number of files affected
    pub fn file_count(&self) -> usize {
        self.edits.file_count()
    }
}

/// Errors that can occur during rename
#[derive(Debug, Clone, thiserror::Error)]
pub enum RenameError {
    #[error("Invalid name: {0}")]
    InvalidName(String),

    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),

    #[error("Invalid position: {0}")]
    InvalidPosition(usize),

    #[error("Name conflict: {0} already exists in scope")]
    Conflict(String),

    #[error("Cannot rename: {0}")]
    CannotRename(String),

    #[error("No module available")]
    NoModule,

    #[error("No source available")]
    NoSource,
}

/// Helper to find all references to a symbol
struct ReferenceFinder {
    /// Name to find
    name: String,
    /// Current file
    file: String,
    /// Found locations
    locations: Vec<SymbolLocation>,
    /// Scopes for tracking local variables
    scopes: Vec<HashSet<String>>,
}

impl ReferenceFinder {
    fn new(name: &str, file: &str) -> Self {
        Self {
            name: name.to_string(),
            file: file.to_string(),
            locations: Vec::new(),
            scopes: Vec::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashSet::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn add_to_scope(&mut self, name: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string());
        }
    }

    fn is_in_scope(&self, name: &str) -> bool {
        self.scopes.iter().any(|s| s.contains(name))
    }

    fn add_location(&mut self, span: Span, kind: ReferenceKind) {
        self.locations.push(SymbolLocation {
            file: self.file.clone(),
            span,
            kind,
        });
    }

    fn visit_ast(&mut self, ast: &Ast) {
        for item in &ast.items {
            self.visit_item(item);
        }
    }

    fn visit_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => {
                // Check function name
                if f.name == self.name {
                    self.add_location(
                        Span {
                            file: f.span.file,
                            start: f.span.start,
                            end: f.span.start + f.name.len(),
                        },
                        ReferenceKind::Definition,
                    );
                }

                self.push_scope();

                // Add parameters to scope
                for param in &f.params {
                    if param.name == self.name {
                        self.add_location(param.span.clone(), ReferenceKind::Definition);
                    }
                    self.add_to_scope(&param.name);
                    self.visit_type(&param.ty);
                }

                // Visit return type
                if let Some(ret) = &f.return_type {
                    self.visit_type(ret);
                }

                // Visit body
                if let Some(body) = &f.body {
                    self.visit_expr(body);
                }

                self.pop_scope();
            }
            Item::Struct(s) => {
                if s.name == self.name {
                    self.add_location(
                        Span {
                            file: s.span.file,
                            start: s.span.start,
                            end: s.span.start + s.name.len(),
                        },
                        ReferenceKind::Definition,
                    );
                }
                for field in &s.fields {
                    if field.name == self.name {
                        self.add_location(field.span.clone(), ReferenceKind::Definition);
                    }
                    self.visit_type(&field.ty);
                }
            }
            Item::Enum(e) => {
                if e.name == self.name {
                    self.add_location(
                        Span {
                            file: e.span.file,
                            start: e.span.start,
                            end: e.span.start + e.name.len(),
                        },
                        ReferenceKind::Definition,
                    );
                }
                for variant in &e.variants {
                    if variant.name == self.name {
                        self.add_location(variant.span.clone(), ReferenceKind::Definition);
                    }
                    for field in &variant.fields {
                        self.visit_type(&field.ty);
                    }
                }
            }
            Item::Const(c) => {
                if c.name == self.name {
                    self.add_location(
                        Span {
                            file: c.span.file,
                            start: c.span.start,
                            end: c.span.start + c.name.len(),
                        },
                        ReferenceKind::Definition,
                    );
                }
                self.visit_type(&c.ty);
                self.visit_expr(&c.value);
            }
            Item::TypeAlias(t) => {
                if t.name == self.name {
                    self.add_location(
                        Span {
                            file: t.span.file,
                            start: t.span.start,
                            end: t.span.start + t.name.len(),
                        },
                        ReferenceKind::Definition,
                    );
                }
                self.visit_type(&t.ty);
            }
            Item::Trait(t) => {
                if t.name == self.name {
                    self.add_location(
                        Span {
                            file: t.span.file,
                            start: t.span.start,
                            end: t.span.start + t.name.len(),
                        },
                        ReferenceKind::Definition,
                    );
                }
                for method in &t.methods {
                    self.visit_item(&Item::Function(method.clone()));
                }
            }
            Item::Impl(i) => {
                self.visit_type(&i.self_type);
                if let Some(trait_name) = &i.trait_name {
                    if trait_name == &self.name {
                        self.add_location(i.span.clone(), ReferenceKind::Reference);
                    }
                }
                for method in &i.methods {
                    self.visit_item(&Item::Function(method.clone()));
                }
            }
            Item::Import(imp) => {
                // Check import path
                for segment in &imp.path {
                    if segment == &self.name {
                        self.add_location(imp.span.clone(), ReferenceKind::Import);
                        break;
                    }
                }
            }
            Item::Module(m) => {
                if m.name == self.name {
                    self.add_location(m.span.clone(), ReferenceKind::Definition);
                }
            }
            Item::Effect(_) | Item::Handler(_) => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Identifier(name, span) => {
                if name == &self.name {
                    self.add_location(span.clone(), ReferenceKind::Reference);
                }
            }
            Expr::Call { callee, args, .. } => {
                self.visit_expr(callee);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                span,
                ..
            } => {
                self.visit_expr(receiver);
                if method == &self.name {
                    // Method name reference
                    self.add_location(span.clone(), ReferenceKind::Reference);
                }
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::FieldAccess { expr, field, span } => {
                self.visit_expr(expr);
                if field == &self.name {
                    self.add_location(span.clone(), ReferenceKind::Reference);
                }
            }
            Expr::Index { expr, index, .. } => {
                self.visit_expr(expr);
                self.visit_expr(index);
            }
            Expr::Binary { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            Expr::Unary { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.visit_expr(condition);
                self.visit_expr(then_branch);
                if let Some(else_br) = else_branch {
                    self.visit_expr(else_br);
                }
            }
            Expr::Match { expr, arms, .. } => {
                self.visit_expr(expr);
                for arm in arms {
                    self.push_scope();
                    self.visit_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                    self.pop_scope();
                }
            }
            Expr::Block { stmts, expr, .. } => {
                self.push_scope();
                for stmt in stmts {
                    self.visit_stmt(stmt);
                }
                if let Some(e) = expr {
                    self.visit_expr(e);
                }
                self.pop_scope();
            }
            Expr::Loop { body, .. } => {
                self.visit_expr(body);
            }
            Expr::While {
                condition, body, ..
            } => {
                self.visit_expr(condition);
                self.visit_expr(body);
            }
            Expr::For {
                pattern,
                iterator,
                body,
                ..
            } => {
                self.visit_expr(iterator);
                self.push_scope();
                self.visit_pattern(pattern);
                self.visit_expr(body);
                self.pop_scope();
            }
            Expr::Return { value, .. } => {
                if let Some(v) = value {
                    self.visit_expr(v);
                }
            }
            Expr::Break { value, .. } => {
                if let Some(v) = value {
                    self.visit_expr(v);
                }
            }
            Expr::Array { elements, .. } => {
                for elem in elements {
                    self.visit_expr(elem);
                }
            }
            Expr::Tuple { elements, .. } => {
                for elem in elements {
                    self.visit_expr(elem);
                }
            }
            Expr::StructInit { name, fields, span } => {
                if name == &self.name {
                    self.add_location(span.clone(), ReferenceKind::Reference);
                }
                for (field_name, value) in fields {
                    if field_name == &self.name {
                        // Field reference
                    }
                    self.visit_expr(value);
                }
            }
            Expr::Lambda { params, body, span } => {
                self.push_scope();
                for param in params {
                    if param.name == self.name {
                        self.add_location(span.clone(), ReferenceKind::Definition);
                    }
                    self.add_to_scope(&param.name);
                }
                self.visit_expr(body);
                self.pop_scope();
            }
            Expr::Reference { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::Dereference { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::Cast { expr, ty, .. } => {
                self.visit_expr(expr);
                self.visit_type(ty);
            }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.visit_expr(s);
                }
                if let Some(e) = end {
                    self.visit_expr(e);
                }
            }
            Expr::Try { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::Await { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::Perform { effect, args, span } => {
                if effect == &self.name {
                    self.add_location(span.clone(), ReferenceKind::Reference);
                }
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::Handle {
                expr,
                handlers,
                finally,
                ..
            } => {
                self.visit_expr(expr);
                for handler in handlers {
                    self.push_scope();
                    for param in &handler.params {
                        self.add_to_scope(&param.name);
                    }
                    self.visit_expr(&handler.body);
                    self.pop_scope();
                }
                if let Some(f) = finally {
                    self.visit_expr(f);
                }
            }
            Expr::Resume { value, .. } => {
                if let Some(v) = value {
                    self.visit_expr(v);
                }
            }
            // Literals have no references
            _ => {}
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                pattern, ty, value, ..
            } => {
                if let Some(t) = ty {
                    self.visit_type(t);
                }
                if let Some(v) = value {
                    self.visit_expr(v);
                }
                self.visit_pattern(pattern);
            }
            Stmt::Expr(expr) => {
                self.visit_expr(expr);
            }
            Stmt::Item(item) => {
                self.visit_item(item);
            }
        }
    }

    fn visit_pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Identifier(name, span) => {
                if name == &self.name {
                    self.add_location(span.clone(), ReferenceKind::PatternBinding);
                }
                self.add_to_scope(name);
            }
            Pattern::Tuple(patterns, _) => {
                for p in patterns {
                    self.visit_pattern(p);
                }
            }
            Pattern::Struct {
                name, fields, span, ..
            } => {
                if name == &self.name {
                    self.add_location(span.clone(), ReferenceKind::Reference);
                }
                for (field_name, pattern) in fields {
                    if field_name == &self.name {
                        // Field reference in pattern
                    }
                    self.visit_pattern(pattern);
                }
            }
            Pattern::Enum { name, fields, span } => {
                if name == &self.name {
                    self.add_location(span.clone(), ReferenceKind::Reference);
                }
                for field in fields {
                    self.visit_pattern(field);
                }
            }
            Pattern::Or(patterns, _) => {
                for p in patterns {
                    self.visit_pattern(p);
                }
            }
            Pattern::Wildcard | Pattern::Literal(_) => {}
        }
    }

    fn visit_type(&mut self, ty: &TypeExpr) {
        match ty {
            TypeExpr::Named { path, args, .. } => {
                // Check if the type name matches
                if let Some(name) = path.segments.last() {
                    if name == &self.name {
                        // We don't have a span on TypeExpr, so we can't add location precisely
                        // This would require AST enhancement to track spans
                    }
                }
                for arg in args {
                    self.visit_type(arg);
                }
            }
            TypeExpr::Array { element, .. } => {
                self.visit_type(element);
            }
            TypeExpr::Tuple(elements) => {
                for elem in elements {
                    self.visit_type(elem);
                }
            }
            TypeExpr::Function {
                params,
                return_type,
                ..
            } => {
                for param in params {
                    self.visit_type(param);
                }
                self.visit_type(return_type);
            }
            TypeExpr::Reference { inner, .. } => {
                self.visit_type(inner);
            }
            TypeExpr::Unit | TypeExpr::SelfType | TypeExpr::Infer => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rename_refactoring_new() {
        let rename = RenameRefactoring::new("old", "new");
        assert_eq!(rename.old_name, "old");
        assert_eq!(rename.new_name, "new");
        assert!(!rename.preview_only);
    }

    #[test]
    fn test_rename_at_position() {
        let rename = RenameRefactoring::at_position(42, "newName");
        assert_eq!(rename.position, 42);
        assert_eq!(rename.new_name, "newName");
    }

    #[test]
    fn test_rename_preview() {
        let rename = RenameRefactoring::new("old", "new").preview();
        assert!(rename.preview_only);
    }

    #[test]
    fn test_find_symbol_at_position() {
        let rename = RenameRefactoring::new("", "");
        let source = "let myVariable = 42";

        // Position in "myVariable"
        let result = rename.find_symbol_at_position(source, 5);
        assert_eq!(result.unwrap(), "myVariable");

        // Position in "let"
        let result = rename.find_symbol_at_position(source, 1);
        assert_eq!(result.unwrap(), "let");
    }
}
