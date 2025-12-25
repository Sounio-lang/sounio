//! Ownership checker implementation
//!
//! Checks ownership, borrowing, and linearity rules.

use crate::ast::{self, Ast, BinaryOp, Expr, Item, Stmt, TypeExpr, UnaryOp};
use crate::common::{NodeId, Span};
use crate::diagnostics::{CompileError, SourceFile};
use crate::resolve::{DefId, SymbolTable};

use super::state::*;
use std::collections::HashMap;

/// Ownership and borrow checker
pub struct OwnershipChecker<'a> {
    symbols: &'a SymbolTable,
    source: &'a SourceFile,
    /// Node span map from AST
    node_spans: &'a HashMap<NodeId, Span>,
    /// Scope stack
    scopes: Vec<ScopeState>,
    /// Type linearity cache (for structs)
    linearity_cache: HashMap<DefId, Linearity>,
    /// Closure capture tracking
    closure_captures: HashMap<NodeId, Vec<CapturedVar>>,
    /// Errors
    errors: Vec<CompileError>,
}

/// A captured variable in a closure
#[derive(Debug, Clone)]
pub struct CapturedVar {
    pub def_id: DefId,
    pub name: String,
    pub capture_kind: CaptureKind,
    pub span: Span,
}

/// How a variable is captured
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    /// Captured by shared reference
    ByRef,
    /// Captured by exclusive reference
    ByRefMut,
    /// Captured by move (ownership transferred)
    ByMove,
}

impl<'a> OwnershipChecker<'a> {
    pub fn new(
        symbols: &'a SymbolTable,
        source: &'a SourceFile,
        node_spans: &'a HashMap<NodeId, Span>,
    ) -> Self {
        Self {
            symbols,
            source,
            node_spans,
            scopes: vec![ScopeState::new()],
            linearity_cache: HashMap::new(),
            closure_captures: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Get span for a NodeId
    fn span_for(&self, id: NodeId) -> Span {
        self.node_spans
            .get(&id)
            .copied()
            .unwrap_or_else(Span::dummy)
    }

    /// Check entire program
    pub fn check_program(&mut self, ast: &Ast) -> Result<(), Vec<CompileError>> {
        // Build linearity cache from struct definitions
        for item in &ast.items {
            if let Item::Struct(s) = item
                && let Some(def_id) = self.symbols.def_for_node(s.id)
            {
                let linearity = if s.modifiers.linear {
                    Linearity::Linear
                } else if s.modifiers.affine {
                    Linearity::Affine
                } else {
                    Linearity::Unrestricted
                };
                self.linearity_cache.insert(def_id, linearity);
            }
        }

        // Check functions
        for item in &ast.items {
            if let Item::Function(f) = item {
                self.check_function(f);
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn check_function(&mut self, f: &ast::FnDef) {
        self.push_scope();

        // Track parameters with ownership annotations
        for param in &f.params {
            if let Some(def_id) = self.symbols.def_for_node(param.id) {
                let linearity = self.get_type_linearity(&param.ty);
                let name = self.get_pattern_name(&param.pattern);
                let param_span = self.span_for(param.id);
                self.track_value(def_id, name, linearity, param_span);
            }
        }

        // Check body
        self.check_block(&f.body);

        // Check linear values consumed
        self.check_scope_end(f.span);
        self.pop_scope();
    }

    fn check_block(&mut self, block: &ast::Block) {
        self.push_scope();

        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }

        self.check_scope_end(Span::dummy());
        self.pop_scope();
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                pattern, ty, value, ..
            } => {
                // Check initializer first
                if let Some(init) = value {
                    self.check_expr(init, UseKind::Move);
                }

                // Track the binding
                let name = self.get_pattern_name(pattern);
                if let Some(def_id) = self.get_pattern_def_id(pattern) {
                    let linearity = if let Some(ty_expr) = ty {
                        self.get_type_linearity(ty_expr)
                    } else {
                        Linearity::Unrestricted
                    };
                    self.track_value(def_id, name, linearity, Span::dummy());
                }
            }

            Stmt::Expr { expr, .. } => {
                self.check_expr(expr, UseKind::Move);
            }

            Stmt::Assign { target, value, .. } => {
                self.check_expr(value, UseKind::Move);
                // Target is being written to, not consumed
            }

            Stmt::Empty | Stmt::MacroInvocation(_) => {}
        }
    }

    fn check_expr(&mut self, expr: &Expr, use_kind: UseKind) {
        let expr_span = self.span_for(get_expr_id(expr));

        match expr {
            Expr::Literal { .. } => {}

            Expr::Path { path, id } => {
                if path.is_simple()
                    && let Some(def_id) = self.symbols.ref_for_node(*id)
                {
                    self.use_value(def_id, use_kind, expr_span);
                }
            }

            Expr::Binary {
                op, left, right, ..
            } => {
                // Assignment: RHS is moved, LHS is target
                if matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                ) {
                    self.check_expr(left, UseKind::Copy);
                    self.check_expr(right, UseKind::Copy);
                } else {
                    self.check_expr(left, UseKind::Copy);
                    self.check_expr(right, UseKind::Copy);
                }
            }

            Expr::Unary {
                op, expr: inner, ..
            } => {
                match op {
                    UnaryOp::Ref => {
                        // Shared borrow
                        if let Some(place) = self.expr_to_place(inner) {
                            self.borrow_shared(place, expr_span);
                        }
                    }
                    UnaryOp::RefMut => {
                        // Exclusive borrow (&!)
                        if let Some(place) = self.expr_to_place(inner) {
                            self.borrow_exclusive(place, expr_span);
                        }
                    }
                    UnaryOp::Deref => {
                        self.check_expr(inner, UseKind::Copy);
                    }
                    _ => {
                        self.check_expr(inner, use_kind);
                    }
                }
            }

            Expr::Call { callee, args, .. } => {
                self.check_expr(callee, UseKind::Copy);
                // Check arguments - by default, arguments are moved unless the
                // parameter type is a reference or the type is Copy
                for arg in args {
                    // For now, assume move semantics for non-primitive arguments
                    // A more complete implementation would look up the function
                    // signature and check parameter ownership annotations
                    self.check_expr(arg, UseKind::Move);
                }
            }

            Expr::MethodCall {
                receiver,
                method,
                args,
                id,
            } => {
                // Check receiver ownership based on method signature
                // Methods with &self take a shared borrow, &!self takes exclusive
                // For now, default to Copy (shared borrow semantics)
                // A full implementation would look up the method signature
                let receiver_use = self.infer_method_receiver_use(method, *id);
                self.check_expr(receiver, receiver_use);
                for arg in args {
                    self.check_expr(arg, UseKind::Move);
                }
            }

            Expr::Field { base, .. } => {
                self.check_expr(base, UseKind::Copy);
            }

            Expr::TupleField { base, .. } => {
                self.check_expr(base, UseKind::Copy);
            }

            Expr::Index { base, index, .. } => {
                self.check_expr(base, UseKind::Copy);
                self.check_expr(index, UseKind::Copy);
            }

            Expr::Cast { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::Block { block, .. } => {
                self.check_block(block);
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.check_expr(condition, UseKind::Copy);
                self.check_block(then_branch);
                if let Some(else_expr) = else_branch {
                    self.check_expr(else_expr, use_kind);
                }
            }

            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.check_expr(scrutinee, UseKind::Move);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard, UseKind::Copy);
                    }
                    self.check_expr(&arm.body, use_kind);
                }
            }

            Expr::Loop { body, .. } => {
                self.check_block(body);
            }

            Expr::While {
                condition, body, ..
            } => {
                self.check_expr(condition, UseKind::Copy);
                self.check_block(body);
            }

            Expr::For {
                pattern,
                iter,
                body,
                id,
            } => {
                self.check_expr(iter, UseKind::Move);
                self.push_scope();

                // Bind pattern variables for the loop iteration variable
                self.bind_pattern_variables(pattern);

                self.check_block(body);
                let for_span = self.span_for(*id);
                self.check_scope_end(for_span);
                self.pop_scope();
            }

            Expr::Return { value, .. } => {
                if let Some(val) = value {
                    self.check_expr(val, UseKind::Move);
                }
            }

            Expr::Break { value, .. } => {
                if let Some(val) = value {
                    self.check_expr(val, UseKind::Move);
                }
            }

            Expr::Continue { .. } => {}

            Expr::Closure {
                params, body, id, ..
            } => {
                // Analyze captures: find free variables used in the closure body
                let captures = self.analyze_closure_captures(body, params);

                // Record captures for this closure
                self.closure_captures.insert(*id, captures.clone());

                // Check each captured variable
                for capture in &captures {
                    match capture.capture_kind {
                        CaptureKind::ByRef => {
                            // Shared borrow of the captured variable
                            let place = Place::var(capture.def_id);
                            self.borrow_shared(place, capture.span);
                        }
                        CaptureKind::ByRefMut => {
                            // Exclusive borrow of the captured variable
                            let place = Place::var(capture.def_id);
                            self.borrow_exclusive(place, capture.span);
                        }
                        CaptureKind::ByMove => {
                            // Move the captured variable
                            self.use_value(capture.def_id, UseKind::Move, capture.span);
                        }
                    }
                }

                // Check the closure body in a new scope
                self.push_scope();
                self.shadow_captures(&captures);
                self.check_expr(body, UseKind::Move);
                let closure_span = self.span_for(*id);
                self.check_scope_end(closure_span);
                self.pop_scope();
            }

            Expr::Tuple { elements, .. } => {
                for elem in elements {
                    self.check_expr(elem, use_kind);
                }
            }

            Expr::Array { elements, .. } => {
                for elem in elements {
                    self.check_expr(elem, use_kind);
                }
            }

            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.check_expr(s, use_kind);
                }
                if let Some(e) = end {
                    self.check_expr(e, use_kind);
                }
            }

            Expr::StructLit { fields, .. } => {
                for (_, field_expr) in fields {
                    self.check_expr(field_expr, UseKind::Move);
                }
            }

            Expr::Try { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::Perform { args, .. } => {
                for arg in args {
                    self.check_expr(arg, UseKind::Move);
                }
            }

            Expr::Handle { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::Sample { distribution, .. } => {
                self.check_expr(distribution, UseKind::Move);
            }

            Expr::Await { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::AsyncBlock { block, .. } => {
                self.check_block(block);
            }

            Expr::AsyncClosure {
                params, body, id, ..
            } => {
                // Analyze captures for async closure (same as regular closure)
                let captures = self.analyze_closure_captures(body, params);
                self.closure_captures.insert(*id, captures.clone());

                for capture in &captures {
                    match capture.capture_kind {
                        CaptureKind::ByRef => {
                            let place = Place::var(capture.def_id);
                            self.borrow_shared(place, capture.span);
                        }
                        CaptureKind::ByRefMut => {
                            let place = Place::var(capture.def_id);
                            self.borrow_exclusive(place, capture.span);
                        }
                        CaptureKind::ByMove => {
                            self.use_value(capture.def_id, UseKind::Move, capture.span);
                        }
                    }
                }

                self.push_scope();
                self.shadow_captures(&captures);
                self.check_expr(body, UseKind::Move);
                let closure_span = self.span_for(*id);
                self.check_scope_end(closure_span);
                self.pop_scope();
            }

            Expr::Spawn { expr, .. } => {
                self.check_expr(expr, UseKind::Move);
            }

            Expr::Select { arms, .. } => {
                for arm in arms {
                    self.check_expr(&arm.future, UseKind::Move);
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard, UseKind::Copy);
                    }
                    self.check_expr(&arm.body, use_kind);
                }
            }

            Expr::Join { futures, .. } => {
                for future in futures {
                    self.check_expr(future, UseKind::Move);
                }
            }

            Expr::MacroInvocation(_) => {}

            // Epistemic expressions
            Expr::Do { interventions, .. } => {
                for (_, value) in interventions {
                    self.check_expr(value, UseKind::Copy);
                }
            }

            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => {
                self.check_expr(factual, UseKind::Copy);
                self.check_expr(intervention, UseKind::Copy);
                self.check_expr(outcome, use_kind);
            }

            Expr::KnowledgeExpr {
                value,
                epsilon,
                validity,
                provenance,
                ..
            } => {
                self.check_expr(value, use_kind);
                if let Some(e) = epsilon {
                    self.check_expr(e, UseKind::Copy);
                }
                if let Some(v) = validity {
                    self.check_expr(v, UseKind::Copy);
                }
                if let Some(p) = provenance {
                    self.check_expr(p, UseKind::Copy);
                }
            }

            Expr::Uncertain {
                value, uncertainty, ..
            } => {
                self.check_expr(value, use_kind);
                self.check_expr(uncertainty, UseKind::Copy);
            }

            Expr::GpuAnnotated { expr, .. } => {
                self.check_expr(expr, use_kind);
            }

            Expr::Observe {
                data, distribution, ..
            } => {
                self.check_expr(data, UseKind::Copy);
                self.check_expr(distribution, UseKind::Copy);
            }

            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                self.check_expr(target, UseKind::Copy);
                for g in given {
                    self.check_expr(g, UseKind::Copy);
                }
                for (_, value) in interventions {
                    self.check_expr(value, UseKind::Copy);
                }
            }

            // Ontology term literals are value types, no ownership concerns
            Expr::OntologyTerm { .. } => {}
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(ScopeState::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn current_scope(&mut self) -> &mut ScopeState {
        self.scopes.last_mut().unwrap()
    }

    fn track_value(&mut self, def_id: DefId, name: String, linearity: Linearity, span: Span) {
        let value = TrackedValue::new(def_id, name, linearity, span);
        self.current_scope().track(value);
    }

    fn use_value(&mut self, def_id: DefId, use_kind: UseKind, span: Span) {
        // Look up in all scopes
        for scope in self.scopes.iter_mut().rev() {
            if let Some(value) = scope.get_mut(def_id) {
                // Check if already moved
                if let OwnershipState::Moved { to } = &value.state {
                    self.errors.push(CompileError::UseAfterMove {
                        name: value.name.clone(),
                        use_span: span.into(),
                        move_span: (*to).into(),
                        src: self.source.to_named_source(),
                    });
                    return;
                }

                // Record use
                value.record_use(span);

                // Update state for move
                if use_kind == UseKind::Move && value.linearity != Linearity::Unrestricted {
                    value.state = OwnershipState::Moved { to: span };
                }

                return;
            }
        }
    }

    fn borrow_shared(&mut self, place: Place, span: Span) {
        // Check if exclusively borrowed
        for scope in &self.scopes {
            let active = scope.active_borrows(&place);
            for borrow in active {
                if borrow.exclusive {
                    self.errors.push(CompileError::AlreadyBorrowed {
                        name: place.to_string(),
                        span: span.into(),
                        prev_span: borrow.span.into(),
                        src: self.source.to_named_source(),
                    });
                    return;
                }
            }
        }

        // Add borrow
        self.current_scope().add_borrow(BorrowState {
            place,
            exclusive: false,
            span,
            active: true,
        });
    }

    fn borrow_exclusive(&mut self, place: Place, span: Span) {
        // Check if any borrows exist
        for scope in &self.scopes {
            let active = scope.active_borrows(&place);
            if !active.is_empty() {
                let prev = active[0];
                if prev.exclusive {
                    self.errors.push(CompileError::DoubleMutBorrow {
                        name: place.to_string(),
                        span: span.into(),
                        first_span: prev.span.into(),
                        src: self.source.to_named_source(),
                    });
                } else {
                    self.errors.push(CompileError::AlreadyBorrowed {
                        name: place.to_string(),
                        span: span.into(),
                        prev_span: prev.span.into(),
                        src: self.source.to_named_source(),
                    });
                }
                return;
            }
        }

        // Add borrow
        self.current_scope().add_borrow(BorrowState {
            place,
            exclusive: true,
            span,
            active: true,
        });
    }

    fn check_scope_end(&mut self, scope_end_span: Span) {
        let errors = self.current_scope().check_all_linear();

        for error in errors {
            match error {
                LinearityError::NotConsumed {
                    name, decl_span, ..
                } => {
                    self.errors.push(CompileError::LinearNotConsumed {
                        name,
                        decl_span: decl_span.into(),
                        scope_end: scope_end_span.into(),
                        src: self.source.to_named_source(),
                    });
                }
                LinearityError::MultipleUse {
                    name,
                    first,
                    second,
                    ..
                } => {
                    self.errors.push(CompileError::LinearMultipleUse {
                        name,
                        first_span: first.into(),
                        second_span: second.into(),
                        src: self.source.to_named_source(),
                    });
                }
            }
        }
    }

    fn get_type_linearity(&self, ty: &TypeExpr) -> Linearity {
        match ty {
            TypeExpr::Named { path, .. } => {
                if let Some(name) = path.name() {
                    // Check if it's a known linear/affine type
                    for (def_id, linearity) in &self.linearity_cache {
                        if let Some(sym) = self.symbols.get(*def_id)
                            && sym.name == name
                        {
                            return *linearity;
                        }
                    }
                }
                Linearity::Unrestricted
            }
            TypeExpr::Reference { .. } => Linearity::Unrestricted,
            TypeExpr::Tuple(elems) => {
                // If any element is linear, the tuple is linear
                for elem in elems {
                    let elem_lin = self.get_type_linearity(elem);
                    if elem_lin == Linearity::Linear {
                        return Linearity::Linear;
                    }
                }
                Linearity::Unrestricted
            }
            _ => Linearity::Unrestricted,
        }
    }

    fn expr_to_place(&self, expr: &Expr) -> Option<Place> {
        match expr {
            Expr::Path { path, id } => {
                if path.is_simple()
                    && let Some(def_id) = self.symbols.ref_for_node(*id)
                {
                    return Some(Place::var(def_id));
                }
                None
            }
            Expr::Field { base, field, .. } => {
                self.expr_to_place(base).map(|p| p.field(field.clone()))
            }
            Expr::Unary { op, expr, .. } if matches!(op, UnaryOp::Deref) => {
                self.expr_to_place(expr).map(|p| p.deref())
            }
            _ => None,
        }
    }

    fn get_pattern_name(&self, pattern: &ast::Pattern) -> String {
        match pattern {
            ast::Pattern::Binding { name, .. } => name.clone(),
            _ => "<pattern>".to_string(),
        }
    }

    fn get_pattern_def_id(&self, pattern: &ast::Pattern) -> Option<DefId> {
        match pattern {
            ast::Pattern::Binding { name, .. } => {
                // Look up the binding in symbols
                // This is a simplification - in practice we'd have NodeId on patterns
                self.symbols.lookup(name)
            }
            _ => None,
        }
    }

    /// Bind pattern variables in the current scope
    fn bind_pattern_variables(&mut self, pattern: &ast::Pattern) {
        match pattern {
            ast::Pattern::Binding { name, .. } => {
                // Look up the binding in the symbol table
                if let Some(def_id) = self.symbols.lookup(name) {
                    // Default to unrestricted linearity for pattern bindings
                    // A more complete implementation would infer from the matched type
                    self.track_value(def_id, name.clone(), Linearity::Unrestricted, Span::dummy());
                }
            }
            ast::Pattern::Tuple(elements) => {
                for elem in elements {
                    self.bind_pattern_variables(elem);
                }
            }
            ast::Pattern::Struct { fields, .. } => {
                for (_, field_pattern) in fields {
                    self.bind_pattern_variables(field_pattern);
                }
            }
            ast::Pattern::Enum { patterns, .. } => {
                if let Some(field_patterns) = patterns {
                    for field_pattern in field_patterns {
                        self.bind_pattern_variables(field_pattern);
                    }
                }
            }
            ast::Pattern::Or(patterns) => {
                // For or-patterns, bind variables from first pattern
                // (all alternatives should bind the same variables)
                if let Some(first) = patterns.first() {
                    self.bind_pattern_variables(first);
                }
            }
            // These patterns don't introduce bindings
            ast::Pattern::Wildcard | ast::Pattern::Literal(_) => {}
        }
    }

    fn shadow_captures(&mut self, captures: &[CapturedVar]) {
        for capture in captures {
            let linearity = self
                .scopes
                .iter()
                .rev()
                .find_map(|scope| scope.get(capture.def_id).map(|value| value.linearity))
                .unwrap_or(Linearity::Unrestricted);
            self.track_value(
                capture.def_id,
                capture.name.clone(),
                linearity,
                capture.span,
            );
        }
    }

    /// Infer the use kind for a method receiver based on method name
    /// This is a heuristic - a full implementation would look up the method signature
    fn infer_method_receiver_use(&self, method: &str, _id: NodeId) -> UseKind {
        // Methods that typically mutate (take &!self)
        let mutating_methods = [
            "push", "pop", "insert", "remove", "clear", "set", "write", "extend", "append",
            "drain", "truncate", "reserve", "shrink",
        ];

        // Methods that consume (take self)
        let consuming_methods = [
            "into_iter",
            "into_inner",
            "into_boxed",
            "into_vec",
            "unwrap",
            "expect",
            "take",
        ];

        if consuming_methods.iter().any(|&m| method.starts_with(m)) {
            UseKind::Move
        } else if mutating_methods.contains(&method) {
            // For mutating methods, we still use Copy here because
            // the exclusive borrow is handled separately
            UseKind::Copy
        } else {
            // Default: shared borrow semantics
            UseKind::Copy
        }
    }

    /// Analyze closure captures by finding free variables in the body
    fn analyze_closure_captures(
        &self,
        body: &Expr,
        params: &[(String, Option<TypeExpr>)],
    ) -> Vec<CapturedVar> {
        let mut captures = Vec::new();
        let mut param_names: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect parameter names to exclude from captures
        for (name, _) in params {
            param_names.insert(name.clone());
        }

        // Find free variables in the body
        self.find_free_variables(body, &param_names, &mut captures);

        captures
    }

    /// Recursively find free variables in an expression
    fn find_free_variables(
        &self,
        expr: &Expr,
        bound_vars: &std::collections::HashSet<String>,
        captures: &mut Vec<CapturedVar>,
    ) {
        match expr {
            Expr::Path { path, id } => {
                if path.is_simple()
                    && let Some(name) = path.name()
                {
                    // Check if this is a free variable (not bound locally)
                    if !bound_vars.contains(name)
                        && let Some(def_id) = self.symbols.ref_for_node(*id)
                    {
                        // Check if already captured
                        if !captures.iter().any(|c| c.def_id == def_id) {
                            let span = self.span_for(*id);
                            // Default to ByRef capture; a more sophisticated
                            // analysis would determine if ByRefMut or ByMove is needed
                            captures.push(CapturedVar {
                                def_id,
                                name: name.to_string(),
                                capture_kind: CaptureKind::ByRef,
                                span,
                            });
                        }
                    }
                }
            }
            Expr::Binary { left, right, .. } => {
                self.find_free_variables(left, bound_vars, captures);
                self.find_free_variables(right, bound_vars, captures);
            }
            Expr::Unary { expr, .. } => {
                self.find_free_variables(expr, bound_vars, captures);
            }
            Expr::Call { callee, args, .. } => {
                self.find_free_variables(callee, bound_vars, captures);
                for arg in args {
                    self.find_free_variables(arg, bound_vars, captures);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.find_free_variables(receiver, bound_vars, captures);
                for arg in args {
                    self.find_free_variables(arg, bound_vars, captures);
                }
            }
            Expr::Field { base, .. } => {
                self.find_free_variables(base, bound_vars, captures);
            }
            Expr::Index { base, index, .. } => {
                self.find_free_variables(base, bound_vars, captures);
                self.find_free_variables(index, bound_vars, captures);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.find_free_variables(condition, bound_vars, captures);
                for stmt in &then_branch.stmts {
                    self.find_free_variables_stmt(stmt, bound_vars, captures);
                }
                if let Some(else_expr) = else_branch {
                    self.find_free_variables(else_expr, bound_vars, captures);
                }
            }
            Expr::Block { block, .. } => {
                for stmt in &block.stmts {
                    self.find_free_variables_stmt(stmt, bound_vars, captures);
                }
            }
            Expr::Closure { body, .. } => {
                // Nested closure - recurse but note that it creates a new scope
                self.find_free_variables(body, bound_vars, captures);
            }
            Expr::AsyncClosure { body, .. } => {
                self.find_free_variables(body, bound_vars, captures);
            }
            Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
                for elem in elements {
                    self.find_free_variables(elem, bound_vars, captures);
                }
            }
            Expr::StructLit { fields, .. } => {
                for (_, field_expr) in fields {
                    self.find_free_variables(field_expr, bound_vars, captures);
                }
            }
            // Other expressions - recurse into sub-expressions as needed
            _ => {}
        }
    }

    /// Find free variables in a statement
    fn find_free_variables_stmt(
        &self,
        stmt: &Stmt,
        bound_vars: &std::collections::HashSet<String>,
        captures: &mut Vec<CapturedVar>,
    ) {
        match stmt {
            Stmt::Let { value, .. } => {
                if let Some(init) = value {
                    self.find_free_variables(init, bound_vars, captures);
                }
            }
            Stmt::Expr { expr, .. } => {
                self.find_free_variables(expr, bound_vars, captures);
            }
            Stmt::Assign { target, value, .. } => {
                self.find_free_variables(target, bound_vars, captures);
                self.find_free_variables(value, bound_vars, captures);
            }
            Stmt::Empty | Stmt::MacroInvocation(_) => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UseKind {
    Move,
    Copy,
}

/// Extract NodeId from an expression
fn get_expr_id(expr: &Expr) -> NodeId {
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
        Expr::Range { id, .. } => *id,
        Expr::StructLit { id, .. } => *id,
        Expr::Try { id, .. } => *id,
        Expr::Perform { id, .. } => *id,
        Expr::Handle { id, .. } => *id,
        Expr::Sample { id, .. } => *id,
        Expr::Await { id, .. } => *id,
        Expr::AsyncBlock { id, .. } => *id,
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

/// Extract NodeId from a pattern (patterns don't have NodeIds in the current AST)
/// This returns a dummy NodeId - in practice patterns would need to be extended
/// to include NodeId for proper span tracking
fn get_pattern_id(_pattern: &ast::Pattern) -> NodeId {
    // The current Pattern enum doesn't include NodeId fields
    // For now, return a dummy NodeId
    NodeId::dummy()
}
