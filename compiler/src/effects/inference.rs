//! Effect inference pass
//!
//! Infers effects for expressions and checks that all effects are declared
//! in function signatures.

use crate::ast::{self, Ast, BinaryOp, Expr, ImplItem, Item, Stmt};
use crate::common::Span;
use crate::resolve::{DefId, SymbolTable};
use crate::types::core::{Effect, EffectSet};
use std::collections::HashMap;

/// Effect inference context
pub struct EffectChecker<'a> {
    symbols: &'a SymbolTable,
    /// Inferred effects per function DefId
    fn_effects: HashMap<DefId, EffectSet>,
    /// Method effects: (type_name, method_name) -> EffectSet
    method_effects: HashMap<(String, String), EffectSet>,
    /// Current function's declared effects
    declared: EffectSet,
    /// Current function's inferred effects
    inferred: EffectSet,
    /// Current function span (for error reporting)
    current_fn_span: Span,
    /// Errors
    errors: Vec<EffectError>,
}

/// Effect error
#[derive(Debug, Clone)]
pub struct EffectError {
    pub kind: EffectErrorKind,
    pub span: Span,
    pub fn_span: Span,
}

/// Kind of effect error
#[derive(Debug, Clone)]
pub enum EffectErrorKind {
    /// Effect used but not declared in function signature
    UndeclaredEffect { effect: String },
    /// Effect not handled
    UnhandledEffect { effect: String },
    /// Effectful operation in pure context
    EffectInPureContext { effect: String },
}

impl<'a> EffectChecker<'a> {
    pub fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            fn_effects: HashMap::new(),
            method_effects: HashMap::new(),
            declared: EffectSet::new(),
            inferred: EffectSet::new(),
            current_fn_span: Span::dummy(),
            errors: Vec::new(),
        }
    }

    /// Check effects for entire program
    pub fn check_program(&mut self, ast: &Ast) -> Result<(), Vec<EffectError>> {
        // First pass: collect declared effects for all functions and methods
        for item in &ast.items {
            match item {
                Item::Function(f) => {
                    self.collect_function_effects(f);
                }
                Item::Impl(impl_def) => {
                    self.collect_impl_method_effects(impl_def);
                }
                _ => {}
            }
        }

        // Second pass: infer and check effects in function bodies
        for item in &ast.items {
            if let Item::Function(f) = item {
                self.check_function(f);
            }
        }

        // Also check impl method bodies
        for item in &ast.items {
            if let Item::Impl(impl_def) = item {
                for impl_item in &impl_def.items {
                    if let ImplItem::Fn(f) = impl_item {
                        self.check_function(f);
                    }
                }
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn collect_impl_method_effects(&mut self, impl_def: &ast::ImplDef) {
        // Extract type name from target_type
        let type_name = self.extract_type_name(&impl_def.target_type);

        for impl_item in &impl_def.items {
            if let ImplItem::Fn(f) = impl_item {
                let mut effects = EffectSet::new();
                for eff_ref in &f.effects {
                    let effect = self.resolve_effect_ref(eff_ref);
                    effects.add(effect);
                }

                // Store method effects keyed by (type_name, method_name)
                self.method_effects
                    .insert((type_name.clone(), f.name.clone()), effects.clone());

                // Also store by DefId for consistency
                if let Some(def_id) = self.symbols.def_for_node(f.id) {
                    self.fn_effects.insert(def_id, effects);
                }
            }
        }
    }

    fn extract_type_name(&self, ty: &ast::TypeExpr) -> String {
        match ty {
            ast::TypeExpr::Named { path, .. } => {
                // Get the last segment of the path as the type name
                path.segments
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string())
            }
            ast::TypeExpr::Reference { inner, .. } => {
                // For reference types like &T or &!T, extract the inner type name
                self.extract_type_name(inner)
            }
            _ => "Unknown".to_string(),
        }
    }

    fn collect_function_effects(&mut self, f: &ast::FnDef) {
        let mut effects = EffectSet::new();
        for eff_ref in &f.effects {
            let effect = self.resolve_effect_ref(eff_ref);
            effects.add(effect);
        }

        if let Some(def_id) = self.symbols.def_for_node(f.id) {
            self.fn_effects.insert(def_id, effects);
        }
    }

    fn check_function(&mut self, f: &ast::FnDef) {
        // Set declared effects for this function
        self.declared = EffectSet::new();
        for eff_ref in &f.effects {
            let effect = self.resolve_effect_ref(eff_ref);
            self.declared.add(effect);
        }

        // Reset inferred effects
        self.inferred = EffectSet::new();
        self.current_fn_span = f.span;

        // Infer effects from body
        self.infer_block(&f.body);

        // Check that all inferred effects are declared
        for effect_name in &self.inferred.effects.clone() {
            if !self.declared.contains(effect_name) {
                self.errors.push(EffectError {
                    kind: EffectErrorKind::UndeclaredEffect {
                        effect: effect_name.clone(),
                    },
                    span: f.span, // TODO: more precise span
                    fn_span: f.span,
                });
            }
        }
    }

    fn infer_block(&mut self, block: &ast::Block) -> EffectSet {
        let mut block_effects = EffectSet::new();

        for stmt in &block.stmts {
            let stmt_effects = self.infer_stmt(stmt);
            block_effects = block_effects.union(&stmt_effects);
        }

        self.inferred = self.inferred.union(&block_effects);
        block_effects
    }

    fn infer_stmt(&mut self, stmt: &Stmt) -> EffectSet {
        match stmt {
            Stmt::Let { value, .. } => {
                if let Some(init) = value {
                    self.infer_expr(init)
                } else {
                    EffectSet::new()
                }
            }
            Stmt::Expr { expr, .. } => self.infer_expr(expr),
            Stmt::Assign { target, value, .. } => {
                let mut effects = self.infer_expr(target);
                effects = effects.union(&self.infer_expr(value));
                // Assignment implies Mut effect if target is mutable
                effects
            }
            Stmt::Empty | Stmt::MacroInvocation(_) => EffectSet::new(),
        }
    }

    fn infer_expr(&mut self, expr: &Expr) -> EffectSet {
        match expr {
            Expr::Literal { .. } => EffectSet::new(),

            Expr::Path { .. } => EffectSet::new(),

            Expr::Binary {
                op, left, right, ..
            } => {
                let mut effects = self.infer_expr(left);
                effects = effects.union(&self.infer_expr(right));

                // Division and remainder may panic
                if matches!(op, BinaryOp::Div | BinaryOp::Rem) {
                    effects.add(Effect {
                        name: "Panic".to_string(),
                        args: Vec::new(),
                    });
                }

                effects
            }

            Expr::Unary { expr, .. } => self.infer_expr(expr),

            Expr::Call { callee, args, .. } => {
                let mut effects = self.infer_expr(callee);

                // Get callee's declared effects
                let callee_effects = self.get_callee_effects(callee);
                effects = effects.union(&callee_effects);

                // Infer argument effects
                for arg in args {
                    effects = effects.union(&self.infer_expr(arg));
                }

                effects
            }

            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let mut effects = self.infer_expr(receiver);
                for arg in args {
                    effects = effects.union(&self.infer_expr(arg));
                }

                // Look up method effects based on receiver type
                let method_effects = self.get_method_effects(receiver, method);
                effects = effects.union(&method_effects);
                effects
            }

            Expr::Field { base, .. } => self.infer_expr(base),

            Expr::TupleField { base, .. } => self.infer_expr(base),

            Expr::Index { base, index, .. } => {
                let mut effects = self.infer_expr(base);
                effects = effects.union(&self.infer_expr(index));
                // Indexing may panic
                effects.add(Effect {
                    name: "Panic".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Cast { expr, .. } => self.infer_expr(expr),

            Expr::Block { block, .. } => self.infer_block(block),

            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let mut effects = self.infer_expr(condition);
                effects = effects.union(&self.infer_block(then_branch));
                if let Some(else_expr) = else_branch {
                    effects = effects.union(&self.infer_expr(else_expr));
                }
                effects
            }

            Expr::Match {
                scrutinee, arms, ..
            } => {
                let mut effects = self.infer_expr(scrutinee);
                for arm in arms {
                    effects = effects.union(&self.infer_expr(&arm.body));
                    if let Some(guard) = &arm.guard {
                        effects = effects.union(&self.infer_expr(guard));
                    }
                }
                effects
            }

            Expr::Loop { body, .. } => {
                let mut effects = self.infer_block(body);
                // Loops may diverge
                effects.add(Effect {
                    name: "Div".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::While {
                condition, body, ..
            } => {
                let mut effects = self.infer_expr(condition);
                effects = effects.union(&self.infer_block(body));
                // Loops may diverge
                effects.add(Effect {
                    name: "Div".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::For { iter, body, .. } => {
                let mut effects = self.infer_expr(iter);
                effects = effects.union(&self.infer_block(body));
                effects
            }

            Expr::Return { value, .. } => {
                if let Some(val) = value {
                    self.infer_expr(val)
                } else {
                    EffectSet::new()
                }
            }

            Expr::Break { value, .. } => {
                if let Some(val) = value {
                    self.infer_expr(val)
                } else {
                    EffectSet::new()
                }
            }

            Expr::Continue { .. } => EffectSet::new(),

            Expr::Closure { body, .. } => {
                // Closures capture their effects
                self.infer_expr(body)
            }

            Expr::Tuple { elements, .. } => {
                let mut effects = EffectSet::new();
                for elem in elements {
                    effects = effects.union(&self.infer_expr(elem));
                }
                effects
            }

            Expr::Array { elements, .. } => {
                let mut effects = EffectSet::new();
                for elem in elements {
                    effects = effects.union(&self.infer_expr(elem));
                }
                effects
            }

            Expr::Range { start, end, .. } => {
                let mut effects = EffectSet::new();
                if let Some(s) = start {
                    effects = effects.union(&self.infer_expr(s));
                }
                if let Some(e) = end {
                    effects = effects.union(&self.infer_expr(e));
                }
                effects
            }

            Expr::StructLit { fields, .. } => {
                let mut effects = EffectSet::new();
                for (_, expr) in fields {
                    effects = effects.union(&self.infer_expr(expr));
                }
                effects
            }

            Expr::Try { expr, .. } => {
                let mut effects = self.infer_expr(expr);
                // Try can propagate Panic
                effects.add(Effect {
                    name: "Panic".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Perform { effect, args, .. } => {
                let mut effects = EffectSet::new();
                // Add the performed effect
                if let Some(name) = effect.name() {
                    effects.add(Effect {
                        name: name.to_string(),
                        args: Vec::new(),
                    });
                }
                for arg in args {
                    effects = effects.union(&self.infer_expr(arg));
                }
                effects
            }

            Expr::Handle { expr, handler, .. } => {
                let body_effects = self.infer_expr(expr);
                // Handler removes the handled effect
                let handled_name = handler.name().unwrap_or("").to_string();
                let mut result = EffectSet::new();
                for eff in &body_effects.effects {
                    if eff != &handled_name {
                        result.effects.insert(eff.clone());
                    }
                }
                result
            }

            Expr::Sample { distribution, .. } => {
                let mut effects = self.infer_expr(distribution);
                // Sample has Prob effect
                effects.add(Effect {
                    name: "Prob".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Await { expr, .. } => {
                let mut effects = self.infer_expr(expr);
                // Await has Async effect
                effects.add(Effect {
                    name: "Async".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::AsyncBlock { block, .. } => {
                let mut effects = self.infer_block(block);
                // Async block has Async effect
                effects.add(Effect {
                    name: "Async".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::AsyncClosure { body, .. } => {
                let mut effects = self.infer_expr(body);
                // Async closure has Async effect
                effects.add(Effect {
                    name: "Async".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Spawn { expr, .. } => {
                let mut effects = self.infer_expr(expr);
                // Spawn has Async effect
                effects.add(Effect {
                    name: "Async".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Select { arms, .. } => {
                let mut effects = EffectSet::new();
                // Select has Async effect
                effects.add(Effect {
                    name: "Async".to_string(),
                    args: Vec::new(),
                });
                for arm in arms {
                    effects = effects.union(&self.infer_expr(&arm.future));
                    effects = effects.union(&self.infer_expr(&arm.body));
                    if let Some(guard) = &arm.guard {
                        effects = effects.union(&self.infer_expr(guard));
                    }
                }
                effects
            }

            Expr::Join { futures, .. } => {
                let mut effects = EffectSet::new();
                // Join has Async effect
                effects.add(Effect {
                    name: "Async".to_string(),
                    args: Vec::new(),
                });
                for future in futures {
                    effects = effects.union(&self.infer_expr(future));
                }
                effects
            }

            Expr::MacroInvocation(_) => EffectSet::new(),

            // Epistemic expressions
            Expr::Do { interventions, .. } => {
                let mut effects = EffectSet::new();
                // Causal interventions have a Causal effect
                effects.add(Effect {
                    name: "Causal".to_string(),
                    args: Vec::new(),
                });
                for (_, value) in interventions {
                    effects = effects.union(&self.infer_expr(value));
                }
                effects
            }

            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => {
                let mut effects = EffectSet::new();
                // Counterfactuals have a Causal effect
                effects.add(Effect {
                    name: "Causal".to_string(),
                    args: Vec::new(),
                });
                effects = effects.union(&self.infer_expr(factual));
                effects = effects.union(&self.infer_expr(intervention));
                effects = effects.union(&self.infer_expr(outcome));
                effects
            }

            Expr::KnowledgeExpr {
                value,
                epsilon,
                validity,
                provenance,
                ..
            } => {
                let mut effects = self.infer_expr(value);
                if let Some(e) = epsilon {
                    effects = effects.union(&self.infer_expr(e));
                }
                if let Some(v) = validity {
                    effects = effects.union(&self.infer_expr(v));
                }
                if let Some(p) = provenance {
                    effects = effects.union(&self.infer_expr(p));
                }
                effects
            }

            Expr::Uncertain {
                value, uncertainty, ..
            } => {
                let mut effects = self.infer_expr(value);
                effects = effects.union(&self.infer_expr(uncertainty));
                effects
            }

            Expr::GpuAnnotated { expr, .. } => {
                let mut effects = self.infer_expr(expr);
                // GPU annotated expressions have GPU effect
                effects.add(Effect {
                    name: "GPU".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Observe {
                data, distribution, ..
            } => {
                let mut effects = self.infer_expr(data);
                effects = effects.union(&self.infer_expr(distribution));
                // Observe has Prob effect
                effects.add(Effect {
                    name: "Prob".to_string(),
                    args: Vec::new(),
                });
                effects
            }

            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                let mut effects = self.infer_expr(target);
                // Query has Prob effect
                effects.add(Effect {
                    name: "Prob".to_string(),
                    args: Vec::new(),
                });
                for g in given {
                    effects = effects.union(&self.infer_expr(g));
                }
                for (_, value) in interventions {
                    effects = effects.union(&self.infer_expr(value));
                }
                // If there are interventions, also add Causal effect
                if !interventions.is_empty() {
                    effects.add(Effect {
                        name: "Causal".to_string(),
                        args: Vec::new(),
                    });
                }
                effects
            }

            // Ontology term literals have no effects
            Expr::OntologyTerm { .. } => EffectSet::new(),
        }
    }

    fn get_callee_effects(&self, callee: &Expr) -> EffectSet {
        if let Expr::Path { path, id } = callee
            && path.is_simple()
        {
            // Look up the function by NodeId reference
            if let Some(def_id) = self.symbols.ref_for_node(*id)
                && let Some(effects) = self.fn_effects.get(&def_id)
            {
                return effects.clone();
            }
        }
        EffectSet::new()
    }

    fn get_method_effects(&self, receiver: &Expr, method_name: &str) -> EffectSet {
        // Try to infer the receiver's type name
        let type_name = self.infer_receiver_type_name(receiver);

        if let Some(name) = type_name {
            // Look up method effects by (type_name, method_name)
            if let Some(effects) = self
                .method_effects
                .get(&(name.clone(), method_name.to_string()))
            {
                return effects.clone();
            }
        }

        // Fallback: no effects known for this method
        EffectSet::new()
    }

    fn infer_receiver_type_name(&self, receiver: &Expr) -> Option<String> {
        match receiver {
            // Simple path like `foo.method()` - we need type info from the checker
            Expr::Path { path, id } => {
                // Try to look up the variable's type from symbol table
                if path.is_simple()
                    && let Some(def_id) = self.symbols.ref_for_node(*id)
                    && let Some(symbol) = self.symbols.get(def_id)
                {
                    // Return the symbol's type name if available
                    // For now, return the symbol name as a heuristic
                    return Some(symbol.name.clone());
                }
                None
            }
            // Struct literal: `MyStruct { ... }.method()`
            Expr::StructLit { path, .. } => path.segments.last().cloned(),
            // Method chain: `x.foo().bar()` - would need type inference
            Expr::MethodCall { .. } => None,
            // Function call result: `get_foo().method()`
            Expr::Call { .. } => None,
            // Field access: `x.field.method()`
            Expr::Field { .. } => None,
            _ => None,
        }
    }

    fn resolve_effect_ref(&self, eff_ref: &ast::EffectRef) -> Effect {
        let name = eff_ref.name.name().unwrap_or("Unknown").to_string();
        Effect {
            name,
            args: Vec::new(),
        }
    }

    /// Get the inferred effects for a function
    pub fn get_function_effects(&self, def_id: DefId) -> Option<&EffectSet> {
        self.fn_effects.get(&def_id)
    }
}

impl std::fmt::Display for EffectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            EffectErrorKind::UndeclaredEffect { effect } => {
                write!(f, "Effect `{}` not declared in function signature", effect)
            }
            EffectErrorKind::UnhandledEffect { effect } => {
                write!(f, "Unhandled effect `{}`", effect)
            }
            EffectErrorKind::EffectInPureContext { effect } => {
                write!(f, "Cannot perform `{}` in pure context", effect)
            }
        }
    }
}
