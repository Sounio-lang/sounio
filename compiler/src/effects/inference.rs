//! Effect inference pass
//!
//! Infers effects for expressions and checks that all effects are declared
//! in function signatures.
//!
//! # Type-Informed Effect Inference
//!
//! This module supports type-aware effect inference when provided with resolved
//! type information from the type checker. This enables:
//!
//! - Method chain effect tracking (e.g., `x.foo().bar()`)
//! - Higher-order function effect propagation
//! - Closure effect inference at call site
//! - Pattern match effect inference for refutable patterns

use crate::ast::{self, Ast, BinaryOp, Expr, ImplItem, Item, Pattern, Stmt};
use crate::common::{NodeId, Span};
use crate::resolve::{DefId, SymbolTable};
use crate::types::core::{Effect, EffectSet, Type};
use std::collections::HashMap;

/// Type information for expressions, provided by the type checker.
///
/// This enables type-aware effect inference, particularly for:
/// - Method chains where receiver types need to be resolved
/// - Higher-order functions where function types carry effect information
/// - Closures that capture effects from their environment
#[derive(Debug, Clone, Default)]
pub struct TypeInfo {
    /// Map from AST NodeId to resolved Type
    pub expr_types: HashMap<NodeId, Type>,
    /// Map from variable name to Type (for local bindings)
    pub binding_types: HashMap<String, Type>,
    /// Method return types: (receiver_type, method_name) -> return_type
    pub method_return_types: HashMap<(String, String), Type>,
}

impl TypeInfo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the type of an expression by its NodeId
    pub fn get_expr_type(&self, id: NodeId) -> Option<&Type> {
        self.expr_types.get(&id)
    }

    /// Get the type of a binding by name
    pub fn get_binding_type(&self, name: &str) -> Option<&Type> {
        self.binding_types.get(name)
    }

    /// Get the return type of a method call
    pub fn get_method_return_type(&self, receiver_type: &str, method: &str) -> Option<&Type> {
        self.method_return_types
            .get(&(receiver_type.to_string(), method.to_string()))
    }

    /// Register a type for an expression
    pub fn set_expr_type(&mut self, id: NodeId, ty: Type) {
        self.expr_types.insert(id, ty);
    }

    /// Register a binding type
    pub fn set_binding_type(&mut self, name: String, ty: Type) {
        self.binding_types.insert(name, ty);
    }

    /// Register a method return type
    pub fn set_method_return_type(&mut self, receiver_type: String, method: String, ty: Type) {
        self.method_return_types.insert((receiver_type, method), ty);
    }

    /// Extract type name from a Type
    pub fn type_name(ty: &Type) -> Option<String> {
        match ty {
            Type::Named { name, .. } => Some(name.clone()),
            Type::Ref { inner, .. } => Self::type_name(inner),
            Type::Array { element, .. } => {
                Self::type_name(element).map(|n| format!("[{}]", n))
            }
            Type::String => Some("String".to_string()),
            Type::Str => Some("str".to_string()),
            Type::I32 => Some("i32".to_string()),
            Type::I64 => Some("i64".to_string()),
            Type::F32 => Some("f32".to_string()),
            Type::F64 => Some("f64".to_string()),
            Type::Bool => Some("bool".to_string()),
            Type::Vec2 => Some("vec2".to_string()),
            Type::Vec3 => Some("vec3".to_string()),
            Type::Vec4 => Some("vec4".to_string()),
            Type::Mat2 => Some("mat2".to_string()),
            Type::Mat3 => Some("mat3".to_string()),
            Type::Mat4 => Some("mat4".to_string()),
            _ => None,
        }
    }
}

/// Effect inference context
pub struct EffectChecker<'a> {
    symbols: &'a SymbolTable,
    /// Optional type information from the type checker
    type_info: Option<&'a TypeInfo>,
    /// Inferred effects per function DefId
    fn_effects: HashMap<DefId, EffectSet>,
    /// Method effects: (type_name, method_name) -> EffectSet
    method_effects: HashMap<(String, String), EffectSet>,
    /// Higher-order function effects: function_type_id -> EffectSet
    /// Used for tracking effects through closures and function parameters
    hof_effects: HashMap<String, EffectSet>,
    /// Current function's declared effects
    declared: EffectSet,
    /// Current function's inferred effects
    inferred: EffectSet,
    /// Current function span (for error reporting)
    current_fn_span: Span,
    /// Current function name (for error messages)
    current_fn_name: Option<String>,
    /// Errors
    errors: Vec<EffectError>,
    /// Effect source tracking for better error messages
    /// Maps effect name to the expression span where it was first inferred
    effect_sources: HashMap<String, Span>,
}

/// Effect error with detailed source information
#[derive(Debug, Clone)]
pub struct EffectError {
    pub kind: EffectErrorKind,
    /// Span where the effect was used/required
    pub span: Span,
    /// Span of the containing function
    pub fn_span: Span,
    /// Name of the containing function (for error messages)
    pub fn_name: Option<String>,
    /// Span where the effect originated (for chains)
    pub source_span: Option<Span>,
    /// Additional context for the error
    pub context: Option<String>,
}

/// Kind of effect error
#[derive(Debug, Clone)]
pub enum EffectErrorKind {
    /// Effect used but not declared in function signature
    UndeclaredEffect {
        effect: String,
        /// Where the effect requirement came from
        source: EffectSource,
    },
    /// Effect not handled
    UnhandledEffect {
        effect: String,
    },
    /// Effectful operation in pure context
    EffectInPureContext {
        effect: String,
    },
    /// Higher-order function called with effectful closure
    EffectfulClosureArg {
        effect: String,
        /// The HOF being called (e.g., "map", "filter")
        hof_name: String,
    },
    /// Refutable pattern may panic
    RefutablePatternPanic {
        pattern_desc: String,
    },
}

/// Source of an effect requirement
#[derive(Debug, Clone)]
pub enum EffectSource {
    /// Direct operation (e.g., division, indexing)
    DirectOperation(String),
    /// Method call on a type
    MethodCall { receiver_type: String, method: String },
    /// Function call
    FunctionCall(String),
    /// Closure invocation
    ClosureCall,
    /// Higher-order function application
    HigherOrderFunction { hof: String, closure_effect: String },
    /// Pattern matching
    PatternMatch,
    /// Unknown source
    Unknown,
}

impl<'a> EffectChecker<'a> {
    /// Create a new effect checker without type information.
    ///
    /// This mode has limited capability for method chain inference and
    /// higher-order function tracking.
    pub fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            type_info: None,
            fn_effects: HashMap::new(),
            method_effects: HashMap::new(),
            hof_effects: HashMap::new(),
            declared: EffectSet::new(),
            inferred: EffectSet::new(),
            current_fn_span: Span::dummy(),
            current_fn_name: None,
            errors: Vec::new(),
            effect_sources: HashMap::new(),
        }
    }

    /// Create a new effect checker with type information.
    ///
    /// This enables full effect inference including:
    /// - Method chain tracking
    /// - Higher-order function effect propagation
    /// - Precise receiver type resolution
    pub fn with_type_info(symbols: &'a SymbolTable, type_info: &'a TypeInfo) -> Self {
        Self {
            symbols,
            type_info: Some(type_info),
            fn_effects: HashMap::new(),
            method_effects: HashMap::new(),
            hof_effects: HashMap::new(),
            declared: EffectSet::new(),
            inferred: EffectSet::new(),
            current_fn_span: Span::dummy(),
            current_fn_name: None,
            errors: Vec::new(),
            effect_sources: HashMap::new(),
        }
    }

    /// Determine which effect a handler handles based on its name.
    ///
    /// Uses the following logic:
    /// 1. If handler name ends with "Handler", strip suffix and use as effect name
    ///    (e.g., "IOHandler" -> "IO", "MutHandler" -> "Mut")
    /// 2. Otherwise, use the handler name as-is
    ///
    /// This aligns with the convention used throughout the Sounio effect system where
    /// handler names typically follow the pattern `{EffectName}Handler`.
    fn determine_handled_effect(&self, handler_name: &str) -> Option<String> {
        if handler_name.is_empty() {
            return None;
        }

        // Check for XHandler pattern
        if handler_name.ends_with("Handler") && handler_name.len() > 7 {
            Some(handler_name[..handler_name.len() - 7].to_string())
        } else {
            // Use handler name as-is (for custom handlers or alternative naming)
            Some(handler_name.to_string())
        }
    }

    /// Record an effect with its source location for better error messages
    fn record_effect(&mut self, effect: Effect, span: Span) {
        if !self.effect_sources.contains_key(&effect.name) {
            self.effect_sources.insert(effect.name.clone(), span);
        }
        self.inferred.add(effect);
    }

    /// Record an effect set with source tracking
    fn record_effects(&mut self, effects: &EffectSet, span: Span) {
        for effect_name in &effects.effects {
            if !self.effect_sources.contains_key(effect_name) {
                self.effect_sources.insert(effect_name.clone(), span);
            }
        }
        self.inferred = self.inferred.union(effects);
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

        // Reset inferred effects and source tracking
        self.inferred = EffectSet::new();
        self.effect_sources.clear();
        self.current_fn_span = f.span;
        self.current_fn_name = Some(f.name.clone());

        // Register parameter types for higher-order function tracking
        self.register_function_params(f);

        // Infer effects from body
        self.infer_block(&f.body);

        // Check that all inferred effects are declared
        for effect_name in &self.inferred.effects.clone() {
            if !self.declared.contains(effect_name) {
                // Get the source span for this effect if available
                let source_span = self.effect_sources.get(effect_name).copied();
                let effect_span = source_span.unwrap_or(f.span);

                self.errors.push(EffectError {
                    kind: EffectErrorKind::UndeclaredEffect {
                        effect: effect_name.clone(),
                        source: EffectSource::Unknown,
                    },
                    span: effect_span,
                    fn_span: f.span,
                    fn_name: Some(f.name.clone()),
                    source_span,
                    context: Some(format!(
                        "Function `{}` uses effect `{}` but does not declare it. \
                         Add `with {}` to the function signature.",
                        f.name, effect_name, effect_name
                    )),
                });
            }
        }
    }

    /// Register function parameter types for higher-order function effect tracking
    fn register_function_params(&mut self, f: &ast::FnDef) {
        for param in &f.params {
            // Check if parameter is a function type
            if let ast::TypeExpr::Function { effects, .. } = &param.ty {
                // Extract parameter name
                let param_name = self.pattern_to_name(&param.pattern);
                if let Some(name) = param_name {
                    // Collect effects declared on this function parameter
                    let mut param_effects = EffectSet::new();
                    for eff_ref in effects {
                        let effect = self.resolve_effect_ref(eff_ref);
                        param_effects.add(effect);
                    }
                    // Store for HOF tracking
                    self.hof_effects.insert(name, param_effects);
                }
            }
        }
    }

    /// Extract name from a pattern (for parameter binding)
    fn pattern_to_name(&self, pattern: &Pattern) -> Option<String> {
        match pattern {
            Pattern::Binding { name, .. } => Some(name.clone()),
            _ => None,
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
            Stmt::Let { pattern, value, .. } => {
                let mut effects = EffectSet::new();

                // Check for refutable patterns that may panic
                if self.is_refutable_pattern(pattern) {
                    effects.add(Effect {
                        name: "Panic".to_string(),
                        args: Vec::new(),
                    });
                }

                if let Some(init) = value {
                    effects = effects.union(&self.infer_expr(init));
                }
                effects
            }
            Stmt::Expr { expr, .. } => self.infer_expr(expr),
            Stmt::Assign { target, value, .. } => {
                let mut effects = self.infer_expr(target);
                effects = effects.union(&self.infer_expr(value));
                // Assignment implies Mut effect if target is mutable
                effects
            }
            Stmt::Empty | Stmt::MacroInvocation(_) | Stmt::LocalExtern(_) => EffectSet::new(),
        }
    }

    /// Check if a pattern is refutable (may fail to match)
    fn is_refutable_pattern(&self, pattern: &Pattern) -> bool {
        match pattern {
            // Irrefutable patterns
            Pattern::Binding { .. } => false,
            Pattern::Wildcard => false,

            // Literal patterns are refutable - can fail to match
            Pattern::Literal(_) => true,

            // Enum patterns are refutable - may not match the variant
            Pattern::Enum { .. } => true,

            // Tuple patterns: refutable if any element is refutable
            Pattern::Tuple(elements) => {
                elements.iter().any(|p| self.is_refutable_pattern(p))
            }

            // Struct patterns: refutable if any field is refutable
            Pattern::Struct { fields, .. } => {
                fields.iter().any(|(_, p)| self.is_refutable_pattern(p))
            }

            // Or patterns: refutable only if all alternatives are refutable
            Pattern::Or(patterns) => {
                patterns.iter().all(|p| self.is_refutable_pattern(p))
            }
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

            Expr::Call { id, callee, args, .. } => {
                let mut effects = self.infer_expr(callee);

                // Get callee's declared effects
                let callee_effects = self.get_callee_effects(callee);
                effects = effects.union(&callee_effects);

                // Check for higher-order function patterns
                let callee_name = self.get_callee_name(callee);

                // Infer argument effects
                for (i, arg) in args.iter().enumerate() {
                    let arg_effects = self.infer_expr(arg);
                    effects = effects.union(&arg_effects);

                    // Track closure argument effects for HOF patterns
                    if let Expr::Closure { .. } = arg {
                        if !arg_effects.is_pure() {
                            // When a closure is passed to a HOF, the enclosing function
                            // must declare the closure's effects
                            for effect_name in &arg_effects.effects {
                                if !self.effect_sources.contains_key(effect_name) {
                                    self.effect_sources.insert(
                                        effect_name.clone(),
                                        self.expr_span_from_id(*id),
                                    );
                                }
                            }
                        }
                    }
                }

                // Check if calling a function parameter (HOF invocation)
                if let Some(name) = &callee_name {
                    if let Some(param_effects) = self.hof_effects.get(name).cloned() {
                        // Add the declared effects from the function parameter's type
                        effects = effects.union(&param_effects);
                    }
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
                // Handler removes the handled effect(s)
                let handler_name = handler.name().unwrap_or("").to_string();
                let handled_effect = self.determine_handled_effect(&handler_name);

                let mut result = EffectSet::new();
                for eff in &body_effects.effects {
                    // Remove the effect that this handler handles
                    if !handled_effect.as_ref().map_or(false, |he| eff == he) {
                        result.effects.insert(eff.clone());
                    }
                }
                result
            }

            Expr::Resume { value, .. } => {
                // Resume continues a continuation with a value
                // The effects of resume are the effects of the value expression
                self.infer_expr(value)
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

            Expr::UnsafeBlock { block, .. } => {
                // Unsafe blocks can have any effects - we just infer from the block
                self.infer_block(block)
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

    /// Get the name of a callee expression (for function calls)
    fn get_callee_name(&self, callee: &Expr) -> Option<String> {
        match callee {
            Expr::Path { path, .. } if path.is_simple() => {
                Some(path.segments[0].clone())
            }
            _ => None,
        }
    }

    /// Get a span from a NodeId, falling back to the current function span
    fn expr_span_from_id(&self, id: NodeId) -> Span {
        // In the future, we could look up the actual span from the AST
        // For now, use the current function span as a fallback
        self.current_fn_span
    }

    /// Infer effects from a closure body.
    ///
    /// Note: The current AST Closure variant doesn't have declared effects,
    /// so we simply infer from the body.
    fn infer_closure_body_effects(&mut self, body: &Expr) -> EffectSet {
        self.infer_expr(body)
    }

    fn get_callee_effects(&self, callee: &Expr) -> EffectSet {
        match callee {
            Expr::Path { path, id } if path.is_simple() => {
                // Look up the function by NodeId reference
                if let Some(def_id) = self.symbols.ref_for_node(*id) {
                    if let Some(effects) = self.fn_effects.get(&def_id) {
                        return effects.clone();
                    }
                }

                // Check if it's a HOF parameter
                let name = &path.segments[0];
                if let Some(effects) = self.hof_effects.get(name) {
                    return effects.clone();
                }

                // Check type info for function type with effects
                if let Some(type_info) = self.type_info {
                    if let Some(ty) = type_info.get_binding_type(name) {
                        if let Type::Function { effects, .. } = ty {
                            return effects.clone();
                        }
                    }
                }

                EffectSet::new()
            }
            // Note: Inline closure calls are rare but we handle the body inference
            // through the argument effect tracking in Call handling
            _ => EffectSet::new(),
        }
    }

    fn get_method_effects(&self, receiver: &Expr, method_name: &str) -> EffectSet {
        // Try to infer the receiver's type name using type information if available
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

    /// Infer the type name of a receiver expression.
    ///
    /// This uses type information from the type checker when available,
    /// enabling method chain effect inference.
    fn infer_receiver_type_name(&self, receiver: &Expr) -> Option<String> {
        // First, try to use type information if available
        if let Some(type_info) = self.type_info {
            let node_id = self.expr_node_id(receiver);
            if let Some(ty) = type_info.get_expr_type(node_id) {
                return TypeInfo::type_name(ty);
            }
        }

        // Fallback to AST-based inference
        match receiver {
            // Simple path like `foo.method()`
            Expr::Path { path, id } => {
                // First try type info for the binding
                if let Some(type_info) = self.type_info {
                    if path.is_simple() {
                        let name = &path.segments[0];
                        if let Some(ty) = type_info.get_binding_type(name) {
                            return TypeInfo::type_name(ty);
                        }
                    }
                }

                // Fall back to symbol table lookup
                if path.is_simple()
                    && let Some(def_id) = self.symbols.ref_for_node(*id)
                    && let Some(symbol) = self.symbols.get(def_id)
                {
                    return Some(symbol.name.clone());
                }
                None
            }

            // Struct literal: `MyStruct { ... }.method()`
            Expr::StructLit { path, .. } => path.segments.last().cloned(),

            // Method chain: `x.foo().bar()` - use type info for the method call result
            Expr::MethodCall { receiver: inner_receiver, method, id, .. } => {
                if let Some(type_info) = self.type_info {
                    // Try to get type from this method call's node
                    if let Some(ty) = type_info.get_expr_type(*id) {
                        return TypeInfo::type_name(ty);
                    }

                    // Otherwise, try to look up the return type based on receiver type
                    if let Some(receiver_type) = self.infer_receiver_type_name(inner_receiver) {
                        if let Some(return_ty) = type_info.get_method_return_type(&receiver_type, method) {
                            return TypeInfo::type_name(return_ty);
                        }
                    }
                }
                None
            }

            // Function call result: `get_foo().method()`
            Expr::Call { id, callee, .. } => {
                if let Some(type_info) = self.type_info {
                    // Try to get the return type from the call expression
                    if let Some(ty) = type_info.get_expr_type(*id) {
                        return TypeInfo::type_name(ty);
                    }

                    // Try to get function return type
                    if let Expr::Path { path, .. } = callee.as_ref() {
                        if path.is_simple() {
                            let fn_name = &path.segments[0];
                            if let Some(ty) = type_info.get_binding_type(fn_name) {
                                // Extract return type from function type
                                if let Type::Function { return_type, .. } = ty {
                                    return TypeInfo::type_name(return_type);
                                }
                            }
                        }
                    }
                }
                None
            }

            // Field access: `x.field.method()`
            Expr::Field { id, .. } => {
                if let Some(type_info) = self.type_info {
                    if let Some(ty) = type_info.get_expr_type(*id) {
                        return TypeInfo::type_name(ty);
                    }
                }
                None
            }

            // Index expression: `arr[i].method()`
            Expr::Index { id, .. } => {
                if let Some(type_info) = self.type_info {
                    if let Some(ty) = type_info.get_expr_type(*id) {
                        return TypeInfo::type_name(ty);
                    }
                }
                None
            }

            _ => None,
        }
    }

    /// Extract the NodeId from an expression
    fn expr_node_id(&self, expr: &Expr) -> NodeId {
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
            Expr::Continue { id } => *id,
            Expr::Closure { id, .. } => *id,
            Expr::Tuple { id, .. } => *id,
            Expr::Array { id, .. } => *id,
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
            Expr::OntologyTerm { id, .. } => *id,
            Expr::Do { id, .. } => *id,
            Expr::Counterfactual { id, .. } => *id,
            Expr::KnowledgeExpr { id, .. } => *id,
            Expr::Uncertain { id, .. } => *id,
            Expr::GpuAnnotated { id, .. } => *id,
            Expr::Observe { id, .. } => *id,
            Expr::Query { id, .. } => *id,
            Expr::MacroInvocation(_) => NodeId::dummy(),
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
            EffectErrorKind::UndeclaredEffect { effect, source } => {
                write!(f, "effect `{}` not declared in function signature", effect)?;
                match source {
                    EffectSource::DirectOperation(op) => {
                        write!(f, " (from {} operation)", op)?;
                    }
                    EffectSource::MethodCall { receiver_type, method } => {
                        write!(f, " (from {}.{}())", receiver_type, method)?;
                    }
                    EffectSource::FunctionCall(name) => {
                        write!(f, " (from call to {})", name)?;
                    }
                    EffectSource::ClosureCall => {
                        write!(f, " (from closure invocation)")?;
                    }
                    EffectSource::HigherOrderFunction { hof, closure_effect } => {
                        write!(f, " (from {} with effectful closure having {})", hof, closure_effect)?;
                    }
                    EffectSource::PatternMatch => {
                        write!(f, " (from refutable pattern)")?;
                    }
                    EffectSource::Unknown => {}
                }
                if let Some(fn_name) = &self.fn_name {
                    write!(f, "\n  help: add `with {}` to function `{}`", effect, fn_name)?;
                }
                Ok(())
            }
            EffectErrorKind::UnhandledEffect { effect } => {
                write!(f, "unhandled effect `{}`", effect)?;
                write!(f, "\n  help: wrap in a handler or propagate with `with {}`", effect)
            }
            EffectErrorKind::EffectInPureContext { effect } => {
                write!(f, "cannot perform `{}` in pure context", effect)?;
                write!(f, "\n  help: this context requires pure computations")
            }
            EffectErrorKind::EffectfulClosureArg { effect, hof_name } => {
                write!(f, "closure passed to `{}` has effect `{}`", hof_name, effect)?;
                write!(f, "\n  help: the enclosing function must declare `with {}`", effect)
            }
            EffectErrorKind::RefutablePatternPanic { pattern_desc } => {
                write!(f, "refutable pattern `{}` may panic", pattern_desc)?;
                write!(f, "\n  help: add `with Panic` or use match/if-let for safe handling")
            }
        }
    }
}

impl std::fmt::Display for EffectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectSource::DirectOperation(op) => write!(f, "{} operation", op),
            EffectSource::MethodCall { receiver_type, method } => {
                write!(f, "{}.{}()", receiver_type, method)
            }
            EffectSource::FunctionCall(name) => write!(f, "call to {}", name),
            EffectSource::ClosureCall => write!(f, "closure invocation"),
            EffectSource::HigherOrderFunction { hof, closure_effect } => {
                write!(f, "{} with {} closure", hof, closure_effect)
            }
            EffectSource::PatternMatch => write!(f, "pattern match"),
            EffectSource::Unknown => write!(f, "unknown"),
        }
    }
}
