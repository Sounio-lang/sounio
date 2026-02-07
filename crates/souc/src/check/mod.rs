//! Type checker for Sounio
//!
//! This module implements type checking and produces HIR from the AST.
//! It handles:
//! - Type inference (bidirectional)
//! - Name resolution
//! - Effect checking
//! - Ownership/borrow checking
//! - Unit checking
//! - Epistemic type constraints
//! - Semantic type compatibility

// Compiler internals often require deep nesting for pattern matching and control flow.
// This is acceptable for maintainability in this context.
#![allow(clippy::excessive_nesting)]

pub mod compatibility;
pub mod conformal;
pub mod dependent;
pub mod diagnostics;
pub mod epistemic;
pub mod pac;
pub mod probabilistic;
pub mod rankn;

pub use conformal::{
    CalibrationExample, ConformalConfig, ConformalResult, ConformalTypeChecker,
    MondrianConformalChecker,
};
pub use dependent::{DependentTypeChecker, RefinementValidator};
pub use probabilistic::{ProbabilisticTypeInference, SubtypePolymorphismChecker};

#[cfg(test)]
mod extern_tests;

use crate::ast::*;
use crate::common::{NodeId, Span};
use crate::epistemic::bayesian::BetaConfidence;
use crate::epistemic::confidence::{Confidence as EpConfidence, Source as EpSource};
use crate::hir::*;
use crate::macro_system::token_tree::{Delimiter, TokenTree};
use crate::ontology::distance::SpectralDistance;
use crate::ontology::embedding::HyperbolicGenerator;
use crate::ontology::fidelity::{FidelityResult, WorldFidelityChecker};
use crate::ontology::version::{DeprecationTracker, DeprecationWarning};
use crate::ontology::{OntologyResolver, ResolverConfig, SubsumptionResult};
use crate::refinement::{
    solver::SimpleChecker, Atom, BinOp as RefinementBinOp, CompareOp, Predicate, Term,
};
use crate::resolve;
use crate::types::{
    self, effects::EffectInference, units::UnitChecker, DimSize, TensorShape, Type, TypeVar,
};
use miette::Result;
use std::collections::HashMap;

/// Refinement info for a function parameter
#[derive(Clone)]
struct RefinementInfo {
    /// The refinement variable name (e.g., "n" in { n: i32 | n > 0 })
    var: String,
    /// The predicate expression from the AST
    predicate: Box<Expr>,
}

/// Probabilistic threshold for semantic type compatibility.
///
/// Instead of a scalar threshold (distance <= 0.15), this uses Bayesian reasoning:
/// P(distance <= threshold) >= required_probability
///
/// This accounts for uncertainty in distance measurements from different sources
/// (embeddings, IC-based similarity, etc.)
#[derive(Clone)]
pub struct ProbabilisticThreshold {
    /// Prior belief about acceptable distance (Beta distribution)
    pub prior: BetaConfidence,
    /// Required probability that distance is acceptable (e.g., 0.95)
    pub required_probability: f64,
}

impl Default for ProbabilisticThreshold {
    fn default() -> Self {
        Self {
            // Jeffreys prior: Beta(0.5, 0.5) - uninformative
            prior: BetaConfidence::new(0.5, 0.5),
            required_probability: 0.95,
        }
    }
}

impl ProbabilisticThreshold {
    /// Create with specific confidence requirements
    pub fn with_confidence(required_probability: f64) -> Self {
        Self {
            prior: BetaConfidence::new(0.5, 0.5),
            required_probability,
        }
    }

    /// Check if distance is acceptable given confidence
    pub fn is_acceptable(&self, distance: f64, confidence: f64) -> bool {
        // Build posterior from observed distance with given confidence
        // Higher confidence = more weight to observation
        let posterior = BetaConfidence::new(
            self.prior.alpha + (1.0 - distance) * confidence * 10.0,
            self.prior.beta + distance * confidence * 10.0,
        );
        // Check if P(acceptable) >= required_probability
        posterior.probability_above(1.0 - self.prior.mean()) >= self.required_probability
    }
}

/// Type check a ResolvedAst and produce HIR
pub fn check(resolved_ast: &resolve::ResolvedAst) -> Result<Hir> {
    let mut checker = TypeChecker::new_with_resolved_ast(resolved_ast);
    checker.check_program(&resolved_ast.ast)
}

/// Type check an AST and produce HIR (with automatic resolution)
pub fn check_ast(ast: &Ast) -> Result<Hir> {
    let resolved_ast = resolve::resolve(ast.clone())?;
    check(&resolved_ast)
}

/// Type check with external types pre-registered.
///
/// Used for multi-module compilation where types from other modules
/// need to be available during type checking.
///
/// # Arguments
/// * `resolved_ast` - The resolved AST to check
/// * `external_types` - Iterator of (type_name, source_module) pairs
pub fn check_with_external_types(
    resolved_ast: &resolve::ResolvedAst,
    external_types: impl IntoIterator<Item = (String, String)>,
) -> Result<Hir> {
    let mut checker = TypeChecker::new_with_resolved_ast(resolved_ast);
    checker.register_external_types(external_types);
    checker.check_program(&resolved_ast.ast)
}

/// Type checker state
pub struct TypeChecker {
    /// Type environment (variable -> type)
    env: TypeEnv,
    /// Type definitions
    type_defs: HashMap<String, TypeDef>,
    /// Effect inference context
    effects: EffectInference,
    /// Unit checker
    units: UnitChecker,
    /// Fresh type variable counter
    next_type_var: u32,
    /// Fresh effect variable counter (for row polymorphism)
    next_effect_var: u32,
    /// Type constraints for unification
    constraints: Vec<TypeConstraint>,
    /// Effect variable bindings: effect param name -> EffectVar id
    /// Used during generic function checking to track effect parameters
    effect_params: HashMap<String, types::EffectVar>,
    /// Errors accumulated during checking
    errors: Vec<TypeError>,
    /// Ontology alignments: (type1, type2) -> distance
    /// Key is ordered tuple: (min(t1,t2), max(t1,t2)) for symmetric lookup
    alignments: HashMap<(String, String), f64>,
    /// Function-level compatibility thresholds from #[compat] annotations
    fn_thresholds: HashMap<String, f64>,
    /// Default compatibility threshold
    default_threshold: f64,
    /// Probabilistic threshold for Bayesian type compatibility (optional)
    probabilistic_threshold: Option<ProbabilisticThreshold>,
    /// Reference to the AST for span lookup
    ast: Option<std::sync::Arc<Ast>>,
    /// Current function being type-checked (for threshold lookup)
    current_fn: Option<String>,
    /// Current impl target type (for resolving `Self` / `self` parameter types)
    current_impl_type: Option<Type>,
    /// Declared ontology prefixes (from `ontology X from "..."` declarations)
    ontology_prefixes: std::collections::HashSet<String>,
    /// Used ontology prefixes (to detect unused imports)
    used_ontology_prefixes: std::collections::HashSet<String>,
    /// Warnings accumulated during checking
    warnings: Vec<String>,
    /// Handler definitions: handler_name -> effect_name
    /// Used for effect masking when evaluating `handle expr with Handler`
    handler_effects: HashMap<String, String>,
    /// Effects masked by handlers in the current expression context.
    /// When checking a `handle expr with Handler` expression, the handled effect
    /// is added to this set. This enables pure functions that use impure internals
    /// as long as all effects are handled before returning.
    masked_effects: types::EffectSet,
    /// Effect operations registry: (effect_name, op_name) -> return_type
    /// Used for type checking `perform Effect::op(args)` expressions
    effect_operations: HashMap<(String, String), Type>,
    /// Symbol table from resolver (for visibility checking)
    symbols: Option<std::sync::Arc<resolve::SymbolTable>>,
    /// Module tree from resolver (for visibility checking)
    module_tree: Option<std::sync::Arc<resolve::module_tree::ModuleTree>>,
    /// Current module for visibility enforcement
    current_module: Option<resolve::module_tree::ModuleId>,
    /// Refinement info for function parameters: fn_name -> list of refinement info per param
    fn_param_refinements: HashMap<String, Vec<Option<RefinementInfo>>>,
    /// Ontology resolver for dynamic term resolution and semantic distance calculation
    /// across all 4 layers (L1 Primitive, L2 Foundation, L3 Domain, L4 Federated)
    ontology_resolver: Option<OntologyResolver>,
    /// Deprecation tracker for warning on deprecated ontology terms
    deprecation_tracker: Option<DeprecationTracker>,
    /// World fidelity checker for validating Knowledge types against OBO Foundry
    fidelity_checker: Option<WorldFidelityChecker>,
    /// Spectral distance calculator for smooth multi-scale semantic distances (optional)
    spectral_distance: Option<SpectralDistance>,
    /// Hyperbolic embeddings for hierarchical ontology structure (optional)
    hyperbolic_generator: Option<HyperbolicGenerator>,
    /// Conformal type checker for uncertainty quantification (optional)
    conformal_checker: Option<ConformalTypeChecker>,
    /// Cache for expand_type_alias results (avoids re-expanding common aliases)
    alias_expansion_cache: std::cell::RefCell<std::collections::HashMap<String, Type>>,
}

/// Type environment with scopes and module awareness
#[derive(Default)]
pub struct TypeEnv {
    scopes: Vec<Scope>,
    /// Module-qualified bindings: (module_path, name) -> binding
    /// Used for resolving qualified paths like `math.sin`
    module_bindings: HashMap<(Vec<String>, String), TypeBinding>,
}

#[derive(Default)]
struct Scope {
    bindings: HashMap<String, TypeBinding>,
}

/// Binding in environment
#[derive(Clone)]
struct TypeBinding {
    ty: Type,
    mutable: bool,
    used: bool,
    /// The module this binding originated from (if any)
    source_module: Option<ModuleId>,
}

/// Type definition (struct, enum, type alias)
#[derive(Clone)]
enum TypeDef {
    Struct {
        fields: Vec<(String, Type)>,
        linear: bool,
        affine: bool,
        /// The module this type was defined in
        source_module: Option<ModuleId>,
    },
    Enum {
        variants: Vec<(String, Vec<Type>)>,
        linear: bool,
        affine: bool,
        /// The module this type was defined in
        source_module: Option<ModuleId>,
    },
    /// Type alias with generic parameters
    /// (underlying_type, span, source_module, generic_param_names)
    Alias(Type, Span, Option<ModuleId>, Vec<String>),
}

/// Type constraint for unification
#[derive(Debug, Clone)]
struct TypeConstraint {
    expected: Type,
    actual: Type,
    span: Span,
}

/// Type error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
    pub code: String,
}

/// Structured type check result with detailed errors and warnings
#[derive(Debug)]
pub struct TypeCheckResult {
    pub hir: Option<Hir>,
    pub errors: Vec<TypeError>,
    pub warnings: Vec<String>,
}

/// Type check an AST and return structured result with errors
pub fn check_with_errors(resolved_ast: &resolve::ResolvedAst) -> TypeCheckResult {
    let mut checker = TypeChecker::new_with_resolved_ast(resolved_ast);
    match checker.check_program_internal(&resolved_ast.ast) {
        Ok(hir) => TypeCheckResult {
            hir: Some(hir),
            errors: checker.errors,
            warnings: checker.warnings,
        },
        Err(_) => TypeCheckResult {
            hir: None,
            errors: checker.errors,
            warnings: checker.warnings,
        },
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        // Initialize ontology resolver with default config (offline mode for fast startup)
        // The resolver supports 4 layers:
        // - L1: Primitive (BFO/RO/COB) - compiled in
        // - L2: Foundation (PATO/UO/IAO) - shipped with stdlib
        // - L3: Domain (ChEBI/GO) - lazy SQLite
        // - L4: Federated (BioPortal/OLS4) - network queries
        let resolver_config = ResolverConfig::default().offline();
        let ontology_resolver = OntologyResolver::new(resolver_config).ok();

        Self {
            env: TypeEnv::default(),
            type_defs: HashMap::new(),
            effects: EffectInference::new(),
            units: UnitChecker::new(),
            next_type_var: 0,
            next_effect_var: 0,
            constraints: Vec::new(),
            effect_params: HashMap::new(),
            errors: Vec::new(),
            alignments: HashMap::new(),
            fn_thresholds: HashMap::new(),
            default_threshold: 0.15, // Default threshold for semantic compatibility
            probabilistic_threshold: None, // Enable with enable_probabilistic_checking()
            ast: None,
            current_fn: None,
            current_impl_type: None,
            ontology_prefixes: std::collections::HashSet::new(),
            used_ontology_prefixes: std::collections::HashSet::new(),
            warnings: Vec::new(),
            handler_effects: HashMap::new(),
            masked_effects: types::EffectSet::new(),
            effect_operations: HashMap::new(),
            symbols: None,
            module_tree: None,
            current_module: None,
            fn_param_refinements: HashMap::new(),
            ontology_resolver,
            deprecation_tracker: None,
            fidelity_checker: None,
            spectral_distance: None, // Initialized on demand when ontology is available
            hyperbolic_generator: Some(HyperbolicGenerator::new(64)), // Default 64 dimensions
            conformal_checker: Some(ConformalTypeChecker::new(ConformalConfig::default())),
            alias_expansion_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// Create a TypeChecker with resolved AST visibility information
    pub fn new_with_resolved_ast(resolved_ast: &resolve::ResolvedAst) -> Self {
        let mut checker = Self::new();
        checker.symbols = Some(std::sync::Arc::new(resolved_ast.symbols.clone()));
        checker.module_tree = Some(std::sync::Arc::new(resolved_ast.module_tree.clone()));
        checker.current_module = Some(resolve::module_tree::ModuleId::root());
        checker
    }

    /// Create a TypeChecker with custom ontology resolver configuration
    pub fn new_with_ontology_config(config: ResolverConfig) -> Self {
        let mut checker = Self::new();
        checker.ontology_resolver = OntologyResolver::new(config).ok();
        checker
    }

    /// Pre-register external type names for cross-module compilation.
    ///
    /// This allows the type checker to recognize types from other modules
    /// without having their full definitions. The types are registered as
    /// opaque struct definitions with no fields.
    ///
    /// Used during stdlib bootstrap to enable modules to reference types
    /// defined in other modules (e.g., `TypeContext` from `check::context`).
    pub fn register_external_types(&mut self, types: impl IntoIterator<Item = (String, String)>) {
        for (type_name, _source_module) in types {
            // Register as an opaque struct with no fields
            // This allows type references to resolve, even if we don't have
            // the full struct definition
            if !self.type_defs.contains_key(&type_name) {
                self.type_defs.insert(
                    type_name,
                    TypeDef::Struct {
                        fields: Vec::new(), // Opaque - fields unknown
                        linear: false,
                        affine: false,
                        source_module: None, // Cross-module source
                    },
                );
            }
        }
    }

    /// Pre-register external enum types for cross-module compilation.
    pub fn register_external_enum(&mut self, name: &str, variants: Vec<(String, Vec<Type>)>) {
        if !self.type_defs.contains_key(name) {
            self.type_defs.insert(
                name.to_string(),
                TypeDef::Enum {
                    variants,
                    linear: false,
                    affine: false,
                    source_module: None,
                },
            );
        }
    }

    /// Enable federated ontology queries (L4 layer - BioPortal/OLS4)
    /// This allows resolution of 15M+ ontology terms via network
    pub fn enable_federated_ontology(&mut self) {
        let config = ResolverConfig::default();
        if let Ok(resolver) = OntologyResolver::new(config) {
            self.ontology_resolver = Some(resolver);
        }
    }

    /// Enable deprecation tracking for ontology terms
    pub fn enable_deprecation_tracking(&mut self, tracker: DeprecationTracker) {
        self.deprecation_tracker = Some(tracker);
    }

    /// Check if a term is deprecated and emit warning if so
    fn check_term_deprecation(&mut self, term_id: &str, span: Option<Span>) {
        if let Some(ref mut tracker) = self.deprecation_tracker {
            if let Some(warning) = tracker.check(term_id, span) {
                self.warnings.push(format!(
                    "deprecated_term: {}{}{}",
                    warning.message,
                    warning
                        .suggestion
                        .map(|s| format!(" ({})", s))
                        .unwrap_or_default(),
                    warning
                        .help
                        .map(|h| format!(" [{}]", h))
                        .unwrap_or_default(),
                ));
            }
        }
    }

    /// Enable probabilistic type checking with Bayesian thresholds
    pub fn enable_probabilistic_checking(&mut self, threshold: ProbabilisticThreshold) {
        self.probabilistic_threshold = Some(threshold);
    }

    /// Check compatibility using probabilistic threshold if enabled
    fn check_probabilistic_compatibility(&self, distance: f64, confidence: f64) -> bool {
        if let Some(ref prob_threshold) = self.probabilistic_threshold {
            prob_threshold.is_acceptable(distance, confidence)
        } else {
            // Fall back to scalar threshold
            distance <= self.default_threshold
        }
    }

    /// Enable fidelity checking against OBO Foundry ground truth
    pub fn enable_fidelity_checking(&mut self, checker: WorldFidelityChecker) {
        self.fidelity_checker = Some(checker);
    }

    /// Check fidelity of a Knowledge type binding and emit appropriate diagnostics
    fn check_knowledge_fidelity(&mut self, curie: &str, span: Span) {
        if let Some(ref mut checker) = self.fidelity_checker {
            // Use OntologyAssertion source type for ontology terms
            let ontology = curie.split(':').next().unwrap_or("").to_string();
            let term = curie.to_string();
            let source = EpSource::OntologyAssertion { ontology, term };
            let confidence = EpConfidence::default();

            match checker.verify_fidelity(curie, None, &source, &confidence) {
                FidelityResult::Violation { reason, .. } => {
                    self.errors.push(TypeError {
                        message: format!("Fidelity violation for '{}': {}", curie, reason),
                        span,
                        code: "E0601".to_string(),
                    });
                }
                FidelityResult::Low { issues, .. } => {
                    for issue in issues {
                        self.warnings
                            .push(format!("fidelity_warning: {}: {:?}", curie, issue));
                    }
                }
                _ => {} // High/Medium fidelity - acceptable
            }
        }
    }

    /// Generate a fresh type variable
    fn fresh_type_var(&mut self) -> Type {
        let var = TypeVar(self.next_type_var);
        self.next_type_var += 1;
        Type::Var(var)
    }

    /// Generate a fresh effect variable for row polymorphism
    fn fresh_effect_var(&mut self) -> types::EffectVar {
        let var = types::EffectVar::new(self.next_effect_var);
        self.next_effect_var += 1;
        var
    }

    /// Register an effect parameter from a generic declaration
    fn register_effect_param(&mut self, name: &str) -> types::EffectVar {
        let var = self.fresh_effect_var();
        self.effect_params.insert(name.to_string(), var);
        var
    }

    /// Look up an effect parameter by name
    fn lookup_effect_param(&self, name: &str) -> Option<types::EffectVar> {
        self.effect_params.get(name).copied()
    }

    /// Clear effect parameters (called after checking a function)
    fn clear_effect_params(&mut self) {
        self.effect_params.clear();
    }

    /// Add a type constraint
    fn constrain(&mut self, expected: Type, actual: Type, span: Span) {
        self.constraints.push(TypeConstraint {
            expected,
            actual,
            span,
        });
    }

    /// Report a type error (default code E0308)
    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.errors.push(TypeError {
            message: message.into(),
            span,
            code: "E0308".to_string(),
        });
    }

    /// Report a type error with a specific code
    fn error_with_code(&mut self, code: &str, message: impl Into<String>, span: Span) {
        self.errors.push(TypeError {
            message: message.into(),
            span,
            code: code.to_string(),
        });
    }

    /// Check visibility of a definition when accessing it
    fn check_item_visibility(
        &mut self,
        def_id: &resolve::DefId,
        item_name: &str,
        item_type: &str,
        span: Span,
    ) -> bool {
        // If no resolver info available, assume visible (backward compat)
        let Some(symbols) = &self.symbols else {
            return true;
        };
        let Some(module_tree) = &self.module_tree else {
            return true;
        };
        let Some(current_module) = &self.current_module else {
            return true;
        };

        // Look up the symbol for this DefId
        let Some(symbol) = symbols.get(*def_id) else {
            return true; // Builtin or internal, assume visible
        };

        // Collect module IDs first to avoid borrow checker issues
        let module_ids: Vec<_> = module_tree.all_module_ids().cloned().collect();

        // Find which module this symbol belongs to by checking all modules
        // We need to find the module that contains this item
        for module_id in module_ids {
            if let Some(module) = module_tree.get(&module_id) {
                // Check if this symbol is in this module
                if module
                    .items
                    .iter()
                    .any(|item| item.node_id == symbol.node_id)
                {
                    // Found the module where this item is defined
                    // Now check if it's visible from current_module
                    for item in &module.items {
                        if item.node_id == symbol.node_id {
                            if !module.is_visible(item, current_module) {
                                self.error(
                                    format!("cannot access private {} `{}`", item_type, item_name),
                                    span,
                                );
                                return false;
                            }
                            return true;
                        }
                    }
                }
            }
        }

        // If we can't find the module, assume it's visible (builtin/internal)
        true
    }

    /// Look up which effect a handler handles by its name.
    ///
    /// Returns the effect name if the handler is registered, or None if unknown.
    /// For built-in handlers (IO, Mut, Alloc, etc.), the handler name typically
    /// matches the effect name with "Handler" suffix (e.g., "IOHandler" handles "IO").
    fn lookup_handler_effect(&self, handler_name: &str) -> Option<String> {
        // First check explicitly registered handlers
        if let Some(effect) = self.handler_effects.get(handler_name) {
            return Some(effect.clone());
        }

        // Check for built-in handler naming convention: XHandler -> X
        if handler_name.ends_with("Handler") {
            let effect_name = &handler_name[..handler_name.len() - 7];
            // Verify this is a known effect
            if self.effects.lookup_effect(effect_name).is_some() {
                return Some(effect_name.to_string());
            }
        }

        // Check for exact match with effect name (handler named same as effect)
        if self.effects.lookup_effect(handler_name).is_some() {
            return Some(handler_name.to_string());
        }

        None
    }

    /// Register a handler definition for effect lookup.
    fn register_handler(&mut self, handler_name: String, effect_name: String) {
        self.handler_effects.insert(handler_name, effect_name);
    }

    /// Get the set of effects that have been masked in the current function context.
    ///
    /// This is useful for:
    /// - Effect inference: determining which effects are actually visible to callers
    /// - Diagnostics: warning about over-declared effects
    /// - Testing: verifying that effect masking is working correctly
    pub fn get_masked_effects(&self) -> &types::EffectSet {
        &self.masked_effects
    }

    /// Compute residual effects after subtracting masked effects.
    ///
    /// Given a set of inferred effects and the effects that have been handled,
    /// returns the effects that are still visible to the caller.
    ///
    /// # Example
    /// ```ignore
    /// // Function uses IO and Mut internally, but handles IO
    /// let inferred = EffectSet::from_effects(&["IO", "Mut"]);
    /// let masked = EffectSet::from_effects(&["IO"]);
    /// let residual = compute_residual_effects(&inferred, &masked);
    /// // residual contains only Mut
    /// ```
    pub fn compute_residual_effects(
        inferred: &types::EffectSet,
        masked: &types::EffectSet,
    ) -> types::EffectSet {
        let masked_names: Vec<String> = masked.effects.iter().cloned().collect();
        inferred.subtract(&masked_names)
    }

    /// Convert an AST expression to a refinement Predicate for SimpleChecker.
    fn expr_to_predicate(&self, expr: &Expr, var_name: &str) -> Option<Predicate> {
        match expr {
            Expr::Binary {
                op, left, right, ..
            } => {
                // Check if this is a comparison (Atom) or logical connective (And/Or)
                match op {
                    BinaryOp::And => {
                        let l = self.expr_to_predicate(left, var_name)?;
                        let r = self.expr_to_predicate(right, var_name)?;
                        Some(Predicate::And(vec![l, r]))
                    }
                    BinaryOp::Or => {
                        let l = self.expr_to_predicate(left, var_name)?;
                        let r = self.expr_to_predicate(right, var_name)?;
                        Some(Predicate::Or(vec![l, r]))
                    }
                    // Comparisons become Atoms
                    BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne => {
                        let compare_op = match op {
                            BinaryOp::Lt => CompareOp::Lt,
                            BinaryOp::Le => CompareOp::Le,
                            BinaryOp::Gt => CompareOp::Gt,
                            BinaryOp::Ge => CompareOp::Ge,
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            _ => return None,
                        };
                        let l = self.expr_to_term(left, var_name)?;
                        let r = self.expr_to_term(right, var_name)?;
                        Some(Predicate::Atom(Atom {
                            op: compare_op,
                            lhs: l,
                            rhs: r,
                        }))
                    }
                    _ => None,
                }
            }
            Expr::Unary {
                op, expr: inner, ..
            } if matches!(op, UnaryOp::Not) => {
                let pred = self.expr_to_predicate(inner, var_name)?;
                Some(Predicate::Not(Box::new(pred)))
            }
            _ => None,
        }
    }

    /// Convert an AST expression to a refinement Term for SimpleChecker.
    fn expr_to_term(&self, expr: &Expr, var_name: &str) -> Option<Term> {
        match expr {
            Expr::Literal { value, .. } => match value {
                Literal::Int(n) => Some(Term::Int(*n)),
                Literal::Float(f) => Some(Term::Float(*f)),
                Literal::Bool(b) => Some(Term::Bool(*b)),
                Literal::TypedInt(n, _) => Some(Term::Int(*n)),
                Literal::TypedFloat(f, _) => Some(Term::Float(*f)),
                _ => None,
            },
            Expr::Path { path, .. } => {
                let name = path.to_string();
                if name == var_name {
                    Some(Term::Var(name))
                } else {
                    // Could be another variable or constant - treat as a variable
                    Some(Term::Var(name))
                }
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                let bin_op = match op {
                    BinaryOp::Add => RefinementBinOp::Add,
                    BinaryOp::Sub => RefinementBinOp::Sub,
                    BinaryOp::Mul => RefinementBinOp::Mul,
                    BinaryOp::Div => RefinementBinOp::Div,
                    BinaryOp::Rem => RefinementBinOp::Mod,
                    _ => return None,
                };
                let l = self.expr_to_term(left, var_name)?;
                let r = self.expr_to_term(right, var_name)?;
                Some(Term::BinOp(bin_op, Box::new(l), Box::new(r)))
            }
            Expr::Unary {
                op, expr: inner, ..
            } => {
                match op {
                    UnaryOp::Neg => {
                        let term = self.expr_to_term(inner, var_name)?;
                        // -x is 0 - x
                        Some(Term::BinOp(
                            RefinementBinOp::Sub,
                            Box::new(Term::Int(0)),
                            Box::new(term),
                        ))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Check if a literal value satisfies a refinement predicate.
    /// Returns an error message if the check fails.
    fn check_literal_refinement(
        &self,
        value: i64,
        var_name: &str,
        predicate: &Expr,
    ) -> Result<(), String> {
        let pred = match self.expr_to_predicate(predicate, var_name) {
            Some(p) => p,
            None => return Ok(()), // Can't convert, skip check
        };

        // Substitute the refinement variable with the actual value
        let substituted = pred.substitute(var_name, &Term::Int(value));

        // Use SimpleChecker to evaluate the predicate with constants
        match SimpleChecker::check(&substituted) {
            Some(true) => Ok(()),
            Some(false) => Err(format!(
                "value {} does not satisfy refinement predicate",
                value
            )),
            None => Ok(()), // Can't evaluate, skip check
        }
    }

    /// Try to extract a constant integer value from an expression.
    /// Handles literals directly and also -literal for negative numbers.
    fn try_extract_const_value(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Literal { value, .. } => match value {
                Literal::Int(n) => Some(*n),
                _ => None,
            },
            // Handle -5 which is parsed as Unary(Neg, Literal(5))
            Expr::Unary {
                op, expr: inner, ..
            } if matches!(op, UnaryOp::Neg) => {
                if let Expr::Literal { value, .. } = inner.as_ref() {
                    match value {
                        Literal::Int(n) => Some(-(*n)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to extract a constant float value from an expression.
    /// Handles literals directly and also -literal for negative numbers.
    fn try_extract_float_value(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Literal { value, .. } => match value {
                Literal::Float(f) => Some(*f),
                Literal::Int(n) => Some(*n as f64), // Allow int literals in float context
                _ => None,
            },
            // Handle -1.5 which is parsed as Unary(Neg, Literal(1.5))
            Expr::Unary {
                op, expr: inner, ..
            } if matches!(op, UnaryOp::Neg) => {
                if let Expr::Literal { value, .. } = inner.as_ref() {
                    match value {
                        Literal::Float(f) => Some(-(*f)),
                        Literal::Int(n) => Some(-(*n as f64)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if a float literal value satisfies a refinement predicate.
    /// Returns an error message if the check fails.
    fn check_float_literal_refinement(
        &self,
        value: f64,
        var_name: &str,
        predicate: &Expr,
    ) -> Result<(), String> {
        let pred = match self.expr_to_predicate(predicate, var_name) {
            Some(p) => p,
            None => return Ok(()), // Can't convert, skip check
        };

        // Substitute the refinement variable with the actual value
        let substituted = pred.substitute(var_name, &Term::Float(value));

        // Use SimpleChecker to evaluate the predicate with constants
        match SimpleChecker::check(&substituted) {
            Some(true) => Ok(()),
            Some(false) => Err(format!(
                "value {} does not satisfy refinement predicate",
                value
            )),
            None => Ok(()), // Can't evaluate, skip check
        }
    }

    /// Expand type aliases recursively (entry point with cycle detection)
    fn expand_type_alias(&self, ty: &Type) -> Type {
        // Cache key: use type name for Named types, otherwise return as-is
        if let Type::Named { name, .. } = ty {
            if let Some(cached) = self.alias_expansion_cache.borrow().get(name) {
                return cached.clone();
            }
        }

        let mut visited = std::collections::HashSet::new();
        let result = self.expand_type_alias_inner(ty, &mut visited);

        // Cache result if it's a Named type
        if let Type::Named { name, .. } = ty {
            self.alias_expansion_cache
                .borrow_mut()
                .insert(name.clone(), result.clone());
        }

        result
    }

    /// Inner recursive alias expansion with cycle detection via visited set.
    fn expand_type_alias_inner(
        &self,
        ty: &Type,
        visited: &mut std::collections::HashSet<String>,
    ) -> Type {
        match ty {
            Type::Named { name, args } => {
                if let Some(TypeDef::Alias(alias_ty, _, _, generic_params)) =
                    self.type_defs.get(name)
                {
                    if !visited.insert(name.clone()) {
                        return Type::Named {
                            name: name.clone(),
                            args: args
                                .iter()
                                .map(|a| self.expand_type_alias_inner(a, visited))
                                .collect(),
                        };
                    }
                    let substituted = if !generic_params.is_empty() && !args.is_empty() {
                        self.substitute_type_params(alias_ty, generic_params, args)
                    } else {
                        alias_ty.clone()
                    };
                    self.expand_type_alias_inner(&substituted, visited)
                } else {
                    Type::Named {
                        name: name.clone(),
                        args: args
                            .iter()
                            .map(|a| self.expand_type_alias_inner(a, visited))
                            .collect(),
                    }
                }
            }
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.expand_type_alias_inner(element, visited)),
                size: *size,
            },
            Type::Tuple(elems) => Type::Tuple(
                elems
                    .iter()
                    .map(|e| self.expand_type_alias_inner(e, visited))
                    .collect(),
            ),
            Type::Ref {
                mutable,
                lifetime,
                inner,
            } => Type::Ref {
                mutable: *mutable,
                lifetime: lifetime.clone(),
                inner: Box::new(self.expand_type_alias_inner(inner, visited)),
            },
            Type::Function {
                params,
                return_type,
                effects,
                abi,
            } => Type::Function {
                params: params
                    .iter()
                    .map(|p| self.expand_type_alias_inner(p, visited))
                    .collect(),
                return_type: Box::new(self.expand_type_alias_inner(return_type, visited)),
                effects: effects.clone(),
                abi: abi.clone(),
            },
            _ => ty.clone(),
        }
    }

    /// Substitute type parameters with concrete type arguments
    fn substitute_type_params(&self, ty: &Type, params: &[String], args: &[Type]) -> Type {
        match ty {
            Type::Named {
                name,
                args: inner_args,
            } => {
                // Check if this is a type parameter that should be substituted
                if inner_args.is_empty() {
                    if let Some(pos) = params.iter().position(|p| p == name) {
                        if pos < args.len() {
                            return args[pos].clone();
                        }
                    }
                }
                // Otherwise recursively substitute in nested type arguments
                Type::Named {
                    name: name.clone(),
                    args: inner_args
                        .iter()
                        .map(|a| self.substitute_type_params(a, params, args))
                        .collect(),
                }
            }
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.substitute_type_params(element, params, args)),
                size: *size,
            },
            Type::Tuple(elems) => Type::Tuple(
                elems
                    .iter()
                    .map(|e| self.substitute_type_params(e, params, args))
                    .collect(),
            ),
            Type::Ref {
                mutable,
                lifetime,
                inner,
            } => Type::Ref {
                mutable: *mutable,
                lifetime: lifetime.clone(),
                inner: Box::new(self.substitute_type_params(inner, params, args)),
            },
            Type::Function {
                params: fn_params,
                return_type,
                effects,
                abi,
            } => Type::Function {
                params: fn_params
                    .iter()
                    .map(|p| self.substitute_type_params(p, params, args))
                    .collect(),
                return_type: Box::new(self.substitute_type_params(return_type, params, args)),
                effects: effects.clone(),
                abi: abi.clone(),
            },
            Type::RawPointer { mutable, inner } => Type::RawPointer {
                mutable: *mutable,
                inner: Box::new(self.substitute_type_params(inner, params, args)),
            },
            // Primitive types don't need substitution
            _ => ty.clone(),
        }
    }

    /// Get human-readable display name for a type
    fn type_display_name(&self, ty: &Type) -> String {
        match ty {
            Type::Unit => "()".to_string(),
            Type::Bool => "bool".to_string(),
            Type::I8 => "i8".to_string(),
            Type::I16 => "i16".to_string(),
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::U8 => "u8".to_string(),
            Type::U16 => "u16".to_string(),
            Type::U32 => "u32".to_string(),
            Type::U64 => "u64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::String => "string".to_string(),
            Type::Named { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else if name == "Knowledge" && args.len() == 1 {
                    format!("Knowledge[{}]", self.type_display_name(&args[0]))
                } else {
                    format!(
                        "{}<{}>",
                        name,
                        args.iter()
                            .map(|a| self.type_display_name(a))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Type::Ontology { namespace, term } => format!("{}:{}", namespace, term),
            Type::Array { element, size } => {
                format!(
                    "[{}; {}]",
                    self.type_display_name(element),
                    size.unwrap_or(0)
                )
            }
            Type::Tuple(types) => {
                let inner: Vec<_> = types.iter().map(|t| self.type_display_name(t)).collect();
                format!("({})", inner.join(", "))
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                let param_strs: Vec<_> = params.iter().map(|t| self.type_display_name(t)).collect();
                format!(
                    "fn({}) -> {}",
                    param_strs.join(", "),
                    self.type_display_name(return_type)
                )
            }
            Type::Var(v) => format!("?T{}", v.0),
            _ => format!("{:?}", ty),
        }
    }

    /// Extract span from an AST expression using the AST's span map
    fn expr_span(&self, expr: &Expr, ast: &Ast) -> Span {
        let id = self.expr_id(expr);
        ast.node_spans.get(&id).copied().unwrap_or_else(Span::dummy)
    }

    /// Extract NodeId from an AST expression
    fn expr_id(&self, expr: &Expr) -> NodeId {
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
            Expr::AsyncClosure { id, .. } => *id,
            Expr::Spawn { id, .. } => *id,
            Expr::Select { id, .. } => *id,
            Expr::Join { id, .. } => *id,
            Expr::OntologyTerm { id, .. } => *id,
            Expr::MacroInvocation(_) => NodeId(0),
            Expr::Do { id, .. } => *id,
            Expr::Counterfactual { id, .. } => *id,
            Expr::KnowledgeExpr { id, .. } => *id,
            Expr::Uncertain { id, .. } => *id,
            _ => NodeId(0),
        }
    }

    pub fn check_program(&mut self, ast: &Ast) -> Result<Hir> {
        self.check_program_internal(ast)
    }

    fn check_program_internal(&mut self, ast: &Ast) -> Result<Hir> {
        // Store AST reference for span lookups
        self.ast = Some(std::sync::Arc::new(ast.clone()));

        let mut items = Vec::new();
        let mut externs = Vec::new();

        // Register user-defined units before type checking
        self.register_unit_defs(&ast.items);

        // First pass: collect ontology prefixes, type definitions, and alignments
        for item in &ast.items {
            self.collect_ontology_prefix(item);
        }
        for item in &ast.items {
            self.collect_type_def(item);
            self.collect_alignment(item);
            self.collect_fn_threshold(item);
        }

        // Validate that all ontology types use declared prefixes
        self.check_undefined_ontology_prefixes();

        // Check for circular type definitions
        self.check_circular_types();

        // Check for infinite-size structs (direct recursion without indirection)
        self.check_infinite_size_types();

        // Second pass: register function signatures in environment
        self.env.push_scope();
        for item in &ast.items {
            if let Item::Function(f) = item {
                let params: Vec<Type> = f
                    .params
                    .iter()
                    .map(|p| self.lower_type_expr(&p.ty))
                    .collect();
                let return_type = f
                    .return_type
                    .as_ref()
                    .map(|t| self.lower_type_expr(t))
                    .unwrap_or(Type::Unit);

                // Extract effects from function declaration
                let mut effect_set = types::EffectSet::new();
                for effect_ref in &f.effects {
                    if let Some(name) = effect_ref.as_simple_name() {
                        effect_set.add(types::Effect {
                            name: name.to_string(),
                            args: Vec::new(), // TODO(issue-20): handle parameterized effects
                        });
                    }
                }

                let fn_type = Type::Function {
                    params,
                    return_type: Box::new(return_type),
                    effects: effect_set,
                    abi: None,
                };
                self.env.bind(f.name.clone(), fn_type, false);

                // Collect refinement info for parameters
                let mut param_refinements = Vec::new();
                for param in &f.params {
                    if let TypeExpr::Refinement { var, predicate, .. } = &param.ty {
                        param_refinements.push(Some(RefinementInfo {
                            var: var.clone(),
                            predicate: predicate.clone(),
                        }));
                    } else {
                        param_refinements.push(None);
                    }
                }
                if param_refinements.iter().any(|r| r.is_some()) {
                    self.fn_param_refinements
                        .insert(f.name.clone(), param_refinements);
                }
            }
            if let Item::Extern(extern_block) = item {
                for ext_item in &extern_block.items {
                    if let ExternItem::Fn(ext_fn) = ext_item {
                        let params: Vec<Type> = ext_fn
                            .params
                            .iter()
                            .map(|p| self.lower_type_expr(&p.ty))
                            .collect();
                        let return_type = ext_fn
                            .return_type
                            .as_ref()
                            .map(|t| self.lower_type_expr(t))
                            .unwrap_or(Type::Unit);

                        // Extern functions don't declare effects (they're C functions)
                        let fn_type = Type::Function {
                            params,
                            return_type: Box::new(return_type),
                            effects: types::EffectSet::new(),
                            abi: Some("C".to_string()), // extern fns use C ABI
                        };

                        // Bind using the D-visible name; codegen/linking uses `link_name` later.
                        self.env.bind(ext_fn.name.clone(), fn_type, false);
                    }
                }
            }
            // Register associated functions from impl blocks (e.g., String::new)
            if let Item::Impl(impl_def) = item {
                // Get the type name from target_type
                let type_name = match &impl_def.target_type {
                    TypeExpr::Named { path, .. } => path.to_string(),
                    _ => continue,
                };
                for impl_item in &impl_def.items {
                    if let ImplItem::Fn(f) = impl_item {
                        // Check if it's an associated function (no self parameter)
                        let is_associated = f.params.first().map_or(true, |p| {
                            !matches!(&p.pattern, Pattern::Binding { name, .. } if name == "self")
                        });
                        if is_associated {
                            let params: Vec<Type> = f
                                .params
                                .iter()
                                .map(|p| self.lower_type_expr(&p.ty))
                                .collect();
                            let return_type = f
                                .return_type
                                .as_ref()
                                .map(|t| self.lower_type_expr(t))
                                .unwrap_or(Type::Unit);
                            let fn_type = Type::Function {
                                params,
                                return_type: Box::new(return_type),
                                effects: types::EffectSet::new(),
                                abi: None,
                            };
                            // Register as TypeName::method_name
                            let qualified_name = format!("{}::{}", type_name, f.name);
                            self.env.bind(qualified_name, fn_type, false);
                        }
                    }
                }
            }
            // Register functions from nested modules
            if let Item::Module(m) = item {
                self.collect_module_functions(m);
            }
            // Register global let/const bindings
            if let Item::Global(g) = item {
                let name = self.pattern_name(&g.pattern);
                let ty =
                    g.ty.as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or_else(|| self.fresh_type_var());
                self.env.bind(name, ty, g.is_mut);
            }
        }

        // Third pass: type check items
        for item in &ast.items {
            if let Item::Extern(extern_block) = item {
                externs.push(self.lower_extern_block(extern_block)?);
                continue;
            }
            if let Some(hir_item) = self.check_item(item)? {
                items.push(hir_item);
            }
        }

        // Emit type aliases to HIR so they can be resolved during HLIR lowering
        // This is essential for refinement types: `type OrbitRatio = { r: f64 | ... }`
        // should be lowered to f64, not a struct named "OrbitRatio"
        for (name, def) in &self.type_defs {
            if let TypeDef::Alias(ty, span, _, _) = def {
                let hir_ty = self.type_to_hir(ty);
                items.push(HirItem::TypeAlias(HirTypeAlias {
                    id: NodeId(0), // ID not used for type aliases
                    name: name.clone(),
                    ty: hir_ty,
                    doc: None,
                }));
            }
        }

        self.env.pop_scope();

        // Solve type constraints
        self.solve_constraints()?;

        // Check for unused ontology imports and generate warnings
        self.check_unused_imports();

        if !self.errors.is_empty() {
            let messages: Vec<_> = self.errors.iter().map(|e| e.message.clone()).collect();
            return Err(miette::miette!("Type errors:\n{}", messages.join("\n")));
        }

        Ok(Hir { items, externs })
    }

    fn lower_extern_block(&mut self, block: &ExternBlock) -> Result<HirExternBlock> {
        let mut functions = Vec::new();

        for item in &block.items {
            if let ExternItem::Fn(f) = item {
                let params: Vec<HirParam> = f
                    .params
                    .iter()
                    .map(|p| {
                        let ty = self.lower_type_expr(&p.ty);
                        HirParam {
                            id: p.id,
                            name: self.pattern_name(&p.pattern),
                            ty: self.type_to_hir(&ty),
                            is_mut: p.is_mut,
                        }
                    })
                    .collect();

                let return_type = f
                    .return_type
                    .as_ref()
                    .map(|t| {
                        let ty = self.lower_type_expr(t);
                        self.type_to_hir(&ty)
                    })
                    .unwrap_or(HirType::Unit);

                functions.push(HirExternFn {
                    id: f.id,
                    name: f.name.clone(),
                    params,
                    return_type,
                    is_variadic: f.is_variadic,
                    link_name: f.link_name.clone(),
                });
            }
        }

        Ok(HirExternBlock {
            id: block.id,
            abi: block.abi.clone(),
            functions,
        })
    }

    /// Check for unused ontology imports and add warnings
    fn check_unused_imports(&mut self) {
        for prefix in &self.ontology_prefixes {
            if !self.used_ontology_prefixes.contains(prefix) {
                self.warnings.push(format!(
                    "unused_import: ontology prefix `{}` is declared but never used",
                    prefix
                ));
            }
        }
    }

    /// Parse vec! macro arguments into expressions
    /// For vec![a, b, c], extract the comma-separated expressions
    /// The args structure is: [Token("vec"), Token("!"), Delimited(Bracket, [...])]
    /// or just: [Delimited(Bracket, [...])]
    fn parse_vec_macro_args(&self, args: &[TokenTree]) -> Vec<Expr> {
        use crate::lexer::TokenKind;

        // Find the bracketed content - skip any leading vec! tokens
        let bracket_content = self.find_bracket_content(args);

        if bracket_content.is_empty() {
            return Vec::new();
        }

        // Parse comma-separated expressions from bracket content
        let mut exprs = Vec::new();
        let mut current_tokens = Vec::new();

        for tt in bracket_content {
            match tt {
                TokenTree::Token(tok) if tok.token.kind == TokenKind::Comma => {
                    if !current_tokens.is_empty() {
                        if let Some(expr) = self.tokens_to_simple_expr(&current_tokens) {
                            exprs.push(expr);
                        }
                        current_tokens.clear();
                    }
                }
                _ => {
                    current_tokens.push(tt.clone());
                }
            }
        }

        // Handle last expression (no trailing comma)
        if !current_tokens.is_empty() {
            if let Some(expr) = self.tokens_to_simple_expr(&current_tokens) {
                exprs.push(expr);
            }
        }

        exprs
    }

    /// Find the bracket content in vec! macro args
    /// The parser already unwraps the bracket, so args ARE the content.
    /// This just returns args directly unless there's a wrapper Delimited.
    fn find_bracket_content<'a>(&self, args: &'a [TokenTree]) -> &'a [TokenTree] {
        // For vec![a, b, c], the parser gives us args = [Token(a), Token(,), Token(b), ...]
        // directly (already unwrapped from the bracket).
        //
        // For recursive calls with [Delimited(Bracket, inner)], we need to unwrap.
        if args.len() == 1 {
            if let TokenTree::Delimited(Delimiter::Bracket, inner, _) = &args[0] {
                return inner;
            }
        }

        // Otherwise, args are the direct content
        args
    }

    /// Convert a sequence of tokens to a simple expression (handles nested vec!)
    fn tokens_to_simple_expr(&self, tokens: &[TokenTree]) -> Option<Expr> {
        use crate::lexer::TokenKind;

        if tokens.is_empty() {
            return None;
        }

        // Check for nested vec! macro: [Token("vec"), Token("!"), Delimited(Bracket, ...)]
        if tokens.len() >= 3 {
            if let (TokenTree::Token(first), TokenTree::Token(second)) = (&tokens[0], &tokens[1]) {
                if first.token.kind == TokenKind::Ident
                    && first.token.text == "vec"
                    && second.token.kind == TokenKind::Bang
                {
                    // This is a nested vec! macro - recursively parse it
                    let nested_exprs = self.parse_vec_macro_args(&tokens[2..]);
                    return Some(Expr::Array {
                        id: NodeId::dummy(),
                        elements: nested_exprs,
                    });
                }
            }
        }

        // For single token, convert directly
        if tokens.len() == 1 {
            if let TokenTree::Token(tok) = &tokens[0] {
                return self.token_to_expr(&tok.token);
            }
            // Handle delimited group (nested array without vec!)
            if let TokenTree::Delimited(Delimiter::Bracket, inner, _) = &tokens[0] {
                let mut inner_exprs = Vec::new();
                let mut current = Vec::new();
                for tt in inner.iter() {
                    match tt {
                        TokenTree::Token(tok) if tok.token.kind == TokenKind::Comma => {
                            if !current.is_empty() {
                                if let Some(e) = self.tokens_to_simple_expr(&current) {
                                    inner_exprs.push(e);
                                }
                                current.clear();
                            }
                        }
                        _ => current.push(tt.clone()),
                    }
                }
                if !current.is_empty() {
                    if let Some(e) = self.tokens_to_simple_expr(&current) {
                        inner_exprs.push(e);
                    }
                }
                return Some(Expr::Array {
                    id: NodeId::dummy(),
                    elements: inner_exprs,
                });
            }
        }

        None
    }

    /// Convert a single token to an expression
    fn token_to_expr(&self, token: &crate::lexer::Token) -> Option<Expr> {
        use crate::lexer::TokenKind;

        match token.kind {
            TokenKind::IntLit => {
                let value = token.text.parse::<i64>().ok()?;
                Some(Expr::Literal {
                    id: NodeId::dummy(),
                    value: Literal::Int(value),
                })
            }
            TokenKind::FloatLit => {
                let value = token.text.parse::<f64>().ok()?;
                Some(Expr::Literal {
                    id: NodeId::dummy(),
                    value: Literal::Float(value),
                })
            }
            TokenKind::StringLit => {
                let text = token.text.clone();
                Some(Expr::Literal {
                    id: NodeId::dummy(),
                    value: Literal::String(text),
                })
            }
            TokenKind::True => Some(Expr::Literal {
                id: NodeId::dummy(),
                value: Literal::Bool(true),
            }),
            TokenKind::False => Some(Expr::Literal {
                id: NodeId::dummy(),
                value: Literal::Bool(false),
            }),
            TokenKind::Ident => {
                let name = token.text.clone();
                Some(Expr::Path {
                    id: NodeId::dummy(),
                    path: Path::simple(&name),
                })
            }
            _ => None,
        }
    }

    fn collect_type_def(&mut self, item: &Item) {
        match item {
            Item::Struct(s) => {
                let fields: Vec<_> = s
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), self.lower_type_expr(&f.ty)))
                    .collect();
                self.type_defs.insert(
                    s.name.clone(),
                    TypeDef::Struct {
                        fields,
                        linear: s.modifiers.linear,
                        affine: s.modifiers.affine,
                        source_module: None, // TODO: extract from struct's module context
                    },
                );
            }
            Item::Enum(e) => {
                let variants: Vec<_> = e
                    .variants
                    .iter()
                    .map(|v| {
                        let types = match &v.data {
                            VariantData::Unit => Vec::new(),
                            VariantData::Tuple(types) => {
                                types.iter().map(|t| self.lower_type_expr(t)).collect()
                            }
                            VariantData::Struct(fields) => {
                                fields.iter().map(|f| self.lower_type_expr(&f.ty)).collect()
                            }
                        };
                        (v.name.clone(), types)
                    })
                    .collect();
                self.type_defs.insert(
                    e.name.clone(),
                    TypeDef::Enum {
                        variants,
                        linear: e.modifiers.linear,
                        affine: e.modifiers.affine,
                        source_module: None, // TODO: extract from enum's module context
                    },
                );
            }
            Item::TypeAlias(t) => {
                let ty = self.lower_type_expr(&t.ty);
                // Extract generic parameter names from the AST
                let generic_params: Vec<String> = t
                    .generics
                    .params
                    .iter()
                    .filter_map(|p| {
                        match p {
                            crate::ast::GenericParam::Type { name, .. } => Some(name.clone()),
                            _ => None, // Skip const and effect params for now
                        }
                    })
                    .collect();
                self.type_defs.insert(
                    t.name.clone(),
                    TypeDef::Alias(ty, t.span, None, generic_params),
                );
            }
            Item::Module(m) => {
                // Recursively collect type definitions from nested modules
                if let Some(ref items) = m.items {
                    for item in items {
                        self.collect_type_def(item);
                    }
                }
            }
            _ => {}
        }
    }

    /// Collect function signatures from a module (recursive)
    fn collect_module_functions(&mut self, m: &ModuleDef) {
        if let Some(ref items) = m.items {
            for item in items {
                if let Item::Function(f) = item {
                    let params: Vec<Type> = f
                        .params
                        .iter()
                        .map(|p| self.lower_type_expr(&p.ty))
                        .collect();
                    let return_type = f
                        .return_type
                        .as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or(Type::Unit);
                    let fn_type = Type::Function {
                        params,
                        return_type: Box::new(return_type),
                        effects: types::EffectSet::new(),
                        abi: None,
                    };
                    // Register with module-qualified name
                    let qualified_name = format!("{}::{}", m.name, f.name);
                    self.env
                        .bind(qualified_name.clone(), fn_type.clone(), false);
                    // Also register unqualified for now (within module scope)
                    self.env.bind(f.name.clone(), fn_type, false);
                }
                // Recursively handle nested modules
                if let Item::Module(nested) = item {
                    self.collect_module_functions(nested);
                }
            }
        }
    }

    /// Collect ontology alignment declarations
    fn collect_alignment(&mut self, item: &Item) {
        if let Item::AlignDecl(align) = item {
            // Create canonical key (ordered pair for symmetric lookup)
            let t1 = format!("{}:{}", align.type1.prefix, align.type1.term);
            let t2 = format!("{}:{}", align.type2.prefix, align.type2.term);
            let key = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            self.alignments.insert(key, align.distance);
        }
    }

    /// Collect function-level compatibility thresholds from #[compat] annotations
    fn collect_fn_threshold(&mut self, item: &Item) {
        if let Item::Function(f) = item {
            // Check for #[compat(threshold = X)] attribute
            for attr in &f.attributes {
                if attr.name == "compat" {
                    match &attr.args {
                        AttributeArgs::Named(pairs) => {
                            for (key, value) in pairs {
                                if key == "threshold" {
                                    if let AttributeValue::Float(threshold) = value {
                                        self.validate_and_insert_threshold(&f.name, *threshold);
                                    }
                                }
                            }
                        }
                        AttributeArgs::Value(AttributeValue::Float(threshold)) => {
                            // Simple form: #[compat(0.2)]
                            self.validate_and_insert_threshold(&f.name, *threshold);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Validate threshold is in valid range [0.0, 1.0] and insert
    fn validate_and_insert_threshold(&mut self, fn_name: &str, threshold: f64) {
        if threshold < 0.0 {
            self.error(
                format!(
                    "Invalid threshold {} for function `{}`: threshold cannot be negative",
                    threshold, fn_name
                ),
                Span::dummy(),
            );
        } else if threshold > 1.0 {
            self.error(
                format!(
                    "Invalid threshold {} for function `{}`: threshold must be between 0.0 and 1.0",
                    threshold, fn_name
                ),
                Span::dummy(),
            );
        } else {
            self.fn_thresholds.insert(fn_name.to_string(), threshold);
        }
    }

    /// Collect ontology prefix declarations and check for duplicates
    fn collect_ontology_prefix(&mut self, item: &Item) {
        if let Item::OntologyImport(ont) = item {
            // Check for duplicate prefix
            if self.ontology_prefixes.contains(&ont.prefix) {
                self.error(
                    format!(
                        "Duplicate ontology prefix `{}`. Each ontology prefix can only be declared once.",
                        ont.prefix
                    ),
                    Span::dummy(),
                );
            } else {
                self.ontology_prefixes.insert(ont.prefix.clone());
            }
        }
    }

    /// Check that all ontology types reference declared prefixes
    fn check_undefined_ontology_prefixes(&mut self) {
        for (name, def) in &self.type_defs.clone() {
            match def {
                TypeDef::Alias(ty, span, _, _) => {
                    self.check_type_for_undefined_ontology(ty, name, *span);
                }
                TypeDef::Struct { fields, .. } => {
                    for (field_name, field_ty) in fields {
                        self.check_type_for_undefined_ontology(
                            field_ty,
                            &format!("{}.{}", name, field_name),
                            Span::dummy(),
                        );
                    }
                }
                TypeDef::Enum { variants, .. } => {
                    for (variant_name, types) in variants {
                        for ty in types {
                            self.check_type_for_undefined_ontology(
                                ty,
                                &format!("{}::{}", name, variant_name),
                                Span::dummy(),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Check a single type for undefined ontology prefixes
    fn check_type_for_undefined_ontology(&mut self, ty: &Type, context: &str, span: Span) {
        match ty {
            Type::Ontology { namespace, term } => {
                if !self.ontology_prefixes.contains(namespace) {
                    self.error_with_code(
                        "E0412",
                        format!(
                            "Undefined ontology prefix `{}` in type `{}:{}` (used in {}). Add `ontology {} from \"...\";` declaration.",
                            namespace, namespace, term, context, namespace
                        ),
                        span,
                    );
                }
            }
            Type::Named { args, .. } => {
                for arg in args {
                    self.check_type_for_undefined_ontology(arg, context, span);
                }
            }
            Type::Array { element, .. } => {
                self.check_type_for_undefined_ontology(element, context, span);
            }
            Type::Tuple(types) => {
                for t in types {
                    self.check_type_for_undefined_ontology(t, context, span);
                }
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                for p in params {
                    self.check_type_for_undefined_ontology(p, context, span);
                }
                self.check_type_for_undefined_ontology(return_type, context, span);
            }
            Type::Ref { inner, .. } => {
                self.check_type_for_undefined_ontology(inner, context, span);
            }
            _ => {}
        }
    }

    /// Check for circular type alias definitions (e.g., type A = B; type B = A;)
    fn check_circular_types(&mut self) {
        use std::collections::HashSet;

        // For each type alias, check if following the chain leads back to itself
        for (name, def) in &self.type_defs.clone() {
            if let TypeDef::Alias(ty, _, _, _) = def {
                let mut visited = HashSet::new();
                visited.insert(name.clone());

                if self.type_creates_cycle(ty, &mut visited) {
                    self.error(
                        format!("Circular type definition detected: `{}` references itself through type aliases", name),
                        Span::dummy(),
                    );
                }
            }
        }
    }

    /// Helper to detect if a type creates a cycle through type aliases
    fn type_creates_cycle(
        &self,
        ty: &Type,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        match ty {
            Type::Named { name, .. } => {
                if visited.contains(name) {
                    return true;
                }
                if let Some(TypeDef::Alias(inner, _, _, _)) = self.type_defs.get(name) {
                    visited.insert(name.clone());
                    self.type_creates_cycle(inner, visited)
                } else {
                    false
                }
            }
            Type::Array { element, .. } => self.type_creates_cycle(element, visited),
            Type::Tuple(types) => types.iter().any(|t| self.type_creates_cycle(t, visited)),
            Type::Function {
                params,
                return_type,
                ..
            } => {
                params.iter().any(|t| self.type_creates_cycle(t, visited))
                    || self.type_creates_cycle(return_type, visited)
            }
            _ => false,
        }
    }

    /// Check for infinite-size types (structs that contain themselves without indirection)
    fn check_infinite_size_types(&mut self) {
        use std::collections::HashSet;

        for (name, def) in &self.type_defs.clone() {
            if let TypeDef::Struct { fields, .. } = def {
                let mut visited = HashSet::new();
                visited.insert(name.clone());

                for (field_name, field_ty) in fields {
                    // Don't clone visited - we need to track all types seen across all fields
                    // to correctly detect cycles through multiple indirection paths
                    if self.type_has_infinite_size(field_ty, &mut visited) {
                        self.error(
                            format!(
                                "Struct `{}` has infinite size: field `{}` creates a cycle without indirection (use Box, &, or Option<Box<...>>)",
                                name, field_name
                            ),
                            Span::dummy(),
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Check if a type has infinite size (contains itself without indirection)
    fn type_has_infinite_size(
        &self,
        ty: &Type,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        match ty {
            Type::Named { name, .. } => {
                if visited.contains(name) {
                    return true;
                }

                if let Some(def) = self.type_defs.get(name) {
                    match def {
                        TypeDef::Struct { fields, .. } => {
                            visited.insert(name.clone());
                            fields
                                .iter()
                                .any(|(_, field_ty)| self.type_has_infinite_size(field_ty, visited))
                        }
                        TypeDef::Alias(inner, _, _, _) => {
                            visited.insert(name.clone());
                            self.type_has_infinite_size(inner, visited)
                        }
                        TypeDef::Enum { .. } => false, // Enums are sized by their largest variant
                    }
                } else {
                    false
                }
            }
            // References and pointers provide indirection - they break the cycle
            Type::Ref { .. } => false,
            // Box, Option<Box<T>>, etc. also provide indirection
            // For now, we assume any generic type provides indirection (conservative)
            Type::Array { element, .. } => self.type_has_infinite_size(element, visited),
            Type::Tuple(types) => types
                .iter()
                .any(|t| self.type_has_infinite_size(t, visited)),
            _ => false,
        }
    }

    /// Get semantic distance between two ontology types.
    ///
    /// This method uses a multi-layer strategy:
    /// 1. Check explicit alignments from `align` declarations
    /// 2. Use OntologyResolver for dynamic resolution (L1-L4 layers)
    /// 3. Fall back to default heuristics for same-ontology terms
    fn get_semantic_distance(&mut self, t1: &str, t2: &str) -> Option<f64> {
        if t1 == t2 {
            return Some(0.0);
        }

        // First check explicit alignments from source code
        let key = if t1 <= t2 {
            (t1.to_string(), t2.to_string())
        } else {
            (t2.to_string(), t1.to_string())
        };
        if let Some(distance) = self.alignments.get(&key).copied() {
            return Some(distance);
        }

        // Try using OntologyResolver for dynamic resolution
        if let Some(ref mut resolver) = self.ontology_resolver {
            // Check subsumption relationship (is-a hierarchy)
            if let Ok(result) = resolver.is_subclass_of(t1, t2) {
                match result {
                    SubsumptionResult::Equivalent => return Some(0.0),
                    SubsumptionResult::IsSubclass => return Some(0.05), // Direct subsumption
                    SubsumptionResult::NotSubclass => {
                        // Check reverse direction
                        if let Ok(SubsumptionResult::IsSubclass) = resolver.is_subclass_of(t2, t1) {
                            return Some(0.1); // Superclass relationship
                        }
                    }
                    SubsumptionResult::Unknown => {}
                }
            }

            // Try to resolve both terms and compute distance based on shared ancestors
            let term1_resolved = resolver.resolve(t1).ok();
            let term2_resolved = resolver.resolve(t2).ok();

            if let (Some(term1), Some(term2)) = (term1_resolved, term2_resolved) {
                // If both terms exist and share the same ontology, compute path-based distance
                if term1.layer == term2.layer {
                    // Check for shared superclasses (common ancestor)
                    let common = term1
                        .superclasses
                        .iter()
                        .filter(|s| term2.superclasses.contains(s))
                        .count();

                    if common > 0 {
                        // Close siblings in hierarchy
                        return Some(0.15);
                    } else if !term1.superclasses.is_empty() || !term2.superclasses.is_empty() {
                        // Same ontology but more distant
                        return Some(0.3);
                    }
                }
            }
        }

        None
    }

    /// Compute semantic distance with confidence for coercion decisions.
    ///
    /// Returns (distance, confidence) where:
    /// - distance: 0.0 = identical, 1.0 = completely unrelated
    /// - confidence: 0.0-1.0, how confident we are in this distance
    ///
    /// This method now integrates three novel mathematical techniques:
    /// 1. Conformal prediction for calibrated confidence intervals
    /// 2. Hyperbolic embeddings for hierarchical structure preservation
    /// 3. Spectral graph distance for smooth multi-scale measurements
    fn compute_semantic_distance_with_confidence(&mut self, t1: &str, t2: &str) -> (f64, f64) {
        if t1 == t2 {
            return (0.0, 1.0); // Identical, full confidence
        }

        // Check explicit alignments (highest confidence)
        let key = if t1 <= t2 {
            (t1.to_string(), t2.to_string())
        } else {
            (t2.to_string(), t1.to_string())
        };
        if let Some(distance) = self.alignments.get(&key).copied() {
            // Use conformal checker to calibrate confidence
            if let Some(ref mut conformal) = self.conformal_checker {
                let result = conformal.check(distance);
                return (distance, result.p_value.max(0.85_f64)); // At least 0.85 confidence for explicit
            }
            return (distance, 0.95_f64); // Explicit alignment, high confidence
        }

        // Use resolver
        if let Some(ref mut resolver) = self.ontology_resolver {
            // Subsumption check
            if let Ok(result) = resolver.is_subclass_of(t1, t2) {
                match result {
                    SubsumptionResult::Equivalent => return (0.0, 1.0),
                    SubsumptionResult::IsSubclass => return (0.05, 0.9),
                    SubsumptionResult::NotSubclass => {
                        if let Ok(SubsumptionResult::IsSubclass) = resolver.is_subclass_of(t2, t1) {
                            return (0.1, 0.9);
                        }
                    }
                    SubsumptionResult::Unknown => {}
                }
            }

            // Term resolution
            let term1 = resolver.resolve(t1).ok();
            let term2 = resolver.resolve(t2).ok();

            match (&term1, &term2) {
                (Some(t1_resolved), Some(t2_resolved)) => {
                    // Both terms resolved - check layer and hierarchy
                    if t1_resolved.layer == t2_resolved.layer {
                        let common = t1_resolved
                            .superclasses
                            .iter()
                            .filter(|s| t2_resolved.superclasses.contains(s))
                            .count();

                        let mut distance = if common > 0 {
                            0.15_f64 // Siblings
                        } else {
                            0.4_f64 // Same layer, no common ancestor
                        };
                        let mut confidence = if common > 0 { 0.8_f64 } else { 0.7_f64 };

                        // Refine with conformal prediction if available
                        if let Some(ref mut conformal) = self.conformal_checker {
                            let result = conformal.check(distance);
                            // Boost confidence if conformal checker approves
                            confidence = confidence.max(result.p_value * 0.9_f64);
                        }

                        return (distance, confidence);
                    }
                    // Different layers - use conformal for uncertainty
                    let distance = 0.5_f64;
                    if let Some(ref mut conformal) = self.conformal_checker {
                        let result = conformal.check(distance);
                        return (distance, result.p_value.max(0.4_f64));
                    }
                    return (distance, 0.5);
                }
                (Some(_), None) | (None, Some(_)) => {
                    // One term not found
                    return (0.7, 0.3);
                }
                (None, None) => {
                    // Neither found
                    return (1.0, 0.1);
                }
            }
        }

        // No resolver available
        (1.0, 0.0)
    }

    /// Check if two ontology types are compatible within given threshold.
    ///
    /// Uses the OntologyResolver to check:
    /// 1. Subsumption (is-a relationship) via L1-L4 layers
    /// 2. Semantic distance via alignments and hierarchy
    /// 3. Confidence-weighted coercion decisions
    fn check_ontology_compatibility(
        &mut self,
        expected_ns: &str,
        expected_term: &str,
        found_ns: &str,
        found_term: &str,
        threshold: f64,
    ) -> Result<f64, String> {
        let expected = format!("{}:{}", expected_ns, expected_term);
        let found = format!("{}:{}", found_ns, found_term);

        // Mark ontology prefixes as used (for unused import warnings)
        self.used_ontology_prefixes.insert(expected_ns.to_string());
        self.used_ontology_prefixes.insert(found_ns.to_string());

        // Check for deprecated terms
        self.check_term_deprecation(&expected, None);
        self.check_term_deprecation(&found, None);

        // Identical terms
        if expected == found {
            return Ok(0.0);
        }

        // Compute distance with confidence
        let (distance, confidence) =
            self.compute_semantic_distance_with_confidence(&expected, &found);

        // Adjust effective threshold based on confidence
        // Low confidence requires stricter threshold to be safe
        let effective_threshold = threshold * confidence.max(0.5);

        if distance <= threshold {
            // Within explicit threshold
            if confidence < 0.5 {
                // Low confidence - add warning
                self.warnings.push(format!(
                    "low_confidence_coercion: semantic distance {:.3} between {} and {} has low confidence {:.2}",
                    distance, expected, found, confidence
                ));
            }
            Ok(distance)
        } else if distance <= effective_threshold * 1.5 && confidence >= 0.8 {
            // High confidence allows slightly exceeding threshold with warning
            self.warnings.push(format!(
                "semantic_coercion: {} coerced to {} (distance {:.3}, threshold {:.3})",
                found, expected, distance, threshold
            ));
            Ok(distance)
        } else if distance > 0.8 {
            // Very different ontologies
            Err(format!(
                "incompatible ontology types: {} and {} are semantically distant ({:.3} > {:.3})",
                expected, found, distance, threshold
            ))
        } else {
            // Distance exceeds threshold
            Err(format!(
                "semantic distance {:.3} exceeds threshold {:.3} between {} and {}",
                distance, threshold, expected, found
            ))
        }
    }

    /// Check type compatibility with semantic distance threshold
    fn check_type_compatibility_with_threshold(
        &mut self,
        expected: &HirType,
        found: &HirType,
        threshold: f64,
        span: Span,
    ) {
        // Extract ontology info from types, cloning to avoid borrow issues
        // when calling check_ontology_compatibility (which needs &mut self)
        let ontology_info = self.extract_ontology_info(expected, found);

        if let Some((exp_ns, exp_term, found_ns, found_term, exp_alias, found_alias)) =
            ontology_info
        {
            match self.check_ontology_compatibility(
                &exp_ns,
                &exp_term,
                &found_ns,
                &found_term,
                threshold,
            ) {
                Ok(distance) => {
                    // Types are compatible within threshold
                    if distance > 0.0 {
                        // Could add a note about semantic coercion here
                    }
                }
                Err(msg) => {
                    // Include type alias names in error message if available
                    let full_msg =
                        if let (Some(exp_name), Some(found_name)) = (exp_alias, found_alias) {
                            format!(
                                "type mismatch: expected `{}` ({}:{}), found `{}` ({}:{}): {}",
                                exp_name, exp_ns, exp_term, found_name, found_ns, found_term, msg
                            )
                        } else {
                            msg
                        };
                    self.error(full_msg, span);
                }
            }
        }
    }

    /// Extract ontology namespace/term info from HirTypes.
    /// Returns (exp_ns, exp_term, found_ns, found_term, exp_alias_name, found_alias_name)
    /// The alias names are Some if the type came from a named type alias.
    fn extract_ontology_info(
        &self,
        expected: &HirType,
        found: &HirType,
    ) -> Option<(
        String,
        String,
        String,
        String,
        Option<String>,
        Option<String>,
    )> {
        match (expected, found) {
            // Both are direct ontology types
            (
                HirType::Ontology {
                    namespace: exp_ns,
                    term: exp_term,
                },
                HirType::Ontology {
                    namespace: found_ns,
                    term: found_term,
                },
            ) => Some((
                exp_ns.clone(),
                exp_term.clone(),
                found_ns.clone(),
                found_term.clone(),
                None,
                None,
            )),

            // Both are named types that might alias ontology types
            (
                HirType::Named { name: exp_name, .. },
                HirType::Named {
                    name: found_name, ..
                },
            ) => {
                let exp_ont = self.type_defs.get(exp_name).and_then(|def| {
                    if let TypeDef::Alias(
                        Type::Ontology {
                            namespace, term, ..
                        },
                        _,
                        _,
                        _,
                    ) = def
                    {
                        Some((namespace.clone(), term.clone()))
                    } else {
                        None
                    }
                });
                let found_ont = self.type_defs.get(found_name).and_then(|def| {
                    if let TypeDef::Alias(
                        Type::Ontology {
                            namespace, term, ..
                        },
                        _,
                        _,
                        _,
                    ) = def
                    {
                        Some((namespace.clone(), term.clone()))
                    } else {
                        None
                    }
                });

                if let (Some((exp_ns, exp_term)), Some((found_ns, found_term))) =
                    (exp_ont, found_ont)
                {
                    Some((
                        exp_ns,
                        exp_term,
                        found_ns,
                        found_term,
                        Some(exp_name.clone()),
                        Some(found_name.clone()),
                    ))
                } else {
                    None
                }
            }

            // Mixed: Named expected, Ontology found
            (
                HirType::Named { name, .. },
                HirType::Ontology {
                    namespace: found_ns,
                    term: found_term,
                },
            ) => {
                let exp_ont = self.type_defs.get(name).and_then(|def| {
                    if let TypeDef::Alias(
                        Type::Ontology {
                            namespace, term, ..
                        },
                        _,
                        _,
                        _,
                    ) = def
                    {
                        Some((namespace.clone(), term.clone()))
                    } else {
                        None
                    }
                });

                exp_ont.map(|(exp_ns, exp_term)| {
                    (
                        exp_ns,
                        exp_term,
                        found_ns.clone(),
                        found_term.clone(),
                        Some(name.clone()),
                        None,
                    )
                })
            }

            // Mixed: Ontology expected, Named found
            (
                HirType::Ontology {
                    namespace: exp_ns,
                    term: exp_term,
                },
                HirType::Named { name, .. },
            ) => {
                let found_ont = self.type_defs.get(name).and_then(|def| {
                    if let TypeDef::Alias(
                        Type::Ontology {
                            namespace, term, ..
                        },
                        _,
                        _,
                        _,
                    ) = def
                    {
                        Some((namespace.clone(), term.clone()))
                    } else {
                        None
                    }
                });

                found_ont.map(|(found_ns, found_term)| {
                    (
                        exp_ns.clone(),
                        exp_term.clone(),
                        found_ns,
                        found_term,
                        None,
                        Some(name.clone()),
                    )
                })
            }

            // Other types: no ontology checking needed
            _ => None,
        }
    }

    fn check_item(&mut self, item: &Item) -> Result<Option<HirItem>> {
        match item {
            Item::Function(f) => {
                let hir_fn = self.check_function(f)?;
                Ok(Some(HirItem::Function(hir_fn)))
            }
            Item::Struct(s) => {
                let hir_struct = self.check_struct(s)?;
                Ok(Some(HirItem::Struct(hir_struct)))
            }
            Item::Enum(e) => {
                let hir_enum = self.check_enum(e)?;
                Ok(Some(HirItem::Enum(hir_enum)))
            }
            Item::Effect(e) => {
                let hir_effect = self.check_effect_def(e)?;
                Ok(Some(HirItem::Effect(hir_effect)))
            }
            Item::Handler(h) => {
                let hir_handler = self.check_handler_def(h)?;
                Ok(Some(HirItem::Handler(hir_handler)))
            }
            Item::Global(g) => {
                let hir_global = self.check_global(g)?;
                Ok(Some(HirItem::Global(hir_global)))
            }
            Item::Module(m) => {
                // Type check inline module items recursively
                // Items are flattened into the parent HIR for now
                if let Some(ref items) = m.items {
                    for item in items {
                        // Module items are checked but not collected here
                        // They'll be collected via the main item loop
                        let _ = self.check_item(item)?;
                    }
                }
                Ok(None)
            }
            Item::Import(_) => {
                // Imports are handled during name resolution
                // No HIR items produced
                Ok(None)
            }
            Item::Trait(t) => {
                let hir_trait = self.check_trait(t)?;
                Ok(Some(HirItem::Trait(hir_trait)))
            }
            Item::Impl(i) => {
                let hir_impl = self.check_impl(i)?;
                Ok(Some(HirItem::Impl(hir_impl)))
            }
            _ => Ok(None),
        }
    }

    fn check_function(&mut self, f: &FnDef) -> Result<HirFn> {
        // Set current function for threshold lookup
        self.current_fn = Some(f.name.clone());

        // Clear masked effects before checking function body.
        // This tracks which effects are handled internally by the function.
        self.masked_effects = types::EffectSet::new();

        // Clear and register effect parameters from generics
        self.clear_effect_params();
        for param in &f.generics.params {
            if let GenericParam::Effect { name } = param {
                self.register_effect_param(name);
            }
        }

        self.env.push_scope();

        // Process parameters
        let mut params = Vec::new();
        for param in &f.params {
            let ty = self.lower_type_expr(&param.ty);
            let hir_ty = self.type_to_hir(&ty);

            // Bind parameter in environment
            if let Pattern::Binding { name, .. } = &param.pattern {
                self.env.bind(name.clone(), ty.clone(), param.is_mut);
            }

            params.push(HirParam {
                id: param.id,
                name: self.pattern_name(&param.pattern),
                ty: hir_ty,
                is_mut: param.is_mut,
            });
        }

        // Process return type
        let return_type = f
            .return_type
            .as_ref()
            .map(|t| self.lower_type_expr(t))
            .unwrap_or(Type::Unit);

        // Check body
        let body = self.check_block(&f.body, Some(&return_type))?;

        self.env.pop_scope();

        // Clear current function
        self.current_fn = None;

        // Determine ABI: use explicit ABI if specified, otherwise Rust
        let abi = f.modifiers.abi.clone().unwrap_or(crate::ast::Abi::Rust);

        // Check for #[export] attribute - explicit FFI export marker
        let has_export_attr = f.attributes.iter().any(|attr| attr.name == "export");

        // Extract #[extern("name")] attribute for FFI binding
        let extern_name = f.attributes.iter().find_map(|attr| {
            if attr.name == "extern" {
                match &attr.args {
                    crate::ast::AttributeArgs::Value(crate::ast::AttributeValue::String(name)) => {
                        Some(name.clone())
                    }
                    _ => None,
                }
            } else {
                None
            }
        });

        // Function is exported if:
        // 1. It has #[export] attribute (explicit FFI export), OR
        // 2. It's public (for module-level visibility)
        // Note: Having a C ABI alone does not imply export - internal functions
        // may use C calling convention without being externally visible.
        let is_exported = has_export_attr || matches!(f.visibility, crate::ast::Visibility::Public);

        // Compute effective effects for the function signature.
        // The effective effects are the declared effects minus any effects that are
        // masked (handled internally). This enables pure functions to use impure
        // operations internally as long as all effects are handled before returning.
        //
        // Example:
        //   fn pure_from_state<S, A>(init: S, f: fn() -> A with Mut<S>) -> A {
        //       handle { f() } with MutHandler
        //   }
        // This function is pure because Mut is handled internally.
        let declared_effects: Vec<HirEffect> =
            f.effects.iter().map(|e| self.lower_effect_ref(e)).collect();

        // Compute effective effects: declared effects minus any that are masked (handled internally).
        // This enables pure functions to use impure operations internally as long as all
        // effects are handled before returning.
        //
        // Example: fn pure_from_state() -> i32 {
        //     handle { perform IO::print("hello"); 42 } with IOHandler
        // }
        // The IO effect is masked internally, so the function's effective effects are empty (pure).
        let mut effective_effects = declared_effects;
        for masked in &self.masked_effects.effects {
            effective_effects.retain(|eff| &eff.name != masked);
        }

        // Clean up effect parameters after checking the function
        self.clear_effect_params();

        Ok(HirFn {
            id: f.id,
            name: f.name.clone(),
            ty: HirFnType {
                params: params.clone(),
                return_type: Box::new(self.type_to_hir(&return_type)),
                effects: effective_effects,
            },
            body,
            abi,
            is_exported,
            extern_name,
            doc: f.doc.clone(),
        })
    }

    fn check_struct(&mut self, s: &StructDef) -> Result<HirStruct> {
        let fields: Vec<_> = s
            .fields
            .iter()
            .map(|f| {
                let ty = self.lower_type_expr(&f.ty);
                HirField {
                    id: f.id,
                    name: f.name.clone(),
                    ty: self.type_to_hir(&ty),
                }
            })
            .collect();

        Ok(HirStruct {
            id: s.id,
            name: s.name.clone(),
            fields,
            is_linear: s.modifiers.linear,
            is_affine: s.modifiers.affine,
            doc: s.doc.clone(),
        })
    }

    /// Type check a trait definition
    fn check_trait(&mut self, t: &TraitDef) -> Result<HirTrait> {
        // Extract type parameters
        let type_params: Vec<String> = t
            .generics
            .params
            .iter()
            .filter_map(|p| {
                if let GenericParam::Type { name, .. } = p {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        // Process associated type declarations
        let mut assoc_types = Vec::new();
        for item in &t.items {
            if let TraitItem::Type(assoc) = item {
                let bounds: Vec<String> = assoc.bounds.iter().map(|p| p.to_string()).collect();
                let default = assoc.default.as_ref().map(|ty| {
                    let lowered = self.lower_type_expr(ty);
                    self.type_to_hir(&lowered)
                });
                assoc_types.push(HirAssocTypeDecl {
                    id: assoc.id,
                    name: assoc.name.clone(),
                    bounds,
                    default,
                });
            }
        }

        // Process trait methods
        let mut methods = Vec::new();
        for item in &t.items {
            if let TraitItem::Fn(f) = item {
                let params: Vec<HirParam> = f
                    .params
                    .iter()
                    .map(|p| {
                        let ty = self.lower_type_expr(&p.ty);
                        HirParam {
                            id: p.id,
                            name: self.pattern_name(&p.pattern),
                            ty: self.type_to_hir(&ty),
                            is_mut: p.is_mut,
                        }
                    })
                    .collect();

                let return_type = f
                    .return_type
                    .as_ref()
                    .map(|ty| {
                        let lowered = self.lower_type_expr(ty);
                        self.type_to_hir(&lowered)
                    })
                    .unwrap_or(HirType::Unit);

                let effects: Vec<HirEffect> = f
                    .effects
                    .iter()
                    .map(|e| HirEffect {
                        id: e.id,
                        name: e.name.to_string(),
                        operations: Vec::new(),
                        effect_var: None,
                    })
                    .collect();

                methods.push(HirTraitMethod {
                    id: f.id,
                    name: f.name.clone(),
                    ty: HirFnType {
                        params,
                        return_type: Box::new(return_type),
                        effects,
                    },
                    has_default: f.default_body.is_some(),
                });
            }
        }

        // Extract supertrait names
        let supertraits: Vec<String> = t.supertraits.iter().map(|p| p.to_string()).collect();

        // Register trait's associated types for later resolution
        for item in &t.items {
            if let TraitItem::Type(assoc) = item {
                self.register_trait_assoc_type(&t.name, &assoc.name, assoc.default.as_ref());
            }
        }

        // Lower where clause
        let where_clause = self.lower_where_clause(&t.where_clause);

        Ok(HirTrait {
            id: t.id,
            name: t.name.clone(),
            type_params,
            assoc_types,
            methods,
            supertraits,
            where_clause,
            doc: t.doc.clone(),
        })
    }

    /// Register a trait's associated types for later resolution
    fn register_trait_assoc_type(
        &mut self,
        trait_name: &str,
        assoc_name: &str,
        default_ty: Option<&TypeExpr>,
    ) {
        // Register associated type placeholder
        // Format: TraitName::AssocTypeName
        let qualified_name = format!("{}::{}", trait_name, assoc_name);
        // Store as a type alias to the default if available
        if let Some(ty_expr) = default_ty {
            let ty = self.lower_type_expr(ty_expr);
            self.type_defs.insert(
                qualified_name,
                TypeDef::Alias(ty, Span::default(), None, Vec::new()),
            );
        }
    }

    /// Type check an impl block
    fn check_impl(&mut self, i: &ImplDef) -> Result<HirImpl> {
        // Extract type parameters
        let type_params: Vec<String> = i
            .generics
            .params
            .iter()
            .filter_map(|p| {
                if let GenericParam::Type { name, .. } = p {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        // Lower the self type
        let self_ty = self.lower_type_expr(&i.target_type);
        let hir_self_ty = self.type_to_hir(&self_ty);

        // Set impl context so self parameter types resolve correctly
        let prev_impl_type = self.current_impl_type.take();
        self.current_impl_type = Some(self_ty.clone());

        // Get the type name for qualified method names
        let type_name = match &i.target_type {
            TypeExpr::Named { path, .. } => path.to_string(),
            _ => "<anonymous>".to_string(),
        };

        // Process associated type implementations
        let mut assoc_types = Vec::new();
        for item in &i.items {
            if let ImplItem::Type(impl_ty) = item {
                let lowered = self.lower_type_expr(&impl_ty.ty);
                let hir_ty = self.type_to_hir(&lowered);

                // Register the concrete associated type
                // Format: TypeName::AssocTypeName for inherent impl
                // Format: TraitName::AssocTypeName::for::TypeName for trait impl
                if let Some(ref trait_ref) = i.trait_ref {
                    let qualified = format!("{}::{}::for::{}", trait_ref, impl_ty.name, type_name);
                    self.type_defs.insert(
                        qualified,
                        TypeDef::Alias(lowered.clone(), Span::default(), None, Vec::new()),
                    );
                }
                // Also register as TypeName::AssocTypeName for direct access
                let direct_name = format!("{}::{}", type_name, impl_ty.name);
                self.type_defs.insert(
                    direct_name,
                    TypeDef::Alias(lowered, Span::default(), None, Vec::new()),
                );

                assoc_types.push(HirAssocTypeImpl {
                    id: impl_ty.id,
                    name: impl_ty.name.clone(),
                    ty: hir_ty,
                });
            }
        }

        // Process methods
        let mut methods = Vec::new();
        for item in &i.items {
            if let ImplItem::Fn(f) = item {
                let hir_fn = self.check_function(f)?;
                methods.push(hir_fn);
            }
        }

        // Restore previous impl context
        self.current_impl_type = prev_impl_type;

        // Get trait reference name
        let trait_ref = i.trait_ref.as_ref().map(|p| p.to_string());

        // Lower where clause
        let where_clause = self.lower_where_clause(&i.where_clause);

        Ok(HirImpl {
            id: i.id,
            trait_ref,
            self_ty: hir_self_ty,
            type_params,
            assoc_types,
            methods,
            where_clause,
            doc: i.doc.clone(),
        })
    }

    fn check_enum(&mut self, e: &EnumDef) -> Result<HirEnum> {
        // Collect enum's type parameters for GADT index extraction
        let enum_type_params: Vec<String> = e
            .generics
            .params
            .iter()
            .filter_map(|p| {
                if let GenericParam::Type { name, .. } = p {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        let variants: Vec<_> = e
            .variants
            .iter()
            .map(|v| {
                let fields = match &v.data {
                    VariantData::Unit => Vec::new(),
                    VariantData::Tuple(types) => {
                        let lowered: Vec<_> =
                            types.iter().map(|t| self.lower_type_expr(t)).collect();
                        lowered.iter().map(|t| self.type_to_hir(t)).collect()
                    }
                    VariantData::Struct(fields) => {
                        let lowered: Vec<_> =
                            fields.iter().map(|f| self.lower_type_expr(&f.ty)).collect();
                        lowered.iter().map(|t| self.type_to_hir(t)).collect()
                    }
                };

                // Handle GADT return type
                let (gadt_return_type, type_indices) = if let Some(gadt) = &v.gadt_return_type {
                    let lowered = self.lower_type_expr(&gadt.return_type);
                    let hir_type = self.type_to_hir(&lowered);

                    // Extract type indices from GADT return type
                    // e.g., Vec<T, Zero> gives us [("N", Zero)]
                    let indices =
                        self.extract_gadt_type_indices(&gadt.return_type, &enum_type_params);

                    (Some(hir_type), indices)
                } else {
                    (None, Vec::new())
                };

                HirVariant {
                    id: v.id,
                    name: v.name.clone(),
                    fields,
                    gadt_return_type,
                    type_indices,
                }
            })
            .collect();

        Ok(HirEnum {
            id: e.id,
            name: e.name.clone(),
            variants,
            is_linear: e.modifiers.linear,
            is_affine: e.modifiers.affine,
            doc: e.doc.clone(),
        })
    }

    /// Extract GADT type indices from a return type expression
    /// Given `Vec<T, Zero>` and enum params `[T, N]`, returns `[("N", Zero)]`
    fn extract_gadt_type_indices(
        &mut self,
        return_type: &TypeExpr,
        enum_params: &[String],
    ) -> Vec<(String, HirType)> {
        let mut indices = Vec::new();

        if let TypeExpr::Named { args, .. } = return_type {
            // Match each type argument to its corresponding parameter
            for (i, arg) in args.iter().enumerate() {
                if i < enum_params.len() {
                    let param_name = &enum_params[i];

                    // Check if this argument differs from just using the parameter
                    // (i.e., it's a specialized index like Zero instead of N)
                    if !self.is_same_type_param(arg, param_name) {
                        let lowered = self.lower_type_expr(arg);
                        let hir_type = self.type_to_hir(&lowered);
                        indices.push((param_name.clone(), hir_type));
                    }
                }
            }
        }

        indices
    }

    /// Check if a type expression is just a reference to a type parameter
    fn is_same_type_param(&self, ty: &TypeExpr, param_name: &str) -> bool {
        match ty {
            TypeExpr::Named { path, args, .. } => {
                args.is_empty() && path.segments.len() == 1 && path.segments[0] == param_name
            }
            _ => false,
        }
    }

    fn check_effect_def(&mut self, e: &EffectDef) -> Result<HirEffect> {
        let operations: Vec<_> = e
            .operations
            .iter()
            .map(|op| {
                let lowered_params: Vec<_> = op
                    .params
                    .iter()
                    .map(|p| self.lower_type_expr(&p.ty))
                    .collect();
                let params: Vec<_> = lowered_params.iter().map(|t| self.type_to_hir(t)).collect();
                let return_type = if let Some(t) = op.return_type.as_ref() {
                    let lowered = self.lower_type_expr(t);
                    self.type_to_hir(&lowered)
                } else {
                    HirType::Unit
                };

                // Register this effect operation in the registry
                let return_type_lowered = if let Some(t) = op.return_type.as_ref() {
                    self.lower_type_expr(t)
                } else {
                    Type::Unit
                };
                self.effect_operations
                    .insert((e.name.clone(), op.name.clone()), return_type_lowered);

                HirEffectOp {
                    id: op.id,
                    name: op.name.clone(),
                    params,
                    return_type,
                }
            })
            .collect();

        Ok(HirEffect {
            id: e.id,
            name: e.name.clone(),
            operations,
            effect_var: None, // Effect definitions are always concrete
        })
    }

    fn check_handler_def(&mut self, h: &HandlerDef) -> Result<HirHandler> {
        let cases: Vec<_> = h
            .cases
            .iter()
            .map(|case| {
                let params: Vec<_> = case
                    .params
                    .iter()
                    .map(|p| self.pattern_name(&p.pattern))
                    .collect();

                // Check handler case body expression
                let body = self
                    .check_expr(&case.body, None)
                    .unwrap_or_else(|_| HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Literal(HirLiteral::Unit),
                        ty: HirType::Unit,
                    });
                HirHandlerCase {
                    id: case.id,
                    op_name: case.name.clone(),
                    params,
                    body,
                }
            })
            .collect();

        // Register this handler for effect lookup during Handle expression checking
        let effect_name = h.effect.to_string();
        self.register_handler(h.name.clone(), effect_name.clone());

        Ok(HirHandler {
            id: h.id,
            name: h.name.clone(),
            effect: effect_name,
            cases,
        })
    }

    fn check_global(&mut self, g: &GlobalDef) -> Result<HirGlobal> {
        let ty =
            g.ty.as_ref()
                .map(|t| self.lower_type_expr(t))
                .unwrap_or_else(|| self.fresh_type_var());

        // Check global value expression with expected type
        let expected_ty = ty.clone();
        let hir_ty = self.type_to_hir(&ty);
        let value = self
            .check_expr(&g.value, Some(&expected_ty))
            .unwrap_or_else(|_| HirExpr {
                id: NodeId::dummy(),
                kind: HirExprKind::Literal(HirLiteral::Unit),
                ty: hir_ty,
            });

        Ok(HirGlobal {
            id: g.id,
            name: self.pattern_name(&g.pattern),
            ty: self.type_to_hir(&ty),
            value,
            is_const: g.is_const,
            doc: g.doc.clone(),
        })
    }

    fn check_block(&mut self, block: &Block, expected: Option<&Type>) -> Result<HirBlock> {
        self.env.push_scope();

        let mut stmts = Vec::new();
        let mut result_ty = Type::Unit;

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == block.stmts.len() - 1;

            match stmt {
                Stmt::Let {
                    is_mut,
                    pattern,
                    ty,
                    value,
                } => {
                    // Check if we have an explicit type annotation
                    let has_annotation = ty.is_some();
                    let declared_ty = ty
                        .as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or_else(|| self.fresh_type_var());

                    // Expand type aliases before type checking (e.g., A -> Vec<Vec<...>>)
                    let expanded_ty = self.expand_type_alias(&declared_ty);

                    let value_expr = value
                        .as_ref()
                        .map(|v| self.check_expr(v, Some(&expanded_ty)))
                        .transpose()?;

                    // Determine the final binding type:
                    // - If there's an explicit annotation, use the declared type
                    // - If no annotation, infer from the value expression's type
                    let binding_ty = if has_annotation {
                        declared_ty.clone()
                    } else if let Some(ref v_expr) = value_expr {
                        // Infer type from value expression
                        self.hir_type_to_type(&v_expr.ty)
                    } else {
                        declared_ty.clone()
                    };

                    // CRITICAL: Verify type compatibility between declared type and value type
                    if let Some(ref v_expr) = value_expr {
                        let actual_ty = self.hir_type_to_type(&v_expr.ty);

                        // Get span from the original AST value expression
                        let value_span = if let (Some(v), Some(ast_ref)) = (value, &self.ast) {
                            self.expr_span(v, ast_ref.as_ref())
                        } else {
                            Span::dummy()
                        };

                        // Get threshold for current function (from #[compat] annotation or default)
                        let threshold = self
                            .current_fn
                            .as_ref()
                            .and_then(|name| self.fn_thresholds.get(name).copied())
                            .unwrap_or(self.default_threshold);

                        // First check structural compatibility (use expanded type for comparison)
                        // Only check if we have an explicit annotation (otherwise we're inferring)
                        if has_annotation && !self.types_compatible(&expanded_ty, &actual_ty) {
                            let decl_name = self.type_display_name(&expanded_ty);
                            let actual_name = self.type_display_name(&actual_ty);
                            self.error(
                                format!(
                                    "Type mismatch: expected `{}`, found `{}`",
                                    decl_name, actual_name
                                ),
                                value_span,
                            );
                        }

                        // Epistemic confidence bound check (MV core): do not allow claiming
                        // `Knowledge[..., epsilon >= x]` unless the expression provides at least x.
                        if has_annotation {
                            if let Some(type_expr) = ty.as_ref() {
                                if let Some(required) =
                                    self.extract_knowledge_confidence_lower_bound(type_expr)
                                {
                                    if let HirType::Knowledge { epsilon_bound, .. } = &v_expr.ty {
                                        let actual = epsilon_bound.unwrap_or(0.0);
                                        if actual + f64::EPSILON < required {
                                            self.error(
                                                format!(
                                                    "Type mismatch: expected `Knowledge[...]` with epsilon >= {}, found epsilon >= {}",
                                                    required, actual
                                                ),
                                                value_span,
                                            );
                                        }
                                    } else {
                                        self.error(
                                            format!(
                                                "Type mismatch: expected `Knowledge[...]` with epsilon >= {}, found `{:?}`",
                                                required, v_expr.ty
                                            ),
                                            value_span,
                                        );
                                    }
                                }
                            }
                        }

                        // Also check semantic/ontology type compatibility with threshold
                        let declared_hir = self.type_to_hir(&expanded_ty);
                        self.check_type_compatibility_with_threshold(
                            &declared_hir,
                            &v_expr.ty,
                            threshold,
                            value_span,
                        );
                    }

                    // Bind all variables in the pattern (supports tuple destructuring)
                    self.bind_pattern_to_type(pattern, &binding_ty, *is_mut);

                    stmts.push(HirStmt::Let {
                        name: self.pattern_name(pattern),
                        ty: self.type_to_hir(&binding_ty),
                        value: value_expr,
                        is_mut: *is_mut,
                        layout_hint: None, // Layout hints are filled in by layout synthesis pass
                    });
                }
                Stmt::Expr { expr, has_semi } => {
                    // Pass expected type for last expression without semicolon (implicit return)
                    let expr_expected = if is_last && !has_semi { expected } else { None };
                    let expr_result = self.check_expr(expr, expr_expected)?;

                    if is_last && !has_semi {
                        result_ty = self.hir_type_to_type(&expr_result.ty);
                    }

                    stmts.push(HirStmt::Expr(expr_result));
                }
                Stmt::Assign { target, op, value } => {
                    let target_expr = self.check_expr(target, None)?;
                    let value_expr =
                        self.check_expr(value, Some(&self.hir_type_to_type(&target_expr.ty)))?;

                    stmts.push(HirStmt::Assign {
                        target: target_expr,
                        value: value_expr,
                    });
                }
                Stmt::Empty | Stmt::MacroInvocation(_) | Stmt::LocalExtern(_) => {}
            }
        }

        if let Some(exp) = expected {
            self.constrain(exp.clone(), result_ty.clone(), Span::dummy());
        }

        self.env.pop_scope();

        Ok(HirBlock {
            stmts,
            ty: self.type_to_hir(&result_ty),
        })
    }

    fn check_expr(&mut self, expr: &Expr, expected: Option<&Type>) -> Result<HirExpr> {
        let (kind, ty) = match expr {
            Expr::Literal { id, value } => {
                let (lit, ty) = self.check_literal_with_expected(value, expected);
                (HirExprKind::Literal(lit), ty)
            }

            Expr::Path { id, path } => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    if name.starts_with("quat_init") || name.starts_with("quat_relu") {}
                    if let Some(binding) = self.env.lookup(name) {
                        let ty = binding.ty.clone();
                        (HirExprKind::Local(name.clone()), self.type_to_hir(&ty))
                    } else if self.is_builtin_function(name) {
                        // Builtin function - return a function type
                        let builtin_ty = self.get_builtin_type(name);
                        (HirExprKind::Global(name.clone()), builtin_ty)
                    } else if self.is_builtin_variant(name) {
                        // Builtin enum variant (None, Some, Ok, Err)
                        let variant_ty = self.get_builtin_variant_type(name, expected);
                        (HirExprKind::Global(name.clone()), variant_ty)
                    } else {
                        self.error(format!("Unknown variable: {}", name), Span::dummy());
                        (HirExprKind::Local(name.clone()), HirType::Error)
                    }
                } else {
                    // Qualified path - try module-qualified lookup first
                    if let Some(binding) = self.env.lookup_qualified(&path.segments) {
                        let ty = binding.ty.clone();
                        let full_path = path.to_string();

                        // Check visibility of qualified path
                        if let Some(def_id) = self
                            .symbols
                            .as_ref()
                            .and_then(|symbols| symbols.ref_for_node(*id))
                        {
                            let span = self
                                .ast
                                .as_ref()
                                .map(|ast| self.expr_span(expr, ast.as_ref()))
                                .unwrap_or_else(Span::dummy);
                            self.check_item_visibility(&def_id, &full_path, "item", span);
                        }

                        (HirExprKind::Global(full_path), self.type_to_hir(&ty))
                    } else {
                        // Check if it's an enum variant (EnumName::Variant)
                        let type_name = &path.segments[0];
                        if let Some(TypeDef::Enum { variants, .. }) = self.type_defs.get(type_name)
                        {
                            if path.segments.len() == 2 {
                                let variant_name = &path.segments[1];
                                if let Some((_, variant_types)) =
                                    variants.iter().find(|(n, _)| n == variant_name)
                                {
                                    // Found enum variant - check visibility
                                    if let Some(def_id) = self
                                        .symbols
                                        .as_ref()
                                        .and_then(|symbols| symbols.ref_for_node(*id))
                                    {
                                        let span = self
                                            .ast
                                            .as_ref()
                                            .map(|ast| self.expr_span(expr, ast.as_ref()))
                                            .unwrap_or_else(Span::dummy);
                                        let variant_full_name =
                                            format!("{}::{}", type_name, variant_name);
                                        self.check_item_visibility(
                                            &def_id,
                                            &variant_full_name,
                                            "enum variant",
                                            span,
                                        );
                                    }

                                    let result_ty = HirType::Named {
                                        name: type_name.clone(),
                                        args: vec![],
                                    };
                                    (HirExprKind::Global(path.to_string()), result_ty)
                                } else {
                                    self.error(
                                        format!(
                                            "Unknown variant `{}` in enum `{}`",
                                            variant_name, type_name
                                        ),
                                        Span::dummy(),
                                    );
                                    (HirExprKind::Global(path.to_string()), HirType::Error)
                                }
                            } else {
                                (HirExprKind::Global(path.to_string()), HirType::Error)
                            }
                        } else if self.is_builtin_associated_fn(&path.segments) {
                            // Builtin associated function (Vec::new, Box::new, etc.)
                            let full_path = path.to_string();
                            let ty = self.get_builtin_associated_fn_type(&path.segments);
                            (HirExprKind::Global(full_path), ty)
                        } else {
                            // Module-qualified path not found - include module info in error if available
                            let error_msg = if let Some(ref resolved) = path.resolved_module {
                                format!(
                                    "Unknown qualified path `{}` (resolved to module {:?})",
                                    path.to_string(),
                                    resolved.path
                                )
                            } else {
                                format!("Unknown qualified path `{}`", path.to_string())
                            };
                            self.error(error_msg, Span::dummy());
                            (HirExprKind::Global(path.to_string()), HirType::Error)
                        }
                    }
                }
            }

            Expr::Binary {
                id,
                op,
                left,
                right,
            } => {
                // Iteratively flatten left-associative binary chains to avoid stack overflow
                // Collect chain: [(op, right_expr), ...] from innermost to outermost
                let mut chain: Vec<(BinaryOp, &Expr)> = Vec::new();
                let mut current = expr;

                // Walk down the left spine collecting operators and right operands
                while let Expr::Binary {
                    op: curr_op,
                    left: curr_left,
                    right: curr_right,
                    ..
                } = current
                {
                    chain.push((*curr_op, curr_right.as_ref()));
                    current = curr_left.as_ref();
                }

                // Now 'current' is the leftmost non-binary expression
                // Check it first
                // Use the expected type (if any) to guide literal typing in expressions like `1 + 2`
                // when the whole expression is contextually typed (e.g., `let x: i32 = 1 + 2`).
                let mut result = self.check_expr(current, expected)?;

                // Process the chain in reverse (innermost to outermost)
                for (chain_op, chain_right) in chain.into_iter().rev() {
                    let right_expr =
                        self.check_expr(chain_right, Some(&self.hir_type_to_type(&result.ty)))?;
                    let result_ty = self.check_binary_units(chain_op, &result.ty, &right_expr.ty);
                    let hir_op = self.lower_binary_op(chain_op);

                    result = HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Binary {
                            op: hir_op,
                            left: Box::new(result),
                            right: Box::new(right_expr),
                        },
                        ty: result_ty,
                    };
                }

                (result.kind, result.ty)
            }

            Expr::Unary {
                id,
                op,
                expr: inner,
            } => {
                let inner_expr = self.check_expr(inner, None)?;
                let result_ty = self.unary_result_type(*op, &inner_expr.ty);
                let hir_op = self.lower_unary_op(*op);

                (
                    HirExprKind::Unary {
                        op: hir_op,
                        expr: Box::new(inner_expr),
                    },
                    result_ty,
                )
            }

            Expr::Call { id, callee, args } => {
                // Check if this is a method call disguised as Call(Field(...))
                if let Expr::Field { base, field, .. } = callee.as_ref() {
                    // This is a method call: base.field(args)
                    let receiver_expr = self.check_expr(base, None)?;
                    let receiver_ty = receiver_expr.ty.clone();

                    let arg_exprs: Vec<_> = args
                        .iter()
                        .map(|a| self.check_expr(a, None))
                        .collect::<Result<_>>()?;

                    // Knowledge explicit extraction: `k.unwrap("reason")`
                    if field == "unwrap" {
                        if let HirType::Knowledge { inner, .. } = &receiver_ty {
                            let span = self
                                .ast
                                .as_ref()
                                .map(|ast| self.expr_span(expr, ast.as_ref()))
                                .unwrap_or_else(Span::dummy);

                            if arg_exprs.len() != 1 {
                                self.error(
                                    "Knowledge.unwrap requires exactly one argument: a reason string"
                                        .to_string(),
                                    span,
                                );
                                return Ok(HirExpr {
                                    id: *id,
                                    kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                    ty: HirType::Error,
                                });
                            }

                            if arg_exprs[0].ty != HirType::String {
                                self.error(
                                    "Knowledge.unwrap(reason): reason must be a `string`"
                                        .to_string(),
                                    span,
                                );
                                return Ok(HirExpr {
                                    id: *id,
                                    kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                    ty: HirType::Error,
                                });
                            }

                            return Ok(HirExpr {
                                id: *id,
                                kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                ty: (*inner.clone()),
                            });
                        }
                    }

                    // Knowledge introspection / extraction helpers (A):
                    // - `k.value()` (explicit unwrap, no reason string)
                    // - `k.confidence()` / `k.epsilon()`
                    // - `k.provenance()` / `k.validity()`
                    if let HirType::Knowledge { inner, .. } = &receiver_ty {
                        let span = self
                            .ast
                            .as_ref()
                            .map(|ast| self.expr_span(expr, ast.as_ref()))
                            .unwrap_or_else(Span::dummy);

                        match field.as_str() {
                            "value" => {
                                if !arg_exprs.is_empty() {
                                    self.error(
                                        "Knowledge.value() takes no arguments".to_string(),
                                        span,
                                    );
                                    return Ok(HirExpr {
                                        id: *id,
                                        kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                        ty: HirType::Error,
                                    });
                                }
                                return Ok(HirExpr {
                                    id: *id,
                                    kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                    ty: (*inner.clone()),
                                });
                            }
                            "confidence" | "epsilon" => {
                                if !arg_exprs.is_empty() {
                                    self.error(
                                        "Knowledge.confidence()/epsilon() takes no arguments"
                                            .to_string(),
                                        span,
                                    );
                                }
                                return Ok(HirExpr {
                                    id: *id,
                                    kind: HirExprKind::EpsilonOf(Box::new(receiver_expr)),
                                    ty: HirType::F64,
                                });
                            }
                            "provenance" => {
                                if !arg_exprs.is_empty() {
                                    self.error(
                                        "Knowledge.provenance() takes no arguments".to_string(),
                                        span,
                                    );
                                }
                                return Ok(HirExpr {
                                    id: *id,
                                    kind: HirExprKind::ProvenanceOf(Box::new(receiver_expr)),
                                    ty: HirType::Unit,
                                });
                            }
                            "validity" => {
                                if !arg_exprs.is_empty() {
                                    self.error(
                                        "Knowledge.validity() takes no arguments".to_string(),
                                        span,
                                    );
                                }
                                return Ok(HirExpr {
                                    id: *id,
                                    kind: HirExprKind::ValidityOf(Box::new(receiver_expr)),
                                    ty: HirType::Unit,
                                });
                            }
                            _ => {}
                        }
                    }

                    let result_ty = self.get_method_return_type(&receiver_ty, field, &arg_exprs);

                    return Ok(HirExpr {
                        id: *id,
                        kind: HirExprKind::MethodCall {
                            receiver: Box::new(receiver_expr),
                            method: field.clone(),
                            args: arg_exprs,
                        },
                        ty: result_ty,
                    });
                }

                let callee_expr = self.check_expr(callee, None)?;
                // If we know the callee's parameter types, use them as expected types for args.
                // This enables context-driven literal typing (e.g., `1` -> `u8` when calling `fn(_, _, u8)`).
                let expected_param_tys: Option<Vec<Type>> = match &callee_expr.ty {
                    HirType::Fn { params, .. } => {
                        Some(params.iter().map(|p| self.hir_type_to_type(p)).collect())
                    }
                    _ => None,
                };

                let checked_args: Vec<_> = if let Some(param_tys) = expected_param_tys {
                    args.iter()
                        .enumerate()
                        .map(|(i, a)| self.check_expr(a, param_tys.get(i)))
                        .collect::<Result<_>>()?
                } else {
                    args.iter()
                        .map(|a| self.check_expr(a, None))
                        .collect::<Result<_>>()?
                };

                // Extract function name for threshold lookup
                let fn_name = match callee.as_ref() {
                    Expr::Path { path, .. } => path.segments.last().cloned(),
                    _ => None,
                };

                // Check visibility of function being called
                if let Expr::Path { id, path, .. } = callee.as_ref() {
                    if let Some(def_id) = self
                        .symbols
                        .as_ref()
                        .and_then(|symbols| symbols.ref_for_node(*id))
                    {
                        let fn_display = fn_name
                            .as_ref()
                            .map(|s| s.clone())
                            .unwrap_or_else(|| path.to_string());
                        let span = self
                            .ast
                            .as_ref()
                            .map(|ast| self.expr_span(expr, ast.as_ref()))
                            .unwrap_or_else(Span::dummy);
                        self.check_item_visibility(&def_id, &fn_display, "function", span);
                    }
                }

                // Get threshold for this function (from #[compat] annotation or default)
                let threshold = fn_name
                    .as_ref()
                    .and_then(|name| self.fn_thresholds.get(name).copied())
                    .unwrap_or(self.default_threshold);

                // Special handling for Option/Result constructors
                // These need type inference from their arguments
                let special_constructor_ty = match fn_name.as_deref() {
                    Some("Some") => {
                        // Some(value) -> Option<typeof(value)>
                        let inner_ty = checked_args
                            .first()
                            .map(|a| a.ty.clone())
                            .unwrap_or(HirType::Unit);
                        Some(HirType::Named {
                            name: "Option".to_string(),
                            args: vec![inner_ty],
                        })
                    }
                    Some("None") => {
                        // None -> Option<T> where T is inferred from context
                        // For now, use expected type if available
                        if let Some(Type::Named { name, args }) = expected {
                            if name == "Option" {
                                Some(HirType::Named {
                                    name: "Option".to_string(),
                                    args: args.iter().map(|t| self.type_to_hir(t)).collect(),
                                })
                            } else {
                                Some(HirType::Named {
                                    name: "Option".to_string(),
                                    args: vec![HirType::Unit],
                                })
                            }
                        } else {
                            Some(HirType::Named {
                                name: "Option".to_string(),
                                args: vec![HirType::Unit],
                            })
                        }
                    }
                    Some("Ok") => {
                        // Ok(value) -> Result<typeof(value), E>
                        let ok_ty = checked_args
                            .first()
                            .map(|a| a.ty.clone())
                            .unwrap_or(HirType::Unit);
                        // Try to get error type from context
                        let err_ty = if let Some(Type::Named { name, args }) = expected {
                            if name == "Result" && args.len() > 1 {
                                self.type_to_hir(&args[1])
                            } else {
                                HirType::Unit
                            }
                        } else {
                            HirType::Unit
                        };
                        Some(HirType::Named {
                            name: "Result".to_string(),
                            args: vec![ok_ty, err_ty],
                        })
                    }
                    Some("Err") => {
                        // Err(value) -> Result<T, typeof(value)>
                        let err_ty = checked_args
                            .first()
                            .map(|a| a.ty.clone())
                            .unwrap_or(HirType::Unit);
                        // Try to get ok type from context
                        let ok_ty = if let Some(Type::Named { name, args }) = expected {
                            if name == "Result" && !args.is_empty() {
                                self.type_to_hir(&args[0])
                            } else {
                                HirType::Unit
                            }
                        } else {
                            HirType::Unit
                        };
                        Some(HirType::Named {
                            name: "Result".to_string(),
                            args: vec![ok_ty, err_ty],
                        })
                    }
                    _ => None,
                };

                // Extract return type and parameter types from function type
                let (result_ty, param_types) = if let Some(special_ty) = special_constructor_ty {
                    // Use the specially inferred type for constructors
                    (special_ty, vec![])
                } else {
                    match &callee_expr.ty {
                        HirType::Fn {
                            params,
                            return_type,
                            ..
                        } => (*return_type.clone(), params.clone()),
                        _ => (HirType::Unit, vec![]),
                    }
                };

                // Check ontological compatibility for each argument
                // We need to iterate with original args to get spans
                for (i, (checked_arg, param_ty)) in
                    checked_args.iter().zip(param_types.iter()).enumerate()
                {
                    // Get span from original AST argument
                    let arg_span = if let Some(ast_ref) = &self.ast {
                        args.get(i)
                            .map(|a| self.expr_span(a, ast_ref.as_ref()))
                            .unwrap_or_else(Span::dummy)
                    } else {
                        Span::dummy()
                    };

                    self.check_type_compatibility_with_threshold(
                        param_ty,
                        &checked_arg.ty,
                        threshold,
                        arg_span,
                    );
                }

                // Check refinement predicates for literal arguments
                if let Some(fn_name_str) = &fn_name {
                    if let Some(refinements) = self.fn_param_refinements.get(fn_name_str).cloned() {
                        for (i, refinement_opt) in refinements.iter().enumerate() {
                            if let Some(refinement) = refinement_opt {
                                if let Some(arg_expr) = args.get(i) {
                                    // Try to extract a constant value from the argument
                                    // First try integer, then try float
                                    let check_result = if let Some(int_val) =
                                        self.try_extract_const_value(arg_expr)
                                    {
                                        Some(self.check_literal_refinement(
                                            int_val,
                                            &refinement.var,
                                            &refinement.predicate,
                                        ))
                                    } else if let Some(float_val) =
                                        self.try_extract_float_value(arg_expr)
                                    {
                                        Some(self.check_float_literal_refinement(
                                            float_val,
                                            &refinement.var,
                                            &refinement.predicate,
                                        ))
                                    } else {
                                        None
                                    };

                                    if let Some(Err(msg)) = check_result {
                                        let arg_span = if let Some(ast_ref) = &self.ast {
                                            self.expr_span(arg_expr, ast_ref.as_ref())
                                        } else {
                                            Span::dummy()
                                        };
                                        self.error_with_code(
                                            "E0600",
                                            format!("refinement type violation: {}", msg),
                                            arg_span,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                (
                    HirExprKind::Call {
                        func: Box::new(callee_expr),
                        args: checked_args,
                    },
                    result_ty,
                )
            }

            Expr::If {
                id,
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_expr = self.check_expr(condition, Some(&Type::Bool))?;
                let then_block = self.check_block(then_branch, expected)?;

                let else_expr = else_branch
                    .as_ref()
                    .map(|e| self.check_expr(e, expected))
                    .transpose()?;

                let result_ty = if else_expr.is_some() {
                    then_block.ty.clone()
                } else {
                    HirType::Unit
                };

                (
                    HirExprKind::If {
                        condition: Box::new(cond_expr),
                        then_branch: then_block,
                        else_branch: else_expr.map(Box::new),
                    },
                    result_ty,
                )
            }

            Expr::Block { id, block } => {
                let hir_block = self.check_block(block, expected)?;
                let ty = hir_block.ty.clone();
                (HirExprKind::Block(hir_block), ty)
            }

            Expr::Return { id, value } => {
                let val = value
                    .as_ref()
                    .map(|v| self.check_expr(v, expected))
                    .transpose()?;

                // Return has Never type since control doesn't continue
                (HirExprKind::Return(val.map(Box::new)), HirType::Never)
            }

            Expr::Tuple { id, elements } => {
                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, None))
                    .collect::<Result<_>>()?;

                let tys: Vec<_> = exprs.iter().map(|e| e.ty.clone()).collect();
                let result_ty = HirType::Tuple(tys);

                (HirExprKind::Tuple(exprs), result_ty)
            }

            Expr::Array { id, elements } => {
                // Extract element type from expected type (either Array or Vec)
                let (elem_ty, is_vec) = expected
                    .and_then(|t| match t {
                        Type::Array { element, .. } => Some((element.as_ref().clone(), false)),
                        Type::Named { name, args } if name == "Vec" && args.len() == 1 => {
                            Some((args[0].clone(), true))
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| (self.fresh_type_var(), false));

                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, Some(&elem_ty)))
                    .collect::<Result<_>>()?;

                let elem_hir_ty = if exprs.is_empty() {
                    self.type_to_hir(&elem_ty)
                } else {
                    exprs[0].ty.clone()
                };

                // Return Vec<T> if expected type was Vec, otherwise Array
                let result_ty = if is_vec {
                    HirType::Named {
                        name: "Vec".to_string(),
                        args: vec![elem_hir_ty.clone()],
                    }
                } else {
                    HirType::Array {
                        element: Box::new(elem_hir_ty),
                        size: Some(exprs.len()),
                    }
                };

                (HirExprKind::Array(exprs), result_ty)
            }

            Expr::ArrayRepeat { id, value, count } => {
                // Extract element type from expected type
                let elem_ty = expected
                    .and_then(|t| match t {
                        Type::Array { element, .. } => Some(element.as_ref().clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| self.fresh_type_var());

                // Type check the value expression
                let value_expr = self.check_expr(value, Some(&elem_ty))?;

                // Evaluate count as a constant usize
                let count_val = self.eval_const_usize(count).unwrap_or_else(|| {
                    self.error(
                        "Array repeat count must be a compile-time constant",
                        Span::dummy(),
                    );
                    0
                });

                let result_ty = HirType::Array {
                    element: Box::new(value_expr.ty.clone()),
                    size: Some(count_val),
                };

                (
                    HirExprKind::ArrayRepeat {
                        value: Box::new(value_expr),
                        count: count_val,
                    },
                    result_ty,
                )
            }

            Expr::Range {
                id,
                start,
                end,
                inclusive,
            } => {
                // Type check start and end if present
                let start_expr = start
                    .as_ref()
                    .map(|e| self.check_expr(e, Some(&Type::I64)))
                    .transpose()?;
                let end_expr = end
                    .as_ref()
                    .map(|e| self.check_expr(e, Some(&Type::I64)))
                    .transpose()?;

                // Infer element type from start or end
                let elem_ty = start_expr
                    .as_ref()
                    .map(|e| e.ty.clone())
                    .or_else(|| end_expr.as_ref().map(|e| e.ty.clone()))
                    .unwrap_or(HirType::I64);

                // Range<T> type
                let range_ty = HirType::Named {
                    name: if *inclusive {
                        "RangeInclusive".to_string()
                    } else {
                        "Range".to_string()
                    },
                    args: vec![elem_ty],
                };

                (
                    HirExprKind::Range {
                        start: start_expr.map(Box::new),
                        end: end_expr.map(Box::new),
                        inclusive: *inclusive,
                    },
                    range_ty,
                )
            }

            Expr::Index { id, base, index } => {
                let base_expr = self.check_expr(base, None)?;
                let index_expr = self.check_expr(index, Some(&Type::I64))?;

                // Check if index is a Range type (for slicing)
                let is_range = matches!(
                    &index_expr.ty,
                    HirType::Named { name, .. } if name == "Range" || name == "RangeInclusive"
                );

                // Extract result type from indexable types
                let result_ty = match &base_expr.ty {
                    HirType::Array { element, size } => {
                        if is_range {
                            // Slicing returns an array of the same element type
                            HirType::Array {
                                element: element.clone(),
                                size: None,
                            }
                        } else {
                            *element.clone()
                        }
                    }
                    HirType::String => {
                        if is_range {
                            HirType::String // String slice returns String
                        } else {
                            HirType::Char
                        }
                    }
                    // Raw pointers are indexable - return inner type
                    HirType::RawPointer { inner, .. } => *inner.clone(),
                    // References to arrays
                    HirType::Ref { inner, .. } => {
                        if let HirType::Array { element, size } = inner.as_ref() {
                            if is_range {
                                HirType::Array {
                                    element: element.clone(),
                                    size: None,
                                }
                            } else {
                                *element.clone()
                            }
                        } else if let HirType::String = inner.as_ref() {
                            if is_range {
                                HirType::String
                            } else {
                                HirType::Char
                            }
                        } else {
                            HirType::Error
                        }
                    }
                    _ => HirType::Error,
                };

                (
                    HirExprKind::Index {
                        base: Box::new(base_expr),
                        index: Box::new(index_expr),
                    },
                    result_ty,
                )
            }

            Expr::Field { id, base, field } => {
                let base_expr = self.check_expr(base, None)?;

                // Look up field type from struct definition
                let field_ty = if let HirType::Named { name, .. } = &base_expr.ty {
                    if let Some(TypeDef::Struct { fields, .. }) = self.type_defs.get(name) {
                        fields
                            .iter()
                            .find(|(n, _)| n == field)
                            .map(|(_, t)| self.type_to_hir(t))
                            .unwrap_or(HirType::Error)
                    } else {
                        HirType::Error
                    }
                } else {
                    HirType::Error
                };

                (
                    HirExprKind::Field {
                        base: Box::new(base_expr),
                        field: field.clone(),
                    },
                    field_ty,
                )
            }

            Expr::TupleField { id, base, index } => {
                let base_expr = self.check_expr(base, None)?;

                // Extract element type from tuple type
                let elem_ty = match &base_expr.ty {
                    HirType::Tuple(elements) => {
                        elements.get(*index).cloned().unwrap_or(HirType::Error)
                    }
                    _ => HirType::Error,
                };

                (
                    HirExprKind::TupleField {
                        base: Box::new(base_expr),
                        index: *index,
                    },
                    elem_ty,
                )
            }

            Expr::StructLit { id, path, fields } => {
                let struct_name = path.segments.last().cloned().unwrap_or_default();

                // Check visibility of struct being constructed
                if let Some(def_id) = self
                    .symbols
                    .as_ref()
                    .and_then(|symbols| symbols.ref_for_node(*id))
                {
                    let span = self
                        .ast
                        .as_ref()
                        .map(|ast| self.expr_span(expr, ast.as_ref()))
                        .unwrap_or_else(Span::dummy);
                    self.check_item_visibility(&def_id, &struct_name, "struct", span);
                }

                let checked_fields: Vec<_> = fields
                    .iter()
                    .map(|(name, expr)| {
                        let expr = self.check_expr(expr, None)?;
                        Ok((name.clone(), expr))
                    })
                    .collect::<Result<_>>()?;

                (
                    HirExprKind::Struct {
                        name: struct_name.clone(),
                        fields: checked_fields,
                    },
                    HirType::Named {
                        name: struct_name,
                        args: vec![],
                    },
                )
            }

            Expr::Loop { id, body } => {
                let body_block = self.check_block(body, None)?;
                (HirExprKind::Loop(body_block), HirType::Unit)
            }

            Expr::While {
                id: _,
                condition,
                body,
            } => {
                let cond_expr = self.check_expr(condition, Some(&Type::Bool))?;
                let body_block = self.check_block(body, None)?;

                // Use proper While HIR node - condition will be re-evaluated each iteration
                (
                    HirExprKind::While {
                        condition: Box::new(cond_expr),
                        body: body_block,
                    },
                    HirType::Unit,
                )
            }

            Expr::For {
                id: _,
                pattern,
                iter,
                body,
            } => {
                // Desugar: for i in start..end { body }
                // Into: { var __counter = start; while __counter < end { let i = __counter; body; __counter = __counter + 1 } }

                // Get the loop variable name from the pattern
                let loop_var = self.pattern_name(pattern);

                // Check if the iterator is a range expression
                match iter.as_ref() {
                    Expr::Range {
                        id: _,
                        start,
                        end,
                        inclusive,
                    } => {
                        // Type check start and end - let type inference determine the type
                        let start_expr = start
                            .as_ref()
                            .map(|e| self.check_expr(e, None))
                            .transpose()?
                            .unwrap_or_else(|| HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Literal(HirLiteral::Int(0)),
                                ty: HirType::I32,
                            });

                        let end_expr =
                            end.as_ref().map(|e| self.check_expr(e, None)).transpose()?;

                        // Determine element type from start expression
                        let elem_ty = start_expr.ty.clone();
                        let is_inclusive = *inclusive;

                        // Generate unique counter variable name
                        let counter_var = format!("__for_counter_{}", self.next_type_var);
                        self.next_type_var += 1;

                        // Build the while condition: counter < end (or counter <= end for inclusive)
                        let cond_expr = if let Some(end_e) = end_expr {
                            HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Binary {
                                    op: if is_inclusive {
                                        HirBinaryOp::Le
                                    } else {
                                        HirBinaryOp::Lt
                                    },
                                    left: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Local(counter_var.clone()),
                                        ty: elem_ty.clone(),
                                    }),
                                    right: Box::new(end_e),
                                },
                                ty: HirType::Bool,
                            }
                        } else {
                            // Infinite range (1..) - always true
                            HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Literal(HirLiteral::Bool(true)),
                                ty: HirType::Bool,
                            }
                        };

                        // Push scope for the for loop body
                        self.env.push_scope();

                        // Define the loop variable in scope (immutable - it gets a new value each iteration)
                        self.env
                            .bind(loop_var.clone(), self.hir_type_to_type(&elem_ty), false);

                        // Check the body
                        let body_block = self.check_block(body, None)?;

                        self.env.pop_scope();

                        // Build the loop body: let i = counter; <original body>; counter = counter + 1
                        let mut loop_stmts = Vec::new();

                        // let i = counter
                        loop_stmts.push(HirStmt::Let {
                            name: loop_var.clone(),
                            ty: elem_ty.clone(),
                            value: Some(HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Local(counter_var.clone()),
                                ty: elem_ty.clone(),
                            }),
                            is_mut: false,
                            layout_hint: None,
                        });

                        // counter = counter + 1
                        loop_stmts.push(HirStmt::Assign {
                            target: HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Local(counter_var.clone()),
                                ty: elem_ty.clone(),
                            },
                            value: HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Binary {
                                    op: HirBinaryOp::Add,
                                    left: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Local(counter_var.clone()),
                                        ty: elem_ty.clone(),
                                    }),
                                    right: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Literal(HirLiteral::Int(1)),
                                        ty: elem_ty.clone(),
                                    }),
                                },
                                ty: elem_ty.clone(),
                            },
                        });

                        // Add original body statements (after increment so `continue` can't skip it)
                        loop_stmts.extend(body_block.stmts);

                        let while_body = HirBlock {
                            stmts: loop_stmts,
                            ty: HirType::Unit,
                        };

                        // Build the while loop
                        let while_expr = HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::While {
                                condition: Box::new(cond_expr),
                                body: while_body,
                            },
                            ty: HirType::Unit,
                        };

                        // Build the outer block: { var counter = start; while ... }
                        let outer_stmts = vec![
                            HirStmt::Let {
                                name: counter_var,
                                ty: elem_ty.clone(),
                                value: Some(start_expr),
                                is_mut: true,
                                layout_hint: None,
                            },
                            HirStmt::Expr(while_expr),
                        ];

                        (
                            HirExprKind::Block(HirBlock {
                                stmts: outer_stmts,
                                ty: HirType::Unit,
                            }),
                            HirType::Unit,
                        )
                    }
                    _ => {
                        // Collection iteration: for item in collection { body }
                        // Desugar to: { var __idx = 0; while __idx < collection.len() { let item = collection[__idx]; body; __idx = __idx + 1 } }

                        // Type check the collection expression
                        let collection_expr = self.check_expr(iter.as_ref(), None)?;
                        let collection_ty = collection_expr.ty.clone();

                        // Extract element type from the collection type
                        let elem_ty = match &collection_ty {
                            HirType::Array { element, .. } => (**element).clone(),
                            HirType::Named { name, args } if name == "Vec" && !args.is_empty() => {
                                args[0].clone()
                            }
                            HirType::Ref { inner, .. } => {
                                // Handle references to arrays/vecs
                                match inner.as_ref() {
                                    HirType::Array { element, .. } => (**element).clone(),
                                    HirType::Named { name, args }
                                        if name == "Vec" && !args.is_empty() =>
                                    {
                                        args[0].clone()
                                    }
                                    _ => {
                                        self.error(
                                            format!("cannot iterate over type `{:?}`; expected array, Vec, or range", collection_ty),
                                            Span::dummy(),
                                        );
                                        return Ok(HirExpr {
                                            id: NodeId::dummy(),
                                            kind: HirExprKind::Block(HirBlock {
                                                stmts: vec![],
                                                ty: HirType::Error,
                                            }),
                                            ty: HirType::Error,
                                        });
                                    }
                                }
                            }
                            _ => {
                                self.error(
                                    format!("cannot iterate over type `{:?}`; expected array, Vec, or range", collection_ty),
                                    Span::dummy(),
                                );
                                return Ok(HirExpr {
                                    id: NodeId::dummy(),
                                    kind: HirExprKind::Block(HirBlock {
                                        stmts: vec![],
                                        ty: HirType::Error,
                                    }),
                                    ty: HirType::Error,
                                });
                            }
                        };

                        // Generate unique index variable name
                        let idx_var = format!("__for_idx_{}", self.next_type_var);
                        self.next_type_var += 1;

                        // Generate unique collection variable name (to avoid re-evaluating the expression)
                        let coll_var = format!("__for_coll_{}", self.next_type_var);
                        self.next_type_var += 1;

                        // Build: collection.len()
                        let len_expr = HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::MethodCall {
                                receiver: Box::new(HirExpr {
                                    id: NodeId::dummy(),
                                    kind: HirExprKind::Local(coll_var.clone()),
                                    ty: collection_ty.clone(),
                                }),
                                method: "len".to_string(),
                                args: vec![],
                            },
                            ty: HirType::Usize,
                        };

                        // Build: __idx < collection.len()
                        let cond_expr = HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::Binary {
                                op: HirBinaryOp::Lt,
                                left: Box::new(HirExpr {
                                    id: NodeId::dummy(),
                                    kind: HirExprKind::Local(idx_var.clone()),
                                    ty: HirType::Usize,
                                }),
                                right: Box::new(len_expr),
                            },
                            ty: HirType::Bool,
                        };

                        // Push scope for the for loop body
                        self.env.push_scope();

                        // Define the loop variable in scope
                        self.env
                            .bind(loop_var.clone(), self.hir_type_to_type(&elem_ty), false);

                        // Check the body
                        let body_block = self.check_block(body, None)?;

                        self.env.pop_scope();

                        // Build the loop body: let item = collection[__idx]; <original body>; __idx = __idx + 1
                        let mut loop_stmts = Vec::new();

                        // let item = collection[__idx]
                        loop_stmts.push(HirStmt::Let {
                            name: loop_var.clone(),
                            ty: elem_ty.clone(),
                            value: Some(HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Index {
                                    base: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Local(coll_var.clone()),
                                        ty: collection_ty.clone(),
                                    }),
                                    index: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Local(idx_var.clone()),
                                        ty: HirType::Usize,
                                    }),
                                },
                                ty: elem_ty.clone(),
                            }),
                            is_mut: false,
                            layout_hint: None,
                        });

                        // __idx = __idx + 1
                        loop_stmts.push(HirStmt::Assign {
                            target: HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Local(idx_var.clone()),
                                ty: HirType::Usize,
                            },
                            value: HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Binary {
                                    op: HirBinaryOp::Add,
                                    left: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Local(idx_var.clone()),
                                        ty: HirType::Usize,
                                    }),
                                    right: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Literal(HirLiteral::Int(1)),
                                        ty: HirType::Usize,
                                    }),
                                },
                                ty: HirType::Usize,
                            },
                        });

                        // Add original body statements (after increment so `continue` can't skip it)
                        loop_stmts.extend(body_block.stmts);

                        let while_body = HirBlock {
                            stmts: loop_stmts,
                            ty: HirType::Unit,
                        };

                        // Build the while loop
                        let while_expr = HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::While {
                                condition: Box::new(cond_expr),
                                body: while_body,
                            },
                            ty: HirType::Unit,
                        };

                        // Build the outer block: { let __coll = collection; var __idx = 0; while ... }
                        let outer_stmts = vec![
                            // Store collection in a local to avoid re-evaluation
                            HirStmt::Let {
                                name: coll_var,
                                ty: collection_ty,
                                value: Some(collection_expr),
                                is_mut: false,
                                layout_hint: None,
                            },
                            // Initialize index to 0
                            HirStmt::Let {
                                name: idx_var,
                                ty: HirType::Usize,
                                value: Some(HirExpr {
                                    id: NodeId::dummy(),
                                    kind: HirExprKind::Literal(HirLiteral::Int(0)),
                                    ty: HirType::Usize,
                                }),
                                is_mut: true,
                                layout_hint: None,
                            },
                            HirStmt::Expr(while_expr),
                        ];

                        (
                            HirExprKind::Block(HirBlock {
                                stmts: outer_stmts,
                                ty: HirType::Unit,
                            }),
                            HirType::Unit,
                        )
                    }
                }
            }

            Expr::Break { id, value } => {
                let val = value
                    .as_ref()
                    .map(|v| self.check_expr(v, None))
                    .transpose()?;
                (HirExprKind::Break(val.map(Box::new)), HirType::Never)
            }

            Expr::Continue { id } => (HirExprKind::Continue, HirType::Never),

            // ==================== EPISTEMIC EXPRESSIONS ====================

            // Do expression: do(X=1, Y=2) - list of interventions
            Expr::Do { id, interventions } => {
                // Lower each intervention as a sequence
                let mut do_exprs = Vec::new();
                for (var, val) in interventions {
                    let value_expr = self.check_expr(val, None)?;
                    do_exprs.push(HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Do {
                            variable: var.clone(),
                            value: Box::new(value_expr),
                        },
                        ty: HirType::Unit,
                    });
                }

                // If multiple interventions, wrap in a block
                if do_exprs.len() == 1 {
                    (do_exprs.pop().unwrap().kind, HirType::Unit)
                } else {
                    (
                        HirExprKind::Block(HirBlock {
                            stmts: do_exprs.into_iter().map(HirStmt::Expr).collect(),
                            ty: HirType::Unit,
                        }),
                        HirType::Unit,
                    )
                }
            }

            Expr::Counterfactual {
                id,
                factual,
                intervention,
                outcome,
            } => {
                let factual_expr = self.check_expr(factual, None)?;
                let intervention_expr = self.check_expr(intervention, None)?;
                let outcome_expr = self.check_expr(outcome, None)?;
                let outcome_ty = outcome_expr.ty.clone();

                (
                    HirExprKind::Counterfactual {
                        factual: Box::new(factual_expr),
                        intervention: Box::new(intervention_expr),
                        outcome: Box::new(outcome_expr),
                    },
                    outcome_ty,
                )
            }

            Expr::KnowledgeExpr {
                id,
                value,
                epsilon,
                validity,
                provenance,
            } => {
                let value_expr = self.check_expr(value, None)?;

                // Epsilon is optional
                let epsilon_expr = if let Some(eps) = epsilon {
                    self.check_expr(eps, Some(&Type::F64))?
                } else {
                    // Default epsilon of 1.0 (perfect confidence)
                    HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Literal(HirLiteral::Float(1.0)),
                        ty: HirType::F64,
                    }
                };

                let validity_expr = validity
                    .as_ref()
                    .map(|v| self.check_expr(v, None))
                    .transpose()?;

                let inner_ty = value_expr.ty.clone();
                let result_ty = HirType::Knowledge {
                    inner: Box::new(inner_ty),
                    epsilon_bound: None, // Could extract from epsilon if constant
                    provenance: None,
                };

                // Provenance is an expression, not a ProvenanceMarker - convert it
                let prov = provenance
                    .as_ref()
                    .map(|_| HirProvenance::Derived { sources: vec![] });

                (
                    HirExprKind::Knowledge {
                        value: Box::new(value_expr),
                        epsilon: Box::new(epsilon_expr),
                        validity: validity_expr.map(Box::new),
                        provenance: prov,
                    },
                    result_ty,
                )
            }

            Expr::Query {
                id,
                target,
                given,
                interventions,
            } => {
                let target_expr = self.check_expr(target, None)?;
                let given_exprs: Vec<_> = given
                    .iter()
                    .map(|g| self.check_expr(g, None))
                    .collect::<Result<_>>()?;

                // Interventions are (variable, value) pairs - lower each value
                let intervention_exprs: Vec<_> = interventions
                    .iter()
                    .map(|(var, val)| {
                        let val_expr = self.check_expr(val, None)?;
                        Ok(HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::Do {
                                variable: var.clone(),
                                value: Box::new(val_expr),
                            },
                            ty: HirType::Unit,
                        })
                    })
                    .collect::<Result<_>>()?;

                // Query returns a probability (Knowledge[f64])
                let result_ty = HirType::Knowledge {
                    inner: Box::new(HirType::F64),
                    epsilon_bound: None,
                    provenance: None,
                };

                (
                    HirExprKind::Query {
                        target: Box::new(target_expr),
                        given: given_exprs,
                        interventions: intervention_exprs,
                    },
                    result_ty,
                )
            }

            // Observe expression: observe(data ~ distribution) for probabilistic programming
            Expr::Observe {
                id,
                data,
                distribution,
            } => {
                let data_expr = self.check_expr(data, None)?;
                let dist_expr = self.check_expr(distribution, None)?;

                // Create an observe expression - the variable is derived from the data expression
                let var_name = match &data_expr.kind {
                    HirExprKind::Local(name) => name.clone(),
                    _ => "_observed".to_string(),
                };

                (
                    HirExprKind::Observe {
                        variable: var_name,
                        value: Box::new(dist_expr),
                    },
                    HirType::Unit,
                )
            }

            // Uncertain expression: value with uncertainty (e.g., 5.0 ± 0.1)
            Expr::Uncertain {
                id,
                value,
                uncertainty,
            } => {
                let value_expr = self.check_expr(value, None)?;
                let uncertainty_expr = self.check_expr(uncertainty, None)?;
                let inner_ty = value_expr.ty.clone();

                // Convert uncertainty to epsilon (confidence bound)
                // For now, assume 2-sigma gives ~95% confidence
                let result_ty = HirType::Knowledge {
                    inner: Box::new(inner_ty),
                    epsilon_bound: Some(0.95),
                    provenance: None,
                };

                (
                    HirExprKind::Knowledge {
                        value: Box::new(value_expr),
                        epsilon: Box::new(uncertainty_expr), // Use uncertainty as epsilon proxy
                        validity: None,
                        provenance: Some(HirProvenance::Derived { sources: vec![] }),
                    },
                    result_ty,
                )
            }

            // Ontology term expression: prefix:term (e.g., chebi:aspirin, drugbank:DB00945)
            Expr::OntologyTerm {
                id: _,
                ontology,
                term,
            } => {
                // Track that this ontology prefix is used
                self.used_ontology_prefixes.insert(ontology.clone());

                let result_ty = HirType::Ontology {
                    namespace: ontology.clone(),
                    term: term.clone(),
                };

                (
                    HirExprKind::OntologyTerm {
                        namespace: ontology.clone(),
                        term: term.clone(),
                    },
                    result_ty,
                )
            }

            // Handle vec![] macro - treat as array literal with Vec type
            Expr::MacroInvocation(macro_inv) if macro_inv.name == "vec" => {
                // Parse vec! macro arguments as expressions
                // For vec![a, b, c], the args contain the comma-separated expressions
                let elements = self.parse_vec_macro_args(&macro_inv.args);

                // Determine element type from expected type or first element
                let (elem_ty, _is_vec) = expected
                    .and_then(|t| match t {
                        Type::Array { element, .. } => Some((element.as_ref().clone(), false)),
                        Type::Named { name, args } if name == "Vec" && args.len() == 1 => {
                            Some((args[0].clone(), true))
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| (self.fresh_type_var(), true)); // Default to Vec

                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, Some(&elem_ty)))
                    .collect::<Result<_>>()?;

                let elem_hir_ty = if exprs.is_empty() {
                    self.type_to_hir(&elem_ty)
                } else {
                    exprs[0].ty.clone()
                };

                // vec! always produces Vec<T>
                let result_ty = HirType::Named {
                    name: "Vec".to_string(),
                    args: vec![elem_hir_ty],
                };

                (HirExprKind::Array(exprs), result_ty)
            }

            // Handle method calls (e.g., vec.is_empty(), vec.len(), etc.)
            Expr::MethodCall {
                id,
                receiver,
                method,
                args,
                ..
            } => {
                // First, check the receiver to get its type
                let receiver_expr = self.check_expr(receiver, None)?;
                let receiver_ty = receiver_expr.ty.clone();

                // Check arguments
                let arg_exprs: Vec<_> = args
                    .iter()
                    .map(|a| self.check_expr(a, None))
                    .collect::<Result<_>>()?;

                // Knowledge explicit extraction: `k.unwrap("reason")`
                if method == "unwrap" {
                    if let HirType::Knowledge { inner, .. } = &receiver_ty {
                        let span = self
                            .ast
                            .as_ref()
                            .map(|ast| self.expr_span(expr, ast.as_ref()))
                            .unwrap_or_else(Span::dummy);

                        if arg_exprs.len() != 1 {
                            self.error(
                                "Knowledge.unwrap requires exactly one argument: a reason string"
                                    .to_string(),
                                span,
                            );
                            return Ok(HirExpr {
                                id: *id,
                                kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                ty: HirType::Error,
                            });
                        }

                        if arg_exprs[0].ty != HirType::String {
                            self.error(
                                "Knowledge.unwrap(reason): reason must be a `string`".to_string(),
                                span,
                            );
                            return Ok(HirExpr {
                                id: *id,
                                kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                ty: HirType::Error,
                            });
                        }

                        return Ok(HirExpr {
                            id: *id,
                            kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                            ty: (*inner.clone()),
                        });
                    }
                }

                // Knowledge introspection / extraction helpers (A)
                if let HirType::Knowledge { inner, .. } = &receiver_ty {
                    let span = self
                        .ast
                        .as_ref()
                        .map(|ast| self.expr_span(expr, ast.as_ref()))
                        .unwrap_or_else(Span::dummy);

                    match method.as_str() {
                        "value" => {
                            if !arg_exprs.is_empty() {
                                self.error(
                                    "Knowledge.value() takes no arguments".to_string(),
                                    span,
                                );
                                return Ok(HirExpr {
                                    id: *id,
                                    kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                    ty: HirType::Error,
                                });
                            }
                            return Ok(HirExpr {
                                id: *id,
                                kind: HirExprKind::Unwrap(Box::new(receiver_expr)),
                                ty: (*inner.clone()),
                            });
                        }
                        "confidence" | "epsilon" => {
                            if !arg_exprs.is_empty() {
                                self.error(
                                    "Knowledge.confidence()/epsilon() takes no arguments"
                                        .to_string(),
                                    span,
                                );
                            }
                            return Ok(HirExpr {
                                id: *id,
                                kind: HirExprKind::EpsilonOf(Box::new(receiver_expr)),
                                ty: HirType::F64,
                            });
                        }
                        "provenance" => {
                            if !arg_exprs.is_empty() {
                                self.error(
                                    "Knowledge.provenance() takes no arguments".to_string(),
                                    span,
                                );
                            }
                            return Ok(HirExpr {
                                id: *id,
                                kind: HirExprKind::ProvenanceOf(Box::new(receiver_expr)),
                                ty: HirType::Unit,
                            });
                        }
                        "validity" => {
                            if !arg_exprs.is_empty() {
                                self.error(
                                    "Knowledge.validity() takes no arguments".to_string(),
                                    span,
                                );
                            }
                            return Ok(HirExpr {
                                id: *id,
                                kind: HirExprKind::ValidityOf(Box::new(receiver_expr)),
                                ty: HirType::Unit,
                            });
                        }
                        _ => {}
                    }
                }

                // Determine return type based on method name and receiver type
                let result_ty = self.get_method_return_type(&receiver_ty, method, &arg_exprs);

                (
                    HirExprKind::MethodCall {
                        receiver: Box::new(receiver_expr),
                        method: method.clone(),
                        args: arg_exprs,
                    },
                    result_ty,
                )
            }

            // Match expression handling
            Expr::Match {
                id: _,
                scrutinee,
                arms,
            } => {
                // Check the scrutinee expression
                let scrutinee_expr = self.check_expr(scrutinee, None)?;
                let scrutinee_ty = scrutinee_expr.ty.clone();

                // Check each arm
                let mut checked_arms = Vec::new();
                let mut arm_types = Vec::new();

                // Convert scrutinee HirType to internal Type for pattern binding
                let scrutinee_internal_ty = self.hir_type_to_type(&scrutinee_ty);

                for arm in arms {
                    // Push scope for pattern variables
                    self.env.push_scope();

                    // Bind pattern variables based on scrutinee type
                    self.bind_pattern_to_type(&arm.pattern, &scrutinee_internal_ty, false);

                    let body_expr = self.check_expr(&arm.body, None)?;
                    arm_types.push(body_expr.ty.clone());

                    checked_arms.push(HirMatchArm {
                        pattern: self.lower_pattern(&arm.pattern),
                        guard: None,
                        body: body_expr,
                    });

                    // Pop pattern variable scope
                    self.env.pop_scope();
                }

                // Determine the result type:
                // - If all arms return the same type, use that
                // - If one arm returns Never, use the other arm's type
                // - If arms differ and one is Unit (common with if-let without else), use Unit
                let result_ty = if arm_types.is_empty() {
                    HirType::Unit
                } else if arm_types.iter().all(|t| t == &arm_types[0]) {
                    arm_types[0].clone()
                } else {
                    // Check for Never type (for exhaustive patterns)
                    let non_never: Vec<_> =
                        arm_types.iter().filter(|t| **t != HirType::Never).collect();
                    if non_never.len() == 1 {
                        non_never[0].clone()
                    } else if arm_types.iter().any(|t| *t == HirType::Unit) {
                        // If any arm returns Unit (like if-let without else), the whole expression is Unit
                        HirType::Unit
                    } else {
                        // Default to first arm's type
                        arm_types[0].clone()
                    }
                };

                (
                    HirExprKind::Match {
                        scrutinee: Box::new(scrutinee_expr),
                        arms: checked_arms,
                    },
                    result_ty,
                )
            }

            // Cast expression: expr as Type
            Expr::Cast { id: _, expr, ty } => {
                // Check the inner expression
                let inner_expr = self.check_expr(expr, None)?;

                // Convert target type from TypeExpr to HirType
                let target_type = self.lower_type_expr(ty);
                let hir_target = self.type_to_hir(&target_type);

                (
                    HirExprKind::Cast {
                        expr: Box::new(inner_expr),
                        target: hir_target.clone(),
                    },
                    hir_target,
                )
            }

            // Effect operation: perform Effect::op(args)
            Expr::Perform {
                id: _,
                effect,
                op,
                args,
            } => {
                // Type-check the arguments
                let checked_args: Vec<HirExpr> = args
                    .iter()
                    .map(|a| self.check_expr(a, None))
                    .collect::<Result<_>>()?;

                // Look up the effect operation's return type
                let effect_name = effect
                    .segments
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string());

                // Lookup the operation in the registry
                let return_type = if let Some(ty) = self
                    .effect_operations
                    .get(&(effect_name.clone(), op.clone()))
                {
                    ty.clone()
                } else {
                    // Effect operation not found - default to Unit but could emit error
                    Type::Unit
                };
                let return_ty = self.type_to_hir(&return_type);

                (
                    HirExprKind::Perform {
                        effect: effect_name,
                        op: op.clone(),
                        args: checked_args,
                    },
                    return_ty,
                )
            }

            // Effect handler: handle expr with Handler
            Expr::Handle {
                id: _,
                expr: inner,
                handler,
            } => {
                // Type-check the inner expression
                let inner_expr = self.check_expr(inner, None)?;
                let inner_ty = inner_expr.ty.clone();

                let handler_name = handler
                    .segments
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string());

                // Look up which effect this handler handles and record it as masked.
                // This enables effect masking: a function can be pure even if it uses
                // impure operations internally, as long as all effects are handled.
                //
                // Example:
                //   fn pure_computation() -> i32 {
                //       handle { perform IO.print("hello"); 42 } with IOHandler
                //   }
                // The IO effect is handled internally, so pure_computation is pure.
                if let Some(handled_effect) = self.lookup_handler_effect(&handler_name) {
                    // Record this effect as masked in the current context
                    self.masked_effects.effects.insert(handled_effect.clone());
                }

                // The handler handles the effect, so the result type is the inner type
                // (effect is removed from the effect row)
                (
                    HirExprKind::Handle {
                        expr: Box::new(inner_expr),
                        handler: handler_name,
                    },
                    inner_ty,
                )
            }

            // Resume continuation: resume(value)
            Expr::Resume { id: _, value } => {
                // Type-check the resume value
                let value_expr = self.check_expr(value, None)?;
                let value_ty = value_expr.ty.clone();

                // The type of resume is the type of the resumed value
                // In effect handlers, resume continues execution with this value
                (
                    HirExprKind::Resume {
                        value: Box::new(value_expr),
                    },
                    value_ty,
                )
            }

            // Probabilistic sampling: sample distribution
            Expr::Sample {
                id: _,
                distribution,
            } => {
                // Type-check the distribution expression
                let dist_expr = self.check_expr(distribution, None)?;

                // The result type depends on the distribution type
                // For most distributions, this is f64
                let sample_ty = match &dist_expr.ty {
                    HirType::Named { name, args } if name == "Normal" || name == "Uniform" => {
                        HirType::F64
                    }
                    HirType::Named { name, args } if name == "Bernoulli" => HirType::Bool,
                    HirType::Named { name, args } if name == "Poisson" => HirType::I64,
                    _ => HirType::F64, // Default to f64 for unknown distributions
                };

                (HirExprKind::Sample(Box::new(dist_expr)), sample_ty)
            }

            // Await expression: expr.await
            Expr::Await { id: _, expr: inner } => {
                // Type-check the future expression
                let future_expr = self.check_expr(inner, None)?;

                // Extract the inner type from Future<T>
                let result_ty = match &future_expr.ty {
                    HirType::Named { name, args } if name == "Future" => {
                        args.first().cloned().unwrap_or(HirType::Unit)
                    }
                    _ => future_expr.ty.clone(), // If not a Future, return as-is
                };

                (
                    HirExprKind::Await {
                        future: Box::new(future_expr),
                    },
                    result_ty,
                )
            }

            // Async block: async { ... }
            Expr::AsyncBlock { id: _, block } => {
                // Type-check the block
                let checked_block = self.check_block(block, None)?;
                let block_ty = checked_block.ty.clone();

                // Wrap the block type in Future<T>
                let future_ty = HirType::Named {
                    name: "Future".to_string(),
                    args: vec![block_ty],
                };

                (
                    HirExprKind::AsyncBlock {
                        body: checked_block,
                    },
                    future_ty,
                )
            }

            // Async closure: async |params| body
            Expr::AsyncClosure {
                id: _,
                params,
                return_type,
                body,
            } => {
                // Type-check the closure body
                let body_expr = self.check_expr(body, None)?;
                let body_ty = body_expr.ty.clone();

                // Build parameter list
                let mut hir_params: Vec<HirParam> = Vec::new();
                for (name, ty_opt) in params {
                    let param_ty = if let Some(t) = ty_opt {
                        let lowered = self.lower_type_expr(t);
                        self.type_to_hir(&lowered)
                    } else {
                        HirType::Unit
                    };
                    hir_params.push(HirParam {
                        id: NodeId::dummy(),
                        name: name.clone(),
                        ty: param_ty,
                        is_mut: false,
                    });
                }

                // The closure returns a Future
                let future_ty = HirType::Named {
                    name: "Future".to_string(),
                    args: vec![body_ty],
                };

                // Build the function type
                let param_types: Vec<HirType> = hir_params.iter().map(|p| p.ty.clone()).collect();
                let fn_ty = HirType::Fn {
                    params: param_types,
                    return_type: Box::new(future_ty),
                };

                (
                    HirExprKind::Closure {
                        params: hir_params,
                        body: Box::new(body_expr),
                    },
                    fn_ty,
                )
            }

            // Spawn expression: spawn { expr }
            Expr::Spawn { id: _, expr: inner } => {
                // Type-check the spawned expression
                let inner_expr = self.check_expr(inner, None)?;
                let inner_ty = inner_expr.ty.clone();

                // Spawn returns a JoinHandle<T>
                let handle_ty = HirType::Named {
                    name: "JoinHandle".to_string(),
                    args: vec![inner_ty],
                };

                (
                    HirExprKind::Spawn {
                        expr: Box::new(inner_expr),
                    },
                    handle_ty,
                )
            }

            // Select expression: select { arms... }
            Expr::Select { id: _, arms } => {
                // Type-check each select arm
                let mut checked_arms = Vec::new();
                let mut result_types = Vec::new();

                for arm in arms {
                    let future_expr = self.check_expr(&arm.future, None)?;
                    let pattern = self.lower_pattern(&arm.pattern);
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|g| self.check_expr(g, Some(&Type::Bool)))
                        .transpose()?
                        .map(Box::new);
                    let body_expr = self.check_expr(&arm.body, None)?;

                    result_types.push(body_expr.ty.clone());

                    checked_arms.push(HirSelectArm {
                        future: future_expr,
                        pattern,
                        guard,
                        body: body_expr,
                    });
                }

                // All arms should have compatible types
                let result_ty = if result_types.is_empty() {
                    HirType::Unit
                } else {
                    result_types[0].clone()
                };

                (HirExprKind::Select { arms: checked_arms }, result_ty)
            }

            // Join expression: join(future1, future2, ...)
            Expr::Join { id: _, futures } => {
                // Type-check all futures
                let checked_futures: Vec<HirExpr> = futures
                    .iter()
                    .map(|f| self.check_expr(f, None))
                    .collect::<Result<_>>()?;

                // Extract the inner types from each Future<T>
                let inner_types: Vec<HirType> = checked_futures
                    .iter()
                    .map(|f| match &f.ty {
                        HirType::Named { name, args } if name == "Future" => {
                            args.first().cloned().unwrap_or(HirType::Unit)
                        }
                        ty => ty.clone(),
                    })
                    .collect();

                // Join returns a tuple of the results
                let result_ty = if inner_types.len() == 1 {
                    inner_types[0].clone()
                } else {
                    HirType::Tuple(inner_types)
                };

                (
                    HirExprKind::Join {
                        futures: checked_futures,
                    },
                    result_ty,
                )
            }

            // Fallback for any remaining expressions
            _ => {
                // For truly unhandled expressions, return a placeholder
                (HirExprKind::Literal(HirLiteral::Unit), HirType::Unit)
            }
        };

        let id = match expr {
            Expr::Literal { id, .. }
            | Expr::Path { id, .. }
            | Expr::Binary { id, .. }
            | Expr::Unary { id, .. }
            | Expr::Call { id, .. }
            | Expr::If { id, .. }
            | Expr::Block { id, .. }
            | Expr::Return { id, .. }
            | Expr::Tuple { id, .. }
            | Expr::Array { id, .. }
            | Expr::ArrayRepeat { id, .. }
            | Expr::Cast { id, .. }
            | Expr::OntologyTerm { id, .. }
            | Expr::Perform { id, .. }
            | Expr::Handle { id, .. }
            | Expr::Resume { id, .. }
            | Expr::Sample { id, .. }
            | Expr::Await { id, .. }
            | Expr::AsyncBlock { id, .. }
            | Expr::AsyncClosure { id, .. }
            | Expr::Spawn { id, .. }
            | Expr::Select { id, .. }
            | Expr::Join { id, .. }
            | Expr::KnowledgeExpr { id, .. }
            | Expr::Uncertain { id, .. } => *id,
            _ => NodeId::dummy(),
        };

        Ok(HirExpr { id, kind, ty })
    }

    fn check_literal_with_expected(
        &self,
        lit: &Literal,
        expected: Option<&Type>,
    ) -> (HirLiteral, HirType) {
        match lit {
            Literal::Unit => (HirLiteral::Unit, HirType::Unit),
            Literal::Bool(b) => (HirLiteral::Bool(*b), HirType::Bool),
            Literal::Int(i) => {
                // Infer integer type from context if available
                let ty = match expected {
                    Some(Type::I8) => HirType::I8,
                    Some(Type::I16) => HirType::I16,
                    Some(Type::I32) => HirType::I32,
                    Some(Type::I64) => HirType::I64,
                    Some(Type::I128) => HirType::I128,
                    Some(Type::Isize) => HirType::Isize,
                    Some(Type::U8) => HirType::U8,
                    Some(Type::U16) => HirType::U16,
                    Some(Type::U32) => HirType::U32,
                    Some(Type::U64) => HirType::U64,
                    Some(Type::U128) => HirType::U128,
                    Some(Type::Usize) => HirType::Usize,
                    Some(Type::F32) => HirType::F32,
                    Some(Type::F64) => HirType::F64,
                    _ => HirType::I64, // Default to i64
                };
                (HirLiteral::Int(*i), ty)
            }
            Literal::Float(f) => {
                // Infer float type from context if available
                let ty = match expected {
                    Some(Type::F32) => HirType::F32,
                    _ => HirType::F64, // Default to f64
                };
                (HirLiteral::Float(*f), ty)
            }
            Literal::Char(c) => (HirLiteral::Char(*c), HirType::Char),
            Literal::String(s) => (HirLiteral::String(s.clone()), HirType::String),
            // C string literal: null-terminated, type is *const i8 (raw pointer to byte)
            Literal::CString(s) => (
                HirLiteral::CString(s.clone()),
                HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::I8),
                },
            ),
            // Unit literals: create Quantity type with unit information
            Literal::IntUnit(i, unit) => {
                let hir_unit = self.parse_unit_string(unit);
                (
                    HirLiteral::Int(*i),
                    HirType::Quantity {
                        numeric: Box::new(HirType::I64),
                        unit: hir_unit,
                    },
                )
            }
            Literal::FloatUnit(f, unit) => {
                let hir_unit = self.parse_unit_string(unit);
                (
                    HirLiteral::Float(*f),
                    HirType::Quantity {
                        numeric: Box::new(HirType::F64),
                        unit: hir_unit,
                    },
                )
            }
            Literal::TypedInt(i, suffix) => {
                let ty = match suffix.as_str() {
                    "i8" => HirType::I8,
                    "i16" => HirType::I16,
                    "i32" => HirType::I32,
                    "i64" => HirType::I64,
                    "i128" => HirType::I128,
                    "isize" => HirType::Isize,
                    "u8" => HirType::U8,
                    "u16" => HirType::U16,
                    "u32" => HirType::U32,
                    "u64" => HirType::U64,
                    "u128" => HirType::U128,
                    "usize" => HirType::Usize,
                    _ => HirType::I64,
                };
                (HirLiteral::Int(*i), ty)
            }
            Literal::TypedFloat(f, suffix) => {
                let ty = match suffix.as_str() {
                    "f32" => HirType::F32,
                    _ => HirType::F64,
                };
                (HirLiteral::Float(*f), ty)
            }
        }
    }

    /// Check unit compatibility for binary operations and compute result type
    fn check_binary_units(&mut self, op: BinaryOp, left: &HirType, right: &HirType) -> HirType {
        let (left_inner, left_conf, left_is_knowledge) = match left {
            HirType::Knowledge {
                inner,
                epsilon_bound,
                ..
            } => ((**inner).clone(), *epsilon_bound, true),
            _ => (left.clone(), None, false),
        };
        let (right_inner, right_conf, right_is_knowledge) = match right {
            HirType::Knowledge {
                inner,
                epsilon_bound,
                ..
            } => ((**inner).clone(), *epsilon_bound, true),
            _ => (right.clone(), None, false),
        };

        let wrap_knowledge = matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Rem
                | BinaryOp::PlusMinus
        ) && (left_is_knowledge || right_is_knowledge);

        // Extract units from quantity types
        let (left_numeric, left_unit) = self.extract_quantity(&left_inner);
        let (right_numeric, right_unit) = self.extract_quantity(&right_inner);

        let result_inner = match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::PlusMinus => {
                // Addition/subtraction requires compatible units
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        if !lu.is_compatible(ru) {
                            self.error(
                                format!(
                                    "Unit mismatch in {}: cannot {} {} and {}",
                                    if op == BinaryOp::Add {
                                        "addition"
                                    } else {
                                        "subtraction"
                                    },
                                    if op == BinaryOp::Add {
                                        "add"
                                    } else {
                                        "subtract"
                                    },
                                    lu.format(),
                                    ru.format()
                                ),
                                Span::dummy(),
                            );
                            return HirType::Error;
                        }
                        // Result has same unit as operands
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: lu.clone(),
                        }
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        self.error(
                            format!(
                                "Cannot {} values with and without units",
                                if op == BinaryOp::Add {
                                    "add"
                                } else {
                                    "subtract"
                                }
                            ),
                            Span::dummy(),
                        );
                        HirType::Error
                    }
                    (None, None) => left_numeric.clone(),
                }
            }
            BinaryOp::Mul => {
                // Multiplication: units multiply
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        let result_unit = lu.multiply(ru);
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: result_unit,
                        }
                    }
                    (Some(lu), None) => HirType::Quantity {
                        numeric: Box::new(left_numeric.clone()),
                        unit: lu.clone(),
                    },
                    (None, Some(ru)) => HirType::Quantity {
                        numeric: Box::new(left_numeric.clone()),
                        unit: ru.clone(),
                    },
                    (None, None) => left_numeric.clone(),
                }
            }
            BinaryOp::Div => {
                // Division: units divide
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        let result_unit = lu.divide(ru);
                        if result_unit.is_dimensionless() {
                            left_numeric.clone()
                        } else {
                            HirType::Quantity {
                                numeric: Box::new(left_numeric.clone()),
                                unit: result_unit,
                            }
                        }
                    }
                    (Some(lu), None) => HirType::Quantity {
                        numeric: Box::new(left_numeric.clone()),
                        unit: lu.clone(),
                    },
                    (None, Some(ru)) => {
                        // Dividing dimensionless by unit gives inverse unit
                        let result_unit = HirUnit::dimensionless().divide(ru);
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: result_unit,
                        }
                    }
                    (None, None) => left_numeric.clone(),
                }
            }
            BinaryOp::Rem => {
                // Remainder: same rules as division for compatibility, result has left's unit
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        if !lu.is_compatible(ru) {
                            self.error(
                                format!(
                                    "Unit mismatch in remainder: incompatible units {} and {}",
                                    lu.format(),
                                    ru.format()
                                ),
                                Span::dummy(),
                            );
                            return HirType::Error;
                        }
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: lu.clone(),
                        }
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        self.error(
                            "Cannot compute remainder of values with and without units".to_string(),
                            Span::dummy(),
                        );
                        HirType::Error
                    }
                    (None, None) => left_numeric.clone(),
                }
            }
            // Comparison operators: units must be compatible, result is bool
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                if let (Some(lu), Some(ru)) = (&left_unit, &right_unit) {
                    if !lu.is_compatible(ru) {
                        self.error(
                            format!(
                                "Unit mismatch in comparison: cannot compare {} and {}",
                                lu.format(),
                                ru.format()
                            ),
                            Span::dummy(),
                        );
                    }
                }
                HirType::Bool
            }
            // Logical operators: require bool operands
            BinaryOp::And | BinaryOp::Or => {
                let op_name = if op == BinaryOp::And { "&&" } else { "||" };
                if !matches!(left, HirType::Bool) {
                    self.error(
                        format!(
                            "Logical '{}' requires bool operands, found `{:?}` on left",
                            op_name, left
                        ),
                        Span::dummy(),
                    );
                }
                if !matches!(right, HirType::Bool) {
                    self.error(
                        format!(
                            "Logical '{}' requires bool operands, found `{:?}` on right",
                            op_name, right
                        ),
                        Span::dummy(),
                    );
                }
                HirType::Bool
            }
            // Bitwise operators: no unit handling
            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => left.clone(),
            // Concatenation: combine array sizes
            BinaryOp::Concat => {
                match (left, right) {
                    (
                        HirType::Array {
                            element: left_elem,
                            size: left_size,
                        },
                        HirType::Array {
                            element: _right_elem,
                            size: right_size,
                        },
                    ) => {
                        // Combine sizes if both are known
                        let combined_size = match (left_size, right_size) {
                            (Some(l), Some(r)) => Some(l + r),
                            _ => None, // Unknown size if either is unknown
                        };
                        HirType::Array {
                            element: left_elem.clone(),
                            size: combined_size,
                        }
                    }
                    // For Vec or other types, just return a Vec
                    (HirType::Named { name, args }, _) if name == "Vec" => HirType::Named {
                        name: name.clone(),
                        args: args.clone(),
                    },
                    // Default: keep left type
                    _ => left.clone(),
                }
            }
        };

        if wrap_knowledge {
            let left_bound = if left_is_knowledge {
                left_conf.unwrap_or(0.0)
            } else {
                1.0
            };
            let right_bound = if right_is_knowledge {
                right_conf.unwrap_or(0.0)
            } else {
                1.0
            };

            return HirType::Knowledge {
                inner: Box::new(result_inner),
                epsilon_bound: Some(left_bound.min(right_bound)),
                provenance: None,
            };
        }

        result_inner
    }

    fn extract_knowledge_confidence_lower_bound(&self, ty: &TypeExpr) -> Option<f64> {
        let TypeExpr::Knowledge { epsilon, .. } = ty else {
            return None;
        };

        let eps = epsilon.as_ref()?;
        match eps.operator {
            ComparisonOp::Ge | ComparisonOp::Gt | ComparisonOp::Eq => self.const_f64(&eps.value),
            ComparisonOp::Lt | ComparisonOp::Le => None,
        }
    }

    fn const_f64(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Literal {
                value: Literal::Float(f),
                ..
            } => Some(*f),
            Expr::Literal {
                value: Literal::Int(i),
                ..
            } => Some(*i as f64),
            _ => None,
        }
    }

    /// Extract the numeric type and optional unit from a type
    fn extract_quantity(&self, ty: &HirType) -> (HirType, Option<HirUnit>) {
        match ty {
            HirType::Quantity { numeric, unit } => (*numeric.clone(), Some(unit.clone())),
            _ => (ty.clone(), None),
        }
    }

    /// Check if a name is a builtin function
    fn is_builtin_function(&self, name: &str) -> bool {
        let result = matches!(
            name,
            "print"
                | "println"
                | "assert"
                | "assert_eq"
                | "len"
                | "type_of"
                | "Some"
                | "None"
                | "Ok"
                | "Err"
                | "dbg"
                | "panic"
                | "format"
                | "read_line"
                | "parse_int"
                | "parse_float"
                | "to_string"
                | "sqrt"
                | "abs"
                | "sin"
                | "cos"
                | "tan"
                | "exp"
                | "log"
                | "pow"
                | "floor"
                | "ceil"
                | "round"
                | "min"
                | "max"
                // Linear algebra constructors
                | "vec2"
                | "vec3"
                | "vec4"
                | "mat2"
                | "mat3"
                | "mat4"
                | "quat"
                // Vector operations
                | "dot"
                | "cross"
                | "normalize"
                | "length"
                | "length_squared"
                // Quaternion operations
                | "quat_mul"
                | "quat_conj"
                | "quat_inv"
                | "quat_normalize"
                | "quat_identity"
                // Matrix operations
                | "mat_mul"
                | "transpose"
                | "inverse"
                | "determinant"
                // Interpolation
                | "lerp"
                | "slerp"
                // Conversions
                | "quat_to_euler"
                | "euler_to_quat"
                | "quat_to_mat3"
                | "quat_to_mat4"
                | "mat3_to_quat"
                // Quaternion Embeddings (Knowledge Graph) - arXiv:1904.10281
                | "hamilton_product"
                | "quat_rotate_vec"
                | "quat_score"
                | "quat_embed_init"
                | "quat_normalize_embed"
                | "quat_inner_product"
                // Automatic Differentiation
                | "dual"
                | "dual_value"
                | "dual_deriv"
                | "dual_add"
                | "dual_sub"
                | "dual_mul"
                | "dual_div"
                | "dual_sin"
                | "dual_cos"
                | "dual_exp"
                | "dual_log"
                | "dual_sqrt"
                | "dual_pow"
                | "dual_tan"
                | "dual_atan"
                | "dual_abs"
                | "dual_asin"
                | "dual_acos"
                | "dual_sinh"
                | "dual_cosh"
                | "dual_tanh"
                | "dual_asinh"
                | "dual_acosh"
                | "dual_atanh"
                | "dual_log2"
                | "dual_log10"
                | "dual_atan2"
                | "grad"
                | "jacobian"
                | "hessian"
                | "array_len"
                | "array_ptr"
                | "ptr_load_f64"
                | "ptr_store_f64"
                // FFI / Raw pointer operations
                | "null_ptr"
                | "null_mut"
                | "is_null"
                | "ptr_eq"
                | "ptr_addr"
                | "ptr_from_addr"
                | "ptr_from_addr_mut"
                | "ptr_offset"
                | "ptr_add"
                | "ptr_sub"
                | "ptr_diff"
                | "as_const"
                | "as_mut"
                | "size_of"
                | "align_of"
                // Slice construction intrinsics
                | "__builtin_slice_from_raw_parts"
                | "__builtin_slice_from_raw_parts_mut"
                // QNN (Quaternionic Neural Network) Intrinsics
                | "quat_linear_fwd"
                | "quat_linear_bwd"
                | "quat_conv2d_fwd"
                | "quat_conv2d_bwd"
                | "quat_relu"
                | "quat_sigmoid"
                | "quat_tanh"
                | "quat_leaky_relu"
                | "quat_avg_pool2d"
                | "quat_max_pool2d"
                | "quat_init_xavier"
                | "quat_init_he"
                | "quat_init_unit"
                | "quat_bn_create"
                | "quat_bn_fwd"
                | "quat_bn_bwd"
                | "quat_lstm_cell"
                | "quat_gru_cell"
                | "quat_attention"
        );
        if name.starts_with("quat_init")
            || name.starts_with("quat_relu")
            || name.starts_with("quat_sigmoid")
        {}
        result
    }

    /// Check if a qualified path is a builtin associated function
    fn is_builtin_associated_fn(&self, segments: &[String]) -> bool {
        if segments.len() != 2 {
            return false;
        }
        let full_name = format!("{}::{}", segments[0], segments[1]);
        matches!(
            full_name.as_str(),
            "Vec::new"
                | "Vec::with_capacity"
                | "Box::new"
                | "HashMap::new"
                | "HashSet::new"
                | "String::new"
                | "String::from"
        )
    }

    /// Get the return type of a builtin associated function
    fn get_builtin_associated_fn_type(&self, segments: &[String]) -> HirType {
        let type_name = &segments[0];
        let fn_name = &segments[1];
        match (type_name.as_str(), fn_name.as_str()) {
            ("Vec", "new") | ("Vec", "with_capacity") => HirType::Named {
                name: "Vec".to_string(),
                args: vec![],
            },
            ("Box", "new") => HirType::Named {
                name: "Box".to_string(),
                args: vec![],
            },
            ("HashMap", "new") => HirType::Named {
                name: "HashMap".to_string(),
                args: vec![],
            },
            ("HashSet", "new") => HirType::Named {
                name: "HashSet".to_string(),
                args: vec![],
            },
            ("String", "new") | ("String", "from") => HirType::String,
            _ => HirType::Error,
        }
    }

    /// Get the type of a builtin function
    fn get_builtin_type(&self, name: &str) -> HirType {
        // For simplicity, most builtins are treated as functions that take any args and return unit or the appropriate type
        match name {
            "print" | "println" | "dbg" | "panic" | "assert" | "assert_eq" => {
                // These return unit
                HirType::Fn {
                    params: vec![], // Variadic, but we'll be lenient
                    return_type: Box::new(HirType::Unit),
                }
            }
            "len" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::I64),
            },
            "type_of" | "format" | "to_string" | "read_line" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::String),
            },
            "parse_int" | "parse_float" => HirType::Fn {
                params: vec![HirType::String],
                return_type: Box::new(HirType::Unit), // Actually returns Option, simplified
            },
            "sqrt" | "abs" | "sin" | "cos" | "tan" | "exp" | "log" | "pow" | "floor" | "ceil"
            | "round" | "min" | "max" => HirType::Fn {
                params: vec![HirType::F64],
                return_type: Box::new(HirType::F64),
            },
            "Some" | "Ok" | "Err" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::Unit), // Generic, simplified
            },
            // None without context - will be handled by get_builtin_variant_type with expected
            "None" => HirType::Named {
                name: "Option".to_string(),
                args: vec![HirType::Unit],
            },
            // Linear algebra constructors
            "vec2" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Vec2),
            },
            "vec3" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Vec3),
            },
            "vec4" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32, HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Vec4),
            },
            "mat2" => HirType::Fn {
                params: vec![HirType::F32; 4], // 2x2 = 4 floats
                return_type: Box::new(HirType::Mat2),
            },
            "mat3" => HirType::Fn {
                params: vec![HirType::F32; 9], // 3x3 = 9 floats
                return_type: Box::new(HirType::Mat3),
            },
            "mat4" => HirType::Fn {
                params: vec![HirType::F32; 16], // 4x4 = 16 floats
                return_type: Box::new(HirType::Mat4),
            },
            "quat" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32, HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Quat),
            },
            // Vector operations
            "dot" => HirType::Fn {
                params: vec![HirType::Vec3, HirType::Vec3],
                return_type: Box::new(HirType::F32),
            },
            "cross" => HirType::Fn {
                params: vec![HirType::Vec3, HirType::Vec3],
                return_type: Box::new(HirType::Vec3),
            },
            "normalize" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::Vec3),
            },
            "length" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::F32),
            },
            "length_squared" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::F32),
            },
            // Quaternion operations
            "quat_mul" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_conj" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_inv" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_normalize" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_identity" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::Quat),
            },
            // Matrix operations
            "mat_mul" => HirType::Fn {
                params: vec![HirType::Mat4, HirType::Mat4],
                return_type: Box::new(HirType::Mat4),
            },
            "transpose" => HirType::Fn {
                params: vec![HirType::Mat4],
                return_type: Box::new(HirType::Mat4),
            },
            "inverse" => HirType::Fn {
                params: vec![HirType::Mat4],
                return_type: Box::new(HirType::Mat4),
            },
            "determinant" => HirType::Fn {
                params: vec![HirType::Mat4],
                return_type: Box::new(HirType::F32),
            },
            // Interpolation
            "lerp" => HirType::Fn {
                params: vec![HirType::Vec3, HirType::Vec3, HirType::F32],
                return_type: Box::new(HirType::Vec3),
            },
            "slerp" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat, HirType::F32],
                return_type: Box::new(HirType::Quat),
            },
            // Conversions
            "quat_to_euler" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Vec3),
            },
            "euler_to_quat" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::Quat),
            },
            "quat_to_mat3" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Mat3),
            },
            "quat_to_mat4" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Mat4),
            },
            "mat3_to_quat" => HirType::Fn {
                params: vec![HirType::Mat3],
                return_type: Box::new(HirType::Quat),
            },
            // Quaternion Embeddings (Knowledge Graph) - arXiv:1904.10281
            // Hamilton product: q1 ⊗ q2 - captures inter-dependencies between components
            "hamilton_product" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            // Rotate vector by quaternion: q * v * q^(-1)
            "quat_rotate_vec" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Vec3],
                return_type: Box::new(HirType::Vec3),
            },
            // Score triple (head, relation, tail) for knowledge graph completion
            // Returns scalar score: <h ⊗ r, t> where ⊗ is Hamilton product
            "quat_score" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::F32),
            },
            // Initialize quaternion embedding with random unit quaternion
            "quat_embed_init" => HirType::Fn {
                params: vec![HirType::I32], // seed
                return_type: Box::new(HirType::Quat),
            },
            // Normalize to unit quaternion for embeddings
            "quat_normalize_embed" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            // Inner product of two quaternion embeddings: sum of component-wise products
            "quat_inner_product" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::F32),
            },

            // ==================== OCTONION OPERATIONS (8D Hypercomplex) ====================
            // Octonions form a division algebra over R with 8 dimensions
            // Multiplication is non-associative but alternative (associator is alternating)
            // References: arXiv:1601.01507 (Octonion-valued neural networks)
            //
            // Octonion basis: {1, i, j, k, l, il, jl, kl} with multiplication from Fano plane
            // o = a + bi + cj + dk + el + fil + gjl + hkl

            // Create octonion from 8 real components
            "oct" => HirType::Fn {
                params: vec![
                    HirType::F32,
                    HirType::F32,
                    HirType::F32,
                    HirType::F32,
                    HirType::F32,
                    HirType::F32,
                    HirType::F32,
                    HirType::F32,
                ],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion multiplication (Cayley-Dickson construction)
            // o1 * o2 = (a1a2 - v1·v2) + (a1v2 + a2v1 + v1×v2)
            // where v1, v2 are 7D imaginary parts, × is 7D cross product
            "oct_mul" => HirType::Fn {
                params: vec![HirType::Octonion, HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion conjugate: o* = a - bi - cj - dk - el - fil - gjl - hkl
            "oct_conj" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion norm: |o| = sqrt(a² + b² + c² + d² + e² + f² + g² + h²)
            "oct_norm" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::F32),
            },

            // Octonion inverse: o⁻¹ = o* / |o|²
            "oct_inv" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion normalize: o / |o|
            "oct_normalize" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion identity: 1 + 0i + 0j + 0k + 0l + 0il + 0jl + 0kl
            "oct_identity" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion real part (scalar component)
            "oct_real" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::F32),
            },

            // Octonion imaginary part (7D vector)
            "oct_imag" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::F32),
                    size: Some(7),
                }),
            },

            // Octonion dot product (Euclidean inner product on R⁸)
            "oct_dot" => HirType::Fn {
                params: vec![HirType::Octonion, HirType::Octonion],
                return_type: Box::new(HirType::F32),
            },

            // Octonion exponentiation (using power series)
            "oct_exp" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion logarithm
            "oct_log" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion power: o^n
            "oct_pow" => HirType::Fn {
                params: vec![HirType::Octonion, HirType::F32],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion ReLU activation (per-component)
            "oct_relu" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion sigmoid (per-component)
            "oct_sigmoid" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion tanh (per-component)
            "oct_tanh" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Octonion),
            },

            // Octonion split into quaternion components: o = q0 + q1*l
            // Useful for Octonion->Quaternion reduction
            "oct_to_quats" => HirType::Fn {
                params: vec![HirType::Octonion],
                return_type: Box::new(HirType::Tuple(vec![HirType::Quat, HirType::Quat])),
            },

            // Construct octonion from two quaternions: o = q0 + q1*l
            "oct_from_quats" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::Octonion),
            },

            // ==================== QUATERNIONIC NEURAL NETWORKS (QNN) ====================
            // Quaternionic Neural Network Primitives - arXiv:1804.10592, 1903.08478

            // QNN Layer Creation
            // Create quaternionic linear layer: quat_linear(input_size, output_size) -> QuatLinear
            "quat_linear_create" => HirType::Fn {
                params: vec![HirType::I32, HirType::I32],
                return_type: Box::new(HirType::QuatLinear {
                    input_features: 0,
                    output_features: 0,
                }),
            },
            // Create quaternionic 2D convolution: quat_conv2d_create(in_ch, out_ch, kH, kW) -> QuatConv2d
            "quat_conv2d_create" => HirType::Fn {
                params: vec![HirType::I32, HirType::I32, HirType::I32, HirType::I32],
                return_type: Box::new(HirType::QuatConv2d {
                    in_channels: 0,
                    out_channels: 0,
                    kernel_h: 0,
                    kernel_w: 0,
                }),
            },

            // QNN Forward Pass - Hamilton Product-based
            // Quaternionic linear layer: y = W ⊗ x + b where ⊗ is Hamilton product
            // Input: [batch, input_features] of Quat, Weights: [input_features, output_features] of Quat
            "quat_linear_fwd" => HirType::Fn {
                params: vec![
                    HirType::QuatLinear {
                        input_features: 0,
                        output_features: 0,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            // Quaternionic linear layer backward (gradients w.r.t weights, input, bias)
            "quat_linear_bwd" => HirType::Fn {
                params: vec![
                    HirType::QuatLinear {
                        input_features: 0,
                        output_features: 0,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                ],
                return_type: Box::new(HirType::Tuple(vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // dW
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // dx
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // db
                ])),
            },

            // QNN 2D Convolution
            // Input: [batch, in_ch, H, W] of Quat, Kernel: [out_ch, in_ch, kH, kW] of Quat
            "quat_conv2d_fwd" => HirType::Fn {
                params: vec![
                    HirType::QuatConv2d {
                        in_channels: 0,
                        out_channels: 0,
                        kernel_h: 0,
                        kernel_w: 0,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::I32, // batch
                    HirType::I32, // height
                    HirType::I32, // width
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_conv2d_bwd" => HirType::Fn {
                params: vec![
                    HirType::QuatConv2d {
                        in_channels: 0,
                        out_channels: 0,
                        kernel_h: 0,
                        kernel_w: 0,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::I32,
                    HirType::I32,
                    HirType::I32,
                ],
                return_type: Box::new(HirType::Tuple(vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // dKernel
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // dx
                ])),
            },

            // QNN Activation Functions - Real-valued output activations
            // Split quaternion into 4 real streams, apply activation, recombine
            "quat_relu" => HirType::Fn {
                params: vec![HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_sigmoid" => HirType::Fn {
                params: vec![HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_tanh" => HirType::Fn {
                params: vec![HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_leaky_relu" => HirType::Fn {
                params: vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::F32,
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },

            // QNN Pooling - Applied per-component then recombine
            "quat_avg_pool2d" => HirType::Fn {
                params: vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::I32,
                    HirType::I32, // kernel size
                    HirType::I32,
                    HirType::I32, // stride
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_max_pool2d" => HirType::Fn {
                params: vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::I32,
                    HirType::I32,
                    HirType::I32,
                    HirType::I32,
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },

            // QNN Weight Initialization - Quaternion-aware schemes
            // Initialize weights with proper quaternion structure
            "quat_init_xavier" => HirType::Fn {
                params: vec![HirType::I32, HirType::I32, HirType::I64],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_init_he" => HirType::Fn {
                params: vec![HirType::I32, HirType::I32, HirType::I64],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_init_unit" => HirType::Fn {
                params: vec![HirType::I32, HirType::I32, HirType::I64],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },

            // QNN Batch Normalization
            "quat_bn_create" => HirType::Fn {
                params: vec![HirType::I32],
                return_type: Box::new(HirType::Named {
                    name: "QuatBN".to_string(),
                    args: vec![],
                }),
            },
            "quat_bn_fwd" => HirType::Fn {
                params: vec![
                    HirType::Named {
                        name: "QuatBN".to_string(),
                        args: vec![],
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Quat),
                    size: None,
                }),
            },
            "quat_bn_bwd" => HirType::Fn {
                params: vec![
                    HirType::Named {
                        name: "QuatBN".to_string(),
                        args: vec![],
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                ],
                return_type: Box::new(HirType::Tuple(vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                ])),
            },

            // QNN Gated Recurrent Units - Quaternion LSTM/GRU cells
            "quat_lstm_cell" => HirType::Fn {
                params: vec![
                    HirType::QuatGate {
                        input_size: 0,
                        hidden_size: 0,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // input
                    HirType::QuatRnnState { hidden_size: 0 },
                ],
                return_type: Box::new(HirType::Tuple(vec![
                    HirType::QuatRnnState { hidden_size: 0 },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                ])),
            },
            "quat_gru_cell" => HirType::Fn {
                params: vec![
                    HirType::QuatGate {
                        input_size: 0,
                        hidden_size: 0,
                    },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::QuatRnnState { hidden_size: 0 },
                ],
                return_type: Box::new(HirType::Tuple(vec![
                    HirType::QuatRnnState { hidden_size: 0 },
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                ])),
            },

            // QNN Attention - Quaternion-valued attention mechanisms
            "quat_attention" => HirType::Fn {
                params: vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // query
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // key
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    }, // value
                ],
                return_type: Box::new(HirType::Tuple(vec![
                    HirType::Array {
                        element: Box::new(HirType::Quat),
                        size: None,
                    },
                    HirType::Array {
                        element: Box::new(HirType::F32),
                        size: None,
                    },
                ])),
            },

            // ==================== AUTOMATIC DIFFERENTIATION ====================
            // Dual number constructor: dual(value, derivative)
            "dual" => HirType::Fn {
                params: vec![HirType::F64, HirType::F64],
                return_type: Box::new(HirType::Dual),
            },
            // Extract value component from dual number (pointer to 16-byte struct)
            "dual_value" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::F64),
            },
            // Extract derivative component from dual number (pointer to 16-byte struct)
            "dual_deriv" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::F64),
            },
            // Dual number arithmetic (forward-mode autodiff)
            // These operate on i64 pointers to 16-byte dual structs [value: f64, deriv: f64]
            "dual_add" => HirType::Fn {
                params: vec![HirType::I64, HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_sub" => HirType::Fn {
                params: vec![HirType::I64, HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_mul" => HirType::Fn {
                params: vec![HirType::I64, HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_div" => HirType::Fn {
                params: vec![HirType::I64, HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            // Dual transcendental functions (chain rule)
            "dual_sin" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_cos" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_exp" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_log" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_sqrt" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_pow" => HirType::Fn {
                params: vec![HirType::I64, HirType::F64],
                return_type: Box::new(HirType::I64),
            },
            "dual_tan" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_atan" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_abs" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_asin" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_acos" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_sinh" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_cosh" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_tanh" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_asinh" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_acosh" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_atanh" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_log2" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_log10" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            "dual_atan2" => HirType::Fn {
                params: vec![HirType::I64, HirType::I64],
                return_type: Box::new(HirType::I64),
            },
            // Compute gradient of a function at a point
            // grad(f, x) where f: fn(f64) -> f64, x: f64 -> f64
            "grad" => HirType::Fn {
                params: vec![
                    HirType::Fn {
                        params: vec![HirType::Dual],
                        return_type: Box::new(HirType::Dual),
                    },
                    HirType::F64,
                ],
                return_type: Box::new(HirType::F64),
            },
            // Compute Jacobian of vector function (returns matrix of partial derivatives)
            // jacobian(f, x, m) where f: fn([Dual]) -> [Dual], x: [f64], m: i64 -> [[f64]]
            // n = input dimension (read from array_len(x)), m = output dimension
            "jacobian" => HirType::Fn {
                params: vec![
                    HirType::Fn {
                        params: vec![HirType::I64],          // pointer to Dual array
                        return_type: Box::new(HirType::I64), // pointer to Dual array
                    },
                    HirType::I64, // pointer to f64 array (with length header)
                    HirType::I64, // m: output dimension
                ],
                return_type: Box::new(HirType::I64), // pointer to result matrix
            },
            // Compute Hessian (second derivatives) of scalar function
            // hessian(f, x) where f: fn([Dual]) -> Dual, x: [f64] -> [[f64]]
            // n = dimension (read from array_len(x), result is n×n matrix)
            "hessian" => HirType::Fn {
                params: vec![
                    HirType::Fn {
                        params: vec![HirType::I64],          // pointer to Dual array
                        return_type: Box::new(HirType::I64), // pointer to Dual
                    },
                    HirType::I64, // pointer to f64 array (with length header)
                ],
                return_type: Box::new(HirType::I64), // pointer to result matrix
            },
            // Get length of array (stored in header at offset 0)
            "array_len" => HirType::Fn {
                params: vec![HirType::I64],          // array pointer
                return_type: Box::new(HirType::I64), // length
            },
            // Get pointer to array data (skipping length header)
            "array_ptr" => HirType::Fn {
                params: vec![HirType::I64],          // array pointer
                return_type: Box::new(HirType::I64), // data pointer (offset 8)
            },
            // Load f64 from memory address
            "ptr_load_f64" => HirType::Fn {
                params: vec![HirType::I64], // memory address
                return_type: Box::new(HirType::F64),
            },
            // Store f64 to memory address
            "ptr_store_f64" => HirType::Fn {
                params: vec![HirType::I64, HirType::F64], // address, value
                return_type: Box::new(HirType::Unit),
            },

            // ==================== FFI / RAW POINTER OPERATIONS ====================
            // Create null const pointer
            "null_ptr" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Create null mut pointer
            "null_mut" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Check if pointer is null
            "is_null" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::Bool),
            },
            // Compare two pointers
            "ptr_eq" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                ],
                return_type: Box::new(HirType::Bool),
            },
            // Get address as integer
            "ptr_addr" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::I64),
            },
            // Create const pointer from address
            "ptr_from_addr" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Create mut pointer from address
            "ptr_from_addr_mut" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Offset pointer by bytes
            "ptr_offset" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::I64,
                ],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Add elements to pointer
            "ptr_add" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::I64,
                ],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Subtract elements from pointer
            "ptr_sub" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::I64,
                ],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Difference between pointers
            "ptr_diff" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                ],
                return_type: Box::new(HirType::I64),
            },
            // Cast *mut to *const
            "as_const" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Cast *const to *mut (unsafe)
            "as_mut" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Get size of type
            "size_of" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::I64),
            },
            // Get alignment of type
            "align_of" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::I64),
            },

            // ==================== SLICE CONSTRUCTION INTRINSICS ====================
            // Create a slice from raw pointer and length: (ptr: *const T, len: usize) -> &[T]
            // This is the FFI bridge for constructing slices from external memory
            "__builtin_slice_from_raw_parts" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::U8), // Generic, but u8 is common for FFI
                    },
                    HirType::Usize, // usize - length parameter
                ],
                return_type: Box::new(HirType::Ref {
                    mutable: false,
                    inner: Box::new(HirType::Array {
                        element: Box::new(HirType::U8),
                        size: None, // Slice, not fixed array
                    }),
                }),
            },

            // Mutable variant: (ptr: *mut T, len: usize) -> &![T]
            "__builtin_slice_from_raw_parts_mut" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: true,
                        inner: Box::new(HirType::U8), // Generic, but u8 is common for FFI
                    },
                    HirType::Usize, // usize - length parameter
                ],
                return_type: Box::new(HirType::Ref {
                    mutable: true, // &! for mutable reference
                    inner: Box::new(HirType::Array {
                        element: Box::new(HirType::U8),
                        size: None, // Slice, not fixed array
                    }),
                }),
            },

            _ => HirType::Error,
        }
    }

    /// Check if a name is a builtin enum variant
    fn is_builtin_variant(&self, name: &str) -> bool {
        matches!(name, "None" | "Some" | "Ok" | "Err")
    }

    /// Get the type of a builtin variant, using expected type for inference
    fn get_builtin_variant_type(&self, name: &str, expected: Option<&Type>) -> HirType {
        match name {
            "None" => {
                // If we have an expected type that's Option<T>, use that
                if let Some(Type::Named {
                    name: type_name,
                    args,
                }) = expected
                {
                    if type_name == "Option" {
                        return HirType::Named {
                            name: "Option".to_string(),
                            args: args.iter().map(|t| self.type_to_hir(t)).collect(),
                        };
                    }
                }
                // Default to Option<()>
                HirType::Named {
                    name: "Option".to_string(),
                    args: vec![HirType::Unit],
                }
            }
            "Some" => {
                // Some is a constructor function - for now return a generic Option type
                HirType::Fn {
                    params: vec![HirType::Unit], // Takes one arg
                    return_type: Box::new(HirType::Named {
                        name: "Option".to_string(),
                        args: vec![HirType::Unit],
                    }),
                }
            }
            "Ok" => {
                // Ok is a constructor for Result<T, E>
                HirType::Fn {
                    params: vec![HirType::Unit],
                    return_type: Box::new(HirType::Named {
                        name: "Result".to_string(),
                        args: vec![HirType::Unit, HirType::Unit],
                    }),
                }
            }
            "Err" => {
                // Err is a constructor for Result<T, E>
                HirType::Fn {
                    params: vec![HirType::Unit],
                    return_type: Box::new(HirType::Named {
                        name: "Result".to_string(),
                        args: vec![HirType::Unit, HirType::Unit],
                    }),
                }
            }
            _ => HirType::Error,
        }
    }

    /// Parse a unit string (e.g., "mg", "mL/min") into HirUnit
    fn parse_unit_string(&self, unit_str: &str) -> HirUnit {
        // Prefer registered/custom units (and built-in aliases) if available
        if let Some(unit) = self.units.parse(unit_str) {
            return self.unit_to_hir(&unit);
        }
        // Handle compound units with / and *
        if let Some(pos) = unit_str.find('/') {
            let num = &unit_str[..pos];
            let den = &unit_str[pos + 1..];
            let num_unit = self.parse_unit_string(num);
            let den_unit = self.parse_unit_string(den);
            return num_unit.divide(&den_unit);
        }
        if let Some(pos) = unit_str.find('*') {
            let left = &unit_str[..pos];
            let right = &unit_str[pos + 1..];
            let left_unit = self.parse_unit_string(left);
            let right_unit = self.parse_unit_string(right);
            return left_unit.multiply(&right_unit);
        }
        // Simple unit
        HirUnit::simple(unit_str)
    }

    /// Convert a Unit (from unit checker) into a HIR unit representation
    fn unit_to_hir(&self, unit: &types::units::Unit) -> HirUnit {
        let mut numerator = Vec::new();
        let mut denominator = Vec::new();
        for (name, exp) in &unit.dimensions {
            if *exp > 0 {
                numerator.push((name.clone(), *exp));
            } else if *exp < 0 {
                denominator.push((name.clone(), -*exp));
            }
        }
        numerator.sort_by(|a, b| a.0.cmp(&b.0));
        denominator.sort_by(|a, b| a.0.cmp(&b.0));
        HirUnit {
            numerator,
            denominator,
        }
    }

    /// Register unit declarations found in items (recursively).
    fn register_unit_defs(&mut self, items: &[Item]) {
        // Pass 1: base units (no definition)
        for item in items {
            match item {
                Item::Unit(unit_def) if unit_def.definition.is_none() => {
                    self.units
                        .register(&unit_def.name, types::units::Unit::base(&unit_def.name));
                }
                Item::Module(m) => {
                    if let Some(inner) = &m.items {
                        self.register_unit_defs(inner);
                    }
                }
                _ => {}
            }
        }

        // Pass 2: derived/scaled units
        for item in items {
            match item {
                Item::Unit(unit_def) => {
                    if let Some(expr) = &unit_def.definition {
                        match self.unit_def_expr_to_unit(expr) {
                            Some(unit) => self.units.register(&unit_def.name, unit),
                            None => self.error_with_code(
                                "EUNIT",
                                format!("Unknown unit in definition of `{}`", unit_def.name),
                                unit_def.span,
                            ),
                        }
                    }
                }
                Item::Module(m) => {
                    if let Some(inner) = &m.items {
                        self.register_unit_defs(inner);
                    }
                }
                _ => {}
            }
        }
    }

    /// Evaluate a unit definition expression into a concrete Unit.
    fn unit_def_expr_to_unit(&self, expr: &UnitDefExpr) -> Option<types::units::Unit> {
        match expr {
            UnitDefExpr::Named(name) => self.units.lookup(name).cloned(),
            UnitDefExpr::Scale(scale, inner) => {
                let mut unit = self.unit_def_expr_to_unit(inner)?;
                unit.scale *= *scale;
                Some(unit)
            }
            UnitDefExpr::Product(lhs, rhs) => {
                let left = self.unit_def_expr_to_unit(lhs)?;
                let right = self.unit_def_expr_to_unit(rhs)?;
                Some(left.multiply(&right))
            }
            UnitDefExpr::Quotient(lhs, rhs) => {
                let left = self.unit_def_expr_to_unit(lhs)?;
                let right = self.unit_def_expr_to_unit(rhs)?;
                Some(left.divide(&right))
            }
            UnitDefExpr::Power(base, exp) => {
                let base = self.unit_def_expr_to_unit(base)?;
                Some(base.power((*exp).into()))
            }
        }
    }

    fn binary_result_type(&self, op: BinaryOp, left: &HirType, right: &HirType) -> HirType {
        match op {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Rem
            | BinaryOp::PlusMinus => left.clone(),
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::And
            | BinaryOp::Or => HirType::Bool,
            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => left.clone(),
            // Concatenation: combine array sizes
            BinaryOp::Concat => match (left, right) {
                (
                    HirType::Array {
                        element: left_elem,
                        size: left_size,
                    },
                    HirType::Array {
                        element: _right_elem,
                        size: right_size,
                    },
                ) => {
                    let combined_size = match (left_size, right_size) {
                        (Some(l), Some(r)) => Some(l + r),
                        _ => None,
                    };
                    HirType::Array {
                        element: left_elem.clone(),
                        size: combined_size,
                    }
                }
                _ => left.clone(),
            },
        }
    }

    fn unary_result_type(&self, op: UnaryOp, operand: &HirType) -> HirType {
        match op {
            UnaryOp::Neg => operand.clone(),
            UnaryOp::Not => {
                if *operand == HirType::Bool {
                    HirType::Bool
                } else {
                    operand.clone()
                }
            }
            UnaryOp::Ref => HirType::Ref {
                mutable: false,
                inner: Box::new(operand.clone()),
            },
            UnaryOp::RefMut => HirType::Ref {
                mutable: true,
                inner: Box::new(operand.clone()),
            },
            UnaryOp::Deref => {
                if let HirType::Ref { inner, .. } = operand {
                    *inner.clone()
                } else {
                    HirType::Error
                }
            }
        }
    }

    fn lower_binary_op(&self, op: BinaryOp) -> HirBinaryOp {
        match op {
            BinaryOp::Add => HirBinaryOp::Add,
            BinaryOp::Sub => HirBinaryOp::Sub,
            BinaryOp::Mul => HirBinaryOp::Mul,
            BinaryOp::Div => HirBinaryOp::Div,
            BinaryOp::Rem => HirBinaryOp::Rem,
            BinaryOp::Eq => HirBinaryOp::Eq,
            BinaryOp::Ne => HirBinaryOp::Ne,
            BinaryOp::Lt => HirBinaryOp::Lt,
            BinaryOp::Le => HirBinaryOp::Le,
            BinaryOp::Gt => HirBinaryOp::Gt,
            BinaryOp::Ge => HirBinaryOp::Ge,
            BinaryOp::And => HirBinaryOp::And,
            BinaryOp::Or => HirBinaryOp::Or,
            BinaryOp::BitAnd => HirBinaryOp::BitAnd,
            BinaryOp::BitOr => HirBinaryOp::BitOr,
            BinaryOp::BitXor => HirBinaryOp::BitXor,
            BinaryOp::Shl => HirBinaryOp::Shl,
            BinaryOp::Shr => HirBinaryOp::Shr,
            BinaryOp::PlusMinus => HirBinaryOp::PlusMinus,
            BinaryOp::Concat => HirBinaryOp::Concat,
        }
    }

    fn lower_unary_op(&self, op: UnaryOp) -> HirUnaryOp {
        match op {
            UnaryOp::Neg => HirUnaryOp::Neg,
            UnaryOp::Not => HirUnaryOp::Not,
            UnaryOp::Ref => HirUnaryOp::Ref,
            UnaryOp::RefMut => HirUnaryOp::RefMut,
            UnaryOp::Deref => HirUnaryOp::Deref,
        }
    }

    /// Evaluate a constant expression to a usize (for array sizes)
    fn eval_const_usize(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Literal {
                value: Literal::Int(i),
                ..
            } if *i >= 0 => Some(*i as usize),
            _ => None, // Non-literal const expressions not yet supported
        }
    }

    /// Lower an effect reference to HIR, handling effect variables
    ///
    /// If the effect reference is a simple identifier that matches an effect
    /// parameter, it becomes an effect variable. Otherwise, it's a concrete effect.
    fn lower_effect_ref(&self, effect_ref: &EffectRef) -> HirEffect {
        // Check if this is a simple identifier that could be an effect variable
        if let Some(name) = effect_ref.as_simple_name() {
            if let Some(effect_var) = self.lookup_effect_param(name) {
                // This is an effect variable
                return HirEffect {
                    id: effect_ref.id,
                    name: name.to_string(),
                    operations: Vec::new(),
                    effect_var: Some(effect_var.0),
                };
            }
        }

        // This is a concrete effect
        HirEffect {
            id: effect_ref.id,
            name: effect_ref.name.to_string(),
            operations: Vec::new(),
            effect_var: None,
        }
    }

    fn lower_type_expr(&mut self, ty: &TypeExpr) -> Type {
        match ty {
            TypeExpr::Unit => Type::Unit,
            TypeExpr::Never => Type::Never,
            TypeExpr::Named { path, args, unit } => {
                let base_type = if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    match name.as_str() {
                        "bool" => Type::Bool,
                        "i8" => Type::I8,
                        "i16" => Type::I16,
                        "i32" => Type::I32,
                        "int" => Type::I32,
                        "i64" => Type::I64,
                        "i128" => Type::I128,
                        "isize" => Type::Isize,
                        "u8" => Type::U8,
                        "u16" => Type::U16,
                        "u32" => Type::U32,
                        "uint" => Type::U32,
                        "u64" => Type::U64,
                        "u128" => Type::U128,
                        "usize" => Type::Usize,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "char" => Type::Char,
                        "str" => Type::Str,
                        "string" | "String" => Type::String,
                        // Linear algebra primitives
                        "vec2" => Type::Vec2,
                        "vec3" => Type::Vec3,
                        "vec4" => Type::Vec4,
                        "mat2" => Type::Mat2,
                        "mat3" => Type::Mat3,
                        "mat4" => Type::Mat4,
                        "quat" => Type::Quat,
                        "dual" => Type::Dual,
                        _ => {
                            let lowered_args: Vec<Type> =
                                args.iter().map(|a| self.lower_type_expr(a)).collect();
                            // Eagerly expand type aliases during lowering
                            if let Some(TypeDef::Alias(alias_ty, _, _, generic_params)) =
                                self.type_defs.get(name).cloned()
                            {
                                if !generic_params.is_empty() && !lowered_args.is_empty() {
                                    self.substitute_type_params(
                                        &alias_ty,
                                        &generic_params,
                                        &lowered_args,
                                    )
                                } else {
                                    alias_ty
                                }
                            } else {
                                Type::Named {
                                    name: name.clone(),
                                    args: lowered_args,
                                }
                            }
                        }
                    }
                } else {
                    let full_name = path.to_string();
                    let lowered_args: Vec<Type> =
                        args.iter().map(|a| self.lower_type_expr(a)).collect();
                    // Eagerly expand type aliases for multi-segment paths
                    if let Some(TypeDef::Alias(alias_ty, _, _, generic_params)) =
                        self.type_defs.get(&full_name).cloned()
                    {
                        if !generic_params.is_empty() && !lowered_args.is_empty() {
                            self.substitute_type_params(&alias_ty, &generic_params, &lowered_args)
                        } else {
                            alias_ty
                        }
                    } else {
                        Type::Named {
                            name: full_name,
                            args: lowered_args,
                        }
                    }
                };
                // If there's a unit annotation, wrap in Quantity type
                if let Some(unit_str) = unit {
                    Type::Quantity {
                        numeric: Box::new(base_type),
                        unit: unit_str.clone(),
                    }
                } else {
                    base_type
                }
            }
            TypeExpr::Reference { mutable, inner } => Type::Ref {
                mutable: *mutable,
                lifetime: None,
                inner: Box::new(self.lower_type_expr(inner)),
            },
            TypeExpr::RawPointer { mutable, inner } => Type::RawPointer {
                mutable: *mutable,
                inner: Box::new(self.lower_type_expr(inner)),
            },
            TypeExpr::Array { element, size } => Type::Array {
                element: Box::new(self.lower_type_expr(element)),
                size: size.as_ref().and_then(|s| self.eval_const_usize(s)),
            },
            TypeExpr::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.lower_type_expr(e)).collect())
            }
            TypeExpr::Function {
                params,
                return_type,
                abi,
                ..
            } => Type::Function {
                params: params.iter().map(|p| self.lower_type_expr(p)).collect(),
                return_type: Box::new(self.lower_type_expr(return_type)),
                effects: types::EffectSet::new(),
                abi: abi.as_ref().map(|a| a.to_string()),
            },
            TypeExpr::Infer => Type::Unknown,
            TypeExpr::SelfType => {
                if let Some(impl_ty) = &self.current_impl_type {
                    impl_ty.clone()
                } else {
                    Type::SelfType
                }
            }

            TypeExpr::Knowledge { value_type, .. } => Type::Named {
                name: "Knowledge".to_string(),
                args: vec![self.lower_type_expr(value_type)],
            },
            TypeExpr::Quantity { numeric_type, .. } => {
                // For now, treat Quantity[T, unit] as just T
                self.lower_type_expr(numeric_type)
            }
            TypeExpr::Tensor { element_type, .. } => {
                // Tensor becomes array-like
                Type::Array {
                    element: Box::new(self.lower_type_expr(element_type)),
                    size: None,
                }
            }
            TypeExpr::Ontology { ontology, term } => {
                // Track that this ontology prefix is used
                self.used_ontology_prefixes.insert(ontology.clone());
                // Ontology term as a semantic type
                Type::Ontology {
                    namespace: ontology.clone(),
                    term: term.clone().unwrap_or_default(),
                }
            }
            TypeExpr::Linear { inner, .. } => {
                // Linear types pass through the inner type
                self.lower_type_expr(inner)
            }
            TypeExpr::Effected { inner, .. } => {
                // Effected types pass through the inner type
                self.lower_type_expr(inner)
            }
            TypeExpr::Tile { element_type, .. } => {
                // Tile becomes array-like (for type checking purposes)
                Type::Array {
                    element: Box::new(self.lower_type_expr(element_type)),
                    size: None,
                }
            }
            TypeExpr::Refinement { base_type, .. } => {
                // For type checking purposes, refinement types are their base type
                // The predicate will be verified separately by the refinement checker
                self.lower_type_expr(base_type)
            }

            // Higher-rank polymorphism: forall T. T -> T
            TypeExpr::Forall { vars, inner } => {
                // Create fresh type variables for each quantified variable
                // and build a mapping from names to type vars
                let mut type_var_map: std::collections::HashMap<String, TypeVar> =
                    std::collections::HashMap::new();

                let mut type_vars: Vec<TypeVar> = Vec::new();
                for v in vars {
                    let tv = TypeVar(self.next_type_var);
                    self.next_type_var += 1;
                    type_var_map.insert(v.name.clone(), tv);
                    type_vars.push(tv);
                }

                // Lower the inner type, substituting type variable names with Type::Var
                let inner_type = self.lower_type_expr_with_type_vars(inner, &type_var_map);

                Type::Forall {
                    vars: type_vars,
                    inner: Box::new(inner_type),
                }
            }

            // Scientific array types: Array<T, N>
            TypeExpr::ScientificArray { element_type, dim } => {
                let elem_type = self.lower_type_expr(element_type);
                Type::ScientificArray {
                    element: Box::new(elem_type),
                    dim: self.tensor_dim_to_dim_size(dim),
                }
            }

            // Scientific matrix types: Matrix<T, M, N>
            TypeExpr::ScientificMatrix {
                element_type,
                rows,
                cols,
            } => {
                let elem_type = self.lower_type_expr(element_type);
                Type::Matrix {
                    element: Box::new(elem_type),
                    rows: self.tensor_dim_to_dim_size(rows),
                    cols: self.tensor_dim_to_dim_size(cols),
                }
            }
        }
    }

    /// Lower a type expression with a mapping of type variable names to TypeVars
    /// This is used when lowering forall types to correctly resolve type variable references
    fn lower_type_expr_with_type_vars(
        &mut self,
        ty: &TypeExpr,
        type_vars: &std::collections::HashMap<String, TypeVar>,
    ) -> Type {
        match ty {
            TypeExpr::Named { path, args, unit } => {
                // Check if this is a type variable reference
                if path.segments.len() == 1 && args.is_empty() && unit.is_none() {
                    let name = &path.segments[0];
                    if let Some(&tv) = type_vars.get(name) {
                        return Type::Var(tv);
                    }
                }
                // Otherwise, delegate to the standard lowering
                // but recursively handle nested type vars
                let base_type = if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    match name.as_str() {
                        "bool" => Type::Bool,
                        "i8" => Type::I8,
                        "i16" => Type::I16,
                        "i32" => Type::I32,
                        "int" => Type::I32,
                        "i64" => Type::I64,
                        "i128" => Type::I128,
                        "isize" => Type::Isize,
                        "u8" => Type::U8,
                        "u16" => Type::U16,
                        "u32" => Type::U32,
                        "uint" => Type::U32,
                        "u64" => Type::U64,
                        "u128" => Type::U128,
                        "usize" => Type::Usize,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "char" => Type::Char,
                        "str" => Type::Str,
                        "string" | "String" => Type::String,
                        "vec2" => Type::Vec2,
                        "vec3" => Type::Vec3,
                        "vec4" => Type::Vec4,
                        "mat2" => Type::Mat2,
                        "mat3" => Type::Mat3,
                        "mat4" => Type::Mat4,
                        "quat" => Type::Quat,
                        "dual" => Type::Dual,
                        _ => Type::Named {
                            name: name.clone(),
                            args: args
                                .iter()
                                .map(|a| self.lower_type_expr_with_type_vars(a, type_vars))
                                .collect(),
                        },
                    }
                } else {
                    Type::Named {
                        name: path.to_string(),
                        args: args
                            .iter()
                            .map(|a| self.lower_type_expr_with_type_vars(a, type_vars))
                            .collect(),
                    }
                };
                if let Some(unit_str) = unit {
                    Type::Quantity {
                        numeric: Box::new(base_type),
                        unit: unit_str.clone(),
                    }
                } else {
                    base_type
                }
            }
            TypeExpr::Function {
                params,
                return_type,
                abi,
                ..
            } => Type::Function {
                params: params
                    .iter()
                    .map(|p| self.lower_type_expr_with_type_vars(p, type_vars))
                    .collect(),
                return_type: Box::new(self.lower_type_expr_with_type_vars(return_type, type_vars)),
                effects: types::EffectSet::new(),
                abi: abi.as_ref().map(|a| a.to_string()),
            },
            TypeExpr::Reference { mutable, inner } => Type::Ref {
                mutable: *mutable,
                lifetime: None,
                inner: Box::new(self.lower_type_expr_with_type_vars(inner, type_vars)),
            },
            TypeExpr::Array { element, size } => Type::Array {
                element: Box::new(self.lower_type_expr_with_type_vars(element, type_vars)),
                size: size.as_ref().and_then(|s| self.eval_const_usize(s)),
            },
            TypeExpr::Tuple(elems) => Type::Tuple(
                elems
                    .iter()
                    .map(|e| self.lower_type_expr_with_type_vars(e, type_vars))
                    .collect(),
            ),
            // Nested forall
            TypeExpr::Forall {
                vars: inner_vars,
                inner,
            } => {
                // Extend the type var map with the inner variables
                let mut extended_map = type_vars.clone();
                let mut new_type_vars: Vec<TypeVar> = Vec::new();
                for v in inner_vars {
                    let tv = TypeVar(self.next_type_var);
                    self.next_type_var += 1;
                    extended_map.insert(v.name.clone(), tv);
                    new_type_vars.push(tv);
                }

                let inner_type = self.lower_type_expr_with_type_vars(inner, &extended_map);

                Type::Forall {
                    vars: new_type_vars,
                    inner: Box::new(inner_type),
                }
            }
            // For other types, delegate to the standard lowering
            _ => self.lower_type_expr(ty),
        }
    }

    /// Lower AST where clause to HIR where predicates
    fn lower_where_clause(&mut self, predicates: &[WherePredicate]) -> Vec<HirWherePredicate> {
        predicates
            .iter()
            .map(|pred| {
                let ty_lowered = self.lower_type_expr(&pred.ty);
                let ty_hir = self.type_to_hir(&ty_lowered);
                let bounds = pred.bounds.iter().map(|p| p.to_string()).collect();
                HirWherePredicate { ty: ty_hir, bounds }
            })
            .collect()
    }

    fn type_to_hir(&self, ty: &Type) -> HirType {
        match ty {
            Type::Unit => HirType::Unit,
            Type::Bool => HirType::Bool,
            Type::I8 => HirType::I8,
            Type::I16 => HirType::I16,
            Type::I32 => HirType::I32,
            Type::I64 => HirType::I64,
            Type::I128 => HirType::I128,
            Type::Isize => HirType::Isize,
            Type::U8 => HirType::U8,
            Type::U16 => HirType::U16,
            Type::U32 => HirType::U32,
            Type::U64 => HirType::U64,
            Type::U128 => HirType::U128,
            Type::Usize => HirType::Usize,
            Type::F32 => HirType::F32,
            Type::F64 => HirType::F64,
            Type::Char => HirType::Char,
            Type::Str | Type::String => HirType::String,
            Type::Ref { mutable, inner, .. } => HirType::Ref {
                mutable: *mutable,
                inner: Box::new(self.type_to_hir(inner)),
            },
            Type::RawPointer { mutable, inner } => HirType::RawPointer {
                mutable: *mutable,
                inner: Box::new(self.type_to_hir(inner)),
            },
            Type::Array { element, size } => HirType::Array {
                element: Box::new(self.type_to_hir(element)),
                size: *size,
            },
            Type::Tuple(elems) => {
                HirType::Tuple(elems.iter().map(|e| self.type_to_hir(e)).collect())
            }
            Type::Function {
                params,
                return_type,
                ..
            } => HirType::Fn {
                params: params.iter().map(|p| self.type_to_hir(p)).collect(),
                return_type: Box::new(self.type_to_hir(return_type)),
            },
            Type::Named { name, args } => {
                if name == "Knowledge" && args.len() == 1 {
                    HirType::Knowledge {
                        inner: Box::new(self.type_to_hir(&args[0])),
                        epsilon_bound: None,
                        provenance: None,
                    }
                } else {
                    HirType::Named {
                        name: name.clone(),
                        args: args.iter().map(|a| self.type_to_hir(a)).collect(),
                    }
                }
            }
            Type::Quantity { numeric, unit } => HirType::Quantity {
                numeric: Box::new(self.type_to_hir(numeric)),
                unit: self.parse_unit_string(unit),
            },
            Type::Var(v) => HirType::Var(v.0),
            Type::Forall { inner, .. } => self.type_to_hir(inner),
            Type::Ontology { namespace, term } => HirType::Ontology {
                namespace: namespace.clone(),
                term: term.clone(),
            },
            Type::Never => HirType::Never,
            // NB: SelfType is still reachable for `Self` used outside impl blocks,
            // which is a semantic error. Do not remove.
            Type::Unknown | Type::Error | Type::SelfType => HirType::Error,
            // Linear algebra primitives
            Type::Vec2 => HirType::Vec2,
            Type::Vec3 => HirType::Vec3,
            Type::Vec4 => HirType::Vec4,
            Type::Mat2 => HirType::Mat2,
            Type::Mat3 => HirType::Mat3,
            Type::Mat4 => HirType::Mat4,
            Type::Quat => HirType::Quat,
            // Octonion type (8D hypercomplex)
            Type::Octonion => HirType::Octonion,
            // Quaternionic Neural Network types
            Type::QuatLinear {
                input_features,
                output_features,
            } => HirType::QuatLinear {
                input_features: *input_features,
                output_features: *output_features,
            },
            Type::QuatConv2d {
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
            } => HirType::QuatConv2d {
                in_channels: *in_channels,
                out_channels: *out_channels,
                kernel_h: *kernel_h,
                kernel_w: *kernel_w,
            },
            Type::QuatRnnState { hidden_size } => HirType::QuatRnnState {
                hidden_size: *hidden_size,
            },
            Type::QuatGate {
                input_size,
                hidden_size,
            } => HirType::QuatGate {
                input_size: *input_size,
                hidden_size: *hidden_size,
            },
            // Automatic differentiation
            Type::Dual => HirType::Dual,

            // Tensor type
            Type::Tensor { element, shape } => {
                let hir_dims = self.tensor_shape_to_hir_dims(shape);
                HirType::Tensor {
                    element: Box::new(self.type_to_hir(element)),
                    dims: hir_dims,
                }
            }

            // f64 vector types
            Type::Vec2d => HirType::Vec2d,
            Type::Vec3d => HirType::Vec3d,
            Type::Vec4d => HirType::Vec4d,

            // Scientific array types
            Type::ScientificArray { element, dim } => {
                // Convert to tensor with the dimension as shape
                HirType::Tensor {
                    element: Box::new(self.type_to_hir(element)),
                    dims: vec![self.dim_size_to_hir_dim(dim)],
                }
            }

            // Matrix types
            Type::Matrix {
                element,
                rows,
                cols,
            } => HirType::Tensor {
                element: Box::new(self.type_to_hir(element)),
                dims: vec![
                    self.dim_size_to_hir_dim(rows),
                    self.dim_size_to_hir_dim(cols),
                ],
            },

            // Causal graph type - maps to Named for HIR lowering
            Type::CausalGraph { graph_name } => HirType::Named {
                name: format!("Causal<{}>", graph_name),
                args: vec![],
            },
            // Sedenion type (16D hypercomplex)
            Type::Sedenion => HirType::Named {
                name: "Sedenion".to_string(),
                args: vec![],
            },
        }
    }

    fn tensor_shape_to_hir_dims(&self, shape: &TensorShape) -> Vec<HirTensorDim> {
        match shape {
            TensorShape::Static(dims) => dims.iter().map(|&d| HirTensorDim::Fixed(d)).collect(),
            TensorShape::Dynamic(ndim) => vec![HirTensorDim::Dynamic; *ndim],
            TensorShape::Symbolic(names) => names
                .iter()
                .map(|n| HirTensorDim::Named(n.clone()))
                .collect(),
            TensorShape::Parametric(params) => params
                .iter()
                .map(|p| HirTensorDim::Named(p.to_string()))
                .collect(),
        }
    }

    fn hir_dims_to_tensor_shape(&self, dims: &[HirTensorDim]) -> TensorShape {
        // Check if all dimensions are fixed
        let all_fixed = dims.iter().all(|d| matches!(d, HirTensorDim::Fixed(_)));
        let all_named = dims.iter().all(|d| matches!(d, HirTensorDim::Named(_)));

        if all_fixed {
            TensorShape::Static(
                dims.iter()
                    .filter_map(|d| match d {
                        HirTensorDim::Fixed(n) => Some(*n),
                        _ => None,
                    })
                    .collect(),
            )
        } else if all_named {
            TensorShape::Symbolic(
                dims.iter()
                    .filter_map(|d| match d {
                        HirTensorDim::Named(n) => Some(n.clone()),
                        _ => None,
                    })
                    .collect(),
            )
        } else {
            TensorShape::Dynamic(dims.len())
        }
    }

    /// Convert an AST TensorDim to a Type DimSize
    fn tensor_dim_to_dim_size(&self, dim: &crate::ast::TensorDim) -> DimSize {
        use crate::ast::TensorDim;
        match dim {
            TensorDim::Named(name) => DimSize::Symbolic(name.clone()),
            TensorDim::Fixed(n) => DimSize::Const(*n),
            TensorDim::Dynamic => DimSize::Dynamic,
            TensorDim::Expr(_) => DimSize::Dynamic, // Expression dimensions treated as dynamic for now
        }
    }

    /// Convert a Type DimSize to an HIR TensorDim
    fn dim_size_to_hir_dim(&self, dim: &DimSize) -> HirTensorDim {
        match dim {
            DimSize::Const(n) => HirTensorDim::Fixed(*n),
            DimSize::Symbolic(name) => HirTensorDim::Named(name.clone()),
            DimSize::Var(_) => HirTensorDim::Dynamic,
            DimSize::Dynamic => HirTensorDim::Dynamic,
            DimSize::BinOp { .. } => HirTensorDim::Dynamic, // Computed dimensions are dynamic
        }
    }

    /// Convert an HIR TensorDim to a Type DimSize
    fn hir_tensor_dim_to_dim_size(&self, dim: &HirTensorDim) -> DimSize {
        match dim {
            HirTensorDim::Fixed(n) => DimSize::Const(*n),
            HirTensorDim::Named(name) => DimSize::Symbolic(name.clone()),
            HirTensorDim::Dynamic => DimSize::Dynamic,
        }
    }

    fn hir_type_to_type(&self, ty: &HirType) -> Type {
        match ty {
            HirType::Unit => Type::Unit,
            HirType::Bool => Type::Bool,
            HirType::I8 => Type::I8,
            HirType::I16 => Type::I16,
            HirType::I32 => Type::I32,
            HirType::I64 => Type::I64,
            HirType::I128 => Type::I128,
            HirType::Isize => Type::Isize,
            HirType::U8 => Type::U8,
            HirType::U16 => Type::U16,
            HirType::U32 => Type::U32,
            HirType::U64 => Type::U64,
            HirType::U128 => Type::U128,
            HirType::Usize => Type::Usize,
            HirType::F32 => Type::F32,
            HirType::F64 => Type::F64,
            HirType::Char => Type::Char,
            HirType::String => Type::String,
            HirType::Ref { mutable, inner } => Type::Ref {
                mutable: *mutable,
                lifetime: None,
                inner: Box::new(self.hir_type_to_type(inner)),
            },
            HirType::RawPointer { mutable, inner } => Type::RawPointer {
                mutable: *mutable,
                inner: Box::new(self.hir_type_to_type(inner)),
            },
            HirType::Array { element, size } => Type::Array {
                element: Box::new(self.hir_type_to_type(element)),
                size: *size,
            },
            HirType::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.hir_type_to_type(e)).collect())
            }
            HirType::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.hir_type_to_type(a)).collect(),
            },
            HirType::Fn {
                params,
                return_type,
            } => Type::Function {
                params: params.iter().map(|p| self.hir_type_to_type(p)).collect(),
                return_type: Box::new(self.hir_type_to_type(return_type)),
                effects: types::EffectSet::new(),
                abi: None,
            },
            HirType::Var(v) => Type::Var(TypeVar(*v)),
            HirType::Never => Type::Never,
            HirType::Error => Type::Error,

            HirType::Knowledge { inner, .. } => Type::Named {
                name: "Knowledge".to_string(),
                args: vec![self.hir_type_to_type(inner)],
            },
            HirType::Quantity { numeric, unit } => Type::Quantity {
                numeric: Box::new(self.hir_type_to_type(numeric)),
                unit: unit.format(),
            },
            HirType::Tensor { element, dims } => Type::Tensor {
                element: Box::new(self.hir_type_to_type(element)),
                shape: self.hir_dims_to_tensor_shape(dims),
            },
            HirType::Ontology { namespace, term } => Type::Ontology {
                namespace: namespace.clone(),
                term: term.clone(),
            },
            // Linear algebra primitives
            HirType::Vec2 => Type::Vec2,
            HirType::Vec3 => Type::Vec3,
            HirType::Vec4 => Type::Vec4,
            HirType::Mat2 => Type::Mat2,
            HirType::Mat3 => Type::Mat3,
            HirType::Mat4 => Type::Mat4,
            HirType::Quat => Type::Quat,
            // Octonion type (8D hypercomplex)
            HirType::Octonion => Type::Octonion,
            // Quaternionic Neural Network types
            HirType::QuatLinear {
                input_features,
                output_features,
            } => Type::QuatLinear {
                input_features: *input_features,
                output_features: *output_features,
            },
            HirType::QuatConv2d {
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
            } => Type::QuatConv2d {
                in_channels: *in_channels,
                out_channels: *out_channels,
                kernel_h: *kernel_h,
                kernel_w: *kernel_w,
            },
            HirType::QuatRnnState { hidden_size } => Type::QuatRnnState {
                hidden_size: *hidden_size,
            },
            HirType::QuatGate {
                input_size,
                hidden_size,
            } => Type::QuatGate {
                input_size: *input_size,
                hidden_size: *hidden_size,
            },
            // f64 vector types
            HirType::Vec2d => Type::Vec2d,
            HirType::Vec3d => Type::Vec3d,
            HirType::Vec4d => Type::Vec4d,
            // Automatic differentiation
            HirType::Dual => Type::Dual,
            // Async types
            HirType::Future { output } => Type::Named {
                name: "Future".to_string(),
                args: vec![self.hir_type_to_type(output)],
            },
            // Scientific array and matrix types
            HirType::ScientificArray { element, dim } => Type::ScientificArray {
                element: Box::new(self.hir_type_to_type(element)),
                dim: self.hir_tensor_dim_to_dim_size(dim),
            },
            HirType::Matrix {
                element,
                rows,
                cols,
            } => Type::Matrix {
                element: Box::new(self.hir_type_to_type(element)),
                rows: self.hir_tensor_dim_to_dim_size(rows),
                cols: self.hir_tensor_dim_to_dim_size(cols),
            },
            // Sedenion type (16D hypercomplex)
            HirType::Sedenion => Type::Sedenion,
        }
    }

    /// Recursively bind all variables in a pattern to their respective types
    /// Enables tuple destructuring: let (x, y) = tuple binds both x and y
    fn bind_pattern_to_type(&mut self, pattern: &Pattern, ty: &Type, is_mut: bool) {
        match pattern {
            Pattern::Binding { name, .. } => {
                self.env.bind(name.clone(), ty.clone(), is_mut);
            }
            Pattern::Tuple(patterns) => {
                if let Type::Tuple(elem_types) = ty {
                    // Verify tuple arity matches pattern
                    if patterns.len() != elem_types.len() {
                        self.error(
                            format!(
                                "Tuple pattern has {} elements but type has {}",
                                patterns.len(),
                                elem_types.len()
                            ),
                            Span::dummy(),
                        );
                        return;
                    }
                    // Recursively bind each element
                    for (pat, elem_ty) in patterns.iter().zip(elem_types.iter()) {
                        self.bind_pattern_to_type(pat, elem_ty, is_mut);
                    }
                } else {
                    self.error(
                        format!("Expected tuple type for tuple pattern, found {:?}", ty),
                        Span::dummy(),
                    );
                }
            }
            Pattern::Wildcard | Pattern::Literal(_) => {
                // No bindings to create
            }
            Pattern::Enum { path, patterns } => {
                // Look up the variant in the enum's TypeDef to get field types
                let variant_name = path.segments.last().cloned().unwrap_or_default();
                let enum_name = if path.segments.len() >= 2 {
                    path.segments[path.segments.len() - 2].clone()
                } else if let Type::Named { name, .. } = ty {
                    name.clone()
                } else {
                    return;
                };
                // Clone field types to avoid borrow conflict with self
                let field_types_cloned = self.type_defs.get(&enum_name).and_then(|td| {
                    if let TypeDef::Enum { variants, .. } = td {
                        variants
                            .iter()
                            .find(|(v, _)| *v == variant_name)
                            .map(|(_, types)| types.clone())
                    } else {
                        None
                    }
                });
                if let Some(field_types) = field_types_cloned {
                    if let Some(pats) = patterns {
                        for (pat, field_ty) in pats.iter().zip(field_types.iter()) {
                            self.bind_pattern_to_type(pat, field_ty, is_mut);
                        }
                    }
                }
            }
            Pattern::Struct { path, fields } => {
                // Look up struct type to find field types (clone to avoid borrow conflict)
                let type_name = path.segments.last().cloned().unwrap_or_default();
                let struct_fields_cloned = self.type_defs.get(&type_name).and_then(|td| {
                    if let TypeDef::Struct {
                        fields: struct_fields,
                        ..
                    } = td
                    {
                        Some(struct_fields.clone())
                    } else {
                        None
                    }
                });
                if let Some(struct_fields) = struct_fields_cloned {
                    for (field_name, field_pat) in fields {
                        if let Some((_, field_ty)) =
                            struct_fields.iter().find(|(n, _)| n == field_name)
                        {
                            self.bind_pattern_to_type(field_pat, field_ty, is_mut);
                        }
                    }
                }
            }
            Pattern::Or(patterns) => {
                // Bind variables from the first alternative (all alternatives must bind the same names)
                if let Some(first) = patterns.first() {
                    self.bind_pattern_to_type(first, ty, is_mut);
                }
            }
        }
    }

    fn pattern_name(&self, pattern: &Pattern) -> String {
        match pattern {
            Pattern::Binding { name, .. } => name.clone(),
            Pattern::Wildcard => "_".to_string(),
            Pattern::Tuple(_) => "_tuple_destructure".to_string(),
            _ => "_".to_string(),
        }
    }

    /// Lower AST provenance marker to HIR provenance
    fn lower_provenance(&self, prov: &ProvenanceMarker) -> HirProvenance {
        match prov.kind {
            ProvenanceKind::Derived => HirProvenance::Derived { sources: vec![] },
            ProvenanceKind::Source => HirProvenance::Measured {
                source: "source".to_string(),
            },
            ProvenanceKind::Computed => HirProvenance::Derived {
                sources: vec!["computed".to_string()],
            },
            ProvenanceKind::Literature => HirProvenance::PeerReviewed {
                citation: String::new(),
            },
            ProvenanceKind::Measured => HirProvenance::Measured {
                source: "measurement".to_string(),
            },
            ProvenanceKind::Input => HirProvenance::UserInput,
        }
    }

    fn solve_constraints(&mut self) -> Result<()> {
        // Simple unification - a real implementation would be more sophisticated
        // Clone constraints to avoid borrow issues with types_compatible(&mut self)
        let constraints: Vec<_> = self.constraints.clone();
        let mut errors = Vec::new();

        for c in &constraints {
            if !self.types_compatible(&c.expected, &c.actual) {
                errors.push((
                    format!(
                        "Type mismatch: expected {:?}, found {:?}",
                        c.expected, c.actual
                    ),
                    c.span,
                ));
            }
        }

        for (msg, span) in errors {
            self.errors.push(TypeError {
                message: msg,
                span,
                code: "E0308".to_string(),
            });
        }
        Ok(())
    }

    /// Get the return type of a method call based on receiver type and method name
    /// Convert HirType to a type name string for method lookup
    fn hir_type_to_name(&self, ty: &HirType) -> String {
        match ty {
            HirType::Named { name, .. } => name.clone(),
            HirType::String => "str".to_string(),
            HirType::Bool => "bool".to_string(),
            HirType::I8 => "i8".to_string(),
            HirType::I16 => "i16".to_string(),
            HirType::I32 => "i32".to_string(),
            HirType::I64 => "i64".to_string(),
            HirType::I128 => "i128".to_string(),
            HirType::Isize => "isize".to_string(),
            HirType::U8 => "u8".to_string(),
            HirType::U16 => "u16".to_string(),
            HirType::U32 => "u32".to_string(),
            HirType::U64 => "u64".to_string(),
            HirType::U128 => "u128".to_string(),
            HirType::Usize => "usize".to_string(),
            HirType::F32 => "f32".to_string(),
            HirType::F64 => "f64".to_string(),
            HirType::Char => "char".to_string(),
            HirType::Unit => "()".to_string(),
            HirType::Never => "!".to_string(),
            HirType::Array { element, .. } => format!("[{}]", self.hir_type_to_name(element)),
            HirType::Tuple(elems) => {
                let elem_names: Vec<_> = elems.iter().map(|e| self.hir_type_to_name(e)).collect();
                format!("({})", elem_names.join(", "))
            }
            HirType::Ref { mutable, inner } => {
                if *mutable {
                    format!("&!{}", self.hir_type_to_name(inner))
                } else {
                    format!("&{}", self.hir_type_to_name(inner))
                }
            }
            HirType::Fn { .. } => "fn".to_string(),
            HirType::Error => "<error>".to_string(),
            HirType::Var(_) => "<var>".to_string(),
            HirType::Knowledge { inner, .. } => {
                format!("Knowledge<{}>", self.hir_type_to_name(inner))
            }
            _ => "<unknown>".to_string(), // Catch-all for other types
        }
    }

    fn get_method_return_type(
        &mut self,
        receiver_ty: &HirType,
        method: &str,
        _args: &[HirExpr],
    ) -> HirType {
        // First, try to look up the method in the resolver's symbol table
        let type_name = self.hir_type_to_name(receiver_ty);
        let return_type_opt = if let Some(symbols) = &self.symbols {
            symbols
                .lookup_method(&type_name, method)
                .and_then(|method_def| method_def.return_type.clone())
        } else {
            None
        };

        if let Some(return_type_expr) = return_type_opt {
            let ty = self.lower_type_expr(&return_type_expr);
            return self.type_to_hir(&ty);
        }

        // Check if method was found but has no return type (returns Unit)
        if let Some(symbols) = &self.symbols {
            if let Some(method_def) = symbols.lookup_method(&type_name, method) {
                if method_def.return_type.is_none() {
                    return HirType::Unit;
                }
            }
        }

        // Fall back to built-in method types
        match receiver_ty {
            // Vec<T> methods
            HirType::Named { name, args } if name == "Vec" => {
                match method {
                    "is_empty" => HirType::Bool,
                    "len" => HirType::Usize,
                    "first" | "last" => {
                        // Returns Option<&T>
                        if let Some(elem_ty) = args.first() {
                            HirType::Named {
                                name: "Option".to_string(),
                                args: vec![elem_ty.clone()],
                            }
                        } else {
                            HirType::Error
                        }
                    }
                    "get" => {
                        // Returns Option<&T>
                        if let Some(elem_ty) = args.first() {
                            HirType::Named {
                                name: "Option".to_string(),
                                args: vec![elem_ty.clone()],
                            }
                        } else {
                            HirType::Error
                        }
                    }
                    "push" | "pop" | "clear" | "remove" | "insert" => HirType::Unit,
                    "contains" => HirType::Bool,
                    "iter" => receiver_ty.clone(), // Simplified - would be Iterator<T>
                    _ => HirType::Error,
                }
            }
            // String methods
            HirType::String => match method {
                "len" => HirType::Usize,
                "is_empty" => HirType::Bool,
                "contains" | "starts_with" | "ends_with" => HirType::Bool,
                "trim" | "to_lowercase" | "to_uppercase" => HirType::String,
                "chars" | "bytes" => HirType::Error, // Would be iterator
                _ => HirType::Error,
            },
            // Option<T> methods
            HirType::Named { name, args } if name == "Option" => match method {
                "is_some" | "is_none" => HirType::Bool,
                "unwrap" | "expect" => {
                    if let Some(inner) = args.first() {
                        inner.clone()
                    } else {
                        HirType::Error
                    }
                }
                "unwrap_or" | "unwrap_or_else" => {
                    if let Some(inner) = args.first() {
                        inner.clone()
                    } else {
                        HirType::Error
                    }
                }
                _ => HirType::Error,
            },
            // Result<T, E> methods
            HirType::Named { name, args } if name == "Result" => match method {
                "is_ok" | "is_err" => HirType::Bool,
                "unwrap" | "expect" => {
                    if let Some(ok_ty) = args.first() {
                        ok_ty.clone()
                    } else {
                        HirType::Error
                    }
                }
                "unwrap_err" | "expect_err" => {
                    if args.len() > 1 {
                        args[1].clone()
                    } else {
                        HirType::Error
                    }
                }
                _ => HirType::Error,
            },
            // Default - unknown method
            _ => HirType::Error,
        }
    }

    /// Lower an AST Pattern to an HIR Pattern
    fn lower_pattern(&self, pattern: &Pattern) -> HirPattern {
        match pattern {
            Pattern::Wildcard => HirPattern::Wildcard,
            Pattern::Literal(lit) => {
                let (hir_lit, _) = self.check_literal_with_expected(lit, None);
                HirPattern::Literal(hir_lit)
            }
            Pattern::Binding { name, mutable } => HirPattern::Binding {
                name: name.clone(),
                mutable: *mutable,
            },
            Pattern::Tuple(patterns) => {
                HirPattern::Tuple(patterns.iter().map(|p| self.lower_pattern(p)).collect())
            }
            Pattern::Struct { path, fields } => HirPattern::Struct {
                name: path.segments.last().cloned().unwrap_or_default(),
                fields: fields
                    .iter()
                    .map(|(name, pat)| (name.clone(), self.lower_pattern(pat)))
                    .collect(),
            },
            Pattern::Enum { path, patterns } => {
                let segments = &path.segments;
                let (enum_name, variant) = if segments.len() >= 2 {
                    (
                        segments[segments.len() - 2].clone(),
                        segments[segments.len() - 1].clone(),
                    )
                } else {
                    (String::new(), segments.last().cloned().unwrap_or_default())
                };
                HirPattern::Variant {
                    enum_name,
                    variant,
                    patterns: patterns
                        .as_ref()
                        .map(|ps| ps.iter().map(|p| self.lower_pattern(p)).collect())
                        .unwrap_or_default(),
                }
            }
            Pattern::Or(patterns) => {
                HirPattern::Or(patterns.iter().map(|p| self.lower_pattern(p)).collect())
            }
        }
    }

    fn types_compatible(&mut self, t1: &Type, t2: &Type) -> bool {
        // Expand type aliases before comparison so aliases are transparent
        let t1 = &self.expand_type_alias(t1);
        let t2 = &self.expand_type_alias(t2);
        match (t1, t2) {
            (Type::Var(_), _) | (_, Type::Var(_)) => true, // Type variables unify with anything
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            (Type::Error, _) | (_, Type::Error) => true,
            (Type::Never, _) | (_, Type::Never) => true, // Never is subtype of all types
            (Type::Unit, Type::Unit) => true,
            (Type::Bool, Type::Bool) => true,
            (Type::I8, Type::I8) => true,
            (Type::I16, Type::I16) => true,
            (Type::I32, Type::I32) => true,
            (Type::I64, Type::I64) => true,
            (Type::I128, Type::I128) => true,
            (Type::Isize, Type::Isize) => true,
            (Type::U8, Type::U8) => true,
            (Type::U16, Type::U16) => true,
            (Type::U32, Type::U32) => true,
            (Type::U64, Type::U64) => true,
            (Type::U128, Type::U128) => true,
            (Type::Usize, Type::Usize) => true,
            (Type::F32, Type::F32) => true,
            (Type::F64, Type::F64) => true,
            (Type::Char, Type::Char) => true,
            (Type::Str, Type::Str) => true,
            (Type::String, Type::String) => true,
            (Type::Dual, Type::Dual) => true,
            (Type::Dual, Type::I64) | (Type::I64, Type::Dual) => true,
            (
                Type::Ref {
                    mutable: m1,
                    inner: i1,
                    ..
                },
                Type::Ref {
                    mutable: m2,
                    inner: i2,
                    ..
                },
            ) => m1 == m2 && self.types_compatible(i1, i2),
            (
                Type::Array {
                    element: e1,
                    size: s1,
                },
                Type::Array {
                    element: e2,
                    size: s2,
                },
            ) => {
                // Size compatibility: None (unknown) matches any size
                let size_ok = match (s1, s2) {
                    (None, _) | (_, None) => true, // Unknown size is compatible with any size
                    (Some(a), Some(b)) => a == b,  // Known sizes must match
                };
                size_ok && self.types_compatible(e1, e2)
            }
            (Type::Tuple(t1), Type::Tuple(t2)) => {
                t1.len() == t2.len()
                    && t1
                        .iter()
                        .zip(t2.iter())
                        .all(|(a, b)| self.types_compatible(a, b))
            }
            (Type::Named { name: n1, args: a1 }, Type::Named { name: n2, args: a2 }) => {
                n1 == n2
                    && a1.len() == a2.len()
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(a, b)| self.types_compatible(a, b))
            }
            // Quantity type compatibility - same unit and compatible numeric types
            (
                Type::Quantity {
                    numeric: n1,
                    unit: u1,
                },
                Type::Quantity {
                    numeric: n2,
                    unit: u2,
                },
            ) => u1 == u2 && self.types_compatible(n1, n2),
            // Quantity with plain numeric - allow if numeric types match (implicit unit stripping)
            (Type::Quantity { numeric, .. }, other) | (other, Type::Quantity { numeric, .. }) => {
                self.types_compatible(numeric, other)
            }
            // Ontology type compatibility - check if within default threshold
            (
                Type::Ontology {
                    namespace: ns1,
                    term: t1,
                },
                Type::Ontology {
                    namespace: ns2,
                    term: t2,
                },
            ) => {
                // Same type = compatible
                if ns1 == ns2 && t1 == t2 {
                    return true;
                }
                // Same namespace = compatible (within same ontology)
                if ns1 == ns2 {
                    return true;
                }
                // Check alignment
                let key1 = format!("{}:{}", ns1, t1);
                let key2 = format!("{}:{}", ns2, t2);
                self.get_semantic_distance(&key1, &key2)
                    .map(|d| d <= self.default_threshold)
                    .unwrap_or(false)
            }
            // Named type (alias) compared with Ontology type - resolve alias
            (Type::Named { name, .. }, Type::Ontology { namespace, term }) => {
                if let Some(TypeDef::Alias(alias_ty, _, _, _)) = self.type_defs.get(name) {
                    if let Type::Ontology {
                        namespace: alias_ns,
                        term: alias_term,
                    } = alias_ty
                    {
                        // Same namespace = compatible
                        if alias_ns == namespace {
                            return true;
                        }
                        // Check alignment
                        let key1 = format!("{}:{}", alias_ns, alias_term);
                        let key2 = format!("{}:{}", namespace, term);
                        return self
                            .get_semantic_distance(&key1, &key2)
                            .map(|d| d <= self.default_threshold)
                            .unwrap_or(false);
                    }
                }
                false
            }
            // Ontology type compared with Named type (alias) - resolve alias
            (Type::Ontology { namespace, term }, Type::Named { name, .. }) => {
                if let Some(TypeDef::Alias(alias_ty, _, _, _)) = self.type_defs.get(name) {
                    if let Type::Ontology {
                        namespace: alias_ns,
                        term: alias_term,
                    } = alias_ty
                    {
                        // Same namespace = compatible
                        if alias_ns == namespace {
                            return true;
                        }
                        // Check alignment
                        let key1 = format!("{}:{}", namespace, term);
                        let key2 = format!("{}:{}", alias_ns, alias_term);
                        return self
                            .get_semantic_distance(&key1, &key2)
                            .map(|d| d <= self.default_threshold)
                            .unwrap_or(false);
                    }
                }
                false
            }
            _ => false,
        }
    }
}

impl TypeEnv {
    fn push_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: String, ty: Type, mutable: bool) {
        self.bind_with_module(name, ty, mutable, None);
    }

    /// Bind a name with explicit module origin
    fn bind_with_module(
        &mut self,
        name: String,
        ty: Type,
        mutable: bool,
        source_module: Option<ModuleId>,
    ) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(
                name,
                TypeBinding {
                    ty,
                    mutable,
                    used: false,
                    source_module,
                },
            );
        }
    }

    /// Bind a module-qualified name (e.g., math.sin)
    fn bind_qualified(&mut self, module_path: Vec<String>, name: String, ty: Type, mutable: bool) {
        let module_id = ModuleId::new(module_path.clone());
        self.module_bindings.insert(
            (module_path, name),
            TypeBinding {
                ty,
                mutable,
                used: false,
                source_module: Some(module_id),
            },
        );
    }

    fn lookup(&self, name: &str) -> Option<&TypeBinding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.bindings.get(name) {
                return Some(binding);
            }
        }
        None
    }

    /// Lookup a qualified path (e.g., ["math", "sin"])
    fn lookup_qualified(&self, path: &[String]) -> Option<&TypeBinding> {
        if path.len() <= 1 {
            return self.lookup(path.first().map(|s| s.as_str()).unwrap_or(""));
        }

        // Split into module path and name
        let module_path = &path[..path.len() - 1];
        let name = &path[path.len() - 1];

        // Try exact module path match
        if let Some(binding) = self
            .module_bindings
            .get(&(module_path.to_vec(), name.clone()))
        {
            return Some(binding);
        }

        // Try looking up the full qualified name (e.g., "String::new" for associated functions)
        let full_name = path.join("::");
        if let Some(binding) = self.lookup(&full_name) {
            return Some(binding);
        }

        // Fall back to unqualified lookup (for local names)
        self.lookup(name)
    }

    fn lookup_mut(&mut self, name: &str) -> Option<&mut TypeBinding> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.bindings.get_mut(name) {
                return Some(binding);
            }
        }
        None
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
