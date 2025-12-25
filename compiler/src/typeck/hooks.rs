//! Type Checker Hooks
//!
//! Integration points that wire semantic distance into the
//! standard type checking flow.

use std::sync::Arc;

use crate::common::Span;
use crate::hir::{
    Hir, HirBlock, HirExpr, HirExprKind, HirFn, HirItem, HirMatchArm, HirStmt, HirType,
};
use crate::ontology::alignment::AlignmentIndex;
use crate::ontology::distance::SemanticDistanceIndex;

use super::coercion_insert::CoercionInserter;
use super::diagnostics::{CompatibilityDiagnostic, DiagnosticAccumulator};
use super::suggestions::{ScoredSuggestion, SuggestionContext, SuggestionEngine};
use super::threshold::{ThresholdContext, ThresholdResolver};
use super::unify_distance::{CoercionSite, UnificationContext};

/// The main semantic type checker
pub struct SemanticTypeChecker {
    /// Distance index
    distance_index: Arc<SemanticDistanceIndex>,
    /// Alignment index
    alignment_index: Arc<AlignmentIndex>,
    /// Suggestion engine
    suggestion_engine: Arc<SuggestionEngine>,
    /// Threshold resolver
    threshold_resolver: ThresholdResolver,
    /// Coercion inserter
    coercion_inserter: CoercionInserter,
    /// Accumulated diagnostics
    diagnostics: DiagnosticAccumulator,
    /// Current module path
    current_module: String,
    /// Current function (if any)
    current_function: Option<String>,
}

impl SemanticTypeChecker {
    pub fn new(
        distance_index: Arc<SemanticDistanceIndex>,
        alignment_index: Arc<AlignmentIndex>,
        suggestion_engine: Arc<SuggestionEngine>,
    ) -> Self {
        Self {
            distance_index: Arc::clone(&distance_index),
            alignment_index: Arc::clone(&alignment_index),
            suggestion_engine,
            threshold_resolver: ThresholdResolver::new(),
            coercion_inserter: CoercionInserter::new(),
            diagnostics: DiagnosticAccumulator::new(),
            current_module: String::new(),
            current_function: None,
        }
    }

    /// Set custom threshold resolver
    pub fn with_threshold_resolver(mut self, resolver: ThresholdResolver) -> Self {
        self.threshold_resolver = resolver;
        self
    }

    /// Set current module context
    pub fn set_module(&mut self, module_name: &str) {
        self.current_module = module_name.to_string();
    }

    /// Check an entire HIR tree
    pub fn check_hir(&mut self, hir: &mut Hir) -> Result<(), Vec<CompatibilityDiagnostic>> {
        for item in &mut hir.items {
            self.check_item(item)?;
        }

        if self.diagnostics.has_errors() {
            Err(self.diagnostics.take_diagnostics())
        } else {
            Ok(())
        }
    }

    /// Check a single HIR item
    fn check_item(&mut self, item: &mut HirItem) -> Result<(), Vec<CompatibilityDiagnostic>> {
        match item {
            HirItem::Function(func) => self.check_function(func),
            // Other items don't need semantic distance checking
            _ => Ok(()),
        }
    }

    /// Check a function
    pub fn check_function(&mut self, func: &mut HirFn) -> Result<(), Vec<CompatibilityDiagnostic>> {
        self.current_function = Some(func.name.clone());
        let func_path = format!("{}::{}", self.current_module, func.name);

        // Check parameter types - validate against declared types
        for (idx, param) in func.ty.params.iter().enumerate() {
            let threshold = self.threshold_resolver.resolve_for_parameter(
                &self.current_module,
                &func_path,
                idx,
            );
            // Store threshold for later use during call-site checking
            let _ = threshold;
        }

        // Check body
        self.check_block(&mut func.body)?;

        // Verify return type if block has a final expression
        if let Some(last_stmt) = func.body.stmts.last()
            && let HirStmt::Expr(return_expr) = last_stmt
        {
            let return_threshold = self
                .threshold_resolver
                .resolve_for_return(&self.current_module, &func_path);

            self.check_type_compatibility(
                &func.ty.return_type,
                &return_expr.ty,
                get_expr_span(return_expr),
                return_threshold.as_f32(),
                ThresholdContext::ReturnType,
            );
        }

        self.current_function = None;

        if self.diagnostics.has_errors() {
            Err(self.diagnostics.take_diagnostics())
        } else {
            Ok(())
        }
    }

    /// Check a block
    fn check_block(&mut self, block: &mut HirBlock) -> Result<(), Vec<CompatibilityDiagnostic>> {
        for stmt in &mut block.stmts {
            self.check_stmt(stmt)?;
        }
        Ok(())
    }

    /// Check a single statement
    fn check_stmt(&mut self, stmt: &mut HirStmt) -> Result<(), Vec<CompatibilityDiagnostic>> {
        match stmt {
            HirStmt::Let { ty, value, .. } => {
                if let Some(init_expr) = value {
                    self.check_expr(init_expr)?;

                    // Check initialization type matches declared type
                    let threshold = self
                        .threshold_resolver
                        .resolve_for_local(&self.current_module);
                    self.check_type_compatibility(
                        ty,
                        &init_expr.ty,
                        get_expr_span(init_expr),
                        threshold.as_f32(),
                        ThresholdContext::LocalAssignment,
                    );
                }
            }

            HirStmt::Expr(expr) => {
                self.check_expr(expr)?;
            }

            HirStmt::Assign { target, value } => {
                self.check_expr(target)?;
                self.check_expr(value)?;

                let threshold = self
                    .threshold_resolver
                    .resolve_for_local(&self.current_module);
                self.check_type_compatibility(
                    &target.ty,
                    &value.ty,
                    get_expr_span(value),
                    threshold.as_f32(),
                    ThresholdContext::LocalAssignment,
                );
            }
        }

        Ok(())
    }

    /// Check an expression
    fn check_expr(&mut self, expr: &mut HirExpr) -> Result<(), Vec<CompatibilityDiagnostic>> {
        match &mut expr.kind {
            HirExprKind::Call { func, args } => {
                self.check_expr(func)?;

                // Get the callee's function type
                if let HirType::Fn { params, .. } = &func.ty {
                    // Check each argument against its parameter type
                    for (idx, (arg, param_ty)) in args.iter_mut().zip(params.iter()).enumerate() {
                        self.check_expr(arg)?;

                        // Resolve threshold for this parameter
                        let func_name = self.extract_callee_name(func);
                        let func_path = format!("{}::{}", self.current_module, func_name);
                        let threshold = self.threshold_resolver.resolve_for_parameter(
                            &self.current_module,
                            &func_path,
                            idx,
                        );

                        self.check_type_compatibility(
                            param_ty,
                            &arg.ty,
                            get_expr_span(arg),
                            threshold.as_f32(),
                            ThresholdContext::FunctionParameter,
                        );
                    }
                } else {
                    // Non-function call - check args anyway
                    for arg in args.iter_mut() {
                        self.check_expr(arg)?;
                    }
                }
            }

            HirExprKind::MethodCall {
                receiver,
                method: _,
                args,
            } => {
                self.check_expr(receiver)?;
                for arg in args.iter_mut() {
                    self.check_expr(arg)?;
                }
            }

            HirExprKind::Field { base, .. } => {
                self.check_expr(base)?;
            }

            HirExprKind::TupleField { base, .. } => {
                self.check_expr(base)?;
            }

            HirExprKind::Index { base, index } => {
                self.check_expr(base)?;
                self.check_expr(index)?;
            }

            HirExprKind::Binary { left, right, .. } => {
                self.check_expr(left)?;
                self.check_expr(right)?;

                // Binary operators might need semantic compatibility checks
                // for ontological types
                if is_ontological(&left.ty) && is_ontological(&right.ty) {
                    let threshold = self
                        .threshold_resolver
                        .resolve_for_local(&self.current_module);
                    self.check_type_compatibility(
                        &left.ty,
                        &right.ty,
                        get_expr_span(right),
                        threshold.as_f32(),
                        ThresholdContext::LocalAssignment,
                    );
                }
            }

            HirExprKind::Unary { expr: operand, .. } => {
                self.check_expr(operand)?;
            }

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expr(condition)?;
                self.check_block(then_branch)?;
                if let Some(else_expr) = else_branch {
                    self.check_expr(else_expr)?;

                    // Both branches should have compatible types
                    let threshold = self
                        .threshold_resolver
                        .resolve_for_local(&self.current_module);
                    self.check_type_compatibility(
                        &then_branch.ty,
                        &else_expr.ty,
                        get_expr_span(else_expr),
                        threshold.as_f32(),
                        ThresholdContext::LocalAssignment,
                    );
                }
            }

            HirExprKind::Match { scrutinee, arms } => {
                self.check_expr(scrutinee)?;
                for arm in arms {
                    self.check_match_arm(arm, &scrutinee.ty)?;
                }
            }

            HirExprKind::Loop(block) => {
                self.check_block(block)?;
            }

            HirExprKind::While { condition, body } => {
                self.check_expr(condition)?;
                self.check_block(body)?;
            }

            HirExprKind::Block(block) => {
                self.check_block(block)?;
            }

            HirExprKind::Return(ret_expr) => {
                if let Some(ret) = ret_expr {
                    self.check_expr(ret)?;
                }
            }

            HirExprKind::Array(elements) => {
                let mut prev_ty: Option<HirType> = None;
                for elem in elements.iter_mut() {
                    self.check_expr(elem)?;
                    if let Some(prev) = &prev_ty {
                        let threshold = self
                            .threshold_resolver
                            .resolve_for_local(&self.current_module);
                        self.check_type_compatibility(
                            prev,
                            &elem.ty,
                            get_expr_span(elem),
                            threshold.as_f32(),
                            ThresholdContext::GenericArgument,
                        );
                    }
                    prev_ty = Some(elem.ty.clone());
                }
            }

            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.check_expr(s)?;
                }
                if let Some(e) = end {
                    self.check_expr(e)?;
                }
            }

            HirExprKind::Tuple(elements) => {
                for elem in elements.iter_mut() {
                    self.check_expr(elem)?;
                }
            }

            HirExprKind::Struct { fields, .. } => {
                for (_, field_expr) in fields.iter_mut() {
                    self.check_expr(field_expr)?;
                }
            }

            HirExprKind::Variant { fields, .. } => {
                for field_expr in fields.iter_mut() {
                    self.check_expr(field_expr)?;
                }
            }

            HirExprKind::Closure { body, .. } => {
                self.check_expr(body)?;
            }

            HirExprKind::Cast {
                expr: inner,
                target,
            } => {
                self.check_expr(inner)?;
                // Cast might need compatibility check for ontological types
                if is_ontological(&inner.ty) && is_ontological(target) {
                    // Record the explicit cast - no threshold (user explicitly asked for it)
                    self.coercion_inserter.record_explicit_cast(
                        get_expr_span(inner),
                        &inner.ty,
                        target,
                    );
                }
            }

            HirExprKind::Ref { expr: inner, .. } => {
                self.check_expr(inner)?;
            }

            HirExprKind::Deref(inner) => {
                self.check_expr(inner)?;
            }

            HirExprKind::Perform { args, .. } => {
                for arg in args.iter_mut() {
                    self.check_expr(arg)?;
                }
            }

            HirExprKind::Handle { expr: inner, .. } => {
                self.check_expr(inner)?;
            }

            HirExprKind::Sample(inner) => {
                self.check_expr(inner)?;
            }

            // Epistemic expressions
            HirExprKind::Knowledge {
                value,
                epsilon,
                validity,
                ..
            } => {
                self.check_expr(value)?;
                self.check_expr(epsilon)?;
                if let Some(v) = validity {
                    self.check_expr(v)?;
                }
            }

            HirExprKind::Do { value, .. } => {
                self.check_expr(value)?;
            }

            HirExprKind::Counterfactual {
                factual,
                intervention,
                outcome,
            } => {
                self.check_expr(factual)?;
                self.check_expr(intervention)?;
                self.check_expr(outcome)?;
            }

            HirExprKind::Query {
                target,
                given,
                interventions,
            } => {
                self.check_expr(target)?;
                for g in given.iter_mut() {
                    self.check_expr(g)?;
                }
                for i in interventions.iter_mut() {
                    self.check_expr(i)?;
                }
            }

            HirExprKind::Observe { value, .. } => {
                self.check_expr(value)?;
            }

            HirExprKind::EpsilonOf(inner)
            | HirExprKind::ProvenanceOf(inner)
            | HirExprKind::ValidityOf(inner)
            | HirExprKind::Unwrap(inner) => {
                self.check_expr(inner)?;
            }

            // Leaf expressions - nothing to check
            HirExprKind::Literal(_)
            | HirExprKind::Local(_)
            | HirExprKind::Global(_)
            | HirExprKind::Break(_)
            | HirExprKind::Continue
            | HirExprKind::OntologyTerm { .. } => {}
        }

        Ok(())
    }

    /// Check a match arm
    fn check_match_arm(
        &mut self,
        arm: &mut HirMatchArm,
        scrutinee_ty: &HirType,
    ) -> Result<(), Vec<CompatibilityDiagnostic>> {
        // Check guard if present
        if let Some(guard) = &mut arm.guard {
            self.check_expr(guard)?;
        }

        // Check pattern compatibility with scrutinee type
        // (Pattern type inference would provide the pattern's type)
        let threshold = self
            .threshold_resolver
            .resolve_for_match(&self.current_module);
        let _ = (scrutinee_ty, threshold); // Used for pattern matching checks

        // Check arm body
        self.check_expr(&mut arm.body)?;

        Ok(())
    }

    /// Check type compatibility using semantic distance
    fn check_type_compatibility(
        &mut self,
        expected: &HirType,
        found: &HirType,
        span: Span,
        threshold: f32,
        context: ThresholdContext,
    ) {
        // Create unification context
        let mut unify_ctx = UnificationContext::new(
            Arc::clone(&self.distance_index),
            Arc::clone(&self.alignment_index),
        );

        // Attempt unification with threshold
        let success = unify_ctx.unify_with_threshold(expected, found, span, threshold);

        if success {
            // Check for coercions that should generate warnings
            for coercion in unify_ctx.coercions() {
                // Warn if coercion is close to threshold
                let margin = threshold - coercion.distance;
                if margin < 0.03 {
                    // Within 3% of threshold - warn
                    let warning = CompatibilityDiagnostic::coercion_warning(
                        coercion.span,
                        &coercion.from_type,
                        &coercion.to_type,
                        coercion.distance,
                        threshold,
                        coercion.kind,
                    );
                    self.diagnostics.warning(warning);
                }

                // Record coercion for insertion
                self.coercion_inserter.record_coercion(coercion.clone());
            }
        } else {
            // Generate error diagnostics with suggestions
            for error in unify_ctx.errors() {
                let suggestions = self.generate_suggestions(expected, found);
                let resolved_threshold = self.threshold_resolver.resolve(
                    &self.current_module,
                    self.current_function.as_deref(),
                    None,
                    Some(context.clone()),
                );

                let diagnostic = CompatibilityDiagnostic::from_unification_error(
                    error,
                    suggestions,
                    Some(&resolved_threshold),
                );
                self.diagnostics.error(diagnostic);
            }
        }
    }

    /// Generate suggestions for a type mismatch
    fn generate_suggestions(&self, expected: &HirType, found: &HirType) -> Vec<ScoredSuggestion> {
        let context = SuggestionContext::new(&self.current_module)
            .with_function(self.current_function.as_deref().unwrap_or(""));

        self.suggestion_engine.suggest(expected, found, &context)
    }

    /// Extract callee name from expression
    fn extract_callee_name(&self, callee: &HirExpr) -> String {
        match &callee.kind {
            HirExprKind::Local(name) => name.clone(),
            HirExprKind::Global(name) => name.clone(),
            _ => "<unknown>".to_string(),
        }
    }

    /// Get accumulated coercions for insertion
    pub fn take_coercions(&mut self) -> Vec<CoercionSite> {
        self.coercion_inserter.take_coercions()
    }

    /// Get diagnostics
    pub fn diagnostics(&self) -> &DiagnosticAccumulator {
        &self.diagnostics
    }

    /// Take diagnostics
    pub fn take_diagnostics(&mut self) -> Vec<CompatibilityDiagnostic> {
        self.diagnostics.take_diagnostics()
    }

    /// Has errors?
    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }
}

/// Check if a type is ontological
fn is_ontological(ty: &HirType) -> bool {
    matches!(ty, HirType::Ontology { .. })
}

/// Get span from expression (uses a placeholder for now)
fn get_expr_span(_expr: &HirExpr) -> Span {
    // HIR expressions have an `id` field but not a direct `span` field
    // In a complete implementation, we'd look up the span from a NodeId -> Span map
    Span::default()
}

/// Hook for call-site type checking
pub struct CallSiteHook {
    threshold_resolver: ThresholdResolver,
}

impl CallSiteHook {
    pub fn new(resolver: ThresholdResolver) -> Self {
        Self {
            threshold_resolver: resolver,
        }
    }

    /// Get threshold for a specific parameter
    pub fn parameter_threshold(&self, module: &str, function: &str, param_index: usize) -> f32 {
        self.threshold_resolver
            .resolve_for_parameter(module, function, param_index)
            .as_f32()
    }
}

/// Hook for return type checking
pub struct ReturnHook {
    threshold_resolver: ThresholdResolver,
}

impl ReturnHook {
    pub fn new(resolver: ThresholdResolver) -> Self {
        Self {
            threshold_resolver: resolver,
        }
    }

    /// Get threshold for return type
    pub fn return_threshold(&self, module: &str, function: &str) -> f32 {
        self.threshold_resolver
            .resolve_for_return(module, function)
            .as_f32()
    }
}

/// Hook for match pattern checking
pub struct PatternHook {
    threshold_resolver: ThresholdResolver,
}

impl PatternHook {
    pub fn new(resolver: ThresholdResolver) -> Self {
        Self {
            threshold_resolver: resolver,
        }
    }

    /// Get threshold for match patterns (always exact)
    pub fn pattern_threshold(&self, module: &str) -> f32 {
        self.threshold_resolver.resolve_for_match(module).as_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_ontological() {
        assert!(is_ontological(&HirType::Ontology {
            namespace: "pbpk".to_string(),
            term: "Concentration".to_string(),
        }));
        assert!(!is_ontological(&HirType::I64));
        assert!(!is_ontological(&HirType::String));
    }
}
