// Sounio Compiler - Dependent Type Checking with Epistemic Constraints
// Validates refinement predicates and dependent type satisfaction
//
// This module bridges the refinement type system (crate::refinement) with the
// epistemic proof search engine (crate::dependent). When a refinement predicate
// involves epistemic constructs (confidence, ontology, causality, temporality),
// we delegate to the ProofSearcher for compile-time verification via decision
// procedures or SMT. For plain arithmetic predicates we fall back to the
// SimpleChecker from the refinement module.

use crate::check::TypeEnv;
use crate::common::Span;
use crate::dependent::{
    ProofResult, ProofSearchConfig, ProofSearcher, SearchStrategy, TypeContext as DepTypeContext,
};
use crate::dependent::predicates::{
    ConfidencePredicate, Predicate as EpistemicPredicate, PredicateKind,
};
use crate::dependent::types::ConfidenceType;
use crate::refinement::{Predicate, solver::SimpleChecker};
use crate::types::Type;
use std::collections::HashMap;

/// Configuration for dependent type checking
#[derive(Clone, Debug)]
pub struct DependentTypeConfig {
    /// Minimum conformal score for type satisfaction
    pub min_conformal_score: f64,
    /// Maximum refinement predicate depth to check
    pub max_predicate_depth: usize,
    /// Enable constraint tracking
    pub track_constraints: bool,
    /// Allow gradual typing fallback when proofs fail
    pub allow_gradual: bool,
    /// Maximum proof search depth
    pub proof_search_depth: usize,
}

impl Default for DependentTypeConfig {
    fn default() -> Self {
        Self {
            min_conformal_score: 0.8,
            max_predicate_depth: 10,
            track_constraints: true,
            allow_gradual: true,
            proof_search_depth: 10,
        }
    }
}

/// Dependent type with epistemic constraints
#[derive(Clone, Debug)]
pub struct DependentType {
    /// Base type
    pub base_type: Type,
    /// Refinement predicate (if any)
    pub refinement: Option<Predicate>,
    /// Epistemic predicate (confidence, ontology, causal, temporal)
    pub epistemic_predicate: Option<EpistemicPredicate>,
    /// Required conformal score
    pub conformal_threshold: f64,
    /// Constraint variables (e.g., "n" in {x: i32 | n > 0})
    pub constraint_vars: HashMap<String, String>,
}

/// Result of dependent type checking
#[derive(Debug, Clone)]
pub enum CheckResult {
    /// Predicate proven true at compile time
    Proven,
    /// Predicate proven false — type error
    Disproven { reason: String },
    /// Cannot determine — deferred to runtime (gradual typing)
    Deferred { reason: String },
}

impl CheckResult {
    pub fn is_satisfied(&self) -> bool {
        matches!(self, Self::Proven | Self::Deferred { .. })
    }
}

/// Dependent type checker with integrated proof search
pub struct DependentTypeChecker {
    config: DependentTypeConfig,
    /// Satisfied constraints cache
    satisfied_constraints: HashMap<String, bool>,
    /// Epistemic type context for proof search
    dep_ctx: DepTypeContext,
}

impl DependentTypeChecker {
    pub fn new(config: DependentTypeConfig) -> Self {
        let dep_ctx = if config.allow_gradual {
            DepTypeContext::with_gradual(
                crate::dependent::GradualMode::Permissive,
            )
        } else {
            DepTypeContext::new()
        };
        Self {
            config,
            satisfied_constraints: HashMap::new(),
            dep_ctx,
        }
    }

    /// Register a known confidence binding (e.g., from a function signature)
    pub fn bind_confidence(&mut self, name: impl Into<String>, level: f64) {
        self.dep_ctx
            .bind_confidence(name, ConfidenceType::Literal(level));
    }

    /// Register an assumed predicate in the proof context
    pub fn assume_predicate(&mut self, pred: EpistemicPredicate) {
        self.dep_ctx.assume(pred);
    }

    /// Check if a refinement predicate is satisfiable.
    ///
    /// First attempts simple syntactic checking. If the predicate involves
    /// epistemic constructs, delegates to the full proof search engine.
    pub fn check_refinement(
        &mut self,
        predicate: &Predicate,
        _env: &TypeEnv,
        _span: Span,
    ) -> Result<bool, String> {
        // Stage 1: Try simple syntactic check (handles True, False, basic arithmetic)
        if let Some(result) = SimpleChecker::check(predicate) {
            let key = format!("{:?}", predicate);
            self.satisfied_constraints.insert(key, result);
            return Ok(result);
        }

        // Stage 2: Try to interpret as an epistemic predicate and use proof search
        if let Some(ep) = translate_to_epistemic(predicate) {
            let result = self.prove_epistemic(&ep);
            let key = format!("{:?}", predicate);
            match result {
                CheckResult::Proven => {
                    self.satisfied_constraints.insert(key, true);
                    Ok(true)
                }
                CheckResult::Disproven { reason } => {
                    self.satisfied_constraints.insert(key, false);
                    Err(reason)
                }
                CheckResult::Deferred { .. } => {
                    // Gradual typing: accept with runtime check
                    self.satisfied_constraints.insert(key, true);
                    Ok(true)
                }
            }
        } else {
            // Stage 3: Cannot analyze — accept conservatively if gradual mode
            if self.config.allow_gradual {
                Ok(true)
            } else {
                Err("Cannot prove refinement predicate statically".to_string())
            }
        }
    }

    /// Prove an epistemic predicate using the proof search engine
    pub fn prove_epistemic(&self, predicate: &EpistemicPredicate) -> CheckResult {
        let search_config = ProofSearchConfig {
            max_depth: self.config.proof_search_depth,
            allow_gradual: self.config.allow_gradual,
            debug: false,
            strategy: SearchStrategy::DepthFirst,
        };
        let mut searcher = ProofSearcher::with_config(&self.dep_ctx, search_config);
        let result = searcher.search(predicate);

        match result {
            ProofResult::Proven(_proof) => CheckResult::Proven,
            ProofResult::Disproven { reason } => CheckResult::Disproven { reason },
            ProofResult::Unknown { reason } => {
                if self.config.allow_gradual {
                    CheckResult::Deferred { reason }
                } else {
                    CheckResult::Disproven {
                        reason: format!("Cannot prove: {}", reason),
                    }
                }
            }
        }
    }

    /// Check a confidence bound: verify that confidence >= threshold
    pub fn check_confidence_bound(
        &self,
        var_name: &str,
        threshold: f64,
    ) -> CheckResult {
        let pred = EpistemicPredicate::confidence_geq(
            ConfidenceType::Var(var_name.to_string()),
            ConfidenceType::Literal(threshold),
        );
        self.prove_epistemic(&pred)
    }

    /// Validate that a value satisfies a dependent type
    pub fn validate_dependent_type(
        &mut self,
        _ty: &Type,
        conformal_score: f64,
        threshold: f64,
    ) -> bool {
        conformal_score >= threshold
    }

    /// Check if a value satisfies a predicate constraint
    pub fn satisfies_constraint(&self, _value: &str, constraint: &Predicate) -> bool {
        // Try simple check first
        if let Some(result) = SimpleChecker::check(constraint) {
            return result;
        }
        // Try epistemic proof search
        if let Some(ep) = translate_to_epistemic(constraint) {
            return self.prove_epistemic(&ep).is_satisfied();
        }
        // Conservative: accept in gradual mode
        self.config.allow_gradual
    }

    /// Register a constraint variable and its type
    pub fn register_constraint_var(&mut self, name: String, ty: String) {
        // Bind to epistemic context if it looks like a confidence variable
        if ty == "confidence" || ty == "f64" {
            self.dep_ctx
                .bind_confidence(&name, ConfidenceType::Var(name.clone()));
        }
    }

    /// Check if dependent type satisfies all constraints
    pub fn check_dependent_type_constraints(
        &mut self,
        dep_type: &DependentType,
    ) -> Result<bool, String> {
        // Check epistemic predicate first (if any)
        if let Some(ref ep) = dep_type.epistemic_predicate {
            match self.prove_epistemic(ep) {
                CheckResult::Proven => {}
                CheckResult::Disproven { reason } => return Err(reason),
                CheckResult::Deferred { .. } => {
                    // Gradual: proceed with runtime check
                }
            }
        }

        // Check refinement predicate (if any)
        if let Some(ref pred) = dep_type.refinement {
            if let Some(result) = SimpleChecker::check(pred) {
                if !result {
                    return Err("Refinement predicate unsatisfiable".to_string());
                }
            } else if let Some(ep) = translate_to_epistemic(pred) {
                match self.prove_epistemic(&ep) {
                    CheckResult::Proven => {}
                    CheckResult::Disproven { reason } => return Err(reason),
                    CheckResult::Deferred { .. } => {}
                }
            }
        }

        Ok(true)
    }

    /// Get the underlying epistemic type context (for advanced callers)
    pub fn dep_context(&self) -> &DepTypeContext {
        &self.dep_ctx
    }

    /// Get a mutable reference to the epistemic type context
    pub fn dep_context_mut(&mut self) -> &mut DepTypeContext {
        &mut self.dep_ctx
    }
}

/// Refinement type validation helper
pub struct RefinementValidator {
    max_depth: usize,
    config: DependentTypeConfig,
}

impl RefinementValidator {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            config: DependentTypeConfig::default(),
        }
    }

    /// Validate refinement type: {x: T | P}
    pub fn validate_refinement(
        &self,
        _base_type: &Type,
        predicate: &Predicate,
        depth: usize,
    ) -> Result<(), String> {
        if depth > self.max_depth {
            return Err(format!(
                "Refinement depth {} exceeds maximum {}",
                depth, self.max_depth
            ));
        }

        // Try simple check
        if let Some(false) = SimpleChecker::check(predicate) {
            return Err("Refinement predicate is unsatisfiable".to_string());
        }

        // Try epistemic proof search for complex predicates
        if let Some(ep) = translate_to_epistemic(predicate) {
            let checker = DependentTypeChecker::new(self.config.clone());
            match checker.prove_epistemic(&ep) {
                CheckResult::Disproven { reason } => return Err(reason),
                _ => {}
            }
        }

        Ok(())
    }

    /// Check if predicate is provable from type
    pub fn is_provable(&self, predicate: &Predicate, _from_type: &Type) -> bool {
        // Simple check
        if let Some(result) = SimpleChecker::check(predicate) {
            return result;
        }

        // Epistemic proof search
        if let Some(ep) = translate_to_epistemic(predicate) {
            let checker = DependentTypeChecker::new(self.config.clone());
            return checker.prove_epistemic(&ep).is_satisfied();
        }

        // Conservative: accept in gradual mode
        self.config.allow_gradual
    }
}

/// Translate a refinement predicate to an epistemic predicate where possible.
///
/// This bridge function maps `crate::refinement::Predicate` (basic logical/arithmetic)
/// to `crate::dependent::predicates::Predicate` (epistemic/causal/temporal).
///
/// Returns `None` if the predicate has no epistemic interpretation.
fn translate_to_epistemic(pred: &Predicate) -> Option<EpistemicPredicate> {
    match pred {
        Predicate::True => Some(EpistemicPredicate::true_()),
        Predicate::False => Some(EpistemicPredicate::false_()),

        Predicate::Not(inner) => {
            translate_to_epistemic(inner).map(EpistemicPredicate::not)
        }

        Predicate::And(preds) => {
            let eps: Vec<_> = preds.iter().filter_map(translate_to_epistemic).collect();
            if eps.len() == preds.len() {
                // All translated successfully — fold into conjunction
                eps.into_iter()
                    .reduce(EpistemicPredicate::and)
            } else {
                None
            }
        }

        Predicate::Or(preds) => {
            let eps: Vec<_> = preds.iter().filter_map(translate_to_epistemic).collect();
            if eps.len() == preds.len() {
                eps.into_iter()
                    .reduce(EpistemicPredicate::or)
            } else {
                None
            }
        }

        Predicate::Implies(p, q) => {
            let ep = translate_to_epistemic(p)?;
            let eq = translate_to_epistemic(q)?;
            Some(EpistemicPredicate::implies(ep, eq))
        }

        // Translate arithmetic comparisons involving confidence variables
        Predicate::Atom(atom) => translate_atom_to_epistemic(atom),

        // Quantifiers
        Predicate::Forall(var, ty, body) => {
            let eb = translate_to_epistemic(body)?;
            Some(EpistemicPredicate::forall(var.clone(), ty.clone(), eb))
        }

        Predicate::Exists(var, ty, body) => {
            let eb = translate_to_epistemic(body)?;
            Some(EpistemicPredicate::exists(var.clone(), ty.clone(), eb))
        }

        // Predicate application and ITE — no epistemic interpretation
        Predicate::App(_, _) | Predicate::Ite(_, _, _) => None,
    }
}

/// Translate an atomic comparison to a confidence predicate if it involves
/// confidence-like terms (variables named "confidence", "epsilon", "ε", etc.)
fn translate_atom_to_epistemic(atom: &crate::refinement::Atom) -> Option<EpistemicPredicate> {
    use crate::refinement::CompareOp;

    let lhs_conf = term_to_confidence(&atom.lhs);
    let rhs_conf = term_to_confidence(&atom.rhs);

    match (lhs_conf, rhs_conf) {
        (Some(lc), Some(rc)) => {
            let pred = match atom.op {
                CompareOp::Ge | CompareOp::Gt => {
                    ConfidencePredicate::Geq(lc, rc)
                }
                CompareOp::Le | CompareOp::Lt => {
                    ConfidencePredicate::Leq(lc, rc)
                }
                CompareOp::Eq => {
                    ConfidencePredicate::Eq(lc, rc)
                }
                CompareOp::Ne => {
                    // ε ≠ threshold → ¬(ε = threshold)
                    return Some(EpistemicPredicate::not(
                        EpistemicPredicate::new(PredicateKind::Confidence(
                            ConfidencePredicate::Eq(lc, rc),
                        )),
                    ));
                }
            };
            Some(EpistemicPredicate::new(PredicateKind::Confidence(pred)))
        }
        _ => None,
    }
}

/// Try to interpret a refinement Term as a ConfidenceType
fn term_to_confidence(term: &crate::refinement::Term) -> Option<ConfidenceType> {
    use crate::refinement::Term;

    match term {
        Term::Var(name) => {
            // Recognize confidence-related variable names
            let lower = name.to_lowercase();
            if lower == "confidence"
                || lower == "epsilon"
                || lower == "ε"
                || lower.starts_with("conf")
                || lower.starts_with("eps")
            {
                Some(ConfidenceType::Var(name.clone()))
            } else {
                // Any variable could be a confidence in epistemic context
                Some(ConfidenceType::Var(name.clone()))
            }
        }
        Term::Float(n) => {
            // Float literal in [0, 1] range is likely a confidence threshold
            let val = *n;
            if (0.0..=1.0).contains(&val) {
                Some(ConfidenceType::Literal(val))
            } else {
                None
            }
        }
        Term::Int(n) => {
            if *n == 0 || *n == 1 {
                Some(ConfidenceType::Literal(*n as f64))
            } else {
                None
            }
        }
        // Complex terms (BinOp, Field, etc.) — could extend later
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependent_type_config_default() {
        let config = DependentTypeConfig::default();
        assert_eq!(config.min_conformal_score, 0.8);
        assert_eq!(config.max_predicate_depth, 10);
        assert!(config.allow_gradual);
    }

    #[test]
    fn test_dependent_type_checker_creation() {
        let checker = DependentTypeChecker::new(DependentTypeConfig::default());
        assert!(checker.satisfied_constraints.is_empty());
    }

    #[test]
    fn test_validate_dependent_type_above_threshold() {
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        let result = checker.validate_dependent_type(&Type::I32, 0.85_f64, 0.8_f64);
        assert!(result);
    }

    #[test]
    fn test_validate_dependent_type_below_threshold() {
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        let result = checker.validate_dependent_type(&Type::I32, 0.75_f64, 0.8_f64);
        assert!(!result);
    }

    #[test]
    fn test_refinement_validator_depth_check() {
        let validator = RefinementValidator::new(5);
        let result = validator.validate_refinement(&Type::I32, &Predicate::True, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_prove_trivially_true() {
        let checker = DependentTypeChecker::new(DependentTypeConfig::default());
        let pred = EpistemicPredicate::true_();
        match checker.prove_epistemic(&pred) {
            CheckResult::Proven => {}
            other => panic!("Expected Proven, got {:?}", other),
        }
    }

    #[test]
    fn test_prove_trivially_false() {
        let mut config = DependentTypeConfig::default();
        config.allow_gradual = false;
        let checker = DependentTypeChecker::new(config);
        let pred = EpistemicPredicate::false_();
        match checker.prove_epistemic(&pred) {
            CheckResult::Disproven { .. } => {}
            other => panic!("Expected Disproven, got {:?}", other),
        }
    }

    #[test]
    fn test_confidence_bound_with_known_binding() {
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        checker.bind_confidence("ε", 0.95);
        let result = checker.check_confidence_bound("ε", 0.90);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_confidence_bound_insufficient() {
        let mut config = DependentTypeConfig::default();
        config.allow_gradual = false;
        let mut checker = DependentTypeChecker::new(config);
        checker.bind_confidence("ε", 0.80);
        let result = checker.check_confidence_bound("ε", 0.95);
        // With a literal binding of 0.80, proof search should identify 0.80 < 0.95
        // The result depends on whether the decision procedure can evaluate this
        match result {
            CheckResult::Proven => panic!("Should not prove 0.80 >= 0.95"),
            CheckResult::Disproven { .. } | CheckResult::Deferred { .. } => {}
        }
    }

    #[test]
    fn test_translate_true_false() {
        assert!(translate_to_epistemic(&Predicate::True).is_some());
        assert!(translate_to_epistemic(&Predicate::False).is_some());
    }

    #[test]
    fn test_check_refinement_simple_true() {
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        let env = TypeEnv::default();
        let result = checker.check_refinement(&Predicate::True, &env, Span::default());
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_check_refinement_simple_false() {
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        let env = TypeEnv::default();
        let result = checker.check_refinement(&Predicate::False, &env, Span::default());
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_dependent_type_with_epistemic_predicate() {
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        checker.bind_confidence("ε", 0.95);

        let dep = DependentType {
            base_type: Type::F64,
            refinement: None,
            epistemic_predicate: Some(EpistemicPredicate::confidence_geq(
                ConfidenceType::Var("ε".to_string()),
                ConfidenceType::Literal(0.90),
            )),
            conformal_threshold: 0.8,
            constraint_vars: HashMap::new(),
        };

        let result = checker.check_dependent_type_constraints(&dep);
        assert!(result.is_ok());
    }

    #[test]
    fn test_refinement_validator_provable() {
        let validator = RefinementValidator::new(10);
        assert!(validator.is_provable(&Predicate::True, &Type::I32));
    }

    #[test]
    fn test_refinement_validator_not_provable() {
        let mut config = DependentTypeConfig::default();
        config.allow_gradual = false;
        let validator = RefinementValidator {
            max_depth: 10,
            config,
        };
        assert!(!validator.is_provable(&Predicate::False, &Type::I32));
    }
}
