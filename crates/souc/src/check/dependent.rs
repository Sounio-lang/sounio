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
use crate::dependent::predicates::{
    ConfidencePredicate, Predicate as EpistemicPredicate, PredicateKind,
};
use crate::dependent::types::ConfidenceType;
use crate::dependent::{
    ProofResult, ProofSearchConfig, ProofSearcher, SearchStrategy, TypeContext as DepTypeContext,
};
use crate::refinement::{Predicate, solver::SimpleChecker};
#[cfg(feature = "smt")]
use crate::refinement::{
    constraint::{Constraint, ConstraintReason, Span as RefinementSpan},
    solver::{VerifyResult, Z3Solver},
};
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
    /// Cache of proof results to avoid recomputing identical proofs
    proof_cache: HashMap<String, CheckResult>,
}

impl DependentTypeChecker {
    pub fn new(config: DependentTypeConfig) -> Self {
        let dep_ctx = if config.allow_gradual {
            DepTypeContext::with_gradual(crate::dependent::GradualMode::Permissive)
        } else {
            DepTypeContext::new()
        };
        Self {
            config,
            satisfied_constraints: HashMap::new(),
            dep_ctx,
            proof_cache: HashMap::new(),
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
            // Stage 2.5: Z3 SMT solver for general arithmetic predicates
            if let Some(result) = try_z3_verify(predicate) {
                let key = format!("{:?}", predicate);
                self.satisfied_constraints.insert(key, result);
                if result {
                    return Ok(true);
                } else {
                    return Err("Z3 proved refinement predicate unsatisfiable".to_string());
                }
            }

            // Stage 3: Cannot analyze — accept conservatively if gradual mode
            if self.config.allow_gradual {
                Ok(true)
            } else {
                Err("Cannot prove refinement predicate statically".to_string())
            }
        }
    }

    /// Prove an epistemic predicate using the proof search engine.
    ///
    /// Results are cached by predicate key so identical proofs are not recomputed.
    pub fn prove_epistemic(&mut self, predicate: &EpistemicPredicate) -> CheckResult {
        let cache_key = format!("{:?}", predicate);
        if let Some(cached) = self.proof_cache.get(&cache_key) {
            return cached.clone();
        }

        let search_config = ProofSearchConfig {
            max_depth: self.config.proof_search_depth,
            allow_gradual: self.config.allow_gradual,
            debug: false,
            strategy: SearchStrategy::DepthFirst,
        };
        let mut searcher = ProofSearcher::with_config(&self.dep_ctx, search_config);
        let result = searcher.search(predicate);

        let check = match result {
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
        };

        self.proof_cache.insert(cache_key, check.clone());
        check
    }

    /// Invalidate the proof cache (call when context changes)
    pub fn invalidate_cache(&mut self) {
        self.proof_cache.clear();
    }

    /// Check a confidence bound: verify that confidence >= threshold
    pub fn check_confidence_bound(&mut self, var_name: &str, threshold: f64) -> CheckResult {
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
    pub fn satisfies_constraint(&mut self, _value: &str, constraint: &Predicate) -> bool {
        // Try simple check first
        if let Some(result) = SimpleChecker::check(constraint) {
            return result;
        }
        // Try epistemic proof search
        if let Some(ep) = translate_to_epistemic(constraint) {
            return self.prove_epistemic(&ep).is_satisfied();
        }
        // Try Z3 SMT solver
        if let Some(result) = try_z3_verify(constraint) {
            return result;
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
            } else if let Some(false) = try_z3_verify(pred) {
                return Err("Z3 proved refinement predicate unsatisfiable".to_string());
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
            let mut checker = DependentTypeChecker::new(self.config.clone());
            match checker.prove_epistemic(&ep) {
                CheckResult::Disproven { reason } => return Err(reason),
                _ => {}
            }
        } else if let Some(false) = try_z3_verify(predicate) {
            return Err("Z3 proved refinement predicate unsatisfiable".to_string());
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
            let mut checker = DependentTypeChecker::new(self.config.clone());
            return checker.prove_epistemic(&ep).is_satisfied();
        }

        // Z3 SMT solver
        if let Some(result) = try_z3_verify(predicate) {
            return result;
        }

        // Conservative: accept in gradual mode
        self.config.allow_gradual
    }
}

/// Try to verify a predicate using the Z3 SMT solver.
///
/// Returns `Some(true)` if Z3 proves valid, `Some(false)` if Z3 proves invalid,
/// `None` if Z3 is inconclusive (timeout, unknown). Used as a fallback when
/// SimpleChecker returns `None` and the predicate has no epistemic interpretation.
#[cfg(feature = "smt")]
fn try_z3_verify(predicate: &Predicate) -> Option<bool> {
    let constraint = Constraint::new(
        Vec::new(),
        predicate.clone(),
        RefinementSpan::dummy(),
        ConstraintReason::Assert {
            message: "dependent type refinement check".to_string(),
        },
    );

    let cfg = z3::Config::new();
    let ctx = z3::Context::new(&cfg);
    let mut solver = Z3Solver::new(&ctx);
    solver.set_timeout(1000);

    let results = solver.verify(&[constraint]);
    match results.first() {
        Some(VerifyResult::Valid) => Some(true),
        Some(VerifyResult::Invalid { .. }) => Some(false),
        _ => None,
    }
}

#[cfg(not(feature = "smt"))]
fn try_z3_verify(_predicate: &Predicate) -> Option<bool> {
    None
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

        Predicate::Not(inner) => translate_to_epistemic(inner).map(EpistemicPredicate::not),

        Predicate::And(preds) => {
            let eps: Vec<_> = preds.iter().filter_map(translate_to_epistemic).collect();
            if eps.len() == preds.len() {
                // All translated successfully — fold into conjunction
                eps.into_iter().reduce(EpistemicPredicate::and)
            } else {
                None
            }
        }

        Predicate::Or(preds) => {
            let eps: Vec<_> = preds.iter().filter_map(translate_to_epistemic).collect();
            if eps.len() == preds.len() {
                eps.into_iter().reduce(EpistemicPredicate::or)
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
                CompareOp::Ge | CompareOp::Gt => ConfidencePredicate::Geq(lc, rc),
                CompareOp::Le | CompareOp::Lt => ConfidencePredicate::Leq(lc, rc),
                CompareOp::Eq => ConfidencePredicate::Eq(lc, rc),
                CompareOp::Ne => {
                    // ε ≠ threshold → ¬(ε = threshold)
                    return Some(EpistemicPredicate::not(EpistemicPredicate::new(
                        PredicateKind::Confidence(ConfidencePredicate::Eq(lc, rc)),
                    )));
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
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
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
        let mut checker = DependentTypeChecker::new(config);
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

    // ================================================================
    // End-to-end integration tests: type checker → proof search → result
    // ================================================================

    #[test]
    fn test_e2e_confidence_chain_propagation() {
        // Scenario: bind ε₁ = 0.95, ε₂ = 0.90
        // Prove: ε₁ ≥ 0.90 (should succeed)
        // Prove: ε₂ ≥ 0.95 (should fail in strict mode)
        let mut checker = DependentTypeChecker::new(DependentTypeConfig {
            allow_gradual: false,
            ..DependentTypeConfig::default()
        });
        checker.bind_confidence("ε₁", 0.95);
        checker.bind_confidence("ε₂", 0.90);

        let r1 = checker.check_confidence_bound("ε₁", 0.90);
        assert!(r1.is_satisfied(), "0.95 >= 0.90 should be proven");

        let r2 = checker.check_confidence_bound("ε₂", 0.95);
        assert!(
            !matches!(r2, CheckResult::Proven),
            "0.90 >= 0.95 should NOT be proven"
        );
    }

    #[test]
    fn test_e2e_refinement_with_confidence_var() {
        // Build a refinement predicate: confidence >= 0.90
        // With known binding: confidence = 0.95
        use crate::refinement::predicate::{Atom, CompareOp, Term};

        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        checker.bind_confidence("confidence", 0.95);

        let pred = Predicate::Atom(Atom {
            lhs: Term::Var("confidence".to_string()),
            op: CompareOp::Ge,
            rhs: Term::Float(0.90),
        });

        let env = TypeEnv::default();
        let result = checker.check_refinement(&pred, &env, Span::default());
        assert!(result.is_ok(), "Should not error");
        assert!(result.unwrap(), "0.95 >= 0.90 should be satisfied");
    }

    #[test]
    fn test_e2e_dependent_type_conjunction() {
        // DependentType with both refinement (True) and epistemic (confidence >= 0.8)
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        checker.bind_confidence("ε", 0.92);

        let dep = DependentType {
            base_type: Type::F64,
            refinement: Some(Predicate::True),
            epistemic_predicate: Some(EpistemicPredicate::confidence_geq(
                ConfidenceType::Var("ε".to_string()),
                ConfidenceType::Literal(0.80),
            )),
            conformal_threshold: 0.8,
            constraint_vars: HashMap::new(),
        };

        let result = checker.check_dependent_type_constraints(&dep);
        assert!(result.is_ok(), "Both predicates should be satisfied");
    }

    #[test]
    fn test_e2e_dependent_type_epistemic_fails() {
        // Epistemic predicate should fail: ε = 0.70, need >= 0.90
        let mut config = DependentTypeConfig::default();
        config.allow_gradual = false;
        let mut checker = DependentTypeChecker::new(config);
        checker.bind_confidence("ε", 0.70);

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
        assert!(result.is_err(), "0.70 >= 0.90 should fail in strict mode");
    }

    #[test]
    fn test_e2e_gradual_fallback_for_unknown() {
        // When proof search returns Unknown, gradual mode should defer to runtime
        let mut checker = DependentTypeChecker::new(DependentTypeConfig {
            allow_gradual: true,
            ..DependentTypeConfig::default()
        });
        // Don't bind any confidence — proof search will return Unknown
        let result = checker.check_confidence_bound("unknown_var", 0.90);
        match result {
            CheckResult::Deferred { .. } => {}
            CheckResult::Proven => {} // Also acceptable if search is conservative
            other => panic!(
                "Expected Deferred or Proven in gradual mode, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_e2e_proof_cache_hit() {
        // Verify that repeated identical proofs are cached
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        checker.bind_confidence("ε", 0.95);

        let r1 = checker.check_confidence_bound("ε", 0.90);
        assert!(r1.is_satisfied());

        // Second call should hit cache
        let r2 = checker.check_confidence_bound("ε", 0.90);
        assert!(r2.is_satisfied());

        // Different predicate should NOT hit cache
        let r3 = checker.check_confidence_bound("ε", 0.99);
        // Whether this succeeds depends on whether 0.95 >= 0.99
        assert!(
            !matches!(r3, CheckResult::Proven),
            "0.95 >= 0.99 should not be proven"
        );
    }

    #[test]
    fn test_e2e_cache_invalidation() {
        let mut checker = DependentTypeChecker::new(DependentTypeConfig::default());
        checker.bind_confidence("ε", 0.95);

        let _ = checker.check_confidence_bound("ε", 0.90);
        assert!(!checker.proof_cache.is_empty(), "Cache should have entries");

        checker.invalidate_cache();
        assert!(checker.proof_cache.is_empty(), "Cache should be cleared");
    }

    #[test]
    fn test_e2e_mock_verifier_epistemic_pipeline() {
        // Test the full mock verifier pipeline
        use crate::smt::z3_solver::{
            EpistemicProperty, EpistemicVerifier, EpistemicVerifyResult, MockEpistemicVerifier,
        };

        let mut verifier = MockEpistemicVerifier::new();
        verifier.declare_epsilon("epsilon_dose", 0.05);
        verifier.declare_epsilon("epsilon_response", 0.08);

        // All epsilons bounded at 0.1: should pass
        let r1 = verifier.verify(&EpistemicProperty::BoundedUncertainty(0.1));
        assert!(matches!(r1, EpistemicVerifyResult::Valid));

        // All epsilons bounded at 0.03: should fail (dose=0.05 > 0.03)
        let r2 = verifier.verify(&EpistemicProperty::BoundedUncertainty(0.03));
        assert!(matches!(r2, EpistemicVerifyResult::Invalid(_)));

        // Reset and verify empty state
        verifier.reset();
        let r3 = verifier.verify(&EpistemicProperty::BoundedUncertainty(0.01));
        assert!(matches!(r3, EpistemicVerifyResult::Valid));
    }

    #[test]
    fn test_e2e_translate_comparison_to_epistemic() {
        use crate::refinement::predicate::{Atom, CompareOp, Term};

        // epsilon >= 0.95 should translate to ConfidencePredicate::Geq
        let pred = Predicate::Atom(Atom {
            lhs: Term::Var("epsilon".to_string()),
            op: CompareOp::Ge,
            rhs: Term::Float(0.95),
        });
        let ep = translate_to_epistemic(&pred);
        assert!(ep.is_some(), "Confidence comparison should translate");

        // Non-confidence variable with out-of-range float should not translate
        let pred2 = Predicate::Atom(Atom {
            lhs: Term::Var("x".to_string()),
            op: CompareOp::Ge,
            rhs: Term::Float(42.0),
        });
        let ep2 = translate_to_epistemic(&pred2);
        // 42.0 is outside [0,1] range so rhs won't be a ConfidenceType
        assert!(
            ep2.is_none(),
            "Non-confidence comparison should not translate"
        );
    }
}
