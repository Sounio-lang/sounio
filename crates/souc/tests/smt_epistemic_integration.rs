//! SMT Integration Tests for Epistemic Confidence Predicates
//!
//! Verifies end-to-end: ConfidencePredicate → translate_confidence_to_smt →
//! Z3Solver::verify → ProofResult.
//!
//! These tests are gated on `--features smt` (requires Z3).
//!
//! Run with: BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/llvm-18/lib/clang/18/include" \
//!           cargo test --features smt --test smt_epistemic_integration -- --nocapture

#![cfg(feature = "smt")]

use sounio::dependent::TypeContext;
use sounio::dependent::predicates::{ConfidencePredicate, Predicate};
use sounio::dependent::proof_search::{ProofResult, ProofSearchConfig, ProofSearcher};
use sounio::dependent::types::ConfidenceType;

// =============================================================================
// Built-in decision procedures (baseline — these don't need Z3)
// =============================================================================

#[test]
fn builtin_literal_geq_proven() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);
    let pred =
        Predicate::confidence_geq(ConfidenceType::literal(0.97), ConfidenceType::literal(0.95));
    assert!(s.search(&pred).is_proven());
}

#[test]
fn builtin_literal_geq_disproven() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);
    let pred =
        Predicate::confidence_geq(ConfidenceType::literal(0.80), ConfidenceType::literal(0.95));
    assert!(s.search(&pred).is_disproven());
}

#[test]
fn builtin_product_bound() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);
    // 0.9 * 0.9 = 0.81 >= 0.80
    let pred = Predicate::confidence_geq(
        ConfidenceType::product(ConfidenceType::literal(0.9), ConfidenceType::literal(0.9)),
        ConfidenceType::literal(0.80),
    );
    assert!(s.search(&pred).is_proven());
}

#[test]
fn builtin_ds_combination() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);
    // DS(0.6, 0.7) = 1 - 0.4*0.3 = 0.88 >= 0.85
    let pred = Predicate::confidence_geq(
        ConfidenceType::dempster_shafer(ConfidenceType::literal(0.6), ConfidenceType::literal(0.7)),
        ConfidenceType::literal(0.85),
    );
    assert!(s.search(&pred).is_proven());
}

#[test]
fn builtin_variable_with_binding() {
    let mut ctx = TypeContext::new();
    ctx.bind_confidence("epsilon", ConfidenceType::literal(0.97));
    let mut s = ProofSearcher::new(&ctx);
    let pred = Predicate::confidence_geq(
        ConfidenceType::var("epsilon"),
        ConfidenceType::literal(0.95),
    );
    assert!(s.search(&pred).is_proven());
}

// =============================================================================
// Z3 fallback path — predicates the built-in procedures can't solve
// =============================================================================

#[test]
fn z3_unbound_variable_unknown_without_gradual() {
    // An unbound variable: built-in returns Unknown, Z3 also returns Unknown
    // (since it's universally quantified)
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);
    let pred = Predicate::confidence_geq(ConfidenceType::var("x"), ConfidenceType::literal(0.95));
    let result = s.search(&pred);
    // Z3 should say Unknown or Unsat (not all x satisfy x >= 0.95)
    assert!(!result.is_proven(), "unbound variable cannot be proven");
}

#[test]
fn z3_unbound_variable_gradual_fallback() {
    // With gradual typing enabled, unbound variables fall back to runtime check
    let ctx = TypeContext::new();
    let config = ProofSearchConfig {
        allow_gradual: true,
        ..Default::default()
    };
    let mut s = ProofSearcher::with_config(&ctx, config);
    let pred = Predicate::confidence_geq(ConfidenceType::var("x"), ConfidenceType::literal(0.95));
    let result = s.search(&pred);
    assert!(
        result.is_proven(),
        "gradual typing should allow runtime check"
    );
}

// =============================================================================
// Z3 Solver direct API tests
// =============================================================================

#[test]
fn z3_solver_simple_satisfiable() {
    use sounio::smt::formula::{SmtFormula, SmtTerm};
    use sounio::smt::solver::VerificationResult;
    use sounio::smt::z3_solver::Z3Solver;

    let config = z3::Config::new();
    let ctx = z3::Context::new(&config);
    let mut solver = Z3Solver::new(&ctx);

    // Verify: 0.97 >= 0.95 (trivially true)
    let formula = SmtFormula::Ge(Box::new(SmtTerm::Real(0.97)), Box::new(SmtTerm::Real(0.95)));
    let result = solver.verify(&formula).expect("Z3 should not error");
    assert_eq!(result, VerificationResult::Sat, "0.97 >= 0.95 should hold");
}

#[test]
fn z3_solver_unsatisfiable() {
    use sounio::smt::formula::{SmtFormula, SmtTerm};
    use sounio::smt::solver::VerificationResult;
    use sounio::smt::z3_solver::Z3Solver;

    let config = z3::Config::new();
    let ctx = z3::Context::new(&config);
    let mut solver = Z3Solver::new(&ctx);

    // Verify: 0.80 >= 0.95 (false)
    let formula = SmtFormula::Ge(Box::new(SmtTerm::Real(0.80)), Box::new(SmtTerm::Real(0.95)));
    let result = solver.verify(&formula).expect("Z3 should not error");
    assert_eq!(
        result,
        VerificationResult::Unsat,
        "0.80 >= 0.95 should not hold"
    );
}

#[test]
fn z3_solver_ds_combination_formula() {
    use sounio::smt::formula::{SmtFormula, SmtTerm};
    use sounio::smt::solver::VerificationResult;
    use sounio::smt::z3_solver::Z3Solver;

    let config = z3::Config::new();
    let ctx = z3::Context::new(&config);
    let mut solver = Z3Solver::new(&ctx);

    // DS(0.6, 0.7) = 1 - (1-0.6)(1-0.7) = 1 - 0.12 = 0.88
    // Verify: 0.88 >= 0.85
    let one = SmtTerm::Real(1.0);
    let a = SmtTerm::Real(0.6);
    let b = SmtTerm::Real(0.7);
    let one_minus_a = SmtTerm::Sub(Box::new(one.clone()), Box::new(a));
    let one_minus_b = SmtTerm::Sub(Box::new(one.clone()), Box::new(b));
    let product = SmtTerm::Mul(Box::new(one_minus_a), Box::new(one_minus_b));
    let ds_result = SmtTerm::Sub(Box::new(one), Box::new(product));

    let formula = SmtFormula::Ge(Box::new(ds_result), Box::new(SmtTerm::Real(0.85)));
    let result = solver.verify(&formula).expect("Z3 should not error");
    assert_eq!(result, VerificationResult::Sat, "DS(0.6, 0.7) >= 0.85");
}

#[test]
fn z3_solver_product_formula() {
    use sounio::smt::formula::{SmtFormula, SmtTerm};
    use sounio::smt::solver::VerificationResult;
    use sounio::smt::z3_solver::Z3Solver;

    let config = z3::Config::new();
    let ctx = z3::Context::new(&config);
    let mut solver = Z3Solver::new(&ctx);

    // 0.9 * 0.9 = 0.81 >= 0.80
    let product = SmtTerm::Mul(Box::new(SmtTerm::Real(0.9)), Box::new(SmtTerm::Real(0.9)));
    let formula = SmtFormula::Ge(Box::new(product), Box::new(SmtTerm::Real(0.80)));
    let result = solver.verify(&formula).expect("Z3 should not error");
    assert_eq!(result, VerificationResult::Sat, "0.9 * 0.9 >= 0.80");
}

#[test]
fn z3_solver_bayesian_update_formula() {
    use sounio::smt::formula::{SmtFormula, SmtTerm};
    use sounio::smt::solver::VerificationResult;
    use sounio::smt::z3_solver::Z3Solver;

    let config = z3::Config::new();
    let ctx = z3::Context::new(&config);
    let mut solver = Z3Solver::new(&ctx);

    // Bayesian: P(H|E) = P(E|H)*P(H)/P(E) = 0.9*0.3/0.4 = 0.675
    // Verify: 0.675 >= 0.60
    let likelihood = SmtTerm::Real(0.9);
    let prior = SmtTerm::Real(0.3);
    let evidence = SmtTerm::Real(0.4);
    let numerator = SmtTerm::Mul(Box::new(likelihood), Box::new(prior));
    let posterior = SmtTerm::Div(Box::new(numerator), Box::new(evidence));

    let formula = SmtFormula::Ge(Box::new(posterior), Box::new(SmtTerm::Real(0.60)));
    let result = solver.verify(&formula).expect("Z3 should not error");
    assert_eq!(
        result,
        VerificationResult::Sat,
        "Bayesian posterior >= 0.60"
    );
}

// =============================================================================
// Translation correctness tests
// =============================================================================

#[test]
fn translation_literal_roundtrip() {
    // Verify that translating a literal through the full pipeline gives correct result
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // Leq: 0.5 <= 0.8 (true)
    let pred =
        Predicate::confidence_leq(ConfidenceType::literal(0.5), ConfidenceType::literal(0.8));
    assert!(s.search(&pred).is_proven());

    // Leq: 0.9 <= 0.8 (false)
    let pred2 =
        Predicate::confidence_leq(ConfidenceType::literal(0.9), ConfidenceType::literal(0.8));
    assert!(s.search(&pred2).is_disproven());
}

#[test]
fn translation_equality() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // 0.95 = 0.95 (true by reflexivity)
    let pred =
        Predicate::confidence_eq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.95));
    assert!(s.search(&pred).is_proven());

    // 0.95 = 0.90 (false)
    let pred2 =
        Predicate::confidence_eq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
    assert!(s.search(&pred2).is_disproven());
}

#[test]
fn translation_conjunction_of_confidences() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // (0.97 >= 0.95) AND (0.88 >= 0.85)
    let p1 =
        Predicate::confidence_geq(ConfidenceType::literal(0.97), ConfidenceType::literal(0.95));
    let p2 =
        Predicate::confidence_geq(ConfidenceType::literal(0.88), ConfidenceType::literal(0.85));
    let conj = Predicate::and(p1, p2);
    assert!(s.search(&conj).is_proven());
}

#[test]
fn translation_disjunction_partial() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // (0.97 >= 0.95) OR (0.70 >= 0.90)
    // First is true, second is false; disjunction is true
    let p1 =
        Predicate::confidence_geq(ConfidenceType::literal(0.97), ConfidenceType::literal(0.95));
    let p2 =
        Predicate::confidence_geq(ConfidenceType::literal(0.70), ConfidenceType::literal(0.90));
    let disj = Predicate::or(p1, p2);
    assert!(s.search(&disj).is_proven());
}

// =============================================================================
// EpistemicVerifier API tests (Z3 backend)
// =============================================================================

#[test]
fn epistemic_verifier_bounded_uncertainty() {
    use sounio::smt::z3_solver::{
        EpistemicProperty, EpistemicVerifier, EpistemicVerifyResult, Z3EpistemicVerifier,
    };

    let mut verifier = Z3EpistemicVerifier::new();

    // Variable names must start with "epsilon_" or end with "_eps"
    // (required by verify_bounded_uncertainty's variable filter)
    verifier.declare_epsilon("epsilon_sensor_a", 0.03);
    verifier.declare_epsilon("epsilon_sensor_b", 0.04);

    // Verify all uncertainties are bounded by 0.05
    let result = verifier.verify(&EpistemicProperty::BoundedUncertainty(0.05));
    assert!(
        matches!(result, EpistemicVerifyResult::Valid),
        "All epsilons (0.03, 0.04) should be bounded by 0.05, got: {:?}",
        result
    );
}

#[test]
fn epistemic_verifier_bounded_uncertainty_fails() {
    use sounio::smt::z3_solver::{
        EpistemicProperty, EpistemicVerifier, EpistemicVerifyResult, Z3EpistemicVerifier,
    };

    let mut verifier = Z3EpistemicVerifier::new();

    verifier.declare_epsilon("epsilon_sensor_a", 0.03);
    verifier.declare_epsilon("epsilon_sensor_b", 0.06); // exceeds bound

    // sensor_b can be up to 0.06, which exceeds the 0.05 bound
    let result = verifier.verify(&EpistemicProperty::BoundedUncertainty(0.05));
    assert!(
        matches!(
            result,
            EpistemicVerifyResult::Invalid(_) | EpistemicVerifyResult::Unknown
        ),
        "Epsilon 0.06 exceeds bound 0.05 — should fail or be unknown, got: {:?}",
        result
    );
}

#[test]
fn epistemic_verifier_quadrature_correctness() {
    use sounio::smt::z3_solver::{
        EpistemicProperty, EpistemicVerifier, EpistemicVerifyResult, Z3EpistemicVerifier,
    };

    let mut verifier = Z3EpistemicVerifier::new();

    // RSS: sqrt(0.03^2 + 0.04^2) = 0.05
    verifier.declare_epsilon("epsilon_in_a", 0.03);
    verifier.declare_epsilon("epsilon_in_b", 0.04);
    verifier.declare_epsilon("epsilon_out_sum", 0.05);

    let result = verifier.verify(&EpistemicProperty::QuadratureCorrectness);
    assert!(
        matches!(result, EpistemicVerifyResult::Valid),
        "RSS check: sqrt(0.03^2 + 0.04^2) = 0.05 — output bounded by RSS of inputs, got: {:?}",
        result
    );
}

#[test]
fn epistemic_verifier_epsilon_non_widening() {
    use sounio::smt::z3_solver::{
        EpistemicProperty, EpistemicVerifier, EpistemicVerifyResult, Z3EpistemicVerifier,
    };

    let mut verifier = Z3EpistemicVerifier::new();

    // Output epsilon should not exceed input
    verifier.declare_epsilon("epsilon_in_x", 0.05);
    verifier.declare_epsilon("epsilon_out_x", 0.04); // output smaller — valid

    let result = verifier.verify(&EpistemicProperty::EpsilonNonWidening);
    assert!(
        matches!(result, EpistemicVerifyResult::Valid),
        "Output uncertainty 0.04 <= input 0.05 — non-widening holds, got: {:?}",
        result
    );
}

// =============================================================================
// Decay confidence tests
// =============================================================================

#[test]
fn builtin_decay_confidence() {
    use std::time::Duration;

    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // decay(0.95, 0.01, 10s) = 0.95 * exp(-0.1) ≈ 0.8591 >= 0.85
    let pred = Predicate::confidence_geq(
        ConfidenceType::decay(ConfidenceType::literal(0.95), 0.01, Duration::from_secs(10)),
        ConfidenceType::literal(0.85),
    );
    assert!(s.search(&pred).is_proven());
}

#[test]
fn builtin_decay_expired() {
    use std::time::Duration;

    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // decay(0.95, 0.1, 30s) = 0.95 * exp(-3.0) ≈ 0.0473 — below 0.50
    let pred = Predicate::confidence_geq(
        ConfidenceType::decay(ConfidenceType::literal(0.95), 0.1, Duration::from_secs(30)),
        ConfidenceType::literal(0.50),
    );
    assert!(s.search(&pred).is_disproven());
}

// =============================================================================
// Min/Max confidence tests
// =============================================================================

#[test]
fn builtin_min_confidence() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // min(0.95, 0.90) = 0.90 >= 0.85
    let pred = Predicate::confidence_geq(
        ConfidenceType::min(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90)),
        ConfidenceType::literal(0.85),
    );
    assert!(s.search(&pred).is_proven());
}

#[test]
fn builtin_max_confidence() {
    let ctx = TypeContext::new();
    let mut s = ProofSearcher::new(&ctx);

    // max(0.80, 0.90) = 0.90 >= 0.85
    let pred = Predicate::confidence_geq(
        ConfidenceType::max(ConfidenceType::literal(0.80), ConfidenceType::literal(0.90)),
        ConfidenceType::literal(0.85),
    );
    assert!(s.search(&pred).is_proven());
}
