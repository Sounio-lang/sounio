//! E2E Composition Tests: Causal + Epistemic + SMT
//!
//! Proves that the three type-checking subsystems compose correctly:
//!   1. Causal type checker (check/causal.rs)  → identification + confidence
//!   2. Epistemic proof searcher (dependent/)   → confidence predicates
//!   3. SMT solver (smt/z3_solver.rs)          → formal verification
//!
//! Flow: causal graph → Pearl's ID → confidence factor → epistemic predicate
//!       → proof search → Z3 verification.
//!
//! Run with: cargo test --features smt --test causal_epistemic_smt_composition

#![cfg(feature = "smt")]

use sounio::check::causal::{
    CausalCheckResult, CausalModelCheckResult, check_causal_model_def, check_counterfactual_expr,
    check_do_expr,
};
use sounio::common::Span;
use sounio::dependent::TypeContext;
use sounio::dependent::predicates::{CausalPredicate, Predicate};
use sounio::dependent::proof_search::ProofSearcher;
use sounio::dependent::types::{CausalGraphType, ConfidenceType};
use sounio::smt::causal_encoding::{
    CONFIDENCE_BACKDOOR, CONFIDENCE_CAUSAL_MARKOV, CONFIDENCE_FRONTDOOR, CONFIDENCE_POSITIVITY,
    CONFIDENCE_SUTVA,
};
use sounio::types::causal::{CausalEdgeDef, CausalGraphDef, CausalGraphRegistry, CausalNodeDef};
use sounio::types::causal_id::IdentificationFormula;

// ===========================================================================
// Helpers
// ===========================================================================

fn dummy_span() -> Span {
    Span::new(0, 0)
}

fn node(name: &str) -> CausalNodeDef {
    CausalNodeDef {
        name: name.to_string(),
        span: dummy_span(),
    }
}

fn edge_def(from: &str, to: &str) -> CausalEdgeDef {
    CausalEdgeDef {
        from: from.to_string(),
        to: to.to_string(),
        span: dummy_span(),
    }
}

fn edge_tuple(from: &str, to: &str) -> (String, String) {
    (from.to_string(), to.to_string())
}

fn make_graph(name: &str, nodes: Vec<CausalNodeDef>, edges: Vec<CausalEdgeDef>) -> CausalGraphDef {
    CausalGraphDef {
        name: name.to_string(),
        nodes,
        edges,
        topological_order: None,
        span: dummy_span(),
    }
}

fn registry_with(graph: CausalGraphDef) -> CausalGraphRegistry {
    use sounio::ast::{CausalEdge, CausalModelDef, CausalNode, Visibility};
    use sounio::common::NodeId;

    let mut reg = CausalGraphRegistry::new();

    let model = CausalModelDef {
        id: NodeId(0),
        visibility: Visibility::Private,
        name: graph.name.clone(),
        nodes: graph
            .nodes
            .iter()
            .map(|n| CausalNode {
                id: NodeId(0),
                name: n.name.clone(),
                ty: None,
                span: dummy_span(),
            })
            .collect(),
        edges: graph
            .edges
            .iter()
            .map(|e| CausalEdge {
                id: NodeId(0),
                from: e.from.clone(),
                to: e.to.clone(),
                span: dummy_span(),
            })
            .collect(),
        equations: vec![],
        span: dummy_span(),
    };

    reg.register(&model)
        .expect("test graph should be a valid DAG");
    reg
}

// ===========================================================================
// 1. Causal confidence flows into epistemic predicates
// ===========================================================================

/// Backdoor-identified causal effect produces confidence that satisfies
/// an epistemic bound, verified end-to-end via proof search.
#[test]
fn causal_backdoor_confidence_satisfies_epistemic_bound() {
    // Confounded graph: U→A, U→B, A→B
    let g = make_graph(
        "BackdoorGraph",
        vec![node("A"), node("B"), node("U")],
        vec![edge_def("U", "A"), edge_def("U", "B"), edge_def("A", "B")],
    );
    let reg = registry_with(g);

    let result = check_do_expr(
        &reg,
        &[("A".to_string(), ())],
        Some("BackdoorGraph"),
        Some("B"),
    );

    let confidence = match &result {
        CausalCheckResult::Identified {
            formula,
            confidence,
            ..
        } => {
            assert!(
                matches!(formula, IdentificationFormula::Adjustment { .. }),
                "Expected adjustment formula, got: {:?}",
                formula
            );
            *confidence
        }
        other => panic!("Expected Identified, got: {:?}", other),
    };

    // confidence = BACKDOOR * POSITIVITY * CAUSAL_MARKOV
    let expected = CONFIDENCE_BACKDOOR * CONFIDENCE_POSITIVITY * CONFIDENCE_CAUSAL_MARKOV;
    assert!(
        (confidence - expected).abs() < 1e-10,
        "Backdoor confidence mismatch: {} vs {}",
        confidence,
        expected
    );

    // Feed into epistemic proof searcher
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    // Should satisfy >= 0.80
    let pred = Predicate::confidence_geq(
        ConfidenceType::literal(confidence),
        ConfidenceType::literal(0.80),
    );
    assert!(
        searcher.search(&pred).is_proven(),
        "Backdoor confidence {:.4} should satisfy >= 0.80",
        confidence
    );

    // Should NOT satisfy >= 0.90 (BACKDOOR * POSITIVITY * MARKOV ≈ 0.824)
    let strict_pred = Predicate::confidence_geq(
        ConfidenceType::literal(confidence),
        ConfidenceType::literal(0.90),
    );
    assert!(
        searcher.search(&strict_pred).is_disproven(),
        "Backdoor confidence {:.4} should NOT satisfy >= 0.90",
        confidence
    );
}

/// Frontdoor confidence constant exceeds backdoor, verified via proof search.
/// Note: When U is observed, the ID algorithm finds backdoor adjustment via U
/// for both graph types. We test the confidence constants directly because
/// CausalGraphDef doesn't model latent (bidirected) confounding.
#[test]
fn frontdoor_confidence_exceeds_backdoor_in_epistemic_system() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let fd_conf = CONFIDENCE_FRONTDOOR * CONFIDENCE_CAUSAL_MARKOV;
    let bd_conf = CONFIDENCE_BACKDOOR * CONFIDENCE_POSITIVITY * CONFIDENCE_CAUSAL_MARKOV;

    // Frontdoor confidence should exceed backdoor
    assert!(
        fd_conf > bd_conf,
        "FRONTDOOR*MARKOV ({:.4}) > BACKDOOR*POSITIVITY*MARKOV ({:.4})",
        fd_conf,
        bd_conf
    );

    // Verify this ordering through the epistemic proof search
    let pred = Predicate::confidence_geq(
        ConfidenceType::literal(fd_conf),
        ConfidenceType::literal(bd_conf),
    );
    assert!(
        searcher.search(&pred).is_proven(),
        "Epistemic system should prove frontdoor >= backdoor confidence"
    );

    // Reverse should be disproven
    let reverse = Predicate::confidence_geq(
        ConfidenceType::literal(bd_conf),
        ConfidenceType::literal(fd_conf),
    );
    assert!(
        searcher.search(&reverse).is_disproven(),
        "Epistemic system should disprove backdoor >= frontdoor confidence"
    );
}

/// Unconfounded direct effect yields maximal confidence.
#[test]
fn causal_unconfounded_maximal_confidence() {
    let g = make_graph(
        "Direct",
        vec![node("A"), node("B")],
        vec![edge_def("A", "B")],
    );
    let reg = registry_with(g);

    let result = check_do_expr(&reg, &[("A".to_string(), ())], Some("Direct"), Some("B"));

    match &result {
        CausalCheckResult::Identified { confidence, .. } => {
            assert!(
                *confidence >= CONFIDENCE_CAUSAL_MARKOV - 1e-10,
                "Unconfounded should have high confidence, got: {}",
                confidence
            );
        }
        other => panic!(
            "Expected Identified for unconfounded graph, got: {:?}",
            other
        ),
    }
}

// ===========================================================================
// 2. Causal + SMT proof search composition
// ===========================================================================

/// Causal identifiability predicate through proof search.
#[test]
fn causal_identifiability_via_proof_search() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let mut graph = CausalGraphType::new();
    graph.add_edge("A", "B");

    let pred = Predicate::causal(CausalPredicate::Identifiable {
        graph,
        treatment: "A".to_string(),
        outcome: "B".to_string(),
    });

    assert!(
        searcher.search(&pred).is_proven(),
        "Simple chain should be identifiable"
    );
}

/// Confounded graph with backdoor through proof search.
#[test]
fn causal_confounded_backdoor_via_proof_search() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let mut graph = CausalGraphType::new();
    graph.add_edge("U", "A");
    graph.add_edge("U", "B");
    graph.add_edge("A", "B");

    let pred = Predicate::causal(CausalPredicate::Identifiable {
        graph,
        treatment: "A".to_string(),
        outcome: "B".to_string(),
    });

    assert!(
        searcher.search(&pred).is_proven(),
        "Confounded graph with backdoor should be identifiable"
    );
}

/// Non-identifiable bidirected graph is disproven.
#[test]
fn causal_bidirected_unidentifiable_disproven() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let mut graph = CausalGraphType::new();
    graph.add_edge("A", "B");
    graph.add_bidirected("A", "B");

    let pred = Predicate::causal(CausalPredicate::Identifiable {
        graph,
        treatment: "A".to_string(),
        outcome: "B".to_string(),
    });

    assert!(
        searcher.search(&pred).is_disproven(),
        "Bidirected confounding with no adjustment should be disproven"
    );
}

/// Frontdoor criterion through proof search.
#[test]
fn causal_frontdoor_via_proof_search() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let mut graph = CausalGraphType::new();
    graph.add_edge("A", "M");
    graph.add_edge("M", "B");
    graph.add_bidirected("A", "B");

    let pred = Predicate::causal(CausalPredicate::Identifiable {
        graph,
        treatment: "A".to_string(),
        outcome: "B".to_string(),
    });

    assert!(
        searcher.search(&pred).is_proven(),
        "Frontdoor criterion should make this identifiable"
    );
}

// ===========================================================================
// 3. d-separation verified by proof search
// ===========================================================================

/// d-separation in a chain: conditioning on mediator blocks the path.
#[test]
fn d_separation_chain_blocks_via_proof_search() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let mut graph = CausalGraphType::new();
    graph.add_edge("A", "Z");
    graph.add_edge("Z", "B");

    let pred = Predicate::causal(CausalPredicate::DSeparated {
        graph,
        x: ["A".to_string()].into_iter().collect(),
        y: ["B".to_string()].into_iter().collect(),
        z: ["Z".to_string()].into_iter().collect(),
    });

    assert!(
        searcher.search(&pred).is_proven(),
        "Conditioning on Z in chain A→Z→B should d-separate A and B"
    );
}

/// d-separation in a fork: conditioning on common cause blocks both paths.
#[test]
fn d_separation_fork_blocks_via_proof_search() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let mut graph = CausalGraphType::new();
    graph.add_edge("Z", "A");
    graph.add_edge("Z", "B");

    let pred = Predicate::causal(CausalPredicate::DSeparated {
        graph,
        x: ["A".to_string()].into_iter().collect(),
        y: ["B".to_string()].into_iter().collect(),
        z: ["Z".to_string()].into_iter().collect(),
    });

    assert!(
        searcher.search(&pred).is_proven(),
        "Conditioning on Z in fork Z→A, Z→B should d-separate A and B"
    );
}

// ===========================================================================
// 4. Counterfactual confidence flows through epistemic verification
// ===========================================================================

/// Counterfactual confidence combines CAUSAL_MARKOV and SUTVA,
/// and the resulting value satisfies the correct epistemic bounds.
#[test]
fn counterfactual_confidence_composes_with_epistemic() {
    let g = make_graph("CF", vec![node("A"), node("B")], vec![edge_def("A", "B")]);
    let reg = registry_with(g);

    let result = check_counterfactual_expr(&reg, Some("CF"), &["A".to_string()], Some("B"), &[]);

    let confidence = match &result {
        CausalCheckResult::Identified { confidence, .. } => *confidence,
        other => panic!("Expected Identified counterfactual, got: {:?}", other),
    };

    // Expected: CAUSAL_MARKOV * SUTVA = 0.99 * 0.95 = 0.9405
    let expected = CONFIDENCE_CAUSAL_MARKOV * CONFIDENCE_SUTVA;
    assert!(
        (confidence - expected).abs() < 1e-10,
        "Counterfactual confidence mismatch: {} vs {}",
        confidence,
        expected
    );

    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    // Should satisfy >= 0.90
    let pred = Predicate::confidence_geq(
        ConfidenceType::literal(confidence),
        ConfidenceType::literal(0.90),
    );
    assert!(
        searcher.search(&pred).is_proven(),
        "Counterfactual confidence {:.4} should satisfy >= 0.90",
        confidence
    );

    // Should NOT satisfy >= 0.95 (0.9405 < 0.95)
    let strict_pred = Predicate::confidence_geq(
        ConfidenceType::literal(confidence),
        ConfidenceType::literal(0.95),
    );
    assert!(
        searcher.search(&strict_pred).is_disproven(),
        "Counterfactual confidence {:.4} should NOT satisfy >= 0.95",
        confidence
    );
}

// ===========================================================================
// 5. Product of causal and epistemic confidence
// ===========================================================================

/// Combining causal confidence with sensor epistemic confidence via product.
#[test]
fn product_of_causal_and_sensor_confidence() {
    let causal_confidence = CONFIDENCE_BACKDOOR * CONFIDENCE_POSITIVITY * CONFIDENCE_CAUSAL_MARKOV;
    let sensor_confidence = 0.95;

    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let combined = ConfidenceType::product(
        ConfidenceType::literal(causal_confidence),
        ConfidenceType::literal(sensor_confidence),
    );

    // Combined ≈ 0.824 * 0.95 ≈ 0.783: should satisfy >= 0.75
    let pred = Predicate::confidence_geq(combined.clone(), ConfidenceType::literal(0.75));
    assert!(
        searcher.search(&pred).is_proven(),
        "Combined causal*sensor confidence should satisfy >= 0.75"
    );

    // Should NOT satisfy >= 0.85
    let strict = Predicate::confidence_geq(combined, ConfidenceType::literal(0.85));
    assert!(
        searcher.search(&strict).is_disproven(),
        "Combined causal*sensor confidence should NOT satisfy >= 0.85"
    );
}

// ===========================================================================
// 6. Causal model validation + clinical epistemic guard
// ===========================================================================

/// Valid DAG passes model validation and its identification confidence
/// satisfies the required epistemic bound for clinical use.
#[test]
fn clinical_pipeline_causal_to_epistemic() {
    let g = make_graph(
        "PBPK",
        vec![
            node("Dose"),
            node("Absorption"),
            node("Plasma"),
            node("Effect"),
        ],
        vec![
            edge_def("Dose", "Absorption"),
            edge_def("Absorption", "Plasma"),
            edge_def("Plasma", "Effect"),
        ],
    );

    // Validate DAG
    let model_check = check_causal_model_def(
        "PBPK",
        &g.nodes,
        &[
            edge_tuple("Dose", "Absorption"),
            edge_tuple("Absorption", "Plasma"),
            edge_tuple("Plasma", "Effect"),
        ],
    );
    assert!(
        matches!(model_check, CausalModelCheckResult::Valid { .. }),
        "PBPK graph should be a valid DAG"
    );

    let reg = registry_with(g);

    let result = check_do_expr(
        &reg,
        &[("Dose".to_string(), ())],
        Some("PBPK"),
        Some("Effect"),
    );

    let confidence = match &result {
        CausalCheckResult::Identified { confidence, .. } => *confidence,
        other => panic!("Expected Identified, got: {:?}", other),
    };

    // Clinical bound: 0.90
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let clinical_bound = Predicate::confidence_geq(
        ConfidenceType::literal(confidence),
        ConfidenceType::literal(0.90),
    );
    assert!(
        searcher.search(&clinical_bound).is_proven(),
        "PBPK unconfounded chain should satisfy clinical bound (confidence={:.4})",
        confidence
    );
}

// ===========================================================================
// 7. Cyclic graph rejection
// ===========================================================================

/// Cyclic graph is rejected at the model validation level.
#[test]
fn cyclic_graph_rejected_before_identification() {
    let result = check_causal_model_def(
        "CyclicDrug",
        &[node("Drug"), node("Side_Effect")],
        &[
            edge_tuple("Drug", "Side_Effect"),
            edge_tuple("Side_Effect", "Drug"),
        ],
    );

    assert!(
        matches!(result, CausalModelCheckResult::CyclicGraph { .. }),
        "Cyclic graph should be rejected"
    );
}

// ===========================================================================
// 8. Dempster-Shafer combination of causal confidences
// ===========================================================================

/// Two independent causal analyses combined via Dempster-Shafer
/// produces higher confidence than either alone.
#[test]
fn dempster_shafer_combines_two_causal_confidences() {
    let ctx = TypeContext::new();
    let mut searcher = ProofSearcher::new(&ctx);

    let causal_a = CONFIDENCE_BACKDOOR * CONFIDENCE_POSITIVITY * CONFIDENCE_CAUSAL_MARKOV;
    let causal_b = CONFIDENCE_FRONTDOOR * CONFIDENCE_CAUSAL_MARKOV;

    // DS(a, b) = 1 - (1-a)(1-b)
    let combined = ConfidenceType::dempster_shafer(
        ConfidenceType::literal(causal_a),
        ConfidenceType::literal(causal_b),
    );

    // DS(0.824, 0.9405) ≈ 0.9895: should satisfy >= 0.98
    let pred = Predicate::confidence_geq(combined.clone(), ConfidenceType::literal(0.98));
    assert!(
        searcher.search(&pred).is_proven(),
        "DS combination of two causal analyses should satisfy >= 0.98"
    );

    // Each alone should NOT satisfy >= 0.98
    let alone_a = Predicate::confidence_geq(
        ConfidenceType::literal(causal_a),
        ConfidenceType::literal(0.98),
    );
    assert!(
        searcher.search(&alone_a).is_disproven(),
        "Backdoor alone should NOT satisfy >= 0.98"
    );

    let alone_b = Predicate::confidence_geq(
        ConfidenceType::literal(causal_b),
        ConfidenceType::literal(0.98),
    );
    assert!(
        searcher.search(&alone_b).is_disproven(),
        "Frontdoor alone should NOT satisfy >= 0.98"
    );
}
