//! Integration tests for PAC Learning Types
//!
//! These tests verify the complete PAC learning pipeline:
//! - Type definitions and VC dimension calculations
//! - Sample complexity computations
//! - Generalization bound derivations
//! - Integration with epistemic types
//! - Qualifier-based refinement type checking

use sounio::check::pac::{PACChecker, PACConstraint, PACErrorKind, PACVerifyResult};
use sounio::types::epistemic::{ConfidenceOp, KnowledgeType};
use sounio::types::pac::{
    compute_generalization_gap, compute_sample_complexity, compute_sample_complexity_tight,
    GeneralizationBound, HypothesisClass, VCDimExpr, VCDimension,
};
use sounio::types::Type;

// ============================================================================
// Sample Complexity Tests
// ============================================================================

#[test]
fn test_sample_complexity_practical_scenarios() {
    // Scenario 1: Simple binary classification (decision stump)
    // VC dim = 2, want 95% accuracy, 95% confidence
    // Formula: (1/ε²) * (d + ln(1/δ)) ≈ 400 * (2 + 3) = 2000
    let samples_stump = compute_sample_complexity(2, 0.05, 0.05);
    assert!(
        samples_stump > 0 && samples_stump < 3000,
        "Decision stump should need reasonable samples: {}",
        samples_stump
    );

    // Scenario 2: 10D linear classifier (same ε, δ as stump to compare)
    // VC dim = 11, want 95% accuracy, 95% confidence
    let samples_linear = compute_sample_complexity(11, 0.05, 0.05);
    assert!(
        samples_linear > samples_stump,
        "More complex class needs more samples: linear={} vs stump={}",
        samples_linear,
        samples_stump
    );

    // Scenario 3: Deep neural network (1000 weights)
    // VC dim ≈ 1000, want 95% accuracy, 99% confidence
    let samples_nn = compute_sample_complexity(1000, 0.05, 0.01);
    assert!(
        samples_nn > 100000,
        "Neural networks need lots of data: {}",
        samples_nn
    );
}

#[test]
fn test_tight_vs_standard_bounds() {
    // Test various scenarios where tight bounds should be better
    let scenarios = [
        (10, 0.1, 0.1),  // Moderate case
        (5, 0.05, 0.05), // High precision
        (100, 0.2, 0.2), // Large class, loose precision
    ];

    for (d, eps, delta) in scenarios {
        let standard = compute_sample_complexity(d, eps, delta);
        let tight = compute_sample_complexity_tight(d, eps, delta);

        assert!(
            tight <= standard,
            "Tight bound should be <= standard for d={}, eps={}, delta={}: tight={}, standard={}",
            d,
            eps,
            delta,
            tight,
            standard
        );
    }
}

#[test]
fn test_sample_complexity_edge_cases() {
    // Very high precision requirement
    let high_precision = compute_sample_complexity(10, 0.001, 0.001);
    assert!(
        high_precision > 1_000_000,
        "High precision needs many samples"
    );

    // Very low precision (loose) - should need fewer samples
    let low_precision = compute_sample_complexity(10, 0.3, 0.3);
    assert!(low_precision < 1000, "Loose precision needs few samples");
}

// ============================================================================
// Generalization Bound Tests
// ============================================================================

#[test]
fn test_generalization_gap_decreases_with_samples() {
    let vc_dim = 10;
    let delta = 0.05;

    let gap_1k = compute_generalization_gap(vc_dim, 1000, delta);
    let gap_10k = compute_generalization_gap(vc_dim, 10000, delta);
    let gap_100k = compute_generalization_gap(vc_dim, 100000, delta);

    assert!(
        gap_10k < gap_1k,
        "Gap should decrease: 10k={} vs 1k={}",
        gap_10k,
        gap_1k
    );
    assert!(
        gap_100k < gap_10k,
        "Gap should decrease: 100k={} vs 10k={}",
        gap_100k,
        gap_10k
    );
}

#[test]
fn test_generalization_bound_from_pac() {
    let bound = GeneralizationBound::from_pac(
        10,   // VC dimension
        5000, // samples
        0.05, // delta (95% confidence)
        0.02, // training error
    );

    assert!(bound.confidence > 0.94);
    assert!(bound.confidence < 0.96);

    let test_bound = bound.test_error_bound().unwrap();
    assert!(test_bound > 0.02, "Test bound >= train error");
    assert!(test_bound < 0.5, "Test bound should be reasonable");
}

// ============================================================================
// PAC Checker Tests
// ============================================================================

#[test]
fn test_pac_checker_basic_workflow() {
    let mut checker = PACChecker::new();

    // Register and use a custom hypothesis class
    let my_class = HypothesisClass {
        name: "MyLinear5D".to_string(),
        vc_dim: VCDimension::Const(6), // d + 1 for 5D
        model_type: "Linear".to_string(),
        is_finite: true,
        rademacher: None,
    };
    checker.register_hypothesis("MyLinear5D", my_class);

    // Check with sufficient samples
    let result = checker.check_train_call(
        "MyLinear5D",
        5000, // samples
        0.1,  // epsilon
        0.1,  // delta
        None,
    );

    assert!(result.is_ok());
    let bound = result.unwrap();
    assert!(bound.confidence > 0.89);
}

#[test]
fn test_pac_checker_insufficient_samples() {
    let mut checker = PACChecker::new();

    // Try with too few samples
    let result = checker.check_train_call(
        "LinearClassifier", // VC dim = 11
        100,                // samples (too few!)
        0.01,               // epsilon (very strict)
        0.01,               // delta (very strict)
        None,
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind, PACErrorKind::InsufficientSamples);
    assert!(err.suggestion.is_some());
}

#[test]
fn test_pac_checker_all_standard_classes() {
    let checker = PACChecker::new();

    // All standard classes should be registered
    let standard_classes = [
        "LinearClassifier",
        "DecisionStump",
        "AxisAlignedRectangle",
        "NeuralNetwork",
        "DecisionTree",
    ];

    for class_name in &standard_classes {
        assert!(
            checker.lookup_hypothesis(class_name).is_some(),
            "Standard class {} should be registered",
            class_name
        );
    }
}

// ============================================================================
// VC Dimension Expression Tests
// ============================================================================

#[test]
fn test_vc_dim_expression_evaluation() {
    use std::collections::HashMap;

    // Test: W * log(W) for neural networks
    let expr = VCDimExpr::mul(
        VCDimExpr::Var("W".to_string()),
        VCDimExpr::Log(Box::new(VCDimExpr::Var("W".to_string()))),
    );

    let mut bindings = HashMap::new();
    bindings.insert("W".to_string(), 1024u64);

    // 1024 * log2(1024) = 1024 * 10 = 10240
    let result = expr.try_eval(&bindings);
    assert!(result.is_some());
    let val = result.unwrap();
    assert!(val >= 10000 && val <= 11000, "Got {}", val);
}

#[test]
fn test_vc_dim_linear_classifier() {
    // Linear classifier: d + 1
    let expr = VCDimExpr::add(VCDimExpr::Var("d".to_string()), VCDimExpr::Const(1));

    let mut bindings = std::collections::HashMap::new();
    bindings.insert("d".to_string(), 10u64);

    let result = expr.try_eval(&bindings).unwrap();
    assert_eq!(result, 11);
}

// ============================================================================
// Epistemic Integration Tests
// ============================================================================

#[test]
fn test_knowledge_type_from_pac() {
    let model_type = Type::Named {
        name: "LinearClassifier".to_string(),
        args: vec![],
    };

    let knowledge = KnowledgeType::from_pac_learning(
        model_type, // inner type
        0.95,       // confidence (1 - delta)
        0.05,       // epsilon
        10000,      // samples
    );

    // Verify PAC provenance
    assert!(knowledge.is_pac_derived());
    assert_eq!(knowledge.pac_confidence(), Some(0.95));

    // Verify confidence bound
    let bound = knowledge.confidence_bound.unwrap();
    assert_eq!(bound.operator, ConfidenceOp::GreaterEq);
    assert!((bound.value - 0.95).abs() < f64::EPSILON);
}

#[test]
fn test_knowledge_type_pac_provenance() {
    let model_type = Type::Named {
        name: "Model".to_string(),
        args: vec![],
    };

    // Create PAC-derived knowledge
    let pac_knowledge = KnowledgeType::from_pac_learning(model_type.clone(), 0.99, 0.01, 50000);
    assert!(pac_knowledge.is_pac_derived());

    // Create regular knowledge
    let regular_knowledge = KnowledgeType::new(model_type);
    assert!(!regular_knowledge.is_pac_derived());
}

// ============================================================================
// Constraint Verification Tests
// ============================================================================

#[test]
fn test_pac_verify_result_types() {
    let mut checker = PACChecker::new();

    // Test satisfied constraint
    let result = checker.check_train_call("DecisionStump", 1000, 0.1, 0.1, None);
    assert!(result.is_ok());

    let results = checker.verify_all();
    assert_eq!(results.len(), 1);

    match &results[0] {
        PACVerifyResult::Satisfied {
            bound,
            samples,
            min_samples,
        } => {
            assert!(*samples >= *min_samples);
            assert!(bound.confidence > 0.89);
        }
        _ => panic!("Expected Satisfied result"),
    }
}

// ============================================================================
// Hypothesis Class Factory Tests
// ============================================================================

#[test]
fn test_hypothesis_class_factories() {
    // Linear classifier
    let linear = HypothesisClass::linear_classifier(10);
    assert_eq!(linear.name, "LinearClassifier<10>");
    assert!(matches!(linear.vc_dim, VCDimension::Const(11)));

    // Decision stump: VC dimension = 2 (can shatter 2 points on a line)
    let stump = HypothesisClass::decision_stump();
    assert!(matches!(stump.vc_dim, VCDimension::Const(2)));

    // Axis-aligned rectangle
    let rect = HypothesisClass::axis_aligned_rectangle(3);
    assert!(matches!(rect.vc_dim, VCDimension::Const(6))); // 2 * 3

    // Neural network (symbolic VC dim)
    let nn = HypothesisClass::neural_network(4, 128);
    match &nn.vc_dim {
        VCDimension::Expr(_) => {} // Expected
        _ => panic!("Neural network should have symbolic VC dim"),
    }

    // Decision tree (symbolic VC dim)
    let tree = HypothesisClass::decision_tree(5, 10);
    match &tree.vc_dim {
        VCDimension::Expr(_) => {} // Expected
        _ => panic!("Decision tree should have symbolic VC dim"),
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_pac_error_kinds() {
    let mut checker = PACChecker::new();

    // Invalid epsilon
    let result = checker.check_train_call("DecisionStump", 1000, 1.5, 0.1, None);
    assert_eq!(result.unwrap_err().kind, PACErrorKind::InvalidEpsilon);

    // Invalid delta
    let result = checker.check_train_call("DecisionStump", 1000, 0.1, -0.1, None);
    assert_eq!(result.unwrap_err().kind, PACErrorKind::InvalidDelta);

    // Unknown hypothesis class
    let result = checker.check_train_call("NonExistent", 1000, 0.1, 0.1, None);
    assert_eq!(
        result.unwrap_err().kind,
        PACErrorKind::UnknownHypothesisClass
    );
}

// ============================================================================
// Performance/Scale Tests
// ============================================================================

#[test]
fn test_large_scale_sample_complexity() {
    // ImageNet-scale scenario
    // Large neural network, high accuracy requirement
    let samples_needed = compute_sample_complexity(
        10_000, // VC dim for large network
        0.05,   // 95% accuracy
        0.01,   // 99% confidence
    );

    // Should need millions of samples
    assert!(
        samples_needed > 1_000_000,
        "Large network needs many samples: {}",
        samples_needed
    );
}

#[test]
fn test_checker_can_handle_many_constraints() {
    let mut checker = PACChecker::new();

    // Add many constraints
    for i in 0..100 {
        let _ = checker.check_train_call("DecisionStump", 1000 + i * 100, 0.1, 0.1, None);
    }

    let results = checker.verify_all();
    assert_eq!(results.len(), 100);
    assert!(results.iter().all(|r| r.is_satisfied()));
}
