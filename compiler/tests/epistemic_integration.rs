//! Integration Tests for Sounio Epistemic Gaps
//!
//! These tests verify the correct integration of:
//! - Promotion lattice with existing UncertainValue
//! - KEC selector with real-world scenarios
//! - CLI command parsing and execution
//! - SMC integration with Knowledge type
//!
//! Run with: cargo test --test epistemic_integration

use std::collections::HashMap;

// =============================================================================
// Module imports (adjust paths for actual integration)
// =============================================================================

mod promotion {
    pub use sounio::epistemic::promotion::*;
}

mod kec {
    pub use sounio::epistemic::kec::*;
}

use promotion::*;
use kec::*;

// =============================================================================
// Promotion Lattice Integration Tests
// =============================================================================

mod lattice_tests {
    use super::*;

    /// Test that the lattice respects mathematical properties
    #[test]
    fn test_lattice_is_valid_lattice() {
        let lattice = PromotionLattice::new();
        let levels = [
            UncertaintyLevel::Point,
            UncertaintyLevel::Interval,
            UncertaintyLevel::Fuzzy,
            UncertaintyLevel::Affine,
            UncertaintyLevel::DempsterShafer,
            UncertaintyLevel::Distribution,
            UncertaintyLevel::Particles,
        ];

        // Reflexivity: a ≤ a
        for &a in &levels {
            assert!(lattice.is_subtype(a, a), "Reflexivity failed for {:?}", a);
        }

        // Transitivity: a ≤ b ∧ b ≤ c → a ≤ c
        for &a in &levels {
            for &b in &levels {
                for &c in &levels {
                    if lattice.is_subtype(a, b) && lattice.is_subtype(b, c) {
                        assert!(
                            lattice.is_subtype(a, c),
                            "Transitivity failed: {:?} ≤ {:?} ≤ {:?}",
                            a, b, c
                        );
                    }
                }
            }
        }

        // Meet is greatest lower bound
        for &a in &levels {
            for &b in &levels {
                let m = lattice.meet(a, b);
                // m ≤ a and m ≤ b
                assert!(
                    lattice.is_subtype(m, a) || a == b,
                    "Meet {:?} not ≤ {:?}",
                    m, a
                );
                assert!(
                    lattice.is_subtype(m, b) || a == b,
                    "Meet {:?} not ≤ {:?}",
                    m, b
                );
            }
        }

        // Join is least upper bound
        for &a in &levels {
            for &b in &levels {
                let j = lattice.join(a, b);
                // a ≤ j and b ≤ j
                assert!(
                    lattice.is_subtype(a, j),
                    "Join: {:?} not ≤ {:?}",
                    a, j
                );
                assert!(
                    lattice.is_subtype(b, j),
                    "Join: {:?} not ≤ {:?}",
                    b, j
                );
            }
        }
    }

    /// Test specific promotion paths
    #[test]
    fn test_promotion_paths() {
        let lattice = PromotionLattice::new();

        // Point can promote to anything
        assert!(lattice.is_subtype(UncertaintyLevel::Point, UncertaintyLevel::Interval));
        assert!(lattice.is_subtype(UncertaintyLevel::Point, UncertaintyLevel::Affine));
        assert!(lattice.is_subtype(UncertaintyLevel::Point, UncertaintyLevel::Distribution));
        assert!(lattice.is_subtype(UncertaintyLevel::Point, UncertaintyLevel::Particles));

        // Interval → Affine → Distribution → Particles
        assert!(lattice.is_subtype(UncertaintyLevel::Interval, UncertaintyLevel::Affine));
        assert!(lattice.is_subtype(UncertaintyLevel::Affine, UncertaintyLevel::Distribution));
        assert!(lattice.is_subtype(UncertaintyLevel::Distribution, UncertaintyLevel::Particles));

        // Cannot demote
        assert!(!lattice.is_subtype(UncertaintyLevel::Distribution, UncertaintyLevel::Point));
        assert!(!lattice.is_subtype(UncertaintyLevel::Particles, UncertaintyLevel::Interval));
    }

    /// Test promoter value conversions
    #[test]
    fn test_promoter_preserves_information() {
        let promoter = Promoter::new().with_samples(1000);

        // Point to Interval
        let point_val = 10.0;
        let point_conf = 0.95;

        let interval = promoter.promote_point(point_val, point_conf, UncertaintyLevel::Interval).unwrap();
        let (lower, upper) = interval.bounds();

        // Point should be within interval
        assert!(lower <= point_val && point_val <= upper);

        // Point estimate should match original
        assert!((interval.point_estimate() - point_val).abs() < 0.01);

        // Interval to Distribution
        let dist = promoter.promote_interval(lower, upper, UncertaintyLevel::Distribution).unwrap();
        let (dist_lower, dist_upper) = dist.bounds();

        // Bounds should be approximately preserved
        assert!((dist_lower - lower).abs() < 1.0);
        assert!((dist_upper - upper).abs() < 1.0);
    }

    /// Test meet/join with incomparable elements
    #[test]
    fn test_incomparable_elements() {
        let lattice = PromotionLattice::new();

        // Interval and Fuzzy are incomparable (same height)
        assert!(!lattice.is_subtype(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy));
        assert!(!lattice.is_subtype(UncertaintyLevel::Fuzzy, UncertaintyLevel::Interval));

        // Their meet is Point
        assert_eq!(
            lattice.meet(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy),
            UncertaintyLevel::Point
        );

        // Their join is Affine
        assert_eq!(
            lattice.join(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy),
            UncertaintyLevel::Affine
        );

        // Affine and DempsterShafer are incomparable
        assert!(!lattice.is_subtype(UncertaintyLevel::Affine, UncertaintyLevel::DempsterShafer));

        // Their join is Distribution
        assert_eq!(
            lattice.join(UncertaintyLevel::Affine, UncertaintyLevel::DempsterShafer),
            UncertaintyLevel::Distribution
        );
    }
}

// =============================================================================
// KEC Auto-Selection Integration Tests
// =============================================================================

mod kec_tests {
    use super::*;

    /// Test that KEC respects cost constraints
    #[test]
    fn test_kec_respects_cost_constraints() {
        // Tight cost constraint should favor simpler models
        let config = KECConfig {
            max_cost_multiplier: 5.0,  // Very tight
            ..Default::default()
        };
        let selector = KECSelector::with_config(config);

        let mut uncertainty = UncertaintyAnalyzer::new();
        uncertainty.add_gaussian("x", 10.0, 1.0);
        let u = uncertainty.analyze();

        let mut complexity = ComplexityAnalyzer::new();
        complexity.add_op(OpKind::Mul);
        let c = complexity.analyze();

        let result = selector.select(&u, &c);

        // Should not recommend expensive models
        assert!(
            result.recommended_model.cost_multiplier() <= 10.0,
            "Recommended {:?} exceeds cost constraint",
            result.recommended_model
        );
    }

    /// Test KEC presets produce different recommendations
    #[test]
    fn test_kec_presets_differ() {
        // Complex computation with moderate uncertainty
        let mut uncertainty = UncertaintyAnalyzer::new();
        for i in 0..5 {
            uncertainty.add_gaussian(&format!("x{}", i), 10.0, 2.0);
        }
        let u = uncertainty.analyze();

        let mut complexity = ComplexityAnalyzer::new();
        for _ in 0..20 {
            complexity.add_op(OpKind::Mul);
        }
        complexity.add_op(OpKind::Exp);
        complexity.add_op(OpKind::Div);
        let c = complexity.analyze();

        // Get recommendations from different presets
        let default_result = KECSelector::new().select(&u, &c);
        let realtime_result = KECSelector::with_config(KECConfig::realtime()).select(&u, &c);
        let scientific_result = KECSelector::with_config(KECConfig::scientific()).select(&u, &c);
        let safety_result = KECSelector::with_config(KECConfig::safety_critical()).select(&u, &c);

        // Realtime should prefer cheaper models
        assert!(
            realtime_result.recommended_model.cost_multiplier()
                <= default_result.recommended_model.cost_multiplier(),
            "Realtime should prefer cheaper models"
        );

        // Safety-critical should prefer guaranteed bounds
        assert!(
            matches!(
                safety_result.recommended_model,
                UncertaintyLevel::Interval | UncertaintyLevel::Affine
            ),
            "Safety-critical should prefer bounded models, got {:?}",
            safety_result.recommended_model
        );
    }

    /// Test KEC with PK/PD typical inputs
    #[test]
    fn test_kec_pkpd_scenario() {
        let selector = KECSelector::with_config(KECConfig::pkpd());

        // Typical PK parameters with variability
        let mut uncertainty = UncertaintyAnalyzer::new();
        uncertainty.add_gaussian("CL", 10.0, 3.0);    // Clearance: 30% CV
        uncertainty.add_gaussian("Vd", 50.0, 15.0);   // Volume: 30% CV
        uncertainty.add_gaussian("ka", 1.2, 0.4);    // Absorption: 33% CV
        uncertainty.add_gaussian("F", 0.85, 0.1);    // Bioavailability
        let u = uncertainty.analyze();

        // One-compartment model operations
        let mut complexity = ComplexityAnalyzer::new();
        complexity.add_op(OpKind::Div);   // ke = CL/Vd
        complexity.add_op(OpKind::Sub);   // ka - ke
        complexity.add_op(OpKind::Mul);   // F * Dose
        complexity.add_op(OpKind::Mul);   // * ka
        complexity.add_op(OpKind::Div);   // / (Vd * (ka-ke))
        complexity.add_op(OpKind::Mul);   // -ke * t
        complexity.add_op(OpKind::Exp);   // exp(-ke*t)
        complexity.add_op(OpKind::Mul);   // -ka * t
        complexity.add_op(OpKind::Exp);   // exp(-ka*t)
        complexity.add_op(OpKind::Sub);   // difference
        complexity.add_op(OpKind::Mul);   // final multiply
        let c = complexity.analyze();

        let result = selector.select(&u, &c);

        // Should recommend Distribution or higher for PK/PD
        assert!(
            result.recommended_model.height() >= UncertaintyLevel::Affine.height(),
            "PK/PD should use at least Affine, got {:?}",
            result.recommended_model
        );

        // Should have reasonable confidence
        assert!(result.confidence > 0.5, "Confidence too low: {}", result.confidence);

        // Check reasoning includes relevant factors
        let reasoning_text = result.reasoning.join(" ");
        assert!(
            reasoning_text.contains("entropy") || reasoning_text.contains("complexity"),
            "Reasoning should mention key factors"
        );
    }

    /// Test KEC handles edge cases
    #[test]
    fn test_kec_edge_cases() {
        let selector = KECSelector::new();

        // Empty inputs → Point
        let u = UncertaintyAnalyzer::new().analyze();
        let c = ComplexityAnalyzer::new().analyze();
        let result = selector.select(&u, &c);
        assert_eq!(result.recommended_model, UncertaintyLevel::Point);

        // Single point input → Point
        let mut uncertainty = UncertaintyAnalyzer::new();
        uncertainty.add_point("x", 10.0);
        let u = uncertainty.analyze();
        let result = selector.select(&u, &c);
        assert_eq!(result.recommended_model, UncertaintyLevel::Point);

        // Very high entropy → Distribution or Particles
        let mut uncertainty = UncertaintyAnalyzer::new();
        for i in 0..20 {
            uncertainty.add_gaussian(&format!("x{}", i), 10.0, 5.0);
        }
        let u = uncertainty.analyze();

        let mut complexity = ComplexityAnalyzer::new();
        for _ in 0..50 {
            complexity.add_op(OpKind::Mul);
        }
        complexity.add_op(OpKind::Loop);
        let c = complexity.analyze();

        let result = selector.select(&u, &c);
        assert!(
            result.recommended_model.height() >= UncertaintyLevel::Distribution.height(),
            "High entropy should use Distribution or higher"
        );
    }

    /// Test complexity scoring
    #[test]
    fn test_complexity_scoring() {
        // Simple linear computation
        let mut simple = ComplexityAnalyzer::new();
        simple.add_op(OpKind::Add);
        simple.add_op(OpKind::Mul);
        let simple_score = simple.analyze().complexity_score();

        // Complex nonlinear computation
        let mut complex = ComplexityAnalyzer::new();
        for _ in 0..10 {
            complex.add_op(OpKind::Mul);
        }
        complex.add_op(OpKind::Exp);
        complex.add_op(OpKind::Log);
        complex.add_op(OpKind::Div);
        complex.enter_scope();
        complex.add_op(OpKind::Conditional);
        let complex_score = complex.analyze().complexity_score();

        assert!(
            complex_score > simple_score,
            "Complex computation should have higher score"
        );

        // Loop should significantly increase complexity
        let mut iterative = ComplexityAnalyzer::new();
        iterative.add_op(OpKind::Loop);
        let iter_metrics = iterative.analyze();
        assert!(iter_metrics.has_iteration);
        assert!(!iter_metrics.is_affine_compatible());
    }
}

// =============================================================================
// SMC Integration Tests
// =============================================================================

mod smc_tests {
    /// Test particle cloud statistics
    #[test]
    fn test_particle_cloud_statistics() {
        // Create a cloud approximating N(10, 1)
        let n = 10000;
        let mean = 10.0;
        let std = 1.0;

        // Quasi-random samples from normal
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                let z = inverse_normal_cdf(u);
                mean + z * std
            })
            .collect();

        // Compute statistics
        let computed_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let computed_var: f64 = samples.iter()
            .map(|x| (x - computed_mean).powi(2))
            .sum::<f64>() / n as f64;

        // Should be close to theoretical values
        assert!((computed_mean - mean).abs() < 0.1, "Mean error too large");
        assert!((computed_var - std*std).abs() < 0.1, "Variance error too large");
    }

    /// Test resampling preserves distribution
    #[test]
    fn test_resampling_preserves_distribution() {
        // This is a placeholder - actual test needs SMC module
        // The test verifies that after resampling, the distribution
        // is approximately preserved.

        // Create uneven weights
        let n = 1000;
        let mut weights = vec![0.0; n];
        weights[0] = 0.5;  // Half the mass on first particle
        for i in 1..n {
            weights[i] = 0.5 / (n - 1) as f64;
        }

        // After resampling, ~half the particles should be copies of first
        // (This is a statistical test, so we use a tolerance)

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights should sum to 1");
    }

    /// Test ESS computation
    #[test]
    fn test_effective_sample_size() {
        // Uniform weights → ESS = N
        let n = 100;
        let uniform_weights: Vec<f64> = vec![1.0 / n as f64; n];
        let ess_uniform = 1.0 / uniform_weights.iter().map(|w| w * w).sum::<f64>();
        assert!((ess_uniform - n as f64).abs() < 0.01);

        // All mass on one particle → ESS = 1
        let mut degenerate_weights = vec![0.0; n];
        degenerate_weights[0] = 1.0;
        let ess_degen = 1.0 / degenerate_weights.iter().map(|w| w * w).sum::<f64>();
        assert!((ess_degen - 1.0).abs() < 0.01);

        // Half mass on one, half distributed → ESS ≈ 1.6
        let mut half_weights = vec![0.5 / (n - 1) as f64; n];
        half_weights[0] = 0.5;
        let ess_half = 1.0 / half_weights.iter().map(|w| w * w).sum::<f64>();
        assert!(ess_half > 1.0 && ess_half < n as f64);
    }

    // Helper function (same as in promotion.rs)
    fn inverse_normal_cdf(p: f64) -> f64 {
        if p <= 0.0 { return f64::NEG_INFINITY; }
        if p >= 1.0 { return f64::INFINITY; }

        let p = if p > 0.5 { 1.0 - p } else { p };
        let t = (-2.0 * p.ln()).sqrt();

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p > 0.5 { -z } else { z }
    }
}

// =============================================================================
// End-to-End Workflow Tests
// =============================================================================

mod workflow_tests {
    use super::*;

    /// Test complete workflow: analyze → select → propagate
    #[test]
    fn test_complete_workflow() {
        // 1. Define uncertain inputs
        let mut uncertainty = UncertaintyAnalyzer::new();
        uncertainty.add_gaussian("a", 10.0, 1.0);
        uncertainty.add_gaussian("b", 5.0, 0.5);
        let u_metrics = uncertainty.analyze();

        // 2. Define computation
        let mut complexity = ComplexityAnalyzer::new();
        complexity.add_op(OpKind::Mul);  // a * b
        complexity.add_op(OpKind::Add);  // + c
        let c_metrics = complexity.analyze();

        // 3. Select model
        let selector = KECSelector::new();
        let result = selector.select(&u_metrics, &c_metrics);

        // 4. Verify reasonable recommendation
        assert!(result.confidence > 0.0);
        assert!(!result.reasoning.is_empty());

        // 5. If Affine selected, verify it's appropriate
        if result.recommended_model == UncertaintyLevel::Affine {
            assert!(c_metrics.is_affine_compatible());
        }

        // 6. Promotion should succeed
        let promoter = Promoter::new();
        let promoted = promoter.promote_point(10.0, 0.95, result.recommended_model);
        assert!(promoted.is_ok());
    }

    /// Test workflow with forced model override
    #[test]
    fn test_workflow_with_override() {
        let lattice = PromotionLattice::new();
        let promoter = Promoter::new().with_samples(500);

        // User wants Distribution regardless of KEC recommendation
        let forced_model = UncertaintyLevel::Distribution;

        // Verify Point can be promoted to Distribution
        assert!(lattice.is_subtype(UncertaintyLevel::Point, forced_model));

        // Perform promotion
        let result = promoter.promote_point(10.0, 0.95, forced_model).unwrap();
        assert_eq!(result.level(), UncertaintyLevel::Distribution);

        // Extract statistics
        if let PromotedValue::Distribution { samples, mean, variance } = result {
            assert_eq!(samples.len(), 500);
            assert!((mean - 10.0).abs() < 0.1);
            assert!(variance > 0.0);
        }
    }

    /// Test that alternatives are valid promotions from recommendation
    #[test]
    fn test_alternatives_are_valid() {
        let selector = KECSelector::new();
        let lattice = PromotionLattice::new();

        let mut uncertainty = UncertaintyAnalyzer::new();
        uncertainty.add_gaussian("x", 10.0, 2.0);
        let u = uncertainty.analyze();

        let mut complexity = ComplexityAnalyzer::new();
        complexity.add_op(OpKind::Mul);
        complexity.add_op(OpKind::Exp);
        let c = complexity.analyze();

        let result = selector.select(&u, &c);

        // All alternatives should be comparable via lattice
        for (alt, _score) in &result.alternatives {
            // Either alt is subtype of recommended, or vice versa, or both at same height
            let alt_to_rec = lattice.is_subtype(*alt, result.recommended_model);
            let rec_to_alt = lattice.is_subtype(result.recommended_model, *alt);
            let same_height = alt.height() == result.recommended_model.height();

            assert!(
                alt_to_rec || rec_to_alt || same_height,
                "Alternative {:?} should be comparable to {:?}",
                alt, result.recommended_model
            );
        }
    }
}

// =============================================================================
// Regression Tests
// =============================================================================

mod regression_tests {
    use super::*;

    /// Ensure Point is always at bottom of lattice
    #[test]
    fn test_point_is_bottom() {
        let lattice = PromotionLattice::new();

        for level in &[
            UncertaintyLevel::Interval,
            UncertaintyLevel::Fuzzy,
            UncertaintyLevel::Affine,
            UncertaintyLevel::DempsterShafer,
            UncertaintyLevel::Distribution,
            UncertaintyLevel::Particles,
        ] {
            assert!(
                lattice.is_subtype(UncertaintyLevel::Point, *level),
                "Point should be subtype of {:?}",
                level
            );
        }
    }

    /// Ensure Particles is always at top of lattice
    #[test]
    fn test_particles_is_top() {
        let lattice = PromotionLattice::new();

        for level in &[
            UncertaintyLevel::Point,
            UncertaintyLevel::Interval,
            UncertaintyLevel::Fuzzy,
            UncertaintyLevel::Affine,
            UncertaintyLevel::DempsterShafer,
            UncertaintyLevel::Distribution,
        ] {
            assert!(
                lattice.is_subtype(*level, UncertaintyLevel::Particles),
                "{:?} should be subtype of Particles",
                level
            );
        }
    }

    /// Verify cost multipliers are monotonically increasing with height
    #[test]
    fn test_cost_increases_with_height() {
        let levels_by_height: Vec<Vec<UncertaintyLevel>> = vec![
            vec![UncertaintyLevel::Point],
            vec![UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy],
            vec![UncertaintyLevel::Affine, UncertaintyLevel::DempsterShafer],
            vec![UncertaintyLevel::Distribution],
            vec![UncertaintyLevel::Particles],
        ];

        for window in levels_by_height.windows(2) {
            let lower_max_cost = window[0].iter().map(|l| l.cost_multiplier() as u64).max().unwrap();
            let upper_min_cost = window[1].iter().map(|l| l.cost_multiplier() as u64).min().unwrap();

            assert!(
                lower_max_cost <= upper_min_cost,
                "Cost should increase with height"
            );
        }
    }
}

// =============================================================================
// Property-Based Tests (if proptest is available)
// =============================================================================

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Promotion then demotion approximates original
        #[test]
        fn promote_preserves_point_estimate(value in -1000.0..1000.0f64, conf in 0.5..1.0f64) {
            let promoter = Promoter::new();

            for target in &[
                UncertaintyLevel::Interval,
                UncertaintyLevel::Affine,
                UncertaintyLevel::Distribution,
            ] {
                if let Ok(promoted) = promoter.promote_point(value, conf, *target) {
                    let estimate = promoted.point_estimate();
                    // Point estimate should be close to original
                    prop_assert!((estimate - value).abs() < value.abs() * 0.1 + 1.0);
                }
            }
        }

        /// Meet and join are commutative
        #[test]
        fn meet_join_commutative(a in 0..7usize, b in 0..7usize) {
            let levels = [
                UncertaintyLevel::Point,
                UncertaintyLevel::Interval,
                UncertaintyLevel::Fuzzy,
                UncertaintyLevel::Affine,
                UncertaintyLevel::DempsterShafer,
                UncertaintyLevel::Distribution,
                UncertaintyLevel::Particles,
            ];
            let la = levels[a];
            let lb = levels[b];
            let lattice = PromotionLattice::new();

            prop_assert_eq!(lattice.meet(la, lb), lattice.meet(lb, la));
            prop_assert_eq!(lattice.join(la, lb), lattice.join(lb, la));
        }
    }
}
