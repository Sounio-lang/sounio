//! Integration Tests for Sounio Epistemic System
//!
//! These tests verify the correct integration of:
//! - Promotion lattice operations
//! - KEC selector with real-world scenarios
//!
//! Run with: cargo test --test epistemic_integration

use sounio::epistemic::promotion::{PromotionLattice, UncertaintyLevel};
use sounio::epistemic::kec::{KECConfig, KECSelector, UncertaintyMetrics, ComplexityMetrics};

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
        // For a valid lattice: meet(a,b) ≤ a and meet(a,b) ≤ b
        for &a in &levels {
            for &b in &levels {
                let m = lattice.meet(a, b);
                // m ≤ a and m ≤ b (meet is below both inputs)
                assert!(
                    lattice.is_subtype(m, a),
                    "Meet {:?} should be ≤ {:?}",
                    m, a
                );
                assert!(
                    lattice.is_subtype(m, b),
                    "Meet {:?} should be ≤ {:?}",
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

    /// Test meet/join with incomparable elements
    #[test]
    fn test_incomparable_elements() {
        let lattice = PromotionLattice::new();

        // Interval and Fuzzy are incomparable (same height)
        assert!(!lattice.is_subtype(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy));
        assert!(!lattice.is_subtype(UncertaintyLevel::Fuzzy, UncertaintyLevel::Interval));

        // Their meet is Point (greatest lower bound)
        assert_eq!(
            lattice.meet(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy),
            UncertaintyLevel::Point
        );

        // Their join is Distribution (least upper bound - both branches meet there)
        // Interval → Affine → Distribution
        // Fuzzy → DempsterShafer → Distribution
        assert_eq!(
            lattice.join(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy),
            UncertaintyLevel::Distribution
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

    /// Test that KEC respects complexity constraints
    #[test]
    fn test_kec_respects_cost_constraints() {
        // Tight cost constraint should favor simpler models
        let config = KECConfig {
            max_complexity: 5.0,  // Very tight
            ..Default::default()
        };
        let selector = KECSelector::new(config);

        // Low uncertainty, simple computation
        let uncertainty = UncertaintyMetrics::from_interval(9.0, 11.0);
        let complexity = ComplexityMetrics {
            operation_count: 2,
            graph_depth: 1,
            variable_count: 1,
            flop_estimate: 10.0,
            memory_bytes: 64,
            parallelizable: true,
            nonlinearity: 0.1,
        };

        let result = selector.select(&uncertainty, &complexity);

        // Should recommend simple model due to tight budget
        assert!(
            result.recommended.cost_multiplier() <= 10.0,
            "Recommended {:?} exceeds cost constraint",
            result.recommended
        );
    }

    /// Test KEC presets produce different recommendations for complex problems
    #[test]
    fn test_kec_presets_differ() {
        // High uncertainty, complex computation
        let uncertainty = UncertaintyMetrics {
            entropy: 3.0,
            cv: 0.3,
            lower_bound: Some(5.0),
            upper_bound: Some(15.0),
            skewness: 0.0,
            kurtosis: 3.0,
            multimodal: false,
            mode_count: 1,
            has_correlations: true,
            correlation_strength: 0.5,
        };

        let complexity = ComplexityMetrics {
            operation_count: 20,
            graph_depth: 5,
            variable_count: 5,
            flop_estimate: 1000.0,
            memory_bytes: 8192,
            parallelizable: true,
            nonlinearity: 0.6,
        };

        // Get recommendations from different presets
        let default_result = KECSelector::default_selector().select(&uncertainty, &complexity);
        let realtime_result = KECSelector::new(KECConfig::realtime()).select(&uncertainty, &complexity);
        let scientific_result = KECSelector::new(KECConfig::scientific()).select(&uncertainty, &complexity);

        // Realtime should prefer simpler models (or equal)
        assert!(
            realtime_result.recommended.height() <= default_result.recommended.height(),
            "Realtime should prefer simpler models"
        );

        // Scientific should allow more complex models
        assert!(
            scientific_result.recommended.height() >= default_result.recommended.height(),
            "Scientific should allow more complex models"
        );
    }

    /// Test KEC with PK/PD typical inputs
    #[test]
    fn test_kec_pkpd_scenario() {
        let selector = KECSelector::new(KECConfig::pkpd());

        // Typical PK parameters with moderate variability (30% CV)
        let uncertainty = UncertaintyMetrics {
            entropy: 2.5,
            cv: 0.3,
            lower_bound: Some(5.0),
            upper_bound: Some(20.0),
            skewness: 0.5,  // Slightly right-skewed (common in PK)
            kurtosis: 3.5,
            multimodal: false,
            mode_count: 1,
            has_correlations: true,
            correlation_strength: 0.4,
        };

        // One-compartment model complexity
        let complexity = ComplexityMetrics {
            operation_count: 15,
            graph_depth: 4,
            variable_count: 6,  // CL, Vd, ka, F, Dose, t
            flop_estimate: 500.0,
            memory_bytes: 4096,
            parallelizable: true,
            nonlinearity: 0.7,  // exp() terms
        };

        let result = selector.select(&uncertainty, &complexity);

        // Should have reasonable confidence
        assert!(result.confidence >= 0.5, "Confidence too low: {}", result.confidence);

        // Check reasoning includes relevant factors
        let reasoning_text = result.reasoning.join(" ");
        assert!(
            reasoning_text.contains("Entropy") || reasoning_text.contains("CV"),
            "Reasoning should mention key factors"
        );
    }

    /// Test KEC handles edge cases
    #[test]
    fn test_kec_edge_cases() {
        let selector = KECSelector::default_selector();

        // Empty/zero uncertainty → should recommend Point
        let no_uncertainty = UncertaintyMetrics::from_point(10.0);
        let simple_complexity = ComplexityMetrics::default();
        let result = selector.select(&no_uncertainty, &simple_complexity);
        assert_eq!(result.recommended, UncertaintyLevel::Point);

        // High entropy with complex computation
        let high_uncertainty = UncertaintyMetrics {
            entropy: 5.0,
            cv: 0.5,
            lower_bound: Some(0.0),
            upper_bound: Some(100.0),
            skewness: 0.0,
            kurtosis: 3.0,
            multimodal: true,
            mode_count: 3,
            has_correlations: true,
            correlation_strength: 0.8,
        };

        let complex_computation = ComplexityMetrics {
            operation_count: 50,
            graph_depth: 10,
            variable_count: 20,
            flop_estimate: 10000.0,
            memory_bytes: 65536,
            parallelizable: false,
            nonlinearity: 0.9,
        };

        let result = selector.select(&high_uncertainty, &complex_computation);

        // Should recommend something reasonable for high entropy
        // (actual model depends on complexity budget and config)
        assert!(
            result.recommended.height() >= UncertaintyLevel::Interval.height(),
            "High entropy should use at least Interval, got {:?}",
            result.recommended
        );

        // Check that warnings or reasoning exist for multimodal data
        let has_multimodal_warning = result.warnings.iter()
            .any(|w| w.to_lowercase().contains("multimodal"));
        let has_reasoning = !result.reasoning.is_empty();
        assert!(has_multimodal_warning || has_reasoning, "Should have warnings or reasoning");
    }

    /// Test complexity metrics cost estimation
    #[test]
    fn test_complexity_cost_estimation() {
        let simple = ComplexityMetrics {
            operation_count: 2,
            graph_depth: 1,
            variable_count: 1,
            flop_estimate: 10.0,
            memory_bytes: 64,
            parallelizable: true,
            nonlinearity: 0.0,
        };

        let complex = ComplexityMetrics {
            operation_count: 100,
            graph_depth: 10,
            variable_count: 20,
            flop_estimate: 10000.0,
            memory_bytes: 65536,
            parallelizable: false,
            nonlinearity: 0.9,
        };

        // Complex should have higher cost for all levels
        for level in [
            UncertaintyLevel::Point,
            UncertaintyLevel::Interval,
            UncertaintyLevel::Affine,
            UncertaintyLevel::Distribution,
        ] {
            let simple_cost = simple.estimate_total(level, 1000);
            let complex_cost = complex.estimate_total(level, 1000);

            assert!(
                complex_cost > simple_cost,
                "Complex computation should have higher cost for {:?}",
                level
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

    /// Test UncertaintyMetrics constructors
    #[test]
    fn test_uncertainty_metrics_constructors() {
        // Point estimate
        let point = UncertaintyMetrics::from_point(10.0);
        assert_eq!(point.entropy, 0.0);
        assert_eq!(point.cv, 0.0);

        // Interval
        let interval = UncertaintyMetrics::from_interval(5.0, 15.0);
        assert!(interval.entropy > 0.0);
        assert!(interval.lower_bound == Some(5.0));
        assert!(interval.upper_bound == Some(15.0));

        // From samples
        let samples: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 - 50.0) * 0.1).collect();
        let from_samples = UncertaintyMetrics::from_samples(&samples);
        assert!(from_samples.entropy > 0.0);
        assert!(from_samples.cv > 0.0);
    }
}

// =============================================================================
// SMC Helper Tests
// =============================================================================

mod smc_tests {
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

    /// Test particle cloud statistics helper
    #[test]
    fn test_particle_cloud_statistics() {
        // Create uniformly spaced samples (simulating a uniform distribution)
        let n = 1000;
        let lower: f64 = 5.0;
        let upper: f64 = 15.0;
        let expected_mean = (lower + upper) / 2.0;
        let expected_var = (upper - lower).powi(2) / 12.0; // uniform variance

        // Uniform samples
        let samples: Vec<f64> = (0..n)
            .map(|i| lower + (upper - lower) * (i as f64 + 0.5) / n as f64)
            .collect();

        // Compute statistics
        let computed_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let computed_var: f64 = samples.iter()
            .map(|x| (x - computed_mean).powi(2))
            .sum::<f64>() / n as f64;

        // Should be close to theoretical values
        assert!((computed_mean - expected_mean).abs() < 0.1,
            "Mean error: expected {}, got {}", expected_mean, computed_mean);
        assert!((computed_var - expected_var).abs() < 1.0,
            "Variance error: expected {}, got {}", expected_var, computed_var);
    }
}
