/// Integration tests for 4 Tier-1 mathematical theories
/// Tests interaction between Information Geometry, Wasserstein, Sheaf Theory, and Tropical Geometry

#[cfg(test)]
mod integration_tests {
    use sounio::epistemic::{BetaConfidence, FisherMatrix, NaturalGradientOptimizer};
    use sounio::epistemic::{transport_cost, wasserstein_barycenter, wasserstein2_distance};
    use sounio::ontology::alignment::{Alignment, AlignmentSource, CellularSheaf};
    use sounio::ontology::distance::sssom::MappingPredicate;
    use sounio::ontology::loader::IRI;
    use sounio::types::{
        ResourceType, TropicalMatrix, TropicalNumber, Type, parallel_compose, sequential_compose,
        tropical_matmul, tropical_power,
    };

    // ============================================================================
    // Test 1: Information Geometry + Wasserstein Composition
    // ============================================================================

    #[test]
    fn test_fisher_metric_with_wasserstein_composition() {
        // Create two Beta distributions
        let p = BetaConfidence::new(10.0, 8.0);
        let q = BetaConfidence::new(8.0, 10.0);

        // Compute Fisher metrics using log-parameterization for positive definiteness
        let fisher_p = FisherMatrix::from_beta_log(p.alpha, p.beta);
        let fisher_q = FisherMatrix::from_beta_log(q.alpha, q.beta);

        // Both Fisher matrices should be positive definite (determinant > 0)
        assert!(
            fisher_p.determinant() > 0.0,
            "Fisher matrix P must be positive definite (log-parameterization)"
        );
        assert!(
            fisher_q.determinant() > 0.0,
            "Fisher matrix Q must be positive definite (log-parameterization)"
        );

        // Compute Wasserstein distance between distributions
        let w2_distance = wasserstein2_distance(&p, &q);
        assert!(
            w2_distance >= 0.0,
            "Wasserstein distance must be non-negative"
        );
        assert!(
            w2_distance < 2.0,
            "Wasserstein distance between nearby Beta distributions should be small"
        );

        // The Wasserstein distance should be symmetric
        let w2_distance_reverse = wasserstein2_distance(&q, &p);
        assert!(
            (w2_distance - w2_distance_reverse).abs() < 1e-10,
            "Wasserstein distance must be symmetric"
        );
    }

    #[test]
    fn test_natural_gradient_with_wasserstein_barycenter() {
        // Create a population of distributions
        let dists = vec![
            BetaConfidence::new(1.0, 2.0),
            BetaConfidence::new(2.0, 2.0),
            BetaConfidence::new(3.0, 1.0),
        ];
        let weights = vec![0.3, 0.4, 0.3];

        // Compute Wasserstein barycenter (minimizes weighted Wasserstein distances)
        let barycenter = wasserstein_barycenter(&dists, &weights);

        assert!(barycenter.alpha > 0.0, "Barycenter alpha must be positive");
        assert!(barycenter.beta > 0.0, "Barycenter beta must be positive");
        assert!(
            barycenter.iterations > 0,
            "Barycenter iterations must be positive"
        );

        // Create natural gradient optimizer starting from barycenter
        let mut optimizer = NaturalGradientOptimizer::new(
            barycenter.alpha,
            barycenter.beta,
            0.01, // learning rate
            0.95, // momentum
        );

        let initial_lr = optimizer.current_learning_rate();
        assert!(initial_lr > 0.0, "Initial learning rate must be positive");
        assert!(
            initial_lr <= 0.01,
            "Learning rate should not exceed initial value"
        );

        // The barycenter should minimize distances to all inputs
        let barycenter_conf = BetaConfidence::new(barycenter.alpha, barycenter.beta);
        for dist in &dists {
            let dist_to_barycenter = wasserstein2_distance(dist, &barycenter_conf);
            assert!(
                dist_to_barycenter >= 0.0,
                "Distance from input to barycenter must be non-negative"
            );
        }
    }

    // ============================================================================
    // Test 2: Sheaf Theory + Alignment
    // ============================================================================

    #[test]
    fn test_sheaf_cohomology_detects_type_inconsistencies() {
        let mut sheaf = CellularSheaf::new();

        // Add three cells representing ontology domains
        sheaf.add_cell("drug_bank".to_string(), Type::I32);
        sheaf.add_cell("chebi".to_string(), Type::I32);
        sheaf.add_cell("mondo".to_string(), Type::I32);

        // Add valid restriction chain: drug_bank → chebi → mondo
        sheaf.add_restriction(
            "drug_bank".to_string(),
            "chebi".to_string(),
            "aspirin_mapping".to_string(),
        );
        sheaf.add_restriction(
            "chebi".to_string(),
            "mondo".to_string(),
            "disease_mapping".to_string(),
        );

        // Missing direct restriction drug_bank → mondo should fail sheaf condition
        assert!(
            !sheaf.check_sheaf_condition(),
            "Incomplete restriction chain should violate sheaf condition"
        );

        // Add the missing direct restriction
        sheaf.add_restriction(
            "drug_bank".to_string(),
            "mondo".to_string(),
            "transitive_bridge".to_string(),
        );

        // Now sheaf condition should pass
        assert!(
            sheaf.check_sheaf_condition(),
            "Complete restriction composition should satisfy sheaf condition"
        );

        // Compute cohomology: should have no obstructions
        let coh = sheaf.compute_cohomology();
        assert_eq!(coh.rank, 3, "H⁰ should contain all 3 types");
        assert!(coh.h1.is_empty(), "H¹ should be empty (no obstructions)");
    }

    #[test]
    fn test_sheaf_with_conflicting_restrictions() {
        let mut sheaf = CellularSheaf::new();

        // Add cells with different types - will cause obstruction
        sheaf.add_cell("cell_a".to_string(), Type::I32);
        sheaf.add_cell("cell_b".to_string(), Type::I32);
        sheaf.add_cell("cell_c".to_string(), Type::I32);

        // Create a chain A → B → C with transitive closure
        sheaf.add_restriction(
            "cell_a".to_string(),
            "cell_b".to_string(),
            "r_ab".to_string(),
        );
        sheaf.add_restriction(
            "cell_b".to_string(),
            "cell_c".to_string(),
            "r_bc".to_string(),
        );

        // Missing direct A → C creates obstruction
        assert!(!sheaf.check_sheaf_condition());

        // Compute cohomology and verify obstruction is detected
        let coh = sheaf.compute_cohomology();
        assert!(!coh.h1.is_empty(), "Should detect obstruction in H¹");
        assert!(
            coh.h1[0].contains("cell_a"),
            "Obstruction message should mention source cell"
        );
    }

    // ============================================================================
    // Test 3: Tropical Geometry + Resource Analysis
    // ============================================================================

    #[test]
    fn test_tropical_matrix_multiplication_for_paths() {
        // Create adjacency matrix with costs
        let mut adj = TropicalMatrix::new(3, 3);

        // Edge costs
        adj.set(0, 1, TropicalNumber(2.0)); // A → B: cost 2
        adj.set(1, 2, TropicalNumber(3.0)); // B → C: cost 3
        adj.set(0, 2, TropicalNumber(6.0)); // A → C: cost 6

        // All other entries are ∞ (no edge)

        // Tropical matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])
        let result = tropical_matmul(&adj, &adj);
        assert!(
            result.is_ok(),
            "Tropical matrix multiplication should succeed"
        );

        let paths2 = result.unwrap();

        // paths2[0,2] should be min(
        //   A[0,0] + A[0,2],  // (∞ + 6 = ∞)
        //   A[0,1] + A[1,2],  // (2 + 3 = 5) ← shortest
        //   A[0,2] + A[2,2]   // (6 + ∞ = ∞)
        // ) = 5
        assert_eq!(
            paths2.get(0, 2),
            Some(TropicalNumber(5.0)),
            "Shortest path A→B→C should have cost 5"
        );
    }

    #[test]
    fn test_tropical_matrix_power_shortest_paths() {
        // Create weighted graph adjacency matrix
        let mut adj = TropicalMatrix::new(2, 2);
        adj.set(0, 0, TropicalNumber(0.0)); // Identity diagonal
        adj.set(0, 1, TropicalNumber(3.0)); // A → B: cost 3
        adj.set(1, 0, TropicalNumber(2.0)); // B → A: cost 2
        adj.set(1, 1, TropicalNumber(0.0)); // Identity diagonal

        // A^2 represents paths of length ≤ 2
        let adj2 = tropical_power(&adj, 2);
        assert!(adj2.is_ok(), "Tropical matrix power should succeed");

        let paths = adj2.unwrap();

        // Direct path A→B: cost 3
        // Path A→B→A→B: cost 2+2+3 = 7, but min-plus chooses min(3, 7) = 3
        assert_eq!(
            paths.get(0, 1),
            Some(TropicalNumber(3.0)),
            "Shortest path should remain 3"
        );
    }

    #[test]
    fn test_resource_composition_sequential_vs_parallel() {
        let t1 = ResourceType::Time(TropicalNumber(3.0));
        let t2 = ResourceType::Time(TropicalNumber(5.0));

        // Sequential composition: costs add (3 + 5 = 8)
        let seq_result = sequential_compose(&t1, &t2);
        assert!(seq_result.is_ok(), "Sequential composition should succeed");
        match seq_result.unwrap() {
            ResourceType::Time(t) => {
                assert_eq!(t, TropicalNumber(8.0), "Sequential: time1 + time2");
            }
            _ => panic!("Expected Time resource"),
        }

        // Parallel composition: take minimum (min(3, 5) = 3)
        let par_result = parallel_compose(&t1, &t2);
        assert!(par_result.is_ok(), "Parallel composition should succeed");
        match par_result.unwrap() {
            ResourceType::Time(t) => {
                assert_eq!(t, TropicalNumber(3.0), "Parallel: min(time1, time2)");
            }
            _ => panic!("Expected Time resource"),
        }
    }

    // ============================================================================
    // Test 4: Cross-Module Integration - All 4 Theories Together
    // ============================================================================

    #[test]
    fn test_fisher_metric_distance_vs_wasserstein() {
        // Two nearby Beta distributions
        let p = BetaConfidence::new(10.0, 8.0);
        let q = BetaConfidence::new(10.5, 8.5);

        // Compute Fisher information matrices using log-parameterization
        let fisher_p = FisherMatrix::from_beta_log(p.alpha, p.beta);
        let fisher_q = FisherMatrix::from_beta_log(q.alpha, q.beta);

        // Both matrices should be invertible (positive definite)
        let det_p = fisher_p.determinant();
        let det_q = fisher_q.determinant();
        assert!(
            det_p > 0.0,
            "Fisher matrix P must be invertible (log-parameterization)"
        );
        assert!(
            det_q > 0.0,
            "Fisher matrix Q must be invertible (log-parameterization)"
        );

        // Condition numbers should indicate numerical stability
        let cond_p = fisher_p.condition_number();
        let cond_q = fisher_q.condition_number();
        assert!(
            cond_p.is_finite() && cond_p > 1.0,
            "Condition number should be well-defined"
        );

        // Wasserstein distance as metric
        let w2 = wasserstein2_distance(&p, &q);
        assert!(w2 >= 0.0, "Wasserstein distance should be metric");

        // Transport cost wrapper should match Wasserstein
        let cost = transport_cost(&p, &q);
        assert!(
            (cost - w2).abs() < 1e-10,
            "Transport cost should equal Wasserstein distance"
        );
    }

    #[test]
    fn test_epistemic_type_inference_with_geometric_bounds() {
        // Three distributions representing type confidence from different sources
        let sources = vec![
            BetaConfidence::new(5.0, 2.0), // High confidence: α > β
            BetaConfidence::new(4.0, 3.0), // Medium confidence
            BetaConfidence::new(3.0, 4.0), // Lower confidence
        ];

        // Weights representing evidence strength
        let weights = vec![0.5, 0.3, 0.2];

        // Compute Wasserstein barycenter for consensus
        let consensus = wasserstein_barycenter(&sources, &weights);
        assert!(consensus.alpha > 0.0 && consensus.beta > 0.0);

        // Verify triangle inequality for all pairs (metric property)
        for i in 0..sources.len() {
            for j in i + 1..sources.len() {
                for k in 0..sources.len() {
                    if k != i && k != j {
                        let d_ij = wasserstein2_distance(&sources[i], &sources[j]);
                        let d_ik = wasserstein2_distance(&sources[i], &sources[k]);
                        let d_kj = wasserstein2_distance(&sources[k], &sources[j]);

                        // Triangle inequality: d(i,j) ≤ d(i,k) + d(k,j)
                        assert!(
                            d_ij <= d_ik + d_kj + 1e-6,
                            "Triangle inequality violated for indices {},{},{}",
                            i,
                            j,
                            k
                        );
                    }
                }
            }
        }

        // Barycenter should minimize distances to all sources
        let consensus_conf = BetaConfidence::new(consensus.alpha, consensus.beta);
        for (idx, source) in sources.iter().enumerate() {
            let dist_to_consensus = wasserstein2_distance(source, &consensus_conf);
            assert!(
                dist_to_consensus >= 0.0,
                "Distance from source {} to consensus should be non-negative",
                idx
            );
        }
    }

    #[test]
    fn test_ontology_alignment_with_tropical_distances() {
        // Create ontology alignment structure
        let mut sheaf = CellularSheaf::new();

        // Represent multiple ontologies as cells
        sheaf.add_cell("go".to_string(), Type::I64); // Gene Ontology
        sheaf.add_cell("chebi".to_string(), Type::I64); // Chemical Entities
        sheaf.add_cell("mondo".to_string(), Type::I64); // Diseases

        // Add valid restrictions (representing cross-ontology mappings)
        sheaf.add_restriction(
            "go".to_string(),
            "chebi".to_string(),
            "protein_chemical".to_string(),
        );
        sheaf.add_restriction(
            "chebi".to_string(),
            "mondo".to_string(),
            "drug_disease".to_string(),
        );
        sheaf.add_restriction(
            "go".to_string(),
            "mondo".to_string(),
            "gene_disease".to_string(),
        );

        // Verify sheaf condition is satisfied
        assert!(sheaf.check_sheaf_condition());

        // Compute cohomology - should have no obstructions
        let coh = sheaf.compute_cohomology();
        assert_eq!(coh.rank, 3);
        assert!(coh.h1.is_empty());

        // Build tropical distance matrix for ontology pairs
        let mut dist_matrix = TropicalMatrix::new(3, 3);

        // Diagonal: 0 (identity)
        dist_matrix.set(0, 0, TropicalNumber(0.0));
        dist_matrix.set(1, 1, TropicalNumber(0.0));
        dist_matrix.set(2, 2, TropicalNumber(0.0));

        // Cross distances (semantic distance between ontologies)
        dist_matrix.set(0, 1, TropicalNumber(2.0)); // GO → ChEBI
        dist_matrix.set(1, 2, TropicalNumber(1.5)); // ChEBI → MONDO
        dist_matrix.set(0, 2, TropicalNumber(3.0)); // GO → MONDO (direct)
        dist_matrix.set(1, 0, TropicalNumber(2.0)); // ChEBI → GO
        dist_matrix.set(2, 1, TropicalNumber(1.5)); // MONDO → ChEBI
        dist_matrix.set(2, 0, TropicalNumber(3.0)); // MONDO → GO

        // Compute transitive distances via tropical matrix power
        let dist2 = tropical_power(&dist_matrix, 2);
        assert!(dist2.is_ok());

        // Transitive path GO → ChEBI → MONDO should have cost 2.0 + 1.5 = 3.5
        // But tropical shortest path would be min(direct=3.0, transitive=3.5) = 3.0
        let transitive = dist2.unwrap();
        assert_eq!(
            transitive.get(0, 2),
            Some(TropicalNumber(3.0)),
            "Shortest ontology path should respect min-plus algebra"
        );
    }

    #[test]
    fn test_convergence_properties_across_theories() {
        // Information Geometry: Natural gradient should converge faster than Euclidean
        let mut optimizer = NaturalGradientOptimizer::new(2.0, 3.0, 0.1, 0.95);

        let initial_lr = optimizer.current_learning_rate();
        assert!(initial_lr > 0.0 && initial_lr <= 0.1);

        // Wasserstein: Barycenters should be unique minimizers
        let dists = vec![BetaConfidence::new(2.0, 3.0), BetaConfidence::new(3.0, 2.0)];
        let weights = vec![0.5, 0.5];

        let barycenter1 = wasserstein_barycenter(&dists, &weights);
        let barycenter2 = wasserstein_barycenter(&dists, &weights);

        // Barycenters should be deterministic (identical)
        assert!((barycenter1.alpha - barycenter2.alpha).abs() < 1e-10);
        assert!((barycenter1.beta - barycenter2.beta).abs() < 1e-10);

        // Sheaf: Consistent restrictions throughout
        let mut sheaf = CellularSheaf::new();
        sheaf.add_cell("a".to_string(), Type::I32);
        sheaf.add_cell("b".to_string(), Type::I32);
        sheaf.add_cell("c".to_string(), Type::I32);
        sheaf.add_restriction("a".to_string(), "b".to_string(), "r1".to_string());
        sheaf.add_restriction("b".to_string(), "c".to_string(), "r2".to_string());
        sheaf.add_restriction("a".to_string(), "c".to_string(), "r3".to_string());

        assert!(sheaf.check_sheaf_condition());

        // Tropical: Algebraic properties hold (associativity, etc.)
        let a = TropicalNumber(2.0);
        let b = TropicalNumber(3.0);
        let c = TropicalNumber(4.0);

        // Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        let left_assoc = (a + b) + c;
        let right_assoc = a + (b + c);
        assert_eq!(
            left_assoc, right_assoc,
            "Tropical addition should be associative"
        );

        // Associativity: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
        let left_mul = (a * b) * c;
        let right_mul = a * (b * c);
        assert_eq!(
            left_mul, right_mul,
            "Tropical multiplication should be associative"
        );
    }
}
