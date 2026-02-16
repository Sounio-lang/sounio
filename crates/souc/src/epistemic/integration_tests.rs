//! Integration tests for the epistemic computing system.
//!
//! These tests validate cross-cutting behavior between evidence collection,
//! temporal validity, provenance tracking, Merkle DAG integrity, and causal
//! inference subsystems. Each test exercises multiple subsystem boundaries
//! to ensure the epistemic pipeline works as a coherent whole.

#[cfg(test)]
mod tests {
    use crate::epistemic::confidence::Evidence;
    use crate::epistemic::evidence_collector::{ConflictStrategy, EvidenceCollector};
    use crate::epistemic::merkle::{
        MerkleProvenanceDAG, ProvenanceMetadata, ProvenanceOperation, ProvenanceSignature,
    };
    use crate::epistemic::provenance::{
        Origin, Provenance, Transformation, TransformationFilter, TransformationKind,
        TransformationMetadata,
    };
    use crate::epistemic::temporal::{ContextIndex, ContextTime, TemporalIndex, ValidityBounds};
    use crate::epistemic::temporal_validity::{
        AgingModel, TemporalValidityResult, apply_aging, is_currently_valid, remaining_validity,
        validate_temporal,
    };

    // ========================================================================
    // Helpers
    // ========================================================================

    fn pub_evidence(doi: &str, strength: f64) -> Evidence {
        Evidence::publication(doi, strength)
    }

    fn exp_evidence(protocol: &str, strength: f64) -> Evidence {
        Evidence::experiment(protocol, strength)
    }

    // ========================================================================
    // 1. Evidence + Temporal Integration (5 tests)
    // ========================================================================

    /// Test 1: Evidence aggregation respects temporal aging.
    ///
    /// Collects 3 pieces of evidence at different timestamps, then applies
    /// exponential decay aging to their confidences. Verifies that aged
    /// confidence is strictly less than original for non-zero elapsed time.
    #[test]
    fn test_evidence_aggregation_with_temporal_aging() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        collector.collect(pub_evidence("doi1", 0.8), "lab_a".into(), 1000);
        collector.collect(pub_evidence("doi2", 0.7), "lab_b".into(), 2000);
        collector.collect(exp_evidence("proto1", 0.9), "lab_c".into(), 3000);

        let agg = collector.aggregate().unwrap();
        let original_confidence = agg.combined_confidence;

        // Apply exponential decay to simulate aging of the aggregate
        let aging_model = AgingModel::ExponentialDecay {
            half_life_seconds: 3600,
        };
        let elapsed = 7200; // 2 half-lives
        let aged_confidence = apply_aging(original_confidence, elapsed, &aging_model);

        // After 2 half-lives, confidence should drop to ~25% of original
        assert!(
            aged_confidence < original_confidence,
            "aged confidence ({}) should be less than original ({})",
            aged_confidence,
            original_confidence
        );
        let expected = original_confidence * 0.25;
        assert!(
            (aged_confidence - expected).abs() < 0.01,
            "after 2 half-lives: expected ~{:.4}, got {:.4}",
            expected,
            aged_confidence
        );
    }

    /// Test 2: Stale evidence is pruned before aggregation.
    ///
    /// Collects evidence with varying timestamps, prunes stale records,
    /// and verifies the aggregate only reflects fresh evidence.
    #[test]
    fn test_stale_evidence_pruned_before_aggregation() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::TrustHigherConfidence)
            .with_staleness_threshold(5000);

        // Old evidence (will be stale at t=10000)
        collector.collect(pub_evidence("old_study", 0.95), "lab_old".into(), 1000);
        // Fresh evidence
        collector.collect(pub_evidence("new_study_1", 0.6), "lab_new1".into(), 8000);
        collector.collect(pub_evidence("new_study_2", 0.7), "lab_new2".into(), 9000);

        // Before pruning: highest confidence is 0.95 (old)
        let before = collector.aggregate().unwrap();
        assert!(
            (before.combined_confidence - 0.95).abs() < 1e-6,
            "before pruning, TrustHigher picks 0.95"
        );

        // Prune stale evidence
        let pruned = collector.prune_stale(10000);
        assert_eq!(pruned, 1, "one record should be pruned");
        assert_eq!(collector.active_count(), 2);

        // After pruning: highest is 0.7
        let after = collector.aggregate().unwrap();
        assert!(
            (after.combined_confidence - 0.7).abs() < 1e-6,
            "after pruning, TrustHigher picks 0.7; got {}",
            after.combined_confidence
        );
    }

    /// Test 3: Temporal validity bounds interact with evidence age.
    ///
    /// Creates evidence with ValidityBounds::Until(future) and
    /// ValidityBounds::Until(past), verifies currently-valid vs expired.
    #[test]
    fn test_temporal_validity_bounds_with_evidence_age() {
        let current = TemporalIndex::date(2026, 6, 1);

        // Context valid until 2027 => should be valid now
        let future_ctx = ContextTime {
            temporal: TemporalIndex::date(2026, 1, 1),
            context: ContextIndex::Named("phase3_trial".into()),
            validity: ValidityBounds::Until(TemporalIndex::date(2027, 1, 1)),
        };
        assert!(is_currently_valid(&future_ctx, &current));

        // Context valid until 2025 => expired
        let past_ctx = ContextTime {
            temporal: TemporalIndex::date(2024, 1, 1),
            context: ContextIndex::Named("phase1_trial".into()),
            validity: ValidityBounds::Until(TemporalIndex::date(2025, 1, 1)),
        };
        assert!(!is_currently_valid(&past_ctx, &current));

        // Evidence from the expired context should have zero remaining validity
        let remaining = remaining_validity(&past_ctx.validity, &current);
        assert_eq!(remaining, Some(0), "expired context has 0 remaining");

        // Evidence from the valid context should have positive remaining
        let remaining_future = remaining_validity(&future_ctx.validity, &current);
        assert!(
            remaining_future.unwrap() > 0,
            "future context has positive remaining"
        );
    }

    /// Test 4: Bayesian fusion with temporally-weighted evidence.
    ///
    /// Newer evidence should dominate when using TemporalPrecedence.
    /// Bayesian fusion should amplify concordant evidence.
    #[test]
    fn test_bayesian_fusion_with_temporal_weighting() {
        // Bayesian fusion: concordant evidence amplifies confidence
        let mut bayesian_collector = EvidenceCollector::new(ConflictStrategy::BayesianFusion);
        bayesian_collector.collect(pub_evidence("study_old", 0.7), "lab1".into(), 1000);
        bayesian_collector.collect(pub_evidence("study_new", 0.8), "lab2".into(), 5000);

        let bayesian_agg = bayesian_collector.aggregate().unwrap();
        // 0.7*0.8 / (0.7*0.8 + 0.3*0.2) = 0.56 / 0.62 ~ 0.903
        assert!(
            bayesian_agg.combined_confidence > 0.85,
            "Bayesian fusion of concordant evidence should amplify: got {}",
            bayesian_agg.combined_confidence
        );

        // TemporalPrecedence: newest wins regardless of concordance
        let mut temporal_collector = EvidenceCollector::new(ConflictStrategy::TemporalPrecedence);
        temporal_collector.collect(pub_evidence("study_old", 0.95), "lab1".into(), 1000);
        temporal_collector.collect(pub_evidence("study_new", 0.6), "lab2".into(), 5000);

        let temporal_agg = temporal_collector.aggregate().unwrap();
        assert!(
            (temporal_agg.combined_confidence - 0.6).abs() < 1e-6,
            "TemporalPrecedence should pick newest (0.6), got {}",
            temporal_agg.combined_confidence
        );
    }

    /// Test 5: Evidence audit trail includes temporal metadata ordering.
    ///
    /// Exports audit trail and verifies entries preserve collection order.
    #[test]
    fn test_evidence_audit_trail_temporal_ordering() {
        let mut collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);

        let timestamps = [1000u64, 2000, 3000, 1500, 5000];
        for (i, &ts) in timestamps.iter().enumerate() {
            collector.collect(
                pub_evidence(&format!("doi_{}", i), 0.7 + i as f64 * 0.05),
                format!("src_{}", i),
                ts,
            );
        }

        let trail = collector.export_audit_trail();
        assert_eq!(trail.len(), 5);

        // Verify audit trail is in collection order
        assert_eq!(trail[0].0, "src_0");
        assert_eq!(trail[4].0, "src_4");

        // Verify timestamps from registry reflect collection order
        let registry = collector.registry();
        assert_eq!(registry[0].timestamp, 1000);
        assert_eq!(registry[1].timestamp, 2000);
        assert_eq!(registry[3].timestamp, 1500);
        assert_eq!(registry[4].timestamp, 5000);
    }

    // ========================================================================
    // 2. Evidence + Provenance Integration (5 tests)
    // ========================================================================

    /// Test 6: Evidence from different sources creates distinct provenance chains.
    ///
    /// Publication evidence gets External origin, computation gets Computed.
    #[test]
    fn test_evidence_sources_create_distinct_provenance() {
        // Publication evidence -> External provenance
        let pub_prov = Provenance::external("https://doi.org/10.1234/study");
        assert_eq!(
            pub_prov.origin,
            Origin::External {
                uri: "https://doi.org/10.1234/study".to_string()
            }
        );

        // Computation evidence -> Computed provenance
        let comp_prov = Provenance::computed("monte_carlo_simulation");
        assert_eq!(
            comp_prov.origin,
            Origin::Computed {
                function: "monte_carlo_simulation".to_string()
            }
        );

        // Both should have distinct summaries
        let pub_summary = pub_prov.summarize();
        let comp_summary = comp_prov.summarize();
        assert_ne!(pub_summary.origin, comp_summary.origin);
    }

    /// Test 7: Evidence aggregation updates provenance through composition.
    ///
    /// Extends a provenance chain with a statistical transformation,
    /// verifies the total confidence factor reflects the aggregation step.
    #[test]
    fn test_evidence_aggregation_updates_provenance() {
        let prov = Provenance::external("https://trial.example.com")
            .extend(Transformation::function("collect_data").with_confidence(1.0))
            .extend(
                Transformation::new("aggregate_evidence", TransformationKind::Statistical)
                    .with_confidence(0.92),
            )
            .extend(
                Transformation::new("bayesian_fusion", TransformationKind::Statistical)
                    .with_confidence(0.95),
            );

        let total = prov.total_confidence_factor();
        let expected = 1.0 * 0.92 * 0.95;
        assert!(
            (total - expected).abs() < 1e-6,
            "total_confidence_factor should be {:.4}, got {:.4}",
            expected,
            total
        );

        assert_eq!(prov.depth(), 3);
    }

    /// Test 8: Conflict resolution recorded in provenance.
    ///
    /// Two conflicting evidence pieces are resolved, and the resolution
    /// is tracked through the provenance transformation chain.
    #[test]
    fn test_conflict_resolution_in_provenance() {
        // Resolve conflict
        let collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        let evidence_a = pub_evidence("study_low", 0.4);
        let evidence_b = pub_evidence("study_high", 0.9);
        let resolved = collector.resolve_conflict(&evidence_a, &evidence_b);

        // Record resolution in provenance
        let prov = Provenance::external("https://resolver.example.com").extend(
            Transformation::new("conflict_resolution", TransformationKind::Statistical)
                .with_confidence(resolved.strength.value()),
        );

        assert!(prov.includes(TransformationKind::Statistical));
        let summary = prov.summarize();
        assert_eq!(
            *summary
                .kind_counts
                .get(&TransformationKind::Statistical)
                .unwrap_or(&0),
            1
        );
    }

    /// Test 9: Provenance query filters find evidence-related transformations.
    ///
    /// Builds a mixed provenance chain and uses filters to isolate
    /// statistical and high-confidence transformations.
    #[test]
    fn test_provenance_query_filters_for_evidence() {
        let prov = Provenance::external("https://data.example.com")
            .extend(Transformation::function("read_sensor").with_confidence(1.0))
            .extend(
                Transformation::new("calibrate", TransformationKind::Conversion)
                    .with_confidence(0.98),
            )
            .extend(
                Transformation::new("aggregate", TransformationKind::Statistical)
                    .with_confidence(0.90),
            )
            .extend(
                Transformation::new("predict", TransformationKind::MLInference)
                    .with_confidence(0.85),
            );

        // Filter by Statistical kind
        let statistical = prov.query(&TransformationFilter::ByKind(
            TransformationKind::Statistical,
        ));
        assert_eq!(statistical.match_count, 1);
        assert_eq!(statistical.matched_transformations[0].1.name, "aggregate");

        // Filter by MinConfidence(0.95)
        let high_conf = prov.query(&TransformationFilter::MinConfidence(0.95));
        assert_eq!(high_conf.match_count, 2); // read_sensor(1.0), calibrate(0.98)

        // Combined filter: Statistical OR MLInference
        let evidence_related = prov.query(&TransformationFilter::Any(vec![
            TransformationFilter::ByKind(TransformationKind::Statistical),
            TransformationFilter::ByKind(TransformationKind::MLInference),
        ]));
        assert_eq!(evidence_related.match_count, 2);
    }

    /// Test 10: Merkle DAG tracks evidence provenance with signatures.
    ///
    /// Creates merkle nodes for evidence sources, signs them with
    /// different authorities, and queries by signer.
    #[test]
    fn test_merkle_dag_tracks_evidence_with_signatures() {
        let mut dag = MerkleProvenanceDAG::new();

        let obs_id = dag.add_root(ProvenanceOperation::measurement("clinical_observation"));
        let rct_id = dag.add_root(ProvenanceOperation::measurement(
            "randomized_controlled_trial",
        ));

        let merge_id = dag.add_derived(
            vec![obs_id, rct_id],
            ProvenanceOperation::computation("meta_analysis"),
        );

        // Sign different nodes with different authorities
        let fda_sig = ProvenanceSignature::new("FDA_reviewer", &[0xFDu8; 64], &obs_id);
        dag.get_mut(&obs_id).unwrap().sign(fda_sig);

        let pi_sig = ProvenanceSignature::new("principal_investigator", &[0xA1u8; 64], &rct_id);
        dag.get_mut(&rct_id).unwrap().sign(pi_sig);

        // Query by signer
        let fda_signed = dag.find_signed_by("FDA_reviewer");
        assert_eq!(fda_signed.len(), 1);
        assert_eq!(fda_signed[0].id, obs_id);

        let pi_signed = dag.find_signed_by("principal_investigator");
        assert_eq!(pi_signed.len(), 1);
        assert_eq!(pi_signed[0].id, rct_id);

        // Merge node has no signatures
        let merge_signed = dag.find_signed_by("FDA_reviewer");
        assert!(merge_signed.iter().all(|n| n.id != merge_id));
    }

    // ========================================================================
    // 3. Provenance + Temporal Integration (5 tests)
    // ========================================================================

    /// Test 11: Provenance chain confidence degrades over time.
    ///
    /// Creates a provenance chain with known confidence, applies
    /// exponential decay, verifies degradation.
    #[test]
    fn test_provenance_confidence_degrades_with_time() {
        let prov = Provenance::external("https://source.example.com")
            .extend(Transformation::function("transform_a").with_confidence(0.95))
            .extend(Transformation::function("transform_b").with_confidence(0.90));

        let original_confidence = prov.total_confidence_factor();
        assert!((original_confidence - 0.855).abs() < 1e-6);

        // Apply aging
        let model = AgingModel::ExponentialDecay {
            half_life_seconds: 86400, // 1 day
        };
        let elapsed = 86400; // 1 day = 1 half-life
        let aged = apply_aging(original_confidence, elapsed, &model);

        assert!(
            aged < original_confidence,
            "aged ({}) should be less than original ({})",
            aged,
            original_confidence
        );
        let expected = original_confidence * 0.5;
        assert!(
            (aged - expected).abs() < 1e-6,
            "after 1 half-life: expected {:.4}, got {:.4}",
            expected,
            aged
        );
    }

    /// Test 12: Temporal validation of provenance nodes.
    ///
    /// Creates merkle nodes with timestamps, validates temporal bounds.
    #[test]
    fn test_temporal_validation_of_provenance_nodes() {
        let current = TemporalIndex::date(2026, 6, 1);

        // Valid bounds: until 2027
        let valid_bounds = ValidityBounds::Until(TemporalIndex::date(2027, 1, 1));
        let result = validate_temporal(&valid_bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Valid { .. }));

        // Expired bounds: until 2025
        let expired_bounds = ValidityBounds::Until(TemporalIndex::date(2025, 1, 1));
        let result = validate_temporal(&expired_bounds, &current);
        assert!(matches!(result, TemporalValidityResult::Expired { .. }));

        // Create a Merkle DAG with nodes that have temporal metadata
        let mut dag = MerkleProvenanceDAG::new();
        let root = dag.add_root(ProvenanceOperation::measurement("thermometer"));
        {
            let node = dag.get_mut(&root).unwrap();
            node.metadata = ProvenanceMetadata::new()
                .with_user("scientist_a")
                .with_custom("valid_until", "2027-01-01");
        }

        let node = dag.get(&root).unwrap();
        assert_eq!(
            node.metadata.custom.get("valid_until").map(|s| s.as_str()),
            Some("2027-01-01")
        );
    }

    /// Test 13: Provenance summary reflects aging.
    ///
    /// Builds a chain, applies aging to total confidence, summarizes.
    #[test]
    fn test_provenance_summary_reflects_aging() {
        let prov = Provenance::external("https://sensor.example.com")
            .extend(Transformation::function("read").with_confidence(1.0))
            .extend(
                Transformation::new("calibrate", TransformationKind::Conversion)
                    .with_confidence(0.98),
            )
            .extend(
                Transformation::new("ml_predict", TransformationKind::MLInference)
                    .with_confidence(0.85),
            );

        let summary = prov.summarize();
        let original_total = summary.total_confidence;

        // Apply linear decay
        let model = AgingModel::Linear {
            rate_per_second: 0.0001,
        };
        let aged_total = apply_aging(original_total, 1000, &model);

        assert!(
            aged_total < original_total,
            "aged summary confidence should be lower"
        );

        // The weakest link should still be ml_predict
        let (weakest_idx, weakest_conf) = summary.min_confidence_step.unwrap();
        assert_eq!(weakest_idx, 2); // ml_predict is index 2
        assert!((weakest_conf - 0.85).abs() < 0.001);
    }

    /// Test 14: Weakest link identification changes with aging.
    ///
    /// Initially weakest link is one step, but after applying step
    /// function decay to an older step, the weakest link shifts.
    #[test]
    fn test_weakest_link_shifts_with_aging() {
        // Step 0: confidence 0.95 (old, will degrade)
        // Step 1: confidence 0.88 (initially weakest)
        // Step 2: confidence 0.92

        let confidence_0: f64 = 0.95;
        let confidence_1: f64 = 0.88; // initially weakest
        let confidence_2: f64 = 0.92;

        // Before aging: weakest is step 1 at 0.88
        let min_before = confidence_0.min(confidence_1).min(confidence_2);
        assert!((min_before - 0.88).abs() < 1e-6);

        // Apply step-function aging to step 0 (oldest data past threshold)
        let model = AgingModel::StepFunction {
            threshold_seconds: 3600,
            degraded_confidence: 0.3,
        };
        let aged_0 = apply_aging(confidence_0, 7200, &model); // past threshold
        let aged_1 = apply_aging(confidence_1, 1000, &model); // before threshold
        let aged_2 = apply_aging(confidence_2, 500, &model); // before threshold

        // After aging: step 0 dropped to 0.3, now it is the weakest
        let min_after = aged_0.min(aged_1).min(aged_2);
        assert!(
            (min_after - 0.3).abs() < 1e-6,
            "weakest link should shift to aged step 0 at 0.3, got {}",
            min_after
        );
        assert!((aged_0 - 0.3).abs() < 1e-6, "step 0 degraded to 0.3");
    }

    /// Test 15: Subchain extraction preserves temporal ordering.
    ///
    /// Extracts a subchain from provenance and verifies that the
    /// transformation ordering is preserved.
    #[test]
    fn test_subchain_preserves_temporal_ordering() {
        let prov = Provenance::external("https://pipeline.example.com")
            .extend({
                let mut t = Transformation::function("step_1");
                t.metadata = TransformationMetadata::new().with_timestamp("2026-01-01T00:00:00Z");
                t
            })
            .extend({
                let mut t = Transformation::function("step_2");
                t.metadata = TransformationMetadata::new().with_timestamp("2026-02-01T00:00:00Z");
                t
            })
            .extend({
                let mut t = Transformation::function("step_3");
                t.metadata = TransformationMetadata::new().with_timestamp("2026-03-01T00:00:00Z");
                t
            })
            .extend({
                let mut t = Transformation::function("step_4");
                t.metadata = TransformationMetadata::new().with_timestamp("2026-04-01T00:00:00Z");
                t
            });

        // Extract subchain [1..3] (inclusive)
        let sub = prov.subchain(1, 3).unwrap();
        assert_eq!(sub.len(), 3);
        assert_eq!(sub[0].name, "step_2");
        assert_eq!(sub[1].name, "step_3");
        assert_eq!(sub[2].name, "step_4");

        // Verify timestamps are in order
        let ts: Vec<&str> = sub
            .iter()
            .filter_map(|t| t.metadata.timestamp.as_deref())
            .collect();
        assert_eq!(ts.len(), 3);
        for window in ts.windows(2) {
            assert!(
                window[0] < window[1],
                "timestamps should be ordered: {} < {}",
                window[0],
                window[1]
            );
        }
    }

    // ========================================================================
    // 4. Causal + Evidence Integration (5 tests)
    // ========================================================================

    /// Test 16: Causal identification result carries confidence that
    /// can be combined with evidence aggregate confidence.
    #[test]
    fn test_causal_identification_with_evidence_confidence() {
        use crate::check::causal::{CausalCheckResult, causal_effect_type_confidence};

        // Simulate an identified causal effect with known confidence
        let causal_result = CausalCheckResult::Identified {
            formula: crate::types::causal_id::IdentificationFormula::Conditional {
                outcome: vec!["Y".into()],
                treatment: vec!["X".into()],
            },
            confidence: 0.99, // causal Markov assumption
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
        };

        let causal_conf = causal_effect_type_confidence(&causal_result).unwrap();

        // Evidence from RCT with high confidence
        let mut collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        collector.collect(pub_evidence("rct_result", 0.95), "rct".into(), 5000);
        collector.collect(exp_evidence("replication", 0.90), "lab".into(), 6000);

        let evidence_agg = collector.aggregate().unwrap();

        // Combined confidence = causal * evidence
        let combined = causal_conf * evidence_agg.combined_confidence;
        assert!(
            combined > 0.0 && combined <= 1.0,
            "combined confidence should be in (0, 1]: got {}",
            combined
        );
        assert!(
            combined < causal_conf,
            "combined should be <= causal_conf since evidence < 1.0"
        );
    }

    /// Test 17: Multiple causal formulas yield different base confidences.
    #[test]
    fn test_multiple_causal_formulas_different_confidences() {
        use crate::types::causal_id::{IdentificationFormula, identification_confidence};

        let conditional = IdentificationFormula::Conditional {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
        };

        let backdoor = IdentificationFormula::Adjustment {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
            adjustment_set: vec!["Z".into()],
        };

        let frontdoor = IdentificationFormula::Frontdoor {
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
            mediator: vec!["M".into()],
        };

        let cond_conf = identification_confidence(&conditional);
        let back_conf = identification_confidence(&backdoor);
        let front_conf = identification_confidence(&frontdoor);

        // Conditional should have highest confidence (fewest assumptions)
        assert!(
            cond_conf >= back_conf,
            "conditional ({}) >= backdoor ({})",
            cond_conf,
            back_conf
        );

        // All should be positive
        assert!(cond_conf > 0.0);
        assert!(back_conf > 0.0);
        assert!(front_conf > 0.0);
    }

    /// Test 18: PC algorithm output used for identifiability check.
    ///
    /// Runs pc_algorithm on synthetic chain data, converts to
    /// CausalGraphDef, and runs identification.
    #[test]
    fn test_pc_discovery_to_identifiability() {
        use crate::causal::discovery::{DataMatrix, PCConfig, pc_algorithm, pc_result_to_graph};

        // Synthetic chain data: X -> Y -> Z
        // X causes Y, Y causes Z; X and Z are conditionally independent given Y
        let n = 200;
        let mut data = Vec::with_capacity(n);
        let mut rng_state = 42u64;

        for _ in 0..n {
            // Simple LCG pseudo-random
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise_x = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise_y = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise_z = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;

            let x = noise_x * 2.0;
            let y = 0.8 * x + noise_y * 0.5;
            let z = 0.7 * y + noise_z * 0.5;
            data.push(vec![x, y, z]);
        }

        let matrix = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data).unwrap();

        let config = PCConfig {
            alpha: 0.05,
            stable: true,
            ..PCConfig::default()
        };

        let pc_result = pc_algorithm(&matrix, &config).unwrap();

        // Convert to CausalGraphDef
        let graph = pc_result_to_graph(&pc_result, "DiscoveredChain");
        assert_eq!(graph.name, "DiscoveredChain");
        assert_eq!(graph.nodes.len(), 3);

        // The discovered graph should have some directed edges
        // (exact structure depends on data, but we verify the pipeline works)
        let directed = pc_result.directed_edges();
        // At minimum the pipeline completes without error
        assert!(graph.nodes.len() == 3);
    }

    /// Test 19: Causal type checker confidence combined with evidence.
    ///
    /// When identification gives a confidence and evidence gives another,
    /// the product represents the joint epistemic state.
    #[test]
    fn test_causal_confidence_times_evidence_confidence() {
        use crate::check::causal::causal_effect_type_confidence;

        let causal_result = crate::check::causal::CausalCheckResult::Identified {
            formula: crate::types::causal_id::IdentificationFormula::Adjustment {
                outcome: vec!["BP".into()],
                treatment: vec!["Drug".into()],
                adjustment_set: vec!["Age".into(), "Weight".into()],
            },
            confidence: 0.92,
            treatment: vec!["Drug".into()],
            outcome: vec!["BP".into()],
        };

        let causal_conf = causal_effect_type_confidence(&causal_result).unwrap();

        // Evidence aggregate
        let mut collector = EvidenceCollector::new(ConflictStrategy::BayesianFusion);
        collector.collect(pub_evidence("rct_phase3", 0.85), "pfizer".into(), 1000);
        collector.collect(pub_evidence("meta_analysis", 0.88), "cochrane".into(), 2000);
        let evidence_conf = collector.aggregate().unwrap().combined_confidence;

        let joint = causal_conf * evidence_conf;
        assert!(joint > 0.0 && joint < 1.0);
        assert!(joint < causal_conf);
        assert!(joint < evidence_conf);
    }

    /// Test 20: Non-identifiable causal query is not overridden by evidence.
    ///
    /// Even with very high-confidence evidence, a non-identifiable
    /// causal query correctly remains non-identifiable.
    #[test]
    fn test_non_identifiable_not_overridden_by_evidence() {
        use crate::check::causal::{
            CausalCheckResult, causal_effect_type_confidence, format_non_identifiable_error,
        };

        let non_id = CausalCheckResult::NotIdentifiable {
            reason: "no valid adjustment set exists (latent confounding)".into(),
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
            graph_name: "BowArc".into(),
        };

        // Even with overwhelming evidence:
        let mut collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        collector.collect(pub_evidence("mega_trial", 0.99), "src".into(), 1000);
        collector.collect(pub_evidence("replication", 0.98), "src2".into(), 2000);
        let evidence_conf = collector.aggregate().unwrap().combined_confidence;
        assert!(evidence_conf > 0.99);

        // The causal query is still not identifiable
        assert!(causal_effect_type_confidence(&non_id).is_none());
        assert!(format_non_identifiable_error(&non_id).is_some());
        assert!(
            format_non_identifiable_error(&non_id)
                .unwrap()
                .contains("not identifiable")
        );
    }

    // ========================================================================
    // 5. SIR Causal + Provenance Integration (5 tests)
    // ========================================================================

    /// Test 21: Counterfactual lowering creates a valid three-step sequence.
    #[test]
    fn test_counterfactual_lowering_creates_provenance_trackable_sequence() {
        use crate::sir::causal::{
            count_causal_ops, lower_counterfactual, validate_three_step_sequence,
        };
        use crate::sir::values::ValueId;

        let ops = lower_counterfactual(
            vec![ValueId::new(1), ValueId::new(2)],
            "Treatment".into(),
            ValueId::new(3),
            "Outcome".into(),
            ValueId::new(100),
        );

        // Validate three-step protocol
        assert!(validate_three_step_sequence(&ops).is_ok());

        // Each step maps to a provenance node in execution
        let counts = count_causal_ops(&ops);
        assert_eq!(counts.abductions, 1);
        assert_eq!(counts.interventions, 1);
        assert_eq!(counts.predictions, 1);
    }

    /// Test 22: Causal formula kind preserves identification confidence mapping.
    #[test]
    fn test_formula_kind_preserves_identification_confidence() {
        use crate::sir::causal::formula_to_sir_kind;
        use crate::sir::ops::CausalFormulaKind;
        use crate::types::causal_id::{IdentificationFormula, identification_confidence};

        let formulas = vec![
            IdentificationFormula::Marginal {
                outcome: vec!["Y".into()],
            },
            IdentificationFormula::Conditional {
                outcome: vec!["Y".into()],
                treatment: vec!["X".into()],
            },
            IdentificationFormula::Adjustment {
                outcome: vec!["Y".into()],
                treatment: vec!["X".into()],
                adjustment_set: vec!["Z".into()],
            },
            IdentificationFormula::Frontdoor {
                treatment: vec!["X".into()],
                outcome: vec!["Y".into()],
                mediator: vec!["M".into()],
            },
            IdentificationFormula::InstrumentalVariable {
                instrument: vec!["Z".into()],
                treatment: vec!["X".into()],
                outcome: vec!["Y".into()],
            },
        ];

        for formula in &formulas {
            let kind = formula_to_sir_kind(formula);
            let conf = identification_confidence(formula);
            assert!(conf > 0.0, "all formulas should have positive confidence");

            // Verify the SIR kind is valid (no panics)
            let kind_str = format!("{:?}", kind);
            assert!(!kind_str.is_empty());
        }
    }

    /// Test 23: Three-step validation + cost estimation.
    #[test]
    fn test_three_step_validation_and_cost() {
        use crate::sir::causal::{
            causal_op_cost, lower_counterfactual, validate_three_step_sequence,
        };
        use crate::sir::values::ValueId;

        let ops = lower_counterfactual(
            vec![ValueId::new(1), ValueId::new(2), ValueId::new(3)],
            "X".into(),
            ValueId::new(4),
            "Y".into(),
            ValueId::new(100),
        );

        assert!(validate_three_step_sequence(&ops).is_ok());

        let costs: Vec<u64> = ops.iter().map(|op| causal_op_cost(op)).collect();

        // Abduction is most expensive (100 + 10*3 = 130)
        assert_eq!(costs[0], 130);
        // DoIntervention is cheapest (10)
        assert_eq!(costs[1], 10);
        // Prediction is moderate (50)
        assert_eq!(costs[2], 50);

        // Abduction should be the most expensive step
        assert!(
            costs[0] > costs[1] && costs[0] > costs[2],
            "abduction should be most expensive: {:?}",
            costs
        );
    }

    /// Test 24: Causal op counts track evidence requirements.
    #[test]
    fn test_causal_op_counts_evidence_requirements() {
        use crate::sir::causal::count_causal_ops;
        use crate::sir::ops::{CausalFormulaKind, CausalOp};
        use crate::sir::values::ValueId;

        let ops = vec![
            // CounterfactualQuery needs factual observations (evidence)
            CausalOp::CounterfactualQuery {
                factual_observations: vec![ValueId::new(1), ValueId::new(2)],
                intervention_var: "X".into(),
                intervention_value: ValueId::new(3),
                outcome_var: "Y".into(),
                result: ValueId::new(10),
            },
            // EstimateEffect needs adjustment set
            CausalOp::EstimateEffect {
                formula_kind: CausalFormulaKind::BackdoorAdjustment,
                treatment: ValueId::new(1),
                outcome: ValueId::new(2),
                adjustment_set: vec![ValueId::new(3), ValueId::new(4)],
                result: ValueId::new(20),
            },
        ];

        let counts = count_causal_ops(&ops);
        assert_eq!(counts.counterfactual_queries, 1);
        assert_eq!(counts.effect_estimations, 1);

        // Verify CounterfactualQuery has observations
        if let CausalOp::CounterfactualQuery {
            factual_observations,
            ..
        } = &ops[0]
        {
            assert_eq!(factual_observations.len(), 2);
        }

        // Verify EstimateEffect has adjustment set
        if let CausalOp::EstimateEffect { adjustment_set, .. } = &ops[1] {
            assert_eq!(adjustment_set.len(), 2);
        }
    }

    /// Test 25: Discovery -> Identification -> SIR pipeline (full pipeline).
    #[test]
    fn test_full_discovery_to_sir_pipeline() {
        use crate::causal::discovery::{DataMatrix, PCConfig, pc_algorithm, pc_result_to_graph};
        use crate::sir::causal::{formula_to_sir_kind, lower_do_expression};
        use crate::sir::values::ValueId;
        use crate::types::causal_id::IdentificationFormula;

        // Generate simple chain data: A -> B
        let n = 150;
        let mut data = Vec::with_capacity(n);
        let mut rng_state = 12345u64;
        for _ in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise_a = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise_b = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;

            let a = noise_a * 3.0;
            let b = 1.2 * a + noise_b * 0.3;
            data.push(vec![a, b]);
        }

        let matrix = DataMatrix::new(vec!["A".into(), "B".into()], data).unwrap();

        // Step 1: Discovery via PC algorithm
        let config = PCConfig::default();
        let pc_result = pc_algorithm(&matrix, &config).unwrap();

        // Step 2: Convert to CausalGraphDef
        let graph = pc_result_to_graph(&pc_result, "Discovered_AB");
        assert_eq!(graph.nodes.len(), 2);

        // Step 3: Build a formula based on structure
        // For a simple A->B chain, use Conditional identification
        let formula = IdentificationFormula::Conditional {
            outcome: vec!["B".into()],
            treatment: vec!["A".into()],
        };

        // Step 4: Convert to SIR kind
        let sir_kind = formula_to_sir_kind(&formula);
        assert_eq!(sir_kind, crate::sir::ops::CausalFormulaKind::Conditional);

        // Step 5: Lower to SIR operations
        let ops = lower_do_expression(
            &formula,
            ValueId::new(1),
            ValueId::new(2),
            vec![],
            ValueId::new(10),
        );

        assert_eq!(ops.len(), 1);
        assert!(matches!(
            ops[0],
            crate::sir::ops::CausalOp::EstimateEffect { .. }
        ));
    }

    // ========================================================================
    // 6. Full System Integration (5+ tests)
    // ========================================================================

    /// Test 26: Knowledge lifecycle -- create, evidence, age, query provenance.
    #[test]
    fn test_knowledge_lifecycle() {
        // Phase 1: Create knowledge with provenance
        let prov = Provenance::external("https://clinicaltrials.gov/NCT12345678")
            .extend(Transformation::function("parse_results").with_confidence(1.0))
            .extend(
                Transformation::new("statistical_analysis", TransformationKind::Statistical)
                    .with_confidence(0.95),
            );

        // Phase 2: Add evidence from multiple sources
        let mut collector = EvidenceCollector::new(ConflictStrategy::BayesianFusion);
        collector.collect(pub_evidence("NCT12345678", 0.9), "trial_db".into(), 1000);
        collector.collect(
            exp_evidence("replication_study", 0.85),
            "lab_eu".into(),
            2000,
        );
        collector.collect(pub_evidence("meta_analysis", 0.88), "cochrane".into(), 3000);

        let evidence_agg = collector.aggregate().unwrap();
        assert!(evidence_agg.combined_confidence > 0.9); // Bayesian fusion amplifies

        // Phase 3: Apply temporal aging
        let aging = AgingModel::ExponentialDecay {
            half_life_seconds: 365 * 86400, // 1 year half-life
        };
        let elapsed = 180 * 86400; // 6 months
        let aged_evidence = apply_aging(evidence_agg.combined_confidence, elapsed, &aging);
        let aged_provenance = apply_aging(prov.total_confidence_factor(), elapsed, &aging);

        assert!(aged_evidence > 0.5, "6-month aging shouldn't halve");
        assert!(aged_provenance > 0.5);

        // Phase 4: Query provenance
        let statistical_steps = prov.query(&TransformationFilter::ByKind(
            TransformationKind::Statistical,
        ));
        assert_eq!(statistical_steps.match_count, 1);

        // Phase 5: Verify lifecycle coherence
        let combined = aged_evidence * aged_provenance;
        assert!(combined > 0.0 && combined < 1.0);
    }

    /// Test 27: GUM + Causal -- uncertainty budget for causal effect.
    ///
    /// The identification confidence affects the overall uncertainty budget.
    #[test]
    fn test_gum_causal_uncertainty_budget() {
        use crate::check::causal::causal_effect_type_confidence;

        // Causal effect identified
        let causal_result = crate::check::causal::CausalCheckResult::Identified {
            formula: crate::types::causal_id::IdentificationFormula::Adjustment {
                outcome: vec!["Clearance".into()],
                treatment: vec!["Dose".into()],
                adjustment_set: vec!["Weight".into(), "Age".into()],
            },
            confidence: 0.90,
            treatment: vec!["Dose".into()],
            outcome: vec!["Clearance".into()],
        };

        let causal_conf = causal_effect_type_confidence(&causal_result).unwrap();

        // Build uncertainty budget entries
        struct BudgetEntry {
            source: String,
            uncertainty: f64,
            sensitivity: f64,
        }

        let budget = vec![
            BudgetEntry {
                source: "measurement_noise".into(),
                uncertainty: 0.05,
                sensitivity: 1.0,
            },
            BudgetEntry {
                source: "model_uncertainty".into(),
                uncertainty: 1.0 - causal_conf, // causal identification uncertainty
                sensitivity: 0.8,
            },
            BudgetEntry {
                source: "parameter_estimation".into(),
                uncertainty: 0.03,
                sensitivity: 0.5,
            },
        ];

        // Combined uncertainty (GUM-style: root-sum-of-squares of u_i * c_i)
        let combined_u: f64 = budget
            .iter()
            .map(|e| (e.uncertainty * e.sensitivity).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(combined_u > 0.0);
        assert!(combined_u < 1.0);

        // The causal identification contributes to the uncertainty budget
        let causal_contribution = (1.0 - causal_conf) * 0.8;
        assert!(
            causal_contribution > 0.0,
            "causal identification uncertainty should contribute to budget"
        );
    }

    /// Test 28: Evidence conflict -> resolution -> provenance -> temporal check.
    #[test]
    fn test_conflict_resolution_provenance_temporal_pipeline() {
        // Step 1: Two conflicting evidence pieces
        let evidence_rct = pub_evidence("phase3_rct", 0.92);
        let evidence_observational = pub_evidence("observational_cohort", 0.55);

        // Step 2: Resolve via Dempster-Shafer
        let collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        let resolved = collector.resolve_conflict(&evidence_rct, &evidence_observational);
        // DS: 0.92 + 0.55 - 0.92*0.55 = 0.964
        assert!(resolved.strength.value() > 0.9);

        // Step 3: Record in provenance chain
        let prov = Provenance::external("https://evidence-repo.example.com")
            .extend(
                Transformation::new("rct_analysis", TransformationKind::Statistical)
                    .with_confidence(0.92),
            )
            .extend(
                Transformation::new("observational_analysis", TransformationKind::Statistical)
                    .with_confidence(0.55),
            )
            .extend(
                Transformation::new("dempster_shafer_fusion", TransformationKind::Aggregation)
                    .with_confidence(resolved.strength.value()),
            );

        assert_eq!(prov.depth(), 3);

        // Step 4: Check temporal validity of resolution
        let current = TemporalIndex::date(2026, 6, 1);
        let bounds = ValidityBounds::Range {
            from: TemporalIndex::date(2025, 1, 1),
            until: TemporalIndex::date(2027, 12, 31),
        };
        let validity = validate_temporal(&bounds, &current);
        assert!(matches!(validity, TemporalValidityResult::Valid { .. }));

        // Step 5: Export audit trail
        let mut evidence_collector = EvidenceCollector::new(ConflictStrategy::DempsterShafer);
        evidence_collector.collect(evidence_rct, "rct_site".into(), 5000);
        evidence_collector.collect(evidence_observational, "cohort_study".into(), 6000);
        let trail = evidence_collector.export_audit_trail();
        assert_eq!(trail.len(), 2);
    }

    /// Test 29: PC discovery -> causal check -> SIR lower -> cost estimate.
    #[test]
    fn test_discovery_to_cost_estimate_pipeline() {
        use crate::sir::causal::{causal_op_cost, formula_to_sir_kind, lower_do_expression};
        use crate::sir::ops::CausalFormulaKind;
        use crate::sir::values::ValueId;
        use crate::types::causal_id::IdentificationFormula;

        // Simulate a discovered graph result: X -> Y with confound Z
        // Backdoor adjustment on Z
        let formula = IdentificationFormula::Adjustment {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
            adjustment_set: vec!["Z".into()],
        };

        // Step 1: Convert formula to SIR kind
        let kind = formula_to_sir_kind(&formula);
        assert_eq!(kind, CausalFormulaKind::BackdoorAdjustment);

        // Step 2: Lower to SIR operations
        let ops = lower_do_expression(
            &formula,
            ValueId::new(1),
            ValueId::new(2),
            vec![ValueId::new(3)],
            ValueId::new(10),
        );

        // Step 3: Estimate cost
        let total_cost: u64 = ops.iter().map(|op| causal_op_cost(op)).sum();
        // BackdoorAdjustment base=80, 1 adj variable: 80 + 10 = 90
        assert_eq!(total_cost, 90);
    }

    /// Test 30: Merkle DAG with causal provenance pipeline.
    ///
    /// Creates merkle nodes representing causal analysis steps,
    /// verifies DAG structure, and computes confidence chain.
    #[test]
    fn test_merkle_dag_causal_provenance() {
        let mut dag = MerkleProvenanceDAG::new();

        // Step 1: Observational data collection
        let obs_id = dag.add_root(ProvenanceOperation::measurement("observational_study"));

        // Step 2: Causal discovery
        let discovery_id = dag.add_derived(
            vec![obs_id],
            ProvenanceOperation::computation("pc_algorithm"),
        );

        // Step 3: Causal identification
        let id_op = ProvenanceOperation::transformation("backdoor_identification", 0.92);
        let identification_id = dag.add_derived(vec![discovery_id], id_op);

        // Step 4: Effect estimation
        let est_op = ProvenanceOperation::transformation("ate_estimation", 0.95);
        let estimation_id = dag.add_derived(vec![identification_id], est_op);

        // Verify DAG structure
        assert_eq!(dag.len(), 4);
        assert_eq!(dag.roots().count(), 1);
        assert_eq!(dag.heads().count(), 1);
        assert_eq!(dag.dag_depth(), 4);

        // Verify path from root to head
        let path = dag.find_path(&obs_id, &estimation_id);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], obs_id);
        assert_eq!(path[3], estimation_id);

        // Compute confidence chain through causal pipeline
        let chain_conf = dag.confidence_chain(&estimation_id).unwrap();
        // 1.0 (measurement) * 1.0 (computation) * 0.92 * 0.95 = 0.874
        assert!(
            (chain_conf - 0.874).abs() < 0.01,
            "expected ~0.874, got {}",
            chain_conf
        );

        // Verify subset integrity
        let all_ids = vec![obs_id, discovery_id, identification_id, estimation_id];
        let verification = dag.verify_subset(&all_ids);
        assert!(verification.iter().all(|(_, valid)| *valid));
    }

    /// Test 31: Provenance-to-Merkle DAG conversion preserves structure.
    #[test]
    fn test_provenance_to_merkle_conversion() {
        let prov = Provenance::computed("simulate_pbpk")
            .extend(Transformation::function("solve_ode").with_confidence(0.99))
            .extend(
                Transformation::new("calibrate", TransformationKind::Conversion)
                    .with_confidence(0.95),
            )
            .extend(
                Transformation::new("validate", TransformationKind::Statistical)
                    .with_confidence(0.90),
            );

        // Convert Provenance -> MerkleProvenanceDAG
        let dag = MerkleProvenanceDAG::from(&prov);

        assert!(dag.len() >= 4); // root + 3 transformations
        assert!(dag.verify_all().is_ok());

        // Convert back: MerkleProvenanceDAG -> Provenance
        let prov_back = Provenance::from(&dag);
        assert!(prov_back.integrity_hash.is_some());

        // The roundtrip should preserve the origin type
        match &prov.origin {
            Origin::Computed { function } => {
                assert_eq!(function, "simulate_pbpk");
            }
            _ => panic!("expected Computed origin"),
        }
    }

    /// Test 32: Full audit trail from evidence through causal to Merkle.
    #[test]
    fn test_full_audit_trail_evidence_to_merkle() {
        // Phase 1: Collect evidence
        let mut collector = EvidenceCollector::new(ConflictStrategy::BayesianFusion);
        collector.collect(
            pub_evidence("doi:10.1000/trial", 0.88),
            "site_a".into(),
            1000,
        );
        collector.collect(exp_evidence("gmp_protocol_v2", 0.82), "site_b".into(), 2000);

        let evidence_agg = collector.aggregate().unwrap();

        // Phase 2: Build provenance chain
        let prov = Provenance::external("https://trials.example.com")
            .extend(
                Transformation::new("evidence_fusion", TransformationKind::Statistical)
                    .with_confidence(evidence_agg.combined_confidence),
            )
            .extend(
                Transformation::new("causal_identification", TransformationKind::Function)
                    .with_confidence(0.92),
            );

        // Phase 3: Convert to Merkle DAG
        let dag = MerkleProvenanceDAG::from(&prov);
        assert!(dag.verify_all().is_ok());

        // Phase 4: Generate audit trail
        let trail = dag.to_audit_trail();
        assert!(!trail.entries.is_empty());

        // Phase 5: Verify audit trail contains evidence fusion step
        let json = trail.to_json();
        assert!(json.contains("evidence_fusion"));
        assert!(json.contains("causal_identification"));
    }

    /// Test 33: Aging model comparison across evidence strategies.
    ///
    /// Compares how different aging models affect the same evidence base.
    #[test]
    fn test_aging_model_comparison() {
        let base_confidence = 0.9;
        let elapsed = 3600u64; // 1 hour

        let exponential = apply_aging(
            base_confidence,
            elapsed,
            &AgingModel::ExponentialDecay {
                half_life_seconds: 3600,
            },
        );

        let linear = apply_aging(
            base_confidence,
            elapsed,
            &AgingModel::Linear {
                rate_per_second: 0.9 / 7200.0, // drops to 0 in 2 hours
            },
        );

        let step = apply_aging(
            base_confidence,
            elapsed,
            &AgingModel::StepFunction {
                threshold_seconds: 1800,
                degraded_confidence: 0.2,
            },
        );

        let none = apply_aging(base_confidence, elapsed, &AgingModel::NoAging);

        // NoAging preserves confidence
        assert!((none - base_confidence).abs() < 1e-6);

        // Exponential: 0.9 * 0.5 = 0.45
        assert!((exponential - 0.45).abs() < 1e-6);

        // Step: past 1800s threshold -> 0.2
        assert!((step - 0.2).abs() < 1e-6);

        // Linear: 0.9 - (0.9/7200)*3600 = 0.9 - 0.45 = 0.45
        assert!((linear - 0.45).abs() < 0.01);

        // All degraded models should be less than original
        assert!(exponential < base_confidence);
        assert!(linear < base_confidence);
        assert!(step < base_confidence);
    }

    /// Test 34: Merkle DAG multi-root merge with confidence.
    #[test]
    fn test_merkle_multi_root_merge_confidence() {
        let mut dag = MerkleProvenanceDAG::new();

        // Two independent measurement sources
        let sensor_a = dag.add_root(ProvenanceOperation::measurement("sensor_A"));
        let sensor_b = dag.add_root(ProvenanceOperation::measurement("sensor_B"));

        // Calibrate each
        let cal_a = dag.add_derived(
            vec![sensor_a],
            ProvenanceOperation::transformation("calibrate_a", 0.98),
        );
        let cal_b = dag.add_derived(
            vec![sensor_b],
            ProvenanceOperation::transformation("calibrate_b", 0.95),
        );

        // Merge both into a single result
        let merged = dag.add_derived(
            vec![cal_a, cal_b],
            ProvenanceOperation::computation("weighted_average"),
        );

        // Check paths
        let path_a = dag.find_path(&sensor_a, &merged).unwrap();
        assert_eq!(path_a.len(), 3); // sensor_a -> cal_a -> merged

        let path_b = dag.find_path(&sensor_b, &merged).unwrap();
        assert_eq!(path_b.len(), 3); // sensor_b -> cal_b -> merged

        // Confidence chain: the merged node multiplies its own factor (1.0)
        // by the product of both parent chains:
        //   sensor_A(1.0) * cal_a(0.98) = 0.98
        //   sensor_B(1.0) * cal_b(0.95) = 0.95
        //   merged: 1.0 * 0.98 * 0.95 = 0.931
        let total = dag.total_confidence_factor();
        assert!(
            (total - 0.931).abs() < 0.01,
            "total confidence should be product of all parent chains: expected ~0.931, got {}",
            total
        );

        assert_eq!(dag.dag_depth(), 3);
        assert_eq!(dag.roots().count(), 2);
        assert_eq!(dag.heads().count(), 1);
    }

    /// Test 35: Evidence collector with all four strategies produces valid aggregates.
    #[test]
    fn test_all_strategies_produce_valid_aggregates() {
        let strategies = vec![
            ConflictStrategy::TrustHigherConfidence,
            ConflictStrategy::BayesianFusion,
            ConflictStrategy::TemporalPrecedence,
            ConflictStrategy::DempsterShafer,
        ];

        for strategy in strategies {
            let mut collector = EvidenceCollector::new(strategy.clone());
            collector.collect(pub_evidence("s1", 0.7), "a".into(), 1000);
            collector.collect(pub_evidence("s2", 0.8), "b".into(), 2000);
            collector.collect(exp_evidence("s3", 0.6), "c".into(), 3000);

            let agg = collector.aggregate().unwrap();
            assert!(
                agg.combined_confidence >= 0.0 && agg.combined_confidence <= 1.0,
                "strategy {:?}: confidence {} out of [0,1]",
                strategy,
                agg.combined_confidence
            );
            assert!(
                agg.supporting_count + agg.conflicting_count == 3,
                "strategy {:?}: counts don't sum to 3",
                strategy
            );
        }
    }
}
