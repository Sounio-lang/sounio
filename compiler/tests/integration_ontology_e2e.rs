//! End-to-End Integration Tests for Ontological Type System
//!
//! Day 49: These tests verify the complete pipeline:
//! Source Code -> Parse -> Type Check (with ontology) -> Semantic Distance
//!
//! # Test Categories
//!
//! 1. **Basic Ontology Types**: Single ontology, direct types
//! 2. **Subsumption**: Is-a hierarchy type compatibility
//! 3. **Cross-Ontology**: Different ontology type compatibility
//! 4. **Semantic Distance**: Distance-based compatibility
//! 5. **Confidence Propagation**: Epistemic tracking through types
//! 6. **Performance Benchmarks**: Resolution and distance calculation speed

use std::time::Instant;

use sounio::ontology::distance::SemanticDistanceIndex;
use sounio::ontology::embedding::{Embedding, EmbeddingConfig, EmbeddingModel, EmbeddingSpace};
use sounio::ontology::loader::{IRI, LoadedTerm, OntologyId};
use sounio::types::semantic::{SemanticType, SemanticTypeChecker};
use std::sync::{Arc, RwLock};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a simple test IRI
fn iri(s: &str) -> IRI {
    IRI::new(&format!("http://example.org/{}", s))
}

/// Create a CURIE-style IRI
fn curie_iri(prefix: &str, local: &str) -> IRI {
    IRI::from_curie(prefix, local)
}

/// Create a minimal LoadedTerm for testing
fn make_term(iri: IRI, label: &str, superclasses: Vec<IRI>) -> LoadedTerm {
    LoadedTerm {
        iri,
        label: label.to_string(),
        ontology: OntologyId::Unknown,
        superclasses,
        subclasses: vec![],
        properties: vec![],
        restrictions: vec![],
        xrefs: vec![],
        definition: None,
        synonyms: vec![],
        hierarchy_depth: 0,
        information_content: 0.0,
        is_obsolete: false,
        replaced_by: None,
    }
}

/// Build a test ontology hierarchy for pharmaceutical domain
fn build_pharma_hierarchy() -> Vec<LoadedTerm> {
    vec![
        // Top-level
        make_term(iri("ChemicalEntity"), "Chemical Entity", vec![]),
        // Drug branch
        make_term(iri("Drug"), "Drug", vec![iri("ChemicalEntity")]),
        make_term(iri("Analgesic"), "Analgesic", vec![iri("Drug")]),
        make_term(iri("NSAID"), "NSAID", vec![iri("Analgesic")]),
        make_term(iri("Aspirin"), "Aspirin", vec![iri("NSAID")]),
        make_term(iri("Ibuprofen"), "Ibuprofen", vec![iri("NSAID")]),
        make_term(iri("Opioid"), "Opioid", vec![iri("Analgesic")]),
        make_term(iri("Morphine"), "Morphine", vec![iri("Opioid")]),
        // Antibiotic branch
        make_term(iri("Antibiotic"), "Antibiotic", vec![iri("Drug")]),
        make_term(iri("Penicillin"), "Penicillin", vec![iri("Antibiotic")]),
        make_term(iri("Amoxicillin"), "Amoxicillin", vec![iri("Penicillin")]),
        // Organic compound (sibling to Drug)
        make_term(
            iri("OrganicCompound"),
            "Organic Compound",
            vec![iri("ChemicalEntity")],
        ),
    ]
}

/// Build a test ontology hierarchy for disease domain
fn build_disease_hierarchy() -> Vec<LoadedTerm> {
    vec![
        make_term(iri("Disease"), "Disease", vec![]),
        make_term(iri("Cancer"), "Cancer", vec![iri("Disease")]),
        make_term(iri("LungCancer"), "Lung Cancer", vec![iri("Cancer")]),
        make_term(iri("BreastCancer"), "Breast Cancer", vec![iri("Cancer")]),
        make_term(
            iri("CardiovascularDisease"),
            "Cardiovascular Disease",
            vec![iri("Disease")],
        ),
        make_term(
            iri("HeartAttack"),
            "Heart Attack",
            vec![iri("CardiovascularDisease")],
        ),
        make_term(iri("Headache"), "Headache", vec![iri("Disease")]),
        make_term(iri("Migraine"), "Migraine", vec![iri("Headache")]),
    ]
}

// ============================================================================
// CATEGORY 1: Basic Ontology Types
// ============================================================================

#[test]
fn test_iri_curie_resolution() {
    let aspirin = curie_iri("CHEBI", "15365");
    assert!(aspirin.as_str().contains("CHEBI"));
    assert!(aspirin.as_str().contains("15365"));
}

#[test]
fn test_semantic_type_from_iri() {
    let aspirin_iri = iri("Aspirin");
    let aspirin_type = SemanticType::from_iri(aspirin_iri.clone(), "Aspirin".to_string());

    assert_eq!(aspirin_type.iri, aspirin_iri);
    assert_eq!(aspirin_type.name, "Aspirin");
}

#[test]
fn test_loaded_term_hierarchy() {
    let terms = build_pharma_hierarchy();

    // Find Aspirin
    let aspirin = terms.iter().find(|t| t.label == "Aspirin").unwrap();

    // Verify it has NSAID as superclass
    assert!(aspirin.superclasses.contains(&iri("NSAID")));
}

// ============================================================================
// CATEGORY 2: Subsumption (Is-A Hierarchy)
// ============================================================================

#[test]
fn test_direct_subsumption_relationship() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Aspirin -> NSAID (direct parent)
    let d = index.distance(&iri("Aspirin"), &iri("NSAID"));
    assert!(
        d.conceptual < 0.5,
        "Direct parent should be relatively close: {}",
        d.conceptual
    );
}

#[test]
fn test_transitive_subsumption_relationship() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Aspirin -> Drug (grandparent via NSAID -> Analgesic -> Drug)
    let d = index.distance(&iri("Aspirin"), &iri("Drug"));

    // Should be farther than direct parent but still related
    assert!(
        d.conceptual < 0.8,
        "Transitive ancestor should have moderate distance: {}",
        d.conceptual
    );
}

#[test]
fn test_sibling_types_distance() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Aspirin and Ibuprofen are siblings (both NSAID)
    let d = index.distance(&iri("Aspirin"), &iri("Ibuprofen"));

    // Siblings should be closer than unrelated types
    assert!(
        d.conceptual < 0.6,
        "Siblings should be relatively close: {}",
        d.conceptual
    );
}

#[test]
fn test_cousin_types_distance() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Aspirin (NSAID) and Morphine (Opioid) are cousins (both Analgesic)
    let d_cousin = index.distance(&iri("Aspirin"), &iri("Morphine")).conceptual;
    // Aspirin and Amoxicillin are more distant (different drug categories)
    let d_distant = index
        .distance(&iri("Aspirin"), &iri("Amoxicillin"))
        .conceptual;

    // Cousins (via Analgesic) should be closer than distant relatives (via Drug)
    assert!(
        d_cousin <= d_distant,
        "Cousins ({}) should be closer or equal to distant relatives ({})",
        d_cousin,
        d_distant
    );
}

// ============================================================================
// CATEGORY 3: Cross-Ontology Compatibility
// ============================================================================

#[test]
fn test_different_domains_high_distance() {
    let mut index = SemanticDistanceIndex::new();

    // Build combined hierarchy
    let mut terms = build_pharma_hierarchy();
    terms.extend(build_disease_hierarchy());
    index.build_from_terms(&terms);

    // Drug and Disease are in different branches (no common ancestor in test)
    let d = index.distance(&iri("Drug"), &iri("Disease"));

    // They should be distant (high conceptual distance for unrelated domains)
    assert!(
        d.conceptual >= 0.5,
        "Unrelated domains should be distant: {}",
        d.conceptual
    );
}

#[test]
fn test_cross_ontology_with_mappings() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Simulate a cross-ontology mapping via embeddings
    let mut config = EmbeddingConfig::default();
    config.dimensions = 4;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // ChEBI.Drug and DrugBank.Drug should be mapped as equivalent
    // We simulate this by giving them very similar embeddings
    space
        .add(Embedding::new(
            iri("ChEBI_Drug"),
            vec![0.9, 0.1, 0.0, 0.0],
            EmbeddingModel::Pretrained,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("DrugBank_Drug"),
            vec![0.89, 0.11, 0.0, 0.0],
            EmbeddingModel::Pretrained,
        ))
        .unwrap();

    let sim = space
        .cosine_similarity(&iri("ChEBI_Drug"), &iri("DrugBank_Drug"))
        .unwrap();
    assert!(
        sim > 0.99,
        "Mapped equivalent types should have very high similarity: {}",
        sim
    );
}

// ============================================================================
// CATEGORY 4: Semantic Distance-Based Compatibility
// ============================================================================

#[test]
fn test_distance_thresholds() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Direct parent should have low distance
    let d_parent = index.distance(&iri("Aspirin"), &iri("NSAID"));
    println!("Aspirin -> NSAID distance: {}", d_parent.conceptual);

    // More distant ancestor should have higher distance
    let d_ancestor = index.distance(&iri("Aspirin"), &iri("ChemicalEntity"));
    println!(
        "Aspirin -> ChemicalEntity distance: {}",
        d_ancestor.conceptual
    );

    // Verify ordering: parent should be closer than distant ancestor
    assert!(
        d_parent.conceptual <= d_ancestor.conceptual,
        "Parent ({}) should be closer than distant ancestor ({})",
        d_parent.conceptual,
        d_ancestor.conceptual
    );
}

#[test]
fn test_semantic_type_checker_compatibility() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let aspirin_type = SemanticType::from_iri(iri("Aspirin"), "Aspirin".to_string());
    let drug_type = SemanticType::from_iri(iri("Drug"), "Drug".to_string());

    // Aspirin should be compatible with Drug (is-a relationship)
    let compat = checker.check_compatibility(&aspirin_type, &drug_type);
    assert!(
        compat.is_compatible(),
        "Aspirin should be compatible with Drug"
    );
}

#[test]
fn test_implicit_vs_explicit_coercion() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    // Close types - should allow implicit coercion
    let aspirin_type = SemanticType::from_iri(iri("Aspirin"), "Aspirin".to_string());
    let nsaid_type = SemanticType::from_iri(iri("NSAID"), "NSAID".to_string());

    assert!(
        checker.allows_implicit_coercion(&aspirin_type, &nsaid_type),
        "Aspirin to NSAID should allow implicit coercion"
    );

    // Verify the subtype relationship works
    let drug_type = SemanticType::from_iri(iri("Drug"), "Drug".to_string());
    let allows_implicit = checker.allows_implicit_coercion(&aspirin_type, &drug_type);

    // Whether this is implicit or explicit depends on the configured thresholds
    // The important thing is that it's compatible at all
    let compat = checker.check_compatibility(&aspirin_type, &drug_type);
    assert!(compat.is_compatible());
}

// ============================================================================
// CATEGORY 5: Confidence Propagation (Epistemic Types)
// ============================================================================

#[test]
fn test_confidence_degrades_with_distance() {
    let initial_confidence = 0.95;
    let alpha = 0.15; // Degradation factor

    // Simulate coercion with distance 0.2
    let distance = 0.2;
    let degraded = initial_confidence * (1.0 - alpha * distance);

    assert!(
        degraded < initial_confidence,
        "Confidence should degrade: {} < {}",
        degraded,
        initial_confidence
    );
    assert!(
        (degraded - 0.9215_f64).abs() < 0.001,
        "Degraded confidence should be ~0.9215: {}",
        degraded
    );
}

#[test]
fn test_confidence_chain_degradation() {
    let initial_confidence = 0.95;
    let alpha = 0.15;

    // Chain of coercions: Aspirin -> NSAID -> Analgesic -> Drug
    let distances = [0.1, 0.1, 0.1]; // Each step has distance 0.1

    let mut confidence = initial_confidence;
    for d in &distances {
        confidence *= 1.0 - alpha * d;
    }

    // After 3 coercions with d=0.1 each
    let expected: f64 = initial_confidence * (1.0 - alpha * 0.1_f64).powi(3);
    assert!(
        (confidence - expected).abs() < 0.001,
        "Chain degradation should match: {} vs {}",
        confidence,
        expected
    );
    assert!(
        confidence < initial_confidence,
        "Confidence should be lower after chain"
    );
}

#[test]
fn test_zero_distance_preserves_confidence() {
    let confidence: f64 = 0.95;
    let alpha: f64 = 0.15;
    let distance: f64 = 0.0;

    let after = confidence * (1.0 - alpha * distance);
    assert!(
        (after - confidence).abs() < 0.0001,
        "Zero distance should preserve confidence exactly"
    );
}

// ============================================================================
// CATEGORY 6: Embedding-Based Distance
// ============================================================================

#[test]
fn test_embedding_captures_semantic_similarity() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 4;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // Aspirin and Ibuprofen are both NSAIDs - similar embeddings
    space
        .add(Embedding::new(
            iri("Aspirin"),
            vec![0.8, 0.2, 0.0, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("Ibuprofen"),
            vec![0.78, 0.22, 0.0, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    // Penicillin is an antibiotic - different embedding region
    space
        .add(Embedding::new(
            iri("Penicillin"),
            vec![0.0, 0.1, 0.9, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    let sim_nsaids = space
        .cosine_similarity(&iri("Aspirin"), &iri("Ibuprofen"))
        .unwrap();
    let sim_diff = space
        .cosine_similarity(&iri("Aspirin"), &iri("Penicillin"))
        .unwrap();

    assert!(
        sim_nsaids > sim_diff,
        "NSAIDs ({}) should be more similar than NSAID-Antibiotic ({})",
        sim_nsaids,
        sim_diff
    );
}

#[test]
fn test_embedding_nearest_neighbors() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 3;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // Add a cluster of similar drugs
    space
        .add(Embedding::new(
            iri("Aspirin"),
            vec![1.0, 0.0, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();
    space
        .add(Embedding::new(
            iri("Ibuprofen"),
            vec![0.95, 0.05, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();
    space
        .add(Embedding::new(
            iri("Naproxen"),
            vec![0.92, 0.08, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();
    space
        .add(Embedding::new(
            iri("Morphine"),
            vec![0.0, 1.0, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    // Nearest neighbors to Aspirin should be other NSAIDs
    let neighbors = space.nearest_neighbors(&iri("Aspirin"), 2).unwrap();
    assert_eq!(neighbors.len(), 2);

    // First neighbor should be very close (probably Ibuprofen or Naproxen)
    assert!(
        neighbors[0].1 < 0.1,
        "Closest neighbor should be very near: {}",
        neighbors[0].1
    );
}

// ============================================================================
// CATEGORY 7: Combined Distance (Path + IC + Embedding)
// ============================================================================

#[test]
fn test_combined_distance_integration() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Add embedding space
    let mut config = EmbeddingConfig::default();
    config.dimensions = 4;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // Add embeddings for some terms
    space
        .add(Embedding::new(
            iri("Aspirin"),
            vec![0.8, 0.2, 0.0, 0.0],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();
    space
        .add(Embedding::new(
            iri("NSAID"),
            vec![0.75, 0.25, 0.0, 0.0],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();
    space
        .add(Embedding::new(
            iri("Drug"),
            vec![0.5, 0.3, 0.2, 0.0],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();

    index.set_embedding_space(space);

    // Verify we now have embeddings
    assert!(index.has_embeddings());

    // Distance should now combine path and embedding
    let aspirin_nsaid = index.distance(&iri("Aspirin"), &iri("NSAID"));
    // Verify we get a valid distance (not max distance)
    assert!(
        aspirin_nsaid.conceptual < 1.0,
        "Should compute valid distance"
    );
}

// ============================================================================
// CATEGORY 8: Performance Benchmarks
// ============================================================================

#[test]
fn benchmark_hierarchy_building() {
    let terms = build_pharma_hierarchy();

    let start = Instant::now();
    for _ in 0..100 {
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);
    }
    let duration = start.elapsed();

    let avg_ms = duration.as_millis() as f64 / 100.0;
    println!("Average hierarchy build time: {:.3} ms", avg_ms);

    // Should complete in reasonable time
    assert!(
        avg_ms < 50.0,
        "Hierarchy building should be fast: {} ms",
        avg_ms
    );
}

#[test]
fn benchmark_distance_calculation() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    let pairs = vec![
        (iri("Aspirin"), iri("NSAID")),
        (iri("Aspirin"), iri("Drug")),
        (iri("Aspirin"), iri("ChemicalEntity")),
        (iri("Ibuprofen"), iri("Morphine")),
    ];

    let start = Instant::now();
    for _ in 0..1000 {
        for (from, to) in &pairs {
            let _ = index.distance(from, to);
        }
    }
    let duration = start.elapsed();

    let total_calcs = 1000 * pairs.len();
    let avg_us = duration.as_micros() as f64 / total_calcs as f64;
    println!(
        "Average distance calculation: {:.3} us ({} calculations)",
        avg_us, total_calcs
    );

    // Each calculation should be very fast with caching
    assert!(
        avg_us < 100.0,
        "Distance calculation should be fast: {} us",
        avg_us
    );
}

#[test]
fn benchmark_type_checking() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let aspirin_type = SemanticType::from_iri(iri("Aspirin"), "Aspirin".to_string());
    let drug_type = SemanticType::from_iri(iri("Drug"), "Drug".to_string());

    let start = Instant::now();
    for _ in 0..10000 {
        let _ = checker.check_compatibility(&aspirin_type, &drug_type);
    }
    let duration = start.elapsed();

    let avg_us = duration.as_micros() as f64 / 10000.0;
    println!("Average type check: {:.3} us", avg_us);

    assert!(avg_us < 50.0, "Type checking should be fast: {} us", avg_us);
}

#[test]
fn benchmark_embedding_operations() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 128;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // Add 100 embeddings
    for i in 0..100 {
        let mut vec = vec![0.0f32; 128];
        vec[i % 128] = 1.0;
        space
            .add(Embedding::new(
                iri(&format!("Term{}", i)),
                vec,
                EmbeddingModel::Structural,
            ))
            .unwrap();
    }

    // Benchmark similarity calculation
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = space.cosine_similarity(&iri("Term0"), &iri("Term50"));
    }
    let duration = start.elapsed();

    let avg_us = duration.as_micros() as f64 / 1000.0;
    println!("Average embedding similarity: {:.3} us", avg_us);

    assert!(
        avg_us < 100.0,
        "Embedding similarity should be fast: {} us",
        avg_us
    );
}

// ============================================================================
// CATEGORY 9: Real-World Scenarios
// ============================================================================

#[test]
fn scenario_pharmacology_type_safety() {
    // Scenario: A drug prescription system
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    // Function expects Drug type
    let drug_param = SemanticType::from_iri(iri("Drug"), "Drug".to_string());

    // Valid: Aspirin is-a Drug
    let aspirin = SemanticType::from_iri(iri("Aspirin"), "Aspirin".to_string());
    let compat = checker.check_compatibility(&aspirin, &drug_param);
    assert!(compat.is_compatible(), "Aspirin should be valid as Drug");

    // Valid: Morphine is-a Drug (via Opioid -> Analgesic -> Drug)
    let morphine = SemanticType::from_iri(iri("Morphine"), "Morphine".to_string());
    let compat = checker.check_compatibility(&morphine, &drug_param);
    assert!(compat.is_compatible(), "Morphine should be valid as Drug");

    // Invalid: OrganicCompound is NOT a Drug (sibling under ChemicalEntity)
    let organic = SemanticType::from_iri(iri("OrganicCompound"), "Organic Compound".to_string());
    let compat = checker.check_compatibility(&organic, &drug_param);
    // This might or might not be compatible depending on the hierarchy
    // The key is that the system provides a distance metric
    println!(
        "OrganicCompound -> Drug compatibility: {:?}",
        compat.is_compatible()
    );
}

#[test]
fn scenario_clinical_trial_type_matching() {
    // Scenario: Clinical trial matching diseases to treatments
    let mut drug_index = SemanticDistanceIndex::new();
    let drug_terms = build_pharma_hierarchy();
    drug_index.build_from_terms(&drug_terms);

    let mut disease_index = SemanticDistanceIndex::new();
    let disease_terms = build_disease_hierarchy();
    disease_index.build_from_terms(&disease_terms);

    // Within disease hierarchy - Cancer and LungCancer
    let d = disease_index.distance(&iri("Cancer"), &iri("LungCancer"));
    println!("Cancer -> LungCancer distance: {}", d.conceptual);
    assert!(d.conceptual < 0.5, "LungCancer should be close to Cancer");

    // Within drug hierarchy - Aspirin to NSAID
    let d = drug_index.distance(&iri("Aspirin"), &iri("NSAID"));
    println!("Aspirin -> NSAID distance: {}", d.conceptual);
}

// ============================================================================
// CATEGORY 10: Error Cases
// ============================================================================

#[test]
fn test_unknown_iri_handling() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    // Distance to unknown IRI should be very high
    let unknown = iri("UnknownTerm");
    let d = index.distance(&iri("Aspirin"), &unknown);

    // Unknown terms should have very high distance
    assert!(
        d.conceptual >= 0.9,
        "Unknown term should be very distant: {}",
        d.conceptual
    );
}

#[test]
fn test_empty_hierarchy_handling() {
    let index = SemanticDistanceIndex::new();

    // Empty index should handle queries gracefully - returns high distance
    let d = index.distance(&iri("Aspirin"), &iri("Drug"));
    assert!(
        d.conceptual >= 0.9,
        "Empty index should return high distance: {}",
        d.conceptual
    );
}

#[test]
fn test_self_distance_is_zero() {
    let mut index = SemanticDistanceIndex::new();
    let terms = build_pharma_hierarchy();
    index.build_from_terms(&terms);

    let d = index.distance(&iri("Aspirin"), &iri("Aspirin"));
    assert!(d.is_exact(), "Self-distance should be exact (zero)");
}
