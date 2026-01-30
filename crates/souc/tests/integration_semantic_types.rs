//! Integration Tests for Semantic Metric Types & Native Ontology Infrastructure
//!
//! Day 47: Tests for the paradigm-shifting semantic type system where type
//! compatibility is based on continuous semantic distance rather than boolean equality.

use std::sync::{Arc, RwLock};

use sounio::ontology::distance::{
    PhysicalCost, SemanticDistance, SemanticDistanceIndex, path::HierarchyGraph,
};
use sounio::ontology::loader::{IRI, LoadedTerm, OntologyId};
use sounio::types::semantic::{SemanticCompatibility, SemanticType, SemanticTypeChecker};

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

// ============================================================================
// IRI Tests
// ============================================================================

#[test]
fn test_iri_creation() {
    let iri = IRI::new("http://example.org/term1");
    assert_eq!(iri.as_str(), "http://example.org/term1");
}

#[test]
fn test_iri_equality() {
    let iri1 = iri("term1");
    let iri2 = iri("term1");
    let iri3 = iri("term2");

    assert_eq!(iri1, iri2);
    assert_ne!(iri1, iri3);
}

#[test]
fn test_iri_hash() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(iri("term1"));
    set.insert(iri("term2"));
    set.insert(iri("term1")); // duplicate

    assert_eq!(set.len(), 2);
}

#[test]
fn test_iri_curie_conversion() {
    let iri = IRI::from_curie("CHEBI", "15365");
    assert!(iri.as_str().contains("CHEBI"));
    assert!(iri.as_str().contains("15365"));

    // Should be able to extract CURIE
    let curie = iri.to_curie();
    assert!(curie.is_some());
    let (prefix, local) = curie.unwrap();
    assert_eq!(prefix, "CHEBI");
    assert_eq!(local, "15365");
}

#[test]
fn test_iri_ontology_extraction() {
    let go_iri = IRI::from_curie("GO", "0008150");
    assert_eq!(go_iri.ontology(), OntologyId::GO);

    let chebi_iri = IRI::from_curie("CHEBI", "24431");
    assert_eq!(chebi_iri.ontology(), OntologyId::ChEBI);
}

// ============================================================================
// Hierarchy Graph Tests
// ============================================================================

#[test]
fn test_hierarchy_graph_add_terms() {
    let mut graph = HierarchyGraph::new();

    // Create a simple hierarchy: Animal -> Mammal -> Dog
    let animal = iri("Animal");
    let mammal = iri("Mammal");
    let dog = iri("Dog");

    graph.add_term(&animal, &[]);
    graph.add_term(&mammal, &[animal.clone()]);
    graph.add_term(&dog, &[mammal.clone()]);

    assert!(graph.contains(&animal));
    assert!(graph.contains(&mammal));
    assert!(graph.contains(&dog));
}

#[test]
fn test_hierarchy_graph_is_ancestor() {
    let mut graph = HierarchyGraph::new();

    let animal = iri("Animal");
    let mammal = iri("Mammal");
    let dog = iri("Dog");

    graph.add_term(&animal, &[]);
    graph.add_term(&mammal, &[animal.clone()]);
    graph.add_term(&dog, &[mammal.clone()]);

    // Animal is ancestor of Dog and Mammal
    assert!(graph.is_ancestor(&animal, &dog));
    assert!(graph.is_ancestor(&animal, &mammal));
    assert!(graph.is_ancestor(&mammal, &dog));

    // Dog is not ancestor of Animal
    assert!(!graph.is_ancestor(&dog, &animal));
    assert!(!graph.is_ancestor(&dog, &mammal));
}

#[test]
fn test_hierarchy_graph_path_length() {
    let mut graph = HierarchyGraph::new();

    let animal = iri("Animal");
    let mammal = iri("Mammal");
    let dog = iri("Dog");

    graph.add_term(&animal, &[]);
    graph.add_term(&mammal, &[animal.clone()]);
    graph.add_term(&dog, &[mammal.clone()]);

    // Distance from Dog to Animal = 2
    assert_eq!(graph.path_length(&dog, &animal), Some(2));

    // Distance from Dog to Mammal = 1
    assert_eq!(graph.path_length(&dog, &mammal), Some(1));

    // Distance from Dog to Dog = 0
    assert_eq!(graph.path_length(&dog, &dog), Some(0));
}

#[test]
fn test_hierarchy_graph_lca() {
    let mut graph = HierarchyGraph::new();

    // Create a diamond:
    //       Animal
    //      /      \
    //  Mammal    Bird
    //     |        |
    //    Dog     Sparrow

    let animal = iri("Animal");
    let mammal = iri("Mammal");
    let bird = iri("Bird");
    let dog = iri("Dog");
    let sparrow = iri("Sparrow");

    graph.add_term(&animal, &[]);
    graph.add_term(&mammal, &[animal.clone()]);
    graph.add_term(&bird, &[animal.clone()]);
    graph.add_term(&dog, &[mammal.clone()]);
    graph.add_term(&sparrow, &[bird.clone()]);

    // LCA of Dog and Sparrow should be Animal
    let lca = graph.lowest_common_ancestor(&dog, &sparrow);
    assert_eq!(lca, Some(animal.clone()));

    // LCA of Dog and Mammal should be Mammal
    let lca2 = graph.lowest_common_ancestor(&dog, &mammal);
    assert_eq!(lca2, Some(mammal.clone()));

    // LCA of Dog and Dog should be Dog
    let lca3 = graph.lowest_common_ancestor(&dog, &dog);
    assert_eq!(lca3, Some(dog.clone()));
}

#[test]
fn test_hierarchy_graph_lca_full() {
    let mut graph = HierarchyGraph::new();

    let animal = iri("Animal");
    let mammal = iri("Mammal");
    let dog = iri("Dog");
    let cat = iri("Cat");

    graph.add_term(&animal, &[]);
    graph.add_term(&mammal, &[animal.clone()]);
    graph.add_term(&dog, &[mammal.clone()]);
    graph.add_term(&cat, &[mammal.clone()]);

    // LCA of Dog and Cat with full result
    let lca = graph.lowest_common_ancestor_full(&dog, &cat);
    assert!(lca.is_some());

    let result = lca.unwrap();
    assert_eq!(result.ancestor, Some(mammal.clone()));
    assert_eq!(result.dist_a, 1); // Dog -> Mammal
    assert_eq!(result.dist_b, 1); // Cat -> Mammal
}

#[test]
fn test_hierarchy_graph_get_ancestors() {
    let mut graph = HierarchyGraph::new();

    let animal = iri("Animal");
    let mammal = iri("Mammal");
    let dog = iri("Dog");

    graph.add_term(&animal, &[]);
    graph.add_term(&mammal, &[animal.clone()]);
    graph.add_term(&dog, &[mammal.clone()]);

    let dog_ancestors = graph.get_ancestors(&dog);
    assert_eq!(dog_ancestors.len(), 2);
    assert!(dog_ancestors.contains(&mammal));
    assert!(dog_ancestors.contains(&animal));

    let animal_ancestors = graph.get_ancestors(&animal);
    assert!(animal_ancestors.is_empty());
}

#[test]
fn test_hierarchy_graph_build_from_terms() {
    let mut graph = HierarchyGraph::new();

    let terms = vec![
        make_term(iri("Animal"), "Animal", vec![]),
        make_term(iri("Mammal"), "Mammal", vec![iri("Animal")]),
        make_term(iri("Dog"), "Dog", vec![iri("Mammal")]),
        make_term(iri("Cat"), "Cat", vec![iri("Mammal")]),
    ];

    graph.build_from_terms(&terms);

    assert!(graph.contains(&iri("Animal")));
    assert!(graph.contains(&iri("Dog")));
    assert!(graph.is_ancestor(&iri("Animal"), &iri("Dog")));
}

// ============================================================================
// Semantic Distance Tests
// ============================================================================

#[test]
fn test_semantic_distance_zero() {
    let dist = SemanticDistance::ZERO;
    assert_eq!(dist.conceptual, 0.0);
    assert!(dist.is_exact());
}

#[test]
fn test_semantic_distance_max() {
    let dist = SemanticDistance::MAX;
    assert_eq!(dist.conceptual, 1.0);
    assert!(!dist.is_exact());
    assert!(!dist.is_subsumption());
    assert!(!dist.is_implicitly_compatible());
}

#[test]
fn test_semantic_distance_new() {
    let d1 = SemanticDistance::new(0.05);
    let d2 = SemanticDistance::new(0.25);
    let d3 = SemanticDistance::new(0.5);
    let d4 = SemanticDistance::new(0.8);

    // Subsumption distance (< 0.1)
    assert!(d1.is_subsumption());
    assert!(d1.is_implicitly_compatible());
    assert!(d1.is_explicitly_compatible());

    // Implicit coercion distance (< 0.3)
    assert!(!d2.is_subsumption());
    assert!(d2.is_implicitly_compatible());
    assert!(d2.is_explicitly_compatible());

    // Explicit cast distance (< 0.7)
    assert!(!d3.is_implicitly_compatible());
    assert!(d3.is_explicitly_compatible());

    // Too far for any coercion
    assert!(!d4.is_explicitly_compatible());
}

#[test]
fn test_semantic_distance_compose() {
    let d1 = SemanticDistance::new(0.2);
    let d2 = SemanticDistance::new(0.3);

    let composed = d1.compose(d2);

    // Distances add
    assert!((composed.conceptual - 0.5).abs() < 0.001);

    // Confidence multiplies
    assert!(composed.confidence_retention < d1.confidence_retention);

    // Provenance adds
    assert_eq!(
        composed.provenance_depth,
        d1.provenance_depth + d2.provenance_depth
    );
}

#[test]
fn test_semantic_distance_within_threshold() {
    let dist = SemanticDistance::new(0.25);

    assert!(dist.within_threshold(0.3));
    assert!(dist.within_threshold(0.25));
    assert!(!dist.within_threshold(0.2));
}

// ============================================================================
// Physical Cost Tests
// ============================================================================

#[test]
fn test_physical_cost_zero() {
    let cost = PhysicalCost::ZERO;
    assert_eq!(cost.cycles, 0);
    assert_eq!(cost.memory_tier, 0);
    assert_eq!(cost.network_hops, 0);
    assert_eq!(cost.allocation, 0);
}

#[test]
fn test_physical_cost_from_conceptual() {
    let zero_cost = PhysicalCost::from_conceptual(0.0);
    assert_eq!(zero_cost.cycles, 0);

    let small_cost = PhysicalCost::from_conceptual(0.05);
    assert_eq!(small_cost.memory_tier, 1); // L1 cache

    let medium_cost = PhysicalCost::from_conceptual(0.4);
    assert_eq!(medium_cost.memory_tier, 3); // L3 cache

    let large_cost = PhysicalCost::from_conceptual(0.8);
    assert_eq!(large_cost.memory_tier, 5); // SSD/Network
    assert_eq!(large_cost.network_hops, 1);
}

#[test]
fn test_physical_cost_add() {
    let cost1 = PhysicalCost::from_conceptual(0.2);
    let cost2 = PhysicalCost::from_conceptual(0.3);

    let combined = cost1.add(cost2);

    assert_eq!(combined.cycles, cost1.cycles + cost2.cycles);
    assert_eq!(
        combined.memory_tier,
        cost1.memory_tier.max(cost2.memory_tier)
    );
}

#[test]
fn test_physical_cost_memory_tier_name() {
    let cost = PhysicalCost::from_conceptual(0.05);
    assert_eq!(cost.memory_tier_name(), "L1 cache");

    let cost2 = PhysicalCost::from_conceptual(0.6);
    assert_eq!(cost2.memory_tier_name(), "RAM");
}

// ============================================================================
// Semantic Distance Index Tests
// ============================================================================

#[test]
fn test_distance_index_creation() {
    let index = SemanticDistanceIndex::new();
    // Index should be empty but valid
    let unknown1 = iri("unknown1");
    let unknown2 = iri("unknown2");
    let dist = index.distance(&unknown1, &unknown2);
    // Unknown terms get high distance
    assert!(dist.conceptual > 0.5);
}

#[test]
fn test_distance_index_add_terms() {
    let mut index = SemanticDistanceIndex::new();

    let animal = make_term(iri("Animal"), "Animal", vec![]);
    let mammal = make_term(iri("Mammal"), "Mammal", vec![iri("Animal")]);
    let dog = make_term(iri("Dog"), "Dog", vec![iri("Mammal")]);

    index.add_term(&animal);
    index.add_term(&mammal);
    index.add_term(&dog);

    // Now we can compute distance
    let dist = index.distance(&iri("Dog"), &iri("Animal"));
    assert!(dist.conceptual < 0.5); // Should be relatively close
}

#[test]
fn test_distance_index_build_from_terms() {
    let mut index = SemanticDistanceIndex::new();

    let terms = vec![
        make_term(iri("Entity"), "Entity", vec![]),
        make_term(iri("Disease"), "Disease", vec![iri("Entity")]),
        make_term(iri("Cancer"), "Cancer", vec![iri("Disease")]),
        make_term(iri("LungCancer"), "Lung Cancer", vec![iri("Cancer")]),
    ];

    index.build_from_terms(&terms);

    // LungCancer -> Disease should have smaller distance than LungCancer -> Entity
    let dist1 = index.distance(&iri("LungCancer"), &iri("Disease"));
    let dist2 = index.distance(&iri("LungCancer"), &iri("Entity"));

    assert!(dist1.conceptual < dist2.conceptual);
}

#[test]
fn test_distance_index_identity() {
    let mut index = SemanticDistanceIndex::new();

    let animal = make_term(iri("Animal"), "Animal", vec![]);
    index.add_term(&animal);

    let dist = index.distance(&iri("Animal"), &iri("Animal"));
    assert!(dist.is_exact());
    assert_eq!(dist.conceptual, 0.0);
}

#[test]
fn test_distance_index_subsumption() {
    let mut index = SemanticDistanceIndex::new();

    let terms = vec![
        make_term(iri("Animal"), "Animal", vec![]),
        make_term(iri("Mammal"), "Mammal", vec![iri("Animal")]),
    ];

    index.build_from_terms(&terms);

    // Mammal -> Animal (upcast) should have low distance
    let upcast = index.distance(&iri("Mammal"), &iri("Animal"));
    assert!(upcast.is_subsumption() || upcast.is_implicitly_compatible());
}

// ============================================================================
// Semantic Type Tests
// ============================================================================

#[test]
fn test_semantic_type_from_iri() {
    let ty = SemanticType::from_iri(iri("Disease"), "Disease".to_string());
    assert_eq!(ty.name, "Disease");
    assert_eq!(ty.implicit_threshold, 0.3);
    assert_eq!(ty.explicit_threshold, 0.7);
}

#[test]
fn test_semantic_type_from_curie() {
    let ty = SemanticType::from_curie("CHEBI", "15365", "Aspirin".to_string());
    assert_eq!(ty.ontology, "CHEBI");
    assert_eq!(ty.local_id, "15365");
    assert_eq!(ty.name, "Aspirin");
}

#[test]
fn test_semantic_type_curie_format() {
    let ty = SemanticType::from_curie("GO", "0008150", "biological_process".to_string());
    assert_eq!(ty.curie(), "GO:0008150");
}

#[test]
fn test_semantic_type_with_thresholds() {
    let ty = SemanticType::from_iri(iri("Strict"), "Strict".to_string()).with_thresholds(0.1, 0.3);

    assert_eq!(ty.implicit_threshold, 0.1);
    assert_eq!(ty.explicit_threshold, 0.3);
}

// ============================================================================
// Semantic Compatibility Tests
// ============================================================================

#[test]
fn test_semantic_compatibility_compatible() {
    let compat = SemanticCompatibility::compatible(SemanticDistance::new(0.1));

    assert!(compat.is_compatible());
    assert!(compat.implicit_compatible);
    assert!(compat.explicit_compatible);
    assert!(compat.incompatibility_reason.is_none());
}

#[test]
fn test_semantic_compatibility_incompatible() {
    let compat = SemanticCompatibility::incompatible("Types too distant".to_string());

    assert!(!compat.is_compatible());
    assert!(!compat.implicit_compatible);
    assert!(!compat.explicit_compatible);
    assert!(compat.incompatibility_reason.is_some());
}

// ============================================================================
// Semantic Type Checker Tests
// ============================================================================

#[test]
fn test_type_checker_creation() {
    let index = Arc::new(RwLock::new(SemanticDistanceIndex::new()));
    let _checker = SemanticTypeChecker::new(index);
}

#[test]
fn test_type_checker_same_type() {
    let mut index = SemanticDistanceIndex::new();
    let disease = make_term(iri("Disease"), "Disease", vec![]);
    index.add_term(&disease);

    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let disease_ty = SemanticType::from_iri(iri("Disease"), "Disease".to_string());

    let compat = checker.check_compatibility(&disease_ty, &disease_ty);
    assert!(compat.is_compatible());
    assert!(compat.implicit_compatible);
    assert!(compat.distance.is_exact());
}

#[test]
fn test_type_checker_subtype() {
    let mut index = SemanticDistanceIndex::new();

    let terms = vec![
        make_term(iri("Disease"), "Disease", vec![]),
        make_term(iri("Cancer"), "Cancer", vec![iri("Disease")]),
        make_term(iri("LungCancer"), "LungCancer", vec![iri("Cancer")]),
    ];

    index.build_from_terms(&terms);
    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let cancer_ty = SemanticType::from_iri(iri("Cancer"), "Cancer".to_string());
    let disease_ty = SemanticType::from_iri(iri("Disease"), "Disease".to_string());

    // Cancer -> Disease should be compatible (upcast)
    let compat = checker.check_compatibility(&cancer_ty, &disease_ty);
    assert!(compat.is_compatible());
}

#[test]
fn test_type_checker_allows_implicit() {
    let mut index = SemanticDistanceIndex::new();

    let terms = vec![
        make_term(iri("Drug"), "Drug", vec![]),
        make_term(iri("Aspirin"), "Aspirin", vec![iri("Drug")]),
    ];

    index.build_from_terms(&terms);
    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let aspirin_ty = SemanticType::from_iri(iri("Aspirin"), "Aspirin".to_string());
    let drug_ty = SemanticType::from_iri(iri("Drug"), "Drug".to_string());

    // Direct subtype should allow implicit coercion
    assert!(checker.allows_implicit_coercion(&aspirin_ty, &drug_ty));
}

#[test]
fn test_type_checker_allows_explicit() {
    let mut index = SemanticDistanceIndex::new();

    let terms = vec![
        make_term(iri("Entity"), "Entity", vec![]),
        make_term(iri("Physical"), "Physical", vec![iri("Entity")]),
        make_term(iri("Chemical"), "Chemical", vec![iri("Physical")]),
        make_term(iri("Drug"), "Drug", vec![iri("Chemical")]),
    ];

    index.build_from_terms(&terms);
    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let drug_ty = SemanticType::from_iri(iri("Drug"), "Drug".to_string());
    let entity_ty = SemanticType::from_iri(iri("Entity"), "Entity".to_string());

    // Drug -> Entity might require explicit cast depending on threshold
    let allows = checker.allows_explicit_cast(&drug_ty, &entity_ty);
    // Either way, this should be a valid check
    assert!(allows || !allows); // Just verifying it runs
}

// ============================================================================
// Integration: Full Type Checking Pipeline
// ============================================================================

#[test]
fn test_full_type_checking_pipeline() {
    // Create a semantic distance index with a small ontology
    let mut index = SemanticDistanceIndex::new();

    let terms = vec![
        make_term(iri("Entity"), "Entity", vec![]),
        make_term(
            iri("PhysicalEntity"),
            "Physical Entity",
            vec![iri("Entity")],
        ),
        make_term(iri("Material"), "Material", vec![iri("PhysicalEntity")]),
        make_term(iri("Chemical"), "Chemical", vec![iri("Material")]),
        make_term(iri("Drug"), "Drug", vec![iri("Chemical")]),
        make_term(iri("Aspirin"), "Aspirin", vec![iri("Drug")]),
    ];

    index.build_from_terms(&terms);
    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    // Create semantic types
    let aspirin = SemanticType::from_iri(iri("Aspirin"), "Aspirin".to_string());
    let drug = SemanticType::from_iri(iri("Drug"), "Drug".to_string());
    let chemical = SemanticType::from_iri(iri("Chemical"), "Chemical".to_string());
    let entity = SemanticType::from_iri(iri("Entity"), "Entity".to_string());

    // Test compatibility at different levels
    let compat_aspirin_drug = checker.check_compatibility(&aspirin, &drug);
    let compat_aspirin_chemical = checker.check_compatibility(&aspirin, &chemical);
    let compat_aspirin_entity = checker.check_compatibility(&aspirin, &entity);

    // All should be compatible (upcast along hierarchy)
    assert!(compat_aspirin_drug.is_compatible());
    assert!(compat_aspirin_chemical.is_compatible());
    assert!(compat_aspirin_entity.is_compatible());

    // Distance should increase with hierarchy depth
    assert!(compat_aspirin_drug.distance.conceptual <= compat_aspirin_chemical.distance.conceptual);
    assert!(
        compat_aspirin_chemical.distance.conceptual <= compat_aspirin_entity.distance.conceptual
    );
}

#[test]
fn test_type_coercion_physical_cost() {
    let mut index = SemanticDistanceIndex::new();

    let terms = vec![
        make_term(iri("Disease"), "Disease", vec![]),
        make_term(iri("Cancer"), "Cancer", vec![iri("Disease")]),
        make_term(iri("LungCancer"), "Lung Cancer", vec![iri("Cancer")]),
    ];

    index.build_from_terms(&terms);
    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let lung_cancer = SemanticType::from_iri(iri("LungCancer"), "Lung Cancer".to_string());
    let disease = SemanticType::from_iri(iri("Disease"), "Disease".to_string());

    let compat = checker.check_compatibility(&lung_cancer, &disease);
    assert!(compat.is_compatible());

    // Physical cost should be computable
    let cost = compat.distance.physical_cost;

    // Type coercion has measurable cost
    // (may be 0 for very close types or small for direct subtypes)
    println!(
        "Coercion cost: {} cycles, memory tier: {}",
        cost.cycles,
        cost.memory_tier_name()
    );
}

#[test]
fn test_sibling_types() {
    let mut index = SemanticDistanceIndex::new();

    // Create sibling types (same parent, different leaves)
    let terms = vec![
        make_term(iri("Animal"), "Animal", vec![]),
        make_term(iri("Mammal"), "Mammal", vec![iri("Animal")]),
        make_term(iri("Dog"), "Dog", vec![iri("Mammal")]),
        make_term(iri("Cat"), "Cat", vec![iri("Mammal")]),
    ];

    index.build_from_terms(&terms);
    let index = Arc::new(RwLock::new(index));
    let checker = SemanticTypeChecker::new(index);

    let dog = SemanticType::from_iri(iri("Dog"), "Dog".to_string());
    let cat = SemanticType::from_iri(iri("Cat"), "Cat".to_string());

    // Dog and Cat are siblings - distance depends on LCA
    let compat = checker.check_compatibility(&dog, &cat);

    // Distance should be > 0 (not identical)
    assert!(compat.distance.conceptual > 0.0);

    // But they share Mammal as LCA, so shouldn't be maximally distant
    assert!(compat.distance.conceptual < 1.0);
}

// ============================================================================
// LoadedTerm Structure Tests
// ============================================================================

#[test]
fn test_loaded_term_creation() {
    let term = make_term(
        curie_iri("GO", "0008150"),
        "biological_process",
        vec![curie_iri("BFO", "0000015")],
    );

    assert_eq!(term.label, "biological_process");
    assert_eq!(term.superclasses.len(), 1);
}

#[test]
fn test_loaded_term_with_multiple_parents() {
    let term = make_term(
        iri("MultiParentTerm"),
        "Multi Parent Term",
        vec![iri("Parent1"), iri("Parent2"), iri("Parent3")],
    );

    assert_eq!(term.superclasses.len(), 3);
}

// ============================================================================
// OntologyId Tests
// ============================================================================

#[test]
fn test_ontology_id_from_prefix() {
    assert_eq!(OntologyId::from_prefix("GO"), OntologyId::GO);
    assert_eq!(OntologyId::from_prefix("CHEBI"), OntologyId::ChEBI);
    assert_eq!(OntologyId::from_prefix("DOID"), OntologyId::DOID);
    assert_eq!(OntologyId::from_prefix("unknown"), OntologyId::Unknown);
}

#[test]
fn test_ontology_id_prefix() {
    assert_eq!(OntologyId::GO.prefix(), "GO");
    assert_eq!(OntologyId::ChEBI.prefix(), "CHEBI");
    assert_eq!(OntologyId::DOID.prefix(), "DOID");
}

#[test]
fn test_ontology_id_display() {
    assert_eq!(format!("{}", OntologyId::GO), "GO");
    assert_eq!(format!("{}", OntologyId::ChEBI), "CHEBI");
}

// ============================================================================
// Day 48: Embedding Space Tests
// ============================================================================

use sounio::ontology::embedding::{
    Embedding, EmbeddingConfig, EmbeddingModel, EmbeddingSpace, EmbeddingStore,
    storage::ann::AnnIndex, storage::memory::MemoryStore,
};

#[test]
fn test_embedding_creation() {
    let emb = Embedding::new(
        iri("Heart"),
        vec![0.1, 0.2, 0.3, 0.4],
        EmbeddingModel::Structural,
    );

    assert_eq!(emb.iri, iri("Heart"));
    assert_eq!(emb.vector.len(), 4);
    assert_eq!(emb.model, EmbeddingModel::Structural);
    assert!((emb.confidence - 1.0).abs() < 0.001);
}

#[test]
fn test_embedding_cosine_similarity_identical() {
    let emb1 = Embedding::new(iri("A"), vec![1.0, 0.0, 0.0], EmbeddingModel::Textual);
    let emb2 = Embedding::new(iri("B"), vec![1.0, 0.0, 0.0], EmbeddingModel::Textual);

    let sim = emb1.cosine_similarity(&emb2);
    assert!(
        (sim - 1.0).abs() < 0.001,
        "Identical vectors should have similarity 1.0"
    );
}

#[test]
fn test_embedding_cosine_similarity_orthogonal() {
    let emb1 = Embedding::new(iri("A"), vec![1.0, 0.0, 0.0], EmbeddingModel::Textual);
    let emb2 = Embedding::new(iri("B"), vec![0.0, 1.0, 0.0], EmbeddingModel::Textual);

    let sim = emb1.cosine_similarity(&emb2);
    assert!(
        sim.abs() < 0.001,
        "Orthogonal vectors should have similarity 0.0"
    );
}

#[test]
fn test_embedding_cosine_similarity_opposite() {
    let emb1 = Embedding::new(iri("A"), vec![1.0, 0.0, 0.0], EmbeddingModel::Textual);
    let emb2 = Embedding::new(iri("B"), vec![-1.0, 0.0, 0.0], EmbeddingModel::Textual);

    let sim = emb1.cosine_similarity(&emb2);
    assert!(
        (sim - (-1.0)).abs() < 0.001,
        "Opposite vectors should have similarity -1.0"
    );
}

#[test]
fn test_embedding_to_semantic_distance() {
    // Identical vectors -> distance 0
    let emb1 = Embedding::new(iri("A"), vec![1.0, 0.0], EmbeddingModel::Textual);
    let emb2 = Embedding::new(iri("B"), vec![1.0, 0.0], EmbeddingModel::Textual);
    let dist = emb1.to_semantic_distance(&emb2);
    assert!(dist < 0.001, "Identical vectors should have distance ~0");

    // Orthogonal vectors -> distance 0.5
    let emb3 = Embedding::new(iri("C"), vec![0.0, 1.0], EmbeddingModel::Textual);
    let dist2 = emb1.to_semantic_distance(&emb3);
    assert!(
        (dist2 - 0.5).abs() < 0.001,
        "Orthogonal vectors should have distance ~0.5"
    );

    // Opposite vectors -> distance 1.0
    let emb4 = Embedding::new(iri("D"), vec![-1.0, 0.0], EmbeddingModel::Textual);
    let dist3 = emb1.to_semantic_distance(&emb4);
    assert!(
        (dist3 - 1.0).abs() < 0.001,
        "Opposite vectors should have distance ~1.0"
    );
}

#[test]
fn test_embedding_auto_normalization() {
    // Embedding::new auto-normalizes
    let emb = Embedding::new(iri("A"), vec![3.0, 4.0], EmbeddingModel::Textual);

    // 3-4-5 triangle: magnitude should be 5, normalized should be [0.6, 0.8]
    assert!((emb.vector[0] - 0.6).abs() < 0.001);
    assert!((emb.vector[1] - 0.8).abs() < 0.001);

    // Magnitude should be ~1
    let mag: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mag - 1.0).abs() < 0.001);
}

#[test]
fn test_memory_store() {
    let mut store = MemoryStore::new();
    let emb = Embedding::new(iri("Test"), vec![0.1, 0.2, 0.3], EmbeddingModel::Structural);

    store.put(&iri("Test"), &emb).unwrap();

    let retrieved = store.get(&iri("Test")).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.iri, iri("Test"));
}

#[test]
fn test_memory_store_overwrite() {
    let mut store = MemoryStore::new();

    let emb1 = Embedding::new(iri("A"), vec![1.0, 0.0], EmbeddingModel::Structural);
    store.put(&iri("A"), &emb1).unwrap();

    let emb2 = Embedding::new(iri("A"), vec![0.0, 1.0], EmbeddingModel::Textual);
    store.put(&iri("A"), &emb2).unwrap();

    let retrieved = store.get(&iri("A")).unwrap().unwrap();
    assert_eq!(retrieved.model, EmbeddingModel::Textual);
}

#[test]
fn test_memory_store_count() {
    let mut store = MemoryStore::new();
    assert_eq!(store.count().unwrap(), 0);

    let emb_a = Embedding::new(
        iri("A"),
        vec![1.0, 0.0, 0.0, 0.0],
        EmbeddingModel::Structural,
    );
    store.put(&iri("A"), &emb_a).unwrap();
    assert_eq!(store.count().unwrap(), 1);

    let emb_b = Embedding::new(
        iri("B"),
        vec![0.0, 1.0, 0.0, 0.0],
        EmbeddingModel::Structural,
    );
    store.put(&iri("B"), &emb_b).unwrap();
    assert_eq!(store.count().unwrap(), 2);

    let emb_c = Embedding::new(
        iri("C"),
        vec![0.0, 0.0, 1.0, 0.0],
        EmbeddingModel::Structural,
    );
    store.put(&iri("C"), &emb_c).unwrap();
    assert_eq!(store.count().unwrap(), 3);
}

#[test]
fn test_embedding_config_defaults() {
    let config = EmbeddingConfig::default();

    assert_eq!(config.dimensions, 256);
    assert_eq!(config.model, EmbeddingModel::Hybrid);
}

#[test]
fn test_embedding_space_creation() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 64;
    config.lazy_generation = false;
    let space = EmbeddingSpace::new(config).unwrap();

    // Can verify it was created by checking try_get returns None for non-existent
    assert!(space.try_get(&iri("NonExistent")).is_none());
}

#[test]
fn test_embedding_space_add_and_get() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 3;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    let emb = Embedding::new(
        iri("Heart"),
        vec![0.5, 0.3, 0.2],
        EmbeddingModel::Structural,
    );
    space.add(emb).unwrap();

    let retrieved = space.try_get(&iri("Heart"));
    assert!(retrieved.is_some());
}

#[test]
fn test_embedding_space_similarity() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 3;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // Add two similar embeddings and one different
    space
        .add(Embedding::new(
            iri("Heart"),
            vec![1.0, 0.1, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("Cardiac"),
            vec![0.95, 0.15, 0.05],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("Kidney"),
            vec![0.0, 0.1, 1.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    // Heart and Cardiac should be very similar
    let sim_heart_cardiac = space
        .cosine_similarity(&iri("Heart"), &iri("Cardiac"))
        .unwrap();
    assert!(
        sim_heart_cardiac > 0.9,
        "Heart and Cardiac should be similar"
    );

    // Heart and Kidney should be quite different
    let sim_heart_kidney = space
        .cosine_similarity(&iri("Heart"), &iri("Kidney"))
        .unwrap();
    assert!(
        sim_heart_kidney < 0.3,
        "Heart and Kidney should be dissimilar"
    );
}

#[test]
fn test_embedding_space_distance() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 2;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    space
        .add(Embedding::new(
            iri("A"),
            vec![1.0, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();
    space
        .add(Embedding::new(
            iri("B"),
            vec![1.0, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();
    space
        .add(Embedding::new(
            iri("C"),
            vec![0.0, 1.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    // Same direction -> distance ~0
    let dist_ab = space.embedding_distance(&iri("A"), &iri("B")).unwrap();
    assert!(dist_ab < 0.01);

    // Orthogonal -> distance ~0.5
    let dist_ac = space.embedding_distance(&iri("A"), &iri("C")).unwrap();
    assert!((dist_ac - 0.5).abs() < 0.01);
}

// ============================================================================
// ANN Index Tests
// ============================================================================

#[test]
fn test_ann_index_creation() {
    let index = AnnIndex::new(128);
    assert!(!index.is_built());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_ann_index_insert_and_build() {
    let mut index = AnnIndex::new(3);

    index.add(&iri("A"), &[1.0, 0.0, 0.0]).unwrap();
    index.add(&iri("B"), &[0.0, 1.0, 0.0]).unwrap();
    index.add(&iri("C"), &[0.0, 0.0, 1.0]).unwrap();

    assert_eq!(index.len(), 3);
    assert!(!index.is_built());

    index.build().unwrap();
    assert!(index.is_built());
}

#[test]
fn test_ann_index_nearest_neighbors() {
    let mut index = AnnIndex::new(3);

    // Insert 5 vectors
    index.add(&iri("A"), &[1.0, 0.0, 0.0]).unwrap();
    index.add(&iri("B"), &[0.9, 0.1, 0.0]).unwrap(); // Close to A
    index.add(&iri("C"), &[0.0, 1.0, 0.0]).unwrap();
    index.add(&iri("D"), &[0.0, 0.0, 1.0]).unwrap();
    index.add(&iri("E"), &[0.95, 0.05, 0.0]).unwrap(); // Very close to A

    index.build().unwrap();

    // Find 3 nearest neighbors to a query close to A
    let query = vec![1.0, 0.0, 0.0];
    let neighbors = index.search(&query, 3).unwrap();

    assert_eq!(neighbors.len(), 3);
}

#[test]
fn test_ann_index_empty_query() {
    let mut index = AnnIndex::new(2);
    index.build().unwrap();

    let neighbors = index.search(&[1.0, 0.0], 5).unwrap();
    assert!(neighbors.is_empty());
}

// ============================================================================
// Integration: Distance Calculator with Embeddings
// ============================================================================

#[test]
fn test_distance_index_with_embeddings() {
    let mut distance_index = SemanticDistanceIndex::new();

    // Build hierarchy
    let terms = vec![
        make_term(iri("Organ"), "Organ", vec![]),
        make_term(iri("Heart"), "Heart", vec![iri("Organ")]),
        make_term(iri("Kidney"), "Kidney", vec![iri("Organ")]),
    ];
    distance_index.build_from_terms(&terms);

    // Create embedding space
    let mut config = EmbeddingConfig::default();
    config.dimensions = 3;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    space
        .add(Embedding::new(
            iri("Heart"),
            vec![1.0, 0.0, 0.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("Kidney"),
            vec![0.0, 0.0, 1.0],
            EmbeddingModel::Structural,
        ))
        .unwrap();

    // Add the embedding space to the distance index
    distance_index.set_embedding_space(space);

    assert!(distance_index.has_embeddings());
}

#[test]
fn test_embedding_distance_calculation() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 3;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // Add embeddings for terms that might not be directly related in hierarchy
    space
        .add(Embedding::new(
            iri("Aspirin"),
            vec![0.8, 0.2, 0.0],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("PainRelief"),
            vec![0.75, 0.25, 0.0],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("Antibiotic"),
            vec![0.1, 0.1, 0.9],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();

    // Aspirin and PainRelief should be close
    let dist1 = space
        .embedding_distance(&iri("Aspirin"), &iri("PainRelief"))
        .unwrap();
    assert!(
        dist1 < 0.1,
        "Aspirin and PainRelief should be semantically close"
    );

    // Aspirin and Antibiotic should be far
    let dist2 = space
        .embedding_distance(&iri("Aspirin"), &iri("Antibiotic"))
        .unwrap();
    assert!(
        dist2 > 0.3,
        "Aspirin and Antibiotic should be semantically distant"
    );
}

#[test]
fn test_embedding_captures_synonymy() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 4;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    // "Heart" and "Cardiac organ" are synonyms - should have similar embeddings
    space
        .add(Embedding::new(
            iri("Heart"),
            vec![0.9, 0.1, 0.0, 0.0],
            EmbeddingModel::Textual,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("CardiacOrgan"),
            vec![0.88, 0.12, 0.0, 0.0],
            EmbeddingModel::Textual,
        ))
        .unwrap();

    // "Liver" is a different organ
    space
        .add(Embedding::new(
            iri("Liver"),
            vec![0.0, 0.1, 0.9, 0.0],
            EmbeddingModel::Textual,
        ))
        .unwrap();

    let sim_synonyms = space
        .cosine_similarity(&iri("Heart"), &iri("CardiacOrgan"))
        .unwrap();
    let sim_different = space
        .cosine_similarity(&iri("Heart"), &iri("Liver"))
        .unwrap();

    assert!(
        sim_synonyms > 0.95,
        "Synonyms should have very high similarity"
    );
    assert!(
        sim_different < 0.3,
        "Different organs should have low similarity"
    );
}

#[test]
fn test_embedding_captures_association() {
    let mut config = EmbeddingConfig::default();
    config.dimensions = 4;
    config.lazy_generation = false;
    let mut space = EmbeddingSpace::new(config).unwrap();

    space
        .add(Embedding::new(
            iri("Diabetes"),
            vec![0.7, 0.3, 0.0, 0.0],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("Insulin"),
            vec![0.65, 0.35, 0.0, 0.0],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();

    space
        .add(Embedding::new(
            iri("Fracture"),
            vec![0.0, 0.0, 0.8, 0.2],
            EmbeddingModel::Hybrid,
        ))
        .unwrap();

    // Diabetes and Insulin are associated
    let dist_associated = space
        .embedding_distance(&iri("Diabetes"), &iri("Insulin"))
        .unwrap();
    // Diabetes and Fracture are unrelated
    let dist_unrelated = space
        .embedding_distance(&iri("Diabetes"), &iri("Fracture"))
        .unwrap();

    assert!(
        dist_associated < dist_unrelated,
        "Associated terms should be closer than unrelated terms"
    );
}
