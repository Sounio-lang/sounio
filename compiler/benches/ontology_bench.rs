//! Ontology Module Benchmarks
//!
//! Benchmarks for the semantic distance and ontology integration system:
//! - Hierarchy building and indexing
//! - Distance calculation (path, IC, embedding)
//! - Type checking with semantic types
//! - ANN index operations

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use sounio::ontology::{
    distance::SemanticDistanceIndex,
    embedding::{Embedding, EmbeddingConfig, EmbeddingModel, EmbeddingSpace},
    loader::{IRI, LoadedTerm, OntologyId},
};
use sounio::types::semantic::{SemanticType, SemanticTypeChecker};
use std::sync::{Arc, RwLock};

// ============================================================================
// Test Data Generation
// ============================================================================

/// Create a simple IRI from a local name
fn iri(s: &str) -> IRI {
    IRI::new(&format!("http://example.org/{}", s))
}

/// Create a LoadedTerm for testing
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

/// Build a pharmaceutical hierarchy for benchmarking
fn build_pharma_hierarchy() -> Vec<LoadedTerm> {
    vec![
        make_term(iri("ChemicalEntity"), "Chemical Entity", vec![]),
        make_term(iri("Drug"), "Drug", vec![iri("ChemicalEntity")]),
        make_term(iri("Analgesic"), "Analgesic", vec![iri("Drug")]),
        make_term(iri("NSAID"), "NSAID", vec![iri("Analgesic")]),
        make_term(iri("Aspirin"), "Aspirin", vec![iri("NSAID")]),
        make_term(iri("Ibuprofen"), "Ibuprofen", vec![iri("NSAID")]),
        make_term(iri("Opioid"), "Opioid", vec![iri("Analgesic")]),
        make_term(iri("Morphine"), "Morphine", vec![iri("Opioid")]),
        make_term(iri("Antibiotic"), "Antibiotic", vec![iri("Drug")]),
        make_term(iri("Penicillin"), "Penicillin", vec![iri("Antibiotic")]),
        make_term(iri("Amoxicillin"), "Amoxicillin", vec![iri("Penicillin")]),
    ]
}

/// Build a large hierarchy for scalability testing
fn build_large_hierarchy(depth: usize, branching: usize) -> Vec<LoadedTerm> {
    let mut terms = vec![make_term(iri("Root"), "Root", vec![])];
    let mut current_level = vec![iri("Root")];

    for level in 0..depth {
        let mut next_level = Vec::new();
        for (idx, parent) in current_level.iter().enumerate() {
            for branch in 0..branching {
                let name = format!("L{}_P{}_B{}", level, idx, branch);
                let term_iri = iri(&name);
                terms.push(make_term(term_iri.clone(), &name, vec![parent.clone()]));
                next_level.push(term_iri);
            }
        }
        current_level = next_level;
    }

    terms
}

// ============================================================================
// Hierarchy Building Benchmarks
// ============================================================================

/// Benchmark building the semantic distance index from terms
fn benchmark_hierarchy_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchy_building");

    // Small hierarchy
    let small_terms = build_pharma_hierarchy();
    group.bench_function("small_11_terms", |b| {
        b.iter(|| {
            let mut index = SemanticDistanceIndex::new();
            index.build_from_terms(&small_terms);
            black_box(index)
        });
    });

    // Medium hierarchy (depth=4, branching=3 = ~120 terms)
    let medium_terms = build_large_hierarchy(4, 3);
    group.bench_with_input(
        BenchmarkId::new("medium", medium_terms.len()),
        &medium_terms,
        |b, terms| {
            b.iter(|| {
                let mut index = SemanticDistanceIndex::new();
                index.build_from_terms(terms);
                black_box(index)
            });
        },
    );

    // Large hierarchy (depth=5, branching=4 = ~1365 terms)
    let large_terms = build_large_hierarchy(5, 4);
    group.bench_with_input(
        BenchmarkId::new("large", large_terms.len()),
        &large_terms,
        |b, terms| {
            b.iter(|| {
                let mut index = SemanticDistanceIndex::new();
                index.build_from_terms(terms);
                black_box(index)
            });
        },
    );

    group.finish();
}

// ============================================================================
// Distance Calculation Benchmarks
// ============================================================================

/// Benchmark distance calculations between concepts
fn benchmark_distance_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculation");

    let terms = build_pharma_hierarchy();
    let mut index = SemanticDistanceIndex::new();
    index.build_from_terms(&terms);

    // Direct parent-child distance
    group.bench_function("direct_subsumption", |b| {
        b.iter(|| black_box(index.distance(&iri("Aspirin"), &iri("NSAID"))));
    });

    // Sibling distance
    group.bench_function("sibling_distance", |b| {
        b.iter(|| black_box(index.distance(&iri("Aspirin"), &iri("Ibuprofen"))));
    });

    // Cousin distance (different branches)
    group.bench_function("cousin_distance", |b| {
        b.iter(|| black_box(index.distance(&iri("Aspirin"), &iri("Morphine"))));
    });

    // Cross-branch distance
    group.bench_function("cross_branch_distance", |b| {
        b.iter(|| black_box(index.distance(&iri("Aspirin"), &iri("Amoxicillin"))));
    });

    // Self distance
    group.bench_function("self_distance", |b| {
        b.iter(|| black_box(index.distance(&iri("Aspirin"), &iri("Aspirin"))));
    });

    group.finish();
}

/// Benchmark batch distance calculations
fn benchmark_batch_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distance");

    let terms = build_pharma_hierarchy();
    let mut index = SemanticDistanceIndex::new();
    index.build_from_terms(&terms);

    let query_pairs: Vec<_> = vec![
        (iri("Aspirin"), iri("NSAID")),
        (iri("Aspirin"), iri("Ibuprofen")),
        (iri("Aspirin"), iri("Morphine")),
        (iri("Morphine"), iri("Opioid")),
        (iri("Penicillin"), iri("Antibiotic")),
    ];

    for batch_size in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    let mut total = 0.0;
                    for i in 0..size {
                        let (from, to) = &query_pairs[i % query_pairs.len()];
                        let d = index.distance(from, to);
                        total += d.conceptual;
                    }
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Type Checking Benchmarks
// ============================================================================

/// Benchmark semantic type checking operations
fn benchmark_type_checking(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_checking");

    let terms = build_pharma_hierarchy();
    let mut index = SemanticDistanceIndex::new();
    index.build_from_terms(&terms);
    let checker = SemanticTypeChecker::new(Arc::new(RwLock::new(index)));

    let aspirin_type = SemanticType::from_iri(iri("Aspirin"), "Aspirin".to_string());
    let nsaid_type = SemanticType::from_iri(iri("NSAID"), "NSAID".to_string());
    let drug_type = SemanticType::from_iri(iri("Drug"), "Drug".to_string());

    // Implicit coercion check
    group.bench_function("allows_implicit", |b| {
        b.iter(|| black_box(checker.allows_implicit_coercion(&aspirin_type, &drug_type)));
    });

    // Check compatibility
    group.bench_function("check_compatibility", |b| {
        b.iter(|| black_box(checker.check_compatibility(&aspirin_type, &nsaid_type)));
    });

    // Semantic distance
    group.bench_function("semantic_distance", |b| {
        b.iter(|| black_box(checker.semantic_distance(&aspirin_type, &nsaid_type)));
    });

    group.finish();
}

// ============================================================================
// Embedding Benchmarks
// ============================================================================

/// Benchmark embedding space operations
fn benchmark_embedding_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_operations");

    // Create embedding space with default config
    let config = EmbeddingConfig::default();
    let mut space = EmbeddingSpace::new(config).expect("Failed to create embedding space");

    // Add embeddings for test concepts
    let dim = 128;
    let concepts: Vec<_> = (0..100).map(|i| iri(&format!("Concept{}", i))).collect();

    for concept in &concepts {
        let vector: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01).sin()).collect();
        let embedding = Embedding::new(concept.clone(), vector, EmbeddingModel::Structural);
        space.add(embedding).expect("Failed to add embedding");
    }

    // Benchmark embedding lookup
    group.bench_function("embedding_lookup", |b| {
        b.iter(|| black_box(space.get(&concepts[50])));
    });

    // Benchmark cosine similarity
    group.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(space.cosine_similarity(&concepts[0], &concepts[50])));
    });

    // Benchmark batch similarity
    group.bench_with_input(BenchmarkId::new("batch_similarity", 100), &100, |b, &n| {
        b.iter(|| {
            let mut total = 0.0f32;
            for i in 0..n {
                if let Ok(sim) = space.cosine_similarity(&concepts[0], &concepts[i]) {
                    total += sim;
                }
            }
            black_box(total)
        });
    });

    group.finish();
}

/// Benchmark nearest neighbor search
fn benchmark_nearest_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_neighbors");

    // Create larger embedding space
    let config = EmbeddingConfig::default();
    let mut space = EmbeddingSpace::new(config).expect("Failed to create embedding space");

    let dim = 128;
    let num_concepts = 1000;
    let concepts: Vec<_> = (0..num_concepts)
        .map(|i| iri(&format!("Concept{}", i)))
        .collect();

    for (i, concept) in concepts.iter().enumerate() {
        let vector: Vec<f32> = (0..dim)
            .map(|d| ((i as f32 + d as f32) * 0.01).sin())
            .collect();
        let embedding = Embedding::new(concept.clone(), vector, EmbeddingModel::Structural);
        space.add(embedding).expect("Failed to add embedding");
    }

    // Build ANN index
    space.build_ann_index().expect("Failed to build ANN index");

    // Benchmark k-NN search for various k
    for k in [5, 10, 20, 50] {
        group.bench_with_input(BenchmarkId::new("knn_search", k), &k, |b, &k| {
            b.iter(|| black_box(space.nearest_neighbors(&concepts[0], k)));
        });
    }

    group.finish();
}

// ============================================================================
// Scalability Benchmarks
// ============================================================================

/// Benchmark scalability with hierarchy size
fn benchmark_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(50); // Reduce sample size for slower benchmarks

    for depth in [3, 4, 5] {
        let terms = build_large_hierarchy(depth, 3);
        let term_count = terms.len();

        group.bench_with_input(
            BenchmarkId::new("build_index", term_count),
            &terms,
            |b, terms| {
                b.iter(|| {
                    let mut index = SemanticDistanceIndex::new();
                    index.build_from_terms(terms);
                    black_box(index)
                });
            },
        );
    }

    // Benchmark distance calculation scaling
    for depth in [3, 4, 5] {
        let terms = build_large_hierarchy(depth, 3);
        let term_count = terms.len();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        // Get leaf nodes for distance queries
        let leaves: Vec<_> = terms
            .iter()
            .filter(|t| t.label.starts_with(&format!("L{}", depth - 1)))
            .map(|t| t.iri.clone())
            .take(10)
            .collect();

        if leaves.len() >= 2 {
            group.bench_with_input(
                BenchmarkId::new("distance_query", term_count),
                &(&index, &leaves),
                |b, (index, leaves)| {
                    b.iter(|| {
                        let mut total = 0.0;
                        for i in 0..leaves.len() {
                            for j in (i + 1)..leaves.len() {
                                let d = index.distance(&leaves[i], &leaves[j]);
                                total += d.conceptual;
                            }
                        }
                        black_box(total)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    benchmark_hierarchy_building,
    benchmark_distance_calculation,
    benchmark_batch_distance,
    benchmark_type_checking,
    benchmark_embedding_operations,
    benchmark_nearest_neighbors,
    benchmark_scalability,
);

criterion_main!(benches);
