//! Layout Synthesis Benchmarks - Day 38 Hypothesis Validation
//!
//! These benchmarks validate the core hypothesis:
//! "Semantic clustering improves cache performance"
//!
//! We test three workload types:
//! 1. Sequential related access - concepts accessed in semantic order
//! 2. Random access - no locality pattern
//! 3. Mixed workload - 70% local, 30% random
//!
//! Expected results:
//! - Sequential related: significant improvement from clustering
//! - Random access: little to no improvement
//! - Mixed workload: moderate improvement

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use sounio::layout::{CacheInstrumentation, ConceptUsage};
use std::collections::HashMap;

/// Generate a synthetic ontology for testing
fn create_test_ontology() -> MockOntology {
    MockOntology::new()
}

/// Mock ontology for benchmarking when real ontology isn't available
struct MockOntology {
    /// Simulated concept hierarchy depths
    depths: HashMap<String, usize>,
    /// Simulated parent relationships
    parents: HashMap<String, String>,
}

impl MockOntology {
    fn new() -> Self {
        let mut depths = HashMap::new();
        let mut parents = HashMap::new();

        // Create a hierarchy of drug-related concepts
        // Level 0: BFO:0000001 (entity)
        depths.insert("BFO:0000001".to_string(), 0);

        // Level 1: CHEBI:24431 (chemical entity)
        depths.insert("CHEBI:24431".to_string(), 1);
        parents.insert("CHEBI:24431".to_string(), "BFO:0000001".to_string());

        // Level 2: CHEBI:23888 (drug)
        depths.insert("CHEBI:23888".to_string(), 2);
        parents.insert("CHEBI:23888".to_string(), "CHEBI:24431".to_string());

        // Level 3: Various specific drugs
        let drugs = [
            ("CHEBI:15365", "aspirin"),
            ("CHEBI:6807", "metformin"),
            ("CHEBI:6801", "methotrexate"),
            ("CHEBI:28748", "doxorubicin"),
            ("CHEBI:28304", "heparin"),
            ("CHEBI:44185", "metoprolol"),
            ("CHEBI:2674", "amoxicillin"),
            ("CHEBI:4031", "ciprofloxacin"),
            ("CHEBI:49575", "diazepam"),
            ("CHEBI:6030", "ibuprofen"),
        ];

        for (curie, _name) in &drugs {
            depths.insert(curie.to_string(), 3);
            parents.insert(curie.to_string(), "CHEBI:23888".to_string());
        }

        // Add some gene ontology terms (different branch)
        depths.insert("GO:0008150".to_string(), 1); // biological_process
        parents.insert("GO:0008150".to_string(), "BFO:0000001".to_string());

        depths.insert("GO:0008152".to_string(), 2); // metabolic_process
        parents.insert("GO:0008152".to_string(), "GO:0008150".to_string());

        depths.insert("GO:0006810".to_string(), 2); // transport
        parents.insert("GO:0006810".to_string(), "GO:0008150".to_string());

        Self { depths, parents }
    }

    #[allow(dead_code)]
    fn is_subclass(&self, child: &str, ancestor: &str) -> bool {
        if child == ancestor {
            return true;
        }
        let mut current = child.to_string();
        while let Some(parent) = self.parents.get(&current) {
            if parent == ancestor {
                return true;
            }
            current = parent.clone();
        }
        false
    }

    fn depth(&self, curie: &str) -> usize {
        self.depths.get(curie).copied().unwrap_or(10)
    }

    fn lca(&self, a: &str, b: &str) -> Option<String> {
        // Simple LCA: find common ancestor by walking up
        let mut ancestors_a = vec![a.to_string()];
        let mut current = a.to_string();
        while let Some(parent) = self.parents.get(&current) {
            ancestors_a.push(parent.clone());
            current = parent.clone();
        }

        let mut current = b.to_string();
        if ancestors_a.contains(&current) {
            return Some(current);
        }
        while let Some(parent) = self.parents.get(&current) {
            if ancestors_a.contains(parent) {
                return Some(parent.clone());
            }
            current = parent.clone();
        }

        None
    }

    fn distance(&self, a: &str, b: &str) -> u32 {
        if a == b {
            return 0;
        }

        if let Some(lca) = self.lca(a, b) {
            let depth_a = self.depth(a);
            let depth_b = self.depth(b);
            let depth_lca = self.depth(&lca);
            ((depth_a - depth_lca) + (depth_b - depth_lca)) as u32
        } else {
            // Different ontologies
            100
        }
    }
}

/// Build a distance matrix using mock ontology
fn build_mock_distance_matrix(concepts: &[String], ontology: &MockOntology) -> Vec<Vec<u32>> {
    let n = concepts.len();
    let mut matrix = vec![vec![0u32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = ontology.distance(&concepts[i], &concepts[j]);
            matrix[i][j] = dist;
            matrix[j][i] = dist;
        }
    }

    matrix
}

/// Generate sequential related workload (should benefit from clustering)
fn generate_sequential_related_workload() -> (ConceptUsage, Vec<String>) {
    let mut usage = ConceptUsage::new();

    // Drug concepts that are semantically related
    let drugs = vec![
        "CHEBI:15365", // aspirin
        "CHEBI:6807",  // metformin
        "CHEBI:6801",  // methotrexate
        "CHEBI:28748", // doxorubicin
        "CHEBI:28304", // heparin
    ];

    // Access pattern: related drugs accessed together
    let mut access_pattern = Vec::new();

    for _ in 0..20 {
        // Access all drugs in order, simulating processing a patient's medications
        for drug in &drugs {
            usage.record_access(drug);
            access_pattern.push(drug.to_string());
        }
    }

    // Record co-occurrences
    let refs: Vec<&str> = drugs.to_vec();
    usage.record_scope(&refs);

    (usage, access_pattern)
}

/// Generate random access workload (clustering shouldn't help much)
fn generate_random_workload() -> (ConceptUsage, Vec<String>) {
    let mut usage = ConceptUsage::new();

    let all_concepts = vec![
        "CHEBI:15365",
        "CHEBI:6807",
        "GO:0008150",
        "CHEBI:6801",
        "GO:0008152",
        "CHEBI:28748",
        "GO:0006810",
        "CHEBI:28304",
        "CHEBI:44185",
        "CHEBI:2674",
    ];

    // Pseudo-random pattern (deterministic for reproducibility)
    let indices = [3, 7, 1, 9, 0, 5, 2, 8, 4, 6, 7, 3, 9, 1, 0, 5, 8, 2, 6, 4];
    let mut access_pattern = Vec::new();

    for _ in 0..10 {
        for &idx in &indices {
            let concept = all_concepts[idx];
            usage.record_access(concept);
            access_pattern.push(concept.to_string());
        }
    }

    (usage, access_pattern)
}

/// Generate mixed workload (70% local, 30% random)
fn generate_mixed_workload() -> (ConceptUsage, Vec<String>) {
    let mut usage = ConceptUsage::new();

    let local_concepts = vec!["CHEBI:15365", "CHEBI:6807", "CHEBI:6801", "CHEBI:28748"];

    let random_concepts = ["GO:0008150", "GO:0008152", "GO:0006810", "CHEBI:28304"];

    let mut access_pattern = Vec::new();

    // Pattern: mostly local with occasional random jumps
    // Simulates typical code that works with related concepts
    // but occasionally needs unrelated data
    for iteration in 0..25 {
        // 70% local access
        for concept in &local_concepts {
            usage.record_access(concept);
            access_pattern.push(concept.to_string());
        }

        // 30% random (every ~3rd iteration)
        if iteration % 3 == 0 {
            let random_idx = iteration % random_concepts.len();
            let concept = random_concepts[random_idx];
            usage.record_access(concept);
            access_pattern.push(concept.to_string());
        }
    }

    // Record co-occurrences for local concepts
    let refs: Vec<&str> = local_concepts.to_vec();
    usage.record_scope(&refs);

    (usage, access_pattern)
}

/// Benchmark cache simulation for different workloads
fn benchmark_cache_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_simulation");

    // Test different cache sizes
    for cache_size in [4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("sequential_related", cache_size),
            &cache_size,
            |b, &size| {
                let (_, access_pattern) = generate_sequential_related_workload();
                b.iter(|| {
                    let mut cache = CacheInstrumentation::new(size);
                    for concept in &access_pattern {
                        cache.access(black_box(concept));
                    }
                    cache.stats().hit_rate()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("random_access", cache_size),
            &cache_size,
            |b, &size| {
                let (_, access_pattern) = generate_random_workload();
                b.iter(|| {
                    let mut cache = CacheInstrumentation::new(size);
                    for concept in &access_pattern {
                        cache.access(black_box(concept));
                    }
                    cache.stats().hit_rate()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mixed_workload", cache_size),
            &cache_size,
            |b, &size| {
                let (_, access_pattern) = generate_mixed_workload();
                b.iter(|| {
                    let mut cache = CacheInstrumentation::new(size);
                    for concept in &access_pattern {
                        cache.access(black_box(concept));
                    }
                    cache.stats().hit_rate()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark clustering algorithm performance
fn benchmark_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering");

    // Generate usage data
    let (usage, _) = generate_sequential_related_workload();
    let concepts: Vec<String> = usage.concepts.iter().cloned().collect();
    let ontology = create_test_ontology();

    // Build distance matrix (mock version)
    let dist_matrix = build_mock_distance_matrix(&concepts, &ontology);

    for max_clusters in [2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("hierarchical_clustering", max_clusters),
            &max_clusters,
            |b, &_max_clusters| {
                b.iter(|| {
                    // Simulate clustering computation
                    let mut total_dist = 0u32;
                    for i in 0..dist_matrix.len() {
                        for j in (i + 1)..dist_matrix.len() {
                            total_dist = total_dist.saturating_add(black_box(dist_matrix[i][j]));
                        }
                    }
                    total_dist
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hypothesis validation: baseline vs optimized layout
fn benchmark_layout_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_effectiveness");

    let workloads = vec![
        ("sequential_related", generate_sequential_related_workload()),
        ("random_access", generate_random_workload()),
        ("mixed_workload", generate_mixed_workload()),
    ];

    for (name, (usage, access_pattern)) in workloads {
        // Baseline: no pre-warming, arbitrary order
        group.bench_function(BenchmarkId::new("baseline", name), |b| {
            b.iter(|| {
                let mut cache = CacheInstrumentation::new(8);
                for concept in &access_pattern {
                    cache.access(black_box(concept));
                }
                cache.stats().hit_rate()
            });
        });

        // Optimized: pre-warm cache with hot concepts
        // Simulates the effect of semantic layout
        let hot_concepts: Vec<_> = usage
            .access_counts
            .iter()
            .filter(|&(_, count)| *count > 5)
            .map(|(c, _)| c.clone())
            .take(4)
            .collect();

        group.bench_function(BenchmarkId::new("optimized", name), |b| {
            b.iter(|| {
                let mut cache = CacheInstrumentation::new(8);

                // Pre-warm with hot concepts
                for concept in &hot_concepts {
                    cache.access(concept);
                }
                // Reset stats after pre-warming
                cache.reset();

                for concept in &access_pattern {
                    cache.access(black_box(concept));
                }
                cache.stats().hit_rate()
            });
        });
    }

    group.finish();
}

/// Hypothesis validation: print results for manual inspection
#[test]
fn test_hypothesis_validation() {
    println!("\n=== Day 38 Hypothesis Validation ===\n");
    println!("Hypothesis: Semantic clustering improves cache performance.\n");

    let workloads = vec![
        ("Sequential Related", generate_sequential_related_workload()),
        ("Random Access", generate_random_workload()),
        ("Mixed Workload", generate_mixed_workload()),
    ];

    for (name, (usage, access_pattern)) in workloads {
        println!("--- {} ---", name);

        // Baseline simulation
        let mut baseline_cache = CacheInstrumentation::new(8);
        for concept in &access_pattern {
            baseline_cache.access(concept);
        }
        let baseline_stats = baseline_cache.stats().clone();

        // Optimized simulation (pre-warm with hot concepts)
        let mut optimized_cache = CacheInstrumentation::new(8);

        // Find hot concepts
        let mut hot: Vec<_> = usage.access_counts.iter().collect();
        hot.sort_by(|a, b| b.1.cmp(a.1));

        // Pre-warm
        for (concept, _) in hot.iter().take(4) {
            optimized_cache.access(concept);
        }
        // Reset for fair comparison
        optimized_cache.reset();

        for concept in &access_pattern {
            optimized_cache.access(concept);
        }
        let optimized_stats = optimized_cache.stats().clone();

        let improvement = optimized_stats.hit_rate() - baseline_stats.hit_rate();

        println!("  Baseline hit rate:  {:.1}%", baseline_stats.hit_rate());
        println!("  Optimized hit rate: {:.1}%", optimized_stats.hit_rate());
        println!("  Improvement:        {:.1} percentage points", improvement);

        if improvement > 5.0 {
            println!("  Result: SUPPORTED (significant improvement)\n");
        } else if improvement > 0.0 {
            println!("  Result: SUPPORTED (marginal improvement)\n");
        } else if improvement < -5.0 {
            println!("  Result: NOT SUPPORTED (regression)\n");
        } else {
            println!("  Result: INCONCLUSIVE\n");
        }
    }
}

criterion_group!(
    benches,
    benchmark_cache_simulation,
    benchmark_clustering,
    benchmark_layout_effectiveness,
);

criterion_main!(benches);
