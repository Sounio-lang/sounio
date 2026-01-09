# Causal Discovery: Learning Causal Structure from Data

Causal discovery algorithms learn the causal structure (the DAG) from observational data. Unlike effect estimation (which assumes a known graph), causal discovery answers: "What causes what?"

## Why Causal Discovery?

In many domains, the causal structure is unknown:

- Which genes regulate other genes?
- What factors drive customer churn?
- How do brain regions influence each other?

Manual construction of causal graphs requires domain expertise and may miss unexpected relationships. Causal discovery algorithms learn structure from data, potentially discovering novel causal mechanisms.

## The Challenge

Learning causal structure from observational data is fundamentally limited:

```sio
// These three graphs produce identical observational distributions:
//
// (1) X -> Y      P(X,Y) = P(X) P(Y|X)
// (2) X <- Y      P(X,Y) = P(Y) P(X|Y)
// (3) X <- U -> Y P(X,Y) = sum_u P(U) P(X|U) P(Y|U)
//
// They are "Markov equivalent" - indistinguishable from correlations alone
```

Causal discovery can identify:
- The **Markov equivalence class** (set of DAGs with same conditional independencies)
- **Definite non-edges** (pairs that are definitely not directly connected)
- Some **edge orientations** (through v-structures and orientation rules)

## Constraint-Based Methods

### PC Algorithm

The PC algorithm learns causal structure by testing conditional independencies.

#### Algorithm Overview

1. Start with complete undirected graph
2. Remove edges between conditionally independent pairs
3. Orient edges using v-structures and propagation rules

```sio
use std::causal::discovery::{DiscoveredGraph, pc_skeleton, fisher_z_test}

/// PC algorithm implementation
fn pc_algorithm(
    n_vars: i64,
    correlations: [[f64]],
    n_samples: i64,
    alpha: f64
) -> DiscoveredGraph with Prob {
    // Phase 1: Skeleton discovery
    var graph = pc_skeleton(n_vars, correlations, n_samples, alpha)

    // Phase 2: Orient v-structures
    graph = orient_v_structures(graph, correlations, n_samples, alpha)

    // Phase 3: Propagate orientations
    graph = propagate_orientations(graph)

    return graph
}

/// Phase 1: Learn skeleton (undirected graph)
fn pc_skeleton(
    n_vars: i64,
    correlations: [[f64]],
    n_samples: i64,
    alpha: f64
) -> DiscoveredGraph {
    var graph = graph_new(n_vars)

    // Start with complete graph
    var i: i64 = 0
    while i < n_vars {
        var j = i + 1
        while j < n_vars {
            // Test marginal independence
            let corr = correlations[i][j]
            let p_value = fisher_z_test(corr, n_samples, 0)

            if p_value < alpha {
                // Significant correlation - keep edge with confidence
                let conf_alpha = (1.0 - p_value) * 10.0 + 1.0
                let conf_beta = p_value * 10.0 + 1.0
                graph = graph_add_edge(graph, i, j, false, beta_new(conf_alpha, conf_beta))
            }
            j = j + 1
        }
        i = i + 1
    }

    // Condition on increasing sets
    var depth: i64 = 1
    while depth < n_vars - 2 {
        graph = remove_conditionally_independent(graph, correlations, n_samples, alpha, depth)
        depth = depth + 1
    }

    return graph
}
```

#### V-Structure Orientation

V-structures (colliders) are the key to orienting edges:

```sio
// A v-structure: X -> Z <- Y
// Identified when:
//   - X and Y are not adjacent
//   - X - Z - Y forms a triple
//   - Z is in no separating set for X and Y

fn orient_v_structures(
    graph: DiscoveredGraph,
    correlations: [[f64]],
    n_samples: i64,
    alpha: f64
) -> DiscoveredGraph {
    var result = graph

    // For each unshielded triple X - Z - Y
    for triple in find_unshielded_triples(graph) {
        let x = triple.x
        let z = triple.z
        let y = triple.y

        // If Z is not in separating set for X and Y
        let sep_set = find_separating_set(graph, x, y, correlations, n_samples, alpha)

        if !contains(sep_set, z) {
            // Orient as v-structure: X -> Z <- Y
            result = orient_edge(result, x, z)
            result = orient_edge(result, y, z)
        }
    }

    return result
}
```

#### Epistemic Uncertainty in Discovery

Sounio tracks uncertainty in discovered edges:

```sio
/// Edge with epistemic confidence
struct DiscoveredEdge {
    from_node: i64,
    to_node: i64,
    is_directed: bool,
    confidence: Beta,       // Confidence that edge exists
    direction_conf: Beta,   // Confidence in orientation (if directed)
}

fn discovery_summary(graph: DiscoveredGraph) -> i64 with IO {
    print("=== Causal Discovery Results ===\n")

    var i: i64 = 0
    while i < graph.edges.len() {
        let edge = graph.edges[i]
        let exist_prob = beta_mean(edge.confidence)

        print("Edge ")
        print(edge.from_node)
        if edge.is_directed {
            print(" -> ")
        } else {
            print(" -- ")
        }
        print(edge.to_node)
        print(" (confidence: ")
        print(exist_prob * 100.0)
        print("%)\n")

        if exist_prob < 0.7 {
            print("  Warning: Low confidence edge\n")
        }

        i = i + 1
    }

    return 0
}
```

### FCI Algorithm

The FCI (Fast Causal Inference) algorithm handles latent confounders and selection bias, outputting Partial Ancestral Graphs (PAGs).

#### PAG Edge Types

```
o-o : Unknown whether edge or arrow at each endpoint
o-> : Arrow at right, unknown at left
<-> : Bidirected (latent confounder)
--> : Directed edge
```

```sio
/// Edge types in PAGs
enum PAGEdgeType {
    Circle_Circle,   // o-o
    Circle_Arrow,    // o->
    Arrow_Arrow,     // <->
    Tail_Arrow,      // -->
    Unknown,
}

/// Run FCI algorithm
fn fci_algorithm(
    n_vars: i64,
    correlations: [[f64]],
    n_samples: i64,
    alpha: f64
) -> PAGGraph with Prob {
    // Phase 1: Skeleton (same as PC)
    let skeleton = pc_skeleton(n_vars, correlations, n_samples, alpha)

    // Phase 2: Orient with possible latent variables
    var pag = skeleton_to_pag(skeleton)

    // Apply FCI orientation rules (R1-R10)
    pag = fci_orient_edges(pag, correlations, n_samples, alpha)

    return pag
}
```

## Score-Based Methods

### GES (Greedy Equivalence Search)

GES searches over equivalence classes using a scoring criterion.

```sio
/// BIC score for model comparison
fn bic_score(data: DataMatrix, graph: CausalDAG) -> f64 {
    let n = data.n_samples as f64
    let k = count_parameters(graph) as f64
    let log_lik = compute_log_likelihood(data, graph)

    // BIC = -2 * log_lik + k * log(n)
    return -2.0 * log_lik + k * ln_f64(n)
}

/// GES algorithm
fn ges_algorithm(data: DataMatrix) -> CausalDAG with Prob, Alloc {
    // Start with empty graph
    var current = empty_dag(data.n_vars)
    var current_score = bic_score(data, current)

    // Phase 1: Forward (add edges)
    var improved = true
    while improved {
        improved = false
        let best_addition = find_best_edge_addition(data, current)

        if best_addition.score < current_score {
            current = apply_addition(current, best_addition)
            current_score = best_addition.score
            improved = true
        }
    }

    // Phase 2: Backward (remove edges)
    improved = true
    while improved {
        improved = false
        let best_deletion = find_best_edge_deletion(data, current)

        if best_deletion.score < current_score {
            current = apply_deletion(current, best_deletion)
            current_score = best_deletion.score
            improved = true
        }
    }

    return current
}
```

## Handling Uncertainty in Discovered Structures

### Bootstrap Confidence

Bootstrap resampling provides confidence estimates:

```sio
/// Bootstrap causal discovery
fn bootstrap_discovery(
    data: DataMatrix,
    n_bootstrap: i64,
    alpha: f64
) -> BootstrapResult with Prob, Alloc {
    var edge_counts = init_edge_counts(data.n_vars)

    var b: i64 = 0
    while b < n_bootstrap {
        // Resample data
        let resampled = bootstrap_sample(data, b * 12345)

        // Run discovery
        let graph = pc_algorithm(
            data.n_vars,
            compute_correlations(resampled),
            resampled.n_samples,
            alpha
        )

        // Count edges
        edge_counts = update_edge_counts(edge_counts, graph)

        b = b + 1
    }

    // Convert counts to confidences
    return counts_to_confidence(edge_counts, n_bootstrap)
}

/// Result with edge-wise confidence
struct BootstrapResult {
    edges: [BootstrapEdge],
    n_bootstrap: i64,
}

struct BootstrapEdge {
    from_node: i64,
    to_node: i64,
    presence_freq: f64,      // How often edge appeared
    direction_freq: f64,      // How often this direction
    confidence: Beta,         // Posterior confidence
}
```

### Bayesian Discovery

Bayesian approaches maintain full posterior over graphs:

```sio
/// Bayesian causal discovery with MCMC
fn bayesian_discovery(
    data: DataMatrix,
    n_samples: i64,
    n_warmup: i64
) -> GraphPosterior with Prob, Alloc {
    // Prior over graphs (uniform over DAGs)
    var current = sample_random_dag(data.n_vars)
    var current_score = log_marginal_likelihood(data, current)

    var samples: [ScoredGraph] = []

    var iter: i64 = 0
    while iter < n_warmup + n_samples {
        // Propose modification (add, delete, or reverse edge)
        let proposed = propose_graph_modification(current)
        let proposed_score = log_marginal_likelihood(data, proposed)

        // MH acceptance
        let log_alpha = proposed_score - current_score
        if ln_f64(random_uniform()) < log_alpha {
            current = proposed
            current_score = proposed_score
        }

        // Save samples after warmup
        if iter >= n_warmup {
            samples = samples ++ [ScoredGraph {
                graph: current,
                log_score: current_score,
            }]
        }

        iter = iter + 1
    }

    return summarize_graph_posterior(samples)
}
```

## Practical Considerations

### Assumptions

Different algorithms make different assumptions:

| Algorithm | Assumptions |
|-----------|-------------|
| PC | Causal sufficiency (no latent confounders), faithfulness |
| FCI | Faithfulness (allows latent confounders) |
| GES | Causal sufficiency, faithfulness |
| LiNGAM | Non-Gaussian errors, linearity |

### Sample Size Requirements

Causal discovery requires substantial data:

```sio
fn check_sample_size(n_samples: i64, n_vars: i64) -> bool with IO {
    // Rule of thumb: need at least 10-20 samples per variable
    let min_samples = n_vars * 20

    if n_samples < min_samples {
        print("Warning: Sample size may be insufficient\n")
        print("Recommended: at least ")
        print(min_samples)
        print(" samples for ")
        print(n_vars)
        print(" variables\n")
        return false
    }

    // For high-dimensional settings (p > n)
    if n_vars > n_samples {
        print("Warning: High-dimensional setting (p > n)\n")
        print("Consider regularized or sparse methods\n")
        return false
    }

    return true
}
```

### Validation Strategies

```sio
/// Validate discovered structure
fn validate_discovery(
    discovered: DiscoveredGraph,
    data: DataMatrix
) -> ValidationReport with Prob, Alloc {
    // 1. Cross-validation
    let cv_score = cross_validate_structure(discovered, data, 5)

    // 2. Stability selection
    let stability = bootstrap_stability(discovered, data, 100)

    // 3. Check implied independencies
    let indep_score = test_implied_independencies(discovered, data)

    // 4. Compare with prior knowledge (if available)
    let prior_consistency = check_prior_knowledge(discovered)

    return ValidationReport {
        cv_score: cv_score,
        stability: stability,
        indep_test_score: indep_score,
        prior_consistency: prior_consistency,
    }
}
```

## Complete Example: Gene Regulatory Network

```sio
use std::causal::discovery::{
    pc_algorithm, bootstrap_discovery, discovery_summary,
    DiscoveredGraph
}

fn discover_gene_network(
    expression_data: [[f64]],
    gene_names: [[u8]],
    n_samples: i64
) -> DiscoveredGraph with Prob, Alloc {
    print("=== Gene Regulatory Network Discovery ===\n\n")

    let n_genes = gene_names.len()

    // Check sample size adequacy
    if !check_sample_size(n_samples, n_genes) {
        print("Proceeding with caution due to sample size...\n\n")
    }

    // Compute correlation matrix
    let correlations = compute_correlation_matrix(expression_data)

    // Run PC algorithm
    print("Running PC algorithm (alpha = 0.01)...\n")
    let discovered = pc_algorithm(n_genes, correlations, n_samples, 0.01)

    print("Initial discovery complete\n\n")

    // Bootstrap for confidence
    print("Running bootstrap (n=100) for confidence estimates...\n")
    let bootstrap_result = bootstrap_discovery(
        DataMatrix { data: expression_data, n_vars: n_genes, n_samples: n_samples },
        100,
        0.01
    )

    // Report results
    print("\n=== Discovered Network ===\n")

    var edge_idx: i64 = 0
    while edge_idx < discovered.edges.len() {
        let edge = discovered.edges[edge_idx]
        let boot_edge = bootstrap_result.edges[edge_idx]

        // Only report edges with sufficient confidence
        if boot_edge.presence_freq > 0.5 {
            print_byte_array(gene_names[edge.from_node])
            if edge.is_directed && boot_edge.direction_freq > 0.7 {
                print(" -> ")
            } else {
                print(" -- ")
            }
            print_byte_array(gene_names[edge.to_node])
            print(" (bootstrap freq: ")
            print(boot_edge.presence_freq * 100.0)
            print("%)\n")
        }

        edge_idx = edge_idx + 1
    }

    // Identify high-confidence edges
    print("\n=== High-Confidence Edges (>80%) ===\n")
    edge_idx = 0
    while edge_idx < bootstrap_result.edges.len() {
        let edge = bootstrap_result.edges[edge_idx]
        if edge.presence_freq > 0.8 {
            print_byte_array(gene_names[edge.from_node])
            print(" -> ")
            print_byte_array(gene_names[edge.to_node])
            print("\n")
        }
        edge_idx = edge_idx + 1
    }

    // Identify uncertain edges
    print("\n=== Uncertain Edges (50-80%) ===\n")
    edge_idx = 0
    while edge_idx < bootstrap_result.edges.len() {
        let edge = bootstrap_result.edges[edge_idx]
        if edge.presence_freq >= 0.5 && edge.presence_freq <= 0.8 {
            print_byte_array(gene_names[edge.from_node])
            print(" -- ")
            print_byte_array(gene_names[edge.to_node])
            print(" (")
            print(edge.presence_freq * 100.0)
            print("%)\n")
        }
        edge_idx = edge_idx + 1
    }

    return discovered
}

fn main() -> i32 {
    // Simulated gene expression data (5 genes, 100 samples)
    let expression_data: [[f64]] = generate_example_data(5, 100)
    let gene_names: [[u8]] = [
        [71, 69, 78, 69, 49],  // "GENE1"
        [71, 69, 78, 69, 50],  // "GENE2"
        [71, 69, 78, 69, 51],  // "GENE3"
        [71, 69, 78, 69, 52],  // "GENE4"
        [71, 69, 78, 69, 53],  // "GENE5"
    ]

    let network = discover_gene_network(expression_data, gene_names, 100)

    print("\nDiscovery complete.\n")
    return 0
}
```

## Summary

| Method | Approach | Output | Handles Latent? |
|--------|----------|--------|-----------------|
| PC | Constraint-based | CPDAG (equivalence class) | No |
| FCI | Constraint-based | PAG | Yes |
| GES | Score-based | CPDAG | No |
| LiNGAM | Functional | Full DAG | No |

Key takeaways:

1. **Causal discovery has fundamental limits** - Markov equivalent DAGs cannot be distinguished from observational data alone
2. **Uncertainty quantification is essential** - Bootstrap and Bayesian methods provide confidence in discovered edges
3. **Sample size matters** - Reliable discovery requires sufficient data
4. **Domain knowledge helps** - Combine algorithmic discovery with expert knowledge
5. **Validation is critical** - Cross-validation, stability selection, and testing implied independencies

Sounio's causal discovery module tracks epistemic uncertainty throughout the discovery process, providing honest assessments of what can and cannot be learned from the data.
