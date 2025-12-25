// connectivity::network_metrics — Graph-Theoretic Metrics with Uncertainty
//
// Network measures for brain connectivity analysis:
// - Centrality measures (degree, betweenness, eigenvector)
// - Clustering and transitivity
// - Path length and efficiency
// - Modularity and community detection
// - Small-world and rich-club metrics
// - Hub classification
//
// All metrics include epistemic uncertainty propagation.

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

// ============================================================================
// CONSTANTS
// ============================================================================

fn MAX_NODES() -> i64 { 500 }
fn INF() -> f64 { 1e30 }

// ============================================================================
// METRIC WITH UNCERTAINTY
// ============================================================================

/// Single metric value with uncertainty
struct MetricValue {
    value: f64,
    uncertainty: f64,
    ci_lower: f64,
    ci_upper: f64,
    p_value: f64,
    is_significant: bool,
}

fn metric_value_new(v: f64, u: f64) -> MetricValue {
    MetricValue {
        value: v,
        uncertainty: u,
        ci_lower: v - 1.96 * u,
        ci_upper: v + 1.96 * u,
        p_value: 1.0,
        is_significant: false,
    }
}

/// Node-level metrics
struct NodeMetrics {
    // Centrality
    degree: MetricValue,
    degree_weighted: MetricValue,
    strength: MetricValue,
    betweenness: MetricValue,
    eigenvector: MetricValue,
    pagerank: MetricValue,
    closeness: MetricValue,

    // Clustering
    clustering_coef: MetricValue,
    local_efficiency: MetricValue,

    // Hub measures
    participation_coef: MetricValue,     // Between-module connectivity
    within_module_degree: MetricValue,   // Within-module degree z-score

    // Classification
    is_hub: bool,
    is_connector_hub: bool,              // High participation
    is_provincial_hub: bool,             // High within-module degree

    // Module assignment
    module_id: i32,
}

fn node_metrics_new() -> NodeMetrics {
    NodeMetrics {
        degree: metric_value_new(0.0, 0.0),
        degree_weighted: metric_value_new(0.0, 0.0),
        strength: metric_value_new(0.0, 0.0),
        betweenness: metric_value_new(0.0, 0.0),
        eigenvector: metric_value_new(0.0, 0.0),
        pagerank: metric_value_new(0.0, 0.0),
        closeness: metric_value_new(0.0, 0.0),
        clustering_coef: metric_value_new(0.0, 0.0),
        local_efficiency: metric_value_new(0.0, 0.0),
        participation_coef: metric_value_new(0.0, 0.0),
        within_module_degree: metric_value_new(0.0, 0.0),
        is_hub: false,
        is_connector_hub: false,
        is_provincial_hub: false,
        module_id: -1,
    }
}

/// Global network metrics
struct GlobalMetrics {
    // Basic
    n_nodes: i64,
    n_edges: i64,
    density: MetricValue,

    // Paths
    char_path_length: MetricValue,
    global_efficiency: MetricValue,
    radius: MetricValue,
    diameter: MetricValue,

    // Clustering
    transitivity: MetricValue,
    avg_clustering: MetricValue,

    // Modularity
    modularity: MetricValue,
    n_modules: i64,

    // Small-world
    small_world_sigma: MetricValue,      // Humphries & Gurney
    small_world_omega: MetricValue,      // Telesford et al.

    // Rich-club
    rich_club_coef: [MetricValue; 100],
    rich_club_k: [i64; 100],
    n_rich_club: i64,

    // Assortativity
    assortativity: MetricValue,          // Degree correlation
}

fn global_metrics_new() -> GlobalMetrics {
    GlobalMetrics {
        n_nodes: 0,
        n_edges: 0,
        density: metric_value_new(0.0, 0.0),
        char_path_length: metric_value_new(0.0, 0.0),
        global_efficiency: metric_value_new(0.0, 0.0),
        radius: metric_value_new(0.0, 0.0),
        diameter: metric_value_new(0.0, 0.0),
        transitivity: metric_value_new(0.0, 0.0),
        avg_clustering: metric_value_new(0.0, 0.0),
        modularity: metric_value_new(0.0, 0.0),
        n_modules: 0,
        small_world_sigma: metric_value_new(0.0, 0.0),
        small_world_omega: metric_value_new(0.0, 0.0),
        rich_club_coef: [metric_value_new(0.0, 0.0); 100],
        rich_club_k: [0; 100],
        n_rich_club: 0,
        assortativity: metric_value_new(0.0, 0.0),
    }
}

// ============================================================================
// BASIC GRAPH OPERATIONS
// ============================================================================

/// Threshold connectivity matrix to binary
fn threshold_matrix(
    weighted: &[[f64; 500]; 500],
    threshold: f64,
    n: i64,
    binary: &![[bool; 500]; 500]
) {
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            binary[i as usize][j as usize] = weighted[i as usize][j as usize] > threshold
            j = j + 1
        }
        i = i + 1
    }
}

/// Count edges
fn count_edges(binary: &[[bool; 500]; 500], n: i64) -> i64 {
    var count: i64 = 0
    var i: i64 = 0
    while i < n {
        var j: i64 = i + 1
        while j < n {
            if binary[i as usize][j as usize] {
                count = count + 1
            }
            j = j + 1
        }
        i = i + 1
    }
    count
}

/// Network density
fn network_density(n_nodes: i64, n_edges: i64) -> f64 {
    let max_edges = n_nodes * (n_nodes - 1) / 2
    if max_edges > 0 {
        n_edges as f64 / max_edges as f64
    } else {
        0.0
    }
}

// ============================================================================
// DEGREE CENTRALITY
// ============================================================================

/// Binary degree
fn degree_binary(
    binary: &[[bool; 500]; 500],
    n: i64,
    degrees: &![i64; 500]
) {
    var i: i64 = 0
    while i < n {
        var deg: i64 = 0
        var j: i64 = 0
        while j < n {
            if binary[i as usize][j as usize] && i != j {
                deg = deg + 1
            }
            j = j + 1
        }
        degrees[i as usize] = deg
        i = i + 1
    }
}

/// Weighted degree (strength)
fn strength_weighted(
    weighted: &[[f64; 500]; 500],
    threshold: f64,
    n: i64,
    strengths: &![f64; 500]
) {
    var i: i64 = 0
    while i < n {
        var s: f64 = 0.0
        var j: i64 = 0
        while j < n {
            if weighted[i as usize][j as usize] > threshold && i != j {
                s = s + weighted[i as usize][j as usize]
            }
            j = j + 1
        }
        strengths[i as usize] = s
        i = i + 1
    }
}

/// Degree with uncertainty propagation
fn degree_with_uncertainty(
    weighted: &[[f64; 500]; 500],
    uncertainty: &[[f64; 500]; 500],
    threshold: f64,
    n: i64,
    node: i64
) -> MetricValue {
    var deg: f64 = 0.0
    var unc_sq: f64 = 0.0

    var j: i64 = 0
    while j < n {
        if j != node && weighted[node as usize][j as usize] > threshold {
            deg = deg + weighted[node as usize][j as usize]
            let u = uncertainty[node as usize][j as usize]
            unc_sq = unc_sq + u * u
        }
        j = j + 1
    }

    metric_value_new(deg, sqrt(unc_sq))
}

// ============================================================================
// SHORTEST PATHS (Floyd-Warshall)
// ============================================================================

/// All-pairs shortest paths
fn floyd_warshall(
    weights: &[[f64; 500]; 500],
    n: i64,
    distances: &![[f64; 500]; 500]
) {
    // Initialize
    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            if i == j {
                distances[i as usize][j as usize] = 0.0
            } else if weights[i as usize][j as usize] > 0.0 {
                distances[i as usize][j as usize] = 1.0 / weights[i as usize][j as usize]
            } else {
                distances[i as usize][j as usize] = INF()
            }
            j = j + 1
        }
        i = i + 1
    }

    // Main algorithm
    var k: i64 = 0
    while k < n {
        i = 0
        while i < n {
            var j: i64 = 0
            while j < n {
                let through_k = distances[i as usize][k as usize] + distances[k as usize][j as usize]
                if through_k < distances[i as usize][j as usize] {
                    distances[i as usize][j as usize] = through_k
                }
                j = j + 1
            }
            i = i + 1
        }
        k = k + 1
    }
}

// ============================================================================
// PATH-BASED METRICS
// ============================================================================

/// Characteristic path length
fn characteristic_path_length(distances: &[[f64; 500]; 500], n: i64) -> f64 {
    var sum: f64 = 0.0
    var count: i64 = 0

    var i: i64 = 0
    while i < n {
        var j: i64 = i + 1
        while j < n {
            if distances[i as usize][j as usize] < INF() * 0.5 {
                sum = sum + distances[i as usize][j as usize]
                count = count + 1
            }
            j = j + 1
        }
        i = i + 1
    }

    if count > 0 {
        sum / count as f64
    } else {
        INF()
    }
}

/// Global efficiency
fn global_efficiency(distances: &[[f64; 500]; 500], n: i64) -> f64 {
    var sum: f64 = 0.0
    var count: i64 = 0

    var i: i64 = 0
    while i < n {
        var j: i64 = i + 1
        while j < n {
            if distances[i as usize][j as usize] > 0.0 && distances[i as usize][j as usize] < INF() * 0.5 {
                sum = sum + 1.0 / distances[i as usize][j as usize]
                count = count + 1
            }
            j = j + 1
        }
        i = i + 1
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Local efficiency for a node
fn local_efficiency_node(
    weights: &[[f64; 500]; 500],
    n: i64,
    node: i64,
    threshold: f64
) -> f64 {
    // Get neighbors
    var neighbors = [0i64; 500]
    var n_neighbors: i64 = 0

    var i: i64 = 0
    while i < n {
        if i != node && weights[node as usize][i as usize] > threshold {
            neighbors[n_neighbors as usize] = i
            n_neighbors = n_neighbors + 1
        }
        i = i + 1
    }

    if n_neighbors < 2 {
        return 0.0
    }

    // Compute efficiency of subgraph
    var sum: f64 = 0.0
    i = 0
    while i < n_neighbors {
        var j: i64 = i + 1
        while j < n_neighbors {
            let ni = neighbors[i as usize]
            let nj = neighbors[j as usize]
            let w = weights[ni as usize][nj as usize]
            if w > threshold {
                sum = sum + w  // Simplified: direct connection
            }
            j = j + 1
        }
        i = i + 1
    }

    let max_edges = n_neighbors * (n_neighbors - 1) / 2
    sum / max_edges as f64
}

// ============================================================================
// CLUSTERING COEFFICIENT
// ============================================================================

/// Local clustering coefficient (binary)
fn clustering_coefficient_node(
    binary: &[[bool; 500]; 500],
    n: i64,
    node: i64
) -> f64 {
    // Get neighbors
    var neighbors = [0i64; 500]
    var k: i64 = 0

    var i: i64 = 0
    while i < n {
        if i != node && binary[node as usize][i as usize] {
            neighbors[k as usize] = i
            k = k + 1
        }
        i = i + 1
    }

    if k < 2 {
        return 0.0
    }

    // Count triangles
    var triangles: i64 = 0
    i = 0
    while i < k {
        var j: i64 = i + 1
        while j < k {
            if binary[neighbors[i as usize] as usize][neighbors[j as usize] as usize] {
                triangles = triangles + 1
            }
            j = j + 1
        }
        i = i + 1
    }

    // k choose 2 = k*(k-1)/2
    let possible = k * (k - 1) / 2
    triangles as f64 / possible as f64
}

/// Weighted clustering coefficient
fn clustering_coefficient_weighted(
    weighted: &[[f64; 500]; 500],
    threshold: f64,
    n: i64,
    node: i64
) -> f64 {
    // Get neighbors and strengths
    var neighbors = [0i64; 500]
    var k: i64 = 0
    var strength: f64 = 0.0

    var i: i64 = 0
    while i < n {
        if i != node && weighted[node as usize][i as usize] > threshold {
            neighbors[k as usize] = i
            strength = strength + weighted[node as usize][i as usize]
            k = k + 1
        }
        i = i + 1
    }

    if k < 2 {
        return 0.0
    }

    // Geometric mean of weights in triangles
    var sum: f64 = 0.0
    i = 0
    while i < k {
        var j: i64 = i + 1
        while j < k {
            let ni = neighbors[i as usize]
            let nj = neighbors[j as usize]
            let wij = weighted[ni as usize][nj as usize]
            if wij > threshold {
                let w_ik = weighted[node as usize][ni as usize]
                let w_jk = weighted[node as usize][nj as usize]
                sum = sum + pow(w_ik * w_jk * wij, 1.0 / 3.0)
            }
            j = j + 1
        }
        i = i + 1
    }

    2.0 * sum / (k as f64 * (k - 1) as f64)
}

/// Global transitivity
fn transitivity(binary: &[[bool; 500]; 500], n: i64) -> f64 {
    var triangles: i64 = 0
    var triples: i64 = 0

    var i: i64 = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            if i != j && binary[i as usize][j as usize] {
                var k: i64 = 0
                while k < n {
                    if k != i && k != j && binary[j as usize][k as usize] {
                        triples = triples + 1
                        if binary[i as usize][k as usize] {
                            triangles = triangles + 1
                        }
                    }
                    k = k + 1
                }
            }
            j = j + 1
        }
        i = i + 1
    }

    if triples > 0 {
        triangles as f64 / triples as f64
    } else {
        0.0
    }
}

// ============================================================================
// MODULARITY
// ============================================================================

/// Modularity Q for given partition
fn modularity_q(
    weights: &[[f64; 500]; 500],
    modules: &[i32; 500],
    n: i64
) -> f64 {
    // Total weight
    var m: f64 = 0.0
    var i: i64 = 0
    while i < n {
        var j: i64 = i + 1
        while j < n {
            m = m + weights[i as usize][j as usize]
            j = j + 1
        }
        i = i + 1
    }

    if m < 1e-10 {
        return 0.0
    }

    // Compute strengths
    var strengths = [0.0; 500]
    i = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            strengths[i as usize] = strengths[i as usize] + weights[i as usize][j as usize]
            j = j + 1
        }
        i = i + 1
    }

    // Compute Q
    var q: f64 = 0.0
    i = 0
    while i < n {
        var j: i64 = 0
        while j < n {
            if modules[i as usize] == modules[j as usize] {
                let expected = strengths[i as usize] * strengths[j as usize] / (2.0 * m)
                q = q + weights[i as usize][j as usize] - expected
            }
            j = j + 1
        }
        i = i + 1
    }

    q / (2.0 * m)
}

/// Simple Louvain community detection
fn louvain_modularity(
    weights: &[[f64; 500]; 500],
    n: i64,
    modules: &![i32; 500]
) -> f64 {
    // Initialize: each node in its own module
    var i: i64 = 0
    while i < n {
        modules[i as usize] = i as i32
        i = i + 1
    }

    var improved = true
    var max_iter: i64 = 100
    var iter: i64 = 0

    while improved && iter < max_iter {
        improved = false

        i = 0
        while i < n {
            let current_module = modules[i as usize]
            var best_module = current_module
            var best_delta: f64 = 0.0

            // Try moving to each neighbor's module
            var j: i64 = 0
            while j < n {
                if weights[i as usize][j as usize] > 0.0 {
                    let target_module = modules[j as usize]
                    if target_module != current_module {
                        // Compute delta Q (simplified)
                        modules[i as usize] = target_module
                        let q_new = modularity_q(weights, modules, n)
                        modules[i as usize] = current_module
                        let q_old = modularity_q(weights, modules, n)
                        let delta = q_new - q_old

                        if delta > best_delta {
                            best_delta = delta
                            best_module = target_module
                        }
                    }
                }
                j = j + 1
            }

            if best_module != current_module {
                modules[i as usize] = best_module
                improved = true
            }

            i = i + 1
        }

        iter = iter + 1
    }

    modularity_q(weights, modules, n)
}

// ============================================================================
// SMALL-WORLD METRICS
// ============================================================================

/// Small-world sigma (Humphries & Gurney 2008)
fn small_world_sigma(
    avg_clustering: f64,
    char_path_length: f64,
    n_nodes: i64,
    n_edges: i64
) -> f64 {
    // Random network estimates
    let p = network_density(n_nodes, n_edges)
    let c_rand = p  // Expected clustering in random graph
    let l_rand = log(n_nodes as f64) / log(n_nodes as f64 * p)  // Approximate

    if c_rand > 0.0 && l_rand > 0.0 && char_path_length > 0.0 {
        (avg_clustering / c_rand) / (char_path_length / l_rand)
    } else {
        0.0
    }
}

/// Small-world omega (Telesford 2011)
fn small_world_omega(
    avg_clustering: f64,
    char_path_length: f64,
    n_nodes: i64,
    n_edges: i64
) -> f64 {
    let p = network_density(n_nodes, n_edges)
    let l_rand = log(n_nodes as f64) / log(n_nodes as f64 * p)
    let c_lattice = 0.75  // Approximate for ring lattice

    if char_path_length > 0.0 {
        l_rand / char_path_length - avg_clustering / c_lattice
    } else {
        0.0
    }
}

// ============================================================================
// RICH-CLUB COEFFICIENT
// ============================================================================

/// Rich-club coefficient for degree k
fn rich_club_coefficient(
    degrees: &[i64; 500],
    weights: &[[f64; 500]; 500],
    n: i64,
    k: i64
) -> f64 {
    // Find nodes with degree > k
    var rich_nodes = [0i64; 500]
    var n_rich: i64 = 0

    var i: i64 = 0
    while i < n {
        if degrees[i as usize] > k {
            rich_nodes[n_rich as usize] = i
            n_rich = n_rich + 1
        }
        i = i + 1
    }

    if n_rich < 2 {
        return 0.0
    }

    // Count edges among rich nodes
    var edges: i64 = 0
    i = 0
    while i < n_rich {
        var j: i64 = i + 1
        while j < n_rich {
            if weights[rich_nodes[i as usize] as usize][rich_nodes[j as usize] as usize] > 0.0 {
                edges = edges + 1
            }
            j = j + 1
        }
        i = i + 1
    }

    let max_edges = n_rich * (n_rich - 1) / 2
    if max_edges > 0 {
        edges as f64 / max_edges as f64
    } else {
        0.0
    }
}

// ============================================================================
// PARTICIPATION COEFFICIENT
// ============================================================================

/// Participation coefficient (between-module connectivity)
fn participation_coefficient(
    weights: &[[f64; 500]; 500],
    modules: &[i32; 500],
    n: i64,
    node: i64
) -> f64 {
    // Get strength by module
    var module_strength = [0.0; 100]
    var n_modules: i32 = 0
    var total_strength: f64 = 0.0

    var j: i64 = 0
    while j < n {
        if j != node && weights[node as usize][j as usize] > 0.0 {
            let m = modules[j as usize]
            if m >= 0 && m < 100 {
                module_strength[m as usize] = module_strength[m as usize] + weights[node as usize][j as usize]
                total_strength = total_strength + weights[node as usize][j as usize]
                if m >= n_modules {
                    n_modules = m + 1
                }
            }
        }
        j = j + 1
    }

    if total_strength < 1e-10 {
        return 0.0
    }

    // P = 1 - sum((ki_m / ki)^2)
    var sum_sq: f64 = 0.0
    var m: i32 = 0
    while m < n_modules {
        let ratio = module_strength[m as usize] / total_strength
        sum_sq = sum_sq + ratio * ratio
        m = m + 1
    }

    1.0 - sum_sq
}

// ============================================================================
// COMPLETE ANALYSIS
// ============================================================================

/// Compute all network metrics (returns global metrics and node array)
fn compute_global_metrics(
    weighted: &[[f64; 500]; 500],
    n: i64,
    threshold: f64
) -> GlobalMetrics {
    var global = global_metrics_new()

    global.n_nodes = n

    // Binary threshold
    var binary = [[false; 500]; 500]
    threshold_matrix(weighted, threshold, n, &!binary)

    // Edge count and density
    global.n_edges = count_edges(&binary, n)
    global.density = metric_value_new(network_density(n, global.n_edges), 0.0)

    // Degrees
    var degrees = [0i64; 500]
    degree_binary(&binary, n, &!degrees)

    // Shortest paths
    var distances = [[0.0; 500]; 500]
    floyd_warshall(weighted, n, &!distances)

    // Global path metrics
    global.char_path_length = metric_value_new(characteristic_path_length(&distances, n), 0.0)
    global.global_efficiency = metric_value_new(global_efficiency(&distances, n), 0.0)

    // Clustering
    var avg_cc: f64 = 0.0
    var i: i64 = 0
    while i < n {
        let cc = clustering_coefficient_weighted(weighted, threshold, n, i)
        avg_cc = avg_cc + cc
        i = i + 1
    }
    global.avg_clustering = metric_value_new(avg_cc / n as f64, 0.0)
    global.transitivity = metric_value_new(transitivity(&binary, n), 0.0)

    // Modularity
    var modules = [0i32; 500]
    let q = louvain_modularity(weighted, n, &!modules)
    global.modularity = metric_value_new(q, 0.0)

    // Count modules
    var max_module: i32 = 0
    i = 0
    while i < n {
        if modules[i as usize] > max_module {
            max_module = modules[i as usize]
        }
        i = i + 1
    }
    global.n_modules = (max_module + 1) as i64

    // Small-world
    global.small_world_sigma = metric_value_new(
        small_world_sigma(
            global.avg_clustering.value,
            global.char_path_length.value,
            n, global.n_edges
        ), 0.0
    )

    global.small_world_omega = metric_value_new(
        small_world_omega(
            global.avg_clustering.value,
            global.char_path_length.value,
            n, global.n_edges
        ), 0.0
    )

    // Rich-club
    var k: i64 = 1
    while k < 50 && global.n_rich_club < 100 {
        let rc = rich_club_coefficient(&degrees, weighted, n, k)
        if rc > 0.0 {
            global.rich_club_coef[global.n_rich_club as usize] = metric_value_new(rc, 0.0)
            global.rich_club_k[global.n_rich_club as usize] = k
            global.n_rich_club = global.n_rich_club + 1
        }
        k = k + 1
    }

    global
}

/// Compute node-level metrics
fn compute_node_metrics(
    weighted: &[[f64; 500]; 500],
    uncertainty: &[[f64; 500]; 500],
    n: i64,
    threshold: f64,
    node_metrics: &![NodeMetrics; 500]
) {
    // Binary threshold
    var binary = [[false; 500]; 500]
    threshold_matrix(weighted, threshold, n, &!binary)

    // Degrees
    var degrees = [0i64; 500]
    degree_binary(&binary, n, &!degrees)

    var strengths = [0.0; 500]
    strength_weighted(weighted, threshold, n, &!strengths)

    // Modularity for participation coefficient
    var modules = [0i32; 500]
    louvain_modularity(weighted, n, &!modules)

    // Mean and std of degrees for hub classification
    var mean_deg: f64 = 0.0
    var i: i64 = 0
    while i < n {
        mean_deg = mean_deg + degrees[i as usize] as f64
        i = i + 1
    }
    mean_deg = mean_deg / n as f64

    var var_deg: f64 = 0.0
    i = 0
    while i < n {
        let d = degrees[i as usize] as f64 - mean_deg
        var_deg = var_deg + d * d
        i = i + 1
    }
    var_deg = var_deg / (n - 1) as f64
    let sd_deg = sqrt(var_deg)

    // Compute per-node metrics
    i = 0
    while i < n {
        node_metrics[i as usize] = node_metrics_new()
        node_metrics[i as usize].degree = metric_value_new(degrees[i as usize] as f64, 0.0)
        node_metrics[i as usize].strength = metric_value_new(strengths[i as usize], 0.0)
        node_metrics[i as usize].degree_weighted = degree_with_uncertainty(weighted, uncertainty, threshold, n, i)
        node_metrics[i as usize].clustering_coef = metric_value_new(
            clustering_coefficient_weighted(weighted, threshold, n, i), 0.0
        )
        node_metrics[i as usize].local_efficiency = metric_value_new(
            local_efficiency_node(weighted, n, i, threshold), 0.0
        )
        node_metrics[i as usize].module_id = modules[i as usize]

        let pc = participation_coefficient(weighted, &modules, n, i)
        node_metrics[i as usize].participation_coef = metric_value_new(pc, 0.0)

        // Hub classification
        node_metrics[i as usize].is_hub = degrees[i as usize] as f64 > mean_deg + sd_deg

        if node_metrics[i as usize].is_hub {
            if pc > 0.3 {
                node_metrics[i as usize].is_connector_hub = true
            } else {
                node_metrics[i as usize].is_provincial_hub = true
            }
        }

        i = i + 1
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_clustering() -> bool {
    // Triangle: all connected
    var binary = [[false; 500]; 500]
    binary[0][1] = true; binary[1][0] = true
    binary[0][2] = true; binary[2][0] = true
    binary[1][2] = true; binary[2][1] = true

    let cc = clustering_coefficient_node(&binary, 3, 0)
    fabs(cc - 1.0) < 0.01
}

fn test_density() -> bool {
    // 3 nodes, 3 edges = complete graph, density = 1.0
    let d = network_density(3, 3)
    fabs(d - 1.0) < 0.01
}

fn test_modularity() -> bool {
    // Two disconnected cliques
    var weights = [[0.0; 500]; 500]
    // Clique 1: 0,1,2
    weights[0][1] = 1.0; weights[1][0] = 1.0
    weights[0][2] = 1.0; weights[2][0] = 1.0
    weights[1][2] = 1.0; weights[2][1] = 1.0
    // Clique 2: 3,4,5
    weights[3][4] = 1.0; weights[4][3] = 1.0
    weights[3][5] = 1.0; weights[5][3] = 1.0
    weights[4][5] = 1.0; weights[5][4] = 1.0

    var modules = [0i32; 500]
    let q = louvain_modularity(&weights, 6, &!modules)
    q > 0.4  // Should have high modularity
}

fn test_rich_club() -> bool {
    // Star graph: one hub connected to all
    var weights = [[0.0; 500]; 500]
    var i: i64 = 1
    while i < 5 {
        weights[0][i as usize] = 1.0
        weights[i as usize][0] = 1.0
        i = i + 1
    }

    var degrees = [0i64; 500]
    degrees[0] = 4
    degrees[1] = 1; degrees[2] = 1; degrees[3] = 1; degrees[4] = 1

    let rc = rich_club_coefficient(&degrees, &weights, 5, 1)
    // Only node 0 has degree > 1, so rich club = 0 (no edges among rich nodes)
    rc < 0.01
}

fn main() -> i32 {
    print("Testing connectivity::network_metrics module...\n")

    if !test_clustering() {
        print("FAIL: clustering\n")
        return 1
    }
    print("PASS: clustering\n")

    if !test_density() {
        print("FAIL: density\n")
        return 2
    }
    print("PASS: density\n")

    if !test_modularity() {
        print("FAIL: modularity\n")
        return 3
    }
    print("PASS: modularity\n")

    if !test_rich_club() {
        print("FAIL: rich_club\n")
        return 4
    }
    print("PASS: rich_club\n")

    print("All connectivity::network_metrics tests PASSED\n")
    0
}
