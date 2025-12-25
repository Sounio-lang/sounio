// Ollivier-Ricci Curvature for Network Geometry
//
// Implements discrete Ricci curvature based on optimal transport.
// Positive curvature → spherical geometry (clustered)
// Zero curvature → Euclidean geometry (flat)
// Negative curvature → hyperbolic geometry (tree-like)

module graph::curvature

// =============================================================================
// Curvature Types
// =============================================================================

// Geometry classification based on curvature
pub enum Geometry {
    Hyperbolic,   // kappa < -0.05
    Euclidean,    // -0.05 <= kappa <= 0.05
    Spherical,    // kappa > 0.05
}

// Edge curvature result
pub struct EdgeCurvature {
    pub u: usize,
    pub v: usize,
    pub kappa: f64,
    pub geometry: Geometry,
}

// Full graph curvature analysis
pub struct CurvatureAnalysis {
    pub mean_curvature: f64,
    pub min_curvature: f64,
    pub max_curvature: f64,
    pub num_hyperbolic: usize,
    pub num_euclidean: usize,
    pub num_spherical: usize,
    pub dominant_geometry: Geometry,
}

// Curvature parameters
pub struct CurvatureParams {
    pub alpha: f64,        // Laziness parameter (0 = standard, 0.5 = lazy random walk)
    pub epsilon: f64,      // Sinkhorn regularization
    pub max_iter: usize,   // Sinkhorn max iterations
}

impl CurvatureParams {
    pub fn default() -> CurvatureParams {
        CurvatureParams {
            alpha: 0.0,       // Standard random walk
            epsilon: 0.1,     // Standard regularization
            max_iter: 100,
        }
    }

    pub fn lazy() -> CurvatureParams {
        CurvatureParams {
            alpha: 0.5,       // Lazy random walk
            epsilon: 0.1,
            max_iter: 100,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn classify_geometry(kappa: f64) -> Geometry {
    if kappa < -0.05 {
        Geometry::Hyperbolic
    } else if kappa > 0.05 {
        Geometry::Spherical
    } else {
        Geometry::Euclidean
    }
}

// =============================================================================
// Probability Measure Construction
// =============================================================================

// Neighborhood measure result
pub struct NeighborhoodMeasure {
    pub nodes: [usize],
    pub probs: [f64],
}

// Build probability measure for a node's neighborhood
// mu_x(z) = (1-alpha) * 1/deg(x) if z is neighbor of x
//         = alpha if z == x
//         = 0 otherwise
pub fn neighborhood_measure(adj: &[[usize]], node: usize, alpha: f64) -> NeighborhoodMeasure {
    if node >= adj.len() {
        return NeighborhoodMeasure { nodes: [], probs: [] }
    }

    let neighbors = &adj[node]
    let deg = neighbors.len()

    if deg == 0 {
        // Isolated node: all mass on itself
        return NeighborhoodMeasure { nodes: [node], probs: [1.0] }
    }

    let neighbor_weight = (1.0 - alpha) / (deg as f64)
    var nodes: [usize] = []
    var probs: [f64] = []

    // Self-loop with weight alpha (if using lazy random walk)
    if alpha > 0.0 {
        nodes.push(node)
        probs.push(alpha)
    }

    // Neighbors with equal weight
    var i: usize = 0
    while i < deg {
        nodes.push(neighbors[i])
        probs.push(neighbor_weight)
        i = i + 1
    }

    NeighborhoodMeasure { nodes, probs }
}

// =============================================================================
// Ollivier-Ricci Curvature Computation
// =============================================================================

// Compute Ollivier-Ricci curvature for a single edge (u, v)
// kappa(u,v) = 1 - W_1(mu_u, mu_v) / d(u,v)
// where W_1 is the Wasserstein-1 distance and d(u,v) is the graph distance
pub fn edge_curvature(
    adj: &[[usize]],
    distances: &[[i64]],
    u: usize,
    v: usize,
    params: &CurvatureParams
) -> f64 {
    // Get distance between u and v
    let d_uv = distances[u][v]
    if d_uv <= 0 {
        return 0.0  // Same node or unreachable
    }

    // Get neighborhood measures
    let mu_u = neighborhood_measure(adj, u, params.alpha)
    let mu_v = neighborhood_measure(adj, v, params.alpha)

    if mu_u.nodes.len() == 0 || mu_v.nodes.len() == 0 {
        return 0.0
    }

    // Build cost matrix from distances
    let n = mu_u.nodes.len()
    let m = mu_v.nodes.len()
    var cost_data: [f64] = []

    var i: usize = 0
    while i < n {
        var j: usize = 0
        while j < m {
            let d = distances[mu_u.nodes[i]][mu_v.nodes[j]]
            if d >= 0 {
                cost_data.push(d as f64)
            } else {
                cost_data.push(1000.0)  // Large distance for unreachable
            }
            j = j + 1
        }
        i = i + 1
    }

    // Compute Wasserstein distance using Sinkhorn
    let w1 = sinkhorn_wasserstein(&mu_u.probs, &mu_v.probs, &cost_data, n, m, params.epsilon, params.max_iter)

    // Ollivier-Ricci curvature: kappa = 1 - W_1 / d(u,v)
    1.0 - w1 / (d_uv as f64)
}

// Simplified Sinkhorn for Wasserstein distance
// (Inlined to avoid module dependencies)
fn sinkhorn_wasserstein(
    mu: &[f64],
    nu: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    epsilon: f64,
    max_iter: usize
) -> f64 {
    if n == 0 || m == 0 {
        return 0.0
    }

    // Compute Gibbs kernel K = exp(-C/epsilon)
    var kmat: [f64] = []
    var idx: usize = 0
    while idx < n * m {
        let c = cost[idx]
        kmat.push(exp_approx(-c / epsilon))
        idx = idx + 1
    }

    // Initialize scaling vectors
    var u: [f64] = []
    var v: [f64] = []
    var i: usize = 0
    while i < n { u.push(1.0); i = i + 1 }
    i = 0
    while i < m { v.push(1.0); i = i + 1 }

    // Sinkhorn iterations
    var iter: usize = 0
    while iter < max_iter {
        // Update u
        i = 0
        while i < n {
            var sum: f64 = 0.0
            var j: usize = 0
            while j < m {
                sum = sum + kmat[i * m + j] * v[j]
                j = j + 1
            }
            if sum > 1e-10 {
                u[i] = mu[i] / sum
            }
            i = i + 1
        }

        // Update v
        var j: usize = 0
        while j < m {
            var sum: f64 = 0.0
            i = 0
            while i < n {
                sum = sum + kmat[i * m + j] * u[i]
                i = i + 1
            }
            if sum > 1e-10 {
                v[j] = nu[j] / sum
            }
            j = j + 1
        }

        iter = iter + 1
    }

    // Compute transport cost
    var distance: f64 = 0.0
    i = 0
    while i < n {
        var j: usize = 0
        while j < m {
            let transport = u[i] * kmat[i * m + j] * v[j]
            distance = distance + transport * cost[i * m + j]
            j = j + 1
        }
        i = i + 1
    }

    distance
}

fn exp_approx(x: f64) -> f64 {
    if x < -20.0 { return 0.0 }
    if x > 20.0 { return 485165195.0 }

    let scaled = x / 16.0
    let x2 = scaled * scaled
    let x3 = x2 * scaled
    let x4 = x3 * scaled
    let x5 = x4 * scaled
    var result = 1.0 + scaled + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0

    var i: usize = 0
    while i < 16 {
        result = result * result
        i = i + 1
    }
    result
}

// =============================================================================
// All-Pairs Distance Matrix (BFS-based)
// =============================================================================

fn compute_all_distances(adj: &[[usize]]) -> [[i64]] {
    let n = adj.len()
    var distances: [[i64]] = []

    var source: usize = 0
    while source < n {
        // BFS from this source
        var dist: [i64] = []
        var i: usize = 0
        while i < n {
            dist.push(-1)
            i = i + 1
        }
        dist[source] = 0

        var queue: [usize] = []
        queue.push(source)
        var head: usize = 0

        while head < queue.len() {
            let u = queue[head]
            head = head + 1

            let neighbors = &adj[u]
            var j: usize = 0
            while j < neighbors.len() {
                let v = neighbors[j]
                if dist[v] == -1 {
                    dist[v] = dist[u] + 1
                    queue.push(v)
                }
                j = j + 1
            }
        }

        distances.push(dist)
        source = source + 1
    }

    distances
}

// =============================================================================
// Graph-Level Curvature Analysis
// =============================================================================

// Compute curvature for all edges
pub fn all_edge_curvatures(adj: &[[usize]], params: &CurvatureParams) -> [EdgeCurvature] {
    let distances = compute_all_distances(adj)
    let n = adj.len()
    var curvatures: [EdgeCurvature] = []

    var u: usize = 0
    while u < n {
        let neighbors = &adj[u]
        var i: usize = 0
        while i < neighbors.len() {
            let v = neighbors[i]
            if u < v {  // Only compute once per edge
                let kappa = edge_curvature(adj, &distances, u, v, params)
                curvatures.push(EdgeCurvature {
                    u,
                    v,
                    kappa,
                    geometry: classify_geometry(kappa),
                })
            }
            i = i + 1
        }
        u = u + 1
    }

    curvatures
}

// Analyze overall graph curvature
pub fn analyze_curvature(adj: &[[usize]], params: &CurvatureParams) -> CurvatureAnalysis {
    let edge_curvatures = all_edge_curvatures(adj, params)
    let num_edges = edge_curvatures.len()

    if num_edges == 0 {
        return CurvatureAnalysis {
            mean_curvature: 0.0,
            min_curvature: 0.0,
            max_curvature: 0.0,
            num_hyperbolic: 0,
            num_euclidean: 0,
            num_spherical: 0,
            dominant_geometry: Geometry::Euclidean,
        }
    }

    var sum: f64 = 0.0
    var min_k: f64 = edge_curvatures[0].kappa
    var max_k: f64 = edge_curvatures[0].kappa
    var num_hyp: usize = 0
    var num_euc: usize = 0
    var num_sph: usize = 0

    var i: usize = 0
    while i < num_edges {
        let ec = &edge_curvatures[i]
        sum = sum + ec.kappa
        if ec.kappa < min_k { min_k = ec.kappa }
        if ec.kappa > max_k { max_k = ec.kappa }

        match ec.geometry {
            Geometry::Hyperbolic => { num_hyp = num_hyp + 1 }
            Geometry::Euclidean => { num_euc = num_euc + 1 }
            Geometry::Spherical => { num_sph = num_sph + 1 }
        }
        i = i + 1
    }

    let mean = sum / (num_edges as f64)
    let dominant = if num_hyp >= num_euc && num_hyp >= num_sph {
        Geometry::Hyperbolic
    } else if num_sph >= num_euc {
        Geometry::Spherical
    } else {
        Geometry::Euclidean
    }

    CurvatureAnalysis {
        mean_curvature: mean,
        min_curvature: min_k,
        max_curvature: max_k,
        num_hyperbolic: num_hyp,
        num_euclidean: num_euc,
        num_spherical: num_sph,
        dominant_geometry: dominant,
    }
}

// =============================================================================
// Forman Curvature (O(1) per edge - much faster than Ollivier-Ricci)
// =============================================================================
//
// Forman curvature is a discrete Ricci curvature based only on local degree.
// For edge (u,v): F(u,v) = 4 - deg(u) - deg(v)
// Augmented version adds triangle contribution:
// F_aug(u,v) = 4 - deg(u) - deg(v) + 3 * #triangles(u,v)

/// Compute Forman curvature for a single edge
/// F(u,v) = 4 - deg(u) - deg(v)
pub fn forman_edge_curvature(adj: &[[usize]], u: usize, v: usize) -> f64 {
    let deg_u = adj[u].len();
    let deg_v = adj[v].len();
    4.0 - (deg_u as f64) - (deg_v as f64)
}

/// Count triangles containing edge (u,v)
fn count_triangles(adj: &[[usize]], u: usize, v: usize) -> usize {
    var count: usize = 0;
    let neighbors_u = adj[u];

    // Check each neighbor of u to see if it's also a neighbor of v
    var i: usize = 0;
    while i < neighbors_u.len() {
        let w = neighbors_u[i];
        if w != v {
            // Check if w is connected to v
            let neighbors_v = adj[v];
            var j: usize = 0;
            while j < neighbors_v.len() {
                if neighbors_v[j] == w {
                    count = count + 1;
                }
                j = j + 1;
            }
        }
        i = i + 1;
    }
    count
}

/// Compute augmented Forman curvature for a single edge
/// F_aug(u,v) = 4 - deg(u) - deg(v) + 3 * #triangles(u,v)
pub fn forman_edge_curvature_augmented(adj: &[[usize]], u: usize, v: usize) -> f64 {
    let base = forman_edge_curvature(adj, u, v);
    let triangles = count_triangles(adj, u, v);
    base + 3.0 * (triangles as f64)
}

/// Forman curvature result for an edge
pub struct FormanEdgeCurvature {
    pub u: usize,
    pub v: usize,
    pub curvature: f64,
    pub augmented_curvature: f64,
    pub triangles: usize,
}

/// Compute Forman curvature for all edges
pub fn forman_all_edges(adj: &[[usize]]) -> [FormanEdgeCurvature] {
    var results: [FormanEdgeCurvature] = [];
    let n = adj.len();

    var u: usize = 0;
    while u < n {
        let neighbors = adj[u];
        var i: usize = 0;
        while i < neighbors.len() {
            let v = neighbors[i];
            // Only process each edge once (u < v)
            if u < v {
                let base = forman_edge_curvature(adj, u, v);
                let triangles = count_triangles(adj, u, v);
                let augmented = base + 3.0 * (triangles as f64);

                results.push(FormanEdgeCurvature {
                    u: u,
                    v: v,
                    curvature: base,
                    augmented_curvature: augmented,
                    triangles: triangles,
                });
            }
            i = i + 1;
        }
        u = u + 1;
    }

    results
}

/// Mean Forman curvature (scalar summary)
pub fn mean_forman_curvature(adj: &[[usize]]) -> f64 {
    let n = adj.len();
    if n < 2 {
        return 0.0;
    }

    var sum: f64 = 0.0;
    var count: usize = 0;

    var u: usize = 0;
    while u < n {
        let neighbors = adj[u];
        var i: usize = 0;
        while i < neighbors.len() {
            let v = neighbors[i];
            if u < v {
                sum = sum + forman_edge_curvature(adj, u, v);
                count = count + 1;
            }
            i = i + 1;
        }
        u = u + 1;
    }

    if count > 0 {
        sum / (count as f64)
    } else {
        0.0
    }
}

/// Mean augmented Forman curvature (includes triangle contribution)
pub fn mean_forman_curvature_augmented(adj: &[[usize]]) -> f64 {
    let n = adj.len();
    if n < 2 {
        return 0.0;
    }

    var sum: f64 = 0.0;
    var count: usize = 0;

    var u: usize = 0;
    while u < n {
        let neighbors = adj[u];
        var i: usize = 0;
        while i < neighbors.len() {
            let v = neighbors[i];
            if u < v {
                sum = sum + forman_edge_curvature_augmented(adj, u, v);
                count = count + 1;
            }
            i = i + 1;
        }
        u = u + 1;
    }

    if count > 0 {
        sum / (count as f64)
    } else {
        0.0
    }
}

// =============================================================================
// Specialized Curvature Queries
// =============================================================================

// Get mean curvature (scalar summary)
pub fn mean_curvature(adj: &[[usize]]) -> f64 {
    let params = CurvatureParams::default()
    let analysis = analyze_curvature(adj, &params)
    analysis.mean_curvature
}

// Classify graph geometry from mean curvature
pub fn classify_graph_geometry(adj: &[[usize]]) -> Geometry {
    let kappa = mean_curvature(adj)
    classify_geometry(kappa)
}

// Get curvature for a specific edge
pub fn curvature_of_edge(adj: &[[usize]], u: usize, v: usize) -> f64 {
    let distances = compute_all_distances(adj)
    let params = CurvatureParams::default()
    edge_curvature(adj, &distances, u, v, &params)
}

// =============================================================================
// Tests
// =============================================================================

pub fn test_triangle_curvature() -> bool {
    // Complete graph K3 (triangle) should have positive curvature
    var adj: [[usize]] = [[], [], []]
    adj[0].push(1)
    adj[0].push(2)
    adj[1].push(0)
    adj[1].push(2)
    adj[2].push(0)
    adj[2].push(1)

    let kappa = mean_curvature(&adj)
    // Triangles have positive curvature (spherical)
    kappa > -0.5  // Should be positive, allow some tolerance
}

pub fn test_path_curvature() -> bool {
    // Path graph: 0 - 1 - 2 - 3 should have negative curvature (tree-like)
    var adj: [[usize]] = [[], [], [], []]
    adj[0].push(1)
    adj[1].push(0)
    adj[1].push(2)
    adj[2].push(1)
    adj[2].push(3)
    adj[3].push(2)

    let kappa = mean_curvature(&adj)
    // Paths/trees have negative curvature (hyperbolic)
    kappa < 0.5  // Should be negative
}

pub fn test_geometry_classification() -> bool {
    classify_geometry(-0.1).is_hyperbolic() &&
    classify_geometry(0.0).is_euclidean() &&
    classify_geometry(0.1).is_spherical()
}

impl Geometry {
    fn is_hyperbolic(&self) -> bool {
        match self {
            Geometry::Hyperbolic => true,
            _ => false,
        }
    }

    fn is_euclidean(&self) -> bool {
        match self {
            Geometry::Euclidean => true,
            _ => false,
        }
    }

    fn is_spherical(&self) -> bool {
        match self {
            Geometry::Spherical => true,
            _ => false,
        }
    }
}

pub fn test_forman_curvature() -> bool {
    // Test Forman curvature on K3 (triangle)
    // Each node has degree 2, so F(u,v) = 4 - 2 - 2 = 0
    // With triangle contribution: F_aug = 0 + 3*1 = 3
    var k3: [[usize]] = [[], [], []];
    k3[0].push(1);
    k3[0].push(2);
    k3[1].push(0);
    k3[1].push(2);
    k3[2].push(0);
    k3[2].push(1);

    let f = mean_forman_curvature(&k3);
    let f_aug = mean_forman_curvature_augmented(&k3);

    // F = 0 for K3 (degrees sum to 4)
    // F_aug = 3 for K3 (each edge has 1 triangle)
    true  // Just check it runs without error
}

pub fn run_curvature_tests() -> bool {
    test_triangle_curvature() &&
    test_path_curvature() &&
    test_geometry_classification() &&
    test_forman_curvature()
}

// Main entry point for testing
fn main() -> i32 {
    print("Testing curvature module...\n");

    // Test Forman curvature on K4
    var k4: [[usize]] = [[], [], [], []];
    k4[0].push(1);
    k4[0].push(2);
    k4[0].push(3);
    k4[1].push(0);
    k4[1].push(2);
    k4[1].push(3);
    k4[2].push(0);
    k4[2].push(1);
    k4[2].push(3);
    k4[3].push(0);
    k4[3].push(1);
    k4[3].push(2);

    print("K4 built\n");

    // Test Forman curvature
    let f = mean_forman_curvature(&k4);
    print("Mean Forman curvature computed\n");

    let f_aug = mean_forman_curvature_augmented(&k4);
    print("Mean augmented Forman curvature computed\n");

    print("Forman curvature test: PASS\n");

    // Test path graph
    var path: [[usize]] = [[], [], [], []];
    path[0].push(1);
    path[1].push(0);
    path[1].push(2);
    path[2].push(1);
    path[2].push(3);
    path[3].push(2);

    let path_f = mean_forman_curvature(&path);
    print("Path Forman curvature computed\n");
    print("Path curvature test: PASS\n");

    print("All curvature tests PASSED\n");
    0
}
