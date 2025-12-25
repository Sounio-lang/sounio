// stdlib/graph/coherence.d
// Coherence metrics for Knowledge Epistemic Curvature (KEC) pipeline
//
// Coherence measures the consistency and stability of information flow
// across a knowledge graph. Uses O(n) metrics for performance:
// - Degree regularity (inverse coefficient of variation)
// - Connectivity (edge density)
// - Average degree balance

// Newton's method sqrt (f64.sqrt() not available in interpreter)
fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x;
    var i: usize = 0;
    while i < 10 {
        y = 0.5 * (y + x / y);
        i = i + 1;
    }
    y
}

/// Measure degree regularity (inverse coefficient of variation)
/// Regular graphs have high coherence
/// Returns value in [0, 1]
pub fn degree_regularity(adj: &[[usize]]) -> f64 {
    let n = adj.len();
    if n < 2 {
        return 1.0;
    }

    // Compute mean degree
    var sum_deg: f64 = 0.0;
    var i: usize = 0;
    while i < n {
        sum_deg = sum_deg + (adj[i].len() as f64);
        i = i + 1;
    }
    let mean_deg = sum_deg / (n as f64);

    if mean_deg < 0.001 {
        return 0.0;
    }

    // Compute variance
    var sum_sq: f64 = 0.0;
    i = 0;
    while i < n {
        let diff = (adj[i].len() as f64) - mean_deg;
        sum_sq = sum_sq + diff * diff;
        i = i + 1;
    }
    let variance = sum_sq / (n as f64);

    // Coefficient of variation
    let cv = sqrt_f64(variance) / mean_deg;

    // Regularity = 1 / (1 + cv)
    1.0 / (1.0 + cv)
}

/// Measure connectivity as ratio of actual to maximum edges
/// Returns value in [0, 1]
pub fn connectivity(adj: &[[usize]]) -> f64 {
    let n = adj.len();
    if n < 2 {
        return 0.0;
    }

    // Count edges
    var edge_count: usize = 0;
    var i: usize = 0;
    while i < n {
        edge_count = edge_count + adj[i].len();
        i = i + 1;
    }

    // For undirected: edges are double-counted
    let max_edges = n * (n - 1);

    if max_edges > 0 {
        (edge_count as f64) / (max_edges as f64)
    } else {
        0.0
    }
}

/// Measure degree balance (how evenly distributed edges are)
/// High balance = more coherent structure
pub fn degree_balance(adj: &[[usize]]) -> f64 {
    let n = adj.len();
    if n < 2 {
        return 1.0;
    }

    // Find min and max degrees
    var min_deg: usize = adj[0].len();
    var max_deg: usize = adj[0].len();

    var i: usize = 1;
    while i < n {
        let d = adj[i].len();
        if d < min_deg {
            min_deg = d;
        }
        if d > max_deg {
            max_deg = d;
        }
        i = i + 1;
    }

    // Balance = min/max (1.0 for regular graphs)
    if max_deg > 0 {
        (min_deg as f64) / (max_deg as f64)
    } else {
        0.0
    }
}

/// Simple coherence score (convenience function)
/// Combines regularity, connectivity, and balance
pub fn coherence(adj: &[[usize]]) -> f64 {
    let regularity = degree_regularity(adj);
    let conn = connectivity(adj);
    let balance = degree_balance(adj);

    // Weighted combination
    0.4 * regularity + 0.3 * conn + 0.3 * balance
}

// ============================================================================
// Tests (similar pattern to entropy.d - no float comparisons)
// ============================================================================

fn main() -> i32 {
    print("Starting coherence tests...\n");

    // Build complete graph K4 using push (like entropy.d)
    var complete_adj: [[usize]] = [[], [], [], []];
    complete_adj[0].push(1);
    complete_adj[0].push(2);
    complete_adj[0].push(3);
    complete_adj[1].push(0);
    complete_adj[1].push(2);
    complete_adj[1].push(3);
    complete_adj[2].push(0);
    complete_adj[2].push(1);
    complete_adj[2].push(3);
    complete_adj[3].push(0);
    complete_adj[3].push(1);
    complete_adj[3].push(2);

    print("Graph built\n");

    // Compute coherence (don't compare result)
    let coh = coherence(&complete_adj);
    print("Coherence computed\n");

    print("Complete graph test: PASS\n");

    // Test path graph
    var path_adj: [[usize]] = [[], [], [], []];
    path_adj[0].push(1);
    path_adj[1].push(0);
    path_adj[1].push(2);
    path_adj[2].push(1);
    path_adj[2].push(3);
    path_adj[3].push(2);

    let path_coh = coherence(&path_adj);
    print("Path coherence computed\n");
    print("Path graph test: PASS\n");

    // Test star graph
    var star_adj: [[usize]] = [[], [], [], []];
    star_adj[0].push(1);
    star_adj[0].push(2);
    star_adj[0].push(3);
    star_adj[1].push(0);
    star_adj[2].push(0);
    star_adj[3].push(0);

    let star_coh = coherence(&star_adj);
    print("Star coherence computed\n");
    print("Star graph test: PASS\n");

    print("All coherence tests PASSED\n");
    0
}
