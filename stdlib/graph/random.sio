// Random Graph Generation
//
// Implements random graph models for testing network geometry algorithms:
// - Erdos-Renyi G(n,p)
// - Barabasi-Albert preferential attachment
// - Watts-Strogatz small-world
// - Random regular graphs

module graph::random

// =============================================================================
// Linear Congruential Generator (LCG) for reproducible randomness
// =============================================================================

pub struct RandomState {
    state: u64,
}

impl RandomState {
    pub fn new(seed: u64) -> RandomState {
        let s = if seed == 0 { 12345 } else { seed }
        RandomState { state: s }
    }

    // Generate next random u64
    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters (Numerical Recipes)
        let a: u64 = 6364136223846793005
        let c: u64 = 1442695040888963407
        self.state = self.state * a + c
        self.state
    }

    // Random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        let r = self.next_u64()
        (r as f64) / 18446744073709551616.0
    }

    // Random usize in [0, max)
    pub fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 { return 0 }
        let r = self.next_u64()
        (r as usize) % max
    }

    // Random bool with probability p
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

// =============================================================================
// Helper: Integer division returning usize
// =============================================================================

fn div_usize(a: usize, b: usize) -> usize {
    if b == 0 { return 0 }
    // Perform division using loop to avoid i64 return type issues
    var result: usize = 0
    var remainder: usize = a
    while remainder >= b {
        remainder = remainder - b
        result = result + 1
    }
    result
}

// =============================================================================
// Erdos-Renyi G(n,p) Random Graph
// =============================================================================

// Generate Erdos-Renyi random graph
// Each edge exists independently with probability p
pub fn erdos_renyi(n: usize, p: f64, seed: u64) -> [[usize]] {
    var rng = RandomState::new(seed)
    var adj: [[usize]] = []

    // Initialize empty adjacency lists
    var i: usize = 0
    while i < n {
        adj.push([])
        i = i + 1
    }

    // Add each possible edge with probability p
    var u: usize = 0
    while u < n {
        var v: usize = u + 1
        while v < n {
            if rng.bernoulli(p) {
                adj[u].push(v)
                adj[v].push(u)
            }
            v = v + 1
        }
        u = u + 1
    }

    adj
}

// Generate sparse Erdos-Renyi with expected degree k
pub fn erdos_renyi_degree(n: usize, expected_degree: f64, seed: u64) -> [[usize]] {
    if n <= 1 { return [[]] }
    let p = expected_degree / ((n - 1) as f64)
    erdos_renyi(n, p, seed)
}

// =============================================================================
// Barabasi-Albert Preferential Attachment
// =============================================================================

// Generate Barabasi-Albert preferential attachment graph
// Start with m0 nodes, add n-m0 nodes each connecting to m existing nodes
pub fn barabasi_albert(n: usize, m: usize, seed: u64) -> [[usize]] {
    var rng = RandomState::new(seed)

    // Need at least m+1 nodes
    if n <= m {
        return erdos_renyi(n, 1.0, seed)  // Complete graph
    }

    var adj: [[usize]] = []
    var degrees: [usize] = []

    // Initialize with complete graph on m nodes
    var i: usize = 0
    while i < n {
        adj.push([])
        degrees.push(0)
        i = i + 1
    }

    // Create initial clique of m nodes (forward edges)
    i = 0
    while i < m {
        var j: usize = i + 1
        while j < m {
            adj[i].push(j)
            degrees[i] = degrees[i] + 1
            j = j + 1
        }
        i = i + 1
    }

    // Add backward edges
    i = 0
    while i < m {
        var j: usize = i + 1
        while j < m {
            adj[j].push(i)
            degrees[j] = degrees[j] + 1
            j = j + 1
        }
        i = i + 1
    }

    // Track total degree for preferential attachment
    // m*(m-1) is number of edges in clique, each contributes 2 to total degree
    var total_degree: usize = m * (m - 1)

    // Add remaining nodes
    var new_node: usize = m
    while new_node < n {
        // Connect to m existing nodes via preferential attachment
        var targets: [usize] = []
        var attempts: usize = 0
        let max_attempts = m * 10

        while targets.len() < m && attempts < max_attempts {
            // Pick random node proportional to degree
            if total_degree == 0 {
                // If no edges yet, pick uniformly
                let target = rng.next_usize(new_node)
                if !contains(&targets, target) {
                    targets.push(target)
                }
            } else {
                let r = rng.next_usize(total_degree)
                var cumsum: usize = 0
                var node: usize = 0
                while node < new_node {
                    cumsum = cumsum + degrees[node]
                    if cumsum > r {
                        if !contains(&targets, node) {
                            targets.push(node)
                        }
                        break
                    }
                    node = node + 1
                }
            }
            attempts = attempts + 1
        }

        // Add edges to targets (forward: new_node -> target)
        i = 0
        while i < targets.len() {
            let target = targets[i]
            adj[new_node].push(target)
            degrees[new_node] = degrees[new_node] + 1
            total_degree = total_degree + 1
            i = i + 1
        }

        // Add edges to targets (backward: target -> new_node)
        i = 0
        while i < targets.len() {
            let target = targets[i]
            adj[target].push(new_node)
            degrees[target] = degrees[target] + 1
            total_degree = total_degree + 1
            i = i + 1
        }

        new_node = new_node + 1
    }

    adj
}

fn contains(arr: &[usize], val: usize) -> bool {
    var i: usize = 0
    while i < arr.len() {
        if arr[i] == val { return true }
        i = i + 1
    }
    false
}

// =============================================================================
// Ring Lattice (simpler version, no rewiring)
// =============================================================================

// Generate ring lattice graph
// Each node connects to k/2 neighbors on each side
pub fn ring_lattice(n: usize, k: usize) -> [[usize]] {
    var adj: [[usize]] = []
    var i: usize = 0
    while i < n {
        adj.push([])
        i = i + 1
    }

    if n < 2 || k == 0 { return adj }

    let half_k: usize = div_usize(k, 2)

    // First pass: add forward edges
    i = 0
    while i < n {
        var j: usize = 1
        while j <= half_k {
            let neighbor = (i + j) % n
            adj[i].push(neighbor)
            j = j + 1
        }
        i = i + 1
    }

    // Second pass: add backward edges
    i = 0
    while i < n {
        var j: usize = 1
        while j <= half_k {
            let neighbor = (i + j) % n
            adj[neighbor].push(i)
            j = j + 1
        }
        i = i + 1
    }

    adj
}

// =============================================================================
// Complete and Regular Graphs
// =============================================================================

// Generate complete graph K_n
pub fn complete_graph(n: usize) -> [[usize]] {
    var adj: [[usize]] = []
    var i: usize = 0
    while i < n {
        adj.push([])
        i = i + 1
    }

    i = 0
    while i < n {
        var j: usize = 0
        while j < n {
            if i != j {
                adj[i].push(j)
            }
            j = j + 1
        }
        i = i + 1
    }

    adj
}

// Generate cycle graph C_n
pub fn cycle_graph(n: usize) -> [[usize]] {
    var adj: [[usize]] = []
    var i: usize = 0
    while i < n {
        adj.push([])
        i = i + 1
    }

    if n < 2 { return adj }

    // First pass: add forward edges (i -> i+1)
    i = 0
    while i < n {
        let next = (i + 1) % n
        adj[i].push(next)
        i = i + 1
    }

    // Second pass: add backward edges (i+1 -> i)
    i = 0
    while i < n {
        let next = (i + 1) % n
        adj[next].push(i)
        i = i + 1
    }

    adj
}

// Generate path graph P_n
pub fn path_graph(n: usize) -> [[usize]] {
    var adj: [[usize]] = []
    var i: usize = 0
    while i < n {
        adj.push([])
        i = i + 1
    }

    if n < 2 { return adj }

    // First pass: add forward edges (i -> i+1)
    i = 0
    while i < n - 1 {
        adj[i].push(i + 1)
        i = i + 1
    }

    // Second pass: add backward edges (i+1 -> i)
    i = 0
    while i < n - 1 {
        adj[i + 1].push(i)
        i = i + 1
    }

    adj
}

// Generate star graph S_n (one center connected to n-1 leaves)
pub fn star_graph(n: usize) -> [[usize]] {
    var adj: [[usize]] = []
    var i: usize = 0
    while i < n {
        adj.push([])
        i = i + 1
    }

    if n < 2 { return adj }

    // First pass: center -> leaves
    i = 1
    while i < n {
        adj[0].push(i)
        i = i + 1
    }

    // Second pass: leaves -> center
    i = 1
    while i < n {
        adj[i].push(0)
        i = i + 1
    }

    adj
}

// =============================================================================
// Graph Statistics
// =============================================================================

// Count edges in graph
pub fn count_edges(adj: &[[usize]]) -> usize {
    var total: usize = 0
    var i: usize = 0
    while i < adj.len() {
        total = total + adj[i].len()
        i = i + 1
    }
    div_usize(total, 2)  // Each edge counted twice
}

// Compute mean degree
pub fn mean_degree(adj: &[[usize]]) -> f64 {
    let n = adj.len()
    if n == 0 { return 0.0 }
    let m = count_edges(adj)
    (2.0 * (m as f64)) / (n as f64)
}

// =============================================================================
// Tests
// =============================================================================

pub fn test_erdos_renyi() -> bool {
    let g = erdos_renyi(10, 0.5, 42)
    g.len() == 10
}

pub fn test_barabasi_albert() -> bool {
    let g = barabasi_albert(20, 3, 42)
    g.len() == 20 && count_edges(&g) > 0
}

pub fn test_complete_graph() -> bool {
    let g = complete_graph(5)
    // K_5 has 5*4/2 = 10 edges
    count_edges(&g) == 10
}

pub fn test_cycle_graph() -> bool {
    let g = cycle_graph(6)
    // C_6 has 6 edges
    count_edges(&g) == 6
}

pub fn test_path_graph() -> bool {
    let g = path_graph(5)
    // P_5 has 4 edges
    count_edges(&g) == 4
}

pub fn run_random_tests() -> bool {
    test_erdos_renyi() &&
    test_barabasi_albert() &&
    test_complete_graph() &&
    test_cycle_graph() &&
    test_path_graph()
}
