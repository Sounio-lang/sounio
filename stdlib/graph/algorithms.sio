// Graph Algorithms Module - BFS, shortest paths, connected components
//
// Core graph algorithms for network geometry analysis.
// Uses adjacency list representation directly.

module graph::algorithms

// =============================================================================
// Constants
// =============================================================================

let INF_DIST: i64 = -1  // Use -1 for unreachable (cleaner than large number)

// =============================================================================
// BFS Result
// =============================================================================

pub struct BfsResult {
    pub distances: [i64],   // -1 means unreachable
    pub parents: [i64],     // -1 means no parent
    pub source: usize,
}

impl BfsResult {
    pub fn distance_to(&self, target: usize) -> i64 {
        if target < self.distances.len() {
            self.distances[target]
        } else {
            INF_DIST
        }
    }

    pub fn is_reachable(&self, target: usize) -> bool {
        target < self.distances.len() && self.distances[target] >= 0
    }
}

// =============================================================================
// Breadth-First Search
// =============================================================================

// Perform BFS from a source node on an adjacency list
pub fn bfs(adj: &[[usize]], source: usize) -> BfsResult {
    let n = adj.len()

    // Initialize distances to -1 (unreachable)
    var distances: [i64] = []
    var parents: [i64] = []
    var i: usize = 0
    while i < n {
        distances.push(-1)
        parents.push(-1)
        i = i + 1
    }

    if source >= n {
        return BfsResult { distances, parents, source }
    }

    // BFS using array-based queue
    var queue: [usize] = []
    queue.push(source)
    distances[source] = 0

    var head: usize = 0
    while head < queue.len() {
        let u = queue[head]
        head = head + 1

        let neighbors = &adj[u]
        var j: usize = 0
        while j < neighbors.len() {
            let v = neighbors[j]
            if distances[v] == -1 {
                distances[v] = distances[u] + 1
                parents[v] = u as i64
                queue.push(v)
            }
            j = j + 1
        }
    }

    BfsResult { distances, parents, source }
}

// =============================================================================
// Shortest Paths
// =============================================================================

// Compute shortest path distance between two nodes
pub fn shortest_path_distance(adj: &[[usize]], u: usize, v: usize) -> i64 {
    let result = bfs(adj, u)
    result.distance_to(v)
}

// Reconstruct path from BFS result
pub fn reconstruct_path(result: &BfsResult, target: usize) -> [usize] {
    var path: [usize] = []

    if target >= result.distances.len() || result.distances[target] < 0 {
        return path
    }

    // Build path backwards
    var current = target as i64
    while current >= 0 {
        path.push(current as usize)
        if (current as usize) == result.source {
            break
        }
        current = result.parents[current as usize]
    }

    // Reverse the path
    reverse_array(&mut path)
    path
}

fn reverse_array(arr: &mut [usize]) {
    let n = arr.len()
    if n <= 1 { return }

    var i: usize = 0
    var j: usize = n - 1
    while i < j {
        let tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp
        i = i + 1
        j = j - 1
    }
}

// Compute all-pairs shortest paths
pub fn all_pairs_shortest_paths(adj: &[[usize]]) -> [[i64]] {
    let n = adj.len()
    var distances: [[i64]] = []

    var u: usize = 0
    while u < n {
        let bfs_result = bfs(adj, u)
        distances.push(bfs_result.distances)
        u = u + 1
    }
    distances
}

// =============================================================================
// Connected Components
// =============================================================================

// Find connected components using BFS
// Returns array mapping each node to its component ID
pub fn connected_components(adj: &[[usize]]) -> [usize] {
    let n = adj.len()
    let unvisited: usize = n  // Use n as "unvisited" marker

    var labels: [usize] = []
    var i: usize = 0
    while i < n {
        labels.push(unvisited)
        i = i + 1
    }

    var current_label: usize = 0
    var node: usize = 0

    while node < n {
        if labels[node] == unvisited {
            // BFS from this node
            var queue: [usize] = []
            queue.push(node)
            labels[node] = current_label

            var head: usize = 0
            while head < queue.len() {
                let u = queue[head]
                head = head + 1

                let neighbors = &adj[u]
                var j: usize = 0
                while j < neighbors.len() {
                    let v = neighbors[j]
                    if labels[v] == unvisited {
                        labels[v] = current_label
                        queue.push(v)
                    }
                    j = j + 1
                }
            }
            current_label = current_label + 1
        }
        node = node + 1
    }

    labels
}

// Count number of connected components
pub fn num_components(adj: &[[usize]]) -> usize {
    let labels = connected_components(adj)
    var max_label: usize = 0
    var i: usize = 0
    while i < labels.len() {
        if labels[i] < labels.len() && labels[i] > max_label {
            max_label = labels[i]
        }
        i = i + 1
    }
    if labels.len() == 0 { 0 } else { max_label + 1 }
}

// Check if graph is connected
pub fn is_connected(adj: &[[usize]]) -> bool {
    let n = adj.len()
    if n <= 1 { return true }

    let bfs_result = bfs(adj, 0)
    var i: usize = 0
    while i < n {
        if bfs_result.distances[i] < 0 { return false }
        i = i + 1
    }
    true
}

// Get size of largest connected component
pub fn largest_component_size(adj: &[[usize]]) -> usize {
    let labels = connected_components(adj)
    let n = labels.len()
    if n == 0 { return 0 }

    // Count nodes in each component
    var counts: [usize] = []
    var i: usize = 0
    while i < n {
        counts.push(0)
        i = i + 1
    }

    i = 0
    while i < n {
        if labels[i] < n {
            counts[labels[i]] = counts[labels[i]] + 1
        }
        i = i + 1
    }

    var max_size: usize = 0
    i = 0
    while i < n {
        if counts[i] > max_size { max_size = counts[i] }
        i = i + 1
    }
    max_size
}

// =============================================================================
// Neighborhood Operations (for Ollivier-Ricci curvature)
// =============================================================================

// Get k-hop neighborhood of a node (excluding the node itself)
pub fn k_hop_neighborhood(adj: &[[usize]], source: usize, k: usize) -> [usize] {
    let result = bfs(adj, source)

    var neighbors: [usize] = []
    var i: usize = 0
    while i < result.distances.len() {
        let d = result.distances[i]
        if d > 0 && d <= (k as i64) {
            neighbors.push(i)
        }
        i = i + 1
    }
    neighbors
}

// Get 1-hop neighbors (direct neighbors)
pub fn direct_neighbors(adj: &[[usize]], node: usize) -> [usize] {
    if node >= adj.len() {
        return []
    }

    var result: [usize] = []
    let neighbors = &adj[node]
    var i: usize = 0
    while i < neighbors.len() {
        result.push(neighbors[i])
        i = i + 1
    }
    result
}

// =============================================================================
// Graph Statistics
// =============================================================================

// Compute degree distribution (index = degree, value = count)
pub fn degree_distribution(adj: &[[usize]]) -> [usize] {
    let n = adj.len()
    if n == 0 { return [] }

    // Find max degree
    var max_degree: usize = 0
    var i: usize = 0
    while i < n {
        let deg = adj[i].len()
        if deg > max_degree { max_degree = deg }
        i = i + 1
    }

    // Count nodes with each degree
    var counts: [usize] = []
    i = 0
    while i <= max_degree {
        counts.push(0)
        i = i + 1
    }

    i = 0
    while i < n {
        let deg = adj[i].len()
        counts[deg] = counts[deg] + 1
        i = i + 1
    }
    counts
}

// Compute average path length (only considers reachable pairs)
pub fn average_path_length(adj: &[[usize]]) -> f64 {
    let n = adj.len()
    if n <= 1 { return 0.0 }

    let distances = all_pairs_shortest_paths(adj)

    var sum: f64 = 0.0
    var count: usize = 0

    var i: usize = 0
    while i < n {
        var j: usize = i + 1
        while j < n {
            let d = distances[i][j]
            if d > 0 {
                sum = sum + (d as f64)
                count = count + 1
            }
            j = j + 1
        }
        i = i + 1
    }

    if count == 0 { 0.0 } else { sum / (count as f64) }
}

// Compute graph diameter (longest shortest path)
pub fn diameter(adj: &[[usize]]) -> i64 {
    let n = adj.len()
    if n == 0 { return 0 }

    let distances = all_pairs_shortest_paths(adj)

    var max_dist: i64 = 0
    var i: usize = 0
    while i < n {
        var j: usize = 0
        while j < n {
            let d = distances[i][j]
            if d > max_dist { max_dist = d }
            j = j + 1
        }
        i = i + 1
    }
    max_dist
}

// =============================================================================
// Tests
// =============================================================================

pub fn test_bfs_path() -> bool {
    // Path graph: 0 - 1 - 2 - 3
    var adj: [[usize]] = [[], [], [], []]
    adj[0].push(1)
    adj[1].push(0)
    adj[1].push(2)
    adj[2].push(1)
    adj[2].push(3)
    adj[3].push(2)

    let result = bfs(&adj, 0)
    result.distances[0] == 0 &&
    result.distances[1] == 1 &&
    result.distances[2] == 2 &&
    result.distances[3] == 3
}

pub fn test_bfs_triangle() -> bool {
    // Triangle: 0 - 1 - 2 - 0
    var adj: [[usize]] = [[], [], []]
    adj[0].push(1)
    adj[0].push(2)
    adj[1].push(0)
    adj[1].push(2)
    adj[2].push(0)
    adj[2].push(1)

    shortest_path_distance(&adj, 0, 2) == 1
}

pub fn test_components() -> bool {
    // Two disconnected edges: 0-1 and 2-3
    var adj: [[usize]] = [[], [], [], []]
    adj[0].push(1)
    adj[1].push(0)
    adj[2].push(3)
    adj[3].push(2)

    num_components(&adj) == 2
}

pub fn test_connectivity() -> bool {
    // Connected path: 0 - 1 - 2
    var adj1: [[usize]] = [[], [], []]
    adj1[0].push(1)
    adj1[1].push(0)
    adj1[1].push(2)
    adj1[2].push(1)

    // Disconnected: 0 - 1, 2 isolated
    var adj2: [[usize]] = [[], [], []]
    adj2[0].push(1)
    adj2[1].push(0)

    is_connected(&adj1) && !is_connected(&adj2)
}

pub fn run_algorithm_tests() -> bool {
    test_bfs_path() &&
    test_bfs_triangle() &&
    test_components() &&
    test_connectivity()
}
