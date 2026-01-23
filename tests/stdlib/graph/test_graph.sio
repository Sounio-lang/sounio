// Graph Module Test Runner

module graph::test_graph

// Import test functions from each module
// Note: Since module imports aren't fully working, we inline tests here

fn main() -> i32 {
    print("=== Graph Module Tests ===\n\n")

    var passed: i32 = 0
    var failed: i32 = 0

    // Test 1: Graph creation
    print("Test: Graph creation... ")
    if test_graph_creation() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test 2: BFS path
    print("Test: BFS path graph... ")
    if test_bfs_path() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test 3: BFS triangle
    print("Test: BFS triangle... ")
    if test_bfs_triangle() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test 4: Connected components
    print("Test: Connected components... ")
    if test_components() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test 5: Uniform measure
    print("Test: Uniform measure... ")
    if test_uniform_measure() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test 6: Exp approximation
    print("Test: Exp approximation... ")
    if test_exp_approx() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test 7: Complete graph
    print("Test: Complete graph K5... ")
    if test_complete_graph() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test 8: Path graph
    print("Test: Path graph P5... ")
    if test_path_graph() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    print("\n=== Results ===\n")
    print("Passed: ")
    print_i32(passed)
    print("\nFailed: ")
    print_i32(failed)
    print("\n")

    if failed == 0 {
        print("\nAll tests passed!\n")
        0
    } else {
        print("\nSome tests failed.\n")
        1
    }
}

// ============================================================================
// Inlined Tests from types.d
// ============================================================================

fn test_graph_creation() -> bool {
    var adj: [[usize]] = [[], [], [], [], []]
    adj[0].push(1)
    adj[1].push(0)
    adj[1].push(2)
    adj[2].push(1)
    adj[2].push(3)
    adj[3].push(2)
    adj[3].push(4)
    adj[4].push(3)
    adj[4].push(0)
    adj[0].push(4)

    adj.len() == 5 && adj[0].len() == 2
}

// ============================================================================
// Inlined Tests from algorithms.d
// ============================================================================

fn test_bfs_path() -> bool {
    // Path graph: 0 - 1 - 2 - 3
    var adj: [[usize]] = [[], [], [], []]
    adj[0].push(1)
    adj[1].push(0)
    adj[1].push(2)
    adj[2].push(1)
    adj[2].push(3)
    adj[3].push(2)

    // Inline BFS
    var dist: [i64] = [-1, -1, -1, -1]
    dist[0] = 0
    var queue: [usize] = [0]
    var head: usize = 0
    while head < queue.len() {
        let u = queue[head]
        head = head + 1
        var j: usize = 0
        while j < adj[u].len() {
            let v = adj[u][j]
            let neg_one: i64 = -1
            if dist[v] == neg_one {
                dist[v] = dist[u] + 1
                queue.push(v)
            }
            j = j + 1
        }
    }

    let d0: i64 = 0
    let d1: i64 = 1
    let d2: i64 = 2
    let d3: i64 = 3
    dist[0] == d0 && dist[1] == d1 && dist[2] == d2 && dist[3] == d3
}

fn test_bfs_triangle() -> bool {
    // Triangle: 0 - 1 - 2 - 0
    var adj: [[usize]] = [[], [], []]
    adj[0].push(1)
    adj[0].push(2)
    adj[1].push(0)
    adj[1].push(2)
    adj[2].push(0)
    adj[2].push(1)

    // Inline BFS from 0
    var dist: [i64] = [-1, -1, -1]
    dist[0] = 0
    var queue: [usize] = [0]
    var head: usize = 0
    while head < queue.len() {
        let u = queue[head]
        head = head + 1
        var j: usize = 0
        while j < adj[u].len() {
            let v = adj[u][j]
            let neg_one: i64 = -1
            if dist[v] == neg_one {
                dist[v] = dist[u] + 1
                queue.push(v)
            }
            j = j + 1
        }
    }

    let expected: i64 = 1
    dist[2] == expected
}

fn test_components() -> bool {
    // Two disconnected edges: 0-1 and 2-3
    var adj: [[usize]] = [[], [], [], []]
    adj[0].push(1)
    adj[1].push(0)
    adj[2].push(3)
    adj[3].push(2)

    let n: usize = 4
    let unvisited: usize = n

    var labels: [usize] = [n, n, n, n]
    var current_label: usize = 0
    var node: usize = 0

    while node < n {
        if labels[node] == unvisited {
            var queue: [usize] = []
            queue.push(node)
            labels[node] = current_label

            var head: usize = 0
            while head < queue.len() {
                let u = queue[head]
                head = head + 1

                var j: usize = 0
                while j < adj[u].len() {
                    let v = adj[u][j]
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

    let expected: usize = 2
    current_label == expected
}

// BFS implementation
struct BfsResult {
    distances: [i64],
    parents: [i64],
    source: usize,
}

fn bfs(adj: &[[usize]], source: usize) -> BfsResult {
    let n = adj.len()
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

    var queue: [usize] = []
    queue.push(source)
    distances[source] = 0

    var head: usize = 0
    while head < queue.len() {
        let u = queue[head]
        head = head + 1

        var j: usize = 0
        while j < adj[u].len() {
            let v = adj[u][j]
            let neg_one: i64 = -1
            if distances[v] == neg_one {
                distances[v] = distances[u] + 1
                parents[v] = u as i64
                queue.push(v)
            }
            j = j + 1
        }
    }

    BfsResult { distances, parents, source }
}

fn shortest_path_distance(adj: &[[usize]], u: usize, v: usize) -> i64 {
    let result = bfs(adj, u)
    if v < result.distances.len() {
        result.distances[v]
    } else {
        -1
    }
}

fn num_components(adj: &[[usize]]) -> usize {
    let n = adj.len()
    let unvisited: usize = n

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
            var queue: [usize] = []
            queue.push(node)
            labels[node] = current_label

            var head: usize = 0
            while head < queue.len() {
                let u = queue[head]
                head = head + 1

                var j: usize = 0
                while j < adj[u].len() {
                    let v = adj[u][j]
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

    var max_label: usize = 0
    i = 0
    while i < labels.len() {
        if labels[i] < labels.len() && labels[i] > max_label {
            max_label = labels[i]
        }
        i = i + 1
    }
    if labels.len() == 0 { 0 } else { max_label + 1 }
}

// ============================================================================
// Inlined Tests from sinkhorn.d
// ============================================================================

fn test_uniform_measure() -> bool {
    let n: usize = 4
    var weights: [f64] = []
    let w = 1.0 / (n as f64)
    var i: usize = 0
    while i < n {
        weights.push(w)
        i = i + 1
    }

    var sum: f64 = 0.0
    i = 0
    while i < weights.len() {
        sum = sum + weights[i]
        i = i + 1
    }

    abs_f64(sum - 1.0) < 1e-10
}

fn test_exp_approx() -> bool {
    // Simple test: exp(0) should be 1
    let e0 = exp_approx(0.0)
    abs_f64(e0 - 1.0) < 0.001
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
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

// ============================================================================
// Inlined Tests from random.d
// ============================================================================

fn test_complete_graph() -> bool {
    // K_5 complete graph
    var adj: [[usize]] = [[], [], [], [], []]
    var i: usize = 0
    while i < 5 {
        var j: usize = 0
        while j < 5 {
            if i != j {
                adj[i].push(j)
            }
            j = j + 1
        }
        i = i + 1
    }

    // K_5 has 5*4/2 = 10 edges, each node has degree 4
    adj[0].len() == 4 && adj[4].len() == 4
}

fn test_path_graph() -> bool {
    // P_5 path graph: 0 - 1 - 2 - 3 - 4
    var adj: [[usize]] = [[], [], [], [], []]

    // Forward edges
    adj[0].push(1)
    adj[1].push(2)
    adj[2].push(3)
    adj[3].push(4)

    // Backward edges
    adj[1].push(0)
    adj[2].push(1)
    adj[3].push(2)
    adj[4].push(3)

    // P_5 has 4 edges, endpoints have degree 1, middle nodes degree 2
    adj[0].len() == 1 && adj[2].len() == 2 && adj[4].len() == 1
}

// ============================================================================
// Helper: Print i32
// ============================================================================

fn print_i32(n: i32) {
    if n == 0 {
        print("0")
        return
    }

    var val = n
    if val < 0 {
        print("-")
        val = -val
    }

    var digits: [u8] = []
    while val > 0 {
        digits.push((48 + (val % 10)) as u8)
        val = val / 10
    }

    var i = digits.len()
    while i > 0 {
        i = i - 1
        print_char(digits[i])
    }
}

fn print_char(c: u8) {
    // Single character - use string literal
    if c == 48 { print("0") }
    else if c == 49 { print("1") }
    else if c == 50 { print("2") }
    else if c == 51 { print("3") }
    else if c == 52 { print("4") }
    else if c == 53 { print("5") }
    else if c == 54 { print("6") }
    else if c == 55 { print("7") }
    else if c == 56 { print("8") }
    else if c == 57 { print("9") }
}
