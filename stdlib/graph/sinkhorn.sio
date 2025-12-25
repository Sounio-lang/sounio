// Sinkhorn Algorithm for Optimal Transport
//
// Implements the Sinkhorn-Knopp algorithm for computing
// the regularized Wasserstein distance between probability measures.
// Used for Ollivier-Ricci curvature computation.

module graph::sinkhorn

// =============================================================================
// Probability Measure
// =============================================================================

pub struct ProbMeasure {
    pub weights: [f64],  // Probability weights (must sum to 1)
}

impl ProbMeasure {
    pub fn new(weights: [f64]) -> ProbMeasure {
        ProbMeasure { weights }
    }

    pub fn uniform(n: usize) -> ProbMeasure {
        var weights: [f64] = []
        let w = 1.0 / (n as f64)
        var i: usize = 0
        while i < n {
            weights.push(w)
            i = i + 1
        }
        ProbMeasure { weights }
    }

    pub fn total(&self) -> f64 {
        var sum: f64 = 0.0
        var i: usize = 0
        while i < self.weights.len() {
            sum = sum + self.weights[i]
            i = i + 1
        }
        sum
    }

    pub fn normalize(&mut self) {
        let total = self.total()
        if total > 0.0 {
            var i: usize = 0
            while i < self.weights.len() {
                self.weights[i] = self.weights[i] / total
                i = i + 1
            }
        }
    }

    pub fn size(&self) -> usize {
        self.weights.len()
    }
}

// =============================================================================
// Cost Matrix (Distance Matrix)
// =============================================================================

pub struct CostMatrix {
    pub n: usize,         // rows
    pub m: usize,         // columns
    pub data: [f64],      // row-major
}

impl CostMatrix {
    pub fn new(n: usize, m: usize) -> CostMatrix {
        var data: [f64] = []
        let size = n * m
        var i: usize = 0
        while i < size {
            data.push(0.0)
            i = i + 1
        }
        CostMatrix { n, m, data }
    }

    pub fn from_distances(distances: &[[i64]], nodes_a: &[usize], nodes_b: &[usize]) -> CostMatrix {
        let n = nodes_a.len()
        let m = nodes_b.len()
        var mat = CostMatrix::new(n, m)

        var i: usize = 0
        while i < n {
            var j: usize = 0
            while j < m {
                let d = distances[nodes_a[i]][nodes_b[j]]
                mat.set(i, j, d as f64)
                j = j + 1
            }
            i = i + 1
        }
        mat
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.n && j < self.m {
            self.data[i * self.m + j]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        if i < self.n && j < self.m {
            self.data[i * self.m + j] = val
        }
    }
}

// =============================================================================
// Sinkhorn Algorithm
// =============================================================================

// Sinkhorn parameters
pub struct SinkhornParams {
    pub epsilon: f64,     // Regularization parameter
    pub max_iter: usize,  // Maximum iterations
    pub tolerance: f64,   // Convergence tolerance
}

impl SinkhornParams {
    pub fn default() -> SinkhornParams {
        SinkhornParams {
            epsilon: 0.1,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }
}

// Sinkhorn result
pub struct SinkhornResult {
    pub distance: f64,        // Wasserstein distance
    pub converged: bool,      // Whether algorithm converged
    pub iterations: usize,    // Number of iterations used
}

// Compute exp(-c/epsilon) for Gibbs kernel
fn compute_gibbs_kernel(cost: &CostMatrix, epsilon: f64) -> CostMatrix {
    var kmat = CostMatrix::new(cost.n, cost.m)
    var i: usize = 0
    while i < cost.n {
        var j: usize = 0
        while j < cost.m {
            let c = cost.get(i, j)
            // exp(-c/epsilon)
            let val = exp_approx(-c / epsilon)
            kmat.set(i, j, val)
            j = j + 1
        }
        i = i + 1
    }
    kmat
}

// Approximate exp function using Taylor series
fn exp_approx(x: f64) -> f64 {
    // For large negative x, return small value
    if x < -20.0 { return 0.0 }
    // For large positive x, cap it
    if x > 20.0 { return 485165195.0 }  // exp(20) approx

    // Use identity: exp(x) = exp(x/n)^n for better range
    let n = 16  // Number of squarings
    let scaled = x / 16.0

    // Taylor: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + ...
    let x2 = scaled * scaled
    let x3 = x2 * scaled
    let x4 = x3 * scaled
    let x5 = x4 * scaled
    var result = 1.0 + scaled + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0

    // Square n times
    var i: usize = 0
    while i < n {
        result = result * result
        i = i + 1
    }
    result
}

// Natural log approximation
fn log_approx(x: f64) -> f64 {
    if x <= 0.0 { return -1e10 }
    if x > 1e10 { return 23.0 }

    // Use log(x) = log(m * 2^e) = log(m) + e*log(2)
    // For simplicity, use Newton's method: find y where exp(y) = x
    var y = 0.0
    if x > 1.0 { y = x - 1.0 }
    else { y = x - 1.0 }

    // Newton iterations: y_new = y - (exp(y) - x) / exp(y)
    var i: usize = 0
    while i < 20 {
        let ey = exp_approx(y)
        let delta = (ey - x) / ey
        y = y - delta
        if abs_f64(delta) < 1e-10 { break }
        i = i + 1
    }
    y
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

// Main Sinkhorn algorithm
// Computes regularized Wasserstein distance between two distributions
pub fn sinkhorn(mu: &ProbMeasure, nu: &ProbMeasure, cost: &CostMatrix, params: &SinkhornParams) -> SinkhornResult {
    let n = mu.size()
    let m = nu.size()

    if n == 0 || m == 0 {
        return SinkhornResult { distance: 0.0, converged: true, iterations: 0 }
    }

    // Compute Gibbs kernel K = exp(-C/epsilon)
    let kmat = compute_gibbs_kernel(cost, params.epsilon)

    // Initialize scaling vectors u and v
    var u: [f64] = []
    var v: [f64] = []
    var i: usize = 0
    while i < n { u.push(1.0); i = i + 1 }
    i = 0
    while i < m { v.push(1.0); i = i + 1 }

    var iter: usize = 0
    var converged = false

    while iter < params.max_iter {
        // Store old u for convergence check
        var u_old: [f64] = []
        i = 0
        while i < n { u_old.push(u[i]); i = i + 1 }

        // Update u: u = mu ./ (K * v)
        i = 0
        while i < n {
            var sum: f64 = 0.0
            var j: usize = 0
            while j < m {
                sum = sum + kmat.get(i, j) * v[j]
                j = j + 1
            }
            if sum > 1e-10 {
                u[i] = mu.weights[i] / sum
            } else {
                u[i] = 1.0
            }
            i = i + 1
        }

        // Update v: v = nu ./ (K^T * u)
        var j: usize = 0
        while j < m {
            var sum: f64 = 0.0
            i = 0
            while i < n {
                sum = sum + kmat.get(i, j) * u[i]
                i = i + 1
            }
            if sum > 1e-10 {
                v[j] = nu.weights[j] / sum
            } else {
                v[j] = 1.0
            }
            j = j + 1
        }

        // Check convergence: ||u - u_old||_inf < tolerance
        var max_diff: f64 = 0.0
        i = 0
        while i < n {
            let diff = abs_f64(u[i] - u_old[i])
            if diff > max_diff { max_diff = diff }
            i = i + 1
        }

        if max_diff < params.tolerance {
            converged = true
            break
        }

        iter = iter + 1
    }

    // Compute transport cost: sum_ij u_i * K_ij * v_j * C_ij
    var distance: f64 = 0.0
    i = 0
    while i < n {
        var j: usize = 0
        while j < m {
            let transport = u[i] * kmat.get(i, j) * v[j]
            distance = distance + transport * cost.get(i, j)
            j = j + 1
        }
        i = i + 1
    }

    SinkhornResult {
        distance,
        converged,
        iterations: iter,
    }
}

// Compute 1-Wasserstein distance using Sinkhorn
pub fn wasserstein_distance(mu: &ProbMeasure, nu: &ProbMeasure, cost: &CostMatrix) -> f64 {
    let params = SinkhornParams::default()
    let result = sinkhorn(mu, nu, cost, &params)
    result.distance
}

// Compute 1-Wasserstein with custom epsilon
pub fn wasserstein_with_epsilon(mu: &ProbMeasure, nu: &ProbMeasure, cost: &CostMatrix, epsilon: f64) -> f64 {
    var params = SinkhornParams::default()
    params.epsilon = epsilon
    let result = sinkhorn(mu, nu, cost, &params)
    result.distance
}

// =============================================================================
// Tests
// =============================================================================

pub fn test_uniform_measure() -> bool {
    let mu = ProbMeasure::uniform(4)
    let total = mu.total()
    abs_f64(total - 1.0) < 1e-10
}

pub fn test_exp_approx() -> bool {
    // exp(0) = 1
    let e0 = exp_approx(0.0)
    // exp(1) ≈ 2.718
    let e1 = exp_approx(1.0)
    // exp(-1) ≈ 0.368
    let em1 = exp_approx(-1.0)

    abs_f64(e0 - 1.0) < 1e-6 &&
    abs_f64(e1 - 2.718281828) < 0.01 &&
    abs_f64(em1 - 0.367879441) < 0.01
}

pub fn test_sinkhorn_identity() -> bool {
    // Transport from uniform to uniform with zero cost
    let mu = ProbMeasure::uniform(2)
    let nu = ProbMeasure::uniform(2)
    var cost = CostMatrix::new(2, 2)
    // Zero cost matrix
    let params = SinkhornParams::default()
    let result = sinkhorn(&mu, &nu, &cost, &params)
    result.distance < 0.01  // Should be ~0
}

pub fn test_sinkhorn_simple() -> bool {
    // Two points, transport from [1,0] to [0,1] with distance 1
    var mu = ProbMeasure::new([1.0, 0.0])
    var nu = ProbMeasure::new([0.0, 1.0])
    var cost = CostMatrix::new(2, 2)
    cost.set(0, 0, 0.0)
    cost.set(0, 1, 1.0)
    cost.set(1, 0, 1.0)
    cost.set(1, 1, 0.0)

    let params = SinkhornParams::default()
    let result = sinkhorn(&mu, &nu, &cost, &params)
    // Should transport 1 unit at cost 1
    abs_f64(result.distance - 1.0) < 0.1
}

pub fn run_sinkhorn_tests() -> bool {
    test_uniform_measure() &&
    test_exp_approx() &&
    test_sinkhorn_identity() &&
    test_sinkhorn_simple()
}
