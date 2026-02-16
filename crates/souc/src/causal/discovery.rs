//! Causal Discovery: PC (Peter-Clark) Algorithm
//!
//! Constraint-based causal structure learning from observational data.
//! The PC algorithm discovers causal graphs by testing conditional
//! independence relationships in data, producing a CPDAG (Completed
//! Partially Directed Acyclic Graph).
//!
//! # Algorithm Overview
//!
//! 1. **Skeleton Discovery**: Start with complete undirected graph,
//!    remove edges where conditional independence is found
//! 2. **V-Structure Orientation**: Orient colliders X → Z ← Y where
//!    Z is not in the separating set of X and Y
//! 3. **Edge Orientation (Meek's Rules)**: Apply orientation rules
//!    R1-R4 to propagate directions without creating new v-structures
//!    or cycles
//!
//! # References
//!
//! - Spirtes, Glymour, Scheines (2000). "Causation, Prediction, and Search"
//! - Meek (1995). "Causal inference and causal explanation with background knowledge"
//! - Kalisch & Bühlmann (2007). "Estimating high-dimensional DAGs with PC"

use std::collections::{BTreeSet, HashMap};

use crate::common::Span;
use crate::types::causal::{CausalEdgeDef, CausalGraphDef, CausalNodeDef};

// ============================================================================
// Data Structures
// ============================================================================

/// Matrix of observational data: rows = samples, columns = variables.
#[derive(Clone, Debug)]
pub struct DataMatrix {
    /// Column (variable) names
    pub variables: Vec<String>,
    /// Row-major data: data[i][j] = value of variable j in sample i
    pub data: Vec<Vec<f64>>,
    /// Number of samples (rows)
    pub n_samples: usize,
    /// Number of variables (columns)
    pub n_vars: usize,
}

impl DataMatrix {
    /// Create a new data matrix from column names and row-major data.
    pub fn new(variables: Vec<String>, data: Vec<Vec<f64>>) -> Result<Self, PCError> {
        let n_samples = data.len();
        let n_vars = variables.len();

        if n_samples == 0 {
            return Err(PCError::InsufficientData {
                n_samples: 0,
                min_required: 1,
            });
        }

        for (i, row) in data.iter().enumerate() {
            if row.len() != n_vars {
                return Err(PCError::DimensionMismatch {
                    row: i,
                    expected: n_vars,
                    got: row.len(),
                });
            }
        }

        Ok(DataMatrix {
            variables,
            data,
            n_samples,
            n_vars,
        })
    }

    /// Get the column index for a variable name.
    pub fn var_index(&self, name: &str) -> Option<usize> {
        self.variables.iter().position(|v| v == name)
    }

    /// Get column data for a variable by index.
    pub fn column(&self, idx: usize) -> Vec<f64> {
        self.data.iter().map(|row| row[idx]).collect()
    }
}

/// Configuration for the PC algorithm.
#[derive(Clone, Debug)]
pub struct PCConfig {
    /// Significance level for conditional independence tests (default: 0.05)
    pub alpha: f64,
    /// Maximum conditioning set size (None = unlimited)
    pub max_cond_size: Option<usize>,
    /// Use stable (order-independent) variant
    pub stable: bool,
    /// Maximum depth for skeleton search (-1 = unlimited)
    pub max_depth: i32,
}

impl Default for PCConfig {
    fn default() -> Self {
        PCConfig {
            alpha: 0.05,
            max_cond_size: None,
            stable: true,
            max_depth: -1,
        }
    }
}

/// An undirected edge in the skeleton.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UndirectedEdge {
    /// One endpoint (always the smaller index)
    pub a: usize,
    /// Other endpoint (always the larger index)
    pub b: usize,
}

impl UndirectedEdge {
    /// Create an undirected edge, normalizing order.
    pub fn new(a: usize, b: usize) -> Self {
        if a <= b {
            UndirectedEdge { a, b }
        } else {
            UndirectedEdge { a: b, b: a }
        }
    }
}

/// A directed edge in the CPDAG.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DirectedEdge {
    /// Source node
    pub from: usize,
    /// Target node
    pub to: usize,
}

/// Edge mark in a CPDAG: directed, undirected, or absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeMark {
    /// No edge
    Absent,
    /// Undirected (present in skeleton, not yet oriented)
    Undirected,
    /// Directed from row to column
    Directed,
}

/// Separation sets: for each pair (i,j), the conditioning set that
/// rendered them conditionally independent.
pub type SepSets = HashMap<(usize, usize), BTreeSet<usize>>;

/// Result of the PC algorithm.
#[derive(Clone, Debug)]
pub struct PCResult {
    /// Variable names
    pub variables: Vec<String>,
    /// Adjacency matrix: adj[i][j] = EdgeMark
    pub adjacency: Vec<Vec<EdgeMark>>,
    /// Separation sets found during skeleton discovery
    pub sep_sets: SepSets,
    /// Number of independence tests performed
    pub n_tests: usize,
    /// Maximum conditioning set size reached
    pub max_cond_reached: usize,
}

impl PCResult {
    /// Check if there is any edge (directed or undirected) between i and j.
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        matches!(
            self.adjacency[i][j],
            EdgeMark::Directed | EdgeMark::Undirected
        ) || matches!(
            self.adjacency[j][i],
            EdgeMark::Directed | EdgeMark::Undirected
        )
    }

    /// Check if there is a directed edge from i to j.
    pub fn has_directed(&self, from: usize, to: usize) -> bool {
        self.adjacency[from][to] == EdgeMark::Directed
    }

    /// Get all directed edges.
    pub fn directed_edges(&self) -> Vec<DirectedEdge> {
        let n = self.variables.len();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if self.adjacency[i][j] == EdgeMark::Directed {
                    edges.push(DirectedEdge { from: i, to: j });
                }
            }
        }
        edges
    }

    /// Get all undirected edges.
    pub fn undirected_edges(&self) -> Vec<UndirectedEdge> {
        let n = self.variables.len();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.adjacency[i][j] == EdgeMark::Undirected
                    && self.adjacency[j][i] == EdgeMark::Undirected
                {
                    edges.push(UndirectedEdge::new(i, j));
                }
            }
        }
        edges
    }

    /// Get the neighbors (adjacent nodes) of a given node.
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        let n = self.variables.len();
        let mut nbrs = Vec::new();
        for j in 0..n {
            if j != node && self.has_edge(node, j) {
                nbrs.push(j);
            }
        }
        nbrs
    }
}

/// Errors from the PC algorithm.
#[derive(Clone, Debug)]
pub enum PCError {
    /// Not enough data samples
    InsufficientData {
        n_samples: usize,
        min_required: usize,
    },
    /// Row dimension mismatch
    DimensionMismatch {
        row: usize,
        expected: usize,
        got: usize,
    },
    /// Singular matrix encountered during partial correlation
    SingularMatrix,
    /// Numerical error
    NumericalError { message: String },
}

impl std::fmt::Display for PCError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PCError::InsufficientData {
                n_samples,
                min_required,
            } => write!(
                f,
                "insufficient data: {} samples, need at least {}",
                n_samples, min_required
            ),
            PCError::DimensionMismatch { row, expected, got } => {
                write!(f, "row {} has {} columns, expected {}", row, got, expected)
            }
            PCError::SingularMatrix => write!(f, "singular matrix in partial correlation"),
            PCError::NumericalError { message } => write!(f, "numerical error: {}", message),
        }
    }
}

impl std::error::Error for PCError {}

// ============================================================================
// PC Algorithm — Main Entry Point
// ============================================================================

/// Run the PC algorithm on observational data.
///
/// Returns a CPDAG (Completed Partially Directed Acyclic Graph) representing
/// the Markov equivalence class of DAGs consistent with the data.
///
/// # Arguments
///
/// * `data` — Observational data matrix (samples x variables)
/// * `config` — Algorithm configuration (significance level, etc.)
///
/// # Example
///
/// ```ignore
/// let data = DataMatrix::new(
///     vec!["X".into(), "Y".into(), "Z".into()],
///     generate_chain_data(1000),
/// )?;
/// let config = PCConfig { alpha: 0.05, ..Default::default() };
/// let result = pc_algorithm(&data, &config)?;
/// ```
pub fn pc_algorithm(data: &DataMatrix, config: &PCConfig) -> Result<PCResult, PCError> {
    let n = data.n_vars;

    // Need at least 4 samples for meaningful correlation
    if data.n_samples < 4 {
        return Err(PCError::InsufficientData {
            n_samples: data.n_samples,
            min_required: 4,
        });
    }

    // Step 0: Compute correlation matrix once
    let corr = correlation_matrix(data)?;

    // Step 1: Discover skeleton
    let (adj, sep_sets, n_tests, max_cond) = discover_skeleton(data, config, &corr)?;

    // Step 2: Orient v-structures
    let mut adjacency = orient_v_structures(n, &adj, &sep_sets);

    // Step 3: Apply Meek's rules
    apply_meek_rules(&mut adjacency, n);

    Ok(PCResult {
        variables: data.variables.clone(),
        adjacency,
        sep_sets,
        n_tests,
        max_cond_reached: max_cond,
    })
}

// ============================================================================
// Phase 1: Skeleton Discovery
// ============================================================================

/// Discover the skeleton (undirected graph) by removing edges where
/// conditional independence is detected.
///
/// Returns (adjacency, separation_sets, n_tests, max_conditioning_size).
fn discover_skeleton(
    data: &DataMatrix,
    config: &PCConfig,
    corr: &[Vec<f64>],
) -> Result<(Vec<Vec<bool>>, SepSets, usize, usize), PCError> {
    let n = data.n_vars;

    // Start with complete undirected graph
    let mut adj = vec![vec![true; n]; n];
    for i in 0..n {
        adj[i][i] = false;
    }

    let mut sep_sets: SepSets = HashMap::new();
    let mut n_tests: usize = 0;
    let mut max_cond: usize = 0;

    // Increase conditioning set size from 0 upward
    let mut cond_size = 0;
    loop {
        // Check termination: max conditioning size
        if let Some(max) = config.max_cond_size {
            if cond_size > max {
                break;
            }
        }
        if config.max_depth >= 0 && cond_size > config.max_depth as usize {
            break;
        }

        let mut found_testable = false;

        // Snapshot adjacency for stable variant
        let adj_snapshot = if config.stable {
            adj.clone()
        } else {
            Vec::new()
        };

        // Collect edges to remove (for stable variant)
        let mut removals: Vec<(usize, usize, BTreeSet<usize>)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                // Use snapshot for stable variant, current for default
                let check_adj = if config.stable { &adj_snapshot } else { &adj };
                if !check_adj[i][j] {
                    continue;
                }

                // Get adjacencies of i excluding j
                let nbrs_i: Vec<usize> = (0..n)
                    .filter(|&k| k != i && k != j && check_adj[i][k])
                    .collect();

                if nbrs_i.len() < cond_size {
                    continue;
                }

                found_testable = true;

                // Test all subsets of size cond_size from neighbors of i
                let subsets = subsets_of_size(&nbrs_i, cond_size);
                for subset in &subsets {
                    n_tests += 1;
                    let cond_set: Vec<usize> = subset.iter().copied().collect();

                    let (independent, _p_value) =
                        conditional_independence_test(data, i, j, &cond_set, corr, config.alpha)?;

                    if independent {
                        let sep: BTreeSet<usize> = subset.clone();
                        if config.stable {
                            removals.push((i, j, sep));
                        } else {
                            adj[i][j] = false;
                            adj[j][i] = false;
                            let sep_clone: BTreeSet<usize> = subset.clone();
                            sep_sets.insert((i, j), sep_clone.clone());
                            sep_sets.insert((j, i), sep_clone);
                        }
                        break;
                    }
                }

                // Also test from j's neighbors
                if config.stable && adj_snapshot[i][j] || !config.stable && adj[i][j] {
                    let nbrs_j: Vec<usize> = (0..n)
                        .filter(|&k| {
                            let check = if config.stable { &adj_snapshot } else { &adj };
                            k != i && k != j && check[j][k]
                        })
                        .collect();

                    if nbrs_j.len() >= cond_size {
                        let subsets_j = subsets_of_size(&nbrs_j, cond_size);
                        for subset in &subsets_j {
                            n_tests += 1;
                            let cond_set: Vec<usize> = subset.iter().copied().collect();

                            let (independent, _p_value) = conditional_independence_test(
                                data,
                                i,
                                j,
                                &cond_set,
                                corr,
                                config.alpha,
                            )?;

                            if independent {
                                let sep: BTreeSet<usize> = subset.clone();
                                if config.stable {
                                    removals.push((i, j, sep));
                                } else {
                                    adj[i][j] = false;
                                    adj[j][i] = false;
                                    let sep_clone: BTreeSet<usize> = subset.clone();
                                    sep_sets.insert((i, j), sep_clone.clone());
                                    sep_sets.insert((j, i), sep_clone);
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Apply removals for stable variant
        if config.stable {
            for (i, j, sep) in removals {
                adj[i][j] = false;
                adj[j][i] = false;
                sep_sets.insert((i, j), sep.clone());
                sep_sets.insert((j, i), sep);
            }
        }

        if !found_testable {
            break;
        }

        max_cond = cond_size;
        cond_size += 1;
    }

    Ok((adj, sep_sets, n_tests, max_cond))
}

// ============================================================================
// Phase 2: V-Structure Orientation
// ============================================================================

/// Orient v-structures (colliders): for each triple X - Z - Y where
/// X and Y are non-adjacent, orient as X → Z ← Y if Z is NOT in
/// sep(X, Y).
fn orient_v_structures(n: usize, adj: &[Vec<bool>], sep_sets: &SepSets) -> Vec<Vec<EdgeMark>> {
    let mut mark = vec![vec![EdgeMark::Absent; n]; n];

    // Initialize with undirected edges from skeleton
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i][j] {
                mark[i][j] = EdgeMark::Undirected;
                mark[j][i] = EdgeMark::Undirected;
            }
        }
    }

    // Find v-structures
    for z in 0..n {
        // Get all nodes adjacent to z
        let neighbors: Vec<usize> = (0..n).filter(|&k| adj[z][k]).collect();

        // Check all pairs of neighbors
        for idx_a in 0..neighbors.len() {
            for idx_b in (idx_a + 1)..neighbors.len() {
                let x = neighbors[idx_a];
                let y = neighbors[idx_b];

                // x and y must NOT be adjacent
                if adj[x][y] {
                    continue;
                }

                // Check if z is in sep(x, y)
                let key = if x < y { (x, y) } else { (y, x) };
                let z_in_sep = sep_sets.get(&key).map_or(false, |s| s.contains(&z));

                if !z_in_sep {
                    // Orient as x -> z <- y (collider)
                    mark[x][z] = EdgeMark::Directed;
                    mark[z][x] = EdgeMark::Absent;
                    mark[y][z] = EdgeMark::Directed;
                    mark[z][y] = EdgeMark::Absent;
                }
            }
        }
    }

    mark
}

// ============================================================================
// Phase 3: Meek's Rules
// ============================================================================

/// Apply Meek's orientation rules R1-R4 until no more edges can be oriented.
///
/// - R1: X → Y — Z  ⇒  X → Y → Z  (if X and Z not adjacent)
/// - R2: X → Z → Y  with X — Y  ⇒  X → Y
/// - R3: X — Y with two directed paths X → Z₁ → Y and X → Z₂ → Y
///        where Z₁ and Z₂ are not adjacent  ⇒  X → Y
/// - R4: X — Y with X — Z → W → Y  ⇒  X → Y  (if X and W not adjacent)
fn apply_meek_rules(adjacency: &mut [Vec<EdgeMark>], n: usize) {
    let mut changed = true;
    let mut iterations = 0;
    let max_iterations = n * n; // Safety bound

    while changed && iterations < max_iterations {
        changed = false;
        iterations += 1;

        // Rule R1: X → Y — Z, X and Z not adjacent ⇒ Y → Z
        for x in 0..n {
            for y in 0..n {
                if adjacency[x][y] != EdgeMark::Directed {
                    continue;
                }
                for z in 0..n {
                    if z == x || z == y {
                        continue;
                    }
                    if adjacency[y][z] == EdgeMark::Undirected
                        && adjacency[z][y] == EdgeMark::Undirected
                        && adjacency[x][z] == EdgeMark::Absent
                        && adjacency[z][x] == EdgeMark::Absent
                    {
                        adjacency[y][z] = EdgeMark::Directed;
                        adjacency[z][y] = EdgeMark::Absent;
                        changed = true;
                    }
                }
            }
        }

        // Rule R2: X → Z → Y with X — Y ⇒ X → Y
        for x in 0..n {
            for y in 0..n {
                if x == y {
                    continue;
                }
                if adjacency[x][y] != EdgeMark::Undirected
                    || adjacency[y][x] != EdgeMark::Undirected
                {
                    continue;
                }
                // Find z such that X → Z → Y
                for z in 0..n {
                    if z == x || z == y {
                        continue;
                    }
                    if adjacency[x][z] == EdgeMark::Directed
                        && adjacency[z][y] == EdgeMark::Directed
                    {
                        adjacency[x][y] = EdgeMark::Directed;
                        adjacency[y][x] = EdgeMark::Absent;
                        changed = true;
                        break;
                    }
                }
            }
        }

        // Rule R3: X — Y, two paths X — Z₁ → Y and X — Z₂ → Y,
        //          Z₁ and Z₂ not adjacent ⇒ X → Y
        for x in 0..n {
            for y in 0..n {
                if x == y {
                    continue;
                }
                if adjacency[x][y] != EdgeMark::Undirected
                    || adjacency[y][x] != EdgeMark::Undirected
                {
                    continue;
                }
                // Collect z's where X — Z → Y
                let intermediates: Vec<usize> = (0..n)
                    .filter(|&z| {
                        z != x
                            && z != y
                            && adjacency[x][z] == EdgeMark::Undirected
                            && adjacency[z][x] == EdgeMark::Undirected
                            && adjacency[z][y] == EdgeMark::Directed
                    })
                    .collect();

                // Check if any two intermediates are non-adjacent
                let mut orient = false;
                for ia in 0..intermediates.len() {
                    for ib in (ia + 1)..intermediates.len() {
                        let z1 = intermediates[ia];
                        let z2 = intermediates[ib];
                        if adjacency[z1][z2] == EdgeMark::Absent
                            && adjacency[z2][z1] == EdgeMark::Absent
                        {
                            orient = true;
                            break;
                        }
                    }
                    if orient {
                        break;
                    }
                }

                if orient {
                    adjacency[x][y] = EdgeMark::Directed;
                    adjacency[y][x] = EdgeMark::Absent;
                    changed = true;
                }
            }
        }

        // Rule R4: X — Y, X — Z → W → Y, X and W not adjacent ⇒ X → Y
        for x in 0..n {
            for y in 0..n {
                if x == y {
                    continue;
                }
                if adjacency[x][y] != EdgeMark::Undirected
                    || adjacency[y][x] != EdgeMark::Undirected
                {
                    continue;
                }
                let mut orient = false;
                for z in 0..n {
                    if z == x || z == y {
                        continue;
                    }
                    // X — Z (undirected)
                    if adjacency[x][z] != EdgeMark::Undirected
                        || adjacency[z][x] != EdgeMark::Undirected
                    {
                        continue;
                    }
                    for w in 0..n {
                        if w == x || w == y || w == z {
                            continue;
                        }
                        // Z → W → Y
                        if adjacency[z][w] == EdgeMark::Directed
                            && adjacency[w][y] == EdgeMark::Directed
                            // X and W not adjacent
                            && adjacency[x][w] == EdgeMark::Absent
                            && adjacency[w][x] == EdgeMark::Absent
                        {
                            orient = true;
                            break;
                        }
                    }
                    if orient {
                        break;
                    }
                }
                if orient {
                    adjacency[x][y] = EdgeMark::Directed;
                    adjacency[y][x] = EdgeMark::Absent;
                    changed = true;
                }
            }
        }
    }
}

// ============================================================================
// Statistical Tests
// ============================================================================

/// Test conditional independence of X ⊥⊥ Y | Z using partial correlation
/// and Fisher's z-transform.
///
/// Returns (is_independent, p_value).
fn conditional_independence_test(
    data: &DataMatrix,
    x: usize,
    y: usize,
    cond_set: &[usize],
    corr: &[Vec<f64>],
    alpha: f64,
) -> Result<(bool, f64), PCError> {
    let n = data.n_samples;

    if cond_set.is_empty() {
        // Marginal independence test
        let r = corr[x][y];
        let z_stat = fisher_z(r, n, 0);
        let p = z_to_pvalue(z_stat);
        return Ok((p > alpha, p));
    }

    // Partial correlation via matrix inversion
    let r_partial = partial_correlation(x, y, cond_set, corr)?;

    let z_stat = fisher_z(r_partial, n, cond_set.len());
    let p = z_to_pvalue(z_stat);

    Ok((p > alpha, p))
}

/// Compute Pearson correlation matrix from data.
pub fn correlation_matrix(data: &DataMatrix) -> Result<Vec<Vec<f64>>, PCError> {
    let n = data.n_samples;
    let p = data.n_vars;

    // Compute means
    let mut means = vec![0.0; p];
    for row in &data.data {
        for (j, val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }

    // Compute correlation matrix
    let mut corr = vec![vec![0.0; p]; p];

    // Compute standard deviations
    let mut stds = vec![0.0; p];
    for row in &data.data {
        for (j, val) in row.iter().enumerate() {
            let d = val - means[j];
            stds[j] += d * d;
        }
    }
    for s in &mut stds {
        *s = (*s / (n - 1) as f64).sqrt();
        if *s < 1e-15 {
            *s = 1e-15; // Avoid division by zero
        }
    }

    // Compute correlations
    for i in 0..p {
        corr[i][i] = 1.0;
        for j in (i + 1)..p {
            let mut sum = 0.0;
            for row in &data.data {
                sum += (row[i] - means[i]) * (row[j] - means[j]);
            }
            let r = sum / ((n - 1) as f64 * stds[i] * stds[j]);
            // Clamp to [-1, 1] for numerical stability
            let r = r.max(-1.0).min(1.0);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    Ok(corr)
}

/// Compute partial correlation of X and Y given Z using matrix inversion.
///
/// Uses the formula: r_{xy|Z} = -P_{xy} / sqrt(P_{xx} * P_{yy})
/// where P = (R_S)^{-1} and S = {x, y} ∪ Z.
fn partial_correlation(
    x: usize,
    y: usize,
    cond_set: &[usize],
    corr: &[Vec<f64>],
) -> Result<f64, PCError> {
    // Build the submatrix indices: [x, y, z1, z2, ...]
    let mut indices = vec![x, y];
    for &z in cond_set {
        if z != x && z != y && !indices.contains(&z) {
            indices.push(z);
        }
    }
    let k = indices.len();

    // Extract sub-correlation matrix
    let mut sub = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            sub[i][j] = corr[indices[i]][indices[j]];
        }
    }

    // Invert the sub-matrix
    let inv = invert_matrix(&sub)?;

    // Partial correlation: -P_{01} / sqrt(P_{00} * P_{11})
    let denom = (inv[0][0] * inv[1][1]).sqrt();
    if denom < 1e-15 {
        return Err(PCError::NumericalError {
            message: "degenerate partial correlation denominator".into(),
        });
    }

    let r = -inv[0][1] / denom;
    // Clamp for numerical stability
    Ok(r.max(-1.0 + 1e-10).min(1.0 - 1e-10))
}

/// Invert a square matrix using Gauss-Jordan elimination.
fn invert_matrix(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PCError> {
    let n = m.len();
    // Augment with identity
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(PCError::SingularMatrix);
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Ok(inv)
}

/// Fisher's z-transform: z = √(n - |Z| - 3) * 0.5 * ln((1+r)/(1-r))
fn fisher_z(r: f64, n: usize, cond_size: usize) -> f64 {
    let dof = n as f64 - cond_size as f64 - 3.0;
    if dof <= 0.0 {
        return 0.0;
    }
    let r_clamped = r.max(-1.0 + 1e-10).min(1.0 - 1e-10);
    let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    dof.sqrt() * z
}

/// Convert z-statistic to two-sided p-value.
fn z_to_pvalue(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Standard normal CDF approximation (Abramowitz & Stegun, formula 26.2.17).
///
/// Absolute error < 7.5e-8.
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let t = x.abs();

    // Horner form of the rational approximation
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;
    let p = 0.2316419;

    let k = 1.0 / (1.0 + p * t);
    let poly = k * (b1 + k * (b2 + k * (b3 + k * (b4 + k * b5))));
    let pdf = (-0.5 * t * t).exp() / (2.0 * std::f64::consts::PI).sqrt();

    0.5 + sign * (0.5 - pdf * poly)
}

// ============================================================================
// Conversion to CausalGraphDef
// ============================================================================

/// Convert a PC algorithm result to a CausalGraphDef for integration
/// with the Sounio type system.
///
/// Only directed edges are included (undirected edges in the CPDAG
/// represent edges whose direction could not be determined).
pub fn pc_result_to_graph(result: &PCResult, name: &str) -> CausalGraphDef {
    let dummy_span = Span::new(0, 0);

    let nodes: Vec<CausalNodeDef> = result
        .variables
        .iter()
        .map(|v| CausalNodeDef {
            name: v.clone(),
            span: dummy_span,
        })
        .collect();

    let mut edges = Vec::new();
    let n = result.variables.len();
    for i in 0..n {
        for j in 0..n {
            if result.adjacency[i][j] == EdgeMark::Directed {
                edges.push(CausalEdgeDef {
                    from: result.variables[i].clone(),
                    to: result.variables[j].clone(),
                    span: dummy_span,
                });
            }
        }
    }

    // Compute topological order from directed edges
    let topo = compute_topological_order(&result.variables, &edges);

    CausalGraphDef {
        name: name.to_string(),
        nodes,
        edges,
        topological_order: topo,
        span: dummy_span,
    }
}

/// Compute topological order from directed edges using Kahn's algorithm.
fn compute_topological_order(variables: &[String], edges: &[CausalEdgeDef]) -> Option<Vec<String>> {
    let n = variables.len();
    let var_idx: HashMap<&str, usize> = variables
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i))
        .collect();

    let mut in_degree = vec![0usize; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];

    for e in edges {
        if let (Some(&from), Some(&to)) = (var_idx.get(e.from.as_str()), var_idx.get(e.to.as_str()))
        {
            in_degree[to] += 1;
            children[from].push(to);
        }
    }

    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        order.push(variables[node].clone());
        for &child in &children[node] {
            in_degree[child] -= 1;
            if in_degree[child] == 0 {
                queue.push_back(child);
            }
        }
    }

    if order.len() == n {
        Some(order)
    } else {
        None // Cycle detected
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Generate all subsets of `items` with exactly `size` elements.
fn subsets_of_size(items: &[usize], size: usize) -> Vec<BTreeSet<usize>> {
    if size == 0 {
        return vec![BTreeSet::new()];
    }
    if items.len() < size {
        return Vec::new();
    }

    let mut result = Vec::new();
    generate_subsets(items, size, 0, &mut BTreeSet::new(), &mut result);
    result
}

fn generate_subsets(
    items: &[usize],
    size: usize,
    start: usize,
    current: &mut BTreeSet<usize>,
    result: &mut Vec<BTreeSet<usize>>,
) {
    if current.len() == size {
        result.push(current.clone());
        return;
    }
    if start >= items.len() {
        return;
    }
    // Pruning: not enough remaining items
    if items.len() - start < size - current.len() {
        return;
    }

    // Include items[start]
    current.insert(items[start]);
    generate_subsets(items, size, start + 1, current, result);
    current.remove(&items[start]);

    // Exclude items[start]
    generate_subsets(items, size, start + 1, current, result);
}

// ============================================================================
// Synthetic Data Generation (for testing)
// ============================================================================

/// Linear congruential generator for deterministic pseudorandom numbers.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Map to [0, 1)
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Generate standard normal variate using Box-Muller transform.
fn randn(state: &mut u64) -> f64 {
    let u1 = lcg_next(state).max(1e-10);
    let u2 = lcg_next(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate synthetic data from a chain: X → Y → Z
/// with linear Gaussian structural equations.
pub fn generate_chain_data(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let x = randn(&mut state);
        let y = 0.8 * x + 0.3 * randn(&mut state);
        let z = 0.7 * y + 0.3 * randn(&mut state);
        data.push(vec![x, y, z]);
    }
    data
}

/// Generate synthetic data from a fork: X ← Z → Y
pub fn generate_fork_data(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let z = randn(&mut state);
        let x = 0.8 * z + 0.3 * randn(&mut state);
        let y = 0.7 * z + 0.3 * randn(&mut state);
        data.push(vec![x, y, z]);
    }
    data
}

/// Generate synthetic data from a collider: X → Z ← Y
pub fn generate_collider_data(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let x = randn(&mut state);
        let y = randn(&mut state);
        let z = 0.6 * x + 0.7 * y + 0.3 * randn(&mut state);
        data.push(vec![x, y, z]);
    }
    data
}

/// Generate synthetic data from a diamond: X → M1, X → M2, M1 → Y, M2 → Y
pub fn generate_diamond_data(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let x = randn(&mut state);
        let m1 = 0.8 * x + 0.3 * randn(&mut state);
        let m2 = 0.7 * x + 0.3 * randn(&mut state);
        let y = 0.5 * m1 + 0.6 * m2 + 0.3 * randn(&mut state);
        data.push(vec![x, m1, m2, y]);
    }
    data
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Data Matrix Tests ---

    #[test]
    fn test_data_matrix_creation() {
        let dm = DataMatrix::new(
            vec!["X".into(), "Y".into()],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        );
        assert!(dm.is_ok());
        let dm = dm.unwrap();
        assert_eq!(dm.n_samples, 2);
        assert_eq!(dm.n_vars, 2);
    }

    #[test]
    fn test_data_matrix_empty() {
        let dm = DataMatrix::new(vec!["X".into()], vec![]);
        assert!(dm.is_err());
    }

    #[test]
    fn test_data_matrix_dimension_mismatch() {
        let dm = DataMatrix::new(
            vec!["X".into(), "Y".into()],
            vec![vec![1.0, 2.0], vec![3.0]],
        );
        assert!(dm.is_err());
    }

    #[test]
    fn test_data_matrix_var_index() {
        let dm = DataMatrix::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![vec![1.0, 2.0, 3.0]],
        )
        .unwrap();
        assert_eq!(dm.var_index("X"), Some(0));
        assert_eq!(dm.var_index("Y"), Some(1));
        assert_eq!(dm.var_index("Z"), Some(2));
        assert_eq!(dm.var_index("W"), None);
    }

    #[test]
    fn test_data_matrix_column() {
        let dm = DataMatrix::new(
            vec!["X".into(), "Y".into()],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        )
        .unwrap();
        assert_eq!(dm.column(0), vec![1.0, 3.0, 5.0]);
        assert_eq!(dm.column(1), vec![2.0, 4.0, 6.0]);
    }

    // --- Correlation Tests ---

    #[test]
    fn test_correlation_matrix_identity() {
        // Independent columns
        let data = DataMatrix::new(
            vec!["X".into(), "Y".into()],
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![-1.0, 0.0],
                vec![0.0, -1.0],
            ],
        )
        .unwrap();
        let corr = correlation_matrix(&data).unwrap();
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);
        assert!(corr[0][1].abs() < 1e-10);
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let data = DataMatrix::new(
            vec!["X".into(), "Y".into()],
            vec![
                vec![1.0, 2.0],
                vec![2.0, 4.0],
                vec![3.0, 6.0],
                vec![4.0, 8.0],
            ],
        )
        .unwrap();
        let corr = correlation_matrix(&data).unwrap();
        assert!((corr[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let data = DataMatrix::new(
            vec!["X".into(), "Y".into()],
            vec![
                vec![1.0, -1.0],
                vec![2.0, -2.0],
                vec![3.0, -3.0],
                vec![4.0, -4.0],
            ],
        )
        .unwrap();
        let corr = correlation_matrix(&data).unwrap();
        assert!((corr[0][1] + 1.0).abs() < 1e-10);
    }

    // --- Partial Correlation Tests ---

    #[test]
    fn test_partial_correlation_chain() {
        // X → Y → Z: controlling for Y should make X and Z independent
        let data_raw = generate_chain_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let corr = correlation_matrix(&data).unwrap();

        // Marginal correlation between X and Z should be significant
        assert!(corr[0][2].abs() > 0.1);

        // Partial correlation X,Z | Y should be near zero
        let r_partial = partial_correlation(0, 2, &[1], &corr).unwrap();
        assert!(
            r_partial.abs() < 0.1,
            "partial r(X,Z|Y) = {} should be small",
            r_partial
        );
    }

    // --- Fisher Z Tests ---

    #[test]
    fn test_fisher_z_zero() {
        let z = fisher_z(0.0, 100, 0);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_fisher_z_positive() {
        let z = fisher_z(0.5, 100, 0);
        assert!(z > 0.0);
    }

    #[test]
    fn test_fisher_z_negative() {
        let z = fisher_z(-0.5, 100, 0);
        assert!(z < 0.0);
    }

    #[test]
    fn test_fisher_z_insufficient_dof() {
        let z = fisher_z(0.5, 3, 0);
        assert!(z.abs() < 1e-10); // dof = 0
    }

    // --- Normal CDF Tests ---

    #[test]
    fn test_normal_cdf_zero() {
        let p = normal_cdf(0.0);
        assert!((p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf_positive() {
        let p = normal_cdf(1.96);
        assert!((p - 0.975).abs() < 0.001);
    }

    #[test]
    fn test_normal_cdf_negative() {
        let p = normal_cdf(-1.96);
        assert!((p - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_normal_cdf_extremes() {
        assert!(normal_cdf(-10.0) < 1e-6);
        assert!(normal_cdf(10.0) > 1.0 - 1e-6);
    }

    // --- CI Test ---

    #[test]
    fn test_ci_test_independent() {
        // Two genuinely independent variables
        let mut state: u64 = 123;
        let n = 500;
        let mut data_raw = Vec::with_capacity(n);
        for _ in 0..n {
            let x = randn(&mut state);
            let y = randn(&mut state);
            data_raw.push(vec![x, y]);
        }
        let data = DataMatrix::new(vec!["X".into(), "Y".into()], data_raw).unwrap();
        let corr = correlation_matrix(&data).unwrap();

        let (indep, p) = conditional_independence_test(&data, 0, 1, &[], &corr, 0.05).unwrap();
        assert!(
            indep,
            "independent variables should test as independent, p={}",
            p
        );
    }

    #[test]
    fn test_ci_test_dependent() {
        // Two strongly dependent variables
        let mut state: u64 = 456;
        let n = 500;
        let mut data_raw = Vec::with_capacity(n);
        for _ in 0..n {
            let x = randn(&mut state);
            let y = 0.9 * x + 0.1 * randn(&mut state);
            data_raw.push(vec![x, y]);
        }
        let data = DataMatrix::new(vec!["X".into(), "Y".into()], data_raw).unwrap();
        let corr = correlation_matrix(&data).unwrap();

        let (indep, _p) = conditional_independence_test(&data, 0, 1, &[], &corr, 0.05).unwrap();
        assert!(!indep, "dependent variables should not test as independent");
    }

    // --- Matrix Inversion Tests ---

    #[test]
    fn test_invert_identity() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let inv = invert_matrix(&m).unwrap();
        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[1][1] - 1.0).abs() < 1e-10);
        assert!(inv[0][1].abs() < 1e-10);
        assert!(inv[1][0].abs() < 1e-10);
    }

    #[test]
    fn test_invert_2x2() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let inv = invert_matrix(&m).unwrap();
        // Check M * M^{-1} = I
        let prod00 = m[0][0] * inv[0][0] + m[0][1] * inv[1][0];
        let prod01 = m[0][0] * inv[0][1] + m[0][1] * inv[1][1];
        let prod10 = m[1][0] * inv[0][0] + m[1][1] * inv[1][0];
        let prod11 = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];
        assert!((prod00 - 1.0).abs() < 1e-10);
        assert!(prod01.abs() < 1e-10);
        assert!(prod10.abs() < 1e-10);
        assert!((prod11 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_singular() {
        let m = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(invert_matrix(&m).is_err());
    }

    // --- Subset Generation Tests ---

    #[test]
    fn test_subsets_size_0() {
        let subs = subsets_of_size(&[1, 2, 3], 0);
        assert_eq!(subs.len(), 1);
        assert!(subs[0].is_empty());
    }

    #[test]
    fn test_subsets_size_1() {
        let subs = subsets_of_size(&[1, 2, 3], 1);
        assert_eq!(subs.len(), 3);
    }

    #[test]
    fn test_subsets_size_2() {
        let subs = subsets_of_size(&[1, 2, 3], 2);
        assert_eq!(subs.len(), 3);
    }

    #[test]
    fn test_subsets_too_large() {
        let subs = subsets_of_size(&[1, 2], 3);
        assert!(subs.is_empty());
    }

    // --- Skeleton Discovery Tests ---

    #[test]
    fn test_skeleton_chain() {
        // X → Y → Z: skeleton should have X-Y and Y-Z but not X-Z
        let data_raw = generate_chain_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig {
            alpha: 0.05,
            ..Default::default()
        };
        let corr = correlation_matrix(&data).unwrap();
        let (adj, _sep, _n, _m) = discover_skeleton(&data, &config, &corr).unwrap();

        // X-Y should be present
        assert!(adj[0][1], "X-Y edge should be in skeleton");
        // Y-Z should be present
        assert!(adj[1][2], "Y-Z edge should be in skeleton");
        // X-Z should be removed (conditional on Y)
        assert!(!adj[0][2], "X-Z edge should be removed (chain)");
    }

    #[test]
    fn test_skeleton_fork() {
        // X ← Z → Y: skeleton should have Z-X and Z-Y but not X-Y
        let data_raw = generate_fork_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig {
            alpha: 0.05,
            ..Default::default()
        };
        let corr = correlation_matrix(&data).unwrap();
        let (adj, _sep, _n, _m) = discover_skeleton(&data, &config, &corr).unwrap();

        // Z-X should be present
        assert!(adj[0][2], "X-Z edge should be in skeleton");
        // Z-Y should be present
        assert!(adj[1][2], "Y-Z edge should be in skeleton");
        // X-Y should be removed (conditional on Z)
        assert!(!adj[0][1], "X-Y edge should be removed (fork)");
    }

    #[test]
    fn test_skeleton_collider() {
        // X → Z ← Y: skeleton should have X-Z and Y-Z but not X-Y
        let data_raw = generate_collider_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig {
            alpha: 0.05,
            ..Default::default()
        };
        let corr = correlation_matrix(&data).unwrap();
        let (adj, _sep, _n, _m) = discover_skeleton(&data, &config, &corr).unwrap();

        // X-Z should be present
        assert!(adj[0][2], "X-Z edge should be in skeleton");
        // Y-Z should be present
        assert!(adj[1][2], "Y-Z edge should be in skeleton");
        // X-Y should be absent (marginal independence)
        assert!(!adj[0][1], "X-Y edge should be absent (collider)");
    }

    // --- V-Structure Orientation Tests ---

    #[test]
    fn test_v_structure_collider() {
        // Collider: X → Z ← Y
        // Skeleton: X-Z, Y-Z (X and Y not adjacent)
        // sep(X,Y) = {} (empty), Z not in sep ⇒ orient as X→Z←Y
        let n = 3;
        let adj = vec![
            vec![false, false, true], // X: adj to Z
            vec![false, false, true], // Y: adj to Z
            vec![true, true, false],  // Z: adj to X, Y
        ];
        let sep_sets: SepSets = {
            let mut s = HashMap::new();
            s.insert((0, 1), BTreeSet::new());
            s.insert((1, 0), BTreeSet::new());
            s
        };

        let mark = orient_v_structures(n, &adj, &sep_sets);

        // X → Z
        assert_eq!(mark[0][2], EdgeMark::Directed);
        assert_eq!(mark[2][0], EdgeMark::Absent);
        // Y → Z
        assert_eq!(mark[1][2], EdgeMark::Directed);
        assert_eq!(mark[2][1], EdgeMark::Absent);
    }

    #[test]
    fn test_v_structure_non_collider() {
        // Chain: X-Y-Z with sep(X,Z) = {Y}
        // Y IS in sep(X,Z), so no v-structure
        let n = 3;
        let adj = vec![
            vec![false, true, false], // X: adj to Y
            vec![true, false, true],  // Y: adj to X, Z
            vec![false, true, false], // Z: adj to Y
        ];
        let sep_sets: SepSets = {
            let mut s = HashMap::new();
            let mut sep = BTreeSet::new();
            sep.insert(1); // Y is in sep(X,Z)
            s.insert((0, 2), sep.clone());
            s.insert((2, 0), sep);
            s
        };

        let mark = orient_v_structures(n, &adj, &sep_sets);

        // All edges should remain undirected (no v-structure)
        assert_eq!(mark[0][1], EdgeMark::Undirected);
        assert_eq!(mark[1][0], EdgeMark::Undirected);
        assert_eq!(mark[1][2], EdgeMark::Undirected);
        assert_eq!(mark[2][1], EdgeMark::Undirected);
    }

    // --- Meek Rules Tests ---

    #[test]
    fn test_meek_r1() {
        // R1: X → Y — Z (X,Z not adj) ⇒ Y → Z
        let n = 3;
        let mut adj = vec![vec![EdgeMark::Absent; n]; n];
        adj[0][1] = EdgeMark::Directed; // X → Y
        adj[1][2] = EdgeMark::Undirected; // Y — Z
        adj[2][1] = EdgeMark::Undirected;
        // X and Z not adjacent (already absent)

        apply_meek_rules(&mut adj, n);

        assert_eq!(adj[1][2], EdgeMark::Directed, "R1: Y → Z");
        assert_eq!(adj[2][1], EdgeMark::Absent, "R1: Z → Y removed");
    }

    #[test]
    fn test_meek_r2() {
        // R2: X → Z → Y with X — Y ⇒ X → Y
        let n = 3;
        let mut adj = vec![vec![EdgeMark::Absent; n]; n];
        adj[0][2] = EdgeMark::Directed; // X → Z
        adj[2][1] = EdgeMark::Directed; // Z → Y
        adj[0][1] = EdgeMark::Undirected; // X — Y
        adj[1][0] = EdgeMark::Undirected;

        apply_meek_rules(&mut adj, n);

        assert_eq!(adj[0][1], EdgeMark::Directed, "R2: X → Y");
        assert_eq!(adj[1][0], EdgeMark::Absent, "R2: Y → X removed");
    }

    // --- Full PC Algorithm Tests ---

    #[test]
    fn test_pc_chain() {
        let data_raw = generate_chain_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        // Skeleton: X-Y, Y-Z (no X-Z)
        assert!(result.has_edge(0, 1), "chain: X-Y edge");
        assert!(result.has_edge(1, 2), "chain: Y-Z edge");
        assert!(!result.has_edge(0, 2), "chain: no X-Z edge");
        assert!(result.n_tests > 0);
    }

    #[test]
    fn test_pc_fork() {
        // Data layout: [X, Y, Z] where Z is the common cause
        let data_raw = generate_fork_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        // Skeleton: X-Z, Y-Z (no X-Y)
        assert!(result.has_edge(0, 2), "fork: X-Z edge");
        assert!(result.has_edge(1, 2), "fork: Y-Z edge");
        assert!(!result.has_edge(0, 1), "fork: no X-Y edge");
    }

    #[test]
    fn test_pc_collider() {
        let data_raw = generate_collider_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        // Skeleton: X-Z, Y-Z (no X-Y)
        assert!(result.has_edge(0, 2), "collider: X-Z edge");
        assert!(result.has_edge(1, 2), "collider: Y-Z edge");
        assert!(!result.has_edge(0, 1), "collider: no X-Y edge");

        // V-structure: X → Z ← Y
        assert!(result.has_directed(0, 2), "collider: X → Z directed");
        assert!(result.has_directed(1, 2), "collider: Y → Z directed");
    }

    #[test]
    fn test_pc_diamond() {
        let data_raw = generate_diamond_data(3000, 42);
        let data = DataMatrix::new(
            vec!["X".into(), "M1".into(), "M2".into(), "Y".into()],
            data_raw,
        )
        .unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        // X should connect to M1 and M2
        assert!(result.has_edge(0, 1), "diamond: X-M1 edge");
        assert!(result.has_edge(0, 2), "diamond: X-M2 edge");
        // M1 and M2 should connect to Y
        assert!(result.has_edge(1, 3), "diamond: M1-Y edge");
        assert!(result.has_edge(2, 3), "diamond: M2-Y edge");
    }

    #[test]
    fn test_pc_independent() {
        // Two completely independent variables
        let mut state: u64 = 789;
        let n = 500;
        let mut data_raw = Vec::with_capacity(n);
        for _ in 0..n {
            let x = randn(&mut state);
            let y = randn(&mut state);
            data_raw.push(vec![x, y]);
        }
        let data = DataMatrix::new(vec!["X".into(), "Y".into()], data_raw).unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        // No edges
        assert!(!result.has_edge(0, 1), "independent: no edge");
    }

    // --- Error Cases ---

    #[test]
    fn test_pc_insufficient_data() {
        let data = DataMatrix::new(
            vec!["X".into(), "Y".into()],
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        )
        .unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config);
        assert!(result.is_err());
    }

    // --- Conversion Tests ---

    #[test]
    fn test_pc_result_to_graph() {
        let data_raw = generate_collider_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        let graph = pc_result_to_graph(&result, "test_graph");
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.nodes.len(), 3);
        // Should have directed edges
        assert!(!graph.edges.is_empty());
    }

    #[test]
    fn test_pc_result_to_graph_topological_order() {
        let data_raw = generate_chain_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        let graph = pc_result_to_graph(&result, "chain_graph");
        // If directed edges form a DAG, topological order should exist
        if !graph.edges.is_empty() {
            // May or may not have topo order depending on orientation
            // Just check it doesn't crash
            let _ = graph.topological_order;
        }
    }

    // --- PCResult Method Tests ---

    #[test]
    fn test_pc_result_methods() {
        let data_raw = generate_collider_data(2000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig::default();
        let result = pc_algorithm(&data, &config).unwrap();

        let directed = result.directed_edges();
        assert!(!directed.is_empty());

        let nbrs = result.neighbors(2); // Z's neighbors
        assert!(!nbrs.is_empty());
    }

    // --- PCConfig Tests ---

    #[test]
    fn test_pc_config_default() {
        let config = PCConfig::default();
        assert!((config.alpha - 0.05).abs() < 1e-10);
        assert!(config.stable);
        assert!(config.max_cond_size.is_none());
    }

    #[test]
    fn test_pc_config_max_depth() {
        let data_raw = generate_chain_data(1000, 42);
        let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
        let config = PCConfig {
            max_depth: 0,
            ..Default::default()
        };
        let result = pc_algorithm(&data, &config).unwrap();
        assert_eq!(result.max_cond_reached, 0);
    }

    // --- Undirected Edge Tests ---

    #[test]
    fn test_undirected_edge_normalization() {
        let e1 = UndirectedEdge::new(3, 1);
        let e2 = UndirectedEdge::new(1, 3);
        assert_eq!(e1, e2);
        assert_eq!(e1.a, 1);
        assert_eq!(e1.b, 3);
    }

    // --- RNG Tests ---

    #[test]
    fn test_lcg_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        let v1 = lcg_next(&mut s1);
        let v2 = lcg_next(&mut s2);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_randn_distribution() {
        // Basic sanity: mean should be near 0
        let mut state: u64 = 12345;
        let n = 10000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += randn(&mut state);
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "randn mean = {}", mean);
    }

    // --- Synthetic Data Tests ---

    #[test]
    fn test_generate_chain_data() {
        let data = generate_chain_data(100, 42);
        assert_eq!(data.len(), 100);
        assert_eq!(data[0].len(), 3);
    }

    #[test]
    fn test_generate_fork_data() {
        let data = generate_fork_data(100, 42);
        assert_eq!(data.len(), 100);
        assert_eq!(data[0].len(), 3);
    }

    #[test]
    fn test_generate_collider_data() {
        let data = generate_collider_data(100, 42);
        assert_eq!(data.len(), 100);
        assert_eq!(data[0].len(), 3);
    }

    #[test]
    fn test_generate_diamond_data() {
        let data = generate_diamond_data(100, 42);
        assert_eq!(data.len(), 100);
        assert_eq!(data[0].len(), 4);
    }

    // --- PCError Display Tests ---

    #[test]
    fn test_pc_error_display() {
        let e = PCError::InsufficientData {
            n_samples: 2,
            min_required: 4,
        };
        let s = format!("{}", e);
        assert!(s.contains("insufficient data"));

        let e = PCError::SingularMatrix;
        let s = format!("{}", e);
        assert!(s.contains("singular"));
    }

    // --- Accuracy Benchmark (meta-test) ---

    #[test]
    fn test_pc_accuracy_chain_above_80_percent() {
        // Run PC on chain data multiple times with different seeds
        let mut correct = 0;
        let total = 10;
        for seed in 0..total {
            let data_raw = generate_chain_data(2000, 100 + seed);
            let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
            let config = PCConfig::default();
            let result = pc_algorithm(&data, &config).unwrap();

            // Check skeleton: X-Y, Y-Z present; X-Z absent
            let has_xy = result.has_edge(0, 1);
            let has_yz = result.has_edge(1, 2);
            let no_xz = !result.has_edge(0, 2);

            if has_xy && has_yz && no_xz {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy >= 0.8,
            "chain accuracy = {:.1}%, expected >= 80%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_pc_accuracy_fork_above_80_percent() {
        let mut correct = 0;
        let total = 10;
        for seed in 0..total {
            let data_raw = generate_fork_data(2000, 200 + seed);
            let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
            let config = PCConfig::default();
            let result = pc_algorithm(&data, &config).unwrap();

            let has_xz = result.has_edge(0, 2);
            let has_yz = result.has_edge(1, 2);
            let no_xy = !result.has_edge(0, 1);

            if has_xz && has_yz && no_xy {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy >= 0.8,
            "fork accuracy = {:.1}%, expected >= 80%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_pc_accuracy_collider_above_80_percent() {
        let mut correct = 0;
        let total = 10;
        for seed in 0..total {
            let data_raw = generate_collider_data(2000, 300 + seed);
            let data = DataMatrix::new(vec!["X".into(), "Y".into(), "Z".into()], data_raw).unwrap();
            let config = PCConfig::default();
            let result = pc_algorithm(&data, &config).unwrap();

            let has_xz = result.has_edge(0, 2);
            let has_yz = result.has_edge(1, 2);
            let no_xy = !result.has_edge(0, 1);
            // Collider orientation
            let x_to_z = result.has_directed(0, 2);
            let y_to_z = result.has_directed(1, 2);

            if has_xz && has_yz && no_xy && x_to_z && y_to_z {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy >= 0.8,
            "collider accuracy = {:.1}%, expected >= 80%",
            accuracy * 100.0
        );
    }
}
