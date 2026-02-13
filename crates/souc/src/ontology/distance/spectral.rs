//! Spectral Distance on Ontology Graphs
//!
//! Uses spectral graph theory to compute smooth semantic distances.
//! The key insight is that the graph Laplacian's eigendecomposition
//! provides a natural "frequency" basis for signals on the graph.
//!
//! # The Graph Laplacian
//!
//! For an ontology graph G with adjacency matrix A and degree matrix D:
//! - Unnormalized Laplacian: L = D - A
//! - Normalized Laplacian: L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2)
//!
//! The eigenvalues 0 = λ₀ ≤ λ₁ ≤ ... ≤ λₙ measure graph connectivity:
//! - λ₁ (Fiedler value): algebraic connectivity
//! - Small eigenvalues: global structure (coarse graph features)
//! - Large eigenvalues: local structure (fine graph features)
//!
//! # Spectral Distance
//!
//! Given eigenvectors φ₀, φ₁, ..., φₖ and eigenvalues λ₀, λ₁, ..., λₖ:
//!
//! ```text
//! d_spectral(u, v) = sqrt( Σᵢ w(λᵢ) * (φᵢ(u) - φᵢ(v))² )
//! ```
//!
//! where w(λ) is a weighting function. Common choices:
//! - w(λ) = 1: Euclidean in spectral embedding
//! - w(λ) = 1/λ: Commute time distance (random walk interpretation)
//! - w(λ) = exp(-tλ): Heat kernel distance (diffusion interpretation)
//!
//! # Heat Kernel Signature
//!
//! The heat kernel h_t(u, v) measures how heat diffuses from u to v at time t:
//!
//! ```text
//! h_t(u, v) = Σᵢ exp(-t * λᵢ) * φᵢ(u) * φᵢ(v)
//! ```
//!
//! - t → 0: Local structure (direct neighbors matter most)
//! - t → ∞: Global structure (cluster membership)
//!
//! The Heat Kernel Signature (HKS) is h_t(u, u) - invariant to graph isomorphism.
//!
//! # References
//!
//! - Chung, F. (1997): Spectral Graph Theory
//! - Sun, J. et al. (2009): A Concise and Provably Informative Multi-Scale Signature
//! - Coifman & Lafon (2006): Diffusion Maps

use super::path::HierarchyGraph;
use crate::ontology::loader::IRI;
use std::collections::HashMap;

/// Configuration for spectral distance computation
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// Number of eigenvectors to use (top-k smallest non-zero eigenvalues)
    pub num_eigenvectors: usize,
    /// Heat diffusion time parameter (larger = more global)
    pub diffusion_time: f64,
    /// Distance type to compute
    pub distance_type: SpectralDistanceType,
    /// Whether to use normalized Laplacian
    pub normalized: bool,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            num_eigenvectors: 50,
            diffusion_time: 1.0,
            distance_type: SpectralDistanceType::HeatKernel,
            normalized: true,
        }
    }
}

/// Type of spectral distance to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralDistanceType {
    /// Euclidean distance in spectral embedding space
    Euclidean,
    /// Commute time distance (w = 1/λ)
    CommuteTime,
    /// Heat kernel distance (w = exp(-tλ))
    HeatKernel,
    /// Biharmonic distance (w = 1/λ²)
    Biharmonic,
}

/// Spectral embedding and distance computation for ontology graphs
pub struct SpectralDistance {
    /// Configuration
    config: SpectralConfig,
    /// IRI to index mapping
    iri_to_idx: HashMap<IRI, usize>,
    /// Index to IRI mapping
    idx_to_iri: Vec<IRI>,
    /// Eigenvalues (sorted ascending, excluding λ₀ = 0)
    eigenvalues: Vec<f64>,
    /// Eigenvectors: rows are nodes, columns are eigenvector components
    /// Shape: (n_nodes, k) where k = num_eigenvectors
    eigenvectors: Vec<Vec<f64>>,
    /// Degree of each node (for normalized Laplacian)
    degrees: Vec<f64>,
    /// Precomputed spectral coordinates for each node
    spectral_coords: Vec<Vec<f64>>,
}

impl SpectralDistance {
    /// Create a new spectral distance calculator
    pub fn new(config: SpectralConfig) -> Self {
        Self {
            config,
            iri_to_idx: HashMap::new(),
            idx_to_iri: Vec::new(),
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            degrees: Vec::new(),
            spectral_coords: Vec::new(),
        }
    }

    /// Build spectral decomposition from hierarchy graph
    pub fn build(&mut self, hierarchy: &HierarchyGraph) {
        // Build index mappings
        self.iri_to_idx.clear();
        self.idx_to_iri.clear();

        // `HierarchyGraph::all_terms()` iterates a HashMap keyset, so the order is not stable
        // across process runs. The spectral embedding depends on index assignment (initial
        // vectors, deflation order), so we sort terms to make results deterministic.
        let mut terms: Vec<IRI> = hierarchy.all_terms().cloned().collect();
        terms.sort_by(|a, b| a.0.cmp(&b.0));
        for (i, iri) in terms.into_iter().enumerate() {
            self.iri_to_idx.insert(iri.clone(), i);
            self.idx_to_iri.push(iri);
        }

        let n = self.idx_to_iri.len();
        if n == 0 {
            return;
        }

        // Build adjacency matrix and compute degrees
        let (adjacency, degrees) = self.build_adjacency_matrix(hierarchy, n);
        self.degrees = degrees;

        // Compute Laplacian
        let laplacian = self.compute_laplacian(&adjacency, n);

        // Compute eigendecomposition using power iteration
        let k = self.config.num_eigenvectors.min(n - 1);
        let (eigenvalues, eigenvectors) = self.power_iteration_eigenvectors(&laplacian, n, k);

        self.eigenvalues = eigenvalues;
        self.eigenvectors = eigenvectors;

        // Precompute spectral coordinates
        self.compute_spectral_coordinates();
    }

    /// Build adjacency matrix from hierarchy graph
    fn build_adjacency_matrix(
        &self,
        hierarchy: &HierarchyGraph,
        n: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut adj = vec![vec![0.0; n]; n];
        let mut degrees = vec![0.0; n];

        for (iri, &i) in &self.iri_to_idx {
            // Add edges for parents
            for parent in hierarchy.get_parents(iri) {
                if let Some(&j) = self.iri_to_idx.get(&parent) {
                    adj[i][j] = 1.0;
                    adj[j][i] = 1.0;
                    degrees[i] += 1.0;
                    degrees[j] += 1.0;
                }
            }
        }

        // Avoid double counting (we added both directions)
        for d in &mut degrees {
            *d /= 2.0;
            // Ensure minimum degree of 1 for isolated nodes
            if *d < 1.0 {
                *d = 1.0;
            }
        }

        (adj, degrees)
    }

    /// Compute Laplacian matrix
    fn compute_laplacian(&self, adjacency: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
        let mut laplacian = vec![vec![0.0; n]; n];

        if self.config.normalized {
            // Normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        laplacian[i][j] = 1.0;
                    } else {
                        let d_i = self.degrees[i].max(1.0);
                        let d_j = self.degrees[j].max(1.0);
                        laplacian[i][j] = -adjacency[i][j] / (d_i * d_j).sqrt();
                    }
                }
            }
        } else {
            // Unnormalized Laplacian: L = D - A
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        laplacian[i][j] = self.degrees[i];
                    } else {
                        laplacian[i][j] = -adjacency[i][j];
                    }
                }
            }
        }

        laplacian
    }

    /// Compute k smallest non-trivial eigenvectors using power iteration
    ///
    /// We use inverse power iteration with deflation to find eigenvectors
    /// corresponding to the k smallest non-zero eigenvalues.
    fn power_iteration_eigenvectors(
        &self,
        laplacian: &[Vec<f64>],
        n: usize,
        k: usize,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);

        // Already found eigenvectors (for deflation)
        let mut found_vecs: Vec<Vec<f64>> = Vec::new();

        // The smallest eigenvalue of the Laplacian is always 0 with eigenvector [1,1,...,1]
        // We skip it by starting deflation with the constant vector
        let const_vec: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
        found_vecs.push(const_vec);

        for _ in 0..k {
            // Random initial vector
            let mut v: Vec<f64> = (0..n)
                .map(|i| ((i * 13 + 7) % 100) as f64 / 100.0 - 0.5)
                .collect();

            // Orthogonalize against found vectors
            for fv in &found_vecs {
                let dot: f64 = v.iter().zip(fv.iter()).map(|(a, b)| a * b).sum();
                for (vi, fvi) in v.iter_mut().zip(fv.iter()) {
                    *vi -= dot * fvi;
                }
            }
            normalize_vec(&mut v);

            // Power iteration to find smallest non-zero eigenvalue
            // We use (L + shift*I)^(-1) iteration where shift avoids the zero eigenvalue
            let shift = 0.01;
            let shifted_laplacian: Vec<Vec<f64>> = laplacian
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &x)| if i == j { x + shift } else { x })
                        .collect()
                })
                .collect();

            const MAX_ITER: usize = 100;
            const TOL: f64 = 1e-8;

            for _ in 0..MAX_ITER {
                // Solve (L + shift*I) * w = v using Gauss-Seidel
                let w = solve_linear_system(&shifted_laplacian, &v, 50);

                // Orthogonalize against found vectors
                let mut w_orth = w;
                for fv in &found_vecs {
                    let dot: f64 = w_orth.iter().zip(fv.iter()).map(|(a, b)| a * b).sum();
                    for (wi, fvi) in w_orth.iter_mut().zip(fv.iter()) {
                        *wi -= dot * fvi;
                    }
                }

                normalize_vec(&mut w_orth);

                // Check convergence
                let diff: f64 = v
                    .iter()
                    .zip(w_orth.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
                    / n as f64;

                v = w_orth;

                if diff < TOL {
                    break;
                }
            }

            // Compute eigenvalue: λ = v^T L v / v^T v
            let lv = mat_vec_mul(laplacian, &v);
            let lambda: f64 = v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum();

            if lambda > 1e-6 {
                // Only keep non-trivial eigenvalues
                eigenvalues.push(lambda);
                eigenvectors.push(v.clone());
            }

            found_vecs.push(v);
        }

        (eigenvalues, eigenvectors)
    }

    /// Compute spectral coordinates for each node
    fn compute_spectral_coordinates(&mut self) {
        let n = self.idx_to_iri.len();
        let k = self.eigenvectors.len();

        self.spectral_coords = vec![vec![0.0; k]; n];

        for (v_idx, eigenvec) in self.eigenvectors.iter().enumerate() {
            let weight = self.eigenvalue_weight(self.eigenvalues[v_idx]);

            for (node_idx, &coord) in eigenvec.iter().enumerate() {
                self.spectral_coords[node_idx][v_idx] = coord * weight.sqrt();
            }
        }
    }

    /// Compute weight for eigenvalue based on distance type
    fn eigenvalue_weight(&self, lambda: f64) -> f64 {
        match self.config.distance_type {
            SpectralDistanceType::Euclidean => 1.0,
            SpectralDistanceType::CommuteTime => {
                if lambda > 1e-10 {
                    1.0 / lambda
                } else {
                    0.0
                }
            }
            SpectralDistanceType::HeatKernel => (-self.config.diffusion_time * lambda).exp(),
            SpectralDistanceType::Biharmonic => {
                if lambda > 1e-10 {
                    1.0 / (lambda * lambda)
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute spectral distance between two IRIs
    pub fn distance(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let i = *self.iri_to_idx.get(a)?;
        let j = *self.iri_to_idx.get(b)?;

        if i == j {
            return Some(0.0);
        }

        let coords_i = &self.spectral_coords[i];
        let coords_j = &self.spectral_coords[j];

        let dist_sq: f64 = coords_i
            .iter()
            .zip(coords_j.iter())
            .map(|(xi, xj)| (xi - xj).powi(2))
            .sum();

        Some(dist_sq.sqrt())
    }

    /// Compute heat kernel value h_t(a, b)
    pub fn heat_kernel(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let i = *self.iri_to_idx.get(a)?;
        let j = *self.iri_to_idx.get(b)?;

        let t = self.config.diffusion_time;

        let hk: f64 = self
            .eigenvalues
            .iter()
            .zip(self.eigenvectors.iter())
            .map(|(&lambda, eigenvec)| (-t * lambda).exp() * eigenvec[i] * eigenvec[j])
            .sum();

        Some(hk)
    }

    /// Compute Heat Kernel Signature at a node
    ///
    /// HKS(a) = h_t(a, a) captures local structure information
    pub fn heat_kernel_signature(&self, a: &IRI) -> Option<f64> {
        self.heat_kernel(a, a)
    }

    /// Compute diffusion distance
    ///
    /// This is derived from heat kernel: d²(a,b) = h_t(a,a) + h_t(b,b) - 2*h_t(a,b)
    pub fn diffusion_distance(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let hk_aa = self.heat_kernel(a, a)?;
        let hk_bb = self.heat_kernel(b, b)?;
        let hk_ab = self.heat_kernel(a, b)?;

        let dist_sq = hk_aa + hk_bb - 2.0 * hk_ab;
        Some(dist_sq.max(0.0).sqrt())
    }

    /// Convert spectral distance to semantic distance [0, 1]
    pub fn semantic_distance(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let raw_dist = self.distance(a, b)?;

        // Sigmoid normalization with adaptive scaling
        // Based on distribution of distances in the graph
        let scale = self.estimate_distance_scale();
        let normalized = 1.0 - (-raw_dist / scale).exp();

        Some(normalized.clamp(0.0, 1.0))
    }

    /// Estimate typical distance scale for normalization
    fn estimate_distance_scale(&self) -> f64 {
        // Use the Fiedler value (second smallest eigenvalue) as a reference
        if !self.eigenvalues.is_empty() {
            // Typical distance ~ 1/sqrt(λ₁)
            1.0 / self.eigenvalues[0].sqrt().max(0.1)
        } else {
            1.0
        }
    }

    /// Get the Fiedler value (algebraic connectivity)
    ///
    /// The second smallest eigenvalue of the Laplacian measures how
    /// "well-connected" the graph is. Larger = more connected.
    pub fn fiedler_value(&self) -> Option<f64> {
        self.eigenvalues.first().copied()
    }

    /// Get number of terms in the index
    pub fn num_terms(&self) -> usize {
        self.idx_to_iri.len()
    }

    /// Get number of eigenvectors computed
    pub fn num_eigenvectors(&self) -> usize {
        self.eigenvectors.len()
    }

    /// Get spectral embedding for a term
    pub fn embedding(&self, a: &IRI) -> Option<&[f64]> {
        let i = *self.iri_to_idx.get(a)?;
        Some(&self.spectral_coords[i])
    }
}

/// Normalize a vector to unit length
fn normalize_vec(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Matrix-vector multiplication
fn mat_vec_mul(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Solve linear system Ax = b using Gauss-Seidel iteration
fn solve_linear_system(a: &[Vec<f64>], b: &[f64], max_iter: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = b.to_vec();

    for _ in 0..max_iter {
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..n {
                if i != j {
                    sum -= a[i][j] * x[j];
                }
            }
            if a[i][i].abs() > 1e-10 {
                x[i] = sum / a[i][i];
            }
        }
    }

    x
}

/// Wavelet-based multi-scale spectral distance
///
/// Uses spectral graph wavelets to capture features at multiple scales.
/// This is more robust than single-scale methods.
pub struct SpectralWaveletDistance {
    /// Base spectral distance calculator
    spectral: SpectralDistance,
    /// Wavelet scales (diffusion times)
    scales: Vec<f64>,
}

impl SpectralWaveletDistance {
    /// Create with logarithmically spaced scales
    pub fn new(config: SpectralConfig, num_scales: usize) -> Self {
        let scales: Vec<f64> = (0..num_scales)
            .map(|i| 0.1 * 2.0_f64.powi(i as i32))
            .collect();

        Self {
            spectral: SpectralDistance::new(config),
            scales,
        }
    }

    /// Build from hierarchy graph
    pub fn build(&mut self, hierarchy: &HierarchyGraph) {
        self.spectral.build(hierarchy);
    }

    /// Compute multi-scale distance
    ///
    /// Averages heat kernel distances across multiple scales
    pub fn distance(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let i = *self.spectral.iri_to_idx.get(a)?;
        let j = *self.spectral.iri_to_idx.get(b)?;

        if i == j {
            return Some(0.0);
        }

        let mut total_dist = 0.0;

        for &t in &self.scales {
            // Heat kernel at scale t
            let hk: f64 = self
                .spectral
                .eigenvalues
                .iter()
                .zip(self.spectral.eigenvectors.iter())
                .map(|(&lambda, eigenvec)| (-t * lambda).exp() * eigenvec[i] * eigenvec[j])
                .sum();

            // Convert to distance contribution
            let hk_ii: f64 = self
                .spectral
                .eigenvalues
                .iter()
                .zip(self.spectral.eigenvectors.iter())
                .map(|(&lambda, eigenvec)| (-t * lambda).exp() * eigenvec[i] * eigenvec[i])
                .sum();

            let hk_jj: f64 = self
                .spectral
                .eigenvalues
                .iter()
                .zip(self.spectral.eigenvectors.iter())
                .map(|(&lambda, eigenvec)| (-t * lambda).exp() * eigenvec[j] * eigenvec[j])
                .sum();

            let dist_sq = hk_ii + hk_jj - 2.0 * hk;
            total_dist += dist_sq.max(0.0).sqrt();
        }

        Some(total_dist / self.scales.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_hierarchy() -> HierarchyGraph {
        let mut h = HierarchyGraph::new();

        // Create a simple tree:
        //      root
        //     /    \
        //    a      b
        //   / \      \
        //  c   d      e

        let root = IRI::new("http://test/root");
        let a = IRI::new("http://test/a");
        let b = IRI::new("http://test/b");
        let c = IRI::new("http://test/c");
        let d = IRI::new("http://test/d");
        let e = IRI::new("http://test/e");

        h.add_term(&root, &[]);
        h.add_term(&a, &[root.clone()]);
        h.add_term(&b, &[root.clone()]);
        h.add_term(&c, &[a.clone()]);
        h.add_term(&d, &[a.clone()]);
        h.add_term(&e, &[b.clone()]);

        h
    }

    #[test]
    fn test_spectral_distance_self() {
        let mut sd = SpectralDistance::new(SpectralConfig::default());
        let h = make_test_hierarchy();
        sd.build(&h);

        let root = IRI::new("http://test/root");
        assert_eq!(sd.distance(&root, &root), Some(0.0));
    }

    #[test]
    fn test_spectral_distance_symmetric() {
        let mut sd = SpectralDistance::new(SpectralConfig::default());
        let h = make_test_hierarchy();
        sd.build(&h);

        let a = IRI::new("http://test/a");
        let b = IRI::new("http://test/b");

        let d_ab = sd.distance(&a, &b);
        let d_ba = sd.distance(&b, &a);

        assert!(d_ab.is_some());
        assert!(d_ba.is_some());
        assert!((d_ab.unwrap() - d_ba.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_siblings_closer_than_cousins() {
        let mut sd = SpectralDistance::new(SpectralConfig::default());
        let h = make_test_hierarchy();
        sd.build(&h);

        let c = IRI::new("http://test/c");
        let d = IRI::new("http://test/d");
        let e = IRI::new("http://test/e");

        // c and d are siblings (under a)
        // c and e are cousins (c under a, e under b)
        let d_cd = sd.distance(&c, &d).unwrap();
        let d_ce = sd.distance(&c, &e).unwrap();

        // Siblings should be closer than cousins
        assert!(
            d_cd < d_ce,
            "Siblings {} should be closer than cousins {}",
            d_cd,
            d_ce
        );
    }

    #[test]
    fn test_heat_kernel_signature() {
        let mut sd = SpectralDistance::new(SpectralConfig::default());
        let h = make_test_hierarchy();
        sd.build(&h);

        let root = IRI::new("http://test/root");
        let hks = sd.heat_kernel_signature(&root);

        assert!(hks.is_some());
        assert!(hks.unwrap() > 0.0);
    }

    #[test]
    fn test_fiedler_value() {
        let mut sd = SpectralDistance::new(SpectralConfig::default());
        let h = make_test_hierarchy();
        sd.build(&h);

        let fiedler = sd.fiedler_value();
        assert!(fiedler.is_some());
        assert!(fiedler.unwrap() > 0.0);
    }

    #[test]
    fn test_semantic_distance_bounds() {
        let mut sd = SpectralDistance::new(SpectralConfig::default());
        let h = make_test_hierarchy();
        sd.build(&h);

        let a = IRI::new("http://test/a");
        let e = IRI::new("http://test/e");

        let sem_dist = sd.semantic_distance(&a, &e);
        assert!(sem_dist.is_some());
        let d = sem_dist.unwrap();
        assert!(d >= 0.0 && d <= 1.0);
    }
}
