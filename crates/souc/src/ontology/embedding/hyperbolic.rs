//! Hyperbolic Embeddings for Ontology Hierarchies
//!
//! Hyperbolic spaces are mathematically ideal for embedding hierarchical structures
//! because they have exponentially more "room" near the boundary (at the leaves of
//! a tree) compared to Euclidean space. This allows hierarchical data to be embedded
//! with much lower distortion.
//!
//! # The Poincaré Ball Model
//!
//! We use the Poincaré ball model: B^n = {x ∈ ℝⁿ : ||x|| < 1}
//!
//! Key properties:
//! - Points near the center represent general concepts (root)
//! - Points near the boundary represent specific concepts (leaves)
//! - The metric expands exponentially near the boundary
//! - Geodesics are circular arcs perpendicular to the boundary
//!
//! # Hyperbolic Distance
//!
//! The distance in the Poincaré ball is:
//!
//! ```text
//! d(x, y) = arcosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
//! ```
//!
//! This distance grows logarithmically as points approach the boundary,
//! naturally capturing the tree-like structure of ontologies.
//!
//! # References
//!
//! - Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
//! - Sala et al. (2018): "Representation Tradeoffs for Hyperbolic Embeddings"

use super::{Embedding, EmbeddingError, EmbeddingGenerator, EmbeddingModel};
use crate::ontology::distance::path::HierarchyGraph;
use crate::ontology::loader::IRI;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Curvature parameter for the hyperbolic space
/// Negative curvature: K = -1/c² where c is the "radius" of curvature
const DEFAULT_CURVATURE: f64 = 1.0;

/// Epsilon for numerical stability near the boundary
const BOUNDARY_EPS: f64 = 1e-5;

/// Maximum norm before clamping (to stay inside the ball)
const MAX_NORM: f64 = 1.0 - BOUNDARY_EPS;

/// A point in the Poincaré ball
#[derive(Debug, Clone)]
pub struct PoincarePoint {
    /// Coordinates (||coords|| < 1)
    pub coords: Vec<f64>,
    /// Curvature parameter (typically 1.0)
    pub curvature: f64,
}

impl PoincarePoint {
    /// Create a new Poincaré point, clamping to stay inside the ball
    pub fn new(coords: Vec<f64>, curvature: f64) -> Self {
        let mut point = Self { coords, curvature };
        point.clamp_to_ball();
        point
    }

    /// Create from Euclidean coordinates with default curvature
    pub fn from_euclidean(coords: Vec<f64>) -> Self {
        Self::new(coords, DEFAULT_CURVATURE)
    }

    /// Create the origin point (most general concept)
    pub fn origin(dimensions: usize) -> Self {
        Self::new(vec![0.0; dimensions], DEFAULT_CURVATURE)
    }

    /// Get dimensionality
    pub fn dimensions(&self) -> usize {
        self.coords.len()
    }

    /// Compute squared Euclidean norm
    pub fn squared_norm(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum()
    }

    /// Compute Euclidean norm
    pub fn norm(&self) -> f64 {
        self.squared_norm().sqrt()
    }

    /// Clamp point to stay inside the Poincaré ball
    fn clamp_to_ball(&mut self) {
        let norm = self.norm();
        if norm >= MAX_NORM {
            let scale = MAX_NORM / norm;
            for x in &mut self.coords {
                *x *= scale;
            }
        }
    }

    /// Conformal factor: λ_x = 2 / (1 - ||x||²)
    /// This measures how much the metric is "stretched" at this point
    pub fn conformal_factor(&self) -> f64 {
        2.0 / (1.0 - self.squared_norm()).max(BOUNDARY_EPS)
    }

    /// Hyperbolic distance to another point in the Poincaré ball
    ///
    /// d(x, y) = (1/√c) * arcosh(1 + 2c * ||x - y||² / ((1 - c||x||²)(1 - c||y||²)))
    pub fn distance(&self, other: &Self) -> f64 {
        debug_assert_eq!(
            self.coords.len(),
            other.coords.len(),
            "Dimensions must match"
        );

        let c = self.curvature;
        let x_sq = self.squared_norm();
        let y_sq = other.squared_norm();

        // ||x - y||²
        let diff_sq: f64 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let numerator = 2.0 * c * diff_sq;
        let denominator = (1.0 - c * x_sq).max(BOUNDARY_EPS) * (1.0 - c * y_sq).max(BOUNDARY_EPS);

        // arcosh(1 + x) = ln(1 + x + sqrt((1+x)² - 1)) = ln(1 + x + sqrt(x(x+2)))
        let arg = 1.0 + numerator / denominator;
        arcosh(arg) / c.sqrt()
    }

    /// Möbius addition: x ⊕ y
    ///
    /// This is the hyperbolic analog of vector addition.
    /// In the Poincaré ball, straight lines become circular arcs.
    ///
    /// Formula:
    /// x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
    pub fn mobius_add(&self, other: &Self) -> Self {
        let c = self.curvature;
        let x_sq = self.squared_norm();
        let y_sq = other.squared_norm();

        // Inner product ⟨x, y⟩
        let xy_dot: f64 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum();

        let num_coeff_x = 1.0 + 2.0 * c * xy_dot + c * y_sq;
        let num_coeff_y = 1.0 - c * x_sq;
        let denom = 1.0 + 2.0 * c * xy_dot + c * c * x_sq * y_sq;

        let coords: Vec<f64> = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(xi, yi)| (num_coeff_x * xi + num_coeff_y * yi) / denom)
            .collect();

        Self::new(coords, c)
    }

    /// Möbius scalar multiplication: r ⊗ x
    ///
    /// Scales a point along the geodesic from origin
    pub fn mobius_scalar(&self, r: f64) -> Self {
        let norm = self.norm();
        if norm < BOUNDARY_EPS {
            return self.clone();
        }

        let c = self.curvature;
        let sqrt_c = c.sqrt();

        // tanh(r * arctanh(√c * ||x||)) / (√c * ||x||) * x
        let arg = sqrt_c * norm;
        let new_norm = (r * arg.atanh()).tanh() / sqrt_c;
        let scale = new_norm / norm;

        let coords: Vec<f64> = self.coords.iter().map(|x| x * scale).collect();
        Self::new(coords, c)
    }

    /// Project from tangent space at origin to Poincaré ball (exponential map)
    ///
    /// exp_0(v) = tanh(√c * ||v|| / 2) * v / (√c * ||v||)
    pub fn exp_map_at_origin(tangent: &[f64], curvature: f64) -> Self {
        let norm: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < BOUNDARY_EPS {
            return Self::origin(tangent.len());
        }

        let sqrt_c = curvature.sqrt();
        let scale = (sqrt_c * norm / 2.0).tanh() / (sqrt_c * norm);

        let coords: Vec<f64> = tangent.iter().map(|x| x * scale).collect();
        Self::new(coords, curvature)
    }

    /// Project from Poincaré ball to tangent space at origin (logarithmic map)
    ///
    /// log_0(y) = 2 / √c * arctanh(√c * ||y||) * y / ||y||
    pub fn log_map_at_origin(&self) -> Vec<f64> {
        let norm = self.norm();
        if norm < BOUNDARY_EPS {
            return vec![0.0; self.coords.len()];
        }

        let sqrt_c = self.curvature.sqrt();
        let scale = 2.0 / sqrt_c * (sqrt_c * norm).atanh() / norm;

        self.coords.iter().map(|x| x * scale).collect()
    }

    /// Geodesic interpolation: move t ∈ [0, 1] along geodesic from self to other
    pub fn geodesic(&self, other: &Self, t: f64) -> Self {
        // Move to tangent space, interpolate, move back
        let neg_self = self.mobius_scalar(-1.0);
        let translated = neg_self.mobius_add(other);
        let scaled = translated.mobius_scalar(t);
        self.mobius_add(&scaled)
    }

    /// Convert to f32 vector for storage compatibility
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.coords.iter().map(|&x| x as f32).collect()
    }

    /// Create from f32 vector
    pub fn from_f32_vec(vec: &[f32], curvature: f64) -> Self {
        Self::new(vec.iter().map(|&x| x as f64).collect(), curvature)
    }
}

/// Inverse hyperbolic cosine
fn arcosh(x: f64) -> f64 {
    // arcosh(x) = ln(x + sqrt(x² - 1))
    (x + (x * x - 1.0).max(0.0).sqrt()).ln()
}

/// Generator for hyperbolic ontology embeddings
///
/// This generates Poincaré ball embeddings from the ontology hierarchy.
/// Parent concepts are placed closer to the origin; children radiate outward.
pub struct HyperbolicGenerator {
    /// Embedding dimensionality
    dimensions: usize,
    /// Curvature parameter
    curvature: f64,
    /// Pre-computed embeddings (cache)
    embeddings: HashMap<IRI, PoincarePoint>,
    /// Random initialization seed
    seed: u64,
}

impl HyperbolicGenerator {
    /// Create a new hyperbolic embedding generator
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            curvature: DEFAULT_CURVATURE,
            embeddings: HashMap::new(),
            seed: 42,
        }
    }

    /// Set curvature parameter
    pub fn with_curvature(mut self, curvature: f64) -> Self {
        self.curvature = curvature;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build embeddings from a hierarchy graph
    ///
    /// Algorithm:
    /// 1. Find roots (terms with no parents)
    /// 2. Place roots near origin
    /// 3. Recursively place children at increasing radii
    /// 4. Spread siblings angularly to avoid collision
    pub fn build_from_hierarchy(&mut self, hierarchy: &HierarchyGraph) {
        // Clear existing
        self.embeddings.clear();

        // Find roots (nodes with no parents)
        let roots = hierarchy.get_roots();

        // Place roots near origin with small angular spread
        let root_radius = 0.1;
        for (i, root) in roots.iter().enumerate() {
            let angle = 2.0 * PI * (i as f64) / (roots.len() as f64);
            let point = self.place_at_polar(root_radius, angle, 0.0);
            self.embeddings.insert(root.clone(), point);
        }

        // BFS from roots to place children
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        for root in &roots {
            visited.insert(root.clone());
            queue.push_back((root.clone(), 1));
        }

        while let Some((iri, depth)) = queue.pop_front() {
            let children = hierarchy.get_children(&iri);
            if children.is_empty() {
                continue;
            }

            // Place children at larger radius than parent
            let parent_point = self.embeddings.get(&iri).unwrap().clone();
            let child_radius = self.depth_to_radius(depth);

            // Angular spread for children
            let base_angle = self.compute_base_angle(&parent_point);

            for (i, child) in children.iter().enumerate() {
                if visited.contains(child) {
                    continue;
                }
                visited.insert(child.clone());

                // Angular offset for this child
                let angle_spread = PI / (depth as f64 + 1.0);
                let angle = base_angle + angle_spread * (i as f64 - children.len() as f64 / 2.0);

                // Add some noise based on hash of IRI for determinism
                let noise = self.hash_to_noise(child) * 0.1;

                let point = self.place_at_polar(child_radius + noise, angle, 0.0);
                self.embeddings.insert(child.clone(), point);

                queue.push_back((child.clone(), depth + 1));
            }
        }
    }

    /// Convert depth in hierarchy to radius in Poincaré ball
    ///
    /// We use an exponential mapping so leaves approach (but don't reach) the boundary.
    /// radius = 1 - exp(-α * depth) where α controls the "spread"
    fn depth_to_radius(&self, depth: usize) -> f64 {
        let alpha = 0.3; // Spread parameter
        (1.0 - (-alpha * depth as f64).exp()) * MAX_NORM
    }

    /// Compute base angle from a parent point
    fn compute_base_angle(&self, point: &PoincarePoint) -> f64 {
        if point.dimensions() >= 2 {
            point.coords[1].atan2(point.coords[0])
        } else {
            0.0
        }
    }

    /// Place a point at polar coordinates (for 2D projection)
    fn place_at_polar(&self, radius: f64, angle: f64, _height: f64) -> PoincarePoint {
        let mut coords = vec![0.0; self.dimensions];

        if self.dimensions >= 2 {
            coords[0] = radius * angle.cos();
            coords[1] = radius * angle.sin();
        } else if self.dimensions == 1 {
            coords[0] = radius;
        }

        // Add small random components for higher dimensions
        let hash = (angle * 1000.0) as u64 ^ self.seed;
        for i in 2..self.dimensions {
            coords[i] = self.pseudo_random(hash, i) * 0.1 * radius;
        }

        PoincarePoint::new(coords, self.curvature)
    }

    /// Generate deterministic pseudo-random value from hash and index
    fn pseudo_random(&self, hash: u64, index: usize) -> f64 {
        let x = hash.wrapping_mul(index as u64 + 1).wrapping_add(self.seed);
        ((x % 10000) as f64 / 10000.0) * 2.0 - 1.0 // [-1, 1]
    }

    /// Hash IRI to small noise value
    fn hash_to_noise(&self, iri: &IRI) -> f64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        iri.as_str().hash(&mut hasher);
        let h = hasher.finish();
        ((h % 1000) as f64 / 1000.0) * 0.2 - 0.1 // [-0.1, 0.1]
    }

    /// Get embedding for an IRI
    pub fn get(&self, iri: &IRI) -> Option<&PoincarePoint> {
        self.embeddings.get(iri)
    }

    /// Compute hyperbolic distance between two IRIs
    pub fn hyperbolic_distance(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let pa = self.embeddings.get(a)?;
        let pb = self.embeddings.get(b)?;
        Some(pa.distance(pb))
    }

    /// Convert hyperbolic distance to semantic distance [0, 1]
    ///
    /// We use a sigmoid-like mapping that saturates at large distances
    pub fn semantic_distance(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let hyp_dist = self.hyperbolic_distance(a, b)?;
        // Sigmoid mapping: σ(d) = 2 / (1 + exp(-d/scale)) - 1
        let scale = 3.0;
        Some(2.0 / (1.0 + (-hyp_dist / scale).exp()) - 1.0)
    }
}

impl EmbeddingGenerator for HyperbolicGenerator {
    fn generate(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        if let Some(point) = self.embeddings.get(iri) {
            Ok(Embedding::with_confidence(
                iri.clone(),
                point.to_f32_vec(),
                EmbeddingModel::Structural,
                0.95, // High confidence for hyperbolic embeddings
            ))
        } else {
            // Generate a default embedding at moderate radius
            let hash = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                iri.as_str().hash(&mut hasher);
                hasher.finish()
            };

            let angle = (hash % 1000) as f64 / 1000.0 * 2.0 * PI;
            let radius = 0.5 + (hash % 500) as f64 / 2000.0; // [0.5, 0.75]

            let mut coords = vec![0.0; self.dimensions];
            if self.dimensions >= 2 {
                coords[0] = radius * angle.cos();
                coords[1] = radius * angle.sin();
            }

            let point = PoincarePoint::new(coords, self.curvature);

            Ok(Embedding::with_confidence(
                iri.clone(),
                point.to_f32_vec(),
                EmbeddingModel::Structural,
                0.5, // Lower confidence for default embeddings
            ))
        }
    }

    fn generate_batch(&self, iris: &[IRI]) -> Result<Vec<Embedding>, EmbeddingError> {
        iris.iter().map(|iri| self.generate(iri)).collect()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Hyperbolic centroid (Fréchet mean in hyperbolic space)
///
/// This computes the point that minimizes the sum of squared hyperbolic distances
/// to a set of points. Useful for finding the "center" of a set of concepts.
pub fn hyperbolic_centroid(points: &[PoincarePoint], max_iter: usize) -> Option<PoincarePoint> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        return Some(points[0].clone());
    }

    let dims = points[0].dimensions();
    let c = points[0].curvature;

    // Start at Euclidean centroid projected to Poincaré ball
    let mut centroid = {
        let mut sum = vec![0.0; dims];
        for p in points {
            for (i, x) in p.coords.iter().enumerate() {
                sum[i] += x;
            }
        }
        for x in &mut sum {
            *x /= points.len() as f64;
        }
        PoincarePoint::new(sum, c)
    };

    // Iteratively refine using gradient descent in tangent space
    let lr = 0.1;
    for _ in 0..max_iter {
        // Compute gradient: sum of log maps
        let mut grad = vec![0.0; dims];
        for p in points {
            let log = log_map(&centroid, p);
            for (i, x) in log.iter().enumerate() {
                grad[i] += x;
            }
        }

        // Scale by learning rate
        for x in &mut grad {
            *x *= lr / points.len() as f64;
        }

        // Move centroid via exponential map
        centroid = exp_map(&centroid, &grad);
    }

    Some(centroid)
}

/// Exponential map at a point (not just origin)
fn exp_map(base: &PoincarePoint, tangent: &[f64]) -> PoincarePoint {
    let c = base.curvature;
    let lambda = base.conformal_factor();

    let norm: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < BOUNDARY_EPS {
        return base.clone();
    }

    let sqrt_c = c.sqrt();
    let scale = (sqrt_c * norm * lambda / 2.0).tanh() / (sqrt_c * norm);

    let v: Vec<f64> = tangent.iter().map(|x| x * scale).collect();
    let v_point = PoincarePoint::new(v, c);

    base.mobius_add(&v_point)
}

/// Logarithmic map at a point (not just origin)
fn log_map(base: &PoincarePoint, target: &PoincarePoint) -> Vec<f64> {
    let c = base.curvature;
    let lambda = base.conformal_factor();

    // -base ⊕ target
    let neg_base = base.mobius_scalar(-1.0);
    let diff = neg_base.mobius_add(target);

    let norm = diff.norm();
    if norm < BOUNDARY_EPS {
        return vec![0.0; base.dimensions()];
    }

    let sqrt_c = c.sqrt();
    let scale = 2.0 / (sqrt_c * lambda) * (sqrt_c * norm).atanh() / norm;

    diff.coords.iter().map(|x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_point_creation() {
        let p = PoincarePoint::from_euclidean(vec![0.5, 0.3]);
        assert!(p.norm() < 1.0);
        assert_eq!(p.dimensions(), 2);
    }

    #[test]
    fn test_poincare_clamp() {
        // Point outside the ball should be clamped
        let p = PoincarePoint::from_euclidean(vec![2.0, 2.0]);
        assert!(p.norm() < 1.0);
    }

    #[test]
    fn test_hyperbolic_distance_self() {
        let p = PoincarePoint::from_euclidean(vec![0.3, 0.4]);
        assert!((p.distance(&p) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_distance_origin() {
        let origin = PoincarePoint::origin(2);
        let p = PoincarePoint::from_euclidean(vec![0.5, 0.0]);

        let d = origin.distance(&p);
        assert!(d > 0.0);

        // arcosh(1 + 2*0.25/0.75) = arcosh(5/3) ≈ 1.099
        assert!((d - 1.099).abs() < 0.05);
    }

    #[test]
    fn test_mobius_add_origin() {
        let origin = PoincarePoint::origin(2);
        let p = PoincarePoint::from_euclidean(vec![0.3, 0.4]);

        // 0 ⊕ p = p
        let sum = origin.mobius_add(&p);
        for (a, b) in sum.coords.iter().zip(p.coords.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_geodesic_endpoints() {
        let p1 = PoincarePoint::from_euclidean(vec![0.2, 0.1]);
        let p2 = PoincarePoint::from_euclidean(vec![0.6, 0.4]);

        // geodesic(0) = p1
        let g0 = p1.geodesic(&p2, 0.0);
        for (a, b) in g0.coords.iter().zip(p1.coords.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // geodesic(1) = p2
        let g1 = p1.geodesic(&p2, 1.0);
        for (a, b) in g1.coords.iter().zip(p2.coords.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_exp_log_inverse() {
        let v = vec![0.3, 0.4];
        let p = PoincarePoint::exp_map_at_origin(&v, 1.0);
        let v_back = p.log_map_at_origin();

        for (a, b) in v.iter().zip(v_back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hyperbolic_generator() {
        let generator = HyperbolicGenerator::new(64);
        assert_eq!(generator.dimensions(), 64);
    }

    #[test]
    fn test_conformal_factor() {
        let origin = PoincarePoint::origin(2);
        assert!((origin.conformal_factor() - 2.0).abs() < 1e-10);

        let near_boundary = PoincarePoint::from_euclidean(vec![0.9, 0.0]);
        assert!(near_boundary.conformal_factor() > 10.0);
    }

    #[test]
    fn test_distance_increases_near_boundary() {
        // Points near boundary should have larger distances
        let inner1 = PoincarePoint::from_euclidean(vec![0.1, 0.0]);
        let inner2 = PoincarePoint::from_euclidean(vec![0.2, 0.0]);
        let inner_dist = inner1.distance(&inner2);

        let outer1 = PoincarePoint::from_euclidean(vec![0.8, 0.0]);
        let outer2 = PoincarePoint::from_euclidean(vec![0.9, 0.0]);
        let outer_dist = outer1.distance(&outer2);

        // Same Euclidean distance (0.1), but hyperbolic distance larger near boundary
        assert!(outer_dist > inner_dist);
    }
}
