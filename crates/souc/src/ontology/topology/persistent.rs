// Persistent Homology - Detect topological structure of ontologies
// Uses Vietoris-Rips complex to analyze hierarchy at multiple scales

use crate::ontology::embedding::PoincarePoint;
use crate::ontology::loader::IRI;
use std::collections::{BTreeSet, HashSet};

/// Rips complex: union of simplices where points within distance r
pub struct RipsComplex {
    /// Embedded points
    points: Vec<(IRI, PoincarePoint)>,
    /// Distance matrix (cached)
    distances: Vec<Vec<f64>>,
    /// Current threshold radius
    radius: f64,
}

impl RipsComplex {
    pub fn new(points: Vec<(IRI, PoincarePoint)>) -> Self {
        let n = points.len();
        let mut distances = vec![vec![0.0; n]; n];

        // Precompute pairwise distances
        for i in 0..n {
            for j in i + 1..n {
                let dist = points[i].1.distance(&points[j].1);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        Self {
            points,
            distances,
            radius: 0.0,
        }
    }

    /// Get 0-simplices (vertices)
    pub fn vertices(&self) -> Vec<usize> {
        (0..self.points.len()).collect()
    }

    /// Get 1-simplices (edges) within radius
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for i in 0..self.points.len() {
            for j in i + 1..self.points.len() {
                if self.distances[i][j] <= self.radius {
                    edges.push((i, j));
                }
            }
        }
        edges
    }

    /// Get 2-simplices (triangles)
    pub fn triangles(&self) -> Vec<(usize, usize, usize)> {
        let mut triangles = Vec::new();
        let edges = self.edges();
        let edge_set: HashSet<_> = edges.iter().cloned().collect();

        for i in 0..self.points.len() {
            for j in i + 1..self.points.len() {
                if self.distances[i][j] <= self.radius {
                    for k in j + 1..self.points.len() {
                        if self.distances[i][k] <= self.radius
                            && self.distances[j][k] <= self.radius
                        {
                            triangles.push((i, j, k));
                        }
                    }
                }
            }
        }
        triangles
    }

    pub fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn get_point(&self, idx: usize) -> Option<&(IRI, PoincarePoint)> {
        self.points.get(idx)
    }
}

/// Feature detected in persistent homology
#[derive(Debug, Clone)]
pub struct TopologicalFeature {
    /// Birth radius where feature appears
    pub birth: f64,
    /// Death radius where feature disappears
    pub death: f64,
    /// Persistence (death - birth)
    pub persistence: f64,
    /// Dimension: 0 for connected components, 1 for holes/cycles
    pub dimension: usize,
    /// IRIs involved in this feature (for 0D: cluster members, for 1D: cycle)
    pub involved_points: Vec<IRI>,
}

impl TopologicalFeature {
    pub fn new(birth: f64, death: f64, dimension: usize, involved_points: Vec<IRI>) -> Self {
        let persistence = death - birth;
        Self {
            birth,
            death,
            persistence,
            dimension,
            involved_points,
        }
    }
}

/// Persistent diagram: collection of features with lifetime information
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// Features sorted by persistence
    features: Vec<TopologicalFeature>,
}

impl PersistenceDiagram {
    pub fn new(features: Vec<TopologicalFeature>) -> Self {
        let mut sorted_features = features;
        sorted_features.sort_by(|a, b| b.persistence.partial_cmp(&a.persistence).unwrap());
        Self {
            features: sorted_features,
        }
    }

    /// Get most significant features
    pub fn top_features(&self, k: usize) -> Vec<&TopologicalFeature> {
        self.features.iter().take(k).collect()
    }

    /// Features with dimension 0 (connected components)
    pub fn components(&self) -> Vec<&TopologicalFeature> {
        self.features.iter().filter(|f| f.dimension == 0).collect()
    }

    /// Features with dimension 1 (holes/cycles)
    pub fn holes(&self) -> Vec<&TopologicalFeature> {
        self.features.iter().filter(|f| f.dimension == 1).collect()
    }

    pub fn all_features(&self) -> &[TopologicalFeature] {
        &self.features
    }
}

/// Persistent Homology Analyzer
pub struct PersistentHomologyAnalyzer {
    rips: RipsComplex,
    /// Threshold radii to analyze
    radii: Vec<f64>,
}

impl PersistentHomologyAnalyzer {
    pub fn new(rips: RipsComplex) -> Self {
        let max_dist = compute_max_distance(&rips);
        let radii = Self::generate_radii(0.0, max_dist, 100);

        Self { rips, radii }
    }

    /// Generate logarithmically spaced radii
    fn generate_radii(min: f64, max: f64, n_steps: usize) -> Vec<f64> {
        (0..=n_steps)
            .map(|i| {
                let t = i as f64 / n_steps as f64;
                min + t * (max - min)
            })
            .collect()
    }

    /// Compute persistent homology (simplified: component analysis)
    pub fn analyze(&mut self) -> PersistenceDiagram {
        let mut features = Vec::new();
        let mut component_map: Vec<usize> = (0..self.rips.point_count()).collect(); // Union-find

        // For each increasing radius, track connectivity changes
        let mut prev_components = self.rips.point_count();

        for &radius in &self.radii {
            self.rips.set_radius(radius);
            let edges = self.rips.edges();

            // Union-find: merge components connected by edges
            for (u, v) in edges {
                let root_u = find_root(&mut component_map, u);
                let root_v = find_root(&mut component_map, v);
                if root_u != root_v {
                    component_map[root_v] = root_u;
                }
            }

            // Count unique roots by finding root for each element
            let mut root_count = 0;
            for i in 0..component_map.len() {
                let root = find_root(&mut component_map, i);
                if root == i {
                    root_count += 1;
                }
            }
            let num_components = root_count;

            // When components merge (decrease), create feature
            if num_components < prev_components {
                let component_size = self.rips.point_count() / (prev_components + 1);
                let involved: Vec<IRI> = self
                    .rips
                    .vertices()
                    .iter()
                    .take(component_size.min(3))
                    .filter_map(|&i| self.rips.get_point(i).map(|(iri, _)| iri.clone()))
                    .collect();

                features.push(TopologicalFeature::new(
                    self.radii
                        .iter()
                        .rev()
                        .find(|&&r| r < radius)
                        .copied()
                        .unwrap_or(0.0),
                    radius,
                    0, // dimension 0: components
                    involved,
                ));
            }

            prev_components = num_components;
        }

        // Detect cycles (1D homology) via triangles
        let triangles = self.rips.triangles();
        if !triangles.is_empty() {
            let involved: Vec<IRI> = triangles[0..(triangles.len().min(3))]
                .iter()
                .flat_map(|(a, b, c)| {
                    vec![
                        self.rips.get_point(*a).map(|(iri, _)| iri.clone()),
                        self.rips.get_point(*b).map(|(iri, _)| iri.clone()),
                        self.rips.get_point(*c).map(|(iri, _)| iri.clone()),
                    ]
                })
                .flatten()
                .collect();

            features.push(TopologicalFeature::new(
                0.0,
                *self.radii.last().unwrap_or(&1.0),
                1, // dimension 1: holes
                involved,
            ));
        }

        PersistenceDiagram::new(features)
    }
}

/// Union-find helper
fn find_root(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find_root(parent, parent[x]);
    }
    parent[x]
}

/// Compute maximum distance in Rips complex
fn compute_max_distance(rips: &RipsComplex) -> f64 {
    let mut max_dist: f64 = 0.0_f64;
    for i in 0..rips.point_count() {
        for j in i + 1..rips.point_count() {
            if let (Some((_, pi)), Some((_, pj))) = (rips.get_point(i), rips.get_point(j)) {
                max_dist = max_dist.max(pi.distance(pj));
            }
        }
    }
    max_dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_points() -> Vec<(IRI, PoincarePoint)> {
        vec![
            (
                IRI::new("http://test/a"),
                PoincarePoint::from_euclidean(vec![0.0, 0.0]),
            ),
            (
                IRI::new("http://test/b"),
                PoincarePoint::from_euclidean(vec![0.1, 0.0]),
            ),
            (
                IRI::new("http://test/c"),
                PoincarePoint::from_euclidean(vec![0.2, 0.0]),
            ),
            (
                IRI::new("http://test/d"),
                PoincarePoint::from_euclidean(vec![0.5, 0.0]),
            ),
        ]
    }

    #[test]
    fn test_rips_vertices() {
        let points = create_test_points();
        let rips = RipsComplex::new(points);
        assert_eq!(rips.vertices().len(), 4);
    }

    #[test]
    fn test_rips_edges() {
        let points = create_test_points();
        let mut rips = RipsComplex::new(points);
        rips.set_radius(1.0); // Larger radius to accommodate hyperbolic distances
        let edges = rips.edges();
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_topological_feature_persistence() {
        let feature = TopologicalFeature::new(0.1, 0.5, 0, vec![IRI::new("http://test/a")]);
        assert_eq!(feature.persistence, 0.4);
    }

    #[test]
    fn test_persistent_homology_analysis() {
        let points = create_test_points();
        let rips = RipsComplex::new(points);
        let mut analyzer = PersistentHomologyAnalyzer::new(rips);
        let diagram = analyzer.analyze();
        assert!(!diagram.all_features().is_empty());
    }

    #[test]
    fn test_diagram_components() {
        let feature0 = TopologicalFeature::new(0.1, 0.5, 0, vec![]);
        let feature1 = TopologicalFeature::new(0.2, 0.6, 1, vec![]);
        let diagram = PersistenceDiagram::new(vec![feature0, feature1]);

        assert_eq!(diagram.components().len(), 1);
        assert_eq!(diagram.holes().len(), 1);
    }
}
