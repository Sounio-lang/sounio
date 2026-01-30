//! Distance Matrix Computation
//!
//! Builds a distance matrix from ontology hierarchy distances.
//! Uses the native ontology's O(1) LCA queries to compute semantic distance.

use std::collections::HashMap;

use crate::ontology::native::NativeOntology;

/// Distance matrix for concepts
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// Concepts in order
    pub concepts: Vec<String>,
    /// Index lookup: concept -> index
    pub index: HashMap<String, usize>,
    /// Distances stored as flat array (row-major order)
    /// Distance[i][j] = distances[i * n + j]
    distances: Vec<u32>,
    /// Number of concepts (matrix dimension)
    n: usize,
}

impl DistanceMatrix {
    /// Build a distance matrix for the given concepts using ontology hierarchy
    pub fn build(concepts: &[String], ontology: &NativeOntology) -> Self {
        let n = concepts.len();
        let mut index = HashMap::new();
        for (i, concept) in concepts.iter().enumerate() {
            index.insert(concept.clone(), i);
        }

        // Initialize with max distance
        let max_dist = u32::MAX / 2; // Use half max to avoid overflow in additions
        let mut distances = vec![max_dist; n * n];

        // Diagonal is 0
        for i in 0..n {
            distances[i * n + i] = 0;
        }

        // Compute pairwise distances using ontology hierarchy
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = compute_ontology_distance(&concepts[i], &concepts[j], ontology);
                distances[i * n + j] = dist;
                distances[j * n + i] = dist;
            }
        }

        Self {
            concepts: concepts.to_vec(),
            index,
            distances,
            n,
        }
    }

    /// Build an empty distance matrix
    pub fn empty() -> Self {
        Self {
            concepts: Vec::new(),
            index: HashMap::new(),
            distances: Vec::new(),
            n: 0,
        }
    }

    /// Get the distance between two concepts by index
    pub fn get(&self, i: usize, j: usize) -> u32 {
        if i >= self.n || j >= self.n {
            return u32::MAX / 2;
        }
        self.distances[i * self.n + j]
    }

    /// Get the distance between two concepts by name
    pub fn get_by_name(&self, a: &str, b: &str) -> u32 {
        match (self.index.get(a), self.index.get(b)) {
            (Some(&i), Some(&j)) => self.get(i, j),
            _ => u32::MAX / 2,
        }
    }

    /// Get the number of concepts
    pub fn len(&self) -> usize {
        self.n
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Get distances as a slice (row-major order)
    pub fn as_slice(&self) -> &[u32] {
        &self.distances
    }

    /// Convert to condensed form (upper triangle) for clustering algorithms
    pub fn to_condensed(&self) -> Vec<f32> {
        let mut condensed = Vec::with_capacity(self.n * (self.n - 1) / 2);
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                condensed.push(self.get(i, j) as f32);
            }
        }
        condensed
    }
}

/// Compute semantic distance between two concepts using the ontology
fn compute_ontology_distance(a: &str, b: &str, ontology: &NativeOntology) -> u32 {
    // If they're the same, distance is 0
    if a == b {
        return 0;
    }

    // If one is a subclass of the other, use hierarchy distance
    if ontology.is_subclass(a, b) {
        // a is below b, count steps
        return count_hierarchy_steps(a, b, ontology);
    }
    if ontology.is_subclass(b, a) {
        return count_hierarchy_steps(b, a, ontology);
    }

    // Otherwise, find LCA and compute distance through it
    if let Some(lca) = ontology.lca(a, b) {
        let dist_a = count_hierarchy_steps(a, lca, ontology);
        let dist_b = count_hierarchy_steps(b, lca, ontology);
        return dist_a.saturating_add(dist_b);
    }

    // Different ontologies or no common ancestor: high distance
    // Check if same ontology prefix
    let prefix_a = a.split(':').next().unwrap_or("");
    let prefix_b = b.split(':').next().unwrap_or("");

    if prefix_a == prefix_b {
        // Same ontology but no LCA found: moderate distance
        10
    } else {
        // Different ontologies: high distance
        100
    }
}

/// Count hierarchy steps between ancestor and descendant
fn count_hierarchy_steps(descendant: &str, ancestor: &str, ontology: &NativeOntology) -> u32 {
    // Use depth difference as proxy for steps
    let depth_d = ontology.depth(descendant);
    let depth_a = ontology.depth(ancestor);
    depth_d.saturating_sub(depth_a) as u32
}

/// Weighted distance combining ontology distance and co-occurrence
pub fn weighted_distance(
    a: &str,
    b: &str,
    ontology_dist: u32,
    co_occurrence: u32,
    max_co_occurrence: u32,
) -> f32 {
    // Higher co-occurrence = lower effective distance
    let co_factor = if max_co_occurrence > 0 {
        1.0 - (co_occurrence as f32 / max_co_occurrence as f32) * 0.5
    } else {
        1.0
    };

    ontology_dist as f32 * co_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix_empty() {
        let matrix = DistanceMatrix::empty();
        assert!(matrix.is_empty());
        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn test_distance_matrix_basic() {
        // Create a mock ontology for testing
        let ontology = NativeOntology::empty("test");
        let concepts = vec![
            "A:001".to_string(),
            "A:002".to_string(),
            "A:003".to_string(),
        ];

        let matrix = DistanceMatrix::build(&concepts, &ontology);

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix.get(0, 0), 0); // Self-distance
        assert_eq!(matrix.get(1, 1), 0);
        assert_eq!(matrix.get(2, 2), 0);
    }

    #[test]
    fn test_condensed_form() {
        let ontology = NativeOntology::empty("test");
        let concepts = vec![
            "A:001".to_string(),
            "A:002".to_string(),
            "A:003".to_string(),
        ];

        let matrix = DistanceMatrix::build(&concepts, &ontology);
        let condensed = matrix.to_condensed();

        // For 3 concepts, condensed should have 3 elements: (0,1), (0,2), (1,2)
        assert_eq!(condensed.len(), 3);
    }

    #[test]
    fn test_weighted_distance() {
        let dist = weighted_distance("A", "B", 10, 5, 10);
        // High co-occurrence should reduce distance
        assert!(dist < 10.0);

        let dist_no_co = weighted_distance("A", "B", 10, 0, 10);
        assert_eq!(dist_no_co, 10.0);
    }
}
