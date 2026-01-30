//! Approximate Nearest Neighbor Index
//!
//! Fast similarity search for millions of embeddings using a VP-tree.
//!
//! # Algorithm
//!
//! We use a Vantage Point tree (VP-tree) for efficient similarity search.
//! This is a metric tree that partitions space using distance from vantage points.
//!
//! Production systems could use more sophisticated approaches:
//! - HNSW (Hierarchical Navigable Small World graphs)
//! - IVF (Inverted File Index)
//! - LSH (Locality Sensitive Hashing)

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::super::EmbeddingError;
use crate::ontology::loader::IRI;

/// Approximate Nearest Neighbor index using VP-tree
pub struct AnnIndex {
    dimensions: usize,
    vectors: Vec<(IRI, Vec<f32>)>,
    tree: Option<VpTree>,
    built: bool,
}

impl AnnIndex {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            vectors: Vec::new(),
            tree: None,
            built: false,
        }
    }

    pub fn with_capacity(dimensions: usize, capacity: usize) -> Self {
        Self {
            dimensions,
            vectors: Vec::with_capacity(capacity),
            tree: None,
            built: false,
        }
    }

    /// Add a vector to the index
    pub fn add(&mut self, iri: &IRI, vector: &[f32]) -> Result<(), EmbeddingError> {
        if vector.len() != self.dimensions {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.dimensions,
                found: vector.len(),
            });
        }

        self.vectors.push((iri.clone(), vector.to_vec()));
        self.built = false;

        Ok(())
    }

    /// Build the index (must call before search)
    pub fn build(&mut self) -> Result<(), EmbeddingError> {
        if self.vectors.is_empty() {
            self.built = true;
            return Ok(());
        }

        let indices: Vec<usize> = (0..self.vectors.len()).collect();
        self.tree = Some(VpTree::build(&self.vectors, &indices));
        self.built = true;

        Ok(())
    }

    /// Check if index is built
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Get number of vectors in index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(IRI, f64)>, EmbeddingError> {
        if !self.built {
            return Err(EmbeddingError::IndexNotBuilt);
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let tree = self.tree.as_ref().ok_or(EmbeddingError::IndexNotBuilt)?;
        let mut results = BinaryHeap::new();

        tree.search(query, k, &self.vectors, &mut results);

        // into_sorted_vec gives ascending order per our reversed Ord,
        // which means farthest first. We need closest first, so reverse.
        let mut sorted = results.into_sorted_vec();
        sorted.reverse();

        let neighbors: Vec<(IRI, f64)> = sorted
            .into_iter()
            .map(|item| (self.vectors[item.index].0.clone(), item.distance as f64))
            .collect();

        Ok(neighbors)
    }

    /// Search with distance threshold
    pub fn search_radius(
        &self,
        query: &[f32],
        max_distance: f64,
    ) -> Result<Vec<(IRI, f64)>, EmbeddingError> {
        if !self.built {
            return Err(EmbeddingError::IndexNotBuilt);
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let tree = self.tree.as_ref().ok_or(EmbeddingError::IndexNotBuilt)?;
        let mut results = Vec::new();

        tree.search_radius(query, max_distance as f32, &self.vectors, &mut results);

        let mut neighbors: Vec<(IRI, f64)> = results
            .into_iter()
            .map(|(idx, dist)| (self.vectors[idx].0.clone(), dist as f64))
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(neighbors)
    }
}

/// VP-Tree node
struct VpTree {
    vantage_point: usize,
    radius: f32,
    inside: Option<Box<VpTree>>,
    outside: Option<Box<VpTree>>,
}

impl VpTree {
    fn build(vectors: &[(IRI, Vec<f32>)], indices: &[usize]) -> Self {
        if indices.is_empty() {
            panic!("Cannot build VP-tree from empty set");
        }

        if indices.len() == 1 {
            return Self {
                vantage_point: indices[0],
                radius: 0.0,
                inside: None,
                outside: None,
            };
        }

        // Choose vantage point (first element for simplicity)
        // Better heuristics exist (e.g., random sample, maximize spread)
        let vp = indices[0];
        let vp_vec = &vectors[vp].1;

        // Calculate distances from vantage point
        let mut distances: Vec<(usize, f32)> = indices[1..]
            .iter()
            .map(|&i| (i, euclidean_distance(vp_vec, &vectors[i].1)))
            .collect();

        if distances.is_empty() {
            return Self {
                vantage_point: vp,
                radius: 0.0,
                inside: None,
                outside: None,
            };
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Median distance as radius
        let median_idx = distances.len() / 2;
        let radius = distances[median_idx].1;

        // Split into inside (< radius) and outside (>= radius)
        let inside_indices: Vec<usize> = distances[..median_idx].iter().map(|&(i, _)| i).collect();
        let outside_indices: Vec<usize> = distances[median_idx..].iter().map(|&(i, _)| i).collect();

        Self {
            vantage_point: vp,
            radius,
            inside: if inside_indices.is_empty() {
                None
            } else {
                Some(Box::new(Self::build(vectors, &inside_indices)))
            },
            outside: if outside_indices.is_empty() {
                None
            } else {
                Some(Box::new(Self::build(vectors, &outside_indices)))
            },
        }
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        vectors: &[(IRI, Vec<f32>)],
        results: &mut BinaryHeap<SearchResult>,
    ) {
        let d = euclidean_distance(query, &vectors[self.vantage_point].1);

        // Consider this point
        if results.len() < k {
            results.push(SearchResult {
                distance: d,
                index: self.vantage_point,
            });
        } else if d < results.peek().unwrap().distance {
            results.pop();
            results.push(SearchResult {
                distance: d,
                index: self.vantage_point,
            });
        }

        // Current search radius (tau)
        let tau = if results.len() < k {
            f32::MAX
        } else {
            results.peek().unwrap().distance
        };

        // Search order depends on which side query is likely in
        if d < self.radius {
            // Query is likely inside
            if d - tau <= self.radius
                && let Some(ref inside) = self.inside
            {
                inside.search(query, k, vectors, results);
            }
            if d + tau >= self.radius
                && let Some(ref outside) = self.outside
            {
                outside.search(query, k, vectors, results);
            }
        } else {
            // Query is likely outside
            if d + tau >= self.radius
                && let Some(ref outside) = self.outside
            {
                outside.search(query, k, vectors, results);
            }
            if d - tau <= self.radius
                && let Some(ref inside) = self.inside
            {
                inside.search(query, k, vectors, results);
            }
        }
    }

    fn search_radius(
        &self,
        query: &[f32],
        max_distance: f32,
        vectors: &[(IRI, Vec<f32>)],
        results: &mut Vec<(usize, f32)>,
    ) {
        let d = euclidean_distance(query, &vectors[self.vantage_point].1);

        // Include this point if within radius
        if d <= max_distance {
            results.push((self.vantage_point, d));
        }

        // Search inside if query could have results there
        if d - max_distance <= self.radius
            && let Some(ref inside) = self.inside
        {
            inside.search_radius(query, max_distance, vectors, results);
        }

        // Search outside if query could have results there
        if d + max_distance >= self.radius
            && let Some(ref outside) = self.outside
        {
            outside.search_radius(query, max_distance, vectors, results);
        }
    }
}

#[derive(Clone)]
struct SearchResult {
    distance: f32,
    index: usize,
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max-heap (we want smallest distances at top)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.index == other.index
    }
}

impl Eq for SearchResult {}

/// Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine distance (1 - cosine similarity)
#[allow(dead_code)]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ann_index_basic() {
        let mut index = AnnIndex::new(3);

        let iri1 = IRI::new("http://example.org/term1");
        let iri2 = IRI::new("http://example.org/term2");
        let iri3 = IRI::new("http://example.org/term3");

        index.add(&iri1, &[1.0, 0.0, 0.0]).unwrap();
        index.add(&iri2, &[0.0, 1.0, 0.0]).unwrap();
        index.add(&iri3, &[1.0, 1.0, 0.0]).unwrap();

        index.build().unwrap();

        // Search for nearest to [1, 1, 0] should find term3 first
        let results = index.search(&[1.0, 1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, iri3); // Exact match
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - std::f32::consts::SQRT_2).abs() < 0.001);
    }

    #[test]
    fn test_ann_index_empty() {
        let mut index = AnnIndex::new(3);
        index.build().unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }
}
